/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator() : f_manager{Rs} {
  ROS_INFO("init begins");
  initThreadFlag = false;
  clearState();
}

Estimator::~Estimator() {
  if (MULTIPLE_THREAD) {
    processThread.join();
    printf("join thread \n");
  }
}

void Estimator::clearState() {
  mProcess.lock();
  while (!accBuf.empty())
    accBuf.pop();
  while (!gyrBuf.empty())
    gyrBuf.pop();
  while (!featureBuf.empty())
    featureBuf.pop();

  prevTime = -1;
  curTime = 0;
  openExEstimation = 0;
  initP = Eigen::Vector3d(0, 0, 0);
  initR = Eigen::Matrix3d::Identity();
  inputImageCnt = 0;
  initFirstPoseFlag = false;

  for (int i = 0; i < WINDOW_SIZE + 1; i++) {
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();

    if (pre_integrations[i] != nullptr) {
      delete pre_integrations[i];
    }
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = Vector3d::Zero();
    ric[i] = Matrix3d::Identity();
  }

  first_imu = false, sum_of_back = 0;
  sum_of_front = 0;
  frame_count = 0;
  solver_flag = INITIAL;
  initial_timestamp = 0;
  all_image_frame.clear();

  if (tmp_pre_integration != nullptr)
    delete tmp_pre_integration;
  if (last_marginalization_info != nullptr)
    delete last_marginalization_info;

  tmp_pre_integration = nullptr;
  last_marginalization_info = nullptr;
  last_marginalization_parameter_blocks.clear();

  f_manager.clearState();

  failure_occur = 0;

  mProcess.unlock();
}

void Estimator::setParameter() {
  mProcess.lock();
  for (int i = 0; i < NUM_OF_CAM; i++) {
    tic[i] = TIC[i];
    ric[i] = RIC[i];
    cout << " exitrinsic cam " << i << endl
         << ric[i] << endl
         << tic[i].transpose() << endl;
  }
  f_manager.setRic(ric);
  // ?
  ProjectionTwoFrameOneCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionTwoFrameTwoCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionOneFrameTwoCamFactor::sqrt_info =
      FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  td = TD;
  g = G;
  cout << "set g " << g.transpose() << endl;
  featureTracker.readIntrinsicParameter(CAM_NAMES);

  std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
  if (MULTIPLE_THREAD && !initThreadFlag) {
    initThreadFlag = true;
    processThread = std::thread(&Estimator::processMeasurements, this);
  }
  mProcess.unlock();
}
/**
 * @brief Estimator::changeSensorType
 * 改变传感器类型
 * @param use_imu
 * @param use_stereo
 */
void Estimator::changeSensorType(int use_imu, int use_stereo) {
  bool restart = false;
  mProcess.lock();
  if (!use_imu && !use_stereo)
    printf("at least use two sensors! \n");
  else {
    if (USE_IMU != use_imu) {
      USE_IMU = use_imu;
      if (USE_IMU) {
        // reuse imu; restart system
        restart = true;
      } else {
        if (last_marginalization_info != nullptr)
          delete last_marginalization_info;

        tmp_pre_integration = nullptr;
        last_marginalization_info = nullptr;
        last_marginalization_parameter_blocks.clear();
      }
    }

    STEREO = use_stereo;
    printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
  }
  mProcess.unlock();
  if (restart) {
    clearState();
    setParameter();
  }
}

void Estimator::inputImage(double t, const cv::Mat &_img,
                           const cv::Mat &_img1) {
  // 统计输入的帧数
  inputImageCnt++;
  map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
  TicToc featureTrackerTime;

  // 进行特征跟踪
  // featureFrame数据类型为map<int, vector<pair<int, Eigen::Matrix<double, 7,
  // 1>>>>
  // 依次对应featureId_cameraId_xyzuvVxVy
  if (_img1.empty())
    featureFrame = featureTracker.trackImage(t, _img);
  else
    featureFrame = featureTracker.trackImage(t, _img, _img1);
  // printf("featureTracker time: %f\n", featureTrackerTime.toc());

  if (SHOW_TRACK) {
    cv::Mat imgTrack = featureTracker.getTrackImage();
    pubTrackImage(imgTrack, t);
  }

  if (MULTIPLE_THREAD) {
    if (inputImageCnt % 2 == 0) {
      // 偶数帧
      mBuf.lock();
      featureBuf.push(make_pair(t, featureFrame));
      mBuf.unlock();
    }
  } else {
    // 奇数帧
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();
    TicToc processTime;
    // NOTE: processMeasurements()是一个循环函数
    // 其在setParameter()函数中作为一个线程，这里重复调用
    processMeasurements();
    printf("process time: %f\n", processTime.toc());
  }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration,
                         const Vector3d &angularVelocity) {
  mBuf.lock();
  // 将acc和gyr分别存储在accBuf和gyrBuf中
  accBuf.push(make_pair(t, linearAcceleration));
  gyrBuf.push(make_pair(t, angularVelocity));
  // printf("input imu with time %f \n", t);
  mBuf.unlock();

  // VINS初始化已完成，正处于滑动窗口非线性状态；如果VINS还在初始化，则不发布里程计信息
  // 表示系统处于非线性优化状态
  if (solver_flag == NON_LINEAR) {
    mPropagate.lock();
    fastPredictIMU(t, linearAcceleration, angularVelocity);
    // 发布里程计信息，每次获取IMU数据都会及时进行更新，而且发布的是当前的里程计信息
    pubLatestOdometry(latest_P, latest_Q, latest_V, t);
    mPropagate.unlock();
  }
}

void Estimator::inputFeature(
    double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
                  &featureFrame) {
  mBuf.lock();
  featureBuf.push(make_pair(t, featureFrame));
  mBuf.unlock();

  // 如果开启了多线程，将不会在此处处理测量
  if (!MULTIPLE_THREAD)
    processMeasurements();
}

/**
 * @brief Estimator::getIMUInterval
 *        获得两帧图像之间的IMU数据
 * @param t0
 * @param t1
 * @param accVector
 * @param gyrVector
 * @return
 */
bool Estimator::getIMUInterval(
    double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
    vector<pair<double, Eigen::Vector3d>> &gyrVector) {
  if (accBuf.empty()) {
    printf("not receive imu\n");
    return false;
  }
  // printf("get imu from %f %f\n", t0, t1);
  // printf("imu fornt time %f   imu end time %f\n", accBuf.front().first,
  // accBuf.back().first);

  // imu队列的最后一个imu数据的时间要大于当前帧的时间
  if (t1 <= accBuf.back().first) {
    // 剔除imu队列中imu数据的时间小于等于先前帧的时间的imu数据
    // NOTE:其会使得该区间的IMU数据中的第一个imu数据大于先前帧的时间，从而导致该区间的预积分不准确
    // 特别是当IMU数据与图像数据不同步时
    // 时间戳对应关系如下：
    //                       imu_t
    // *     *     *     *     *        (IMU数据）
    //                       +          (Img数据）
    //                     img_t
    while (accBuf.front().first <= t0) {
      accBuf.pop();
      gyrBuf.pop();
    }
    // 将imu队列中剩下的imu数据的时间小于当前帧
    while (accBuf.front().first < t1) {
      accVector.push_back(accBuf.front());
      accBuf.pop();
      gyrVector.push_back(gyrBuf.front());
      gyrBuf.pop();
    }
    // 该IMU数据的时间大于或等于当前帧的时间
    accVector.push_back(accBuf.front());
    gyrVector.push_back(gyrBuf.front());
    // 保留imu数据的时间戳对应关系如下：
    //     imu_t0                  imu_t1
    // *     *     *     *     *     *     *    (IMU数据）
    //    +                       +             (Img数据)
    //  prev                     cur
  } else {
    printf("wait for imu\n");
    return false;
  }
  return true;
}

/**
 * @brief Estimator::IMUAvailable
 *        判断imu数据是否有效
 *        首先imu数据队列不为空，其次imu数据队列的最后一个imu数据的时间要大于当前帧的时间
 * @param t
 * @return
 */
bool Estimator::IMUAvailable(double t) {
  if (!accBuf.empty() && t <= accBuf.back().first)
    return true;
  else
    return false;
}

void Estimator::processMeasurements() {
  while (1) {
    // printf("process measurments\n");
    pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>
        feature;
    vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
    if (!featureBuf.empty()) {
      feature = featureBuf.front();
      curTime = feature.first + td;
      // 不使用IMU数据则跳出循环，使用IMU数据则需要判断IMU数据是否有效
      while (1) {
        if ((!USE_IMU || IMUAvailable(feature.first + td)))
          break;
        else {
          printf("wait for imu ... \n");
          if (!MULTIPLE_THREAD)
            return;
          std::chrono::milliseconds dura(5);
          std::this_thread::sleep_for(dura);
        }
      }
      mBuf.lock();
      // 其通过prevTime和currTime来控制IMU的积分区间
      // prev表示上一帧的的时间，curTime表示当前帧的时间
      if (USE_IMU)
        getIMUInterval(prevTime, curTime, accVector, gyrVector);

      featureBuf.pop();
      mBuf.unlock();

      if (USE_IMU) {
        // 初始化标志位
        if (!initFirstPoseFlag)
          initFirstIMUPose(accVector);
        for (size_t i = 0; i < accVector.size(); i++) {
          double dt;
          // 时间间隔示意图
          //         t_begin                                    t_end
          //     d_begin                                      d_end
          //     ----|-------|-------|-------|-------|-------|---|
          // *       *       *       *       *       *       *       *
          //     ^                                               ^
          //   prev                                             end
          if (i == 0) // 第一个imu数据
            dt = accVector[i].first - prevTime;
          else if (i == accVector.size() - 1) // 最后一个imu数据
            dt = curTime - accVector[i - 1].first;
          else
            dt = accVector[i].first - accVector[i - 1].first;
          processIMU(accVector[i].first, dt, accVector[i].second,
                     gyrVector[i].second);
        }
      }
      mProcess.lock();
      processImage(feature.second, feature.first);
      prevTime = curTime;

      printStatistics(*this, 0);

      std_msgs::Header header;
      header.frame_id = "world";
      header.stamp = ros::Time(feature.first);

      pubOdometry(*this, header);
      pubKeyPoses(*this, header);
      pubCameraPose(*this, header);
      pubPointCloud(*this, header);
      pubKeyframe(*this);
      pubTF(*this, header);
      mProcess.unlock();
    }

    if (!MULTIPLE_THREAD)
      break;

    std::chrono::milliseconds dura(2);
    std::this_thread::sleep_for(dura);
  }
}

void Estimator::initFirstIMUPose(
    vector<pair<double, Eigen::Vector3d>> &accVector) {
  printf("init first imu pose\n");
  initFirstPoseFlag = true;
  // return;
  Eigen::Vector3d averAcc(0, 0, 0);
  int n = (int)accVector.size();
  for (size_t i = 0; i < accVector.size(); i++) {
    averAcc = averAcc + accVector[i].second;
  }
  // 求平均加速度
  averAcc = averAcc / n;
  printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
  // 重力占加速度的绝大一部分
  Matrix3d R0 = Utility::g2R(averAcc);
  double yaw = Utility::R2ypr(R0).x();
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  Rs[0] = R0;
  cout << "init R0 " << endl << Rs[0] << endl;
  // Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r) {
  Ps[0] = p;
  Rs[0] = r;
  initP = p;
  initR = r;
}

/**
 * @brief Estimator::processIMU
 *        处理IMU数据
 * @param t
 * @param dt
 * @param linear_acceleration
 * @param angular_velocity
 */
void Estimator::processIMU(double t, double dt,
                           const Vector3d &linear_acceleration,
                           const Vector3d &angular_velocity) {
  // 第一个imu数据，只执行一次
  if (!first_imu) {
    first_imu = true;
    // NOTE:acc_0仅在processIMU函数中进行了赋值，猜测其存储前一时刻的IMU数据
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  // 初始化后将滑动窗口中的预积分量对象pre_integration都设置为nullptr
  // pre_integrations容器的大小为WINDOW_SIZE + 1，即11
  // 构建滑动窗口中的预积分
  if (!pre_integrations[frame_count]) {
    pre_integrations[frame_count] =
        new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
  }
  // 第一个frame_count将不进行预积分，只会更新加速度和角速度
  if (frame_count != 0) {
    pre_integrations[frame_count]->push_back(dt, linear_acceleration,
                                             angular_velocity);
    // if(solver_flag != NON_LINEAR)
    // temp_pre_integration与帧关系较大
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

    // 将时间差，线加速度，角速度保存到窗口的容器中
    // 这样不会超出窗口的大小吗
    // 存储该帧imu数据所有的间隔时间
    dt_buf[frame_count].push_back(dt);
    // 存储该帧imu数据所有的加速度
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    // 存储该帧imu数据所有的角速度
    angular_velocity_buf[frame_count].push_back(angular_velocity);

    // TODO: 不明白这部分积分的意义
    int j = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
    Vs[j] += dt * un_acc;
  }
  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}
/**
 * @brief   处理图像特征数据
 * @abstract
 * addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧
 * 判断并进行外参标定
 * 进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param[in]   image
 * 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   header 某帧图像的头信息
 * @return  void
*/
void Estimator::processImage(
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    const double header) {
  ROS_DEBUG("new image coming ------------------------------------------");
  ROS_DEBUG("Adding feature points %lu", image.size());
  // 添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
  // 通过检测两帧之间的视差决定次新帧是否作为关键帧

  // 在构造estimator对象时，就使用Rs(存储窗口帧旋转矩阵的容器)对f_manager进行构造
  if (f_manager.addFeatureCheckParallax(frame_count, image, td)) {
    // 新的信息较多，则边缘化旧的
    marginalization_flag = MARGIN_OLD;
    // printf("keyframe\n");
  } else {
    // 旧的信息占主导，则边缘化第二新的帧
    marginalization_flag = MARGIN_SECOND_NEW;
    // printf("non-keyframe\n");
  }

  ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
  ROS_DEBUG("Solving %d", frame_count);
  // 窗口内跟踪次数大于4的特征点数量
  ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
  Headers[frame_count] = header; // 存储当前帧的时间戳

  // 用featureId_cameraId_xyzuvVxVy和时间构造图像帧
  ImageFrame imageframe(image, header);
  imageframe.pre_integration = tmp_pre_integration;
  // 存储图像数据及其对应的时间戳
  all_image_frame.insert(make_pair(header, imageframe));
  // Bas和Bgs是什么时候更新的
  tmp_pre_integration =
      new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

  // 相机与IMU在线标定，至少会迭代计算WINDOW_SIZE次
  if (ESTIMATE_EXTRINSIC == 2) {
    ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    if (frame_count != 0) {
      // 前后两帧之间归一化的特征点
      vector<pair<Vector3d, Vector3d>> corres =
          f_manager.getCorresponding(frame_count - 1, frame_count);
      Matrix3d calib_ric;
      // 如果标定结果好，就不用再标定了
      if (initial_ex_rotation.CalibrationExRotation(
              corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
        ROS_WARN("initial extrinsic rotation calib success");
        ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
        ric[0] = calib_ric;
        RIC[0] = calib_ric;
        ESTIMATE_EXTRINSIC = 1;
      }
    }
  }
  // 同时标定且初始化
  if (solver_flag == INITIAL) {
    // monocular + IMU initilization
    if (!STEREO && USE_IMU) {
      // frame_count是滑动窗口中图像帧的数量，一开始初始化为0，滑动窗口总帧数WINDOW_SIZE=10
      //确保有足够的frame参与初始化
      if (frame_count == WINDOW_SIZE) {
        bool result = false;
        //有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
        if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1) {
          // 视觉惯性联合初始化（松耦合）
          result = initialStructure();
          initial_timestamp = header;
        }
        if (result) {
          optimization();
          updateLatestStates();
          solver_flag = NON_LINEAR;
          slideWindow();
          ROS_INFO("Initialization finish!");
        } else
          slideWindow();
      }
    }

    // stereo + IMU initilization
    if (STEREO && USE_IMU) {
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
      f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
      if (frame_count == WINDOW_SIZE) {
        map<double, ImageFrame>::iterator frame_it;
        int i = 0;
        for (frame_it = all_image_frame.begin();
             frame_it != all_image_frame.end(); frame_it++) {
          frame_it->second.R = Rs[i];
          frame_it->second.T = Ps[i];
          i++;
        }
        solveGyroscopeBias(all_image_frame, Bgs);
        for (int i = 0; i <= WINDOW_SIZE; i++) {
          pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
        }
        optimization();
        updateLatestStates();
        solver_flag = NON_LINEAR;
        slideWindow();
        ROS_INFO("Initialization finish!");
      }
    }

    // stereo only initilization
    if (STEREO && !USE_IMU) {
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
      f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
      optimization();

      if (frame_count == WINDOW_SIZE) {
        optimization();
        updateLatestStates();
        solver_flag = NON_LINEAR;
        slideWindow();
        ROS_INFO("Initialization finish!");
      }
    }

    // frame_count会不断累加，知道等于WINDOW_SIZE
    if (frame_count < WINDOW_SIZE) {
      frame_count++;
      int prev_frame = frame_count - 1;
      Ps[frame_count] = Ps[prev_frame];
      Vs[frame_count] = Vs[prev_frame];
      Rs[frame_count] = Rs[prev_frame];
      Bas[frame_count] = Bas[prev_frame];
      Bgs[frame_count] = Bgs[prev_frame];
    }

  } else {
    TicToc t_solve;
    if (!USE_IMU)
      f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
    optimization();
    set<int> removeIndex;
    outliersRejection(removeIndex);
    f_manager.removeOutlier(removeIndex);
    if (!MULTIPLE_THREAD) {
      featureTracker.removeOutliers(removeIndex);
      predictPtsInNextFrame();
    }

    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    if (failureDetection()) {
      ROS_WARN("failure detection!");
      failure_occur = 1;
      clearState();
      setParameter();
      ROS_WARN("system reboot!");
      return;
    }

    slideWindow();
    f_manager.removeFailures();
    // prepare output of VINS
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
      key_poses.push_back(Ps[i]);

    last_R = Rs[WINDOW_SIZE];
    last_P = Ps[WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];
    updateLatestStates();
  }
}
/**
 * @brief   视觉的结构初始化
 * @abstract  确保IMU有充分运动激励
 *            relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
 *            sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
 *            visualInitialAlign()视觉惯性联合初始化
 * @return  bool true:初始化成功
*/

/**
 * vins系统初始化
 * 1.确保IMU有足够的excitation
 * 2.检查当前帧（滑动窗口中的最新帧）与滑动窗口中所有图像帧之间的特征点匹配关系，
 *   选择跟当前帧中有足够多数量的特征点（30个）被跟踪，且由足够视差（20-pixels）的某一帧
 *   利用五点法恢复相对旋转和平移量。如果找不到，则在滑动窗口中保留当前帧，然后等待新的图像帧
 * 3.sfm.construct 全局SFM 恢复滑动窗口中所有帧的位姿，以及特特征点三角化
 * 4.利用pnp恢复其他帧
 * 5.visual-inertial alignment：视觉SFM的结果与IMU预积分结果对齐
 * 6.给滑动窗口中要优化的变量一个合理的初始值以便进行非线性优化
 */

//基本流程：
// (1)SFM恢复出旋转以及带尺度因子的平移
// (2)利用SFM恢复出的旋转矫正优化陀螺仪的偏差,并使用优化后的数值积分
// (3)利用加速度计的积分恢复出速度,尺度因子,C0坐标系下的重力
// (4)利用已知的重力与c0坐标系下的重力,恢复出c0相对世界坐标系的旋转
// (5)将恢复的旋转作用于所有帧,转换为世界坐标系的位姿状态.
bool Estimator::initialStructure() {
  TicToc t_sfm;
  // check imu observibility
  //通过加速度标准差判断IMU是否有充分运动以初始化。
  {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g;
    // 遍历所有的图像数据
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      // 用预积分求得的速度求平均加速度
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++;
         frame_it != all_image_frame.end(); frame_it++) {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
      // cout << "frame g " << tmp_g.transpose() << endl;
    }

    var = sqrt(var / ((int)all_image_frame.size() - 1));
    // ROS_WARN("IMU variation %f!", var);
    // NOTE: 就只是提醒一下吗
    if (var < 0.25) {
      ROS_INFO("IMU excitation not enouth!");
      // return false;
    }
  }
  // global sfm
  //将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中

  // 用于记录滑动窗口中每一帧的姿态
  Quaterniond Q[frame_count + 1];
  // 用于记录滑动窗口中每一帧的位置
  Vector3d T[frame_count + 1];
  // sfm跟踪的点
  map<int, Vector3d> sfm_tracked_points;

  vector<SFMFeature> sfm_f; //用于sfm的所有特征点
  for (auto &it_per_id : f_manager.feature) {
    // TODO imu_j可以不用减1，imu_j++放在循环最后就可以了
    // 该特征点被观测到的起始帧
    int imu_j = it_per_id.start_frame - 1;
    // 临时的特征点数据结构
    SFMFeature tmp_feature;
    // 该特征点的状态：未被初始化
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      Vector3d pts_j = it_per_frame.point;
      tmp_feature.observation.push_back(
          make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
    }
    sfm_f.push_back(tmp_feature);
  }

  Matrix3d relative_R;
  Vector3d relative_T;
  int l;

  // 保证具有足够的视差,由F矩阵恢复Rt
  // 第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
  // 此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
  if (!relativePose(relative_R, relative_T, l)) {
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
  }

  //对窗口中每个图像帧求解sfm问题
  //得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
  // frame_count+1表示当前窗口帧数，因为frame_count是从0开始索引的
  GlobalSFM sfm;
  if (!sfm.construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f,
                     sfm_tracked_points)) {
    ROS_DEBUG("global SFM failed!");
    marginalization_flag = MARGIN_OLD;
    return false;
  }

  // solve pnp for all frame
  // 对于所有的图像帧，包括不在滑动窗口中的（貌似只会多出一帧），提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
  // 最后把世界坐标系从帧l的相机坐标系，转到帧l的IMU坐标系

  // NOTE：几乎
  // 4.对于非滑动窗口的所有帧，提供一个初始的R,T，然后solve pnp求解pose
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  // all_image_frame的范围比滑动窗口大，滑动窗口有的它都有，它有的滑动窗口不一定有。
  frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end(); frame_it++) {
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    // Headers[i]:记录滑动窗口的帧，表示该帧在滑动窗口中
    if ((frame_it->first) == Headers[i]) {
      // 设置滑动窗口中的帧都是关键帧
      frame_it->second.is_key_frame = true;
      // NOTE: 这里应该是为了便于实现相机与IMU紧耦合
      // 根据各帧相机坐标系的姿态和外参，得到各帧相机相对参考帧IMU坐标系的姿态
      frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
      // T实际上应该是-R*Rci*Tic + T，但是-R*Rci*Tic太小了，忽略了
      frame_it->second.T = T[i];
      i++;
      continue;
    }
    // 对于不在滑动窗口的帧，i不能大于滑动窗口+1，否则会超出Headers的索引范围
    // headers[WINDOW_SIZE+1] = 0
    // 可能是滑动窗口之前丢弃的帧
    // NOTE: 这种情况几乎不可能出现，而且就算满足了，i++的话，不会超出边界吗
    if ((frame_it->first) > Headers[i]) {
      i++;
    }
    // 初始位姿，Q[WINDOWS+1]也没有被赋值
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    // 转换格式，并将旋转矩阵变换为旋转向量
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    // 位于滑动窗口外的帧是非关键帧
    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    // 遍历该帧的特征点
    for (auto &id_pts : frame_it->second.points) {
      // 该特征点id
      int feature_id = id_pts.first;
      // 一个特征点对应双目观测
      for (auto &i_p : id_pts.second) {
        // 从sfm_tracked_points中找到该特征点
        it = sfm_tracked_points.find(feature_id);
        if (it != sfm_tracked_points.end()) {
          Vector3d world_pts = it->second;
          cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2(img_pts(0), img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }
    // 因为使用的是归一化坐标，所以内参为I
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6) {
      cout << "pts_3_vector size " << pts_3_vector.size() << endl;
      ROS_DEBUG("Not enough points for solve pnp !");
      return false;
    }
    // NOTE: 用已经三角化的空间点和双目分别观测的特征点可以恢复相机的位姿吗
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) {
      ROS_DEBUG("solve pnp fail!");
      return false;
    }
    // 成功求解该帧位姿
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp, tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    frame_it->second.R = R_pnp * RIC[0].transpose();
    frame_it->second.T = T_pnp;
  }
  // camera与IMU对齐:矫正陀螺仪偏差 -> 恢复尺度,速度,重力
  // ->细化重力(解算初始旋转与东北天坐标系的旋转) ->将旋转阵作用于所有帧
  if (visualInitialAlign())
    return true;
  else {
    ROS_INFO("misalign visual structure with IMU");
    return false;
  }
}

/**
 * @brief   视觉惯性联合初始化
 * @abstract  陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *            更新了Bgs后，IMU测量量需要repropagate
 *            得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlign() {
  TicToc t_g;
  VectorXd x;
  // solve scale
  // 计算陀螺仪偏置，尺度，重力加速度和速度
  bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
  if (!result) {
    ROS_DEBUG("solve g failed!");
    return false;
  }

  // change state
  // 得到所有图像帧的位姿Ps,Rs，并将其设置为关键帧
  // 其实还是把窗口中的帧设置为关键帧
  for (int i = 0; i <= frame_count; i++) {
    // NOTE: 帧的IMU相对世界的旋转
    Matrix3d Ri = all_image_frame[Headers[i]].R;
    // NOTE: 依旧是帧之间的位移
    Vector3d Pi = all_image_frame[Headers[i]].T;
    Ps[i] = Pi;
    // NOTE:
    Rs[i] = Ri;
    all_image_frame[Headers[i]].is_key_frame = true;
  }

  // 取出尺度
  double s = (x.tail<1>())(0);
  // 遍历窗口
  // 再次反向传播
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    // NOTE: 在VisualIMUAligment()函数中已经重新计算过了，再次计算有什么不一样吗
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }
  // 将Ps、Vs、depth尺度s缩放，为什么
  // 将位置position转换到以第b0为基础(即初始帧体坐标)
  // NOTE: 变化后的Ps[i]为第i帧imu坐标系到第0帧imu坐标系的变换
  // 变化前的Ps[i]是帧i的相对于帧l的相机的位移（帧与帧之间）
  // Rs[i]是帧i的IMU相对于帧l的相机的旋转（IMU与相机之间）
  // TIC是帧与IMU固定的位移
  // Ps[0]和Rs[0]同理
  // s*Ps[i] - Rs[i]*TIC[0]就是帧i的相机相对于帧l的IMU的位移
  // T^(c_l)_(b_k) * TIC = T^(c_l)_(c_k) (其结果为相机c_k到相机c的变换）
  // | R   P |  *  | RIC   TIC |   =  | R*RIC   R*TIC+P|
  // | 0   1 |     |  0     1  |      |  0         1   |
  // 可知Rs[i]*TIC + P = Ps[i] ---> P = Ps[i] - Rs[i]*TIC
  // P为帧i的IMU到帧l的相机的平移向量
  // 从而得出帧i的IMU到帧l的相机的平移向量，以此也计算出了帧0的IMU到帧l之间的平移向量
  // P^l_i - P^l_0 --> P^0_l 向量相减，减数指向被减数
  for (int i = frame_count; i >= 0; i--)
    Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
  // NOTE: 更新速度以b0（帧0的Body坐标系）为参考坐标系，R不是参考l帧的吗
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end();
       frame_i++) {
    if (frame_i->second.is_key_frame) {
      kv++;
      // Vs为优化得到的速度
      // R为帧i的IMU相对于帧l的相机的旋转，这样会把速度转换到l帧（参考帧）相机坐标系下
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }

  // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
  // 当前参考坐标系与世界坐标系（依靠g构建坐标系）
  // NOTE: 不是十分明白怎么变换的
  Matrix3d R0 = Utility::g2R(g);
  // 将R0作用于Rs[0]，再求其相对z轴的旋转角
  double yaw = Utility::R2ypr(R0 * Rs[0]).x();
  // 再将该旋转角变换为旋转矩阵作用与R0
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  g = R0 * g;
  // Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count; i++) {
    Ps[i] = rot_diff * Ps[i];
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  ROS_DEBUG_STREAM("g0     " << g.transpose());
  ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

  // 将所有特征点的深度置为-1
  f_manager.clearDepth();
  // 重新计算特征点的深度
  f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

  return true;
}

/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @abstract    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 保存滑动窗口中与当前帧满足初始化条件的那一帧
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
// 在滑动窗口中，寻找与最新帧有足够多数量的特征点对应关系和视差的帧，然后用5点法恢复相对位姿
// 保证具有足够的视差,由E矩阵恢复R、t
// 这里的第L帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用，得到的Rt为当前帧到第l帧的坐标系变换Rt
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T,
                             int &l) {
  // find previous frame which contians enough correspondance and parallex with
  // newest frame
  // 寻找第i帧到窗口最后一帧的对应特征点
  for (int i = 0; i < WINDOW_SIZE; i++) {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.getCorresponding(i, WINDOW_SIZE);
    if (corres.size() > 20) {
      // 计算平均视差
      double sum_parallax = 0;
      double average_parallax;
      for (int j = 0; j < int(corres.size()); j++) {
        //第j个对应点在第i帧和最后一帧的(x,y)
        Vector2d pts_0(corres[j].first(0), corres[j].first(1));
        Vector2d pts_1(corres[j].second(0), corres[j].second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax = sum_parallax + parallax;
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());

      // 判断是否满足初始化条件：视差>30和内点数满足要求
      // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
      // 五点法求解位姿
      if (average_parallax * 460 > 30 &&
          m_estimator.solveRelativeRT(corres, relative_R, relative_T)) {
        // 返回最新匹配的帧
        l = i;
        ROS_DEBUG("average_parallax %f choose l %d and newest frame to "
                  "triangulate the whole structure",
                  average_parallax * 460, l);
        return true;
      }
    }
  }
  return false;
}

void Estimator::vector2double() {
  for (int i = 0; i <= WINDOW_SIZE; i++) {
    para_Pose[i][0] = Ps[i].x();
    para_Pose[i][1] = Ps[i].y();
    para_Pose[i][2] = Ps[i].z();
    Quaterniond q{Rs[i]};
    para_Pose[i][3] = q.x();
    para_Pose[i][4] = q.y();
    para_Pose[i][5] = q.z();
    para_Pose[i][6] = q.w();

    if (USE_IMU) {
      para_SpeedBias[i][0] = Vs[i].x();
      para_SpeedBias[i][1] = Vs[i].y();
      para_SpeedBias[i][2] = Vs[i].z();

      para_SpeedBias[i][3] = Bas[i].x();
      para_SpeedBias[i][4] = Bas[i].y();
      para_SpeedBias[i][5] = Bas[i].z();

      para_SpeedBias[i][6] = Bgs[i].x();
      para_SpeedBias[i][7] = Bgs[i].y();
      para_SpeedBias[i][8] = Bgs[i].z();
    }
  }

  for (int i = 0; i < NUM_OF_CAM; i++) {
    para_Ex_Pose[i][0] = tic[i].x();
    para_Ex_Pose[i][1] = tic[i].y();
    para_Ex_Pose[i][2] = tic[i].z();
    Quaterniond q{ric[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    para_Feature[i][0] = dep(i);

  para_Td[0][0] = td;
}

void Estimator::double2vector() {
  Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
  Vector3d origin_P0 = Ps[0];

  if (failure_occur) {
    origin_R0 = Utility::R2ypr(last_R0);
    origin_P0 = last_P0;
    failure_occur = 0;
  }

  if (USE_IMU) {
    Vector3d origin_R00 =
        Utility::R2ypr(Quaterniond(para_Pose[0][6], para_Pose[0][3],
                                   para_Pose[0][4], para_Pose[0][5])
                           .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 ||
        abs(abs(origin_R00.y()) - 90) < 1.0) {
      ROS_DEBUG("euler singular point!");
      rot_diff = Rs[0] *
                 Quaterniond(para_Pose[0][6], para_Pose[0][3], para_Pose[0][4],
                             para_Pose[0][5])
                     .toRotationMatrix()
                     .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {

      Rs[i] = rot_diff *
              Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4],
                          para_Pose[i][5])
                  .normalized()
                  .toRotationMatrix();

      Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                  para_Pose[i][1] - para_Pose[0][1],
                                  para_Pose[i][2] - para_Pose[0][2]) +
              origin_P0;

      Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0], para_SpeedBias[i][1],
                                  para_SpeedBias[i][2]);

      Bas[i] = Vector3d(para_SpeedBias[i][3], para_SpeedBias[i][4],
                        para_SpeedBias[i][5]);

      Bgs[i] = Vector3d(para_SpeedBias[i][6], para_SpeedBias[i][7],
                        para_SpeedBias[i][8]);
    }
  } else {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
      Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4],
                          para_Pose[i][5])
                  .normalized()
                  .toRotationMatrix();

      Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
    }
  }

  if (USE_IMU) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      tic[i] =
          Vector3d(para_Ex_Pose[i][0], para_Ex_Pose[i][1], para_Ex_Pose[i][2]);
      ric[i] = Quaterniond(para_Ex_Pose[i][6], para_Ex_Pose[i][3],
                           para_Ex_Pose[i][4], para_Ex_Pose[i][5])
                   .toRotationMatrix();
    }
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    dep(i) = para_Feature[i][0];
  f_manager.setDepth(dep);

  if (USE_IMU)
    td = para_Td[0][0];
}

bool Estimator::failureDetection() {
  return false;
  if (f_manager.last_track_num < 2) {
    ROS_INFO(" little feature %d", f_manager.last_track_num);
    // return true;
  }
  if (Bas[WINDOW_SIZE].norm() > 2.5) {
    ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
    return true;
  }
  if (Bgs[WINDOW_SIZE].norm() > 1.0) {
    ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
    return true;
  }
  /*
  if (tic(0) > 1)
  {
      ROS_INFO(" big extri param estimation %d", tic(0) > 1);
      return true;
  }
  */
  Vector3d tmp_P = Ps[WINDOW_SIZE];
  if ((tmp_P - last_P).norm() > 5) {
    // ROS_INFO(" big translation");
    // return true;
  }
  if (abs(tmp_P.z() - last_P.z()) > 1) {
    // ROS_INFO(" big z translation");
    // return true;
  }
  Matrix3d tmp_R = Rs[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) {
    ROS_INFO(" big delta_angle ");
    // return true;
  }
  return false;
}

void Estimator::optimization() {
  TicToc t_whole, t_prepare;
  vector2double();

  ceres::Problem problem;
  ceres::LossFunction *loss_function;
  // loss_function = NULL;
  loss_function = new ceres::HuberLoss(1.0);
  // loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
  // ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
  for (int i = 0; i < frame_count + 1; i++) {
    ceres::LocalParameterization *local_parameterization =
        new PoseLocalParameterization();
    problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    if (USE_IMU)
      problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }
  if (!USE_IMU)
    problem.SetParameterBlockConstant(para_Pose[0]);

  for (int i = 0; i < NUM_OF_CAM; i++) {
    ceres::LocalParameterization *local_parameterization =
        new PoseLocalParameterization();
    problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE,
                              local_parameterization);
    if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE &&
         Vs[0].norm() > 0.2) ||
        openExEstimation) {
      // ROS_INFO("estimate extinsic param");
      openExEstimation = 1;
    } else {
      // ROS_INFO("fix extinsic param");
      problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
  }
  problem.AddParameterBlock(para_Td[0], 1);

  if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
    problem.SetParameterBlockConstant(para_Td[0]);

  if (last_marginalization_info && last_marginalization_info->valid) {
    // construct new marginlization_factor
    MarginalizationFactor *marginalization_factor =
        new MarginalizationFactor(last_marginalization_info);
    problem.AddResidualBlock(marginalization_factor, NULL,
                             last_marginalization_parameter_blocks);
  }
  if (USE_IMU) {
    for (int i = 0; i < frame_count; i++) {
      int j = i + 1;
      if (pre_integrations[j]->sum_dt > 10.0)
        continue;
      IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
      problem.AddResidualBlock(imu_factor, NULL, para_Pose[i],
                               para_SpeedBias[i], para_Pose[j],
                               para_SpeedBias[j]);
    }
  }

  int f_m_cnt = 0;
  int feature_index = -1;
  for (auto &it_per_id : f_manager.feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)
      continue;

    ++feature_index;

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point;
        ProjectionTwoFrameOneCamFactor *f_td =
            new ProjectionTwoFrameOneCamFactor(
                pts_i, pts_j, it_per_id.feature_per_frame[0].velocity,
                it_per_frame.velocity, it_per_id.feature_per_frame[0].cur_td,
                it_per_frame.cur_td);
        problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i],
                                 para_Pose[imu_j], para_Ex_Pose[0],
                                 para_Feature[feature_index], para_Td[0]);
      }

      if (STEREO && it_per_frame.is_stereo) {
        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {
          ProjectionTwoFrameTwoCamFactor *f =
              new ProjectionTwoFrameTwoCamFactor(
                  pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
                  it_per_frame.velocityRight,
                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
          problem.AddResidualBlock(f, loss_function, para_Pose[imu_i],
                                   para_Pose[imu_j], para_Ex_Pose[0],
                                   para_Ex_Pose[1], para_Feature[feature_index],
                                   para_Td[0]);
        } else {
          ProjectionOneFrameTwoCamFactor *f =
              new ProjectionOneFrameTwoCamFactor(
                  pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity,
                  it_per_frame.velocityRight,
                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
          problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0],
                                   para_Ex_Pose[1], para_Feature[feature_index],
                                   para_Td[0]);
        }
      }
      f_m_cnt++;
    }
  }

  ROS_DEBUG("visual measurement count: %d", f_m_cnt);
  // printf("prepare for ceres: %f \n", t_prepare.toc());

  ceres::Solver::Options options;

  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = NUM_ITERATIONS;
  // options.use_explicit_schur_complement = true;
  // options.minimizer_progress_to_stdout = true;
  // options.use_nonmonotonic_steps = true;
  if (marginalization_flag == MARGIN_OLD)
    options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  else
    options.max_solver_time_in_seconds = SOLVER_TIME;
  TicToc t_solver;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // cout << summary.BriefReport() << endl;
  ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
  // printf("solver costs: %f \n", t_solver.toc());

  double2vector();
  // printf("frame_count: %d \n", frame_count);

  if (frame_count < WINDOW_SIZE)
    return;

  TicToc t_whole_marginalization;
  if (marginalization_flag == MARGIN_OLD) {
    MarginalizationInfo *marginalization_info = new MarginalizationInfo();
    vector2double();

    if (last_marginalization_info && last_marginalization_info->valid) {
      vector<int> drop_set;
      for (int i = 0;
           i < static_cast<int>(last_marginalization_parameter_blocks.size());
           i++) {
        if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
            last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
          drop_set.push_back(i);
      }
      // construct new marginlization_factor
      MarginalizationFactor *marginalization_factor =
          new MarginalizationFactor(last_marginalization_info);
      ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
          marginalization_factor, NULL, last_marginalization_parameter_blocks,
          drop_set);
      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    if (USE_IMU) {
      if (pre_integrations[1]->sum_dt < 10.0) {
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            imu_factor, NULL, vector<double *>{para_Pose[0], para_SpeedBias[0],
                                               para_Pose[1], para_SpeedBias[1]},
            vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }
    }

    {
      int feature_index = -1;
      for (auto &it_per_id : f_manager.feature) {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
          continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        if (imu_i != 0)
          continue;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame) {
          imu_j++;
          if (imu_i != imu_j) {
            Vector3d pts_j = it_per_frame.point;
            ProjectionTwoFrameOneCamFactor *f_td =
                new ProjectionTwoFrameOneCamFactor(
                    pts_i, pts_j, it_per_id.feature_per_frame[0].velocity,
                    it_per_frame.velocity,
                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                f_td, loss_function,
                vector<double *>{para_Pose[imu_i], para_Pose[imu_j],
                                 para_Ex_Pose[0], para_Feature[feature_index],
                                 para_Td[0]},
                vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
          if (STEREO && it_per_frame.is_stereo) {
            Vector3d pts_j_right = it_per_frame.pointRight;
            if (imu_i != imu_j) {
              ProjectionTwoFrameTwoCamFactor *f =
                  new ProjectionTwoFrameTwoCamFactor(
                      pts_i, pts_j_right,
                      it_per_id.feature_per_frame[0].velocity,
                      it_per_frame.velocityRight,
                      it_per_id.feature_per_frame[0].cur_td,
                      it_per_frame.cur_td);
              ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                  f, loss_function,
                  vector<double *>{para_Pose[imu_i], para_Pose[imu_j],
                                   para_Ex_Pose[0], para_Ex_Pose[1],
                                   para_Feature[feature_index], para_Td[0]},
                  vector<int>{0, 4});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            } else {
              ProjectionOneFrameTwoCamFactor *f =
                  new ProjectionOneFrameTwoCamFactor(
                      pts_i, pts_j_right,
                      it_per_id.feature_per_frame[0].velocity,
                      it_per_frame.velocityRight,
                      it_per_id.feature_per_frame[0].cur_td,
                      it_per_frame.cur_td);
              ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                  f, loss_function,
                  vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1],
                                   para_Feature[feature_index], para_Td[0]},
                  vector<int>{2});
              marginalization_info->addResidualBlockInfo(residual_block_info);
            }
          }
        }
      }
    }

    TicToc t_pre_margin;
    marginalization_info->preMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

    TicToc t_margin;
    marginalization_info->marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());

    std::unordered_map<long, double *> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++) {
      addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
      if (USE_IMU)
        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
            para_SpeedBias[i - 1];
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
      addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

    addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

    vector<double *> parameter_blocks =
        marginalization_info->getParameterBlocks(addr_shift);

    if (last_marginalization_info)
      delete last_marginalization_info;
    last_marginalization_info = marginalization_info;
    last_marginalization_parameter_blocks = parameter_blocks;

  } else {
    if (last_marginalization_info &&
        std::count(std::begin(last_marginalization_parameter_blocks),
                   std::end(last_marginalization_parameter_blocks),
                   para_Pose[WINDOW_SIZE - 1])) {

      MarginalizationInfo *marginalization_info = new MarginalizationInfo();
      vector2double();
      if (last_marginalization_info && last_marginalization_info->valid) {
        vector<int> drop_set;
        for (int i = 0;
             i < static_cast<int>(last_marginalization_parameter_blocks.size());
             i++) {
          ROS_ASSERT(last_marginalization_parameter_blocks[i] !=
                     para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks[i] ==
              para_Pose[WINDOW_SIZE - 1])
            drop_set.push_back(i);
        }
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor =
            new MarginalizationFactor(last_marginalization_info);
        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
            marginalization_factor, NULL, last_marginalization_parameter_blocks,
            drop_set);

        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      TicToc t_pre_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->preMarginalize();
      ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

      TicToc t_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->marginalize();
      ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

      std::unordered_map<long, double *> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1)
          continue;
        else if (i == WINDOW_SIZE) {
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
          if (USE_IMU)
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
                para_SpeedBias[i - 1];
        } else {
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
          if (USE_IMU)
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] =
                para_SpeedBias[i];
        }
      }
      for (int i = 0; i < NUM_OF_CAM; i++)
        addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

      addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

      vector<double *> parameter_blocks =
          marginalization_info->getParameterBlocks(addr_shift);
      if (last_marginalization_info)
        delete last_marginalization_info;
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
    }
  }
  // printf("whole marginalization costs: %f \n",
  // t_whole_marginalization.toc());
  // printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow() {
  TicToc t_margin;
  if (marginalization_flag == MARGIN_OLD) {
    double t_0 = Headers[0];
    back_R0 = Rs[0];
    back_P0 = Ps[0];
    if (frame_count == WINDOW_SIZE) {
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Headers[i] = Headers[i + 1];
        Rs[i].swap(Rs[i + 1]);
        Ps[i].swap(Ps[i + 1]);
        if (USE_IMU) {
          std::swap(pre_integrations[i], pre_integrations[i + 1]);

          dt_buf[i].swap(dt_buf[i + 1]);
          linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
          angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

          Vs[i].swap(Vs[i + 1]);
          Bas[i].swap(Bas[i + 1]);
          Bgs[i].swap(Bgs[i + 1]);
        }
      }
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
      Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

      if (USE_IMU) {
        Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
        Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
        Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

        delete pre_integrations[WINDOW_SIZE];
        pre_integrations[WINDOW_SIZE] = new IntegrationBase{
            acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();
      }

      if (true || solver_flag == INITIAL) {
        map<double, ImageFrame>::iterator it_0;
        it_0 = all_image_frame.find(t_0);
        delete it_0->second.pre_integration;
        all_image_frame.erase(all_image_frame.begin(), it_0);
      }
      slideWindowOld();
    }
  } else {
    if (frame_count == WINDOW_SIZE) {
      Headers[frame_count - 1] = Headers[frame_count];
      Ps[frame_count - 1] = Ps[frame_count];
      Rs[frame_count - 1] = Rs[frame_count];

      if (USE_IMU) {
        for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
          double tmp_dt = dt_buf[frame_count][i];
          Vector3d tmp_linear_acceleration =
              linear_acceleration_buf[frame_count][i];
          Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

          pre_integrations[frame_count - 1]->push_back(
              tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

          dt_buf[frame_count - 1].push_back(tmp_dt);
          linear_acceleration_buf[frame_count - 1].push_back(
              tmp_linear_acceleration);
          angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
        }

        Vs[frame_count - 1] = Vs[frame_count];
        Bas[frame_count - 1] = Bas[frame_count];
        Bgs[frame_count - 1] = Bgs[frame_count];

        delete pre_integrations[WINDOW_SIZE];
        pre_integrations[WINDOW_SIZE] = new IntegrationBase{
            acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();
      }
      slideWindowNew();
    }
  }
}

void Estimator::slideWindowNew() {
  sum_of_front++;
  f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld() {
  sum_of_back++;

  bool shift_depth = solver_flag == NON_LINEAR ? true : false;
  if (shift_depth) {
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0 * ric[0];
    R1 = Rs[0] * ric[0];
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    f_manager.removeBackShiftDepth(R0, P0, R1, P1);
  } else
    f_manager.removeBack();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = Rs[frame_count];
  T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T) {
  T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = Rs[index];
  T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame() {
  // printf("predict pts in next frame\n");
  if (frame_count < 2)
    return;
  // predict next pose. Assume constant velocity motion
  Eigen::Matrix4d curT, prevT, nextT;
  getPoseInWorldFrame(curT);
  getPoseInWorldFrame(frame_count - 1, prevT);
  nextT = curT * (prevT.inverse() * curT);
  map<int, Eigen::Vector3d> predictPts;

  for (auto &it_per_id : f_manager.feature) {
    if (it_per_id.estimated_depth > 0) {
      int firstIndex = it_per_id.start_frame;
      int lastIndex =
          it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
      // printf("cur frame index  %d last frame index %d\n", frame_count,
      // lastIndex);
      if ((int)it_per_id.feature_per_frame.size() >= 2 &&
          lastIndex == frame_count) {
        double depth = it_per_id.estimated_depth;
        Vector3d pts_j =
            ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
        Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
        Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() *
                             (pts_w - nextT.block<3, 1>(0, 3));
        Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
        int ptsIndex = it_per_id.feature_id;
        predictPts[ptsIndex] = pts_cam;
      }
    }
  }
  featureTracker.setPrediction(predictPts);
  // printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici,
                                    Vector3d &tici, Matrix3d &Rj, Vector3d &Pj,
                                    Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi,
                                    Vector3d &uvj) {
  Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
  Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
  Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
  double rx = residual.x();
  double ry = residual.y();
  return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex) {
  // return;
  int feature_index = -1;
  for (auto &it_per_id : f_manager.feature) {
    double err = 0;
    int errCnt = 0;
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)
      continue;
    feature_index++;
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
    double depth = it_per_id.estimated_depth;
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;
      if (imu_i != imu_j) {
        Vector3d pts_j = it_per_frame.point;
        double tmp_error =
            reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j],
                              Ps[imu_j], ric[0], tic[0], depth, pts_i, pts_j);
        err += tmp_error;
        errCnt++;
        // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
      }
      // need to rewrite projecton factor.........
      if (STEREO && it_per_frame.is_stereo) {

        Vector3d pts_j_right = it_per_frame.pointRight;
        if (imu_i != imu_j) {
          double tmp_error = reprojectionError(
              Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
              ric[1], tic[1], depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
          // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
        } else {
          double tmp_error = reprojectionError(
              Rs[imu_i], Ps[imu_i], ric[0], tic[0], Rs[imu_j], Ps[imu_j],
              ric[1], tic[1], depth, pts_i, pts_j_right);
          err += tmp_error;
          errCnt++;
          // printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
        }
      }
    }
    double ave_err = err / errCnt;
    if (ave_err * FOCAL_LENGTH > 3)
      removeIndex.insert(it_per_id.feature_id);
  }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration,
                               Eigen::Vector3d angular_velocity) {
  double dt = t - latest_time;
  latest_time = t;
  Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
  Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
  latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
  Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
  latest_V = latest_V + dt * un_acc;
  latest_acc_0 = linear_acceleration;
  latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates() {
  mPropagate.lock();
  latest_time = Headers[frame_count] + td;
  latest_P = Ps[frame_count];
  latest_Q = Rs[frame_count];
  latest_V = Vs[frame_count];
  latest_Ba = Bas[frame_count];
  latest_Bg = Bgs[frame_count];
  latest_acc_0 = acc_0;
  latest_gyr_0 = gyr_0;
  mBuf.lock();
  queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
  queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
  mBuf.unlock();
  while (!tmp_accBuf.empty()) {
    double t = tmp_accBuf.front().first;
    Eigen::Vector3d acc = tmp_accBuf.front().second;
    Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
    fastPredictIMU(t, acc, gyr);
    tmp_accBuf.pop();
    tmp_gyrBuf.pop();
  }
  mPropagate.unlock();
}
