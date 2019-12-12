/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame() {
  return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[]) : Rs(_Rs) {
  for (int i = 0; i < NUM_OF_CAM; i++)
    ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[]) {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ric[i] = _ric[i];
  }
}

void FeatureManager::clearState() { feature.clear(); }

int FeatureManager::getFeatureCount() {
  int cnt = 0;
  for (auto &it : feature) {
    it.used_num = it.feature_per_frame.size();
    if (it.used_num >= 4) {
      cnt++;
    }
  }
  return cnt;
}

/**
 * @brief
 * 把特征点放入feature的list容器中，计算每一个点跟踪次数和它在次新帧和次次新帧间的视差，返回是否是关键帧
 * @param[in]   frame_count 窗口内帧的个数
 * @param[in]   image
 * 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   td IMU和cam同步时间差
 * @return  bool true：次新帧是关键帧;false：非关键帧
 * @cite feature会清除一些旧的数据吗？其是否也只是维护窗口中的数据
*/
bool FeatureManager::addFeatureCheckParallax(
    int frame_count,
    const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
    double td) {
  ROS_DEBUG("input feature: %d", (int)image.size());
  ROS_DEBUG("num of feature: %d", getFeatureCount());
  double parallax_sum = 0; // 总视差
  int parallax_num = 0;    // 计算视差特征点的数量
  last_track_num = 0;      // 跟踪特征点的数量
  last_average_parallax = 0;
  new_feature_num = 0; // 新特征的数量
  long_track_num = 0;  // 连续跟踪四帧特征点的数量

  // 把当前帧图像特征点数据image添加到feature容器中
  // feature容器按照特征点id组织特征点数据，对于每个id的特征点，记录它被滑动窗口中哪些图像帧观测到了
  for (auto &id_pts : image) {
    // 特征对象
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td);
    // id_pts.second[0].first表示camera_id，0即是第一个相机
    assert(id_pts.second[0].first == 0);
    // 如果是双目
    if (id_pts.second.size() == 2) {
      f_per_fra.rightObservation(id_pts.second[1].second);
      // 确保是第二个相机
      assert(id_pts.second[1].first == 1);
    }

    int feature_id = id_pts.first;
    /**
    * STL find_if的用法：
    * find_if (begin, end, func)
    * 就是从begin开始 ，到end为止，返回第一个让
    * func这个函数返回true的iterator
    */
    // 迭代器寻找feature list中是否有这feature_id
    auto it = find_if(feature.begin(), feature.end(),
                      [feature_id](const FeaturePerId &it) {
                        return it.feature_id == feature_id;
                      });
    // 如果返回尾部迭代器，说明该特征点第一次出现（新检测的特征点）
    // 因为使用的是光流法，通过光流法跟踪得到的特征点id不变
    // 如果是新提取的特征点，其id不同
    if (it == feature.end()) {
      // 将该特征点的id和当前的窗口的帧数（或则是其出现在窗口的第一帧ID)存储在FeaturePerId对象中
      feature.push_back(FeaturePerId(feature_id, frame_count));
      feature.back().feature_per_frame.push_back(f_per_fra);
      new_feature_num++;
    } else if (it->feature_id == feature_id) {
      // 将其添加到对应的feature_id的feature_per_frame容器中
      it->feature_per_frame.push_back(f_per_fra);
      last_track_num++;
      if (it->feature_per_frame.size() >= 4)
        long_track_num++;
    }
  }

  // if (frame_count < 2 || last_track_num < 20)
  // if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 *
  // last_track_num)
  // 1.滑动窗口的帧数少于2
  // 2.跟踪的特征点数量少于20个（那么下一帧提取到的新的特征点会比较多，旧的信息已经没用了）
  // 3.长时间跟踪的特征点少于40个（我认为有了这个条件可以不用考虑第二个条件）
  // 4.新的特征点数量大于跟踪特征点数量的一半
  if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 ||
      new_feature_num > 0.5 * last_track_num)
    return true;

  // 计算每个特征在次新帧和次次新帧的视差
  for (auto &it_per_id : feature) {
    // 1.当前窗口帧数比该特征点出现在窗口的首帧大于2
    // 2.该特征点出现在窗口的首帧与该特征点出现的次数之和大于当前的滑动窗口帧数
    // 这两个条件保证了该特征点在窗口多帧观测到，feature_per_frame中有足够多的数据
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >=
            frame_count - 1) {
      // 对于给定id的特征点，计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
      //（需要使用IMU数据补偿由于旋转造成的视差）
      parallax_sum += compensatedParallax2(it_per_id, frame_count);
      parallax_num++;
    }
  }

  if (parallax_num == 0) {
    // 如果视差点的数量为0（这个条件是不是太极端了），则边缘化旧的信息
    // 那就说明较少的特征点持续跟踪，不满足计算视差的前提条件
    return true;
  } else {
    ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum,
              parallax_num);
    ROS_DEBUG("current parallax: %lf",
              parallax_sum / parallax_num * FOCAL_LENGTH);
    // 平均每点的视差乘以焦距就是最新的平均视差，为什么要乘以焦距
    last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
    // 只要平均视差大于阈值，就边缘旧的信息，MIN_PARALLAX只有4+
    return parallax_sum / parallax_num >= MIN_PARALLAX;
  }
}

/**
 * @brief FeatureManager::getCorresponding
 *        计算滑动窗口某两帧之间关联的特征点
 * @param frame_count_l 表达不太合理
 * @param frame_count_r
 * @return
 */
vector<pair<Vector3d, Vector3d>>
FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
  vector<pair<Vector3d, Vector3d>> corres;
  for (auto &it : feature) {
    // 范围示意图：
    //         l                  r
    //         +                  +
    //    -                              -
    //  start                           end
    // 特征点在窗口的范围大于frame_count_l到frame_count_r之间的范围
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
      // frame_count_l与start_frame之间的帧数
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;

      // a是frame_count_l时该特征点对应的位置
      a = it.feature_per_frame[idx_l].point;

      // b是frame_count_r时该特征点对应的位置
      b = it.feature_per_frame[idx_r].point;

      corres.push_back(make_pair(a, b));
    }
  }
  return corres;
}

void FeatureManager::setDepth(const VectorXd &x) {
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)
      continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    // ROS_INFO("feature id %d , start_frame %d, depth %f ",
    // it_per_id->feature_id, it_per_id-> start_frame,
    // it_per_id->estimated_depth);
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = 2;
    } else
      it_per_id.solve_flag = 1;
  }
}

void FeatureManager::removeFailures() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == 2)
      feature.erase(it);
  }
}

void FeatureManager::clearDepth() {
  for (auto &it_per_id : feature)
    it_per_id.estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector() {
  VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)
      continue;
#if 1
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
    dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
  }
  return dep_vec;
}

void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                                      Eigen::Matrix<double, 3, 4> &Pose1,
                                      Eigen::Vector2d &point0,
                                      Eigen::Vector2d &point1,
                                      Eigen::Vector3d &point_3d) {
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Eigen::Vector4d triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P,
                                    vector<cv::Point2f> &pts2D,
                                    vector<cv::Point3f> &pts3D) {
  Eigen::Matrix3d R_initial;
  Eigen::Vector3d P_initial;

  // w_T_cam ---> cam_T_w
  R_initial = R.inverse();
  P_initial = -(R_initial * P);

  // printf("pnp size %d \n",(int)pts2D.size() );
  if (int(pts2D.size()) < 4) {
    printf("feature tracking not enough, please slowly move you device! \n");
    return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
  // pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 /
  // focalLength, 0.99, inliers);

  if (!pnp_succ) {
    printf("pnp failed ! \n");
    return false;
  }
  cv::Rodrigues(rvec, r);
  // cout << "r " << endl << r << endl;
  Eigen::MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  Eigen::MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);

  // cam_T_w ---> w_T_cam
  R = R_pnp.transpose();
  P = R * (-T_pnp);

  return true;
}

void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[],
                                        Matrix3d Rs[], Vector3d tic[],
                                        Matrix3d ric[]) {

  if (frameCnt > 0) {
    vector<cv::Point2f> pts2D;
    vector<cv::Point3f> pts3D;
    for (auto &it_per_id : feature) {
      if (it_per_id.estimated_depth > 0) {
        int index = frameCnt - it_per_id.start_frame;
        if ((int)it_per_id.feature_per_frame.size() >= index + 1) {
          Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point *
                                        it_per_id.estimated_depth) +
                              tic[0];
          Vector3d ptsInWorld =
              Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];

          cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
          cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(),
                              it_per_id.feature_per_frame[index].point.y());
          pts3D.push_back(point3d);
          pts2D.push_back(point2d);
        }
      }
    }
    Eigen::Matrix3d RCam;
    Eigen::Vector3d PCam;
    // trans to w_T_cam
    RCam = Rs[frameCnt - 1] * ric[0];
    PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];

    if (solvePoseByPnP(RCam, PCam, pts2D, pts3D)) {
      // trans to w_T_imu
      Rs[frameCnt] = RCam * ric[0].transpose();
      Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

      Eigen::Quaterniond Q(Rs[frameCnt]);
      // cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " <<
      // Q.vec().transpose() << endl;
      // cout << "frameCnt: " << frameCnt << " pnp P " <<
      // Ps[frameCnt].transpose() << endl;
    }
  }
}

/**
 * @brief FeatureManager::triangulate
 *        对所有的特征点三角化，求特征点的深度
 * @param frameCnt
 * @param Ps
 * @param Rs
 * @param tic
 * @param ric
 */
// NOTE:
void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[],
                                 Vector3d tic[], Matrix3d ric[]) {
  // 每个特征点的id
  for (auto &it_per_id : feature) {
    if (it_per_id.estimated_depth > 0)
      continue;

    if (STEREO && it_per_id.feature_per_frame[0].is_stereo) {
      int imu_i = it_per_id.start_frame;
      // NOTE: leftPose
      Eigen::Matrix<double, 3, 4> leftPose;
      // Ps[i]是帧i的imu坐标系到帧0的imu坐标系的位移，tic就是内参
      // t0为i帧相机到世界坐标系（0帧或者l帧）IMU的位移
      Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
      // R0为i帧相机相对参考系相机的旋转
      Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
      // leftPose是R0
      leftPose.leftCols<3>() = R0.transpose();
      leftPose.rightCols<1>() = -R0.transpose() * t0;
      // cout << "left pose " << leftPose << endl;

      Eigen::Matrix<double, 3, 4> rightPose;
      //
      Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
      Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
      rightPose.leftCols<3>() = R1.transpose();
      rightPose.rightCols<1>() = -R1.transpose() * t1;
      // cout << "right pose " << rightPose << endl;

      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;
      point0 = it_per_id.feature_per_frame[0].point.head(2);
      point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
      // cout << "point0 " << point0.transpose() << endl;
      // cout << "point1 " << point1.transpose() << endl;

      triangulatePoint(leftPose, rightPose, point0, point1, point3d);
      Eigen::Vector3d localPoint;
      localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
      double depth = localPoint.z();
      if (depth > 0)
        it_per_id.estimated_depth = depth;
      else
        it_per_id.estimated_depth = INIT_DEPTH;
      /*
      Vector3d ptsGt = pts_gt[it_per_id.feature_id];
      printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id,
      point3d.x(), point3d.y(), point3d.z(),
                                                      ptsGt.x(), ptsGt.y(),
      ptsGt.z());
      */
      continue;
    } else if (it_per_id.feature_per_frame.size() > 1) {
      // 需要有多帧观测到该特征点
      int imu_i = it_per_id.start_frame;
      Eigen::Matrix<double, 3, 4> leftPose;
      // Ps[i]是帧i的imu坐标系到帧0的imu坐标系的位移，tic就是内参
      // 可以发现Rs[i]已经变成帧i的imu坐标系相对于帧0相机坐标系的旋转
      // NOTE: Ps[i] Rs[i]还需要多研究一下
      Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
      Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
      leftPose.leftCols<3>() = R0.transpose();
      leftPose.rightCols<1>() = -R0.transpose() * t0;

      imu_i++;
      Eigen::Matrix<double, 3, 4> rightPose;
      Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
      Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
      rightPose.leftCols<3>() = R1.transpose();
      rightPose.rightCols<1>() = -R1.transpose() * t1;

      Eigen::Vector2d point0, point1;
      Eigen::Vector3d point3d;
      point0 = it_per_id.feature_per_frame[0].point.head(2);
      point1 = it_per_id.feature_per_frame[1].point.head(2);
      triangulatePoint(leftPose, rightPose, point0, point1, point3d);
      Eigen::Vector3d localPoint;
      localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
      double depth = localPoint.z();
      // INIT_DEPTH是什么鬼，一个设定的深度能准确吗
      if (depth > 0)
        it_per_id.estimated_depth = depth;
      else
        it_per_id.estimated_depth = INIT_DEPTH;
      /*
      Vector3d ptsGt = pts_gt[it_per_id.feature_id];
      printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id,
      point3d.x(), point3d.y(), point3d.z(),
                                                      ptsGt.x(), ptsGt.y(),
      ptsGt.z());
      */
      continue;
    }
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (it_per_id.used_num < 4)
      continue;

    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    Eigen::Matrix<double, 3, 4> P0;
    Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
    Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    // 对于观测到该id特征点的每一个图像帧
    // NOTE: 这里已经分不清Rs和t是相对与谁与谁之间的旋转和平移了
    for (auto &it_per_frame : it_per_id.feature_per_frame) {
      imu_j++;

      Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
      Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized();
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      // 这个continue有什么作用呢
      if (imu_i == imu_j)
        continue;
    }
    ROS_ASSERT(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V =
        Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV)
            .matrixV()
            .rightCols<1>();
    double svd_method = svd_V[2] / svd_V[3];
    // it_per_id->estimated_depth = -b / A;
    // it_per_id->estimated_depth = svd_V[2] / svd_V[3];

    it_per_id.estimated_depth = svd_method;
    // it_per_id->estimated_depth = INIT_DEPTH;

    if (it_per_id.estimated_depth < 0.1) {
      it_per_id.estimated_depth = INIT_DEPTH;
    }
  }
}

void FeatureManager::removeOutlier(set<int> &outlierIndex) {
  std::set<int>::iterator itSet;
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;
    int index = it->feature_id;
    itSet = outlierIndex.find(index);
    if (itSet != outlierIndex.end()) {
      feature.erase(it);
      // printf("remove outlier %d \n", index);
    }
  }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R,
                                          Eigen::Vector3d marg_P,
                                          Eigen::Matrix3d new_R,
                                          Eigen::Vector3d new_P) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() < 2) {
        feature.erase(it);
        continue;
      } else {
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth = dep_j;
        else
          it->estimated_depth = INIT_DEPTH;
      }
    }
    // remove tracking-lost feature after marginalize
    /*
    if (it->endFrame() < WINDOW_SIZE - 1)
    {
        feature.erase(it);
    }
    */
  }
}

void FeatureManager::removeBack() {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0)
        feature.erase(it);
    }
  }
}

void FeatureManager::removeFront(int frame_count) {
  for (auto it = feature.begin(), it_next = feature.begin();
       it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame == frame_count) {
      it->start_frame--;
    } else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1)
        continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0)
        feature.erase(it);
    }
  }
}

/**
 * @brief FeatureManager::compensatedParallax2
 * 计算某个特征点it_per_id在次新帧和次次新帧的视差
 * @param it_per_id
 * @param frame_count
 * @return
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id,
                                            int frame_count) {
  // check the second last frame is keyframe or not
  // parallax betwwen seconde last frame and third last frame
  // 第i帧为窗口的帧数减去特征点在窗口的起始帧再减去2
  // NOTE: frame_i会不会不存在
  const FeaturePerFrame &frame_i =
      it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame &frame_j =
      it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

  double ans = 0; // 初始化时差

  Vector3d p_j = frame_j.point;

  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point;
  Vector3d p_i_comp;

  // int r_i = frame_count - 2;
  // int r_j = frame_count - 1;
  // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] *
  // ric[camera_id_i] * p_i;
  p_i_comp = p_i;
  // p_i(2)不就是为1吗
  double dep_i = p_i(2);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  // 归一化平面坐标差值
  double du = u_i - u_j, dv = v_i - v_j;

  // du_comp和du有什么区别吗
  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  // 有什么意义
  ans = max(
      ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}
