/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "initial_sfm.h"

GlobalSFM::GlobalSFM() {}

/**
 * @brief GlobalSFM::triangulatePoint
 *        三角化，用最小二乘的方法，OpenCV有现成的函数实现
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @param point_3d
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                                 Eigen::Matrix<double, 3, 4> &Pose1,
                                 Vector2d &point0, Vector2d &point1,
                                 Vector3d &point_3d) {
  Matrix4d design_matrix = Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Vector4d triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}
/**
 * @brief GlobalSFM::solveFrameByPnP
 *        PNP方法得到第l帧到第i帧的R_initial、P_initial
 * @param R_initial
 * @param P_initial
 * @param i
 * @param sfm_f
 * @return
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                                vector<SFMFeature> &sfm_f) {
  vector<cv::Point2f> pts_2_vector; // 用于pnp求解的3d点
  vector<cv::Point3f> pts_3_vector; // 用于pnp求解位姿的2d点，也就是像素坐标
  for (int j = 0; j < feature_num; j++) {
    // 如果该特征点未被三角化，则跳过该特征点
    if (sfm_f[j].state != true)
      continue;
    Vector2d point2d;
    // 遍历观测到该特征点的图像帧
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
      // 从obserbation中找到该特征点的图像帧
      if (sfm_f[j].observation[k].first == i) {
        // 获取2d归一化坐标
        Vector2d img_pts = sfm_f[j].observation[k].second;
        cv::Point2f pts_2(img_pts(0), img_pts(1));
        pts_2_vector.push_back(pts_2);

        // 获取3d空间坐标
        cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1],
                          sfm_f[j].position[2]);
        pts_3_vector.push_back(pts_3);
        break;
      }
    }
  }
  if (int(pts_2_vector.size()) < 15) {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10)
      return false;
  }
  cv::Mat r, rvec, t, D, tmp_r;
  // 初始旋转
  cv::eigen2cv(R_initial, tmp_r);
  // Rodigues变换如果输入是旋转矩阵，输出则是旋转向量，如果输入是旋转向量，输出则是旋转矩阵
  // rvec是旋转向量
  cv::Rodrigues(tmp_r, rvec);
  // 初始的平移
  cv::eigen2cv(P_initial, t);
  // 由于2D坐标是归一化平面的坐标，所以内参使用单位矩阵
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  bool pnp_succ;
  // 其也没有畸变系数，输出为revec和t，分别为旋转向量和平移向量，1表示EPnP
  pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
  if (!pnp_succ) {
    return false;
  }
  // 再转换为旋转矩阵
  cv::Rodrigues(rvec, r);
  // cout << "r " << endl << r << endl;
  // 转换为Eigen类型
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;
}
/**
 * @brief GlobalSFM::triangulateTwoFrames
 *        三角化frame0和frame1之间的对应点
 * @param frame0
 * @param Pose0
 * @param frame1
 * @param Pose1
 * @param sfm_f
 */
void GlobalSFM::triangulateTwoFrames(int frame0,
                                     Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1,
                                     Eigen::Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_f) {
  assert(frame0 != frame1); // 确保不是同一帧
  // 遍历sfm_f中的特征点
  for (int j = 0; j < feature_num; j++) {
    // 如果已经被三角化，则跳过
    if (sfm_f[j].state == true)
      continue;
    bool has_0 = false, has_1 = false;
    Vector2d point0; // frame0中的2D点
    Vector2d point1; // frame1中的2D点
    // 遍历观测到该特征点的图像帧
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) {
      // 该特征点是否被frame0观测
      if (sfm_f[j].observation[k].first == frame0) {
        // 若观测到，则记录下来
        point0 = sfm_f[j].observation[k].second;
        has_0 = true;
      }
      // 该特征点是否被fram1观测
      if (sfm_f[j].observation[k].first == frame1) {
        // 若观测到，则记录下来
        point1 = sfm_f[j].observation[k].second;
        has_1 = true;
      }
    }
    // 同时观测到
    if (has_0 && has_1) {
      Vector3d point_3d;
      // 三角化
      triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
      // 设置标志位，表示已经被初始化
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
      // cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " <<
      // point_3d.transpose() << endl;
    }
  }
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
/**
 * @brief   纯视觉sfm，求解窗口中的所有图像帧的位姿和特征点坐标
 * @param[in]   frame_num	窗口总帧数（frame_count + 1）
 * @param[out]  q 	窗口内图像帧的旋转四元数q（相对于第l帧）
 * @param[out]	T 	窗口内图像帧的平移向量T（相对于第l帧）
 * @param[in]  	l 	第l帧
 * @param[in]  	relative_R	当前帧到第l帧的旋转矩阵
 * @param[in]  	relative_T 	当前帧到第l帧的平移向量
 * @param[in]  	sfm_f		所有特征点
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return  bool true:sfm求解成功
*/
bool GlobalSFM::construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                          const Matrix3d relative_R, const Vector3d relative_T,
                          vector<SFMFeature> &sfm_f,
                          map<int, Vector3d> &sfm_tracked_points) {
  feature_num = sfm_f.size();
  // cout << "set 0 and " << l << " as known " << endl;
  // have relative_r relative_t
  // intial two view
  //假设第l帧为原点，根据当前帧到第l帧的relative_R，relative_T，得到当前帧位姿
  q[l].w() = 1;
  q[l].x() = 0;
  q[l].y() = 0;
  q[l].z() = 0;
  T[l].setZero();
  // 当前帧相对于L帧的位姿
  q[frame_num - 1] = q[l] * Quaterniond(relative_R);
  T[frame_num - 1] = relative_T;
  // cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
  // cout << "init t_l " << T[l].transpose() << endl;

  // rotate to cam frame
  Matrix3d c_Rotation[frame_num];
  Vector3d c_Translation[frame_num];
  Quaterniond c_Quat[frame_num];
  double c_rotation[frame_num][4];
  double c_translation[frame_num][3];
  // 这里的pose表示的是第l帧到每一帧的变换矩阵
  Eigen::Matrix<double, 3, 4> Pose[frame_num];

  // 第l帧到第l帧的变换，有何意义
  c_Quat[l] = q[l].inverse();
  c_Rotation[l] = c_Quat[l].toRotationMatrix();
  c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
  Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
  Pose[l].block<3, 1>(0, 3) = c_Translation[l];

  // 第l帧到当前帧的位姿
  c_Quat[frame_num - 1] = q[frame_num - 1].inverse(); // 第l帧到当前帧的旋转矩阵
  c_Rotation[frame_num - 1] =
      c_Quat[frame_num - 1].toRotationMatrix(); // 第l帧到当前帧的位移
  c_Translation[frame_num - 1] =
      -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
  Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
  Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

  // 现在只有当前帧和第l帧的位姿是知道的
  // 1: trangulate between l ----- frame_num - 1
  // 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
  // 1、先三角化第l帧（参考帧）与第frame_num-1帧（当前帧）的路标点
  // 2、pnp求解从第l+1开始的每一帧到当前帧的变换矩阵R_initial,
  // P_initial，保存在Pose中,并与当前帧进行三角化
  for (int i = l; i < frame_num - 1; i++) {
    // 会先对第l帧和当前帧进行三角化
    // 对于l以后的帧在用PnP进行求解
    // solve pnp
    if (i > l) {
      // 使用上一帧的位姿作为初始的位姿，能够保证收敛
      Matrix3d R_initial = c_Rotation[i - 1];
      Vector3d P_initial = c_Translation[i - 1];
      // 其求解的是世界坐标系到各帧的位姿
      if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
        return false;
      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      // 四元数可以直接用旋转矩阵赋值
      c_Quat[i] = c_Rotation[i];
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result
    // 遍历l帧到第(frame_num-2)帧，寻找与第(frame_num-1)帧的匹配，并三角化特征点
    // 特征点都是以当前帧为参考的，那么其求解的位姿不也是相对于当前帧吗，也是应该还是相对于l帧的
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
  }
  // 3: triangulate l-----l+1 l+2 ... frame_num -2
  // 再对(l+1)帧到frame-2帧进行三角化，可以获得更多的地图点
  // 上面求解PnP后已经获得了第(l+1)帧到(frame_num-2)帧的位姿
  // 然后由以l帧为参考三角化特征点
  for (int i = l + 1; i < frame_num - 1; i++)
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
  // 4: solve pnp l-1; triangulate l-1 ----- l
  //             l-2              l-2 ----- l
  // 反向遍历0到l-1帧
  // 首先通过PnP求解相对于l帧的位姿，然后再三角化，获得更多的地图点
  for (int i = l - 1; i >= 0; i--) {
    // solve pnp
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
      return false;
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    c_Quat[i] = c_Rotation[i];
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    // triangulate
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
  }
  // 5: triangulate all other points
  // 三角化其他未三角化的特征点
  // 至此得到了滑动窗口中所有图像帧的位姿以及特征点的3d坐标
  for (int j = 0; j < feature_num; j++) {
    if (sfm_f[j].state == true)
      continue;
    if ((int)sfm_f[j].observation.size() >= 2) {
      Vector2d point0, point1;
      int frame_0 = sfm_f[j].observation[0].first;
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
      // cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point :
      // "  << j << "  " << point_3d.transpose() << endl;
    }
  }

  /*
          for (int i = 0; i < frame_num; i++)
          {
                  q[i] = c_Rotation[i].transpose();
                  cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  "
     << q[i].vec().transpose() << endl;
          }
          for (int i = 0; i < frame_num; i++)
          {
                  Vector3d t_tmp;
                  t_tmp = -1 * (q[i] * c_Translation[i]);
                  cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"
     "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
          }
  */
  // full BA
  // 使用ceres进行全局BA优化
  ceres::Problem problem;
  // LocalParameterization类的作用是解决非线性优化的过参数问题
  ceres::LocalParameterization *local_parameterization =
      new ceres::QuaternionParameterization();
  // cout << " begin full BA " << endl;
  // 优化的变量为滑动窗口内各帧的位姿c_translation,c_ratation
  for (int i = 0; i < frame_num; i++) {
    // double array for ceres
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    c_rotation[i][0] = c_Quat[i].w();
    c_rotation[i][1] = c_Quat[i].x();
    c_rotation[i][2] = c_Quat[i].y();
    c_rotation[i][3] = c_Quat[i].z();
    // 添加参数模块，对于四元数，其实际自由度为3，其会被重构
    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[i], 3);
    // 固定第一帧，这里把第l帧当做第一帧
    if (i == l) {
      problem.SetParameterBlockConstant(c_rotation[i]);
    }
    // 这是固定了第一帧到最后一帧之间的平移了吗
    if (i == l || i == frame_num - 1) {
      problem.SetParameterBlockConstant(c_translation[i]);
    }
  }

  // 遍历所有的特征点
  for (int i = 0; i < feature_num; i++) {
    // 如果特征点未被三角化则跳过
    if (sfm_f[i].state != true)
      continue;
    // 遍历观测到该特征点的帧
    for (int j = 0; j < int(sfm_f[i].observation.size()); j++) {
      int l = sfm_f[i].observation[j].first;
      // 定义重投影误差的代价函数
      // 这里使用了归一化平面的2d坐标
      ceres::CostFunction *cost_function =
          ReprojectionError3D::Create(sfm_f[i].observation[j].second.x(),
                                      sfm_f[i].observation[j].second.y());

      // 残差模块
      // 第二个参数为损失函数：用于处理参数中含有野值的情况，常用参数包括HuberLoss和CauchyLoss
      // 这里取NULL，其实应该改为nullptr，表示损失函数为单位函数
      // 后面3个参数都是优化变量，传入前面的cost_function中。这是一个自动求导的代价函数。
      problem.AddResidualBlock(cost_function, NULL, c_rotation[l],
                               c_translation[l], sfm_f[i].position);
    }
  }
  // 求解器配置
  ceres::Solver::Options options;
  // 稠密舒尔补
  options.linear_solver_type = ceres::DENSE_SCHUR;
  // options.minimizer_progress_to_stdout = true;
  //  options.num_threads = 2; //用于评估Jacobian的线程数，线程越多优化速度越快
  options.max_solver_time_in_seconds = 0.2;
  // 优化信息
  ceres::Solver::Summary summary;
  // 开始优化
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << "\n";
  if (summary.termination_type == ceres::CONVERGENCE ||
      summary.final_cost < 5e-03) {
    // cout << "vision only BA converge" << endl;
  } else {
    // cout << "vision only BA not converge " << endl;
    return false;
  }
  // 更新优化后的R和T，并且把位姿转化为参考为第l帧的
  for (int i = 0; i < frame_num; i++) {
    q[i].w() = c_rotation[i][0];
    q[i].x() = c_rotation[i][1];
    q[i].y() = c_rotation[i][2];
    q[i].z() = c_rotation[i][3];
    q[i] = q[i].inverse();
    // cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " <<
    // q[i].vec().transpose() << endl;
  }
  for (int i = 0; i < frame_num; i++) {

    T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1],
                                 c_translation[i][2]));
    // cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"
    // "<< T[i](2) << endl;
  }
  // 把三角化成功的点存储在sfm_tracked_points中
  for (int i = 0; i < (int)sfm_f.size(); i++) {
    if (sfm_f[i].state)
      sfm_tracked_points[sfm_f[i].id] = Vector3d(
          sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
  }
  return true;
}
