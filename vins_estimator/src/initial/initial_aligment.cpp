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

#include "initial_alignment.h"
/**
 * @brief   陀螺仪偏置校正
 * @abstract    根据视觉SFM的结果来校正陀螺仪Bias -> Paper V-B-1
 *              主要是将相邻帧之间SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐
 *              注意得到了新的Bias后对应的预积分需要repropagate
 * @param[in]
 * all_image_frame所有图像帧构成的map,图像帧保存了位姿、预积分量和关于角点的信息
 * @param[out]  Bgs 陀螺仪偏置
 * @return      void
*/
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame,
                        Vector3d *Bgs) {
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  // 遍历所有的图像帧
  // for循环会先判断条件，再执行代码块，next(frame_i)是保证fram_j不会指向end()
  // 循环范围
  //   frame_i                             frame_i       end()
  //      *     *     *     *     *     *     *     *     *
  //          fram_j                             frame_j
  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++) {
    // 迭代器的下一个或上一个分别用next和prev获取
    frame_j = next(frame_i);
    // J_bw
    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    // 用两帧之间的旋转和相机与IMU外参估计IMU的旋转，并不是实际IMU预积分的旋转
    // R_ij = (R^c0_bk)^-1 * (R^c0_bk+1) 转换为四元数 q_ij = (q^c0_bk)^-1 *
    // (q^c0_bk+1)
    Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    // IMU预积分旋转对角速度偏置的雅克比矩阵
    // NOTE: 不明白代码语法
    tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(
        O_R, O_BG);
    // IMU测量的两帧实际的旋转与通过相机位姿和外参计算得到的两帧之间IMU旋转的偏差
    tmp_b =
        2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
    // 构造最小二乘矩阵
    // tmp_A * delta_bg = tmp_b
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  // LDLT分解求解delta_bg
  // NOTE: LDLT这方法好吗
  delta_bg = A.ldlt().solve(b);
  ROS_WARN_STREAM("gyroscope bias initial calibration "
                  << delta_bg.transpose());

  // 这里对Bgs进行运算，更新滑动窗口中的随机游走项
  for (int i = 0; i <= WINDOW_SIZE; i++)
    Bgs[i] += delta_bg;

  // 使用更新后的随机游走项重新预积分（首帧除外，因为首帧前没有预积分数据）
  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
  }
}

//在半径为G的半球找到切面的一对正交基 -> Algorithm 1
MatrixXd TangentBasis(Vector3d &g0) {
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if (a == tmp)
    tmp << 1, 0, 0;
  // 和Algorithm1有所不同
  // 如果a = 0, 0, 1，那么tmp = 1, 0, 0, b = 1, 0, 0
  // NOTE: 感觉有点问题
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

/**
 * @brief       重力矢量细化
 * @abstract    重力细化，在其切线空间上用两个变量重新参数化重力 -> Paper V-B-3
 *              g^ = ||g|| * (g^-) + w1b1 + w2b2
 * @param[in]   all_image_frame
 *              所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x
 *              待优化变量，窗口中每帧的速度V[0:n]、二自由度重力参数w[w1,w2]^T、尺度s
 * @return      void
*/
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g,
                   VectorXd &x) {
  // norm()表示向量大小，也是L2距离
  // g0 = (g^-)*||g||
  Vector3d g0 = g.normalized() * G.norm();
  Vector3d lx, ly;
  // VectorXd x;
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  // 现在改为优化变量w了
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  // 迭代四次
  for (int k = 0; k < 4; k++) {
    // lxly = b = [b1, b2]
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin();
         next(frame_i) != all_image_frame.end(); frame_i++, i++) {
      frame_j = next(frame_i);

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;

      // tmp_A * x = tmp_b 求解最小二乘问题
      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 *
                                Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() *
                                (frame_j->second.T - frame_i->second.T) / 100.0;
      tmp_b.block<3, 1>(0, 0) =
          frame_j->second.pre_integration->delta_p +
          frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] -
          frame_i->second.R.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) =
          frame_i->second.R.transpose() * frame_j->second.R;
      tmp_A.block<3, 2>(3, 6) =
          frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) =
          frame_j->second.pre_integration->delta_v -
          frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
      // MatrixXd cov_inv = cov.inverse();
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    // dg = [w1, w2]^T
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
    // double s = x(n_state - 1);
  }
  g = g0;
}

/**
 * @brief       计算尺度，重力加速度和速度
 * @abstract    速度、重力向量和尺度初始化Paper -> V-B-2
 *              相邻帧之间的位置和速度与IMU预积分出来的位置和速度对齐，求解最小二乘
 *              重力细化 -> Paper V-B-3 重力方向与世界坐标系的轴对齐
 * @param[in]   all_image_frame
 *              所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、重力g、尺度s
 * @return      void
*/
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g,
                     VectorXd &x) {
  int all_frame_count = all_image_frame.size();
  // 优化变量x的总维度，包括每帧的速度3维，重力向量3维，尺度1维
  int n_state = all_frame_count * 3 + 3 + 1;

  // 建立相应大小的A矩阵
  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  // 参考 https://blog.csdn.net/qq_41839222/article/details/89106128
  for (frame_i = all_image_frame.begin();
       next(frame_i) != all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    // 两帧之间预积分的时间
    double dt = frame_j->second.pre_integration->sum_dt;

    // 论文公式（10）（11)
    // tmp_A * x = tmp_b 求解最小二乘问题
    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) =
        frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
    // NOTE: 为什么要除以100
    tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() *
                              (frame_j->second.T - frame_i->second.T) / 100.0;
    tmp_b.block<3, 1>(0, 0) =
        frame_j->second.pre_integration->delta_p +
        frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
    // cout << "delta_p   " <<
    // frame_j->second.pre_integration->delta_p.transpose() << endl;
    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    tmp_A.block<3, 3>(3, 6) =
        frame_i->second.R.transpose() * dt * Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
    // cout << "delta_v   " <<
    // frame_j->second.pre_integration->delta_v.transpose() << endl;

    // NOTE: 这个有什么作用
    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
    // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
    // MatrixXd cov_inv = cov.inverse();
    cov_inv.setIdentity();

    // 构建H矩阵
    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    // 构建b矩阵
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;
    // H.t()*H*X = H.t()*b

    // 将H矩阵有关速度的部分累计到A矩阵中
    // NOTE: 为什么跨度只有3，中间有重叠部分，不会影响结果吗
    // 应该不会有影响，应该就是相当于联立方程
    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  // NOTE: 矩阵的数值过大不是不好求解吗
  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);
  // 先前对尺度的方程乘以了100，现在再将其恢复
  double s = x(n_state - 1) / 100.0;
  ROS_DEBUG("estimated scale: %f", s);
  // 重力
  g = x.segment<3>(n_state - 4);
  ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
  if (fabs(g.norm() - G.norm()) > 0.5 || s < 0) {
    return false;
  }

  // 重力细化，在正切空间微调重力向量，同时优化尺度
  // 因为通过上面求解出来的重力并没有加入模长的限制
  RefineGravity(all_image_frame, g, x);
  // 重新计算了尺度
  // NOTE: 应该直接将其返回，而不是将其作为一个判别条件
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;
  ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
  if (s < 0.0)
    return false;
  else
    return true;
}

/**
 * @brief VisualIMUAlignment
 *        视觉和IMU对其
 * @param all_image_frame
 * @param Bgs
 * @param g
 * @param x
 * @return
 */
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs,
                        Vector3d &g, VectorXd &x) {
  // 计算陀螺仪偏置
  solveGyroscopeBias(all_image_frame, Bgs);

  if (LinearAlignment(all_image_frame, g, x))
    return true;
  else
    return false;
}
