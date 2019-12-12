/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g) {
  Eigen::Matrix3d R0;
  // g^ = ||g||*g + w_1*b_1 + w_2*b_2
  Eigen::Vector3d ng1 = g.normalized(); // 单位化，是表示g的方向向量
  Eigen::Vector3d ng2{0, 0, 1.0};       // 理想的旋转向量
  // R0表示ng1向量旋转到ng2向量的旋转矩阵，其就是b_1
  R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
  double yaw = Utility::R2ypr(R0).x(); // 返回R绕z轴的旋转角度
  // R0是从当前坐标系到世界坐标系的旋转矩阵（当前坐标系的z轴，与世界坐标系的重力g的方向对应）
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
  return R0;
}
