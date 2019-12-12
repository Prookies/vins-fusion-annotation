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

#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;

// 用于视觉初始化的图像特征点数据
struct SFMFeature {
  bool state;                              // 状态
  int id;                                  // id
  vector<pair<int, Vector2d>> observation; // 观测点（多帧）
  double position[3];                      // 空间位置点
  double depth;                            // 深度
};

/**
 * @brief The ReprojectionError3D struct
 *        其可以当做一个函数使用，其对()进行了重载
 */
struct ReprojectionError3D {
  ReprojectionError3D(double observed_u, double observed_v)
      : observed_u(observed_u), observed_v(observed_v) {}

  // 模板函数
  // camera_R表示旋转，camera_T表示平移，point表示特征点空间位置，这三个变量都是优化变量
  // 它们在AddResidualBlock是传入
  // residuals是误差输出
  template <typename T>
  bool operator()(const T *const camera_R, const T *const camera_T,
                  const T *point, T *residuals) const {
    T p[3];
    ceres::QuaternionRotatePoint(camera_R, point, p);
    p[0] += camera_T[0];
    p[1] += camera_T[1];
    p[2] += camera_T[2];
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];
    residuals[0] = xp - T(observed_u);
    residuals[1] = yp - T(observed_v);
    return true;
  }

  /**
   * @brief Create
   *        创建重投影误差代价函数
   * @param observed_x
   * @param observed_y
   * @return
   */
  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError3D, 2, 4, 3, 3>(
        new ReprojectionError3D(observed_x, observed_y)));
  }

  double observed_u;
  double observed_v;
};

class GlobalSFM {
public:
  GlobalSFM();
  bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                 const Matrix3d relative_R, const Vector3d relative_T,
                 vector<SFMFeature> &sfm_f,
                 map<int, Vector3d> &sfm_tracked_points);

private:
  bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                       vector<SFMFeature> &sfm_f);

  void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0,
                        Eigen::Matrix<double, 3, 4> &Pose1, Vector2d &point0,
                        Vector2d &point1, Vector3d &point_3d);
  void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                            int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                            vector<SFMFeature> &sfm_f);

  int feature_num;
};
