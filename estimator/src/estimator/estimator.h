/*******************************************************
 * Copyright (C) 2020, RAM-LAB, Hong Kong University of Science and Technology
 *
 * This file is part of M-LOAM (https://ram-lab.com/file/jjiao/m-loam).
 * If you use this code, please cite the respective publications as
 * listed on the above websites.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Jianhao JIAO (jiaojh1994@gmail.com)
 *******************************************************/

#pragma once

#include <thread>
#include <mutex>
#include <unordered_map>
#include <queue>

#include <omp.h>
#include <time.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <ceres/ceres.h>

#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/filters/extract_indices.h>

#include "common/common.hpp"
#include "common/color.hpp"
#include "common/types/type.h"
#include "common/random_generator.hpp"

#include "parameters.h"
#include "../imageSegmenter/image_segmenter.hpp"
#include "../featureExtract/feature_extract.hpp"
#include "../lidarTracker/lidar_tracker.h"
#include "../initial/initial_extrinsics.h"
#include "../utility/utility.h"
#include "../utility/cloud_visualizer.h"
#include "../utility/tic_toc.h"
#include "../utility/CircularBuffer.h"
#include "../factor/lidar_online_calib_factor.hpp"
#include "../factor/lidar_pure_odom_factor.hpp"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/prior_factor.hpp"
#include "../factor/impl_loss_function.hpp"
#include "mloam_pcl/point_with_time.hpp"

#define MAX_FEATURE_SELECT_TIME 7 // 7ms
#define MAX_RANDOM_QUEUE_TIME 10

class Estimator
{
  public:
    Estimator();
    ~Estimator();

    void clearState();
    void setParameter();

    void inputCloud(const double &t, const std::vector<common::PointCloud> &v_laser_cloud_in);
    void inputCloud(const double &t, const std::vector<common::PointITimeCloud> &v_laser_cloud_in);
    void inputCloud(const double &t, const common::PointCloud &laser_cloud_in);  //not defined

    // process measurements
    void processMeasurements();
    void undistortMeasurements(const std::vector<Pose> &pose_undist);
    void process();

    // build global map (for online calibration) and local map (for local optimization)
    void buildCalibMap();
    void buildLocalMap();

    // process localmap optimization
    void optimizeMap();

    // apply good feature
    void evaluateFeatJacobian(const Pose &pose_pivot,
                              const Pose &pose_i,
                              const Pose &pose_ext,
                              PointPlaneFeature &feature);
                              
    void goodFeatureMatching(const pcl::KdTreeFLANN<PointI>::Ptr &kdtree_from_map,
                             const PointICloud &laser_map,
                             const PointICloud &laser_cloud,
                             std::vector<PointPlaneFeature> &all_features,
                             std::vector<size_t> &sel_feature_idx,
                             const char feature_type,
                             const Pose &pose_pivot,
                             const Pose &pose_i,
                             const Pose &pose_ext,
                             const double &gf_ratio = 0.5);

    void vector2Double();
    void double2Vector();

    // slide window and marginalization
    void slideWindow();

    void evalResidual(ceres::Problem &problem,
                      std::vector<PoseLocalParameterization *> &local_param_ids,
                      const std::vector<double *> &para_ids,
                      const std::vector<ceres::internal::ResidualBlock *> &res_ids_proj,
                      const MarginalizationInfo *last_marginalization_info_,
                      const std::vector<ceres::internal::ResidualBlock *> &res_ids_marg);

    void evalDegenracy(std::vector<PoseLocalParameterization *> &local_param_ids,
                       const ceres::CRSMatrix &jaco);

    void evalCalib();

    void printParameter();
    void printSlideWindow();
    void visualizePCL();

    void changeSensorType(int use_imu, int use_stereo);


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    std::mutex m_process_;
    std::mutex m_buf_;

    std::thread track_thread_;
    std::thread process_thread_;

    omp_lock_t omp_lock_{};

    bool init_thread_flag_;

    SolverFlag solver_flag_;

    bool b_system_inited_{};

    Pose pose_laser_prev_;

    // pose from laser at k=0 to laser at k=K
    std::vector<Pose> pose_laser_cur_; //2个

    // pose from laser at k=K-1 to laser at k=K
    std::vector<Pose> pose_rlt_; //2个

    std::vector<Eigen::Quaterniond> qbl_; //body系(主雷达)到各雷达的外参, 大小为2，主雷达到主雷达为q=I,t=0
    std::vector<Eigen::Vector3d> tbl_;
    std::vector<double> tdbl_; //时间戳offset
    std::vector<Eigen::Matrix<double, 6, 6> > covbl_; //外参的cov

    // slide window
    // xx[cir_buf_cnt_] indicates the newest variables and measurements
    bool ini_fixed_local_map_{};

    size_t cir_buf_cnt_{};

    CircularBuffer<Eigen::Quaterniond> Qs_; //WINDOW_SIZE + 1， 主雷达每帧在odom下的位姿
    CircularBuffer<Eigen::Vector3d> Ts_;
    CircularBuffer<std_msgs::Header> Header_;

    std::vector<CircularBuffer<common::PointICloud> > surf_points_stack_, corner_points_stack_; 
    //2个，分别表示左右两个雷达; 每个对象 WINDOW_SIZE + 1大小，保存窗口中对应帧raw feature points(没有去畸变的)的less surf和less corner

    std::vector<CircularBuffer<int> > surf_points_stack_size_, corner_points_stack_size_;
    //2个，分别表示左右两个雷达；每个对象 WINDOW_SIZE + 1大小,保存窗口中对应帧less surf和less corner点的个数

    pcl::VoxelGrid<PointI> down_size_filter_corner_, down_size_filter_surf_;

    std::vector<common::PointICloud> surf_points_local_map_, surf_points_local_map_filtered_; //2个， 
    //surf_points_local_map_[n]: n号雷达在主雷达pivot下的local surf map， 把n号雷达在滑窗内的所有帧surf points都转换到主雷达pivot帧下

    std::vector<common::PointICloud> surf_points_pivot_map_;

    std::vector<common::PointICloud> corner_points_local_map_, corner_points_local_map_filtered_; //2个，
    //corner_points_local_map_[n]: n号雷达在主雷达pivot下的local corner map, 把n号雷达在滑窗内的所有帧corner points都转换到主雷达pivot帧下

    std::vector<common::PointICloud> corner_points_pivot_map_;

    std::vector<std::vector<Pose> > pose_local_; //2个， 每个对象 WINDOW_SIZE + 1大小

    double prev_time_{}, cur_time_{};
    double td_{};

    int frame_cnt_{};

    ImageSegmenter img_segment_;
    FeatureExtract f_extract_;
    LidarTracker lidar_tracker_;
    InitialExtrinsics initial_extrinsics_;

    std::queue<std::pair<double, std::vector<cloudFeature> > > feature_buf_; //每帧features

    pair<double, std::vector<cloudFeature> > prev_feature_, cur_feature_; //k, k+1帧左右雷达features

    std::vector<std::vector<std::vector<PointPlaneFeature> > > surf_map_features_, corner_map_features_;//2个，每个对象WINDOW_SIZE + 1大小
    //surf_map_features_[n][i]: “n号雷达在滑窗中i帧下surf points”在“n号雷达的local surf map”中的correspondances.这些features是在local map下的points
    //corner_map_features_[n][i]: 同理

    std::vector<std::vector<PointPlaneFeature> > cumu_surf_map_features_, cumu_corner_map_features_; //2个， 每个雷达在pivot帧下的points
    //cumu_surf_map_features_[n]: n号雷达在pivot帧下的surf points在n号雷达的local surf map中的correspondances.
    //cumu_corner_map_features_[n]: 同理

    size_t cumu_surf_feature_cnt_, cumu_corner_feature_cnt_;

    std::vector<std::vector<std::vector<size_t> > > sel_surf_feature_idx_, sel_corner_feature_idx_; //2个，每个对象WINDOW_SIZE + 1大小
    //ESTIMATE_EXTRINSIC == 0下使用
    //sel_surf_feature_idx_[n][i][j]: 挑选出n号雷达在i帧下的第j个好point在自己点云帧下的index放进 sel_surf_feature_idx_[n][i][j]
    //sel_corner_feature_idx_[n][i][j]: 同理



    double **para_pose_{}; //OPT_WINDOW_SIZE + 1, Xv。每个位姿的顺序是[tx, ty, tz, qx, qy, qz, qw]
    double **para_ex_pose_{}; //2个, Xe
    double *para_td_{}; //2个, time offset

    Eigen::VectorXd eig_thre_; //大小：OPT_WINDOW_SIZE + 1 + 2
    std::vector<double> log_lambda_;
    std::vector<Pose> log_extrinsics_;

    std::vector<double> d_factor_calib_; //2个
    std::vector<double> cur_eig_calib_;  //2个，没有使用
    std::vector<std::vector<std::pair<double, Pose> > > pose_calib_; //2个
    std::vector<bool> calib_converge_; //2个
    std::vector<size_t> num_residuals_;

    // for marginalization
    MarginalizationInfo *last_marginalization_info_{};
    vector<double *> last_marginalization_parameter_blocks_;

    PlaneNormalVisualizer plane_normal_vis_;

    std::vector<double> total_measurement_pre_time_, total_feat_matching_time_, 
      total_solver_time_, total_marginalization_time_, total_whole_odom_time_;
    int total_corner_feature_, total_surf_feature_;

    std::vector<nav_msgs::Path> v_laser_path_; //2个

    pcl::PCDWriter pcd_writer_;

    common::RandomGeneratorInt<size_t> rgi_;
};





//
