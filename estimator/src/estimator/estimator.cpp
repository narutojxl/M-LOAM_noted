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

#include "estimator.h"
#include "../utility/visualization.h"

using namespace common;

Estimator::Estimator()
{
    ROS_INFO("init begins");
    init_thread_flag_ = false;
    clearState();
}

Estimator::~Estimator()
{
    delete[] para_pose_;
    delete[] para_td_;
    delete[] para_ex_pose_;
    if (MULTIPLE_THREAD)
    {
        process_thread_.join();
        printf("join thread \n");
    }
}

void Estimator::setParameter()
{
    m_process_.lock();

    pose_rlt_.resize(NUM_OF_LASER);
    pose_laser_cur_.resize(NUM_OF_LASER);
    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        pose_rlt_[i] = Pose();
        pose_laser_cur_[i] = Pose();
    }

    qbl_.resize(NUM_OF_LASER);
    tbl_.resize(NUM_OF_LASER);
    tdbl_.resize(NUM_OF_LASER);
    covbl_.resize(NUM_OF_LASER);
    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        qbl_[i] = QBL[i]; //QBL, TBL, TDBL, COV_EXT: read from config file
        tbl_[i] = TBL[i];
        tdbl_[i] = TDBL[i];
        covbl_[i] = COV_EXT[i];
        cout << "Given extrinsic Laser_" << i << ": " << Pose(QBL[i], TBL[i], TDBL[i]) << endl;
    }

    initial_extrinsics_.setParameter();

    Qs_.resize(WINDOW_SIZE + 1);
    Ts_.resize(WINDOW_SIZE + 1);
    Header_.resize(WINDOW_SIZE + 1);
    surf_points_stack_.resize(NUM_OF_LASER);
    surf_points_stack_size_.resize(NUM_OF_LASER);
    corner_points_stack_.resize(NUM_OF_LASER);
    corner_points_stack_size_.resize(NUM_OF_LASER);

    down_size_filter_surf_.setLeafSize(0.4, 0.4, 0.4);
    down_size_filter_corner_.setLeafSize(0.2, 0.2, 0.2);

    pose_local_.resize(NUM_OF_LASER);
    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        surf_points_stack_[i].resize(WINDOW_SIZE + 1);
        surf_points_stack_size_[i].resize(WINDOW_SIZE + 1);
        corner_points_stack_[i].resize(WINDOW_SIZE + 1);
        corner_points_stack_size_[i].resize(WINDOW_SIZE + 1);
        pose_local_[i].resize(WINDOW_SIZE + 1);
    }

    surf_points_local_map_.resize(NUM_OF_LASER);
    surf_points_local_map_filtered_.resize(NUM_OF_LASER);
    surf_points_pivot_map_.resize(NUM_OF_LASER);
    corner_points_local_map_.resize(NUM_OF_LASER);
    corner_points_local_map_filtered_.resize(NUM_OF_LASER);
    corner_points_pivot_map_.resize(NUM_OF_LASER);

    cumu_surf_map_features_.resize(NUM_OF_LASER);
    cumu_corner_map_features_.resize(NUM_OF_LASER);

    printf("MULTIPLE_THREAD is %d\n", MULTIPLE_THREAD);
    if (MULTIPLE_THREAD && !init_thread_flag_)
    {
        init_thread_flag_ = true;
        process_thread_ = std::thread(&Estimator::processMeasurements, this);
    }

    para_pose_ = new double *[OPT_WINDOW_SIZE + 1];
    for (size_t i = 0; i < OPT_WINDOW_SIZE + 1; i++)
    {
        para_pose_[i] = new double[SIZE_POSE];
    }
    para_ex_pose_ = new double *[NUM_OF_LASER];
    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        para_ex_pose_[i] = new double[SIZE_POSE];
    }
    para_td_ = new double[NUM_OF_LASER];

    eig_thre_ = Eigen::VectorXd::Constant(OPT_WINDOW_SIZE + 1 + NUM_OF_LASER, 1, LAMBDA_INITIAL); //初值：100
    eig_thre_.block(OPT_WINDOW_SIZE + 1, 0, 1, NUM_OF_LASER) = Eigen::VectorXd::Zero(NUM_OF_LASER); //[100 100 100 0 0]
    //TODO(jxl): block(OPT_WINDOW_SIZE + 1, 0, NUM_OF_LASER, 1)

    d_factor_calib_ = std::vector<double>(NUM_OF_LASER, 0);
    cur_eig_calib_ = std::vector<double>(NUM_OF_LASER, 0);
    pose_calib_.resize(NUM_OF_LASER);
    calib_converge_.resize(NUM_OF_LASER, false);

    img_segment_.setParameter(N_SCANS, HORIZON_SCAN, MIN_CLUSTER_SIZE, SEGMENT_VALID_POINT_NUM, SEGMENT_VALID_LINE_NUM);
    v_laser_path_.resize(NUM_OF_LASER);

    m_process_.unlock();
}

void Estimator::clearState()
{
    printf("[estimator] clear state\n");
    m_process_.lock();

    b_system_inited_ = false;

    prev_time_ = -1;
    cur_time_ = 0;
    frame_cnt_ = 0;

    td_ = 0;

    solver_flag_ = INITIAL;

    pose_rlt_.clear();
    pose_laser_cur_.clear();

    qbl_.clear();
    tbl_.clear();
    tdbl_.clear();
    covbl_.clear();

    initial_extrinsics_.clearState();

    ini_fixed_local_map_ = false;

    cir_buf_cnt_ = 0;

    Qs_.clear();
    Ts_.clear();
    Header_.clear();
    surf_points_stack_.clear();
    surf_points_stack_size_.clear();
    corner_points_stack_.clear();
    corner_points_stack_size_.clear();

    surf_points_local_map_.clear();
    surf_points_local_map_filtered_.clear();
    surf_points_pivot_map_.clear();
    corner_points_local_map_.clear();
    corner_points_local_map_filtered_.clear();
    corner_points_pivot_map_.clear();

    surf_map_features_.clear();
    corner_map_features_.clear();

    cumu_surf_map_features_.clear();
    cumu_corner_map_features_.clear();
    cumu_surf_feature_cnt_ = 0;
    cumu_corner_feature_cnt_ = 0;

    pose_local_.clear();

    last_marginalization_info_ = nullptr;

    d_factor_calib_.clear();
    cur_eig_calib_.clear();
    pose_calib_.clear();
    calib_converge_.clear();

    total_measurement_pre_time_.clear();
    total_feat_matching_time_.clear();   
    total_solver_time_.clear(); 
    total_marginalization_time_.clear(); 
    total_whole_odom_time_.clear();

    total_corner_feature_ = 0;
    total_surf_feature_ = 0;

    m_process_.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    m_process_.lock();
    m_process_.unlock();
    if (restart)
    {
        clearState();
        setParameter();
    }
}


void Estimator::inputCloud(const double &t, const std::vector<PointCloud> &v_laser_cloud_in)
{
    assert(v_laser_cloud_in.size() == NUM_OF_LASER);
 
    common::timing::Timer mea_pre_timer("odom_mea_pre");
    std::vector<cloudFeature> feature_frame(NUM_OF_LASER);

    if (NUM_OF_LASER == 1)
    {
        PointICloud laser_cloud;
        f_extract_.calTimestamp(v_laser_cloud_in[0], laser_cloud);

        PointICloud laser_cloud_segment, laser_cloud_outlier;
        ScanInfo scan_info(N_SCANS, SEGMENT_CLOUD);
        if (ESTIMATE_EXTRINSIC != 0) scan_info.segment_flag_ = false;
        img_segment_.segmentCloud(laser_cloud, laser_cloud_segment, laser_cloud_outlier, scan_info);

        f_extract_.extractCloud(laser_cloud_segment, scan_info, feature_frame[0]);
        feature_frame[0].insert(pair<std::string, PointICloud>("laser_cloud_outlier", laser_cloud_outlier));
        total_corner_feature_ += feature_frame[0]["corner_points_less_sharp"].size();
        total_surf_feature_ += feature_frame[0]["surf_points_less_flat"].size();

        // PointICloud laser_cloud_segment, laser_cloud_outlier;
        // ScanInfo scan_info(N_SCANS, SEGMENT_CLOUD);
        // if (ESTIMATE_EXTRINSIC != 0) scan_info.segment_flag_ = false;
        // img_segment_.segmentCloud(laser_cloud, laser_cloud_segment, laser_cloud_outlier, scan_info);

        // f_extract_.extractCloud_aloam(laser_cloud, scan_info, feature_frame[0]);
        // laser_cloud_outlier.push_back(laser_cloud[0]);
        // feature_frame[0].insert(pair<std::string, PointICloud>("laser_cloud_outlier", laser_cloud_outlier));
        // total_corner_feature_ += feature_frame[0]["corner_points_less_sharp"].size();
        // total_surf_feature_ += feature_frame[0]["surf_points_less_flat"].size();        
    } 
    else 
    {
        std::vector<cloudFeature *> feature_frame_ptr(NUM_OF_LASER);
        #pragma omp parallel for num_threads(NUM_OF_LASER)
        for (size_t i = 0; i < v_laser_cloud_in.size(); i++)
        {
            PointICloud laser_cloud;
            f_extract_.calTimestamp(v_laser_cloud_in[i], laser_cloud); //laser_cloud：每个点的强度是在一帧中的时间比例

            PointICloud laser_cloud_segment, laser_cloud_outlier;
            ScanInfo scan_info(N_SCANS, SEGMENT_CLOUD); //16*1
            if (ESTIMATE_EXTRINSIC != 0) scan_info.segment_flag_ = false; //TODO(jxl):当需要对外参提纯或者估计外参时，不移除没有聚类的点
            img_segment_.segmentCloud(laser_cloud, laser_cloud_segment, laser_cloud_outlier, scan_info);
            //laser_cloud_outlier: 没有形成聚类的points
            //对点云进行聚类，把没有聚类的点移除； 点的强度为：线号(最底下线束为0，最上面线束最大)+时间比例

            feature_frame_ptr[i] = new cloudFeature;
            f_extract_.extractCloud(laser_cloud_segment, scan_info, *feature_frame_ptr[i]);
            //依次对一帧的scan提取corner sharp, less corner sharp, surf flat, less surf flat

            feature_frame_ptr[i]->insert(pair<std::string, PointICloud>("laser_cloud_outlier", laser_cloud_outlier));
        }

        for (size_t i = 0; i < NUM_OF_LASER; i++) 
        {
            feature_frame[i] = *feature_frame_ptr[i];
            total_corner_feature_ += feature_frame[i]["corner_points_less_sharp"].size();
            total_surf_feature_ += feature_frame[i]["surf_points_less_flat"].size();
        }
        for (auto &frame_ptr : feature_frame_ptr) delete frame_ptr;    
    }

    double mea_pre_time = mea_pre_timer.Stop();
    // printf("meaPre time: %fms (%lu*%fms)\n", mea_pre_time * 1000, v_laser_cloud_in.size(), 
    //                                          mea_pre_time * 1000 / v_laser_cloud_in.size());
    m_buf_.lock();
    feature_buf_.push(make_pair(t, feature_frame)); //把每帧的features压入到队列中
    m_buf_.unlock();
    if (!MULTIPLE_THREAD) processMeasurements();
}

void Estimator::inputCloud(const double &t, const std::vector<PointITimeCloud> &v_laser_cloud_in)
{
    assert(v_laser_cloud_in.size() == NUM_OF_LASER);

    common::timing::Timer mea_pre_timer("odom_mea_pre");
    std::vector<cloudFeature> feature_frame(NUM_OF_LASER);

    if (NUM_OF_LASER == 1)
    {
        PointICloud laser_cloud;
        f_extract_.calTimestamp(v_laser_cloud_in[0], laser_cloud);

        PointICloud laser_cloud_segment, laser_cloud_outlier;
        ScanInfo scan_info(N_SCANS, SEGMENT_CLOUD);
        if (ESTIMATE_EXTRINSIC != 0) scan_info.segment_flag_ = false;
        img_segment_.segmentCloud(laser_cloud, laser_cloud_segment, laser_cloud_outlier, scan_info);

        f_extract_.extractCloud(laser_cloud_segment, scan_info, feature_frame[0]);
        feature_frame[0].insert(pair<std::string, PointICloud>("laser_cloud_outlier", laser_cloud_outlier));
        total_corner_feature_ += feature_frame[0]["corner_points_less_sharp"].size();
        total_surf_feature_ += feature_frame[0]["surf_points_less_flat"].size();
    } 
    else
    {
        std::vector<cloudFeature *> feature_frame_ptr(NUM_OF_LASER);
        #pragma omp parallel for num_threads(NUM_OF_LASER)
        for (size_t i = 0; i < v_laser_cloud_in.size(); i++)
        {
            PointICloud laser_cloud;
            f_extract_.calTimestamp(v_laser_cloud_in[i], laser_cloud);

            PointICloud laser_cloud_segment, laser_cloud_outlier;
            ScanInfo scan_info(N_SCANS, SEGMENT_CLOUD);
            if (ESTIMATE_EXTRINSIC != 0) scan_info.segment_flag_ = false;
            img_segment_.segmentCloud(laser_cloud, laser_cloud_segment, laser_cloud_outlier, scan_info);

            feature_frame_ptr[i] = new cloudFeature;
            f_extract_.extractCloud(laser_cloud_segment, scan_info, *feature_frame_ptr[i]);
            feature_frame_ptr[i]->insert(pair<std::string, PointICloud>("laser_cloud_outlier", laser_cloud_outlier));
        }

        for (size_t i = 0; i < NUM_OF_LASER; i++)
        {
            feature_frame[i] = *feature_frame_ptr[i];
            total_corner_feature_ += feature_frame[i]["corner_points_less_sharp"].size();
            total_surf_feature_ += feature_frame[i]["surf_points_less_flat"].size();
        }
        for (auto &frame_ptr : feature_frame_ptr) delete frame_ptr;
    }

    double mea_pre_time = mea_pre_timer.Stop();
    // printf("meaPre time: %fms (%lu*%fms)\n", mea_pre_time * 1000, v_laser_cloud_in.size(), 
    //                                          mea_pre_time * 1000 / v_laser_cloud_in.size());

    m_buf_.lock();
    feature_buf_.push(make_pair(t, feature_frame));
    m_buf_.unlock();
    if (!MULTIPLE_THREAD) processMeasurements();
}

void Estimator::processMeasurements()
{
    while (1)
    {
        if (!feature_buf_.empty())
        {
            cur_feature_ = feature_buf_.front();
            cur_time_ = cur_feature_.first + td_; //td_ = 0
            assert(cur_feature_.second.size() == NUM_OF_LASER);

            m_buf_.lock();
            feature_buf_.pop(); //处理一帧，pop一次
            m_buf_.unlock();

            m_process_.lock();
            common::timing::Timer odom_process_timer("odom_process");

            process(); //前端里程计模块

            double time_process = odom_process_timer.Stop() * 1000;
            std::cout << common::RED << "frame: " << frame_cnt_
                      << ", odom process time: " << time_process << "ms" << common::RESET << std::endl << std::endl;
            LOG_EVERY_N(INFO, 20) << "odom process time: " << time_process << "ms";

            // printStatistics(*this, 0);
            pubOdometry(*this, cur_time_);
            if (frame_cnt_ % SKIP_NUM_ODOM_PUB == 0) pubPointCloud(*this, cur_time_); 
            frame_cnt_++;
            m_process_.unlock();
        }
        if (!MULTIPLE_THREAD) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void Estimator::undistortMeasurements(const std::vector<Pose> &pose_undist)
{
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        if (ESTIMATE_EXTRINSIC == 2) // initialization
        {
            // for (PointI &point : cur_feature_.second[n]["corner_points_sharp"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
            // for (PointI &point : cur_feature_.second[n]["surf_points_flat"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
            for (PointI &point : cur_feature_.second[n]["corner_points_less_sharp"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
            for (PointI &point : cur_feature_.second[n]["surf_points_less_flat"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
			for (PointI &point : cur_feature_.second[n]["laser_cloud"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
        } else
        // if (ESTIMATE_EXTRINSIC == 1) // online calibration
        // {
        //     if (n != IDX_REF) continue;
        //     // for (PointI &point : cur_feature_.second[n]["corner_points_sharp"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
        //     // for (PointI &point : cur_feature_.second[n]["surf_points_flat"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
        //     for (PointI &point : cur_feature_.second[n]["corner_points_less_sharp"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
        //     for (PointI &point : cur_feature_.second[n]["surf_points_less_flat"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
		// 	for (PointI &point : cur_feature_.second[n]["laser_cloud"]) TransformToEnd(point, point, pose_undist[n], true, SCAN_PERIOD);
        // } else
        // if (ESTIMATE_EXTRINSIC == 0) // pure odometry with accurate extrinsics
        {
            // Pose pose_ext(qbl_[n], tbl_[n]);
            // Pose pose_undist = pose_ext.inverse() * pose_rlt_[IDX_REF] * pose_ext;
            // for (PointI &point : cur_feature_.second[n]["corner_points_sharp"]) TransformToEnd(point, point, pose_undist, true, SCAN_PERIOD);
            // for (PointI &point : cur_feature_.second[n]["surf_points_flat"]) TransformToEnd(point, point, pose_undist, true, SCAN_PERIOD);

            //把当前帧的feature points转换到当前帧的end下  //TODO(jxl): pose_undist[n]，作者还没测试 https://github.com/gogojjh/M-LOAM/issues/6
            for (PointI &point : cur_feature_.second[n]["corner_points_less_sharp"]) TransformToEnd(point, point, pose_undist[IDX_REF], true, SCAN_PERIOD);
            for (PointI &point : cur_feature_.second[n]["surf_points_less_flat"]) TransformToEnd(point, point, pose_undist[IDX_REF], true, SCAN_PERIOD);
			for (PointI &point : cur_feature_.second[n]["laser_cloud"]) TransformToEnd(point, point, pose_undist[IDX_REF], true, SCAN_PERIOD);
        }
    }
}

void Estimator::process()
{
    if (!b_system_inited_) //第一帧scan
    {
        b_system_inited_ = true;
        // printf("System initialization finished \n");
    } else 
    {
        common::timing::Timer tracker_timer("odom_tracker");
        // -----------------
        // tracker and initialization
        if (ESTIMATE_EXTRINSIC == 2) //外参的初值还没有计算
        {
            #pragma omp parallel for num_threads(NUM_OF_LASER)
            for (size_t n = 0; n < NUM_OF_LASER; n++)
            {
                cloudFeature &cur_cloud_feature = cur_feature_.second[n];
                cloudFeature &prev_cloud_feature = prev_feature_.second[n];
                pose_rlt_[n] = lidar_tracker_.trackCloud(prev_cloud_feature, cur_cloud_feature, pose_rlt_[n]); 
                //在n雷达之前相邻两帧delta_T基础上，用n雷达curr和prev相邻两帧scan(点到平面，点到直线)匹配，计算delta_T

                pose_laser_cur_[n] = pose_laser_cur_[n] * pose_rlt_[n];
            }
            // printf("lidarTracker: %fms\n", tracker_timer.Stop() * 1000);
            // for (size_t n = 0; n < NUM_OF_LASER; n++)
            //     std::cout << "laser_" << n << ", pose_rlt: " << pose_rlt_[n] << std::endl;

            // initialize extrinsics
            printf("calibrating extrinsic param, sufficient movement is needed\n");
            if (initial_extrinsics_.addPose(pose_rlt_) && (cir_buf_cnt_ == WINDOW_SIZE)) //标定的时候需要机器人做“螺丝”运动，给予充分激励
            {
                // TicToc t_calib_ext;
                for (size_t n = 0; n < NUM_OF_LASER; n++)
                {
                    if (initial_extrinsics_.cov_rot_state_[n]) continue; //忽略主雷达，n=0时
                    Pose calib_result;
                    if (initial_extrinsics_.calibExRotation(IDX_REF, n, calib_result)) //IDX_REF=0
                    {
                        if (initial_extrinsics_.calibExTranslation(IDX_REF, n, calib_result))
                        {
                            std::cout << common::YELLOW << "Initial extrinsic of laser_" << n << ": " << calib_result 
                                      << common::RESET << std::endl;
                            qbl_[n] = calib_result.q_;
                            tbl_[n] = calib_result.t_;
                            // tdbl_[n] = calib_result.td_;
                            QBL[n] = calib_result.q_;
                            TBL[n] = calib_result.t_;
                            // TDBL[n] = calib_result.td_;
                        }
                    }
                }
                if ((initial_extrinsics_.full_cov_rot_state_) && (initial_extrinsics_.full_cov_pos_state_))
                {
                    std::cout << common::YELLOW << "All initial extrinsic rotation calib success" << common::RESET << std::endl;
                    ESTIMATE_EXTRINSIC = 1; //标定状态由2转为1，进入refine阶段; 后面对外参refine收敛结束后，转为 ESTIMATE_EXTRINSIC = 0，见evalCalib()
                    initial_extrinsics_.saveStatistics();
                }
                // LOG_EVERY_N(INFO, 20) << "initialize extrinsics: " << t_calib_ext.toc() << "ms";
                // printf("initialize extrinsics: %fms\n", t_calib_ext.toc());
            }
        }
        else if (ESTIMATE_EXTRINSIC != 2)
        {
            cloudFeature &cur_cloud_feature = cur_feature_.second[IDX_REF]; //k+1帧主雷达features，当前帧的points还是在当前帧各个时刻下采集的points
            cloudFeature &prev_cloud_feature = prev_feature_.second[IDX_REF]; //k帧主雷达features，在上一个周期末尾已经转换到了k帧end下
            pose_rlt_[IDX_REF] = lidar_tracker_.trackCloud(prev_cloud_feature, cur_cloud_feature, pose_rlt_[IDX_REF]);
            //在之前主雷达相邻两scan的delta_T初值基础上，用主雷达当前帧scan和主雷达上一帧scan，计算delta_T，作为返回值返回
            //没有用副雷达的feature points

            pose_laser_cur_[IDX_REF] = Pose(Qs_[cir_buf_cnt_ - 1], Ts_[cir_buf_cnt_ - 1]) * pose_rlt_[IDX_REF];
            //里程计位姿累积

            // std::cout << "pose_rlt: " << pose_rlt_[IDX_REF] << std::endl;
            // LOG_EVERY_N(INFO, 20) << "lidarTracker: " << t_mloam_tracker.toc() << "ms";
            printf("lidarTracker: %fms\n", tracker_timer.Stop() * 1000);
        }
    }

    //----------------- update pose and point cloud
    //xx[cir_buf_cnt_] indicates the newest variables and measurements
    //cir_buf_cnt_ 初值=0
    Qs_[cir_buf_cnt_] = pose_laser_cur_[IDX_REF].q_;
    Ts_[cir_buf_cnt_] = pose_laser_cur_[IDX_REF].t_;
    Header_[cir_buf_cnt_].stamp = ros::Time(cur_feature_.first);
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        PointICloud &corner_points = cur_feature_.second[n]["corner_points_less_sharp"];
        down_size_filter_corner_.setInputCloud(boost::make_shared<PointICloud>(corner_points));
        down_size_filter_corner_.filter(corner_points_stack_[n][cir_buf_cnt_]); //raw curr feature points(没有畸变的)
        corner_points_stack_size_[n][cir_buf_cnt_] = corner_points_stack_[n][cir_buf_cnt_].size();

        PointICloud &surf_points = cur_feature_.second[n]["surf_points_less_flat"];
        down_size_filter_surf_.setInputCloud(boost::make_shared<PointICloud>(surf_points));
        down_size_filter_surf_.filter(surf_points_stack_[n][cir_buf_cnt_]); //raw curr feature points(没有畸变的)
        surf_points_stack_size_[n][cir_buf_cnt_] = surf_points_stack_[n][cir_buf_cnt_].size();
    }
    // printSlideWindow();

    switch (solver_flag_)
    {
        // INITIAL: multi-LiDAR individual tracking
        case INITIAL:
        {
            printf("[INITIAL]\n");
            slideWindow();
            if (cir_buf_cnt_ < WINDOW_SIZE)
            {
                cir_buf_cnt_++;
                if (cir_buf_cnt_ == WINDOW_SIZE) //执行第4帧scan时，WINDOW_SIZE=4，slideWindow()执行了两次
                {
                    slideWindow(); 
                }
            }
            if ((cir_buf_cnt_ == WINDOW_SIZE) && (ESTIMATE_EXTRINSIC != 2))
            {
                solver_flag_ = NON_LINEAR;
            }
            break;
        }
        // NON_LINEAR: single LiDAR tracking and perform scan-to-map constrains
        case NON_LINEAR:
        {
            // local optimization: optimize the relative LiDAR measurments
            printf("[NON_LINEAR]\n");
            if (LM_OPT_ENABLE) optimizeMap(); 

            slideWindow();

            if (ESTIMATE_EXTRINSIC) evalCalib();

            break;
        }
    }

    ROS_WARN("cir_buf_cnt_ = %d, Qs_.size()=%d", cir_buf_cnt_, Qs_.size());

    // pass cur_feature to prev_feature
    prev_time_ = cur_time_;
    prev_feature_.first = prev_time_;
    prev_feature_.second.clear();
    prev_feature_.second.resize(NUM_OF_LASER);
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        prev_feature_.second[n].insert(make_pair("corner_points_less_sharp", 
            cur_feature_.second[n].find("corner_points_less_sharp")->second));
        prev_feature_.second[n].insert(make_pair("surf_points_less_flat", 
            cur_feature_.second[n].find("surf_points_less_flat")->second));
    }

    if (DISTORTION)
    {
        Pose pose_laser_cur = Pose(Qs_[cir_buf_cnt_ - 1], Ts_[cir_buf_cnt_ - 1]);
        std::vector<Pose> pose_undist = pose_rlt_;
        pose_undist[IDX_REF] = pose_laser_prev_.inverse() * pose_laser_cur;

        // {
        //     Pose pose_ext;
        //     Pose pose_local;
    
        //     ofstream fpose("/tmp/pose_rlt.txt");
        //     fpose << "rlt_0: " << pose_rlt_[IDX_REF] << std::endl;
        //     pose_ext = Pose(qbl_[1], tbl_[1]);
        //     pose_local = pose_ext.inverse() * pose_rlt_[IDX_REF] * pose_ext;
        //     fpose << "rlt_1: " << pose_local << std::endl;
    
        //     fpose << "update rlt_0: " << pose_rlt_[IDX_REF] << std::endl;
        //     pose_ext = Pose(qbl_[1], tbl_[1]);
        //     pose_local = pose_ext.inverse() * pose_rlt_[IDX_REF] * pose_ext;
        //     fpose << "update rlt_1: " << pose_local << std::endl;
        //     fpose.close();
        // }

        // for (size_t n = 0; n < NUM_OF_LASER; n++)
        // {
        //     stringstream ss;
        //     ss << "/tmp/raw_pc_" << n << ".pcd";
        //     pcl::io::savePCDFileASCII(ss.str(), cur_feature_.second[n]["laser_cloud"]);
        // }

        for (size_t n = 0; n < NUM_OF_LASER; n++)
        {
            Pose pose_ext(qbl_[n], tbl_[n]);
            pose_undist[n] = pose_ext.inverse() * pose_rlt_[IDX_REF] * pose_ext;
        }
        undistortMeasurements(pose_undist);  
        //pose_undist[0], [1]: 主雷达k帧到k+1帧(相邻两帧laser)的delta_T, 副雷达k帧到k+1帧的delta_T
        //把当前帧的feature points转换到当前帧的end下
        

        // for (size_t n = 0; n < NUM_OF_LASER; n++)
        // {
        //     stringstream ss;
        //     ss << "/tmp/undistort_raw_pc_" << n << ".pcd";
        //     pcl::io::savePCDFileASCII(ss.str(), cur_feature_.second[n]["laser_cloud"]);
        // }

        pose_laser_prev_ = pose_laser_cur;
        //TODO(jxl): 为何不在对当前帧feature points去畸变后，再和prev feature points交换
        //作者设计deskewed points不用来计算odometry，用来计算mapping. https://github.com/gogojjh/M-LOAM/issues/6
    }
}

void Estimator::optimizeMap()
{
    int pivot_idx = WINDOW_SIZE - OPT_WINDOW_SIZE; //2

    ceres::Problem problem;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    // loss_function = new ceres::GemanMcClureLoss(1.0);
    loss_function = new ceres::HuberLoss(1.0);
    // loss_function = new ceres::CauchyLoss(1.0);
    // ceres: set options and solve the non-linear equation
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // options.num_threads = 1;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.gradient_check_relative_precision = 1e-4;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    options.max_num_iterations = NUM_ITERATIONS;
    options.max_solver_time_in_seconds = SOLVER_TIME;

    vector2Double(); //给Xv，Xe赋值

    // ceres: add parameter block
    std::vector<double *> para_ids;
    std::vector<PoseLocalParameterization *> local_param_ids;
    for (size_t i = 0; i < OPT_WINDOW_SIZE + 1; i++)
    {
        PoseLocalParameterization *local_parameterization = new PoseLocalParameterization();
        local_parameterization->setParameter();
        problem.AddParameterBlock(para_pose_[i], SIZE_POSE, local_parameterization);
        local_param_ids.push_back(local_parameterization);
        para_ids.push_back(para_pose_[i]);
    }
    problem.SetParameterBlockConstant(para_pose_[0]); //主雷达在poivot_idx + 1 处为固定值

    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        PoseLocalParameterization *local_parameterization = new PoseLocalParameterization();
        local_parameterization->setParameter();
        problem.AddParameterBlock(para_ex_pose_[i], SIZE_POSE, local_parameterization);
        local_param_ids.push_back(local_parameterization);
        para_ids.push_back(para_ex_pose_[i]);
        if (ESTIMATE_EXTRINSIC == 0) problem.SetParameterBlockConstant(para_ex_pose_[i]);
    }
    problem.SetParameterBlockConstant(para_ex_pose_[IDX_REF]); //主雷达到主雷达的外参为const value

    // for (size_t i = 0; i < NUM_OF_LASER; i++)
    // {
    //     problem.AddParameterBlock(&para_td_[i], 1);
    //     para_ids.push_back(&para_td_[i]);
    //     if (!ESTIMATE_TD)
    //     {
    //         problem.SetParameterBlockConstant(&para_td_[i]);
    //     }
    // }
    // problem.SetParameterBlockConstant(&para_td_[IDX_REF]);

    
    // ceres: add the prior residual into future optimization
    std::vector<ceres::internal::ResidualBlock *> res_ids_marg;
    if ((MARGINALIZATION_FACTOR) && (last_marginalization_info_))
    {
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
        ceres::internal::ResidualBlock *res_id_marg = problem.AddResidualBlock(marginalization_factor,
                                                                               NULL,
                                                                               last_marginalization_parameter_blocks_);
        res_ids_marg.push_back(res_id_marg);
    }

    // ceres: add residual block within the sliding window
    std::vector<ceres::internal::ResidualBlock *> res_ids_proj;
    if (ESTIMATE_EXTRINSIC == 1)
    {
        buildCalibMap(); //1. 构建n号雷达在主雷达pivot下的local surf, corner map; 
                         //2. 构建“n号雷达在滑窗中每一帧下points” 在 “n号雷达的local surf, corner map”中的correspondances
        //主雷达local map:   主雷达滑窗内所有帧转到主雷达pivot帧下形成的local map
        //副雷达m local map：副雷达m滑窗内所有帧转到主雷达pivot帧下形成的local map
        //副雷达n local map：副雷达n滑窗内所有帧转到主雷达pivot帧下形成的local map
        std::cout << common::YELLOW << "optimization with online calibration" << common::RESET << std::endl;
        
        if (PRIOR_FACTOR) //外参变量(q,t)与外参初值变量(q0,t0)之间的残差，[t-t0, 2(q0.inv * q)]或者 [t-t0, Log({R0}^-1 * R)]
        {
            for (size_t n = 0; n < NUM_OF_LASER; n++)
            {
                PriorFactor *f = new PriorFactor(tbl_[n], qbl_[n], PRIOR_FACTOR_POS, PRIOR_FACTOR_ROT);
                ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f,
                                                                                  NULL,
                                                                                  para_ex_pose_[n]);
                res_ids_proj.push_back(res_id);
            }
        }

        if (POINT_PLANE_FACTOR)
        {
            CHECK_JACOBIAN = 0;  
            for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)//i =3,4
            {
                std::vector<PointPlaneFeature> &features_frame = surf_map_features_[IDX_REF][i];
                for (const PointPlaneFeature &feature : features_frame)
                {
                    LidarPureOdomPlaneNormFactor *f = new LidarPureOdomPlaneNormFactor(
                                         feature.point_,  //主雷达在滑窗中对应帧下的surf point
                                         feature.coeffs_, //该点在主雷达pivot下的local surf map中的correspondances形成的平面方程
                                         1.0);
                    
                    ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f,
                                                                                      loss_function,
                                                                                      para_pose_[0], //主雷达pivot pose, Xv[0]
                                                                                      para_pose_[i - pivot_idx], //主雷达依次在Xv[]中除了pivot帧pose
                                                                                      para_ex_pose_[IDX_REF]); //主雷达到主雷达的外参，const value
                    res_ids_proj.push_back(res_id); //对应论文中pure odometry：见笔记红色约束
                    if (CHECK_JACOBIAN)
                    {
                        double **tmp_param = new double *[3];
                        tmp_param[0] = para_pose_[0];
                        tmp_param[1] = para_pose_[i - pivot_idx];
                        tmp_param[2] = para_ex_pose_[IDX_REF];
                        f->check(tmp_param);
                        CHECK_JACOBIAN = 0;
                    }
                }
            }

            for (size_t n = 0; n < NUM_OF_LASER; n++) 
            {
                if (n == IDX_REF) continue; //忽略主雷达
                cumu_surf_map_features_[n].insert(cumu_surf_map_features_[n].end(),
                                                  surf_map_features_[n][pivot_idx].begin(), 
                                                  surf_map_features_[n][pivot_idx].end());
            }
            if (frame_cnt_ % N_CUMU_FEATURE == 0)
            {
                std::cout << common::YELLOW << "Start Calibration !" << common::RESET << std::endl;
                for (size_t n = 0; n < NUM_OF_LASER; n++)
                {
                    if (n == IDX_REF) continue; //忽略主雷达
                    for (const PointPlaneFeature &feature : cumu_surf_map_features_[n])
                    {
                        LidarOnlineCalibPlaneNormFactor *f = new LidarOnlineCalibPlaneNormFactor(
                                            feature.point_,  //n雷达在pivot帧下的点 
                                            feature.coeffs_, //n雷达在pivot帧下的点,在自己local map下的correspondances形成的平面方程；
                                            1.0);
                        ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f,
                                                                                          loss_function,
                                                                                          para_ex_pose_[n]); //主雷达到每个副雷达的外参
                        res_ids_proj.push_back(res_id);
                    }
                }
                if (!MARGINALIZATION_FACTOR)
                {
                    cumu_surf_map_features_.clear();
                    cumu_surf_map_features_.resize(NUM_OF_LASER);
                }
            }
        }

        if (POINT_EDGE_FACTOR)
        {
            for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
            {
                std::vector<PointPlaneFeature> &features_frame = corner_map_features_[IDX_REF][i];
                for (const PointPlaneFeature &feature : features_frame)
                {
                    LidarPureOdomEdgeFactor *f = new LidarPureOdomEdgeFactor(feature.point_, feature.coeffs_, 1.0);
                    // ceres::CostFunction *f = LidarPureOdomEdgeFactor::Create(feature.point_, feature.coeffs_, 1.0);
                    ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f,
                                                                                      loss_function,
                                                                                      para_pose_[0], //主雷达pivot pose, Xv[0]
                                                                                      para_pose_[i - pivot_idx], //主雷达依次在Xv[]中除了pivot帧pose
                                                                                      para_ex_pose_[IDX_REF]); //主雷达到主雷达的外参，const value
                    res_ids_proj.push_back(res_id);
                }
            }            

            for (size_t n = 0; n < NUM_OF_LASER; n++) 
            {
                if (n == IDX_REF) continue; //忽略主雷达
                cumu_corner_map_features_[n].insert(cumu_corner_map_features_[n].end(),
                                                    corner_map_features_[n][pivot_idx].begin(), 
                                                    corner_map_features_[n][pivot_idx].end());
            }
            if (frame_cnt_ % N_CUMU_FEATURE == 0)
            {
                for (size_t n = 0; n < NUM_OF_LASER; n++)
                {
                    if (n == IDX_REF) continue; //忽略主雷达
                    for (const PointPlaneFeature &feature : cumu_corner_map_features_[n])
                    {
                        LidarOnlineCalibEdgeFactor *f = new LidarOnlineCalibEdgeFactor(
                            feature.point_,  //n雷达在pivot帧下的点 
                            feature.coeffs_, //n雷达在pivot帧下的点,在自己local map下的correspondances
                            1.0);
                        ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f, loss_function, para_ex_pose_[n]); //主雷达到每个副雷达的外参
                        res_ids_proj.push_back(res_id);
                    }
                }
                if (!MARGINALIZATION_FACTOR)
                {
                    cumu_corner_map_features_.clear();
                    cumu_corner_map_features_.resize(NUM_OF_LASER);
                }
            }
        }
    }
    else if (ESTIMATE_EXTRINSIC == 0)
    {
        buildLocalMap();
        std::cout << common::YELLOW << "optimization with pure odometry" << common::RESET << std::endl;

        if (POINT_PLANE_FACTOR)
        {
            for (size_t n = 0; n < NUM_OF_LASER; n++)
            {
                for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
                {
                    for (const size_t &fid : sel_surf_feature_idx_[n][i])
                    {
                        const PointPlaneFeature &feature = surf_map_features_[n][i][fid];
                        // if (feature.type_ == 'n') continue;
                        LidarPureOdomPlaneNormFactor *f = new LidarPureOdomPlaneNormFactor(feature.point_, feature.coeffs_, 1.0);
                        ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f,
                                                                                          loss_function,
                                                                                          para_pose_[0], //主雷达pivot pose, Xv[0]
                                                                                          para_pose_[i - pivot_idx], //主雷达依次在Xv[]中除了pivot帧pose
                                                                                          para_ex_pose_[n]); //主雷达到每个副雷达的外参
                        res_ids_proj.push_back(res_id);
                    }
                }
            }
        }

        CHECK_JACOBIAN = 0; //default: 0
        
        ROS_WARN("TEST JACOBIAN");
        if (POINT_EDGE_FACTOR)
        {
            for (size_t n = 0; n < NUM_OF_LASER; n++)
            {
                for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
                {
                    for (const size_t &fid : sel_corner_feature_idx_[n][i])
                    {
                        const PointPlaneFeature &feature = corner_map_features_[n][i][fid];
                        // if (feature.type_ == 'n') continue;
                        LidarPureOdomEdgeFactor *f = new LidarPureOdomEdgeFactor(feature.point_, feature.coeffs_, 1.0);
                        // ceres::CostFunction *f = LidarPureOdomEdgeFactor::Create(feature.point_, feature.coeffs_, 1.0);
                        ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f,
                                                                                          loss_function,
                                                                                          para_pose_[0],
                                                                                          para_pose_[i - pivot_idx],
                                                                                          para_ex_pose_[n]);
                        res_ids_proj.push_back(res_id);
                        if (CHECK_JACOBIAN && frame_cnt_== 100) //TODO(jxl): 测试雅克比计算是否正确
                        {
                            double **tmp_param = new double *[3];
                            tmp_param[0] = para_pose_[0];
                            tmp_param[1] = para_pose_[i - pivot_idx];
                            tmp_param[2] = para_ex_pose_[n];
                            f->check(tmp_param);
                            CHECK_JACOBIAN = 0;
                        }
                    }
                }
            }
        }
        ROS_WARN("TEST JACOBIAN DONE");

    }
    
    common::timing::Timer eval_deg_timer("odom_eval_residual");
    evalResidual(problem,
                 local_param_ids, //PoseLocalParameterization
                 para_ids, //Xv[], double raw pointer
                 res_ids_proj, //all laser残差块
                 last_marginalization_info_, //上一次边缘化信息
                 res_ids_marg); //上一次边缘化残差块
    printf("evaluate residual: %fms\n", eval_deg_timer.Stop() * 1000);

    common::timing::Timer solver_timer("odom_solver");
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    // std::cout << summary.FullReport() << std::endl;
    printf("ceres solver costs: %fms\n", solver_timer.Stop() * 1000);

    double2Vector();

    // **************************************************** marginalization
    // ceres: marginalization of current parameter block
    // prepare all the residuals, jacobians, and dropped parameter blocks to construct marginalization prior 
    if (MARGINALIZATION_FACTOR)
    {
        common::timing::Timer marg_timer("odom_marg");
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2Double();

        // indicate the prior error
        if (last_marginalization_info_) //跟lio-mapping, lili-om一样
        {
            std::vector<int> drop_set;
            for (size_t i = 0; i < static_cast<int>(last_marginalization_parameter_blocks_.size()); i++)
            {
                // indicate the dropped pose to calculate the related residuals
                if (last_marginalization_parameter_blocks_[i] == para_pose_[0]) drop_set.push_back(i);
            }
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info_);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, 
                                                                           NULL,
                                                                           last_marginalization_parameter_blocks_, 
                                                                           drop_set); //!@第一类
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (PRIOR_FACTOR) //外参
        {
            for (size_t n = 0; n < NUM_OF_LASER; n++)
            {
                PriorFactor *f = new PriorFactor(tbl_[n], qbl_[n], PRIOR_FACTOR_POS, PRIOR_FACTOR_ROT);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, 
                                                                               NULL,
                                                                               std::vector<double *>{para_ex_pose_[n]}, //主雷达到n雷达的外参
                                                                               std::vector<int>{}); //!@第二类
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        if (ESTIMATE_EXTRINSIC == 1)
        {
            if (POINT_PLANE_FACTOR)
            {
                for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
                {
                    std::vector<PointPlaneFeature> &features_frame = surf_map_features_[IDX_REF][i];
                    for (const PointPlaneFeature &feature: features_frame)
                    {
                        LidarPureOdomPlaneNormFactor *f = new LidarPureOdomPlaneNormFactor(feature.point_, feature.coeffs_, 1.0);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f,
                                                                                       loss_function,
                                                                                       std::vector<double *>{para_pose_[0], //主雷达pivot pose, Xv[0]
                                                                                                             para_pose_[i - pivot_idx], //主雷达依次在Xv[]中除了pivot帧pose
                                                                                                             para_ex_pose_[IDX_REF]}, //主雷达到主雷达的外参
                                                                                       std::vector<int>{0}); //!@第三类                      
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }

                if (frame_cnt_ % N_CUMU_FEATURE == 0)
                {
                    for (size_t n = 0; n < NUM_OF_LASER; n++)
                    {
                        if (n == IDX_REF) continue;
                        for (const PointPlaneFeature &feature : cumu_surf_map_features_[n])
                        {
                            LidarOnlineCalibPlaneNormFactor *f = new LidarOnlineCalibPlaneNormFactor(feature.point_, feature.coeffs_, 1.0);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f,
                                                                                           loss_function,
                                                                                           std::vector<double *>{para_ex_pose_[n]}, //主雷达到每个副雷达的外参
                                                                                           std::vector<int>{}); //!@第四类  
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                    cumu_surf_map_features_.clear();
                    cumu_surf_map_features_.resize(NUM_OF_LASER);
                }
            }

            if (POINT_EDGE_FACTOR)
            {
                for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
                {
                    std::vector<PointPlaneFeature> &features_frame = corner_map_features_[IDX_REF][i];
                    for (const PointPlaneFeature &feature: features_frame)
                    {
                        // if (feature.type_ == 'n') continue;
                        LidarPureOdomEdgeFactor *f = new LidarPureOdomEdgeFactor(feature.point_, feature.coeffs_, 1.0);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f,
                                                                                       loss_function,
                                                                                       std::vector<double *>{para_pose_[0], //主雷达pivot pose, Xv[0]
                                                                                                             para_pose_[i - pivot_idx], //主雷达依次在Xv[]中除了pivot帧pose
                                                                                                             para_ex_pose_[IDX_REF]}, //主雷达到主雷达的外参
                                                                                       std::vector<int>{0}); //!@第三类 
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }                
                
                if (frame_cnt_ % N_CUMU_FEATURE == 0)
                {
                    for (size_t n = 0; n < NUM_OF_LASER; n++)
                    {
                        if (n == IDX_REF) continue;
                        for (const PointPlaneFeature &feature : cumu_corner_map_features_[n])
                        {
                            LidarOnlineCalibEdgeFactor *f = new LidarOnlineCalibEdgeFactor(feature.point_, feature.coeffs_, 1.0);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f,
                                                                                           loss_function,
                                                                                           std::vector<double *>{para_ex_pose_[n]}, //主雷达到每个副雷达的外参
                                                                                           std::vector<int>{}); //!@第四类 
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                    cumu_corner_map_features_.clear();
                    cumu_corner_map_features_.resize(NUM_OF_LASER);
                }
            }
        }
        else if (ESTIMATE_EXTRINSIC == 0)
        {
            if (POINT_PLANE_FACTOR)
            {
                for (size_t n = 0; n < NUM_OF_LASER; n++)
                {
                    for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
                    {
                        for (const size_t &fid : sel_surf_feature_idx_[n][i])
                        {
                            const PointPlaneFeature &feature = surf_map_features_[n][i][fid];
                            // if (feature.type_ == 'n') continue;
                            LidarPureOdomPlaneNormFactor *f = new LidarPureOdomPlaneNormFactor(feature.point_, feature.coeffs_, 1.0);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f,
                                                                                           loss_function,
                                                                                           vector<double *>{para_pose_[0], //主雷达pivot pose, Xv[0]
                                                                                                            para_pose_[i - pivot_idx], //主雷达依次在Xv[]中除了pivot帧pose
                                                                                                            para_ex_pose_[n]}, //主雷达到每个副雷达的外参
                                                                                           std::vector<int>{0}); //!@第三类
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
            if (POINT_EDGE_FACTOR)
            {
                for (size_t n = 0; n < NUM_OF_LASER; n++)
                {
                    for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
                    {
                        for (const size_t &fid : sel_corner_feature_idx_[n][i])
                        {
                            const PointPlaneFeature &feature = corner_map_features_[n][i][fid];
                            if (feature.type_ == 'n') continue;
                            LidarPureOdomEdgeFactor *f = new LidarPureOdomEdgeFactor(feature.point_, feature.coeffs_, 1.0);
                            // ceres::CostFunction *f = LidarPureOdomEdgeFactor::Create(feature.point_, feature.coeffs_, 1.0);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f,
                                                                                           loss_function,
                                                                                           vector<double *>{para_pose_[0], //主雷达pivot pose, Xv[0]
                                                                                                            para_pose_[i - pivot_idx], //主雷达依次在Xv[]中除了pivot帧pose
                                                                                                            para_ex_pose_[n]}, //主雷达到每个副雷达的外参
                                                                                           std::vector<int>{0}); //!@第三类
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        //! calculate the residuals and jacobian of all ResidualBlockInfo over the marginalized parameter blocks,
        //! for next iteration, the linearization posize_t is assured and fixed
        //! adjust the memory of H and b to implement the Schur complement
       
        marginalization_info->preMarginalize(); // add parameter block given residual info
        

        // marginalize some states and keep the remaining states with prior residuals
        marginalization_info->marginalize(); // compute linear residuals and jacobian
        

        //! indicate shared memory of parameter blocks except for the dropped state
        std::unordered_map<long, double *> addr_shift;
        for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++)
        {
            addr_shift[reinterpret_cast<long>(para_pose_[i - pivot_idx])] = para_pose_[i - pivot_idx - 1];
        }
        for (size_t n = 0; n < NUM_OF_LASER; n++)
        {
            addr_shift[reinterpret_cast<long>(para_ex_pose_[n])] = para_ex_pose_[n];
        }
        // for (size_t n = 0; n < NUM_OF_LASER; n++)
        // {
        //     addr_shift[reinterpret_cast<long>(&para_td_[n])] = &para_td_[n];
        // }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
        if (last_marginalization_info_)
        {
            delete last_marginalization_info_;
        }
        last_marginalization_info_ = marginalization_info;
        last_marginalization_parameter_blocks_ = parameter_blocks; // save parameter_blocks at the last optimization
        printf("whole marginalization costs: %fms\n", marg_timer.Stop() * 1000);
    }
}

//ESTIMATE_EXTRINSIC == 1
void Estimator::buildCalibMap()
{
    common::timing::Timer build_map_timer("odom_build_calib_map");
    int pivot_idx = WINDOW_SIZE - OPT_WINDOW_SIZE;
    Pose pose_pivot(Qs_[pivot_idx], Ts_[pivot_idx]); //pivot pose: Xv[0] //TODO(jxl): 好像跟作者论文中pivot的位置不一样
    // build the whole local map using all poses except the newest pose
    surf_points_local_map_.clear();
    surf_points_local_map_.resize(NUM_OF_LASER);
    surf_points_local_map_filtered_.clear();
    surf_points_local_map_filtered_.resize(NUM_OF_LASER); //surf_points_local_map_[n]: n号雷达在主雷达pivot下的local surf map
    corner_points_local_map_.clear(); 
    corner_points_local_map_.resize(NUM_OF_LASER);
    corner_points_local_map_filtered_.clear(); 
    corner_points_local_map_filtered_.resize(NUM_OF_LASER);//corner_points_local_map_[n]: n号雷达在主雷达pivot下的local corner map

    // #pragma omp parallel for num_threads(NUM_OF_LASER)
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        Pose pose_ext = Pose(qbl_[n], tbl_[n]);
        for (size_t i = 0; i < WINDOW_SIZE + 1; i++)//i=0,1,2,3,4
        {
            Pose pose_i(Qs_[i], Ts_[i]);
            pose_local_[n][i] = Pose(pose_pivot.T_.inverse() * pose_i.T_ * pose_ext.T_); //主雷达pivot到各雷达n(包括自己)i帧的变换
            PointICloud surf_points_trans, corner_points_trans;
            if (i == WINDOW_SIZE) continue;
            // if ((n != IDX_REF) && (i > pivot_idx)) continue;
            
            //此处正确，因为buildCalibMap()函数是在ESTIMATE_EXTRINSIC == 1时调用的；
            //副雷达的local map是用主雷达在窗口内的所有帧构建的，后面只对副雷达pivot帧下的points在local map中找correspondances.
            //`ESTIMATE_EXTRINSIC == 0`, which calls  `buildLocalMap(); 在该函数中用的是副雷达的所有帧构建的副雷达的local map.
            //https://github.com/gogojjh/M-LOAM/issues/7
            pcl::transformPointCloud(surf_points_stack_[IDX_REF][i], surf_points_trans, pose_local_[IDX_REF][i].T_.cast<float>());
            // for (auto &p: surf_points_trans.points) p.intensity = i;
            surf_points_local_map_[n] += surf_points_trans;

            pcl::transformPointCloud(corner_points_stack_[IDX_REF][i], corner_points_trans, pose_local_[IDX_REF][i].T_.cast<float>());
            // for (auto &p: corner_points_trans.points) p.intensity = i;
            corner_points_local_map_[n] += corner_points_trans;
        }
        float ratio = (n == IDX_REF ? 0.4 : 0.2);
        pcl::VoxelGrid<PointI> down_size_filter;
        down_size_filter.setLeafSize(ratio, ratio, ratio);
        down_size_filter.setInputCloud(boost::make_shared<PointICloud>(surf_points_local_map_[n]));
        down_size_filter.filter(surf_points_local_map_filtered_[n]);
        down_size_filter.setInputCloud(boost::make_shared<PointICloud>(corner_points_local_map_[n]));
        down_size_filter.filter(corner_points_local_map_filtered_[n]);
    }

    // calculate features and correspondences from p+1 to j
    surf_map_features_.clear(); 
    surf_map_features_.resize(NUM_OF_LASER);
    corner_map_features_.clear(); 
    corner_map_features_.resize(NUM_OF_LASER);
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        surf_map_features_[n].resize(WINDOW_SIZE + 1);
        corner_map_features_[n].resize(WINDOW_SIZE + 1);
    }

    // #pragma omp parallel for num_threads(NUM_OF_LASER)
    pcl::KdTreeFLANN<PointI>::Ptr kdtree_surf_points_local_map(new pcl::KdTreeFLANN<PointI>());
    pcl::KdTreeFLANN<PointI>::Ptr kdtree_corner_points_local_map(new pcl::KdTreeFLANN<PointI>());
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        // if (calib_converge_[n]) continue;
        kdtree_surf_points_local_map->setInputCloud(boost::make_shared<PointICloud>(surf_points_local_map_filtered_[n]));
        kdtree_corner_points_local_map->setInputCloud(boost::make_shared<PointICloud>(corner_points_local_map_filtered_[n]));
        for (size_t i = pivot_idx; i < WINDOW_SIZE + 1; i++)//i=2,3,4
        {
            if (((n == IDX_REF) && (i == pivot_idx))
             || ((n != IDX_REF) && (i != pivot_idx))) continue; 
             //忽略主雷达的pivot帧
             //忽略副雷达不是pivot帧的所有帧，即只考虑副雷达的pivot帧。对于副雷达只找在pivot帧在local map下的correspondances,其他帧不管。

            int n_neigh = (n == IDX_REF ? 5:10);
            f_extract_.matchSurfFromMap(kdtree_surf_points_local_map, //n号雷达在主雷达pivot下的local surf map kdtree
                                        surf_points_local_map_filtered_[n], //n号雷达在主雷达pivot下的local surf map
                                        surf_points_stack_[n][i], //n号雷达在i帧下的surf points
                                        pose_local_[n][i], //主雷达pivot到各雷达n(包括自己)的变换
                                        surf_map_features_[n][i], //[out]：在“n号雷达在主雷达pivot下的local surf map”中找“n号雷达在i帧下的surf points”的correspondances
                                        n_neigh, //在local map kdtree中找最近点的个数，要对它们构成的cov valid分析
                                        true); //FOV检测
            f_extract_.matchCornerFromMap(kdtree_corner_points_local_map, //类似
                                          corner_points_local_map_filtered_[n],
                                          corner_points_stack_[n][i],
                                          pose_local_[n][i],
                                          corner_map_features_[n][i],//[out]
                                          n_neigh,
                                          true);
        }
    }
    // LOG_EVERY_N(INFO, 20) << "build map(extract map): " << t_build_map.toc() << "ms("
    //                       << t_extract_map.toc() << ")ms";
    printf("build map: %fms\n", build_map_timer.Stop() * 1000);
    // if (PCL_VIEWER) visualizePCL();
}

//ESTIMATE_EXTRINSIC == 0
void Estimator::buildLocalMap()
{
    common::timing::Timer build_map_timer("odom_build_local_map");
    int pivot_idx = WINDOW_SIZE - OPT_WINDOW_SIZE; //4-2
    Pose pose_pivot(Qs_[pivot_idx], Ts_[pivot_idx]);

    // build the whole local map using all poses except the newest pose
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        surf_points_local_map_[n].clear();
        surf_points_local_map_filtered_[n].clear();
        corner_points_local_map_[n].clear();
        corner_points_local_map_filtered_[n].clear();
    }

    // #pragma omp parallel for num_threads(NUM_OF_LASER)
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        Pose pose_ext = Pose(qbl_[n], tbl_[n]);
        for (size_t i = 0; i < WINDOW_SIZE + 1; i++)//0,1,2,3,4
        {
            Pose pose_i(Qs_[i], Ts_[i]);
            pose_local_[n][i] = Pose(pose_pivot.T_.inverse() * pose_i.T_ * pose_ext.T_); //主雷达pivot到各雷达n(包括自己)i帧的变换
            if (i == WINDOW_SIZE) continue;
            PointICloud surf_points_trans, corner_points_trans;

            pcl::transformPointCloud(surf_points_stack_[n][i], surf_points_trans, pose_local_[n][i].T_.cast<float>());
            // for (auto &p: surf_points_trans.points) p.intensity = i;
            surf_points_local_map_[n] += surf_points_trans;

            pcl::transformPointCloud(corner_points_stack_[n][i], corner_points_trans, pose_local_[n][i].T_.cast<float>());
            // for (auto &p: surf_points_trans.points) p.intensity = i;
            corner_points_local_map_[n] += corner_points_trans;
        }

        float ratio;
        pcl::VoxelGrid<PointI> down_size_filter;
        ratio = 0.4 * std::min(2.0, std::max(0.75, 1.0 / 192 * float(N_SCANS * NUM_OF_LASER * WINDOW_SIZE)));
        down_size_filter.setLeafSize(ratio, ratio, ratio);
        down_size_filter.setInputCloud(boost::make_shared<PointICloud>(surf_points_local_map_[n]));
        down_size_filter.filter(surf_points_local_map_filtered_[n]);
        ratio = 0.4 * std::min(2.0, std::max(0.75, 1.0 / 192 * float(N_SCANS * NUM_OF_LASER * WINDOW_SIZE)));
        down_size_filter.setLeafSize(ratio, ratio, ratio);
        down_size_filter.setInputCloud(boost::make_shared<PointICloud>(corner_points_local_map_[n]));
        down_size_filter.filter(corner_points_local_map_filtered_[n]);
    }

    // calculate features and correspondences from p+1 to j
    surf_map_features_.clear();
    surf_map_features_.resize(NUM_OF_LASER);
    corner_map_features_.clear();
    corner_map_features_.resize(NUM_OF_LASER);
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        surf_map_features_[n].resize(WINDOW_SIZE + 1);
        corner_map_features_[n].resize(WINDOW_SIZE + 1);
    }

    sel_surf_feature_idx_.clear();
    sel_surf_feature_idx_.resize(NUM_OF_LASER);
    sel_corner_feature_idx_.clear();
    sel_corner_feature_idx_.resize(NUM_OF_LASER);
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        sel_surf_feature_idx_[n].resize(WINDOW_SIZE + 1);
        sel_corner_feature_idx_[n].resize(WINDOW_SIZE + 1);
    }

    // #pragma omp parallel for num_threads(NUM_OF_LASER)
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        pcl::KdTreeFLANN<PointI>::Ptr kdtree_surf_points_local_map(new pcl::KdTreeFLANN<PointI>());
        kdtree_surf_points_local_map->setInputCloud(boost::make_shared<PointICloud>(surf_points_local_map_filtered_[n]));
        pcl::KdTreeFLANN<PointI>::Ptr kdtree_corner_points_local_map(new pcl::KdTreeFLANN<PointI>());
        kdtree_corner_points_local_map->setInputCloud(boost::make_shared<PointICloud>(corner_points_local_map_filtered_[n]));
        Pose pose_ext = Pose(qbl_[n], tbl_[n]);
        int n_neigh = 5;
        for (size_t i = pivot_idx + 1; i < WINDOW_SIZE + 1; i++) //Xv[]中除过Xv[0]
        {
            Pose pose_i(Qs_[i], Ts_[i]);
            if (POINT_PLANE_FACTOR)
            {
                goodFeatureMatching(kdtree_surf_points_local_map, //n号雷达在主雷达pivot下的local surf map kdtree
                                    surf_points_local_map_filtered_[n], //n号雷达在主雷达pivot下的local surf map
                                    surf_points_stack_[n][i], //n号雷达在i帧下的surf points
                                    surf_map_features_[n][i], //[out]：在“n号雷达在主雷达pivot下的local surf map”中找“n号雷达在i帧下的surf points”的correspondances
                                    sel_surf_feature_idx_[n][i], //[out]: 挑选出第j个好point在自己点云帧下的index放进sel_surf_feature_idx_[n][i][j]
                                    's',
                                    pose_pivot, //主雷达pivot帧pose
                                    pose_i, //主雷达i帧的pose
                                    pose_ext, //主雷达到n雷达的外参
                                    ODOM_GF_RATIO); //0.8
            }
            if (POINT_EDGE_FACTOR)
            {
                goodFeatureMatching(kdtree_corner_points_local_map,
                                    corner_points_local_map_filtered_[n],
                                    corner_points_stack_[n][i],
                                    corner_map_features_[n][i],
                                    sel_corner_feature_idx_[n][i],
                                    'c',
                                    pose_pivot,
                                    pose_i,
                                    pose_ext,
                                    ODOM_GF_RATIO);
            }
        }
    }
    // LOG_EVERY_N(INFO, 20) << "build map(extract map): " << t_build_map.toc() << "ms("
    //                        << t_extract_map.toc() << ")ms";
    printf("build map: %fms\n", build_map_timer.Stop() * 1000);
    // if (PCL_VIEWER) visualizePCL();
}

void Estimator::evaluateFeatJacobian(const Pose &pose_pivot, //主雷达pivot帧pose
                                     const Pose &pose_i,  //主雷达i帧的pose
                                     const Pose &pose_ext, //主雷达到n雷达的外参
                                     PointPlaneFeature &feature)//n号雷达i帧下的point在local map中的correspondances
{
    if (feature.type_ == 's')
    {
        LidarPureOdomPlaneNormFactor f(feature.point_, feature.coeffs_, 1.0);

        double **param = new double *[3];

        param[0] = new double[SIZE_POSE];
        param[0][0] = pose_pivot.t_(0); //tp, p: pivot
        param[0][1] = pose_pivot.t_(1);
        param[0][2] = pose_pivot.t_(2);
        param[0][3] = pose_pivot.q_.x(); //Rp, JPL
        param[0][4] = pose_pivot.q_.y();
        param[0][5] = pose_pivot.q_.z();
        param[0][6] = pose_pivot.q_.w();

        param[1] = new double[SIZE_POSE];
        param[1][0] = pose_i.t_(0); //ti
        param[1][1] = pose_i.t_(1);
        param[1][2] = pose_i.t_(2);
        param[1][3] = pose_i.q_.x(); //Ri
        param[1][4] = pose_i.q_.y();
        param[1][5] = pose_i.q_.z();
        param[1][6] = pose_i.q_.w();

        param[2] = new double[SIZE_POSE];
        param[2][0] = pose_ext.t_(0); //delta_t, 外参
        param[2][1] = pose_ext.t_(1);
        param[2][2] = pose_ext.t_(2);
        param[2][3] = pose_ext.q_.x(); //delta_R
        param[2][4] = pose_ext.q_.y();
        param[2][5] = pose_ext.q_.z();
        param[2][6] = pose_ext.q_.w();

        double *res = new double[1];
        double **jaco = new double *[3];
        jaco[0] = new double[1 * 7];
        jaco[1] = new double[1 * 7];
        jaco[2] = new double[1 * 7];
        f.Evaluate(param, res, jaco); //计算jacobian

        // Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> mat_jacobian_1(jaco[0]);
        // Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> mat_jacobian_2(jaco[1]);
        // Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> mat_jacobian_3(jaco[2]);
        // Eigen::Matrix<double, 3, 7> mat_jacobian;
        // mat_jacobian.row(0) = mat_jacobian_1;
        // mat_jacobian.row(1) = mat_jacobian_2;
        // mat_jacobian.row(2) = mat_jacobian_3;
        // feature.jaco_ = mat_jacobian.topLeftCorner<3, 6>();

        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> mat_jacobian(jaco[1]);
        feature.jaco_ = mat_jacobian.topLeftCorner<1, 6>();

        delete[] jaco[0];
        delete[] jaco[1];
        delete[] jaco[2];
        delete[] jaco;
        delete[] res;
        delete[] param[0];
        delete[] param[1];
        delete[] param[2];
        delete[] param;
    } 
    else if (feature.type_ == 'c') //TODO(jxl): 对于corner point，为何不计算jacobian
    {                              //https://github.com/gogojjh/M-LOAM/issues/9
        feature.jaco_ = Eigen::Matrix<double, 1, 6>::Identity();  //1 0 0 0 0 0
    }

}

void Estimator::goodFeatureMatching(const pcl::KdTreeFLANN<PointI>::Ptr &kdtree_from_map,
                                    const PointICloud &laser_map,
                                    const PointICloud &laser_cloud,
                                    std::vector<PointPlaneFeature> &all_features, //[out]
                                    std::vector<size_t> &sel_feature_idx,  //[out]
                                    const char feature_type,
                                    const Pose &pose_pivot,
                                    const Pose &pose_i,
                                    const Pose &pose_ext,
                                    const double &gf_ratio)
{
    Pose pose_local(pose_pivot.T_.inverse() * pose_i.T_ * pose_ext.T_); //主雷达pivot到副雷达i的变换

    size_t num_all_features = laser_cloud.size();
    all_features.resize(num_all_features);
    std::vector<size_t> all_feature_idx(num_all_features);
    std::vector<int> feature_visited(num_all_features, -1); //-1: 还未用过该feature
    std::iota(all_feature_idx.begin(), all_feature_idx.end(), 0); //初始化为0,1，2，...
    
    size_t num_use_features;
    num_use_features = static_cast<size_t>(num_all_features * gf_ratio);
    sel_feature_idx.resize(num_use_features);

    //假设有100个points， gf_ratio =0.2，  num_use_features =50个, size_rnd_subset = 5
    //all_feature_idx:  0  1  2  3 ... 99
    //feature_visited: -1 -1 -1 -1 ... -1

    size_t size_rnd_subset = static_cast<size_t>(1.0 * num_all_features / num_use_features);
    Eigen::Matrix<double, 6, 6> sub_mat_H = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
    size_t num_sel_features = 0; //正在挑选出第几个好point
    common::timing::Timer gfm_timer("odom_match_feat");

    size_t n_neigh = 5;
    bool b_match;  
    size_t num_rnd_que;
    if (gf_ratio == 1.0)
    {
        for (size_t j = 0; j < all_feature_idx.size(); j++)
        {
            size_t que_idx = all_feature_idx[j];
            b_match = false;
            if (feature_type == 's')
            {
                b_match = f_extract_.matchSurfPointFromMap(kdtree_from_map,
                                                           laser_map,
                                                           laser_cloud.points[que_idx],
                                                           pose_local,
                                                           all_features[que_idx], 
                                                           que_idx,
                                                           n_neigh,
                                                           false);
            }
            else if (feature_type == 'c')
            {
                b_match = f_extract_.matchCornerPointFromMap(kdtree_from_map,
                                                             laser_map,
                                                             laser_cloud.points[que_idx],
                                                             pose_local,
                                                             all_features[que_idx],
                                                             que_idx,
                                                             n_neigh,
                                                             false);
            }
            if (b_match)
            {
                sel_feature_idx[num_sel_features] = que_idx;
                num_sel_features++;
            }
        }
    } 
    else
    {
        while (true)
        {
            if ((num_sel_features >= num_use_features) ||
                (all_feature_idx.size() == 0) ||
                (gfm_timer.GetCountTime() * 1000 > MAX_FEATURE_SELECT_TIME)) //const 7ms
                    break;

            std::priority_queue<FeatureWithScore, 
                                std::vector<FeatureWithScore>, 
                                std::less<FeatureWithScore>> heap_subset; //分数从大到小，优先级由高到低，所以FeatureWithScore需要重载operator < 
           //在挑选第1个好point时，heap_subset[]里已经存放了挑选完第0个好point剩余点的score, jacobian
           //在挑选第2个好point时，heap_subset[]里已经存放了挑选完第1个好point剩余点的score, jacobian
           //依次类推
           //比如第0个好point: 挑选的j集合为{0 1 2 3 4}，1号为好point，heap_subset里{0 1 2 3 4}
           //比如第1个好point: 挑选的j集合为{0 2 3 4 5}，heap_subset里{0 1 2 3 4 5}, 0号为好point，heap_subset里剩余{0 1 2 3 4 5}

            while (true)
            {
                if (all_feature_idx.size() == 0) break;
                num_rnd_que = 0;
                size_t j;
                while (num_rnd_que < MAX_RANDOM_QUEUE_TIME)//const 10
                {
                    j = rgi_.geneRandUniform(0, all_feature_idx.size() - 1);
                    if (feature_visited[j] < int(num_sel_features))
                    {
                        feature_visited[j] = int(num_sel_features);
                        break;
                    }
                    num_rnd_que++;
                }
                if (num_rnd_que >= MAX_RANDOM_QUEUE_TIME || gfm_timer.GetCountTime() * 1000 > MAX_FEATURE_SELECT_TIME)
                    break;

                size_t que_idx = all_feature_idx[j];

                //在本次挑选第i个好point时，即使挑选到之前曾经处理过的point，也不会重复计算jacobian.
                //因为PointPlaneFeature的构造函数默认type为‘n’，如果之前处理过该point，类型就变为‘s’,或者‘c’。
                //如果没有correspondance, 早已经在之前步骤中从all_feature_idx[]中移除了,本次不会选到之前曾经处理过的point
                if (all_features[que_idx].type_ == 'n') //这句妙啊！
                {
                    b_match = false;
                    if (feature_type == 's')
                    {
                        b_match = f_extract_.matchSurfPointFromMap(kdtree_from_map, //n号雷达在主雷达pivot下的local surf map kdtree
                                                                   laser_map, //n号雷达在主雷达pivot下的local surf map
                                                                   laser_cloud.points[que_idx], //n号雷达在i帧下的surf points[que_idx]
                                                                   pose_local,  //主雷达pivot到副雷达i的变换
                                                                   all_features[que_idx], //[out]: n号雷达在i帧下的surf points[que_idx]在local map中的correspondances放置在all_features[que_idx]
                                                                   que_idx, //que_idx
                                                                   n_neigh, //5
                                                                   false); 
                    }
                    else if (feature_type == 'c')
                    {
                        b_match = f_extract_.matchCornerPointFromMap(kdtree_from_map,
                                                                     laser_map,
                                                                     laser_cloud.points[que_idx],
                                                                     pose_local,
                                                                     all_features[que_idx],
                                                                     que_idx,
                                                                     n_neigh,
                                                                     false);
                    }
                    if (b_match) //是否找到了在local map中对应的correspondances
                    {
                        evaluateFeatJacobian(pose_pivot,
                                             pose_i,
                                             pose_ext,
                                             all_features[que_idx]); //计算残差(点到面，点到线)对point i的jacobian，维数：1*6
                    } 
                    else //没有找到correspondances
                    {
                        all_feature_idx.erase(all_feature_idx.begin() + j);
                        feature_visited.erase(feature_visited.begin() + j);
                        continue;
                    }
                }

                const Eigen::MatrixXd &jaco = all_features[que_idx].jaco_;
                double cur_det = common::logDet(sub_mat_H + jaco.transpose() * jaco, //sub_mat_H: 当前时刻之前，所有好points的J^T*J
                                                true); 
                //TODO(jxl): J^T*J分解，这块打分的依据是什么？ https://github.com/gogojjh/M-LOAM/issues/10

                //按照分数从最大到最小排序，挑出最好的point，累加好points的J^T*J, 把好point在点云中的idx放到sel_feature_idx[]
                //同时从all_feature_idx[], feature_visited[]中移除
                heap_subset.push(FeatureWithScore(que_idx, cur_det, jaco));
                if (heap_subset.size() >= size_rnd_subset)
                {
                    const FeatureWithScore &fws = heap_subset.top();
                    std::vector<size_t>::iterator iter = std::find(all_feature_idx.begin(), all_feature_idx.end(), fws.idx_);
                    if (iter == all_feature_idx.end())
                    {
                        std::cerr << "odometry [goodFeatureMatching]: not exist feature idx !" << std::endl;
                        break;
                    }
                    sub_mat_H += fws.jaco_.transpose() * fws.jaco_; //每挑选出一个好point, sub_mat_H累加所有好points的J^T*J

                    size_t position = iter - all_feature_idx.begin();
                    all_feature_idx.erase(all_feature_idx.begin() + position);
                    feature_visited.erase(feature_visited.begin() + position);
                    sel_feature_idx[num_sel_features] = fws.idx_; //把挑选出第i个好point在自己点云帧下的index放进sel_feature_idx[i]
                    num_sel_features++;
                    // printf("position: %lu, num: %lu\n", position, num_rnd_que);
                    break;
                }
                if (num_rnd_que >= MAX_RANDOM_QUEUE_TIME || gfm_timer.GetCountTime() * 1000 > MAX_FEATURE_SELECT_TIME)
                    break;
            }
            if (num_rnd_que >= MAX_RANDOM_QUEUE_TIME || gfm_timer.GetCountTime() * 1000 > MAX_FEATURE_SELECT_TIME)
            {
                std::cout << "odometry [goodFeatureMatching]: early termination!" << std::endl;
                LOG(INFO) << "early termination: feature_type " << feature_type << ", " << num_rnd_que << ", " << gfm_timer.GetCountTime() * 1000;
            }
        }
    }
    gfm_timer.Stop();
    sel_feature_idx.resize(num_sel_features);
    // printf("num of all features: %lu, selected features: %lu\n", num_all_features, num_use_features);
}

// push new state and measurements in the sliding window
// move the localmap in the pivot frame to the pivot+1 frame, and remove the first point cloud
void Estimator::slideWindow()
{
    // TicToc t_solid_window;
    // printf("size of sliding window: %lu\n", cir_buf_cnt_);
    Qs_.push(Qs_[cir_buf_cnt_]);
    Ts_.push(Ts_[cir_buf_cnt_]);
    Header_.push(Header_[cir_buf_cnt_]);
    for (size_t n = 0; n < NUM_OF_LASER; n++)
    {
        surf_points_stack_[n].push(surf_points_stack_[n][cir_buf_cnt_]);
        surf_points_stack_size_[n].push(surf_points_stack_size_[n][cir_buf_cnt_]);
        corner_points_stack_[n].push(corner_points_stack_[n][cir_buf_cnt_]);
        corner_points_stack_size_[n].push(corner_points_stack_size_[n][cir_buf_cnt_]);
    }
    // printf("slide window: %fms\n", t_solid_window.toc());
}

void Estimator::vector2Double()
{
    int pivot_idx = WINDOW_SIZE - OPT_WINDOW_SIZE; //2=4-2
    for (size_t i = pivot_idx; i < WINDOW_SIZE + 1; i++)//i = 2,3,4
    {
        para_pose_[i - pivot_idx][0] = Ts_[i](0);
        para_pose_[i - pivot_idx][1] = Ts_[i](1);
        para_pose_[i - pivot_idx][2] = Ts_[i](2);
        para_pose_[i - pivot_idx][3] = Qs_[i].x();
        para_pose_[i - pivot_idx][4] = Qs_[i].y();
        para_pose_[i - pivot_idx][5] = Qs_[i].z();
        para_pose_[i - pivot_idx][6] = Qs_[i].w();
    }
    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        para_ex_pose_[i][0] = tbl_[i](0);
        para_ex_pose_[i][1] = tbl_[i](1);
        para_ex_pose_[i][2] = tbl_[i](2);
        para_ex_pose_[i][3] = qbl_[i].x(); //TODO(jxl): m-loam的四元数定义为[qx, qy, qz, qw], JPL惯例, 和作者邮件已确认
        para_ex_pose_[i][4] = qbl_[i].y();
        para_ex_pose_[i][5] = qbl_[i].z();
        para_ex_pose_[i][6] = qbl_[i].w();
    }
    // for (size_t i = 0; i < NUM_OF_LASER; i++)
    // {
    //     para_td_[i] = tdbl_[i];
    // }
}

void Estimator::double2Vector()
{
    int pivot_idx = WINDOW_SIZE - OPT_WINDOW_SIZE;
    for (size_t i = 0; i < OPT_WINDOW_SIZE + 1; i++)
    {
        Ts_[i + pivot_idx] = Eigen::Vector3d(para_pose_[i][0], para_pose_[i][1], para_pose_[i][2]);
        Qs_[i + pivot_idx] = Eigen::Quaterniond(para_pose_[i][6], para_pose_[i][3], para_pose_[i][4], para_pose_[i][5]);
    }
    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        tbl_[i] = Eigen::Vector3d(para_ex_pose_[i][0], para_ex_pose_[i][1], para_ex_pose_[i][2]);
        qbl_[i] = Eigen::Quaterniond(para_ex_pose_[i][6], para_ex_pose_[i][3], para_ex_pose_[i][4], para_ex_pose_[i][5]);
    }
    // for (size_t i = 0; i < NUM_OF_LASER; i++)
    // {
    //     tdbl_[i] = para_td_[i];
    // }
}

void Estimator::evalResidual(ceres::Problem &problem,
                             std::vector<PoseLocalParameterization *> &local_param_ids,
                             const std::vector<double *> &para_ids,
                             const std::vector<ceres::internal::ResidualBlock *> &res_ids_proj,
                             const MarginalizationInfo *last_marginalization_info_,
                             const std::vector<ceres::internal::ResidualBlock *> &res_ids_marg)
{
	double cost;
    ceres::CRSMatrix jaco; 
    // A compressed row sparse matrix used primarily for communicating the
    // Jacobian matrix to the user.

    ceres::Problem::EvaluateOptions e_option;
	if ((PRIOR_FACTOR) || (POINT_PLANE_FACTOR) || (POINT_EDGE_FACTOR))
	{
		e_option.parameter_blocks = para_ids;
		e_option.residual_blocks = res_ids_proj;
        problem.Evaluate(e_option, &cost, nullptr, nullptr, &jaco);
        // printf("cost res: %f\n", cost);
        evalDegenracy(local_param_ids, jaco);
    }
	// if (MARGINALIZATION_FACTOR)
	// {
	// 	if (last_marginalization_info_ && !res_ids_marg.empty())
	// 	{
	// 		e_option.parameter_blocks = para_ids;
	// 		e_option.residual_blocks = res_ids_marg;
    //         problem.Evaluate(e_option, &cost, nullptr, nullptr, &jaco);
    //         printf("cost marg: %f\n", cost);
	// 	}
	// }
}

// A^TA is not only symmetric and invertiable: https://math.stackexchange.com/questions/2352684/when-is-a-symmetric-matrix-invertible
void Estimator::evalDegenracy(std::vector<PoseLocalParameterization *> &local_param_ids,
                              const ceres::CRSMatrix &jaco)
{
    // printf("jacob: %d constraints, %d parameters 6 * (%d pose_param_block, %d ext_param_block)\n",
    //        jaco.num_rows, jaco.num_cols, OPT_WINDOW_SIZE + 1, NUM_OF_LASER); // 1555(feature_size) * 48(para_size)
    if (jaco.num_rows == 0) return;
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_J; // Jacobian is a diagonal matrix
    CRSMatrix2EigenMatrix(jaco, mat_J);
    Eigen::SparseMatrix<double, Eigen::RowMajor> mat_Jt = mat_J.transpose();
    Eigen::MatrixXd mat_JtJ = mat_Jt * mat_J;
    // bool b_vis = false; // to verify the structure of A^T*A
    // if (b_vis)
    // {
    //     printf("visualize the structure of H(J^T*J)\n");
    //     for (size_t i = 0; i < mat_JtJ.rows(); i++)
    //     {
    //         for (auto j = 0; j < mat_JtJ.cols(); j++)
    //         {
    //             if (mat_JtJ(i, j) == 0) std::cout << "0 ";
    //                                else std::cout << "1 ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    // calculate the degeneracy factor of poses
    for (size_t i = 0; i < OPT_WINDOW_SIZE + 1; i++) //Xv[0], Xv[1], Xv[2]
    {
        Eigen::Matrix<double, 6, 6> mat_H = mat_JtJ.block(6 * i, 6 * i, 6, 6);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6> > esolver(mat_H);
        Eigen::Matrix<double, 1, 6> mat_E = esolver.eigenvalues().real(); // 6*1， 特征值从小到大
        Eigen::Matrix<double, 6, 6> mat_V_f = esolver.eigenvectors().real(); // 6*6, column is the corresponding eigenvector
        Eigen::Matrix<double, 6, 6> mat_V_p = mat_V_f;
        for (auto j = 0; j < mat_E.cols(); j++)
        {
            if (mat_E(0, j) < eig_thre_(i)) //[100 100 100 0 0]
            {
                mat_V_p.col(j) = Eigen::Matrix<double, 6, 1>::Zero();
                local_param_ids[i]->is_degenerate_ = true;
            } else
            {
                break;
            }
        }
        std::cout << i << " D factor: " << mat_E(0, 0) << ": " << mat_V_f.col(0).transpose() << std::endl;
        LOG(INFO) << i << " D factor: " << mat_E(0, 0) << ": " << mat_V_f.col(0).transpose();
        Eigen::Matrix<double, 6, 6> mat_P = (mat_V_f.transpose()).inverse() * mat_V_p.transpose(); // 6*6， 
        //和该文章作者已经讨论过了
        //https://zhuanlan.zhihu.com/p/258159552

        if (local_param_ids[i]->is_degenerate_)
        {
            local_param_ids[i]->V_update_ = mat_P;
            // std::cout << "param " << i << " is degenerate !" << std::endl;
            // std::cout << mat_P << std::endl;
        }
    }

    // calculate the degeneracy factor of extrinsics
    if (ESTIMATE_EXTRINSIC != 0)
    {
        d_factor_calib_ = std::vector<double>(NUM_OF_LASER, 0);
        for (size_t i = OPT_WINDOW_SIZE + 1; i < local_param_ids.size(); i++) //i=0,1,2分别对应Xv，所以i=3，4，对应主雷达到每个副雷达的外参
        {
            if (frame_cnt_ % N_CUMU_FEATURE == 0) // need to optimize the extriniscs
            {
                Eigen::Matrix<double, 6, 6> mat_H = mat_JtJ.block(6 * i, 6 * i, 6, 6);
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> esolver(mat_H);
                Eigen::Matrix<double, 1, 6> mat_E = esolver.eigenvalues().real(); // 6*1，特征值从小到大
                double lambda = mat_E(0, 0) / N_CUMU_FEATURE;
                // std::cout << mat_H << std::endl;
                // double lambda = mat_E(0, 0);
                printf("%lu: calib eig is %f\n", i - OPT_WINDOW_SIZE - 1, lambda);
                log_lambda_.push_back(lambda);
                if (lambda >= LAMBDA_THRE_CALIB) //没有退化，标定成功
                {
                    eig_thre_(i) = LAMBDA_THRE_CALIB;
                    d_factor_calib_[i - OPT_WINDOW_SIZE - 1] = lambda;
                }
                else if (lambda > eig_thre_(i))
                {
                    eig_thre_(i) = lambda;
                }
                else
                {
                    // degenerate cases for calibration, not update the extrinsics
                    local_param_ids[i]->is_degenerate_ = true;
                    local_param_ids[i]->V_update_.setZero();
                }
                Pose tmp_pose(qbl_[i - OPT_WINDOW_SIZE - 1], tbl_[i - OPT_WINDOW_SIZE - 1]);
                log_extrinsics_.push_back(tmp_pose);
            }
            else
            {
                // no enough cumu features for calibration, not update the extrinsics
                local_param_ids[i]->is_degenerate_ = true;
                local_param_ids[i]->V_update_.setZero();
            }           
        }
    }
    std::cout << eig_thre_.transpose() << std::endl;
}

void Estimator::evalCalib()
{
    if (solver_flag_ == NON_LINEAR)
    {
        for (size_t n = 0; n < NUM_OF_LASER; n++)
        {
            if (d_factor_calib_[n] != 0) // with high constraints
            {
                double weight = pow(d_factor_calib_[n] / LAMBDA_THRE_CALIB, 1.0);
                Pose pose_ext = Pose(qbl_[n], tbl_[n]);
                pose_calib_[n].push_back(make_pair(weight, pose_ext));
            }
        }

        // check if all lidars are coveraged
        bool is_converage = true;
        for (size_t n = 0; n < NUM_OF_LASER; n++)
        {
            if (n == IDX_REF) continue;
            std::cout << common::YELLOW
                      << "laser_" << n
                      << ", eligible calib size: " << pose_calib_[n].size() 
                      << common::RESET << std::endl;
            if (pose_calib_[n].size() >= N_CALIB) calib_converge_[n] = true; //对每个雷达，25次以上标定成功
            else is_converage = false;
        }

        if (is_converage) //所有外参都收敛
        {
            std::cout << common::YELLOW << "Finish nonlinear calibration !" << common::RESET << std::endl;

            ESTIMATE_EXTRINSIC = 0; //当外参refine收敛结束后，置 ESTIMATE_EXTRINSIC = 0
            for (size_t n = 0; n < NUM_OF_LASER; n++)
            {
                Pose pose_mean;
                if (n != IDX_REF)
                {
                    LOG(INFO) << n << ":";
                    Eigen::Matrix<double, 6, 6> pose_cov;

                    computeMeanPose(pose_calib_[n], pose_mean, pose_cov); // compute the mean calibration parameters
                    //在李代数空间计算均值

                    qbl_[n] = pose_mean.q_;
                    tbl_[n] = pose_mean.t_;
                    covbl_[n] = pose_cov.diagonal().asDiagonal();
                }
                log_lambda_.push_back(0.0);
                log_extrinsics_.push_back(pose_mean);
            }
            // ini_fixed_local_map_ = false; // reconstruct new optimized map

            //online refine阶段结束，进入 ESTIMATE_EXTRINSIC = 0阶段，所以丢弃在本次滑窗末尾计算的边缘化残差
            if (last_marginalization_info_ != nullptr) delete last_marginalization_info_;
            last_marginalization_info_ = nullptr; // meaning that the prior errors in online calibration are discarded
            last_marginalization_parameter_blocks_.clear();
        }
    }
}

void Estimator::printParameter()
{
    printf("print optimized window (p -> j) [qx qy qz qw x y z]\n");
    for (size_t i = 0; i < OPT_WINDOW_SIZE + 1; i++)
    {
        std::cout << "Pose " << WINDOW_SIZE - OPT_WINDOW_SIZE + i << ": " <<
            para_pose_[i][3] << " " <<
            para_pose_[i][4] << " " <<
            para_pose_[i][5] << " " <<
            para_pose_[i][6] << " " <<
            para_pose_[i][0] << " " <<
            para_pose_[i][1] << " " <<
            para_pose_[i][2] << std::endl;
    }
    for (size_t i = 0; i < NUM_OF_LASER; i++)
    {
        std::cout << "Ext: " << " " <<
            para_ex_pose_[i][3] << " " <<
            para_ex_pose_[i][4] << " " <<
            para_ex_pose_[i][5] << " " <<
            para_ex_pose_[i][6] << " " <<
            para_ex_pose_[i][0] << " " <<
            para_ex_pose_[i][1] << " " <<
            para_ex_pose_[i][2] << std::endl;
    }
    // for (size_t i = 0; i < NUM_OF_LASER; i++)
    // {
    //     std::cout << "dt: " <<
    //         para_td_[i] << std::endl;
    // }
}

void Estimator::printSlideWindow()
{
    printf("print slide window (0 -> j) ************************\n");
    for (size_t i = 0; i < cir_buf_cnt_ + 1; i++)
    {
        Pose pose(Qs_[i], Ts_[i]);
        std::cout << i << ": " << pose << std::endl;
    }
}

//
