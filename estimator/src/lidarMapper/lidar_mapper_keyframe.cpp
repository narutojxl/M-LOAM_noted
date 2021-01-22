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

#define PCL_NO_PRECOMPILE

#include "lidar_mapper.h"

using namespace common;

SaveStatistics save_statistics;

int frame_cnt = 0;
int frame_drop_cnt = 0;

double time_laser_cloud_surf_last = 0;
double time_laser_cloud_corner_last = 0;
double time_laser_cloud_full_res = 0;
double time_laser_cloud_outlier = 0;
double time_laser_odometry = 0;
double time_ext = 0;

// thread data buffer
std::queue<sensor_msgs::PointCloud2ConstPtr> surf_last_buf;
std::queue<sensor_msgs::PointCloud2ConstPtr> corner_last_buf;
std::queue<sensor_msgs::PointCloud2ConstPtr> full_res_buf;
std::queue<sensor_msgs::PointCloud2ConstPtr> outlier_buf;
std::queue<nav_msgs::Odometry::ConstPtr> odometry_buf; //odom
std::queue<mloam_msgs::ExtrinsicsConstPtr> ext_buf; //外参
std::queue<mloam_msgs::KeyframesConstPtr> loop_info_buf; //闭环
std::mutex m_buf;

PointICloud::Ptr laser_cloud_surf_last(new PointICloud());
PointICloud::Ptr laser_cloud_corner_last(new PointICloud());
PointICloud::Ptr laser_cloud_surf_last_ds(new PointICloud());
PointICloud::Ptr laser_cloud_corner_last_ds(new PointICloud());
PointICloud::Ptr laser_cloud_full_res(new PointICloud());
PointICloud::Ptr laser_cloud_outlier(new PointICloud());
PointICloud::Ptr laser_cloud_outlier_ds(new PointICloud());

PointICovCloud::Ptr laser_cloud_surf_from_map_cov(new PointICovCloud()); //local surf map
PointICovCloud::Ptr laser_cloud_corner_from_map_cov(new PointICovCloud()); //local corner map
PointICovCloud::Ptr laser_cloud_surf_from_map_cov_ds(new PointICovCloud());
PointICovCloud::Ptr laser_cloud_corner_from_map_cov_ds(new PointICovCloud());

PointICovCloud::Ptr laser_cloud_surf_cov(new PointICovCloud());
PointICovCloud::Ptr laser_cloud_corner_cov(new PointICovCloud());
PointICovCloud::Ptr laser_cloud_outlier_cov(new PointICovCloud());

pcl::KdTreeFLANN<PointI>::Ptr kdtree_surrounding_keyframes(new pcl::KdTreeFLANN<PointI>());
pcl::KdTreeFLANN<PointI>::Ptr kdtree_global_map_keyframes(new pcl::KdTreeFLANN<PointI>());
pcl::KdTreeFLANN<PointIWithCov>::Ptr kdtree_surf_from_map(new pcl::KdTreeFLANN<PointIWithCov>());
pcl::KdTreeFLANN<PointIWithCov>::Ptr kdtree_corner_from_map(new pcl::KdTreeFLANN<PointIWithCov>());

bool save_new_keyframe;
PointICloud::Ptr surrounding_keyframes(new PointICloud());
PointICloud::Ptr surrounding_keyframes_ds(new PointICloud());

PointICloud::Ptr global_map_keyframes(new PointICloud());
PointICloud::Ptr global_map_keyframes_ds(new PointICloud());

std::vector<int> surrounding_existing_keyframes_id; //当期帧周围的关键帧index
std::vector<PointICovCloud::Ptr> surrounding_surf_cloud_keyframes; //当期帧周围的关键帧 surf points转换到map下，即local surf map
std::vector<PointICovCloud::Ptr> surrounding_corner_cloud_keyframes; //当期帧周围的关键帧 corner points转换到map下，即local corner map
std::vector<PointICovCloud::Ptr> surf_cloud_keyframes_cov;  //所有keyframes surf points, points在每个关键帧下
std::vector<PointICovCloud::Ptr> corner_cloud_keyframes_cov;//所有keyframes corner points
std::vector<PointICovCloud::Ptr> outlier_cloud_keyframes_cov;//所有keyframes outlier points

// downsampling voxel grid
pcl::VoxelGridCovarianceMLOAM<PointI> down_size_filter_surf;
pcl::VoxelGridCovarianceMLOAM<PointI> down_size_filter_corner;
pcl::VoxelGridCovarianceMLOAM<PointI> down_size_filter_outlier;
pcl::VoxelGridCovarianceMLOAM<PointI> down_size_filter_surrounding_keyframes;
pcl::VoxelGridCovarianceMLOAM<PointI> down_size_filter_global_map_keyframes;
pcl::VoxelGridCovarianceMLOAM<PointIWithCov> down_size_filter_surf_map_cov;
pcl::VoxelGridCovarianceMLOAM<PointIWithCov> down_size_filter_corner_map_cov;
pcl::VoxelGridCovarianceMLOAM<PointIWithCov> down_size_filter_outlier_map_cov;
pcl::VoxelGridCovarianceMLOAM<PointIWithCov> down_size_filter_global_map_cov;

std::vector<int> point_search_ind;
std::vector<float> point_search_sq_dis;

PointICloud::Ptr pose_keyframes_3d(new PointICloud());
PointI pose_point_cur, pose_point_prev;
Eigen::Quaterniond q_ori_cur, q_ori_prev;
std::vector<std::pair<double, Pose> > pose_keyframes_6d;
mloam_msgs::Keyframes laser_keyframes_6d;

mloam_msgs::Keyframes loop_info;

// wmap_T_curr = wmap_T_odom * wodom_T_curr;
// transformation between odom's world and map's world frame
double para_pose[SIZE_POSE];
Pose pose_wmap_prev, pose_wmap_curr, pose_wmap_wodom, pose_wodom_curr;
//主雷达T_map_prev, T_map_curr, T_map_odom, T_odom_curr

ros::Publisher pub_laser_cloud_surrounding, pub_laser_cloud_map;
ros::Publisher pub_laser_cloud_full_res;
ros::Publisher pub_laser_cloud_surf_last_res, pub_laser_cloud_corner_last_res;
ros::Publisher pub_good_surf_feature;
ros::Publisher pub_odom_aft_mapped, pub_odom_aft_mapped_high_frec, pub_laser_after_mapped_path;
ros::Publisher pub_keyframes, pub_keyframes_6d;

nav_msgs::Path laser_after_mapped_path;

// extrinsics
mloam_msgs::Extrinsics extrinsics;
std::vector<Pose> pose_ext; //外参

std::vector<Eigen::Matrix<double, 1, 6> > d_factor_list;
std::vector<Eigen::Matrix<double, 6, 6> > d_eigvec_list;

Eigen::Matrix<double, 6, 6> mat_P;

std::vector<double> gf_logdet_H_list; //good feature points累加的Hessian矩阵
std::vector<double> gf_deg_factor_list;
std::vector<std::vector<double> > mapping_sp_list;
std::vector<double> total_match_feature;
std::vector<double> total_solver;
std::vector<double> total_mapping;

bool is_degenerate;
bool with_ua_flag; //true, <!-- awareness of uncertainty propagation-->  

Eigen::Matrix<double, 6, 6> cov_mapping;

pcl::PCDWriter pcd_writer;

double lambda = 10.0;
double gf_ratio_cur;

ActiveFeatureSelection afs;

std::mutex m_process;

// set current pose after odom
void transformAssociateToMap()
{
	// q_w_curr = q_wmap_wodom * q_wodom_curr;
	// t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
	pose_wmap_curr = pose_wmap_wodom * pose_wodom_curr;
    //pose_wmap_wodom： 上一帧和local map匹配得到的T_map_odom
    //pose_wodom_curr: 前端中滑窗对相邻两scan-to-scan匹配得到的laser odom refine后得到的curr frame in odom
    //pose_wmap_curr:  curr frame in map 预测值
}

// update the transformation between map's world to odom's world after map
void transformUpdate()
{
	// q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	// t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
	pose_wmap_wodom = pose_wmap_curr * pose_wodom_curr.inverse();
    //pose_wmap_curr: 在初值的基础上，curr scan与local map匹配后refine的值, curr frame in map
    //pose_wmap_wodom: 本帧结束末尾，更新T_map_odom，为下一帧做准备
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laser_cloud_surf_last_msg)
{
	m_buf.lock();
	surf_last_buf.push(laser_cloud_surf_last_msg);
	m_buf.unlock();
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laser_cloud_corner_last_msg)
{
	m_buf.lock();
	corner_last_buf.push(laser_cloud_corner_last_msg);
	m_buf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laser_cloud_full_res_msg)
{
	m_buf.lock();
	full_res_buf.push(laser_cloud_full_res_msg);
	m_buf.unlock();
}

void laserCloudOutlierResHandler(const sensor_msgs::PointCloud2ConstPtr &laser_cloud_outlier_msg)
{
    m_buf.lock();
    outlier_buf.push(laser_cloud_outlier_msg);
    m_buf.unlock();
}

void extrinsicsHandler(const mloam_msgs::ExtrinsicsConstPtr &ext)
{
	m_buf.lock();
	ext_buf.push(ext);
	m_buf.unlock();
}

void loopInfoHandler(const mloam_msgs::KeyframesConstPtr &loop_info_msg)
{
    m_buf.lock();
    loop_info_buf.push(loop_info_msg);
    m_buf.unlock();
}

//receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laser_odom)
{
	m_buf.lock();
	odometry_buf.push(laser_odom);
	m_buf.unlock();

	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr; //没有使用
	q_wodom_curr.x() = laser_odom->pose.pose.orientation.x;
	q_wodom_curr.y() = laser_odom->pose.pose.orientation.y;
	q_wodom_curr.z() = laser_odom->pose.pose.orientation.z;
	q_wodom_curr.w() = laser_odom->pose.pose.orientation.w;
	t_wodom_curr.x() = laser_odom->pose.pose.position.x;
	t_wodom_curr.y() = laser_odom->pose.pose.position.y;
	t_wodom_curr.z() = laser_odom->pose.pose.position.z;

	Pose pose_wmap_curr_ini = pose_wmap_wodom * pose_wodom_curr;
	nav_msgs::Odometry odom_aft_mapped;
	odom_aft_mapped.header.frame_id = "/world";
	odom_aft_mapped.child_frame_id = "/aft_mapped";
	odom_aft_mapped.header.stamp = laser_odom->header.stamp;
	odom_aft_mapped.pose.pose.orientation.x = pose_wmap_curr_ini.q_.x();
	odom_aft_mapped.pose.pose.orientation.y = pose_wmap_curr_ini.q_.y();
	odom_aft_mapped.pose.pose.orientation.z = pose_wmap_curr_ini.q_.z();
	odom_aft_mapped.pose.pose.orientation.w = pose_wmap_curr_ini.q_.w();
	odom_aft_mapped.pose.pose.position.x = pose_wmap_curr_ini.t_.x();
	odom_aft_mapped.pose.pose.position.y = pose_wmap_curr_ini.t_.y();
	odom_aft_mapped.pose.pose.position.z = pose_wmap_curr_ini.t_.z();
	pub_odom_aft_mapped_high_frec.publish(odom_aft_mapped); // publish (k-1)th oldest map * kth newest odom
}

void vector2Double()
{
	para_pose[0] = pose_wmap_curr.t_(0);
	para_pose[1] = pose_wmap_curr.t_(1);
	para_pose[2] = pose_wmap_curr.t_(2);
	para_pose[3] = pose_wmap_curr.q_.x(); //JPL conversion
	para_pose[4] = pose_wmap_curr.q_.y();
	para_pose[5] = pose_wmap_curr.q_.z();
	para_pose[6] = pose_wmap_curr.q_.w();
}

void double2Vector()
{
	pose_wmap_curr.t_ = Eigen::Vector3d(para_pose[0], para_pose[1], para_pose[2]);
	pose_wmap_curr.q_ = Eigen::Quaterniond(para_pose[6], para_pose[3], para_pose[4], para_pose[5]);
    pose_wmap_curr.update();
}

void extractSurroundingKeyFrames()
{
    if (pose_keyframes_6d.size() == 0) return;
    if ((!laser_cloud_surf_from_map_cov_ds->size() == 0) && (!laser_cloud_corner_from_map_cov_ds->size() == 0)) 
    {
        printf("not need to construct the map\n");
        return;
    }
    //只有当上一帧是个keyframe，才会重新build local map；否则，local map保持不变，直接return。

    // update the current point
    pose_point_cur.x = pose_wmap_curr.t_[0]; //当前帧在map下pose init值
    pose_point_cur.y = pose_wmap_curr.t_[1];
    pose_point_cur.z = pose_wmap_curr.t_[2];

    surrounding_keyframes->clear();  //在关键帧的kd tree中搜索距离当期帧30m以内, 都有哪些keyframes
    kdtree_surrounding_keyframes->setInputCloud(pose_keyframes_3d);
    kdtree_surrounding_keyframes->radiusSearch(pose_point_cur, SURROUNDING_KF_RADIUS, point_search_ind, point_search_sq_dis, 0);
    for (size_t i = 0; i < point_search_ind.size(); i++)
        surrounding_keyframes->push_back(pose_keyframes_3d->points[point_search_ind[i]]);
    
    for (int i = 0; i < surrounding_existing_keyframes_id.size(); i++) // existing keyframes id
    {
        bool existing_flag = false;
        for (int j = 0; j < surrounding_keyframes->size(); j++) // current surrounding keyframes id
        {
            if (surrounding_existing_keyframes_id[i] == (int)surrounding_keyframes->points[j].intensity)
            {
                existing_flag = true;
                break;
            }
        }
        if (!existing_flag)
        {
            surrounding_existing_keyframes_id.erase(surrounding_existing_keyframes_id.begin() + i);
            surrounding_surf_cloud_keyframes.erase(surrounding_surf_cloud_keyframes.begin() + i);
            surrounding_corner_cloud_keyframes.erase(surrounding_corner_cloud_keyframes.begin() + i);
            i--;
        }
    }

    for (int i = 0; i < surrounding_keyframes->size(); i++)
    {
        bool existing_flag = false;
        for (int j = 0; j < surrounding_existing_keyframes_id.size(); j++)
        {
            if (surrounding_existing_keyframes_id[j] == (int)surrounding_keyframes->points[i].intensity)
            {
                existing_flag = true;
                break;
            }
        }
        if (existing_flag)
        {
            continue;
        }
        else
        {
            int key_ind = (int)surrounding_keyframes->points[i].intensity;
            surrounding_existing_keyframes_id.push_back(key_ind);
            const Pose &pose_local = pose_keyframes_6d[key_ind].second;

            PointICovCloud::Ptr surf_trans(new PointICovCloud());
            cloudUCTAssociateToMap(*surf_cloud_keyframes_cov[key_ind], *surf_trans, pose_local, pose_ext);
            surrounding_surf_cloud_keyframes.push_back(surf_trans); //关键帧points转换到map下, 且计算cov, 依次存放起来

            PointICovCloud::Ptr corner_trans(new PointICovCloud());
            cloudUCTAssociateToMap(*corner_cloud_keyframes_cov[key_ind], *corner_trans, pose_local, pose_ext);
            surrounding_corner_cloud_keyframes.push_back(corner_trans);
        }
    }

    PointICloud::Ptr surrounding_existing_keyframes(new PointICloud());
    PointICloud::Ptr surrounding_existing_keyframes_ds(new PointICloud());
    for (int i = 0; i < surrounding_existing_keyframes_id.size(); i++)
    {
        int key_ind = surrounding_existing_keyframes_id[i];
        PointI point = pose_keyframes_3d->points[key_ind];
        point.intensity = i;
        surrounding_existing_keyframes->push_back(point);
    }
    down_size_filter_surrounding_keyframes.setInputCloud(surrounding_existing_keyframes);
    down_size_filter_surrounding_keyframes.filter(*surrounding_existing_keyframes_ds);
    for (int i = 0; i < surrounding_existing_keyframes_ds->size(); i++)
    {
        int j = (int)surrounding_existing_keyframes_ds->points[i].intensity;
        *laser_cloud_surf_from_map_cov += *surrounding_surf_cloud_keyframes[j]; //构建local map(map由关键帧构成)
        *laser_cloud_corner_from_map_cov += *surrounding_corner_cloud_keyframes[j];
    }

    common::timing::Timer filter_timer("mapping_filter");
    down_size_filter_surf_map_cov.setInputCloud(laser_cloud_surf_from_map_cov);
    down_size_filter_surf_map_cov.filter(*laser_cloud_surf_from_map_cov_ds);
    down_size_filter_corner_map_cov.setInputCloud(laser_cloud_corner_from_map_cov);
    down_size_filter_corner_map_cov.filter(*laser_cloud_corner_from_map_cov_ds); //local map降采样
    // laser_cloud_surf_from_map_cov_ds = laser_cloud_surf_from_map_cov;
    // laser_cloud_corner_from_map_cov_ds = laser_cloud_corner_from_map_cov;
    printf("corner/surf: before ds: %lu, %lu; after ds: %lu, %lu\n", 
           laser_cloud_corner_from_map_cov->size(), laser_cloud_surf_from_map_cov->size(),
           laser_cloud_corner_from_map_cov_ds->size(), laser_cloud_surf_from_map_cov_ds->size());
    printf("filter time: %fms\n", filter_timer.Stop() * 1000); // 10ms
}

void downsampleCurrentScan()
{
    laser_cloud_surf_last_ds->clear();
    down_size_filter_surf.setInputCloud(laser_cloud_surf_last);
    down_size_filter_surf.filter(*laser_cloud_surf_last_ds); //所有雷达curr帧，转到主雷达下

    laser_cloud_corner_last_ds->clear();
    down_size_filter_corner.setInputCloud(laser_cloud_corner_last);
    down_size_filter_corner.filter(*laser_cloud_corner_last_ds); //所有雷达curr帧，转到主雷达下 

    laser_cloud_outlier_ds->clear();
    down_size_filter_outlier.setInputCloud(laser_cloud_outlier);
    down_size_filter_outlier.filter(*laser_cloud_outlier_ds); //所有雷达curr帧，转到主雷达下 

    // propagate the extrinsic uncertainty on points
    laser_cloud_surf_cov->clear(); //所有雷达curr帧，转到主雷达下, 且计算每个点的cov
    laser_cloud_corner_cov->clear();
    laser_cloud_outlier_cov->clear();

    for (PointI &point_ori : *laser_cloud_surf_last_ds)
    {
        int idx = int(point_ori.intensity); // indicate the lidar id， 见visualization.cpp::transformCloudFeature()
        PointI point_sel;
        Eigen::Matrix3d cov_point = Eigen::Matrix3d::Zero();
        if (with_ua_flag)//true
        {
            pointAssociateToMap(point_ori, point_sel, pose_ext[idx].inverse()); //point_sel: 在n雷达curr帧下
            evalPointUncertainty(point_sel, cov_point, pose_ext[idx]); 
            //对于points中是在n雷达下观察到的points:
            //把主雷达到n雷达的外参cov和每个点(landmark)测量的cov一起考虑进去，计算得到这些points的cov
            //只有point的cov在阈值内才会被保留下来

            if (cov_point.trace() > TRACE_THRESHOLD_MAPPING) continue;
        }
        PointIWithCov point_cov(point_ori, cov_point.cast<float>());
        laser_cloud_surf_cov->push_back(point_cov);
    }

    for (PointI &point_ori : *laser_cloud_corner_last_ds)
    {
        int idx = int(point_ori.intensity); // indicate the lidar id
        PointI point_sel;
        Eigen::Matrix3d cov_point = Eigen::Matrix3d::Zero();
        if (with_ua_flag)//true
        {
            pointAssociateToMap(point_ori, point_sel, pose_ext[idx].inverse());
            evalPointUncertainty(point_sel, cov_point, pose_ext[idx]);
            if (cov_point.trace() > TRACE_THRESHOLD_MAPPING) continue;
        }
        PointIWithCov point_cov(point_ori, cov_point.cast<float>());
        laser_cloud_corner_cov->push_back(point_cov);
    }

    for (PointI &point_ori : *laser_cloud_outlier_ds)
    {
        int idx = int(point_ori.intensity); // indicate the lidar id
        PointI point_sel;
        Eigen::Matrix3d cov_point = Eigen::Matrix3d::Zero();
        if (with_ua_flag)//true
        {
            pointAssociateToMap(point_ori, point_sel, pose_ext[idx].inverse());
            evalPointUncertainty(point_sel, cov_point, pose_ext[idx]);
            if (cov_point.trace() > TRACE_THRESHOLD_MAPPING) continue;
        }
        PointIWithCov point_cov(point_ori, cov_point.cast<float>());
        laser_cloud_outlier_cov->push_back(point_cov);
    }
    std::cout << "input surf num: " << laser_cloud_surf_cov->size()
              << " corner num: " << laser_cloud_corner_cov->size() << std::endl;
}

void scan2MapOptimization()
{
    // step 4: perform scan-to-map optimization
    size_t laser_cloud_surf_from_map_num = laser_cloud_surf_from_map_cov_ds->size();
    size_t laser_cloud_corner_from_map_num = laser_cloud_corner_from_map_cov_ds->size();
    printf("map surf num: %lu, corner num: %lu\n", laser_cloud_surf_from_map_num, laser_cloud_corner_from_map_num);
    if ((laser_cloud_surf_from_map_num > 50) && (laser_cloud_corner_from_map_num > 10))
    {
        // pose_wmap_prev = pose_wmap_curr;
        common::timing::Timer t_timer("mapping_kdtree");
        kdtree_surf_from_map->setInputCloud(laser_cloud_surf_from_map_cov_ds);
        kdtree_corner_from_map->setInputCloud(laser_cloud_corner_from_map_cov_ds);
        printf("build time %fms\n", t_timer.Stop() * 1000);
        printf("********************************\n");

        // int max_iter = pose_keyframes_6d.size() <= 5 ? 5 : 2; // should have more iterations at the initial stage
        int max_iter = 2;
        for (int iter_cnt = 0; iter_cnt < max_iter; iter_cnt++) //TODO(jxl): 两轮ceres
        {
            ceres::Problem problem;
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            afs.loss_function_ = loss_function;
            vector2Double();

            std::vector<double *> para_ids;
            std::vector<ceres::internal::ResidualBlock *> res_ids_proj;
            PoseLocalParameterization *local_parameterization = new PoseLocalParameterization(); //自定义参数块
            local_parameterization->setParameter();
            problem.AddParameterBlock(para_pose, SIZE_POSE, local_parameterization);
            para_ids.push_back(para_pose);

            // ******************************************************
            // evaluate the full hessian matrix
            if (iter_cnt == 0)
            {
                if (frame_cnt % 10 == 0) //每隔10帧重新计算一次好points的比例 gf_ratio_cur
                {
                    int total_feat_num = 0;
                    Eigen::Matrix<double, 6, 6> mat_H = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
                    if (POINT_PLANE_FACTOR)
                        afs.evalFullHessian(kdtree_surf_from_map, *laser_cloud_surf_from_map_cov_ds,
                                            *laser_cloud_surf_cov, pose_wmap_curr, 's', mat_H, total_feat_num);
                        //mat_H：累加curr surf points每个点对curr pose的hessian矩阵

                    if (POINT_EDGE_FACTOR)
                        afs.evalFullHessian(kdtree_corner_from_map, *laser_cloud_corner_from_map_cov_ds,
                                            *laser_cloud_corner_cov, pose_wmap_curr, 'c', mat_H, total_feat_num);
                    // std::cout << mat_H << std::endl;
                    // std::cout << common::logDet(mat_H, true) << std::endl;
                    // std::cout << total_feat_num << " " << std::log(1.0 * total_feat_num) << std::endl;
                    // double gf_deg_factor = common::logDet(mat_H, true) - mat_H.rows() * std::log(1.0 * total_feat_num);
                    double gf_deg_factor = common::logDet(mat_H, true);
                    //TODO(jxl): J^T*J分解，这块打分的依据是什么？ https://github.com/gogojjh/M-LOAM/issues/10

                    gf_deg_factor_list.push_back(gf_deg_factor);
                    if (FLAGS_gf_method == "wo_gf") //feature选择方法
                    {
                        gf_ratio_cur = 1.0;
                    }
                    else if (FLAGS_gf_method == "rnd" || FLAGS_gf_method == "fps" || FLAGS_gf_method == "gd_fix")
                    {
                        gf_ratio_cur = FLAGS_gf_ratio_ini;
                    }
                    else if (FLAGS_gf_method == "gd_float")
                    {
                        if (gf_deg_factor > MAP_DEG_THRE)
                        {
                            gf_ratio_cur = FLAGS_gf_ratio_ini;
                        } 
                        else if (gf_deg_factor <= MAP_DEG_THRE)
                        {
                            gf_ratio_cur = 0.8;
                        }
                    }
                    std::cout << common::YELLOW << "gf_deg_factor: " << gf_deg_factor << ", gf_ratio: " << gf_ratio_cur << common::RESET << std::endl;
                }
            }

            // ******************************************************
            std::vector<PointPlaneFeature> all_surf_features, all_corner_features;
            std::vector<size_t> sel_surf_feature_idx, sel_corner_feature_idx;
            size_t surf_num = 0, corner_num = 0;
            common::timing::Timer gfs_timer("mapping_match_feat");
            Eigen::Matrix<double, 6, 6> sub_mat_H;  
            if (POINT_EDGE_FACTOR)
            {
                sub_mat_H = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
                afs.goodFeatureMatching(kdtree_corner_from_map,
                                        *laser_cloud_corner_from_map_cov_ds,
                                        *laser_cloud_corner_cov,
                                        pose_wmap_curr,
                                        all_corner_features, //all_corner_features[i]: index = i point对应的correspondance, 所有point都有, 如果有的话。
                                        sel_corner_feature_idx, //第i个好corner point在点云中的idx放到sel_corner_feature_idx[i]
                                        'c',
                                        FLAGS_gf_method,
                                        gf_ratio_cur, 
                                        sub_mat_H); //累加好points的残差对pose的雅克比
                corner_num = sel_corner_feature_idx.size();
            }
            if (POINT_PLANE_FACTOR)
            {
                sub_mat_H = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
                afs.goodFeatureMatching(kdtree_surf_from_map,
                                        *laser_cloud_surf_from_map_cov_ds,
                                        *laser_cloud_surf_cov,
                                        pose_wmap_curr,
                                        all_surf_features,
                                        sel_surf_feature_idx,
                                        's',
                                        FLAGS_gf_method,
                                        gf_ratio_cur, 
                                        sub_mat_H);
                surf_num = sel_surf_feature_idx.size();
            }
            gf_logdet_H_list.push_back(common::logDet(sub_mat_H, true));
            printf("matching features time: %fms\n", gfs_timer.Stop() * 1000);
            // printf("matching surf & corner num: %lu, %lu\n", surf_num, corner_num);
            
            //把好points的残差加入ceres
            for (const size_t &fid : sel_surf_feature_idx)
            {
                const PointPlaneFeature &feature = all_surf_features[fid];
                // if (feature.type_ == 'n') continue;
                Eigen::Matrix3d cov_matrix = Eigen::Matrix3d::Zero();
                if (with_ua_flag)
                    extractCov(laser_cloud_surf_cov->points[feature.idx_], cov_matrix);
                else 
                    cov_matrix = COV_MEASUREMENT;
                LidarMapPlaneNormFactor *f = new LidarMapPlaneNormFactor(feature.point_, feature.coeffs_, cov_matrix);
                ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f, loss_function, para_pose); //对当前帧在map下的初值进行refine
                res_ids_proj.push_back(res_id);
            }

            CHECK_JACOBIAN = 0;
            for (const size_t &fid : sel_corner_feature_idx)
            {
                const PointPlaneFeature &feature = all_corner_features[fid];
                // if (feature.type_ == 'n') continue;
                Eigen::Matrix3d cov_matrix = Eigen::Matrix3d::Zero();
                if (with_ua_flag)
                    extractCov(laser_cloud_corner_cov->points[feature.idx_], cov_matrix);
                else
                    cov_matrix = COV_MEASUREMENT;
                LidarMapEdgeFactor *f = new LidarMapEdgeFactor(feature.point_, feature.coeffs_, cov_matrix);
                ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f, loss_function, para_pose);
                res_ids_proj.push_back(res_id);
                if (CHECK_JACOBIAN)
                {
                    double **tmp_param = new double *[1];
                    tmp_param[0] = para_pose;
                    f->check(tmp_param);
                    CHECK_JACOBIAN = 0;
                }
            }
            // printf("add constraints: %fms\n", t_add_constraints.toc());

            // ******************************************************
            ceres::Problem::EvaluateOptions e_option;
            e_option.parameter_blocks = para_ids;
            e_option.residual_blocks = res_ids_proj;
            ceres::CRSMatrix jaco;
            problem.Evaluate(e_option, nullptr, nullptr, nullptr, &jaco);

            Eigen::Matrix<double, 6, 6> mat_H; // mat_H / 134 = normlized_mat_H
            evalHessian(jaco, mat_H); //所有残差的hessian
            evalDegenracy(mat_H, local_parameterization); // the hessian matrix is already normized to evaluate degeneracy
            //判断解是否发生退化

            // evalDegenracy(mat_H / 25, local_parameterization); // the hessian matrix is already normized to evaluate degeneracy

            // *********************************************************
            common::timing::Timer solver_timer("mapping_solver");
            ceres::Solver::Summary summary;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.max_num_iterations = 30; //TODO(jxl): 后端迭代30次
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            // options.update_state_every_iteration = false;
            // options.max_solver_time_in_seconds = 0.04;

            ceres::Solve(options, &problem, &summary); //求解时会考虑到是否发生退化
            std::cout << summary.BriefReport() << std::endl;
            printf("mapping solver time: %fms\n", solver_timer.Stop() * 1000);

            if (iter_cnt == max_iter - 1) //最后一次ceres
            {
                if (with_ua_flag)
                {
                    common::timing::Timer eval_deg_timer("mapping_eval_deg");
                    problem.Evaluate(e_option, nullptr, nullptr, nullptr, &jaco);
                    evalHessian(jaco, mat_H);
                    if (pose_keyframes_6d.size() <= 10)
                        cov_mapping.setZero();
                    else
                        cov_mapping = (mat_H).inverse();

                    double tr = cov_mapping.trace();
                    std::vector<double> sp{tr};
                    mapping_sp_list.push_back(sp);
                    LOG_EVERY_N(INFO, 20) << "trace: " << tr;
                    std::cout << common::YELLOW << "trace: " << tr << common::RESET << std::endl;
                    printf("evaluate H: %fms\n", eval_deg_timer.Stop() * 1000);
                }
                else
                {
                    cov_mapping.setZero();
                } 
                // if (MLOAM_RESULT_SAVE && frame_cnt == 500)
                //     afs.writeFeature(*laser_cloud_surf_cov, sel_surf_feature_idx, all_surf_features);                
                afs.pubFeature(*laser_cloud_surf_cov, sel_surf_feature_idx, all_surf_features, pub_good_surf_feature, time_laser_odometry);
            }

            double2Vector();
            printf("-------------------------------------\n");
        }
        std::cout << "optimization result: " << pose_wmap_curr << std::endl;
        pose_wmap_curr.cov_ = cov_mapping; //当前帧在map下的位姿 

        //不用ceres接口来直接计算cov是因为，作者这的参数块(t, q)是用一个double raw pointer来表示，不像ceres example中是用两个double raw pointer来表示。
        //http://ceres-solver.org/nnls_covariance.html
    }
    else
    {
        std::cout << "Map surf num is not enough" << std::endl;
        pose_wmap_curr.cov_.setZero();
    }
}

void saveKeyframe()
{
    pose_point_cur.x = pose_wmap_curr.t_[0];
    pose_point_cur.y = pose_wmap_curr.t_[1];
    pose_point_cur.z = pose_wmap_curr.t_[2];
    q_ori_cur = pose_wmap_curr.q_;

    save_new_keyframe = false;
    if (sqrt((pose_point_cur.x - pose_point_prev.x) * (pose_point_cur.x - pose_point_prev.x)
           + (pose_point_cur.y - pose_point_prev.y) * (pose_point_cur.y - pose_point_prev.y) 
           + (pose_point_cur.z - pose_point_prev.z) * (pose_point_cur.z - pose_point_prev.z)) > DISTANCE_KEYFRAMES ||
        q_ori_cur.angularDistance(q_ori_prev) / M_PI * 180 > ORIENTATION_KEYFRAMES || 
        pose_keyframes_6d.size() == 0) //后端接收的第一帧scan
    {
        save_new_keyframe = true;
    }

    if (!save_new_keyframe) return;
    pose_point_prev = pose_point_cur;
    q_ori_prev = q_ori_cur;

    PointI pose_3d;
    pose_3d.x = pose_wmap_curr.t_[0];
    pose_3d.y = pose_wmap_curr.t_[1];
    pose_3d.z = pose_wmap_curr.t_[2];
    pose_3d.intensity = pose_keyframes_3d->size(); //关键帧位姿的intensity: keyframe index

    pose_keyframes_3d->push_back(pose_3d);
    pose_keyframes_6d.push_back(std::make_pair(time_laser_odometry, pose_wmap_curr));

    PointICovCloud::Ptr surf_keyframe_cov(new PointICovCloud());
    PointICovCloud::Ptr corner_keyframe_cov(new PointICovCloud());
    PointICovCloud::Ptr outlier_keyframe_cov(new PointICovCloud());

    pcl::copyPointCloud(*laser_cloud_surf_cov, *surf_keyframe_cov);
    pcl::copyPointCloud(*laser_cloud_corner_cov, *corner_keyframe_cov);
    pcl::copyPointCloud(*laser_cloud_outlier_cov, *outlier_keyframe_cov);

    surf_cloud_keyframes_cov.push_back(surf_keyframe_cov);
    corner_cloud_keyframes_cov.push_back(corner_keyframe_cov);
    outlier_cloud_keyframes_cov.push_back(outlier_keyframe_cov);
    printf("current keyframes size: %lu\n", pose_keyframes_3d->size());
}

void updateKeyframe()
{
    std::cout << common::YELLOW << "received loop info, need to update all keyframes" << common::RESET << std::endl;  
}

void pubPointCloud()
{
    // publish registrated laser cloud
    *laser_cloud_full_res += *laser_cloud_outlier;
    for (PointI &point : *laser_cloud_full_res) pointAssociateToMap(point, point, pose_wmap_curr); //转换到map下
    sensor_msgs::PointCloud2 laser_cloud_full_res_msg;
    pcl::toROSMsg(*laser_cloud_full_res, laser_cloud_full_res_msg);
    laser_cloud_full_res_msg.header.stamp = ros::Time().fromSec(time_laser_odometry);
    laser_cloud_full_res_msg.header.frame_id = "/world";
    pub_laser_cloud_full_res.publish(laser_cloud_full_res_msg);

    for (PointI &point : *laser_cloud_surf_last) pointAssociateToMap(point, point, pose_wmap_curr);
    sensor_msgs::PointCloud2 laser_cloud_surf_last_msg;
    pcl::toROSMsg(*laser_cloud_surf_last, laser_cloud_surf_last_msg);
    laser_cloud_surf_last_msg.header.stamp = ros::Time().fromSec(time_laser_odometry);
    laser_cloud_surf_last_msg.header.frame_id = "/world";
    pub_laser_cloud_surf_last_res.publish(laser_cloud_surf_last_msg);

    for (PointI &point : *laser_cloud_corner_last) pointAssociateToMap(point, point, pose_wmap_curr);
    sensor_msgs::PointCloud2 laser_cloud_corner_last_msg;
    pcl::toROSMsg(*laser_cloud_corner_last, laser_cloud_corner_last_msg);
    laser_cloud_corner_last_msg.header.stamp = ros::Time().fromSec(time_laser_odometry);
    laser_cloud_corner_last_msg.header.frame_id = "/world";
    pub_laser_cloud_corner_last_res.publish(laser_cloud_corner_last_msg);
}

void pubOdometry()
{
    // step 5: publish odom
    nav_msgs::Odometry odom_aft_mapped;
    odom_aft_mapped.header.stamp = ros::Time().fromSec(time_laser_odometry);
    odom_aft_mapped.header.frame_id = "/world";
    odom_aft_mapped.child_frame_id = "/aft_mapped";
    odom_aft_mapped.pose.pose.orientation.x = pose_wmap_curr.q_.x();
    odom_aft_mapped.pose.pose.orientation.y = pose_wmap_curr.q_.y();
    odom_aft_mapped.pose.pose.orientation.z = pose_wmap_curr.q_.z();
    odom_aft_mapped.pose.pose.orientation.w = pose_wmap_curr.q_.w();
    odom_aft_mapped.pose.pose.position.x = pose_wmap_curr.t_.x();
    odom_aft_mapped.pose.pose.position.y = pose_wmap_curr.t_.y();
    odom_aft_mapped.pose.pose.position.z = pose_wmap_curr.t_.z();
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 6; j++)
            odom_aft_mapped.pose.covariance[i * 6 + j] = float(pose_wmap_curr.cov_(i, j));
    pub_odom_aft_mapped.publish(odom_aft_mapped); 

    geometry_msgs::PoseStamped laser_after_mapped_pose;
    laser_after_mapped_pose.header = odom_aft_mapped.header;
    laser_after_mapped_pose.header.stamp = ros::Time().fromSec(time_laser_odometry);
    laser_after_mapped_pose.pose = odom_aft_mapped.pose.pose;

    laser_after_mapped_path.header.stamp = ros::Time().fromSec(time_laser_odometry);
    laser_after_mapped_path.header.frame_id = "/world";
    laser_after_mapped_path.poses.push_back(laser_after_mapped_pose);
    pub_laser_after_mapped_path.publish(laser_after_mapped_path);
    publishTF(odom_aft_mapped);

    // publish 3d keyframes
    if (pub_keyframes.getNumSubscribers() != 0 && save_new_keyframe)
    {
        sensor_msgs::PointCloud2 keyframes_msg;
        pcl::toROSMsg(*pose_keyframes_3d, keyframes_msg);
        keyframes_msg.header.stamp = ros::Time().fromSec(time_laser_odometry);
        keyframes_msg.header.frame_id = "/world";
        pub_keyframes.publish(keyframes_msg);
    }

    // publish 6d keyframes with covariance
    // if (pub_keyframes_6d.getNumSubscribers() != 0 && save_new_keyframe)
    if (save_new_keyframe)
    {
        const std::pair<double, Pose> &pkf = pose_keyframes_6d.back();
        geometry_msgs::PoseWithCovarianceStamped laser_keyframes_pose;
        laser_keyframes_pose.header.stamp = ros::Time().fromSec(pkf.first);
        laser_keyframes_pose.header.frame_id = "/world";
        laser_keyframes_pose.pose.pose.position.x = pkf.second.t_.x();
        laser_keyframes_pose.pose.pose.position.y = pkf.second.t_.y();
        laser_keyframes_pose.pose.pose.position.z = pkf.second.t_.z();
        laser_keyframes_pose.pose.pose.orientation.x = pkf.second.q_.x();
        laser_keyframes_pose.pose.pose.orientation.y = pkf.second.q_.y();
        laser_keyframes_pose.pose.pose.orientation.z = pkf.second.q_.z();
        laser_keyframes_pose.pose.pose.orientation.w = pkf.second.q_.w();
        for (size_t i = 0; i < 6; i++)
            for (size_t j = 0; j < 6; j++)
                laser_keyframes_pose.pose.covariance[i * 6 + j] = float(pkf.second.cov_(i, j));
        laser_keyframes_6d.poses.push_back(laser_keyframes_pose);
        laser_keyframes_6d.header = laser_keyframes_pose.header;
        pub_keyframes_6d.publish(laser_keyframes_6d);
    }
}

void pubGlobalMap()
{
    ros::Rate rate(0.5);
    while (ros::ok())
    {
        rate.sleep();
        if (pub_laser_cloud_surrounding.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 laser_cloud_surround_msg;
            pcl::toROSMsg(*laser_cloud_surf_from_map_cov_ds + *laser_cloud_corner_from_map_cov_ds, laser_cloud_surround_msg);
            // pcl::toROSMsg(*laser_cloud_surf_from_map_cov_ds, laser_cloud_surround_msg);
            laser_cloud_surround_msg.header.stamp = ros::Time().fromSec(time_laser_odometry);
            laser_cloud_surround_msg.header.frame_id = "/world";
            pub_laser_cloud_surrounding.publish(laser_cloud_surround_msg);
        }

        if ((pub_laser_cloud_map.getNumSubscribers() != 0) && (!pose_keyframes_3d->points.empty()))
        {
            global_map_keyframes->clear();
            global_map_keyframes_ds->clear();

            std::vector<int> point_search_ind;
            std::vector<float> point_search_sq_dis;

            kdtree_global_map_keyframes->setInputCloud(pose_keyframes_3d);            
            kdtree_global_map_keyframes->radiusSearch(pose_point_cur, (double)GLOBALMAP_KF_RADIUS, point_search_ind, point_search_sq_dis, 0);

            for (int i = 0; i < point_search_ind.size(); i++)
                global_map_keyframes->points.push_back(pose_keyframes_3d->points[point_search_ind[i]]);

            down_size_filter_global_map_keyframes.setInputCloud(global_map_keyframes);
            down_size_filter_global_map_keyframes.filter(*global_map_keyframes_ds);

            PointICovCloud::Ptr laser_cloud_map(new PointICovCloud());
            PointICovCloud::Ptr laser_cloud_map_ds(new PointICovCloud());
            for (int i = 0; i < global_map_keyframes_ds->size(); i++)
            {
                int key_ind = (int)global_map_keyframes_ds->points[i].intensity;
                PointICovCloud surf_trans;
                cloudUCTAssociateToMap(*surf_cloud_keyframes_cov[key_ind], surf_trans, pose_keyframes_6d[key_ind].second, pose_ext);
                *laser_cloud_map += surf_trans;

                PointICovCloud corner_trans;
                cloudUCTAssociateToMap(*corner_cloud_keyframes_cov[key_ind], corner_trans, pose_keyframes_6d[key_ind].second, pose_ext);
                *laser_cloud_map += corner_trans;

                PointICovCloud outlier_trans;
                cloudUCTAssociateToMap(*outlier_cloud_keyframes_cov[key_ind], outlier_trans, pose_keyframes_6d[key_ind].second, pose_ext);
                *laser_cloud_map += outlier_trans;
            }

            // if ((abs(pose_wmap_curr.t_.x()) >= 1000) || (abs(pose_wmap_curr.t_.y()) >= 1000) || (abs(pose_wmap_curr.t_.z()) >= 1000))
            // {
            //     down_size_filter_global_map_cov.setLeafSize(2.0, 2.0, 2.0);
            //     down_size_filter_global_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
            // } else
            // {
            //     down_size_filter_global_map_cov.setLeafSize(0.8, 0.8, 0.8);
            //     down_size_filter_global_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
            // }
            down_size_filter_global_map_cov.setLeafSize(MAP_SURF_RES, MAP_SURF_RES, MAP_SURF_RES);
            down_size_filter_global_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
            down_size_filter_global_map_cov.setInputCloud(laser_cloud_map);
            down_size_filter_global_map_cov.filter(*laser_cloud_map_ds);

            sensor_msgs::PointCloud2 laser_cloud_msg;
            pcl::toROSMsg(*laser_cloud_map_ds, laser_cloud_msg);
            laser_cloud_msg.header.stamp = ros::Time().fromSec(time_laser_odometry);
            laser_cloud_msg.header.frame_id = "/world";
            pub_laser_cloud_map.publish(laser_cloud_msg);
        }
    }
}

void saveGlobalMap()
{
    std::cout << common::YELLOW << "Saving keyframe poses & map cloud (corner + surf) /tmp/mloam_*.pcd" << common::RESET << std::endl;
    pcd_writer.write("/tmp/mloam_mapping_keyframes.pcd", *pose_keyframes_3d);
    
    PointICovCloud::Ptr laser_cloud_map(new PointICovCloud());
    PointICovCloud::Ptr laser_cloud_surf_map(new PointICovCloud());
    PointICovCloud::Ptr laser_cloud_surf_map_ds(new PointICovCloud());
    PointICovCloud::Ptr laser_cloud_corner_map(new PointICovCloud());
    PointICovCloud::Ptr laser_cloud_corner_map_ds(new PointICovCloud());

    printf("global keyframes num: %lu\n", pose_keyframes_3d->size());
    for (size_t i = 0; i < pose_keyframes_3d->size(); i++)
        global_map_keyframes->points.push_back(pose_keyframes_3d->points[i]);

    down_size_filter_global_map_keyframes.setInputCloud(global_map_keyframes);
    down_size_filter_global_map_keyframes.filter(*global_map_keyframes_ds);
    for (int i = 0; i < global_map_keyframes_ds->size(); i++)
    {
        int key_ind = (int)global_map_keyframes_ds->points[i].intensity;
        PointICovCloud surf_trans;
        cloudUCTAssociateToMap(*surf_cloud_keyframes_cov[key_ind], surf_trans, pose_keyframes_6d[key_ind].second, pose_ext);
        *laser_cloud_surf_map += surf_trans;

        PointICovCloud outlier_trans;
        cloudUCTAssociateToMap(*outlier_cloud_keyframes_cov[key_ind], outlier_trans, pose_keyframes_6d[key_ind].second, pose_ext);
        *laser_cloud_surf_map += outlier_trans;

        PointICovCloud corner_trans;
        cloudUCTAssociateToMap(*corner_cloud_keyframes_cov[key_ind], corner_trans, pose_keyframes_6d[key_ind].second, pose_ext);
        *laser_cloud_corner_map += corner_trans;
    }

    // adpatively change the resolution of the global map by checking the range
    // if ((abs(pose_wmap_curr.t_.x()) >= 1000) || (abs(pose_wmap_curr.t_.y()) >= 1000) || (abs(pose_wmap_curr.t_.z()) >= 1000))
    // {
    //     down_size_filter_global_map_cov.setLeafSize(2.0, 2.0, 2.0);
    //     down_size_filter_global_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    // }
    // else
    // {
    //     down_size_filter_global_map_cov.setLeafSize(0.8, 0.8, 0.8);
    //     down_size_filter_global_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    // }
    down_size_filter_global_map_cov.setLeafSize(MAP_SURF_RES * 2, MAP_SURF_RES * 2, MAP_SURF_RES * 2);
    down_size_filter_global_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    down_size_filter_global_map_cov.setInputCloud(laser_cloud_surf_map);
    down_size_filter_global_map_cov.filter(*laser_cloud_surf_map_ds);
    down_size_filter_global_map_cov.setInputCloud(laser_cloud_corner_map);
    down_size_filter_global_map_cov.filter(*laser_cloud_corner_map_ds);

    if (with_ua_flag)
    {
        pcd_writer.write("/tmp/mloam_mapping_surf_cloud.pcd", *laser_cloud_surf_map_ds);
        pcd_writer.write("/tmp/mloam_mapping_corner_cloud.pcd", *laser_cloud_corner_map_ds);
        *laser_cloud_map += *laser_cloud_surf_map_ds;
        *laser_cloud_map += *laser_cloud_corner_map_ds;
        pcd_writer.write("/tmp/mloam_mapping_cloud.pcd", *laser_cloud_map);
    }
    else
    {
        pcd_writer.write("/tmp/mloam_mapping_surf_cloud_wo_ua.pcd", *laser_cloud_surf_map_ds);
        pcd_writer.write("/tmp/mloam_mapping_corner_cloud_wo_ua.pcd", *laser_cloud_corner_map_ds);
        *laser_cloud_map += *laser_cloud_surf_map_ds;
        *laser_cloud_map += *laser_cloud_corner_map_ds;
        pcd_writer.write("/tmp/mloam_mapping_cloud_wo_ua.pcd", *laser_cloud_map);
    }
}

void clearCloud()
{
    laser_cloud_surf_from_map_cov->clear(); //如果当前帧是个keyframe, 在末尾清空，为下次keyframe做准备
    laser_cloud_corner_from_map_cov->clear();
    laser_cloud_surf_from_map_cov_ds->clear();
    laser_cloud_corner_from_map_cov_ds->clear();
}

void process()
{
	while (1)
	{
		if (!ros::ok()) break;
		while (!surf_last_buf.empty() && !corner_last_buf.empty() &&
               !outlier_buf.empty() && !full_res_buf.empty() &&
               !ext_buf.empty() && !odometry_buf.empty())
		{
			//********************* * 100******************************************************
			// step 1: pop up subscribed data
			m_buf.lock();
			while (!corner_last_buf.empty() && corner_last_buf.front()->header.stamp.toSec() < surf_last_buf.front()->header.stamp.toSec())
				corner_last_buf.pop();
			if (corner_last_buf.empty())
			{
				m_buf.unlock();
				break;
			}

			while (!full_res_buf.empty() && full_res_buf.front()->header.stamp.toSec() < surf_last_buf.front()->header.stamp.toSec())
				full_res_buf.pop();
			if (full_res_buf.empty())
			{
				m_buf.unlock();
				break;
			}

			while (!outlier_buf.empty() && outlier_buf.front()->header.stamp.toSec() < surf_last_buf.front()->header.stamp.toSec())
				outlier_buf.pop();
			if (outlier_buf.empty())
			{
				m_buf.unlock();
				break;
			}            

			while (!odometry_buf.empty() && odometry_buf.front()->header.stamp.toSec() < surf_last_buf.front()->header.stamp.toSec())
				odometry_buf.pop();
			if (odometry_buf.empty())
			{
				m_buf.unlock();
				break;
			}

			while (!ext_buf.empty() && ext_buf.front()->header.stamp.toSec() < surf_last_buf.front()->header.stamp.toSec())
				ext_buf.pop();
			if (ext_buf.empty())
			{
				m_buf.unlock();
				break;
			}

			time_laser_cloud_surf_last = surf_last_buf.front()->header.stamp.toSec();
			time_laser_cloud_corner_last = corner_last_buf.front()->header.stamp.toSec();
			time_laser_cloud_full_res = full_res_buf.front()->header.stamp.toSec();
            time_laser_cloud_outlier = outlier_buf.front()->header.stamp.toSec();
            time_laser_odometry = odometry_buf.front()->header.stamp.toSec();
			time_ext = ext_buf.front()->header.stamp.toSec();

            if (std::abs(time_laser_cloud_surf_last - time_laser_cloud_corner_last) > 0.005 ||
                std::abs(time_laser_cloud_surf_last - time_laser_cloud_full_res) > 0.005 ||
                std::abs(time_laser_cloud_surf_last - time_laser_cloud_outlier) > 0.005 ||
                std::abs(time_laser_cloud_surf_last - time_laser_odometry) > 0.005 ||
                std::abs(time_laser_cloud_surf_last - time_ext) > 0.005)
            {
                printf("time surf: %f, corner: %f, full: %f, outlier: %f, odom: %f\n, ext: %f\n",
                       time_laser_cloud_surf_last, time_laser_cloud_corner_last,
                       time_laser_cloud_full_res, time_laser_cloud_outlier,
                       time_laser_odometry, time_ext);
                printf("unsync messeage!");
				m_buf.unlock();
				break;
			}

            //消息是同步的

			laser_cloud_surf_last->clear();
			pcl::fromROSMsg(*surf_last_buf.front(), *laser_cloud_surf_last);
			surf_last_buf.pop();

			laser_cloud_corner_last->clear();
			pcl::fromROSMsg(*corner_last_buf.front(), *laser_cloud_corner_last);
			corner_last_buf.pop();

			laser_cloud_full_res->clear();
			pcl::fromROSMsg(*full_res_buf.front(), *laser_cloud_full_res);
			full_res_buf.pop();

            laser_cloud_outlier->clear();
            pcl::fromROSMsg(*outlier_buf.front(), *laser_cloud_outlier);
            outlier_buf.pop();

            pose_wodom_curr.q_ = Eigen::Quaterniond(odometry_buf.front()->pose.pose.orientation.w,
													odometry_buf.front()->pose.pose.orientation.x,
													odometry_buf.front()->pose.pose.orientation.y,
													odometry_buf.front()->pose.pose.orientation.z);
			pose_wodom_curr.t_ = Eigen::Vector3d(odometry_buf.front()->pose.pose.position.x,
												 odometry_buf.front()->pose.pose.position.y,
												 odometry_buf.front()->pose.pose.position.z);
            //curr帧在odom位姿
			odometry_buf.pop();

			extrinsics = *ext_buf.front();
			if (!extrinsics.status)//visualization.cpp#L223, 最后都转为了0
			{
				std::cout << common::YELLOW << "Accurate extrinsic calibration!" << common::RESET << std::endl;
				for (size_t n = 0; n < NUM_OF_LASER; n++)
				{
					pose_ext[n].q_ = Eigen::Quaterniond(extrinsics.odoms[n].pose.pose.orientation.w,
														extrinsics.odoms[n].pose.pose.orientation.x,
														extrinsics.odoms[n].pose.pose.orientation.y,
														extrinsics.odoms[n].pose.pose.orientation.z);
					pose_ext[n].t_ = Eigen::Vector3d(extrinsics.odoms[n].pose.pose.position.x,
													 extrinsics.odoms[n].pose.pose.position.y,
													 extrinsics.odoms[n].pose.pose.position.z);
					for (size_t i = 0; i < 6; i++)
						for (size_t j = 0; j < 6; j++)
							pose_ext[n].cov_(i, j) = double(extrinsics.odoms[n].pose.covariance[i * 6 + j]);
				}
			}
			ext_buf.pop();

            while (!surf_last_buf.empty()) //TODO(jxl): 一直pop，直到为空
            {
				surf_last_buf.pop();
                frame_drop_cnt++;
				std::cout << common::GREEN << "drop lidar frame in mapping for real time performance" << common::RESET << std::endl;
			}
			m_buf.unlock();
			
			// if (extrinsics.status) continue; // calibration is not finish
            std::lock_guard<std::mutex> lock(m_process);

			frame_cnt++;
			common::timing::Timer process_timer("mapping_process");

			transformAssociateToMap(); //结合当前帧在odom位姿和之前计算的T_map_odom, 预测当前帧在map下位姿

            common::timing::Timer extract_kf_timer("mapping_extract_kf");
            extractSurroundingKeyFrames(); //构建local map,计算每个点的cov
            printf("extract surrounding keyframes: %fms\n", extract_kf_timer.Stop() * 1000);

            common::timing::Timer dscs_timer("mapping_dscs");
            downsampleCurrentScan(); //对curr点云降采样，计算点的cov(传播外参的cov到点的cov)
            // printf("downsample current scan time: %fms\n", t_dscs.toc());

            common::timing::Timer opti_timer("mapping_opti");
            scan2MapOptimization(); //laser残差(挑选出的好point的残差)，对curr帧在map下位姿refine
            printf("optimization time: %fms\n", opti_timer.Stop() * 1000);

			transformUpdate(); //更新T_map_odom

            common::timing::Timer skf_timer("mapping_save_kf");
            saveKeyframe(); //保存关键帧pose，和相应的surf, corner, outlier points
            printf("save keyframes time: %fms\n", skf_timer.Stop() * 1000);

            // TODO: using loop info to update keyframes
            if (!loop_info_buf.empty())
            {
                while (loop_info_buf.size() != 1)
                {
                    loop_info_buf.pop();
                }
                loop_info = *loop_info_buf.front();
                if (loop_info.status)
                {
                    updateKeyframe(); //TODO(jxl): 还没有实现完闭环检测
                }
                loop_info_buf.pop();
            }

            pubPointCloud();

            pubOdometry();

            if (save_new_keyframe) clearCloud(); // if save new keyframe, clear the map point cloud

            double process_time = process_timer.Stop() * 1000;
            std::cout << common::RED << "frame: " << frame_cnt
                      << ", whole mapping time: " << process_time << "ms" << common::RESET << std::endl;
            LOG_EVERY_N(INFO, 20) << "whole mapping time " << process_time << "ms";

            // std::cout << "pose_wmap_curr: " << pose_wmap_curr << std::endl;
			printf("\n");
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}

//把点转换到map下，根据点的cov和点所在位姿的cov计算转换到map下后的cov
//只有cov的迹满足一定要求才能加入到local map中, 为后面scan-local_map-match做准备
void cloudUCTAssociateToMap(const PointICovCloud &cloud_local, //关键帧points
                            PointICovCloud &cloud_global, //[out]关键帧points转换到map下, 且计算cov
                            const Pose &pose_global,  //关键帧位姿
                            const vector<Pose> &pose_ext) //外参
{
    // the compound pose: pose_global * pose_ext with uncertainty
    std::vector<Pose> pose_compound(NUM_OF_LASER);
    for (size_t n = 0; n < NUM_OF_LASER; n++) 
    {
        compoundPoseWithCov(pose_global, pose_ext[n], pose_compound[n]); //pose_compound[n]：计算关键帧时刻n号雷达的位姿和cov
        // if (n == IDX_REF) continue;
        // std::cout << "pose global: " << pose_global << std::endl;
        // std::cout << pose_global.cov_ << std::endl;
        // std::cout << "pose ext: " << pose_ext[n] << std::endl;
        // std::cout << pose_ext[n].cov_ << std::endl;
        // std::cout << "pose compound: " << pose_compound[n] << ", "
        //           << pose_compound[n].cov_.trace() << std::endl;
        // std::cout << pose_compound[n].cov_ << std::endl;
        // std::cout << std::endl;
    }
    // exit(EXIT_FAILURE);

    cloud_global.clear();
    cloud_global.resize(cloud_local.size());
    size_t cloud_size = 0;
    for (const PointIWithCov &point_ori : cloud_local)
    {
        int ind = (int)point_ori.intensity; //雷达index, 见downsampleCurrentScan()
        PointIWithCov point_sel, point_cov;
        Eigen::Matrix3d cov_point = Eigen::Matrix3d::Zero();
        if (with_ua_flag) //true
        {
            pointAssociateToMap(point_ori, point_sel, pose_ext[ind].inverse()); //point_sel: 在n雷达下
            evalPointUncertainty(point_sel, cov_point, pose_compound[ind]); //根据点的cov和雷达位姿的cov，计算转到map下point的cov
            if (cov_point.trace() > TRACE_THRESHOLD_MAPPING) continue;
        }
        pointAssociateToMap(point_ori, point_cov, pose_global);//point_cov：关键帧在map下points
        updateCov(point_cov, cov_point); //把计算的cov赋给point_cov
        cloud_global[cloud_size] = point_cov;
        cloud_size++;
    }
    cloud_global.resize(cloud_size);
}

void evalHessian(const ceres::CRSMatrix &jaco, Eigen::Matrix<double, 6, 6> &mat_H)
{
	// printf("jacob: %d constraints, %d parameters\n", jaco.num_rows, jaco.num_cols); // 2000+, 6
	if (jaco.num_rows == 0) return;
	Eigen::SparseMatrix<double, Eigen::RowMajor> mat_J; // Jacobian is a diagonal matrix
	CRSMatrix2EigenMatrix(jaco, mat_J);
	Eigen::SparseMatrix<double, Eigen::RowMajor> mat_Jt = mat_J.transpose();
	Eigen::MatrixXd mat_JtJ = mat_Jt * mat_J;
	mat_H = mat_JtJ.block(0, 0, 6, 6);  // normalized the hessian matrix for pair uncertainty evaluation
}



// TODO: still have some bugs
void evalDegenracy(const Eigen::Matrix<double, 6, 6> &mat_H, PoseLocalParameterization *local_parameterization)
{
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6> > esolver(mat_H);
	Eigen::Matrix<double, 1, 6> mat_E = esolver.eigenvalues().real();	// 6*1， 特征值从小到大
	Eigen::Matrix<double, 6, 6> mat_V_f = esolver.eigenvectors().real(); // 6*6, column is the corresponding eigenvector
	Eigen::Matrix<double, 6, 6> mat_V_p = mat_V_f;  //和Estimator::evalDegenracy()一样
	for (size_t j = 0; j < mat_E.cols(); j++)
	{
		if (mat_E(0, j) < MAP_EIG_THRE)
		{
			mat_V_p.col(j) = Eigen::Matrix<double, 6, 1>::Zero();
			local_parameterization->is_degenerate_ = true;
		}
		else
		{
			break;
		}
	}
	d_factor_list.push_back(mat_E);
	d_eigvec_list.push_back(mat_V_f);
 	mat_P = mat_V_f.transpose().inverse() * mat_V_p.transpose(); // 6*6
    // std::cout << "D factor: " << mat_E(0, 0) << std::endl;
    LOG(INFO) << "D factor: " << mat_E(0, 0) << ", D vector: " << mat_V_f.col(0).transpose();
	if (local_parameterization->is_degenerate_)
	{
		local_parameterization->V_update_ = mat_P;
		// std::cout << "param " << i << " is degenerate !" << std::endl;
		// std::cout << mat_P.transpose() << std::endl;
	} else
	{
		is_degenerate = false;
	}
}

void sigintHandler(int sig)
{
    printf("[lidar_mapper] press ctrl-c\n");
    std::cout << common::YELLOW << "mapping drop frame: " << frame_drop_cnt << common::RESET << std::endl;
    if (MLOAM_RESULT_SAVE)
    {
        save_statistics.saveMapStatistics(MLOAM_MAP_PATH,
                                          OUTPUT_FOLDER + "others/mapping_gf_deg_factor_" + FLAGS_gf_method + "_" + std::to_string(FLAGS_gf_ratio_ini) + ".txt",
                                          OUTPUT_FOLDER + "others/mapping_gf_logdet_H_" + FLAGS_gf_method + "_" + std::to_string(FLAGS_gf_ratio_ini) + ".txt",
                                          laser_after_mapped_path,
                                          gf_deg_factor_list,
                                          gf_logdet_H_list);
        if (with_ua_flag)                                          
            save_statistics.saveMapTimeStatistics(OUTPUT_FOLDER + "time/time_mloam_mapping_" + FLAGS_gf_method + "_" + std::to_string(FLAGS_gf_ratio_ini) + ".txt");
        else
            save_statistics.saveMapTimeStatistics(OUTPUT_FOLDER + "time/time_mloam_mapping_wo_ua_" + FLAGS_gf_method + "_" + std::to_string(FLAGS_gf_ratio_ini) + ".txt");
    }
    saveGlobalMap();
    ros::shutdown();
}

int main(int argc, char **argv)
{
	// if (argc < 5)
	// {
	// 	printf("please intput: rosrun mloam lidar_mapper [args] \n"
	// 		   "for example: "
	// 		   "rosrun mloam lidar_mapper config_file 1 output_path 1 \n");
	// 	return 1;
	// }
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);

	ros::init(argc, argv, "lidar_mapper");
	ros::NodeHandle nh;

    MLOAM_RESULT_SAVE = FLAGS_result_save;
    OUTPUT_FOLDER = FLAGS_output_path;
	with_ua_flag = FLAGS_with_ua;
    printf("save result (0/1): %d to %s\n", MLOAM_RESULT_SAVE, OUTPUT_FOLDER.c_str());
	printf("with the awareness of uncertainty (0/1): %d\n", with_ua_flag);
    printf("gf method: %s, gf ratio: %f\n", FLAGS_gf_method.c_str(), FLAGS_gf_ratio_ini);
    gf_ratio_cur = std::min(1.0, FLAGS_gf_ratio_ini);

	if (with_ua_flag) //true
        MLOAM_MAP_PATH = OUTPUT_FOLDER + "traj/stamped_mloam_map_estimate_" + FLAGS_gf_method + "_" + to_string(FLAGS_gf_ratio_ini) + ".txt";
    else
        MLOAM_MAP_PATH = OUTPUT_FOLDER + "traj/stamped_mloam_map_wo_ua_estimate_" + FLAGS_gf_method + "_" + to_string(FLAGS_gf_ratio_ini) + ".txt";

    std::cout << "config file: " << FLAGS_config_file << std::endl;
	readParameters(FLAGS_config_file);
	printf("Mapping as %fhz\n", 1.0 / (SCAN_PERIOD * SKIP_NUM_ODOM_PUB));

	ros::Subscriber sub_laser_cloud_full_res = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud", 10, laserCloudFullResHandler); //所有雷达curr points, 转到主雷达下 
    ros::Subscriber sub_laser_cloud_outlier = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_outlier", 10, laserCloudOutlierResHandler); //所有雷达curr outlier points, 包含没有形成聚类的points, 转到主雷达下 
    ros::Subscriber sub_laser_cloud_surf_last = nh.subscribe<sensor_msgs::PointCloud2>("/surf_points_less_flat", 10, laserCloudSurfLastHandler); //所有雷达curr surf_less_flat points, 转到主雷达下 
	ros::Subscriber sub_laser_cloud_corner_last = nh.subscribe<sensor_msgs::PointCloud2>("/corner_points_less_sharp", 10, laserCloudCornerLastHandler); //所有雷达curr corner_less_sharp points, 转到主雷达下
	ros::Subscriber sub_laser_odometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom", 10, laserOdometryHandler); //remapped, "laser_odom_0"
	ros::Subscriber sub_extrinsic = nh.subscribe<mloam_msgs::Extrinsics>("/extrinsics", 10, extrinsicsHandler); //外参
    ros::Subscriber sub_loop_info = nh.subscribe<mloam_msgs::Keyframes>("/loop_info", 10, loopInfoHandler); //mloam_loop模块

	pub_laser_cloud_full_res = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_registered", 5); //每一帧在map下points，包含了outliers(未聚类的points)
	pub_laser_cloud_surf_last_res = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_registered", 5); //每一帧surf在map下points
	pub_laser_cloud_corner_last_res = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_registered", 5);//每一帧corner在map下points
	pub_laser_cloud_surrounding = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 5); //local map
	pub_laser_cloud_map = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 5); //发布的是距离当前帧GLOBALMAP_KF_RADIUS以内的所有关键帧组成的map
    pub_good_surf_feature = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_good", 5);

	pub_odom_aft_mapped = nh.advertise<nav_msgs::Odometry>("/laser_map", 5); // raw pose from odometry in the world   curr主雷达在map下位姿
	pub_odom_aft_mapped_high_frec = nh.advertise<nav_msgs::Odometry>("/laser_map_high_frec", 5); // optimized pose in the world
	pub_laser_after_mapped_path = nh.advertise<nav_msgs::Path>("/laser_map_path", 5); //主雷达在map下path
    pub_keyframes = nh.advertise<sensor_msgs::PointCloud2>("/laser_map_keyframes", 5); //在map下所有keyframes位置
    pub_keyframes_6d = nh.advertise<mloam_msgs::Keyframes>("/laser_map_keyframes_6d", 5); //在map下所有keyframes pose

    down_size_filter_surf.setLeafSize(MAP_SURF_RES, MAP_SURF_RES, MAP_SURF_RES);
    down_size_filter_surf.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    down_size_filter_corner.setLeafSize(MAP_CORNER_RES, MAP_CORNER_RES, MAP_CORNER_RES);
    down_size_filter_corner.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    down_size_filter_outlier.setLeafSize(MAP_OUTLIER_RES, MAP_OUTLIER_RES, MAP_OUTLIER_RES);
    down_size_filter_outlier.setTraceThreshold(TRACE_THRESHOLD_MAPPING);    

    down_size_filter_surf_map_cov.setLeafSize(MAP_SURF_RES, MAP_SURF_RES, MAP_SURF_RES);
    down_size_filter_surf_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    down_size_filter_corner_map_cov.setLeafSize(MAP_CORNER_RES, MAP_CORNER_RES, MAP_CORNER_RES);
    down_size_filter_corner_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    down_size_filter_outlier_map_cov.setLeafSize(MAP_OUTLIER_RES, MAP_OUTLIER_RES, MAP_OUTLIER_RES);
    down_size_filter_outlier_map_cov.setTraceThreshold(TRACE_THRESHOLD_MAPPING);
    down_size_filter_surrounding_keyframes.setLeafSize(MAP_SUR_KF_RES, MAP_SUR_KF_RES, MAP_SUR_KF_RES);
    down_size_filter_global_map_keyframes.setLeafSize(10, 10, 10);

    cov_mapping.setZero();

    pose_ext.resize(NUM_OF_LASER);

    pose_point_prev.x = 0.0;
    pose_point_prev.y = 0.0;
    pose_point_prev.z = 0.0;
    q_ori_prev.setIdentity();

    pose_point_cur.x = 0.0;
    pose_point_cur.y = 0.0;
    pose_point_cur.z = 0.0;
    q_ori_cur.setIdentity();

    pose_keyframes_6d.clear();
    pose_keyframes_3d->clear();
    laser_keyframes_6d.poses.clear();

    signal(SIGINT, sigintHandler);

    std::thread mapping_process{process}; //入口
    std::thread pub_map_process{pubGlobalMap};

    ros::Rate loop_rate(100);
	while (ros::ok()) 
	{
		ros::spinOnce();
		loop_rate.sleep();
    }

    pub_map_process.detach();
    mapping_process.join();
    return 0;
}



