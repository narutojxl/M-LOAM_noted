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

#include <vector>
#include <queue>
#include <map>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <ros/console.h>

#include "common/types/type.h"
#include "common/algos/math.hpp"

#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"

// maintain a priority_queue
// the left (first) should have larger w (higher_priority) than the right (second)
struct rotCmp
{
	bool operator()(const std::pair<size_t, std::vector<Pose> > &pose_r, const std::pair<size_t, std::vector<Pose> > &pose_l)
	{
		return (pose_l.second[0].q_.w() > pose_r.second[0].q_.w());  //主雷达q.w比较
	}
};

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialExtrinsics
{
public:
	InitialExtrinsics();
	void clearState();
	void setParameter();

	bool addPose(const std::vector<Pose> &pose_laser);

	bool calibExRotation(const size_t &idx_ref, const size_t &idx_data, Pose &calib_result);
	bool calibExTranslation(const size_t &idx_ref, const size_t &idx_data, Pose &calib_result);
	bool calibExTranslationPlanar(const size_t &idx_ref, const size_t &idx_data);
	bool calibExTranslationNonPlanar(const size_t &idx_ref, const size_t &idx_data);
	void calibTimeDelay(const size_t &idx_ref, const size_t &idx_data);

	bool setCovRotation(const size_t &idx);
	bool setCovTranslation(const size_t &idx);

	bool checkScrewMotion(const Pose &pose_ref, const Pose &pose_data);
	void saveStatistics();

	void decomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2, cv::Mat_<double> &t1, cv::Mat_<double> &t2);

	std::vector<Pose> calib_ext_; //主雷达到各个雷达的外参

	std::vector<double> v_rd_;
	std::vector<double> v_td_;

	std::vector<std::vector<double> > v_rot_cov_, v_pos_cov_;
	std::vector<bool> cov_rot_state_, cov_pos_state_; //主雷达到n雷达的旋转是否标定收敛， 到n雷达的平移是否标定收敛
	bool full_cov_rot_state_, full_cov_pos_state_; //是否主雷达到所有副雷达的旋转标定收敛， 到所有副雷达的平移标定收敛
	double rot_cov_thre_;

	std::priority_queue<std::pair<size_t, std::vector<Pose> >,  //元素：pair类型，<size_t = k，std::vector<Pose>>, 和v_pose_[k]一一对应
						std::vector<std::pair<size_t, std::vector<Pose> > >, //容器：std::vector
						rotCmp> //比较函数
						pq_pose_;
	std::vector<std::vector<Pose> > v_pose_; //v_pose_[k][]: k帧时刻，所有雷达k-1帧到k帧的delta_T
		
	// v_pose_[idx_ref][indices_[idx_data][i]], v_pose_[idx_data][indices_[idx_data][i]] as the screw motion pair
	std::vector<std::vector<int> > indices_;

	size_t frame_cnt_, pose_cnt_;

	std::vector<Eigen::MatrixXd> Q_; //eq16. Q，大小为2，每个矩阵4K *4

	std::pair<size_t, std::vector<Pose> > pose_laser_add_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
