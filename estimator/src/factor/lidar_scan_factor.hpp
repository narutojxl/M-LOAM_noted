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

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"

// ****************************************************************
// calculate distrance from point to plane (using normal)
class LidarScanPlaneNormFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    LidarScanPlaneNormFactor(const Eigen::Vector3d &point,
                             const Eigen::Vector4d &coeff,
                             const double &s = 1.0)
        : point_(point), coeff_(coeff), s_(s) {}

    bool Evaluate(double const *const *param, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
        double d = coeff_(3);
        double a = w.dot(q_last_curr * point_ + t_last_curr) + d; //残差(点到平面的距离)有正有负。没关系，ceres计算的是残差模的平方
        residuals[0] = a;

        if (jacobians)
        {
            Eigen::Matrix3d R = q_last_curr.toRotationMatrix();
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]); //(r,t)是一个整体
                Eigen::Matrix<double, 1, 6> jaco; // [dy/dt, dy/dR, 1]
                jaco.setZero();
                jaco.leftCols<3>() = w.transpose();
                jaco.rightCols<3>() = -w.transpose() * R * Utility::skewSymmetric(point_); 
                //m-loam paper equ.38
                //Supplementary Material to: Greedy-Based Feature Selection for Efficient LiDAR SLAM  equ.7

                jacobian_pose.setZero();
                jacobian_pose.leftCols<6>() = jaco;
            }
        }
        return true;
    }

    // TODO: check if derived jacobian == perturbation on the raw function
    void check(double **param)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        Evaluate(param, res, jaco);
        std::cout << "[LidarScanPlaneNormFactor] check begins" << std::endl;
        std::cout << "analytical:" << std::endl;

        std::cout << res[0] << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[0]) << std::endl;

        delete[] jaco[0];
        delete[] jaco;
        delete[] res;

        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
        double d = coeff_(3);
        double a = w.dot(q_last_curr * point_ + t_last_curr) + d;
        double r = a;

        std::cout << "perturbation:" << std::endl;
        std::cout << r << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 6> num_jacobian;

        // add random perturbation
        for (int k = 0; k < 6; k++)
        {
            Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
            Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
            q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
            t_last_curr = s_ * t_last_curr;

            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
            if (a == 0)
                t_last_curr += delta;
            else if (a == 1)
                q_last_curr = q_last_curr * Utility::deltaQ(delta);

            Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
            double d = coeff_(3);
            double v = w.dot(q_last_curr * point_ + t_last_curr) + d;
            double tmp_r = v;
            num_jacobian(0, k) = (tmp_r - r) / eps;
        }
        std::cout << num_jacobian << std::endl;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    Eigen::Vector3d point_;
    Eigen::Vector4d coeff_;
    double s_; //1
};

// ****************************************************************
// calculate distrance from point to edge
class LidarScanEdgeFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    LidarScanEdgeFactor(const Eigen::Vector3d &point,
                        const Eigen::VectorXd &coeff,
                        const double &s = 1.0)
        : point_(point), coeff_(coeff), s_(s) {}

    bool Evaluate(double const *const *param, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        Eigen::Vector3d de = lpa - lpb;
        residuals[0] = nu.norm() / de.norm(); //点到直线的距离

        if (jacobians)
        {
            Eigen::Matrix3d R = q_last_curr.toRotationMatrix();
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                Eigen::Matrix<double, 1, 6> jaco; // [dy/dt, dy/dq, 1]

                Eigen::Matrix<double, 1, 3> eta = 1.0 / de.norm() * nu.normalized().transpose();
                jaco.leftCols<3>() = -eta * Utility::skewSymmetric(lpa - lpb);
                jaco.rightCols<3>() = eta * Utility::skewSymmetric(lpa - lpb) * R * Utility::skewSymmetric(point_); //右扰动
                //见Supplementary Material to: Greedy-Based Feature Selection for Efficient LiDAR SLAM  equ.6

                jacobian_pose.setZero();
                jacobian_pose.leftCols<6>() = jaco;
            }
        }
        return true;
    }

    void check(double **param)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        Evaluate(param, res, jaco);
        std::cout << "[LidarScanEdgeFactor] check begins" << std::endl;
        std::cout << "analytical:" << std::endl;
        std::cout << res[0] << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[0]) << std::endl;

        delete[] jaco[0];
        delete[] jaco;
        delete[] res;

        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);

        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        Eigen::Vector3d de = lpa - lpb;
        double r = nu.norm() / de.norm();

        std::cout << "perturbation:" << std::endl;
        std::cout << r << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 6> num_jacobian;

        // add random perturbation
        for (int k = 0; k < 6; k++)
        {
            Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
            Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
            if (a == 0)
                t_last_curr += delta;
            else if (a == 1)
                q_last_curr = q_last_curr * Utility::deltaQ(delta);

            Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
            Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
            Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

            Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
            Eigen::Vector3d de = lpa - lpb;
            double tmp_r = nu.norm() / de.norm();
            num_jacobian(k) = (tmp_r - r) / eps;
        }
        std::cout << num_jacobian << std::endl;
    }

private:
    const Eigen::Vector3d point_;
    const Eigen::VectorXd coeff_;
    const double s_;
};

// ****************************************************************
// calculate distrance from point to edge (using 3*1 vector)
class LidarScanEdgeFactorVector : public ceres::SizedCostFunction<3, 7>
{
public:
    LidarScanEdgeFactorVector(const Eigen::Vector3d &point,
                              const Eigen::VectorXd &coeff,
                              const double &s = 1.0)
        : point_(point), coeff_(coeff), s_(s) {}

    bool Evaluate(double const *const *param, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
        q_last_curr = Eigen::Quaterniond::Identity().slerp(s_, q_last_curr);
        t_last_curr = s_ * t_last_curr;

        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        Eigen::Vector3d de = lpa - lpb;
        residuals[0] = nu.x() / de.norm();
        residuals[1] = nu.y() / de.norm();
        residuals[2] = nu.z() / de.norm();
        // residuals[0] = nu.norm / de.norm();

        if (jacobians)
        {
            Eigen::Matrix3d R = q_last_curr.toRotationMatrix();
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                Eigen::Matrix<double, 3, 6> jaco; // [dy/dt, dy/dq, 1]

                double eta = 1.0 / de.norm();
                jaco.leftCols<3>() = -eta * Utility::skewSymmetric(lpa - lpb);
                jaco.rightCols<3>() = eta * Utility::skewSymmetric(lpa - lpb) * R * Utility::skewSymmetric(point_);
                //类比点到直线的距离的雅克比

                jacobian_pose.setZero();
                jacobian_pose.leftCols<6>() = jaco;
            }
        }
        return true;
    }

    void check(double **param)
    {
        double *res = new double[3];
        double **jaco = new double *[1];
        jaco[0] = new double[3 * 7];
        Evaluate(param, res, jaco);
        std::cout << "[LidarScanEdgeFactor] check begins" << std::endl;
        std::cout << "analytical:" << std::endl;
        std::cout << res[0] << " " << res[1] << " " << res[2] << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>>(jaco[0]) << std::endl;

        delete[] jaco[0];
        delete[] jaco;
        delete[] res;

        Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);

        Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
        Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
        Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

        Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
        Eigen::Vector3d de = lpa - lpb;
        Eigen::Vector3d r = nu / de.norm();

        std::cout << "perturbation:" << std::endl;
        std::cout << r.transpose() << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 3, 6> num_jacobian;

        // add random perturbation
        for (int k = 0; k < 6; k++)
        {
            Eigen::Quaterniond q_last_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
            Eigen::Vector3d t_last_curr(param[0][0], param[0][1], param[0][2]);
            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
            if (a == 0)
                t_last_curr += delta;
            else if (a == 1)
                q_last_curr = q_last_curr * Utility::deltaQ(delta);

            Eigen::Vector3d lpa(coeff_(0), coeff_(1), coeff_(2));
            Eigen::Vector3d lpb(coeff_(3), coeff_(4), coeff_(5));
            Eigen::Vector3d lp = q_last_curr * point_ + t_last_curr;

            Eigen::Vector3d nu = (lp - lpa).cross(lp - lpb);
            Eigen::Vector3d de = lpa - lpb;
            Eigen::Vector3d tmp_r = nu / de.norm();
            num_jacobian.col(k) = (tmp_r - r) / eps;
        }
        std::cout << num_jacobian.block<1, 6>(0, 0) << std::endl;
        std::cout << num_jacobian.block<1, 6>(1, 0) << std::endl;
        std::cout << num_jacobian.block<1, 6>(2, 0) << std::endl;
    }

private:
    const Eigen::Vector3d point_;
    const Eigen::VectorXd coeff_;
    const double s_; //1
};
