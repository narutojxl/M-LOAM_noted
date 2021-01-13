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

#include "marginalization_factor.h"


//*************************************************************************
// 和lio-mapping完全一样。
// 注意：lili-om和lio-mapping有些地方稍微不一样，用beyond compare软件查看。
//*************************************************************************



void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);  //计算每个残差项，以及残差对优化变量的雅克比

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    if (loss_function) //如果指定了loss函数，对残差和雅克比进行修正
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

/********************************************************
 * MarginalizationInfo
 * *******************************************************/
MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");

    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;

        delete factors[i]->cost_function;

        delete factors[i];
    }
}

//给parameter_block_size， parameter_block_idx填充内容
//第一次边缘化时，只添加跟marg帧相关的imu预积分残差和laser残差
//后面边缘化时，还会添加上次边缘化后产生的残差
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks; //这个残差的参数块
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes(); //每个参数块的大小

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i]; //每个参数块的地址
        int size = parameter_block_sizes[i]; //对应参数块的大小
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

//给parameter_block_data填充内容
void MarginalizationInfo::preMarginalize()
{
    for (auto it : factors) //factors: marg对象的所有ResidualBlockInfos
    {
        it->Evaluate(); //计算每个残差项，以及残差对优化变量的雅克比
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end()) // such parameter block exists or not?
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data; //每个参数块的地址，以及指向参数的double*
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors) //每一个残差项
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])]; //该参数块的索引
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]; //该参数块的大小
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j; //累加
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j; //累加
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;  //累加
        }
    }
    return threadsstruct;
}

void MarginalizationInfo::marginalize()
{
    int pos = 0;
    for (auto &it : parameter_block_idx)
    {
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }
    m = pos; // size of marginalized states
    // std::printf("parameter_block_size = %d, parameter_block_idx = %d\n", parameter_block_size.size(), parameter_block_idx.size());

    for (const auto &it : parameter_block_size) //<参数块的地址, 每个参数块的大小>
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos; //parameter_block_idx中没有的参数块，把该参数块的地址和索引添加进来
            pos += localSize(it.second);
        }
    }
    n = pos - m; // size of optimized states  //n: 剩下状态的参数块维数之和； pos：所有优化变量的维数之和
    //ROS_INFO("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    if (m == 0)
    {
        valid = false;
        printf("unstable tracking...\n");
        return;
    }

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread


    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
	
    for (auto it : factors) //factors: marg对象的所有ResidualBlockInfo //把所有的残差项均匀分到threadsstruct[]数组中
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size; //<每个参数块的地址, 每个参数块的大小>
        threadsstruct[i].parameter_block_idx = parameter_block_idx;  //<每个参数块的地址, 在矩阵中的id> 
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        //对每个threadsstruct[i]创建一个线程，每个线程计算一部分残差
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)
    {
        pthread_join( tids[i], NULL );
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    //TODO
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose()); //A的左上角矩阵，保证Amm是对称矩阵
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm); //特征值分解计算实对称矩阵的逆

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //特征值 > thresh, 取特征值的逆； 否则，取0
	//printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    //jxl: 见FEJ(First Estimate Jacobians) paper: Consistency Analysis for Sliding-Window Visual Odometry
    Eigen::VectorXd bmm = b.segment(0, m); //b最上面m*1
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
	
    // remaining part
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    // decompose A,b as Jacobian using the Eigenvalue decomposition
	
	//舒尔补, 消去X_m，剩下X_(r+n) 
    //得到 A * X_(r+n) =  b， 边缘化pivot帧后，得到的对剩余状态的约束方程
    //其中 A = J^T * J, A是一个实对称矩阵。 J = Σ每个残差项ri对X_(r+n)的雅克比
    //其中 b = J^T * r, r = Σ每个残差项。
    //marg掉滑窗内的第一个位姿, 会产生一个残差,下次优化时要把上次边缘化产生的残差考虑进来.

    //FEJ:
    //由于剩余状态Xn还在滑窗内,在下次迭代优化时会被继续优化.
    //当引入新的观测后，这些Xn状态量是会发生变化的，此时就和边缘化时的Xn状态量不一样了，
    //从而导致边缘化时产生的关于Xn的雅可比和残差项都发生了变化,导致H矩阵的N(H)空间秩发生了变化,引入了错误的信息.
    //在每次优化迭代时：
    //对于状态Xn，残差对Xn的雅克比一直是边缘化时Xn的值带入到雅克比表达式中
    //但是状态Xn每次都在更新，只是说Xn的线性化点是固定不变的。
    //更新后的残差 r_new = r_old + J*dx, dx为Xn当前状态与边缘化时状态量的差.
   

    //对marg后的H矩阵进行特征值分解:
    //对A进行特征值分解 A = J^T * J = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose() 
    //J = sqrt(eigenvalues.asDiagonal()) * eigenvectors.transpose()
	
    // A = J^{T}J, b = J^{T}b
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A); //实对称矩阵特征值分解 A = V [\] V^T, V.inv() =V.trans() 
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));  //特征值
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0)); //特征值逆

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    //FEJ,以后计算关于先验的残差和Jacobian都在边缘化的这个线性点展开
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();  
	//J = [sqrt(λ1)...sqrt(λi)...sqrt(λn)].diag() * V^T , J可逆
    //边缘化产生的残差r
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b; 
	//r = [1/sqrt(λ1) ... 1/sqrt(λi) ... 1/sqrt(λn)].diag() * V^T * b 
	//r = (J^T).inv() * b
	
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();  //保存X_(r+n)每个参数块的大小
    keep_block_idx.clear();   //保存X_(r+n)每个参数块的索引，通过索引或地址可以找到该参数块
    keep_block_data.clear();  //保存X_(r+n)每个参数块的数据， 这三个变量一一对应

    for (const auto &it : parameter_block_idx) //<每个参数块的地址, 在矩阵中的id> 
    {
        if (it.second >= m) //只对X_(r+n)的参数块
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

/********************************************************
 * MarginalizationFactor
 * *******************************************************/
//先验残差项
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size) //上一次marg后剩下状态X_(r+n)的参数块
    {
        mutable_parameter_block_sizes()->push_back(it); //每个参数块的大小
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n); // set number of residuals  //n: 上一次marg后剩下状态X_(r+n)的参数块维数之和
};


//利用边缘化得到的雅可比和残差，构造先验信息进入到非线性优化中
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n); // dx = x(update) - x0(initial)  表示当前的状态量和边缘化时的状态量之间的差值
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) //对于上一次marg后剩下状态X_(r+n)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size); //上次边缘化时被保留状态Xn在当前优化中的值
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size); //边缘化时，被保留的状态量的值
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // TODO: the marginalization error
    // ||r+J*dx|| = r^{T}r + dx^{T}*J^{T}J*dx + dx^{T}*J^{T}r = b^{T}H^{+}b + dx^{T}*H*dx + 2dx^{T}*b (fit the algorithm)
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx; 
    //更新边缘化留下的残差
	
	// DLOG(INFO) << "linearized_residuals: " << marginalization_info->linearized_residuals.norm();
    // DLOG(INFO) << "linearized_residuals size: " << marginalization_info->linearized_residuals.rows();
    // DLOG(INFO) << "dr: " << (marginalization_info->linearized_jacobians * dx).norm();
    if (jacobians)
    {
        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size); // using Jacobian representation is good to update
            }
        }
    }
    return true;
}
