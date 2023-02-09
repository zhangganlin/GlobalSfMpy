
#include "GSfM_nonlinear_rotation_estimator.hpp"

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <memory>
#include <unordered_map>

#include "theia/util/hash.h"
#include "theia/util/map_util.h"
#include "theia/sfm/global_pose_estimation/pairwise_rotation_error.h"
#include "theia/sfm/types.h"

#include <ceres/rotation.h>
#include "gamma_values.cpp"

#include "pairwise_rotation_error_quat.hpp"

namespace theia
{

    bool GSfMNonlinearRotationEstimator::EstimateRotations(
        const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
        std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations)
    {
        CHECK_NOTNULL(global_orientations);
        if (global_orientations->size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "initialization was provivded.";
            return false;
        }
        if (view_pairs.size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "relative rotation constraints were provivded.";
            return false;
        }

        // Set up the problem and loss function.
        std::unique_ptr<ceres::Problem> problem(new ceres::Problem());
        ceres::LossFunction *loss_function =
            new ceres::SoftLOneLoss(robust_loss_width_);

        for (const auto &view_pair : view_pairs)
        {
            const ViewIdPair &view_id_pair = view_pair.first;
            Eigen::Vector3d *rotation1 =
                FindOrNull(*global_orientations, view_id_pair.first);
            Eigen::Vector3d *rotation2 =
                FindOrNull(*global_orientations, view_id_pair.second);

            // Do not add the relative rotation constaint if it requires an orientation
            // that we do not have an initialization for.
            if (rotation1 == nullptr || rotation2 == nullptr)
            {
                continue;
            }

            ceres::CostFunction *cost_function =
                PairwiseRotationError::Create(view_pair.second.rotation_2, 1.0);
            problem->AddResidualBlock(cost_function,
                                      loss_function,
                                      rotation1->data(),
                                      rotation2->data());
        }

        // The problem should be relatively sparse so sparse cholesky is a good
        // choice.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;

        ceres::Solver::Summary summary;
        ceres::Solve(options, problem.get(), &summary);
        VLOG(1) << summary.FullReport();
        return true;
    }

    bool GSfMNonlinearRotationEstimator::EstimateRotationsWithCustomizedLoss(
        const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
        std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations,
        ceres::LossFunction *loss_function, int thread_num, 
        RotationErrorType rotation_error_type)
    {
        CHECK_NOTNULL(global_orientations);
        if (global_orientations->size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "initialization was provivded.";
            return false;
        }
        if (view_pairs.size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "relative rotation constraints were provivded.";
            return false;
        }

        // Set up the problem and loss function.
        ceres::Problem::Options ceres_options;
        // Because the loss function is constructed by Python, ceres should not destruct it.
        ceres_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        std::unique_ptr<ceres::Problem> problem(new ceres::Problem(ceres_options));

        std::unordered_map<theia::ViewId,Eigen::Quaterniond> temp_quat_rots;

        for (const auto &view_pair : view_pairs)
        {
            const ViewIdPair &view_id_pair = view_pair.first;
            Eigen::Vector3d *rotation1 =
                FindOrNull(*global_orientations, view_id_pair.first);
            Eigen::Vector3d *rotation2 =
                FindOrNull(*global_orientations, view_id_pair.second);

            // Do not add the relative rotation constaint if it requires an orientation
            // that we do not have an initialization for.
            if (rotation1 == nullptr || rotation2 == nullptr)
            {
                continue;
            }

            double cost_weight = 1.0;
            double q1[4];
            double q2[4];
            double q_relative[4];
            
            ceres::AngleAxisToQuaternion(rotation1->data(),q1);
            ceres::AngleAxisToQuaternion(rotation2->data(),q2);
            ceres::AngleAxisToQuaternion(view_pair.second.rotation_2.data(),q_relative);

            Eigen::Quaterniond q1_xyzw(q1[0],q1[1],q1[2],q1[3]);
            Eigen::Quaterniond q2_xyzw(q2[0],q2[1],q2[2],q2[3]);
            Eigen::Quaterniond q_relative_xyzw(q_relative[0],q_relative[1],q_relative[2],q_relative[3]);


            temp_quat_rots.insert(std::make_pair(view_id_pair.first,q1_xyzw));
            temp_quat_rots.insert(std::make_pair(view_id_pair.second,q2_xyzw));

            Eigen::Quaterniond* quat1 = FindOrNull(temp_quat_rots,view_id_pair.first);
            Eigen::Quaterniond* quat2 = FindOrNull(temp_quat_rots,view_id_pair.second);

            ceres::CostFunction *cost_function = 0;

            if(rotation_error_type == RotationErrorType::ROTATION_MAT_FNORM){
                cost_function = PairwiseRotationErrorRotFNorm::Create(q_relative_xyzw, cost_weight);
            } else if(rotation_error_type == RotationErrorType::QUATERNION_NORM){
                cost_function = PairwiseRotationErrorQuatFNorm::Create(q_relative_xyzw, cost_weight);
            } else if(rotation_error_type == RotationErrorType::QUATERNION_COSINE){
                cost_function = PairwiseRotationErrorQuat::Create(q_relative_xyzw, cost_weight);
            }

            problem->AddResidualBlock(cost_function,
                                      loss_function,
                                      quat1->coeffs().data(),
                                      quat2->coeffs().data());

            // cost_function = PairwiseRotationError::Create(view_pair.second.rotation_2,cost_weight);
            // problem->AddResidualBlock(cost_function,
            //                           loss_function,
            //                           rotation1->data(),
            //                           rotation2->data());

            
        }
        ceres::LocalParameterization* quat_parameterization = new ceres::EigenQuaternionParameterization();
        for(auto& kv_pair: temp_quat_rots){
            Eigen::Quaterniond& quat = kv_pair.second;
            problem->SetParameterization(quat.coeffs().data(),quat_parameterization);
        }

        // The problem should be relatively sparse so sparse cholesky is a good
        // choice.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;
        options.num_threads = thread_num;

        ceres::Solver::Summary summary;
        ceres::Solve(options, problem.get(), &summary);
        VLOG(1) << summary.FullReport();

        for(auto& kv_pair:temp_quat_rots){
            Eigen::Quaterniond& q = kv_pair.second;
            double quat[4] = {q.w(),q.x(),q.y(),q.z()};
            Eigen::Vector3d *rotation =
                FindOrNull(*global_orientations, kv_pair.first);
            if (rotation == nullptr){
                continue;
            }
            ceres::QuaternionToAngleAxis(quat,rotation->data());
        }


        return true;
    }


    bool GSfMNonlinearRotationEstimator::EstimateRotationsWithCustomizedLossAndCovariance(
        const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
        std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations,
        ceres::LossFunction *loss_function, int thread_num, 
        CovarianceMap covariances,
        RotationErrorType rotation_error_type, Reconstruction* reconstruction)
    {
        CHECK_NOTNULL(global_orientations);
        if (global_orientations->size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "initialization was provivded.";
            return false;
        }
        if (view_pairs.size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "relative rotation constraints were provivded.";
            return false;
        }

        // Set up the problem and loss function.
        ceres::Problem::Options ceres_options;
        // Because the loss function is constructed by Python, ceres should not destruct it.
        ceres_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        std::unique_ptr<ceres::Problem> problem(new ceres::Problem(ceres_options));

        for (const auto &view_pair : view_pairs)
        {
            const ViewIdPair &view_id_pair = view_pair.first;
            Eigen::Vector3d *rotation1 =
                FindOrNull(*global_orientations, view_id_pair.first);
            Eigen::Vector3d *rotation2 =
                FindOrNull(*global_orientations, view_id_pair.second);
            std::pair<Eigen::Matrix3d,Eigen::Vector3d> * covariance_rot = 
                FindOrNull(covariances,view_id_pair);
            // Do not add the relative rotation constaint if it requires an orientation
            // that we do not have an initialization for.
            if (rotation1 == nullptr || rotation2 == nullptr || 
                (covariance_rot == nullptr && rotation_error_type == RotationErrorType::ANGLE_AXIS_COVARIANCE) ||
                (covariance_rot == nullptr && rotation_error_type == RotationErrorType::ANGLE_AXIS_COV_INLIERS)||
                (covariance_rot == nullptr && rotation_error_type == RotationErrorType::ANGLE_AXIS_COVTRACE)||
                (covariance_rot == nullptr && rotation_error_type == RotationErrorType::ANGLE_AXIS_COVNORM))

            {
                continue;
            }


            ceres::CostFunction* cost_function = nullptr;
            if(rotation_error_type == RotationErrorType::ANGLE_AXIS_COVARIANCE){
                Eigen::Matrix3d cov = (covariance_rot->first)*1e8;
                Eigen::Matrix3d cov_inv = cov.inverse();
                Eigen::Matrix3d Lt;
                Lt = cov_inv.llt().matrixL().transpose();
                cost_function = PairwiseRotationErrorAngleAxis::Create(view_pair.second.rotation_2,Lt);
            }else if(rotation_error_type == RotationErrorType::ANGLE_AXIS){ 
                double weight = 1.0;
                cost_function = PairwiseRotationError::Create(view_pair.second.rotation_2,weight);
            }else if(rotation_error_type == RotationErrorType::ANGLE_AXIS_INLIERS){ 
                MatchedFeatures features;
                get_matched_features(view_id_pair,*reconstruction,features);
                double weight = features.first.size()/100.0;
                cost_function = PairwiseRotationError::Create(view_pair.second.rotation_2,weight);
            }else if(rotation_error_type == RotationErrorType::ANGLE_AXIS_COV_INLIERS){ 
                MatchedFeatures features;
                get_matched_features(view_id_pair,*reconstruction,features);
                double weight = features.first.size()/100.0;
                Eigen::Matrix3d cov = (covariance_rot->first)*1e8;
                Eigen::Matrix3d cov_inv = cov.inverse();
                Eigen::Matrix3d Lt;
                Lt = cov_inv.llt().matrixL().transpose();
                Lt = Lt*weight;
                cost_function = PairwiseRotationErrorAngleAxis::Create(view_pair.second.rotation_2,Lt);
            }else if(rotation_error_type == RotationErrorType::ANGLE_AXIS_COVTRACE){ 
                Eigen::Matrix3d cov = (covariance_rot->first)*1e8;
                double trace = cov.trace();
                // if (trace <=0.1){
                //     std::cout << trace << std::endl;
                // }
                double weight = sqrt(1.0/trace);
                cost_function = PairwiseRotationError::Create(view_pair.second.rotation_2,weight);
            }else if(rotation_error_type == RotationErrorType::ANGLE_AXIS_COVNORM){ 
                Eigen::Matrix3d cov = (covariance_rot->first)*1e8;
                double fnorm = cov.norm();
                double weight = sqrt(1.0/fnorm);
                cost_function = PairwiseRotationError::Create(view_pair.second.rotation_2,weight);
            }

            problem->AddResidualBlock(cost_function,
                                      loss_function,
                                      rotation1->data(),
                                      rotation2->data());
            
        }

        // The problem should be relatively sparse so sparse cholesky is a good
        // choice.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 200;
        options.num_threads = thread_num;

        ceres::Solver::Summary summary;
        ceres::Solve(options, problem.get(), &summary);
        VLOG(1) << summary.FullReport();

        return true;
    }




    bool GSfMNonlinearRotationEstimator::EstimateRotationsWithSigmaConsensus(
        const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
        std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations,
        ceres::LossFunction *loss_function, int thread_num, int iters_num, double sigma_max)
    {
        CHECK_NOTNULL(global_orientations);
        if (global_orientations->size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "initialization was provivded.";
            return false;
        }
        if (view_pairs.size() == 0)
        {
            LOG(INFO) << "Skipping nonlinear rotation optimization because no "
                         "relative rotation constraints were provivded.";
            return false;
        }

        // Set up the problem and loss function.
        ceres::Problem::Options ceres_options;
        // Because the loss function is constructed by Python, ceres should not destruct it.
        ceres_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;


        std::unique_ptr<ceres::Problem> problem;
        // (new ceres::Problem(ceres_options));

        
        double squared_sigma_max_2 = sigma_max * sigma_max * 2.0;
        double dof_minus_one_per_two = (nu3 - 1.0) / 2.0;
        double C_times_two_ad_dof = C3 * std::pow(2.0, dof_minus_one_per_two);
        double one_over_sigma = C_times_two_ad_dof / sigma_max;
        double gamma_value = tgamma(dof_minus_one_per_two);
        double gamma_difference = gamma_value - upper_incomplete_gamma_of_k3;
        double weight_zero = one_over_sigma * gamma_difference;

        int view_pair_num = view_pairs.size();
        std::vector<double> last_weights(view_pair_num,0.0);
        std::vector<double> weights(view_pair_num,0.0);

        int iter_num = iters_num;
        for(int i = 0; i < iter_num; i++){

            problem.reset(new ceres::Problem(ceres_options));

            int view_pair_idx = -1;
            for (const auto &view_pair : view_pairs)
            {   
                view_pair_idx ++;
                const ViewIdPair &view_id_pair = view_pair.first;
                Eigen::Vector3d *rotation1 =
                    FindOrNull(*global_orientations, view_id_pair.first);
                Eigen::Vector3d *rotation2 =
                    FindOrNull(*global_orientations, view_id_pair.second);

                // Do not add the relative rotation constaint if it requires an orientation
                // that we do not have an initialization for.
                if (rotation1 == nullptr || rotation2 == nullptr)
                {
                    continue;
                }
                double weight = 0.0;

                Eigen::Matrix<double, 3, 3> rotation1_mat, rotation2_mat;
                Eigen::Matrix3d relative_rotation_mat;
                ceres::AngleAxisToRotationMatrix(
                    rotation1->data(), ceres::ColumnMajorAdapter3x3(rotation1_mat.data()));
                ceres::AngleAxisToRotationMatrix(
                    rotation2->data(), ceres::ColumnMajorAdapter3x3(rotation2_mat.data()));
                ceres::AngleAxisToRotationMatrix(view_pair.second.rotation_2.data(),
                    ceres::ColumnMajorAdapter3x3(relative_rotation_mat.data()));

                // Compute the loop rotation from the two global rotations.
                const Eigen::Matrix<double, 3, 3> loop_rotation_mat =
                    rotation2_mat * rotation1_mat.transpose();
                // Compute the error matrix between the expected relative rotation and the
                // observed relative rotation
                const Eigen::Matrix<double, 3, 3> error_rotation_mat =
                    loop_rotation_mat * relative_rotation_mat.cast<double>().transpose();
                Eigen::Matrix<double, 3, 1> error_rotation;
                ceres::RotationMatrixToAngleAxis(
                    ceres::ColumnMajorAdapter3x3(error_rotation_mat.data()),
                    error_rotation.data());
                double residual = error_rotation.norm();

                if (residual < std::numeric_limits<double>::epsilon())
                    weight = weight_zero;
                else
                {
                    // Calculate the squared residual
                    const double squared_residual = residual * residual;
                    // Get the position of the gamma value in the lookup table
                    size_t x = round(precision_of_stored_gamma3 * squared_residual / squared_sigma_max_2);


                    // If the sought gamma value is not stored in the lookup, return the closest element
                    if (stored_gamma_number3 < x)
                        x = stored_gamma_number3;

                    // Calculate the weight of the point
                    weight = one_over_sigma * (stored_gamma_values3[x] - upper_incomplete_gamma_of_k3);
                }
                
                weights[view_pair_idx] = weight;

                ceres::CostFunction *cost_function =
                    PairwiseRotationError::Create(view_pair.second.rotation_2, weight);
                problem->AddResidualBlock(cost_function,
                                        loss_function,
                                        rotation1->data(),
                                        rotation2->data());
            }


            double average_weight_diff = 0.0;
            for(size_t w_idx = 0; w_idx < view_pair_num; w_idx++){
                average_weight_diff += std::abs(weights[w_idx]-last_weights[w_idx]);
            }
            average_weight_diff /= view_pair_num;
            std::cout << "i=" << i << "   average weight error:" << average_weight_diff << std::endl;

            std::swap(weights,last_weights);

            // The problem should be relatively sparse so sparse cholesky is a good
            // choice.
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.max_num_iterations = 200;
            options.num_threads = thread_num;

            ceres::Solver::Summary summary;
            ceres::Solve(options, problem.get(), &summary);

            if(average_weight_diff <= 1e-7){
                break;
            }

        }


        // VLOG(1) << summary.FullReport();
        return true;
    }

} // namespace theia
