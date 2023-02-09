#ifndef THEIA_SFM_GLOBAL_POSE_ESTIMATION_GSfM_NONLINEAR_ROTATION_ESTIMATOR_H_
#define THEIA_SFM_GLOBAL_POSE_ESTIMATION_GSfM_NONLINEAR_ROTATION_ESTIMATOR_H_

#include <Eigen/Core>
#include <unordered_map>

#include "theia/sfm/global_pose_estimation/rotation_estimator.h"
#include "theia/sfm/types.h"
#include "theia/util/hash.h"
#include "theia/sfm/reconstruction.h"

#include "pairwise_rotation_error_quat.hpp"
#include "feature_inliers.hpp"
#include "uncertainty.hpp"

namespace theia
{

    // Computes the global rotations given relative rotations and an initial guess
    // for the global orientations. Nonlinear optimization is performed with Ceres
    // using a SoftL1 loss function to be robust to outliers.
    class GSfMNonlinearRotationEstimator : public RotationEstimator
    {
    public:
        GSfMNonlinearRotationEstimator() : robust_loss_width_(0.1) {}
        explicit GSfMNonlinearRotationEstimator(const double robust_loss_width)
            : robust_loss_width_(robust_loss_width) {}

        // Estimates the global orientations of all views based on an initial
        // guess. Returns true on successful estimation and false otherwise.
        bool EstimateRotations(
            const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
            std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations);
        
        bool EstimateRotationsWithCustomizedLoss(
            const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
            std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations,
            ceres::LossFunction *loss_function,
            int thread_num,
            RotationErrorType rotation_error_type = RotationErrorType::QUATERNION_COSINE);

        bool EstimateRotationsWithCustomizedLossAndCovariance(
            const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
            std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations,
            ceres::LossFunction *loss_function, int thread_num, 
            CovarianceMap covariances,
            RotationErrorType rotation_error_type,
            Reconstruction* reconstruction);

        bool EstimateRotationsWithSigmaConsensus(
            const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
            std::unordered_map<ViewId, Eigen::Vector3d> *global_orientations,
            ceres::LossFunction *loss_function,
            int thread_num,
            int iters_num, double sigma_max);

    private:
        const double robust_loss_width_;
    };

} // namespace theia

#endif // THEIA_SFM_GLOBAL_POSE_ESTIMATION_NONLINEAR_ROTATION_ESTIMATOR_H_
