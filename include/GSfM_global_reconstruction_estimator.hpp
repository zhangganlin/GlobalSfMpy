#ifndef THEIA_SFM_GSfM_GLOBAL_RECONSTRUCTION_ESTIMATOR_H_
#define THEIA_SFM_GSfM_GLOBAL_RECONSTRUCTION_ESTIMATOR_H_

// #include "theia/sfm/bundle_adjustment/bundle_adjustment.h"
// #include "theia/sfm/filter_view_pairs_from_relative_translation.h"
// #include "theia/sfm/reconstruction_estimator.h"
// #include "theia/sfm/reconstruction_estimator_options.h"
// #include "theia/sfm/types.h"
// #include "theia/solvers/sample_consensus_estimator.h"
// #include "theia/util/util.h"
#include <theia/theia.h>
#include "GSfM_nonlinear_rotation_estimator.hpp"
#include "GSfM_nonlinear_position_estimator.hpp"
#include "uncertainty.hpp"

namespace theia
{

    class Reconstruction;
    class ViewGraph;

    // Estimates the camera position and 3D structure of the scene using global
    // methods to estimate camera poses. First, rotation is estimated globally
    // then the position is estimated using a global optimization.
    //
    // The pipeline for estimating camera poses and structure is as follows:
    //   1) Filter potentially bad pairwise geometries by enforcing a loop
    //      constaint on rotations that form a triplet.
    //   2) Initialize focal lengths.
    //   3) Estimate the global rotation for each camera.
    //   4) Remove any pairwise geometries where the relative rotation is not
    //      consistent with the global rotation.
    //   5) Optimize the relative translation given the known rotations.
    //   6) Filter potentially bad relative translations.
    //   7) Estimate positions.
    //   8) Estimate structure.
    //   9) Bundle adjustment.
    //   10) Retriangulate, and bundle adjust.
    //
    // After each filtering step we remove any views which are no longer connected
    // to the largest connected component in the view graph.
    class GSfMGlobalReconstructionEstimator : public ReconstructionEstimator
    {
    public:
        GSfMGlobalReconstructionEstimator(
            const ReconstructionEstimatorOptions &options);

        ReconstructionEstimatorSummary Estimate(ViewGraph *view_graph,
                                                Reconstruction *reconstruction);

        bool Estimate_StepByStep(ViewGraph *view_graph,
                                 Reconstruction *reconstruction);
        bool Estimate_BeforeStep3(ViewGraph *view_graph, Reconstruction *reconstruction);
        bool Estimate_AfterStep3();

        ViewGraph *view_graph_;
        Reconstruction *reconstruction_;
        std::unordered_map<ViewId, Eigen::Vector3d> orientations_;
        std::unordered_map<ViewId, Eigen::Vector3d> positions_;
        
        bool EstimateGlobalRotationsNonLinear(ceres::LossFunction* loss_func,
                                              RotationErrorType rotation_error_type = RotationErrorType::QUATERNION_COSINE);
        bool EstimateGlobalRotationsUncertainty(ceres::LossFunction* loss_func,
                                              CovarianceMap& covariances,
                                              RotationErrorType rotation_error_type);
        bool EstimateGlobalRotationsSigmaConsensus(ceres::LossFunction* loss_func, int iters_num, double sigma_max);
        bool EstimatePositionNonLinear(ceres::LossFunction* loss_func, PositionErrorType error_type);
        void FilterRotations();
        void OptimizePairwiseTranslations();
        void FilterRelativeTranslation();
        bool EstimatePosition();
        void EstimateStructure();
        theia::BundleAdjustmentSummary BundleAdjustment();
        // Bundle adjust only the camera positions and points. The camera orientations
        // and intrinsics are held constant.
        bool BundleAdjustCameraPositionsAndPoints();
        ReconstructionEstimatorOptions options_;

    private:
        bool FilterInitialViewGraph();
        void CalibrateCameras();
        bool EstimateGlobalRotations();
        
        FilterViewPairsFromRelativeTranslationOptions translation_filter_options_;
        BundleAdjustmentOptions bundle_adjustment_options_;
        RansacParameters ransac_params_;

        DISALLOW_COPY_AND_ASSIGN(GSfMGlobalReconstructionEstimator);
    };

    void SetUnderconstrainedAsUnestimated(Reconstruction *reconstruction);

} // namespace theia

#endif // THEIA_SFM_GLOBAL_RECONSTRUCTION_ESTIMATOR_H_
