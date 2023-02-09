// Copyright (C) 2015 The Regents of the University of California (Regents).
// All rights reserved.
//
// Motified by Ganlin Zhang, based on Theia implementation

#include "GSfM_nonlinear_position_estimator.hpp"
#include <theia/sfm/global_pose_estimation/nonlinear_position_estimator.h>

#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <theia/sfm/global_pose_estimation/pairwise_translation_error.h>
#include "pairwise_translation_error_covariance.hpp"

#include <theia/sfm/reconstruction.h>
#include <theia/sfm/types.h>
#include <theia/util/map_util.h>
#include <theia/util/random.h>
#include <theia/util/util.h>

namespace theia
{
    namespace
    {

        using Eigen::Matrix3d;
        using Eigen::Vector3d;

        Vector3d GetRotatedTranslation(const Vector3d &rotation_angle_axis,
                                       const Vector3d &translation)
        {
            Matrix3d rotation;
            ceres::AngleAxisToRotationMatrix(
                rotation_angle_axis.data(),
                ceres::ColumnMajorAdapter3x3(rotation.data()));
            return rotation.transpose() * translation;
        }

        Vector3d GetRotatedFeatureRay(const Camera &camera,
                                      const Vector3d &orientation,
                                      const Feature &feature)
        {
            Camera temp_camera = camera;
            temp_camera.SetOrientationFromAngleAxis(orientation);
            // Get the image ray rotated into the world reference frame.
            return camera.PixelToUnitDepthRay(feature).normalized();
        }

        // Sorts the pairs such that the number of views (i.e. the int) is sorted in
        // descending order.
        bool CompareViewsPerTrack(const std::pair<TrackId, int> &t1,
                                  const std::pair<TrackId, int> &t2)
        {
            return t1.second > t2.second;
        }

    } // namespace

    GSfMNonlinearPositionEstimator::GSfMNonlinearPositionEstimator(
        const NonlinearPositionEstimator::Options &options,
        const Reconstruction &reconstruction)
        : options_(options), reconstruction_(reconstruction)
    {
        CHECK_GT(options_.num_threads, 0);
        CHECK_GE(options_.min_num_points_per_view, 0);
        CHECK_GT(options_.point_to_camera_weight, 0);
        CHECK_GT(options_.robust_loss_width, 0);

        if (options_.rng.get() == nullptr)
        {
            rng_ = std::make_shared<RandomNumberGenerator>();
        }
        else
        {
            rng_ = options_.rng;
        }
        rng_->Seed(1);
    }

    bool GSfMNonlinearPositionEstimator::EstimatePositions(
        const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
        const std::unordered_map<ViewId, Vector3d> &orientations,
        std::unordered_map<ViewId, Vector3d> *positions)
    {
        CHECK_NOTNULL(positions);
        if (view_pairs.empty() || orientations.empty())
        {
            VLOG(2) << "Number of view_pairs = " << view_pairs.size()
                    << " Number of orientations = " << orientations.size();
            return false;
        }
        use_customized_loss = false;
        triangulated_points_.clear();
        problem_.reset(new ceres::Problem());
        view_pairs_ = &view_pairs;

        // Iterative schur is only used if the problem is large enough, otherwise
        // sparse schur is used.
        static const int kMinNumCamerasForIterativeSolve = 1000;

        // Initialize positions to be random.
        InitializeRandomPositions(orientations, positions);

        // Add the constraints to the problem.
        AddCameraToCameraConstraints(orientations, positions);
        if (options_.min_num_points_per_view > 0)
        {
            AddPointToCameraConstraints(orientations, positions);
            AddCamerasAndPointsToParameterGroups(positions);
        }

        // Set one camera to be at the origin to remove the ambiguity of the origin.
        positions->begin()->second.setZero();
        problem_->SetParameterBlockConstant(positions->begin()->second.data());

        // Set the solver options.
        ceres::Solver::Summary summary;
        solver_options_.num_threads = options_.num_threads;
        solver_options_.max_num_iterations = options_.max_num_iterations;

        // Choose the type of linear solver. For sufficiently large problems, we want
        // to use iterative methods (e.g., Conjugate Gradient or Iterative Schur);
        // however, we only want to use a Schur solver if 3D points are used in the
        // optimization.
        if (positions->size() > kMinNumCamerasForIterativeSolve)
        {
            if (options_.min_num_points_per_view > 0)
            {
                solver_options_.linear_solver_type = ceres::ITERATIVE_SCHUR;
                solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;
            }
            else
            {
                solver_options_.linear_solver_type = ceres::CGNR;
                solver_options_.preconditioner_type = ceres::JACOBI;
            }
        }
        else
        {
            if (options_.min_num_points_per_view > 0)
            {
                solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
            }
            else
            {
                solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            }
        }

        ceres::Solve(solver_options_, problem_.get(), &summary);
        LOG(INFO) << summary.FullReport();
        return summary.IsSolutionUsable();
    }

    bool GSfMNonlinearPositionEstimator::EstimatePositions(
        const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
        const std::unordered_map<ViewId, Vector3d> &orientations,
        std::unordered_map<ViewId, Vector3d> *positions,
        PositionErrorType error_type,
        ceres::LossFunction *loss_func)
    {
        CHECK_NOTNULL(positions);
        CHECK_NOTNULL(loss_func);
        if (view_pairs.empty() || orientations.empty())
        {
            VLOG(2) << "Number of view_pairs = " << view_pairs.size()
                    << " Number of orientations = " << orientations.size();
            return false;
        }
        triangulated_points_.clear();

        use_customized_loss = true;
        customized_loss_func = loss_func;

        // Set up the problem and loss function.
        ceres::Problem::Options ceres_options;
        // Because the loss function is constructed by Python, ceres should not destruct it.
        ceres_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        problem_.reset(new ceres::Problem(ceres_options));

        view_pairs_ = &view_pairs;

        // Iterative schur is only used if the problem is large enough, otherwise
        // sparse schur is used.
        static const int kMinNumCamerasForIterativeSolve = 1000;

        // Initialize positions to be random.
        InitializeRandomPositions(orientations, positions);

        // Add the constraints to the problem.
        AddCameraToCameraConstraints(orientations, positions);
        if (options_.min_num_points_per_view > 0)
        {
            AddPointToCameraConstraints(orientations, positions);
            AddCamerasAndPointsToParameterGroups(positions);
        }

        // Set one camera to be at the origin to remove the ambiguity of the origin.
        positions->begin()->second.setZero();
        problem_->SetParameterBlockConstant(positions->begin()->second.data());

        // Set the solver options.
        ceres::Solver::Summary summary;
        // solver_options_.num_threads = options_.num_threads;
        solver_options_.max_num_iterations = options_.max_num_iterations;

        // Choose the type of linear solver. For sufficiently large problems, we want
        // to use iterative methods (e.g., Conjugate Gradient or Iterative Schur);
        // however, we only want to use a Schur solver if 3D points are used in the
        // optimization.
        if (positions->size() > kMinNumCamerasForIterativeSolve)
        {
            if (options_.min_num_points_per_view > 0)
            {
                solver_options_.linear_solver_type = ceres::ITERATIVE_SCHUR;
                solver_options_.preconditioner_type = ceres::SCHUR_JACOBI;
            }
            else
            {
                solver_options_.linear_solver_type = ceres::CGNR;
                solver_options_.preconditioner_type = ceres::JACOBI;
            }
        }
        else
        {
            if (options_.min_num_points_per_view > 0)
            {
                solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
            }
            else
            {
                solver_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            }
        }
        ceres::Solve(solver_options_, problem_.get(), &summary);
        LOG(INFO) << summary.FullReport();
        return summary.IsSolutionUsable();
    }

    void GSfMNonlinearPositionEstimator::InitializeRandomPositions(
        const std::unordered_map<ViewId, Vector3d> &orientations,
        std::unordered_map<ViewId, Vector3d> *positions)
    {
        std::unordered_set<ViewId> constrained_positions;
        constrained_positions.reserve(orientations.size());
        for (const auto &view_pair : *view_pairs_)
        {
            constrained_positions.insert(view_pair.first.first);
            constrained_positions.insert(view_pair.first.second);
        }

        positions->reserve(orientations.size());
        for (const auto &orientation : orientations)
        {
            if (ContainsKey(constrained_positions, orientation.first))
            {
                (*positions)[orientation.first] = 100.0 * rng_->RandVector3d();
                (*positions)[orientation.first] = Eigen::Vector3d(0,0,0);
            }
        }
    }

    void GSfMNonlinearPositionEstimator::AddCameraToCameraConstraints(
        const std::unordered_map<ViewId, Vector3d> &orientations,
        std::unordered_map<ViewId, Vector3d> *positions,
        CovarianceMap& covariances, PositionErrorType error_type)
    {
        for (const auto &view_pair : *view_pairs_)
        {
            const ViewId view_id1 = view_pair.first.first;
            const ViewId view_id2 = view_pair.first.second;
            Vector3d *position1 = FindOrNull(*positions, view_id1);
            Vector3d *position2 = FindOrNull(*positions, view_id2);
            std::pair<Eigen::Matrix3d,Eigen::Vector3d> * covariance_pos = 
                FindOrNull(covariances,view_pair.first);
            // Do not add this view pair if one or both of the positions do not exist.
            if (position1 == nullptr || position2 == nullptr || 
                (covariance_pos==nullptr&&error_type==PositionErrorType::COVARIANCE))
            {
                continue;
            }

            // Rotate the relative translation so that it is aligned to the global
            // orientation frame.
            const Vector3d translation_direction = GetRotatedTranslation(
                FindOrDie(orientations, view_id1), view_pair.second.position_2);

            ceres::CostFunction *cost_function = nullptr;

            if (error_type == PositionErrorType::COVARIANCE){
                Eigen::Matrix3d cov = (covariance_pos->first)*1e8;
                Eigen::Matrix3d cov_inv = cov.completeOrthogonalDecomposition().pseudoInverse();
                Eigen::Matrix3d Lt;
                Lt = cov_inv.llt().matrixL().transpose();
                cost_function = PairwiseTranslationErrorCovariance::Create(translation_direction, Lt);
            }else if (error_type == PositionErrorType::BASELINE){
                cost_function = theia::PairwiseTranslationError::Create(translation_direction,1.0);
            }



            ceres::LossFunction *loss_func;
            if (use_customized_loss)
            {
                loss_func = customized_loss_func;
            }
            else
            {
                loss_func = new ceres::HuberLoss(options_.robust_loss_width);
            }
            problem_->AddResidualBlock(cost_function,
                                       loss_func,
                                       position1->data(),
                                       position2->data());
        }

        VLOG(2) << problem_->NumResidualBlocks()
                << " camera to camera constraints "
                   "were added to the position "
                   "estimation problem.";
    }

    void GSfMNonlinearPositionEstimator::AddCameraToCameraConstraints(
        const std::unordered_map<ViewId, Vector3d> &orientations,
        std::unordered_map<ViewId, Vector3d> *positions)
    {
        for (const auto &view_pair : *view_pairs_)
        {
            const ViewId view_id1 = view_pair.first.first;
            const ViewId view_id2 = view_pair.first.second;
            Vector3d *position1 = FindOrNull(*positions, view_id1);
            Vector3d *position2 = FindOrNull(*positions, view_id2);

            // Do not add this view pair if one or both of the positions do not exist.
            if (position1 == nullptr || position2 == nullptr)
            {
                continue;
            }

            // Rotate the relative translation so that it is aligned to the global
            // orientation frame.
            const Vector3d translation_direction = GetRotatedTranslation(
                FindOrDie(orientations, view_id1), view_pair.second.position_2);

            ceres::CostFunction *cost_function =
                PairwiseTranslationError::Create(translation_direction, 1.0);

            ceres::LossFunction *loss_func;
            if (use_customized_loss)
            {
                loss_func = customized_loss_func;
            }
            else
            {
                loss_func = new ceres::HuberLoss(options_.robust_loss_width);
            }
            problem_->AddResidualBlock(cost_function,
                                       loss_func,
                                       position1->data(),
                                       position2->data());
        }

        VLOG(2) << problem_->NumResidualBlocks()
                << " camera to camera constraints "
                   "were added to the position "
                   "estimation problem.";
    }

    void GSfMNonlinearPositionEstimator::AddPointToCameraConstraints(
        const std::unordered_map<ViewId, Eigen::Vector3d> &orientations,
        std::unordered_map<ViewId, Eigen::Vector3d> *positions)
    {
        const int num_camera_to_camera_constraints = problem_->NumResidualBlocks();
        std::unordered_set<TrackId> tracks_to_add;
        const int num_point_to_camera_constraints =
            FindTracksForProblem(*positions, &tracks_to_add);
        if (num_point_to_camera_constraints == 0)
        {
            return;
        }

        const double point_to_camera_weight =
            options_.point_to_camera_weight *
            static_cast<double>(num_camera_to_camera_constraints) /
            static_cast<double>(num_point_to_camera_constraints);

        triangulated_points_.reserve(tracks_to_add.size());
        for (const TrackId track_id : tracks_to_add)
        {
            triangulated_points_[track_id] = 100.0 * rng_->RandVector3d();
            triangulated_points_[track_id] = Eigen::Vector3d(1,1,1);

            AddTrackToProblem(
                track_id, orientations, point_to_camera_weight, positions);
        }

        VLOG(2) << num_point_to_camera_constraints
                << " point to camera constriants "
                   "were added to the position "
                   "estimation problem.";
    }

    int GSfMNonlinearPositionEstimator::FindTracksForProblem(
        const std::unordered_map<ViewId, Eigen::Vector3d> &positions,
        std::unordered_set<TrackId> *tracks_to_add)
    {
        CHECK_NOTNULL(tracks_to_add)->clear();

        std::unordered_map<ViewId, int> tracks_per_camera;
        for (const auto &position : positions)
        {
            tracks_per_camera[position.first] = 0;
        }

        // Add the tracks that see the most views until each camera has the minimum
        // number of tracks.
        for (const auto &position : positions)
        {
            const View *view = reconstruction_.View(position.first);
            if (view == nullptr ||
                view->NumFeatures() < options_.min_num_points_per_view)
            {
                continue;
            }

            // Get the tracks in sorted order so that we add the tracks that see the
            // most cameras first.
            const std::vector<TrackId> &sorted_tracks =
                GetTracksSortedByNumViews(reconstruction_, *view, *tracks_to_add);

            for (int i = 0;
                 i < sorted_tracks.size() &&
                 tracks_per_camera[position.first] < options_.min_num_points_per_view;
                 i++)
            {
                // Update the number of point to camera constraints for each camera.
                tracks_to_add->insert(sorted_tracks[i]);
                for (const ViewId view_id :
                     reconstruction_.Track(sorted_tracks[i])->ViewIds())
                {
                    if (!ContainsKey(positions, view_id))
                    {
                        continue;
                    }
                    ++tracks_per_camera[view_id];
                }
            }
        }

        int num_point_to_camera_constraints = 0;
        for (const auto &tracks_in_camera : tracks_per_camera)
        {
            num_point_to_camera_constraints += tracks_in_camera.second;
        }
        return num_point_to_camera_constraints;
    }

    std::vector<TrackId> GSfMNonlinearPositionEstimator::GetTracksSortedByNumViews(
        const Reconstruction &reconstruction,
        const View &view,
        const std::unordered_set<TrackId> &existing_tracks)
    {
        std::vector<std::pair<TrackId, int>> views_per_track;
        views_per_track.reserve(view.NumFeatures());
        const auto &track_ids = view.TrackIds();
        for (const auto &track_id : track_ids)
        {
            const Track *track = reconstruction.Track(track_id);

            if (track == nullptr || ContainsKey(existing_tracks, track_id))
            {
                continue;
            }
            views_per_track.emplace_back(track_id, track->NumViews());
        }

        // Return an empty array if no tracks could be found for this view.
        std::vector<TrackId> sorted_tracks(views_per_track.size());
        if (views_per_track.size() == 0)
        {
            return sorted_tracks;
        }

        // Sort the tracks by the number of views. Only sort the first few tracks
        // since those are the ones that will be added to the problem.
        const int num_tracks_to_sort =
            std::min(static_cast<int>(views_per_track.size()),
                     options_.min_num_points_per_view);
        std::partial_sort(views_per_track.begin(),
                          views_per_track.begin() + num_tracks_to_sort,
                          views_per_track.end(),
                          CompareViewsPerTrack);

        for (int i = 0; i < num_tracks_to_sort; i++)
        {
            sorted_tracks[i] = views_per_track[i].first;
        }
        return sorted_tracks;
    }

    void GSfMNonlinearPositionEstimator::AddTrackToProblem(
        const TrackId track_id,
        const std::unordered_map<ViewId, Vector3d> &orientations,
        const double point_to_camera_weight,
        std::unordered_map<ViewId, Vector3d> *positions)
    {
        // For each view in the track add the point to camera correspondences.
        for (const ViewId view_id : reconstruction_.Track(track_id)->ViewIds())
        {
            if (!ContainsKey(*positions, view_id))
            {
                continue;
            }
            Vector3d &camera_position = FindOrDie(*positions, view_id);
            Vector3d &point = FindOrDie(triangulated_points_, track_id);

            // Rotate the feature ray to be in the global orientation frame.
            const Vector3d feature_ray = GetRotatedFeatureRay(
                reconstruction_.View(view_id)->Camera(),
                FindOrDie(orientations, view_id),
                *reconstruction_.View(view_id)->GetFeature(track_id));

            // Rotate the relative translation so that it is aligned to the global
            // orientation frame.
            ceres::CostFunction *cost_function =
                PairwiseTranslationError::Create(feature_ray, point_to_camera_weight);

            ceres::LossFunction *loss_func;
            if (use_customized_loss)
            {
                loss_func = customized_loss_func;
            }
            else
            {
                loss_func = new ceres::HuberLoss(options_.robust_loss_width);
            }
            // Add the residual block
            problem_->AddResidualBlock(cost_function,
                                       loss_func,
                                       camera_position.data(),
                                       point.data());
        }
    }

    void GSfMNonlinearPositionEstimator::AddCamerasAndPointsToParameterGroups(
        std::unordered_map<ViewId, Vector3d> *positions)
    {
        CHECK_GT(triangulated_points_.size(), 0)
            << "Cannot set the Ceres parameter groups for Schur based solvers "
               "because there are no triangulated points.";

        // Create a custom ordering for Schur-based problems.
        solver_options_.linear_solver_ordering.reset(
            new ceres::ParameterBlockOrdering);
        ceres::ParameterBlockOrdering *parameter_ordering =
            solver_options_.linear_solver_ordering.get();
        // Add point parameters to group 0.
        for (auto &point : triangulated_points_)
        {
            parameter_ordering->AddElementToGroup(point.second.data(), 0);
        }

        // Add camera parameters to group 1.
        for (auto &position : *positions)
        {
            parameter_ordering->AddElementToGroup(position.second.data(), 1);
        }
    }

} // namespace theia
