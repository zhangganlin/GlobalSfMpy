#include "compare_reconstructions.hpp"

using theia::Reconstruction;
using theia::TrackId;
using theia::ViewId;

double AngularDifference(const Eigen::Vector3d &rotation1,
                         const Eigen::Vector3d &rotation2)
{
    Eigen::Matrix3d rotation1_mat(
        Eigen::AngleAxisd(rotation1.norm(), rotation1.normalized()));
    Eigen::Matrix3d rotation2_mat(
        Eigen::AngleAxisd(rotation2.norm(), rotation2.normalized()));
    Eigen::Matrix3d rotation_loop = rotation1_mat.transpose() * rotation2_mat;
    return Eigen::AngleAxisd(rotation_loop).angle();
}

namespace {

// A cost function whose error is the difference in rotations after the current
// alignemnt is applied. That is,
//    error = unaligned_rotation * rotation_alignment - gt_rotation.
struct RotationAlignmentError {
  RotationAlignmentError(const Eigen::Vector3d& gt_rotation,
                         const Eigen::Vector3d& unaligned_rotation)
      : gt_rotation_(gt_rotation) {
    // Convert the unaligned rotation to rotation matrix.
    Eigen::Matrix3d unaligned_rotation_mat;
    ceres::AngleAxisToRotationMatrix(
        unaligned_rotation.data(),
        ceres::ColumnMajorAdapter3x3(unaligned_rotation_mat_.data()));
  }

  // Compute the alignment error of the two rotations after applying the
  // rotation transformation.
  template <typename T>
  bool operator()(const T* rotation, T* residuals) const {
    // Convert the rotation transformation to a matrix.
    Eigen::Matrix<T, 3, 3> rotation_mat;
    ceres::AngleAxisToRotationMatrix(
        rotation, ceres::ColumnMajorAdapter3x3(rotation_mat.data()));

    // Apply the rotation transformation.
    const Eigen::Matrix<T, 3, 3> aligned_rotation_mat =
        unaligned_rotation_mat_.cast<T>() * rotation_mat;

    // Convert back to angle axis.
    Eigen::Matrix<T, 3, 1> aligned_rotation;
    ceres::RotationMatrixToAngleAxis(
        ceres::ColumnMajorAdapter3x3(aligned_rotation_mat.data()),
        aligned_rotation.data());

    // Compute the error of the aligned rotation to the gt_rotation.
    residuals[0] = gt_rotation_[0] - aligned_rotation[0];
    residuals[1] = gt_rotation_[1] - aligned_rotation[1];
    residuals[2] = gt_rotation_[2] - aligned_rotation[2];

    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& gt_rotation,
      const Eigen::Vector3d& unaligned_rotation) {
    return new ceres::AutoDiffCostFunction<RotationAlignmentError, 3, 3>(
        new RotationAlignmentError(gt_rotation, unaligned_rotation));
  }

  Eigen::Vector3d gt_rotation_;
  Eigen::Matrix3d unaligned_rotation_mat_;
};


struct PositionAlignmentError {
  PositionAlignmentError(const Eigen::Vector3d& gt_position,
                         const Eigen::Vector3d& unaligned_position)
      : gt_position_(gt_position), unaligned_position_(unaligned_position){
  }

  // Compute the alignment error of the two rotations after applying the
  // rotation transformation.
  template <typename T>
  bool operator()(const T* rotation, const T* translation, const T* scale, T* residuals) const {
    // Convert the rotation transformation to a matrix.
    Eigen::Matrix<T, 3, 3> rotation_mat;
    // ceres::AngleAxisToRotationMatrix(
    //     rotation, ceres::ColumnMajorAdapter3x3(rotation_mat.data()));
    ceres::AngleAxisToRotationMatrix(
        rotation,rotation_mat.data());

    // Convert back to angle axis.
    Eigen::Matrix<T, 3, 1> aligned_position;
    Eigen::Matrix<T, 3, 1> trans;
    trans(0) = translation[0];
    trans(1) = translation[1];
    trans(2) = translation[2];

    aligned_position = rotation_mat*unaligned_position_.cast<T>();
    aligned_position = aligned_position * scale[0];
    aligned_position = aligned_position + trans;

    // Compute the error of the aligned rotation to the gt_rotation.
    residuals[0] = gt_position_[0] - aligned_position(0);
    residuals[1] = gt_position_[1] - aligned_position(1);
    residuals[2] = gt_position_[2] - aligned_position(2);

    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& gt_position,
      const Eigen::Vector3d& unaligned_position) {
    return new ceres::AutoDiffCostFunction<PositionAlignmentError, 3, 3, 3, 1>(
        new PositionAlignmentError(gt_position, unaligned_position));
  }

  Eigen::Vector3d gt_position_;
  Eigen::Vector3d unaligned_position_;
};

// Apply the rotation alignment to all rotations in the vector.
void ApplyRotationTransformation(const Eigen::Vector3d& rotation_alignment,
                                 std::vector<Eigen::Vector3d>* rotation) {
  Eigen::Matrix3d rotation_alignment_mat;
  ceres::AngleAxisToRotationMatrix(
      rotation_alignment.data(),
      ceres::ColumnMajorAdapter3x3(rotation_alignment_mat.data()));

  for (int i = 0; i < rotation->size(); i++) {
    // Convert the current rotation to a rotation matrix.
    Eigen::Matrix3d rotation_mat;
    ceres::AngleAxisToRotationMatrix(
        rotation->at(i).data(),
        ceres::ColumnMajorAdapter3x3(rotation_mat.data()));

    // Apply the rotation transformation.
    const Eigen::Matrix3d aligned_rotation =
        rotation_mat * rotation_alignment_mat;

    // Convert back to angle axis.
    ceres::RotationMatrixToAngleAxis(
        ceres::ColumnMajorAdapter3x3(aligned_rotation.data()),
        rotation->at(i).data());
  }
}

}  // namespace


void AlignRotations(const std::vector<Eigen::Vector3d>& gt_rotation,
                    std::vector<Eigen::Vector3d>* rotation) {
  CHECK_EQ(gt_rotation.size(), rotation->size());

  Eigen::Vector3d rotation_alignment = Eigen::Vector3d::Zero();

  // Set up the nonlinear system and adds all residuals.
  ceres::Problem problem;
  for (int i = 0; i < gt_rotation.size(); i++) {
    ceres::LossFunction* loss_func = new ceres::CauchyLoss(0.1);
    // ceres::LossFunction* loss_func = NULL;

    problem.AddResidualBlock(RotationAlignmentError::Create(gt_rotation[i],
                                                            rotation->at(i)),
                             loss_func,
                             rotation_alignment.data());
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 500;
  options.linear_solver_type = ceres::DENSE_QR;
  options.function_tolerance = 0.0;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  VLOG(2) << summary.FullReport();

  // Apply the solved rotation transformation to the rotations.
  ApplyRotationTransformation(rotation_alignment, rotation);
}


std::vector<std::string> FindCommonEstimatedViewsByName(
        const Reconstruction& reconstruction1,
        const Reconstruction& reconstruction2) {
    std::vector<std::string> common_view_names;
    common_view_names.reserve(reconstruction1.NumViews());

    const auto& view_ids1 = reconstruction1.ViewIds();
    for (const ViewId view_id1 : view_ids1) {
        const std::string name = reconstruction1.View(view_id1)->Name();
        const ViewId view_id2 = reconstruction2.ViewIdFromName(name);
        if (view_id2 != kInvalidViewId && reconstruction2.View(view_id2)->IsEstimated()) {
            common_view_names.emplace_back(name);
        }
    }
    return common_view_names;
}

std::vector<std::string> FindCommonEstimatedViewsByNameColmap(
        const ColmapViewGraph& colmap_viewgraph,
        const Reconstruction& reconstruction){
    std::vector<std::string> common_view_names;
    common_view_names.reserve(colmap_viewgraph.num_view);
    for (const auto&  view_name_pair : colmap_viewgraph.image_names) {
        const std::string name = view_name_pair.second;
        const ViewId view_id2 = reconstruction.ViewIdFromName(name);
        if (view_id2 != kInvalidViewId && reconstruction.View(view_id2)->IsEstimated()) {
            common_view_names.emplace_back(name);
        }
    }
    return common_view_names;
}

std::vector<std::string> FindCommonEstimatedViewsByNameColmapETH3D(
        const ColmapViewGraph& colmap_viewgraph,
        const ColmapViewGraph& ETH3D_viewgraph){
    std::vector<std::string> common_view_names;
    common_view_names.reserve(ETH3D_viewgraph.num_view);
    for (const auto&  view_name_pair : ETH3D_viewgraph.image_names) {
        const std::string name = view_name_pair.second;
        if(colmap_viewgraph.image_ids.find(name)!=colmap_viewgraph.image_ids.end()){
            const ViewId view_id2 = colmap_viewgraph.image_ids.at(name);
            common_view_names.emplace_back(name);
        }
    }
    return common_view_names;
}


CompareInfo compare_orientations(const std::vector<std::string> &common_view_names,
        const Reconstruction &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold){
    CompareInfo result;
    // Gather all the rotations in common with both views.
    std::vector<Eigen::Vector3d> rotations1, rotations2;
    rotations1.reserve(common_view_names.size());
    rotations2.reserve(common_view_names.size());
    for (const std::string &view_name : common_view_names)
    {
        const ViewId view_id1 = reference_reconstruction.ViewIdFromName(view_name);
        const ViewId view_id2 = reconstruction_to_align->ViewIdFromName(view_name);
        rotations1.push_back(reference_reconstruction.View(view_id1)
                                 ->Camera()
                                 .GetOrientationAsAngleAxis());
        rotations2.push_back(reconstruction_to_align->View(view_id2)
                                 ->Camera()
                                 .GetOrientationAsAngleAxis());
    }

    // Align the rotation estimations.
    AlignRotations(rotations1, &rotations2);

    // Measure the difference in rotations.
    for (int i = 0; i < rotations1.size(); i++)
    {
        result.rotation_diff_when_align.push_back(AngularDifference(rotations1[i], rotations2[i]));
    }
    result.common_camera = common_view_names.size();
    return result;
}

CompareInfo compare_orientations_colmap(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold){
    CompareInfo result;
    // Gather all the rotations in common with both views.
    std::vector<Eigen::Vector3d> rotations1, rotations2;

    rotations1.reserve(common_view_names.size());
    rotations2.reserve(common_view_names.size());
    for (const std::string &view_name : common_view_names)
    {
        const ViewId view_id1 = reference_reconstruction.image_ids.at(view_name);
        const ViewId view_id2 = reconstruction_to_align->ViewIdFromName(view_name);
        const std::vector<double>& ref_pose = reference_reconstruction.poses.at(view_id1);
        Eigen::Vector3d ref_orientation = {ref_pose[3],ref_pose[4],ref_pose[5]};

        rotations1.push_back(ref_orientation);
        rotations2.push_back(reconstruction_to_align->View(view_id2)
                                 ->Camera()
                                 .GetOrientationAsAngleAxis());
    }

    // Align the rotation estimations.
    AlignRotations(rotations1, &rotations2);

    // Measure the difference in rotations.
    for (int i = 0; i < rotations1.size(); i++)
    {   
        double error = AngularDifference(rotations1[i], rotations2[i]);
        result.rotation_diff_when_align.push_back(error);
    }
    result.common_camera = common_view_names.size();

    return result;
}

CompareInfo compare_reconstructions_colmap(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold){
    CompareInfo result;

    // Gather all the rotations and positions in common with both views.
    std::vector<Eigen::Vector3d> rotations1, rotations2, positions1, positions2;
    rotations1.reserve(common_view_names.size());
    rotations2.reserve(common_view_names.size());
    positions1.reserve(common_view_names.size());
    positions2.reserve(common_view_names.size());

    for (const std::string &view_name : common_view_names)
    {
        const ViewId view_id1 = reference_reconstruction.image_ids.at(view_name);
        const ViewId view_id2 = reconstruction_to_align->ViewIdFromName(view_name);
        const std::vector<double>& ref_pose = reference_reconstruction.poses.at(view_id1);
        Eigen::Vector3d ref_orientation = {ref_pose[3],ref_pose[4],ref_pose[5]};
        Eigen::Vector3d ref_position = {ref_pose[0],ref_pose[1],ref_pose[2]};


        rotations1.push_back(ref_orientation);
        rotations2.push_back(reconstruction_to_align->View(view_id2)
                                 ->Camera()
                                 .GetOrientationAsAngleAxis());
        positions1.push_back(ref_position);
        positions2.push_back(reconstruction_to_align->View(view_id2)
                                ->Camera()
                                .GetPosition());
    }

    // Align the rotation estimations.
    AlignRotations(rotations1, &rotations2);

    // Measure the difference in rotations.
    for (int i = 0; i < rotations1.size(); i++)
    {   
        double error = AngularDifference(rotations1[i], rotations2[i]);
        result.rotation_diff_when_align.push_back(error);

    }

    // Align the positions.
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d rotation_vec;
    Eigen::Vector3d translation(0,0,0);
    double scale=1.0;
    ceres::RotationMatrixToAngleAxis(rotation.data(),rotation_vec.data());

    // Set up the nonlinear system and adds all residuals.
    ceres::Problem problem;
    for (int i = 0; i < positions1.size(); i++) {
        ceres::LossFunction* loss_func = new ceres::CauchyLoss(0.1);
        // ceres::LossFunction* loss_func = NULL;

        problem.AddResidualBlock(PositionAlignmentError::Create(positions1[i],
                                                                positions2[i]),
                                loss_func,
                                rotation_vec.data(),
                                translation.data(),
                                &scale);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    VLOG(2) << summary.FullReport();

    // ceres::AngleAxisToRotationMatrix(
    //     rotation_vec.data(), ceres::ColumnMajorAdapter3x3(rotation.data()));
    ceres::AngleAxisToRotationMatrix(
        rotation_vec.data(), rotation.data());

    // theia::AlignPointCloudsUmeyama(positions2,
    //                         positions1,
    //                         &rotation,
    //                         &translation,
    //                         &scale);

    // Apply the similarity transformation to the reconstruction.
    theia::TransformReconstruction(rotation, translation, scale, reconstruction_to_align);

     for (int i = 0; i < common_view_names.size(); i++)
    {
        const ViewId view_id1 =
            reference_reconstruction.image_ids.at(common_view_names[i]);
        const ViewId view_id2 =
            reconstruction_to_align->ViewIdFromName(common_view_names[i]);
        const theia::Camera &camera2 =
            reconstruction_to_align->View(view_id2)->Camera();

        // Position error.
        const double position_error =
            (positions1[i] - camera2.GetPosition()).norm();
        result.position_errors.push_back(position_error);

    }
    result.num_3d_points = reconstruction_to_align->NumTracks();
    result.common_camera = (int)common_view_names.size();

    int num_reconstructed_view = 0;

    for(auto view_id: reconstruction_to_align->ViewIds()){
        if(reconstruction_to_align->View(view_id)->IsEstimated()){
            num_reconstructed_view ++;
        }
    }

    result.num_reconstructed_view = num_reconstructed_view;

    return result;
}



CompareInfo compare_reconstructions(const std::vector<std::string> &common_view_names,
        const Reconstruction &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold){
        // CompareInfo *result){
    CompareInfo result;
    // Gather all the rotations in common with both views.
    std::vector<Eigen::Vector3d> rotations1, rotations2;
    rotations1.reserve(common_view_names.size());
    rotations2.reserve(common_view_names.size());
    for (const std::string &view_name : common_view_names)
    {
        const ViewId view_id1 = reference_reconstruction.ViewIdFromName(view_name);
        const ViewId view_id2 = reconstruction_to_align->ViewIdFromName(view_name);
        rotations1.push_back(reference_reconstruction.View(view_id1)
                                 ->Camera()
                                 .GetOrientationAsAngleAxis());
        rotations2.push_back(reconstruction_to_align->View(view_id2)
                                 ->Camera()
                                 .GetOrientationAsAngleAxis());
    }

    // Align the rotation estimations.
    AlignRotations(rotations1, &rotations2);

    // Measure the difference in rotations.
    for (int i = 0; i < rotations1.size(); i++)
    {
        result.rotation_diff_when_align.push_back(AngularDifference(rotations1[i], rotations2[i]));
    }


    //----------------------------------------------------------

    if (robust_alignment_threshold > 0.0)
    {
        AlignReconstructionsRobust(robust_alignment_threshold,
                                   reference_reconstruction,
                                   reconstruction_to_align);
    }
    else
    {
        AlignReconstructions(reference_reconstruction, reconstruction_to_align);
    }

    for (int i = 0; i < common_view_names.size(); i++)
    {
        const ViewId view_id1 =
            reference_reconstruction.ViewIdFromName(common_view_names[i]);
        const ViewId view_id2 =
            reconstruction_to_align->ViewIdFromName(common_view_names[i]);
        const theia::Camera &camera1 =
            reference_reconstruction.View(view_id1)->Camera();
        const theia::Camera &camera2 =
            reconstruction_to_align->View(view_id2)->Camera();

        // Position error.
        const double position_error =
            (camera1.GetPosition() - camera2.GetPosition()).norm();
        result.position_errors.push_back(position_error);

    }
    result.num_3d_points = reconstruction_to_align->NumTracks();
    result.common_camera = (int)common_view_names.size();
    return result;
}

CompareInfo compare_orientations_colmap_eth3d(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        ColmapViewGraph &reconstruction_to_align,
        double robust_alignment_threshold){
    CompareInfo result;
    // Gather all the rotations in common with both views.
    std::vector<Eigen::Vector3d> rotations1, rotations2;

    rotations1.reserve(common_view_names.size());
    rotations2.reserve(common_view_names.size());
    for (const std::string &view_name : common_view_names)
    {
        const ViewId view_id1 = reference_reconstruction.image_ids.at(view_name);
        const ViewId view_id2 = reconstruction_to_align.image_ids.at(view_name);
        const std::vector<double>& ref_pose = reference_reconstruction.poses.at(view_id1);
        const std::vector<double>& to_align_pose = reconstruction_to_align.poses.at(view_id2);

        Eigen::Vector3d ref_orientation = {ref_pose[3],ref_pose[4],ref_pose[5]};
        Eigen::Vector3d to_align_orientation = {to_align_pose[3],to_align_pose[4],to_align_pose[5]};

        rotations1.push_back(ref_orientation);
        rotations2.push_back(to_align_orientation);
    }

    // Align the rotation estimations.
    AlignRotations(rotations1, &rotations2);

    // Measure the difference in rotations.
    for (int i = 0; i < rotations1.size(); i++)
    {   
        double error = AngularDifference(rotations1[i], rotations2[i]);
        result.rotation_diff_when_align.push_back(error);
    }
    result.common_camera = common_view_names.size();

    return result;
}

CompareInfo compare_reconstructions_colmap_eth3d(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        ColmapViewGraph &reconstruction_to_align,
        double robust_alignment_threshold){
    CompareInfo result;

    // Gather all the rotations and positions in common with both views.
    std::vector<Eigen::Vector3d> rotations1, rotations2, positions1, positions2;
    rotations1.reserve(common_view_names.size());
    rotations2.reserve(common_view_names.size());
    positions1.reserve(common_view_names.size());
    positions2.reserve(common_view_names.size());

    for (const std::string &view_name : common_view_names)
    {
        const ViewId view_id1 = reference_reconstruction.image_ids.at(view_name);
        const ViewId view_id2 = reconstruction_to_align.image_ids.at(view_name);
        const std::vector<double>& ref_pose = reference_reconstruction.poses.at(view_id1);
        Eigen::Vector3d ref_orientation = {ref_pose[3],ref_pose[4],ref_pose[5]};
        Eigen::Vector3d ref_position = {ref_pose[0],ref_pose[1],ref_pose[2]};
        const std::vector<double>& to_align_pose = reconstruction_to_align.poses.at(view_id2);
        Eigen::Vector3d to_align_orientation = {to_align_pose[3],to_align_pose[4],to_align_pose[5]};
        Eigen::Vector3d to_align_position = {to_align_pose[0],to_align_pose[1],to_align_pose[2]};


        rotations1.push_back(ref_orientation);
        rotations2.push_back(to_align_orientation);
        positions1.push_back(ref_position);
        positions2.push_back(to_align_position);
    }

    // Align the rotation estimations.
    AlignRotations(rotations1, &rotations2);

    // Measure the difference in rotations.
    for (int i = 0; i < rotations1.size(); i++)
    {   
        double error = AngularDifference(rotations1[i], rotations2[i]);
        result.rotation_diff_when_align.push_back(error);

    }

    // Align the positions.
    Eigen::Matrix3d rotation;
    Eigen::Vector3d rotation_vec(1,1,1);
    Eigen::Vector3d translation(0,0,0);
    double scale = 1.0;

    // Set up the nonlinear system and adds all residuals.
    ceres::Problem problem;
    for (int i = 0; i < positions1.size(); i++) {
        ceres::LossFunction* loss_func = new ceres::CauchyLoss(0.1);
        // ceres::LossFunction* loss_func = NULL;

        problem.AddResidualBlock(PositionAlignmentError::Create(positions1[i],
                                                                positions2[i]),
                                loss_func,
                                rotation_vec.data(),
                                translation.data(),
                                &scale);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    VLOG(2) << summary.FullReport();

    // ceres::AngleAxisToRotationMatrix(
    //     rotation_vec.data(), ceres::ColumnMajorAdapter3x3(rotation.data()));
    ceres::AngleAxisToRotationMatrix(
        rotation_vec.data(), rotation.data());
    
     for (int i = 0; i < common_view_names.size(); i++)
    {
        const ViewId view_id1 =
            reference_reconstruction.image_ids.at(common_view_names[i]);
        const ViewId view_id2 =
            reconstruction_to_align.image_ids.at(common_view_names[i]);

        // Apply the similarity transformation to the reconstruction.
        Eigen::Vector3d camera_position;
        camera_position = scale*rotation*positions2[i]+translation;

        // Position error.
        const double position_error =
            (positions1[i] - camera_position).norm();
        result.position_errors.push_back(position_error);

    }
    result.num_3d_points = -1;
    result.common_camera = (int)common_view_names.size();
    return result;
}

void residuals_of_relative_rot(const ViewGraph& view_graph,
        const Reconstruction& reconstruction_to_eval,
         CovarianceMap& covariances,
         std::vector<double>& residuals){
    residuals.clear();
    for(auto& edge_pair: view_graph.GetAllEdges()){
        auto view_id_pair = edge_pair.first;
        auto two_view_info = edge_pair.second;

        Eigen::Vector3d rotation1 =
                reconstruction_to_eval.View(view_id_pair.first)->Camera().GetOrientationAsAngleAxis();
        Eigen::Vector3d rotation2 =
                reconstruction_to_eval.View(view_id_pair.second)->Camera().GetOrientationAsAngleAxis();
        std::pair<Eigen::Matrix3d,Eigen::Vector3d> * covariance_rot = 
            theia::FindOrNull(covariances,view_id_pair);
        
        if(covariance_rot==0){continue;}
        
        Eigen::Matrix3d cov = (covariance_rot->first)*1e8;
        Eigen::Matrix3d cov_inv = cov.inverse();
        Eigen::Matrix3d Lt;
        Lt = cov_inv.llt().matrixL().transpose();
        theia::PairwiseRotationErrorAngleAxis cost_func(two_view_info.rotation_2,Lt);

        double r[3] = {0,0,0};
        cost_func(rotation1.data(),rotation2.data(),r);
        double residual_norm = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        residuals.push_back(residual_norm);

    }
}

void residuals_of_colmap_relative_rot(const ViewGraph& view_graph,
        const Reconstruction& reconstruction,
        const ColmapViewGraph & gt,
         CovarianceMap& covariances,
         std::vector<double>& residuals){
    residuals.clear();
    for(auto& edge_pair: view_graph.GetAllEdges()){
        auto view_id_pair = edge_pair.first;
        auto two_view_info = edge_pair.second;

        std::string name1 = reconstruction.View(view_id_pair.first)->Name();
        std::string name2 = reconstruction.View(view_id_pair.second)->Name();

        if(gt.image_ids.find(name1) == gt.image_ids.end() || gt.image_ids.find(name2) == gt.image_ids.end()){
            continue;
        }

        const std::vector<double>& pose1 = gt.poses.at(gt.image_ids.at(name1));
        const std::vector<double>& pose2 = gt.poses.at(gt.image_ids.at(name2));

        Eigen::Vector3d rotation1 = {pose1[3],pose1[4],pose1[5]};
        Eigen::Vector3d rotation2 = {pose2[3],pose2[4],pose2[5]};

        std::pair<Eigen::Matrix3d,Eigen::Vector3d> * covariance_rot = 
            theia::FindOrNull(covariances,view_id_pair);
        if(covariance_rot==0){continue;}
        
        // Eigen::Matrix3d cov = (covariance_rot->first)*1e8;
        // Eigen::Matrix3d cov_inv = cov.inverse();
        // Eigen::Matrix3d Lt;
        // Lt = cov_inv.llt().matrixL().transpose();
        theia::PairwiseRotationError cost_func(two_view_info.rotation_2,1.0);
        // theia::PairwiseRotationErrorAngleAxis cost_func(two_view_info.rotation_2,Lt);

        double r[3] = {0,0,0};
        cost_func(rotation1.data(),rotation2.data(),r);
        double residual_norm = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
        residuals.push_back(residual_norm);
    }
}