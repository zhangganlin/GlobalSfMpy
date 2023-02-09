#ifndef COMPARE_RECONSTRUCTIONS_HPP
#define COMPARE_RECONSTRUCTIONS_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <theia/theia.h>

#include <algorithm>
#include <memory>
#include <string>

#include "read_colmap_posegraph.hpp"
#include "pairwise_rotation_error_quat.hpp"
#include "uncertainty.hpp"

using theia::Reconstruction;
using theia::ViewGraph;
using theia::TrackId;
using theia::ViewId;
using theia::kInvalidViewId;

class CompareInfo{
    public:
    CompareInfo(){
        rotation_diff_when_align.clear();
        position_errors.clear();
    }
    std::vector<double> rotation_diff_when_align;
    std::vector<double> position_errors;
    int num_3d_points;
    int common_camera;
    int num_reconstructed_view;
};

std::vector<std::string> FindCommonEstimatedViewsByName(
        const Reconstruction& reconstruction1,
        const Reconstruction& reconstruction2);

std::vector<std::string> FindCommonEstimatedViewsByNameColmap(
        const ColmapViewGraph& colmap_viewgraph,
        const Reconstruction& reconstruction);

std::vector<std::string> FindCommonEstimatedViewsByNameColmapETH3D(
        const ColmapViewGraph& colmap_viewgraph,
        const ColmapViewGraph& ETH3D_viewgraph);

CompareInfo compare_reconstructions(const std::vector<std::string> &common_view_names,
        const Reconstruction &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold);

CompareInfo compare_orientations(const std::vector<std::string> &common_view_names,
        const Reconstruction &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold);

CompareInfo compare_orientations_colmap(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold);

CompareInfo compare_reconstructions_colmap(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        Reconstruction *reconstruction_to_align,
        double robust_alignment_threshold);

CompareInfo compare_orientations_colmap_eth3d(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        ColmapViewGraph &reconstruction_to_align,
        double robust_alignment_threshold);

CompareInfo compare_reconstructions_colmap_eth3d(const std::vector<std::string> &common_view_names,
        const ColmapViewGraph &reference_reconstruction,
        ColmapViewGraph &reconstruction_to_align,
        double robust_alignment_threshold);

void residuals_of_relative_rot(const ViewGraph& view_graph,
        const Reconstruction& reconstruction_to_eval,
         CovarianceMap& covariances,
         std::vector<double>& residuals);

void residuals_of_colmap_relative_rot(const ViewGraph& view_graph,
        const Reconstruction& reconstruction,
        const ColmapViewGraph & gt,
         CovarianceMap& covariances,
         std::vector<double>& residuals);

#endif