#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "GSfM_reconstruction_builder.hpp"
#include <theia/theia.h>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include "command_line_helpers.h"
#include "compare_reconstructions.hpp"
#include "gamma_values.cpp"
#include "feature_inliers.hpp"
#include "uncertainty.hpp"

PYBIND11_MAKE_OPAQUE(std::unordered_map<uint32_t, Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<theia::ViewIdPair, theia::TwoViewInfo>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<uint32_t,std::string>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string,uint32_t>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<uint32_t,std::vector<double>>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<theia::ViewIdPair,std::pair<int,int>>);
PYBIND11_MAKE_OPAQUE(std::unordered_map<theia::ViewIdPair,std::pair<Eigen::Matrix3d,Eigen::Vector3d>>);


using namespace theia;

namespace py = pybind11;

double (*tgamma_ref) (double) = tgamma;

class pyLossFunction: public ceres::LossFunction{
    public:
    using ceres::LossFunction::LossFunction;
    void Evaluate(double sq_norm, double out[3]) const override{
        py::gil_scoped_acquire acquire;
        if(!cached_flag){
            out_wrap = py::array_t<double>(3,out,dummy);
            cached_flag = true;
        }

        // Check if the pointers have changed and if they have then change them
        auto info = out_wrap.request(true);
        if (info.ptr != out) {
            out_wrap = py::array_t<double>(3, out, dummy);
        }

        pybind11::function overload = pybind11::get_overload(
            static_cast<const ceres::LossFunction*>(this), "Evaluate");
        if (overload) {
            overload.operator()<pybind11::return_value_policy::reference>(
                sq_norm, out_wrap);
            return;
        }
        pybind11::pybind11_fail("Tried to call pure virtual function \"" PYBIND11_STRINGIFY(
            Ceres::LossFunction) "::" "Evaluate \"");

    }
    private:
    mutable bool cached_flag = false;
    mutable py::array_t<double> out_wrap;
    mutable py::str dummy;  // Dummy variable for pybind11 so it doesn't make a
                            // copy
};

class pyRotationEstimator: public theia::RotationEstimator{
    public:
    using RotationEstimator::RotationEstimator;
    bool EstimateRotations(
        const std::unordered_map<ViewIdPair, TwoViewInfo> &view_pairs,
        std::unordered_map<ViewId, Eigen::Vector3d>* rotations) override
    {
        py::gil_scoped_acquire acquire;
        PYBIND11_OVERRIDE_PURE(bool, RotationEstimator, EstimateRotations, 
                                view_pairs,rotations); 
    }
};

void SetOrientataions(
    const std::unordered_map<ViewId, Eigen::Vector3d>& orientations,
    Reconstruction* reconstruction) {
    for(const auto& view_id: reconstruction->ViewIds()){
        View* view = reconstruction->MutableView(view_id);
        view->SetEstimated(false);
    }

    for (const auto& orientation : orientations) {
        View* view = reconstruction->MutableView(orientation.first);
        if (view == nullptr) {
        LOG(WARNING) << "Trying to set the pose of View " << orientation.first
                    << " which does not exist in the reconstruction.";
        continue;
        }
        view->MutableCamera()->SetOrientationFromAngleAxis(orientation.second);
        view->SetEstimated(true);
    }
}

void AddMatchesToReconstructionBuilder(
    FeaturesAndMatchesDatabase *features_and_matches_database,
    GSfMReconstructionBuilder *reconstruction_builder)
{
    // Add all the views. When the intrinsics group id is invalid, the
    // reconstruction builder will assume that the view does not share its
    // intrinsics with any other views.
    theia::CameraIntrinsicsGroupId intrinsics_group_id =
        theia::kInvalidCameraIntrinsicsGroupId;

    const auto camera_calibrations_names =
        features_and_matches_database->ImageNamesOfCameraIntrinsicsPriors();
    LOG(INFO) << "Loading " << camera_calibrations_names.size()
              << " intrinsics priors from the DB.";
    for (int i = 0; i < camera_calibrations_names.size(); i++)
    {
        const theia::CameraIntrinsicsPrior camera_intrinsics_prior =
            features_and_matches_database->GetCameraIntrinsicsPrior(
                camera_calibrations_names[i]);
        reconstruction_builder->AddImageWithCameraIntrinsicsPrior(
            camera_calibrations_names[i],
            camera_intrinsics_prior,
            intrinsics_group_id);
    }

    // Add the matches.
    const auto match_keys = features_and_matches_database->ImageNamesOfMatches();
    LOG(INFO) << "Loading " << match_keys.size() << " matches from the DB.";
    for (const auto &match_key : match_keys)
    {
        const theia::ImagePairMatch &match =
            features_and_matches_database->GetImagePairMatch(match_key.first,
                                                             match_key.second);
        CHECK(reconstruction_builder->AddTwoViewMatch(
            match_key.first, match_key.second, match));
    }
}


PYBIND11_MODULE(GlobalSfMpy, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::bind_map<std::unordered_map<uint32_t, Eigen::Vector3d>>(m,"MapViewIdVector3d");
    py::bind_map<std::unordered_map<ViewIdPair, TwoViewInfo>>(m,"MapEdges");
    py::bind_vector<std::vector<double>>(m,"VectorDouble");
    py::bind_map<std::unordered_map<uint32_t,std::string>>(m,"MapViewIdString");
    py::bind_map<std::unordered_map<std::string,uint32_t>>(m,"MapStringViewId");
    py::bind_map<std::unordered_map<uint32_t,std::vector<double>>>(m,"MapViewIdPose");
    py::bind_map<std::unordered_map<theia::ViewIdPair,std::pair<int,int>>>(m,"MapEdgesInlierOutlier");
    py::bind_map<std::unordered_map<theia::ViewIdPair,std::pair<Eigen::Matrix3d,Eigen::Vector3d>>>(m,"MapEdgesCovariance");


    py::class_<ceres::LossFunction, pyLossFunction>(m,"LossFunction")
        .def(py::init<>())
    ;

    py::class_<RotationEstimator,pyRotationEstimator>(m,"RotationEstimator")
        .def(py::init<>())
        .def("EstimateRotations",&RotationEstimator::EstimateRotations)
    ;

    py::class_<GSfMNonlinearRotationEstimator,RotationEstimator>(m,"NonlinearRotationEstimator")
        .def(py::init<>())
        .def(py::init<const double>())
        .def("EstimateRotations",&GSfMNonlinearRotationEstimator::EstimateRotations, py::call_guard<py::gil_scoped_release>())
        .def("EstimateRotationsWithCustomizedLoss",&GSfMNonlinearRotationEstimator::EstimateRotationsWithCustomizedLoss, py::call_guard<py::gil_scoped_release>())
    ;

    m.def("test_loss_with_input_x",[](ceres::LossFunction* loss_func, double x){
        double out[3] = {0,0,0};
        loss_func->Evaluate(x,out);
        std::cout <<"["<< out[0] << ", " << out[1] << ", " << out[2] <<"]"<< std::endl;
    }, py::call_guard<py::gil_scoped_release>());

    m.def("load_1DSFM_config",
    [](std::string flagfile, ReconstructionBuilderOptions& options){
        YAML::Node config = YAML::LoadFile(flagfile);
        options.num_threads = config["num_threads"].as<int>();
        options.min_track_length = config["min_track_length"].as<int>();
        options.max_track_length = config["max_track_length"].as<int>();

        // Reconstruction Estimator Options.
        theia::ReconstructionEstimatorOptions &reconstruction_estimator_options =
            options.reconstruction_estimator_options;
        reconstruction_estimator_options.min_num_two_view_inliers 
            = config["min_num_inliers_for_valid_match"].as<int>();
        reconstruction_estimator_options.num_threads 
            = config["num_threads"].as<int>();
        reconstruction_estimator_options.intrinsics_to_optimize =
            StringToOptimizeIntrinsicsType(config["intrinsics_to_optimize"].as<std::string>());
        options.reconstruct_largest_connected_component 
            = config["reconstruct_largest_connected_component"].as<bool>();
        options.only_calibrated_views 
            = config["only_calibrated_views"].as<bool>();
        reconstruction_estimator_options.max_reprojection_error_in_pixels 
            = config["max_reprojection_error_pixels"].as<double>();

        // Which type of SfM pipeline to use (e.g., incremental, global, etc.);
        reconstruction_estimator_options.reconstruction_estimator_type =
            StringToReconstructionEstimatorType(config["reconstruction_estimator"].as<std::string>());

        // Global SfM Options.
        reconstruction_estimator_options.global_rotation_estimator_type =
            StringToRotationEstimatorType(config["global_rotation_estimator"].as<std::string>());
        reconstruction_estimator_options.global_position_estimator_type =
            StringToPositionEstimatorType(config["global_position_estimator"].as<std::string>());
        reconstruction_estimator_options.num_retriangulation_iterations =
            config["num_retriangulation_iterations"].as<int>();
        reconstruction_estimator_options
            .refine_relative_translations_after_rotation_estimation =
            config["refine_relative_translations_after_rotation_estimation"].as<bool>();
        reconstruction_estimator_options.extract_maximal_rigid_subgraph =
            config["extract_maximal_rigid_subgraph"].as<bool>();
        reconstruction_estimator_options.filter_relative_translations_with_1dsfm =
            config["filter_relative_translations_with_1dsfm"].as<bool>();
        reconstruction_estimator_options.rotation_filtering_max_difference_degrees =
            config["post_rotation_filtering_degrees"].as<double>();
        reconstruction_estimator_options.nonlinear_position_estimator_options
            .min_num_points_per_view =
            config["position_estimation_min_num_tracks_per_view"].as<int>();
        reconstruction_estimator_options
            .refine_camera_positions_and_points_after_position_estimation =
            config["refine_camera_positions_and_points_after_position_estimation"].as<bool>();

        // Incremental SfM Options.
        reconstruction_estimator_options.absolute_pose_reprojection_error_threshold =
            config["absolute_pose_reprojection_error_threshold"].as<double>();
        reconstruction_estimator_options.min_num_absolute_pose_inliers =
            config["min_num_absolute_pose_inliers"].as<int>();
        reconstruction_estimator_options.full_bundle_adjustment_growth_percent =
            config["full_bundle_adjustment_growth_percent"].as<double>();
        reconstruction_estimator_options.partial_bundle_adjustment_num_views =
            config["partial_bundle_adjustment_num_views"].as<int>();

        // Triangulation options (used by all SfM pipelines).
        reconstruction_estimator_options.min_triangulation_angle_degrees =
            config["min_triangulation_angle_degrees"].as<double>();
        reconstruction_estimator_options
            .triangulation_max_reprojection_error_in_pixels =
            config["triangulation_reprojection_error_pixels"].as<double>();
        reconstruction_estimator_options.bundle_adjust_tracks =
            config["bundle_adjust_tracks"].as<bool>();

        // Bundle adjustment options (used by all SfM pipelines).
        reconstruction_estimator_options.bundle_adjustment_loss_function_type =
            StringToLossFunction(config["bundle_adjustment_robust_loss_function"].as<std::string>());
        reconstruction_estimator_options.bundle_adjustment_robust_loss_width =
            config["bundle_adjustment_robust_loss_width"].as<double>();

        // Track subsampling options.
        reconstruction_estimator_options.subsample_tracks_for_bundle_adjustment =
            config["subsample_tracks_for_bundle_adjustment"].as<bool>();
        reconstruction_estimator_options
            .track_subset_selection_long_track_length_threshold =
            config["track_subset_selection_long_track_length_threshold"].as<int>();
        reconstruction_estimator_options.track_selection_image_grid_cell_size_pixels =
            config["track_selection_image_grid_cell_size_pixels"].as<int>();
        reconstruction_estimator_options.min_num_optimized_tracks_per_view =
            config["min_num_optimized_tracks_per_view"].as<int>();
    }   
    , py::call_guard<py::gil_scoped_release>());

    m.def("SetReconstructionFromEstimatedPoses",
            &SetReconstructionFromEstimatedPoses, py::call_guard<py::gil_scoped_release>());

    m.def("SetUnderconstrainedAsUnestimated",
            &SetUnderconstrainedAsUnestimated, py::call_guard<py::gil_scoped_release>());
    m.def("AddMatchesToReconstructionBuilder",[](RocksDbFeaturesAndMatchesDatabase* database, 
                                                 GSfMReconstructionBuilder* reconstruction_builder){
        AddMatchesToReconstructionBuilder(database,reconstruction_builder);
    },py::call_guard<py::gil_scoped_release>());

    m.def("AddColmapMatchesToReconstructionBuilder",&AddColmapMatchesToReconstructionBuilder,
        py::call_guard<py::gil_scoped_release>());

    m.def("store_covariance_rot",
            &store_covariance_rot, py::call_guard<py::gil_scoped_release>());

    py::class_<ReconstructionEstimatorOptions>(m,"ReconstructionEstimatorOptions")
        .def(py::init<>())
        .def_readwrite("num_retriangulation_iterations",
            &ReconstructionEstimatorOptions::num_retriangulation_iterations)
        .def_readwrite("refine_camera_positions_and_points_after_position_estimation",
            &ReconstructionEstimatorOptions::refine_camera_positions_and_points_after_position_estimation)
    ;

    py::class_<BundleAdjustmentSummary>(m,"BundleAdjustmentSummary")
        .def(py::init<>())
        .def_readwrite("success",&BundleAdjustmentSummary::success)
        .def_readwrite("initial_cost",&BundleAdjustmentSummary::initial_cost)
        .def_readwrite("final_cost",&BundleAdjustmentSummary::final_cost)
        .def_readwrite("setup_time_in_seconds",&BundleAdjustmentSummary::setup_time_in_seconds)
        .def_readwrite("solve_time_in_seconds",&BundleAdjustmentSummary::solve_time_in_seconds)
    ;

    py::class_<ReconstructionBuilderOptions>(m,"ReconstructionBuilderOptions")
        .def(py::init<>())
        .def_readwrite("reconstruction_estimator_options",
            &ReconstructionBuilderOptions::reconstruction_estimator_options)
        .def("print",[](ReconstructionBuilderOptions& options){
            theia::ReconstructionEstimatorOptions &reconstruction_estimator_options =
                options.reconstruction_estimator_options;
            std::cout << "num_threads: " << options.num_threads << std::endl
                << "min_track_length: " << options.min_track_length << std::endl
                << "max_track_length: " << options.max_track_length << std::endl
                << "min_num_inliers_for_valid_match: " 
                << reconstruction_estimator_options.min_num_two_view_inliers 
                << std::endl << "intrinsics_to_optimize: "
                << (int)reconstruction_estimator_options.intrinsics_to_optimize 
                << std::endl << "reconstruct_largest_connected_component: "
                << options.reconstruct_largest_connected_component << std::endl
                << "only_calibrated_views: " << options.only_calibrated_views 
                << std::endl << "max_reprojection_error_pixels: "
                << reconstruction_estimator_options.max_reprojection_error_in_pixels 
                << std::endl << "reconstruction_estimator: "
                << (int)reconstruction_estimator_options
                    .reconstruction_estimator_type << std::endl 
                << "global_rotation_estimator: "
                << (int)reconstruction_estimator_options
                    .global_rotation_estimator_type << std::endl 
                << "global_position_estimator: "
                << (int)reconstruction_estimator_options
                    .global_position_estimator_type << std::endl 
                << "num_retriangulation_iterations: "
                << reconstruction_estimator_options
                    .num_retriangulation_iterations << std::endl
                << "refine_relative_translations_after_rotation_estimation: "
                << reconstruction_estimator_options
                    .refine_relative_translations_after_rotation_estimation
                << std::endl << "extract_maximal_rigid_subgraph: "
                << reconstruction_estimator_options.extract_maximal_rigid_subgraph
                << std::endl << "filter_relative_translations_with_1dsfm: " 
                << reconstruction_estimator_options
                    .filter_relative_translations_with_1dsfm << std::endl
                << "post_rotation_filtering_degrees: " 
                << reconstruction_estimator_options
                    .rotation_filtering_max_difference_degrees << std::endl
                << "position_estimation_min_num_tracks_per_view: "
                << reconstruction_estimator_options.nonlinear_position_estimator_options
                    .min_num_points_per_view << std::endl
                << "refine_camera_positions_and_points_after_position_estimation: "
                << reconstruction_estimator_options
                    .refine_camera_positions_and_points_after_position_estimation
                << std::endl << "absolute_pose_reprojection_error_threshold: "
                << reconstruction_estimator_options
                    .absolute_pose_reprojection_error_threshold << std::endl
                << "min_num_absolute_pose_inliers: "
                << reconstruction_estimator_options.min_num_absolute_pose_inliers
                << std::endl << "full_bundle_adjustment_growth_percent: "
                << reconstruction_estimator_options.full_bundle_adjustment_growth_percent
                << std::endl << "partial_bundle_adjustment_num_views: "
                << reconstruction_estimator_options.partial_bundle_adjustment_num_views
                << std::endl << "min_triangulation_angle_degrees: "
                << reconstruction_estimator_options.min_triangulation_angle_degrees
                << std::endl << "triangulation_reprojection_error_pixels: "
                << reconstruction_estimator_options
                    .triangulation_max_reprojection_error_in_pixels << std::endl
                << "bundle_adjust_tracks: " << reconstruction_estimator_options
                                                .bundle_adjust_tracks << std::endl
                << "bundle_adjustment_robust_loss_function: " 
                << (int)reconstruction_estimator_options.bundle_adjustment_loss_function_type
                << std::endl << "bundle_adjustment_robust_loss_width: "
                << reconstruction_estimator_options.bundle_adjustment_robust_loss_width
                << std::endl << "subsample_tracks_for_bundle_adjustment: "
                << reconstruction_estimator_options.subsample_tracks_for_bundle_adjustment
                << std::endl << "track_subset_selection_long_track_length_threshold: "
                << reconstruction_estimator_options
                    .track_subset_selection_long_track_length_threshold << std::endl
                << "track_selection_image_grid_cell_size_pixels: "
                << reconstruction_estimator_options.track_selection_image_grid_cell_size_pixels
                << std::endl << "min_num_optimized_tracks_per_view: "
                << reconstruction_estimator_options.min_num_optimized_tracks_per_view 
                << std::endl << std::endl;
        }, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<CompareInfo>(m,"CompareInfo")
        .def(py::init<>())
        .def_readwrite("rotation_diff_when_align",&CompareInfo::rotation_diff_when_align)
        .def_readwrite("position_errors",&CompareInfo::position_errors)
        .def_readwrite("num_3d_points",&CompareInfo::num_3d_points)
        .def_readwrite("common_camera",&CompareInfo::common_camera)
        .def_readwrite("num_reconstructed_view",&CompareInfo::num_reconstructed_view)
    ;

    py::class_<ColmapViewGraph>(m,"ColmapViewGraph")
        .def(py::init<>())
        .def("read_poses",&ColmapViewGraph::read_poses)
        .def_readwrite("num_view",&ColmapViewGraph::num_view)
        .def_readwrite("image_ids",&ColmapViewGraph::image_ids)
        .def_readwrite("image_names",&ColmapViewGraph::image_names)
        .def_readwrite("poses",&ColmapViewGraph::poses)
    ;

    py::class_<Reconstruction>(m,"Reconstruction")
        .def(py::init<>())
        .def("NumTracks",&Reconstruction::NumTracks)
    ;

    py::class_<ViewGraph>(m,"ViewGraph")
        .def(py::init<>())
        .def("NumViews", &ViewGraph::NumViews)
        .def("NumEdges", &ViewGraph::NumEdges)
        .def("HasView",  &ViewGraph::HasView)
        .def("HasEdge",  &ViewGraph::HasEdge)
        .def("ViewIds",  &ViewGraph::ViewIds)
        .def("GetAllEdges", &ViewGraph::GetAllEdges,py::return_value_policy::reference)
    ;

    py::class_<TwoViewInfo>(m,"TwoViewInfo")
        .def(py::init<>())
        .def_readwrite("focal_length_1",&TwoViewInfo::focal_length_1)
        .def_readwrite("focal_length_2",&TwoViewInfo::focal_length_2)
        .def_readwrite("position_2",&TwoViewInfo::position_2)
        .def_readwrite("rotation_2",&TwoViewInfo::rotation_2)
        .def_readwrite("num_verified_matches",&TwoViewInfo::num_verified_matches)
        .def_readwrite("num_homography_inliers",&TwoViewInfo::num_homography_inliers)
        .def_readwrite("visibility_score",&TwoViewInfo::visibility_score)
    ;

    py::enum_<RotationErrorType>(m, "RotationErrorType", py::arithmetic())
        .value("QUATERNION_COSINE", RotationErrorType::QUATERNION_COSINE)
        .value("QUATERNION_NORM", RotationErrorType::QUATERNION_NORM)
        .value("ROTATION_MAT_FNORM", RotationErrorType::ROTATION_MAT_FNORM)
        .value("ANGLE_AXIS_COVARIANCE",RotationErrorType::ANGLE_AXIS_COVARIANCE)
        .value("ANGLE_AXIS",RotationErrorType::ANGLE_AXIS)
        .value("ANGLE_AXIS_COVTRACE",RotationErrorType::ANGLE_AXIS_COVTRACE)
        .value("ANGLE_AXIS_COVNORM",RotationErrorType::ANGLE_AXIS_COVNORM)
    ;

    py::enum_<PositionErrorType>(m, "PositionErrorType", py::arithmetic())
        .value("BASELINE", PositionErrorType::BASELINE)
    ;

    py::class_<RocksDbFeaturesAndMatchesDatabase>(m,"FeaturesAndMatchesDatabase")
        .def(py::init([](std::string& directory){
            return new RocksDbFeaturesAndMatchesDatabase(directory);
        }))
        .def("ContainsCameraIntrinsicsPrior",&RocksDbFeaturesAndMatchesDatabase::ContainsCameraIntrinsicsPrior, py::call_guard<py::gil_scoped_release>())
        .def("GetCameraIntrinsicsPrior",&RocksDbFeaturesAndMatchesDatabase::GetCameraIntrinsicsPrior, py::call_guard<py::gil_scoped_release>())
        .def("PutCameraIntrinsicsPrior",&RocksDbFeaturesAndMatchesDatabase::PutCameraIntrinsicsPrior, py::call_guard<py::gil_scoped_release>())
        .def("ImageNamesOfCameraIntrinsicsPriors",&RocksDbFeaturesAndMatchesDatabase::ImageNamesOfCameraIntrinsicsPriors, py::call_guard<py::gil_scoped_release>())
        .def("NumCameraIntrinsicsPrior",&RocksDbFeaturesAndMatchesDatabase::NumCameraIntrinsicsPrior, py::call_guard<py::gil_scoped_release>())
        .def("ContainsFeatures",&RocksDbFeaturesAndMatchesDatabase::ContainsFeatures, py::call_guard<py::gil_scoped_release>())
        .def("GetFeatures",&RocksDbFeaturesAndMatchesDatabase::GetFeatures, py::call_guard<py::gil_scoped_release>())
        .def("PutFeatures",&RocksDbFeaturesAndMatchesDatabase::PutFeatures, py::call_guard<py::gil_scoped_release>())
        .def("ImageNamesOfFeatures",&RocksDbFeaturesAndMatchesDatabase::ImageNamesOfFeatures, py::call_guard<py::gil_scoped_release>())
        .def("NumImages",&RocksDbFeaturesAndMatchesDatabase::NumImages, py::call_guard<py::gil_scoped_release>())
        .def("GetImagePairMatch",&RocksDbFeaturesAndMatchesDatabase::GetImagePairMatch, py::call_guard<py::gil_scoped_release>())
        .def("PutImagePairMatch",&RocksDbFeaturesAndMatchesDatabase::PutImagePairMatch, py::call_guard<py::gil_scoped_release>())
        .def("ImageNamesOfMatches",&RocksDbFeaturesAndMatchesDatabase::ImageNamesOfMatches, py::call_guard<py::gil_scoped_release>())
        .def("NumMatches",&RocksDbFeaturesAndMatchesDatabase::NumMatches, py::call_guard<py::gil_scoped_release>())
        .def("RemoveAllMatches",&RocksDbFeaturesAndMatchesDatabase::RemoveAllMatches, py::call_guard<py::gil_scoped_release>())
    ;

    py::class_<GSfMReconstructionBuilder>
        (m, "ReconstructionBuilder")
        // .def(py::init<ReconstructionBuilderOptions,Reconstruction*,ViewGraph* >())
        .def(py::init([](ReconstructionBuilderOptions &options,
            Reconstruction* reconstruction,
            ViewGraph* view_graph){
                return new GSfMReconstructionBuilder(
                options, std::move(reconstruction), std::move(view_graph));
        }
        ))
        .def(py::init([](ReconstructionBuilderOptions &options,
            RocksDbFeaturesAndMatchesDatabase * database){
                return new GSfMReconstructionBuilder(
                options,std::move(database));
        }
        ))
        .def("BuildReconstruction", [](GSfMReconstructionBuilder* builder){
            std::vector<Reconstruction *> reconstructions;
            bool success = builder->BuildReconstruction(&reconstructions);
            return std::make_tuple(success,std::move(reconstructions));
        }, py::call_guard<py::gil_scoped_release>())
        .def("CheckView",[](GSfMReconstructionBuilder* builder){
            std::vector<Reconstruction *> reconstructions;
            builder->CheckView(&reconstructions);
        }, py::call_guard<py::gil_scoped_release>())
        .def("get_view_graph",
            [](GSfMReconstructionBuilder* builder)->ViewGraph*{
                builder->contains_allocated_objects = false;
                return builder->view_graph_;
            }
        , py::call_guard<py::gil_scoped_release>())
        .def("get_reconstruction",
            [](GSfMReconstructionBuilder* builder)->Reconstruction*{
                builder->contains_allocated_objects = false;
                return builder->reconstruction_;
            }
        , py::call_guard<py::gil_scoped_release>())
    ;


    py::class_<GSfMGlobalReconstructionEstimator>
        (m, "GlobalReconstructionEstimator")
        .def(py::init([](ReconstructionEstimatorOptions &options){
            return new GSfMGlobalReconstructionEstimator(options);
        }), py::call_guard<py::gil_scoped_release>())
        .def("get_view_graph",
            [](GSfMGlobalReconstructionEstimator* estimator)->ViewGraph*{
                return estimator->view_graph_;
            }, py::call_guard<py::gil_scoped_release>()
        )
        .def("get_reconstruction",
            [](GSfMGlobalReconstructionEstimator* estimator)->Reconstruction*{
                return estimator->reconstruction_;
            }, py::call_guard<py::gil_scoped_release>()
        )
        .def_readwrite("orientations",&GSfMGlobalReconstructionEstimator::orientations_, py::call_guard<py::gil_scoped_release>())
        .def_readwrite("positions",&GSfMGlobalReconstructionEstimator::positions_, py::call_guard<py::gil_scoped_release>())
        .def_readwrite("options",&GSfMGlobalReconstructionEstimator::options_, py::call_guard<py::gil_scoped_release>())
        .def("FilterInitialViewGraphAndCalibrateCameras",
            &GSfMGlobalReconstructionEstimator::Estimate_BeforeStep3,
            py::call_guard<py::gil_scoped_release>())
        .def("OrientationsFromMaximumSpanningTree",
            [](GSfMGlobalReconstructionEstimator* estimator){
                OrientationsFromMaximumSpanningTree
                    (*(estimator->view_graph_),&(estimator->orientations_));
            },
            py::call_guard<py::gil_scoped_release>()
        )
        .def("EstimateGlobalRotations", 
            &GSfMGlobalReconstructionEstimator::EstimateGlobalRotationsNonLinear,
            py::arg("loss_func")= nullptr,
            py::arg("rotation_error_type")=RotationErrorType::QUATERNION_COSINE,
            py::call_guard<py::gil_scoped_release>()
        )
        .def("EstimateGlobalRotationsUncertainty", 
            &GSfMGlobalReconstructionEstimator::EstimateGlobalRotationsUncertainty,
            py::call_guard<py::gil_scoped_release>()
        )
        .def("EstimateGlobalRotationsWithSigmaConsensus",
        [](GSfMGlobalReconstructionEstimator* estimator, ceres::LossFunction* loss_func, int iters_num, double sigma_max){
            py::gil_scoped_release release;
            return estimator->EstimateGlobalRotationsSigmaConsensus(loss_func,iters_num,sigma_max);
        })
        .def("FilterRotations",[](GSfMGlobalReconstructionEstimator* estimator){
            py::gil_scoped_release release;
            LOG(INFO) << "Filtering any bad rotation estimations.";
            estimator->FilterRotations();    
        })
        .def("OptimizePairwiseTranslations",[](GSfMGlobalReconstructionEstimator* estimator){
            py::gil_scoped_release release;
            LOG(INFO) << "Optimizing the pairwise translation estimations.";
            estimator->OptimizePairwiseTranslations();;    
        })
        .def("FilterRelativeTranslation",[](GSfMGlobalReconstructionEstimator* estimator){
            py::gil_scoped_release release;
            LOG(INFO) << "Filtering any bad relative translations.";
            estimator->FilterRelativeTranslation();;    
        })
        .def("EstimatePosition",
        [](GSfMGlobalReconstructionEstimator* estimator,
           ceres::LossFunction* loss_func,
           PositionErrorType error_type
           )->bool{
            py::gil_scoped_release release;
            LOG(INFO) << "Estimating the positions of all cameras.";
            if (!estimator->EstimatePositionNonLinear(loss_func,error_type))
            {
                LOG(WARNING) << "Position estimation failed!";
                return false;
            }
            LOG(INFO) << estimator->positions_.size()
                    << " camera positions were estimated successfully.";
            return true;
        })
        .def("EstimateStructure",[](GSfMGlobalReconstructionEstimator* estimator){
            py::gil_scoped_release release;
            LOG(INFO) << "Triangulating all features.";
            estimator->EstimateStructure();
        })
        .def("BundleAdjustCameraPositionsAndPoints",[](GSfMGlobalReconstructionEstimator* estimator){
            py::gil_scoped_release release;
            LOG(INFO) << "Performing partial bundle adjustment to optimize only the "
                             "camera positions and 3d points.";
            estimator->BundleAdjustCameraPositionsAndPoints();
        })
        .def("BundleAdjustmentAndRemoveOutlierPoints",
            [](GSfMGlobalReconstructionEstimator* estimator)->BundleAdjustmentSummary{
                py::gil_scoped_release release;
                LOG(INFO) << "Performing bundle adjustment.";
                auto summary = estimator->BundleAdjustment();
                if (!summary.success)
                {
                    LOG(WARNING) << "Bundle adjustment failed!";
                    return summary;
                }
                int num_points_removed = SetOutlierTracksToUnestimated(
                    estimator->options_.max_reprojection_error_in_pixels,
                    estimator->options_.min_triangulation_angle_degrees,
                    estimator->reconstruction_);
                LOG(INFO) << num_points_removed << " outlier points were removed.";
                return summary;
            }
        )
    ;

    m.def("Read1DSFM",[](const std::string &dataset_directory, 
            theia::Reconstruction *reconstruction, 
            theia::ViewGraph *view_graph, 
            std::unordered_map<theia::ViewIdPair,std::pair<Eigen::Matrix3d,Eigen::Vector3d>>& rot_covariances){
        Read1DSFM(dataset_directory,reconstruction,view_graph);
        read_covariance(dataset_directory,rot_covariances);
    }, py::call_guard<py::gil_scoped_release>());
    m.def("ReadCovariance",[](const std::string &dataset_directory, 
            std::unordered_map<theia::ViewIdPair,std::pair<Eigen::Matrix3d,Eigen::Vector3d>>& rot_covariances){
        read_covariance(dataset_directory,rot_covariances);
    }, py::call_guard<py::gil_scoped_release>());

    m.def("CalcCovariance",[](string dataset_path){
        theia::Reconstruction* reconstruction = new theia::Reconstruction;
        theia::ViewGraph* view_graph = new theia::ViewGraph;
        Read1DSFM(dataset_path, reconstruction, view_graph);
        store_covariance_rot(dataset_path,reconstruction,view_graph);
    }, py::call_guard<py::gil_scoped_release>());


    m.def("WriteReconstruction",&WriteReconstruction, py::call_guard<py::gil_scoped_release>());
    m.def("InitGlog",[](int log_level, bool logtostderr, std::string log_dir){
        FLAGS_logtostderr = logtostderr;
        FLAGS_v = log_level;
        FLAGS_log_dir = log_dir;
        google::InitGoogleLogging("GlobalSfM");
    },py::arg("log_level")=0,py::arg("logtostderr")=true,py::arg("log_dir")="./log",py::call_guard<py::gil_scoped_release>());
    m.def("StopGlog",&google::ShutdownGoogleLogging,py::call_guard<py::gil_scoped_release>());


    m.def("num_reproject",[](Reconstruction& reconstruction)->int{
            int res = 0;
            for(auto trackid:reconstruction.TrackIds()){
                auto track = reconstruction.Track(trackid);
                res += track->NumViews();
            }
            return res;
        },py::call_guard<py::gil_scoped_release>());

    m.def("compare_reconstructions",&compare_reconstructions, py::call_guard<py::gil_scoped_release>());
    m.def("compare_orientations",&compare_orientations, py::call_guard<py::gil_scoped_release>());
    m.def("compare_orientations_colmap",&compare_orientations_colmap, py::call_guard<py::gil_scoped_release>());
    m.def("compare_reconstructions_colmap",&compare_reconstructions_colmap, py::call_guard<py::gil_scoped_release>());
    m.def("compare_orientations_colmap_eth3d",&compare_orientations_colmap_eth3d, py::call_guard<py::gil_scoped_release>());
    m.def("compare_reconstructions_colmap_eth3d",&compare_reconstructions_colmap_eth3d, py::call_guard<py::gil_scoped_release>());

    m.def("residuals_of_relative_rot",&residuals_of_relative_rot, py::call_guard<py::gil_scoped_release>());
    m.def("residuals_of_colmap_relative_rot",&residuals_of_colmap_relative_rot, py::call_guard<py::gil_scoped_release>());

    m.def("ReadReconstruction",&theia::ReadReconstruction, py::call_guard<py::gil_scoped_release>());
    m.def("FindCommonViewsByName",&FindCommonEstimatedViewsByName, py::call_guard<py::gil_scoped_release>());
    m.def("FindCommonViewsByNameColmap",&FindCommonEstimatedViewsByNameColmap, py::call_guard<py::gil_scoped_release>());
    m.def("FindCommonViewsByNameColmapETH3D",&FindCommonEstimatedViewsByNameColmapETH3D, py::call_guard<py::gil_scoped_release>());
    
    m.def("SetOrientations",&SetOrientataions, py::call_guard<py::gil_scoped_release>());
    m.def("tgamma",tgamma_ref, py::call_guard<py::gil_scoped_release>());

    m.attr("nu3") = py::float_(nu3);
    m.attr("stored_gamma_values3") = stored_gamma_values3;
    m.attr("C3") = py::float_(C3);
    m.attr("sigma_quantile3") = py::float_(sigma_quantile3);
    m.attr("upper_incomplete_gamma_of_k3") = py::float_(upper_incomplete_gamma_of_k3);
    m.attr("stored_gamma_number3") = py::int_(stored_gamma_number3);
    m.attr("precision_of_stored_gamma3") = py::float_(precision_of_stored_gamma3);

    m.attr("nu4") = py::float_(nu4);
    m.attr("stored_gamma_values4") = stored_gamma_values4;
    m.attr("C4") = py::float_(C4);
    m.attr("sigma_quantile4") = py::float_(sigma_quantile4);
    m.attr("upper_incomplete_gamma_of_k4") = py::float_(upper_incomplete_gamma_of_k4);
    m.attr("stored_gamma_number4") = py::int_(stored_gamma_number4);
    m.attr("precision_of_stored_gamma4") = py::float_(precision_of_stored_gamma4);

    m.attr("nu9") = py::float_(nu9);
    m.attr("stored_gamma_values9") = stored_gamma_values9;
    m.attr("C9") = py::float_(C9);
    m.attr("sigma_quantile9") = py::float_(sigma_quantile9);
    m.attr("upper_incomplete_gamma_of_k9") = py::float_(upper_incomplete_gamma_of_k9);
    m.attr("stored_gamma_number9") = py::int_(stored_gamma_number9);
    m.attr("precision_of_stored_gamma9") = py::float_(precision_of_stored_gamma9);
}
