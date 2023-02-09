#ifndef THEIA_SFM_GSfM_RECONSTRUCTION_BUILDER_H_
#define THEIA_SFM_GSfM_RECONSTRUCTION_BUILDER_H_

#include <memory>
#include <string>
#include <vector>

#include <theia/theia.h>
#include "GSfM_global_reconstruction_estimator.hpp"

namespace theia
{
    class FeatureExtractorAndMatcher;
    class FeaturesAndMatchesDatabase;
    class RandomNumberGenerator;
    class Reconstruction;
    class TrackBuilder;
    class ViewGraph;
    struct CameraIntrinsicsPrior;
    struct ImagePairMatch;

    struct ReconstructionBuilderOptions;

    // Base class for building SfM reconstructions. This class will manage the
    // entire reconstruction estimation process.
    class GSfMReconstructionBuilder
    {
    public:
        GSfMReconstructionBuilder(
            const ReconstructionBuilderOptions &options,
            FeaturesAndMatchesDatabase *features_and_matches_database);

        // If the reconstruction and view graph are already known the reconstruction
        // builder can take ownership of them and estimate the unestimated views and
        // tracks in the reconstruction.
        // GSfMReconstructionBuilder(const ReconstructionBuilderOptions &options,
        //                                 std::unique_ptr<Reconstruction> reconstruction,
        //                                 std::unique_ptr<ViewGraph> view_graph);
        GSfMReconstructionBuilder(
            const ReconstructionBuilderOptions &options,
            Reconstruction* reconstruction,
            ViewGraph* view_graph);
        ~GSfMReconstructionBuilder();

        // Add an image to the reconstruction.
        bool AddImage(const std::string &image_filepath);
        // Same as above, but with the camera intrinsics group specified to enable
        // shared camera intrinsics.
        bool AddImage(const std::string &image_filepath,
                      const CameraIntrinsicsGroupId camera_intrinsics_group);

        // Same as above, but with the camera priors manually specified.
        bool AddImageWithCameraIntrinsicsPrior(
            const std::string &image_filepath,
            const CameraIntrinsicsPrior &camera_intrinsics_prior);
        // Same as above, but with the camera intrinsics group specified to enable
        // shared camera intrinsics.
        bool AddImageWithCameraIntrinsicsPrior(
            const std::string &image_filepath,
            const CameraIntrinsicsPrior &camera_intrinsics_prior,
            const CameraIntrinsicsGroupId camera_intrinsics_group);

        // Add a match to the view graph. Either this method is repeatedly called or
        // ExtractAndMatchFeatures must be called.
        bool AddTwoViewMatch(const std::string &image1,
                             const std::string &image2,
                             const ImagePairMatch &matches);

        // Assignes a mask to an image to indicate the area for keypoints extraction.
        bool AddMaskForFeaturesExtraction(const std::string &image_filepath,
                                          const std::string &mask_filepath);

        // Extracts features and performs matching with geometric verification.
        bool ExtractAndMatchFeatures();

        // Estimates a Structure-from-Motion reconstruction using the specified
        // ReconstructionEstimator. Features are first extracted and matched if
        // necessary, then a reconstruction is estimated. Once a reconstruction has
        // been estimated, all views that have been successfully estimated are added
        // to the output vector and we estimate a reconstruction from the remaining
        // unestimated views. We repeat this process until no more views can be
        // successfully estimated.
        bool BuildReconstruction(std::vector<Reconstruction *> *reconstructions);

        void CheckView(std::vector<Reconstruction *> *reconstructions);

        // std::unique_ptr<Reconstruction> reconstruction_;
        // std::unique_ptr<ViewGraph> view_graph_;
        Reconstruction* reconstruction_;
        ViewGraph* view_graph_;
        bool contains_allocated_objects = false;
    private:
        // Adds the given matches as edges in the view graph.
        void AddMatchToViewGraph(const ViewId view_id1,
                                 const ViewId view_id2,
                                 const ImagePairMatch &image_matches);

        // Builds tracks from the two view inlier correspondences after geometric
        // verification.
        void AddTracksForMatch(const ViewId view_id1,
                               const ViewId view_id2,
                               const ImagePairMatch &image_matches);

        // Removes all uncalibrated views from the reconstruction and view graph.
        void RemoveUncalibratedViews();

        ReconstructionBuilderOptions options_;

        // SfM objects.
        std::unique_ptr<TrackBuilder> track_builder_;

        // Container of image information.
        std::vector<std::string> image_filepaths_;

        // A DB for storing features and matches.
        FeaturesAndMatchesDatabase *features_and_matches_database_;

        // Module for performing feature extraction and matching.
        std::unique_ptr<FeatureExtractorAndMatcher> feature_extractor_and_matcher_;

        DISALLOW_COPY_AND_ASSIGN(GSfMReconstructionBuilder);
    };

    namespace GSfM
    {
        bool AddViewToReconstruction(const std::string &image_filepath,
                                     const CameraIntrinsicsPrior *intrinsics,
                                     const CameraIntrinsicsGroupId intrinsics_group_id,
                                     Reconstruction *reconstruction);

        Reconstruction *CreateEstimatedSubreconstruction(
            const Reconstruction &input_reconstruction);

        void RemoveEstimatedViewsAndTracks(Reconstruction *reconstruction,
                                           ViewGraph *view_graph);

    } // namespace GSfM

} // namespace theia

#endif // THEIA_SFM_RECONSTRUCTION_BUILDER_H_
