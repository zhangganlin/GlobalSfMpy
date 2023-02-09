#ifndef UNCERTAINTY_HPP
#define UNCERTAINTY_HPP

#include <theia/io/read_1dsfm.h>
#include <theia/theia.h>
#include <string>
#include <fstream>
#include <ceres/rotation.h>
#include <Eigen/Core>

typedef std::pair<std::vector<theia::Feature>, std::vector<theia::Feature>> MatchedFeatures;

typedef std::unordered_map<theia::ViewIdPair,std::pair<Eigen::Matrix3d,Eigen::Vector3d>> CovarianceMap;

bool get_matched_features(theia::ViewIdPair view_id_pair,
                          theia::Reconstruction &reconstruction,
                          MatchedFeatures &matched_features);

namespace ceres
{
    class CostFunction;
} // namespace ceres


struct MatchedFeaturesSampsonError
{
    MatchedFeaturesSampsonError(const theia::Feature &feature1,
                                const theia::Feature &feature2,
                                const Eigen::Matrix3d &inv_intrinsic1,
                                const Eigen::Matrix3d &inv_intrinsic2);

    // The error is given by the rotation loop error as specified above. We return
    // 3 residuals to give more opportunity for optimization.
    template <typename T>
    bool operator()(const T *rotation, const T *translation, T *residuals) const;

    static ceres::CostFunction *Create(const theia::Feature &feature1,
                                       const theia::Feature &feature2,
                                       const Eigen::Matrix3d &inv_intrinsic1,
                                       const Eigen::Matrix3d &inv_intrinsic2);
    const theia::Feature feature1_;
    const theia::Feature feature2_;
    const Eigen::Matrix3d inv_intrinsic1_;
    const Eigen::Matrix3d inv_intrinsic2_;

};

bool get_covariance_rot(std::pair<const theia::ViewIdPair, theia::TwoViewInfo> &edge_pair,
                    theia::Reconstruction* reconstruction,
                    double* covariance_rotation, Eigen::Vector3d& rotation);

void store_covariance_rot(std::string dataset_directory, 
    theia::Reconstruction* reconstruction,
    theia::ViewGraph* view_graph);

void read_covariance(const std::string &dataset_directory, 
                   CovarianceMap& covariances);

#endif
