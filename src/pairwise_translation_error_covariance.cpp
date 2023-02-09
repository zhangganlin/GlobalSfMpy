#include "pairwise_translation_error_covariance.hpp"

#include <ceres/ceres.h>
#include <glog/logging.h>

namespace theia {

PairwiseTranslationErrorCovariance::PairwiseTranslationErrorCovariance(
    const Eigen::Vector3d& translation_direction, const Eigen::Matrix3d &position_covariance_inverse)
    : translation_direction_(translation_direction), position_covariance_inverse_(position_covariance_inverse) {}

ceres::CostFunction* PairwiseTranslationErrorCovariance::Create(
    const Eigen::Vector3d& translation_direction, const Eigen::Matrix3d &rotation_covariance_inverse) {
  return (new ceres::AutoDiffCostFunction<PairwiseTranslationErrorCovariance, 3, 3, 3>(
        new PairwiseTranslationErrorCovariance(translation_direction, rotation_covariance_inverse)));
}

}  // namespace theia