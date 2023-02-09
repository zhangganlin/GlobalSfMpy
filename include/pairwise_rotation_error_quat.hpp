// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SFM_GLOBAL_POSE_ESTIMATION_PAIRWISE_ROTATION_ERROR_QUAT_H_
#define THEIA_SFM_GLOBAL_POSE_ESTIMATION_PAIRWISE_ROTATION_ERROR_QUAT_H_

#include <iostream>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <ceres/ceres.h>

namespace ceres
{
    class CostFunction;
} // namespace ceres

namespace theia
{
    enum class RotationErrorType
    {
        QUATERNION_NORM = 0,
        ROTATION_MAT_FNORM = 1,
        QUATERNION_COSINE = 2,
        ANGLE_AXIS_COVARIANCE = 3,
        ANGLE_AXIS = 4,
        ANGLE_AXIS_INLIERS = 5,
        ANGLE_AXIS_COV_INLIERS = 6,
        ANGLE_AXIS_COVTRACE = 7,
        ANGLE_AXIS_COVNORM = 8
    };

    // The error in two global rotations based on the current estimates for the
    // global rotations and the relative rotation such that R{i, j} = R_j * R_i'.
    struct PairwiseRotationErrorQuat
    {
        PairwiseRotationErrorQuat(const Eigen::Quaterniond &relative_rotation,
                                  const double weight);

        // The error is given by the rotation loop error as specified above. We return
        // 3 residuals to give more opportunity for optimization.
        template <typename T>
        bool operator()(const T *rotation1, const T *rotation2, T *residuals) const;

        static ceres::CostFunction *Create(const Eigen::Quaterniond &relative_rotation,
                                           const double weight);

        const Eigen::Quaterniond relative_rotation_;
        const double weight_;
    };

    template <typename T>
    bool PairwiseRotationErrorQuat::operator()(const T *rotation1,
                                               const T *rotation2,
                                               T *residuals_ptr) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> q_a(rotation1);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(rotation2);
        // Eigen::Map<const Eigen::Quaternion<T>> q_relative(relative_rotation_.coeffs().data());

        Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
        Eigen::Quaternion<T> q_estimated_relative = q_b * q_a_inverse;

        // Compute the error between the two orientation estimates.
        Eigen::Quaternion<T> delta_q =
            relative_rotation_.template cast<T>() * q_estimated_relative.conjugate();

        // Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);

        // residuals.template block<3, 1>(0, 0) = weight_ * T(2.0) * delta_q.vec();
        residuals_ptr[0] = weight_ * T(2.0) * delta_q.coeffs()[0];
        residuals_ptr[1] = weight_ * T(2.0) * delta_q.coeffs()[1];
        residuals_ptr[2] = weight_ * T(2.0) * delta_q.coeffs()[2];

        return true;
    }

    struct PairwiseRotationErrorQuatFNorm
    {
        PairwiseRotationErrorQuatFNorm(const Eigen::Quaterniond &relative_rotation,
                                       const double weight);

        // The error is given by the rotation loop error as specified above. We return
        // 3 residuals to give more opportunity for optimization.
        template <typename T>
        bool operator()(const T *rotation1, const T *rotation2, T *residuals) const;

        static ceres::CostFunction *Create(const Eigen::Quaterniond &relative_rotation,
                                           const double weight);

        const Eigen::Quaterniond relative_rotation_;
        const double weight_;
    };

    template <typename T>
    bool PairwiseRotationErrorQuatFNorm::operator()(const T *rotation1,
                                                    const T *rotation2,
                                                    T *residuals_ptr) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> q_a(rotation1);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(rotation2);

        Eigen::Quaternion<T> q_b_estimated = relative_rotation_.template cast<T>() * q_a;
        Eigen::Quaternion<T> q_b_positive = q_b;
        if (q_b.coeffs()[1] < (T)0)
        {
            q_b_positive.coeffs() = -q_b.coeffs();
        }
        if (q_b_estimated.coeffs()[1] < (T)0)
        {
            q_b_estimated.coeffs() = -q_b_estimated.coeffs();
        }

        residuals_ptr[0] = T(weight_) * (q_b_positive.coeffs()[0] - q_b_estimated.coeffs()[0]);
        residuals_ptr[1] = T(weight_) * (q_b_positive.coeffs()[1] - q_b_estimated.coeffs()[1]);
        residuals_ptr[2] = T(weight_) * (q_b_positive.coeffs()[2] - q_b_estimated.coeffs()[2]);
        residuals_ptr[3] = T(weight_) * (q_b_positive.coeffs()[3] - q_b_estimated.coeffs()[3]);

        return true;
    }

    struct PairwiseRotationErrorRotFNorm
    {
        PairwiseRotationErrorRotFNorm(const Eigen::Quaterniond &relative_rotation,
                                      const double weight);

        // The error is given by the rotation loop error as specified above. We return
        // 3 residuals to give more opportunity for optimization.
        template <typename T>
        bool operator()(const T *rotation1, const T *rotation2, T *residuals) const;

        static ceres::CostFunction *Create(const Eigen::Quaterniond &relative_rotation,
                                           const double weight);

        const Eigen::Quaterniond relative_rotation_;
        const double weight_;
    };

    template <typename T>
    bool PairwiseRotationErrorRotFNorm::operator()(const T *rotation1,
                                                   const T *rotation2,
                                                   T *residuals_ptr) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> q_a(rotation1);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(rotation2);
        // Convert angle axis rotations to rotation matrices.
        Eigen::Matrix<T, 3, 3> rotation1_mat, rotation2_mat, rotation2_estimated;
        Eigen::Matrix3d relative_rotation_mat;
        rotation1_mat = q_a.toRotationMatrix();
        rotation2_mat = q_b.toRotationMatrix();
        relative_rotation_mat = relative_rotation_.toRotationMatrix();

        rotation2_estimated = relative_rotation_mat * rotation1_mat;

        residuals_ptr[0] = T(weight_) * (rotation2_estimated(0) - rotation2_mat(0));
        residuals_ptr[1] = T(weight_) * (rotation2_estimated(1) - rotation2_mat(1));
        residuals_ptr[2] = T(weight_) * (rotation2_estimated(2) - rotation2_mat(2));
        residuals_ptr[3] = T(weight_) * (rotation2_estimated(3) - rotation2_mat(3));
        residuals_ptr[4] = T(weight_) * (rotation2_estimated(4) - rotation2_mat(4));
        residuals_ptr[5] = T(weight_) * (rotation2_estimated(5) - rotation2_mat(5));
        residuals_ptr[6] = T(weight_) * (rotation2_estimated(6) - rotation2_mat(6));
        residuals_ptr[7] = T(weight_) * (rotation2_estimated(7) - rotation2_mat(7));
        residuals_ptr[8] = T(weight_) * (rotation2_estimated(8) - rotation2_mat(8));

        return true;
    }

    struct PairwiseRotationErrorAngleAxis
    {
        PairwiseRotationErrorAngleAxis(const Eigen::Vector3d &relative_rotation,
                              const Eigen::Matrix3d &rotation_covariance_inverse);

        // The error is given by the rotation loop error as specified above. We return
        // 3 residuals to give more opportunity for optimization.
        template <typename T>
        bool operator()(const T *rotation1, const T *rotation2, T *residuals) const;

        static ceres::CostFunction *Create(const Eigen::Vector3d &relative_rotation,
                                           const Eigen::Matrix3d &rotation_covariance_inverse);

        const Eigen::Vector3d relative_rotation_;
        const Eigen::Matrix3d rotation_covariance_inverse_;
    };

    template <typename T>
    bool PairwiseRotationErrorAngleAxis::operator()(const T *rotation1,
                                           const T *rotation2,
                                           T *residuals) const
    {
        // Convert angle axis rotations to rotation matrices.
        Eigen::Matrix<T, 3, 3> rotation1_mat, rotation2_mat;
        Eigen::Matrix3d relative_rotation_mat;
        ceres::AngleAxisToRotationMatrix(
            rotation1, ceres::ColumnMajorAdapter3x3(rotation1_mat.data()));
        ceres::AngleAxisToRotationMatrix(
            rotation2, ceres::ColumnMajorAdapter3x3(rotation2_mat.data()));
        ceres::AngleAxisToRotationMatrix(relative_rotation_.data(),
                                         ceres::ColumnMajorAdapter3x3(relative_rotation_mat.data()));

        // Compute the loop rotation from the two global rotations.
        const Eigen::Matrix<T, 3, 3> loop_rotation_mat =
            rotation2_mat * rotation1_mat.transpose();
        // Compute the error matrix between the expected relative rotation and the
        // observed relative rotation
        const Eigen::Matrix<T, 3, 3> error_rotation_mat =
            loop_rotation_mat * relative_rotation_mat.cast<T>().transpose();
        Eigen::Matrix<T, 3, 1> error_rotation;
        ceres::RotationMatrixToAngleAxis(
            ceres::ColumnMajorAdapter3x3(error_rotation_mat.data()),
            error_rotation.data());
        Eigen::Matrix<T, 3, 1> error_cov = rotation_covariance_inverse_ * error_rotation;
        residuals[0] = error_cov(0);
        residuals[1] = error_cov(1);
        residuals[2] = error_cov(2);

        return true;
    }

} // namespace theia

#endif // THEIA_SFM_GLOBAL_POSE_ESTIMATION_PAIRWISE_ROTATION_ERROR_H_
