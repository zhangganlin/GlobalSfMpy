// Copyright (C) 2018 The Regents of the University of California (Regents).
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
// Author: Chris Sweeney (sweeney.chris.m@gmail.com)

#ifndef THEIA_MATCHING_GLOBAL_DESCRIPTOR_EXTRACTOR_H_
#define THEIA_MATCHING_GLOBAL_DESCRIPTOR_EXTRACTOR_H_

#include <Eigen/Core>
#include <vector>

namespace theia {

// Global descriptors provide a summary of an entire image into a single feature
// descriptor. These descriptors may be formed using training data (e.g., SIFT
// features) or may be directly computed from the image itself. Global
// descriptors provide an efficient mechanism for determining the image
// similarity between two images.
class GlobalDescriptorExtractor {
 public:
  virtual ~GlobalDescriptorExtractor() {}

  // Add features to the descriptor extractor for training. This method may be
  // called multiple times to add multiple sets of features (e.g., once per
  // image) to the global descriptor extractor for training.
  virtual void AddFeaturesForTraining(
      const std::vector<Eigen::VectorXf>& features) = 0;

  // Train the global descriptor extracto with the given set of feature
  // descriptors added with AddFeaturesForTraining. It is assumed that all
  // descriptors have the same length.
  virtual bool Train() = 0;

  // Compute a global image descriptor for the set of input features.
  virtual Eigen::VectorXf ExtractGlobalDescriptor(
      const std::vector<Eigen::VectorXf>& features) = 0;
};

}  // namespace theia
#endif  // THEIA_MATCHING_GLOBAL_DESCRIPTOR_EXTRACTOR_H_
