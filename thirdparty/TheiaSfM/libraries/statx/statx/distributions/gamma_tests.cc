// Copyright (C) 2013  Victor Fragoso <vfragoso@cs.ucsb.edu>
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
//     * Neither the name of the University of California, Santa Barbara nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL VICTOR FRAGOSO BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "statx/distributions/gamma.h"

#include <vector>
#include <glog/logging.h>
#include "gtest/gtest.h"

namespace libstatx {
namespace distributions {
using std::vector;
// Data generated w/ Matlab: gamrnd(10,5,1,100);
const vector<double> gamma_data {
  8.281481e+01, 6.301344e+01, 3.072138e+01, 5.385770e+01, 1.125294e+02,
      4.735972e+01, 4.521651e+01, 7.538720e+01, 7.388141e+01, 3.188737e+01,
      7.836387e+01, 6.626858e+01, 4.376795e+01, 3.709848e+01, 3.260446e+01,
      3.680972e+01, 7.431934e+01, 3.752173e+01, 2.631026e+01, 4.467621e+01,
      5.336188e+01, 4.786767e+01, 5.876286e+01, 6.770980e+01, 4.954593e+01,
      3.300750e+01, 7.628839e+01, 5.433957e+01, 6.786724e+01, 4.884123e+01,
      6.754133e+01, 4.968153e+01, 3.768777e+01, 9.485386e+01, 6.092029e+01,
      6.350518e+01, 2.964721e+01, 5.632677e+01, 4.534917e+01, 5.300938e+01,
      7.745481e+01, 5.999174e+01, 4.464276e+01, 3.238049e+01, 4.998208e+01,
      1.006971e+02, 5.130439e+01, 2.408069e+01, 2.545774e+01, 3.580094e+01,
      4.035286e+01, 3.958864e+01, 6.076237e+01, 4.537793e+01, 3.642099e+01,
      3.350846e+01, 5.028753e+01, 3.188314e+01, 6.233033e+01, 3.367892e+01,
      4.422015e+01, 4.414105e+01, 2.293864e+01, 3.662738e+01, 5.666799e+01,
      4.885567e+01, 6.806502e+01, 4.383164e+01, 4.437378e+01, 4.402731e+01,
      3.465310e+01, 4.050428e+01, 6.492585e+01, 4.802266e+01, 3.695687e+01,
      4.629182e+01, 7.253221e+01, 3.974259e+01, 3.631377e+01, 9.919604e+01,
      5.327352e+01, 3.608887e+01, 6.170982e+01, 2.040792e+01, 5.370556e+01,
      5.570049e+01, 5.124550e+01, 6.301063e+01, 5.575775e+01, 4.331198e+01,
      6.635299e+01, 7.069917e+01, 4.728598e+01, 4.502881e+01, 4.869248e+01,
      6.234588e+01, 5.596122e+01, 5.871831e+01, 3.402727e+01, 5.326567e+01 };

// Data generated w/ Matlab: gamrnd(5,5,1,100);
const vector<double> gamma_data2 {
  2.935282e+01, 1.456991e+01, 2.179105e+01, 4.662592e+01, 1.848795e+01,
      1.280956e+01, 2.049646e+01, 2.074559e+01, 4.529118e+01, 2.093993e+01,
      1.885276e+01, 2.644404e+01, 2.845802e+01, 1.226375e+01, 1.620898e+01,
      2.003918e+01, 3.524929e+00, 3.949052e+01, 3.494162e+01, 2.302144e+01,
      1.018251e+01, 4.529940e+01, 2.378308e+01, 2.300209e+01, 2.824850e+01,
      2.087138e+01, 2.717265e+01, 9.578956e+00, 2.045692e+01, 9.542987e+00,
      1.253181e+01, 2.328615e+01, 2.704726e+01, 3.597449e+01, 2.622144e+01,
      1.177030e+01, 2.333387e+01, 3.462306e+01, 2.732402e+01, 3.488596e+01,
      1.664271e+01, 3.872324e+01, 2.307011e+01, 3.618277e+01, 2.334589e+01,
      5.463564e+00, 6.760208e+00, 2.420741e+01, 2.806612e+01, 3.387817e+01,
      2.853151e+01, 3.344155e+01, 3.443736e+01, 2.177922e+01, 7.090731e+00,
      1.212419e+01, 3.106458e+01, 1.399792e+01, 2.484488e+01, 2.674753e+01,
      1.466104e+01, 7.049609e+00, 1.717663e+01, 2.069649e+01, 2.310877e+01,
      5.564597e+01, 1.125541e+01, 2.515453e+01, 2.050660e+01, 1.851458e+01,
      3.127491e+01, 3.420078e+01, 1.584706e+01, 4.971608e+01, 2.446765e+01,
      2.458195e+01, 1.863726e+01, 4.323075e+01, 3.287631e+01, 2.089741e+01,
      2.039685e+01, 1.078862e+01, 1.553083e+01, 2.715651e+01, 2.035624e+01,
      8.590665e+00, 6.053885e+01, 2.008609e+01, 1.383688e+01, 3.504034e+01,
      1.509380e+01, 1.780764e+01, 1.770217e+01, 1.482979e+01, 9.373417e+00,
      2.208395e+01, 2.636874e+01, 1.070021e+01, 1.883059e+01, 3.777177e+01 };

// Density values generated w/ Matlab: y = gampdf(x, 2, 1)
const vector<double> gamma_pdf_data {
  0, 3.032653e-01, 3.678794e-01, 3.346952e-01, 2.706706e-01, 2.052125e-01,
      1.493612e-01, 1.056908e-01, 7.326256e-02, 4.999048e-02, 3.368973e-02,
      2.247724e-02, 1.487251e-02, 9.772355e-03, 6.383174e-03, 4.148133e-03,
      2.683701e-03, 1.729481e-03, 1.110688e-03, 7.110924e-04, 4.539993e-04 };

// Cumulative distribution values generated w/ Matlab: y = gamcdf(x, 2, 1)

const vector<double> gamma_cdf_data {
  0, 9.020401e-02, 2.642411e-01, 4.421746e-01, 5.939942e-01, 7.127025e-01,
      8.008517e-01, 8.641118e-01, 9.084218e-01, 9.389005e-01, 9.595723e-01,
      9.734360e-01, 9.826487e-01, 9.887242e-01, 9.927049e-01, 9.952988e-01,
      9.969808e-01, 9.980671e-01, 9.987659e-01, 9.992141e-01, 9.995006e-01 };

const vector<double> gamma_cdf_data2 {
  0, 1.518978e-06, 1.352664e-05, 4.817452e-05, 1.178957e-04, 2.349085e-04,
      4.109912e-04, 6.573597e-04, 9.845968e-04, 1.402613e-03, 1.920626e-03,
      2.547156e-03, 3.290026e-03, 4.156373e-03, 5.152663e-03, 6.284708e-03,
      7.557685e-03, 8.976161e-03, 1.054411e-02, 1.226496e-02, 1.414157e-02,
      1.617630e-02, 1.837103e-02, 2.072717e-02, 2.324568e-02, 2.592710e-02,
      2.877160e-02, 3.177897e-02, 3.494863e-02, 3.827971e-02, 4.177102e-02,
      4.542107e-02, 4.922813e-02, 5.319021e-02, 5.730510e-02, 6.157036e-02,
      6.598339e-02, 7.054138e-02, 7.524139e-02, 8.008030e-02, 8.505487e-02,
      9.016176e-02, 9.539749e-02, 1.007585e-01, 1.062412e-01, 1.118418e-01,
      1.175565e-01, 1.233816e-01, 1.293131e-01, 1.353472e-01, 1.414799e-01,
      1.477073e-01, 1.540254e-01, 1.604302e-01, 1.669179e-01, 1.734844e-01,
      1.801258e-01, 1.868383e-01, 1.936179e-01, 2.004608e-01, 2.073631e-01,
      2.143211e-01, 2.213310e-01, 2.283892e-01, 2.354919e-01, 2.426357e-01,
      2.498169e-01, 2.570322e-01, 2.642780e-01, 2.715510e-01, 2.788479e-01,
      2.861655e-01, 2.935005e-01, 3.008500e-01, 3.082109e-01, 3.155802e-01,
      3.229549e-01, 3.303324e-01, 3.377098e-01, 3.450844e-01, 3.524537e-01,
      3.598150e-01, 3.671660e-01, 3.745042e-01, 3.818272e-01, 3.891329e-01,
      3.964190e-01, 4.036835e-01, 4.109242e-01, 4.181392e-01, 4.253266e-01,
      4.324844e-01, 4.396111e-01, 4.467047e-01, 4.537637e-01, 4.607865e-01,
      4.677716e-01, 4.747174e-01, 4.816227e-01, 4.884860e-01, 4.953061e-01,
      5.020817e-01, 5.088117e-01, 5.154950e-01, 5.221305e-01, 5.287172e-01,
      5.352542e-01, 5.417405e-01, 5.481754e-01, 5.545580e-01, 5.608875e-01,
      5.671633e-01, 5.733847e-01, 5.795511e-01, 5.856620e-01, 5.917167e-01,
      5.977148e-01, 6.036560e-01, 6.095396e-01, 6.153655e-01, 6.211332e-01,
      6.268425e-01, 6.324932e-01, 6.380849e-01, 6.436175e-01, 6.490908e-01,
      6.545048e-01, 6.598592e-01, 6.651541e-01, 6.703894e-01, 6.755651e-01,
      6.806811e-01, 6.857377e-01, 6.907347e-01, 6.956722e-01, 7.005505e-01,
      7.053696e-01, 7.101296e-01, 7.148308e-01, 7.194734e-01, 7.240574e-01,
      7.285833e-01, 7.330511e-01, 7.374613e-01, 7.418140e-01, 7.461095e-01,
      7.503482e-01, 7.545303e-01, 7.586563e-01, 7.627264e-01, 7.667411e-01,
      7.707006e-01, 7.746053e-01, 7.784557e-01, 7.822522e-01, 7.859951e-01,
      7.896849e-01, 7.933220e-01, 7.969069e-01, 8.004399e-01, 8.039215e-01,
      8.073522e-01, 8.107324e-01, 8.140627e-01, 8.173434e-01, 8.205750e-01,
      8.237580e-01, 8.268929e-01, 8.299802e-01, 8.330204e-01, 8.360139e-01,
      8.389612e-01, 8.418629e-01, 8.447194e-01, 8.475313e-01, 8.502990e-01,
      8.530230e-01, 8.557038e-01, 8.583419e-01, 8.609379e-01, 8.634922e-01,
      8.660053e-01, 8.684777e-01, 8.709099e-01, 8.733025e-01, 8.756558e-01,
      8.779705e-01, 8.802469e-01, 8.824856e-01, 8.846871e-01, 8.868519e-01,
      8.889803e-01, 8.910731e-01, 8.931305e-01, 8.951530e-01, 8.971413e-01,
      8.990956e-01, 9.010165e-01, 9.029045e-01, 9.047600e-01, 9.065835e-01,
      9.083754e-01, 9.101361e-01, 9.118662e-01, 9.135661e-01, 9.152361e-01,
      9.168768e-01, 9.184886e-01, 9.200719e-01, 9.216271e-01, 9.231547e-01,
      9.246550e-01, 9.261285e-01, 9.275756e-01, 9.289967e-01, 9.303922e-01,
      9.317625e-01, 9.331079e-01, 9.344290e-01, 9.357259e-01, 9.369993e-01,
      9.382493e-01, 9.394764e-01, 9.406809e-01, 9.418633e-01, 9.430238e-01,
      9.441629e-01, 9.452808e-01, 9.463779e-01, 9.474546e-01, 9.485112e-01,
      9.495480e-01, 9.505653e-01, 9.515636e-01, 9.525430e-01, 9.535039e-01,
      9.544467e-01, 9.553716e-01, 9.562789e-01, 9.571690e-01, 9.580421e-01,
      9.588985e-01, 9.597385e-01, 9.605624e-01, 9.613705e-01, 9.621631e-01,
      9.629403e-01, 9.637025e-01, 9.644500e-01, 9.651829e-01, 9.659017e-01,
      9.666064e-01, 9.672974e-01, 9.679749e-01, 9.686392e-01, 9.692904e-01,
      9.699289e-01, 9.705548e-01, 9.711683e-01, 9.717698e-01, 9.723594e-01,
      9.729373e-01, 9.735038e-01, 9.740590e-01, 9.746032e-01, 9.751365e-01,
      9.756592e-01, 9.761714e-01, 9.766734e-01, 9.771654e-01, 9.776475e-01,
      9.781198e-01, 9.785827e-01, 9.790363e-01, 9.794807e-01, 9.799161e-01,
      9.803427e-01, 9.807606e-01, 9.811701e-01, 9.815712e-01, 9.819642e-01,
      9.823492e-01, 9.827263e-01, 9.830957e-01, 9.834576e-01, 9.838120e-01,
      9.841592e-01, 9.844992e-01, 9.848322e-01, 9.851584e-01, 9.854779e-01,
      9.857907e-01, 9.860971e-01, 9.863972e-01, 9.866910e-01, 9.869788e-01,
      9.872606e-01, 9.875365e-01, 9.878066e-01, 9.880712e-01, 9.883302e-01,
      9.885838e-01, 9.888321e-01, 9.890752e-01, 9.893132e-01, 9.895462e-01,
      9.897743e-01, 9.899976e-01, 9.902163e-01, 9.904303e-01, 9.906398e-01,
      9.908448e-01, 9.910456e-01, 9.912421e-01, 9.914344e-01, 9.916226e-01,
      9.918069e-01, 9.919872e-01, 9.921637e-01, 9.923365e-01, 9.925055e-01,
      9.926710e-01, 9.928329e-01, 9.929914e-01, 9.931464e-01, 9.932982e-01,
      9.934467e-01, 9.935920e-01, 9.937341e-01, 9.938733e-01, 9.940094e-01,
      9.941426e-01, 9.942729e-01, 9.944005e-01, 9.945252e-01, 9.946473e-01,
      9.947667e-01, 9.948836e-01, 9.949979e-01, 9.951097e-01, 9.952191e-01,
      9.953261e-01, 9.954308e-01, 9.955332e-01, 9.956334e-01, 9.957314e-01,
      9.958273e-01, 9.959211e-01, 9.960128e-01, 9.961025e-01, 9.961903e-01,
      9.962761e-01, 9.963600e-01, 9.964421e-01, 9.965224e-01, 9.966010e-01,
      9.966778e-01, 9.967529e-01, 9.968264e-01, 9.968982e-01, 9.969685e-01,
      9.970372e-01, 9.971044e-01, 9.971701e-01, 9.972343e-01, 9.972971e-01,
      9.973586e-01, 9.974187e-01, 9.974774e-01, 9.975348e-01, 9.975910e-01,
      9.976459e-01, 9.976996e-01, 9.977521e-01, 9.978034e-01, 9.978536e-01,
      9.979026e-01, 9.979506e-01, 9.979975e-01, 9.980433e-01, 9.980882e-01,
      9.981320e-01, 9.981748e-01, 9.982167e-01, 9.982576e-01, 9.982976e-01,
      9.983367e-01, 9.983750e-01, 9.984124e-01, 9.984489e-01, 9.984846e-01,
      9.985195e-01, 9.985536e-01, 9.985870e-01, 9.986196e-01, 9.986514e-01,
      9.986826e-01, 9.987130e-01, 9.987428e-01, 9.987718e-01, 9.988003e-01 };

TEST(Gamma, FitMLE1) {
  double k, theta;
  EXPECT_TRUE(gammafit(gamma_data, &k, &theta));
  VLOG(1) << "k: " << k << " theta: " << theta;
  const double k_gt = 10;
  const double theta_gt = 5.0;
  EXPECT_NEAR(k, k_gt, 1.0);
  EXPECT_NEAR(theta, theta_gt, 1.0);
}

TEST(Gamma, FitMLE2) {
  double k, theta;
  EXPECT_TRUE(gammafit(gamma_data2, &k, &theta));
  VLOG(1) << "k: " << k << " theta: " << theta;
  const double k_gt = 5.0;
  const double theta_gt = 5.0;
  EXPECT_NEAR(k, k_gt, 1.0);
  EXPECT_NEAR(theta, theta_gt, 1.0);
}

TEST(Gamma, PDF) {
  double x = 0.0;
  const double dx = 0.5;
  const int nel = gamma_pdf_data.size();
  const int k = 2;  // Shape
  const int theta = 1;  // Scale
  for (int i = 0; i < nel; i++) {
    const double y = gammapdf(x, k, theta);
    const double y_gt = gamma_pdf_data[i];
    EXPECT_NEAR(y_gt, y,  1e-2);
    x += dx;
  }
}

TEST(Gamma, CDF) {
  double x = 0.0;
  const double dx = 0.5;
  const int nel = gamma_cdf_data.size();
  const int k = 2;  // Shape
  const int theta = 1;  // Scale
  for (int i = 0; i < nel; i++) {
    const double y = gammacdf(x, k, theta);
    const double y_gt = gamma_cdf_data[i];
    EXPECT_NEAR(y_gt, y,  1e-2);
    x += dx;
  }
}

// gamma(k=3.18575, theta=35.2152)
TEST(Gamma, CDF2) {
  const double k = 3.186575;
  const double theta = 35.2152;
  double x = 0.0;
  const double dx = 1.0;
  const int nel = gamma_cdf_data2.size();
  for (int i = 0; i < nel; i++) {
    const double y = gammacdf(x, k, theta);
    const double y_gt = gamma_cdf_data2[i];
    EXPECT_NEAR(y_gt, y,  1e-2);
    x += dx;
  }
}

// Implementation of lower incomplete gamma function
TEST(Gamma, LowerIncompleteGamma) {
  double x = 1.0;
  double a = 1.0;
  double res_gt = 0.6321;
  double res = lower_inc_gamma(a, x);
  EXPECT_NEAR(res_gt, res, 1e-3);
}

}  // namespace distributions.
}  // namespace libstatx
