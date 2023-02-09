#ifndef FEATURE_INLIEARS_HPP
#define FEATURE_INLIEARS_HPP

#include <theia/sfm/pose/util.h>
#include "GSfM_global_reconstruction_estimator.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <ceres/rotation.h>
#include <fstream>
// #include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
// using namespace cv::xfeatures2d;


void get_inliners(theia::View& view1, theia::View& view2, theia::TwoViewInfo& edge, string dataset_prefix,
                  int& inliners_num, int& outliners_num);

void read_inliners(const std::string &dataset_directory, unordered_map<theia::ViewIdPair,pair<int,int>>& inliners);
void store_inliners(const std::string &dataset_directory, 
            theia::Reconstruction& reconstruction,theia::ViewGraph& view_graph);
#endif