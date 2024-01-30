#include "feature_inliers.hpp"

using namespace std;
using namespace cv;

void get_inliners(theia::View& view1, theia::View& view2, theia::TwoViewInfo& edge, string dataset_prefix,
                  int& inliners_num, int& outliners_num){
    inliners_num = 0;
    outliners_num = 0;

    string img1 = dataset_prefix + view1.Name();
    string img2 = dataset_prefix + view2.Name();

    cv::Mat input1 = cv::imread(img1,cv::IMREAD_GRAYSCALE);
    cv::Mat input2 = cv::imread(img2,cv::IMREAD_GRAYSCALE);

    std::vector<KeyPoint> keypoints1,keypoints2;
    Mat descriptors1, descriptors2;

    Ptr<SIFT> detector= cv::SIFT::create();
    detector->detectAndCompute(input1,noArray(),keypoints1,descriptors1);
    detector->detectAndCompute(input2,noArray(),keypoints2,descriptors2);
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    double max_sampson_error_pixels = 6.0;
    const double max_sampson_error_pixels1 =
        theia::ComputeResolutionScaledThreshold(max_sampson_error_pixels,
                                        view1.CameraIntrinsicsPrior().image_width,
                                        view1.CameraIntrinsicsPrior().image_height);
    const double max_sampson_error_pixels2 =
        theia::ComputeResolutionScaledThreshold(max_sampson_error_pixels,
                                        view2.CameraIntrinsicsPrior().image_width,
                                        view2.CameraIntrinsicsPrior().image_height);
    double error_thresh =
        max_sampson_error_pixels1 * max_sampson_error_pixels2;

    double f1 = view1.CameraIntrinsicsPrior().focal_length.value[0];
    double u1 = view1.CameraIntrinsicsPrior().principal_point.value[0];
    double v1 = view1.CameraIntrinsicsPrior().principal_point.value[1];
    double f2 = view2.CameraIntrinsicsPrior().focal_length.value[0];
    double u2 = view2.CameraIntrinsicsPrior().principal_point.value[0];
    double v2 = view2.CameraIntrinsicsPrior().principal_point.value[1];

    Eigen::Matrix3d intrinsic1;
    intrinsic1 << f1, 0,u1,
                   0,f1,v1,
                   0, 0, 1;
    
    Eigen::Matrix3d intrinsic2;
    intrinsic2 << f2, 0,u2,
                   0,f2,v2,
                   0, 0, 1;

    Eigen::Matrix3d inv_intrinsic1 = intrinsic1.inverse();
    Eigen::Matrix3d inv_intrinsic2 = intrinsic2.inverse();

    Eigen::Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(edge.rotation_2.data(),rotation.data());
    Eigen::Vector3d& vec = edge.position_2;
    Eigen::Matrix3d skew_position; 
    skew_position<< 0.0,   -vec[2], vec[1],
                    vec[2], 0.0,   -vec[0],
                    -vec[1], vec[0], 0.0   ;
    Eigen::Matrix3d E = rotation* skew_position;
    Eigen::Matrix3d F = inv_intrinsic2.transpose()*E*inv_intrinsic1;

    // cout << "error_thresh: " << error_thresh << endl;
    std::vector<DMatch> inliers;
    for(auto& match:good_matches){
        Eigen::Vector2d pt1(keypoints1[match.queryIdx].pt.x,keypoints1[match.queryIdx].pt.y);
        Eigen::Vector2d pt2(keypoints2[match.trainIdx].pt.x,keypoints2[match.trainIdx].pt.y);
        Eigen::Vector2d feature1 = (inv_intrinsic1*pt1.homogeneous()).hnormalized();
        Eigen::Vector2d feature2 = (inv_intrinsic2*pt2.homogeneous()).hnormalized();
        
        // double error = theia::SquaredSampsonDistance(E,feature1,feature2);
        double error = theia::SquaredSampsonDistance(F,pt1,pt2);

        if(error <= error_thresh){
            inliers.push_back(match);
            inliners_num ++;
        }
        // cout << error << endl;
    }
    outliners_num =  good_matches.size() - inliners_num;
    
    //-- Draw matches
    // Mat img_matches;
    // drawMatches( input1, keypoints1, input2, keypoints2, inliers, img_matches);

    //-- Show detected matches
    // imshow("Good Matches", img_matches );
    // waitKey();
    // imwrite("/home/zhangganlin/Desktop/CVG/GlobalSfMpy/outlier/colmap/"+view2.Name(),img_matches);

    // drawMatches( input1, keypoints1, input2, keypoints2, good_matches, img_matches);
    // imshow("Good Matches", img_matches );
    // waitKey();
    // imwrite("/home/zhangganlin/Desktop/CVG/GlobalSfMpy/outlier/colmap/"+view2.Name()+"o.jpg",img_matches);
}

void store_inliners(const std::string &dataset_directory, 
            theia::Reconstruction& reconstruction,theia::ViewGraph& view_graph){
    std::ofstream output;
    output.open(dataset_directory+"/inliers.txt",fstream::out);
    for(auto& edge:view_graph.GetAllEdges()){
        theia::TwoViewInfo two_view_info = edge.second;
        theia::View view1 = *reconstruction.View(edge.first.first);
        theia::View view2 = *reconstruction.View(edge.first.second);
        int inlier = 0, outliner = 0;
        get_inliners(view1,view2,two_view_info,dataset_directory+"/images/",inlier,outliner);
        output << edge.first.first << " " << edge.first.second << " " 
               << inlier << " " << outliner << endl;
    }
    output.close();
}

void read_inliners(const std::string &dataset_directory, 
                   unordered_map<theia::ViewIdPair,pair<int,int>>& inliers){
    std::ifstream file(dataset_directory+"/inliers.txt");
    std::string line; 
    inliers.clear();
    while (getline (file, line)) {
        theia::ViewId id1,id2;
        int inlier,outlier;
        sscanf(line.c_str(),
             "%d %d %d %d",
             &id1,
             &id2,
             &inlier,
             &outlier);
        theia::ViewIdPair id_pair(id1,id2);
        std::pair<int,int> inlier_outlier(inlier,outlier);
        inliers.insert(std::make_pair(id_pair,inlier_outlier));
    }

}
