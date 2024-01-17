#include "read_colmap_posegraph.hpp"
#include <opencv2/core/core.hpp>


ColmapViewGraph::ColmapViewGraph(){}

void ColmapViewGraph::read_poses(std::string path){
    std::ifstream fin;
    fin.open(path,std::ios::in);
    std::string temp;
    
    std::getline(fin,temp);
    std::getline(fin,temp);
    std::getline(fin,temp);
    std::getline(fin,temp);

    int num_start = 20;
    int num_len = 0;
    while(temp[num_start+num_len] <= '9' && temp[num_start+num_len] >= '0'){
        num_len ++;
    }
    int num = std::stoi(temp.substr(num_start,num_len));
    // std::cout << num_len << " " << num << std::endl;

    uint32_t image_id, camera_id;
    double qw,qx,qy,qz,tx,ty,tz;
    std::string name;

    for(int i = 0; i < num; i++){
        fin >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> camera_id >> name;
        std::getline(fin,temp);
        std::getline(fin,temp);

        size_t name_start = name.find("/");
        if(name_start != name.npos){
            name = name.substr(name_start+1,name.size());
        }
        
        image_names.insert(std::make_pair(image_id,name));
        image_ids.insert(std::make_pair(name,image_id));
        double q[4] = {qw,qx,qy,qz};
        double rot[3];
        ceres::QuaternionToAngleAxis(q,rot);
        std::vector<double> pose = {tx,ty,tz,rot[0],rot[1],rot[2]};
        poses.insert(std::make_pair(image_id,pose));

        // std::cout << tx << " " << ty << " " << tz << " " << rot[0] << " " <<
        //           rot[1] << " " << rot[2] << std::endl;
        // break;
    }

    num_view = num;

}

void AddColmapMatchesToReconstructionBuilder(
    std::string two_view_geometries_path,
    std::string images_wildcard,
    theia::GSfMReconstructionBuilder *reconstruction_builder){
    
    theia::CameraIntrinsicsGroupId intrinsics_group_id =
        theia::kInvalidCameraIntrinsicsGroupId;

    std::vector<std::string> image_files;
    theia::GetFilepathsFromWildcard(images_wildcard, &image_files);
    // Load calibration file if it is provided.
    std::unordered_map<std::string, theia::CameraIntrinsicsPrior>
        camera_intrinsics_prior;

    // Add images with possible calibration. When the intrinsics group id is
    // invalid, the reconstruction builder will assume that the view does not
    // share its intrinsics with any other views.


    for (const std::string &image_file : image_files)
    {
        std::string image_filename;
        CHECK(theia::GetFilenameFromFilepath(image_file, true, &image_filename));

        theia::CameraIntrinsicsPrior image_camera_intrinsics_prior;
        cv::Mat img = cv::imread(image_file);

        image_camera_intrinsics_prior.image_width=img.cols;
        image_camera_intrinsics_prior.image_height=img.rows;
        theia::Prior<2> pp;
        pp.is_set = true;
        pp.value[0] = image_camera_intrinsics_prior.image_width/2;
        pp.value[1] = image_camera_intrinsics_prior.image_height/2;
        image_camera_intrinsics_prior.principal_point=pp;

        camera_intrinsics_prior.insert(std::make_pair(image_filename,image_camera_intrinsics_prior));
    }




    std::ifstream fin;
    fin.open(two_view_geometries_path,std::ios::in);
    std::string temp;
    
    std::getline(fin,temp);
    std::getline(fin,temp);
    std::getline(fin,temp);
    std::string image_name1,image_name2;
    double f1,f2;
    int num_inliers;
    double rot0,rot1,rot2;
    double t0,t1,t2;
    while(fin>>image_name1>>image_name2>>f1>>f2>>num_inliers>>rot0>>rot1>>rot2>>t0>>t1>>t2){

        theia::CameraIntrinsicsPrior camera_intrinsics_prior1 =
            camera_intrinsics_prior[image_name1];
        theia::CameraIntrinsicsPrior camera_intrinsics_prior2 =
            camera_intrinsics_prior[image_name2];

        theia::Prior<1> focus1;
        focus1.is_set = true;
        focus1.value[0] = f1;
        camera_intrinsics_prior1.focal_length=focus1;
        theia::Prior<1> focus2;
        focus2.is_set = true;
        focus2.value[0] = f2;
        camera_intrinsics_prior2.focal_length=focus2;
        reconstruction_builder->AddImageWithCameraIntrinsicsPrior(
            image_name1,
            camera_intrinsics_prior1,
            intrinsics_group_id);
        reconstruction_builder->AddImageWithCameraIntrinsicsPrior(
            image_name2,
            camera_intrinsics_prior2,
            intrinsics_group_id);

        theia::ImagePairMatch match;
        match.image1 = image_name1;
        match.image2 = image_name2;
        theia::TwoViewInfo two_view_info;
        two_view_info.focal_length_1 = f1;
        two_view_info.focal_length_2 = f2;
        two_view_info.num_homography_inliers = num_inliers;
        two_view_info.num_verified_matches = num_inliers;
        two_view_info.visibility_score = num_inliers;
        two_view_info.position_2 = Eigen::Vector3d(t0,t1,t2);
        two_view_info.rotation_2 = Eigen::Vector3d(rot0,rot1,rot2);
        match.twoview_info = two_view_info;
        match.correspondences.clear();
        std::vector<theia::Feature> features1;
        std::vector<theia::Feature> features2;
        for(int i = 0; i<num_inliers; i++){
            double x,y;
            fin >> x >> y;
            features1.push_back(theia::Feature(x,y));
        }
        for(int i = 0; i<num_inliers; i++){
            double x,y;
            fin >> x >> y;
            features2.push_back(theia::Feature(x,y));
        }
        for(int i = 0; i<num_inliers; i++){
            theia::FeatureCorrespondence correspond(features1[i],features2[i]);
            match.correspondences.push_back(correspond);
        }

        reconstruction_builder->AddTwoViewMatch(image_name1, image_name2, match);
    }
}