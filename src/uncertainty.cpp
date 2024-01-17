#include "uncertainty.hpp"

bool get_matched_features(theia::ViewIdPair view_id_pair, 
                          theia::Reconstruction& reconstruction,                         
                          MatchedFeatures& matched_features){
    theia::ViewId id1 = view_id_pair.first;
    theia::ViewId id2 = view_id_pair.second;
    const theia::View* view1 = reconstruction.View(id1);
    const theia::View* view2 = reconstruction.View(id2);

    bool has_common_features = false;
    matched_features.first.clear();
    matched_features.second.clear();

    for(theia::TrackId trackid:view1->TrackIds()){
        const theia::Track* track = reconstruction.Track(trackid);
        const std::unordered_set<theia::ViewId>& view_ids = track->ViewIds();
        
        std::unordered_set<theia::ViewId>::const_iterator iter = view_ids.find(id2);
        if(iter == view_ids.end()){
            continue;
        }

        has_common_features = true;
        theia::Feature feature1 = *(view1->GetFeature(trackid));
        theia::Feature feature2 = *(view2->GetFeature(trackid));
        matched_features.first.push_back(feature1);
        matched_features.second.push_back(feature2);

    }

    return has_common_features;
}


MatchedFeaturesSampsonError::MatchedFeaturesSampsonError(
                                const theia::Feature &feature1,
                                const theia::Feature &feature2,
                                const Eigen::Matrix3d &inv_intrinsic1,
                                const Eigen::Matrix3d &inv_intrinsic2)
    : feature1_(feature1), feature2_(feature2),
      inv_intrinsic1_(inv_intrinsic1), inv_intrinsic2_(inv_intrinsic2) {}

ceres::CostFunction* MatchedFeaturesSampsonError::Create(
    const theia::Feature &feature1, const theia::Feature &feature2,
    const Eigen::Matrix3d &inv_intrinsic1,const Eigen::Matrix3d &inv_intrinsic2) {
    return new ceres::AutoDiffCostFunction<MatchedFeaturesSampsonError, 1, 3, 3>(
        new MatchedFeaturesSampsonError(feature1, feature2, inv_intrinsic1, inv_intrinsic2));
}

template <typename T>
bool MatchedFeaturesSampsonError::operator()(const T *rotation,
                                       const T *translation,
                                       T *residuals) const
{
    Eigen::Matrix<T, 3, 3> rotation_mat;
    ceres::AngleAxisToRotationMatrix(
        rotation, rotation_mat.data());
    Eigen::Map<const Eigen::Matrix<T,3,1>> trans_eigen(translation);
    
    Eigen::Matrix<T,3,3> translation_hat;
    translation_hat << T(0.0), -trans_eigen[2], trans_eigen[1],
                       trans_eigen[2], T(0.0), -trans_eigen[0],
                        -trans_eigen[1], trans_eigen[0], T(0.0);

    Eigen::Matrix<T,3,3> F = 
        inv_intrinsic2_.transpose() * rotation_mat*translation_hat * inv_intrinsic1_;


    Eigen::Matrix<T,3,1> epiline_x = F * feature1_.homogeneous();
    T numerator_sqrt = feature2_.homogeneous().dot(epiline_x);
    Eigen::Matrix<T,4,1>denominator(feature2_.homogeneous().dot(F.col(0)),
                                feature2_.homogeneous().dot(F.col(1)),
                                epiline_x[0],
                                epiline_x[1]);

    // // Finally, return the complete Sampson distance.
    residuals[0] = ceres::sqrt(numerator_sqrt*numerator_sqrt / denominator.squaredNorm());

    return true;
}
bool get_covariance_rot(std::pair<const theia::ViewIdPair, theia::TwoViewInfo> &edge_pair,
                    theia::Reconstruction* reconstruction,
                    double* covariance_rotation, Eigen::Vector3d& rotation){
    theia::ViewIdPair view_id_pair = edge_pair.first;
    theia::TwoViewInfo two_view_info = edge_pair.second;

    MatchedFeatures matched_features;
    get_matched_features(view_id_pair,*reconstruction,matched_features);

    // Set up the problem and loss function.
    std::unique_ptr<ceres::Problem> problem(new ceres::Problem());
    ceres::LossFunction *loss_function = new ceres::TrivialLoss();
    

    theia::View view1 = *(reconstruction->View(view_id_pair.first));
    theia::View view2 = *(reconstruction->View(view_id_pair.second));

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

    Eigen::Matrix3d k1_inv,k2_inv;
    k1_inv = intrinsic1.inverse();
    k2_inv = intrinsic2.inverse();

    rotation = two_view_info.rotation_2;
    auto translation = two_view_info.position_2;

    if(translation.isZero()){return false;}
    
    problem.get()->AddParameterBlock(rotation.data(), 3);
    problem.get()->AddParameterBlock(translation.data(), 3);
    // problem.get()->SetParameterBlockConstant(translation.data());
    for(size_t i = 0; i < matched_features.first.size(); i++){
        ceres::CostFunction *cost_function =
                MatchedFeaturesSampsonError::Create(matched_features.first[i],
                                                    matched_features.second[i],
                                                    k1_inv,k2_inv);
        problem->AddResidualBlock(cost_function,
                                loss_function,
                                rotation.data(),
                                translation.data());

    }
    ceres::LocalParameterization* trans_para = new ceres::HomogeneousVectorParameterization(3);
    problem.get()->SetParameterization(translation.data(),trans_para);
    ceres::Solver::Options options;
    options.num_threads = 20;
    options.max_num_iterations = 500;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);

    problem.get()->SetParameterBlockConstant(translation.data());
    options.max_num_iterations = 0;
    ceres::Solve(options, problem.get(), &summary);

    ceres::Covariance::Options cov_options;
    cov_options.apply_loss_function = false;
    ceres::Covariance covariance(cov_options);

    std::vector<std::pair<const double*, const double*> > covariance_blocks;
    covariance_blocks.push_back(std::make_pair(rotation.data(),rotation.data()));
    CHECK(covariance.Compute(covariance_blocks, problem.get()));

    covariance.GetCovarianceBlock(rotation.data(), rotation.data(), covariance_rotation);
    return true;
}

void store_covariance_rot(std::string dataset_directory, 
    theia::Reconstruction* reconstruction,
    theia::ViewGraph* view_graph){
    
    std::ofstream output;
    output.open(dataset_directory+"/covariance_rot.txt",std::fstream::out);
    output << "# Stored as uint64, should convert to double first.\n" 
           << "# view_id1 view_id2 C00 C11 C22 C01 C02 C12 R0 R1 R2" << std::endl;

    auto all_edges = view_graph->GetAllEdges();
    int i = 0;
    for(auto& edge_pair: all_edges){
        i ++;
        if(i%100==0 || i==all_edges.size()){
            std::cout << dataset_directory << " " << i << " / " <<all_edges.size() << std::endl;
        }
        double covariance_rotation[9];
        Eigen::Vector3d rotation;
        bool success = get_covariance_rot(edge_pair,reconstruction,covariance_rotation,rotation);
        if(!success){continue;}

        output << edge_pair.first.first << " " << edge_pair.first.second << " " 
               << *(uint64_t*)(&covariance_rotation[0]) << " " 
               << *(uint64_t*)(&covariance_rotation[4]) << " " 
               << *(uint64_t*)(&covariance_rotation[8]) << " " 
               << *(uint64_t*)(&covariance_rotation[1]) << " " 
               << *(uint64_t*)(&covariance_rotation[2]) << " " 
               << *(uint64_t*)(&covariance_rotation[5]) << " " 
               << *(uint64_t*)(&rotation.x()) << " "
               << *(uint64_t*)(&rotation.y()) << " "
               << *(uint64_t*)(&rotation.z()) << " "
               << std::endl;
    }
    output.close();
}

void read_covariance(const std::string &dataset_directory, 
                   CovarianceMap& covariances){
    std::ifstream file;
    file = std::ifstream(dataset_directory+"/covariance_rot.txt");
    
    std::string line; 
    covariances.clear();
    getline (file, line);
    getline (file, line);
    while (getline (file, line)) {
        theia::ViewId id1,id2;
        int inlier,outlier;
        double c00, c11, c22, c01, c02, c12;
        double r0, r1, r2;
        sscanf(line.c_str(),
             "%d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64" %" SCNu64 " %" SCNu64 " %" SCNu64,
             &id1,
             &id2,
             (uint64_t*)&c00,(uint64_t*)&c11,(uint64_t*)&c22,
             (uint64_t*)&c01,(uint64_t*)&c02,(uint64_t*)&c12,
             (uint64_t*)&r0, (uint64_t*)&r1, (uint64_t*)&r2 );
        theia::ViewIdPair id_pair(id1,id2);
        Eigen::Matrix3d C;
        C << c00, c01, c02,
             c01, c11, c12,
             c02, c12, c22;
        Eigen::Vector3d R(r0,r1,r2);
        covariances.insert(std::make_pair(id_pair,std::make_pair(C,R)));
    }

}