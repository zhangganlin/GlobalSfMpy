#ifndef READ_COLMAP_POSEGRAPH_HPP
#define READ_COLMAP_POSEGRAPH_HPP


#include "GSfM_global_reconstruction_estimator.hpp"
#include <fstream>
#include "GSfM_reconstruction_builder.hpp"
#include <theia/theia.h>


class ColmapViewGraph{
    public:
    ColmapViewGraph();
    void read_poses(std::string path);
    std::unordered_map<theia::ViewId,std::string> image_names;
    std::unordered_map<std::string,theia::ViewId> image_ids;
    std::unordered_map<theia::ViewId,std::vector<double>> poses;
    size_t num_view = 0;
};

void AddColmapMatchesToReconstructionBuilder(
    std::string two_view_geometries_path,
    std::string images_wildcard,
    theia::GSfMReconstructionBuilder *reconstruction_builder);

#endif