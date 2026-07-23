#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace meshdic {

struct ImageData {
    int height = 0;
    int width = 0;
    std::vector<float> data;
};

struct MeshdicConfig {
    int subset_radius = 31;
    int search_radius = 20;
    int max_iterations = 30;
    double cutoff_diffnorm = 1e-4;
    double lambda_reg = 1e-6;
    int bcoef_border = 3;
};

struct MeshdicResult {
    std::vector<double> U;
    int num_nodes = 0;
    int num_elements = 0;
    std::vector<double> nodes_coord;
    std::vector<int> elements;
    std::vector<double> norm_of_W;
    int iterations = 0;
    bool converged = false;
};

MeshdicResult solve(
    const std::string& ref_image_path,
    const std::string& def_image_path,
    const std::string& mesh_dir,
    const MeshdicConfig& config
);

} // namespace meshdic
