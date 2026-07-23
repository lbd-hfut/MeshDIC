#pragma once

#include <cstdint>
#include <vector>

#include "shape_func.h"

namespace meshdic {

struct G2LParams {
    double tol_global = 0.1;
    double tol_local = 1e-8;
    int max_iter = 1000;
};

struct G2LOutput {
    std::vector<double> xi;
    std::vector<double> eta;
    std::vector<double> J11, J12, J21, J22;
    std::vector<uint8_t> valid;
    std::vector<int> elem_id;
    int img_h = 0, img_w = 0;
};

G2LOutput compute_global_to_local(
    const double* inform, int n_pixels,
    const double* nodes_coord, int n_nodes,
    const int* elements, int n_elements,
    int img_h, int img_w,
    ElementType element_type,
    const G2LParams& params = G2LParams{});

} // namespace meshdic
