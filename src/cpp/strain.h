#pragma once

#include <vector>

#include "shape_func.h"

namespace meshdic {

void compute_strain(
    ElementType type,
    const double* U, int n_nodes,
    const double* nodes_coord,
    const int* elements, int n_elements,
    std::vector<double>& Exx,
    std::vector<double>& Eyy,
    std::vector<double>& Exy);

// Convenience: backward-compatible Q8 caller
inline void compute_strain_q8(
    const double* U, int n_nodes,
    const double* nodes_coord,
    const int* elements, int n_elements,
    std::vector<double>& Exx,
    std::vector<double>& Eyy,
    std::vector<double>& Exy) {
    compute_strain(ElementType::Q8, U, n_nodes, nodes_coord,
                   elements, n_elements, Exx, Eyy, Exy);
}

} // namespace meshdic
