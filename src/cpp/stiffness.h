#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstdint>
#include <vector>

#include "global2local.h"
#include "shape_func.h"

namespace meshdic {

struct StiffnessCache {
    Eigen::SparseMatrix<double> A;
    int fem_size = 0;
    std::vector<int> free_nodes;

    ElementType element_type = ElementType::Q8;
    std::vector<std::vector<int>> elem_pixels;
    std::vector<std::vector<int>> elem_dofs;
    std::vector<Eigen::MatrixXd> elem_N_cache;
    std::vector<Eigen::MatrixXd> elem_DN_cache;
};

StiffnessCache assemble_stiffness(
    const G2LOutput& g2l,
    int img_h, int img_w,
    const double* fx_ref, const double* fy_ref,
    int n_nodes,
    const int* elements, int n_elements,
    ElementType element_type,
    double alpha, double beta = 0.0);

Eigen::VectorXd assemble_residual(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    const Eigen::VectorXd& U,
    class BsplineEngine* bsp,
    const double* def_bcoef, int def_bcoef_h, int def_bcoef_w,
    double alpha, double beta = 0.0);

// Forward-additive Gauss-Newton with constant Hessian.
// The Hessian is assembled once from the reference image (ICGN-style),
// but displacement updates use forward-additive U += dU rather than
// inverse-compositional warp updates.  Commonly called "Global ICGN"
// in the FE-DIC literature (e.g., Yang et al.).
// beta: optional displacement penalty (soft boundary condition) to
// suppress rigid-body modes when no boundary conditions are applied.
int global_icgn(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    Eigen::VectorXd& U,
    class BsplineEngine* bsp,
    const double* def_bcoef, int def_bcoef_h, int def_bcoef_w,
    double alpha, double tol, int max_iter, double beta = 0.0);

int global_forward_gn(
    const StiffnessCache& cache,
    const G2LOutput& g2l,
    const double* ref_img, int img_h, int img_w,
    const int* elements, int n_elements,
    Eigen::VectorXd& U,
    class BsplineEngine* bsp,
    const double* def_bcoef, int def_bcoef_h, int def_bcoef_w,
    double alpha, double tol, int max_iter, double beta = 0.0);

} // namespace meshdic
