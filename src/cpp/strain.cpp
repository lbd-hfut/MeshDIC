#include "strain.h"
#include "shape_func.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace meshdic {

void compute_strain(
    ElementType type,
    const double* U, int n_nodes,
    const double* nodes_coord,
    const int* elements, int n_elements,
    std::vector<double>& Exx,
    std::vector<double>& Eyy,
    std::vector<double>& Exy) {

    int nn = nodes_per_element(type);
    int elem_stride = (type == ElementType::Q8) ? 9 : nn;

    Exx.assign(n_nodes, 0.0);
    Eyy.assign(n_nodes, 0.0);
    Exy.assign(n_nodes, 0.0);
    std::vector<int> count(n_nodes, 0);

    for (int e = 0; e < n_elements; ++e) {
        // Extract element connectivity (0-based)
        std::vector<int> conn(nn);
        for (int k = 0; k < nn; ++k)
            conn[k] = elements[e * elem_stride + k] - 1;

        // Extract coordinates and displacements
        std::vector<double> elem_coord(2 * nn);
        std::vector<double> elem_U(2 * nn);
        for (int k = 0; k < nn; ++k) {
            int nid = conn[k];
            elem_coord[2 * k]     = nodes_coord[2 * nid];
            elem_coord[2 * k + 1] = nodes_coord[2 * nid + 1];
            elem_U[2 * k]         = U[2 * nid];
            elem_U[2 * k + 1]     = U[2 * nid + 1];
        }

        // Evaluate shape function derivatives at center
        double xi = 0.0, eta = 0.0;
        std::vector<double> N(nn), dN_dxi(nn), dN_deta(nn);
        switch (type) {
            case ElementType::T3:
                xi = 1.0/3.0; eta = 1.0/3.0;
                shape_functions_t3(xi, eta, N.data(), dN_dxi.data(), dN_deta.data());
                break;
            case ElementType::Q4:
                shape_functions_q4(xi, eta, N.data(), dN_dxi.data(), dN_deta.data());
                break;
            case ElementType::Q8:
            default: {
                double n8[8], dx8[8], de8[8];
                shape_functions_q8(xi, eta, n8, dx8, de8);
                for (int i = 0; i < 8; ++i) {
                    N[i] = n8[i]; dN_dxi[i] = dx8[i]; dN_deta[i] = de8[i];
                }
                break;
            }
        }

        // Jacobian
        double J11 = 0.0, J12 = 0.0, J21 = 0.0, J22 = 0.0;
        for (int k = 0; k < nn; ++k) {
            J11 += dN_dxi[k]  * elem_coord[2 * k];
            J12 += dN_deta[k] * elem_coord[2 * k];
            J21 += dN_dxi[k]  * elem_coord[2 * k + 1];
            J22 += dN_deta[k] * elem_coord[2 * k + 1];
        }
        double detJ = J11 * J22 - J12 * J21;
        if (std::abs(detJ) < 1e-12) continue;
        double iJ11 = J22 / detJ, iJ12 = -J12 / detJ;
        double iJ21 = -J21 / detJ, iJ22 = J11 / detJ;

        // Displacement gradients
        double dudx = 0.0, dudy = 0.0;
        double dvdx = 0.0, dvdy = 0.0;
        for (int k = 0; k < nn; ++k) {
            double dN_dx = iJ11 * dN_dxi[k] + iJ12 * dN_deta[k];
            double dN_dy = iJ21 * dN_dxi[k] + iJ22 * dN_deta[k];
            dudx += dN_dx * elem_U[2 * k];
            dudy += dN_dy * elem_U[2 * k];
            dvdx += dN_dx * elem_U[2 * k + 1];
            dvdy += dN_dy * elem_U[2 * k + 1];
        }

        double exx = dudx;
        double eyy = dvdy;
        double exy = 0.5 * (dudy + dvdx);

        for (int k = 0; k < nn; ++k) {
            int nid = conn[k];
            Exx[nid] += exx;
            Eyy[nid] += eyy;
            Exy[nid] += exy;
            count[nid]++;
        }
    }

    for (int i = 0; i < n_nodes; ++i) {
        if (count[i] > 0) {
            Exx[i] /= count[i];
            Eyy[i] /= count[i];
            Exy[i] /= count[i];
        }
    }
}

} // namespace meshdic
