#include "global2local.h"
#include "shape_func.h"

#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

namespace meshdic {

// ============================================================
// T3: direct affine solve
// ============================================================
static bool solve_point_t3(double gx, double gy,
                            const double* elem_nodes,
                            double& xi, double& eta,
                            double& J11, double& J12, double& J21, double& J22) {
    // x = x1 + (x2-x1)*xi + (x3-x1)*eta
    // y = y1 + (y2-y1)*xi + (y3-y1)*eta
    double dx_dxi  = elem_nodes[2] - elem_nodes[0];
    double dx_deta = elem_nodes[4] - elem_nodes[0];
    double dy_dxi  = elem_nodes[3] - elem_nodes[1];
    double dy_deta = elem_nodes[5] - elem_nodes[1];

    double det = dx_dxi * dy_deta - dx_deta * dy_dxi;
    if (std::abs(det) < 1e-12) return false;

    double inv_det = 1.0 / det;
    double dx = gx - elem_nodes[0];
    double dy = gy - elem_nodes[1];

    xi  = inv_det * ( dy_deta * dx - dx_deta * dy);
    eta = inv_det * (-dy_dxi  * dx + dx_dxi  * dy);

    if (xi < -0.05 || eta < -0.05 || xi + eta > 1.05) return false;

    J11 = dx_dxi;  J12 = dx_deta;
    J21 = dy_dxi;  J22 = dy_deta;
    return true;
}

// ============================================================
// Q4: simple Newton iteration
// ============================================================
static bool solve_point_q4(double gx, double gy,
                            const double* elem_nodes,
                            double& xi, double& eta,
                            double& J11, double& J12, double& J21, double& J22,
                            int max_iter) {
    xi = 0.0; eta = 0.0;

    for (int it = 0; it < max_iter; ++it) {
        double N[4], dN_dxi[4], dN_deta[4];
        shape_functions_q4(xi, eta, N, dN_dxi, dN_deta);

        double xp = 0.0, yp = 0.0;
        for (int i = 0; i < 4; ++i) {
            xp += N[i] * elem_nodes[2 * i];
            yp += N[i] * elem_nodes[2 * i + 1];
        }
        double rx = gx - xp, ry = gy - yp;
        if (std::sqrt(rx * rx + ry * ry) < 0.1) {
            if (std::abs(xi) > 1.2 || std::abs(eta) > 1.2) return false;
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 4; ++i) {
                J11 += dN_dxi[i]  * elem_nodes[2 * i];
                J12 += dN_deta[i] * elem_nodes[2 * i];
                J21 += dN_dxi[i]  * elem_nodes[2 * i + 1];
                J22 += dN_deta[i] * elem_nodes[2 * i + 1];
            }
            return true;
        }

        double j11 = 0.0, j12 = 0.0, j21 = 0.0, j22 = 0.0;
        for (int i = 0; i < 4; ++i) {
            j11 += dN_dxi[i]  * elem_nodes[2 * i];
            j12 += dN_deta[i] * elem_nodes[2 * i];
            j21 += dN_dxi[i]  * elem_nodes[2 * i + 1];
            j22 += dN_deta[i] * elem_nodes[2 * i + 1];
        }
        double det = j11 * j22 - j12 * j21;
        if (std::abs(det) < 1e-12) return false;

        double inv_det = 1.0 / det;
        xi  += inv_det * (j22 * rx - j12 * ry);
        eta += inv_det * (-j21 * rx + j11 * ry);

        if (xi  < -2.0) xi  = -2.0;
        if (xi  >  2.0) xi  =  2.0;
        if (eta < -2.0) eta = -2.0;
        if (eta >  2.0) eta =  2.0;
    }
    return false;
}

// ============================================================
// Q8: full Newton-Raphson with fallback
// ============================================================
static bool solve_point_q8(double gx, double gy,
                            const double* elem8,
                            double xi0, double eta0,
                            double& xi, double& eta,
                            double& J11, double& J12, double& J21, double& J22,
                            double tol, int max_iter) {
    xi = xi0; eta = eta0;
    const double max_step = 0.5;
    const double clip = 2.0;
    for (int it = 0; it < max_iter; ++it) {
        double N[8], dN_dxi[8], dN_deta[8];
        shape_functions_q8(xi, eta, N, dN_dxi, dN_deta);

        double xp = 0.0, yp = 0.0;
        for (int i = 0; i < 8; ++i) {
            xp += N[i] * elem8[2 * i];
            yp += N[i] * elem8[2 * i + 1];
        }
        double rx = gx - xp, ry = gy - yp;
        if (std::sqrt(rx * rx + ry * ry) < tol) {
            if (std::abs(xi) > 2.0 || std::abs(eta) > 2.0) return false;
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 8; ++i) {
                J11 += dN_dxi[i]  * elem8[2 * i];
                J12 += dN_deta[i] * elem8[2 * i];
                J21 += dN_dxi[i]  * elem8[2 * i + 1];
                J22 += dN_deta[i] * elem8[2 * i + 1];
            }
            return true;
        }

        double j11 = 0.0, j12 = 0.0, j21 = 0.0, j22 = 0.0;
        for (int i = 0; i < 8; ++i) {
            j11 += dN_dxi[i]  * elem8[2 * i];
            j12 += dN_deta[i] * elem8[2 * i];
            j21 += dN_dxi[i]  * elem8[2 * i + 1];
            j22 += dN_deta[i] * elem8[2 * i + 1];
        }
        double det = j11 * j22 - j12 * j21;
        if (std::abs(det) < 1e-12) return false;
        double inv_det = 1.0 / det;
        double dxi  = inv_det * (j22 * rx - j12 * ry);
        double deta = inv_det * (-j21 * rx + j11 * ry);

        double dn = std::sqrt(dxi * dxi + deta * deta);
        if (dn > max_step) {
            double s = max_step / dn;
            dxi  *= s; deta *= s;
        }
        xi  += dxi;
        eta += deta;
        if (xi  < -clip) xi  = -clip;
        if (xi  >  clip) xi  =  clip;
        if (eta < -clip) eta = -clip;
        if (eta >  clip) eta =  clip;
    }
    return false;
}

static bool solve_point_q8_fallback(double gx, double gy,
                                     const double* elem8,
                                     double xi0, double eta0,
                                     double& xi, double& eta,
                                     double& J11, double& J12, double& J21, double& J22,
                                     double tol_global, double tol_local, int max_iter) {
    double xmin = elem8[0], xmax = elem8[0];
    double ymin = elem8[1], ymax = elem8[1];
    for (int i = 0; i < 8; ++i) {
        xmin = std::min(xmin, elem8[2 * i]);
        xmax = std::max(xmax, elem8[2 * i]);
        ymin = std::min(ymin, elem8[2 * i + 1]);
        ymax = std::max(ymax, elem8[2 * i + 1]);
    }
    double cx = 0.5 * (xmin + xmax), cy = 0.5 * (ymin + ymax);
    double sx = 0.5 * (xmax - xmin) * 2.0;
    double sy = 0.5 * (ymax - ymin) * 2.0;
    sx = (sx != 0.0) ? sx : 1.0;
    sy = (sy != 0.0) ? sy : 1.0;

    double nodes_norm[16];
    for (int i = 0; i < 8; ++i) {
        nodes_norm[2 * i]     = (elem8[2 * i]     - cx) / sx;
        nodes_norm[2 * i + 1] = (elem8[2 * i + 1] - cy) / sy;
    }
    double px = (gx - cx) / sx, py = (gy - cy) / sy;

    xi = xi0; eta = eta0;
    double prev_xi = xi, prev_eta = eta;
    double prev2_xi = xi, prev2_eta = eta;

    for (int it = 0; it < max_iter; ++it) {
        double N[8], dN_dxi[8], dN_deta[8];
        shape_functions_q8(xi, eta, N, dN_dxi, dN_deta);

        double xp = 0.0, yp = 0.0;
        for (int i = 0; i < 8; ++i) {
            xp += N[i] * nodes_norm[2 * i];
            yp += N[i] * nodes_norm[2 * i + 1];
        }
        double rx = px - xp, ry = py - yp;

        double xp_orig = 0.0, yp_orig = 0.0;
        for (int i = 0; i < 8; ++i) {
            xp_orig += N[i] * elem8[2 * i];
            yp_orig += N[i] * elem8[2 * i + 1];
        }
        double res_orig = std::sqrt((gx - xp_orig) * (gx - xp_orig) +
                                     (gy - yp_orig) * (gy - yp_orig));

        if (res_orig < tol_global) {
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 8; ++i) {
                J11 += dN_dxi[i]  * elem8[2 * i];
                J12 += dN_deta[i] * elem8[2 * i];
                J21 += dN_dxi[i]  * elem8[2 * i + 1];
                J22 += dN_deta[i] * elem8[2 * i + 1];
            }
            return true;
        }

        double jn11 = 0.0, jn12 = 0.0, jn21 = 0.0, jn22 = 0.0;
        for (int i = 0; i < 8; ++i) {
            jn11 += dN_dxi[i]  * nodes_norm[2 * i];
            jn12 += dN_deta[i] * nodes_norm[2 * i];
            jn21 += dN_dxi[i]  * nodes_norm[2 * i + 1];
            jn22 += dN_deta[i] * nodes_norm[2 * i + 1];
        }
        double det = jn11 * jn22 - jn12 * jn21;
        if (std::abs(det) < 1e-12) return false;

        double inv_det = 1.0 / det;
        double dxi  = inv_det * (jn22 * rx - jn12 * ry);
        double deta = inv_det * (-jn21 * rx + jn11 * ry);

        double max_step = 0.3;
        if (std::abs(det) < 1e-2) {
            double dn = std::sqrt(dxi * dxi + deta * deta);
            if (dn > max_step) {
                double s = max_step / dn;
                dxi  *= s; deta *= s;
            }
        }

        double alpha = 1.0;
        double xi_trial  = xi  + alpha * dxi;
        double eta_trial = eta + alpha * deta;

        if (std::abs(xi_trial - prev2_xi) < 1e-12 &&
            std::abs(eta_trial - prev2_eta) < 1e-12) {
            alpha = 0.25;
            xi_trial  = xi  + alpha * dxi;
            eta_trial = eta + alpha * deta;
        }

        for (int bt = 0; bt < 10; ++bt) {
            double Nb[8], dNb_dxi[8], dNb_deta[8];
            shape_functions_q8(xi_trial, eta_trial, Nb, dNb_dxi, dNb_deta);
            double xb = 0.0, yb = 0.0;
            for (int i = 0; i < 8; ++i) {
                xb += Nb[i] * elem8[2 * i];
                yb += Nb[i] * elem8[2 * i + 1];
            }
            double res_b = std::sqrt((gx - xb) * (gx - xb) + (gy - yb) * (gy - yb));
            if (res_b <= res_orig * 1.01 || alpha < 1e-4) break;
            alpha *= 0.5;
            xi_trial  = xi  + alpha * dxi;
            eta_trial = eta + alpha * deta;
        }

        prev2_xi = prev_xi; prev2_eta = prev_eta;
        prev_xi = xi; prev_eta = eta;
        xi  = xi_trial; eta = eta_trial;

        double step_norm = alpha * std::sqrt(dxi * dxi + deta * deta);
        if (step_norm < tol_local) {
            double Nf[8], dNf_dxi[8], dNf_deta[8];
            shape_functions_q8(xi, eta, Nf, dNf_dxi, dNf_deta);
            J11 = 0.0; J12 = 0.0; J21 = 0.0; J22 = 0.0;
            for (int i = 0; i < 8; ++i) {
                J11 += dNf_dxi[i]  * elem8[2 * i];
                J12 += dNf_deta[i] * elem8[2 * i];
                J21 += dNf_dxi[i]  * elem8[2 * i + 1];
                J22 += dNf_deta[i] * elem8[2 * i + 1];
            }
            return true;
        }

        if (res_orig > 100.0) return false;  // divergence guard
    }
    return false;
}

// ============================================================
// Main entry point – dispatches by element type
// ============================================================
G2LOutput compute_global_to_local(
    const double* inform, int n_pixels,
    const double* nodes_coord, int n_nodes,
    const int* elements, int n_elements,
    int img_h, int img_w,
    ElementType element_type,
    const G2LParams& params) {

    G2LOutput out;
    int total = img_h * img_w;
    out.xi.assign(total, 0.0);
    out.eta.assign(total, 0.0);
    out.J11.assign(total, 0.0);
    out.J12.assign(total, 0.0);
    out.J21.assign(total, 0.0);
    out.J22.assign(total, 0.0);
    out.valid.assign(total, 0);
    out.elem_id.assign(total, -1);
    out.img_h = img_h;
    out.img_w = img_w;

    int nn = nodes_per_element(element_type);
    int elem_stride = (element_type == ElementType::Q8) ? 9 : nn;  // Q8 stored as quad9

    // Build per-element node arrays
    std::vector<std::vector<double>> elem_nodes(n_elements);
    for (int e = 0; e < n_elements; ++e) {
        elem_nodes[e].resize(2 * nn);
        for (int k = 0; k < nn; ++k) {
            int nid = elements[e * elem_stride + k] - 1;
            elem_nodes[e][2 * k]     = nodes_coord[2 * nid];
            elem_nodes[e][2 * k + 1] = nodes_coord[2 * nid + 1];
        }
    }

    // Process each pixel
    for (int p = 0; p < n_pixels; ++p) {
        double gx = inform[p * 3];
        double gy = inform[p * 3 + 1];
        int eid = static_cast<int>(inform[p * 3 + 2]) - 1;
        int idx = static_cast<int>(gy) * img_w + static_cast<int>(gx);
        if (idx < 0 || idx >= total) continue;

        double xi, eta, J11, J12, J21, J22;
        bool ok = false;

        switch (element_type) {
            case ElementType::T3:
                ok = solve_point_t3(gx, gy, elem_nodes[eid].data(),
                                     xi, eta, J11, J12, J21, J22);
                break;
            case ElementType::Q4:
                ok = solve_point_q4(gx, gy, elem_nodes[eid].data(),
                                     xi, eta, J11, J12, J21, J22,
                                     params.max_iter);
                break;
            case ElementType::Q8:
                {
                double xi0 = 0.0, eta0 = 0.0;
                double q4_J11 = 0.0, q4_J12 = 0.0, q4_J21 = 0.0, q4_J22 = 0.0;
                bool q4_seed_ok = solve_point_q4(
                    gx, gy, elem_nodes[eid].data(),
                    xi0, eta0, q4_J11, q4_J12, q4_J21, q4_J22,
                    params.max_iter);
                if (!q4_seed_ok) {
                    xi0 = 0.0;
                    eta0 = 0.0;
                }

                ok = solve_point_q8(gx, gy, elem_nodes[eid].data(),
                                     xi0, eta0,
                                     xi, eta, J11, J12, J21, J22,
                                     params.tol_global, params.max_iter);
                if (!ok) {
                    ok = solve_point_q8_fallback(gx, gy, elem_nodes[eid].data(),
                                                  xi0, eta0,
                                                  xi, eta, J11, J12, J21, J22,
                                                  params.tol_global, params.tol_local,
                                                  params.max_iter);
                }
                break;
                }
        }

        if (ok) {
            out.xi[idx]  = xi;
            out.eta[idx] = eta;
            out.J11[idx] = J11;
            out.J12[idx] = J12;
            out.J21[idx] = J21;
            out.J22[idx] = J22;
            out.valid[idx] = 1;
            out.elem_id[idx] = eid + 1;
        }
    }

    return out;
}

} // namespace meshdic
