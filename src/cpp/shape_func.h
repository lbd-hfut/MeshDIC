#pragma once

#include <cstdint>

namespace meshdic {

enum class ElementType { T3 = 0, Q4 = 1, Q8 = 2 };

inline int nodes_per_element(ElementType t) {
    switch (t) {
        case ElementType::T3: return 3;
        case ElementType::Q4: return 4;
        case ElementType::Q8: return 8;
    }
    return 8;
}

inline int dofs_per_element(ElementType t) {
    return 2 * nodes_per_element(t);
}

// ---- Q8 (8-node quadratic quadrilateral, serendipity) ----
void shape_functions_q8(double xi, double eta,
                         double N[8], double dN_dxi[8], double dN_deta[8]);

// ---- Q4 (4-node bilinear quadrilateral) ----
void shape_functions_q4(double xi, double eta,
                         double N[4], double dN_dxi[4], double dN_deta[4]);

// ---- T3 (3-node linear triangle) ----
// Note: xi, eta are barycentric-like: N1 = 1 - xi - eta, N2 = xi, N3 = eta
void shape_functions_t3(double xi, double eta,
                         double N[3], double dN_dxi[3], double dN_deta[3]);

// ---- Generic dispatch ----
void shape_functions(ElementType type, double xi, double eta,
                     double* N, double* dN_dxi, double* dN_deta);

} // namespace meshdic
