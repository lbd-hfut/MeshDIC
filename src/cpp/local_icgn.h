#pragma once

#include <cstdint>
#include <vector>

namespace meshdic {

struct LocalICGNParams {
    int max_iter = 25;
    double cutoff_diffnorm = 1e-3;
    double lambda_reg = 1e-3;
};

struct LocalICGNResult {
    bool success = false;
    double u = 0.0;
    double v = 0.0;
    double corr_coef = -1.0;
    double diffnorm = 0.0;
    int iterations = 0;
};

struct CoarseSearchResult {
    double dy = 0.0;
    double dx = 0.0;
};

enum class InterpMode { Ref, Def };

class LocalICGNSolver {
public:
    explicit LocalICGNSolver(const LocalICGNParams& params = LocalICGNParams{})
        : params_(params) {}

    LocalICGNResult solve(
        const double* f_buffer,
        const double* fx_ref,
        const double* fy_ref,
        int n_pixels,
        double xc, double yc,
        const double* dx, const double* dy,
        class BsplineEngine* bsp = nullptr,
        const double* def_bcoef = nullptr,
        int bcoef_h = 0, int bcoef_w = 0,
        double u0 = 0.0, double v0 = 0.0);

    CoarseSearchResult coarse_search(
        const double* ref_img,
        const double* def_img,
        int img_h, int img_w,
        int cx, int cy,
        int subset_radius,
        int search_radius,
        const uint8_t* mask_pad = nullptr,
        int mask_pad_h = 0, int mask_pad_w = 0);

private:
    LocalICGNParams params_;
};

} // namespace meshdic
