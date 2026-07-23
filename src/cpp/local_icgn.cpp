#include "local_icgn.h"
#include "bspline.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

namespace meshdic {

static Eigen::Matrix<double, 6, 1> build_df_dp_row(
    double X, double Y, double fx, double fy) {
    Eigen::Matrix<double, 6, 1> row;
    row << fx, fy,
           X * fx, Y * fx,
           X * fy, Y * fy;
    return row;
}

static bool inverse_compositional_update_affine(
    const Eigen::Matrix<double, 6, 1>& p_old,
    const Eigen::Matrix<double, 6, 1>& delta,
    Eigen::Matrix<double, 6, 1>& p_new) {

    double du = p_old(0), dv = p_old(1);
    double dudx = p_old(2), dudy = p_old(3);
    double dvdx = p_old(4), dvdy = p_old(5);

    double dp0 = delta(0), dp1 = delta(1);
    double dp2 = delta(2), dp3 = delta(3);
    double dp4 = delta(4), dp5 = delta(5);

    double denom = dp2 + dp5 + dp2 * dp5 - dp3 * dp4 + 1.0;
    if (!std::isfinite(denom) || std::abs(denom) < 1e-12) {
        return false;
    }

    p_new(0) = du - ((dudx + 1.0) * (dp0 + dp0 * dp5 - dp1 * dp3)) / denom
                 - (dudy * (dp1 - dp0 * dp4 + dp1 * dp2)) / denom;
    p_new(1) = dv - ((dvdy + 1.0) * (dp1 - dp0 * dp4 + dp1 * dp2)) / denom
                 - (dvdx * (dp0 + dp0 * dp5 - dp1 * dp3)) / denom;
    p_new(2) = ((dp5 + 1.0) * (dudx + 1.0)) / denom
                 - (dp4 * dudy) / denom - 1.0;
    p_new(3) = (dudy * (dp2 + 1.0)) / denom
                 - (dp3 * (dudx + 1.0)) / denom;
    p_new(4) = (dvdx * (dp5 + 1.0)) / denom
                 - (dp4 * (dvdy + 1.0)) / denom;
    p_new(5) = ((dp2 + 1.0) * (dvdy + 1.0)) / denom
                 - (dp3 * dvdx) / denom - 1.0;

    return p_new.allFinite();
}

LocalICGNResult LocalICGNSolver::solve(
    const double* f_buffer,
    const double* fx_ref,
    const double* fy_ref,
    int n_pixels,
    double xc, double yc,
    const double* dx, const double* dy,
    BsplineEngine* bsp,
    const double* def_bcoef,
    int bcoef_h, int bcoef_w,
    double u0, double v0) {

    LocalICGNResult result;
    int n = n_pixels;

    double fm = 0.0;
    for (int i = 0; i < n; ++i) fm += f_buffer[i];
    fm /= n;

    double deltaf_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = f_buffer[i] - fm;
        deltaf_sq += d * d;
    }
    if (deltaf_sq < params_.lambda_reg) {
        return result;
    }
    double deltaf_inv = 1.0 / std::sqrt(deltaf_sq);

    Eigen::MatrixXd df_dp(n, 6);
    for (int i = 0; i < n; ++i) {
        df_dp.row(i) = build_df_dp_row(dx[i], dy[i], fx_ref[i], fy_ref[i]);
    }

    Eigen::Matrix<double, 6, 6> H =
        2.0 * deltaf_inv * deltaf_inv * df_dp.transpose() * df_dp;
    for (int i = 0; i < 6; ++i) H(i, i) += params_.lambda_reg;

    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(H);
    if (ldlt.info() != Eigen::Success) {
        return result;
    }

    Eigen::Matrix<double, 6, 1> p = Eigen::Matrix<double, 6, 1>::Zero();
    p(0) = u0;
    p(1) = v0;

    std::vector<double> xs(n), ys(n);
    std::vector<double> g_buf(n);
    cv::Mat bcoef_mat(bcoef_h, bcoef_w, CV_64FC1, const_cast<double*>(def_bcoef));

    for (int iter = 0; iter < params_.max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double X = dx[i], Y = dy[i];
            xs[i] = xc + X + p(0) + p(2) * X + p(3) * Y;
            ys[i] = yc + Y + p(1) + p(4) * X + p(5) * Y;
            if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]) ||
                    xs[i] < 0.0 || xs[i] > static_cast<double>(bcoef_w - 1) ||
                    ys[i] < 0.0 || ys[i] > static_cast<double>(bcoef_h - 1)) {
                return result;
            }
        }

        if (bsp) {
            bsp->interpolate_batch(xs.data(), ys.data(), n, bcoef_mat, g_buf.data());
        } else {
            // Use constant (no deformed image available)
            std::fill(g_buf.begin(), g_buf.end(), 0.0);
        }

        double gm = 0.0;
        for (int i = 0; i < n; ++i) gm += g_buf[i];
        gm /= n;

        double deltag_sq = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = g_buf[i] - gm;
            deltag_sq += d * d;
        }
        if (deltag_sq < params_.lambda_reg) {
            return result;
        }
        double deltag_inv = 1.0 / std::sqrt(deltag_sq);

        double corr = 0.0;
        Eigen::VectorXd resid(n);
        for (int i = 0; i < n; ++i) {
            double f_norm = (f_buffer[i] - fm) * deltaf_inv;
            double g_norm = (g_buf[i] - gm) * deltag_inv;
            double diff = f_norm - g_norm;
            resid(i) = diff;
            corr += diff * diff;
        }

        Eigen::Matrix<double, 6, 1> grad =
            2.0 * deltaf_inv * df_dp.transpose() * resid;

        Eigen::Matrix<double, 6, 1> delta = -ldlt.solve(grad);

        double diffnorm = delta.norm();
        if (!std::isfinite(diffnorm)) {
            return result;
        }

        if (diffnorm < params_.cutoff_diffnorm) {
            result.success = true;
            result.u = p(0);
            result.v = p(1);
            result.corr_coef = corr;
            result.diffnorm = diffnorm;
            result.iterations = iter + 1;
            return result;
        }

        Eigen::Matrix<double, 6, 1> p_next;
        if (!inverse_compositional_update_affine(p, delta, p_next)) {
            return result;
        }

        p = p_next;
        if (!p.allFinite()) {
            return result;
        }
    }

    result.success = true;
    result.u = p(0);
    result.v = p(1);
    result.iterations = params_.max_iter;
    return result;
}

CoarseSearchResult LocalICGNSolver::coarse_search(
    const double* ref_img, const double* def_img,
    int img_h, int img_w,
    int cx, int cy,
    int subset_radius, int search_radius,
    const uint8_t* mask_pad, int mask_pad_h, int mask_pad_w) {

    CoarseSearchResult result;

    int y0 = std::max(0, cy - subset_radius);
    int y1 = std::min(img_h, cy + subset_radius + 1);
    int x0 = std::max(0, cx - subset_radius);
    int x1 = std::min(img_w, cx + subset_radius + 1);
    int rh = y1 - y0, rw = x1 - x0;
    if (rh <= 0 || rw <= 0) return result;

    cv::Mat ref_patch(rh, rw, CV_64FC1);
    for (int i = 0; i < rh; ++i)
        for (int j = 0; j < rw; ++j)
            ref_patch.at<double>(i, j) = ref_img[(y0 + i) * img_w + (x0 + j)];

    cv::Mat mask_patch;
    if (mask_pad) {
        mask_patch = cv::Mat(rh, rw, CV_8UC1);
        for (int i = 0; i < rh; ++i)
            for (int j = 0; j < rw; ++j)
                mask_patch.at<uint8_t>(i, j) =
                    mask_pad[(y0 + i) * mask_pad_w + (x0 + j)];
    }

    int sy0 = std::max(0, y0 - search_radius);
    int sy1 = std::min(img_h, y1 + search_radius + 1);
    int sx0 = std::max(0, x0 - search_radius);
    int sx1 = std::min(img_w, x1 + search_radius + 1);
    int sh = sy1 - sy0, sw = sx1 - sx0;

    cv::Mat search_def(sh, sw, CV_64FC1);
    for (int i = 0; i < sh; ++i)
        for (int j = 0; j < sw; ++j)
            search_def.at<double>(i, j) = def_img[(sy0 + i) * img_w + (sx0 + j)];

    cv::Mat ref_32f, search_32f;
    ref_patch.convertTo(ref_32f, CV_32F);
    search_def.convertTo(search_32f, CV_32F);

    cv::Mat res;
    if (mask_pad) {
        cv::matchTemplate(search_32f, ref_32f, res,
                          cv::TM_CCOEFF_NORMED, mask_patch);
    } else {
        cv::matchTemplate(search_32f, ref_32f, res, cv::TM_CCOEFF_NORMED);
    }

    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(res, nullptr, &max_val, nullptr, &max_loc);

    result.dy = (sy0 + max_loc.y) - y0;
    result.dx = (sx0 + max_loc.x) - x0;
    return result;
}

} // namespace meshdic
