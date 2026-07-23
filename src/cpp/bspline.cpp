#include "bspline.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace meshdic {

static void complex_divide(cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.size() == b.size() && a.channels() == 2 && b.channels() == 2);
    int total = a.rows * a.cols;
    double* pa = a.ptr<double>(0);
    const double* pb = b.ptr<double>(0);
    for (int i = 0; i < total; ++i) {
        double ar = pa[2 * i], ai = pa[2 * i + 1];
        double br = pb[2 * i], bi = pb[2 * i + 1];
        double denom = br * br + bi * bi + 1e-30;
        pa[2 * i] = (ar * br + ai * bi) / denom;
        pa[2 * i + 1] = (ai * br - ar * bi) / denom;
    }
}

static double my_factorial(int k) {
    double f = 1.0;
    for (int i = 2; i <= k; ++i) f *= i;
    return f;
}

double BsplineEngine::beta5_nth(double x, int n) {
    const int coeffs[7] = {1, -6, 15, -20, 15, -6, 1};
    const int shifts[7] = {-3, -2, -1, 0, 1, 2, 3};
    int order = 5 - n;
    double result = 0.0;
    int factor = static_cast<int>(120.0 / my_factorial(5 - n));
    for (int i = 0; i < 7; ++i) {
        double x_shifted = x + shifts[i];
        if (x_shifted <= 0.0) continue;
        double pow_val = std::pow(x_shifted, order);
        result += coeffs[i] * factor * pow_val;
    }
    return result / 120.0;
}

Eigen::Matrix<double, 6, 6> BsplineEngine::init_QK_matrix() {
    const double x_samples[6] = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    Eigen::Matrix<double, 6, 6> QK;
    auto factorial = [](int k) {
        double f = 1.0;
        for (int i = 2; i <= k; ++i) f *= i;
        return f;
    };
    for (int n = 0; n < 6; ++n) {
        double sign = (n % 2 == 0) ? 1.0 : -1.0;
        for (int i = 0; i < 6; ++i) {
            QK(n, i) = sign * beta5_nth(x_samples[i], n) / factorial(n);
        }
    }
    return QK;
}

BsplineEngine::BsplineEngine(int border)
    : border_(border), QK_(init_QK_matrix()) {
    if (border_ < 3) border_ = 3;
}

cv::Mat BsplineEngine::compute_coefficients(const cv::Mat& image) const {
    CV_Assert(image.type() == CV_64FC1);
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, border_, border_, border_, border_,
                       cv::BORDER_REPLICATE);

    int h = padded.rows;
    int w = padded.cols;

    // Build FFT kernel for rows
    cv::Mat kernel_x = cv::Mat::zeros(1, w, CV_64FC1);
    double* kx = kernel_x.ptr<double>(0);
    kx[0] = beta5_nth(0.0, 0);
    kx[1] = beta5_nth(1.0, 0);
    kx[2] = beta5_nth(2.0, 0);
    kx[w - 2] = beta5_nth(-2.0, 0);
    kx[w - 1] = beta5_nth(-1.0, 0);

    cv::Mat kernel_x_fft;
    cv::dft(kernel_x, kernel_x_fft, cv::DFT_COMPLEX_OUTPUT);

    // Build FFT kernel for columns
    cv::Mat kernel_y = cv::Mat::zeros(h, 1, CV_64FC1);
    for (int i = 0; i < h; ++i) kernel_y.at<double>(i, 0) = 0.0;
    kernel_y.at<double>(0, 0) = beta5_nth(0.0, 0);
    kernel_y.at<double>(1, 0) = beta5_nth(1.0, 0);
    kernel_y.at<double>(2, 0) = beta5_nth(2.0, 0);
    kernel_y.at<double>(h - 2, 0) = beta5_nth(-2.0, 0);
    kernel_y.at<double>(h - 1, 0) = beta5_nth(-1.0, 0);

    cv::Mat kernel_y_fft;
    cv::dft(kernel_y, kernel_y_fft, cv::DFT_COMPLEX_OUTPUT);

    // Row-wise FFT deconvolution
    cv::Mat bcoef_rows;
    {
        cv::Mat tmp;
        padded.convertTo(tmp, CV_64FC1);
        cv::Mat rows_fft;
        cv::dft(tmp, rows_fft, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);
        for (int r = 0; r < h; ++r) {
            cv::Mat row = rows_fft.row(r);
            complex_divide(row, kernel_x_fft);
        }
        cv::idft(rows_fft, bcoef_rows, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE | cv::DFT_ROWS);
    }

    // Column-wise FFT deconvolution (1D per column, matching Python's loop)
    cv::Mat bcoef_all = bcoef_rows.clone();
    for (int c = 0; c < w; ++c) {
        cv::Mat col = bcoef_all.col(c);
        cv::Mat col_fft;
        cv::dft(col, col_fft, cv::DFT_COMPLEX_OUTPUT);
        complex_divide(col_fft, kernel_y_fft);
        cv::Mat col_out;
        cv::idft(col_fft, col_out, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
        col_out.copyTo(bcoef_all.col(c));
    }
    return bcoef_all;
}

Eigen::Matrix<double, 6, 6> BsplineEngine::compute_QK_B_QKT(
    const cv::Mat& bcoef, int y, int x) const {

    int top = y + border_ - kOffset;
    int left = x + border_ - kOffset;

    Eigen::Matrix<double, 6, 6> block;
    for (int i = 0; i < 6; ++i) {
        const double* row = bcoef.ptr<double>(top + i) + left;
        for (int j = 0; j < 6; ++j) {
            block(i, j) = row[j];
        }
    }

    return QK_ * block * QK_.transpose();
}

std::pair<cv::Mat, cv::Mat> BsplineEngine::compute_gradients(
    const cv::Mat& bcoef, const cv::Mat& roi) const {

    int H = roi.rows, W = roi.cols;
    cv::Mat fx = cv::Mat::zeros(H, W, CV_64FC1);
    cv::Mat fy = cv::Mat::zeros(H, W, CV_64FC1);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (roi.at<uchar>(y, x) == 0) continue;
            auto temp = compute_QK_B_QKT(bcoef, y, x);
            fx.at<double>(y, x) = temp(0, 1);
            fy.at<double>(y, x) = temp(1, 0);
        }
    }
    return {fx, fy};
}

double BsplineEngine::interpolate(double x, double y,
                                   const cv::Mat& bcoef) const {
    int x_floor = static_cast<int>(std::floor(x));
    int y_floor = static_cast<int>(std::floor(y));
    double xd = x - x_floor;
    double yd = y - y_floor;

    auto M = compute_QK_B_QKT(bcoef, y_floor, x_floor);

    Eigen::Matrix<double, 1, 6> y_vec, x_vec;
    for (int i = 0; i < 6; ++i) {
        y_vec(i) = std::pow(yd, i);
        x_vec(i) = std::pow(xd, i);
    }

    Eigen::Matrix<double, 1, 6> tmp = y_vec * M;
    return tmp.dot(x_vec);
}

void BsplineEngine::interpolate_with_grad(
    double x, double y, const cv::Mat& bcoef,
    double& value, double& gx, double& gy) const {

    int x_floor = static_cast<int>(std::floor(x));
    int y_floor = static_cast<int>(std::floor(y));
    double xd = x - x_floor;
    double yd = y - y_floor;

    auto M = compute_QK_B_QKT(bcoef, y_floor, x_floor);

    Eigen::Matrix<double, 1, 6> y_vec, x_vec, dy_vec, dx_vec;
    for (int i = 0; i < 6; ++i) {
        y_vec(i) = std::pow(yd, i);
        x_vec(i) = std::pow(xd, i);
        dy_vec(i) = (i == 0) ? 0.0 : i * std::pow(yd, i - 1);
        dx_vec(i) = (i == 0) ? 0.0 : i * std::pow(xd, i - 1);
    }

    value = (y_vec * M).dot(x_vec);
    gx = (y_vec * M).dot(dx_vec);
    gy = (dy_vec * M).dot(x_vec);
}

void BsplineEngine::interpolate_batch(
    const double* xs, const double* ys, int n,
    const cv::Mat& bcoef, double* out) const {

    for (int i = 0; i < n; ++i) {
        out[i] = interpolate(xs[i], ys[i], bcoef);
    }
}

} // namespace meshdic
