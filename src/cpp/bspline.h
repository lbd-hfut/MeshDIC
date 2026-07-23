#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace meshdic {

class BsplineEngine {
public:
    explicit BsplineEngine(int border = 3);

    cv::Mat compute_coefficients(const cv::Mat& image) const;

    std::pair<cv::Mat, cv::Mat> compute_gradients(
        const cv::Mat& bcoef, const cv::Mat& roi) const;

    double interpolate(double x, double y, const cv::Mat& bcoef) const;

    void interpolate_with_grad(
        double x, double y, const cv::Mat& bcoef,
        double& value, double& gx, double& gy) const;

    void interpolate_batch(
        const double* xs, const double* ys, int n,
        const cv::Mat& bcoef, double* out) const;

    const Eigen::Matrix<double, 6, 6>& QK() const { return QK_; }

private:
    int border_;
    static constexpr int kOffset = 2;

    Eigen::Matrix<double, 6, 6> QK_;

    Eigen::Matrix<double, 6, 6> compute_QK_B_QKT(
        const cv::Mat& bcoef, int y, int x) const;

    static double beta5_nth(double x, int n);

    static Eigen::Matrix<double, 6, 6> init_QK_matrix();
};

} // namespace meshdic
