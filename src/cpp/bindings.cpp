#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <opencv2/core.hpp>

#include "mesh_dic.h"
#include "bspline.h"
#include "local_icgn.h"
#include "shape_func.h"
#include "global2local.h"
#include "stiffness.h"
#include "strain.h"

namespace py = pybind11;

namespace {

cv::Mat numpy_to_mat(py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    int type = CV_64FC1;
    return cv::Mat(static_cast<int>(buf.shape[0]), static_cast<int>(buf.shape[1]),
                   type, buf.ptr);
}

py::array_t<double> mat_to_numpy(const cv::Mat& m) {
    auto result = py::array_t<double>({m.rows, m.cols});
    auto buf = result.request();
    std::memcpy(buf.ptr, m.ptr<double>(0), m.rows * m.cols * sizeof(double));
    return result;
}

template<typename T>
py::array_t<T> copy_to_numpy(const std::vector<T>& v) {
    auto arr = py::array_t<T>(v.size());
    auto buf = arr.request();
    std::memcpy(buf.ptr, v.data(), v.size() * sizeof(T));
    return arr;
}

py::dict solve_to_dict(const meshdic::MeshdicResult& r) {
    py::dict d;
    d["converged"] = r.converged;
    d["iterations"] = r.iterations;
    d["num_nodes"] = r.num_nodes;
    d["num_elements"] = r.num_elements;

    py::list norm_w;
    for (auto v : r.norm_of_W) norm_w.append(v);
    d["norm_of_W"] = norm_w;

    return d;
}

} // anonymous namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Mesh-DIC C++ core module";

    py::class_<meshdic::MeshdicConfig>(m, "Config")
        .def(py::init<>())
        .def_readwrite("subset_radius", &meshdic::MeshdicConfig::subset_radius)
        .def_readwrite("search_radius", &meshdic::MeshdicConfig::search_radius)
        .def_readwrite("max_iterations", &meshdic::MeshdicConfig::max_iterations)
        .def_readwrite("cutoff_diffnorm", &meshdic::MeshdicConfig::cutoff_diffnorm)
        .def_readwrite("lambda_reg", &meshdic::MeshdicConfig::lambda_reg)
        .def_readwrite("bcoef_border", &meshdic::MeshdicConfig::bcoef_border);

    m.def("solve", [](const std::string& ref_path,
                       const std::string& def_path,
                       const std::string& mesh_dir,
                       const meshdic::MeshdicConfig& cfg) {
        auto result = meshdic::solve(ref_path, def_path, mesh_dir, cfg);
        return solve_to_dict(result);
    }, py::arg("ref_path"), py::arg("def_path"),
       py::arg("mesh_dir"), py::arg("config"));

    py::class_<meshdic::BsplineEngine>(m, "BsplineEngine")
        .def(py::init<int>(), py::arg("border") = 3)
        .def("compute_coefficients", [](const meshdic::BsplineEngine& eng,
                                         py::array_t<double> img) {
            cv::Mat mat = numpy_to_mat(img);
            cv::Mat result = eng.compute_coefficients(mat);
            return mat_to_numpy(result);
        })
        .def("compute_gradients", [](const meshdic::BsplineEngine& eng,
                                      py::array_t<double> bcoef,
                                      py::array_t<unsigned char> roi) {
            auto buf_roi = roi.request();
            cv::Mat roi_mat(static_cast<int>(buf_roi.shape[0]),
                            static_cast<int>(buf_roi.shape[1]),
                            CV_8UC1, buf_roi.ptr);
            cv::Mat bcoef_mat = numpy_to_mat(bcoef);
            auto [fx, fy] = eng.compute_gradients(bcoef_mat, roi_mat);
            return py::make_tuple(mat_to_numpy(fx), mat_to_numpy(fy));
        })
        .def("interpolate", [](const meshdic::BsplineEngine& eng,
                                double x, double y,
                                py::array_t<double> bcoef) {
            cv::Mat bcoef_mat = numpy_to_mat(bcoef);
            return eng.interpolate(x, y, bcoef_mat);
        })
        .def("interpolate_batch", [](const meshdic::BsplineEngine& eng,
                                      py::array_t<double> xs,
                                      py::array_t<double> ys,
                                      py::array_t<double> bcoef) {
            auto buf_x = xs.request();
            auto buf_y = ys.request();
            int n = static_cast<int>(buf_x.shape[0]);
            cv::Mat bcoef_mat = numpy_to_mat(bcoef);
            py::array_t<double> result(n);
            auto buf_r = result.request();
            eng.interpolate_batch(static_cast<const double*>(buf_x.ptr),
                                   static_cast<const double*>(buf_y.ptr),
                                   n, bcoef_mat,
                                   static_cast<double*>(buf_r.ptr));
            return result;
        });

    py::class_<meshdic::LocalICGNParams>(m, "LocalICGNParams")
        .def(py::init<>())
        .def_readwrite("max_iter", &meshdic::LocalICGNParams::max_iter)
        .def_readwrite("cutoff_diffnorm", &meshdic::LocalICGNParams::cutoff_diffnorm)
        .def_readwrite("lambda_reg", &meshdic::LocalICGNParams::lambda_reg);

    py::class_<meshdic::LocalICGNResult>(m, "LocalICGNResult")
        .def(py::init<>())
        .def_readwrite("success", &meshdic::LocalICGNResult::success)
        .def_readwrite("u", &meshdic::LocalICGNResult::u)
        .def_readwrite("v", &meshdic::LocalICGNResult::v)
        .def_readwrite("corr_coef", &meshdic::LocalICGNResult::corr_coef)
        .def_readwrite("diffnorm", &meshdic::LocalICGNResult::diffnorm)
        .def_readwrite("iterations", &meshdic::LocalICGNResult::iterations);

    py::class_<meshdic::CoarseSearchResult>(m, "CoarseSearchResult")
        .def(py::init<>())
        .def_readwrite("dy", &meshdic::CoarseSearchResult::dy)
        .def_readwrite("dx", &meshdic::CoarseSearchResult::dx);

    py::class_<meshdic::LocalICGNSolver>(m, "LocalICGNSolver")
        .def(py::init<const meshdic::LocalICGNParams&>(),
             py::arg("params") = meshdic::LocalICGNParams{})
        .def("solve", [](meshdic::LocalICGNSolver& solver,
                          py::array_t<double> f_buffer,
                          py::array_t<double> fx_ref,
                          py::array_t<double> fy_ref,
                          double xc, double yc,
                          py::array_t<double> dx,
                          py::array_t<double> dy,
                          meshdic::BsplineEngine* bsp,
                          py::array_t<double> def_bcoef,
                          double u0, double v0) -> meshdic::LocalICGNResult {
            auto buf_f = f_buffer.request();
            auto buf_fx = fx_ref.request();
            auto buf_fy = fy_ref.request();
            auto buf_dx = dx.request();
            auto buf_dy = dy.request();
            auto buf_bcoef = def_bcoef.request();
            return solver.solve(
                static_cast<const double*>(buf_f.ptr),
                static_cast<const double*>(buf_fx.ptr),
                static_cast<const double*>(buf_fy.ptr),
                static_cast<int>(buf_f.shape[0]),
                xc, yc,
                static_cast<const double*>(buf_dx.ptr),
                static_cast<const double*>(buf_dy.ptr),
                bsp,
                static_cast<const double*>(buf_bcoef.ptr),
                static_cast<int>(buf_bcoef.shape[0]),
                static_cast<int>(buf_bcoef.shape[1]),
                u0, v0);
        },
        py::arg("f_buffer"), py::arg("fx_ref"), py::arg("fy_ref"),
        py::arg("xc"), py::arg("yc"),
        py::arg("dx"), py::arg("dy"),
        py::arg("bsp").none(false),
        py::arg("def_bcoef"),
        py::arg("u0") = 0.0, py::arg("v0") = 0.0)
        .def("coarse_search", [](meshdic::LocalICGNSolver& solver,
                                  py::array_t<double> ref_img,
                                  py::array_t<double> def_img,
                                  int img_h, int img_w,
                                  int cx, int cy,
                                  int subset_radius, int search_radius,
                                  py::array_t<uint8_t> mask_pad) -> meshdic::CoarseSearchResult {
            auto buf_ref = ref_img.request();
            auto buf_def = def_img.request();
            const uint8_t* mp = nullptr;
            int mph = 0, mpw = 0;
            if (mask_pad.size() > 0) {
                auto buf_mp = mask_pad.request();
                mp = static_cast<const uint8_t*>(buf_mp.ptr);
                mph = static_cast<int>(buf_mp.shape[0]);
                mpw = static_cast<int>(buf_mp.shape[1]);
            }
            return solver.coarse_search(
                static_cast<const double*>(buf_ref.ptr),
                static_cast<const double*>(buf_def.ptr),
                img_h, img_w, cx, cy,
                subset_radius, search_radius,
                mp, mph, mpw);
        },
        py::arg("ref_img"), py::arg("def_img"),
        py::arg("img_h"), py::arg("img_w"),
        py::arg("cx"), py::arg("cy"),
        py::arg("subset_radius"), py::arg("search_radius"),
        py::arg("mask_pad") = py::array_t<uint8_t>());

    // ---- ElementType enum ----
    py::enum_<meshdic::ElementType>(m, "ElementType")
        .value("T3", meshdic::ElementType::T3)
        .value("Q4", meshdic::ElementType::Q4)
        .value("Q8", meshdic::ElementType::Q8)
        .export_values();

    m.def("nodes_per_element", &meshdic::nodes_per_element);
    m.def("dofs_per_element", &meshdic::dofs_per_element);

    // ---- Shape function bindings ----
    m.def("shape_functions_q8", [](double xi, double eta) {
        double N[8], dN_dxi[8], dN_deta[8];
        meshdic::shape_functions_q8(xi, eta, N, dN_dxi, dN_deta);
        py::dict d;
        d["N"] = py::array_t<double>(8, N);
        d["dN_dxi"] = py::array_t<double>(8, dN_dxi);
        d["dN_deta"] = py::array_t<double>(8, dN_deta);
        return d;
    }, py::arg("xi"), py::arg("eta"));

    m.def("shape_functions_q4", [](double xi, double eta) {
        double N[4], dN_dxi[4], dN_deta[4];
        meshdic::shape_functions_q4(xi, eta, N, dN_dxi, dN_deta);
        py::dict d;
        d["N"] = py::array_t<double>(4, N);
        d["dN_dxi"] = py::array_t<double>(4, dN_dxi);
        d["dN_deta"] = py::array_t<double>(4, dN_deta);
        return d;
    }, py::arg("xi"), py::arg("eta"));

    m.def("shape_functions_t3", [](double xi, double eta) {
        double N[3], dN_dxi[3], dN_deta[3];
        meshdic::shape_functions_t3(xi, eta, N, dN_dxi, dN_deta);
        py::dict d;
        d["N"] = py::array_t<double>(3, N);
        d["dN_dxi"] = py::array_t<double>(3, dN_dxi);
        d["dN_deta"] = py::array_t<double>(3, dN_deta);
        return d;
    }, py::arg("xi"), py::arg("eta"));

    py::class_<meshdic::G2LParams>(m, "G2LParams")
        .def(py::init<>())
        .def_readwrite("tol_global", &meshdic::G2LParams::tol_global)
        .def_readwrite("tol_local", &meshdic::G2LParams::tol_local)
        .def_readwrite("max_iter", &meshdic::G2LParams::max_iter);

    m.def("compute_global_to_local", [](py::array_t<double> inform,
                                         py::array_t<double> nodes_coord,
                                         py::array_t<int> elements,
                                         int img_h, int img_w,
                                         meshdic::ElementType element_type,
                                         const meshdic::G2LParams& params) -> py::dict {
        auto bi = inform.request();
        auto bn = nodes_coord.request();
        auto be = elements.request();
        auto result = meshdic::compute_global_to_local(
            static_cast<const double*>(bi.ptr), static_cast<int>(bi.shape[0]),
            static_cast<const double*>(bn.ptr), static_cast<int>(bn.shape[0]),
            static_cast<const int*>(be.ptr), static_cast<int>(be.shape[0]),
            img_h, img_w, element_type, params);
        py::dict d;
        d["xi"] = copy_to_numpy(result.xi);
        d["eta"] = copy_to_numpy(result.eta);
        d["J11"] = copy_to_numpy(result.J11);
        d["J12"] = copy_to_numpy(result.J12);
        d["J21"] = copy_to_numpy(result.J21);
        d["J22"] = copy_to_numpy(result.J22);
        d["valid"] = copy_to_numpy(result.valid);
        d["elem_id"] = copy_to_numpy(result.elem_id);
        d["h"] = img_h;
        d["w"] = img_w;
        return d;
    }, py::arg("inform"), py::arg("nodes_coord"), py::arg("elements"),
       py::arg("img_h"), py::arg("img_w"),
       py::arg("element_type") = meshdic::ElementType::Q8,
       py::arg("params") = meshdic::G2LParams{});

    m.def("assemble_stiffness", [](py::dict g2l_dict,
                                    py::array_t<double> ref_img,
                                    py::array_t<double> fx_ref,
                                    py::array_t<double> fy_ref,
                                    py::array_t<double> nodes_coord,
                                    py::array_t<int> elements,
                                    meshdic::ElementType element_type,
                                    double alpha) -> py::dict {
        auto buf_r = ref_img.request();
        auto buf_fx = fx_ref.request();
        auto buf_fy = fy_ref.request();
        auto buf_n = nodes_coord.request();
        auto buf_e = elements.request();

        meshdic::G2LOutput g2l;
        g2l.img_h = py::cast<int>(g2l_dict["h"]);
        g2l.img_w = py::cast<int>(g2l_dict["w"]);
        int total = g2l.img_h * g2l.img_w;

        auto arr_xi = py::cast<py::array_t<double>>(g2l_dict["xi"]);
        auto arr_eta = py::cast<py::array_t<double>>(g2l_dict["eta"]);
        auto arr_v = py::cast<py::array_t<uint8_t>>(g2l_dict["valid"]);
        g2l.xi.assign(arr_xi.data(), arr_xi.data() + total);
        g2l.eta.assign(arr_eta.data(), arr_eta.data() + total);
        g2l.valid.assign(arr_v.data(), arr_v.data() + total);
        auto arr_eid = py::cast<py::array_t<int>>(g2l_dict["elem_id"]);
        g2l.elem_id.assign(arr_eid.data(), arr_eid.data() + total);
        auto set_j = [&](std::vector<double>& dst, const py::array_t<double>& src) {
            dst.assign(src.data(), src.data() + total);
        };
        set_j(g2l.J11, py::cast<py::array_t<double>>(g2l_dict["J11"]));
        set_j(g2l.J12, py::cast<py::array_t<double>>(g2l_dict["J12"]));
        set_j(g2l.J21, py::cast<py::array_t<double>>(g2l_dict["J21"]));
        set_j(g2l.J22, py::cast<py::array_t<double>>(g2l_dict["J22"]));

        auto cache = meshdic::assemble_stiffness(
            g2l, g2l.img_h, g2l.img_w,
            static_cast<const double*>(buf_fx.ptr),
            static_cast<const double*>(buf_fy.ptr),
            static_cast<int>(buf_n.shape[0]),
            static_cast<const int*>(buf_e.ptr),
            static_cast<int>(buf_e.shape[0]),
            element_type,
            alpha);

        // Return A in COO format
        py::dict out;
        std::vector<int> rows, cols;
        std::vector<double> vals;
        for (int k = 0; k < cache.A.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(cache.A, k); it; ++it) {
                rows.push_back((int)it.row());
                cols.push_back((int)it.col());
                vals.push_back(it.value());
            }
        }
        out["A_rows"] = copy_to_numpy(rows);
        out["A_cols"] = copy_to_numpy(cols);
        out["A_vals"] = copy_to_numpy(vals);
        out["A_shape"] = py::make_tuple(cache.fem_size, cache.fem_size);
        return out;
    }, py::arg("g2l_dict"), py::arg("ref_img"),
       py::arg("fx_ref"), py::arg("fy_ref"),
       py::arg("nodes_coord"), py::arg("elements"),
       py::arg("element_type") = meshdic::ElementType::Q8,
       py::arg("alpha") = 10.0);

    m.def("global_icgn", [](py::dict g2l_dict,
                             py::array_t<double> ref_img,
                             py::array_t<double> fx_ref,
                             py::array_t<double> fy_ref,
                             py::array_t<double> nodes_coord,
                             py::array_t<int> elements,
                             py::array_t<double> U_init,
                             meshdic::BsplineEngine* bsp,
                             py::array_t<double> def_bcoef,
                             meshdic::ElementType element_type,
                             double alpha, double tol, int max_iter,
                             double beta = 0.0) -> py::dict {
        auto buf_r = ref_img.request();
        auto buf_fx = fx_ref.request();
        auto buf_fy = fy_ref.request();
        auto buf_n = nodes_coord.request();
        auto buf_e = elements.request();
        auto buf_u = U_init.request();
        auto buf_d = def_bcoef.request();
        int img_h = py::cast<int>(g2l_dict["h"]);
        int img_w = py::cast<int>(g2l_dict["w"]);
        int total = img_h * img_w;

        meshdic::G2LOutput g2l; g2l.img_h = img_h; g2l.img_w = img_w;
        auto ld = [&](std::vector<double>& d, const char* k) {
            auto a = py::cast<py::array_t<double>>(g2l_dict[k]);
            d.assign(a.data(), a.data() + total);
        };
        ld(g2l.xi, "xi"); ld(g2l.eta, "eta");
        ld(g2l.J11, "J11"); ld(g2l.J12, "J12");
        ld(g2l.J21, "J21"); ld(g2l.J22, "J22");
        auto arr_v = py::cast<py::array_t<uint8_t>>(g2l_dict["valid"]);
        g2l.valid.assign(arr_v.data(), arr_v.data() + total);
        auto arr_eid = py::cast<py::array_t<int>>(g2l_dict["elem_id"]);
        g2l.elem_id.assign(arr_eid.data(), arr_eid.data() + total);

        int n_nodes = (int)buf_n.shape[0];
        int n_elements = (int)buf_e.shape[0];

        auto cache = meshdic::assemble_stiffness(
            g2l, img_h, img_w,
            (const double*)buf_fx.ptr, (const double*)buf_fy.ptr,
            n_nodes,
            (const int*)buf_e.ptr, n_elements,
            element_type, alpha, beta);

        int dofs = 2 * n_nodes;
        Eigen::VectorXd U(dofs);
        for (int i = 0; i < dofs; ++i) U(i) = ((const double*)buf_u.ptr)[i];

        int iters = meshdic::global_icgn(
            cache, g2l, (const double*)buf_r.ptr, img_h, img_w,
            (const int*)buf_e.ptr, n_elements,
            U, bsp, (const double*)buf_d.ptr,
            (int)buf_d.shape[0], (int)buf_d.shape[1],
            alpha, tol, max_iter, beta);

        py::dict out;
        auto U_copy = py::array_t<double>(dofs);
        std::memcpy(U_copy.request().ptr, U.data(), dofs * sizeof(double));
        out["U"] = U_copy;
        out["iterations"] = iters;
        return out;
    }, py::arg("g2l_dict"), py::arg("ref_img"),
       py::arg("fx_ref"), py::arg("fy_ref"),
       py::arg("nodes_coord"), py::arg("elements"),
       py::arg("U_init"), py::arg("bsp"),
       py::arg("def_bcoef"),
       py::arg("element_type") = meshdic::ElementType::Q8,
       py::arg("alpha") = 10.0, py::arg("tol") = 1e-3,
       py::arg("max_iter") = 30,
       py::arg("beta") = 0.0);

    m.def("global_forward_gn", [](py::dict g2l_dict,
                                   py::array_t<double> ref_img,
                                   py::array_t<double> fx_ref,
                                   py::array_t<double> fy_ref,
                                   py::array_t<double> nodes_coord,
                                   py::array_t<int> elements,
                                   py::array_t<double> U_init,
                                   meshdic::BsplineEngine* bsp,
                                   py::array_t<double> def_bcoef,
                                   meshdic::ElementType element_type,
                                   double alpha, double tol, int max_iter,
                                   double beta = 0.0) -> py::dict {
        auto buf_r = ref_img.request();
        auto buf_fx = fx_ref.request();
        auto buf_fy = fy_ref.request();
        auto buf_n = nodes_coord.request();
        auto buf_e = elements.request();
        auto buf_u = U_init.request();
        auto buf_d = def_bcoef.request();
        int img_h = py::cast<int>(g2l_dict["h"]);
        int img_w = py::cast<int>(g2l_dict["w"]);
        int total = img_h * img_w;

        meshdic::G2LOutput g2l; g2l.img_h = img_h; g2l.img_w = img_w;
        auto ld = [&](std::vector<double>& d, const char* k) {
            auto a = py::cast<py::array_t<double>>(g2l_dict[k]);
            d.assign(a.data(), a.data() + total);
        };
        ld(g2l.xi, "xi"); ld(g2l.eta, "eta");
        ld(g2l.J11, "J11"); ld(g2l.J12, "J12");
        ld(g2l.J21, "J21"); ld(g2l.J22, "J22");
        auto arr_v = py::cast<py::array_t<uint8_t>>(g2l_dict["valid"]);
        g2l.valid.assign(arr_v.data(), arr_v.data() + total);
        auto arr_eid = py::cast<py::array_t<int>>(g2l_dict["elem_id"]);
        g2l.elem_id.assign(arr_eid.data(), arr_eid.data() + total);

        int n_nodes = (int)buf_n.shape[0];
        int n_elements = (int)buf_e.shape[0];

        auto cache = meshdic::assemble_stiffness(
            g2l, img_h, img_w,
            (const double*)buf_fx.ptr, (const double*)buf_fy.ptr,
            n_nodes,
            (const int*)buf_e.ptr, n_elements,
            element_type, alpha, beta);

        int dofs = 2 * n_nodes;
        Eigen::VectorXd U(dofs);
        for (int i = 0; i < dofs; ++i) U(i) = ((const double*)buf_u.ptr)[i];

        int iters = meshdic::global_forward_gn(
            cache, g2l, (const double*)buf_r.ptr, img_h, img_w,
            (const int*)buf_e.ptr, n_elements,
            U, bsp, (const double*)buf_d.ptr,
            (int)buf_d.shape[0], (int)buf_d.shape[1],
            alpha, tol, max_iter, beta);

        py::dict out;
        auto U_copy = py::array_t<double>(dofs);
        std::memcpy(U_copy.request().ptr, U.data(), dofs * sizeof(double));
        out["U"] = U_copy;
        out["iterations"] = iters;
        return out;
    }, py::arg("g2l_dict"), py::arg("ref_img"),
       py::arg("fx_ref"), py::arg("fy_ref"),
       py::arg("nodes_coord"), py::arg("elements"),
       py::arg("U_init"), py::arg("bsp"),
       py::arg("def_bcoef"),
       py::arg("element_type") = meshdic::ElementType::Q8,
       py::arg("alpha") = 10.0, py::arg("tol") = 1e-3,
       py::arg("max_iter") = 30,
       py::arg("beta") = 0.0);

    m.def("_debug_residual", [](py::dict g2l_dict,
                                 py::array_t<double> ref_img,
                                 py::array_t<double> fx_ref,
                                 py::array_t<double> fy_ref,
                                 py::array_t<double> nodes_coord,
                                 py::array_t<int> elements,
                                 py::array_t<double> U_init,
                                 meshdic::BsplineEngine* bsp,
                                 py::array_t<double> def_bcoef,
                                 meshdic::ElementType element_type,
                                 double alpha) -> py::array_t<double> {
        // Same setup as global_icgn but returns residual b only
        auto buf_r = ref_img.request();
        auto buf_fx = fx_ref.request();
        auto buf_fy = fy_ref.request();
        auto buf_n = nodes_coord.request();
        auto buf_e = elements.request();
        auto buf_u = U_init.request();
        auto buf_d = def_bcoef.request();
        int img_h = py::cast<int>(g2l_dict["h"]);
        int img_w = py::cast<int>(g2l_dict["w"]);
        int total = img_h * img_w;

        meshdic::G2LOutput g2l; g2l.img_h = img_h; g2l.img_w = img_w;
        auto ld = [&](std::vector<double>& d, const char* k) {
            auto a = py::cast<py::array_t<double>>(g2l_dict[k]);
            d.assign(a.data(), a.data() + total);
        };
        ld(g2l.xi, "xi"); ld(g2l.eta, "eta");
        ld(g2l.J11, "J11"); ld(g2l.J12, "J12");
        ld(g2l.J21, "J21"); ld(g2l.J22, "J22");
        auto arr_v = py::cast<py::array_t<uint8_t>>(g2l_dict["valid"]);
        g2l.valid.assign(arr_v.data(), arr_v.data() + total);
        auto arr_eid = py::cast<py::array_t<int>>(g2l_dict["elem_id"]);
        g2l.elem_id.assign(arr_eid.data(), arr_eid.data() + total);

        int n_nodes = (int)buf_n.shape[0];
        int n_elements = (int)buf_e.shape[0];
        int dofs = 2 * n_nodes;

        Eigen::VectorXd U(dofs);
        for (int i = 0; i < dofs; ++i) U(i) = ((const double*)buf_u.ptr)[i];

        auto cache = meshdic::assemble_stiffness(
            g2l, img_h, img_w,
            (const double*)buf_fx.ptr, (const double*)buf_fy.ptr,
            n_nodes,
            (const int*)buf_e.ptr, n_elements,
            element_type, alpha);

        Eigen::VectorXd b = meshdic::assemble_residual(
            cache, g2l, (const double*)buf_r.ptr, img_h, img_w,
            (const int*)buf_e.ptr, n_elements,
            U, bsp, (const double*)buf_d.ptr,
            (int)buf_d.shape[0], (int)buf_d.shape[1], alpha);

        py::array_t<double> result(dofs);
        auto br = result.request();
        for (int i = 0; i < dofs; ++i) ((double*)br.ptr)[i] = b(i);
        return result;
    }, py::arg("g2l_dict"), py::arg("ref_img"),
       py::arg("fx_ref"), py::arg("fy_ref"),
       py::arg("nodes_coord"), py::arg("elements"),
       py::arg("U_init"), py::arg("bsp"),
       py::arg("def_bcoef"),
       py::arg("element_type") = meshdic::ElementType::Q8,
       py::arg("alpha") = 10.0);

    m.def("compute_strain", [](py::array_t<double> U,
                                py::array_t<double> nodes_coord,
                                py::array_t<int> elements,
                                meshdic::ElementType element_type) -> py::dict {
        auto bu = U.request();
        auto bn = nodes_coord.request();
        auto be = elements.request();
        int n_nodes = (int)bn.shape[0];
        int n_elements = (int)be.shape[0];

        std::vector<double> Exx, Eyy, Exy;
        meshdic::compute_strain(
            element_type,
            (const double*)bu.ptr, n_nodes,
            (const double*)bn.ptr,
            (const int*)be.ptr, n_elements,
            Exx, Eyy, Exy);

        py::dict out;
        auto make_copy = [](const std::vector<double>& v) {
            auto arr = py::array_t<double>(v.size());
            auto buf = arr.request();
            std::memcpy(buf.ptr, v.data(), v.size() * sizeof(double));
            return arr;
        };
        out["Exx"] = make_copy(Exx);
        out["Eyy"] = make_copy(Eyy);
        out["Exy"] = make_copy(Exy);
        return out;
    }, py::arg("U"), py::arg("nodes_coord"), py::arg("elements"),
       py::arg("element_type") = meshdic::ElementType::Q8);
}
