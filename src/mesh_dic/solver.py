import os
from typing import Optional, Union

import numpy as np
import yaml

from ._core import (
    Config, BsplineEngine, ElementType,
    compute_global_to_local, G2LParams,
    global_icgn, global_forward_gn, compute_strain,
    LocalICGNSolver, LocalICGNParams,
)
from .config import MeshdicConfig, config_from_dict, DEFAULT_CONFIG
from .field import interpolate_displacement_to_pixels
from .mesh_gen import _ensure_mesh_with_loading, T3, Q4, Q8


def _str_to_element_type(s: str) -> ElementType:
    mapping = {"T3": ElementType.T3, "Q4": ElementType.Q4, "Q8": ElementType.Q8}
    if s not in mapping:
        raise ValueError(f"Unknown element_type '{s}'. Use 'T3', 'Q4', or 'Q8'.")
    return mapping[s]


def _init_nodal_displacement(ref_img, def_img, fx, fy, coords, eng,
                               bcoef_def, subset_radius=20, search_radius=15,
                               refine_subpixel=False, roi_mask=None,
                               local_max_iter=50, local_tol=1e-6,
                               local_lambda=1e-6):
    """Build nodal initial displacement from per-node local searches.

    Every node first gets a coarse integer-pixel template match.

    Returns *U_init* as flat array ``[u0, v0, u1, v1, ...]``.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    h, w = ref_img.shape
    n_nodes = len(coords)
    seed_u = np.full(n_nodes, np.nan, dtype=np.float64)
    seed_v = np.full(n_nodes, np.nan, dtype=np.float64)
    stats = {"coarse": 0, "subpixel": 0, "skipped": 0}

    params = LocalICGNParams()
    params.max_iter = int(local_max_iter)
    params.cutoff_diffnorm = float(local_tol)
    params.lambda_reg = float(local_lambda)

    solver = LocalICGNSolver(params)
    mask_pad = np.ones((h, w), dtype=np.uint8)
    local_dx = local_dy = None
    if refine_subpixel:
        offsets = np.arange(-subset_radius, subset_radius + 1, dtype=np.float64)
        local_xx, local_yy = np.meshgrid(offsets, offsets)
        circle = (local_xx * local_xx + local_yy * local_yy) <= subset_radius * subset_radius
        local_dx = local_xx[circle].ravel()
        local_dy = local_yy[circle].ravel()

    for i in tqdm(range(n_nodes), desc="Node init"):
        cx = int(round(coords[i, 0]))
        cy = int(round(coords[i, 1]))

        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            stats["skipped"] += 1
            continue
        if (cx - subset_radius - search_radius < 0
                or cx + subset_radius + search_radius >= w
                or cy - subset_radius - search_radius < 0
                or cy + subset_radius + search_radius >= h):
            stats["skipped"] += 1
            continue

        coarse = solver.coarse_search(
            ref_img, def_img, h, w,
            cx, cy, subset_radius, search_radius, mask_pad)
        u0, v0 = float(coarse.dx), float(coarse.dy)
        seed_u[i] = u0
        seed_v[i] = v0
        stats["coarse"] += 1

        if not refine_subpixel:
            continue

        if local_dx is None or local_dy is None:
            continue

        x_coords = (cx + local_dx).astype(np.int64)
        y_coords = (cy + local_dy).astype(np.int64)
        if (x_coords.min() < 0 or x_coords.max() >= w
                or y_coords.min() < 0 or y_coords.max() >= h):
            continue

        f_buffer = np.ascontiguousarray(ref_img[y_coords, x_coords], dtype=np.float64)
        fx_buffer = np.ascontiguousarray(fx[y_coords, x_coords], dtype=np.float64)
        fy_buffer = np.ascontiguousarray(fy[y_coords, x_coords], dtype=np.float64)

        result = solver.solve(
            f_buffer, fx_buffer, fy_buffer,
            float(cx), float(cy),
            np.ascontiguousarray(local_dx, dtype=np.float64),
            np.ascontiguousarray(local_dy, dtype=np.float64),
            eng, bcoef_def, u0, v0)

        if (np.isfinite([result.u, result.v]).all()
                and abs(result.u) <= search_radius + 2
                and abs(result.v) <= search_radius + 2):
            seed_u[i] = result.u
            seed_v[i] = result.v
            if result.success:
                stats["subpixel"] += 1

    good = np.isfinite(seed_u) & np.isfinite(seed_v)
    if good.sum() >= 3 and good.sum() < n_nodes:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        good_xy = coords[good]
        lin_u = LinearNDInterpolator(good_xy, seed_u[good])
        lin_v = LinearNDInterpolator(good_xy, seed_v[good])
        near_u = NearestNDInterpolator(good_xy, seed_u[good])
        near_v = NearestNDInterpolator(good_xy, seed_v[good])

        miss = ~good
        fill_u = lin_u(coords[miss])
        fill_v = lin_v(coords[miss])
        nan_fill = ~np.isfinite(fill_u) | ~np.isfinite(fill_v)
        if nan_fill.any():
            miss_xy = coords[miss]
            fill_u[nan_fill] = near_u(miss_xy[nan_fill])
            fill_v[nan_fill] = near_v(miss_xy[nan_fill])
        seed_u[miss] = fill_u
        seed_v[miss] = fill_v
    elif good.sum() == 0:
        seed_u[:] = 0.0
        seed_v[:] = 0.0
    else:
        seed_u[~good] = 0.0
        seed_v[~good] = 0.0

    U_init = np.zeros(2 * n_nodes, dtype=np.float64)
    U_init[0::2] = seed_u
    U_init[1::2] = seed_v

    print(
        "  Nodal init: "
        f"{stats['coarse']}/{n_nodes} coarse"
        f", {stats['subpixel']}/{n_nodes} subpixel"
        f", {stats['skipped']} skipped"
    )
    return U_init


def _fd7_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference FE-Global-DIC 7-point finite-difference image gradients."""
    kernel = np.array([-1 / 60, 3 / 20, -3 / 4, 0,
                       3 / 4, -3 / 20, 1 / 60], dtype=np.float64)
    fx = np.zeros_like(img, dtype=np.float64)
    fy = np.zeros_like(img, dtype=np.float64)
    for off, coeff in zip(range(-3, 4), kernel):
        fx[:, 3:-3] += coeff * img[:, 3 + off: img.shape[1] - 3 + off]
        fy[3:-3, :] += coeff * img[3 + off: img.shape[0] - 3 + off, :]
    return fx, fy


def solve(
    config_path: Optional[str] = None,
    ref_image: Optional[str] = None,
    def_image: Optional[str] = None,
    roi_mask: Optional[str] = None,
    mesh_dir: Optional[str] = None,
    mesh_size: Optional[float] = None,
    element_type: str = "Q8",
    external_file: Optional[str] = None,
    init_nodal: bool = True,
    init_subpixel: bool = True,
    init_subset_radius: int = 20,
    init_search_radius: int = 15,
    init_local_max_iter: int = 50,
    init_local_tol: float = 1e-6,
    init_local_lambda: float = 1e-6,
    alpha: Optional[float] = 1.0,
    beta: float = 0.0,
    method: str = "forward_gn",
    max_iter: Optional[int] = 10,
    tol: Optional[float] = 1e-3,
    **kwargs,
) -> dict:
    py_cfg = DEFAULT_CONFIG
    _alpha = alpha if alpha is not None else 1.0
    _max_iter = max_iter if max_iter is not None else 10
    _tol = tol if tol is not None else 1e-3
    _elem_type_str = element_type
    _external = external_file
    _method = method

    if config_path is not None:
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        if "solver" in yaml_cfg:
            py_cfg = config_from_dict(yaml_cfg["solver"])
            ref_image = ref_image or yaml_cfg.get("images", {}).get("ref")
            def_image = def_image or yaml_cfg.get("images", {}).get("def")
            roi_mask = roi_mask or yaml_cfg.get("images", {}).get("roi")
            mesh_dir = mesh_dir or yaml_cfg.get("mesh", {}).get("dir")
            mesh_size = mesh_size or yaml_cfg.get("mesh", {}).get("size", 30.0)
            _elem_type_str = yaml_cfg.get("mesh", {}).get("element_type", _elem_type_str)
            _external = _external or yaml_cfg.get("mesh", {}).get("external_file", "")
            init_nodal = yaml_cfg.get("solver", {}).get("init_nodal", init_nodal)
            init_subpixel = yaml_cfg.get("solver", {}).get("init_subpixel", init_subpixel)
            init_subset_radius = yaml_cfg.get("solver", {}).get("init_subset_radius", init_subset_radius)
            init_search_radius = yaml_cfg.get("solver", {}).get("init_search_radius", init_search_radius)
            init_local_max_iter = yaml_cfg.get("solver", {}).get("init_local_max_iter", init_local_max_iter)
            init_local_tol = yaml_cfg.get("solver", {}).get("init_local_tol", init_local_tol)
            init_local_lambda = yaml_cfg.get("solver", {}).get("init_local_lambda", init_local_lambda)
            _alpha = yaml_cfg.get("solver", {}).get("alpha", _alpha)
            beta = yaml_cfg.get("solver", {}).get("beta", beta)
            _method = yaml_cfg.get("solver", {}).get("method", _method)
            _max_iter = yaml_cfg.get("solver", {}).get("max_iter", _max_iter)
            _tol = yaml_cfg.get("solver", {}).get("tol", _tol)
    else:
        if ref_image is None or def_image is None or roi_mask is None:
            raise ValueError(
                "ref_image, def_image, and roi_mask must be provided "
                "either via config_path or as explicit arguments"
            )

    mesh_dir = mesh_dir or "./mesh/"
    mesh_size = mesh_size if mesh_size is not None else 30.0
    etype = _str_to_element_type(_elem_type_str)

    # 1. Load/generate mesh
    coords, elems, inform = _ensure_mesh_with_loading(
        roi_mask, mesh_size, mesh_dir, element_type=_elem_type_str,
        external_file=_external or None)

    # 2. Load images and normalize: (Img - mean) / std over ROI
    #    (matches funNormalizeImg.m in the reference FE-Global-DIC)
    from PIL import Image
    ref_raw = np.array(Image.open(ref_image).convert("L"), dtype=np.float64)
    def_raw = np.array(Image.open(def_image).convert("L"), dtype=np.float64)
    h, w = ref_raw.shape

    # ROI mask for computing normalization statistics
    roi_mask_img = np.array(Image.open(roi_mask).convert("L"), dtype=np.uint8)
    roi_mask_bool = roi_mask_img > 0

    def _normalize(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        roi_vals = img[mask]
        mean_val = roi_vals.mean()
        std_val = roi_vals.std()
        return (img - mean_val) / std_val

    ref = _normalize(ref_raw, roi_mask_bool)
    def_img = _normalize(def_raw, roi_mask_bool)

    # 3. B-spline processing
    eng = BsplineEngine(3)
    bcoef_ref = eng.compute_coefficients(ref)
    bcoef_def = eng.compute_coefficients(def_img)
    fx, fy = _fd7_gradients(ref)

    # 4. Global-to-local mapping
    g2l_params = G2LParams()
    g2l_params.max_iter = 200
    g2l = compute_global_to_local(inform, coords, elems, h, w, etype, g2l_params)

    # 5. Displacement initialisation
    if init_nodal:
        roi_init = np.array(Image.open(roi_mask).convert("L"), dtype=np.uint8)
        U_init = _init_nodal_displacement(
            ref, def_img, fx, fy, coords, eng, bcoef_def,
            subset_radius=init_subset_radius,
            search_radius=init_search_radius,
            refine_subpixel=init_subpixel,
            roi_mask=roi_init,
            local_max_iter=init_local_max_iter,
            local_tol=init_local_tol,
            local_lambda=init_local_lambda)
    else:
        U_init = np.zeros(2 * len(coords), dtype=np.float64)

    # 6. Global solve
    method_key = _method.lower()
    if method_key in ("forward_gn", "forward", "gn"):
        icgn_result = global_forward_gn(
            g2l, ref, fx, fy, coords, elems, U_init, eng, bcoef_def,
            etype, _alpha, _tol, _max_iter, beta)
    elif method_key in ("icgn", "frozen_gn", "fixed_hessian"):
        icgn_result = global_icgn(
            g2l, ref, fx, fy, coords, elems, U_init, eng, bcoef_def,
            etype, _alpha, _tol, _max_iter, beta)
    else:
        raise ValueError("Unknown solver method. Use 'forward_gn' or 'icgn'.")

    U = icgn_result["U"]
    n_nodes = len(coords)
    U_reshaped = np.column_stack([U[0::2], U[1::2]])
    U_init_reshaped = np.column_stack([U_init[0::2], U_init[1::2]])

    # 7. Strain
    strain = compute_strain(U, coords, elems, etype)
    pixel_field = interpolate_displacement_to_pixels(
        g2l, elems, U_reshaped, _elem_type_str)

    return {
        "U": U_reshaped,
        "U_init": U_init_reshaped,
        "U_pixel": pixel_field["u"],
        "V_pixel": pixel_field["v"],
        "field_mask": pixel_field["mask"],
        "nodes": coords,
        "elements": elems,
        "Exx": strain["Exx"],
        "Eyy": strain["Eyy"],
        "Exy": strain["Exy"],
        "iterations": icgn_result["iterations"],
        "method": method_key,
        "fem_size": n_nodes * 2,
    }
