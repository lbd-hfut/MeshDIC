"""Pixel-field interpolation from FE nodal values."""

from __future__ import annotations

import numpy as np


def _shape_values(element_type: str, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    etype = element_type.upper()
    if etype == "T3":
        return np.column_stack((1.0 - xi - eta, xi, eta))

    if etype == "Q4":
        return np.column_stack((
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ))

    if etype == "Q8":
        xi2 = xi * xi
        eta2 = eta * eta
        return np.column_stack((
            -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
            -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
            -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
            -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
            0.50 * (1.0 - xi2) * (1.0 - eta),
            0.50 * (1.0 + xi) * (1.0 - eta2),
            0.50 * (1.0 - xi2) * (1.0 + eta),
            0.50 * (1.0 - xi) * (1.0 - eta2),
        ))

    raise ValueError(f"Unknown element_type '{element_type}'. Use 'T3', 'Q4', or 'Q8'.")


def _in_bounds(element_type: str, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    etype = element_type.upper()
    if etype == "T3":
        return (xi >= -0.05) & (eta >= -0.05) & ((xi + eta) <= 1.05)
    if etype == "Q4":
        return (np.abs(xi) <= 1.5) & (np.abs(eta) <= 1.5)
    if etype == "Q8":
        return (np.abs(xi) <= 1.05) & (np.abs(eta) <= 1.05)
    raise ValueError(f"Unknown element_type '{element_type}'. Use 'T3', 'Q4', or 'Q8'.")


def interpolate_displacement_to_pixels(
    g2l: dict,
    elements: np.ndarray,
    U: np.ndarray,
    element_type: str,
) -> dict:
    """Interpolate nodal displacement to integer pixels inside FE elements.

    Parameters
    ----------
    g2l
        Output from ``compute_global_to_local``. Pixel ownership and local
        coordinates come from the same mapping used by the global solver.
    elements
        One-based element connectivity. Q8 meshes may contain a ninth center
        node; only the first eight serendipity nodes are used.
    U
        Nodal displacement as ``(n_nodes, 2)`` or flattened ``[u0, v0, ...]``.
    element_type
        ``"T3"``, ``"Q4"``, or ``"Q8"``.
    """
    h = int(g2l["h"])
    w = int(g2l["w"])
    xi = np.asarray(g2l["xi"], dtype=np.float64).reshape(-1)
    eta = np.asarray(g2l["eta"], dtype=np.float64).reshape(-1)
    valid = np.asarray(g2l["valid"], dtype=bool).reshape(-1)
    elem_id = np.asarray(g2l["elem_id"], dtype=np.int32).reshape(-1) - 1

    U_arr = np.asarray(U, dtype=np.float64)
    if U_arr.ndim == 1:
        U_arr = np.column_stack((U_arr[0::2], U_arr[1::2]))

    nn = {"T3": 3, "Q4": 4, "Q8": 8}[element_type.upper()]
    conn = np.asarray(elements, dtype=np.int32)[:, :nn] - 1

    pixel_ok = (
        valid
        & (elem_id >= 0)
        & (elem_id < len(conn))
        & _in_bounds(element_type, xi, eta)
    )

    u_field = np.full(h * w, np.nan, dtype=np.float64)
    v_field = np.full(h * w, np.nan, dtype=np.float64)
    if np.any(pixel_ok):
        pix = np.flatnonzero(pixel_ok)
        N = _shape_values(element_type, xi[pix], eta[pix])
        elem_conn = conn[elem_id[pix]]
        elem_u = U_arr[elem_conn, 0]
        elem_v = U_arr[elem_conn, 1]
        u_field[pix] = np.einsum("ij,ij->i", N, elem_u)
        v_field[pix] = np.einsum("ij,ij->i", N, elem_v)

    return {
        "u": u_field.reshape(h, w),
        "v": v_field.reshape(h, w),
        "mask": pixel_ok.reshape(h, w),
    }
