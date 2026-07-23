"""Run MeshDIC analysis on case images."""

import os
import sys
import warnings

import numpy as np

# Ensure src/ is on path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mesh_dic import solve

CASE_DIR = os.path.join(os.path.dirname(__file__), "..", "case")

ELEMENT_ALPHA = {
    "T3": 0.01,
    "Q4": 0.01,
    "Q8": 0.01,
}


def alpha_for_element(element_type: str) -> float:
    """Return the default regularization used by the formal case runner."""
    return ELEMENT_ALPHA.get(element_type.upper(), 0.01)


def run_case(case_name: str, mesh_size: float = 30.0, alpha: float = 10.0,
             max_iter: int = 30, tol: float = 1e-3,
             init_subset_radius: int = 20,
             init_search_radius: int = 15,
             method: str = "forward_gn",
             element_type: str = "Q8"):
    """Run Mesh-DIC on a case directory.

    Parameters
    ----------
    case_name : str
        Name of the case subdirectory under case/ (e.g. 'ring', 'star').
    mesh_size : float
        Mesh element size for Gmsh generation.
    alpha : float
        Tikhonov regularization strength.
    max_iter : int
        Maximum global ICGN iterations.
    tol : float
        Convergence tolerance for global ICGN.
    """
    case_dir = os.path.join(CASE_DIR, case_name)
    ref_img = os.path.join(case_dir, "001.bmp")
    def_img = os.path.join(case_dir, "002.bmp")
    roi_img = os.path.join(case_dir, "003.bmp")
    mesh_dir = os.path.join(case_dir, "mesh")
    out_dir = os.path.join(case_dir, "result", element_type)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Case '{case_name}':")
    print(f"  ref={ref_img}")
    print(f"  def={def_img}")
    print(f"  roi={roi_img}")
    print(f"  mesh_dir={mesh_dir}, mesh_size={mesh_size}, element_type={element_type}")
    print(f"  solver: method={method}, alpha={alpha}, max_iter={max_iter}, tol={tol}")
    print(
        "  init: "
        f"6-param local ICGN, radius={init_subset_radius}, "
        f"search={init_search_radius}"
    )

    try:
        result = solve(
            ref_image=ref_img,
            def_image=def_img,
            roi_mask=roi_img,
            mesh_dir=mesh_dir,
            mesh_size=mesh_size,
            element_type=element_type,
            alpha=alpha,
            method=method,
            max_iter=max_iter,
            tol=tol,
            init_nodal=True,
            init_subpixel=True,
            init_subset_radius=init_subset_radius,
            init_search_radius=init_search_radius,
            init_local_max_iter=50,
            init_local_tol=1e-6,
            init_local_lambda=1e-6,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    U = result["U"]
    U_init = result["U_init"]
    nodes = result["nodes"]
    elements = result["elements"]
    Exx = result["Exx"]
    Eyy = result["Eyy"]
    Exy = result["Exy"]
    U_pixel = result["U_pixel"]
    V_pixel = result["V_pixel"]
    field_mask = result["field_mask"]
    iterations = result["iterations"]

    n_nodes = len(nodes)
    n_elements = len(elements)

    print(f"  Converged in {iterations} iterations")
    print(f"  Nodes: {n_nodes}, Elements: {n_elements}")
    print(f"  U range: [{U[:, 0].min():.4f}, {U[:, 0].max():.4f}] x "
          f"[{U[:, 1].min():.4f}, {U[:, 1].max():.4f}]")
    print(f"  Exx range: [{Exx.min():.6f}, {Exx.max():.6f}]")
    print(f"  Eyy range: [{Eyy.min():.6f}, {Eyy.max():.6f}]")
    print(f"  Exy range: [{Exy.min():.6f}, {Exy.max():.6f}]")

    # Save results
    np.save(os.path.join(out_dir, "U.npy"), U)
    np.save(os.path.join(out_dir, "U_init.npy"), U_init)
    np.save(os.path.join(out_dir, "nodes.npy"), nodes)
    np.save(os.path.join(out_dir, "elements.npy"), elements)
    np.save(os.path.join(out_dir, "Exx.npy"), Exx)
    np.save(os.path.join(out_dir, "Eyy.npy"), Eyy)
    np.save(os.path.join(out_dir, "Exy.npy"), Exy)
    np.save(os.path.join(out_dir, "U_pixel.npy"), U_pixel)
    np.save(os.path.join(out_dir, "V_pixel.npy"), V_pixel)
    np.save(os.path.join(out_dir, "field_mask.npy"), field_mask)
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="ascii") as f:
        f.write(f"case={case_name}\n")
        f.write(f"element_type={element_type}\n")
        f.write(f"method={method}\n")
        f.write(f"alpha={alpha}\n")
        f.write(f"max_iter={max_iter}\n")
        f.write(f"tol={tol}\n")
        f.write(f"iterations={iterations}\n")

    print(f"  Results saved -> {out_dir}/")

    # Plot
    try:
        import matplotlib.pyplot as plt
        from PIL import Image

        ref = np.array(Image.open(ref_img).convert("L"))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(ref, cmap="gray")
        axes[0, 0].set_title("Reference Image")
        axes[0, 0].axis("off")

        # Pixel displacement interpolated with the FE shape functions.
        u_show = np.ma.masked_invalid(U_pixel)
        v_show = np.ma.masked_invalid(V_pixel)

        im_u = axes[0, 1].imshow(u_show, cmap="jet", origin="upper")
        axes[0, 1].set_title("U Displacement")
        axes[0, 1].axis("off")
        plt.colorbar(im_u, ax=axes[0, 1])

        im_v = axes[0, 2].imshow(v_show, cmap="jet", origin="upper")
        axes[0, 2].set_title("V Displacement")
        axes[0, 2].axis("off")
        plt.colorbar(im_v, ax=axes[0, 2])

        # Exx strain
        im_exx = axes[1, 0].scatter(nodes[:, 0], nodes[:, 1],
                                     c=Exx, cmap="RdBu_r", s=10)
        axes[1, 0].set_title("Exx Strain")
        axes[1, 0].axis("equal")
        axes[1, 0].invert_yaxis()
        plt.colorbar(im_exx, ax=axes[1, 0])

        # Eyy strain
        im_eyy = axes[1, 1].scatter(nodes[:, 0], nodes[:, 1],
                                     c=Eyy, cmap="RdBu_r", s=10)
        axes[1, 1].set_title("Eyy Strain")
        axes[1, 1].axis("equal")
        axes[1, 1].invert_yaxis()
        plt.colorbar(im_eyy, ax=axes[1, 1])

        # Exy strain
        im_exy = axes[1, 2].scatter(nodes[:, 0], nodes[:, 1],
                                     c=Exy, cmap="RdBu_r", s=10)
        axes[1, 2].set_title("Exy Strain")
        axes[1, 2].axis("equal")
        axes[1, 2].invert_yaxis()
        plt.colorbar(im_exy, ax=axes[1, 2])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "overview.png"), dpi=150)
        plt.close()
        print(f"  Overview plot saved")
    except Exception as e:
        print(f"  Plot error: {e}")

    print()


if __name__ == "__main__":
    for elem_type in ("T3", "Q4", "Q8"):
        run_case("ring", mesh_size=30.0, alpha=alpha_for_element(elem_type),
                 max_iter=30, tol=1e-3,
                 init_subset_radius=15, init_search_radius=15,
                 method="forward_gn",
                 element_type=elem_type)
    for elem_type in ("T3", "Q4", "Q8"):
        run_case("star", mesh_size=30.0, alpha=alpha_for_element(elem_type),
                 max_iter=30, tol=1e-3,
                 init_subset_radius=5, init_search_radius=5,
                 method="forward_gn",
                 element_type=elem_type)
