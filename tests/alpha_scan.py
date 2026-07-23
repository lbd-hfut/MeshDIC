"""Alpha sensitivity scan after image normalization fix.

Runs the solver on ring/star cases with a range of alpha values
and reports displacement/strain ranges for comparison.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mesh_dic import solve

CASE_DIR = os.path.join(os.path.dirname(__file__), "..", "case")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "alpha_scan")


def run_alpha_scan(case_name, element_type, alphas, mesh_size=30.0,
                   max_iter=30, tol=1e-3):
    """Run solver for a range of alpha values."""
    case_dir = os.path.join(CASE_DIR, case_name)
    ref_img = os.path.join(case_dir, "001.bmp")
    def_img = os.path.join(case_dir, "002.bmp")
    roi_img = os.path.join(case_dir, "003.bmp")
    mesh_dir = os.path.join(case_dir, "mesh")

    print(f"\n{'='*70}")
    print(f"Case: {case_name}  |  Element: {element_type}  |  Mesh: {mesh_size}px")
    print(f"{'='*70}")
    print(f"{'alpha':>10}  {'iter':>5}  {'Ux min':>10}  {'Ux max':>10}  "
          f"{'Uy min':>10}  {'Uy max':>10}  {'Exx min':>10}  {'Exx max':>10}  "
          f"{'Eyy min':>10}  {'Eyy max':>10}  {'time(s)':>8}")
    print("-" * 70)

    results = {}
    for alpha in alphas:
        t0 = time.perf_counter()
        try:
            result = solve(
                ref_image=ref_img,
                def_image=def_img,
                roi_mask=roi_img,
                mesh_dir=mesh_dir,
                mesh_size=mesh_size,
                element_type=element_type,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
            )
            elapsed = time.perf_counter() - t0
            iterations = result["iterations"]
            U = result["U"]
            Exx = result["Exx"]
            Eyy = result["Eyy"]
            Exy = result["Exy"]

            print(f"{alpha:10.0e}  {iterations:5d}  "
                  f"{U[:, 0].min():10.4f}  {U[:, 0].max():10.4f}  "
                  f"{U[:, 1].min():10.4f}  {U[:, 1].max():10.4f}  "
                  f"{Exx.min():10.6f}  {Exx.max():10.6f}  "
                  f"{Eyy.min():10.6f}  {Eyy.max():10.6f}  "
                  f"{elapsed:8.2f}")

            results[alpha] = {
                "U": U, "Exx": Exx, "Eyy": Eyy, "Exy": Exy,
                "iterations": iterations, "time": elapsed,
                "nodes": result["nodes"], "elements": result["elements"],
            }

            # Save for later plotting
            out_dir = os.path.join(RESULT_DIR, case_name, element_type,
                                   f"alpha_{alpha:.0e}")
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "U.npy"), U)
            np.save(os.path.join(out_dir, "Exx.npy"), Exx)
            np.save(os.path.join(out_dir, "Eyy.npy"), Eyy)
            np.save(os.path.join(out_dir, "Exy.npy"), Exy)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"{alpha:10.0e}  {'FAIL':>5}  -- {e}  "
                  f"{elapsed:8.2f}s")
            results[alpha] = None

    return results


def plot_alpha_trends(results, case_name, element_type):
    """Plot how displacement/strain statistics vary with alpha."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping trend plot.")
        return

    alphas = sorted([a for a, r in results.items() if r is not None])
    if len(alphas) < 2:
        return

    ux_range = [np.ptp(results[a]["U"][:, 0]) for a in alphas]
    uy_range = [np.ptp(results[a]["U"][:, 1]) for a in alphas]
    exx_range = [np.ptp(results[a]["Exx"]) for a in alphas]
    eyy_range = [np.ptp(results[a]["Eyy"]) for a in alphas]
    iters = [results[a]["iterations"] for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(alphas, ux_range, "o-", label="Ux range")
    axes[0].loglog(alphas, uy_range, "s-", label="Uy range")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("Displacement range (px)")
    axes[0].set_title(f"Displacement vs alpha ({case_name}/{element_type})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(alphas, exx_range, "o-", label="Exx range")
    axes[1].loglog(alphas, eyy_range, "s-", label="Eyy range")
    ax2 = axes[1].twinx()
    ax2.semilogx(alphas, iters, "D--", color="red", label="Iterations")
    ax2.set_ylabel("Iterations", color="red")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Strain range")
    axes[1].set_title(f"Strain & convergence vs alpha ({case_name}/{element_type})")
    axes[1].grid(True, alpha=0.3)
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    out_dir = os.path.join(RESULT_DIR, case_name, element_type)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "alpha_trends.png"), dpi=150)
    plt.close()
    print(f"  Trend plot saved -> {out_dir}/alpha_trends.png")


if __name__ == "__main__":
    # Wide sweep on both cases, Q8 element
    alphas_wide = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1]
    
    for case in ["ring"]:
        for etype in ["Q8"]:
            results = run_alpha_scan(case, etype, alphas_wide,
                                     mesh_size=30.0, max_iter=100, tol=1e-3)
            if results:
                plot_alpha_trends(results, case, etype)
    
    print("\nDone.")
