"""Visualize DIC results with FE-interpolated pixel displacement fields."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from mesh_dic import solve

CASES = {'ring': 30.0, 'star': 15.0}
TYPES = ['T3', 'Q4', 'Q8']

for case, mesh_size in CASES.items():
    h, w = np.array(Image.open(f'case/{case}/003.bmp').convert('L')).shape

    for etype in TYPES:
        mesh_dir = f'case/{case}/result/mesh/{etype}'
        out_dir = f'case/{case}/result/{etype}'
        os.makedirs(out_dir, exist_ok=True)

        print(f'{case}/{etype}: solving...', end=' ', flush=True)
        t0 = time.time()
        result = solve(
            ref_image=f'case/{case}/001.bmp',
            def_image=f'case/{case}/002.bmp',
            roi_mask=f'case/{case}/003.bmp',
            mesh_dir=mesh_dir, element_type=etype,
            alpha=1.0, max_iter=10, tol=1e-3)
        t = time.time() - t0

        U = result['U']; nodes = result['nodes']; elems = result['elements']
        U_pixel = result['U_pixel']; V_pixel = result['V_pixel']
        Exx = result['Exx']; Eyy = result['Eyy']; Exy = result['Exy']
        iters = result['iterations']
        print(f'iters={iters} |U|_rms={np.sqrt(np.mean(U**2)):.4f} |U|_max={np.abs(U).max():.3f} {t:.1f}s')

        np.save(os.path.join(out_dir, 'U.npy'), U)
        np.save(os.path.join(out_dir, 'U_pixel.npy'), U_pixel)
        np.save(os.path.join(out_dir, 'V_pixel.npy'), V_pixel)
        np.save(os.path.join(out_dir, 'field_mask.npy'), result['field_mask'])
        np.save(os.path.join(out_dir, 'Exx.npy'), Exx)
        np.save(os.path.join(out_dir, 'Eyy.npy'), Eyy)
        np.save(os.path.join(out_dir, 'Exy.npy'), Exy)

        # Element polygons
        if etype == 'T3':
            boundary = [0, 1, 2]
        elif etype == 'Q4':
            boundary = [0, 1, 2, 3]
        else:
            boundary = [0, 4, 1, 5, 2, 6, 3, 7]

        polys = [nodes[e[boundary] - 1] for e in elems]

        # Use max-over-element-nodes for color (better preserves peaks)
        def elem_vals(vals):
            idx = list(range(len(boundary)))
            return np.array([vals[e[idx] - 1].max() for e in elems])

        fig, axes = plt.subplots(2, 3, figsize=(20, 6.5 if case == 'star' else 14))
        fig.suptitle(f'{case} / {etype}  (alpha=1.0, |U|_max={np.abs(U).max():.2f} px, '
                     f'{len(nodes)}n, {len(elems)}e)', fontsize=13)

        pixel_fields = [(U_pixel, 'U displacement (px)'),
                        (V_pixel, 'V displacement (px)')]

        for (field, title), ax in zip(pixel_fields, axes.flat[:2]):
            im = ax.imshow(np.ma.masked_invalid(field), cmap='jet', origin='upper')
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, w); ax.set_ylim(h, 0)
            ax.set_aspect('equal'); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fields = [(Exx, 'Exx strain'),
                  (Eyy, 'Eyy strain'),
                  (Exy, 'Exy strain')]

        for idx, ((field, title), ax) in enumerate(zip(fields, axes.flat[2:5])):
            ev = elem_vals(field)
            # Use raw data range for colorbar, not element-averaged
            raw_max = max(abs(field.min()), abs(field.max()))
            if raw_max < 1e-12:
                raw_max = 1.0

            pc = PolyCollection(polys, array=ev, cmap='RdBu_r',
                                edgecolors='face', linewidths=0)
            pc.set_clim(-raw_max, raw_max)
            ax.add_collection(pc)
            ax.autoscale_view()
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, w); ax.set_ylim(h, 0)
            ax.set_aspect('equal'); ax.axis('off')
            plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 2]
        ref_img = np.array(Image.open(f'case/{case}/001.bmp').convert('L'))
        ax.imshow(ref_img, cmap='gray', origin='upper')
        step = max(1, len(nodes) // 2000)
        mag = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
        ax.quiver(nodes[::step, 0], nodes[::step, 1],
                  U[::step, 0], U[::step, 1], mag[::step],
                  angles='xy', scale_units='xy', scale=0.15,
                  cmap='jet', width=0.003)
        ax.set_title('Displacement vectors', fontsize=10)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')

        plt.tight_layout()
        fig_path = os.path.join(out_dir, 'overview.png')
        fig.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {fig_path}')

print('\nDone.')
