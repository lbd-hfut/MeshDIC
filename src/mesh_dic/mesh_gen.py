import os
import numpy as np
from skimage import measure

import gmsh
import pygmsh

# Element type constants
T3 = "T3"
Q4 = "Q4"
Q8 = "Q8"


def extract_polygon_from_mask(mask, simplify_eps=2.0):
    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons = []
    for c in contours:
        poly = np.flip(c, axis=1)
        poly = measure.approximate_polygon(poly, tolerance=simplify_eps)
        if len(poly) < 4:
            continue
        polygons.append(poly)

    # Fallback: if mask fills the entire image (no contours found),
    # use the image bounding rectangle.
    if len(polygons) == 0:
        h, w = mask.shape
        rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
        polygons.append(rect)

    areas = [_polygon_area(p) for p in polygons]
    idx = np.argsort(areas)[::-1]
    return [polygons[i] for i in idx]


def _polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clean_polygon(poly, min_edge=1.0):
    """Remove nearly-duplicate consecutive vertices to avoid degenerate Gmsh edges."""
    if len(poly) < 3:
        return poly
    d = np.sqrt(np.sum((np.roll(poly, -1, axis=0) - poly) ** 2, axis=1))
    keep = d > min_edge
    return poly[keep]


def generate_mesh_from_mask(mask, mesh_size=30.0, simplify_eps=2.0,
                             element_type=Q8):
    """Generate mesh from binary mask using Gmsh.

    Parameters
    ----------
    mask : ndarray
        Binary mask (True/False or 0/255).
    mesh_size : float
        Target mesh element size.
    simplify_eps : float
        Polygon simplification tolerance.
    element_type : str
        'T3', 'Q4', or 'Q8'.
    """
    polys = extract_polygon_from_mask(mask, simplify_eps)
    outer = _clean_polygon(polys[0])
    holes = [_clean_polygon(h) for h in polys[1:]] if len(polys) > 1 else []

    with pygmsh.geo.Geometry() as geom:
        outer_loop = _add_polygon_to_geom(geom, outer, mesh_size)
        hole_loops = []
        for h in holes:
            hole_loops.append(_add_polygon_to_geom(geom, h, mesh_size))
        surface = geom.add_plane_surface(outer_loop, holes=hole_loops)

        if element_type in (Q4, Q8):
            geom.set_recombined_surfaces([surface])

        if element_type == Q8:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
        else:
            gmsh.option.setNumber("Mesh.ElementOrder", 1)

        mesh = geom.generate_mesh(dim=2)
    return mesh


def _add_polygon_to_geom(geom, poly, mesh_size):
    pts = [geom.add_point([x, y, 0], mesh_size=mesh_size) for x, y in poly]
    # Piecewise linear boundary — preserves sharp corners
    n = len(pts)
    lines = [geom.add_line(pts[i], pts[(i + 1) % n]) for i in range(n)]
    loop = geom.add_curve_loop(lines)
    return loop


def write_mesh_files(mesh, output_dir, element_type=Q8):
    """Write nodes and elements files from a Gmsh mesh.

    element_type determines which cell type to extract:
    - 'T3':  triangle   cells (3 nodes)
    - 'Q4':  quad       cells (4 nodes)
    - 'Q8':  quad9      cells (9 nodes, only 8 corner+edge used)
    """
    os.makedirs(output_dir, exist_ok=True)

    cell_key = {"T3": "triangle", "Q4": "quad", "Q8": "quad9"}[element_type]
    cells = mesh.cells_dict.get(cell_key)
    if cells is None:
        raise ValueError(
            f"mesh.cells_dict has no '{cell_key}' cells for element_type={element_type}")

    if cells.min() == 0:
        cells = cells + 1

    nodes_path, elements_path, _ = _mesh_files(output_dir, element_type)
    with open(elements_path, "w") as f:
        for i, conn in enumerate(cells, start=1):
            f.write(f"{i}, " + ", ".join(map(str, conn)) + "\n")

    nodes_all = mesh.points
    node_ids = sorted(set(cells.flatten().tolist()))
    with open(nodes_path, "w") as f:
        for nid in node_ids:
            x, y = nodes_all[nid - 1][:2]
            f.write(f"{nid}, {x:.6f}, {y:.6f}\n")

    return nodes_path, elements_path


def read_nodes(node_file):
    node_ids, coords = [], []
    with open(node_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            node_ids.append(int(parts[0]))
            coords.append([float(parts[1]), float(parts[2])])
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    return np.array(coords), id2idx


def read_elements(elem_file):
    elements = []
    with open(elem_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            elements.append([int(v) for v in parts[1:]])
    return elements


def _mesh_files(mesh_dir, element_type):
    """Return (nodes_file, elements_file, inform_file) — all type-specific."""
    return (
        os.path.join(mesh_dir, f"nodes_{element_type}.txt"),
        os.path.join(mesh_dir, f"elements_{element_type}.txt"),
        os.path.join(mesh_dir, f"Inform_{element_type}.npy"),
    )


def load_mesh(mesh_dir, element_type=Q8):
    nodes_file, elements_file, inform_file = _mesh_files(mesh_dir, element_type)
    coords, id2idx = read_nodes(nodes_file)
    elements_raw = read_elements(elements_file)
    elements_remapped = []
    for elem in elements_raw:
        elements_remapped.append([id2idx[nid] + 1 for nid in elem])
    elements_arr = np.array(elements_remapped, dtype=np.int32)
    inform = np.load(inform_file)
    return coords, elements_arr, inform


def build_inform(nodes_file, elements_file, output_dir, element_type=Q8):
    """Build Inform array mapping image pixels to element IDs.

    For T3:  boundary uses all 3 nodes (triangle)
    For Q4:  boundary uses 4 corner nodes (quadrilateral)
    For Q8:  boundary uses 8 corner+edge nodes (8-node polygon)
    """
    from matplotlib.path import Path

    coords, id2idx = read_nodes(nodes_file)
    elements = read_elements(elements_file)
    Inform = []

    for eid, conn in enumerate(elements, start=1):
        if element_type == T3:
            boundary_order = [0, 1, 2]
        elif element_type == Q4:
            boundary_order = [0, 1, 2, 3]
        else:  # Q8: quadratic edges — subdivide for better boundary approximation
            boundary_order = [0, 4, 1, 5, 2, 6, 3, 7]

        nodes_boundary = np.array([coords[id2idx[conn[k]]] for k in boundary_order])

        # For Q8, refine each edge by inserting midpoints → 16-vertex polygon
        if element_type == Q8:
            refined = []
            n = len(nodes_boundary)
            for i in range(n):
                refined.append(nodes_boundary[i])
                refined.append(0.5 * (nodes_boundary[i] +
                                      nodes_boundary[(i + 1) % n]))
            nodes_boundary = np.array(refined)

        path = Path(np.vstack([nodes_boundary, nodes_boundary[0]]))

        xmin = int(np.floor(nodes_boundary[:, 0].min()))
        xmax = int(np.ceil(nodes_boundary[:, 0].max()))
        ymin = int(np.floor(nodes_boundary[:, 1].min()))
        ymax = int(np.ceil(nodes_boundary[:, 1].max()))
        xv, yv = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        points = np.vstack([xv.ravel(), yv.ravel()]).T
        mask_in = path.contains_points(points)
        for pt in points[mask_in]:
            Inform.append([pt[0], pt[1], eid])

    Inform = np.array(Inform)
    _, _, inform_path = _mesh_files(output_dir, element_type)
    np.save(inform_path, Inform)
    return Inform


def create_mesh(mask, mesh_size=30.0, simplify_eps=2.0, output_dir="./mesh/",
                element_type=Q8):
    mesh = generate_mesh_from_mask(
        mask=np.array(mask), mesh_size=mesh_size,
        simplify_eps=simplify_eps, element_type=element_type)
    nodes_path, elements_path = write_mesh_files(mesh, output_dir, element_type)
    return build_inform(nodes_path, elements_path, output_dir, element_type)


def _ensure_mesh_with_loading(roi_mask_path, mesh_size, mesh_dir,
                               element_type=Q8, external_file=None):
    import os as _os
    _os.makedirs(mesh_dir, exist_ok=True)
    nodes_file, elements_file, inform_file = _mesh_files(mesh_dir, element_type)

    # If external mesh file is provided, import it (always, overwriting cache)
    if external_file:
        import_external_mesh(external_file, mesh_dir, element_type)
        return load_mesh(mesh_dir, element_type)

    # Cache hit: all three files exist
    if (_os.path.exists(nodes_file) and _os.path.exists(elements_file)
            and _os.path.exists(inform_file)):
        return load_mesh(mesh_dir, element_type)

    # Cache miss: auto-generate via Gmsh
    from PIL import Image as _Image
    mask = np.array(_Image.open(roi_mask_path).convert("L")) > 0
    create_mesh(mask=mask, mesh_size=mesh_size, output_dir=mesh_dir,
                element_type=element_type)
    return load_mesh(mesh_dir, element_type)


# ============================================================
# Boundary export / external mesh import
# ============================================================

def export_boundary(roi_mask_path, output_path, simplify_eps=2.0):
    """Extract ROI boundary polygon and write to a simple text file.

    Output format (one vertex per line)::

        # outer boundary (N vertices)
        x0, y0
        x1, y1
        ...
        # hole 1 (M vertices)
        x0, y0
        ...

    This file can be used as input for external FEM meshing tools
    (Abaqus, ANSYS, standalone Gmsh, etc.).

    Parameters
    ----------
    roi_mask_path : str
        Path to ROI mask image.
    output_path : str
        Path for the output boundary file.
    simplify_eps : float
        Polygon simplification tolerance passed to
        :func:`extract_polygon_from_mask`.

    Returns
    -------
    list of ndarray
        List of polygon arrays (outer first, then holes).
    """
    from PIL import Image as _Image
    mask = np.array(_Image.open(roi_mask_path).convert("L")) > 0
    polys = extract_polygon_from_mask(mask, simplify_eps)

    with open(output_path, "w") as f:
        for idx, poly in enumerate(polys):
            tag = "outer boundary" if idx == 0 else f"hole {idx}"
            f.write(f"# {tag} ({len(poly)} vertices)\n")
            for x, y in poly:
                f.write(f"{x:.6f}, {y:.6f}\n")
            f.write("\n")

    return polys


def import_external_mesh(mesh_file, output_dir, element_type=Q8,
                          format='auto'):
    """Import a mesh generated by an external FEM tool.

    Uses ``meshio`` to read common formats: Abaqus (.inp), Gmsh (.msh),
    Nastran (.bdf), VTK (.vtu), and many others.  Also supports plain
    nodes/elements text files matching this package's internal format.

    After import, ``nodes.txt``, ``elements.txt``, and ``Inform.npy``
    are written to *output_dir*, making the mesh usable by the solver.

    Parameters
    ----------
    mesh_file : str
        Path to the external mesh file.
    output_dir : str
        Directory to write the internal mesh files.
    element_type : str
        'T3', 'Q4', or 'Q8'. Must match the element type in the file.
    format : str
        'auto' (detect from extension), 'abaqus', 'gmsh', etc.
        See meshio documentation for supported formats.

    Returns
    -------
    tuple
        (nodes_path, elements_path, inform_path) of the written files.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import meshio
    except ImportError:
        raise ImportError(
            "meshio is required for external mesh import. "
            "Install with: pip install mesh_dic[mesh]  or  pip install meshio")

    # Read mesh
    m = meshio.read(mesh_file, file_format=None if format == 'auto' else format)

    # Extract cells of the requested type
    cell_key = {"T3": "triangle", "Q4": "quad", "Q8": "quad9"}[element_type]
    cell_blocks = m.cells_dict.get(cell_key, [])
    # meshio may return a list of cell blocks; concatenate them
    if len(cell_blocks) == 0:
        available = list(m.cells_dict.keys())
        raise ValueError(
            f"No '{cell_key}' cells found in {mesh_file}. "
            f"Available cell types: {available}. "
            f"Check that element_type='{element_type}' matches the mesh.")

    if isinstance(cell_blocks, list) and len(cell_blocks) > 0:
        if hasattr(cell_blocks[0], 'data'):
            cells = np.vstack([cb.data for cb in cell_blocks])
        else:
            cells = np.vstack(cell_blocks)
    else:
        cells = cell_blocks

    # Nodes
    nodes_all = m.points[:, :2]  # x, y only (ignore z)
    if nodes_all.shape[1] < 2:
        raise ValueError("Mesh must have at least 2D coordinates.")

    # Write nodes.txt (1-based IDs)
    nodes_path, elements_path, inform_path = _mesh_files(output_dir, element_type)
    with open(nodes_path, "w") as f:
        for i, (x, y) in enumerate(nodes_all, start=1):
            f.write(f"{i}, {x:.6f}, {y:.6f}\n")

    # Write elements.txt (1-based IDs)
    if cells.min() == 0:
        cells = cells + 1

    with open(elements_path, "w") as f:
        for i, conn in enumerate(cells, start=1):
            f.write(f"{i}, " + ", ".join(map(str, conn)) + "\n")

    # Build Inform
    build_inform(nodes_path, elements_path, output_dir, element_type)

    return nodes_path, elements_path, inform_path
