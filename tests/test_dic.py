"""Smoke tests for MeshDIC package (imports, config, mesh)."""

import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure src/ is on path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestImports:
    """Verify that all core modules can be imported."""

    def test_import_package(self):
        import mesh_dic
        assert mesh_dic.__version__ == "0.1.0"

    def test_import_config(self):
        from mesh_dic.config import MeshdicConfig, DEFAULT_CONFIG, config_from_dict
        assert DEFAULT_CONFIG.subset_radius == 31
        assert DEFAULT_CONFIG.max_iterations == 30

    def test_config_from_dict(self):
        from mesh_dic.config import config_from_dict
        cfg = config_from_dict({"subset_radius": 15, "max_iterations": 50})
        assert cfg.subset_radius == 15
        assert cfg.max_iterations == 50
        # Unchanged defaults
        assert cfg.search_radius == 20


class TestConfigFile:
    """Verify YAML config loading."""

    def test_load_default_yaml(self):
        import yaml

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "default.yaml"
        )
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        assert "solver" in cfg
        assert cfg["solver"]["alpha"] == 0.1
        assert cfg["solver"]["max_iter"] == 10
        assert cfg["mesh"]["size"] == 30.0


class TestMeshIO:
    """Verify mesh file read/write utilities."""

    def test_read_nodes(self):
        from mesh_dic.mesh_gen import read_nodes

        with tempfile.TemporaryDirectory() as tmpdir:
            node_file = os.path.join(tmpdir, "nodes.txt")
            with open(node_file, "w") as f:
                f.write("1, 0.0, 0.0\n")
                f.write("2, 1.0, 0.0\n")
                f.write("3, 1.0, 1.0\n")
                f.write("4, 0.0, 1.0\n")

            coords, id2idx = read_nodes(node_file)
            assert coords.shape == (4, 2)
            assert id2idx[1] == 0
            assert id2idx[4] == 3
            assert np.allclose(coords[0], [0.0, 0.0])

    def test_read_elements(self):
        from mesh_dic.mesh_gen import read_elements

        with tempfile.TemporaryDirectory() as tmpdir:
            elem_file = os.path.join(tmpdir, "elements.txt")
            with open(elem_file, "w") as f:
                f.write("1, 1, 5, 2, 6, 3, 7, 4, 8, 9\n")

            elems = read_elements(elem_file)
            assert len(elems) == 1
            assert elems[0] == [1, 5, 2, 6, 3, 7, 4, 8, 9]

    def test_load_mesh_roundtrip(self):
        from mesh_dic.mesh_gen import read_nodes, read_elements

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write minimal mesh files
            nodes_path = os.path.join(tmpdir, "nodes.txt")
            elems_path = os.path.join(tmpdir, "elements.txt")
            inform_path = os.path.join(tmpdir, "Inform.npy")

            with open(nodes_path, "w") as f:
                f.write("1, 0.0, 0.0\n")
                f.write("2, 1.0, 0.0\n")
                f.write("3, 1.0, 1.0\n")
                f.write("4, 0.0, 1.0\n")
                f.write("5, 0.5, 0.0\n")
                f.write("6, 1.0, 0.5\n")
                f.write("7, 0.5, 1.0\n")
                f.write("8, 0.0, 0.5\n")
                f.write("9, 0.5, 0.5\n")

            with open(elems_path, "w") as f:
                f.write("1, 1, 5, 2, 6, 3, 7, 4, 8, 9\n")

            # Create minimal Inform array: pixel (2,2) in element 1
            inform = np.array([[2, 2, 1]], dtype=np.float64)
            np.save(inform_path, inform)

            coords, id2idx = read_nodes(nodes_path)
            elems = read_elements(elems_path)

            assert len(coords) == 9
            assert len(elems) == 1
            assert elems[0] == [1, 5, 2, 6, 3, 7, 4, 8, 9]


class TestShapeFunctions:
    """Verify shape function computation via C++ binding."""

    def test_shape_functions_q8(self):
        from mesh_dic._core import shape_functions_q8

        result = shape_functions_q8(0.0, 0.0)
        N = np.array(result["N"])
        dN_dxi = np.array(result["dN_dxi"])
        dN_deta = np.array(result["dN_deta"])

        assert N.shape == (8,)
        assert dN_dxi.shape == (8,)
        assert dN_deta.shape == (8,)

        # Sum of N at center should be 1.0
        assert abs(np.sum(N) - 1.0) < 1e-12

    def test_shape_functions_q8_corners(self):
        from mesh_dic._core import shape_functions_q8

        corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        for k, (xi, eta) in enumerate(corners):
            result = shape_functions_q8(xi, eta)
            N = np.array(result["N"])
            # Corner node value should be 1.0, others near 0
            assert abs(N[k] - 1.0) < 1e-12, f"Corner {k}: N[{k}]={N[k]}"
            for j in range(4):
                if j != k:
                    assert abs(N[j]) < 1e-12, f"Corner {k}: N[{j}]={N[j]}"

    def test_shape_functions_q4(self):
        from mesh_dic._core import shape_functions_q4

        result = shape_functions_q4(0.0, 0.0)
        N = np.array(result["N"])
        dN_dxi = np.array(result["dN_dxi"])
        dN_deta = np.array(result["dN_deta"])

        assert N.shape == (4,)
        assert dN_dxi.shape == (4,)
        assert dN_deta.shape == (4,)
        assert abs(np.sum(N) - 1.0) < 1e-12

        # Each N_i(0,0) should be 0.25
        for i in range(4):
            assert abs(N[i] - 0.25) < 1e-12

    def test_shape_functions_q4_corners(self):
        from mesh_dic._core import shape_functions_q4

        corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        for k, (xi, eta) in enumerate(corners):
            result = shape_functions_q4(xi, eta)
            N = np.array(result["N"])
            assert abs(N[k] - 1.0) < 1e-12, f"Corner {k}: N[{k}]={N[k]}"
            for j in range(4):
                if j != k:
                    assert abs(N[j]) < 1e-12, f"Corner {k}: N[{j}]={N[j]}"

    def test_shape_functions_t3(self):
        from mesh_dic._core import shape_functions_t3

        # At centroid: xi=1/3, eta=1/3
        result = shape_functions_t3(1.0 / 3.0, 1.0 / 3.0)
        N = np.array(result["N"])
        dN_dxi = np.array(result["dN_dxi"])
        dN_deta = np.array(result["dN_deta"])

        assert N.shape == (3,)
        assert abs(np.sum(N) - 1.0) < 1e-12
        for i in range(3):
            assert abs(N[i] - 1.0 / 3.0) < 1e-12

        # Derivatives: dN_dxi = [-1, 1, 0], dN_deta = [-1, 0, 1]
        assert abs(dN_dxi[0] + 1.0) < 1e-12
        assert abs(dN_dxi[1] - 1.0) < 1e-12
        assert abs(dN_dxi[2]) < 1e-12
        assert abs(dN_deta[0] + 1.0) < 1e-12
        assert abs(dN_deta[1]) < 1e-12
        assert abs(dN_deta[2] - 1.0) < 1e-12

    def test_shape_functions_t3_vertices(self):
        from mesh_dic._core import shape_functions_t3

        # (0,0): N1=1, N2=0, N3=0
        result = shape_functions_t3(0.0, 0.0)
        N = np.array(result["N"])
        assert abs(N[0] - 1.0) < 1e-12
        assert abs(N[1]) < 1e-12
        assert abs(N[2]) < 1e-12

        # (1,0): N1=0, N2=1, N3=0
        result = shape_functions_t3(1.0, 0.0)
        N = np.array(result["N"])
        assert abs(N[1] - 1.0) < 1e-12

        # (0,1): N1=0, N2=0, N3=1
        result = shape_functions_t3(0.0, 1.0)
        N = np.array(result["N"])
        assert abs(N[2] - 1.0) < 1e-12


class TestPixelFieldInterpolation:
    """Verify FE shape-function interpolation to integer pixels."""

    def test_interpolate_t3_linear_field(self):
        from mesh_dic.field import interpolate_displacement_to_pixels

        g2l = {
            "h": 1,
            "w": 1,
            "xi": np.array([0.25]),
            "eta": np.array([0.25]),
            "valid": np.array([1], dtype=np.uint8),
            "elem_id": np.array([1], dtype=np.int32),
        }
        elements = np.array([[1, 2, 3]], dtype=np.int32)
        U = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        field = interpolate_displacement_to_pixels(g2l, elements, U, "T3")

        assert field["mask"][0, 0]
        assert np.isclose(field["u"][0, 0], 2.5)
        assert np.isclose(field["v"][0, 0], 3.5)

    def test_interpolate_q4_constant_field(self):
        from mesh_dic.field import interpolate_displacement_to_pixels

        g2l = {
            "h": 1,
            "w": 1,
            "xi": np.array([0.3]),
            "eta": np.array([-0.2]),
            "valid": np.array([1], dtype=np.uint8),
            "elem_id": np.array([1], dtype=np.int32),
        }
        elements = np.array([[1, 2, 3, 4]], dtype=np.int32)
        U = np.repeat([[2.0, -1.0]], 4, axis=0)

        field = interpolate_displacement_to_pixels(g2l, elements, U, "Q4")

        assert field["mask"][0, 0]
        assert np.isclose(field["u"][0, 0], 2.0)
        assert np.isclose(field["v"][0, 0], -1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
