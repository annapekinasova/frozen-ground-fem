import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
)
from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
)
from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
    ThermalElement1D,
)


class TestThermalAnalysis1DInvalid(unittest.TestCase):
    def test_z_min_max_setters(self):
        msh = ThermalAnalysis1D((100, -8))
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)
        with self.assertRaises(ValueError):
            msh.z_min = "twelve"
        with self.assertRaises(ValueError):
            msh.z_min = 101.0
        with self.assertRaises(ValueError):
            msh.z_max = "twelve"
        with self.assertRaises(ValueError):
            msh.z_max = -8.0
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)

    def test_grid_size_setter(self):
        msh = ThermalAnalysis1D((100, -8))
        self.assertEqual(msh.grid_size, 0.0)
        with self.assertRaises(ValueError):
            msh.grid_size = "twelve"
        with self.assertRaises(ValueError):
            msh.grid_size = -0.5
        self.assertEqual(msh.grid_size, 0.0)

    def test_set_num_nodes_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_nodes = 5

    def test_set_nodes_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.nodes = ()

    def test_set_num_elements_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_elements = 5

    def test_set_elements_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.elements = ()

    def test_set_num_boundaries_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_boundaries = 3

    def test_set_boundaries_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.boundaries = ()

    def test_set_time_step_invalid_float(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = -0.1

    def test_set_time_step_invalid_int(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = -1

    def test_set_time_step_invalid_str0(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = "-0.1e-10"

    def test_set_time_step_invalid_str1(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = "three"

    def test_set_dt_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.dt = 0.1

    def test_set_over_dt_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.over_dt = 0.1

    def test_set_implicit_factor_invalid_float0(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = -0.1

    def test_set_implicit_factor_invalid_float1(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = 1.1

    def test_set_implicit_factor_invalid_int0(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = -1

    def test_set_implicit_factor_invalid_int1(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = 2

    def test_set_implicit_factor_invalid_str0(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = "-0.1e-10"

    def test_set_implicit_factor_invalid_str1(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = "three"

    def test_set_one_minus_alpha_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.one_minus_alpha = 0.1

    def test_set_implicit_error_tolerance_invalid_float(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = -0.1

    def test_set_implicit_error_tolerance_invalid_int(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = -1

    def test_set_implicit_error_tolerance_invalid_str0(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = "-0.1e-10"

    def test_set_implicit_error_tolerance_invalid_str1(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = "three"

    def test_set_eps_s_not_allowed(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.eps_s = 0.1

    def test_set_max_iterations_invalid_float0(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = -0.1

    def test_set_max_iterations_invalid_float1(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = 0.1

    def test_set_max_iterations_invalid_int(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.max_iterations = -1

    def test_set_max_iterations_invalid_str0(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = "-1"

    def test_set_max_iterations_invalid_str1(self):
        msh = ThermalAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = "three"

    def test_generate_mesh(self):
        msh = ThermalAnalysis1D()
        with self.assertRaises(ValueError):
            msh.generate_mesh()
        with self.assertRaises(ValueError):
            ThermalAnalysis1D(generate=True)
        self.assertFalse(msh.mesh_valid)
        msh.grid_size = np.inf
        msh.z_min = -8
        msh.z_max = 100
        with self.assertRaises(ValueError):
            msh.generate_mesh()
        self.assertFalse(msh.mesh_valid)
        with self.assertRaises(ValueError):
            msh.generate_mesh(order=2)
        with self.assertRaises(ValueError):
            msh.generate_mesh(num_elements=0)

    def test_add_boundary(self):
        msh = ThermalAnalysis1D((-8, 100), generate=True)
        nd = Node1D(0, 5.0)
        ip = IntegrationPoint1D(7.5)
        with self.assertRaises(TypeError):
            msh.add_boundary(nd)
        with self.assertRaises(ValueError):
            msh.add_boundary(ThermalBoundary1D((nd,)))
        with self.assertRaises(ValueError):
            msh.add_boundary(ThermalBoundary1D(
                (msh.nodes[0],),
                (ip,),
            ))

    def test_remove_boundary(self):
        msh = ThermalAnalysis1D((-8, 100), generate=True)
        bnd0 = ThermalBoundary1D((msh.nodes[0],))
        msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            (msh.nodes[-1],),
            (msh.elements[-1].int_pts[-1],),
        )
        with self.assertRaises(KeyError):
            msh.remove_boundary(bnd1)


class TestThermalAnalysis1DDefaults(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D()

    def test_zmin_zmax(self):
        self.assertEqual(self.msh.z_min, -np.inf)
        self.assertEqual(self.msh.z_max, np.inf)

    def test_mesh_valid(self):
        self.assertFalse(self.msh.mesh_valid)

    def test_grid_size(self):
        self.assertEqual(self.msh.grid_size, 0.0)

    def test_num_objects(self):
        self.assertEqual(self.msh.num_nodes, 0)
        self.assertEqual(self.msh.num_elements, 0)
        self.assertEqual(self.msh.num_boundaries, 0)

    def test_object_types(self):
        self.assertIsInstance(self.msh.nodes, tuple)
        self.assertIsInstance(self.msh.elements, tuple)
        self.assertIsInstance(self.msh.boundaries, set)

    def test_object_lens(self):
        self.assertEqual(len(self.msh.nodes), 0)
        self.assertEqual(len(self.msh.elements), 0)
        self.assertEqual(len(self.msh.boundaries), 0)


class TestThermalAnalysis1DLinear(unittest.TestCase):
    def test_create_mesh_no_args(self):
        msh = ThermalAnalysis1D(order=1)
        self.assertFalse(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 0)
        self.assertEqual(msh.num_elements, 0)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertTrue(np.isinf(msh.z_min))
        self.assertTrue(np.isinf(msh.z_max))
        self.assertTrue(msh.z_min < 0)
        self.assertTrue(msh.z_max > 0)
        self.assertEqual(msh.grid_size, 0.0)

    def test_create_mesh_z_range_generate(self):
        msh = ThermalAnalysis1D(
            z_range=(100, -8), num_elements=9, generate=True, order=1)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 10)
        self.assertEqual(msh.num_elements, 9)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)
        self.assertEqual(msh.grid_size, 0.0)
        self.assertAlmostEqual(msh.nodes[1].z - msh.nodes[0].z, 12.0)

    def test_z_min_max_setters(self):
        msh = ThermalAnalysis1D((100, -8), order=1)
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)
        msh.z_min = -7
        self.assertAlmostEqual(msh.z_min, -7.0)
        self.assertIsInstance(msh.z_min, float)
        msh.z_max = 101
        self.assertAlmostEqual(msh.z_max, 101.0)
        self.assertIsInstance(msh.z_max, float)

    def test_grid_size_setter(self):
        msh = ThermalAnalysis1D((100, -8), order=1)
        self.assertEqual(msh.grid_size, 0.0)
        msh.grid_size = 1
        self.assertAlmostEqual(msh.grid_size, 1.0)
        self.assertIsInstance(msh.grid_size, float)
        msh.generate_mesh(order=1)
        self.assertEqual(msh.num_nodes, 109)
        self.assertEqual(msh.num_elements, 108)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.nodes[1].z - msh.nodes[0].z, 1.0)

    def test_generate_mesh(self):
        msh = ThermalAnalysis1D()
        self.assertFalse(msh.mesh_valid)
        msh.z_min = -8
        msh.z_max = 100
        msh.grid_size = 0
        msh.generate_mesh(num_elements=9, order=1)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 10)
        self.assertEqual(msh.num_elements, 9)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.elements[0].jacobian, 12.0)
        msh.grid_size = 1
        self.assertFalse(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 0)
        self.assertEqual(msh.num_elements, 0)
        self.assertEqual(msh.num_boundaries, 0)
        msh.generate_mesh(order=1)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 109)
        self.assertEqual(msh.num_elements, 108)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.elements[0].jacobian, 1.0)


class TestThermalAnalysis1DCubic(unittest.TestCase):
    def test_create_mesh_no_args(self):
        msh = ThermalAnalysis1D()
        self.assertFalse(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 0)
        self.assertEqual(msh.num_elements, 0)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertTrue(np.isinf(msh.z_min))
        self.assertTrue(np.isinf(msh.z_max))
        self.assertTrue(msh.z_min < 0)
        self.assertTrue(msh.z_max > 0)
        self.assertEqual(msh.grid_size, 0.0)

    def test_create_mesh_z_range_generate(self):
        msh = ThermalAnalysis1D(
            z_range=(100, -8), num_elements=9, generate=True)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 28)
        self.assertEqual(msh.num_elements, 9)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)
        self.assertEqual(msh.grid_size, 0.0)
        self.assertAlmostEqual(msh.nodes[1].z - msh.nodes[0].z, 4.0)

    def test_z_min_max_setters(self):
        msh = ThermalAnalysis1D((100, -8))
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)
        msh.z_min = -7
        self.assertAlmostEqual(msh.z_min, -7.0)
        self.assertIsInstance(msh.z_min, float)
        msh.z_max = 101
        self.assertAlmostEqual(msh.z_max, 101.0)
        self.assertIsInstance(msh.z_max, float)

    def test_grid_size_setter(self):
        msh = ThermalAnalysis1D((100, -8))
        self.assertEqual(msh.grid_size, 0.0)
        msh.grid_size = 1
        self.assertAlmostEqual(msh.grid_size, 1.0)
        self.assertIsInstance(msh.grid_size, float)
        msh.generate_mesh()
        self.assertEqual(msh.num_nodes, 325)
        self.assertEqual(msh.num_elements, 108)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.nodes[1].z - msh.nodes[0].z, 1.0/3.0)

    def test_generate_mesh(self):
        msh = ThermalAnalysis1D()
        self.assertFalse(msh.mesh_valid)
        msh.z_min = -8
        msh.z_max = 100
        msh.generate_mesh(num_elements=9)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 28)
        self.assertEqual(msh.num_elements, 9)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.elements[0].jacobian, 12.0)
        msh.grid_size = 1
        self.assertFalse(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 0)
        self.assertEqual(msh.num_elements, 0)
        self.assertEqual(msh.num_boundaries, 0)
        msh.generate_mesh()
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 325)
        self.assertEqual(msh.num_elements, 108)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.elements[0].jacobian, 1.0)


class TestAddBoundaries(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D((-8, 100), generate=True)

    def test_add_boundary_no_int_pt(self):
        bnd = ThermalBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd)
        self.assertEqual(self.msh.num_boundaries, 1)
        self.assertTrue(bnd in self.msh.boundaries)
        bnd1 = ThermalBoundary1D((self.msh.nodes[-1],))
        self.msh.add_boundary(bnd1)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(bnd in self.msh.boundaries)

    def test_add_boundary_with_int_pt(self):
        bnd = ThermalBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd)
        bnd1 = ThermalBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(bnd1)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(bnd in self.msh.boundaries)


class TestRemoveBoundaries(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D((-8, 100), generate=True)
        self.bnd0 = ThermalBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(self.bnd0)
        self.bnd1 = ThermalBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(self.bnd1)

    def test_remove_boundary_by_ref(self):
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(self.bnd0 in self.msh.boundaries)
        self.msh.remove_boundary(self.bnd0)
        self.assertEqual(self.msh.num_boundaries, 1)
        self.assertFalse(self.bnd0 in self.msh.boundaries)
        self.msh.boundaries.discard(self.bnd0)
        self.assertEqual(self.msh.num_boundaries, 1)
        self.assertTrue(self.bnd1 in self.msh.boundaries)

    def test_clear_boundaries(self):
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(self.bnd0 in self.msh.boundaries)
        self.assertTrue(self.bnd1 in self.msh.boundaries)
        self.msh.clear_boundaries()
        self.assertEqual(self.msh.num_boundaries, 0)
        self.assertFalse(self.bnd0 in self.msh.boundaries)
        self.assertFalse(self.bnd1 in self.msh.boundaries)


if __name__ == "__main__":
    unittest.main()
