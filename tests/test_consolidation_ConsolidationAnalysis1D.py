import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
)
from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
)
from frozen_ground_fem.consolidation import (
    ConsolidationAnalysis1D,
    ConsolidationBoundary1D,
)


class TestConsolidationAnalysis1DInvalid(unittest.TestCase):
    def test_z_min_max_setters(self):
        msh = ConsolidationAnalysis1D((100, -8))
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
        msh = ConsolidationAnalysis1D((100, -8))
        self.assertEqual(msh.grid_size, 0.0)
        with self.assertRaises(ValueError):
            msh.grid_size = "twelve"
        with self.assertRaises(ValueError):
            msh.grid_size = -0.5
        self.assertEqual(msh.grid_size, 0.0)

    def test_set_num_nodes_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_nodes = 5

    def test_set_nodes_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.nodes = ()

    def test_set_num_elements_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_elements = 5

    def test_set_elements_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.elements = ()

    def test_set_num_boundaries_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_boundaries = 3

    def test_set_boundaries_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.boundaries = ()

    def test_set_time_step_invalid_float(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = -0.1

    def test_set_time_step_invalid_int(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = -1

    def test_set_time_step_invalid_str0(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = "-0.1e-10"

    def test_set_time_step_invalid_str1(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = "three"

    def test_set_dt_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.dt = 0.1

    def test_set_over_dt_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.over_dt = 0.1

    def test_set_implicit_factor_invalid_float0(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = -0.1

    def test_set_implicit_factor_invalid_float1(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = 1.1

    def test_set_implicit_factor_invalid_int0(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = -1

    def test_set_implicit_factor_invalid_int1(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = 2

    def test_set_implicit_factor_invalid_str0(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = "-0.1e-10"

    def test_set_implicit_factor_invalid_str1(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = "three"

    def test_set_one_minus_alpha_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.one_minus_alpha = 0.1

    def test_set_implicit_error_tolerance_invalid_float(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = -0.1

    def test_set_implicit_error_tolerance_invalid_int(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = -1

    def test_set_implicit_error_tolerance_invalid_str0(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = "-0.1e-10"

    def test_set_implicit_error_tolerance_invalid_str1(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = "three"

    def test_set_eps_s_not_allowed(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.eps_s = 0.1

    def test_set_max_iterations_invalid_float0(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = -0.1

    def test_set_max_iterations_invalid_float1(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = 0.1

    def test_set_max_iterations_invalid_int(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.max_iterations = -1

    def test_set_max_iterations_invalid_str0(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = "-1"

    def test_set_max_iterations_invalid_str1(self):
        msh = ConsolidationAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = "three"

    def test_generate_mesh(self):
        msh = ConsolidationAnalysis1D()
        with self.assertRaises(ValueError):
            msh.generate_mesh()
        with self.assertRaises(ValueError):
            ConsolidationAnalysis1D(generate=True)
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
        msh = ConsolidationAnalysis1D((-8, 100), generate=True)
        nd = Node1D(0, 5.0)
        ip = IntegrationPoint1D(7.5)
        with self.assertRaises(TypeError):
            msh.add_boundary(nd)
        with self.assertRaises(ValueError):
            msh.add_boundary(ConsolidationBoundary1D((nd,)))
        with self.assertRaises(ValueError):
            msh.add_boundary(ConsolidationBoundary1D(
                (msh.nodes[0],),
                (ip,),
            ))

    def test_remove_boundary(self):
        msh = ConsolidationAnalysis1D((-8, 100), generate=True)
        bnd0 = ConsolidationBoundary1D((msh.nodes[0],))
        msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            (msh.nodes[-1],),
            (msh.elements[-1].int_pts[-1],),
        )
        with self.assertRaises(KeyError):
            msh.remove_boundary(bnd1)


class TestConsolidationAnalysis1DDefaults(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D()

    def test_zmin_zmax(self):
        self.assertEqual(self.msh.z_min, -np.inf)
        self.assertEqual(self.msh.z_max, np.inf)

    def test_mesh_valid(self):
        self.assertFalse(self.msh.mesh_valid)

    def test_grid_size(self):
        self.assertEqual(self.msh.grid_size, 0.0)

    def test_time_step(self):
        self.assertEqual(self.msh.time_step, 0.0)
        self.assertEqual(self.msh.dt, 0.0)
        self.assertEqual(self.msh.over_dt, 0.0)

    def test_implicit_factor(self):
        self.assertEqual(self.msh.implicit_factor, 0.5)
        self.assertEqual(self.msh.alpha, 0.5)
        self.assertEqual(self.msh.one_minus_alpha, 0.5)

    def test_implicit_error_tolerance(self):
        self.assertEqual(self.msh.implicit_error_tolerance, 1.0e-3)
        self.assertEqual(self.msh.eps_s, 1.0e-3)

    def test_max_iterations(self):
        self.assertEqual(self.msh.max_iterations, 100)

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


class TestConsolidationAnalysis1DSetters(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D((100, -8))

    def test_z_min_max_setters(self):
        self.assertAlmostEqual(self.msh.z_min, -8.0)
        self.assertAlmostEqual(self.msh.z_max, 100.0)
        self.msh.z_min = -7
        self.assertAlmostEqual(self.msh.z_min, -7.0)
        self.assertIsInstance(self.msh.z_min, float)
        self.msh.z_max = 101
        self.assertAlmostEqual(self.msh.z_max, 101.0)
        self.assertIsInstance(self.msh.z_max, float)

    def test_grid_size_setter(self):
        self.assertEqual(self.msh.grid_size, 0.0)
        self.msh.grid_size = 1
        self.assertAlmostEqual(self.msh.grid_size, 1.0)
        self.assertIsInstance(self.msh.grid_size, float)

    def test_time_step_setter(self):
        self.assertAlmostEqual(self.msh.time_step, 0.0)
        self.assertAlmostEqual(self.msh.dt, 0.0)
        self.assertAlmostEqual(self.msh.over_dt, 0.0)
        self.msh.time_step = 0.1
        self.assertAlmostEqual(self.msh.time_step, 0.1)
        self.assertAlmostEqual(self.msh.dt, 0.1)
        self.assertAlmostEqual(self.msh.over_dt, 10.0)
        self.msh.time_step = 1.5
        self.assertAlmostEqual(self.msh.time_step, 1.5)
        self.assertAlmostEqual(self.msh.dt, 1.5)
        self.assertAlmostEqual(self.msh.over_dt, 1.0/1.5)

    def test_implicit_factor_setter(self):
        self.assertAlmostEqual(self.msh.implicit_factor, 0.5)
        self.assertAlmostEqual(self.msh.alpha, 0.5)
        self.assertAlmostEqual(self.msh.one_minus_alpha, 0.5)
        self.msh.implicit_factor = 0.1
        self.assertAlmostEqual(self.msh.implicit_factor, 0.1)
        self.assertAlmostEqual(self.msh.alpha, 0.1)
        self.assertAlmostEqual(self.msh.one_minus_alpha, 0.9)
        self.msh.implicit_factor = 0.85
        self.assertAlmostEqual(self.msh.implicit_factor, 0.85)
        self.assertAlmostEqual(self.msh.alpha, 0.85)
        self.assertAlmostEqual(self.msh.one_minus_alpha, 0.15)

    def test_implicit_error_tolerance_setter(self):
        self.assertAlmostEqual(self.msh.implicit_error_tolerance, 1.0e-3)
        self.assertAlmostEqual(self.msh.eps_s, 1.0e-3)
        self.msh.implicit_error_tolerance = 0.1
        self.assertAlmostEqual(self.msh.implicit_error_tolerance, 1.0e-1)
        self.assertAlmostEqual(self.msh.eps_s, 1.0e-1)
        self.msh.implicit_error_tolerance = 1.5e-4
        self.assertAlmostEqual(self.msh.implicit_error_tolerance, 0.00015)
        self.assertAlmostEqual(self.msh.eps_s, 1.5e-4)

    def test_max_iterations_setter(self):
        self.assertEqual(self.msh.max_iterations, 100)
        self.msh.max_iterations = 10
        self.assertEqual(self.msh.max_iterations, 10)
        self.msh.max_iterations = 500
        self.assertEqual(self.msh.max_iterations, 500)


class TestConsolidationAnalysis1DLinearNoArgs(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D(order=1)

    def test_create_analysis_no_args(self):
        self.assertFalse(self.msh.mesh_valid)
        self.assertEqual(self.msh.num_nodes, 0)
        self.assertEqual(self.msh.num_elements, 0)
        self.assertEqual(self.msh.num_boundaries, 0)
        self.assertTrue(np.isinf(self.msh.z_min))
        self.assertTrue(np.isinf(self.msh.z_max))
        self.assertTrue(self.msh.z_min < 0)
        self.assertTrue(self.msh.z_max > 0)
        self.assertEqual(self.msh.grid_size, 0.0)


class TestConsolidationAnalysis1DLinearMeshGen(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D(z_range=(100, -8))

    def test_z_range_generate(self):
        nel = 9
        nnod = 10
        self.msh.generate_mesh(nel, order=1)
        self.assertTrue(self.msh.mesh_valid)
        self.assertEqual(self.msh.num_nodes, nnod)
        self.assertEqual(self.msh.num_elements, nel)
        self.assertEqual(self.msh.num_boundaries, 0)
        self.assertAlmostEqual(self.msh.z_min, -8.0)
        self.assertAlmostEqual(self.msh.z_max, 100.0)
        self.assertEqual(self.msh.grid_size, 0.0)
        self.assertAlmostEqual(self.msh.nodes[1].z - self.msh.nodes[0].z, 12.0)
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_mass_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_1.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr

    def test_grid_size_generate(self):
        nel = 108
        nnod = 109
        self.msh.grid_size = 1.0
        self.msh.generate_mesh(order=1)
        self.assertAlmostEqual(self.msh.grid_size, 1.0)
        self.assertIsInstance(self.msh.grid_size, float)
        self.msh.generate_mesh(order=1)
        self.assertEqual(self.msh.num_nodes, nnod)
        self.assertEqual(self.msh.num_elements, nel)
        self.assertEqual(self.msh.num_boundaries, 0)
        self.assertAlmostEqual(self.msh.nodes[1].z - self.msh.nodes[0].z, 1.0)
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_mass_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_1.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr


class TestConsolidationAnalysis1DCubicMeshGen(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D(z_range=(100, -8))

    def test_z_range_generate(self):
        nel = 9
        nnod = nel * 3 + 1
        self.msh.generate_mesh(nel)
        self.assertTrue(self.msh.mesh_valid)
        self.assertEqual(self.msh.num_nodes, nnod)
        self.assertEqual(self.msh.num_elements, nel)
        self.assertEqual(self.msh.num_boundaries, 0)
        self.assertAlmostEqual(self.msh.z_min, -8.0)
        self.assertAlmostEqual(self.msh.z_max, 100.0)
        self.assertEqual(self.msh.grid_size, 0.0)
        self.assertAlmostEqual(self.msh.nodes[1].z - self.msh.nodes[0].z, 4.0)
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_mass_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_1.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr

    def test_grid_size_generate(self):
        nel = 108
        nnod = nel * 3 + 1
        self.msh.grid_size = 1.0
        self.msh.generate_mesh()
        self.assertEqual(self.msh.num_nodes, nnod)
        self.assertEqual(self.msh.num_elements, nel)
        self.assertEqual(self.msh.num_boundaries, 0)
        self.assertAlmostEqual(
            self.msh.nodes[1].z - self.msh.nodes[0].z, 1.0/3.0)
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_mass_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_1.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr


class TestAddBoundaries(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D((-8, 100), generate=True)

    def test_add_boundary_no_int_pt(self):
        bnd = ConsolidationBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd)
        self.assertEqual(self.msh.num_boundaries, 1)
        self.assertTrue(bnd in self.msh.boundaries)
        bnd1 = ConsolidationBoundary1D((self.msh.nodes[-1],))
        self.msh.add_boundary(bnd1)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(bnd1 in self.msh.boundaries)

    def test_add_boundary_with_int_pt(self):
        bnd = ConsolidationBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd)
        bnd1 = ConsolidationBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(bnd1)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(bnd1 in self.msh.boundaries)


class TestRemoveBoundaries(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D((-8, 100), generate=True)
        self.bnd0 = ConsolidationBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(self.bnd0)
        self.bnd1 = ConsolidationBoundary1D(
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


class TestUpdateBoundaries(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D((0, 100), generate=True)
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.6
                ip.void_ratio_0 = 0.9
        per = 365.0 * 86400.0
        om = 2.0 * np.pi / per
        t0 = (7.0/12.0) * per
        eavg = 0.5
        eamp = 0.1
        def f(t): return eavg + eamp * np.cos(om * (t - t0))
        self.f = f
        self.bnd0 = ConsolidationBoundary1D(
            (self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_function=f)
        self.msh.add_boundary(self.bnd0)
        self.water_flux = 0.08
        self.bnd1 = ConsolidationBoundary1D(
            (self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=self.water_flux,
        )
        self.msh.add_boundary(self.bnd1)

    def test_initial_void_ratio_water_flux_vector(self):
        for en, en0 in zip(self.msh._void_ratio_vector,
                           self.msh._void_ratio_vector_0):
            self.assertEqual(en, 0.0)
            self.assertEqual(en0, 0.0)
        for fx, fx0 in zip(self.msh._water_flux_vector,
                           self.msh._water_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_initial_porosity(self):
        expected_porosity = 0.6/1.6
        expected_Sw = 1.0
        expected_Si = 0.0
        for e in self.msh.elements:
            for ip in e.int_pts:
                self.assertAlmostEqual(ip.porosity, expected_porosity)
                self.assertAlmostEqual(ip.deg_sat_water, expected_Sw)
                self.assertAlmostEqual(ip.deg_sat_ice, expected_Si)

    def test_update_consolidation_boundaries(self):
        t = 6307200.0
        expected_void_ratio_0 = self.f(t)
        expected_void_ratio_1 = 0.425685517452261
        self.msh.update_boundary_conditions(t)
        self.assertAlmostEqual(
            self.msh.nodes[0].void_ratio, expected_void_ratio_0)
        self.assertAlmostEqual(
            self.msh.nodes[0].void_ratio, expected_void_ratio_1)
        self.assertAlmostEqual(
            self.msh._void_ratio_vector[0], expected_void_ratio_0)
        self.assertAlmostEqual(
            self.msh._void_ratio_vector[0], expected_void_ratio_1)
        for en in self.msh._void_ratio_vector[1:]:
            self.assertEqual(en, 0.0)
        for en0 in self.msh._void_ratio_vector_0:
            self.assertEqual(en0, 0.0)
        for fx, fx0 in zip(self.msh._water_flux_vector,
                           self.msh._water_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)
        t = 18921600.0
        expected_void_ratio_2 = self.f(t)
        expected_void_ratio_3 = 0.599452189536827
        self.msh.update_boundary_conditions(t)
        self.assertAlmostEqual(
            self.msh.nodes[0].void_ratio, expected_void_ratio_2)
        self.assertAlmostEqual(
            self.msh.nodes[0].void_ratio, expected_void_ratio_3)
        self.assertAlmostEqual(
            self.msh._void_ratio_vector[0], expected_void_ratio_2)
        self.assertAlmostEqual(
            self.msh._void_ratio_vector[0], expected_void_ratio_3)
        for en in self.msh._void_ratio_vector[1:]:
            self.assertEqual(en, 0.0)
        for en0 in self.msh._void_ratio_vector_0:
            self.assertEqual(en0, 0.0)
        for fx, fx0 in zip(self.msh._water_flux_vector,
                           self.msh._water_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_update_water_flux_vector(self):
        t = 6307200.0
        self.msh.update_boundary_conditions(t)
        self.msh.update_water_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._water_flux_vector,
                                          self.msh._water_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, self.water_flux)
            else:
                self.assertEqual(fx, 0.0)
        self.msh.update_boundary_conditions(t)
        self.msh.update_water_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._water_flux_vector,
                                          self.msh._water_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, self.water_flux)
            else:
                self.assertEqual(fx, 0.0)


class TestUpdateNodes(unittest.TestCase):
    def setUp(self):
        self.msh = ConsolidationAnalysis1D((0, 100), generate=True, order=1)
        self.msh._void_ratio_vector[:] = np.linspace(
            2.0, 22.0, self.msh.num_nodes)
        self.msh._void_ratio_vector_0[:] = np.linspace(
            1.0, 11.0, self.msh.num_nodes)
        self.msh.time_step = 0.25
        self.msh.update_nodes()

    def test_initial_node_values(self):
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k+1))

    def test_repeat_update_nodes(self):
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k+1))

    def test_change_void_ratio_vectors_update_nodes(self):
        self.msh._void_ratio_vector[:] = np.linspace(
            4.0, 44.0, self.msh.num_nodes)
        self.msh._void_ratio_vector_0[:] = np.linspace(
            2.0, 22.0, self.msh.num_nodes)
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k+1))
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 4.0 * (k+1))


class TestUpdateGlobalMatricesLinearConstant(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            (0, 100),
            generate=True,
            order=1,
        )
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.6
                ip.void_ratio_0 = 0.9
                sig_p, dsig_de = ip.material.eff_stress(0.6, 0.0)
                ip.eff_stress = sig_p
                ip.eff_stress_gradient = dsig_de
                k, dk_de = ip.material.hyd_cond(0.6, 1.0, False)
                ip.hyd_cond = k
                ip.hyd_cond_gradient = dk_de

    def test_initial_stiffness_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._stiffness_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._stiffness_matrix, expected))

    def test_initial_mass_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._mass_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._mass_matrix, expected))

    def test_update_stiffness_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        k00 = -1.55999984566148E-09
        k11 = -7.82923225956888E-10
        d0 = np.ones((self.msh.num_nodes,)) * (k00 + k11)
        d0[0] = k00
        d0[-1] = k11
        dp1 = -np.ones((self.msh.num_nodes - 1,)) * k00
        dm1 = -np.ones((self.msh.num_nodes - 1,)) * k11
        expected1 = np.diag(d0) + np.diag(dm1, -1) + np.diag(dp1, 1)
        self.msh.update_stiffness_matrix()
        self.assertTrue(np.allclose(
            self.msh._stiffness_matrix_0,
            expected0,
            atol=1e-20, rtol=1e-16,
        ))
        self.assertTrue(np.allclose(
            self.msh._stiffness_matrix,
            expected1,
            atol=1e-20, rtol=1e-16,
        ))

    def test_update_mass_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        m0 = 1.75438596491228
        d0 = np.ones((self.msh.num_nodes,)) * 2.0 * m0
        d0[0] = m0
        d0[-1] = m0
        d1 = np.ones((self.msh.num_nodes - 1,)) * m0 * 0.5
        expected1 = np.diag(d0) + np.diag(d1, -1) + np.diag(d1, 1)
        self.msh.update_mass_matrix()
        self.assertTrue(np.allclose(
            self.msh._mass_matrix_0,
            expected0,
            atol=1e-12, rtol=1e-10,
        ))
        self.assertTrue(np.allclose(
            self.msh._mass_matrix,
            expected1,
            atol=1e-12, rtol=1e-10,
        ))


class TestUpdateIntegrationPointsLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        self.msh._void_ratio_vector[:] = np.array([
            0.6,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        self.msh.update_nodes()
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio_0 = 0.9
        self.msh.update_integration_points()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()

    def test_void_ratio_distribution(self):
        expected_void_ratio_int_pts = np.array([
            0.589433756729741,
            0.560566243270259,
            0.541547005383793,
            0.518452994616208,
            0.503660254037844,
            0.486339745962156,
            0.475773502691896,
            0.464226497308104,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.036178444520940E-10,
            8.332723447117670E-11,
            7.218198340441230E-11,
            6.063323545379980E-11,
            5.422629776125640E-11,
            4.757966757424550E-11,
            4.393169733182270E-11,
            4.026418833655080E-11,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            7.822587016510380E-10,
            6.290755669959050E-10,
            5.449349474417820E-10,
            4.577481445767780E-10,
            4.093792290928700E-10,
            3.592007648723600E-10,
            3.316605619219050E-10,
            3.039728519516280E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            -8.123441622973750E-11,
            -6.330331621462550E-11,
            -5.769218208232310E-11,
            -4.722093512946930E-11,
            -4.545873593535430E-11,
            -3.927183415246080E-11,
            -3.978293540543520E-11,
            -3.627677855006020E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-19, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.670512849594790E+05,
            1.956224690989220E+05,
            2.170676343218650E+05,
            2.462919434227970E+05,
            2.670467867470950E+05,
            2.935815141421210E+05,
            3.110474684266550E+05,
            3.313250233604040E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -9.136574786536750E+05,
            -1.069922520669520E+06,
            -1.187213061665100E+06,
            -1.347050255225330E+06,
            -1.460565202602900E+06,
            -1.605692204375930E+06,
            -1.701219154424590E+06,
            -1.812123657305390E+06,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.670512849594790E+05,
            1.956224690989220E+05,
            2.170676343218650E+05,
            2.462919434227970E+05,
            2.670467867470950E+05,
            2.935815141421210E+05,
            3.110474684266550E+05,
            3.313250233604040E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [-7.99028679973533E-10, 7.99028679973533E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.42998283395020E-10, -8.17080731014650E-10, 6.74082447619630E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.95455548875149E-10,
             -7.92450395857143E-10, 5.96994846981994E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.22272641090901E-10, -7.72457873113186E-10,
             5.50185232022285E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.35477581693084E-10, -2.35477581693084E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))


class TestInitializeGlobalSystemLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_void_ratio_vector = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        for nd, e0 in zip(self.msh.nodes,
                          initial_void_ratio_vector,
                          ):
            nd.void_ratio = e0
            nd.void_ratio_0 = 0.9
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.6,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=-2.0e-11,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5)

    def test_free_indices(self):
        expected_free_vec = [i for i in range(self.msh.num_nodes)][1:]
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array([
            0.6,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.589433756729741,
            0.560566243270259,
            0.541547005383793,
            0.518452994616208,
            0.503660254037844,
            0.486339745962156,
            0.475773502691896,
            0.464226497308104,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.036178444520940E-10,
            8.332723447117670E-11,
            7.218198340441230E-11,
            6.063323545379980E-11,
            5.422629776125640E-11,
            4.757966757424550E-11,
            4.393169733182270E-11,
            4.026418833655080E-11,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            7.822587016510380E-10,
            6.290755669959050E-10,
            5.449349474417820E-10,
            4.577481445767780E-10,
            4.093792290928700E-10,
            3.592007648723600E-10,
            3.316605619219050E-10,
            3.039728519516280E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.670512849594790E+05,
            1.956224690989220E+05,
            2.170676343218650E+05,
            2.462919434227970E+05,
            2.670467867470950E+05,
            2.935815141421210E+05,
            3.110474684266550E+05,
            3.313250233604040E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -9.136574786536750E+05,
            -1.069922520669520E+06,
            -1.187213061665100E+06,
            -1.347050255225330E+06,
            -1.460565202602900E+06,
            -1.605692204375930E+06,
            -1.701219154424590E+06,
            -1.812123657305390E+06,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.670512849594790E+05,
            1.956224690989220E+05,
            2.170676343218650E+05,
            2.462919434227970E+05,
            2.670467867470950E+05,
            2.935815141421210E+05,
            3.110474684266550E+05,
            3.313250233604040E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            -8.123441622973750E-11,
            -6.330331621462550E-11,
            -5.769218208232310E-11,
            -4.722093512946930E-11,
            -4.545873593535430E-11,
            -3.927183415246080E-11,
            -3.978293540543520E-11,
            -3.627677855006020E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [-7.99028679973533E-10, 7.99028679973533E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.42998283395020E-10, -8.17080731014650E-10, 6.74082447619630E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.95455548875149E-10,
             -7.92450395857143E-10, 5.96994846981994E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.22272641090901E-10, -7.72457873113186E-10,
             5.50185232022285E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.35477581693084E-10, -2.35477581693084E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))


class TestInitializeTimeStepLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_void_ratio_vector = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        for nd, e0 in zip(self.msh.nodes,
                          initial_void_ratio_vector,
                          ):
            nd.void_ratio = e0
            nd.void_ratio_0 = 0.9
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.6,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=-2.0e-11,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh.update_weighted_matrices()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.501)

    def test_iteration_variables(self):
        self.assertEqual(self.msh._eps_a, 1.0)
        self.assertEqual(self.msh._iter, 0)

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array([
            0.6,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.589433756729741,
            0.560566243270259,
            0.541547005383793,
            0.518452994616208,
            0.503660254037844,
            0.486339745962156,
            0.475773502691896,
            0.464226497308104,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.036178444520940E-10,
            8.332723447117670E-11,
            7.218198340441230E-11,
            6.063323545379980E-11,
            5.422629776125640E-11,
            4.757966757424550E-11,
            4.393169733182270E-11,
            4.026418833655080E-11,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            7.822587016510380E-10,
            6.290755669959050E-10,
            5.449349474417820E-10,
            4.577481445767780E-10,
            4.093792290928700E-10,
            3.592007648723600E-10,
            3.316605619219050E-10,
            3.039728519516280E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.670512849594790E+05,
            1.956224690989220E+05,
            2.170676343218650E+05,
            2.462919434227970E+05,
            2.670467867470950E+05,
            2.935815141421210E+05,
            3.110474684266550E+05,
            3.313250233604040E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -9.136574786536750E+05,
            -1.069922520669520E+06,
            -1.187213061665100E+06,
            -1.347050255225330E+06,
            -1.460565202602900E+06,
            -1.605692204375930E+06,
            -1.701219154424590E+06,
            -1.812123657305390E+06,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.670512849594790E+05,
            1.956224690989220E+05,
            2.170676343218650E+05,
            2.462919434227970E+05,
            2.670467867470950E+05,
            2.935815141421210E+05,
            3.110474684266550E+05,
            3.313250233604040E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            -8.123441622973750E-11,
            -6.330331621462550E-11,
            -5.769218208232310E-11,
            -4.722093512946930E-11,
            -4.545873593535430E-11,
            -3.927183415246080E-11,
            -3.978293540543520E-11,
            -3.627677855006020E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [-7.99028679973533E-10, 7.99028679973533E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.42998283395020E-10, -8.17080731014650E-10, 6.74082447619630E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.95455548875149E-10,
             -7.92450395857143E-10, 5.96994846981994E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.22272641090901E-10, -7.72457873113186E-10,
             5.50185232022285E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.35477581693084E-10, -2.35477581693084E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    atol=1e-18, rtol=1e-8))


class TestUpdateWeightedMatricesLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_void_ratio_vector = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        for nd, e0 in zip(self.msh.nodes,
                          initial_void_ratio_vector,
                          ):
            nd.void_ratio = e0
            nd.void_ratio_0 = 0.9
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.6,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=-2.0e-11,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh._void_ratio_vector[:] = np.array([
            0.6,
            0.51,
            0.44,
            0.39,
            0.35,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.6,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        expected_void_ratio_vector = np.array([
            0.6,
            0.51,
            0.44,
            0.39,
            0.35,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.580980762113533,
            0.529019237886467,
            0.495207259421637,
            0.454792740578363,
            0.429433756729741,
            0.400566243270259,
            0.381547005383793,
            0.358452994616207,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            9.721198630328660E-11,
            6.566805659823700E-11,
            5.087392179522110E-11,
            3.749631585663170E-11,
            3.096309673105840E-11,
            2.489985421821330E-11,
            2.156942895510420E-11,
            1.811843070467490E-11,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            7.338979361386530E-10,
            4.957583219966880E-10,
            3.840706031076100E-10,
            2.830769112579550E-10,
            2.337546392323530E-10,
            1.879804365265010E-10,
            1.628375264800560E-10,
            1.367843555705900E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.749557388226390E+05,
            2.324621448835880E+05,
            2.796827805717430E+05,
            3.488688635308960E+05,
            4.007719581555580E+05,
            4.693169526892120E+05,
            5.207659484958170E+05,
            5.908778520765940E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -9.568894920350800E+05,
            -1.271410616376180E+06,
            -1.529675537557290E+06,
            -1.908076590441840E+06,
            -2.191951393204330E+06,
            -2.566846126250770E+06,
            -2.848237315784740E+06,
            -3.231701980930900E+06,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.749557388226390E+05,
            2.324621448835880E+05,
            2.796827805717430E+05,
            3.488688635308960E+05,
            4.007719581555580E+05,
            4.693169526892120E+05,
            5.207659484958170E+05,
            5.908778520765940E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            -5.735706086792910E-11,
            -3.064378543793070E-11,
            -2.621434129972450E-11,
            -1.456875920360860E-11,
            -1.626587892219620E-11,
            -1.076851142225290E-11,
            -1.119988732484370E-11,
            -7.982985487033130E-12,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.array([
            [-7.99028679973533E-10, 7.99028679973533E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.42998283395020E-10, -8.17080731014650E-10, 6.74082447619630E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.95455548875149E-10,
             -7.92450395857143E-10, 5.96994846981994E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.22272641090901E-10, -7.72457873113186E-10,
             5.50185232022285E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.35477581693084E-10, -2.35477581693084E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [-7.57840810676873E-10, 7.57840810676873E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.80675969356063E-10, -7.50396717669538E-10, 5.69720748313476E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.40938797717057E-10,
             -7.15788190033667E-10, 4.74849392316610E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.58929683094287E-10, -6.80744426454293E-10,
             4.21814743360006E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.63864017804318E-10, -2.63864017804318E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_weighted(self):
        expected_K = np.array([
            [-7.78434745325203E-10, 7.78434745325203E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.61837126375541E-10, -7.83738724342094E-10, 6.21901597966553E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.18197173296103E-10,
             -7.54119292945405E-10, 5.35922119649302E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.40601162092594E-10, -7.26601149783739E-10,
             4.85999987691145E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.49670799748701E-10, -2.49670799748701E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._weighted_stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_mass_matrix_weighted(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._weighted_mass_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [4.38596491228031E+03, 2.19298245614074E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614043E+03, 8.77192982456101E+03, 2.19298245614066E+03,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614046E+03, 8.77192982456103E+03,
                2.19298245614062E+03, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614047E+03,
                8.77192982456104E+03, 2.19298245614059E+03],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614048E+03, 4.38596491228058E+03],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            atol=1e-11, rtol=1e-16,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [4.38596491228109E+03, 2.19298245613996E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614027E+03, 8.77192982456180E+03, 2.19298245614004E+03,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614024E+03, 8.77192982456178E+03,
                2.19298245614008E+03, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614023E+03,
                8.77192982456177E+03, 2.19298245614011E+03],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614023E+03, 4.38596491228083E+03],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            atol=1e-11, rtol=1e-16,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    atol=1e-18, rtol=1e-8))


class TestVoidRatioCorrectionLinearOneStep(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_void_ratio_vector = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        for nd, e0 in zip(self.msh.nodes,
                          initial_void_ratio_vector,
                          ):
            nd.void_ratio = e0
            nd.void_ratio_0 = 0.9
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.6,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=-2.0e-11,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh._void_ratio_vector[:] = np.array([
            0.6,
            0.51,
            0.44,
            0.39,
            0.35,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.6,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        expected_void_ratio_vector = np.array([
            0.600000000000000,
            0.549999999999998,
            0.510000000000000,
            0.479999999999998,
            0.460000000000007,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.589433756729740,
            0.560566243270258,
            0.541547005383791,
            0.518452994616207,
            0.503660254037844,
            0.486339745962154,
            0.475773502691896,
            0.464226497308109,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.036178444520930E-10,
            8.332723447117580E-11,
            7.218198340441160E-11,
            6.063323545379960E-11,
            5.422629776125630E-11,
            4.757966757424490E-11,
            4.393169733182260E-11,
            4.026418833655240E-11,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            7.822587016510350E-10,
            6.290755669958980E-10,
            5.449349474417760E-10,
            4.577481445767770E-10,
            4.093792290928690E-10,
            3.592007648723560E-10,
            3.316605619219050E-10,
            3.039728519516400E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.371720913096690E+05,
            9.375936845079700E+04,
            7.369175578056780E+04,
            5.583532328286030E+04,
            4.732232895587810E+04,
            3.974636884423720E+04,
            3.457891505553490E+04,
            2.814031327404820E+04,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -3.948130157805790E+06,
            -2.698611551541770E+06,
            -2.121019229211170E+06,
            -1.607069788170220E+06,
            -1.362046115244570E+06,
            -1.143992455017300E+06,
            -9.952611792347770E+05,
            -8.099433232125740E+05,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.749557388226390E+05,
            2.324621448835880E+05,
            2.796827805717430E+05,
            3.488688635308960E+05,
            4.007719581555580E+05,
            4.693169526892120E+05,
            5.207659484958170E+05,
            5.908778520765940E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            -4.605983696099940E-12,
            -2.961668977150410E-11,
            -4.414236745724510E-11,
            -4.400342486575110E-11,
            -4.628448188839970E-11,
            -4.270685035950430E-11,
            -4.303914014310940E-11,
            -4.054681068966590E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.array([
            [-7.99028679973533E-10, 7.99028679973533E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.42998283395020E-10, -8.17080731014650E-10, 6.74082447619630E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.95455548875149E-10,
             -7.92450395857143E-10, 5.96994846981994E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.22272641090901E-10, -7.72457873113186E-10,
             5.50185232022285E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.35477581693084E-10, -2.35477581693084E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [-1.90217926974104E-09, 1.90217926974104E-09, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.24614887316253E-09, -2.12982288189213E-09, 8.83674008729601E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 4.05047109985122E-10,
             -9.31029254298931E-10, 5.25982144313809E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                1.51259938422719E-10, -5.13289248513135E-10,
             3.62029310090416E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                4.73216597612098E-11, -4.73216597612098E-11],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_weighted(self):
        expected_K = np.array([
            [-7.78434745325203E-10, 7.78434745325203E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.61837126375541E-10, -7.83738724342094E-10, 6.21901597966553E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.18197173296103E-10,
             -7.54119292945405E-10, 5.35922119649302E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.40601162092594E-10, -7.26601149783739E-10,
             4.85999987691145E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.49670799748701E-10, -2.49670799748701E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._weighted_stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_mass_matrix_weighted(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._weighted_mass_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [4.38596491228031E+03, 2.19298245614074E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614043E+03, 8.77192982456101E+03, 2.19298245614066E+03,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614046E+03, 8.77192982456103E+03,
                2.19298245614062E+03, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614047E+03,
                8.77192982456104E+03, 2.19298245614059E+03],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614048E+03, 4.38596491228058E+03],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            atol=1e-11, rtol=1e-16,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [4.38596491228109E+03, 2.19298245613996E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614027E+03, 8.77192982456180E+03, 2.19298245614004E+03,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614024E+03, 8.77192982456178E+03,
                2.19298245614008E+03, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614023E+03,
                8.77192982456177E+03, 2.19298245614011E+03],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614023E+03, 4.38596491228083E+03],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            atol=1e-11, rtol=1e-16,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    atol=1e-18, rtol=1e-8))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            8.77192982455595E+01,
            5.04385964912261E+02,
            8.99122807017535E+02,
            1.18421052631578E+03,
            6.79824561403537E+02,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000000,
            0.03999999999999840,
            0.07000000000000010,
            0.08999999999999800,
            0.11000000000000700,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.39879137964099E-01
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
        self.assertEqual(self.msh._iter, 1)


class TestIterativeVoidRatioCorrectionLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_void_ratio_vector = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        for nd, e0 in zip(self.msh.nodes,
                          initial_void_ratio_vector,
                          ):
            nd.void_ratio = e0
            nd.void_ratio_0 = 0.9
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.6,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=-2.0e-11,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh._void_ratio_vector[:] = np.array([
            0.6,
            0.51,
            0.44,
            0.39,
            0.35,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()
        self.msh.iterative_correction_step()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.6,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        expected_void_ratio_vector = np.array([
            0.600000000000000,
            0.549999999999998,
            0.510000000000000,
            0.479999999999998,
            0.460000000000006,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes,
                                    atol=1e-13, rtol=1e-20))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector,
                                    atol=1e-13, rtol=1e-20))
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0,
                                    atol=1e-13, rtol=1e-20))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.589433756729740,
            0.560566243270258,
            0.541547005383791,
            0.518452994616207,
            0.503660254037844,
            0.486339745962154,
            0.475773502691896,
            0.464226497308108,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.036178444520930E-10,
            8.332723447117580E-11,
            7.218198340441160E-11,
            6.063323545379960E-11,
            5.422629776125630E-11,
            4.757966757424490E-11,
            4.393169733182260E-11,
            4.026418833655210E-11,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            7.822587016510350E-10,
            6.290755669958980E-10,
            5.449349474417760E-10,
            4.577481445767770E-10,
            4.093792290928690E-10,
            3.592007648723560E-10,
            3.316605619219050E-10,
            3.039728519516380E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.371720913096690E+05,
            9.375936845079700E+04,
            7.369175578056780E+04,
            5.583532328286050E+04,
            4.732232895587820E+04,
            3.974636884423710E+04,
            3.457891505553490E+04,
            2.814031327404880E+04,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -3.948130157805790E+06,
            -2.698611551541770E+06,
            -2.121019229211170E+06,
            -1.607069788170220E+06,
            -1.362046115244570E+06,
            -1.143992455017290E+06,
            -9.952611792347750E+05,
            -8.099433232125910E+05,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.749557388226390E+05,
            2.324621448835880E+05,
            2.796827805717430E+05,
            3.488688635308960E+05,
            4.007719581555580E+05,
            4.693169526892120E+05,
            5.207659484958170E+05,
            5.908778520765940E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            -4.605983696099940E-12,
            -2.961668977150410E-11,
            -4.414236745724500E-11,
            -4.400342486575100E-11,
            -4.628448188839980E-11,
            -4.270685035950440E-11,
            -4.303914014310920E-11,
            -4.054681068966530E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.array([
            [-7.99028679973533E-10, 7.99028679973533E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.42998283395020E-10, -8.17080731014650E-10, 6.74082447619630E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.95455548875149E-10,
             -7.92450395857143E-10, 5.96994846981994E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                2.22272641090901E-10, -7.72457873113186E-10,
             5.50185232022285E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.35477581693084E-10, -2.35477581693084E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [-1.90217926974104E-09, 1.90217926974104E-09, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.24614887316253E-09, -2.12982288189213E-09, 8.83674008729601E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 4.05047109985122E-10,
             -9.31029254298930E-10, 5.25982144313809E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                1.51259938422718E-10, -5.13289248513135E-10,
             3.62029310090417E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                4.73216597612115E-11, -4.73216597612115E-11],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_weighted(self):
        expected_K = np.array([
            [-1.35060397485729E-09, 1.35060397485729E-09, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [6.94573578278777E-10, -1.47345180645339E-09, 7.78878228174615E-10,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.00251329430135E-10,
             -8.61739825078037E-10, 5.61488495647901E-10,
             0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00,
                1.86766289756810E-10, -6.42873560813160E-10,
             4.56107271056350E-10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.41399620727147E-10, -1.41399620727147E-10],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._weighted_stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_mass_matrix_weighted(self):
        expected_M = np.array([
            [4.38596491228070E+00, 2.19298245614036E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614035E+00, 8.77192982456140E+00, 2.19298245614036E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614035E+00, 8.77192982456140E+00,
                2.19298245614036E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614035E+00,
                8.77192982456140E+00, 2.19298245614036E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614035E+00, 4.38596491228070E+00],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._weighted_mass_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [4.38596491228003E+03, 2.19298245614103E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614070E+03, 8.77192982456067E+03, 2.19298245614074E+03,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614050E+03, 8.77192982456097E+03,
                2.19298245614063E+03, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614044E+03,
                8.77192982456108E+03, 2.19298245614058E+03],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614042E+03, 4.38596491228063E+03],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            atol=1e-11, rtol=1e-16,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [4.38596491228138E+03, 2.19298245613968E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [2.19298245614000E+03, 8.77192982456214E+03, 2.19298245613996E+03,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 2.19298245614020E+03, 8.77192982456184E+03,
                2.19298245614007E+03, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.19298245614026E+03,
                8.77192982456173E+03, 2.19298245614012E+03],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                2.19298245614028E+03, 4.38596491228077E+03],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            atol=1e-11, rtol=1e-16,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    atol=1e-18, rtol=1e-8))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -6.45741238258779E-11,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            -2.73736754432321E-12,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-10, rtol=1e-8,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000E+00,
            1.28684494867066E-17,
            -5.14737979468399E-17,
            1.93026742300689E-16,
            -7.20633171256021E-16,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            atol=1e-14, rtol=1e-8,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 6.40255563958163E-16
        self.assertEqual(self.msh._iter, 2)
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a,
                               delta=1e-14)


class TestDeformedCoordsLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_void_ratio_vector = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        for nd, e0 in zip(self.msh.nodes,
                          initial_void_ratio_vector,
                          ):
            nd.void_ratio = e0
            nd.void_ratio_0 = 0.9
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.6,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=-2.0e-11,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh._void_ratio_vector[:] = np.array([
            0.6,
            0.51,
            0.44,
            0.39,
            0.35,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()
        self.msh.iterative_correction_step()

    def test_calculate_settlement(self):
        expected = 20.1315789473684
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            20.1315789473684,
            40.8552631578947,
            60.9868421052631,
            80.6578947368421,
            100.0000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))


class TestUpdateIntegrationPointsCubic(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        initial_void_ratio_nodes = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        initial_void_ratio_0_nodes = np.array([
            0.802254248593737,
            0.679191704032817,
            0.584150070553881,
            0.530587476655649,
            0.515981351889433,
            0.528766176598625,
            0.554870977120579,
            0.582329942396731,
            0.603536597295654,
            0.615488027450132,
            0.618747094408366,
            0.615894414423534,
            0.610069646102427,
        ])
        for nd, e0, e00 in zip(self.msh.nodes,
                               initial_void_ratio_nodes,
                               initial_void_ratio_0_nodes,
                               ):
            nd.void_ratio = e0
            nd.void_ratio_0 = e00
        initial_void_ratio_int_pts = np.array([
            [0.783745715060955,
             0.714698836928105,
             0.627327140376931,
             0.562625830645097,
             0.535113117201931,],
            [0.526348555862977,
             0.516904981852187,
             0.519829331413518,
             0.536033293625736,
             0.550919150923390,],
            [0.559000789171861,
             0.574407230147396,
             0.593902490791297,
             0.608339304393577,
             0.614478220550266,],
            [0.616433858525457,
             0.618541004706861,
             0.617888494120909,
             0.614272901437460,
             0.610951417714685,],
        ])
        initial_ppc_int_pts = np.array([
            [5.77166646474166E+04,
             8.41992188496132E+04,
             1.35781869341622E+05,
             1.93431234985906E+05,
             2.24842014584614E+05,],
            [2.35882598526983E+05,
             2.48386046851806E+05,
             2.44444909206464E+05,
             2.23713285346802E+05,
             2.06221264170372E+05,],
            [1.97304574489730E+05,
             1.81360323911413E+05,
             1.63017883528454E+05,
             1.50641118488766E+05,
             1.45667203576232E+05,],
            [1.44117447679089E+05,
             1.42466076597825E+05,
             1.42975416619767E+05,
             1.45830873403700E+05,
             1.48504285599951E+05,],
        ])
        initial_void_ratio_int_pts_deform = np.reshape([
            0.774581669776811,
            0.703315641710972,
            0.656319129770595,
            0.601230982260878,
            0.568921538039961,
            0.537781325197891,
            0.524519137940444,
            0.516311951220182,
            0.516873983023993,
            0.524480968356718,
            0.533646230967572,
            0.548943507921394,
            0.561045564063720,
            0.576947164319193,
            0.587433503554867,
            0.599725330730990,
            0.606934542194310,
            0.613882862872818,
            0.616842056839490,
            0.618673327726750,
            0.618547921480358,
            0.616850573624820,
            0.614805527110658,
            0.611392243254117,
        ], (self.msh.num_elements, self.msh.elements[0].order, 2))
        for e, e0s, ppc0s in zip(self.msh.elements,
                                 initial_void_ratio_int_pts,
                                 initial_ppc_int_pts):
            for ip, e0, ppc in zip(e.int_pts, e0s, ppc0s):
                ip.material = self.mtl
                ip.void_ratio_0 = e0
                ip.pre_consol_stress = ppc
        for e, ee00 in zip(self.msh.elements,
                           initial_void_ratio_int_pts_deform):
            for iipp, e0s in zip(e._int_pts_deformed, ee00):
                for ip, e0 in zip(iipp, e0s):
                    ip.void_ratio_0 = e0
        self.msh.update_integration_points()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()

    def test_void_ratio_distribution(self):
        expected_void_ratio_int_pts = np.array([
            0.564605405564335,
            0.485143306872301,
            0.420257096108955,
            0.405867716936220,
            0.418920304310449,
            0.431085818448524,
            0.460255726081957,
            0.509407217025838,
            0.557357527450674,
            0.584326765554389,
            0.595666299923334,
            0.613396690730221,
            0.627874761329268,
            0.631085465808520,
            0.628173039897355,
            0.625458546779438,
            0.618949860617677,
            0.607982680571832,
            0.597283520127575,
            0.591265869710410,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            8.59073106253587E-11,
            4.71518408125654E-11,
            2.88906276938521E-11,
            2.59166393545608E-11,
            2.86005281937074E-11,
            3.13516920990812E-11,
            3.90750935789392E-11,
            5.66307686156151E-11,
            8.13329619654793E-11,
            9.96988893121850E-11,
            1.08609826849242E-10,
            1.24165673963490E-10,
            1.38506652814146E-10,
            1.41904946425873E-10,
            1.38818899333579E-10,
            1.36003040999632E-10,
            1.29481781049291E-10,
            1.19192985299232E-10,
            1.09943987129997E-10,
            1.05060988160446E-10,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            6.48553746967738E-10,
            3.55970904138496E-10,
            2.18108618541327E-10,
            1.95656614551851E-10,
            2.15918524165858E-10,
            2.36688324155684E-10,
            2.94995829449878E-10,
            4.27531683997083E-10,
            6.14019887838572E-10,
            7.52672709240330E-10,
            8.19945469690917E-10,
            9.37383704688169E-10,
            1.04565034114871E-09,
            1.07130562053225E-09,
            1.04800763354538E-09,
            1.02674942559873E-09,
            9.77517438880057E-10,
            8.99842593899900E-10,
            8.30017658458556E-10,
            7.93153656372024E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            1.28755597302353E-10,
            7.46148172878518E-11,
            1.09529229271368E-11,
            -3.69776152422568E-11,
            -6.53485057810324E-11,
            -7.70845503795531E-11,
            -9.56990281646424E-11,
            -1.21388361879674E-10,
            -2.70095500129423E-10,
            -2.15277807907599E-10,
            -2.07812563896064E-10,
            -1.88621892637009E-10,
            -1.66583868356442E-10,
            -1.31873386089425E-10,
            -9.64147261532523E-11,
            -7.80065395723034E-11,
            -4.56162882137969E-11,
            -1.01806828600777E-10,
            -9.56538955097771E-11,
            -9.45298205420664E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-19, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.91348263991600E+05,
            2.95508926399538E+05,
            4.21400086245929E+05,
            4.55904278660068E+05,
            4.24492378919279E+05,
            3.97167030147011E+05,
            3.38599254744269E+05,
            2.58783518415400E+05,
            1.21098524857941E+05,
            7.88384835245203E+04,
            6.86780997290420E+04,
            5.90437626148318E+04,
            6.13171369891120E+04,
            7.82738172392405E+04,
            9.82147105720476E+04,
            1.11149614042132E+05,
            1.40799387466950E+05,
            1.50935229366213E+05,
            1.60031067344958E+05,
            1.65385719998053E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -1.0465455112526E+06,
            -1.6162338450695E+06,
            -2.3047732940054E+06,
            -2.4934878761874E+06,
            -2.3216860422544E+06,
            -2.1722348765920E+06,
            -1.8519087802210E+06,
            -1.4153707169023E+06,
            -3.4854957265183E+06,
            -2.2691539614727E+06,
            -1.9767146081406E+06,
            -1.6994160953899E+06,
            -1.7648490697025E+06,
            -2.2529015593352E+06,
            -2.8268466059490E+06,
            -3.1991430548182E+06,
            -4.0525321335512E+06,
            -8.2551356091753E+05,
            -8.7526163915541E+05,
            -9.0454796546700E+05,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.91348263991600E+05,
            2.95508926399538E+05,
            4.21400086245929E+05,
            4.55904278660068E+05,
            4.24492378919279E+05,
            3.97167030147011E+05,
            3.38599254744269E+05,
            2.58783518415400E+05,
            2.23713285346800E+05,
            2.06221264170372E+05,
            1.97304574489731E+05,
            1.81360323911411E+05,
            1.63017883528454E+05,
            1.50641118488764E+05,
            1.45667203576232E+05,
            1.44117447679089E+05,
            1.42466076597824E+05,
            1.50935229366213E+05,
            1.60031067344958E+05,
            1.65385719998053E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_calculate_settlement(self):
        expected = 2.849717515737840
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.84971751573783,
            10.10193918312510,
            17.38583989564510,
            24.95575711969480,
            32.90437075408170,
            41.17532936190790,
            49.64556242438820,
            58.18107658063250,
            66.69005362206340,
            75.12468060779240,
            83.47575953432560,
            91.75911912707220,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))


class TestInitializeGlobalSystemCubic(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        initial_void_ratio_nodes = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        initial_void_ratio_0_nodes = np.array([
            0.802254248593737,
            0.679191704032817,
            0.584150070553881,
            0.530587476655649,
            0.515981351889433,
            0.528766176598625,
            0.554870977120579,
            0.582329942396731,
            0.603536597295654,
            0.615488027450132,
            0.618747094408366,
            0.615894414423534,
            0.610069646102427,
        ])
        for nd, e0, e00 in zip(self.msh.nodes,
                               initial_void_ratio_nodes,
                               initial_void_ratio_0_nodes,
                               ):
            nd.void_ratio = e0
            nd.void_ratio_0 = e00
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5)

    def test_free_indices(self):
        expected_free_vec = [i for i in range(self.msh.num_nodes)][1:-1]
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.564605405564335,
            0.485143306872301,
            0.420257096108955,
            0.405867716936220,
            0.418920304310449,
            0.431085818448524,
            0.460255726081957,
            0.509407217025838,
            0.557357527450674,
            0.584326765554389,
            0.595666299923334,
            0.613396690730221,
            0.627874761329268,
            0.631085465808520,
            0.628173039897355,
            0.625458546779438,
            0.618949860617677,
            0.607982680571832,
            0.597283520127575,
            0.591265869710410,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            8.59073106253591E-11,
            4.71518408125651E-11,
            2.88906276938520E-11,
            2.59166393545604E-11,
            2.86005281937073E-11,
            3.13516920990812E-11,
            3.90750935789386E-11,
            5.66307686156151E-11,
            8.13329619654780E-11,
            9.96988893121846E-11,
            1.08609826849242E-10,
            1.24165673963488E-10,
            1.38506652814146E-10,
            1.41904946425871E-10,
            1.38818899333578E-10,
            1.36003040999632E-10,
            1.29481781049290E-10,
            1.19192985299232E-10,
            1.09943987129995E-10,
            1.05060988160446E-10,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            6.48553746967741E-10,
            3.55970904138494E-10,
            2.18108618541326E-10,
            1.95656614551848E-10,
            2.15918524165857E-10,
            2.36688324155684E-10,
            2.94995829449874E-10,
            4.27531683997083E-10,
            6.14019887838562E-10,
            7.52672709240327E-10,
            8.19945469690917E-10,
            9.37383704688155E-10,
            1.04565034114871E-09,
            1.07130562053223E-09,
            1.04800763354537E-09,
            1.02674942559873E-09,
            9.77517438880047E-10,
            8.99842593899902E-10,
            8.30017658458543E-10,
            7.93153656372022E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.91348263991599E+05,
            2.95508926399539E+05,
            4.21400086245930E+05,
            4.55904278660072E+05,
            4.24492378919280E+05,
            3.97167030147011E+05,
            3.38599254744272E+05,
            2.58783518415400E+05,
            1.21098524857946E+05,
            7.88384835245214E+04,
            6.86780997290425E+04,
            5.90437626148323E+04,
            6.13171369891121E+04,
            7.82738172392415E+04,
            9.82147105720494E+04,
            1.11149614042132E+05,
            1.40799387466949E+05,
            1.50935229366213E+05,
            1.60031067344960E+05,
            1.65385719998054E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -1.0465455112526E+06,
            -1.6162338450695E+06,
            -2.3047732940054E+06,
            -2.4934878761874E+06,
            -2.3216860422544E+06,
            -2.1722348765920E+06,
            -1.8519087802210E+06,
            -1.4153707169023E+06,
            -3.4854957265184E+06,
            -2.2691539614727E+06,
            -1.9767146081406E+06,
            -1.6994160953899E+06,
            -1.7648490697025E+06,
            -2.2529015593352E+06,
            -2.8268466059491E+06,
            -3.1991430548182E+06,
            -4.0525321335511E+06,
            -8.2551356091753E+05,
            -8.7526163915542E+05,
            -9.0454796546700E+05,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.91348263991599E+05,
            2.95508926399539E+05,
            4.21400086245930E+05,
            4.55904278660072E+05,
            4.24492378919280E+05,
            3.97167030147011E+05,
            3.38599254744272E+05,
            2.58783518415400E+05,
            2.23713285346802E+05,
            2.06221264170372E+05,
            1.97304574489730E+05,
            1.81360323911413E+05,
            1.63017883528454E+05,
            1.50641118488766E+05,
            1.45667203576232E+05,
            1.44117447679089E+05,
            1.42466076597825E+05,
            1.50935229366213E+05,
            1.60031067344960E+05,
            1.65385719998054E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            1.28755597302346E-10,
            7.46148172878523E-11,
            1.09529229271374E-11,
            -3.69776152422559E-11,
            -6.53485057810297E-11,
            -7.70845503795567E-11,
            -9.56990281646412E-11,
            -1.21388361879674E-10,
            -2.70095500129424E-10,
            -2.15277807907589E-10,
            -2.07812563896076E-10,
            -1.88621892637006E-10,
            -1.66583868356443E-10,
            -1.31873386089419E-10,
            -9.64147261532209E-11,
            -7.80065395723413E-11,
            -4.56162882137967E-11,
            -1.01806828600777E-10,
            -9.56538955097747E-11,
            -9.45298205420616E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_calculate_settlement(self):
        expected = 2.849717515737840
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.84971751573783,
            10.10193918312510,
            17.38583989564510,
            24.95575711969480,
            32.90437075408170,
            41.17532936190790,
            49.64556242438820,
            58.18107658063250,
            66.69005362206340,
            75.12468060779240,
            83.47575953432560,
            91.75911912707220,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    ))


class TestInitializeTimeStepCubic(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        initial_void_ratio_nodes = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        initial_void_ratio_0_nodes = np.array([
            0.802254248593737,
            0.679191704032817,
            0.584150070553881,
            0.530587476655649,
            0.515981351889433,
            0.528766176598625,
            0.554870977120579,
            0.582329942396731,
            0.603536597295654,
            0.615488027450132,
            0.618747094408366,
            0.615894414423534,
            0.610069646102427,
        ])
        for nd, e0, e00 in zip(self.msh.nodes,
                               initial_void_ratio_nodes,
                               initial_void_ratio_0_nodes,
                               ):
            nd.void_ratio = e0
            nd.void_ratio_0 = e00
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e+9
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5 + 1e+9)

    def test_iteration_variables(self):
        self.assertEqual(self.msh._eps_a, 1.0)
        self.assertEqual(self.msh._iter, 0)

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.564605405564335,
            0.485143306872301,
            0.420257096108955,
            0.405867716936220,
            0.418920304310449,
            0.431085818448524,
            0.460255726081957,
            0.509407217025838,
            0.557357527450674,
            0.584326765554389,
            0.595666299923334,
            0.613396690730221,
            0.627874761329268,
            0.631085465808520,
            0.628173039897355,
            0.625458546779438,
            0.618949860617677,
            0.607982680571832,
            0.597283520127575,
            0.591265869710410,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            8.59073106253591E-11,
            4.71518408125651E-11,
            2.88906276938520E-11,
            2.59166393545604E-11,
            2.86005281937073E-11,
            3.13516920990812E-11,
            3.90750935789386E-11,
            5.66307686156151E-11,
            8.13329619654780E-11,
            9.96988893121846E-11,
            1.08609826849242E-10,
            1.24165673963488E-10,
            1.38506652814146E-10,
            1.41904946425871E-10,
            1.38818899333578E-10,
            1.36003040999632E-10,
            1.29481781049290E-10,
            1.19192985299232E-10,
            1.09943987129995E-10,
            1.05060988160446E-10,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            6.48553746967741E-10,
            3.55970904138494E-10,
            2.18108618541326E-10,
            1.95656614551848E-10,
            2.15918524165857E-10,
            2.36688324155684E-10,
            2.94995829449874E-10,
            4.27531683997083E-10,
            6.14019887838562E-10,
            7.52672709240327E-10,
            8.19945469690917E-10,
            9.37383704688155E-10,
            1.04565034114871E-09,
            1.07130562053223E-09,
            1.04800763354537E-09,
            1.02674942559873E-09,
            9.77517438880047E-10,
            8.99842593899902E-10,
            8.30017658458543E-10,
            7.93153656372022E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.91348263991599E+05,
            2.95508926399539E+05,
            4.21400086245930E+05,
            4.55904278660072E+05,
            4.24492378919280E+05,
            3.97167030147011E+05,
            3.38599254744272E+05,
            2.58783518415400E+05,
            1.21098524857946E+05,
            7.88384835245214E+04,
            6.86780997290425E+04,
            5.90437626148323E+04,
            6.13171369891121E+04,
            7.82738172392415E+04,
            9.82147105720494E+04,
            1.11149614042132E+05,
            1.40799387466949E+05,
            1.50935229366213E+05,
            1.60031067344960E+05,
            1.65385719998054E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -1.0465455112526E+06,
            -1.6162338450695E+06,
            -2.3047732940054E+06,
            -2.4934878761874E+06,
            -2.3216860422544E+06,
            -2.1722348765920E+06,
            -1.8519087802210E+06,
            -1.4153707169023E+06,
            -3.4854957265184E+06,
            -2.2691539614727E+06,
            -1.9767146081406E+06,
            -1.6994160953899E+06,
            -1.7648490697025E+06,
            -2.2529015593352E+06,
            -2.8268466059491E+06,
            -3.1991430548182E+06,
            -4.0525321335511E+06,
            -8.2551356091753E+05,
            -8.7526163915542E+05,
            -9.0454796546700E+05,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.91348263991599E+05,
            2.95508926399539E+05,
            4.21400086245930E+05,
            4.55904278660072E+05,
            4.24492378919280E+05,
            3.97167030147011E+05,
            3.38599254744272E+05,
            2.58783518415400E+05,
            2.23713285346802E+05,
            2.06221264170372E+05,
            1.97304574489730E+05,
            1.81360323911413E+05,
            1.63017883528454E+05,
            1.50641118488766E+05,
            1.45667203576232E+05,
            1.44117447679089E+05,
            1.42466076597825E+05,
            1.50935229366213E+05,
            1.60031067344960E+05,
            1.65385719998054E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            1.28755597302346E-10,
            7.46148172878523E-11,
            1.09529229271374E-11,
            -3.69776152422559E-11,
            -6.53485057810297E-11,
            -7.70845503795567E-11,
            -9.56990281646412E-11,
            -1.21388361879674E-10,
            -2.70095500129424E-10,
            -2.15277807907589E-10,
            -2.07812563896076E-10,
            -1.88621892637006E-10,
            -1.66583868356443E-10,
            -1.31873386089419E-10,
            -9.64147261532209E-11,
            -7.80065395723413E-11,
            -4.56162882137967E-11,
            -1.01806828600777E-10,
            -9.56538955097747E-11,
            -9.45298205420616E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_calculate_settlement(self):
        expected = 2.849717515737840
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.84971751573783,
            10.10193918312510,
            17.38583989564510,
            24.95575711969480,
            32.90437075408170,
            41.17532936190790,
            49.64556242438820,
            58.18107658063250,
            66.69005362206340,
            75.12468060779240,
            83.47575953432560,
            91.75911912707220,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    atol=1e-18, rtol=1e-8))


class TestUpdateWeightedMatricesCubic(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        initial_void_ratio_nodes = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        initial_void_ratio_0_nodes = np.array([
            0.802254248593737,
            0.679191704032817,
            0.584150070553881,
            0.530587476655649,
            0.515981351889433,
            0.528766176598625,
            0.554870977120579,
            0.582329942396731,
            0.603536597295654,
            0.615488027450132,
            0.618747094408366,
            0.615894414423534,
            0.610069646102427,
        ])
        for nd, e0, e00 in zip(self.msh.nodes,
                               initial_void_ratio_nodes,
                               initial_void_ratio_0_nodes,
                               ):
            nd.void_ratio = e0
            nd.void_ratio_0 = e00
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e+9
        self.msh.initialize_time_step()
        self.msh._void_ratio_vector[:] = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        expected_void_ratio_vector = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.564605405564336,
            0.485143306872301,
            0.420257096108955,
            0.405867716936220,
            0.418920304310449,
            0.431085818448524,
            0.460255726081957,
            0.509407217025838,
            0.557357527450674,
            0.584326765554389,
            0.595666299923334,
            0.613396690730221,
            0.627874761329268,
            0.631085465808520,
            0.628173039897355,
            0.625458546779438,
            0.618949860617677,
            0.607982680571832,
            0.597283520127575,
            0.591265869710410,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            8.59073106253587E-11,
            4.71518408125654E-11,
            2.88906276938521E-11,
            2.59166393545608E-11,
            2.86005281937074E-11,
            3.13516920990812E-11,
            3.90750935789392E-11,
            5.66307686156151E-11,
            8.13329619654793E-11,
            9.96988893121850E-11,
            1.08609826849242E-10,
            1.24165673963490E-10,
            1.38506652814146E-10,
            1.41904946425873E-10,
            1.38818899333579E-10,
            1.36003040999632E-10,
            1.29481781049291E-10,
            1.19192985299232E-10,
            1.09943987129997E-10,
            1.05060988160446E-10,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            6.48553746967738E-10,
            3.55970904138496E-10,
            2.18108618541327E-10,
            1.95656614551851E-10,
            2.15918524165858E-10,
            2.36688324155684E-10,
            2.94995829449878E-10,
            4.27531683997083E-10,
            6.14019887838572E-10,
            7.52672709240330E-10,
            8.19945469690917E-10,
            9.37383704688169E-10,
            1.04565034114871E-09,
            1.07130562053225E-09,
            1.04800763354538E-09,
            1.02674942559873E-09,
            9.77517438880057E-10,
            8.99842593899900E-10,
            8.30017658458556E-10,
            7.93153656372024E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.91348263991600E+05,
            2.95508926399538E+05,
            4.21400086245929E+05,
            4.55904278660068E+05,
            4.24492378919279E+05,
            3.97167030147011E+05,
            3.38599254744269E+05,
            2.58783518415400E+05,
            1.21098524857941E+05,
            7.88384835245203E+04,
            6.86780997290420E+04,
            5.90437626148318E+04,
            6.13171369891120E+04,
            7.82738172392405E+04,
            9.82147105720476E+04,
            1.11149614042132E+05,
            1.40799387466950E+05,
            1.50935229366213E+05,
            1.60031067344958E+05,
            1.65385719998053E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -1.0465455112526E+06,
            -1.6162338450695E+06,
            -2.3047732940054E+06,
            -2.4934878761874E+06,
            -2.3216860422544E+06,
            -2.1722348765920E+06,
            -1.8519087802210E+06,
            -1.4153707169023E+06,
            -3.4854957265183E+06,
            -2.2691539614727E+06,
            -1.9767146081406E+06,
            -1.6994160953899E+06,
            -1.7648490697025E+06,
            -2.2529015593352E+06,
            -2.8268466059490E+06,
            -3.1991430548182E+06,
            -4.0525321335512E+06,
            -8.2551356091753E+05,
            -8.7526163915541E+05,
            -9.0454796546700E+05,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        print(actual_dsigde_int_pts)
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.91348263991600E+05,
            2.95508926399538E+05,
            4.21400086245929E+05,
            4.55904278660068E+05,
            4.24492378919279E+05,
            3.97167030147011E+05,
            3.38599254744269E+05,
            2.58783518415400E+05,
            2.23713285346800E+05,
            2.06221264170372E+05,
            1.97304574489731E+05,
            1.81360323911411E+05,
            1.63017883528454E+05,
            1.50641118488764E+05,
            1.45667203576232E+05,
            1.44117447679089E+05,
            1.42466076597824E+05,
            1.50935229366213E+05,
            1.60031067344958E+05,
            1.65385719998053E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            1.28755597302353E-10,
            7.46148172878518E-11,
            1.09529229271368E-11,
            -3.69776152422568E-11,
            -6.53485057810324E-11,
            -7.70845503795531E-11,
            -9.56990281646424E-11,
            -1.21388361879674E-10,
            -2.70095500129423E-10,
            -2.15277807907599E-10,
            -2.07812563896064E-10,
            -1.88621892637009E-10,
            -1.66583868356442E-10,
            -1.31873386089425E-10,
            -9.64147261532523E-11,
            -7.80065395723034E-11,
            -4.56162882137969E-11,
            -1.01806828600777E-10,
            -9.56538955097771E-11,
            -9.45298205420664E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_calculate_settlement(self):
        expected = 2.849717515737840
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.84971751573783,
            10.10193918312510,
            17.38583989564510,
            24.95575711969480,
            32.90437075408170,
            41.17532936190790,
            49.64556242438820,
            58.18107658063250,
            66.69005362206340,
            75.12468060779240,
            83.47575953432560,
            91.75911912707220,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_weighted(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._weighted_stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_mass_matrix_weighted(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._weighted_mass_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C0[0:4, 0:4] = np.array([
            [1.94111127255236E-10, 1.97717842048361E-09,
             -6.44419004684924E-10, 2.54126566542399E-10],
            [1.62318348377551E-09, 3.80549604455875E-09,
             6.41867138086242E-10, -6.21270142417696E-10],
            [-5.06105282870977E-10, 3.89394907291124E-10,
             4.40356061425780E-09, 1.74767704174649E-09],
            [2.19674929857516E-10, -5.48452471153168E-10,
             1.59528725511373E-09, 1.35271466737085E-09],
        ])
        expected_C0[3:7, 3:7] = np.array([
            [1.35271466737085E-09, 1.88693886993210E-09,
             -7.24465353603536E-10, 2.78339075606081E-10],
            [1.69717287166591E-09, 4.07841410990068E-09,
             1.18842047595217E-09, -7.73777206426125E-10],
            [-6.26419303828751E-10, 7.58907399848305E-10,
             2.76868442113293E-09, 3.20886498009664E-09],
            [2.37798342089257E-10, -6.00404578644066E-10,
             2.75139443896618E-09, -9.38056698373154E-10],
        ])
        expected_C0[6:10, 6:10] = np.array([
            [-9.38056698373154E-10, 3.35077233058000E-09,
             -1.22531897784410E-09, 4.40848734882380E-10],
            [2.78240180037441E-09, 7.48977033286816E-10,
             3.89555748014249E-09, -1.46865805430337E-09],
            [-9.77770430208369E-10, 2.95689074887369E-09,
             -9.38705382336200E-10, 4.77870556050012E-09],
            [3.65688655309741E-10, -1.18926394389017E-09,
             4.09952370604433E-09, -3.78816275321872E-09],
        ])
        expected_C0[9:13, 9:13] = np.array([
            [-3.78816275321872E-09, 5.15582043417614E-09,
             -1.12138292198810E-09, 3.52080400073365E-10],
            [4.50804725877288E-09, -2.37802248680026E-10,
             2.51978236913352E-09, -9.99777333823209E-10],
            [-8.56934566145133E-10, 1.68634004339398E-09,
             2.59965520949223E-09, 2.37721127626317E-09],
            [2.79546394791862E-10, -7.67830065293762E-10,
             1.84567992502723E-09, 5.81409301027040E-10],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C1[0:4, 0:4] = np.array([
            [1.95384166512722E-09, -3.22947371733365E-10,
             6.01143655121627E-11, 8.99884506903008E-11],
            [3.10475649747281E-11, 7.61709331908281E-09,
             -2.09873244930040E-09, -1.00131910754332E-10],
            [-7.81993563017849E-11, -1.84626021850528E-09,
             7.77823802022922E-09, 1.80748835002288E-10],
            [1.24440087375184E-10, -1.72949582018861E-10,
             3.33138621635041E-10, 3.61799767560999E-09],
        ])
        expected_C1[3:7, 3:7] = np.array([
            [3.61799767560999E-09, 5.08857714639332E-11,
             1.59893035031092E-11, 9.05350955551746E-11],
            [2.40651769730115E-10, 8.61786082400174E-09,
             -2.75103809953770E-09, 8.27557568984909E-11],
            [-8.20567462716754E-11, -2.32152502343383E-09,
             9.81956880680082E-09, -1.30594953984620E-09],
            [1.31075829071998E-10, -9.06168708835672E-11,
             -8.48478998715738E-10, 5.83763776728587E-09],
        ])
        expected_C1[6:10, 6:10] = np.array([
            [5.83763776728587E-09, -1.46241233492501E-09,
             5.34696246439430E-10, -8.48680466164861E-11],
            [-8.94041804719421E-10, 1.14582961879649E-08,
             -5.41690107867945E-09, 8.10924954934344E-10],
            [2.87147698803702E-10, -4.47823434741065E-09,
             1.29663319743526E-08, -2.95612482891641E-09],
            [-9.70796704384715E-12, 5.31530844521144E-10,
             -2.27694297446062E-09, 8.50599810662939E-09],
        ])
        expected_C1[9:13, 9:13] = np.array([
            [8.50599810662939E-09, -3.33449859223987E-09,
             4.59848268035733E-10, -1.92410893534812E-12],
            [-2.68672541683661E-09, 1.21567804887976E-08,
             -4.01453997122810E-09, 3.34734944670265E-10],
            [1.95399912192764E-10, -3.18109764548856E-09,
             9.34084366022580E-09, -5.48873963925753E-10],
            [7.06098963461549E-11, 1.02787676140819E-10,
             -1.73426126898161E-11, 1.78275059575521E-09],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    ))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    ))


class TestVoidRatioCorrectionCubicOneStep(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        initial_void_ratio_nodes = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        initial_void_ratio_0_nodes = np.array([
            0.802254248593737,
            0.679191704032817,
            0.584150070553881,
            0.530587476655649,
            0.515981351889433,
            0.528766176598625,
            0.554870977120579,
            0.582329942396731,
            0.603536597295654,
            0.615488027450132,
            0.618747094408366,
            0.615894414423534,
            0.610069646102427,
        ])
        for nd, e0, e00 in zip(self.msh.nodes,
                               initial_void_ratio_nodes,
                               initial_void_ratio_0_nodes,
                               ):
            nd.void_ratio = e0
            nd.void_ratio_0 = e00
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e+9
        self.msh.initialize_time_step()
        self.msh._void_ratio_vector[:] = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        expected_void_ratio_vector = np.array([
            0.590000000000000,
            0.470206975127078,
            0.417194357855208,
            0.431610560346886,
            0.490075955902514,
            0.551866430440789,
            0.582644566423252,
            0.614026858728604,
            0.622756198569330,
            0.619538396028206,
            0.616173393531223,
            0.602516735058181,
            0.590000000000000,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.569128079115627,
            0.499985012736970,
            0.435312589780856,
            0.414418256881911,
            0.425480544594668,
            0.438350670681608,
            0.470137534797798,
            0.522701646894974,
            0.566234372947905,
            0.581474934686134,
            0.588831829690780,
            0.607280247189220,
            0.620554034576872,
            0.622541518497027,
            0.620312424372368,
            0.620115464242184,
            0.618835743140346,
            0.609917047579777,
            0.598013122313993,
            0.591264194921091,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            8.88911574750356E-11,
            5.27424115230687E-11,
            3.23682524028665E-11,
            2.76447866436949E-11,
            3.00526683225420E-11,
            3.31192255629855E-11,
            4.21016774402141E-11,
            6.26095742760141E-11,
            8.69703063956344E-11,
            9.75753358936280E-11,
            1.03148050022034E-10,
            1.18562578819859E-10,
            1.31059422918151E-10,
            1.33040721682186E-10,
            1.30820584958994E-10,
            1.30626206822924E-10,
            1.29370277263680E-10,
            1.20946382768782E-10,
            1.10551240700388E-10,
            1.05059659803572E-10,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            6.71080177380339E-10,
            3.98176690300247E-10,
            2.44362804816751E-10,
            2.08703192212370E-10,
            2.26881396997310E-10,
            2.50032246140451E-10,
            3.17844901193052E-10,
            4.72668434120175E-10,
            6.56578790294139E-10,
            7.36641029090343E-10,
            7.78712007711933E-10,
            8.95083365827997E-10,
            9.89427781992576E-10,
            1.00438551641499E-09,
            9.87624684535542E-10,
            9.86157234704333E-10,
            9.76675645586414E-10,
            9.13079796770326E-10,
            8.34602094585936E-10,
            7.93143627995839E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.67992969558270E+05,
            1.92774364801713E+05,
            2.73212741561396E+05,
            3.56444500730514E+05,
            3.51452708049072E+05,
            3.22227451522590E+05,
            2.54778859355561E+05,
            1.76504685983604E+05,
            9.37946968713443E+04,
            8.55827323412454E+04,
            8.36082528165774E+04,
            7.04092259274252E+04,
            7.56991412958137E+04,
            1.00095800080884E+05,
            1.23149948690044E+05,
            1.29627304618239E+05,
            1.41262612540818E+05,
            1.42761476662028E+05,
            1.56705512638555E+05,
            1.65387234932263E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -4.8352263429084E+06,
            -5.5484922337977E+06,
            -7.8636948241913E+06,
            -1.0259297423272E+07,
            -1.0115622080577E+07,
            -9.2744515803672E+06,
            -7.3331250445268E+06,
            -5.0802132348680E+06,
            -2.6996283852232E+06,
            -2.4632690463331E+06,
            -2.4064389573341E+06,
            -2.0265404253717E+06,
            -2.1787964287524E+06,
            -2.8809887142195E+06,
            -3.5445404507085E+06,
            -3.7309737407369E+06,
            -4.0658648229235E+06,
            -4.1090056001975E+06,
            -4.5103472173941E+06,
            -9.0455625113191E+05,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        print(actual_dsigde_int_pts)
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.91348263991600E+05,
            2.95508926399538E+05,
            4.21400086245929E+05,
            4.55904278660068E+05,
            4.24492378919279E+05,
            3.97167030147011E+05,
            3.38599254744269E+05,
            2.58783518415400E+05,
            2.23713285346800E+05,
            2.06221264170372E+05,
            1.97304574489731E+05,
            1.81360323911411E+05,
            1.63017883528454E+05,
            1.50641118488764E+05,
            1.45667203576232E+05,
            1.44117447679089E+05,
            1.42466076597824E+05,
            1.50935229366213E+05,
            1.60031067344958E+05,
            1.65387234932263E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            7.67937592531328E-10,
            3.81715291051708E-10,
            1.51152823282975E-10,
            -3.65326830878692E-11,
            -1.89003705268055E-10,
            -2.37467103403308E-10,
            -2.92102791443893E-10,
            -3.11302680800663E-10,
            -2.03874023282467E-10,
            -1.35286787782316E-10,
            -2.28077187849400E-10,
            -1.92155987634666E-10,
            -1.57854662335974E-10,
            -1.20977692552541E-10,
            -9.89886847658185E-11,
            -1.44751508323270E-10,
            -8.47109090106859E-11,
            -3.38596727548286E-11,
            -2.38122199274877E-11,
            -9.41123530521890E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_calculate_settlement(self):
        expected = 2.648933407448820
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.64893340744882,
            9.95457785203426,
            17.31318066455180,
            24.92543424093510,
            32.92320592863530,
            41.26427469656750,
            49.75964144772940,
            58.26117572795490,
            66.73187533571180,
            75.12367131839760,
            83.46438769334810,
            91.75664141749700,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-6.97052913516213E-09, 9.25667624542665E-09,
             -2.98537266328442E-09, 6.99225553019894E-10],
            [8.51073397651034E-09, -1.75209877052056E-08,
             1.12391898889759E-08, -2.22893616028064E-09],
            [-2.69630228103718E-09, 1.06843941643020E-08,
             -1.45497799267633E-08, 6.56168804349848E-09],
            [6.27482741975240E-10, -2.07313965683883E-09,
             6.24054684978189E-09, -9.78465371417183E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-9.78465371417183E-09, 6.32487551476772E-09,
             -1.75615789753342E-09, 4.21046162019225E-10],
            [5.92192154519666E-09, -1.37033592059372E-08,
             9.47005733533779E-09, -1.68861967459724E-09],
            [-1.54761987440241E-09, 8.54138741981539E-09,
             -1.21442630562243E-08, 5.15049551081127E-09],
            [3.39328773350114E-10, -1.33925990088167E-09,
             4.22530651519816E-09, -7.28948532995730E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-7.28948532995730E-09, 5.48693690443767E-09,
             -2.02421123098987E-09, 6.01384268842896E-10],
            [4.40108491747705E-09, -1.25329929816024E-08,
             1.07416002313876E-08, -2.60969216726222E-09],
            [-1.55328254328389E-09, 8.95834084364957E-09,
             -1.64193159002880E-08, 9.01425759992227E-09],
            [4.58638121153452E-10, -2.07977054459775E-09,
             7.73071247927224E-09, -1.39899037855766E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.39899037855766E-08, 9.74797091021575E-09,
             -2.00668633379695E-09, 1.39039153329849E-10],
            [8.48082789030296E-09, -2.00427738536990E-08,
             1.18511921791768E-08, -2.89246215780780E-10],
            [-1.48872060931180E-09, 1.01712689768135E-08,
             -1.24856512053271E-08, 3.80310283782546E-09],
            [-3.53702411193186E-12, 1.75566815639992E-10,
             2.73749580378234E-09, -2.90952559531040E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_weighted(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._weighted_stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_mass_matrix_weighted(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._weighted_mass_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C0[0:4, 0:4] = np.array([
            [1.94111127255236E-10, 1.97717842048361E-09,
             -6.44419004684924E-10, 2.54126566542399E-10],
            [1.62318348377551E-09, 3.80549604455875E-09,
             6.41867138086242E-10, -6.21270142417696E-10],
            [-5.06105282870977E-10, 3.89394907291124E-10,
             4.40356061425780E-09, 1.74767704174649E-09],
            [2.19674929857516E-10, -5.48452471153168E-10,
             1.59528725511373E-09, 1.35271466737085E-09],
        ])
        expected_C0[3:7, 3:7] = np.array([
            [1.35271466737085E-09, 1.88693886993210E-09,
             -7.24465353603536E-10, 2.78339075606081E-10],
            [1.69717287166591E-09, 4.07841410990068E-09,
             1.18842047595217E-09, -7.73777206426125E-10],
            [-6.26419303828751E-10, 7.58907399848305E-10,
             2.76868442113293E-09, 3.20886498009664E-09],
            [2.37798342089257E-10, -6.00404578644066E-10,
             2.75139443896618E-09, -9.38056698373154E-10],
        ])
        expected_C0[6:10, 6:10] = np.array([
            [-9.38056698373154E-10, 3.35077233058000E-09,
             -1.22531897784410E-09, 4.40848734882380E-10],
            [2.78240180037441E-09, 7.48977033286816E-10,
             3.89555748014249E-09, -1.46865805430337E-09],
            [-9.77770430208369E-10, 2.95689074887369E-09,
             -9.38705382336200E-10, 4.77870556050012E-09],
            [3.65688655309741E-10, -1.18926394389017E-09,
             4.09952370604433E-09, -3.78816275321872E-09],
        ])
        expected_C0[9:13, 9:13] = np.array([
            [-3.78816275321872E-09, 5.15582043417614E-09,
             -1.12138292198810E-09, 3.52080400073365E-10],
            [4.50804725877288E-09, -2.37802248680026E-10,
             2.51978236913352E-09, -9.99777333823209E-10],
            [-8.56934566145133E-10, 1.68634004339398E-09,
             2.59965520949223E-09, 2.37721127626317E-09],
            [2.79546394791862E-10, -7.67830065293762E-10,
             1.84567992502723E-09, 5.81409301027040E-10],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C1[0:4, 0:4] = np.array([
            [1.95384166512722E-09, -3.22947371733365E-10,
             6.01143655121627E-11, 8.99884506903008E-11],
            [3.10475649747281E-11, 7.61709331908281E-09,
             -2.09873244930040E-09, -1.00131910754332E-10],
            [-7.81993563017849E-11, -1.84626021850528E-09,
             7.77823802022922E-09, 1.80748835002288E-10],
            [1.24440087375184E-10, -1.72949582018861E-10,
             3.33138621635041E-10, 3.61799767560999E-09],
        ])
        expected_C1[3:7, 3:7] = np.array([
            [3.61799767560999E-09, 5.08857714639332E-11,
             1.59893035031092E-11, 9.05350955551746E-11],
            [2.40651769730115E-10, 8.61786082400174E-09,
             -2.75103809953770E-09, 8.27557568984909E-11],
            [-8.20567462716754E-11, -2.32152502343383E-09,
             9.81956880680082E-09, -1.30594953984620E-09],
            [1.31075829071998E-10, -9.06168708835672E-11,
             -8.48478998715738E-10, 5.83763776728587E-09],
        ])
        expected_C1[6:10, 6:10] = np.array([
            [5.83763776728587E-09, -1.46241233492501E-09,
             5.34696246439430E-10, -8.48680466164861E-11],
            [-8.94041804719421E-10, 1.14582961879649E-08,
             -5.41690107867945E-09, 8.10924954934344E-10],
            [2.87147698803702E-10, -4.47823434741065E-09,
             1.29663319743526E-08, -2.95612482891641E-09],
            [-9.70796704384715E-12, 5.31530844521144E-10,
             -2.27694297446062E-09, 8.50599810662939E-09],
        ])
        expected_C1[9:13, 9:13] = np.array([
            [8.50599810662939E-09, -3.33449859223987E-09,
             4.59848268035733E-10, -1.92410893534812E-12],
            [-2.68672541683661E-09, 1.21567804887976E-08,
             -4.01453997122810E-09, 3.34734944670265E-10],
            [1.95399912192764E-10, -3.18109764548856E-09,
             9.34084366022580E-09, -5.48873963925753E-10],
            [7.06098963461549E-11, 1.02787676140819E-10,
             -1.73426126898161E-11, 1.78275059575521E-09],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    ))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    ))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -2.11096312236706E-10,
            1.01796928182927E-10,
            5.66617037880383E-11,
            2.60446276751471E-11,
            6.78604291542197E-11,
            1.00026757939162E-10,
            -4.77243072914896E-11,
            -2.77588933786640E-11,
            -5.89618733068533E-11,
            -5.07320621382569E-11,
            3.01293233046077E-11,
            6.86868124271470E-12,
            6.88837112775813E-12,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-20,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000E+00,
            1.64971121645743E-02,
            1.10391936750544E-02,
            6.93885380624079E-03,
            1.15834854570863E-02,
            1.20047698913215E-02,
            -7.35543357674824E-03,
            -6.38358137522411E-03,
            -8.26512908961414E-03,
            -7.35133256844753E-03,
            1.29270060395583E-03,
            1.32935934700725E-03,
            0.00000000000000E+00,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            rtol=1e-10, atol=1e-13,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.52376731431709E-02
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
        self.assertEqual(self.msh._iter, 1)


class TestIterativeVoidRatioCorrectionCubic(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=4.05e-4,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
        )
        self.msh = ConsolidationAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        initial_void_ratio_nodes = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        initial_void_ratio_0_nodes = np.array([
            0.802254248593737,
            0.679191704032817,
            0.584150070553881,
            0.530587476655649,
            0.515981351889433,
            0.528766176598625,
            0.554870977120579,
            0.582329942396731,
            0.603536597295654,
            0.615488027450132,
            0.618747094408366,
            0.615894414423534,
            0.610069646102427,
        ])
        for nd, e0, e00 in zip(self.msh.nodes,
                               initial_void_ratio_nodes,
                               initial_void_ratio_0_nodes,
                               ):
            nd.void_ratio = e0
            nd.void_ratio_0 = e00
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
        bnd0 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.59,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e+9
        self.msh.initialize_time_step()
        self.msh._void_ratio_vector[:] = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()
        self.msh.iterative_correction_step()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.590000000000000,
            0.453709862962504,
            0.406155164180154,
            0.424671706540645,
            0.478492470445428,
            0.539861660549467,
            0.590000000000000,
            0.620410440103828,
            0.631021327658944,
            0.626889728596653,
            0.614880692927267,
            0.601187375711174,
            0.590000000000000,
        ])
        expected_void_ratio_vector = np.array([
            0.590000000000000,
            0.493837930960843,
            0.426621202596801,
            0.442474208265292,
            0.490847090559195,
            0.542308270456928,
            0.582164160917045,
            0.613154017101859,
            0.621886599194566,
            0.618529165428225,
            0.613118660189295,
            0.601823005638421,
            0.590000000000000,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.576745133720561,
            0.522857285889452,
            0.453228624609594,
            0.420138479530192,
            0.433192442682735,
            0.448544408978053,
            0.474951426032962,
            0.517109992497673,
            0.556490377142573,
            0.577807518044875,
            0.588252092137429,
            0.606461011500498,
            0.619667013770159,
            0.621669175721034,
            0.619351663282738,
            0.618324343279851,
            0.615659081595615,
            0.607871614188826,
            0.597992435016069,
            0.591494971364410,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            9.41526419223176E-11,
            6.26831831522095E-11,
            3.70561546163506E-11,
            2.88647679760228E-11,
            3.18542915038672E-11,
            3.57686266663153E-11,
            4.36598912437473E-11,
            6.00215847106311E-11,
            8.08022534717506E-11,
            9.49108227390427E-11,
            1.02697587916880E-10,
            1.17831557517760E-10,
            1.30184711920662E-10,
            1.32167432007984E-10,
            1.29875146475690E-10,
            1.28871768413599E-10,
            1.26304616683483E-10,
            1.19093084925694E-10,
            1.10533976386396E-10,
            1.05242858210298E-10,
        ])
        actual_hyd_cond_int_pts = np.array([
            ip.hyd_cond for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_int_pts,
            expected_hyd_cond_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array([
            7.10801540184705E-10,
            4.73224141336699E-10,
            2.79753931879644E-10,
            2.17913391653515E-10,
            2.40482677917021E-10,
            2.70033791995826E-10,
            3.29608572916701E-10,
            4.53130512172388E-10,
            6.10013325653703E-10,
            7.16525395418754E-10,
            7.75311262373290E-10,
            8.89564550244800E-10,
            9.82824186899151E-10,
            9.97792651543891E-10,
            9.80487135165054E-10,
            9.72912173301421E-10,
            9.53531565743330E-10,
            8.99088400093697E-10,
            8.34471758349744E-10,
            7.94526676915139E-10,
        ])
        actual_hyd_cond_grad_int_pts = np.array([
            ip.hyd_cond_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_hyd_cond_grad_int_pts,
            expected_hyd_cond_grad_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            1.34920494569552E+05,
            9.98035217371275E+04,
            1.63136264614416E+05,
            3.02335493703559E+05,
            2.81493368408200E+05,
            2.40292827533276E+05,
            2.21814406002999E+05,
            2.07325117329570E+05,
            1.24159007347325E+05,
            9.51105856338515E+04,
            8.50150603362701E+04,
            7.20891684338797E+04,
            7.76566609747067E+04,
            1.02640826388701E+05,
            1.26602927690093E+05,
            1.36485181849816E+05,
            1.43827190515702E+05,
            1.47761954172425E+05,
            1.55996854257109E+05,
            1.64167203410191E+05,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -3.8833239941904E+06,
            -2.8725762672527E+06,
            -4.6954391378486E+06,
            -8.7019150110601E+06,
            -8.1020304234175E+06,
            -6.9161835328939E+06,
            -6.3843318084229E+06,
            -5.9672965570789E+06,
            -3.5735834934861E+06,
            -2.7375027083305E+06,
            -2.4469301326286E+06,
            -2.0748930575274E+06,
            -2.2351383741507E+06,
            -2.9542404596902E+06,
            -3.6439251753576E+06,
            -3.9283593142746E+06,
            -4.1396793106084E+06,
            -4.2529309123637E+06,
            -4.4899503895798E+06,
            -4.7251119416353E+06,
        ])
        actual_dsigde_int_pts = np.array([
            ip.eff_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        print(actual_dsigde_int_pts)
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array([
            1.91348263991600E+05,
            2.95508926399538E+05,
            4.21400086245929E+05,
            4.55904278660068E+05,
            4.24492378919279E+05,
            3.97167030147011E+05,
            3.38599254744269E+05,
            2.58783518415400E+05,
            2.23713285346800E+05,
            2.06221264170372E+05,
            1.97304574489731E+05,
            1.81360323911411E+05,
            1.63017883528454E+05,
            1.50641118488764E+05,
            1.45667203576232E+05,
            1.44117447679089E+05,
            1.44941932159988E+05,
            1.51803313745405E+05,
            1.60223705341556E+05,
            1.65416798965919E+05,
        ])
        actual_ppc_int_pts = np.array([
            ip.pre_consol_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ppc_int_pts,
            expected_ppc_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            3.88164398184163E-10,
            1.74843041142561E-10,
            1.24777899971609E-10,
            -1.02768381367832E-11,
            -2.27150845618871E-10,
            -1.80725558279008E-10,
            -2.25218405500305E-10,
            -2.9185617341162E-10,
            -2.35878425499901E-10,
            -1.98436281036782E-10,
            -2.27303444310517E-10,
            -1.92088456270699E-10,
            -1.57708324407301E-10,
            -1.19763211535513E-10,
            -9.57590621902482E-11,
            -1.13735426701145E-10,
            -7.90666213070898E-11,
            -4.66983466248786E-11,
            -3.48670002680157E-11,
            -3.85208976406866E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-18, rtol=1e-8,
        ))

    def test_calculate_settlement(self):
        expected = 2.510612104602660
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.51061210460266,
            9.89932801458063,
            17.34686286857390,
            24.99459990877970,
            33.02880558270170,
            41.34146347073530,
            49.79645596467100,
            58.29410504883410,
            66.76019350988060,
            75.14730643823020,
            83.47436213903750,
            91.75637506650600,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-1.75973053787199E-09, 2.30012579221698E-09, -
                7.04533370197091E-10, 1.64138115852099E-10],
            [1.59213591880079E-09, -3.81159727452408E-09,
                2.74059958738665E-09, -5.21138231663366E-10],
            [-4.27905926569196E-10, 2.23565512579641E-09, -
                3.37467740597143E-09, 1.56692820674421E-09],
            [9.52348424823326E-11, -3.75502889134308E-10,
                1.26214863347870E-09, -2.26528300823915E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-2.26528300823915E-09, 1.83605309846816E-09, -
                7.40454657106646E-10, 1.87803980050909E-10],
            [1.45652110193580E-09, -4.53944671410105E-09,
                3.93945857548987E-09, -8.56532963324624E-10],
            [-5.44362557557076E-10, 3.08043242328213E-09, -
                7.05088438566787E-09, 4.51481451994281E-09],
            [1.06722513017261E-10, -5.09787707760505E-10,
                3.59987343768189E-09, -6.77569446565898E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-6.77569446565898E-09, 4.81318466550499E-09, -
                1.76001522428352E-09, 5.25716781498861E-10],
            [3.67644360509381E-09, -1.07093191546780E-08,
                9.31245855882189E-09, -2.27958300923769E-09],
            [-1.26491812901206E-09, 7.43512509628429E-09, -
                1.39050373566887E-08, 7.73483038941651E-09],
            [3.75396622353581E-10, -1.72079478841128E-09,
                6.37646668050490E-09, -1.22941608598481E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.22941608598481E-08, 8.49031902641605E-09, -
                1.58123119002382E-09, 3.54004509008710E-10],
            [7.19477267560952E-09, -1.23945827374777E-08,
                6.53432234036163E-09, -1.33451227849348E-09],
            [-1.05233447833788E-09, 4.86743768888254E-09, -
                6.74118845073361E-09, 2.92608524018894E-09],
            [2.08936498445703E-10, -8.70617741434581E-10,
                1.86302253771706E-09, -1.20134129472818E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-5.63135425474188E-09, 7.78243995664323E-09,
             -2.79873865081796E-09, 6.47652948916602E-10],
            [6.97020157689739E-09, -1.41312646027458E-08,
             9.02266945624223E-09, -1.86160643039379E-09],
            [-2.48991229578893E-09, 8.39738862969527E-09,
             -1.14288526123578E-08, 5.52137627845144E-09],
            [5.71494436454711E-10, -1.69146422933953E-09,
             5.18342687254891E-09, -8.24804485383431E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-8.24804485383431E-09, 5.32414926672877E-09,
             -1.48175040957510E-09, 3.42188917016555E-10],
            [4.90071689835126E-09, -1.29065442508259E-08,
             9.53037251633964E-09, -1.52454516386499E-09],
            [-1.27136856609679E-09, 8.63393174855245E-09,
             -1.28312175393721E-08, 5.46865435691640E-09],
            [2.60742311143565E-10, -1.18404318500092E-09,
             4.58051527291278E-09, -7.77313231633711E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-7.77313231633711E-09, 5.55559590349438E-09,
             -2.05020757846383E-09, 6.10529592251139E-10],
            [4.47495308480110E-09, -1.27394083683484E-08,
             1.09203725119806E-08, -2.65591722843331E-09],
            [-1.58160754352372E-09, 9.14789576927411E-09,
             -1.67504576477152E-08, 9.18416942196483E-09],
            [4.68579039103069E-10, -2.12938383505383E-09,
             7.90888649180580E-09, -1.43916719340399E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.43916719340399E-08, 1.03931158844223E-08,
             -2.97809869811374E-09, 7.28573051876312E-10],
            [9.14635342994599E-09, -2.22820983279098E-08,
             1.61491020407110E-08, -3.01335714274714E-09],
            [-2.46623414338129E-09, 1.44910410637981E-08,
             -2.22747957990703E-08, 1.02499888786535E-08],
            [5.86990518639882E-10, -2.55102735783727E-09,
             9.18398599028779E-09, -7.21994915109040E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_weighted(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [-3.69345449176234E-09, 5.03909134887713E-09,
             -1.75182216211486E-09, 4.06185305000065E-10],
            [4.27884039910890E-09, -8.96582604646811E-09,
             5.87950754051203E-09, -1.19252189315282E-09],
            [-1.45905358562669E-09, 5.31419887665779E-09,
             -7.40137962943426E-09, 3.54623433840316E-09],
            [3.33651085265848E-10, -1.03462425256792E-09,
             3.22488814826489E-09, -5.26016363981594E-09],
        ])
        expected_K[3:7, 3:7] = np.array([
            [-5.26016363981594E-09, 3.58308679551943E-09,
             -1.11228378474802E-09, 2.65445648081713E-10],
            [3.18166387779445E-09, -8.72404548461169E-09,
             6.73418843191565E-09, -1.19180682509841E-09],
            [-9.09050296585882E-10, 5.85629528398332E-09,
             -9.93945094480137E-09, 4.99220595740393E-09],
            [1.84187645374258E-10, -8.48181519929124E-10,
             4.09061774951111E-09, -7.27529689831190E-09],
        ])
        expected_K[6:10, 6:10] = np.array([
            [-7.27529689831190E-09, 5.18538615665021E-09,
             -1.90418734464830E-09, 5.67474211353737E-10],
            [4.07681518005153E-09, -1.17218347445569E-08,
             1.01092246276546E-08, -2.46420506314919E-09],
            [-1.42234330621996E-09, 8.28409889712094E-09,
             -1.53084968353989E-08, 8.44674124449792E-09],
            [4.21308548749983E-10, -1.92135380094985E-09,
             7.12934843691023E-09, -1.33216237349816E-08],
        ])
        expected_K[9:13, 9:13] = np.array([
            [-1.33216237349816E-08, 9.42830129215964E-09,
             -2.27609373705245E-09, 5.40112995164061E-10],
            [8.15660844135132E-09, -1.73296601475327E-08,
             1.13443211036209E-08, -2.17126939743955E-09],
            [-1.75559774319384E-09, 9.68234846112700E-09,
             -1.45213954724940E-08, 6.59464475456080E-09],
            [3.96766900939393E-10, -1.70825354171147E-09,
             5.53055999523556E-09, -4.21907335446348E-09],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._weighted_stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
        ))

    def test_global_mass_matrix_weighted(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [1.073976396191230, 0.827115524375121,
             -0.292152319586382, 0.172057508616350],
            [0.827115524375121, 5.711294681820770,
             -0.728432655607079, -0.360701026586015],
            [-0.292152319586382, -0.728432655607079,
                6.090899317243500, 0.964212938374387],
            [0.172057508616350, -0.360701026586015,
                0.964212938374387, 2.485356171490430],
        ])
        expected_M[3:7, 3:7] = np.array([
            [2.485356171490430, 0.968912320698014,
             -0.354238025050214, 0.184437085580628],
            [0.968912320698014, 6.348137466951200,
             -0.781308811792766, -0.345510724763817],
            [-0.354238025050214, -0.781308811792766,
                6.294126613966870, 0.951457720125221],
            [0.184437085580628, -0.345510724763817,
                0.951457720125221, 2.449790534456370],
        ])
        expected_M[6:10, 6:10] = np.array([
            [2.449790534456370, 0.944179997827494,
             -0.345311365702335, 0.177990344132947],
            [0.944179997827494, 6.103636610625840,
             -0.760671799268481, -0.328866549684514],
            [-0.345311365702335, -0.760671799268481,
                6.013813296008200, 0.911290365791853],
            [0.177990344132947, -0.328866549684514,
                0.911290365791853, 2.358917676705340],
        ])
        expected_M[9:13, 9:13] = np.array([
            [2.358917676705340, 0.910660920968132,
             -0.330767326976185, 0.175078145569009],
            [0.910660920968132, 5.959489120058790,
             -0.747378801047292, -0.332521194576472],
            [-0.330767326976185, -0.747378801047292,
                5.970249434859010, 0.914168656168706],
            [0.175078145569009, -0.332521194576472,
                0.914168656168706, 1.182079948391130],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._weighted_mass_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C0[0:4, 0:4] = np.array([
            [-7.72750849689942E-10, 3.34666119881369E-09,
             -1.16806340064381E-09, 3.75150161116382E-10],
            [2.96653572392957E-09, 1.22838165858672E-09,
             2.21132111464894E-09, -9.56961973162426E-10],
            [-1.02167911239973E-09, 1.92866678272182E-09,
             2.39020950252638E-09, 2.73733010757597E-09],
            [3.38883051249274E-10, -8.78013152869972E-10,
             2.57665701250683E-09, -1.44725648417549E-10],
        ])
        expected_C0[3:7, 3:7] = np.array([
            [-1.44725648417549E-10, 2.76045571845773E-09,
             -9.10379917424225E-10, 3.17159909621484E-10],
            [2.55974425959524E-09, 1.98611472464536E-09,
             2.58578540416506E-09, -9.41414137313021E-10],
            [-8.08763173343154E-10, 2.14683883019890E-09,
             1.32440114156619E-09, 3.44756069882718E-09],
            [2.76530908267756E-10, -7.69601484728379E-10,
             2.99676659488078E-09, -1.18785791469959E-09],
        ])
        expected_C0[6:10, 6:10] = np.array([
            [-1.18785791469959E-09, 3.53687307615260E-09,
             -1.29740503802648E-09, 4.61727449809815E-10],
            [2.98258758785326E-09, 2.42719238347371E-10,
             4.29394051455882E-09, -1.56096908125911E-09],
            [-1.05648301881232E-09, 3.38137764929199E-09,
             -1.64043512169125E-09, 5.13466098804081E-09],
            [3.88644618507938E-10, -1.28954345015944E-09,
             4.47596458424697E-09, -4.30189419078548E-09],
        ])
        expected_C0[9:13, 9:13] = np.array([
            [-4.30189419078548E-09, 5.62481156704795E-09,
             -1.46881419550241E-09, 4.45134643151039E-10],
            [4.98896514164379E-09, -2.70534095370755E-09,
             4.92478175076317E-09, -1.41815589329625E-09],
            [-1.20856619857310E-09, 4.09379542951621E-09,
             -1.29044830138797E-09, 4.21149103344911E-09],
            [3.73461596038705E-10, -1.18664796543221E-09,
             3.67944865378648E-09, -9.27456728840614E-10],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C1[0:4, 0:4] = np.array([
            [2.92070364207240E-09, -1.69243015006345E-09,
             5.83758761471049E-10, -3.10351438836824E-11],
            [-1.31230467517933E-09, 1.01942077050548E-08,
             -3.66818642586309E-09, 2.35559919990397E-10],
            [4.37374473226965E-10, -3.38553209393597E-09,
             9.79158913196064E-09, -8.08904230827194E-10],
            [5.23196598342591E-12, 1.56611099697944E-10,
             -6.48231135758056E-10, 5.11543799139839E-09],
        ])
        expected_C1[3:7, 3:7] = np.array([
            [5.11543799139839E-09, -8.22631077061703E-10,
             2.01903867323798E-10, 5.17142615397711E-11],
            [-6.21919618199212E-10, 1.07101602092571E-08,
             -4.14840302775059E-09, 2.50392687785387E-10],
            [1.00287123242728E-10, -3.70945645378442E-09,
             1.12638520863676E-08, -1.54464525857674E-09],
            [9.23432628934985E-11, 7.85800352007450E-11,
             -1.09385115463033E-09, 6.08743898361230E-09],
        ])
        expected_C1[6:10, 6:10] = np.array([
            [6.08743898361230E-09, -1.64851308049761E-09,
             6.06782306621815E-10, -1.05746761543922E-10],
            [-1.09422759219827E-09, 1.19645539829043E-08,
             -5.81528411309578E-09, 9.03235981890081E-10],
            [3.65860287407648E-10, -4.90272124782895E-09,
             1.36680617137076E-08, -3.31208025645711E-09],
            [-3.26639302420449E-11, 6.31810350790411E-10,
             -2.65338385266326E-09, 9.01972954419614E-09],
        ])
        expected_C1[9:13, 9:13] = np.array([
            [9.01972954419614E-09, -3.80348972511169E-09,
             8.07279541550040E-10, -9.49783520130222E-11],
            [-3.16764329970753E-09, 1.46243191938251E-08,
             -6.41953935285775E-09, 7.53113504143303E-10],
            [5.47031544620735E-10, -5.58855303161079E-09,
             1.32309471711060E-08, -2.38315372111170E-09],
            [-2.33053049006883E-11, 5.21605576279264E-10,
             -1.85111134144907E-09, 3.29161662562287E-09],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    ))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_water_flux_vector,
                                    ))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -3.75205305758787E-10,
            -8.54674232552884E-13,
            -3.99653229015751E-13,
            4.47878893044570E-13,
            8.13970460568607E-14,
            -9.03993769200721E-13,
            3.75425894921889E-13,
            2.27093668066429E-13,
            -3.81158049555420E-13,
            -1.17067401873851E-12,
            -7.80442205198545E-13,
            2.53597169472052E-12,
            3.55444964338650E-11,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-20,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000E+00,
            -1.11912142281283E-04,
            -7.28033916928684E-05,
            8.11894375595511E-05,
            -2.01011730134727E-05,
            -8.06157469039566E-05,
            5.09221136049017E-05,
            2.73976044580833E-06,
            -6.92911051357366E-05,
            -1.69295098810003E-04,
            -3.47055681069689E-06,
            1.97203271208842E-04,
            0.00000000000000E+00,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            rtol=1e-10, atol=1e-13,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.61494644164352E-04
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
        self.assertEqual(self.msh._iter, 6)


if __name__ == "__main__":
    unittest.main()
