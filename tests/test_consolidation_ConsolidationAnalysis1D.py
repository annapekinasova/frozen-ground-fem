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
        self.msh.update_consolidation_boundary_conditions(t)
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
        self.msh.update_consolidation_boundary_conditions(t)
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
        self.msh.update_consolidation_boundary_conditions(t)
        self.msh.update_water_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._water_flux_vector,
                                          self.msh._water_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, self.water_flux)
            else:
                self.assertEqual(fx, 0.0)
        self.msh.update_consolidation_boundary_conditions(t)
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
        self.msh.update_consolidation_boundary_conditions(self.msh._t1)
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
        self.msh.update_consolidation_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_water_flux_vector()
        self.msh.update_stiffness_matrix()
        self.msh.update_mass_matrix()
        self.msh.update_weighted_matrices()
        self.msh.calculate_void_ratio_correction()

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
        self.msh.update_consolidation_boundary_conditions(self.msh._t1)
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
        self.msh.update_consolidation_boundary_conditions(self.msh._t1)
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
        self.msh.update_consolidation_boundary_conditions(self.msh._t1)
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


# class TestVoidRatioCorrectionCubicOneStep(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = ConsolidationAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#         )
#         initial_void_ratio_vector = np.array([
#             -2.000000000000000,
#             -9.157452320220460,
#             -10.488299785319000,
#             -7.673205119057850,
#             -3.379831977359920,
#             0.186084957826655,
#             1.975912628300400,
#             2.059737589813890,
#             1.158320034961550,
#             0.100523127786268,
#             -0.548750924584512,
#             -0.609286860003055,
#             -0.205841501790609,
#         ])
#         initial_void_ratio_rate_vector = np.array([
#             -0.02000000000000000,
#             -0.09157452320220460,
#             -0.10488299785319000,
#             -0.07673205119057850,
#             -0.03379831977359920,
#             0.00186084957826655,
#             0.01975912628300400,
#             0.02059737589813890,
#             0.01158320034961550,
#             0.00100523127786268,
#             -0.00548750924584512,
#             -0.00609286860003055,
#             -0.00205841501790609,
#         ])
#         for nd, T0, dTdt0 in zip(self.msh.nodes,
#                                  initial_void_ratio_vector,
#                                  initial_void_ratio_rate_vector,
#                                  ):
#             nd.void_ratio = T0
#             nd.void_ratio_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ConsolidationBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
#             bnd_value=-2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ConsolidationBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 1e-3
#         self.msh.initialize_time_step()
#         self.msh._void_ratio_vector[:] = np.array([
#             -2.000000000000000,
#             -9.157543894743660,
#             -10.488404668316800,
#             -7.673281851109040,
#             -3.379865775679690,
#             0.186086818676234,
#             1.975932387426680,
#             2.059758187189790,
#             1.158331618161900,
#             0.100524133017546,
#             -0.548756412093758,
#             -0.609292952871655,
#             -0.205843560205627,
#         ])
#         self.msh._void_ratio_rate_vector[:] = np.array([
#             0.00000000000000E+00,
#             -9.15745232017429E-02,
#             -1.04882997852940E-01,
#             -7.67320511902980E-02,
#             -3.37983197735703E-02,
#             1.86084957826127E-03,
#             1.97591262829366E-02,
#             2.05973758982125E-02,
#             1.15832003495520E-02,
#             1.00523127785634E-03,
#             -5.48750924589392E-03,
#             -6.09286859998282E-03,
#             -2.05841501790816E-03,
#         ])
#         self.msh.update_consolidation_boundary_conditions(self.msh._t1)
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_water_flux_vector()
#         self.msh.update_stiffness_matrix()
#         self.msh.update_mass_matrix()
#         self.msh.update_weighted_matrices()
#         self.msh.calculate_void_ratio_correction()
#
#     def test_void_ratio_distribution_nodes(self):
#         expected_void_ratio_vector_0 = np.array([
#             -2.000000000000000,
#             -9.157452320220460,
#             -10.488299785319000,
#             -7.673205119057850,
#             -3.379831977359920,
#             0.186084957826655,
#             1.975912628300400,
#             2.059737589813890,
#             1.158320034961550,
#             0.100523127786268,
#             -0.548750924584512,
#             -0.609286860003055,
#             -0.205841501790609,
#         ])
#         expected_void_ratio_vector = np.array([
#             -2.000000000000000,
#             -9.157452320100920,
#             -10.488299785247600,
#             -7.673205119026180,
#             -3.379831977369980,
#             0.186084957800307,
#             1.975912628285290,
#             2.059737589803070,
#             1.158320034961120,
#             0.100523127786127,
#             -0.548750924582867,
#             -0.609286860000696,
#             -0.205841501793126,
#         ])
#         actual_void_ratio_nodes = np.array([
#             nd.void_ratio for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     actual_void_ratio_nodes,
#                                     atol=1e-13, rtol=1e-20))
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     self.msh._void_ratio_vector,
#                                     atol=1e-13, rtol=1e-20))
#         self.assertTrue(np.allclose(expected_void_ratio_vector_0,
#                                     self.msh._void_ratio_vector_0,
#                                     atol=1e-13, rtol=1e-20))
#
#     def test_void_ratio_rate_distribution_nodes(self):
#         expected_void_ratio_rate_vector = np.array([
#             0.00000000000000E+00,
#             1.19538157150600E-07,
#             7.13562542387081E-08,
#             3.16733306249262E-08,
#             -1.00595087815236E-08,
#             -2.63478683315554E-08,
#             -1.51088030975188E-08,
#             -1.08153486166884E-08,
#             -4.32320845789036E-10,
#             -1.40887301824932E-10,
#             1.64468438867971E-09,
#             2.35877983811861E-09,
#             -2.51726417488385E-09,
#         ])
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes,
#                                     atol=1e-12,
#                                     rtol=1e-10,
#                                     ))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector,
#                                     atol=1e-12,
#                                     rtol=1e-10,
#                                     ))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] = np.array([
#             [0.3754127685077310, -0.4791349478258730,
#                 0.1366244789908620, -0.0329022996727200],
#             [-0.4791349496693890, 1.0974938419781100,
#              -0.7555601288938970, 0.1372012365851720],
#             [0.1366244797783630, -0.7555601289255820,
#                 1.1003657455723900, -0.4814300964251700],
#             [-0.0329022998392129, 0.1372012365941040,
#              -0.4814300964261240, 0.7513590334249440],
#         ])
#         expected_H[3:7, 3:7] = np.array([
#             [0.7513590334249440, -0.4704045465033010,
#                 0.1245488443348180, -0.0283721715852281],
#             [-0.4704043870680910, 1.0458043504407100,
#              -0.6891104546310090, 0.1137104912583860],
#             [0.1245485254752460, -0.6891061500305520,
#                 0.9190440219144710, -0.3544863973591650],
#             [-0.0283721538707407, 0.1137101723984170,
#              -0.3544862379296680, 0.5359698450992510],
#         ])
#         expected_H[6:10, 6:10] = np.array([
#             [0.5359698450992510, -0.3407384274106900,
#                 0.0973538364030547, -0.0234370346896243],
#             [-0.3407384274106900, 0.7788306912244320,
#              -0.5354461002167970, 0.0973538364030541],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900],
#             [-0.0234370346896243, 0.0973538364030541,
#              -0.3407384274106900, 0.5959083732909660],
#         ])
#         expected_H[9:13, 9:13] = np.array([
#             [0.5959083732909660, -0.4132037184518070,
#                 0.1111717585476730, -0.0270547876895725],
#             [-0.4134791823328720, 0.9869466717130710,
#              -0.6971967795841350, 0.1237292902039360],
#             [0.1112834973498860, -0.6972094742830320,
#                 1.0421741681109100, -0.4562481911777670],
#             [-0.0270746940945498, 0.1237148240767320,
#              -0.4562007410634830, 0.3595606110813010],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#
#     def test_global_stiffness_matrix_0(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] = np.array([
#             [0.3754127757099620, -0.4791349563299290,
#                 0.1366244808437270, -0.0329023002237587],
#             [-0.4791349623925930, 1.0974938702568300,
#              -0.7555601482973810, 0.1372012404331490],
#             [0.1366244834335360, -0.7555601484015830,
#                 1.1003657741573600, -0.4814301091893070],
#             [-0.0329023007712942, 0.1372012404625230,
#              -0.4814301091924440, 0.7513590624829260],
#         ])
#         expected_H[3:7, 3:7] = np.array([
#             [0.7513590624829260, -0.4704047663661540,
#                 0.1245490529816890, -0.0283721795972456],
#             [-0.4704044695643160, 1.0458062449151100,
#              -0.6891123373729850, 0.1137105620221960],
#             [0.1245484594009230, -0.6891043240405850,
#                 0.9190421952958320, -0.3544863306561690],
#             [-0.0283721466203953, 0.1137099684405900,
#              -0.3544860338663970, 0.5359698377434610],
#         ])
#         expected_H[6:10, 6:10] = np.array([
#             [0.5359698377434610, -0.3407384274106900,
#                 0.0973538364030547, -0.0234370346896243],
#             [-0.3407384274106900, 0.7788306912244320,
#              -0.5354461002167970, 0.0973538364030541],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900],
#             [-0.0234370346896243, 0.0973538364030541,
#              -0.3407384274106900, 0.5958500379379190],
#         ])
#         expected_H[9:13, 9:13] = np.array([
#             [0.5958500379379190, -0.4131199413318780,
#                 0.1111410332402880, -0.0270495041490680],
#             [-0.4135246144137550, 0.9869955536278930,
#              -0.6972079946668100, 0.1237370554526720],
#             [0.1113006435889920, -0.6972323578968160,
#                 1.0422001879261100, -0.4562684736182830],
#             [-0.0270753013289352, 0.1237052894950540,
#              -0.4561656685786790, 0.3595356804125600],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix_0,
#         ))
#
#     def test_global_stiffness_matrix_weighted(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] = np.array([
#             [0.3754127757099620, -0.4791349563299290,
#                 0.1366244808437270, -0.0329023002237587],
#             [-0.4791349623925930, 1.0974938702568300,
#              -0.7555601482973810, 0.1372012404331490],
#             [0.1366244834335360, -0.7555601484015830,
#                 1.1003657741573600, -0.4814301091893070],
#             [-0.0329023007712942, 0.1372012404625230,
#              -0.4814301091924440, 0.7513590624829260],
#         ])
#         expected_H[3:7, 3:7] = np.array([
#             [0.7513590624829260, -0.4704047663661540,
#                 0.1245490529816890, -0.0283721795972456],
#             [-0.4704044695643160, 1.0458062449151100,
#              -0.6891123373729850, 0.1137105620221960],
#             [0.1245484594009230, -0.6891043240405850,
#                 0.9190421952958320, -0.3544863306561690],
#             [-0.0283721466203953, 0.1137099684405900,
#              -0.3544860338663970, 0.5359698377434610],
#         ])
#         expected_H[6:10, 6:10] = np.array([
#             [0.5359698377434610, -0.3407384274106900,
#                 0.0973538364030547, -0.0234370346896243],
#             [-0.3407384274106900, 0.7788306912244320,
#              -0.5354461002167970, 0.0973538364030541],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900],
#             [-0.0234370346896243, 0.0973538364030541,
#              -0.3407384274106900, 0.5958500379379190],
#         ])
#         expected_H[9:13, 9:13] = np.array([
#             [0.5958500379379190, -0.4131199413318780,
#                 0.1111410332402880, -0.0270495041490680],
#             [-0.4135246144137550, 0.9869955536278930,
#              -0.6972079946668100, 0.1237370554526720],
#             [0.1113006435889920, -0.6972323578968160,
#                 1.0422001879261100, -0.4562684736182830],
#             [-0.0270753013289352, 0.1237052894950540,
#              -0.4561656685786790, 0.3595356804125600],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._weighted_stiffness_matrix,
#             rtol=1e-10, atol=1e-15,
#         ))
#
#     def test_global_mass_matrix_0(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [4.73637171623350E+06, 3.48911975196258E+06,
#              -1.29625396875553E+06, 6.28539963771457E+05],
#             [3.48911975196258E+06, 2.04661850623021E+07,
#              -2.68983544335460E+06, -1.08386996099026E+06],
#             [-1.29625396875553E+06, -2.68983544335460E+06,
#                 1.99779575443545E+07, 3.06435173643206E+06],
#             [6.28539963771457E+05, -1.08386996099026E+06,
#                 3.06435173643206E+06, 4.00238174498494E+06],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [4.16942926772367E+06, 2.79778776603154E+06,
#              -1.54215963369236E+06, 7.18714827570726E+05],
#             [2.79778776603154E+06, 2.63422846762999E+07,
#                 8.59594504059493E+05, -1.83532284105249E+06],
#             [-1.54215963369236E+06, 8.59594504059493E+05,
#                 2.79087377467474E+07, 3.38411418075181E+06],
#             [7.18714827570726E+05, -1.83532284105249E+06,
#                 3.38411418075181E+06, 4.88507845309510E+06],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329808E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442682E+05],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857144E+06, -1.36311507936508E+06],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329808E+06],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [1.04462849763812E+09, 5.23841479165073E+08,
#              -2.37985788657434E+08, 5.50357660694582E+07],
#             [5.23841479165073E+08, 3.84440750802666E+08,
#              -1.43609705550615E+08, 1.42597922699745E+07],
#             [-2.37985788657434E+08, -1.43609705550615E+08,
#                 1.69013732188852E+08, 1.93503173102562E+07],
#             [5.50357660694582E+07, 1.42597922699745E+07,
#                 1.93503173102562E+07, 5.14006065936711E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix_0,
#             rtol=1e-10,
#         ))
#
#     def test_global_mass_matrix(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] = np.array([
#             [4.73637171624879E+06, 3.48911975197287E+06,
#              -1.29625396875982E+06, 6.28539963772493E+05],
#             [3.48911975197287E+06, 2.04661850623337E+07,
#              -2.68983544335970E+06, -1.08386996098978E+06],
#             [-1.29625396875982E+06, -2.68983544335970E+06,
#                 1.99779575443663E+07, 3.06435173643278E+06],
#             [6.28539963772493E+05, -1.08386996098978E+06,
#                 3.06435173643278E+06, 8.17181101271068E+06],
#         ])
#         expected_C[3:7, 3:7] = np.array([
#             [8.17181101270964E+06, 2.79778776603809E+06,
#              -1.54215963368644E+06, 7.18714827570090E+05],
#             [2.79778776603809E+06, 2.63422846762452E+07,
#                 8.59594504004377E+05, -1.83532284104636E+06],
#             [-1.54215963368644E+06, 8.59594504004377E+05,
#                 2.79087377466924E+07, 3.38411418075791E+06],
#             [7.18714827570090E+05, -1.83532284104636E+06,
#                 3.38411418075791E+06, 9.73170984639250E+06],
#         ])
#         expected_C[6:10, 6:10] = np.array([
#             [9.73170984639250E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442682E+05],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857144E+06, -1.36311507936508E+06],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 1.04947512903143E+09],
#         ])
#         expected_C[9:13, 9:13] = np.array([
#             [1.04947512903143E+09, 5.23841479165093E+08,
#              -2.37985788657442E+08, 5.50357660694532E+07],
#             [5.23841479165093E+08, 3.84440750802889E+08,
#              -1.43609705550628E+08, 1.42597922700073E+07],
#             [-2.37985788657442E+08, -1.43609705550628E+08,
#                 1.69013732189027E+08, 1.93503173101947E+07],
#             [5.50357660694532E+07, 1.42597922700073E+07,
#                 1.93503173101947E+07, 5.14006065935181E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix,
#             rtol=1e-10,
#         ))
#
#     def test_global_mass_matrix_weighted(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] = np.array([
#             [4.73636734264201E+06, 3.48911679572160E+06,
#              -1.29625283410763E+06, 6.28539549337238E+05],
#             [3.48911679572160E+06, 2.04661736956678E+07,
#              -2.68983338251903E+06, -1.08386957415677E+06],
#             [-1.29625283410763E+06, -2.68983338251903E+06,
#                 1.99779489357607E+07, 3.06435027581987E+06],
#             [6.28539549337238E+05, -1.08386957415677E+06,
#                 3.06435027581987E+06, 8.17180553044345E+06],
#         ])
#         expected_C[3:7, 3:7] = np.array([
#             [8.17180553044345E+06, 2.79778798757569E+06,
#              -1.54215436314838E+06, 7.18714133746307E+05],
#             [2.79778798757569E+06, 2.63422295641292E+07,
#                 8.59562216163159E+05, -1.83531960807053E+06],
#             [-1.54215436314838E+06, 8.59562216163159E+05,
#                 2.79086998878311E+07, 3.38411847742000E+06],
#             [7.18714133746307E+05, -1.83531960807053E+06,
#                 3.38411847742000E+06, 9.73170935158256E+06],
#         ])
#         expected_C[6:10, 6:10] = np.array([
#             [9.73170935158256E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442682E+05],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857144E+06, -1.36311507936508E+06],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 1.04946889343989E+09],
#         ])
#         expected_C[9:13, 9:13] = np.array([
#             [1.04946889343989E+09, 5.23838342703922E+08,
#              -2.37984368299458E+08, 5.50354343989630E+07],
#             [5.23838342703922E+08, 3.84438373169631E+08,
#              -1.43608818315020E+08, 1.42597165566276E+07],
#             [-2.37984368299458E+08, -1.43608818315020E+08,
#                 1.69012674106786E+08, 1.93501729917517E+07],
#             [5.50354343989630E+07, 1.42597165566276E+07,
#                 1.93501729917517E+07, 5.14002478974014E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._weighted_mass_matrix,
#             rtol=1e-10,
#         ))
#
#     def test_global_coef_matrix_0(self):
#         expected_C0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C0[0:4, 0:4] = np.array([
#             [4.73636734245431E+09, 3.48911679596117E+09,
#              -1.29625283417594E+09, 6.28539549353689E+08],
#             [3.48911679596117E+09, 2.04661736951191E+10,
#              -2.68983338214125E+09, -1.08386957422537E+09],
#             [-1.29625283417594E+09, -2.68983338214125E+09,
#                 1.99779489352105E+10, 3.06435027606058E+09],
#             [6.28539549353689E+08, -1.08386957422537E+09,
#                 3.06435027606058E+09, 8.17180553006777E+09],
#         ])
#         expected_C0[3:7, 3:7] = np.array([
#             [8.17180553006777E+09, 2.79778798781089E+09,
#              -1.54215436321065E+09, 7.18714133760493E+08],
#             [2.79778798781089E+09, 2.63422295636063E+10,
#                 8.59562216507716E+08, -1.83531960812739E+09],
#             [-1.54215436321065E+09, 8.59562216507711E+08,
#                 2.79086998873716E+10, 3.38411847759724E+09],
#             [7.18714133760493E+08, -1.83531960812739E+09,
#                 3.38411847759724E+09, 9.73170935131458E+09],
#         ])
#         expected_C0[6:10, 6:10] = np.array([
#             [9.73170935131458E+09, 3.74856646842434E+09,
#              -1.36311507941376E+09, 7.19421847454401E+08],
#             [3.74856646842434E+09, 2.45360714281820E+10,
#              -3.06700892830372E+09, -1.36311507941376E+09],
#             [-1.36311507941376E+09, -3.06700892830372E+09,
#                 2.45360714281820E+10, 3.74856646842434E+09],
#             [7.19421847454401E+08, -1.36311507941376E+09,
#                 3.74856646842434E+09, 1.04946889343959E+12],
#         ])
#         expected_C0[9:13, 9:13] = np.array([
#             [1.04946889343959E+12, 5.23838342704129E+11,
#              -2.37984368299513E+11, 5.50354343989765E+10],
#             [5.23838342704129E+11, 3.84438373169137E+11,
#              -1.43608818314672E+11, 1.42597165565657E+10],
#             [-2.37984368299513E+11, -1.43608818314672E+11,
#                 1.69012674106265E+11, 1.93501729919799E+10],
#             [5.50354343989765E+10, 1.42597165565657E+10,
#                 1.93501729919798E+10, 5.14002478972216E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C0, self.msh._coef_matrix_0,
#             rtol=1e-10,
#         ))
#
#     def test_global_coef_matrix_1(self):
#         expected_C1 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C1[0:4, 0:4] = np.array([
#             [4.73636734282972E+09, 3.48911679548203E+09,
#              -1.29625283403932E+09, 6.28539549320787E+08],
#             [3.48911679548203E+09, 2.04661736962166E+10,
#              -2.68983338289681E+09, -1.08386957408817E+09],
#             [-1.29625283403932E+09, -2.68983338289681E+09,
#                 1.99779489363109E+10, 3.06435027557915E+09],
#             [6.28539549320787E+08, -1.08386957408817E+09,
#                 3.06435027557915E+09, 8.17180553081913E+09],
#         ])
#         expected_C1[3:7, 3:7] = np.array([
#             [8.17180553081913E+09, 2.79778798734049E+09,
#              -1.54215436308610E+09, 7.18714133732121E+08],
#             [2.79778798734049E+09, 2.63422295646521E+10,
#                 8.59562215818603E+08, -1.83531960801368E+09],
#             [-1.54215436308610E+09, 8.59562215818607E+08,
#                 2.79086998882907E+10, 3.38411847724276E+09],
#             [7.18714133732121E+08, -1.83531960801368E+09,
#                 3.38411847724276E+09, 9.73170935185054E+09],
#         ])
#         expected_C1[6:10, 6:10] = np.array([
#             [9.73170935185054E+09, 3.74856646808360E+09,
#              -1.36311507931641E+09, 7.19421847430964E+08],
#             [3.74856646808360E+09, 2.45360714289608E+10,
#              -3.06700892883917E+09, -1.36311507931641E+09],
#             [-1.36311507931641E+09, -3.06700892883917E+09,
#                 2.45360714289608E+10, 3.74856646808360E+09],
#             [7.19421847430964E+08, -1.36311507931641E+09,
#                 3.74856646808360E+09, 1.04946889344019E+12],
#         ])
#         expected_C1[9:13, 9:13] = np.array([
#             [1.04946889344019E+12, 5.23838342703716E+11,
#              -2.37984368299402E+11, 5.50354343989495E+10],
#             [5.23838342703716E+11, 3.84438373170124E+11,
#              -1.43608818315369E+11, 1.42597165566894E+10],
#             [-2.37984368299402E+11, -1.43608818315369E+11,
#                 1.69012674107307E+11, 1.93501729915236E+10],
#             [5.50354343989495E+10, 1.42597165566894E+10,
#                 1.93501729915236E+10, 5.14002478975812E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C1, self.msh._coef_matrix_1,
#             rtol=1e-10,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
#         expected_flux_vector_0[-1] = -2.61109074784318 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector_0,
#                                     self.msh._water_flux_vector_0,
#                                     rtol=1e-15, atol=1e-16))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.61109074784359 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector,
#                                     rtol=1e-15, atol=1e-16))
#
#     def test_global_flux_vector_weighted(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -0.5 * (2.61109074784318
#                                            + 2.61109159734326) * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._weighted_water_flux_vector,
#                                     rtol=1e-15, atol=1e-16))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             2.31785996337891E+05,
#             1.50889699377441E+06,
#             2.08416204809570E+06,
#             9.32412240631104E+05,
#             1.13966773396301E+06,
#             -2.08082869632721E+05,
#             -2.67614536674500E+05,
#             -5.42551380218506E+05,
#             -1.97868066013336E+05,
#             4.53328138000488E+05,
#             7.37393379348755E+05,
#             5.20777461418152E+05,
#             2.46628021972836E+05,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_water_flux_vector,
#             rtol=1e-10,
#         ))
#
#     def test_void_ratio_increment_vector(self):
#         expected_dT = np.array([
#             0.00000000000000E+00,
#             9.15746427389685E-05,
#             1.04883069156525E-04,
#             7.67320828637213E-05,
#             3.37983097104436E-05,
#             -1.86087592687940E-06,
#             -1.97591413889387E-05,
#             -2.05973867154074E-05,
#             -1.15832007823766E-05,
#             -1.00523141887636E-06,
#             5.48751089066735E-06,
#             6.09287095873518E-06,
#             2.05841250072898E-06,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_void_ratio_vector,
#             rtol=1e-10, atol=1e-20,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 9.92791192400449E-06
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#         self.assertEqual(self.msh._iter, 1)
#
#
# class TestIterativeVoidRatioCorrectionCubic(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = ConsolidationAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#         )
#         initial_void_ratio_vector = np.array([
#             -2.000000000000000,
#             -9.157452320220460,
#             -10.488299785319000,
#             -7.673205119057850,
#             -3.379831977359920,
#             0.186084957826655,
#             1.975912628300400,
#             2.059737589813890,
#             1.158320034961550,
#             0.100523127786268,
#             -0.548750924584512,
#             -0.609286860003055,
#             -0.205841501790609,
#         ])
#         initial_void_ratio_rate_vector = np.array([
#             -0.02000000000000000,
#             -0.09157452320220460,
#             -0.10488299785319000,
#             -0.07673205119057850,
#             -0.03379831977359920,
#             0.00186084957826655,
#             0.01975912628300400,
#             0.02059737589813890,
#             0.01158320034961550,
#             0.00100523127786268,
#             -0.00548750924584512,
#             -0.00609286860003055,
#             -0.00205841501790609,
#         ])
#         for nd, T0, dTdt0 in zip(self.msh.nodes,
#                                  initial_void_ratio_vector,
#                                  initial_void_ratio_rate_vector,
#                                  ):
#             nd.void_ratio = T0
#             nd.void_ratio_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ConsolidationBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
#             bnd_value=-2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ConsolidationBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 1e-3
#         self.msh.initialize_time_step()
#         self.msh._void_ratio_vector[:] = np.array([
#             -2.000000000000000,
#             -9.157543894743660,
#             -10.488404668316800,
#             -7.673281851109040,
#             -3.379865775679690,
#             0.186086818676234,
#             1.975932387426680,
#             2.059758187189790,
#             1.158331618161900,
#             0.100524133017546,
#             -0.548756412093758,
#             -0.609292952871655,
#             -0.205843560205627,
#         ])
#         self.msh._void_ratio_rate_vector[:] = np.array([
#             0.00000000000000E+00,
#             -9.15745232017429E-02,
#             -1.04882997852940E-01,
#             -7.67320511902980E-02,
#             -3.37983197735703E-02,
#             1.86084957826127E-03,
#             1.97591262829366E-02,
#             2.05973758982125E-02,
#             1.15832003495520E-02,
#             1.00523127785634E-03,
#             -5.48750924589392E-03,
#             -6.09286859998282E-03,
#             -2.05841501790816E-03,
#         ])
#         self.msh.update_consolidation_boundary_conditions(self.msh._t1)
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_water_flux_vector()
#         self.msh.update_stiffness_matrix()
#         self.msh.update_mass_matrix()
#         self.msh.update_weighted_matrices()
#         self.msh.iterative_correction_step()
#
#     def test_void_ratio_distribution_nodes(self):
#         expected_void_ratio_rate_vector = np.array([
#             0.00000000000000E+00,
#             1.19538157150600E-07,
#             7.13562542387081E-08,
#             3.16733306249262E-08,
#             -1.00595087815236E-08,
#             -2.63478683315554E-08,
#             -1.51088030975188E-08,
#             -1.08153486166884E-08,
#             -4.32320845789036E-10,
#             -1.40887301824932E-10,
#             1.64468438867971E-09,
#             2.35877983811861E-09,
#             -2.51726417488385E-09,
#         ])
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes,
#                                     atol=1e-12,
#                                     rtol=1e-10,
#                                     ))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector,
#                                     atol=1e-12,
#                                     rtol=1e-10,
#                                     ))
#
#     def test_void_ratio_rate_distribution_nodes(self):
#         expected_void_ratio_rate_vector = np.array([
#             0.00000000000000E+00,
#             1.19538157150600E-07,
#             7.13562542387081E-08,
#             3.16733306249262E-08,
#             -1.00595087815236E-08,
#             -2.63478683315554E-08,
#             -1.51088030975188E-08,
#             -1.08153486166884E-08,
#             -4.32320845789036E-10,
#             -1.40887301824932E-10,
#             1.64468438867971E-09,
#             2.35877983811861E-09,
#             -2.51726417488385E-09,
#         ])
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes,
#                                     atol=1e-12,
#                                     rtol=1e-10,
#                                     ))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector,
#                                     atol=1e-12,
#                                     rtol=1e-10,
#                                     ))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] = np.array([
#             [0.3754127685077310, -0.4791349478258730,
#                 0.1366244789908620, -0.0329022996727200],
#             [-0.4791349496693890, 1.0974938419781100,
#              -0.7555601288938970, 0.1372012365851720],
#             [0.1366244797783630, -0.7555601289255820,
#                 1.1003657455723900, -0.4814300964251700],
#             [-0.0329022998392129, 0.1372012365941040,
#              -0.4814300964261240, 0.7513590334249440],
#         ])
#         expected_H[3:7, 3:7] = np.array([
#             [0.7513590334249440, -0.4704045465033010,
#                 0.1245488443348180, -0.0283721715852281],
#             [-0.4704043870680910, 1.0458043504407100,
#              -0.6891104546310090, 0.1137104912583860],
#             [0.1245485254752460, -0.6891061500305520,
#                 0.9190440219144710, -0.3544863973591650],
#             [-0.0283721538707407, 0.1137101723984170,
#              -0.3544862379296680, 0.5359698450992510],
#         ])
#         expected_H[6:10, 6:10] = np.array([
#             [0.5359698450992510, -0.3407384274106900,
#                 0.0973538364030547, -0.0234370346896243],
#             [-0.3407384274106900, 0.7788306912244320,
#              -0.5354461002167970, 0.0973538364030541],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900],
#             [-0.0234370346896243, 0.0973538364030541,
#              -0.3407384274106900, 0.5959083732909660],
#         ])
#         expected_H[9:13, 9:13] = np.array([
#             [0.5959083732909660, -0.4132037184518070,
#                 0.1111717585476730, -0.0270547876895725],
#             [-0.4134791823328720, 0.9869466717130710,
#              -0.6971967795841350, 0.1237292902039360],
#             [0.1112834973498860, -0.6972094742830320,
#                 1.0421741681109100, -0.4562481911777670],
#             [-0.0270746940945498, 0.1237148240767320,
#              -0.4562007410634830, 0.3595606110813010],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#
#     def test_global_stiffness_matrix_0(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] = np.array([
#             [0.3754127757099620, -0.4791349563299290,
#                 0.1366244808437270, -0.0329023002237587],
#             [-0.4791349623925930, 1.0974938702568300,
#              -0.7555601482973810, 0.1372012404331490],
#             [0.1366244834335360, -0.7555601484015830,
#                 1.1003657741573600, -0.4814301091893070],
#             [-0.0329023007712942, 0.1372012404625230,
#              -0.4814301091924440, 0.7513590624829260],
#         ])
#         expected_H[3:7, 3:7] = np.array([
#             [0.7513590624829260, -0.4704047663661540,
#                 0.1245490529816890, -0.0283721795972456],
#             [-0.4704044695643160, 1.0458062449151100,
#              -0.6891123373729850, 0.1137105620221960],
#             [0.1245484594009230, -0.6891043240405850,
#                 0.9190421952958320, -0.3544863306561690],
#             [-0.0283721466203953, 0.1137099684405900,
#              -0.3544860338663970, 0.5359698377434610],
#         ])
#         expected_H[6:10, 6:10] = np.array([
#             [0.5359698377434610, -0.3407384274106900,
#                 0.0973538364030547, -0.0234370346896243],
#             [-0.3407384274106900, 0.7788306912244320,
#              -0.5354461002167970, 0.0973538364030541],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900],
#             [-0.0234370346896243, 0.0973538364030541,
#              -0.3407384274106900, 0.5958500379379190],
#         ])
#         expected_H[9:13, 9:13] = np.array([
#             [0.5958500379379190, -0.4131199413318780,
#                 0.1111410332402880, -0.0270495041490680],
#             [-0.4135246144137550, 0.9869955536278930,
#              -0.6972079946668100, 0.1237370554526720],
#             [0.1113006435889920, -0.6972323578968160,
#                 1.0422001879261100, -0.4562684736182830],
#             [-0.0270753013289352, 0.1237052894950540,
#              -0.4561656685786790, 0.3595356804125600],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix_0,
#         ))
#
#     def test_global_stiffness_matrix_weighted(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] = np.array([
#             [0.3754127757099620, -0.4791349563299290,
#                 0.1366244808437270, -0.0329023002237587],
#             [-0.4791349623925930, 1.0974938702568300,
#              -0.7555601482973810, 0.1372012404331490],
#             [0.1366244834335360, -0.7555601484015830,
#                 1.1003657741573600, -0.4814301091893070],
#             [-0.0329023007712942, 0.1372012404625230,
#              -0.4814301091924440, 0.7513590624829260],
#         ])
#         expected_H[3:7, 3:7] = np.array([
#             [0.7513590624829260, -0.4704047663661540,
#                 0.1245490529816890, -0.0283721795972456],
#             [-0.4704044695643160, 1.0458062449151100,
#              -0.6891123373729850, 0.1137105620221960],
#             [0.1245484594009230, -0.6891043240405850,
#                 0.9190421952958320, -0.3544863306561690],
#             [-0.0283721466203953, 0.1137099684405900,
#              -0.3544860338663970, 0.5359698377434610],
#         ])
#         expected_H[6:10, 6:10] = np.array([
#             [0.5359698377434610, -0.3407384274106900,
#                 0.0973538364030547, -0.0234370346896243],
#             [-0.3407384274106900, 0.7788306912244320,
#              -0.5354461002167970, 0.0973538364030541],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900],
#             [-0.0234370346896243, 0.0973538364030541,
#              -0.3407384274106900, 0.5958500379379190],
#         ])
#         expected_H[9:13, 9:13] = np.array([
#             [0.5958500379379190, -0.4131199413318780,
#                 0.1111410332402880, -0.0270495041490680],
#             [-0.4135246144137550, 0.9869955536278930,
#              -0.6972079946668100, 0.1237370554526720],
#             [0.1113006435889920, -0.6972323578968160,
#                 1.0422001879261100, -0.4562684736182830],
#             [-0.0270753013289352, 0.1237052894950540,
#              -0.4561656685786790, 0.3595356804125600],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._weighted_stiffness_matrix,
#             rtol=1e-10, atol=1e-15,
#         ))
#
#     def test_global_mass_matrix_0(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [4.73637171623350E+06, 3.48911975196258E+06,
#              -1.29625396875553E+06, 6.28539963771457E+05],
#             [3.48911975196258E+06, 2.04661850623021E+07,
#              -2.68983544335460E+06, -1.08386996099026E+06],
#             [-1.29625396875553E+06, -2.68983544335460E+06,
#                 1.99779575443545E+07, 3.06435173643206E+06],
#             [6.28539963771457E+05, -1.08386996099026E+06,
#                 3.06435173643206E+06, 4.00238174498494E+06],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [4.16942926772367E+06, 2.79778776603154E+06,
#              -1.54215963369236E+06, 7.18714827570726E+05],
#             [2.79778776603154E+06, 2.63422846762999E+07,
#                 8.59594504059493E+05, -1.83532284105249E+06],
#             [-1.54215963369236E+06, 8.59594504059493E+05,
#                 2.79087377467474E+07, 3.38411418075181E+06],
#             [7.18714827570726E+05, -1.83532284105249E+06,
#                 3.38411418075181E+06, 4.88507845309510E+06],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329808E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442682E+05],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857144E+06, -1.36311507936508E+06],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329808E+06],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [1.04462849763812E+09, 5.23841479165073E+08,
#              -2.37985788657434E+08, 5.50357660694582E+07],
#             [5.23841479165073E+08, 3.84440750802666E+08,
#              -1.43609705550615E+08, 1.42597922699745E+07],
#             [-2.37985788657434E+08, -1.43609705550615E+08,
#                 1.69013732188852E+08, 1.93503173102562E+07],
#             [5.50357660694582E+07, 1.42597922699745E+07,
#                 1.93503173102562E+07, 5.14006065936711E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix_0,
#             rtol=1e-10,
#         ))
#
#     def test_global_mass_matrix(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] = np.array([
#             [4.73637171624879E+06, 3.48911975197287E+06,
#              -1.29625396875982E+06, 6.28539963772493E+05],
#             [3.48911975197287E+06, 2.04661850623337E+07,
#              -2.68983544335970E+06, -1.08386996098978E+06],
#             [-1.29625396875982E+06, -2.68983544335970E+06,
#                 1.99779575443663E+07, 3.06435173643278E+06],
#             [6.28539963772493E+05, -1.08386996098978E+06,
#                 3.06435173643278E+06, 8.17181101271068E+06],
#         ])
#         expected_C[3:7, 3:7] = np.array([
#             [8.17181101270964E+06, 2.79778776603809E+06,
#              -1.54215963368644E+06, 7.18714827570090E+05],
#             [2.79778776603809E+06, 2.63422846762452E+07,
#                 8.59594504004377E+05, -1.83532284104636E+06],
#             [-1.54215963368644E+06, 8.59594504004377E+05,
#                 2.79087377466924E+07, 3.38411418075791E+06],
#             [7.18714827570090E+05, -1.83532284104636E+06,
#                 3.38411418075791E+06, 9.73170984639250E+06],
#         ])
#         expected_C[6:10, 6:10] = np.array([
#             [9.73170984639250E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442682E+05],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857144E+06, -1.36311507936508E+06],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 1.04947512903143E+09],
#         ])
#         expected_C[9:13, 9:13] = np.array([
#             [1.04947512903143E+09, 5.23841479165093E+08,
#              -2.37985788657442E+08, 5.50357660694532E+07],
#             [5.23841479165093E+08, 3.84440750802889E+08,
#              -1.43609705550628E+08, 1.42597922700073E+07],
#             [-2.37985788657442E+08, -1.43609705550628E+08,
#                 1.69013732189027E+08, 1.93503173101947E+07],
#             [5.50357660694532E+07, 1.42597922700073E+07,
#                 1.93503173101947E+07, 5.14006065935181E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix,
#             rtol=1e-10,
#         ))
#
#     def test_global_mass_matrix_weighted(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] = np.array([
#             [4.73636734264201E+06, 3.48911679572160E+06,
#              -1.29625283410763E+06, 6.28539549337238E+05],
#             [3.48911679572160E+06, 2.04661736956678E+07,
#              -2.68983338251903E+06, -1.08386957415677E+06],
#             [-1.29625283410763E+06, -2.68983338251903E+06,
#                 1.99779489357607E+07, 3.06435027581987E+06],
#             [6.28539549337238E+05, -1.08386957415677E+06,
#                 3.06435027581987E+06, 8.17180553044345E+06],
#         ])
#         expected_C[3:7, 3:7] = np.array([
#             [8.17180553044345E+06, 2.79778798757569E+06,
#              -1.54215436314838E+06, 7.18714133746307E+05],
#             [2.79778798757569E+06, 2.63422295641292E+07,
#                 8.59562216163159E+05, -1.83531960807053E+06],
#             [-1.54215436314838E+06, 8.59562216163159E+05,
#                 2.79086998878311E+07, 3.38411847742000E+06],
#             [7.18714133746307E+05, -1.83531960807053E+06,
#                 3.38411847742000E+06, 9.73170935158256E+06],
#         ])
#         expected_C[6:10, 6:10] = np.array([
#             [9.73170935158256E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442682E+05],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857144E+06, -1.36311507936508E+06],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 1.04946889343989E+09],
#         ])
#         expected_C[9:13, 9:13] = np.array([
#             [1.04946889343989E+09, 5.23838342703922E+08,
#              -2.37984368299458E+08, 5.50354343989630E+07],
#             [5.23838342703922E+08, 3.84438373169631E+08,
#              -1.43608818315020E+08, 1.42597165566276E+07],
#             [-2.37984368299458E+08, -1.43608818315020E+08,
#                 1.69012674106786E+08, 1.93501729917517E+07],
#             [5.50354343989630E+07, 1.42597165566276E+07,
#                 1.93501729917517E+07, 5.14002478974014E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._weighted_mass_matrix,
#             rtol=1e-10,
#         ))
#
#     def test_global_coef_matrix_0(self):
#         expected_C0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C0[0:4, 0:4] = np.array([
#             [4.73636734245431E+09, 3.48911679596117E+09,
#              -1.29625283417594E+09, 6.28539549353689E+08],
#             [3.48911679596117E+09, 2.04661736951191E+10,
#              -2.68983338214125E+09, -1.08386957422537E+09],
#             [-1.29625283417594E+09, -2.68983338214125E+09,
#                 1.99779489352105E+10, 3.06435027606058E+09],
#             [6.28539549353689E+08, -1.08386957422537E+09,
#                 3.06435027606058E+09, 8.17180553006777E+09],
#         ])
#         expected_C0[3:7, 3:7] = np.array([
#             [8.17180553006777E+09, 2.79778798781089E+09,
#              -1.54215436321065E+09, 7.18714133760493E+08],
#             [2.79778798781089E+09, 2.63422295636063E+10,
#                 8.59562216507716E+08, -1.83531960812739E+09],
#             [-1.54215436321065E+09, 8.59562216507711E+08,
#                 2.79086998873716E+10, 3.38411847759724E+09],
#             [7.18714133760493E+08, -1.83531960812739E+09,
#                 3.38411847759724E+09, 9.73170935131458E+09],
#         ])
#         expected_C0[6:10, 6:10] = np.array([
#             [9.73170935131458E+09, 3.74856646842434E+09,
#              -1.36311507941376E+09, 7.19421847454401E+08],
#             [3.74856646842434E+09, 2.45360714281820E+10,
#              -3.06700892830372E+09, -1.36311507941376E+09],
#             [-1.36311507941376E+09, -3.06700892830372E+09,
#                 2.45360714281820E+10, 3.74856646842434E+09],
#             [7.19421847454401E+08, -1.36311507941376E+09,
#                 3.74856646842434E+09, 1.04946889343959E+12],
#         ])
#         expected_C0[9:13, 9:13] = np.array([
#             [1.04946889343959E+12, 5.23838342704129E+11,
#              -2.37984368299513E+11, 5.50354343989765E+10],
#             [5.23838342704129E+11, 3.84438373169137E+11,
#              -1.43608818314672E+11, 1.42597165565657E+10],
#             [-2.37984368299513E+11, -1.43608818314672E+11,
#                 1.69012674106265E+11, 1.93501729919799E+10],
#             [5.50354343989765E+10, 1.42597165565657E+10,
#                 1.93501729919798E+10, 5.14002478972216E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C0, self.msh._coef_matrix_0,
#             rtol=1e-10,
#         ))
#
#     def test_global_coef_matrix_1(self):
#         expected_C1 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C1[0:4, 0:4] = np.array([
#             [4.73636734282972E+09, 3.48911679548203E+09,
#              -1.29625283403932E+09, 6.28539549320787E+08],
#             [3.48911679548203E+09, 2.04661736962166E+10,
#              -2.68983338289681E+09, -1.08386957408817E+09],
#             [-1.29625283403932E+09, -2.68983338289681E+09,
#                 1.99779489363109E+10, 3.06435027557915E+09],
#             [6.28539549320787E+08, -1.08386957408817E+09,
#                 3.06435027557915E+09, 8.17180553081913E+09],
#         ])
#         expected_C1[3:7, 3:7] = np.array([
#             [8.17180553081913E+09, 2.79778798734049E+09,
#              -1.54215436308610E+09, 7.18714133732121E+08],
#             [2.79778798734049E+09, 2.63422295646521E+10,
#                 8.59562215818603E+08, -1.83531960801368E+09],
#             [-1.54215436308610E+09, 8.59562215818607E+08,
#                 2.79086998882907E+10, 3.38411847724276E+09],
#             [7.18714133732121E+08, -1.83531960801368E+09,
#                 3.38411847724276E+09, 9.73170935185054E+09],
#         ])
#         expected_C1[6:10, 6:10] = np.array([
#             [9.73170935185054E+09, 3.74856646808360E+09,
#              -1.36311507931641E+09, 7.19421847430964E+08],
#             [3.74856646808360E+09, 2.45360714289608E+10,
#              -3.06700892883917E+09, -1.36311507931641E+09],
#             [-1.36311507931641E+09, -3.06700892883917E+09,
#                 2.45360714289608E+10, 3.74856646808360E+09],
#             [7.19421847430964E+08, -1.36311507931641E+09,
#                 3.74856646808360E+09, 1.04946889344019E+12],
#         ])
#         expected_C1[9:13, 9:13] = np.array([
#             [1.04946889344019E+12, 5.23838342703716E+11,
#              -2.37984368299402E+11, 5.50354343989495E+10],
#             [5.23838342703716E+11, 3.84438373170124E+11,
#              -1.43608818315369E+11, 1.42597165566894E+10],
#             [-2.37984368299402E+11, -1.43608818315369E+11,
#                 1.69012674107307E+11, 1.93501729915236E+10],
#             [5.50354343989495E+10, 1.42597165566894E+10,
#                 1.93501729915236E+10, 5.14002478975812E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C1, self.msh._coef_matrix_1,
#             rtol=1e-10,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
#         expected_flux_vector_0[-1] = -2.61109074784318 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector_0,
#                                     self.msh._water_flux_vector_0,
#                                     rtol=1e-15, atol=1e-16))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.61109074784359 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector,
#                                     rtol=1e-15, atol=1e-16))
#
#     def test_global_flux_vector_weighted(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -0.5 * (2.61109074784318
#                                            + 2.61109159734326) * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._weighted_water_flux_vector,
#                                     rtol=1e-15, atol=1e-16))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             2.31785996337891E+05,
#             1.50889699377441E+06,
#             2.08416204809570E+06,
#             9.32412240631104E+05,
#             1.13966773396301E+06,
#             -2.08082869632721E+05,
#             -2.67614536674500E+05,
#             -5.42551380218506E+05,
#             -1.97868066013336E+05,
#             4.53328138000488E+05,
#             7.37393379348755E+05,
#             5.20777461418152E+05,
#             2.46628021972836E+05,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_water_flux_vector,
#             rtol=1e-10,
#         ))
#
#     def test_void_ratio_increment_vector(self):
#         expected_dT = np.array([
#             0.00000000000000E+00,
#             9.15746427389685E-05,
#             1.04883069156525E-04,
#             7.67320828637213E-05,
#             3.37983097104436E-05,
#             -1.86087592687940E-06,
#             -1.97591413889387E-05,
#             -2.05973867154074E-05,
#             -1.15832007823766E-05,
#             -1.00523141887636E-06,
#             5.48751089066735E-06,
#             6.09287095873518E-06,
#             2.05841250072898E-06,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_void_ratio_vector,
#             rtol=1e-10, atol=1e-20,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 9.92791192400449E-06
#         self.assertEqual(self.msh._iter, 1)
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#
#
if __name__ == "__main__":
    unittest.main()
