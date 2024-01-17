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


# class TestUpdateGlobalMatricesCubicConstant(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#         )
#         self.msh = ConsolidationAnalysis1D((0, 100), generate=True)
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.deg_sat_water = 0.8
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.water_flux_rate = -1.5e-8
#
#     def test_initial_stiffness_matrix(self):
#         expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         self.assertTrue(np.allclose(self.msh._stiffness_matrix_0, expected))
#         self.assertTrue(np.allclose(self.msh._stiffness_matrix, expected))
#
#     def test_initial_mass_matrix(self):
#         expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         self.assertTrue(np.allclose(self.msh._mass_matrix_0, expected))
#         self.assertTrue(np.allclose(self.msh._mass_matrix, expected))
#
#     def test_update_stiffness_matrix(self):
#         expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         h00 = 0.7465991018961040
#         h11 = 2.0906373785075500
#         h33 = 0.6858746574516590
#         h10 = -0.8713876864303850
#         h20 = 0.2431123389801100
#         h30 = -0.0575993100013845
#         h21 = -1.3758296977239400
#         h01 = -0.9579200197637180
#         h02 = 0.2795470056467770
#         h03 = -0.0682260877791622
#         h12 = -1.4987966977239400
#         d0 = np.ones((self.msh.num_nodes,)) * (h00 + h33)
#         d0[0] = h00
#         d0[-1] = h33
#         d0[1::3] = h11
#         d0[2::3] = h11
#         dm1 = np.ones((self.msh.num_nodes - 1,)) * h10
#         dm1[1::3] = h21
#         dm2 = np.ones((self.msh.num_nodes - 2,)) * h20
#         dm2[2::3] = 0.0
#         dm3 = np.zeros((self.msh.num_nodes - 3,))
#         dm3[0::3] = h30
#         dp1 = np.ones((self.msh.num_nodes - 1,)) * h01
#         dp1[1::3] = h12
#         dp2 = np.ones((self.msh.num_nodes - 2,)) * h02
#         dp2[2::3] = 0.0
#         dp3 = np.zeros((self.msh.num_nodes - 3,))
#         dp3[0::3] = h03
#         expected1 = np.diag(d0)
#         expected1 += np.diag(dm1, -1) + np.diag(dp1, 1)
#         expected1 += np.diag(dm2, -2) + np.diag(dp2, 2)
#         expected1 += np.diag(dm3, -3) + np.diag(dp3, 3)
#         self.msh.update_stiffness_matrix()
#         self.assertTrue(np.allclose(self.msh._stiffness_matrix_0, expected0))
#         self.assertTrue(np.allclose(self.msh._stiffness_matrix, expected1))
#
#     def test_update_mass_matrix(self):
#         expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         c00 = 1.84687971781305e6
#         c11 = 9.34982857142857e6
#         c10 = 1.42844603174603e6
#         c20 = -5.19434920634924e5
#         c30 = 2.74146208112873e5
#         c21 = -1.16872857142856e6
#         d0 = np.ones((self.msh.num_nodes,)) * 2.0 * c00
#         d0[0] = c00
#         d0[-1] = c00
#         d0[1::3] = c11
#         d0[2::3] = c11
#         d1 = np.ones((self.msh.num_nodes - 1,)) * c10
#         d1[1::3] = c21
#         d2 = np.ones((self.msh.num_nodes - 2,)) * c20
#         d2[2::3] = 0.0
#         d3 = np.zeros((self.msh.num_nodes - 3,))
#         d3[0::3] = c30
#         expected1 = np.diag(d0)
#         expected1 += np.diag(d1, -1) + np.diag(d1, 1)
#         expected1 += np.diag(d2, -2) + np.diag(d2, 2)
#         expected1 += np.diag(d3, -3) + np.diag(d3, 3)
#         self.msh.update_mass_matrix()
#         self.assertTrue(np.allclose(
#             self.msh._mass_matrix_0, expected0))
#         self.assertTrue(np.allclose(
#             self.msh._mass_matrix, expected1))
#
#
# class TestUpdateIntegrationPointsCubic(unittest.TestCase):
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
#         self.msh._void_ratio_vector[:] = np.array([
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
#         self.msh._void_ratio_rate_vector[:] = np.array([
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
#         self.msh.update_nodes()
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         self.msh.update_integration_points()
#
#     def test_void_ratio_distribution(self):
#         expected_void_ratio_int_pts = np.array([
#             -3.422539664476490,
#             -7.653704430301370,
#             -10.446160239424800,
#             -9.985642548540930,
#             -8.257070581278590,
#             -7.064308307087920,
#             -4.672124032386330,
#             -1.440401917815120,
#             0.974681570235134,
#             1.870711258948380,
#             2.078338922559240,
#             2.177366336413890,
#             1.680380179180770,
#             0.811005133641826,
#             0.227782988247163,
#             -0.031120907462955,
#             -0.417466130765087,
#             -0.644813855455235,
#             -0.528772037813549,
#             -0.285997082550321,
#         ])
#         actual_void_ratio_int_pts = np.array([
#             ip.void_ratio for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_int_pts,
#                                     expected_void_ratio_int_pts))
#
#     def test_void_ratio_rate_distribution(self):
#         expected_void_ratio_rate_int_pts = np.array([
#             -0.034225396644765,
#             -0.076537044303014,
#             -0.104461602394248,
#             -0.099856425485409,
#             -0.082570705812786,
#             -0.070643083070879,
#             -0.046721240323863,
#             -0.014404019178151,
#             0.009746815702351,
#             0.018707112589484,
#             0.020783389225592,
#             0.021773663364139,
#             0.016803801791808,
#             0.008110051336418,
#             0.002277829882472,
#             -0.000311209074630,
#             -0.004174661307651,
#             -0.006448138554552,
#             -0.005287720378135,
#             -0.002859970825503,
#         ])
#         actual_void_ratio_rate_int_pts = np.array([
#             ip.void_ratio_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_rate_int_pts,
#                                     expected_void_ratio_rate_int_pts))
#
#     def test_void_ratio_gradient_distribution(self):
#         expected_void_ratio_gradient_int_pts = np.array([
#             -1.15093426984199,
#             -0.70037674599536,
#             -0.15129838219301,
#             0.26620714324995,
#             0.47571152668668,
#             0.52108465163990,
#             0.51343382134772,
#             0.43315319751340,
#             0.27077898886023,
#             0.11272541074531,
#             0.07267706952532,
#             -0.02454456350281,
#             -0.11231442240250,
#             -0.13519566470900,
#             -0.11353558171063,
#             -0.10632645781291,
#             -0.06254104067706,
#             -0.00664052813362,
#             0.03949323637823,
#             0.06538510258090,
#         ])
#         actual_void_ratio_gradient_int_pts = np.array([
#             ip.void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_gradient_int_pts,
#                                     expected_void_ratio_gradient_int_pts))
#
#     def test_deg_sat_water_distribution(self):
#         expected_deg_sat_water_int_pts = np.array([
#             0.044857035897862,
#             0.028960004408085,
#             0.024424941557965,
#             0.025036878560037,
#             0.027783692446551,
#             0.030254882699662,
#             0.037889208517624,
#             0.071616670181262,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             0.531172122610449,
#             0.139509906472742,
#             0.110434834954165,
#             0.122871924439420,
#             0.170874577744838,
#         ])
#         actual_deg_sat_water_int_pts = np.array([
#             ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
#                                     expected_deg_sat_water_int_pts))
#
#     def test_deg_sat_water_void_ratio_grad_distribution(self):
#         expected_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             0.007100952173750,
#             0.002066569970404,
#             0.001283854149754,
#             0.001375496464083,
#             0.001839863406448,
#             0.002336484783824,
#             0.004404228952282,
#             0.026828796529360,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             7.683274447247210,
#             0.179434281070916,
#             0.092158984082347,
#             0.124931329787761,
#             0.319815963223536,
#         ])
#         actual_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             ip.deg_sat_water_void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_void_ratio_grad_int_pts,
#                                     expected_deg_sat_water_void_ratio_grad_int_pts))
#
#     def test_water_flux_distribution(self):
#         expected_water_flux_int_pts = np.array([
#             1.8071264681E-15,
#             5.0412534775E-23,
#             1.5503621786E-28,
#             -1.7186479754E-27,
#             -3.0723442020E-24,
#             -3.9524301404E-22,
#             -5.4976497868E-18,
#             -1.8328498927E-12,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             1.0956257621E-10,
#             1.5160234947E-11,
#             6.5849524657E-13,
#             -6.1857084254E-12,
#             -2.6451092292E-11,
#         ])
#         actual_water_flux_int_pts = np.array([
#             ip.water_flux_rate
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_water_flux_int_pts,
#                                     expected_water_flux_int_pts,
#                                     atol=1e-30))


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
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio_0 = 0.9
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
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio_0 = 0.9
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


# class TestUpdateWeightedMatricesLinear(unittest.TestCase):
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
#             order=1
#         )
#         initial_void_ratio_vector = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         initial_void_ratio_rate_vector = np.array([
#             0.05,
#             0.02,
#             0.01,
#             -0.08,
#             -0.05,
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
#             bnd_value=2.0,
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
#             2.0,
#             0.6,
#             -0.2,
#             -0.8,
#             -6,
#         ])
#         self.msh._void_ratio_rate_vector[:] = np.array([
#             0,
#             500,
#             600,
#             700,
#             6000,
#         ])
#         self.msh.update_consolidation_boundary_conditions(self.msh._t1)
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_water_flux_vector()
#         self.msh.update_stiffness_matrix()
#         self.msh.update_mass_matrix()
#         self.msh.update_weighted_matrices()
#
#     def test_void_ratio_distribution_nodes(self):
#         expected_void_ratio_vector_0 = np.array([
#             2.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         expected_void_ratio_vector = np.array([
#             2.0,
#             0.6,
#             -0.2,
#             -0.8,
#             -6,
#         ])
#         actual_void_ratio_nodes = np.array([
#             nd.void_ratio for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     actual_void_ratio_nodes))
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     self.msh._void_ratio_vector))
#         self.assertTrue(np.allclose(expected_void_ratio_vector_0,
#                                     self.msh._void_ratio_vector_0))
#
#     def test_void_ratio_distribution_int_pts(self):
#         expected_void_ratio_int_pts = np.array([
#             1.7041451884327400,
#             0.8958548115672620,
#             0.4309401076758500,
#             -0.0309401076758503,
#             -0.3267949192431120,
#             -0.6732050807568880,
#             -1.8988893001069700,
#             -4.9011106998930300,
#         ])
#         actual_void_ratio_int_pts = np.array([
#             ip.void_ratio for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_int_pts,
#                                     expected_void_ratio_int_pts))
#
#     def test_void_ratio_rate_distribution_nodes(self):
#         expected_void_ratio_rate_vector = np.array([
#             0,
#             500,
#             600,
#             700,
#             6000,
#         ])
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector))
#
#     def test_void_ratio_rate_distribution_int_pts(self):
#         expected_void_ratio_rate_int_pts = np.array([
#             105.6624327025940,
#             394.3375672974060,
#             521.1324865405190,
#             578.8675134594810,
#             621.1324865405190,
#             678.8675134594810,
#             1820.0217866474900,
#             4879.9782133525100,
#         ])
#         actual_void_ratio_rate_int_pts = np.array([
#             ip.void_ratio_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_rate_int_pts,
#                                     expected_void_ratio_rate_int_pts))
#
#     def test_void_ratio_gradient_distribution(self):
#         expected_void_ratio_gradient_int_pts = np.array([
#             -0.0560000,
#             -0.0560000,
#             -0.0320000,
#             -0.0320000,
#             -0.0240000,
#             -0.0240000,
#             -0.2080000,
#             -0.2080000,
#         ])
#         actual_void_ratio_gradient_int_pts = np.array([
#             ip.void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_gradient_int_pts,
#                                     expected_void_ratio_gradient_int_pts))
#
#     def test_deg_sat_water_distribution(self):
#         expected_deg_sat_water_int_pts = np.array([
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             0.532566107142957,
#             0.159095145119459,
#             0.107903535760564,
#             0.061690903893537,
#             0.036917109700501,
#         ])
#         actual_deg_sat_water_int_pts = np.array([
#             ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
#                                     expected_deg_sat_water_int_pts))
#
#     def test_deg_sat_water_void_ratio_grad_distribution(self):
#         expected_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             7.737022063089890,
#             0.260925333912205,
#             0.086263750747031,
#             0.017548502720457,
#             0.004092516398380,
#         ])
#         actual_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             ip.deg_sat_water_void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_void_ratio_grad_int_pts,
#                                     expected_deg_sat_water_void_ratio_grad_int_pts))
#
#     def test_water_flux_distribution(self):
#         expected_water_flux_int_pts = np.array([
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             -1.9136585886E-11,
#             -4.4163827878E-12,
#             -1.1115184284E-12,
#             -7.6323111305E-14,
#             -4.9394064270E-19,
#         ])
#         actual_water_flux_int_pts = np.array([
#             ip.water_flux_rate
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_water_flux_int_pts,
#                                     expected_water_flux_int_pts,
#                                     atol=1e-30))
#
#     def test_spec_grav_distribution(self):
#         expected_spec_grav_int_pts = np.array([
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             2.29587638328360,
#             2.62205400109484,
#             2.67023583947749,
#             2.71449137425298,
#             2.73851722970310,
#         ])
#         actual_spec_grav_int_pts = np.array([
#             ip.spec_grav
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_spec_grav_int_pts,
#                                     expected_spec_grav_int_pts))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1507583313920580, -0.0786443785009067,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0786056432160232, 0.1767637295188870,
#              -0.0981580863028642, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0981468970118543,
#                 0.1992782620987600, -0.1011313650869050],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1011312105966210, 0.1011312105966210],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#
#     def test_global_stiffness_matrix_0(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
#              -0.0999626493104180, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
#                 0.2016437545597640, -0.1016790550734020],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554918020, 0.1016790554918020],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix_0,
#         ))
#
#     def test_global_stiffness_matrix_weighted(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1591542280116300, -0.0870402751204793,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0870153954701239, 0.1860757632767650,
#              -0.0990603678066411, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0990557982491078,
#                 0.2004610083292620, -0.1014052100801540],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1014051330442120, 0.1014051330442120],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._weighted_stiffness_matrix,
#         ))
#
#     def test_global_mass_matrix_0(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
#                 2.15090157481671E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
#                 5.71758161659235E+07, 9.43574147951588E+06],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43574147951588E+06, 1.74614402201079E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix_0,
#         ))
#
#     def test_global_mass_matrix(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 3.82127786353284E+08, 1.27845341703034E+09,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 1.27845341703034E+09, 4.93329220496251E+09,
#                 6.53471451217525E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 6.53471451217525E+07,
#                 1.08389521552969E+08, 1.17642307395528E+07],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 1.17642307395528E+07, 1.96536710434876E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix,
#         ))
#
#     def test_global_mass_matrix_weighted(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 7.66475899599609E+08, 8.00150151716176E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 8.00150151716176E+08, 2.57010076129751E+09,
#                 4.34280804349598E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349598E+07,
#                 8.27826688594461E+07, 1.05999861095344E+07],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 1.05999861095344E+07, 1.85575556317977E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._weighted_mass_matrix,
#         ))
#
#     def test_global_coef_matrix_0(self):
#         expected_C0 = np.array([
#             [2.1204012345643E+10, 1.0602006172876E+10, 0.0000000000000E+00,
#                 0.0000000000000E+00, 0.0000000000000E+00],
#             [1.0602006172876E+10, 7.6647589959953E+11, 8.0015015171622E+11,
#                 0.0000000000000E+00, 0.0000000000000E+00],
#             [0.0000000000000E+00, 8.0015015171622E+11, 2.5701007612974E+12,
#                 4.3428080435009E+10, 0.0000000000000E+00],
#             [0.0000000000000E+00, 0.0000000000000E+00, 4.3428080435009E+10,
#                 8.2782668859346E+10, 1.0599986109585E+10],
#             [0.0000000000000E+00, 0.0000000000000E+00, 0.0000000000000E+00,
#                 1.0599986109585E+10, 1.8557555631747E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C0, self.msh._coef_matrix_0,
#             rtol=1e-14, atol=1e-3,
#         ))
#
#     def test_global_coef_matrix_1(self):
#         expected_C1 = np.array([
#             [2.12040123457151E+10, 1.06020061728035E+10, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728035E+10, 7.66475899599689E+11, 8.00150151716132E+11,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 8.00150151716133E+11, 2.57010076129760E+12,
#                 4.34280804349103E+10, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349103E+10,
#                 8.27826688595464E+10, 1.05999861094837E+10],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 1.05999861094837E+10, 1.85575556318484E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C1, self.msh._coef_matrix_1,
#             rtol=1e-14, atol=1e-3,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
#         expected_flux_vector_0[-1] = -2.74983450612514 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector_0,
#                                     self.msh._water_flux_vector_0))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.73851722970310 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector))
#
#     def test_global_flux_vector_weighted(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -0.5 * (2.73851722970310
#                                            + 2.74983450612514) * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._weighted_water_flux_vector))
#
#
# class TestVoidRatioCorrectionLinearOneStep(unittest.TestCase):
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
#             order=1
#         )
#         initial_void_ratio_vector = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         initial_void_ratio_rate_vector = np.array([
#             0.05,
#             0.02,
#             0.01,
#             -0.08,
#             -0.05,
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
#             bnd_value=2.0,
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
#             2.0,
#             0.6,
#             -0.2,
#             -0.8,
#             -6,
#         ])
#         self.msh._void_ratio_rate_vector[:] = np.array([
#             0,
#             500,
#             600,
#             700,
#             6000,
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
#             2.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         expected_void_ratio_vector = np.array([
#             2.0000000000000000,
#             0.0999999999995405,
#             -0.7999999999994870,
#             -1.5000000000217000,
#             -11.9999999999265000,
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
#             0.000000E+00,
#             -4.595491E-10,
#             5.135892E-10,
#             -2.170086E-08,
#             7.346657E-08,
#         ])
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes,
#                                     atol=1e-12,
#                                     rtol=1e-3,
#                                     ))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector,
#                                     atol=1e-12,
#                                     rtol=1e-3,
#                                     ))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1675205356603570, -0.0954065827692065,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0954368524138330, 0.1954002217027660,
#              -0.0999633692889327, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0999643764055516,
#                 0.2016434316692080, -0.1016790552636560],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554408220, 0.1016790554408220],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#
#     def test_global_stiffness_matrix_0(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
#              -0.0999626493104180, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
#                 0.2016437545597640, -0.1016790550734020],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554918020, 0.1016790554918020],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix_0,
#         ))
#
#     def test_global_stiffness_matrix_weighted(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1591542280116300, -0.0870402751204793,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0870153954701239, 0.1860757632767650,
#              -0.0990603678066411, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0990557982491078,
#                 0.2004610083292620, -0.1014052100801540],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1014051330442120, 0.1014051330442120],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._weighted_stiffness_matrix,
#         ))
#
#     def test_global_mass_matrix_0(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
#                 2.15090157481671E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
#                 5.71758161659235E+07, 9.43574147951588E+06],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43574147951588E+06, 1.74614402201079E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix_0,
#         ))
#
#     def test_global_mass_matrix(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 1.15082401284158E+09, 3.21846886400860E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 3.21846886400860E+08, 2.06909317631999E+08,
#                 2.15090157480156E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157480156E+07,
#                 5.71758161655548E+07, 9.43574147951723E+06],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43574147951723E+06, 1.74614402201153E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix,
#         ))
#
#     def test_global_mass_matrix_weighted(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 7.66475899599609E+08, 8.00150151716176E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 8.00150151716176E+08, 2.57010076129751E+09,
#                 4.34280804349598E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349598E+07,
#                 8.27826688594461E+07, 1.05999861095344E+07],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 1.05999861095344E+07, 1.85575556317977E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._weighted_mass_matrix,
#         ))
#
#     def test_global_coef_matrix_0(self):
#         expected_C0 = np.array([
#             [2.1204012345643E+10, 1.0602006172876E+10, 0.0000000000000E+00,
#                 0.0000000000000E+00, 0.0000000000000E+00],
#             [1.0602006172876E+10, 7.6647589959953E+11, 8.0015015171622E+11,
#                 0.0000000000000E+00, 0.0000000000000E+00],
#             [0.0000000000000E+00, 8.0015015171622E+11, 2.5701007612974E+12,
#                 4.3428080435009E+10, 0.0000000000000E+00],
#             [0.0000000000000E+00, 0.0000000000000E+00, 4.3428080435009E+10,
#                 8.2782668859346E+10, 1.0599986109585E+10],
#             [0.0000000000000E+00, 0.0000000000000E+00, 0.0000000000000E+00,
#                 1.0599986109585E+10, 1.8557555631747E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C0, self.msh._coef_matrix_0,
#             rtol=1e-14, atol=1e-3,
#         ))
#
#     def test_global_coef_matrix_1(self):
#         expected_C1 = np.array([
#             [2.12040123457151E+10, 1.06020061728035E+10, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728035E+10, 7.66475899599689E+11, 8.00150151716132E+11,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 8.00150151716133E+11, 2.57010076129760E+12,
#                 4.34280804349103E+10, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349103E+10,
#                 8.27826688595464E+10, 1.05999861094837E+10],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 1.05999861094837E+10, 1.85575556318484E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C1, self.msh._coef_matrix_1,
#             rtol=1e-14, atol=1e-3,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
#         expected_flux_vector_0[-1] = -2.74983450612514 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector_0,
#                                     self.msh._water_flux_vector_0))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.74983450612506 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector))
#
#     def test_global_flux_vector_weighted(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -0.5 * (2.73851722970310
#                                            + 2.74983450612514) * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._weighted_water_flux_vector))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             -5.30100308654E+09,
#             -8.63328040829E+11,
#             -1.97253518894E+12,
#             -1.47604633121E+11,
#             -1.18765324067E+11,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_water_flux_vector,
#         ))
#
#     def test_void_ratio_increment_vector(self):
#         expected_dT = np.array([
#             0.00000E+00,
#             -5.00000E-01,
#             -6.00000E-01,
#             -7.00000E-01,
#             -6.00000E+00,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_void_ratio_vector,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 4.95840885995792E-01
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#         self.assertEqual(self.msh._iter, 1)
#
#
# class TestIterativeVoidRatioCorrectionLinear(unittest.TestCase):
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
#             order=1
#         )
#         initial_void_ratio_vector = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         initial_void_ratio_rate_vector = np.array([
#             0.05,
#             0.02,
#             0.01,
#             -0.08,
#             -0.05,
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
#             bnd_value=2.0,
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
#             2.0,
#             0.6,
#             -0.2,
#             -0.8,
#             -6,
#         ])
#         self.msh._void_ratio_rate_vector[:] = np.array([
#             0,
#             500,
#             600,
#             700,
#             6000,
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
#         expected_void_ratio_vector_0 = np.array([
#             2.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         expected_void_ratio_vector = np.array([
#             2.0000000000000000,
#             0.0999999999983167,
#             -0.7999999999938220,
#             -1.5000000000335000,
#             -11.9999999999168000,
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
#             0.000000E+00,
#             -1.683265E-09,
#             6.177614E-09,
#             -3.350076E-08,
#             8.317969E-08,
#         ])
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes,
#                                     atol=1e-12,
#                                     rtol=1e-3,
#                                     ))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector,
#                                     atol=1e-12,
#                                     rtol=1e-3,
#                                     ))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#              0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1675252439072830,
#              -0.0954112910161317, 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0954350107457320,
#              0.1953984089517590, -0.0999633982060268, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0999643720200537,
#              0.2016434272652130, -0.1016790552451590],
#             [0.0000000000000000, 0.0000000000000000,
#              0.0000000000000000, -0.1016790554457790, 0.1016790554457790],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#
#     def test_global_stiffness_matrix_0(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
#              -0.0999626493104180, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
#                 0.2016437545597640, -0.1016790550734020],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554918020, 0.1016790554918020],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix_0,
#         ))
#
#     def test_global_stiffness_matrix_weighted(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000],
#             [-0.0721139528911510, 0.1675353295557450, -0.0954213766645937,
#                 0.0000000000000000, 0.0000000000000000],
#             [0.0000000000000000, -0.0954310001289640, 0.1953940094300550,
#              -0.0999630093010912, 0.0000000000000000],
#             [0.0000000000000000, 0.0000000000000000, -0.0999645379456421,
#                 0.2016435931141710, -0.1016790551685290],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554663120, 0.1016790554663120],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._weighted_stiffness_matrix,
#         ))
#
#     def test_global_mass_matrix_0(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
#                 2.15090157481671E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
#                 5.71758161659235E+07, 9.43574147951588E+06],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43574147951588E+06, 1.74614402201079E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix_0,
#         ))
#
#     def test_global_mass_matrix(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 1.15082401284598E+09, 3.21846886402202E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 3.21846886402202E+08, 2.06909317633054E+08,
#                 2.15090157479978E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157479978E+07,
#                 5.71758161653827E+07, 9.43574147951432E+06],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43574147951432E+06, 1.74614402201151E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix,
#         ))
#
#     def test_global_mass_matrix_weighted(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728395E+07, 1.15082401284375E+09, 3.21846886401436E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 3.21846886401436E+08, 2.06909317632248E+08,
#                 2.15090157480912E+07, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157480912E+07,
#                 5.71758161657388E+07, 9.43574147951655E+06],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43574147951655E+06, 1.74614402201115E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._weighted_mass_matrix,
#         ))
#
#     def test_global_coef_matrix_0(self):
#         expected_C0 = np.array([
#             [2.1204012345643E+10, 1.0602006172876E+10, 0.0000000000000E+00,
#                 0.0000000000000E+00, 0.0000000000000E+00],
#             [1.0602006172876E+10, 1.1508240128437E+12, 3.2184688640148E+11,
#                 0.0000000000000E+00, 0.0000000000000E+00],
#             [0.0000000000000E+00, 3.2184688640148E+11, 2.0690931763215E+11,
#                 2.1509015748141E+10, 0.0000000000000E+00],
#             [0.0000000000000E+00, 0.0000000000000E+00, 2.1509015748141E+10,
#                 5.7175816165638E+10, 9.4357414795674E+09],
#             [0.0000000000000E+00, 0.0000000000000E+00, 0.0000000000000E+00,
#                 9.4357414795674E+09, 1.7461440220061E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C0, self.msh._coef_matrix_0,
#             rtol=1e-13, atol=1e-3,
#         ))
#
#     def test_global_coef_matrix_1(self):
#         expected_C1 = np.array([
#             [2.12040123457151E+10, 1.06020061728035E+10, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [1.06020061728035E+10, 1.15082401284384E+12, 3.21846886401388E+11,
#                 0.00000000000000E+00, 0.00000000000000E+00],
#             [0.00000000000000E+00, 3.21846886401388E+11, 2.06909317632346E+11,
#                 2.15090157480413E+10, 0.00000000000000E+00],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157480413E+10,
#                 5.71758161658397E+10, 9.43574147946571E+09],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43574147946571E+09, 1.74614402201624E+10],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C1, self.msh._coef_matrix_1,
#             rtol=1e-13, atol=1e-3,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
#         expected_flux_vector_0[-1] = -2.74983450612514 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector_0,
#                                     self.msh._water_flux_vector_0))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.74983450612506 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector))
#
#     def test_global_flux_vector_weighted(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -0.5 * (2.74983450612514
#                                            + 2.74983450612506) * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._weighted_water_flux_vector))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             -1.3214111328125E-01,
#             4.1467285156250E-01,
#             5.2429199218750E-01,
#             -4.6118164062500E-01,
#             5.8278333356252E-02,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_water_flux_vector,
#             rtol=1e-13, atol=1e-3,
#         ))
#
#     def test_void_ratio_increment_vector(self):
#         expected_dT = np.array([
#             0.00000E+00,
#             -1.22372E-12,
#             5.66406E-12,
#             -1.17999E-11,
#             9.71392E-12,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_void_ratio_vector,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 1.33062137400659E-12
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#         self.assertEqual(self.msh._iter, 2)
#
#
# class TestInitializeGlobalSystemCubic(unittest.TestCase):
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
#
#     def test_time_step_set(self):
#         self.assertAlmostEqual(self.msh._t0, 1.5)
#         self.assertAlmostEqual(self.msh._t1, 1.5)
#
#     def test_free_indices(self):
#         expected_free_vec = [i for i in range(self.msh.num_nodes)][1:]
#         self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
#         self.assertTrue(np.all(expected_free_vec ==
#                         self.msh._free_arr[0].flatten()))
#         self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))
#
#     def test_void_ratio_distribution_nodes(self):
#         expected_void_ratio_vector = np.array([
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
#         actual_void_ratio_nodes = np.array([
#             nd.void_ratio for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     actual_void_ratio_nodes))
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     self.msh._void_ratio_vector))
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     self.msh._void_ratio_vector_0))
#
#     def test_void_ratio_distribution_int_pts(self):
#         expected_void_ratio_int_pts = np.array([
#             -3.422539664476490,
#             -7.653704430301370,
#             -10.446160239424800,
#             -9.985642548540930,
#             -8.257070581278590,
#             -7.064308307087920,
#             -4.672124032386330,
#             -1.440401917815120,
#             0.974681570235134,
#             1.870711258948380,
#             2.078338922559240,
#             2.177366336413890,
#             1.680380179180770,
#             0.811005133641826,
#             0.227782988247163,
#             -0.031120907462955,
#             -0.417466130765087,
#             -0.644813855455235,
#             -0.528772037813549,
#             -0.285997082550321,
#         ])
#         actual_void_ratio_int_pts = np.array([
#             ip.void_ratio for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_int_pts,
#                                     expected_void_ratio_int_pts,
#                                     ))
#
#     def test_void_ratio_rate_distribution_nodes(self):
#         expected_void_ratio_rate_vector = np.array([
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
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes,
#                                     ))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector,
#                                     ))
#
#     def test_void_ratio_rate_distribution_int_pts(self):
#         expected_void_ratio_rate_int_pts = np.array([
#             -0.034225396644765,
#             -0.076537044303014,
#             -0.104461602394248,
#             -0.099856425485409,
#             -0.082570705812786,
#             -0.070643083070879,
#             -0.046721240323863,
#             -0.014404019178151,
#             0.009746815702351,
#             0.018707112589484,
#             0.020783389225592,
#             0.021773663364139,
#             0.016803801791808,
#             0.008110051336418,
#             0.002277829882472,
#             -0.000311209074630,
#             -0.004174661307651,
#             -0.006448138554552,
#             -0.005287720378135,
#             -0.002859970825503,
#         ])
#         actual_void_ratio_rate_int_pts = np.array([
#             ip.void_ratio_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_rate_int_pts,
#                                     expected_void_ratio_rate_int_pts,
#                                     ))
#
#     def test_void_ratio_gradient_distribution(self):
#         expected_void_ratio_gradient_int_pts = np.array([
#             -1.15093426984199,
#             -0.70037674599536,
#             -0.15129838219301,
#             0.26620714324995,
#             0.47571152668668,
#             0.52108465163990,
#             0.51343382134772,
#             0.43315319751340,
#             0.27077898886023,
#             0.11272541074531,
#             0.07267706952532,
#             -0.02454456350281,
#             -0.11231442240250,
#             -0.13519566470900,
#             -0.11353558171063,
#             -0.10632645781291,
#             -0.06254104067706,
#             -0.00664052813362,
#             0.03949323637823,
#             0.06538510258090,
#         ])
#         actual_void_ratio_gradient_int_pts = np.array([
#             ip.void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_gradient_int_pts,
#                                     expected_void_ratio_gradient_int_pts,
#                                     ))
#
#     def test_deg_sat_water_distribution(self):
#         expected_deg_sat_water_int_pts = np.array([
#             0.044857035897863,
#             0.028960004408085,
#             0.024424941557965,
#             0.025036878560037,
#             0.027783692446551,
#             0.030254882699662,
#             0.037889208517624,
#             0.071616670181262,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             0.531172122610449,
#             0.139509906472742,
#             0.110434834954165,
#             0.122871924439420,
#             0.170874577744838,
#         ])
#         actual_deg_sat_water_int_pts = np.array([
#             ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
#                                     expected_deg_sat_water_int_pts,
#                                     ))
#
#     def test_deg_sat_water_void_ratio_grad_distribution(self):
#         expected_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             0.007100952173751,
#             0.002066569970404,
#             0.001283854149754,
#             0.001375496464083,
#             0.001839863406448,
#             0.002336484783824,
#             0.004404228952282,
#             0.026828796529360,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             7.683274447247210,
#             0.179434281070916,
#             0.092158984082347,
#             0.124931329787761,
#             0.319815963223536,
#         ])
#         actual_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             ip.deg_sat_water_void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_void_ratio_grad_int_pts,
#                                     expected_deg_sat_water_void_ratio_grad_int_pts,
#                                     ))
#
#     def test_water_flux_distribution(self):
#         expected_water_flux_int_pts = np.array([
#             1.8071264681E-15,
#             5.0412534775E-23,
#             1.5503621786E-28,
#             -1.7186479754E-27,
#             -3.0723442020E-24,
#             -3.9524301404E-22,
#             -5.4976497868E-18,
#             -1.8328498927E-12,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             1.0956257621E-10,
#             1.5160234947E-11,
#             6.5849524657E-13,
#             -6.1857084254E-12,
#             -2.6451092292E-11,
#         ])
#         actual_water_flux_int_pts = np.array([
#             ip.water_flux_rate
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_water_flux_int_pts,
#                                     expected_water_flux_int_pts,
#                                     atol=1e-30))
#
#     def test_spec_grav_distribution(self):
#         expected_spec_grav_int_pts = np.array([
#             2.73079394984140,
#             2.74627913402004,
#             2.75071278269289,
#             2.75011411247605,
#             2.74742845413695,
#             2.74501452413187,
#             2.73757048307299,
#             2.70492452266610,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             2.29701505120963,
#             2.64038419594985,
#             2.66783269125494,
#             2.65605663067911,
#             2.61109074784318,
#         ])
#         actual_spec_grav_int_pts = np.array([
#             ip.spec_grav
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_spec_grav_int_pts,
#                                     expected_spec_grav_int_pts,
#                                     atol=1e-30))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.37541276634706, -0.47913494460783,
#                 0.13662447765333, -0.03290229939256],
#             [-0.47913495073106, 1.09749384355934,
#              -0.75556012955112, 0.13720123672283],
#             [0.13662448026901, -0.75556012965636,
#                 1.10036574587615, -0.48143009648880],
#             [-0.0329022999455621, 0.13720123675250,
#              -0.48143009649197, 0.37713115968503],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.37422788138613, -0.47040475255461,
#                 0.12454905038507, -0.02837217921659],
#             [-0.47040445574570, 1.04580620489708,
#              -0.68911230909446, 0.11371055994309],
#             [0.12454845679017, -0.68910429557128,
#                 0.91904216745687, -0.35448632867575],
#             [-0.0283721462389552, 0.11370996634735,
#              -0.35448603187892, 0.26914821177052],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.26682162569726, -0.34073842741069,
#                 0.09735383640305, -0.02343703468962],
#             [-0.34073842741069, 0.77883069122443,
#              -0.53544610021680, 0.09735383640305],
#             [0.09735383640305, -0.53544610021680,
#                 0.77883069122443, -0.34073842741069],
#             [-0.0234370346896243, 0.09735383640305,
#              -0.34073842741069, 0.26682162569726],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.32902830441547, -0.41311979105240,
#                 0.11114097767973, -0.02704949104280],
#             [-0.41352446262158, 0.98699528504461,
#              -0.69720784332217, 0.12373702089915],
#             [0.11130058733431, -0.69723220657608,
#                 1.04220001681837, -0.45626839757660],
#             [-0.0270752880707999, 0.12370525491951,
#              -0.45616559239695, 0.35953562554824],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#
#     def test_global_mass_matrix(self):
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
#             expected_C, self.msh._mass_matrix,
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.61109074784318 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector))
#
#
# class TestInitializeTimeStepCubic(unittest.TestCase):
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
#
#     def test_time_step_set(self):
#         self.assertAlmostEqual(self.msh._t0, 1.5)
#         self.assertAlmostEqual(self.msh._t1, 1.501)
#
#     def test_iteration_variables(self):
#         self.assertEqual(self.msh._eps_a, 1.0)
#         self.assertEqual(self.msh._iter, 0)
#
#     def test_void_ratio_distribution_nodes(self):
#         expected_void_ratio_vector = np.array([
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
#         actual_void_ratio_nodes = np.array([
#             nd.void_ratio for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     actual_void_ratio_nodes))
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     self.msh._void_ratio_vector))
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     self.msh._void_ratio_vector_0))
#
#     def test_void_ratio_distribution_int_pts(self):
#         expected_void_ratio_int_pts = np.array([
#             -3.422539664476490,
#             -7.653704430301370,
#             -10.446160239424800,
#             -9.985642548540930,
#             -8.257070581278590,
#             -7.064308307087920,
#             -4.672124032386330,
#             -1.440401917815120,
#             0.974681570235134,
#             1.870711258948380,
#             2.078338922559240,
#             2.177366336413890,
#             1.680380179180770,
#             0.811005133641826,
#             0.227782988247163,
#             -0.031120907462955,
#             -0.417466130765087,
#             -0.644813855455235,
#             -0.528772037813549,
#             -0.285997082550321,
#         ])
#         actual_void_ratio_int_pts = np.array([
#             ip.void_ratio for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_int_pts,
#                                     expected_void_ratio_int_pts))
#
#     def test_void_ratio_rate_distribution_nodes(self):
#         expected_void_ratio_rate_vector = np.array([
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
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector))
#
#     def test_void_ratio_rate_distribution_int_pts(self):
#         expected_void_ratio_rate_int_pts = np.array([
#             -0.034225396644765,
#             -0.076537044303014,
#             -0.104461602394248,
#             -0.099856425485409,
#             -0.082570705812786,
#             -0.070643083070879,
#             -0.046721240323863,
#             -0.014404019178151,
#             0.009746815702351,
#             0.018707112589484,
#             0.020783389225592,
#             0.021773663364139,
#             0.016803801791808,
#             0.008110051336418,
#             0.002277829882472,
#             -0.000311209074630,
#             -0.004174661307651,
#             -0.006448138554552,
#             -0.005287720378135,
#             -0.002859970825503,
#         ])
#         actual_void_ratio_rate_int_pts = np.array([
#             ip.void_ratio_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_rate_int_pts,
#                                     expected_void_ratio_rate_int_pts))
#
#     def test_void_ratio_gradient_distribution(self):
#         expected_void_ratio_gradient_int_pts = np.array([
#             -1.15093426984199,
#             -0.70037674599536,
#             -0.15129838219301,
#             0.26620714324995,
#             0.47571152668668,
#             0.52108465163990,
#             0.51343382134772,
#             0.43315319751340,
#             0.27077898886023,
#             0.11272541074531,
#             0.07267706952532,
#             -0.02454456350281,
#             -0.11231442240250,
#             -0.13519566470900,
#             -0.11353558171063,
#             -0.10632645781291,
#             -0.06254104067706,
#             -0.00664052813362,
#             0.03949323637823,
#             0.06538510258090,
#         ])
#         actual_void_ratio_gradient_int_pts = np.array([
#             ip.void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_gradient_int_pts,
#                                     expected_void_ratio_gradient_int_pts))
#
#     def test_deg_sat_water_distribution(self):
#         expected_deg_sat_water_int_pts = np.array([
#             0.044857035897862,
#             0.028960004408085,
#             0.024424941557965,
#             0.025036878560037,
#             0.027783692446551,
#             0.030254882699662,
#             0.037889208517624,
#             0.071616670181262,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             0.531172122610449,
#             0.139509906472742,
#             0.110434834954165,
#             0.122871924439420,
#             0.170874577744838,
#         ])
#         actual_deg_sat_water_int_pts = np.array([
#             ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
#                                     expected_deg_sat_water_int_pts))
#
#     def test_deg_sat_water_void_ratio_grad_distribution(self):
#         expected_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             0.007100952173750,
#             0.002066569970404,
#             0.001283854149754,
#             0.001375496464083,
#             0.001839863406448,
#             0.002336484783824,
#             0.004404228952282,
#             0.026828796529360,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             7.683274447247210,
#             0.179434281070916,
#             0.092158984082347,
#             0.124931329787761,
#             0.319815963223536,
#         ])
#         actual_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             ip.deg_sat_water_void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_void_ratio_grad_int_pts,
#                                     expected_deg_sat_water_void_ratio_grad_int_pts))
#
#     def test_water_flux_distribution(self):
#         expected_water_flux_int_pts = np.array([
#             1.8071264681E-15,
#             5.0412534775E-23,
#             1.5503621786E-28,
#             -1.7186479754E-27,
#             -3.0723442020E-24,
#             -3.9524301404E-22,
#             -5.4976497868E-18,
#             -1.8328498927E-12,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             1.0956257621E-10,
#             1.5160234947E-11,
#             6.5849524657E-13,
#             -6.1857084254E-12,
#             -2.6451092292E-11,
#         ])
#         actual_water_flux_int_pts = np.array([
#             ip.water_flux_rate
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_water_flux_int_pts,
#                                     expected_water_flux_int_pts,
#                                     atol=1e-30))
#
#     def test_spec_grav_distribution(self):
#         expected_spec_grav_int_pts = np.array([
#             2.73079394984140,
#             2.74627913402004,
#             2.75071278269289,
#             2.75011411247605,
#             2.74742845413695,
#             2.74501452413187,
#             2.73757048307299,
#             2.70492452266610,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             2.29701505120963,
#             2.64038419594985,
#             2.66783269125494,
#             2.65605663067911,
#             2.61109074784318,
#         ])
#         actual_spec_grav_int_pts = np.array([
#             ip.spec_grav
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_spec_grav_int_pts,
#                                     expected_spec_grav_int_pts,
#                                     atol=1e-30))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.37541276634706, -0.47913494460783,
#                 0.13662447765333, -0.03290229939256],
#             [-0.47913495073106, 1.09749384355934,
#              -0.75556012955112, 0.13720123672283],
#             [0.13662448026901, -0.75556012965636,
#                 1.10036574587615, -0.48143009648880],
#             [-0.0329022999455621, 0.13720123675250,
#              -0.48143009649197, 0.37713115968503],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.37422788138613, -0.47040475255461,
#                 0.12454905038507, -0.02837217921659],
#             [-0.47040445574570, 1.04580620489708,
#              -0.68911230909446, 0.11371055994309],
#             [0.12454845679017, -0.68910429557128,
#                 0.91904216745687, -0.35448632867575],
#             [-0.0283721462389552, 0.11370996634735,
#              -0.35448603187892, 0.26914821177052],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.26682162569726, -0.34073842741069,
#                 0.09735383640305, -0.02343703468962],
#             [-0.34073842741069, 0.77883069122443,
#              -0.53544610021680, 0.09735383640305],
#             [0.09735383640305, -0.53544610021680,
#                 0.77883069122443, -0.34073842741069],
#             [-0.0234370346896243, 0.09735383640305,
#              -0.34073842741069, 0.26682162569726],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.32902830441547, -0.41311979105240,
#                 0.11114097767973, -0.02704949104280],
#             [-0.41352446262158, 0.98699528504461,
#              -0.69720784332217, 0.12373702089915],
#             [0.11130058733431, -0.69723220657608,
#                 1.04220001681837, -0.45626839757660],
#             [-0.0270752880707999, 0.12370525491951,
#              -0.45616559239695, 0.35953562554824],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix_0,
#         ))
#
#     def test_global_mass_matrix(self):
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
#             expected_C, self.msh._mass_matrix,
#         ))
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix_0,
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.61109074784318 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector))
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector_0))
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._weighted_water_flux_vector))
#
#
# class TestUpdateWeightedMatricesCubic(unittest.TestCase):
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
#         actual_void_ratio_nodes = np.array([
#             nd.void_ratio for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     actual_void_ratio_nodes))
#         self.assertTrue(np.allclose(expected_void_ratio_vector,
#                                     self.msh._void_ratio_vector))
#         self.assertTrue(np.allclose(expected_void_ratio_vector_0,
#                                     self.msh._void_ratio_vector_0))
#
#     def test_void_ratio_distribution_int_pts(self):
#         expected_void_ratio_int_pts = np.array([
#             -3.422558663172200,
#             -7.653777872055470,
#             -10.446265951027200,
#             -9.985741476399620,
#             -8.257152402542280,
#             -7.064378950170990,
#             -4.672170753626650,
#             -1.440416321834300,
#             0.974691317050836,
#             1.870729966060970,
#             2.078359705948460,
#             2.177388110077260,
#             1.680396982982560,
#             0.811013243693162,
#             0.227785266077045,
#             -0.031121218672030,
#             -0.417470305426395,
#             -0.644820303593790,
#             -0.528777325533927,
#             -0.285999942521147,
#         ])
#         actual_void_ratio_int_pts = np.array([
#             ip.void_ratio for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_int_pts,
#                                     expected_void_ratio_int_pts))
#
#     def test_void_ratio_rate_distribution_nodes(self):
#         expected_void_ratio_rate_vector = np.array([
#             0.00000000000000E+01,
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
#         actual_void_ratio_rate_nodes = np.array([
#             nd.void_ratio_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     actual_void_ratio_rate_nodes))
#         self.assertTrue(np.allclose(expected_void_ratio_rate_vector,
#                                     self.msh._void_ratio_rate_vector))
#
#     def test_void_ratio_rate_distribution_int_pts(self):
#         expected_void_ratio_rate_int_pts = np.array([
#             -0.018998695698926,
#             -0.073441754087105,
#             -0.105711602393866,
#             -0.098927858747365,
#             -0.081821263711586,
#             -0.070643083070656,
#             -0.046721240323791,
#             -0.014404019178151,
#             0.009746815702341,
#             0.018707112589436,
#             0.020783389225579,
#             0.021773663364221,
#             0.016803801791818,
#             0.008110051336330,
#             0.002277829882428,
#             -0.000311209074661,
#             -0.004174661307715,
#             -0.006448138554552,
#             -0.005287720378074,
#             -0.002859970825479,
#         ])
#         actual_void_ratio_rate_int_pts = np.array([
#             ip.void_ratio_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_rate_int_pts,
#                                     expected_void_ratio_rate_int_pts))
#
#     def test_void_ratio_gradient_distribution(self):
#         expected_void_ratio_gradient_int_pts = np.array([
#             -1.15094952744558,
#             -0.70038540187041,
#             -0.15129979517683,
#             0.26621009170330,
#             0.47571579778849,
#             0.52108986248641,
#             0.51343895568593,
#             0.43315752904537,
#             0.27078169665011,
#             0.11272653799941,
#             0.07267779629601,
#             -0.02454480894844,
#             -0.11231554554672,
#             -0.13519701666565,
#             -0.11353671706644,
#             -0.10632752107749,
#             -0.06254166608747,
#             -0.00664059453890,
#             0.03949363131059,
#             0.06538575643192,
#         ])
#         actual_void_ratio_gradient_int_pts = np.array([
#             ip.void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_void_ratio_gradient_int_pts,
#                                     expected_void_ratio_gradient_int_pts))
#
#     def test_deg_sat_water_distribution(self):
#         expected_deg_sat_water_int_pts = np.array([
#             0.044856900989608,
#             0.028959852636677,
#             0.024424805840736,
#             0.025036742486148,
#             0.027783541907744,
#             0.030254717644438,
#             0.037889002748163,
#             0.071616283741731,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             1.000000000000000,
#             0.531169731519830,
#             0.139509157401129,
#             0.110434240704826,
#             0.122871263842543,
#             0.170873663087485,
#         ])
#         actual_deg_sat_water_int_pts = np.array([
#             ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
#                                     expected_deg_sat_water_int_pts))
#
#     def test_deg_sat_water_void_ratio_grad_distribution(self):
#         expected_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             0.007100891659295,
#             0.002066539598856,
#             0.001283834284541,
#             0.001375475622202,
#             0.001839835492879,
#             0.002336448985836,
#             0.004404161382346,
#             0.026828384412124,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             0.000000000000000,
#             7.683182424661650,
#             0.179431534630069,
#             0.092157570309422,
#             0.124929414851821,
#             0.319811086490582,
#         ])
#         actual_deg_sat_water_void_ratio_grad_int_pts = np.array([
#             ip.deg_sat_water_void_ratio_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_void_ratio_grad_int_pts,
#                                     expected_deg_sat_water_void_ratio_grad_int_pts))
#
#     def test_water_flux_distribution(self):
#         expected_water_flux_int_pts = np.array([
#             1.7713796673E-15,
#             5.0330476423E-23,
#             1.5503165575E-28,
#             -1.7174679441E-27,
#             -3.0704550887E-24,
#             -3.9513529639E-22,
#             -5.4966774209E-18,
#             -1.8327626215E-12,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             0.0000000000E+00,
#             1.0956353545E-10,
#             1.5160133394E-11,
#             6.5848484730E-13,
#             -6.1856394494E-12,
#             -2.6451054205E-11,
#         ])
#         actual_water_flux_int_pts = np.array([
#             ip.water_flux_rate
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_water_flux_int_pts,
#                                     expected_water_flux_int_pts,
#                                     atol=1e-30))
#
#     def test_spec_grav_distribution(self):
#         expected_spec_grav_int_pts = np.array([
#             2.73079408088337,
#             2.74627928227786,
#             2.75071291548223,
#             2.75011424558537,
#             2.74742860125224,
#             2.74501468529173,
#             2.73757068344138,
#             2.70492489447492,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             1.94419643704324,
#             2.29701700484279,
#             2.64038489946504,
#             2.66783325516560,
#             2.65605725478295,
#             2.61109159734326,
#         ])
#         actual_spec_grav_int_pts = np.array([
#             ip.spec_grav
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_spec_grav_int_pts,
#                                     expected_spec_grav_int_pts))
#
#     def test_global_stiffness_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] = np.array([
#             [0.3754127850728640, -0.4791349680520250,
#                 0.1366244840341230, -0.0329023010549603],
#             [-0.4791349740541280, 1.0974938969543100,
#              -0.7555601670436450, 0.1372012441434650],
#             [0.1366244865980610, -0.7555601671468060,
#                 1.1003658024385600, -0.4814301218898150],
#             [-0.0329023015970263, 0.1372012441725460,
#              -0.4814301218929200, 0.7513590838946910],
#         ])
#         expected_H[3:7, 3:7] = np.array([
#             [0.7513590838946910, -0.4704047801777000,
#                 0.1245490555783110, -0.0283721799779005],
#             [-0.4704044833829290, 1.0458062849331300,
#              -0.6891123656515080, 0.1137105641013030],
#             [0.1245484620116770, -0.6891043525098910,
#                 0.9190422231347980, -0.3544863326365830],
#             [-0.0283721470018353, 0.1137099705338300,
#              -0.3544860358538770, 0.5359698380191410],
#         ])
#         expected_H[6:10, 6:10] = np.array([
#             [0.5359698380191410, -0.3407384274106900,
#                 0.0973538364030547, -0.0234370346896243],
#             [-0.3407384274106900, 0.7788306912244320,
#              -0.5354461002167970, 0.0973538364030541],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900],
#             [-0.0234370346896243, 0.0973538364030541,
#              -0.3407384274106900, 0.5958501457631080],
#         ])
#         expected_H[9:13, 9:13] = np.array([
#             [0.5958501457631080, -0.4131200916113520,
#                 0.1111410888008420, -0.0270495172553382],
#             [-0.4135247662059270, 0.9869958222111810,
#              -0.6972081460114500, 0.1237370900061950],
#             [0.1113006998436770, -0.6972325092175540,
#                 1.0422003590338500, -0.4562685496599700],
#             [-0.0270753145870706, 0.1237053240705960,
#              -0.4561657447604080, 0.3595357352768820],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._stiffness_matrix,
#         ))
#
#     def test_global_stiffness_matrix_0(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.37541276634706, -0.47913494460783,
#                 0.13662447765333, -0.03290229939256],
#             [-0.47913495073106, 1.09749384355934,
#              -0.75556012955112, 0.13720123672283],
#             [0.13662448026901, -0.75556012965636,
#                 1.10036574587615, -0.48143009648880],
#             [-0.0329022999455621, 0.13720123675250,
#              -0.48143009649197, 0.37713115968503],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.37422788138613, -0.47040475255461,
#                 0.12454905038507, -0.02837217921659],
#             [-0.47040445574570, 1.04580620489708,
#              -0.68911230909446, 0.11371055994309],
#             [0.12454845679017, -0.68910429557128,
#                 0.91904216745687, -0.35448632867575],
#             [-0.0283721462389552, 0.11370996634735,
#              -0.35448603187892, 0.26914821177052],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.26682162569726, -0.34073842741069,
#                 0.09735383640305, -0.02343703468962],
#             [-0.34073842741069, 0.77883069122443,
#              -0.53544610021680, 0.09735383640305],
#             [0.09735383640305, -0.53544610021680,
#                 0.77883069122443, -0.34073842741069],
#             [-0.0234370346896243, 0.09735383640305,
#              -0.34073842741069, 0.26682162569726],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.32902830441547, -0.41311979105240,
#                 0.11114097767973, -0.02704949104280],
#             [-0.41352446262158, 0.98699528504461,
#              -0.69720784332217, 0.12373702089915],
#             [0.11130058733431, -0.69723220657608,
#                 1.04220001681837, -0.45626839757660],
#             [-0.0270752880707999, 0.12370525491951,
#              -0.45616559239695, 0.35953562554824],
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
#         ))
#
#     def test_global_mass_matrix(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] = np.array([
#             [4.73636296905052E+06, 3.48911383948062E+06,
#              -1.29625169945974E+06, 6.28539134903018E+05],
#             [3.48911383948062E+06, 2.04661623290336E+07,
#              -2.68983132168345E+06, -1.08386918732327E+06],
#             [-1.29625169945974E+06, -2.68983132168345E+06,
#                 1.99779403271669E+07, 3.06434881520768E+06],
#             [6.28539134903018E+05, -1.08386918732327E+06,
#                 3.06434881520768E+06, 8.17180004817830E+06],
#         ])
#         expected_C[3:7, 3:7] = np.array([
#             [8.17180004817830E+06, 2.79778820911984E+06,
#              -1.54214909260440E+06, 7.18713439921888E+05],
#             [2.79778820911984E+06, 2.63421744519585E+07,
#                 8.59529928266825E+05, -1.83531637508857E+06],
#             [-1.54214909260440E+06, 8.59529928266825E+05,
#                 2.79086620289149E+07, 3.38412277408819E+06],
#             [7.18713439921888E+05, -1.83531637508857E+06,
#                 3.38412277408819E+06, 9.73170885677195E+06],
#         ])
#         expected_C[6:10, 6:10] = np.array([
#             [9.73170885677195E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442682E+05],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857144E+06, -1.36311507936508E+06],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 1.04946265784836E+09],
#         ])
#         expected_C[9:13, 9:13] = np.array([
#             [1.04946265784836E+09, 5.23835206242772E+08,
#              -2.37982947941482E+08, 5.50351027284678E+07],
#             [5.23835206242772E+08, 3.84435995536596E+08,
#              -1.43607931079425E+08, 1.42596408432806E+07],
#             [-2.37982947941482E+08, -1.43607931079425E+08,
#                 1.69011616024721E+08, 1.93500286732473E+07],
#             [5.50351027284678E+07, 1.42596408432806E+07,
#                 1.93500286732473E+07, 5.13998892011317E+07],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._mass_matrix,
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
#             rtol=1e-14, atol=1e-3,
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
#             rtol=1e-14, atol=1e-3,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
#         expected_flux_vector_0[-1] = -2.61109074784318 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector_0,
#                                     self.msh._water_flux_vector_0))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -2.61109159734326 * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._water_flux_vector))
#
#     def test_global_flux_vector_weighted(self):
#         expected_flux_vector = np.zeros(self.msh.num_nodes)
#         expected_flux_vector[-1] = -0.5 * (2.61109074784318
#                                            + 2.61109159734326) * 25.0e-3
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._weighted_water_flux_vector))
#
#
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
