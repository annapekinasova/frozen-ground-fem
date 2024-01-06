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

    def test_update_heat_flux_vector_no_int_pt(self):
        msh = ThermalAnalysis1D((0, 100), generate=True)
        bnd0 = ThermalBoundary1D((msh.nodes[0],))
        msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            (msh.nodes[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
        )
        msh.add_boundary(bnd1)
        bnd2 = ThermalBoundary1D(
            (msh.nodes[5],),
            bnd_type=ThermalBoundary1D.BoundaryType.heat_flux,
        )
        msh.add_boundary(bnd2)
        with self.assertRaises(AttributeError):
            msh.update_heat_flux_vector()


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


class TestThermalAnalysis1DSetters(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D((100, -8))

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


class TestThermalAnalysis1DLinearNoArgs(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D(order=1)

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


class TestThermalAnalysis1DLinearMeshGen(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D(z_range=(100, -8))

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
        self.assertEqual(self.msh._temp_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_storage_matrix.shape, (nnod, nnod))
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
        self.assertEqual(self.msh._temp_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_storage_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_1.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr


class TestThermalAnalysis1DCubicMeshGen(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D(z_range=(100, -8))

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
        self.assertEqual(self.msh._temp_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_storage_matrix.shape, (nnod, nnod))
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
        self.assertEqual(self.msh._temp_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._weighted_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
        self.assertEqual(
            self.msh._weighted_heat_storage_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._coef_matrix_1.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr


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
        self.assertTrue(bnd1 in self.msh.boundaries)

    def test_add_boundary_with_int_pt(self):
        bnd = ThermalBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd)
        bnd1 = ThermalBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(bnd1)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(bnd1 in self.msh.boundaries)


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


class TestUpdateBoundaries(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
        )
        self.msh = ThermalAnalysis1D((0, 100), generate=True)
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.deg_sat_water = 0.8
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
        per = 365.0 * 86400.0
        om = 2.0 * np.pi / per
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        def f(t): return Tavg + Tamp * np.cos(om * (t - t0))
        self.f = f
        self.bnd0 = ThermalBoundary1D(
            (self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_function=f)
        self.msh.add_boundary(self.bnd0)
        self.geotherm_grad = 25.0 / 1.0e3
        self.flux_geotherm = -0.05218861799159
        self.bnd1 = ThermalBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
            bnd_value=self.geotherm_grad,
        )
        self.msh.add_boundary(self.bnd1)
        self.fixed_flux = 0.08
        self.bnd2 = ThermalBoundary1D(
            (self.msh.nodes[5],),
            bnd_type=ThermalBoundary1D.BoundaryType.heat_flux,
            bnd_value=self.fixed_flux,
        )
        self.msh.add_boundary(self.bnd2)

    def test_initial_temp_heat_flux_vector(self):
        for tn, tn0 in zip(self.msh._temp_vector,
                           self.msh._temp_vector_0):
            self.assertEqual(tn, 0.0)
            self.assertEqual(tn0, 0.0)
        for fx, fx0 in zip(self.msh._heat_flux_vector,
                           self.msh._heat_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_initial_thrm_cond(self):
        expected_thrm_cond = 2.0875447196636
        for e in self.msh.elements:
            for ip in e.int_pts:
                self.assertAlmostEqual(ip.thrm_cond, expected_thrm_cond)

    def test_update_thermal_boundaries(self):
        t = 1.314e7
        expected_temp_0 = self.f(t)
        expected_temp_1 = 15.0
        self.msh.update_thermal_boundary_conditions(t)
        self.assertAlmostEqual(self.msh.nodes[0].temp, expected_temp_0)
        self.assertAlmostEqual(self.msh.nodes[0].temp, expected_temp_1)
        self.assertAlmostEqual(self.msh._temp_vector[0], expected_temp_0)
        self.assertAlmostEqual(self.msh._temp_vector[0], expected_temp_1)
        for tn in self.msh._temp_vector[1:]:
            self.assertEqual(tn, 0.0)
        for tn0 in self.msh._temp_vector_0:
            self.assertEqual(tn0, 0.0)
        for fx, fx0 in zip(self.msh._heat_flux_vector,
                           self.msh._heat_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)
        t = 3.5478e7
        expected_temp_2 = self.f(t)
        expected_temp_3 = -14.3185165257814
        self.msh.update_thermal_boundary_conditions(t)
        self.assertAlmostEqual(self.msh.nodes[0].temp, expected_temp_2)
        self.assertAlmostEqual(self.msh.nodes[0].temp, expected_temp_3)
        self.assertAlmostEqual(self.msh._temp_vector[0], expected_temp_2)
        self.assertAlmostEqual(self.msh._temp_vector[0], expected_temp_3)
        for tn in self.msh._temp_vector[1:]:
            self.assertEqual(tn, 0.0)
        for tn0 in self.msh._temp_vector_0:
            self.assertEqual(tn0, 0.0)
        for fx, fx0 in zip(self.msh._heat_flux_vector,
                           self.msh._heat_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_update_heat_flux_vector(self):
        t = 1.314e7
        self.msh.update_thermal_boundary_conditions(t)
        self.msh.update_heat_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._heat_flux_vector,
                                          self.msh._heat_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == 5:
                self.assertAlmostEqual(fx, self.fixed_flux)
            elif k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, self.flux_geotherm)
            else:
                self.assertEqual(fx, 0.0)
        self.msh.update_thermal_boundary_conditions(t)
        self.msh.update_heat_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._heat_flux_vector,
                                          self.msh._heat_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == 5:
                self.assertAlmostEqual(fx, self.fixed_flux)
            elif k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, self.flux_geotherm)
            else:
                self.assertEqual(fx, 0.0)


class TestUpdateGlobalMatricesLinearConstant(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
        )
        self.msh = ThermalAnalysis1D((0, 100), generate=True, order=1)
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.deg_sat_water = 0.8
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.water_flux_rate = -1.5e-8

    def test_initial_heat_flow_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix, expected))

    def test_initial_heat_storage_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected))

    def test_update_heat_flow_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        k00 = 0.2239397572692170
        k11 = 0.1632153128247730
        d0 = np.ones((self.msh.num_nodes,)) * (k00 + k11)
        d0[0] = k00
        d0[-1] = k11
        dp1 = -np.ones((self.msh.num_nodes - 1,)) * k00
        dm1 = -np.ones((self.msh.num_nodes - 1,)) * k11
        expected1 = np.diag(d0) + np.diag(dm1, -1) + np.diag(dp1, 1)
        self.msh.update_heat_flow_matrix()
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix_0, expected0))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix, expected1))

    def test_update_heat_storage_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        c0 = 8.08009876543210e6
        d0 = np.ones((self.msh.num_nodes,)) * 2.0 * c0
        d0[0] = c0
        d0[-1] = c0
        d1 = np.ones((self.msh.num_nodes - 1,)) * c0 * 0.5
        expected1 = np.diag(d0) + np.diag(d1, -1) + np.diag(d1, 1)
        self.msh.update_heat_storage_matrix()
        self.assertTrue(np.allclose(
            self.msh._heat_storage_matrix_0, expected0))
        self.assertTrue(np.allclose(
            self.msh._heat_storage_matrix, expected1))


class TestUpdateIntegrationPointsLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        self.msh._temp_vector[:] = np.array([
            2,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        self.msh.update_nodes()
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        self.msh.update_integration_points()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()

    def test_temperature_distribution(self):
        expected_temp_int_pts = np.array([
            1.5984827557301400,
            0.5015172442698560,
            -0.0901923788646684,
            -0.6098076211353320,
            -0.9479274057836310,
            -1.3520725942163700,
            -3.7189110867544700,
            -9.7810889132455400,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution(self):
        expected_temp_rate_int_pts = np.array([
            0.04366025403784440,
            0.02633974596215560,
            0.01788675134594810,
            0.01211324865405190,
            -0.00901923788646684,
            -0.06098076211353320,
            -0.07366025403784440,
            -0.05633974596215560,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -0.0760000,
            -0.0760000,
            -0.0360000,
            -0.0360000,
            -0.0280000,
            -0.0280000,
            -0.4200000,
            -0.4200000,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.000000000000000,
            1.000000000000000,
            0.314715929845879,
            0.113801777607921,
            0.089741864676250,
            0.074104172041942,
            0.042882888566470,
            0.025322726744343,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_deg_sat_water_temp_grad_distribution(self):
        expected_deg_sat_water_temp_grad_int_pts = np.array([
            0.000000000000000,
            0.000000000000001,
            1.810113168088841,
            0.100397364897923,
            0.051013678629442,
            0.029567794445951,
            0.006250998062539,
            0.001419738751606,
        ])
        actual_deg_sat_water_temp_grad_int_pts = np.array([
            ip.deg_sat_water_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_temp_grad_int_pts,
                                    expected_deg_sat_water_temp_grad_int_pts))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            0.0000000000E+00,
            0.0000000000E+00,
            -4.8910645169E-12,
            -5.5518497692E-13,
            8.3577053338E-13,
            1.7708810805E-13,
            2.0670411271E-16,
            6.0318175808E-27,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_water_flux_int_pts,
                                    expected_water_flux_int_pts,
                                    atol=1e-30))

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            1.94419643704324,
            1.94419643704324,
            2.48085630059944,
            2.66463955659925,
            2.68754164945741,
            2.70253225701445,
            2.73271219424962,
            2.74983450612514,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts,
                                    atol=1e-30))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
             -0.0999626493104180, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
                0.2016437545597640, -0.1016790550734020],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554918020, 0.1016790554918020],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
                2.15090157481671E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
                5.71758161659235E+07, 9.43574147951588E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951588E+06, 1.74614402201079E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))


class TestUpdateGlobalMatricesCubicConstant(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
        )
        self.msh = ThermalAnalysis1D((0, 100), generate=True)
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.deg_sat_water = 0.8
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.water_flux_rate = -1.5e-8

    def test_initial_heat_flow_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix, expected))

    def test_initial_heat_storage_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected))

    def test_update_heat_flow_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        h00 = 0.7465991018961040
        h11 = 2.0906373785075500
        h33 = 0.6858746574516590
        h10 = -0.8713876864303850
        h20 = 0.2431123389801100
        h30 = -0.0575993100013845
        h21 = -1.3758296977239400
        h01 = -0.9579200197637180
        h02 = 0.2795470056467770
        h03 = -0.0682260877791622
        h12 = -1.4987966977239400
        d0 = np.ones((self.msh.num_nodes,)) * (h00 + h33)
        d0[0] = h00
        d0[-1] = h33
        d0[1::3] = h11
        d0[2::3] = h11
        dm1 = np.ones((self.msh.num_nodes - 1,)) * h10
        dm1[1::3] = h21
        dm2 = np.ones((self.msh.num_nodes - 2,)) * h20
        dm2[2::3] = 0.0
        dm3 = np.zeros((self.msh.num_nodes - 3,))
        dm3[0::3] = h30
        dp1 = np.ones((self.msh.num_nodes - 1,)) * h01
        dp1[1::3] = h12
        dp2 = np.ones((self.msh.num_nodes - 2,)) * h02
        dp2[2::3] = 0.0
        dp3 = np.zeros((self.msh.num_nodes - 3,))
        dp3[0::3] = h03
        expected1 = np.diag(d0)
        expected1 += np.diag(dm1, -1) + np.diag(dp1, 1)
        expected1 += np.diag(dm2, -2) + np.diag(dp2, 2)
        expected1 += np.diag(dm3, -3) + np.diag(dp3, 3)
        self.msh.update_heat_flow_matrix()
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix_0, expected0))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix, expected1))

    def test_update_heat_storage_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        c00 = 1.84687971781305e6
        c11 = 9.34982857142857e6
        c10 = 1.42844603174603e6
        c20 = -5.19434920634924e5
        c30 = 2.74146208112873e5
        c21 = -1.16872857142856e6
        d0 = np.ones((self.msh.num_nodes,)) * 2.0 * c00
        d0[0] = c00
        d0[-1] = c00
        d0[1::3] = c11
        d0[2::3] = c11
        d1 = np.ones((self.msh.num_nodes - 1,)) * c10
        d1[1::3] = c21
        d2 = np.ones((self.msh.num_nodes - 2,)) * c20
        d2[2::3] = 0.0
        d3 = np.zeros((self.msh.num_nodes - 3,))
        d3[0::3] = c30
        expected1 = np.diag(d0)
        expected1 += np.diag(d1, -1) + np.diag(d1, 1)
        expected1 += np.diag(d2, -2) + np.diag(d2, 2)
        expected1 += np.diag(d3, -3) + np.diag(d3, 3)
        self.msh.update_heat_storage_matrix()
        self.assertTrue(np.allclose(
            self.msh._heat_storage_matrix_0, expected0))
        self.assertTrue(np.allclose(
            self.msh._heat_storage_matrix, expected1))


class TestUpdateIntegrationPointsCubic(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        self.msh._temp_vector[:] = np.array([
            -2.000000000000000,
            -9.157452320220460,
            -10.488299785319000,
            -7.673205119057850,
            -3.379831977359920,
            0.186084957826655,
            1.975912628300400,
            2.059737589813890,
            1.158320034961550,
            0.100523127786268,
            -0.548750924584512,
            -0.609286860003055,
            -0.205841501790609,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            -0.02000000000000000,
            -0.09157452320220460,
            -0.10488299785319000,
            -0.07673205119057850,
            -0.03379831977359920,
            0.00186084957826655,
            0.01975912628300400,
            0.02059737589813890,
            0.01158320034961550,
            0.00100523127786268,
            -0.00548750924584512,
            -0.00609286860003055,
            -0.00205841501790609,
        ])
        self.msh.update_nodes()
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        self.msh.update_integration_points()

    def test_temperature_distribution(self):
        expected_temp_int_pts = np.array([
            -3.422539664476490,
            -7.653704430301370,
            -10.446160239424800,
            -9.985642548540930,
            -8.257070581278590,
            -7.064308307087920,
            -4.672124032386330,
            -1.440401917815120,
            0.974681570235134,
            1.870711258948380,
            2.078338922559240,
            2.177366336413890,
            1.680380179180770,
            0.811005133641826,
            0.227782988247163,
            -0.031120907462955,
            -0.417466130765087,
            -0.644813855455235,
            -0.528772037813549,
            -0.285997082550321,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution(self):
        expected_temp_rate_int_pts = np.array([
            -0.034225396644765,
            -0.076537044303014,
            -0.104461602394248,
            -0.099856425485409,
            -0.082570705812786,
            -0.070643083070879,
            -0.046721240323863,
            -0.014404019178151,
            0.009746815702351,
            0.018707112589484,
            0.020783389225592,
            0.021773663364139,
            0.016803801791808,
            0.008110051336418,
            0.002277829882472,
            -0.000311209074630,
            -0.004174661307651,
            -0.006448138554552,
            -0.005287720378135,
            -0.002859970825503,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -1.15093426984199,
            -0.70037674599536,
            -0.15129838219301,
            0.26620714324995,
            0.47571152668668,
            0.52108465163990,
            0.51343382134772,
            0.43315319751340,
            0.27077898886023,
            0.11272541074531,
            0.07267706952532,
            -0.02454456350281,
            -0.11231442240250,
            -0.13519566470900,
            -0.11353558171063,
            -0.10632645781291,
            -0.06254104067706,
            -0.00664052813362,
            0.03949323637823,
            0.06538510258090,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            0.044857035897862,
            0.028960004408085,
            0.024424941557965,
            0.025036878560037,
            0.027783692446551,
            0.030254882699662,
            0.037889208517624,
            0.071616670181262,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            0.531172122610449,
            0.139509906472742,
            0.110434834954165,
            0.122871924439420,
            0.170874577744838,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_deg_sat_water_temp_grad_distribution(self):
        expected_deg_sat_water_temp_grad_int_pts = np.array([
            0.007100952173750,
            0.002066569970404,
            0.001283854149754,
            0.001375496464083,
            0.001839863406448,
            0.002336484783824,
            0.004404228952282,
            0.026828796529360,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            7.683274447247210,
            0.179434281070916,
            0.092158984082347,
            0.124931329787761,
            0.319815963223536,
        ])
        actual_deg_sat_water_temp_grad_int_pts = np.array([
            ip.deg_sat_water_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_temp_grad_int_pts,
                                    expected_deg_sat_water_temp_grad_int_pts))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            1.8071264681E-15,
            5.0412534775E-23,
            1.5503621786E-28,
            -1.7186479754E-27,
            -3.0723442020E-24,
            -3.9524301404E-22,
            -5.4976497868E-18,
            -1.8328498927E-12,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            1.0956257621E-10,
            1.5160234947E-11,
            6.5849524657E-13,
            -6.1857084254E-12,
            -2.6451092292E-11,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_water_flux_int_pts,
                                    expected_water_flux_int_pts,
                                    atol=1e-30))


class TestUpdateNodes(unittest.TestCase):
    def setUp(self):
        self.msh = ThermalAnalysis1D((0, 100), generate=True, order=1)
        self.msh._temp_vector[:] = np.linspace(2.0, 22.0, self.msh.num_nodes)
        self.msh._temp_vector_0[:] = np.linspace(1.0, 11.0, self.msh.num_nodes)
        self.msh.time_step = 0.25
        self.msh._temp_rate_vector[:] = (
            self.msh._temp_vector[:] - self.msh._temp_vector_0[:]
        ) / self.msh.dt
        self.msh.update_nodes()

    def test_initial_node_values(self):
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.temp, 2.0 * (k+1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k+1))

    def test_repeat_update_nodes(self):
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.temp, 2.0 * (k+1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k+1))

    def test_change_temp_vectors_update_nodes(self):
        self.msh._temp_vector[:] = np.linspace(4.0, 44.0, self.msh.num_nodes)
        self.msh._temp_vector_0[:] = np.linspace(2.0, 22.0, self.msh.num_nodes)
        self.msh._temp_rate_vector[:] = (
            self.msh._temp_vector[:] - self.msh._temp_vector_0[:]
        ) / self.msh.dt
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.temp, 2.0 * (k+1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k+1))
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.temp, 4.0 * (k+1))
            self.assertAlmostEqual(nd.temp_rate, 8.0 * (k+1))


class TestInitializeGlobalSystemLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_temp_vector = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        initial_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        for nd, T0, dTdt0 in zip(self.msh.nodes,
                                 initial_temp_vector,
                                 initial_temp_rate_vector,
                                 ):
            nd.temp = T0
            nd.temp_rate = dTdt0
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        bnd0 = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=2.0,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            nodes=(self.msh.nodes[-1],),
            int_pts=(self.msh.elements[-1].int_pts[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
            bnd_value=25.0e-3,
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

    def test_temperature_distribution_nodes(self):
        expected_temp_vector = np.array([
            2.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_vector,
                                    actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            1.5984827557301400,
            0.5015172442698560,
            -0.0901923788646684,
            -0.6098076211353320,
            -0.9479274057836310,
            -1.3520725942163700,
            -3.7189110867544700,
            -9.7810889132455400,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            0.04366025403784440,
            0.02633974596215560,
            0.01788675134594810,
            0.01211324865405190,
            -0.00901923788646684,
            -0.06098076211353320,
            -0.07366025403784440,
            -0.05633974596215560,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -0.0760000,
            -0.0760000,
            -0.0360000,
            -0.0360000,
            -0.0280000,
            -0.0280000,
            -0.4200000,
            -0.4200000,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.000000000000000,
            1.000000000000000,
            0.314715929845879,
            0.113801777607921,
            0.089741864676250,
            0.074104172041942,
            0.042882888566470,
            0.025322726744343,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_deg_sat_water_temp_grad_distribution(self):
        expected_deg_sat_water_temp_grad_int_pts = np.array([
            0.000000000000000,
            0.000000000000001,
            1.810113168088841,
            0.100397364897923,
            0.051013678629442,
            0.029567794445951,
            0.006250998062539,
            0.001419738751606,
        ])
        actual_deg_sat_water_temp_grad_int_pts = np.array([
            ip.deg_sat_water_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_temp_grad_int_pts,
                                    expected_deg_sat_water_temp_grad_int_pts))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            0.0000000000E+00,
            0.0000000000E+00,
            -4.8910645169E-12,
            -5.5518497692E-13,
            8.3577053338E-13,
            1.7708810805E-13,
            2.0670411271E-16,
            6.0318175808E-27,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_water_flux_int_pts,
                                    expected_water_flux_int_pts,
                                    atol=1e-30))

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            1.94419643704324,
            1.94419643704324,
            2.48085630059944,
            2.66463955659925,
            2.68754164945741,
            2.70253225701445,
            2.73271219424962,
            2.74983450612514,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts,
                                    atol=1e-30))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
             -0.0999626493104180, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
                0.2016437545597640, -0.1016790550734020],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554918020, 0.1016790554918020],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
                2.15090157481671E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
                5.71758161659235E+07, 9.43574147951588E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951588E+06, 1.74614402201079E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.74983450612514 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))


class TestInitializeTimeStepLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_temp_vector = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        initial_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        for nd, T0, dTdt0 in zip(self.msh.nodes,
                                 initial_temp_vector,
                                 initial_temp_rate_vector,
                                 ):
            nd.temp = T0
            nd.temp_rate = dTdt0
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        bnd0 = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=2.0,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            nodes=(self.msh.nodes[-1],),
            int_pts=(self.msh.elements[-1].int_pts[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
            bnd_value=25.0e-3,
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

    def test_temperature_distribution_nodes(self):
        expected_temp_vector = np.array([
            2.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_vector,
                                    actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            1.5984827557301400,
            0.5015172442698560,
            -0.0901923788646684,
            -0.6098076211353320,
            -0.9479274057836310,
            -1.3520725942163700,
            -3.7189110867544700,
            -9.7810889132455400,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            0.04366025403784440,
            0.02633974596215560,
            0.01788675134594810,
            0.01211324865405190,
            -0.00901923788646684,
            -0.06098076211353320,
            -0.07366025403784440,
            -0.05633974596215560,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -0.0760000,
            -0.0760000,
            -0.0360000,
            -0.0360000,
            -0.0280000,
            -0.0280000,
            -0.4200000,
            -0.4200000,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.000000000000000,
            1.000000000000000,
            0.314715929845879,
            0.113801777607921,
            0.089741864676250,
            0.074104172041942,
            0.042882888566470,
            0.025322726744343,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_deg_sat_water_temp_grad_distribution(self):
        expected_deg_sat_water_temp_grad_int_pts = np.array([
            0.000000000000000,
            0.000000000000001,
            1.810113168088841,
            0.100397364897923,
            0.051013678629442,
            0.029567794445951,
            0.006250998062539,
            0.001419738751606,
        ])
        actual_deg_sat_water_temp_grad_int_pts = np.array([
            ip.deg_sat_water_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_temp_grad_int_pts,
                                    expected_deg_sat_water_temp_grad_int_pts))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            0.0000000000E+00,
            0.0000000000E+00,
            -4.8910645169E-12,
            -5.5518497692E-13,
            8.3577053338E-13,
            1.7708810805E-13,
            2.0670411271E-16,
            6.0318175808E-27,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_water_flux_int_pts,
                                    expected_water_flux_int_pts,
                                    atol=1e-30))

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            1.94419643704324,
            1.94419643704324,
            2.48085630059944,
            2.66463955659925,
            2.68754164945741,
            2.70253225701445,
            2.73271219424962,
            2.74983450612514,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts,
                                    atol=1e-30))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
             -0.0999626493104180, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
                0.2016437545597640, -0.1016790550734020],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554918020, 0.1016790554918020],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
                2.15090157481671E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
                5.71758161659235E+07, 9.43574147951588E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951588E+06, 1.74614402201079E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.74983450612514 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector_0))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_heat_flux_vector))


class TestUpdateWeightedMatricesLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_temp_vector = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        initial_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        for nd, T0, dTdt0 in zip(self.msh.nodes,
                                 initial_temp_vector,
                                 initial_temp_rate_vector,
                                 ):
            nd.temp = T0
            nd.temp_rate = dTdt0
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        bnd0 = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=2.0,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            nodes=(self.msh.nodes[-1],),
            int_pts=(self.msh.elements[-1].int_pts[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
            bnd_value=25.0e-3,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array([
            2.0,
            0.6,
            -0.2,
            -0.8,
            -6,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            0,
            500,
            600,
            700,
            6000,
        ])
        self.msh.update_thermal_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
            2.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        expected_temp_vector = np.array([
            2.0,
            0.6,
            -0.2,
            -0.8,
            -6,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_vector,
                                    actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector_0,
                                    self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            1.7041451884327400,
            0.8958548115672620,
            0.4309401076758500,
            -0.0309401076758503,
            -0.3267949192431120,
            -0.6732050807568880,
            -1.8988893001069700,
            -4.9011106998930300,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0,
            500,
            600,
            700,
            6000,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            105.6624327025940,
            394.3375672974060,
            521.1324865405190,
            578.8675134594810,
            621.1324865405190,
            678.8675134594810,
            1820.0217866474900,
            4879.9782133525100,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -0.0560000,
            -0.0560000,
            -0.0320000,
            -0.0320000,
            -0.0240000,
            -0.0240000,
            -0.2080000,
            -0.2080000,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            0.532566107142957,
            0.159095145119459,
            0.107903535760564,
            0.061690903893537,
            0.036917109700501,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_deg_sat_water_temp_grad_distribution(self):
        expected_deg_sat_water_temp_grad_int_pts = np.array([
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            7.737022063089890,
            0.260925333912205,
            0.086263750747031,
            0.017548502720457,
            0.004092516398380,
        ])
        actual_deg_sat_water_temp_grad_int_pts = np.array([
            ip.deg_sat_water_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_temp_grad_int_pts,
                                    expected_deg_sat_water_temp_grad_int_pts))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            -1.9136585886E-11,
            -4.4163827878E-12,
            -1.1115184284E-12,
            -7.6323111305E-14,
            -4.9394064270E-19,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_water_flux_int_pts,
                                    expected_water_flux_int_pts,
                                    atol=1e-30))

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            2.29587638328360,
            2.62205400109484,
            2.67023583947749,
            2.71449137425298,
            2.73851722970310,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1507583313920580, -0.0786443785009067,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0786056432160232, 0.1767637295188870,
             -0.0981580863028642, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0981468970118543,
                0.1992782620987600, -0.1011313650869050],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1011312105966210, 0.1011312105966210],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
             -0.0999626493104180, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
                0.2016437545597640, -0.1016790550734020],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554918020, 0.1016790554918020],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix_weighted(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1591542280116300, -0.0870402751204793,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0870153954701239, 0.1860757632767650,
             -0.0990603678066411, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0990557982491078,
                0.2004610083292620, -0.1014052100801540],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1014051330442120, 0.1014051330442120],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._weighted_heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
                2.15090157481671E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
                5.71758161659235E+07, 9.43574147951588E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951588E+06, 1.74614402201079E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 3.82127786353284E+08, 1.27845341703034E+09,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.27845341703034E+09, 4.93329220496251E+09,
                6.53471451217525E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 6.53471451217525E+07,
                1.08389521552969E+08, 1.17642307395528E+07],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.17642307395528E+07, 1.96536710434876E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_heat_storage_matrix_weighted(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 7.66475899599609E+08, 8.00150151716176E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 8.00150151716176E+08, 2.57010076129751E+09,
                4.34280804349598E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349598E+07,
                8.27826688594461E+07, 1.05999861095344E+07],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.05999861095344E+07, 1.85575556317977E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._weighted_heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [2.1204012345643E+10, 1.0602006172876E+10, 0.0000000000000E+00,
                0.0000000000000E+00, 0.0000000000000E+00],
            [1.0602006172876E+10, 7.6647589959953E+11, 8.0015015171622E+11,
                0.0000000000000E+00, 0.0000000000000E+00],
            [0.0000000000000E+00, 8.0015015171622E+11, 2.5701007612974E+12,
                4.3428080435009E+10, 0.0000000000000E+00],
            [0.0000000000000E+00, 0.0000000000000E+00, 4.3428080435009E+10,
                8.2782668859346E+10, 1.0599986109585E+10],
            [0.0000000000000E+00, 0.0000000000000E+00, 0.0000000000000E+00,
                1.0599986109585E+10, 1.8557555631747E+10],
        ])
        print(self.msh._coef_matrix_0)
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-14, atol=1e-3,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [2.12040123457151E+10, 1.06020061728035E+10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728035E+10, 7.66475899599689E+11, 8.00150151716132E+11,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 8.00150151716133E+11, 2.57010076129760E+12,
                4.34280804349103E+10, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349103E+10,
                8.27826688595464E+10, 1.05999861094837E+10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.05999861094837E+10, 1.85575556318484E+10],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-14, atol=1e-3,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
        expected_flux_vector_0[-1] = -2.74983450612514 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector_0,
                                    self.msh._heat_flux_vector_0))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.73851722970310 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -0.5 * (2.73851722970310
                                           + 2.74983450612514) * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_heat_flux_vector))


class TestTemperatureCorrectionLinearOneStep(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_temp_vector = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        initial_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        for nd, T0, dTdt0 in zip(self.msh.nodes,
                                 initial_temp_vector,
                                 initial_temp_rate_vector,
                                 ):
            nd.temp = T0
            nd.temp_rate = dTdt0
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        bnd0 = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=2.0,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            nodes=(self.msh.nodes[-1],),
            int_pts=(self.msh.elements[-1].int_pts[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
            bnd_value=25.0e-3,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array([
            2.0,
            0.6,
            -0.2,
            -0.8,
            -6,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            0,
            500,
            600,
            700,
            6000,
        ])
        self.msh.update_thermal_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()
        self.msh.calculate_temperature_correction()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
            2.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        expected_temp_vector = np.array([
            2.0000000000000000,
            0.0999999999995405,
            -0.7999999999994870,
            -1.5000000000217000,
            -11.9999999999265000,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_vector,
                                    actual_temp_nodes,
                                    atol=1e-13, rtol=1e-20))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector,
                                    atol=1e-13, rtol=1e-20))
        self.assertTrue(np.allclose(expected_temp_vector_0,
                                    self.msh._temp_vector_0,
                                    atol=1e-13, rtol=1e-20))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.000000E+00,
            -4.595491E-10,
            5.135892E-10,
            -2.170086E-08,
            7.346657E-08,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes,
                                    atol=1e-12,
                                    rtol=1e-3,
                                    ))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector,
                                    atol=1e-12,
                                    rtol=1e-3,
                                    ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675205356603570, -0.0954065827692065,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954368524138330, 0.1954002217027660,
             -0.0999633692889327, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999643764055516,
                0.2016434316692080, -0.1016790552636560],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554408220, 0.1016790554408220],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
             -0.0999626493104180, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
                0.2016437545597640, -0.1016790550734020],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554918020, 0.1016790554918020],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix_weighted(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1591542280116300, -0.0870402751204793,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0870153954701239, 0.1860757632767650,
             -0.0990603678066411, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0990557982491078,
                0.2004610083292620, -0.1014052100801540],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1014051330442120, 0.1014051330442120],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._weighted_heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
                2.15090157481671E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
                5.71758161659235E+07, 9.43574147951588E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951588E+06, 1.74614402201079E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284158E+09, 3.21846886400860E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886400860E+08, 2.06909317631999E+08,
                2.15090157480156E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157480156E+07,
                5.71758161655548E+07, 9.43574147951723E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951723E+06, 1.74614402201153E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_heat_storage_matrix_weighted(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 7.66475899599609E+08, 8.00150151716176E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 8.00150151716176E+08, 2.57010076129751E+09,
                4.34280804349598E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349598E+07,
                8.27826688594461E+07, 1.05999861095344E+07],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.05999861095344E+07, 1.85575556317977E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._weighted_heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [2.1204012345643E+10, 1.0602006172876E+10, 0.0000000000000E+00,
                0.0000000000000E+00, 0.0000000000000E+00],
            [1.0602006172876E+10, 7.6647589959953E+11, 8.0015015171622E+11,
                0.0000000000000E+00, 0.0000000000000E+00],
            [0.0000000000000E+00, 8.0015015171622E+11, 2.5701007612974E+12,
                4.3428080435009E+10, 0.0000000000000E+00],
            [0.0000000000000E+00, 0.0000000000000E+00, 4.3428080435009E+10,
                8.2782668859346E+10, 1.0599986109585E+10],
            [0.0000000000000E+00, 0.0000000000000E+00, 0.0000000000000E+00,
                1.0599986109585E+10, 1.8557555631747E+10],
        ])
        print(self.msh._coef_matrix_0)
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-14, atol=1e-3,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [2.12040123457151E+10, 1.06020061728035E+10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728035E+10, 7.66475899599689E+11, 8.00150151716132E+11,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 8.00150151716133E+11, 2.57010076129760E+12,
                4.34280804349103E+10, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 4.34280804349103E+10,
                8.27826688595464E+10, 1.05999861094837E+10],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.05999861094837E+10, 1.85575556318484E+10],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-14, atol=1e-3,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
        expected_flux_vector_0[-1] = -2.74983450612514 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector_0,
                                    self.msh._heat_flux_vector_0))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.74983450612506 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -0.5 * (2.73851722970310
                                           + 2.74983450612514) * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_heat_flux_vector))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -5.30100308654E+09,
            -8.63328040829E+11,
            -1.97253518894E+12,
            -1.47604633121E+11,
            -1.18765324067E+11,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.00000E+00,
            -5.00000E-01,
            -6.00000E-01,
            -7.00000E-01,
            -6.00000E+00,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 4.95840885995792E-01
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
        self.assertEqual(self.msh._iter, 1)


class TestIterativeTemperatureCorrectionLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
            order=1
        )
        initial_temp_vector = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        initial_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        for nd, T0, dTdt0 in zip(self.msh.nodes,
                                 initial_temp_vector,
                                 initial_temp_rate_vector,
                                 ):
            nd.temp = T0
            nd.temp_rate = dTdt0
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        bnd0 = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=2.0,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            nodes=(self.msh.nodes[-1],),
            int_pts=(self.msh.elements[-1].int_pts[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
            bnd_value=25.0e-3,
        )
        self.msh.add_boundary(bnd1)
        self.msh.initialize_global_system(1.5)
        self.msh.time_step = 1e-3
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array([
            2.0,
            0.6,
            -0.2,
            -0.8,
            -6,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            0,
            500,
            600,
            700,
            6000,
        ])
        self.msh.update_thermal_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()
        self.msh.iterative_correction_step()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
            2.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        expected_temp_vector = np.array([
            2.0000000000000000,
            0.0999999999983167,
            -0.7999999999938220,
            -1.5000000000335000,
            -11.9999999999168000,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_vector,
                                    actual_temp_nodes,
                                    atol=1e-13, rtol=1e-20))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector,
                                    atol=1e-13, rtol=1e-20))
        self.assertTrue(np.allclose(expected_temp_vector_0,
                                    self.msh._temp_vector_0,
                                    atol=1e-13, rtol=1e-20))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.000000E+00,
            -1.683265E-09,
            6.177614E-09,
            -3.350076E-08,
            8.317969E-08,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes,
                                    atol=1e-12,
                                    rtol=1e-3,
                                    ))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector,
                                    atol=1e-12,
                                    rtol=1e-3,
                                    ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
             0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675252439072830,
             -0.0954112910161317, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954350107457320,
             0.1953984089517590, -0.0999633982060268, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999643720200537,
             0.2016434272652130, -0.1016790552451590],
            [0.0000000000000000, 0.0000000000000000,
             0.0000000000000000, -0.1016790554457790, 0.1016790554457790],
        ])
        print(self.msh._heat_flow_matrix)
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675501246312030, -0.0954361717400518,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954251477242246, 0.1953877970346430,
             -0.0999626493104180, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999646994863613,
                0.2016437545597640, -0.1016790550734020],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554918020, 0.1016790554918020],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix_weighted(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000],
            [-0.0721139528911510, 0.1675353295557450, -0.0954213766645937,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954310001289640, 0.1953940094300550,
             -0.0999630093010912, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999645379456421,
                0.2016435931141710, -0.1016790551685290],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554663120, 0.1016790554663120],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._weighted_heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284593E+09, 3.21846886402014E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886402014E+08, 2.06909317632500E+08,
                2.15090157481671E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157481671E+07,
                5.71758161659235E+07, 9.43574147951588E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951588E+06, 1.74614402201079E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284598E+09, 3.21846886402202E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886402202E+08, 2.06909317633054E+08,
                2.15090157479978E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157479978E+07,
                5.71758161653827E+07, 9.43574147951432E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951432E+06, 1.74614402201151E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_heat_storage_matrix_weighted(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 1.15082401284375E+09, 3.21846886401436E+08,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886401436E+08, 2.06909317632248E+08,
                2.15090157480912E+07, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157480912E+07,
                5.71758161657388E+07, 9.43574147951655E+06],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147951655E+06, 1.74614402201115E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._weighted_heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [2.1204012345643E+10, 1.0602006172876E+10, 0.0000000000000E+00,
                0.0000000000000E+00, 0.0000000000000E+00],
            [1.0602006172876E+10, 1.1508240128437E+12, 3.2184688640148E+11,
                0.0000000000000E+00, 0.0000000000000E+00],
            [0.0000000000000E+00, 3.2184688640148E+11, 2.0690931763215E+11,
                2.1509015748141E+10, 0.0000000000000E+00],
            [0.0000000000000E+00, 0.0000000000000E+00, 2.1509015748141E+10,
                5.7175816165638E+10, 9.4357414795674E+09],
            [0.0000000000000E+00, 0.0000000000000E+00, 0.0000000000000E+00,
                9.4357414795674E+09, 1.7461440220061E+10],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-13, atol=1e-3,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [2.12040123457151E+10, 1.06020061728035E+10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728035E+10, 1.15082401284384E+12, 3.21846886401388E+11,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 3.21846886401388E+11, 2.06909317632346E+11,
                2.15090157480413E+10, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.15090157480413E+10,
                5.71758161658397E+10, 9.43574147946571E+09],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43574147946571E+09, 1.74614402201624E+10],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-13, atol=1e-3,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
        expected_flux_vector_0[-1] = -2.74983450612514 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector_0,
                                    self.msh._heat_flux_vector_0))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.74983450612506 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -0.5 * (2.74983450612514
                                           + 2.74983450612506) * 25.0e-3
        print(self.msh._heat_flux_vector)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_heat_flux_vector))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -1.3214111328125E-01,
            4.1467285156250E-01,
            5.2429199218750E-01,
            -4.6118164062500E-01,
            5.8278333356252E-02,
        ])
        print(self.msh._residual_heat_flux_vector)
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
            rtol=1e-13, atol=1e-3,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.00000E+00,
            -1.22372E-12,
            5.66406E-12,
            -1.17999E-11,
            9.71392E-12,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.33062137400659E-12
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
        self.assertEqual(self.msh._iter, 2)


class TestInitializeGlobalSystemCubic(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
            spec_grav_solids=2.65,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
        )
        self.msh = ThermalAnalysis1D(
            z_range=(0, 100),
            num_elements=4,
            generate=True,
        )
        initial_temp_vector = np.array([
            -2.000000000000000,
            -9.157452320220460,
            -10.488299785319000,
            -7.673205119057850,
            -3.379831977359920,
            0.186084957826655,
            1.975912628300400,
            2.059737589813890,
            1.158320034961550,
            0.100523127786268,
            -0.548750924584512,
            -0.609286860003055,
            -0.205841501790609,
        ])
        initial_temp_rate_vector = np.array([
            -0.02000000000000000,
            -0.09157452320220460,
            -0.10488299785319000,
            -0.07673205119057850,
            -0.03379831977359920,
            0.00186084957826655,
            0.01975912628300400,
            0.02059737589813890,
            0.01158320034961550,
            0.00100523127786268,
            -0.00548750924584512,
            -0.00609286860003055,
            -0.00205841501790609,
        ])
        # TODO: Start edditing here
        for nd, T0, dTdt0 in zip(self.msh.nodes,
                                 initial_temp_vector,
                                 initial_temp_rate_vector,
                                 ):
            nd.temp = T0
            nd.temp_rate = dTdt0
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
        bnd0 = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=-2.0,
        )
        self.msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            nodes=(self.msh.nodes[-1],),
            int_pts=(self.msh.elements[-1].int_pts[-1],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
            bnd_value=25.0e-3,
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

    def test_temperature_distribution_nodes(self):
        expected_temp_vector = np.array([
            2.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_vector,
                                    actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector,
                                    self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            1.5984827557301400,
            0.5015172442698560,
            -0.0901923788646684,
            -0.6098076211353320,
            -0.9479274057836310,
            -1.3520725942163700,
            -3.7189110867544700,
            -9.7810889132455400,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.05,
            0.02,
            0.01,
            -0.08,
            -0.05,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            0.04366025403784440,
            0.02633974596215560,
            0.01788675134594810,
            0.01211324865405190,
            -0.00901923788646684,
            -0.06098076211353320,
            -0.07366025403784440,
            -0.05633974596215560,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -0.0760000,
            -0.0760000,
            -0.0360000,
            -0.0360000,
            -0.0280000,
            -0.0280000,
            -0.4200000,
            -0.4200000,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.000000000000000,
            1.000000000000000,
            0.314715929845879,
            0.113801777607921,
            0.089741864676250,
            0.074104172041942,
            0.042882888566470,
            0.025322726744343,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_deg_sat_water_temp_grad_distribution(self):
        expected_deg_sat_water_temp_grad_int_pts = np.array([
            0.000000000000000,
            0.000000000000001,
            1.810113168088841,
            0.100397364897923,
            0.051013678629442,
            0.029567794445951,
            0.006250998062539,
            0.001419738751606,
        ])
        actual_deg_sat_water_temp_grad_int_pts = np.array([
            ip.deg_sat_water_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_temp_grad_int_pts,
                                    expected_deg_sat_water_temp_grad_int_pts))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            0.0000000000E+00,
            0.0000000000E+00,
            -4.8910645169E-12,
            -5.5518497692E-13,
            8.3577053338E-13,
            1.7708810805E-13,
            2.0670411271E-16,
            6.0318175808E-27,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_water_flux_int_pts,
                                    expected_water_flux_int_pts,
                                    atol=1e-30))

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            1.94419643704324,
            1.94419643704324,
            2.48085630059944,
            2.66463955659925,
            2.68754164945741,
            2.70253225701445,
            2.73271219424962,
            2.74983450612514,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts,
                                    atol=1e-30))

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array([
            [0.37541276634706, -0.47913494460783,
                0.13662447765333, -0.03290229939256],
            [-0.47913495073106, 1.09749384355934,
             -0.75556012955112, 0.13720123672283],
            [0.13662448026901, -0.75556012965636,
                1.10036574587615, -0.48143009648880],
            [-0.0329022999455621, 0.13720123675250,
             -0.48143009649197, 0.37713115968503],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.37422788138613, -0.47040475255461,
                0.12454905038507, -0.02837217921659],
            [-0.47040445574570, 1.04580620489708,
             -0.68911230909446, 0.11371055994309],
            [0.12454845679017, -0.68910429557128,
                0.91904216745687, -0.35448632867575],
            [-0.0283721462389552, 0.11370996634735,
             -0.35448603187892, 0.26914821177052],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.26682162569726, -0.34073842741069,
                0.09735383640305, -0.02343703468962],
            [-0.34073842741069, 0.77883069122443,
             -0.53544610021680, 0.09735383640305],
            [0.09735383640305, -0.53544610021680,
                0.77883069122443, -0.34073842741069],
            [-0.0234370346896243, 0.09735383640305,
             -0.34073842741069, 0.26682162569726],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.32902830441547, -0.41311979105240,
                0.11114097767973, -0.02704949104280],
            [-0.41352446262158, 0.98699528504461,
             -0.69720784332217, 0.12373702089915],
            [0.11130058733431, -0.69723220657608,
                1.04220001681837, -0.45626839757660],
            [-0.0270752880707999, 0.12370525491951,
             -0.45616559239695, 0.35953562554824],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array([
            [4.73637171623350E+06, 3.48911975196258E+06,
             -1.29625396875553E+06, 6.28539963771457E+05],
            [3.48911975196258E+06, 2.04661850623021E+07,
             -2.68983544335460E+06, -1.08386996099026E+06],
            [-1.29625396875553E+06, -2.68983544335460E+06,
                1.99779575443545E+07, 3.06435173643206E+06],
            [6.28539963771457E+05, -1.08386996099026E+06,
                3.06435173643206E+06, 4.00238174498494E+06],
        ])
        expected_C[3:7, 3:7] += np.array([
            [4.16942926772367E+06, 2.79778776603154E+06,
             -1.54215963369236E+06, 7.18714827570726E+05],
            [2.79778776603154E+06, 2.63422846762999E+07,
                8.59594504059493E+05, -1.83532284105249E+06],
            [-1.54215963369236E+06, 8.59594504059493E+05,
                2.79087377467474E+07, 3.38411418075181E+06],
            [7.18714827570726E+05, -1.83532284105249E+06,
                3.38411418075181E+06, 4.88507845309510E+06],
        ])
        expected_C[6:10, 6:10] += np.array([
            [4.84663139329808E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442682E+05],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857144E+06, -1.36311507936508E+06],
            [-1.36311507936508E+06, -3.06700892857144E+06,
                2.45360714285714E+07, 3.74856646825397E+06],
            [7.19421847442682E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 4.84663139329808E+06],
        ])
        expected_C[9:13, 9:13] += np.array([
            [1.04462849763812E+09, 5.23841479165073E+08,
             -2.37985788657434E+08, 5.50357660694582E+07],
            [5.23841479165073E+08, 3.84440750802666E+08,
             -1.43609705550615E+08, 1.42597922699745E+07],
            [-2.37985788657434E+08, -1.43609705550615E+08,
                1.69013732188852E+08, 1.93503173102562E+07],
            [5.50357660694582E+07, 1.42597922699745E+07,
                1.93503173102562E+07, 5.14006065936711E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.61109074784318 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))


if __name__ == "__main__":
    unittest.main()
