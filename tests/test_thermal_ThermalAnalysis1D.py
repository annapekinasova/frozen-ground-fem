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


if __name__ == "__main__":
    unittest.main()
