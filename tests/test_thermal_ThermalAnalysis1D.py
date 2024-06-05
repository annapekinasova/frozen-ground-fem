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
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
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
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
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
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
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
        self.assertEqual(self.msh._residual_heat_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_temp_vector.shape, (nnod,))
        self.assertEqual(self.msh._heat_flow_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_flow_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._heat_storage_matrix.shape, (nnod, nnod))
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
        self.msh.update_boundary_conditions(t)
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
        self.msh.update_boundary_conditions(t)
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
        self.msh.update_boundary_conditions(t)
        self.msh.update_heat_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._heat_flux_vector,
                                          self.msh._heat_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == 5:
                self.assertAlmostEqual(fx, -self.fixed_flux)
            elif k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, -self.flux_geotherm)
            else:
                self.assertEqual(fx, 0.0)
        self.msh.update_boundary_conditions(t)
        self.msh.update_heat_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._heat_flux_vector,
                                          self.msh._heat_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == 5:
                self.assertAlmostEqual(fx, -self.fixed_flux)
            elif k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, -self.flux_geotherm)
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
                ip.temp_gradient = 0.003

    def test_initial_heat_flow_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix, expected))

    def test_initial_heat_storage_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected))

    def test_initial_heat_flux_vector(self):
        expected = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(self.msh._heat_flux_vector_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_flux_vector, expected))

    def test_update_heat_flow_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        k00 = 0.1935775350469950
        d0 = 2 * np.ones((self.msh.num_nodes,)) * k00
        d0[0] = k00
        d0[-1] = k00
        dp1 = -np.ones((self.msh.num_nodes - 1,)) * k00
        dm1 = -np.ones((self.msh.num_nodes - 1,)) * k00
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

    def test_update_heat_flux_vector(self):
        expected0 = np.zeros(self.msh.num_nodes)
        expected1 = np.ones(self.msh.num_nodes) * 0.0018217333333333300
        expected1[0] = 0.0009108666666666670
        expected1[-1] = 0.0009108666666666670
        self.msh.update_heat_flux_vector()
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector_0, expected0))
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector, expected1,
            rtol=1e-13, atol=1e-16,
        ))


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
        self.msh.update_heat_flux_vector()

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

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.00000000000000000,
            0.00000000000000000,
            1.96985868037600000,
            0.37676651903183400,
            0.24895666952857000,
            0.17754007257781200,
            0.06672422855669150,
            0.02583496685516250,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

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
            [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
                0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
             -0.0999640592805206, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
                0.2016431146839040, -0.1016790554033840],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554033840, 0.1016790554033840],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00],
            [1.06020061728395E+07, 4.75157069027845E+09, 1.49253092352162E+09,
                0.00000000000000E+00, 0.00000000000000E+00],
            [0.00000000000000E+00, 1.49253092352162E+09, 1.87400276780934E+09,
                2.77995607535567E+08, 0.00000000000000E+00],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.77995607535567E+08,
                6.55976597222048E+08, 6.67084599390764E+07],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                6.67084599390764E+07, 8.85939210208664E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array([
            -0.000000000000000E+00,
            -7.240998977095220E-06,
            -1.693636195537800E-06,
            4.516088940272450E-07,
            9.283918373080160E-10,
        ])
        self.assertTrue(np.allclose(
            expected_Phi, self.msh._heat_flux_vector,
            atol=1e-15, rtol=1e-6,
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
                ip.temp_gradient = 0.003

    def test_initial_heat_flow_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_flow_matrix, expected))

    def test_initial_heat_storage_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected))

    def test_initial_heat_flux_vector(self):
        expected = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(self.msh._heat_flux_vector_0, expected))
        self.assertTrue(np.allclose(self.msh._heat_flux_vector, expected))

    def test_update_heat_flow_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        h00 = 0.7162368796738820
        h11 = 2.0906373785075500
        h10 = -0.9146538530970510
        h20 = 0.2613296723134430
        h30 = -0.0629126988902734
        h21 = -1.4373131977239400
        d0 = np.ones((self.msh.num_nodes,)) * (2 * h00)
        d0[0] = h00
        d0[-1] = h00
        d0[1::3] = h11
        d0[2::3] = h11
        d1 = np.ones((self.msh.num_nodes - 1,)) * h10
        d1[1::3] = h21
        d2 = np.ones((self.msh.num_nodes - 2,)) * h20
        d2[2::3] = 0.0
        d3 = np.zeros((self.msh.num_nodes - 3,))
        d3[0::3] = h30
        expected1 = np.diag(d0)
        expected1 += np.diag(d1, -1) + np.diag(d1, 1)
        expected1 += np.diag(d2, -2) + np.diag(d2, 2)
        expected1 += np.diag(d3, -3) + np.diag(d3, 3)
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

    def test_update_heat_flux_vector(self):
        expected0 = np.zeros(self.msh.num_nodes)
        expected1 = np.ones(self.msh.num_nodes) * 0.00068315
        expected1[3::3] = 0.000455433333333333
        expected1[0] = 0.000227716666666667
        expected1[-1] = 0.000227716666666667
        self.msh.update_heat_flux_vector()
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector_0, expected0))
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector, expected1,
            rtol=1e-13, atol=1e-16
        ))


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

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.07235260409982430,
            0.03289271388265000,
            0.02421242458534380,
            0.02531316492062430,
            0.03052609000431350,
            0.03558952835188600,
            0.05338816551016490,
            0.16710056507182700,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            3.90566915045107000,
            0.53439071533528300,
            0.35766602066855600,
            0.43005975899489600,
            0.75161061396167100,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

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
            0.0,
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
            0.0211324865405187,
            0.0788675134594813,
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
            0.0040000,
            0.0040000,
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

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

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
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
             -0.0999640592805206, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
                0.2016431146839040, -0.1016790554033840,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554033840, 0.1016790554033840,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
                8.29817132739774E+06, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
                3.29568618779144E+07, 8.17817064124763E+06,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                8.17817064124763E+06, 1.63181792594571E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            -0.000000000000000E+00,
            -7.240998977095220E-06,
            -1.693636195537800E-06,
            4.516088940272450E-07,
            6.874586265312840E-02,
        ])
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
        self.msh.time_step = 3.024E+05
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5 + 3.024e5)

    def test_iteration_variables(self):
        self.assertEqual(self.msh._eps_a, 1.0)
        self.assertEqual(self.msh._iter, 0)

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        expected_temp_vector_1 = np.array([
            2.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_vector_1,
                                    actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector_0,
                                    self.msh._temp_vector_0))
        self.assertTrue(np.allclose(expected_temp_vector_1,
                                    self.msh._temp_vector))

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

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.00000000000000,
            0.00000000000000,
            0.00000000000000,
            0.00000000000000,
            0.00000000000000,
            0.00000000000000,
            0.00000000000000,
            0.00000000000000,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

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
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
             -0.0999640592805206, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
                0.2016431146839040, -0.1016790554033840,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554033840, 0.1016790554033840,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
                8.29817132739774E+06, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
                3.29568618779144E+07, 8.17817064124763E+06,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                8.17817064124763E+06, 1.63181792594571E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            -0.000000000000000E+00,
            -7.240998977095220E-06,
            -1.693636195537800E-06,
            4.516088940272450E-07,
            6.874586265312840E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector_0))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))


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
        self.msh.time_step = 3.024E+05
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
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
            0.0,
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

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.000000000000000000,
            0.000000000000000000,
            0.340923364017748000,
            0.187553330175011000,
            0.028947898402562300,
            0.012907964849384700,
            0.002679172391015270,
            0.000615976343614986,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

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

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
             -0.0999640592805206, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
                0.2016431146839040, -0.1016790554033840,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554033840, 0.1016790554033840,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1507501456631930, -0.0786361927720423,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0786361927720423, 0.1767867533269640,
             -0.0981505605549217, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0981505605549217,
                0.1992818037997590, -0.1011312432448370,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1011312432448370, 0.1011312432448370,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
                8.29817132739774E+06, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
                3.29568618779144E+07, 8.17817064124763E+06,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                8.17817064124763E+06, 1.63181792594571E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 8.78602658519930E+08, 3.44200503749112E+08,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 3.44200503749112E+08, 6.06873440461168E+08,
                3.48953707771033E+07, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 3.48953707771033E+07,
                7.50250521945132E+07, 1.03000212013861E+07,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.03000212013861E+07, 1.82864440804706E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [5.263894652352350E-04, -5.403980334474360E-04,
                2.370573772919760E-05, -1.493205599042890E-05,
             5.234886473432850E-06,],
            [-2.432926732320180E-05, 5.234640374760480E-05,
             -4.741147545839510E-05,
                2.986411198085780E-05, -1.046977294686570E-05,],
            [1.421074304030880E-05, -5.075229116649230E-05,
                8.574864747653640E-05, -7.577088977750250E-05,
             2.656379042714970E-05,],
            [-7.163599248898670E-06, 2.558410027180900E-05,
             -2.576091232012540E-04,
                5.978940173930280E-04, -3.587053952146850E-04,],
            [4.034968407044770E-06, -1.441049851236240E-05,
                1.451008965421070E-04, -1.172965336484240E-03,
             1.038239970047450E-03,],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-14, atol=1e-17,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [1.000526389465240E+00, -5.403980334474360E-04,
                2.370573772919760E-05, -1.493205599042890E-05,
             5.234886473432850E-06,],
            [-2.432926732320180E-05, 1.000052346403750E+00,
             -4.741147545839510E-05,
                2.986411198085780E-05, -1.046977294686570E-05,],
            [1.421074304030880E-05, -5.075229116649230E-05,
                1.000085748647480E+00, -7.577088977750250E-05,
             2.656379042714970E-05,],
            [-7.163599248898670E-06, 2.558410027180900E-05,
             -2.576091232012540E-04,
                1.000597894017390E+00, -3.587053952146850E-04,],
            [4.034968407044770E-06, -1.441049851236240E-05,
                1.451008965421070E-04, -1.172965336484240E-03,
             1.001038239970050E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-14, atol=1e-14,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array([
            -0.000000000000000E+00,
            -7.240998977095220E-06,
            -1.693636195537800E-06,
            4.516088940272450E-07,
            6.874586265312840E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector_0,
                                    self.msh._heat_flux_vector_0))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            0.000000000000000E+00,
            -6.548583091546750E-06,
            -2.895509358076450E-05,
            -2.831704596426130E-06,
            6.846293074257740E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))


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
        self.msh.time_step = 3.024E+05
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
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        expected_temp_vector = np.array([
            2.0000000000000000,
            0.0988263131322971,
            -0.7971699198965330,
            -1.5126232123463600,
            -11.9736027887043000,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            expected_temp_vector,
            actual_temp_nodes,
        ))
        self.assertTrue(np.allclose(
            expected_temp_vector_0,
            self.msh._temp_vector_0,
        ))
        self.assertTrue(np.allclose(
            expected_temp_vector,
            self.msh._temp_vector,
        ))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            6.61375661375661E-06,
            -3.88123964187482E-09,
            9.35873050088320E-09,
            -4.17434270713069E-08,
            8.72923653959047E-08,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            expected_temp_rate_vector,
            actual_temp_rate_nodes,
            atol=1e-12, rtol=1e-3,
        ))
        self.assertTrue(np.allclose(
            expected_temp_rate_vector,
            self.msh._temp_rate_vector,
            atol=1e-12, rtol=1e-3,
        ))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
             -0.0999640592805206, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
                0.2016431146839040, -0.1016790554033840,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554033840, 0.1016790554033840,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1675482489237170, -0.0954342960325658,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0954342960325658, 0.1954036559952800,
             -0.0999693599627141, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0999693599627141,
                0.2016484400499960, -0.1016790800872820,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790800872820, 0.1016790800872820,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
                8.29817132739774E+06, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
                3.29568618779144E+07, 8.17817064124763E+06,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                8.17817064124763E+06, 1.63181792594571E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 1.14790024626751E+09, 3.21101921370035E+08,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 3.21101921370035E+08, 2.06835155994880E+08,
                2.14800761090158E+07, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.14800761090158E+07,
                5.70746434252292E+07, 9.43514558536205E+06,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43514558536205E+06, 1.74625389157076E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [5.263894652352350E-04, -5.403980334474360E-04,
                2.370573772919760E-05, -1.493205599042890E-05,
             5.234886473432850E-06,],
            [-2.432926732320180E-05, 5.234640374760480E-05,
             -4.741147545839510E-05,
                2.986411198085780E-05, -1.046977294686570E-05,],
            [1.421074304030880E-05, -5.075229116649230E-05,
                8.574864747653640E-05, -7.577088977750250E-05,
             2.656379042714970E-05,],
            [-7.163599248898670E-06, 2.558410027180900E-05,
             -2.576091232012540E-04,
                5.978940173930280E-04, -3.587053952146850E-04,],
            [4.034968407044770E-06, -1.441049851236240E-05,
                1.451008965421070E-04, -1.172965336484240E-03,
             1.038239970047450E-03,],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-14, atol=1e-17,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [1.000526389465240E+00, -5.403980334474360E-04,
                2.370573772919760E-05, -1.493205599042890E-05,
             5.234886473432850E-06,],
            [-2.432926732320180E-05, 1.000052346403750E+00,
             -4.741147545839510E-05,
                2.986411198085780E-05, -1.046977294686570E-05,],
            [1.421074304030880E-05, -5.075229116649230E-05,
                1.000085748647480E+00, -7.577088977750250E-05,
             2.656379042714970E-05,],
            [-7.163599248898670E-06, 2.558410027180900E-05,
             -2.576091232012540E-04,
                1.000597894017390E+00, -3.587053952146850E-04,],
            [4.034968407044770E-06, -1.441049851236240E-05,
                1.451008965421070E-04, -1.172965336484240E-03,
             1.001038239970050E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-14, atol=1e-14,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array([
            -0.000000000000000E+00,
            -7.240998977095210E-06,
            -1.693636195537800E-06,
            4.516088940272450E-07,
            6.874586265312840E-02,
        ])
        self.assertTrue(np.allclose(
            expected_flux_vector_0,
            self.msh._heat_flux_vector_0
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            0.000000000000000E+00,
            2.143145921199510E-05,
            8.271535830408110E-06,
            2.236869459407050E-07,
            6.874523161026890E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -2.00010331808334E+00,
            -5.01130348395403E-01,
            -5.97300376134295E-01,
            -7.10765507610895E-01,
            -5.97904836728674E+00,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.00000000000000E+00,
            -5.01173686867703E-01,
            -5.97169919896533E-01,
            -7.12623212346363E-01,
            -5.97360278870428E+00,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 4.9481302578941E-01
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
        self.msh.time_step = 3.024E+05
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
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()
        self.msh.iterative_correction_step()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
            0.0,
            0.1,
            -0.8,
            -1.5,
            -12,
        ])
        expected_temp_vector = np.array([
            2.0000000000000000,
            0.0986658199546282,
            -0.7965229034856440,
            -1.5139885335425800,
            -11.9724600868138000,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            expected_temp_vector,
            actual_temp_nodes,
        ))
        self.assertTrue(np.allclose(
            expected_temp_vector_0,
            self.msh._temp_vector_0,
        ))
        self.assertTrue(np.allclose(
            expected_temp_vector,
            self.msh._temp_vector,
        ))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            6.61375661375661E-06,
            -4.41197104950990E-09,
            1.14983350342468E-08,
            -4.62583781170062E-08,
            9.10711414888197E-08,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            expected_temp_rate_vector,
            actual_temp_rate_nodes,
            atol=1e-16, rtol=1e-10,
        ))
        self.assertTrue(np.allclose(
            expected_temp_rate_vector,
            self.msh._temp_rate_vector,
            atol=1e-16, rtol=1e-10,
        ))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
             -0.0999640592805206, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
                0.2016431146839040, -0.1016790554033840,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016790554033840, 0.1016790554033840,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
                0.0000000000000000, 0.0000000000000000,],
            [-0.0721139528911510, 0.1675471042926980, -0.0954331514015465,
                0.0000000000000000, 0.0000000000000000,],
            [0.0000000000000000, -0.0954331514015465, 0.1954028010121700,
             -0.0999696496106236, 0.0000000000000000,],
            [0.0000000000000000, 0.0000000000000000, -0.0999696496106236,
                0.2016488078173110, -0.1016791582066870,],
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
             -0.1016791582066870, 0.1016791582066870,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
                8.29817132739774E+06, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
                3.29568618779144E+07, 8.17817064124763E+06,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                8.17817064124763E+06, 1.63181792594571E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.06020061728395E+07, 1.14799363337692E+09, 3.21136209426181E+08,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 3.21136209426181E+08, 2.06883882586324E+08,
                2.14790000604973E+07, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 2.14790000604973E+07,
                5.70646987491469E+07, 9.43497504608584E+06,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.43497504608584E+06, 1.74625357643405E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [5.273775865857460E-04, -5.578042805506230E-04,
                5.829210337869900E-05, -4.063100921329330E-05,
             1.276559979947190E-05,],
            [-2.630551002422450E-05, 8.715889795397970E-05,
             -1.165842067573980E-04,
                8.126201842658660E-05, -2.553119959894370E-05,],
            [4.266935330377370E-05, -2.142702244487380E-04,
                3.699117100917840E-04, -2.891602777006950E-04,
             9.084943875387490E-05,],
            [-1.763366821733360E-05, 8.854997215175700E-05,
             -4.436809550911480E-04,
                8.659072516224760E-04, -4.931426004657510E-04,],
            [9.527608077932960E-06, -4.784423862214370E-05,
                2.397242706197770E-04, -1.348248270501960E-03,
             1.146840630426390E-03,],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-8, atol=1e-11,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [1.000527377586590E+00, -5.578042805506230E-04,
                5.829210337869900E-05, -4.063100921329330E-05,
             1.276559979947190E-05,],
            [-2.630551002422450E-05, 1.000087158897950E+00,
             -1.165842067573980E-04,
                8.126201842658660E-05, -2.553119959894370E-05,],
            [4.266935330377370E-05, -2.142702244487380E-04,
                1.000369911710090E+00, -2.891602777006950E-04,
             9.084943875387490E-05,],
            [-1.763366821733360E-05, 8.854997215175700E-05,
             -4.436809550911480E-04,
                1.000865907251620E+00, -4.931426004657510E-04,],
            [9.527608077932960E-06, -4.784423862214370E-05,
                2.397242706197770E-04, -1.348248270501960E-03,
             1.001146840630430E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-8, atol=1e-11,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array([
            -0.000000000000000E+00,
            -7.240998977095210E-06,
            -1.693636195537800E-06,
            4.516088940272450E-07,
            6.874586265312840E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector_0,
                                    self.msh._heat_flux_vector_0))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            0.000000000000000E+00,
            2.133070477819620E-05,
            8.181590920372290E-06,
            2.187351973225760E-07,
            6.874521027822230E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -2.00025900344127E+00,
            -1.60722721278623E-04,
            6.47788749226980E-04,
            -1.36736823325139E-03,
            1.14601596360676E-03,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.00000000000000E+00,
            -1.60493177668852E-04,
            6.47016410889105E-04,
            -1.36532119621939E-03,
            1.14270189049772E-03,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.5508312528406E-04
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
                                    expected_temp_int_pts,
                                    ))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
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
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes,
                                    ))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector,
                                    ))

    def test_temperature_rate_distribution_int_pts(self):
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
                                    expected_temp_rate_int_pts,
                                    ))

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
                                    expected_temp_gradient_int_pts,
                                    ))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            0.044857035897863,
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
                                    expected_deg_sat_water_int_pts,
                                    ))

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.zeros(
            self.msh.num_elements * 5
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

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

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            2.73079394984140,
            2.74627913402004,
            2.75071278269289,
            2.75011411247605,
            2.74742845413695,
            2.74501452413187,
            2.73757048307299,
            2.70492452266610,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            2.29701505120963,
            2.64038419594985,
            2.66783269125494,
            2.65605663067911,
            2.61109074784318,
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
            [0.3754127694384990, -0.4791349492121190,
                0.1366244795670250, -0.0329022997934044,],
            [-0.4791349492121190, 1.0974938412970900,
             -0.7555601286108480, 0.1372012365258810,],
            [0.1366244795670250, -0.7555601286108480,
                1.1003657454416000, -0.4814300963977770,],
            [-0.0329022997934044, 0.1372012365258810,
             -0.4814300963977770, 0.3771311596653000,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742278648957170, -0.4704043073588020,
                0.1245486051912600, -0.0283721627281755,],
            [-0.4704043073588020, 1.0458021981448800,
             -0.6891083023287820, 0.1137104115427070,],
            [0.1245486051912600, -0.6891083023287820,
                0.9190461742111960, -0.3544864770736730,],
            [-0.0283721627281755, 0.1137104115427070,
             -0.3544864770736730, 0.2691482282591420,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972590, -0.3407384274106880,
                0.0973538364030538, -0.0234370346896240,],
            [-0.3407384274106880, 0.7788306912244300,
             -0.5354461002167960, 0.0973538364030537,],
            [0.0973538364030538, -0.5354461002167960,
                0.7788306912244310, -0.3407384274106880,],
            [-0.0234370346896240, 0.0973538364030537,
             -0.3407384274106880, 0.2668216256972590,],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3292202434390340, -0.4133999362112650,
                0.1112498768967810, -0.0270701841245502,],
            [-0.4133999362112650, 0.9868437905990680,
             -0.6971611631648800, 0.1237173087770760,],
            [0.1112498768967810, -0.6971611631648800,
                1.0421402435851100, -0.4562289573170110,],
            [-0.0270701841245502, 0.1237173087770760,
             -0.4562289573170110, 0.3595818326644850,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array([
            [3.74901860610780E+06, 2.89593744482105E+06,
             -1.05358398557321E+06, 5.54643193641815E+05,],
            [2.89593744482105E+06, 1.88936621803465E+07,
             -2.36630842102390E+06, -1.04792564897764E+06,],
            [-1.05358398557321E+06, -2.36630842102390E+06,
                1.88757400422500E+07, 2.88462077162992E+06,],
            [5.54643193641815E+05, -1.04792564897764E+06,
                2.88462077162992E+06, 3.73110189762532E+06,],
        ])
        expected_C[3:7, 3:7] += np.array([
            [3.74548978664947E+06, 2.83529217357612E+06,
             -8.63799618191817E+05, 6.29634934688451E+05,],
            [2.83529217357612E+06, 1.92531430053028E+07,
             -3.32307881157745E+06, -1.41571111930997E+06,],
            [-8.63799618191817E+05, -3.32307881157745E+06,
                2.30168957941050E+07, 3.93911517581243E+06,],
            [6.29634934688451E+05, -1.41571111930997E+06,
                3.93911517581243E+06, 4.82119852843649E+06,],
        ])
        expected_C[6:10, 6:10] += np.array([
            [4.84663139329806E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442681E+05,],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857143E+06, -1.36311507936508E+06,],
            [-1.36311507936508E+06, -3.06700892857143E+06,
                2.45360714285714E+07, 3.74856646825397E+06,],
            [7.19421847442681E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 4.84663139329806E+06,],
        ])
        expected_C[9:13, 9:13] += np.array([
            [4.26389587430359E+06, 3.18874298722127E+06,
             -1.17518017569044E+06, 5.93504826831539E+05,],
            [3.18874298722127E+06, 1.95951789440892E+07,
             -2.51514790752116E+06, -1.07640022713129E+06,],
            [-1.17518017569044E+06, -2.51514790752116E+06,
                1.94626651593996E+07, 2.99118309010297E+06,],
            [5.93504826831539E+05, -1.07640022713129E+06,
                2.99118309010297E+06, 3.89099624195036E+06,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            1.89850411575195E-08,
            9.32804654257637E-09,
            -4.31099786843920E-09,
            -1.42748247257048E-06,
            1.28559187259822E-05,
            1.28558305047007E-05,
            -1.42842430458346E-06,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
            1.10932858948307E-04,
            7.12664676651715E-05,
            -1.57821060328194E-05,
            6.53002632662146E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))


class TestInitializeTimeStepCubic(unittest.TestCase):
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
        self.msh.time_step = 3.024E+05
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5 + 3.024e5)

    def test_iteration_variables(self):
        self.assertEqual(self.msh._eps_a, 1.0)
        self.assertEqual(self.msh._iter, 0)

    def test_temperature_distribution_nodes(self):
        expected_temp_vector = np.array([
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

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
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
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector))

    def test_temperature_rate_distribution_int_pts(self):
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

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.zeros(
            self.msh.num_elements * 5
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

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

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            2.73079394984140,
            2.74627913402004,
            2.75071278269289,
            2.75011411247605,
            2.74742845413695,
            2.74501452413187,
            2.73757048307299,
            2.70492452266610,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            2.29701505120963,
            2.64038419594985,
            2.66783269125494,
            2.65605663067911,
            2.61109074784318,
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
            [0.3754127694384990, -0.4791349492121190,
                0.1366244795670250, -0.0329022997934044,],
            [-0.4791349492121190, 1.0974938412970900,
             -0.7555601286108480, 0.1372012365258810,],
            [0.1366244795670250, -0.7555601286108480,
                1.1003657454416000, -0.4814300963977770,],
            [-0.0329022997934044, 0.1372012365258810,
             -0.4814300963977770, 0.3771311596653000,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742278648957170, -0.4704043073588020,
                0.1245486051912600, -0.0283721627281755,],
            [-0.4704043073588020, 1.0458021981448800,
             -0.6891083023287820, 0.1137104115427070,],
            [0.1245486051912600, -0.6891083023287820,
                0.9190461742111960, -0.3544864770736730,],
            [-0.0283721627281755, 0.1137104115427070,
             -0.3544864770736730, 0.2691482282591420,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972590, -0.3407384274106880,
                0.0973538364030538, -0.0234370346896240,],
            [-0.3407384274106880, 0.7788306912244300,
             -0.5354461002167960, 0.0973538364030537,],
            [0.0973538364030538, -0.5354461002167960,
                0.7788306912244310, -0.3407384274106880,],
            [-0.0234370346896240, 0.0973538364030537,
             -0.3407384274106880, 0.2668216256972590,],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3292202434390340, -0.4133999362112650,
                0.1112498768967810, -0.0270701841245502,],
            [-0.4133999362112650, 0.9868437905990680,
             -0.6971611631648800, 0.1237173087770760,],
            [0.1112498768967810, -0.6971611631648800,
                1.0421402435851100, -0.4562289573170110,],
            [-0.0270701841245502, 0.1237173087770760,
             -0.4562289573170110, 0.3595818326644850,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array([
            [3.74901860610780E+06, 2.89593744482105E+06,
             -1.05358398557321E+06, 5.54643193641815E+05,],
            [2.89593744482105E+06, 1.88936621803465E+07,
             -2.36630842102390E+06, -1.04792564897764E+06,],
            [-1.05358398557321E+06, -2.36630842102390E+06,
                1.88757400422500E+07, 2.88462077162992E+06,],
            [5.54643193641815E+05, -1.04792564897764E+06,
                2.88462077162992E+06, 3.73110189762532E+06,],
        ])
        expected_C[3:7, 3:7] += np.array([
            [3.74548978664947E+06, 2.83529217357612E+06,
             -8.63799618191817E+05, 6.29634934688451E+05,],
            [2.83529217357612E+06, 1.92531430053028E+07,
             -3.32307881157745E+06, -1.41571111930997E+06,],
            [-8.63799618191817E+05, -3.32307881157745E+06,
                2.30168957941050E+07, 3.93911517581243E+06,],
            [6.29634934688451E+05, -1.41571111930997E+06,
                3.93911517581243E+06, 4.82119852843649E+06,],
        ])
        expected_C[6:10, 6:10] += np.array([
            [4.84663139329806E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442681E+05,],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857143E+06, -1.36311507936508E+06,],
            [-1.36311507936508E+06, -3.06700892857143E+06,
                2.45360714285714E+07, 3.74856646825397E+06,],
            [7.19421847442681E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 4.84663139329806E+06,],
        ])
        expected_C[9:13, 9:13] += np.array([
            [4.26389587430359E+06, 3.18874298722127E+06,
             -1.17518017569044E+06, 5.93504826831539E+05,],
            [3.18874298722127E+06, 1.95951789440892E+07,
             -2.51514790752116E+06, -1.07640022713129E+06,],
            [-1.17518017569044E+06, -2.51514790752116E+06,
                1.94626651593996E+07, 2.99118309010297E+06,],
            [5.93504826831539E+05, -1.07640022713129E+06,
                2.99118309010297E+06, 3.89099624195036E+06,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            1.89850411575195E-08,
            9.32804654257637E-09,
            -4.31099786843920E-09,
            -1.42748247257048E-06,
            1.28559187259822E-05,
            1.28558305047007E-05,
            -1.42842430458346E-06,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
            1.10932858948307E-04,
            7.12664676651715E-05,
            -1.57821060328194E-05,
            6.53002632662146E-02,
        ])
        self.assertTrue(np.allclose(
            expected_flux_vector,
            self.msh._heat_flux_vector,
        ))
        self.assertTrue(np.allclose(
            expected_flux_vector,
            self.msh._heat_flux_vector_0,
        ))


class TestUpdateWeightedMatricesCubic(unittest.TestCase):
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
        self.msh.time_step = 3.024E+05
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array([
            -2.000000000000000,
            -9.157543894743660,
            -10.488404668316800,
            -7.673281851109040,
            -3.379865775679690,
            0.186086818676234,
            1.975932387426680,
            2.059758187189790,
            1.158331618161900,
            0.100524133017546,
            -0.548756412093758,
            -0.609292952871655,
            -0.205843560205627,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            0.00000000000000E+00,
            -9.15745232017429E-02,
            -1.04882997852940E-01,
            -7.67320511902980E-02,
            -3.37983197735703E-02,
            1.86084957826127E-03,
            1.97591262829366E-02,
            2.05973758982125E-02,
            1.15832003495520E-02,
            1.00523127785634E-03,
            -5.48750924589392E-03,
            -6.09286859998282E-03,
            -2.05841501790816E-03,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
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
        expected_temp_vector = np.array([
            -2.000000000000000,
            -9.157543894743660,
            -10.488404668316800,
            -7.673281851109040,
            -3.379865775679690,
            0.186086818676234,
            1.975932387426680,
            2.059758187189790,
            1.158331618161900,
            0.100524133017546,
            -0.548756412093758,
            -0.609292952871655,
            -0.205843560205627,
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
            -3.422558663172200,
            -7.653777872055470,
            -10.446265951027200,
            -9.985741476399620,
            -8.257152402542280,
            -7.064378950170990,
            -4.672170753626650,
            -1.440416321834300,
            0.974691317050836,
            1.870729966060970,
            2.078359705948460,
            2.177388110077260,
            1.680396982982560,
            0.811013243693162,
            0.227785266077045,
            -0.031121218672030,
            -0.417470305426395,
            -0.644820303593790,
            -0.528777325533927,
            -0.285999942521147,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.00000000000000E+01,
            -9.15745232017429E-02,
            -1.04882997852940E-01,
            -7.67320511902980E-02,
            -3.37983197735703E-02,
            1.86084957826127E-03,
            1.97591262829366E-02,
            2.05973758982125E-02,
            1.15832003495520E-02,
            1.00523127785634E-03,
            -5.48750924589392E-03,
            -6.09286859998282E-03,
            -2.05841501790816E-03,
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
            -0.018998695698926,
            -0.073441754087105,
            -0.105711602393866,
            -0.098927858747365,
            -0.081821263711586,
            -0.070643083070656,
            -0.046721240323791,
            -0.014404019178151,
            0.009746815702341,
            0.018707112589436,
            0.020783389225579,
            0.021773663364221,
            0.016803801791818,
            0.008110051336330,
            0.002277829882428,
            -0.000311209074661,
            -0.004174661307715,
            -0.006448138554552,
            -0.005287720378074,
            -0.002859970825479,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -1.15094952744558,
            -0.70038540187041,
            -0.15129979517683,
            0.26621009170330,
            0.47571579778849,
            0.52108986248641,
            0.51343895568593,
            0.43315752904537,
            0.27078169665011,
            0.11272653799941,
            0.07267779629601,
            -0.02454480894844,
            -0.11231554554672,
            -0.13519701666565,
            -0.11353671706644,
            -0.10632752107749,
            -0.06254166608747,
            -0.00664059453890,
            0.03949363131059,
            0.06538575643192,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            0.044856900989608,
            0.028959852636677,
            0.024424805840736,
            0.025036742486148,
            0.027783541907744,
            0.030254717644438,
            0.037889002748163,
            0.071616283741731,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            0.531169731519830,
            0.139509157401129,
            0.110434240704826,
            0.122871263842543,
            0.170873663087485,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.001840979752590500,
            0.000535773463086406,
            0.000332848500784909,
            0.000356607492588793,
            0.000476998005294406,
            0.000605750673746538,
            0.001141828376592430,
            0.006955560494723540,
            0.000000000000000000,
            0.000000000000000000,
            0.000000000000000000,
            0.000000000000000000,
            0.000000000000000000,
            0.000000000000000000,
            0.000000000000000000,
            1.991948285691010000,
            0.046519642504662600,
            0.023892886650615300,
            0.032389355878183000,
            0.082914617808559400,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array([
            1.7713796673E-15,
            5.0330476423E-23,
            1.5503165575E-28,
            -1.7174679441E-27,
            -3.0704550887E-24,
            -3.9513529639E-22,
            -5.4966774209E-18,
            -1.8327626215E-12,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            0.0000000000E+00,
            1.0956353545E-10,
            1.5160133394E-11,
            6.5848484730E-13,
            -6.1856394494E-12,
            -2.6451054205E-11,
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
            2.73079408088337,
            2.74627928227786,
            2.75071291548223,
            2.75011424558537,
            2.74742860125224,
            2.74501468529173,
            2.73757068344138,
            2.70492489447492,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            1.94419643704324,
            2.29701700484279,
            2.64038489946504,
            2.66783325516560,
            2.65605725478295,
            2.61109159734326,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array([
            [0.3754127694384990, -0.4791349492121190,
                0.1366244795670250, -0.0329022997934044,],
            [-0.4791349492121190, 1.0974938412970900,
             -0.7555601286108480, 0.1372012365258810,],
            [0.1366244795670250, -0.7555601286108480,
                1.1003657454416000, -0.4814300963977770,],
            [-0.0329022997934044, 0.1372012365258810,
             -0.4814300963977770, 0.3771311596653000,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742278648957170, -0.4704043073588020,
                0.1245486051912600, -0.0283721627281755,],
            [-0.4704043073588020, 1.0458021981448800,
             -0.6891083023287820, 0.1137104115427070,],
            [0.1245486051912600, -0.6891083023287820,
                0.9190461742111960, -0.3544864770736730,],
            [-0.0283721627281755, 0.1137104115427070,
             -0.3544864770736730, 0.2691482282591420,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972590, -0.3407384274106880,
                0.0973538364030538, -0.0234370346896240,],
            [-0.3407384274106880, 0.7788306912244300,
             -0.5354461002167960, 0.0973538364030537,],
            [0.0973538364030538, -0.5354461002167960,
                0.7788306912244310, -0.3407384274106880,],
            [-0.0234370346896240, 0.0973538364030537,
             -0.3407384274106880, 0.2668216256972590,],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3292202434390340, -0.4133999362112650,
                0.1112498768967810, -0.0270701841245502,],
            [-0.4133999362112650, 0.9868437905990680,
             -0.6971611631648800, 0.1237173087770760,],
            [0.1112498768967810, -0.6971611631648800,
                1.0421402435851100, -0.4562289573170110,],
            [-0.0270701841245502, 0.1237173087770760,
             -0.4562289573170110, 0.3595818326644850,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array([
            [0.3754127881031510, -0.4791349725652340,
                0.1366244859099620, -0.0329023014478784,],
            [-0.4791349725652340, 1.0974938947368000,
             -0.7555601661219740, 0.1372012439504100,],
            [0.1366244859099620, -0.7555601661219740,
                1.1003658020126000, -0.4814301218005920,],
            [-0.0329023014478784, 0.1372012439504100,
             -0.4814301218005920, 0.3771311792980600,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742278880876620, -0.4704043350030930,
                0.1245486104057020, -0.0283721634902704,],
            [-0.4704043350030930, 1.0458022783717200,
             -0.6891083590766130, 0.1137104157079890,],
            [0.1245486104057020, -0.6891083590766130,
                0.9190462296983460, -0.3544864810274360,],
            [-0.0283721634902704, 0.1137104157079890,
             -0.3544864810274360, 0.2691482288097170,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972590, -0.3407384274106880,
                0.0973538364030538, -0.0234370346896240,],
            [-0.3407384274106880, 0.7788306912244300,
             -0.5354461002167960, 0.0973538364030537,],
            [0.0973538364030538, -0.5354461002167960,
                0.7788306912244310, -0.3407384274106880,],
            [-0.0234370346896240, 0.0973538364030537,
             -0.3407384274106880, 0.2668216256972590,],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3292204606981410, -0.4134002391967840,
                0.1112499890634780, -0.0270702105648353,],
            [-0.4134002391967840, 0.9868443266035030,
             -0.6971614651973810, 0.1237173777906620,],
            [0.1112499890634780, -0.6971614651973810,
                1.0421405856543300, -0.4562291095204280,],
            [-0.0270702105648353, 0.1237173777906620,
             -0.4562291095204280, 0.3595819422946010,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array([
            [3.74901860610780E+06, 2.89593744482105E+06,
             -1.05358398557321E+06, 5.54643193641815E+05,],
            [2.89593744482105E+06, 1.88936621803465E+07,
             -2.36630842102390E+06, -1.04792564897764E+06,],
            [-1.05358398557321E+06, -2.36630842102390E+06,
                1.88757400422500E+07, 2.88462077162992E+06,],
            [5.54643193641815E+05, -1.04792564897764E+06,
                2.88462077162992E+06, 3.73110189762532E+06,],
        ])
        expected_C[3:7, 3:7] += np.array([
            [3.74548978664947E+06, 2.83529217357612E+06,
             -8.63799618191817E+05, 6.29634934688451E+05,],
            [2.83529217357612E+06, 1.92531430053028E+07,
             -3.32307881157745E+06, -1.41571111930997E+06,],
            [-8.63799618191817E+05, -3.32307881157745E+06,
                2.30168957941050E+07, 3.93911517581243E+06,],
            [6.29634934688451E+05, -1.41571111930997E+06,
                3.93911517581243E+06, 4.82119852843649E+06,],
        ])
        expected_C[6:10, 6:10] += np.array([
            [4.84663139329806E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442681E+05,],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857143E+06, -1.36311507936508E+06,],
            [-1.36311507936508E+06, -3.06700892857143E+06,
                2.45360714285714E+07, 3.74856646825397E+06,],
            [7.19421847442681E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 4.84663139329806E+06,],
        ])
        expected_C[9:13, 9:13] += np.array([
            [4.26389587430359E+06, 3.18874298722127E+06,
             -1.17518017569044E+06, 5.93504826831539E+05,],
            [3.18874298722127E+06, 1.95951789440892E+07,
             -2.51514790752116E+06, -1.07640022713129E+06,],
            [-1.17518017569044E+06, -2.51514790752116E+06,
                1.94626651593996E+07, 2.99118309010297E+06,],
            [5.93504826831539E+05, -1.07640022713129E+06,
                2.99118309010297E+06, 3.89099624195036E+06,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array([
            [4.73636726267305E+06, 3.48911673034659E+06,
             -1.29625281104576E+06, 6.28539537122451E+05,],
            [3.48911673034659E+06, 2.04661732700568E+07,
             -2.68983332476522E+06, -1.08386955187774E+06,],
            [-1.29625281104576E+06, -2.68983332476522E+06,
                1.99779485371977E+07, 3.06435021201054E+06,],
            [6.28539537122451E+05, -1.08386955187774E+06,
                3.06435021201054E+06, 4.00237952380553E+06,],
        ])
        expected_C[3:7, 3:7] += np.array([
            [4.16942582348791E+06, 2.79778791457803E+06,
             -1.54215430073513E+06, 7.18714123666039E+05,],
            [2.79778791457803E+06, 2.63422288785067E+07,
                8.59562060676672E+05, -1.83531959920816E+06,],
            [-1.54215430073513E+06, 8.59562060676672E+05,
                2.79086996007669E+07, 3.38411851152410E+06,],
            [7.18714123666039E+05, -1.83531959920816E+06,
                3.38411851152410E+06, 4.88507795406508E+06,],
        ])
        expected_C[6:10, 6:10] += np.array([
            [4.84663139329806E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442681E+05,],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857143E+06, -1.36311507936508E+06,],
            [-1.36311507936508E+06, -3.06700892857143E+06,
                2.45360714285714E+07, 3.74856646825397E+06,],
            [7.19421847442681E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 4.84663139329806E+06,],
        ])
        expected_C[9:13, 9:13] += np.array([
            [1.04462235072547E+09, 5.23838386065482E+08,
             -2.37984388388786E+08, 5.50354387236226E+07,],
            [5.23838386065482E+08, 3.84438392077579E+08,
             -1.43608827863727E+08, 1.42597187764393E+07,],
            [-2.37984388388786E+08, -1.43608827863727E+08,
                1.69012676948151E+08, 1.93501717350303E+07,],
            [5.50354387236226E+07, 1.42597187764393E+07,
                1.93501717350303E+07, 5.14002477376637E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [1.78185839183366E-02, -2.77127430345919E-02,
             1.50433610648253E-02, -7.35639689353092E-03,
             3.29022343483873E-03, -1.59799477200926E-03,
             7.23419240758401E-04, -3.09492200436262E-04,
             1.34610513562921E-04, -3.57649137759604E-05,
             5.07197857118833E-06, -4.82351273890191E-06,
             1.94517619009087E-06,],
            [-6.53006244088708E-03, 1.23820752743078E-02,
             -7.58846221606364E-03, 2.42293401259993E-03,
             -1.02332955333098E-03, 4.97010403290629E-04,
             -2.24998789042002E-04, 9.62586649521523E-05,
             -4.18667362402849E-05, 1.11236497958590E-05,
             -1.57749334309657E-06, 1.50021517819160E-06,
             -6.04991217519945E-07,],
            [1.89483958886758E-03, -7.33761755982859E-03,
             1.07628946644645E-02, -6.85273820358912E-03,
             2.28464967298612E-03, -1.10960799641968E-03,
             5.02324405792690E-04, -2.14903719617181E-04,
             9.34702070794457E-05, -2.48342704320413E-05,
             3.52185631570216E-06, -3.34932779485135E-06,
             1.35068217542172E-06,],
            [-3.80660655255157E-03, 9.70282260947883E-03,
             -1.61720895164148E-02, 1.99366753228409E-02,
             -1.44011732007212E-02, 6.99435766029683E-03,
             -3.16637638422469E-03, 1.35463468394310E-03,
             -5.89184703972141E-04, 1.56541562601116E-04,
             -2.21998424485752E-05, 2.11123176782723E-05,
             -8.51395650605432E-06,],
            [4.48095959132533E-04, -1.14217099757077E-03,
             1.90370290782420E-03, -5.17365082004948E-03,
             8.18479371125820E-03, -5.62005813179915E-03,
             1.92430164769261E-03, -7.79493967181258E-04,
             3.39033045400048E-04, -9.00783104899615E-05,
             1.27743984899811E-05, -1.21486068963806E-05,
             4.89916418941459E-06,],
            [-2.84229394733751E-04, 7.24484487542467E-04,
             -1.20752779447341E-03, 2.41825608987232E-03,
             -5.59889822818017E-03, 6.75915087832430E-03,
             -3.57928341729571E-03, 1.14032765432749E-03,
             -4.95974021195527E-04, 1.31776245656198E-04,
             -1.86877647279309E-05, 1.77722894451538E-05,
             -7.16702456143744E-06,],
            [4.95529880937904E-04, -1.26307735408426E-03,
             2.10522245521133E-03, -3.97921328085604E-03,
             6.74344454535633E-03, -1.00652958451163E-02,
             1.17483155404407E-02, -8.58893762367075E-03,
             3.73567186136673E-03, -9.92537495628906E-04,
             1.40756075646251E-04, -1.33860724060288E-04,
             5.39819644572294E-05,],
            [-7.34823762305452E-05, 1.87302378547646E-04,
             -3.12184500781330E-04, 5.90079546468535E-04,
             -9.99988796303864E-04, 1.49258780270258E-03,
             -3.80008781652643E-03, 5.73444569569661E-03,
             -3.30615994627354E-03, 5.14363948539733E-04,
             -6.21405722845615E-05, 5.90964330409250E-05,
             -2.38317965957474E-05,],
            [1.86351266233454E-05, -4.74998730326615E-05,
             7.91699724523563E-05, -1.49644140954120E-04,
             2.53597104461942E-04, -3.78520185745728E-04,
             7.88728548398504E-04, -3.08222237680943E-03,
             4.63575199551331E-03, -2.19167918312994E-03,
             1.70364469334773E-04, -1.62018663241317E-04,
             6.53372061289730E-05,],
            [-1.90463447310565E-06, 4.85480444939994E-06,
             -8.09170025057296E-06, 1.52946312263013E-05,
             -2.59193187790252E-05, 3.86872925046024E-05,
             -7.28270744983748E-05, 1.45672700895190E-04,
             -2.75255357315613E-04, 6.61629659731790E-04,
             -1.11477114909815E-03, 1.06016079586382E-03,
             -4.27530650256261E-04,],
            [2.11614266998179E-06, -5.39392675857693E-06,
             8.99027735490845E-06, -1.69930882889238E-05,
             2.87976392424106E-05, -4.29834866537746E-05,
             8.09144652436439E-05, -1.61849542559816E-04,
             3.05822253551293E-04, -9.17862649135350E-04,
             1.40479095185883E-03, -8.37517038795954E-04,
             1.51168002271328E-04,],
            [-1.09738558673714E-06, 2.79717316074395E-06,
             -4.66216240048254E-06, 8.81224618119773E-06,
             -1.49338296916196E-05, 2.22903017791169E-05,
             -4.19604827105925E-05, 8.39316543939213E-05,
             -1.58592772553255E-04, 3.26393554651366E-04,
             -1.18449431815978E-03, 2.05923794612925E-03,
             -1.09772192519313E-03,],
            [1.86538778028875E-06, -4.75476687179532E-06,
             7.92496354671456E-06, -1.49794717025384E-05,
             2.53852280878675E-05, -3.78901063220117E-05,
             7.13264076449964E-05, -1.42670984910012E-04,
             2.69583475068749E-04, -6.56288714581979E-04,
             1.61373213169533E-03, -3.02006533367110E-03,
             1.88683178423550E-03,],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-8, atol=1e-11,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [1.01781858391834E+00, -2.77127430345919E-02,
             1.50433610648253E-02, -7.35639689353092E-03,
             3.29022343483873E-03, -1.59799477200926E-03,
             7.23419240758401E-04, -3.09492200436262E-04,
             1.34610513562921E-04, -3.57649137759604E-05,
             5.07197857118833E-06, -4.82351273890191E-06,
             1.94517619009087E-06,],
            [-6.53006244088708E-03, 1.01238207527431E+00,
             -7.58846221606364E-03, 2.42293401259993E-03,
             -1.02332955333098E-03, 4.97010403290629E-04,
             -2.24998789042002E-04, 9.62586649521523E-05,
             -4.18667362402849E-05, 1.11236497958590E-05,
             -1.57749334309657E-06, 1.50021517819160E-06,
             -6.04991217519945E-07,],
            [1.89483958886758E-03, -7.33761755982859E-03,
             1.01076289466446E+00, -6.85273820358912E-03,
             2.28464967298612E-03, -1.10960799641968E-03,
             5.02324405792690E-04, -2.14903719617181E-04,
             9.34702070794457E-05, -2.48342704320413E-05,
             3.52185631570216E-06, -3.34932779485135E-06,
             1.35068217542172E-06,],
            [-3.80660655255157E-03, 9.70282260947883E-03,
             -1.61720895164148E-02, 1.01993667532284E+00,
             -1.44011732007212E-02, 6.99435766029683E-03,
             -3.16637638422469E-03, 1.35463468394310E-03,
             -5.89184703972141E-04, 1.56541562601116E-04,
             -2.21998424485752E-05, 2.11123176782723E-05,
             -8.51395650605432E-06,],
            [4.48095959132533E-04, -1.14217099757077E-03,
             1.90370290782420E-03, -5.17365082004948E-03,
             1.00818479371126E+00, -5.62005813179915E-03,
             1.92430164769261E-03, -7.79493967181258E-04,
             3.39033045400048E-04, -9.00783104899615E-05,
             1.27743984899811E-05, -1.21486068963806E-05,
             4.89916418941459E-06,],
            [-2.84229394733751E-04, 7.24484487542467E-04,
             -1.20752779447341E-03, 2.41825608987232E-03,
             -5.59889822818017E-03, 1.00675915087832E+00,
             -3.57928341729571E-03, 1.14032765432749E-03,
             -4.95974021195527E-04, 1.31776245656198E-04,
             -1.86877647279309E-05, 1.77722894451538E-05,
             -7.16702456143744E-06,],
            [4.95529880937904E-04, -1.26307735408426E-03,
             2.10522245521133E-03, -3.97921328085604E-03,
             6.74344454535633E-03, -1.00652958451163E-02,
             1.01174831554044E+00, -8.58893762367075E-03,
             3.73567186136673E-03, -9.92537495628906E-04,
             1.40756075646251E-04, -1.33860724060288E-04,
             5.39819644572294E-05,],
            [-7.34823762305452E-05, 1.87302378547646E-04,
             -3.12184500781330E-04, 5.90079546468535E-04,
             -9.99988796303864E-04, 1.49258780270258E-03,
             -3.80008781652643E-03, 1.00573444569570E+00,
             -3.30615994627354E-03, 5.14363948539733E-04,
             -6.21405722845615E-05, 5.90964330409250E-05,
             -2.38317965957474E-05,],
            [1.86351266233454E-05, -4.74998730326615E-05,
             7.91699724523563E-05, -1.49644140954120E-04,
             2.53597104461942E-04, -3.78520185745728E-04,
             7.88728548398504E-04, -3.08222237680943E-03,
             1.00463575199551E+00, -2.19167918312994E-03,
             1.70364469334773E-04, -1.62018663241317E-04,
             6.53372061289730E-05,],
            [-1.90463447310565E-06, 4.85480444939994E-06,
             -8.09170025057296E-06, 1.52946312263013E-05,
             -2.59193187790252E-05, 3.86872925046024E-05,
             -7.28270744983748E-05, 1.45672700895190E-04,
             -2.75255357315613E-04, 1.00066162965973E+00,
             -1.11477114909815E-03, 1.06016079586382E-03,
             -4.27530650256261E-04,],
            [2.11614266998179E-06, -5.39392675857693E-06,
             8.99027735490845E-06, -1.69930882889238E-05,
             2.87976392424106E-05, -4.29834866537746E-05,
             8.09144652436439E-05, -1.61849542559816E-04,
             3.05822253551293E-04, -9.17862649135350E-04,
             1.00140479095186E+00, -8.37517038795954E-04,
             1.51168002271328E-04,],
            [-1.09738558673714E-06, 2.79717316074395E-06,
             -4.66216240048254E-06, 8.81224618119773E-06,
             -1.49338296916196E-05, 2.22903017791169E-05,
             -4.19604827105925E-05, 8.39316543939213E-05,
             -1.58592772553255E-04, 3.26393554651366E-04,
             -1.18449431815978E-03, 1.00205923794613E+00,
             -1.09772192519313E-03,],
            [1.86538778028875E-06, -4.75476687179532E-06,
             7.92496354671456E-06, -1.49794717025384E-05,
             2.53852280878675E-05, -3.78901063220117E-05,
             7.13264076449964E-05, -1.42670984910012E-04,
             2.69583475068749E-04, -6.56288714581979E-04,
             1.61373213169533E-03, -3.02006533367110E-03,
             1.00188683178424E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-8, atol=1e-11,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array([
            1.89850411575195E-08,
            9.32804654257637E-09,
            -4.31099786843920E-09,
            -1.42748247257048E-06,
            1.28559187259822E-05,
            1.28558305047007E-05,
            -1.42842430458346E-06,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
            1.10932858948307E-04,
            7.12664676651715E-05,
            -1.57821060328194E-05,
            6.53002632662146E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector_0,
                                    self.msh._heat_flux_vector_0))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            1.86097444238256E-08,
            9.14364951726059E-09,
            -4.22577796578408E-09,
            -1.42744721507470E-06,
            1.28554351353358E-05,
            1.28553469287760E-05,
            -1.42837057414327E-06,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
            1.10934871399782E-04,
            7.12674973267367E-05,
            -1.57825186547925E-05,
            6.53002847395233E-02,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector))


class TestTemperatureCorrectionCubicOneStep(unittest.TestCase):
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
        self.msh.time_step = 3.024E+05
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array([
            -2.000000000000000,
            -9.157543894743660,
            -10.488404668316800,
            -7.673281851109040,
            -3.379865775679690,
            0.186086818676234,
            1.975932387426680,
            2.059758187189790,
            1.158331618161900,
            0.100524133017546,
            -0.548756412093758,
            -0.609292952871655,
            -0.205843560205627,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            0.00000000000000E+00,
            -9.15745232017429E-02,
            -1.04882997852940E-01,
            -7.67320511902980E-02,
            -3.37983197735703E-02,
            1.86084957826127E-03,
            1.97591262829366E-02,
            2.05973758982125E-02,
            1.15832003495520E-02,
            1.00523127785634E-03,
            -5.48750924589392E-03,
            -6.09286859998282E-03,
            -2.05841501790816E-03,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array([
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
        expected_temp_vector = np.array([
            -2.000000000000000,
            -9.082587139505380,
            -10.479659191427400,
            -7.632980191952010,
            -3.388514757134900,
            0.178062168417391,
            1.968571182872450,
            2.056890544111210,
            1.158018885633480,
            0.100916692926800,
            -0.547117529099153,
            -0.607000081807634,
            -0.210024671835248,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            expected_temp_vector,
            actual_temp_nodes,
        ))
        self.assertTrue(np.allclose(
            expected_temp_vector,
            self.msh._temp_vector,
        ))
        self.assertTrue(np.allclose(
            expected_temp_vector_0,
            self.msh._temp_vector_0,
        ))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.00000000000000E+00,
            2.47570047890236E-07,
            2.85733795580311E-08,
            1.33019023210993E-07,
            -2.87130590349577E-08,
            -2.65305419384922E-08,
            -2.42771971630921E-08,
            -9.41484392280149E-09,
            -9.95861051620030E-10,
            1.30146777036420E-09,
            5.40137966509952E-09,
            7.56209371863997E-09,
            -1.38333740500700E-08,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            expected_temp_rate_vector,
            actual_temp_rate_nodes,
            atol=1e-12, rtol=1e-10,
        ))
        self.assertTrue(np.allclose(
            expected_temp_rate_vector,
            self.msh._temp_rate_vector,
            atol=1e-12, rtol=1e-10,
        ))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array([
            [0.3754127694384990, -0.4791349492121190,
                0.1366244795670250, -0.0329022997934044,],
            [-0.4791349492121190, 1.0974938412970900,
             -0.7555601286108480, 0.1372012365258810,],
            [0.1366244795670250, -0.7555601286108480,
                1.1003657454416000, -0.4814300963977770,],
            [-0.0329022997934044, 0.1372012365258810,
             -0.4814300963977770, 0.3771311596653000,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742278648957170, -0.4704043073588020,
                0.1245486051912600, -0.0283721627281755,],
            [-0.4704043073588020, 1.0458021981448800,
             -0.6891083023287820, 0.1137104115427070,],
            [0.1245486051912600, -0.6891083023287820,
                0.9190461742111960, -0.3544864770736730,],
            [-0.0283721627281755, 0.1137104115427070,
             -0.3544864770736730, 0.2691482282591420,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972590, -0.3407384274106880,
                0.0973538364030538, -0.0234370346896240,],
            [-0.3407384274106880, 0.7788306912244300,
             -0.5354461002167960, 0.0973538364030537,],
            [0.0973538364030538, -0.5354461002167960,
                0.7788306912244310, -0.3407384274106880,],
            [-0.0234370346896240, 0.0973538364030537,
             -0.3407384274106880, 0.2668216256972590,],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3292202434390340, -0.4133999362112650,
                0.1112498768967810, -0.0270701841245502,],
            [-0.4133999362112650, 0.9868437905990680,
             -0.6971611631648800, 0.1237173087770760,],
            [0.1112498768967810, -0.6971611631648800,
                1.0421402435851100, -0.4562289573170110,],
            [-0.0270701841245502, 0.1237173087770760,
             -0.4562289573170110, 0.3595818326644850,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array([
            [0.3753879416232020, -0.4791042467337650,
                0.1366167050145040, -0.0329003999039413,],
            [-0.4791042467337650, 1.0974433676402300, -
                0.7555344966562600, 0.1371953757497930,],
            [0.1366167050145040, -0.7555344966562600,
                1.1003399482612500, -0.4814221566194900,],
            [-0.0329003999039413, 0.1371953757497930, -
                0.4814221566194900, 0.3771271807736380,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742218218850550, -0.4703963231115380,
                0.1245458628648330, -0.0283713616383489,],
            [-0.4703963231115380, 1.0458235893735000, -
                0.6891377585117370, 0.1137104922497770,],
            [0.1245458628648330, -0.6891377585117370,
                0.9190792031511250, -0.3544873075042210,],
            [-0.0283713616383489, 0.1137104922497770, -
                0.3544873075042210, 0.2691481768927930,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972590, -0.3407384274106880,
                0.0973538364030538, -0.0234370346896240,],
            [-0.3407384274106880, 0.7788306912244300, -
                0.5354461002167960, 0.0973538364030537,],
            [0.0973538364030538, -0.5354461002167960,
                0.7788306912244310, -0.3407384274106880,],
            [-0.0234370346896240, 0.0973538364030537, -
                0.3407384274106880, 0.2668216256972590,],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3289962966660710, -0.4130762140338010,
                0.1111334904406820, -0.0270535730729519,],
            [-0.4130762140338010, 0.9863568813802830, -
                0.6970058557736110, 0.1237251884271290,],
            [0.1111334904406820, -0.6970058557736110,
                1.0421897385013800, -0.4563173731684510,],
            [-0.0270535730729519, 0.1237251884271290, -
                0.4563173731684510, 0.3596457578142740,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array([
            [3.74901860610780E+06, 2.89593744482105E+06,
             -1.05358398557321E+06, 5.54643193641815E+05,],
            [2.89593744482105E+06, 1.88936621803465E+07,
             -2.36630842102390E+06, -1.04792564897764E+06,],
            [-1.05358398557321E+06, -2.36630842102390E+06,
                1.88757400422500E+07, 2.88462077162992E+06,],
            [5.54643193641815E+05, -1.04792564897764E+06,
                2.88462077162992E+06, 3.73110189762532E+06,],
        ])
        expected_C[3:7, 3:7] += np.array([
            [3.74548978664947E+06, 2.83529217357612E+06,
             -8.63799618191817E+05, 6.29634934688451E+05,],
            [2.83529217357612E+06, 1.92531430053028E+07,
             -3.32307881157745E+06, -1.41571111930997E+06,],
            [-8.63799618191817E+05, -3.32307881157745E+06,
                2.30168957941050E+07, 3.93911517581243E+06,],
            [6.29634934688451E+05, -1.41571111930997E+06,
                3.93911517581243E+06, 4.82119852843649E+06,],
        ])
        expected_C[6:10, 6:10] += np.array([
            [4.84663139329806E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442681E+05,],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857143E+06, -1.36311507936508E+06,],
            [-1.36311507936508E+06, -3.06700892857143E+06,
                2.45360714285714E+07, 3.74856646825397E+06,],
            [7.19421847442681E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 4.84663139329806E+06,],
        ])
        expected_C[9:13, 9:13] += np.array([
            [4.26389587430359E+06, 3.18874298722127E+06,
             -1.17518017569044E+06, 5.93504826831539E+05,],
            [3.18874298722127E+06, 1.95951789440892E+07,
             -2.51514790752116E+06, -1.07640022713129E+06,],
            [-1.17518017569044E+06, -2.51514790752116E+06,
                1.94626651593996E+07, 2.99118309010297E+06,],
            [5.93504826831539E+05, -1.07640022713129E+06,
                2.99118309010297E+06, 3.89099624195036E+06,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array([
            [4.74285700814038E+06, 3.49344088380004E+06, -
                1.29807950820942E+06, 6.28942731006372E+05,],
            [3.49344088380004E+06, 2.04775275805259E+07, -
                2.69184560214359E+06, -1.08352184762866E+06,],
            [-1.29807950820942E+06, -2.69184560214359E+06,
                1.99795376252816E+07, 3.06432556263851E+06,],
            [6.28942731006372E+05, -1.08352184762866E+06,
                3.06432556263851E+06, 4.00294353743635E+06,],
        ])
        expected_C[3:7, 3:7] += np.array([
            [4.17010553124093E+06, 2.80148011856716E+06, -
                1.53913344761798E+06, 7.18402432622841E+05,],
            [2.80148011856716E+06, 2.63128374675488E+07,
                8.30360330006419E+05, -1.83207088751921E+06,],
            [-1.53913344761798E+06, 8.30360330006419E+05,
                2.78795373209754E+07, 3.38735499836960E+06,],
            [7.18402432622841E+05, -1.83207088751921E+06,
                3.38735499836960E+06, 4.88471921523720E+06,],
        ])
        expected_C[6:10, 6:10] += np.array([
            [4.84663139329806E+06, 3.74856646825397E+06, -
                1.36311507936508E+06, 7.19421847442681E+05,],
            [3.74856646825397E+06, 2.45360714285714E+07, -
                3.06700892857143E+06, -1.36311507936508E+06,],
            [-1.36311507936508E+06, -3.06700892857143E+06,
                2.45360714285714E+07, 3.74856646825397E+06,],
            [7.19421847442681E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 4.84663139329806E+06,],
        ])
        expected_C[9:13, 9:13] += np.array([
            [1.05184695178840E+09, 5.27408401740155E+08, -
                2.39636875141802E+08, 5.53787670808823E+07,],
            [5.27408401740155E+08, 3.86393873685784E+08, -
                1.44403954369652E+08, 1.44997490061072E+07,],
            [-2.39636875141802E+08, -1.44403954369652E+08,
                1.69498523604074E+08, 1.91351534443363E+07,],
            [5.53787670808823E+07, 1.44997490061072E+07,
                1.91351534443363E+07, 5.11240998577955E+07,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.array([
            [1.78185839183366E-02, -2.77127430345919E-02,
             1.50433610648253E-02, -7.35639689353092E-03,
             3.29022343483873E-03, -1.59799477200926E-03,
             7.23419240758401E-04, -3.09492200436262E-04,
             1.34610513562921E-04, -3.57649137759604E-05,
             5.07197857118833E-06, -4.82351273890191E-06,
             1.94517619009087E-06,],
            [-6.53006244088708E-03, 1.23820752743078E-02,
             -7.58846221606364E-03, 2.42293401259993E-03,
             -1.02332955333098E-03, 4.97010403290629E-04,
             -2.24998789042002E-04, 9.62586649521523E-05,
             -4.18667362402849E-05, 1.11236497958590E-05,
             -1.57749334309657E-06, 1.50021517819160E-06,
             -6.04991217519945E-07,],
            [1.89483958886758E-03, -7.33761755982859E-03,
             1.07628946644645E-02, -6.85273820358912E-03,
             2.28464967298612E-03, -1.10960799641968E-03,
             5.02324405792690E-04, -2.14903719617181E-04,
             9.34702070794457E-05, -2.48342704320413E-05,
             3.52185631570216E-06, -3.34932779485135E-06,
             1.35068217542172E-06,],
            [-3.80660655255157E-03, 9.70282260947883E-03,
             -1.61720895164148E-02, 1.99366753228409E-02,
             -1.44011732007212E-02, 6.99435766029683E-03,
             -3.16637638422469E-03, 1.35463468394310E-03,
             -5.89184703972141E-04, 1.56541562601116E-04,
             -2.21998424485752E-05, 2.11123176782723E-05,
             -8.51395650605432E-06,],
            [4.48095959132533E-04, -1.14217099757077E-03,
             1.90370290782420E-03, -5.17365082004948E-03,
             8.18479371125820E-03, -5.62005813179915E-03,
             1.92430164769261E-03, -7.79493967181258E-04,
             3.39033045400048E-04, -9.00783104899615E-05,
             1.27743984899811E-05, -1.21486068963806E-05,
             4.89916418941459E-06,],
            [-2.84229394733751E-04, 7.24484487542467E-04,
             -1.20752779447341E-03, 2.41825608987232E-03,
             -5.59889822818017E-03, 6.75915087832430E-03,
             -3.57928341729571E-03, 1.14032765432749E-03,
             -4.95974021195527E-04, 1.31776245656198E-04,
             -1.86877647279309E-05, 1.77722894451538E-05,
             -7.16702456143744E-06,],
            [4.95529880937904E-04, -1.26307735408426E-03,
             2.10522245521133E-03, -3.97921328085604E-03,
             6.74344454535633E-03, -1.00652958451163E-02,
             1.17483155404407E-02, -8.58893762367075E-03,
             3.73567186136673E-03, -9.92537495628906E-04,
             1.40756075646251E-04, -1.33860724060288E-04,
             5.39819644572294E-05,],
            [-7.34823762305452E-05, 1.87302378547646E-04,
             -3.12184500781330E-04, 5.90079546468535E-04,
             -9.99988796303864E-04, 1.49258780270258E-03,
             -3.80008781652643E-03, 5.73444569569661E-03,
             -3.30615994627354E-03, 5.14363948539733E-04,
             -6.21405722845615E-05, 5.90964330409250E-05,
             -2.38317965957474E-05,],
            [1.86351266233454E-05, -4.74998730326615E-05,
             7.91699724523563E-05, -1.49644140954120E-04,
             2.53597104461942E-04, -3.78520185745728E-04,
             7.88728548398504E-04, -3.08222237680943E-03,
             4.63575199551331E-03, -2.19167918312994E-03,
             1.70364469334773E-04, -1.62018663241317E-04,
             6.53372061289730E-05,],
            [-1.90463447310565E-06, 4.85480444939994E-06,
             -8.09170025057296E-06, 1.52946312263013E-05,
             -2.59193187790252E-05, 3.86872925046024E-05,
             -7.28270744983748E-05, 1.45672700895190E-04,
             -2.75255357315613E-04, 6.61629659731790E-04,
             -1.11477114909815E-03, 1.06016079586382E-03,
             -4.27530650256261E-04,],
            [2.11614266998179E-06, -5.39392675857693E-06,
             8.99027735490845E-06, -1.69930882889238E-05,
             2.87976392424106E-05, -4.29834866537746E-05,
             8.09144652436439E-05, -1.61849542559816E-04,
             3.05822253551293E-04, -9.17862649135350E-04,
             1.40479095185883E-03, -8.37517038795954E-04,
             1.51168002271328E-04,],
            [-1.09738558673714E-06, 2.79717316074395E-06,
             -4.66216240048254E-06, 8.81224618119773E-06,
             -1.49338296916196E-05, 2.22903017791169E-05,
             -4.19604827105925E-05, 8.39316543939213E-05,
             -1.58592772553255E-04, 3.26393554651366E-04,
             -1.18449431815978E-03, 2.05923794612925E-03,
             -1.09772192519313E-03,],
            [1.86538778028875E-06, -4.75476687179532E-06,
             7.92496354671456E-06, -1.49794717025384E-05,
             2.53852280878675E-05, -3.78901063220117E-05,
             7.13264076449964E-05, -1.42670984910012E-04,
             2.69583475068749E-04, -6.56288714581979E-04,
             1.61373213169533E-03, -3.02006533367110E-03,
             1.88683178423550E-03,],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-8, atol=1e-11,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.array([
            [1.01781858391834E+00, -2.77127430345919E-02,
             1.50433610648253E-02, -7.35639689353092E-03,
             3.29022343483873E-03, -1.59799477200926E-03,
             7.23419240758401E-04, -3.09492200436262E-04,
             1.34610513562921E-04, -3.57649137759604E-05,
             5.07197857118833E-06, -4.82351273890191E-06,
             1.94517619009087E-06,],
            [-6.53006244088708E-03, 1.01238207527431E+00,
             -7.58846221606364E-03, 2.42293401259993E-03,
             -1.02332955333098E-03, 4.97010403290629E-04,
             -2.24998789042002E-04, 9.62586649521523E-05,
             -4.18667362402849E-05, 1.11236497958590E-05,
             -1.57749334309657E-06, 1.50021517819160E-06,
             -6.04991217519945E-07,],
            [1.89483958886758E-03, -7.33761755982859E-03,
             1.01076289466446E+00, -6.85273820358912E-03,
             2.28464967298612E-03, -1.10960799641968E-03,
             5.02324405792690E-04, -2.14903719617181E-04,
             9.34702070794457E-05, -2.48342704320413E-05,
             3.52185631570216E-06, -3.34932779485135E-06,
             1.35068217542172E-06,],
            [-3.80660655255157E-03, 9.70282260947883E-03,
             -1.61720895164148E-02, 1.01993667532284E+00,
             -1.44011732007212E-02, 6.99435766029683E-03,
             -3.16637638422469E-03, 1.35463468394310E-03,
             -5.89184703972141E-04, 1.56541562601116E-04,
             -2.21998424485752E-05, 2.11123176782723E-05,
             -8.51395650605432E-06,],
            [4.48095959132533E-04, -1.14217099757077E-03,
             1.90370290782420E-03, -5.17365082004948E-03,
             1.00818479371126E+00, -5.62005813179915E-03,
             1.92430164769261E-03, -7.79493967181258E-04,
             3.39033045400048E-04, -9.00783104899615E-05,
             1.27743984899811E-05, -1.21486068963806E-05,
             4.89916418941459E-06,],
            [-2.84229394733751E-04, 7.24484487542467E-04,
             -1.20752779447341E-03, 2.41825608987232E-03,
             -5.59889822818017E-03, 1.00675915087832E+00,
             -3.57928341729571E-03, 1.14032765432749E-03,
             -4.95974021195527E-04, 1.31776245656198E-04,
             -1.86877647279309E-05, 1.77722894451538E-05,
             -7.16702456143744E-06,],
            [4.95529880937904E-04, -1.26307735408426E-03,
             2.10522245521133E-03, -3.97921328085604E-03,
             6.74344454535633E-03, -1.00652958451163E-02,
             1.01174831554044E+00, -8.58893762367075E-03,
             3.73567186136673E-03, -9.92537495628906E-04,
             1.40756075646251E-04, -1.33860724060288E-04,
             5.39819644572294E-05,],
            [-7.34823762305452E-05, 1.87302378547646E-04,
             -3.12184500781330E-04, 5.90079546468535E-04,
             -9.99988796303864E-04, 1.49258780270258E-03,
             -3.80008781652643E-03, 1.00573444569570E+00,
             -3.30615994627354E-03, 5.14363948539733E-04,
             -6.21405722845615E-05, 5.90964330409250E-05,
             -2.38317965957474E-05,],
            [1.86351266233454E-05, -4.74998730326615E-05,
             7.91699724523563E-05, -1.49644140954120E-04,
             2.53597104461942E-04, -3.78520185745728E-04,
             7.88728548398504E-04, -3.08222237680943E-03,
             1.00463575199551E+00, -2.19167918312994E-03,
             1.70364469334773E-04, -1.62018663241317E-04,
             6.53372061289730E-05,],
            [-1.90463447310565E-06, 4.85480444939994E-06,
             -8.09170025057296E-06, 1.52946312263013E-05,
             -2.59193187790252E-05, 3.86872925046024E-05,
             -7.28270744983748E-05, 1.45672700895190E-04,
             -2.75255357315613E-04, 1.00066162965973E+00,
             -1.11477114909815E-03, 1.06016079586382E-03,
             -4.27530650256261E-04,],
            [2.11614266998179E-06, -5.39392675857693E-06,
             8.99027735490845E-06, -1.69930882889238E-05,
             2.87976392424106E-05, -4.29834866537746E-05,
             8.09144652436439E-05, -1.61849542559816E-04,
             3.05822253551293E-04, -9.17862649135350E-04,
             1.00140479095186E+00, -8.37517038795954E-04,
             1.51168002271328E-04,],
            [-1.09738558673714E-06, 2.79717316074395E-06,
             -4.66216240048254E-06, 8.81224618119773E-06,
             -1.49338296916196E-05, 2.22903017791169E-05,
             -4.19604827105925E-05, 8.39316543939213E-05,
             -1.58592772553255E-04, 3.26393554651366E-04,
             -1.18449431815978E-03, 1.00205923794613E+00,
             -1.09772192519313E-03,],
            [1.86538778028875E-06, -4.75476687179532E-06,
             7.92496354671456E-06, -1.49794717025384E-05,
             2.53852280878675E-05, -3.78901063220117E-05,
             7.13264076449964E-05, -1.42670984910012E-04,
             2.69583475068749E-04, -6.56288714581979E-04,
             1.61373213169533E-03, -3.02006533367110E-03,
             1.00188683178424E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-8, atol=1e-11,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array([
            1.89850411575195E-08,
            9.32804654257637E-09,
            -4.31099786843920E-09,
            -1.42748247257048E-06,
            1.28559187259822E-05,
            1.28558305047007E-05,
            -1.42842430458346E-06,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
            1.10932858948307E-04,
            7.12664676651715E-05,
            -1.57821060328194E-05,
            6.53002632662146E-02,
        ])
        self.assertTrue(np.allclose(
            expected_flux_vector_0,
            self.msh._heat_flux_vector_0,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.array([
            5.46746653845989E-09,
            2.68636671492506E-09,
            -1.24151622646639E-09,
            -7.60441975353691E-07,
            6.84647489228791E-06,
            6.84643509031610E-06,
            -7.60714420349861E-07,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
            5.41348756985190E-05,
            3.30150985940221E-05,
            -7.78256844391883E-06,
            6.53078244526887E-02,
        ])
        self.assertTrue(np.allclose(
            expected_flux_vector,
            self.msh._heat_flux_vector,
        ))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -2.38461234636085E-01,
            7.59224243604616E-02,
            7.99943496297368E-03,
            4.17791714043375E-02,
            -8.96429211407198E-03,
            -7.86598279264764E-03,
            -7.63908219256820E-03,
            -2.82298628508657E-03,
            -3.20429239722832E-04,
            3.96237468132868E-04,
            1.63711247439950E-03,
            2.30096034332267E-03,
            -4.19458710959875E-03,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.00000000000000E+00,
            7.49567570052084E-02,
            8.74547297620121E-03,
            4.03016846701951E-02,
            -8.64903073239771E-03,
            -8.02469673177830E-03,
            -7.36118354840201E-03,
            -2.86764617815346E-03,
            -3.12731582359419E-04,
            3.92558622480283E-04,
            1.63886471997195E-03,
            2.29287000911671E-03,
            -4.18115389772324E-03,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 5.22652045961174E-03
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
        self.assertEqual(self.msh._iter, 1)


class TestIterativeTemperatureCorrectionCubic(unittest.TestCase):
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
        self.msh.time_step = 3.024E+5
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array([
            -2.000000000000000,
            -9.082587137736640,
            -10.479659195340700,
            -7.632980166438240,
            -3.388514806411960,
            0.178062121944726,
            1.968571203878090,
            2.056890541011660,
            1.158018886579530,
            0.100916691640023,
            -0.547117547373778,
            -0.607000082862531,
            -0.210024714103347,
        ])
        self.msh._temp_rate_vector[:] = np.array([
            0.00000000000000E+00,
            -3.02825804238568E-10,
            -3.46835310360251E-10,
            -2.53743555523472E-10,
            -1.11766930468156E-10,
            6.15360310271584E-12,
            6.53410260679122E-11,
            6.81130155364171E-11,
            3.83042339601586E-11,
            3.32417750613871E-12,
            -1.81465252840407E-11,
            -2.01483749999432E-11,
            -6.80692796927300E-12,
        ])
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_heat_flux_vector()
        self.msh.update_heat_flow_matrix()
        self.msh.update_heat_storage_matrix()
        self.msh.update_weighted_matrices()
        self.msh.iterative_correction_step()

    def test_temperature_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.00000000000000E+00,
            2.47498817526917E-07,
            2.85763254553079E-08,
            1.32984448853519E-07,
            -2.87329006419087E-08,
            -2.65614768071763E-08,
            -2.42586956207760E-08,
            -9.41748664381412E-09,
            -9.95379039448084E-10,
            1.30275115736807E-09,
            5.39880674132476E-09,
            7.56091046420124E-09,
            -1.38388620278036E-08,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes,
                                    ))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector,
                                    ))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array([
            0.00000000000000E+00,
            2.47498817526917E-07,
            2.85763254553079E-08,
            1.32984448853519E-07,
            -2.87329006419087E-08,
            -2.65614768071763E-08,
            -2.42586956207760E-08,
            -9.41748664381412E-09,
            -9.95379039448084E-10,
            1.30275115736807E-09,
            5.39880674132476E-09,
            7.56091046420124E-09,
            -1.38388620278036E-08,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    actual_temp_rate_nodes,
                                    atol=1e-12, rtol=1e-10,
                                    ))
        self.assertTrue(np.allclose(expected_temp_rate_vector,
                                    self.msh._temp_rate_vector,
                                    atol=1e-12, rtol=1e-10,
                                    ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array([
            [0.3753879492274280, -0.4791042561238620,
             0.1366167073707450, -0.0329004004743101,],
            [-0.4791042561238620, 1.0974433825255700,
             -0.7555345038239140, 0.1371953774222090,],
            [0.1366167073707450, -0.7555345038239140,
                1.1003399547735400, -0.4814221583203750,],
            [-0.0329004004743101, 0.1371953774222090,
                -0.4814221583203750, 0.3771271813724750,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742218246395010, -0.4703963271434800,
             0.1245458642680330, -0.0283713617640532,],
            [-0.4703963271434800, 1.0458236191503300,
             -0.6891377855047830, 0.1137104934979320,],
            [0.1245458642680330, -0.6891377855047830,
                0.9190792299556210, -0.3544873087188710,],
            [-0.0283713617640532, 0.1137104934979320,
                -0.3544873087188710, 0.2691481769849920,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972600, -0.3407384274106900,
             0.0973538364030547, -0.0234370346896243,],
            [-0.3407384274106900, 0.7788306912244320,
                -0.5354461002167970, 0.0973538364030541,],
            [0.0973538364030547, -0.5354461002167970,
                0.7788306912244330, -0.3407384274106900,],
            [-0.0234370346896243, 0.0973538364030541,
                -0.3407384274106900, 0.2668216256972600,],

        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3289962995618880, -0.4130762153818610,
             0.1111334919996520, -0.0270535761796790,],
            [-0.4130762153818610, 0.9863568971153020,
             -0.6970058899187200, 0.1237252081852800,],
            [0.1111334919996520, -0.6970058899187200,
                1.0421898250233900, -0.4563174271043250,],
            [-0.0270535761796790, 0.1237252081852800,
             -0.4563174271043250, 0.3596457950987240,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array([
            [0.3753879416232020, -0.4791042467337650,
             0.1366167050145040, -0.0329003999039413,],
            [-0.4791042467337650, 1.0974433676402300, -
                0.7555344966562600, 0.1371953757497930,],
            [0.1366167050145040, -0.7555344966562600,
                1.1003399482612500, -0.4814221566194900,],
            [-0.0329003999039413, 0.1371953757497930, -
                0.4814221566194900, 0.3771271807736380,],
        ])
        expected_H[3:7, 3:7] += np.array([
            [0.3742218218850550, -0.4703963231115380,
                0.1245458628648330, -0.0283713616383489,],
            [-0.4703963231115380, 1.0458235893735000, -
                0.6891377585117370, 0.1137104922497770,],
            [0.1245458628648330, -0.6891377585117370,
                0.9190792031511250, -0.3544873075042210,],
            [-0.0283713616383489, 0.1137104922497770, -
                0.3544873075042210, 0.2691481768927930,],
        ])
        expected_H[6:10, 6:10] += np.array([
            [0.2668216256972590, -0.3407384274106880,
                0.0973538364030538, -0.0234370346896240,],
            [-0.3407384274106880, 0.7788306912244300, -
                0.5354461002167960, 0.0973538364030537,],
            [0.0973538364030538, -0.5354461002167960,
                0.7788306912244310, -0.3407384274106880,],
            [-0.0234370346896240, 0.0973538364030537, -
                0.3407384274106880, 0.2668216256972590,],
        ])
        expected_H[9:13, 9:13] += np.array([
            [0.3289962966660710, -0.4130762140338010,
                0.1111334904406820, -0.0270535730729519,],
            [-0.4130762140338010, 0.9863568813802830, -
                0.6970058557736110, 0.1237251884271290,],
            [0.1111334904406820, -0.6970058557736110,
                1.0421897385013800, -0.4563173731684510,],
            [-0.0270535730729519, 0.1237251884271290, -
                0.4563173731684510, 0.3596457578142740,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_storage_matrix_0(self):
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
            expected_C, self.msh._heat_storage_matrix_0,
            rtol=1e-10,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array([
            [4.73637171624879E+06, 3.48911975197287E+06,
             -1.29625396875982E+06, 6.28539963772493E+05],
            [3.48911975197287E+06, 2.04661850623337E+07,
             -2.68983544335970E+06, -1.08386996098978E+06],
            [-1.29625396875982E+06, -2.68983544335970E+06,
                1.99779575443663E+07, 3.06435173643278E+06],
            [6.28539963772493E+05, -1.08386996098978E+06,
                3.06435173643278E+06, 8.17181101271068E+06],
        ])
        expected_C[3:7, 3:7] = np.array([
            [8.17181101270964E+06, 2.79778776603809E+06,
             -1.54215963368644E+06, 7.18714827570090E+05],
            [2.79778776603809E+06, 2.63422846762452E+07,
                8.59594504004377E+05, -1.83532284104636E+06],
            [-1.54215963368644E+06, 8.59594504004377E+05,
                2.79087377466924E+07, 3.38411418075791E+06],
            [7.18714827570090E+05, -1.83532284104636E+06,
                3.38411418075791E+06, 9.73170984639250E+06],
        ])
        expected_C[6:10, 6:10] = np.array([
            [9.73170984639250E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442682E+05],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857144E+06, -1.36311507936508E+06],
            [-1.36311507936508E+06, -3.06700892857144E+06,
                2.45360714285714E+07, 3.74856646825397E+06],
            [7.19421847442682E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 1.04947512903143E+09],
        ])
        expected_C[9:13, 9:13] = np.array([
            [1.04947512903143E+09, 5.23841479165093E+08,
             -2.37985788657442E+08, 5.50357660694532E+07],
            [5.23841479165093E+08, 3.84440750802889E+08,
             -1.43609705550628E+08, 1.42597922700073E+07],
            [-2.37985788657442E+08, -1.43609705550628E+08,
                1.69013732189027E+08, 1.93503173101947E+07],
            [5.50357660694532E+07, 1.42597922700073E+07,
                1.93503173101947E+07, 5.14006065935181E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
            rtol=1e-10,
        ))

    def test_global_heat_storage_matrix_weighted(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array([
            [4.73636734264201E+06, 3.48911679572160E+06,
             -1.29625283410763E+06, 6.28539549337238E+05],
            [3.48911679572160E+06, 2.04661736956678E+07,
             -2.68983338251903E+06, -1.08386957415677E+06],
            [-1.29625283410763E+06, -2.68983338251903E+06,
                1.99779489357607E+07, 3.06435027581987E+06],
            [6.28539549337238E+05, -1.08386957415677E+06,
                3.06435027581987E+06, 8.17180553044345E+06],
        ])
        expected_C[3:7, 3:7] = np.array([
            [8.17180553044345E+06, 2.79778798757569E+06,
             -1.54215436314838E+06, 7.18714133746307E+05],
            [2.79778798757569E+06, 2.63422295641292E+07,
                8.59562216163159E+05, -1.83531960807053E+06],
            [-1.54215436314838E+06, 8.59562216163159E+05,
                2.79086998878311E+07, 3.38411847742000E+06],
            [7.18714133746307E+05, -1.83531960807053E+06,
                3.38411847742000E+06, 9.73170935158256E+06],
        ])
        expected_C[6:10, 6:10] = np.array([
            [9.73170935158256E+06, 3.74856646825397E+06,
             -1.36311507936508E+06, 7.19421847442682E+05],
            [3.74856646825397E+06, 2.45360714285714E+07,
             -3.06700892857144E+06, -1.36311507936508E+06],
            [-1.36311507936508E+06, -3.06700892857144E+06,
                2.45360714285714E+07, 3.74856646825397E+06],
            [7.19421847442682E+05, -1.36311507936508E+06,
                3.74856646825397E+06, 1.04946889343989E+09],
        ])
        expected_C[9:13, 9:13] = np.array([
            [1.04946889343989E+09, 5.23838342703922E+08,
             -2.37984368299458E+08, 5.50354343989630E+07],
            [5.23838342703922E+08, 3.84438373169631E+08,
             -1.43608818315020E+08, 1.42597165566276E+07],
            [-2.37984368299458E+08, -1.43608818315020E+08,
                1.69012674106786E+08, 1.93501729917517E+07],
            [5.50354343989630E+07, 1.42597165566276E+07,
                1.93501729917517E+07, 5.14002478974014E+07],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._weighted_heat_storage_matrix,
            rtol=1e-10,
        ))

    def test_global_coef_matrix_0(self):
        expected_C0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C0[0:4, 0:4] = np.array([
            [4.73636734245431E+09, 3.48911679596117E+09,
             -1.29625283417594E+09, 6.28539549353689E+08],
            [3.48911679596117E+09, 2.04661736951191E+10,
             -2.68983338214125E+09, -1.08386957422537E+09],
            [-1.29625283417594E+09, -2.68983338214125E+09,
                1.99779489352105E+10, 3.06435027606058E+09],
            [6.28539549353689E+08, -1.08386957422537E+09,
                3.06435027606058E+09, 8.17180553006777E+09],
        ])
        expected_C0[3:7, 3:7] = np.array([
            [8.17180553006777E+09, 2.79778798781089E+09,
             -1.54215436321065E+09, 7.18714133760493E+08],
            [2.79778798781089E+09, 2.63422295636063E+10,
                8.59562216507716E+08, -1.83531960812739E+09],
            [-1.54215436321065E+09, 8.59562216507711E+08,
                2.79086998873716E+10, 3.38411847759724E+09],
            [7.18714133760493E+08, -1.83531960812739E+09,
                3.38411847759724E+09, 9.73170935131458E+09],
        ])
        expected_C0[6:10, 6:10] = np.array([
            [9.73170935131458E+09, 3.74856646842434E+09,
             -1.36311507941376E+09, 7.19421847454401E+08],
            [3.74856646842434E+09, 2.45360714281820E+10,
             -3.06700892830372E+09, -1.36311507941376E+09],
            [-1.36311507941376E+09, -3.06700892830372E+09,
                2.45360714281820E+10, 3.74856646842434E+09],
            [7.19421847454401E+08, -1.36311507941376E+09,
                3.74856646842434E+09, 1.04946889343959E+12],
        ])
        expected_C0[9:13, 9:13] = np.array([
            [1.04946889343959E+12, 5.23838342704129E+11,
             -2.37984368299513E+11, 5.50354343989765E+10],
            [5.23838342704129E+11, 3.84438373169137E+11,
             -1.43608818314672E+11, 1.42597165565657E+10],
            [-2.37984368299513E+11, -1.43608818314672E+11,
                1.69012674106265E+11, 1.93501729919799E+10],
            [5.50354343989765E+10, 1.42597165565657E+10,
                1.93501729919798E+10, 5.14002478972216E+10],
        ])
        self.assertTrue(np.allclose(
            expected_C0, self.msh._coef_matrix_0,
            rtol=1e-10,
        ))

    def test_global_coef_matrix_1(self):
        expected_C1 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C1[0:4, 0:4] = np.array([
            [4.73636734282972E+09, 3.48911679548203E+09,
             -1.29625283403932E+09, 6.28539549320787E+08],
            [3.48911679548203E+09, 2.04661736962166E+10,
             -2.68983338289681E+09, -1.08386957408817E+09],
            [-1.29625283403932E+09, -2.68983338289681E+09,
                1.99779489363109E+10, 3.06435027557915E+09],
            [6.28539549320787E+08, -1.08386957408817E+09,
                3.06435027557915E+09, 8.17180553081913E+09],
        ])
        expected_C1[3:7, 3:7] = np.array([
            [8.17180553081913E+09, 2.79778798734049E+09,
             -1.54215436308610E+09, 7.18714133732121E+08],
            [2.79778798734049E+09, 2.63422295646521E+10,
                8.59562215818603E+08, -1.83531960801368E+09],
            [-1.54215436308610E+09, 8.59562215818607E+08,
                2.79086998882907E+10, 3.38411847724276E+09],
            [7.18714133732121E+08, -1.83531960801368E+09,
                3.38411847724276E+09, 9.73170935185054E+09],
        ])
        expected_C1[6:10, 6:10] = np.array([
            [9.73170935185054E+09, 3.74856646808360E+09,
             -1.36311507931641E+09, 7.19421847430964E+08],
            [3.74856646808360E+09, 2.45360714289608E+10,
             -3.06700892883917E+09, -1.36311507931641E+09],
            [-1.36311507931641E+09, -3.06700892883917E+09,
                2.45360714289608E+10, 3.74856646808360E+09],
            [7.19421847430964E+08, -1.36311507931641E+09,
                3.74856646808360E+09, 1.04946889344019E+12],
        ])
        expected_C1[9:13, 9:13] = np.array([
            [1.04946889344019E+12, 5.23838342703716E+11,
             -2.37984368299402E+11, 5.50354343989495E+10],
            [5.23838342703716E+11, 3.84438373170124E+11,
             -1.43608818315369E+11, 1.42597165566894E+10],
            [-2.37984368299402E+11, -1.43608818315369E+11,
                1.69012674107307E+11, 1.93501729915236E+10],
            [5.50354343989495E+10, 1.42597165566894E+10,
                1.93501729915236E+10, 5.14002478975812E+10],
        ])
        self.assertTrue(np.allclose(
            expected_C1, self.msh._coef_matrix_1,
            rtol=1e-10,
        ))

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.zeros(self.msh.num_nodes)
        expected_flux_vector_0[-1] = -2.61109074784318 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector_0,
                                    self.msh._heat_flux_vector_0,
                                    rtol=1e-15, atol=1e-16))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -2.61109074784359 * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._heat_flux_vector,
                                    rtol=1e-15, atol=1e-16))

    def test_global_flux_vector_weighted(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = -0.5 * (2.61109074784318
                                           + 2.61109159734326) * 25.0e-3
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._weighted_heat_flux_vector,
                                    rtol=1e-15, atol=1e-16))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            2.31785996337891E+05,
            1.50889699377441E+06,
            2.08416204809570E+06,
            9.32412240631104E+05,
            1.13966773396301E+06,
            -2.08082869632721E+05,
            -2.67614536674500E+05,
            -5.42551380218506E+05,
            -1.97868066013336E+05,
            4.53328138000488E+05,
            7.37393379348755E+05,
            5.20777461418152E+05,
            2.46628021972836E+05,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
            rtol=1e-10,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.00000000000000E+00,
            9.15746427389685E-05,
            1.04883069156525E-04,
            7.67320828637213E-05,
            3.37983097104436E-05,
            -1.86087592687940E-06,
            -1.97591413889387E-05,
            -2.05973867154074E-05,
            -1.15832007823766E-05,
            -1.00523141887636E-06,
            5.48751089066735E-06,
            6.09287095873518E-06,
            2.05841250072898E-06,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
            rtol=1e-8, atol=1e-14,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 9.92791192400449E-06
        self.assertEqual(self.msh._iter, 1)
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)


if __name__ == "__main__":
    unittest.main()
