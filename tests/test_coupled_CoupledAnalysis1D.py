import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
)
from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
)
from frozen_ground_fem.coupled import (
    CoupledAnalysis1D,
)
from frozen_ground_fem.thermal import (
    ThermalBoundary1D,
)
from frozen_ground_fem.consolidation import (
    ConsolidationBoundary1D,
    HydraulicBoundary1D,
)


class TestCoupledAnalysis1DInvalid(unittest.TestCase):
    def test_z_min_max_setters(self):
        msh = CoupledAnalysis1D((100, -8))
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
        msh = CoupledAnalysis1D((100, -8))
        self.assertEqual(msh.grid_size, 0.0)
        with self.assertRaises(ValueError):
            msh.grid_size = "twelve"
        with self.assertRaises(ValueError):
            msh.grid_size = -0.5
        self.assertEqual(msh.grid_size, 0.0)

    def test_set_num_nodes_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_nodes = 5

    def test_set_nodes_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.nodes = ()

    def test_set_num_elements_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_elements = 5

    def test_set_elements_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.elements = ()

    def test_set_num_boundaries_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.num_boundaries = 3

    def test_set_boundaries_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.boundaries = ()

    def test_set_time_step_invalid_float(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = -0.1

    def test_set_time_step_invalid_int(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = -1

    def test_set_time_step_invalid_str0(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = "-0.1e-10"

    def test_set_time_step_invalid_str1(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.time_step = "three"

    def test_set_dt_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.dt = 0.1

    def test_set_over_dt_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.over_dt = 0.1

    def test_set_implicit_factor_invalid_float0(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = -0.1

    def test_set_implicit_factor_invalid_float1(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = 1.1

    def test_set_implicit_factor_invalid_int0(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = -1

    def test_set_implicit_factor_invalid_int1(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = 2

    def test_set_implicit_factor_invalid_str0(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = "-0.1e-10"

    def test_set_implicit_factor_invalid_str1(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_factor = "three"

    def test_set_one_minus_alpha_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.one_minus_alpha = 0.1

    def test_set_implicit_error_tolerance_invalid_float(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = -0.1

    def test_set_implicit_error_tolerance_invalid_int(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = -1

    def test_set_implicit_error_tolerance_invalid_str0(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = "-0.1e-10"

    def test_set_implicit_error_tolerance_invalid_str1(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.implicit_error_tolerance = "three"

    def test_set_eps_s_not_allowed(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(AttributeError):
            msh.eps_s = 0.1

    def test_set_max_iterations_invalid_float0(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = -0.1

    def test_set_max_iterations_invalid_float1(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = 0.1

    def test_set_max_iterations_invalid_int(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(ValueError):
            msh.max_iterations = -1

    def test_set_max_iterations_invalid_str0(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = "-1"

    def test_set_max_iterations_invalid_str1(self):
        msh = CoupledAnalysis1D((100, -8))
        with self.assertRaises(TypeError):
            msh.max_iterations = "three"

    def test_generate_mesh(self):
        msh = CoupledAnalysis1D()
        with self.assertRaises(ValueError):
            msh.generate_mesh()
        with self.assertRaises(ValueError):
            CoupledAnalysis1D(generate=True)
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
        msh = CoupledAnalysis1D((-8, 100), generate=True)
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
        msh = CoupledAnalysis1D((-8, 100), generate=True)
        bnd0 = ThermalBoundary1D((msh.nodes[0],))
        msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            (msh.nodes[-1],),
            (msh.elements[-1].int_pts[-1],),
        )
        with self.assertRaises(KeyError):
            msh.remove_boundary(bnd1)

    def test_update_heat_flux_vector_no_int_pt(self):
        msh = CoupledAnalysis1D((0, 100), generate=True)
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


class TestCoupledAnalysis1DDefaults(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D()

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


class TestCoupledAnalysis1DSetters(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D((100, -8))

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


class TestCoupledAnalysis1DLinearNoArgs(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D(order=1)

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


class TestCoupledAnalysis1DLinearMeshGen(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D(z_range=(100, -8))

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
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
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
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr


class TestCoupledAnalysis1DCubicMeshGen(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D(z_range=(100, -8))

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
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
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
        self.assertEqual(self.msh._void_ratio_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector_0.shape, (nnod,))
        self.assertEqual(self.msh._water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._residual_water_flux_vector.shape, (nnod,))
        self.assertEqual(self.msh._delta_void_ratio_vector.shape, (nnod,))
        self.assertEqual(self.msh._stiffness_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._stiffness_matrix.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix_0.shape, (nnod, nnod))
        self.assertEqual(self.msh._mass_matrix.shape, (nnod, nnod))
        with self.assertRaises(AttributeError):
            _ = self.msh._free_vec
        with self.assertRaises(AttributeError):
            _ = self.msh._free_arr


class TestAddBoundaries(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D((-8, 100), generate=True)

    def test_add_boundary_no_int_pt(self):
        bnd0 = ThermalBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd0)
        self.assertEqual(self.msh.num_boundaries, 1)
        self.assertTrue(bnd0 in self.msh.boundaries)
        bnd1 = ThermalBoundary1D((self.msh.nodes[-1],))
        self.msh.add_boundary(bnd1)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(bnd1 in self.msh.boundaries)
        bnd2 = ConsolidationBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd2)
        self.assertEqual(self.msh.num_boundaries, 3)
        self.assertTrue(bnd2 in self.msh.boundaries)
        bnd3 = ConsolidationBoundary1D((self.msh.nodes[-1],))
        self.msh.add_boundary(bnd3)
        self.assertEqual(self.msh.num_boundaries, 4)
        self.assertTrue(bnd3 in self.msh.boundaries)

    def test_add_boundary_with_int_pt(self):
        bnd0 = ThermalBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd0)
        bnd1 = ThermalBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(bnd1)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(bnd1 in self.msh.boundaries)
        bnd2 = ConsolidationBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(bnd2)
        bnd3 = ConsolidationBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(bnd3)
        self.assertEqual(self.msh.num_boundaries, 4)
        self.assertTrue(bnd3 in self.msh.boundaries)


class TestRemoveBoundaries(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D((-8, 100), generate=True)
        self.bnd0 = ThermalBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(self.bnd0)
        self.bnd1 = ThermalBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(self.bnd1)
        self.bnd2 = ConsolidationBoundary1D((self.msh.nodes[0],))
        self.msh.add_boundary(self.bnd2)
        self.bnd3 = ConsolidationBoundary1D(
            (self.msh.nodes[-1],),
            (self.msh.elements[-1].int_pts[-1],),
        )
        self.msh.add_boundary(self.bnd3)

    def test_remove_boundary_by_ref(self):
        self.assertEqual(self.msh.num_boundaries, 4)
        self.assertTrue(self.bnd0 in self.msh.boundaries)
        self.assertTrue(self.bnd1 in self.msh.boundaries)
        self.assertTrue(self.bnd2 in self.msh.boundaries)
        self.assertTrue(self.bnd3 in self.msh.boundaries)
        self.msh.remove_boundary(self.bnd0)
        self.assertEqual(self.msh.num_boundaries, 3)
        self.assertFalse(self.bnd0 in self.msh.boundaries)
        self.msh.boundaries.discard(self.bnd0)
        self.assertEqual(self.msh.num_boundaries, 3)
        self.msh.remove_boundary(self.bnd3)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertFalse(self.bnd3 in self.msh.boundaries)
        self.msh.boundaries.discard(self.bnd3)
        self.assertEqual(self.msh.num_boundaries, 2)
        self.assertTrue(self.bnd1 in self.msh.boundaries)
        self.assertTrue(self.bnd2 in self.msh.boundaries)

    def test_clear_boundaries(self):
        self.assertEqual(self.msh.num_boundaries, 4)
        self.assertTrue(self.bnd0 in self.msh.boundaries)
        self.assertTrue(self.bnd1 in self.msh.boundaries)
        self.assertTrue(self.bnd2 in self.msh.boundaries)
        self.assertTrue(self.bnd3 in self.msh.boundaries)
        self.msh.clear_boundaries()
        self.assertEqual(self.msh.num_boundaries, 0)
        self.assertFalse(self.bnd0 in self.msh.boundaries)
        self.assertFalse(self.bnd1 in self.msh.boundaries)
        self.assertFalse(self.bnd2 in self.msh.boundaries)
        self.assertFalse(self.bnd3 in self.msh.boundaries)


class TestUpdateBoundaries(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
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
        self.msh = CoupledAnalysis1D((0, 100), generate=True)
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.deg_sat_water = 0.8
                ip.void_ratio = 0.6
                ip.void_ratio_0 = 0.9
        per = 365.0 * 86400.0
        om = 2.0 * np.pi / per
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        def f_T(t): return Tavg + Tamp * np.cos(om * (t - t0))
        self.f_T = f_T
        self.bnd0 = ThermalBoundary1D(
            (self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_function=f_T)
        self.msh.add_boundary(self.bnd0)
        self.geotherm_grad = 25.0 / 1.0e3
        self.flux_geotherm = -0.0443884299924575
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
        eavg = 0.5
        eamp = 0.1
        def f_e(t): return eavg + eamp * np.cos(om * (t - t0))
        self.f_e = f_e
        self.bnd0 = ConsolidationBoundary1D(
            (self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_function=f_e)
        self.msh.add_boundary(self.bnd0)
        self.water_flux = 0.08
        self.bnd1 = ConsolidationBoundary1D(
            (self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=self.water_flux,
        )
        self.msh.add_boundary(self.bnd1)

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
        expected_thrm_cond = 1.7755371996983
        for e in self.msh.elements:
            for ip in e.int_pts:
                self.assertAlmostEqual(ip.thrm_cond, expected_thrm_cond)

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
        expected_Sw = 0.8
        expected_Si = 0.2
        expected_thw = expected_Sw * expected_porosity
        for e in self.msh.elements:
            for ip in e.int_pts:
                self.assertAlmostEqual(ip.porosity, expected_porosity)
                self.assertAlmostEqual(ip.deg_sat_water, expected_Sw)
                self.assertAlmostEqual(ip.deg_sat_ice, expected_Si)
                self.assertAlmostEqual(ip.vol_water_cont, expected_thw)

    def test_update_thermal_boundaries(self):
        t = 1.314e7
        expected_temp_0 = self.f_T(t)
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
        expected_temp_2 = self.f_T(t)
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

    def test_update_consolidation_boundaries(self):
        t = 6307200.0
        expected_void_ratio_0 = self.f_e(t)
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
        expected_void_ratio_2 = self.f_e(t)
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
                self.assertAlmostEqual(fx, -self.water_flux)
            else:
                self.assertEqual(fx, 0.0)
        self.msh.update_boundary_conditions(t)
        self.msh.update_water_flux_vector()
        for k, (fx, fx0) in enumerate(zip(self.msh._water_flux_vector,
                                          self.msh._water_flux_vector_0)):
            self.assertEqual(fx0, 0.0)
            if k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, -self.water_flux)
            else:
                self.assertEqual(fx, 0.0)


class TestUpdateNodes(unittest.TestCase):
    def setUp(self):
        self.msh = CoupledAnalysis1D((0, 100), generate=True, order=1)
        self.msh._temp_vector[:] = np.linspace(2.0, 22.0, self.msh.num_nodes)
        self.msh._temp_vector_0[:] = np.linspace(1.0, 11.0, self.msh.num_nodes)
        self.msh._void_ratio_vector[:] = np.linspace(
            2.0, 22.0, self.msh.num_nodes)
        self.msh._void_ratio_vector_0[:] = np.linspace(
            1.0, 11.0, self.msh.num_nodes)
        self.msh.time_step = 0.25
        self.msh._temp_rate_vector[:] = (
            self.msh._temp_vector[:] - self.msh._temp_vector_0[:]
        ) / self.msh.dt
        self.msh.update_nodes()

    def test_initial_node_values(self):
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k+1))
            self.assertAlmostEqual(nd.temp, 2.0 * (k+1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k+1))

    def test_repeat_update_nodes(self):
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k+1))
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
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
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
        self.msh = CoupledAnalysis1D((0, 100), generate=True, order=1)
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.deg_sat_water = 0.8
                ip.water_flux_rate = -1.5e-8
                ip.temp_gradient = 0.003
                ip.void_ratio = 0.6
                ip.void_ratio_0 = 0.9
                sig_p, dsig_de = ip.material.eff_stress(0.6, 0.0)
                ip.eff_stress = sig_p
                ip.eff_stress_gradient = dsig_de
                k, dk_de = ip.material.hyd_cond(0.6, 1.0, False)
                ip.hyd_cond = k
                ip.hyd_cond_gradient = dk_de

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
        k00 = 0.2503784879262050
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
        c0 = 8.68800000000000E+06
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
        expected1 = np.ones(self.msh.num_nodes) * 0.0022465125000000000
        expected1[0] = 0.0011232562500000000
        expected1[-1] = 0.0011232562500000000
        self.msh.update_heat_flux_vector()
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector_0, expected0))
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector, expected1,
            rtol=1e-13, atol=1e-16,
        ))

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
        k00 = 1.55999984566148E-09
        k11 = 7.82923225956888E-10
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
        m0 = 1.72280701754386
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

    def test_update_water_flux_vector(self):
        self.msh.update_water_flux_vector()
        expected = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(self.msh._water_flux_vector_0, expected))
        self.assertTrue(np.allclose(self.msh._water_flux_vector, expected))


class TestInitializeGlobalSystemLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            thrm_cond_solids=2.1,
            spec_heat_cap_solids=874.0,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=8.10e-6,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
            comp_index_frozen_a1=0.021,
            comp_index_frozen_a2=0.01,
            comp_index_frozen_a3=0.23,
        )
        self.msh = CoupledAnalysis1D(
            z_range=(0, 0.1),
            num_elements=4,
            generate=True,
            order=1
        )
        temp_bound = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=5.0,
        )
        self.msh.add_boundary(temp_bound)
        hyd_bound = HydraulicBoundary1D(
            nodes=(self.msh.nodes[0],), bnd_value=0.1,
        )
        self.msh.add_boundary(hyd_bound)
        e_cu0 = self.mtl.void_ratio_0_comp
        Ccu = self.mtl.comp_index_unfrozen
        sig_cu0 = self.mtl.eff_stress_0_comp
        sig_p_ob = 1.50e4
        e_bnd = e_cu0 - Ccu * np.log10(sig_p_ob / sig_cu0)
        void_ratio_bound = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=e_bnd,
            bnd_value_1=sig_p_ob,
        )
        self.msh.add_boundary(void_ratio_bound)
        for nd in self.msh.nodes:
            nd.temp = -5.0
            nd.temp_rate = 0.0
            nd.void_ratio = 2.83
            nd.void_ratio_0 = 2.83
        for e in self.msh.elements:
            e.assign_material(self.mtl)
        self.msh.initialize_global_system(0.0)

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 0.0)
        self.assertAlmostEqual(self.msh._t1, 0.0)

    def test_free_indices(self):
        expected_free_vec = [i for i in range(self.msh.num_nodes)][1:]
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_vec_thrm[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr_thrm[0].flatten()))
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_vec_cnsl[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr_cnsl[0].flatten()))
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array([
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_nodes,
                                    expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_rate_nodes,
                                    expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array([
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
        ])
        actual_vol_water_cont_int_pts = np.array([
            ip.vol_water_cont
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_int_pts,
            expected_vol_water_cont_int_pts,
        ))

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts,
                                    atol=1e-30))

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array([
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
        ])
        actual_vol_heat_cap_int_pts = np.array([
            ip.vol_heat_cap
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_vol_heat_cap_int_pts,
                                    expected_vol_heat_cap_int_pts,
                                    atol=1e-30))

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array([
            2.830000000000000,
            2.830000000000000,
            2.830000000000000,
            2.830000000000000,
            2.830000000000000,
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
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        actual_void_ratio_0_int_pts = np.array([
            ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_0_int_pts,
                                    expected_void_ratio_int_pts))
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array([
            1.5000000000000E+04,
            1.5331990460407E+04,
            1.5663980920814E+04,
            1.5995971381221E+04,
            1.6327961841628E+04,
        ])
        actual_sig_nodes = np.array([
            nd.tot_stress
            for nd in self.msh.nodes
        ])
        Sw_iipp = [
            ip.deg_sat_water
            for e in self.msh.elements
            for iipp in e._int_pts_deformed
            for ip in iipp
        ]
        print(actual_sig_nodes)
        print(Sw_iipp)
        self.assertTrue(np.allclose(
            expected_sig_nodes,
            actual_sig_nodes,
        ))

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array([
            1.50701578393613E+04,
            1.52618326210456E+04,
            1.54021482997682E+04,
            1.55938230814525E+04,
            1.57341387601751E+04,
            1.59258135418595E+04,
            1.60661292205821E+04,
            1.62578040022664E+04,
        ])
        actual_sig_int_pts = np.array([
            ip.tot_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sig_int_pts,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -5.32198647330003E+06,
            -5.38967591665359E+06,
            -5.43922802832446E+06,
            -5.50691747167802E+06,
            -5.55646958334889E+06,
            -5.62415902670245E+06,
            -5.67371113837331E+06,
            -5.74140058172688E+06,
        ])
        actual_dsigde_int_pts = np.array([
            ip.tot_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        print(actual_dsigde_int_pts)
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
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
        expected = 0.0
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            0.00000000000000000,
            0.02500000000000000,
            0.05000000000000000,
            0.07500000000000000,
            0.10000000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [8.43400120829928E+01, -8.43400120829928E+01,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-8.43400120829928E+01, 1.68680024165986E+02,
             -8.43400120829928E+01, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -8.43400120829928E+01,
                1.68680024165986E+02, -8.43400120829928E+01,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -8.43400120829928E+01, 1.68680024165986E+02,
             -8.43400120829928E+01,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -8.43400120829928E+01,
             8.43400120829928E+01,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [1.70489693482649E+04, 8.52448467413248E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [8.52448467413248E+03, 3.40979386965298E+04, 8.52448467413248E+03,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 8.52448467413248E+03, 3.40979386965298E+04,
                8.52448467413248E+03, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 8.52448467413248E+03,
                3.40979386965298E+04, 8.52448467413248E+03,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                8.52448467413248E+03, 1.70489693482649E+04,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(
            expected_Phi, self.msh._heat_flux_vector_0,
            atol=1e-15, rtol=1e-6,
        ))

    def test_global_heat_flux_vector(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(
            expected_Phi, self.msh._heat_flux_vector,
            atol=1e-15, rtol=1e-6,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [1.98713374797455E-03, 9.93566873987276E-04, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [9.93566873987276E-04, 3.97426749594910E-03, 9.93566873987276E-04,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 9.93566873987276E-04, 3.97426749594910E-03,
                9.93566873987276E-04, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 9.93566873987276E-04,
                3.97426749594910E-03, 9.93566873987276E-04,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.93566873987276E-04, 1.98713374797455E-03,],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))


class TestInitializeTimeStepLinear(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            spec_grav_solids=2.6,
            thrm_cond_solids=2.1,
            spec_heat_cap_solids=874.0,
            deg_sat_water_alpha=1.20e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=1.0e-5,
            seg_pot_0=2.0e-9,
            hyd_cond_index=0.305,
            void_ratio_0_hyd_cond=2.6,
            hyd_cond_mult=0.8,
            hyd_cond_0=8.10e-6,
            void_ratio_min=0.3,
            void_ratio_tr=2.6,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
            comp_index_frozen_a1=0.021,
            comp_index_frozen_a2=0.01,
            comp_index_frozen_a3=0.23,
        )
        self.msh = CoupledAnalysis1D(
            z_range=(0, 0.1),
            num_elements=4,
            generate=True,
            order=1
        )
        temp_bound = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=5.0,
        )
        self.msh.add_boundary(temp_bound)
        hyd_bound = HydraulicBoundary1D(
            nodes=(self.msh.nodes[0],), bnd_value=0.1,
        )
        self.msh.add_boundary(hyd_bound)
        e_cu0 = self.mtl.void_ratio_0_comp
        Ccu = self.mtl.comp_index_unfrozen
        sig_cu0 = self.mtl.eff_stress_0_comp
        sig_p_ob = 1.50e4
        e_bnd = e_cu0 - Ccu * np.log10(sig_p_ob / sig_cu0)
        void_ratio_bound = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=e_bnd,
            bnd_value_1=sig_p_ob,
        )
        self.msh.add_boundary(void_ratio_bound)
        for nd in self.msh.nodes:
            nd.temp = -5.0
            nd.temp_rate = 0.0
            nd.void_ratio = 2.83
            nd.void_ratio_0 = 2.83
        for e in self.msh.elements:
            e.assign_material(self.mtl)
        self.msh.time_step = 3.75
        self.msh.initialize_global_system(0.0)
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 0.0)
        self.assertAlmostEqual(self.msh._t1, 3.75)

    def test_free_indices(self):
        expected_free_vec = [i for i in range(self.msh.num_nodes)][1:]
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_vec_thrm[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr_thrm[0].flatten()))
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_vec_cnsl[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr_cnsl[0].flatten()))
        self.assertTrue(np.all(expected_free_vec ==
                               self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec ==
                        self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array([
            5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_nodes,
                                    expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            2.8867513459481300,
            -2.8867513459481300,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
            -5.0,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_rate_nodes,
                                    expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            2.10313369225284000,
            0.56353297441383300,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        print(actual_temp_rate_int_pts)
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -400.00000000000000000,
            -400.00000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.000000000000000,
            0.049189123034829,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array([
            0.5851446432832020,
            0.0349299200049867,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
        ])
        actual_vol_water_cont_int_pts = np.array([
            ip.vol_water_cont
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_int_pts,
            expected_vol_water_cont_int_pts,
        ))

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array([
            0.07077197308170330,
            0.00376019673031735,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
        ])
        actual_vol_water_cont_temp_gradient_int_pts = np.array([
            ip.vol_water_cont_temp_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_vol_water_cont_temp_gradient_int_pts,
            expected_vol_water_cont_temp_gradient_int_pts,
        ))

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array([
            0.97204355293632,
            2.08230418654856,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
        ])
        actual_thrm_cond_int_pts = np.array([
            ip.thrm_cond
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
                                    expected_thrm_cond_int_pts,
                                    atol=1e-30))

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array([
            3.40266539296583E+06,
            2.07560330535707E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
        ])
        actual_vol_heat_cap_int_pts = np.array([
            ip.vol_heat_cap
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_vol_heat_cap_int_pts,
                                    expected_vol_heat_cap_int_pts,
                                    atol=1e-30))

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array([
            1.03011911113263,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes,
                                    atol=1e-8))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector,
                                    atol=1e-8))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector_0,
                                    atol=1e-8))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            1.41047869771790000,
            2.44964041341473000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        actual_void_ratio_0_int_pts = np.array([
            ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_0_int_pts,
                                    expected_void_ratio_int_pts))
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.01956560310148E-09,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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
            7.69716904600309E-00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array([
            1.5057626734725E+04,
            1.5340992813644E+04,
            1.5672983274051E+04,
            1.6004973734458E+04,
            1.6336964194865E+04,
        ])
        actual_sig_nodes = np.array([
            nd.tot_stress
            for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            expected_sig_nodes,
            actual_sig_nodes,
        ))

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array([
            1.51175090332130E+04,
            1.52811105151559E+04,
            1.54111506530051E+04,
            1.56028254346894E+04,
            1.57431411134120E+04,
            1.59348158950964E+04,
            1.60751315738190E+04,
            1.62668063555033E+04,
        ])
        actual_sig_int_pts = np.array([
            ip.tot_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sig_int_pts,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            7.27181296175159E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
        ])
        actual_sigp_int_pts = np.array([
            ip.eff_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sigp_int_pts,
        ))

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            0.00000000000000E+00,
            -4.27081477472239E+06,
            -5.44240718561433E+06,
            -5.51009662896790E+06,
            -5.55964874063876E+06,
            -5.62733818399233E+06,
            -5.67689029566319E+06,
            -5.74457973901676E+06,
        ])
        actual_dsigde_int_pts = np.array([
            ip.tot_stress_gradient
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -2.09299601559626E+02,
            -1.53141051315518E+01,
            -1.53141051315518E+01,
            -1.53141051315518E+01,
            -1.53141051315518E+01,
            -1.53141051315518E+01,
            -1.53141051315518E+01,
            -1.53141051315518E+01,
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
            6.89049609873269E+03,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
            2.80000000000000E+00,
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
            -3.16511899023414E-09,
            -4.50905622831981E-12,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
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
        expected = 0.0
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            0.00587428488533737,
            0.02500000000000000,
            0.05000000000000000,
            0.07500000000000000,
            0.10000000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [8.43400120829928E+01, -8.43400120829928E+01,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-8.43400120829928E+01, 1.68680024165986E+02,
             -8.43400120829928E+01, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -8.43400120829928E+01,
                1.68680024165986E+02, -8.43400120829928E+01,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -8.43400120829928E+01, 1.68680024165986E+02,
             -8.43400120829928E+01,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -8.43400120829928E+01,
             8.43400120829928E+01,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [1.70489693482649E+04, 8.52448467413248E+03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [8.52448467413248E+03, 3.40979386965298E+04, 8.52448467413248E+03,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 8.52448467413248E+03, 3.40979386965298E+04,
                8.52448467413248E+03, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 8.52448467413248E+03,
                3.40979386965298E+04, 8.52448467413248E+03,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                8.52448467413248E+03, 1.70489693482649E+04,],
        ])
        self.assertTrue(np.allclose(
            expected_C, self.msh._heat_storage_matrix,
        ))

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(
            expected_Phi, self.msh._heat_flux_vector_0,
            atol=1e-15, rtol=1e-6,
        ))

    def test_global_heat_flux_vector(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(
            expected_Phi, self.msh._heat_flux_vector,
            atol=1e-15, rtol=1e-6,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [1.98713374797455E-03, 9.93566873987276E-04, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [9.93566873987276E-04, 3.97426749594910E-03, 9.93566873987276E-04,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 9.93566873987276E-04, 3.97426749594910E-03,
                9.93566873987276E-04, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 9.93566873987276E-04,
                3.97426749594910E-03, 9.93566873987276E-04,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.93566873987276E-04, 1.98713374797455E-03,],
        ])
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.array([
            -1.56524098050761E-06,
            -5.84157071020524E-06,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
        ])
        print(self.msh._water_flux_vector)
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))


class TestUpdateGlobalMatricesCubicConstant(unittest.TestCase):
    def setUp(self):
        self.mtl = Material(
            thrm_cond_solids=3.0,
            spec_heat_cap_solids=741.0,
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
        self.msh = CoupledAnalysis1D((0, 100), generate=True)
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.deg_sat_water = 0.8
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.water_flux_rate = -1.5e-8
                ip.temp_gradient = 0.003
                ip.void_ratio = 0.6
                ip.void_ratio_0 = 0.9
                sig_p, dsig_de = ip.material.eff_stress(0.6, 0.0)
                ip.eff_stress = sig_p
                ip.eff_stress_gradient = dsig_de
                k, dk_de = ip.material.hyd_cond(0.6, 1.0, False)
                ip.hyd_cond = k
                ip.hyd_cond_gradient = dk_de

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
        h00 = 0.9264004053269600
        h11 = 2.7040876696030200
        h10 = -1.1830383554513200
        h20 = 0.3380109587003770
        h30 = -0.0813730085760168
        h21 = -1.8590602728520800
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
        c00 = 1.98582857142857E+06
        c11 = 1.00532571428571E+07
        c10 = 1.53591428571429E+06
        c20 = -5.58514285714289E+05
        c30 = 2.94771428571427E+05
        c21 = -1.25665714285713E+06
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
        expected1 = np.ones(self.msh.num_nodes) * 0.0008424421875
        expected1[3::3] = 0.0005616281250
        expected1[0] = 0.0002808140625
        expected1[-1] = 0.0002808140625
        self.msh.update_heat_flux_vector()
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector_0, expected0))
        self.assertTrue(np.allclose(
            self.msh._heat_flux_vector, expected1,
            rtol=1e-13, atol=1e-16
        ))

    def test_initial_stiffness_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._stiffness_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._stiffness_matrix, expected))

    def test_initial_mass_matrix(self):
        expected = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(self.msh._mass_matrix_0, expected))
        self.assertTrue(np.allclose(self.msh._mass_matrix, expected))

    def test_initial_water_flux_vector(self):
        expected = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(self.msh._water_flux_vector_0, expected))
        self.assertTrue(np.allclose(self.msh._water_flux_vector, expected))

    def test_update_stiffness_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected1 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        stiff_el = -np.array([
            [-4.72294599234629E-09, 6.08882284823793E-09,
             -1.81459605925378E-09, 4.48719203362137E-10,],
            [4.98148866515888E-09, -1.26517845867392E-08,
                9.48489198083411E-09, -1.81459605925378E-09,],
            [-1.34835008743102E-09, 7.91131182593230E-09,
             -1.26517845867392E-08, 6.08882284823793E-09,],
            [3.12730794913833E-10, -1.34835008743102E-09,
                4.98148866515888E-09, -3.94586937264169E-09,],
        ])
        for k in range(self.msh.num_elements):
            kmin = 3 * k
            kmax = kmin + 4
            expected1[kmin:kmax, kmin:kmax] += stiff_el
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
        m00 = 0.3937844611528820
        m11 = 1.9935338345864700
        m10 = 0.3045676691729330
        m20 = -0.1107518796992490
        m30 = 0.0584523809523807
        m21 = -0.2491917293233050
        d0 = np.ones((self.msh.num_nodes,)) * 2.0 * m00
        d0[0] = m00
        d0[-1] = m00
        d0[1::3] = m11
        d0[2::3] = m11
        d1 = np.ones((self.msh.num_nodes - 1,)) * m10
        d1[1::3] = m21
        d2 = np.ones((self.msh.num_nodes - 2,)) * m20
        d2[2::3] = 0.0
        d3 = np.zeros((self.msh.num_nodes - 3,))
        d3[0::3] = m30
        expected1 = np.diag(d0)
        expected1 += np.diag(d1, -1) + np.diag(d1, 1)
        expected1 += np.diag(d2, -2) + np.diag(d2, 2)
        expected1 += np.diag(d3, -3) + np.diag(d3, 3)
        self.msh.update_mass_matrix()
        self.assertTrue(np.allclose(
            self.msh._mass_matrix_0, expected0,
            atol=1e-12, rtol=1e-10,
        ))
        self.assertTrue(np.allclose(
            self.msh._mass_matrix, expected1,
            atol=1e-12, rtol=1e-10,
        ))

    def test_update_water_flux_vector(self):
        self.msh.update_water_flux_vector()
        expected = np.zeros(self.msh.num_nodes)
        self.assertTrue(np.allclose(self.msh._water_flux_vector_0, expected))
        self.assertTrue(np.allclose(self.msh._water_flux_vector, expected))


# class TestUpdateIntegrationPointsCubic(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#         )
#         self.msh._temp_vector[:] = np.array([
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
#         self.msh._temp_rate_vector[:] = np.array([
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
#                 ip.vol_water_cont__0 = ip.porosity
#         self.msh.update_integration_points()
#
#     def test_temperature_distribution(self):
#         expected_temp_int_pts = np.array([
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
#         actual_temp_int_pts = np.array([
#             ip.temp for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_int_pts,
#                                     expected_temp_int_pts))
#
#     def test_temperature_rate_distribution(self):
#         expected_temp_rate_int_pts = np.array([
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
#         actual_temp_rate_int_pts = np.array([
#             ip.temp_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_rate_int_pts,
#                                     expected_temp_rate_int_pts))
#
#     def test_temperature_gradient_distribution(self):
#         expected_temp_gradient_int_pts = np.array([
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
#         actual_temp_gradient_int_pts = np.array([
#             ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
#                                     expected_temp_gradient_int_pts))
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
#     def test_vol_water_cont_temp_gradient_distribution(self):
#         expected_vol_water_cont_temp_gradient_int_pts = np.array([
#             0.07235260409982430,
#             0.03289271388265000,
#             0.02421242458534380,
#             0.02531316492062430,
#             0.03052609000431350,
#             0.03558952835188600,
#             0.05338816551016490,
#             0.16710056507182700,
#             0.00000000000000000,
#             0.00000000000000000,
#             0.00000000000000000,
#             0.00000000000000000,
#             0.00000000000000000,
#             0.00000000000000000,
#             0.00000000000000000,
#             3.90566915045107000,
#             0.53439071533528300,
#             0.35766602066855600,
#             0.43005975899489600,
#             0.75161061396167100,
#         ])
#         actual_vol_water_cont_temp_gradient_int_pts = np.array([
#             ip.vol_water_cont_temp_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(
#             actual_vol_water_cont_temp_gradient_int_pts,
#             expected_vol_water_cont_temp_gradient_int_pts,
#         ))
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
#
# class TestInitializeTimeStepLinear(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#             order=1
#         )
#         initial_temp_vector = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         initial_temp_rate_vector = np.array([
#             0.05,
#             0.02,
#             0.01,
#             -0.08,
#             -0.05,
#         ])
#         for nd, T0, dTdt0 in zip(self.msh.nodes,
#                                  initial_temp_vector,
#                                  initial_temp_rate_vector,
#                                  ):
#             nd.temp = T0
#             nd.temp_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp,
#             bnd_value=2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 3.024E+05
#         self.msh.initialize_time_step()
#
#     def test_time_step_set(self):
#         self.assertAlmostEqual(self.msh._t0, 1.5)
#         self.assertAlmostEqual(self.msh._t1, 1.5 + 3.024e5)
#
#     def test_iteration_variables(self):
#         self.assertEqual(self.msh._eps_a, 1.0)
#         self.assertEqual(self.msh._iter, 0)
#
#     def test_temperature_distribution_nodes(self):
#         expected_temp_vector_0 = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         expected_temp_vector_1 = np.array([
#             2.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         actual_temp_nodes = np.array([
#             nd.temp for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_vector_1,
#                                     actual_temp_nodes))
#         self.assertTrue(np.allclose(expected_temp_vector_0,
#                                     self.msh._temp_vector_0))
#         self.assertTrue(np.allclose(expected_temp_vector_1,
#                                     self.msh._temp_vector))
#
#     def test_temperature_distribution_int_pts(self):
#         expected_temp_int_pts = np.array([
#             1.5984827557301400,
#             0.5015172442698560,
#             -0.0901923788646684,
#             -0.6098076211353320,
#             -0.9479274057836310,
#             -1.3520725942163700,
#             -3.7189110867544700,
#             -9.7810889132455400,
#         ])
#         actual_temp_int_pts = np.array([
#             ip.temp for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_int_pts,
#                                     expected_temp_int_pts))
#
#     def test_temperature_rate_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
#             0.05,
#             0.02,
#             0.01,
#             -0.08,
#             -0.05,
#         ])
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     actual_temp_rate_nodes))
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     self.msh._temp_rate_vector))
#
#     def test_temperature_rate_distribution_int_pts(self):
#         expected_temp_rate_int_pts = np.array([
#             0.04366025403784440,
#             0.02633974596215560,
#             0.01788675134594810,
#             0.01211324865405190,
#             -0.00901923788646684,
#             -0.06098076211353320,
#             -0.07366025403784440,
#             -0.05633974596215560,
#         ])
#         actual_temp_rate_int_pts = np.array([
#             ip.temp_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_rate_int_pts,
#                                     expected_temp_rate_int_pts))
#
#     def test_temperature_gradient_distribution(self):
#         expected_temp_gradient_int_pts = np.array([
#             -0.0760000,
#             -0.0760000,
#             -0.0360000,
#             -0.0360000,
#             -0.0280000,
#             -0.0280000,
#             -0.4200000,
#             -0.4200000,
#         ])
#         actual_temp_gradient_int_pts = np.array([
#             ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
#                                     expected_temp_gradient_int_pts))
#
#     def test_deg_sat_water_distribution(self):
#         expected_deg_sat_water_int_pts = np.array([
#             1.000000000000000,
#             1.000000000000000,
#             0.314715929845879,
#             0.113801777607921,
#             0.089741864676250,
#             0.074104172041942,
#             0.042882888566470,
#             0.025322726744343,
#         ])
#         actual_deg_sat_water_int_pts = np.array([
#             ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
#                                     expected_deg_sat_water_int_pts))
#
#     def test_vol_water_cont_temp_gradient_distribution(self):
#         expected_vol_water_cont_temp_gradient_int_pts = np.array([
#             0.00000000000000,
#             0.00000000000000,
#             0.00000000000000,
#             0.00000000000000,
#             0.00000000000000,
#             0.00000000000000,
#             0.00000000000000,
#             0.00000000000000,
#         ])
#         actual_vol_water_cont_temp_gradient_int_pts = np.array([
#             ip.vol_water_cont_temp_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(
#             actual_vol_water_cont_temp_gradient_int_pts,
#             expected_vol_water_cont_temp_gradient_int_pts,
#         ))
#
#     def test_water_flux_distribution(self):
#         expected_water_flux_int_pts = np.array([
#             0.0000000000E+00,
#             0.0000000000E+00,
#             -4.8910645169E-12,
#             -5.5518497692E-13,
#             8.3577053338E-13,
#             1.7708810805E-13,
#             2.0670411271E-16,
#             6.0318175808E-27,
#         ])
#         actual_water_flux_int_pts = np.array([
#             ip.water_flux_rate
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_water_flux_int_pts,
#                                     expected_water_flux_int_pts,
#                                     atol=1e-30))
#
#     def test_thrm_cond_distribution(self):
#         expected_thrm_cond_int_pts = np.array([
#             1.94419643704324,
#             1.94419643704324,
#             2.48085630059944,
#             2.66463955659925,
#             2.68754164945741,
#             2.70253225701445,
#             2.73271219424962,
#             2.74983450612514,
#         ])
#         actual_thrm_cond_int_pts = np.array([
#             ip.thrm_cond
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
#                                     expected_thrm_cond_int_pts,
#                                     atol=1e-30))
#
#     def test_global_heat_flow_matrix(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000,],
#             [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
#                 0.0000000000000000, 0.0000000000000000,],
#             [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
#              -0.0999640592805206, 0.0000000000000000,],
#             [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
#                 0.2016431146839040, -0.1016790554033840,],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554033840, 0.1016790554033840,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix_0,
#         ))
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix,
#         ))
#
#     def test_global_heat_storage_matrix(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
#                 8.29817132739774E+06, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
#                 3.29568618779144E+07, 8.17817064124763E+06,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 8.17817064124763E+06, 1.63181792594571E+07,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix_0,
#         ))
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix,
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.array([
#             -0.000000000000000E+00,
#             -7.240998977095220E-06,
#             -1.693636195537800E-06,
#             4.516088940272450E-07,
#             6.874586265312840E-02,
#         ])
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._heat_flux_vector_0))
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._heat_flux_vector))
#
#
# class TestTemperatureCorrectionLinearOneStep(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#             order=1
#         )
#         initial_temp_vector = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         initial_temp_rate_vector = np.array([
#             0.05,
#             0.02,
#             0.01,
#             -0.08,
#             -0.05,
#         ])
#         for nd, T0, dTdt0 in zip(self.msh.nodes,
#                                  initial_temp_vector,
#                                  initial_temp_rate_vector,
#                                  ):
#             nd.temp = T0
#             nd.temp_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp,
#             bnd_value=2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 3.024E+05
#         self.msh.initialize_time_step()
#         self.msh._temp_vector[:] = np.array([
#             2.0,
#             0.6,
#             -0.2,
#             -0.8,
#             -6,
#         ])
#         self.msh._temp_rate_vector[:] = np.array([
#             0,
#             500,
#             600,
#             700,
#             6000,
#         ])
#         self.msh.update_boundary_conditions(self.msh._t1)
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_global_matrices_and_vectors()
#         self.msh.calculate_solution_vector_correction()
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_global_matrices_and_vectors()
#         self.msh.update_iteration_variables()
#
#     def test_temperature_distribution_nodes(self):
#         expected_temp_vector_0 = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         expected_temp_vector = np.array([
#             2.0000000000000000,
#             0.0988263131322971,
#             -0.7971699198965330,
#             -1.5126232123463600,
#             -11.9736027887043000,
#         ])
#         actual_temp_nodes = np.array([
#             nd.temp for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(
#             expected_temp_vector,
#             actual_temp_nodes,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_vector_0,
#             self.msh._temp_vector_0,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_vector,
#             self.msh._temp_vector,
#         ))
#
#     def test_temperature_rate_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
#             6.61375661375661E-06,
#             -3.88123964187482E-09,
#             9.35873050088320E-09,
#             -4.17434270713069E-08,
#             8.72923653959047E-08,
#         ])
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(
#             expected_temp_rate_vector,
#             actual_temp_rate_nodes,
#             atol=1e-12, rtol=1e-3,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_rate_vector,
#             self.msh._temp_rate_vector,
#             atol=1e-12, rtol=1e-3,
#         ))
#
#     def test_global_heat_flow_matrix_0(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000,],
#             [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
#                 0.0000000000000000, 0.0000000000000000,],
#             [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
#              -0.0999640592805206, 0.0000000000000000,],
#             [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
#                 0.2016431146839040, -0.1016790554033840,],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554033840, 0.1016790554033840,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix_0,
#         ))
#
#     def test_global_heat_flow_matrix(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000,],
#             [-0.0721139528911510, 0.1675482489237170, -0.0954342960325658,
#                 0.0000000000000000, 0.0000000000000000,],
#             [0.0000000000000000, -0.0954342960325658, 0.1954036559952800,
#              -0.0999693599627141, 0.0000000000000000,],
#             [0.0000000000000000, 0.0000000000000000, -0.0999693599627141,
#                 0.2016484400499960, -0.1016790800872820,],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790800872820, 0.1016790800872820,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix,
#         ))
#
#     def test_global_heat_storage_matrix_0(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
#                 8.29817132739774E+06, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
#                 3.29568618779144E+07, 8.17817064124763E+06,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 8.17817064124763E+06, 1.63181792594571E+07,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix_0,
#         ))
#
#     def test_global_heat_storage_matrix(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [1.06020061728395E+07, 1.14790024626751E+09, 3.21101921370035E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 3.21101921370035E+08, 2.06835155994880E+08,
#                 2.14800761090158E+07, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.14800761090158E+07,
#                 5.70746434252292E+07, 9.43514558536205E+06,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43514558536205E+06, 1.74625389157076E+07,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.array([
#             -0.000000000000000E+00,
#             -7.240998977095210E-06,
#             -1.693636195537800E-06,
#             4.516088940272450E-07,
#             6.874586265312840E-02,
#         ])
#         self.assertTrue(np.allclose(
#             expected_flux_vector_0,
#             self.msh._heat_flux_vector_0
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.array([
#             0.000000000000000E+00,
#             2.143145921199510E-05,
#             8.271535830408110E-06,
#             2.236869459407050E-07,
#             6.874523161026890E-02,
#         ])
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._heat_flux_vector))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             -2.00010331808334E+00,
#             -5.01130348395403E-01,
#             -5.97300376134295E-01,
#             -7.10765507610895E-01,
#             -5.97904836728674E+00,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_heat_flux_vector,
#         ))
#
#     def test_temperature_increment_vector(self):
#         expected_dT = np.array([
#             0.00000000000000E+00,
#             -5.01173686867703E-01,
#             -5.97169919896533E-01,
#             -7.12623212346363E-01,
#             -5.97360278870428E+00,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_temp_vector,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 4.9481302578941E-01
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#         self.assertEqual(self.msh._iter, 1)
#
#
# class TestIterativeTemperatureCorrectionLinear(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#             order=1
#         )
#         initial_temp_vector = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         initial_temp_rate_vector = np.array([
#             0.05,
#             0.02,
#             0.01,
#             -0.08,
#             -0.05,
#         ])
#         for nd, T0, dTdt0 in zip(self.msh.nodes,
#                                  initial_temp_vector,
#                                  initial_temp_rate_vector,
#                                  ):
#             nd.temp = T0
#             nd.temp_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp,
#             bnd_value=2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 3.024E+05
#         self.msh.initialize_time_step()
#         self.msh._temp_vector[:] = np.array([
#             2.0,
#             0.6,
#             -0.2,
#             -0.8,
#             -6,
#         ])
#         self.msh._temp_rate_vector[:] = np.array([
#             0,
#             500,
#             600,
#             700,
#             6000,
#         ])
#         self.msh.update_boundary_conditions(self.msh._t1)
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_global_matrices_and_vectors()
#         self.msh.iterative_correction_step()
#
#     def test_temperature_distribution_nodes(self):
#         expected_temp_vector_0 = np.array([
#             0.0,
#             0.1,
#             -0.8,
#             -1.5,
#             -12,
#         ])
#         expected_temp_vector = np.array([
#             2.0000000000000000,
#             0.0986658199546282,
#             -0.7965229034856440,
#             -1.5139885335425800,
#             -11.9724600868138000,
#         ])
#         actual_temp_nodes = np.array([
#             nd.temp for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(
#             expected_temp_vector,
#             actual_temp_nodes,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_vector_0,
#             self.msh._temp_vector_0,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_vector,
#             self.msh._temp_vector,
#         ))
#
#     def test_temperature_rate_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
#             6.61375661375661E-06,
#             -4.41197104950990E-09,
#             1.14983350342468E-08,
#             -4.62583781170062E-08,
#             9.10711414888197E-08,
#         ])
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(
#             expected_temp_rate_vector,
#             actual_temp_rate_nodes,
#             atol=1e-16, rtol=1e-10,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_rate_vector,
#             self.msh._temp_rate_vector,
#             atol=1e-16, rtol=1e-10,
#         ))
#
#     def test_global_heat_flow_matrix_0(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000,],
#             [-0.0721139528911510, 0.1675420790767840, -0.0954281261856329,
#                 0.0000000000000000, 0.0000000000000000,],
#             [0.0000000000000000, -0.0954281261856329, 0.1953921854661540,
#              -0.0999640592805206, 0.0000000000000000,],
#             [0.0000000000000000, 0.0000000000000000, -0.0999640592805206,
#                 0.2016431146839040, -0.1016790554033840,],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016790554033840, 0.1016790554033840,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix_0,
#         ))
#
#     def test_global_heat_flow_matrix(self):
#         expected_H = np.array([
#             [0.0721139528911510, -0.0721139528911510, 0.0000000000000000,
#                 0.0000000000000000, 0.0000000000000000,],
#             [-0.0721139528911510, 0.1675471042926980, -0.0954331514015465,
#                 0.0000000000000000, 0.0000000000000000,],
#             [0.0000000000000000, -0.0954331514015465, 0.1954028010121700,
#              -0.0999696496106236, 0.0000000000000000,],
#             [0.0000000000000000, 0.0000000000000000, -0.0999696496106236,
#                 0.2016488078173110, -0.1016791582066870,],
#             [0.0000000000000000, 0.0000000000000000, 0.0000000000000000,
#              -0.1016791582066870, 0.1016791582066870,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix,
#         ))
#
#     def test_global_heat_storage_matrix_0(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [1.06020061728395E+07, 3.89011555173310E+07, 8.63025666982303E+06,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 8.63025666982303E+06, 3.34542102448499E+07,
#                 8.29817132739774E+06, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 8.29817132739774E+06,
#                 3.29568618779144E+07, 8.17817064124763E+06,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 8.17817064124763E+06, 1.63181792594571E+07,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix_0,
#         ))
#
#     def test_global_heat_storage_matrix(self):
#         expected_C = np.array([
#             [2.12040123456790E+07, 1.06020061728395E+07, 0.00000000000000E+00,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [1.06020061728395E+07, 1.14799363337692E+09, 3.21136209426181E+08,
#                 0.00000000000000E+00, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 3.21136209426181E+08, 2.06883882586324E+08,
#                 2.14790000604973E+07, 0.00000000000000E+00,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 2.14790000604973E+07,
#                 5.70646987491469E+07, 9.43497504608584E+06,],
#             [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
#                 9.43497504608584E+06, 1.74625357643405E+07,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.array([
#             -0.000000000000000E+00,
#             -7.240998977095210E-06,
#             -1.693636195537800E-06,
#             4.516088940272450E-07,
#             6.874586265312840E-02,
#         ])
#         self.assertTrue(np.allclose(expected_flux_vector_0,
#                                     self.msh._heat_flux_vector_0))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.array([
#             0.000000000000000E+00,
#             2.133070477819620E-05,
#             8.181590920372290E-06,
#             2.187351973225760E-07,
#             6.874521027822230E-02,
#         ])
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._heat_flux_vector))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             -2.00025900344127E+00,
#             -1.60722721278623E-04,
#             6.47788749226980E-04,
#             -1.36736823325139E-03,
#             1.14601596360676E-03,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_heat_flux_vector,
#         ))
#
#     def test_temperature_increment_vector(self):
#         expected_dT = np.array([
#             0.00000000000000E+00,
#             -1.60493177668852E-04,
#             6.47016410889105E-04,
#             -1.36532119621939E-03,
#             1.14270189049772E-03,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_temp_vector,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 1.5508312528406E-04
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#         self.assertEqual(self.msh._iter, 2)
#
#
# class TestInitializeGlobalSystemCubic(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#         )
#         initial_temp_vector = np.array([
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
#         initial_temp_rate_vector = np.array([
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
#                                  initial_temp_vector,
#                                  initial_temp_rate_vector,
#                                  ):
#             nd.temp = T0
#             nd.temp_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp,
#             bnd_value=-2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
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
#     def test_temperature_distribution_nodes(self):
#         expected_temp_vector = np.array([
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
#         actual_temp_nodes = np.array([
#             nd.temp for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_vector,
#                                     actual_temp_nodes))
#         self.assertTrue(np.allclose(expected_temp_vector,
#                                     self.msh._temp_vector))
#         self.assertTrue(np.allclose(expected_temp_vector,
#                                     self.msh._temp_vector_0))
#
#     def test_temperature_distribution_int_pts(self):
#         expected_temp_int_pts = np.array([
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
#         actual_temp_int_pts = np.array([
#             ip.temp for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_int_pts,
#                                     expected_temp_int_pts,
#                                     ))
#
#     def test_temperature_rate_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
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
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     actual_temp_rate_nodes,
#                                     ))
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     self.msh._temp_rate_vector,
#                                     ))
#
#     def test_temperature_rate_distribution_int_pts(self):
#         expected_temp_rate_int_pts = np.array([
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
#         actual_temp_rate_int_pts = np.array([
#             ip.temp_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_rate_int_pts,
#                                     expected_temp_rate_int_pts,
#                                     ))
#
#     def test_temperature_gradient_distribution(self):
#         expected_temp_gradient_int_pts = np.array([
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
#         actual_temp_gradient_int_pts = np.array([
#             ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
#                                     expected_temp_gradient_int_pts,
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
#     def test_vol_water_cont_temp_gradient_distribution(self):
#         expected_vol_water_cont_temp_gradient_int_pts = np.zeros(
#             self.msh.num_elements * 5
#         )
#         actual_vol_water_cont_temp_gradient_int_pts = np.array([
#             ip.vol_water_cont_temp_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(
#             actual_vol_water_cont_temp_gradient_int_pts,
#             expected_vol_water_cont_temp_gradient_int_pts,
#         ))
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
#     def test_thrm_cond_distribution(self):
#         expected_thrm_cond_int_pts = np.array([
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
#         actual_thrm_cond_int_pts = np.array([
#             ip.thrm_cond
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
#                                     expected_thrm_cond_int_pts,
#                                     atol=1e-30))
#
#     def test_global_heat_flow_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.3754127694384990, -0.4791349492121190,
#                 0.1366244795670250, -0.0329022997934044,],
#             [-0.4791349492121190, 1.0974938412970900,
#              -0.7555601286108480, 0.1372012365258810,],
#             [0.1366244795670250, -0.7555601286108480,
#                 1.1003657454416000, -0.4814300963977770,],
#             [-0.0329022997934044, 0.1372012365258810,
#              -0.4814300963977770, 0.3771311596653000,],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.3742278648957170, -0.4704043073588020,
#                 0.1245486051912600, -0.0283721627281755,],
#             [-0.4704043073588020, 1.0458021981448800,
#              -0.6891083023287820, 0.1137104115427070,],
#             [0.1245486051912600, -0.6891083023287820,
#                 0.9190461742111960, -0.3544864770736730,],
#             [-0.0283721627281755, 0.1137104115427070,
#              -0.3544864770736730, 0.2691482282591420,],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.2668216256972590, -0.3407384274106880,
#                 0.0973538364030538, -0.0234370346896240,],
#             [-0.3407384274106880, 0.7788306912244300,
#              -0.5354461002167960, 0.0973538364030537,],
#             [0.0973538364030538, -0.5354461002167960,
#                 0.7788306912244310, -0.3407384274106880,],
#             [-0.0234370346896240, 0.0973538364030537,
#              -0.3407384274106880, 0.2668216256972590,],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.3292202434390340, -0.4133999362112650,
#                 0.1112498768967810, -0.0270701841245502,],
#             [-0.4133999362112650, 0.9868437905990680,
#              -0.6971611631648800, 0.1237173087770760,],
#             [0.1112498768967810, -0.6971611631648800,
#                 1.0421402435851100, -0.4562289573170110,],
#             [-0.0270701841245502, 0.1237173087770760,
#              -0.4562289573170110, 0.3595818326644850,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix,
#         ))
#
#     def test_global_heat_storage_matrix(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [3.74901860610780E+06, 2.89593744482105E+06,
#              -1.05358398557321E+06, 5.54643193641815E+05,],
#             [2.89593744482105E+06, 1.88936621803465E+07,
#              -2.36630842102390E+06, -1.04792564897764E+06,],
#             [-1.05358398557321E+06, -2.36630842102390E+06,
#                 1.88757400422500E+07, 2.88462077162992E+06,],
#             [5.54643193641815E+05, -1.04792564897764E+06,
#                 2.88462077162992E+06, 3.73110189762532E+06,],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [3.74548978664947E+06, 2.83529217357612E+06,
#              -8.63799618191817E+05, 6.29634934688451E+05,],
#             [2.83529217357612E+06, 1.92531430053028E+07,
#              -3.32307881157745E+06, -1.41571111930997E+06,],
#             [-8.63799618191817E+05, -3.32307881157745E+06,
#                 2.30168957941050E+07, 3.93911517581243E+06,],
#             [6.29634934688451E+05, -1.41571111930997E+06,
#                 3.93911517581243E+06, 4.82119852843649E+06,],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329806E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442681E+05,],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857143E+06, -1.36311507936508E+06,],
#             [-1.36311507936508E+06, -3.06700892857143E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06,],
#             [7.19421847442681E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329806E+06,],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [4.26389587430359E+06, 3.18874298722127E+06,
#              -1.17518017569044E+06, 5.93504826831539E+05,],
#             [3.18874298722127E+06, 1.95951789440892E+07,
#              -2.51514790752116E+06, -1.07640022713129E+06,],
#             [-1.17518017569044E+06, -2.51514790752116E+06,
#                 1.94626651593996E+07, 2.99118309010297E+06,],
#             [5.93504826831539E+05, -1.07640022713129E+06,
#                 2.99118309010297E+06, 3.89099624195036E+06,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix,
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.array([
#             1.89850411575195E-08,
#             9.32804654257637E-09,
#             -4.31099786843920E-09,
#             -1.42748247257048E-06,
#             1.28559187259822E-05,
#             1.28558305047007E-05,
#             -1.42842430458346E-06,
#             -0.00000000000000E+00,
#             -0.00000000000000E+00,
#             1.10932858948307E-04,
#             7.12664676651715E-05,
#             -1.57821060328194E-05,
#             6.53002632662146E-02,
#         ])
#         self.assertTrue(np.allclose(expected_flux_vector,
#                                     self.msh._heat_flux_vector))
#
#
# class TestInitializeTimeStepCubic(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#         )
#         initial_temp_vector = np.array([
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
#         initial_temp_rate_vector = np.array([
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
#                                  initial_temp_vector,
#                                  initial_temp_rate_vector,
#                                  ):
#             nd.temp = T0
#             nd.temp_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp,
#             bnd_value=-2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 3.024E+05
#         self.msh.initialize_time_step()
#
#     def test_time_step_set(self):
#         self.assertAlmostEqual(self.msh._t0, 1.5)
#         self.assertAlmostEqual(self.msh._t1, 1.5 + 3.024e5)
#
#     def test_iteration_variables(self):
#         self.assertEqual(self.msh._eps_a, 1.0)
#         self.assertEqual(self.msh._iter, 0)
#
#     def test_temperature_distribution_nodes(self):
#         expected_temp_vector = np.array([
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
#         actual_temp_nodes = np.array([
#             nd.temp for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_vector,
#                                     actual_temp_nodes))
#         self.assertTrue(np.allclose(expected_temp_vector,
#                                     self.msh._temp_vector))
#         self.assertTrue(np.allclose(expected_temp_vector,
#                                     self.msh._temp_vector_0))
#
#     def test_temperature_distribution_int_pts(self):
#         expected_temp_int_pts = np.array([
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
#         actual_temp_int_pts = np.array([
#             ip.temp for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_int_pts,
#                                     expected_temp_int_pts))
#
#     def test_temperature_rate_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
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
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     actual_temp_rate_nodes))
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     self.msh._temp_rate_vector))
#
#     def test_temperature_rate_distribution_int_pts(self):
#         expected_temp_rate_int_pts = np.array([
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
#         actual_temp_rate_int_pts = np.array([
#             ip.temp_rate for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_rate_int_pts,
#                                     expected_temp_rate_int_pts))
#
#     def test_temperature_gradient_distribution(self):
#         expected_temp_gradient_int_pts = np.array([
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
#         actual_temp_gradient_int_pts = np.array([
#             ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
#                                     expected_temp_gradient_int_pts))
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
#     def test_vol_water_cont_temp_gradient_distribution(self):
#         expected_vol_water_cont_temp_gradient_int_pts = np.zeros(
#             self.msh.num_elements * 5
#         )
#         actual_vol_water_cont_temp_gradient_int_pts = np.array([
#             ip.vol_water_cont_temp_gradient
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(
#             actual_vol_water_cont_temp_gradient_int_pts,
#             expected_vol_water_cont_temp_gradient_int_pts,
#         ))
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
#     def test_thrm_cond_distribution(self):
#         expected_thrm_cond_int_pts = np.array([
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
#         actual_thrm_cond_int_pts = np.array([
#             ip.thrm_cond
#             for e in self.msh.elements for ip in e.int_pts
#         ])
#         self.assertTrue(np.allclose(actual_thrm_cond_int_pts,
#                                     expected_thrm_cond_int_pts,
#                                     atol=1e-30))
#
#     def test_global_heat_flow_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.3754127694384990, -0.4791349492121190,
#                 0.1366244795670250, -0.0329022997934044,],
#             [-0.4791349492121190, 1.0974938412970900,
#              -0.7555601286108480, 0.1372012365258810,],
#             [0.1366244795670250, -0.7555601286108480,
#                 1.1003657454416000, -0.4814300963977770,],
#             [-0.0329022997934044, 0.1372012365258810,
#              -0.4814300963977770, 0.3771311596653000,],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.3742278648957170, -0.4704043073588020,
#                 0.1245486051912600, -0.0283721627281755,],
#             [-0.4704043073588020, 1.0458021981448800,
#              -0.6891083023287820, 0.1137104115427070,],
#             [0.1245486051912600, -0.6891083023287820,
#                 0.9190461742111960, -0.3544864770736730,],
#             [-0.0283721627281755, 0.1137104115427070,
#              -0.3544864770736730, 0.2691482282591420,],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.2668216256972590, -0.3407384274106880,
#                 0.0973538364030538, -0.0234370346896240,],
#             [-0.3407384274106880, 0.7788306912244300,
#              -0.5354461002167960, 0.0973538364030537,],
#             [0.0973538364030538, -0.5354461002167960,
#                 0.7788306912244310, -0.3407384274106880,],
#             [-0.0234370346896240, 0.0973538364030537,
#              -0.3407384274106880, 0.2668216256972590,],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.3292202434390340, -0.4133999362112650,
#                 0.1112498768967810, -0.0270701841245502,],
#             [-0.4133999362112650, 0.9868437905990680,
#              -0.6971611631648800, 0.1237173087770760,],
#             [0.1112498768967810, -0.6971611631648800,
#                 1.0421402435851100, -0.4562289573170110,],
#             [-0.0270701841245502, 0.1237173087770760,
#              -0.4562289573170110, 0.3595818326644850,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix,
#         ))
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix_0,
#         ))
#
#     def test_global_heat_storage_matrix(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [3.74901860610780E+06, 2.89593744482105E+06,
#              -1.05358398557321E+06, 5.54643193641815E+05,],
#             [2.89593744482105E+06, 1.88936621803465E+07,
#              -2.36630842102390E+06, -1.04792564897764E+06,],
#             [-1.05358398557321E+06, -2.36630842102390E+06,
#                 1.88757400422500E+07, 2.88462077162992E+06,],
#             [5.54643193641815E+05, -1.04792564897764E+06,
#                 2.88462077162992E+06, 3.73110189762532E+06,],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [3.74548978664947E+06, 2.83529217357612E+06,
#              -8.63799618191817E+05, 6.29634934688451E+05,],
#             [2.83529217357612E+06, 1.92531430053028E+07,
#              -3.32307881157745E+06, -1.41571111930997E+06,],
#             [-8.63799618191817E+05, -3.32307881157745E+06,
#                 2.30168957941050E+07, 3.93911517581243E+06,],
#             [6.29634934688451E+05, -1.41571111930997E+06,
#                 3.93911517581243E+06, 4.82119852843649E+06,],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329806E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442681E+05,],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857143E+06, -1.36311507936508E+06,],
#             [-1.36311507936508E+06, -3.06700892857143E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06,],
#             [7.19421847442681E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329806E+06,],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [4.26389587430359E+06, 3.18874298722127E+06,
#              -1.17518017569044E+06, 5.93504826831539E+05,],
#             [3.18874298722127E+06, 1.95951789440892E+07,
#              -2.51514790752116E+06, -1.07640022713129E+06,],
#             [-1.17518017569044E+06, -2.51514790752116E+06,
#                 1.94626651593996E+07, 2.99118309010297E+06,],
#             [5.93504826831539E+05, -1.07640022713129E+06,
#                 2.99118309010297E+06, 3.89099624195036E+06,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix,
#         ))
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix_0,
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.array([
#             1.89850411575195E-08,
#             9.32804654257637E-09,
#             -4.31099786843920E-09,
#             -1.42748247257048E-06,
#             1.28559187259822E-05,
#             1.28558305047007E-05,
#             -1.42842430458346E-06,
#             -0.00000000000000E+00,
#             -0.00000000000000E+00,
#             1.10932858948307E-04,
#             7.12664676651715E-05,
#             -1.57821060328194E-05,
#             6.53002632662146E-02,
#         ])
#         self.assertTrue(np.allclose(
#             expected_flux_vector,
#             self.msh._heat_flux_vector,
#         ))
#         self.assertTrue(np.allclose(
#             expected_flux_vector,
#             self.msh._heat_flux_vector_0,
#         ))
#
#
# class TestTemperatureCorrectionCubicOneStep(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#         )
#         initial_temp_vector = np.array([
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
#         initial_temp_rate_vector = np.array([
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
#                                  initial_temp_vector,
#                                  initial_temp_rate_vector,
#                                  ):
#             nd.temp = T0
#             nd.temp_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp,
#             bnd_value=-2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 3.024E+05
#         self.msh.initialize_time_step()
#         self.msh._temp_vector[:] = np.array([
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
#         self.msh._temp_rate_vector[:] = np.array([
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
#         self.msh.update_boundary_conditions(self.msh._t1)
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_global_matrices_and_vectors()
#         self.msh.calculate_solution_vector_correction()
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_global_matrices_and_vectors()
#         self.msh.update_iteration_variables()
#
#     def test_temperature_distribution_nodes(self):
#         expected_temp_vector_0 = np.array([
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
#         expected_temp_vector = np.array([
#             -2.000000000000000,
#             -9.082587139505380,
#             -10.479659191427400,
#             -7.632980191952010,
#             -3.388514757134900,
#             0.178062168417391,
#             1.968571182872450,
#             2.056890544111210,
#             1.158018885633480,
#             0.100916692926800,
#             -0.547117529099153,
#             -0.607000081807634,
#             -0.210024671835248,
#         ])
#         actual_temp_nodes = np.array([
#             nd.temp for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(
#             expected_temp_vector,
#             actual_temp_nodes,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_vector,
#             self.msh._temp_vector,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_vector_0,
#             self.msh._temp_vector_0,
#         ))
#
#     def test_temperature_rate_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
#             0.00000000000000E+00,
#             2.47570047890236E-07,
#             2.85733795580311E-08,
#             1.33019023210993E-07,
#             -2.87130590349577E-08,
#             -2.65305419384922E-08,
#             -2.42771971630921E-08,
#             -9.41484392280149E-09,
#             -9.95861051620030E-10,
#             1.30146777036420E-09,
#             5.40137966509952E-09,
#             7.56209371863997E-09,
#             -1.38333740500700E-08,
#         ])
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(
#             expected_temp_rate_vector,
#             actual_temp_rate_nodes,
#             atol=1e-12, rtol=1e-10,
#         ))
#         self.assertTrue(np.allclose(
#             expected_temp_rate_vector,
#             self.msh._temp_rate_vector,
#             atol=1e-12, rtol=1e-10,
#         ))
#
#     def test_global_heat_flow_matrix_0(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.3754127694384990, -0.4791349492121190,
#                 0.1366244795670250, -0.0329022997934044,],
#             [-0.4791349492121190, 1.0974938412970900,
#              -0.7555601286108480, 0.1372012365258810,],
#             [0.1366244795670250, -0.7555601286108480,
#                 1.1003657454416000, -0.4814300963977770,],
#             [-0.0329022997934044, 0.1372012365258810,
#              -0.4814300963977770, 0.3771311596653000,],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.3742278648957170, -0.4704043073588020,
#                 0.1245486051912600, -0.0283721627281755,],
#             [-0.4704043073588020, 1.0458021981448800,
#              -0.6891083023287820, 0.1137104115427070,],
#             [0.1245486051912600, -0.6891083023287820,
#                 0.9190461742111960, -0.3544864770736730,],
#             [-0.0283721627281755, 0.1137104115427070,
#              -0.3544864770736730, 0.2691482282591420,],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.2668216256972590, -0.3407384274106880,
#                 0.0973538364030538, -0.0234370346896240,],
#             [-0.3407384274106880, 0.7788306912244300,
#              -0.5354461002167960, 0.0973538364030537,],
#             [0.0973538364030538, -0.5354461002167960,
#                 0.7788306912244310, -0.3407384274106880,],
#             [-0.0234370346896240, 0.0973538364030537,
#              -0.3407384274106880, 0.2668216256972590,],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.3292202434390340, -0.4133999362112650,
#                 0.1112498768967810, -0.0270701841245502,],
#             [-0.4133999362112650, 0.9868437905990680,
#              -0.6971611631648800, 0.1237173087770760,],
#             [0.1112498768967810, -0.6971611631648800,
#                 1.0421402435851100, -0.4562289573170110,],
#             [-0.0270701841245502, 0.1237173087770760,
#              -0.4562289573170110, 0.3595818326644850,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix_0,
#         ))
#
#     def test_global_heat_flow_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.3753879416232020, -0.4791042467337650,
#                 0.1366167050145040, -0.0329003999039413,],
#             [-0.4791042467337650, 1.0974433676402300, -
#                 0.7555344966562600, 0.1371953757497930,],
#             [0.1366167050145040, -0.7555344966562600,
#                 1.1003399482612500, -0.4814221566194900,],
#             [-0.0329003999039413, 0.1371953757497930, -
#                 0.4814221566194900, 0.3771271807736380,],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.3742218218850550, -0.4703963231115380,
#                 0.1245458628648330, -0.0283713616383489,],
#             [-0.4703963231115380, 1.0458235893735000, -
#                 0.6891377585117370, 0.1137104922497770,],
#             [0.1245458628648330, -0.6891377585117370,
#                 0.9190792031511250, -0.3544873075042210,],
#             [-0.0283713616383489, 0.1137104922497770, -
#                 0.3544873075042210, 0.2691481768927930,],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.2668216256972590, -0.3407384274106880,
#                 0.0973538364030538, -0.0234370346896240,],
#             [-0.3407384274106880, 0.7788306912244300, -
#                 0.5354461002167960, 0.0973538364030537,],
#             [0.0973538364030538, -0.5354461002167960,
#                 0.7788306912244310, -0.3407384274106880,],
#             [-0.0234370346896240, 0.0973538364030537, -
#                 0.3407384274106880, 0.2668216256972590,],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.3289962966660710, -0.4130762140338010,
#                 0.1111334904406820, -0.0270535730729519,],
#             [-0.4130762140338010, 0.9863568813802830, -
#                 0.6970058557736110, 0.1237251884271290,],
#             [0.1111334904406820, -0.6970058557736110,
#                 1.0421897385013800, -0.4563173731684510,],
#             [-0.0270535730729519, 0.1237251884271290, -
#                 0.4563173731684510, 0.3596457578142740,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix,
#         ))
#
#     def test_global_heat_storage_matrix_0(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [3.74901860610780E+06, 2.89593744482105E+06,
#              -1.05358398557321E+06, 5.54643193641815E+05,],
#             [2.89593744482105E+06, 1.88936621803465E+07,
#              -2.36630842102390E+06, -1.04792564897764E+06,],
#             [-1.05358398557321E+06, -2.36630842102390E+06,
#                 1.88757400422500E+07, 2.88462077162992E+06,],
#             [5.54643193641815E+05, -1.04792564897764E+06,
#                 2.88462077162992E+06, 3.73110189762532E+06,],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [3.74548978664947E+06, 2.83529217357612E+06,
#              -8.63799618191817E+05, 6.29634934688451E+05,],
#             [2.83529217357612E+06, 1.92531430053028E+07,
#              -3.32307881157745E+06, -1.41571111930997E+06,],
#             [-8.63799618191817E+05, -3.32307881157745E+06,
#                 2.30168957941050E+07, 3.93911517581243E+06,],
#             [6.29634934688451E+05, -1.41571111930997E+06,
#                 3.93911517581243E+06, 4.82119852843649E+06,],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329806E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442681E+05,],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857143E+06, -1.36311507936508E+06,],
#             [-1.36311507936508E+06, -3.06700892857143E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06,],
#             [7.19421847442681E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329806E+06,],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [4.26389587430359E+06, 3.18874298722127E+06,
#              -1.17518017569044E+06, 5.93504826831539E+05,],
#             [3.18874298722127E+06, 1.95951789440892E+07,
#              -2.51514790752116E+06, -1.07640022713129E+06,],
#             [-1.17518017569044E+06, -2.51514790752116E+06,
#                 1.94626651593996E+07, 2.99118309010297E+06,],
#             [5.93504826831539E+05, -1.07640022713129E+06,
#                 2.99118309010297E+06, 3.89099624195036E+06,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix_0,
#         ))
#
#     def test_global_heat_storage_matrix(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [4.74285700814038E+06, 3.49344088380004E+06, -
#                 1.29807950820942E+06, 6.28942731006372E+05,],
#             [3.49344088380004E+06, 2.04775275805259E+07, -
#                 2.69184560214359E+06, -1.08352184762866E+06,],
#             [-1.29807950820942E+06, -2.69184560214359E+06,
#                 1.99795376252816E+07, 3.06432556263851E+06,],
#             [6.28942731006372E+05, -1.08352184762866E+06,
#                 3.06432556263851E+06, 4.00294353743635E+06,],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [4.17010553124093E+06, 2.80148011856716E+06, -
#                 1.53913344761798E+06, 7.18402432622841E+05,],
#             [2.80148011856716E+06, 2.63128374675488E+07,
#                 8.30360330006419E+05, -1.83207088751921E+06,],
#             [-1.53913344761798E+06, 8.30360330006419E+05,
#                 2.78795373209754E+07, 3.38735499836960E+06,],
#             [7.18402432622841E+05, -1.83207088751921E+06,
#                 3.38735499836960E+06, 4.88471921523720E+06,],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329806E+06, 3.74856646825397E+06, -
#                 1.36311507936508E+06, 7.19421847442681E+05,],
#             [3.74856646825397E+06, 2.45360714285714E+07, -
#                 3.06700892857143E+06, -1.36311507936508E+06,],
#             [-1.36311507936508E+06, -3.06700892857143E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06,],
#             [7.19421847442681E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329806E+06,],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [1.05184695178840E+09, 5.27408401740155E+08, -
#                 2.39636875141802E+08, 5.53787670808823E+07,],
#             [5.27408401740155E+08, 3.86393873685784E+08, -
#                 1.44403954369652E+08, 1.44997490061072E+07,],
#             [-2.39636875141802E+08, -1.44403954369652E+08,
#                 1.69498523604074E+08, 1.91351534443363E+07,],
#             [5.53787670808823E+07, 1.44997490061072E+07,
#                 1.91351534443363E+07, 5.11240998577955E+07,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.array([
#             1.89850411575195E-08,
#             9.32804654257637E-09,
#             -4.31099786843920E-09,
#             -1.42748247257048E-06,
#             1.28559187259822E-05,
#             1.28558305047007E-05,
#             -1.42842430458346E-06,
#             -0.00000000000000E+00,
#             -0.00000000000000E+00,
#             1.10932858948307E-04,
#             7.12664676651715E-05,
#             -1.57821060328194E-05,
#             6.53002632662146E-02,
#         ])
#         self.assertTrue(np.allclose(
#             expected_flux_vector_0,
#             self.msh._heat_flux_vector_0,
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.array([
#             5.46746653845989E-09,
#             2.68636671492506E-09,
#             -1.24151622646639E-09,
#             -7.60441975353691E-07,
#             6.84647489228791E-06,
#             6.84643509031610E-06,
#             -7.60714420349861E-07,
#             -0.00000000000000E+00,
#             -0.00000000000000E+00,
#             5.41348756985190E-05,
#             3.30150985940221E-05,
#             -7.78256844391883E-06,
#             6.53078244526887E-02,
#         ])
#         self.assertTrue(np.allclose(
#             expected_flux_vector,
#             self.msh._heat_flux_vector,
#         ))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             -2.38461234636085E-01,
#             7.59224243604616E-02,
#             7.99943496297368E-03,
#             4.17791714043375E-02,
#             -8.96429211407198E-03,
#             -7.86598279264764E-03,
#             -7.63908219256820E-03,
#             -2.82298628508657E-03,
#             -3.20429239722832E-04,
#             3.96237468132868E-04,
#             1.63711247439950E-03,
#             2.30096034332267E-03,
#             -4.19458710959875E-03,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_heat_flux_vector,
#         ))
#
#     def test_temperature_increment_vector(self):
#         expected_dT = np.array([
#             0.00000000000000E+00,
#             7.49567570052084E-02,
#             8.74547297620121E-03,
#             4.03016846701951E-02,
#             -8.64903073239771E-03,
#             -8.02469673177830E-03,
#             -7.36118354840201E-03,
#             -2.86764617815346E-03,
#             -3.12731582359419E-04,
#             3.92558622480283E-04,
#             1.63886471997195E-03,
#             2.29287000911671E-03,
#             -4.18115389772324E-03,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_temp_vector,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 5.22652045961174E-03
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#         self.assertEqual(self.msh._iter, 1)
#
#
# class TestIterativeTemperatureCorrectionCubic(unittest.TestCase):
#     def setUp(self):
#         self.mtl = Material(
#             thrm_cond_solids=3.0,
#             spec_heat_cap_solids=741.0,
#             spec_grav_solids=2.65,
#             deg_sat_water_alpha=1.20e4,
#             deg_sat_water_beta=0.35,
#             water_flux_b1=0.08,
#             water_flux_b2=4.0,
#             water_flux_b3=1.0e-5,
#             seg_pot_0=2.0e-9,
#         )
#         self.msh = CoupledAnalysis1D(
#             z_range=(0, 100),
#             num_elements=4,
#             generate=True,
#         )
#         initial_temp_vector = np.array([
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
#         initial_temp_rate_vector = np.array([
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
#                                  initial_temp_vector,
#                                  initial_temp_rate_vector,
#                                  ):
#             nd.temp = T0
#             nd.temp_rate = dTdt0
#         for e in self.msh.elements:
#             for ip in e.int_pts:
#                 ip.material = self.mtl
#                 ip.void_ratio = 0.35
#                 ip.void_ratio_0 = 0.3
#                 ip.tot_stress = 1.2e5
#         bnd0 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[0],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp,
#             bnd_value=-2.0,
#         )
#         self.msh.add_boundary(bnd0)
#         bnd1 = ThermalBoundary1D(
#             nodes=(self.msh.nodes[-1],),
#             int_pts=(self.msh.elements[-1].int_pts[-1],),
#             bnd_type=ThermalBoundary1D.BoundaryType.temp_grad,
#             bnd_value=25.0e-3,
#         )
#         self.msh.add_boundary(bnd1)
#         self.msh.initialize_global_system(1.5)
#         self.msh.time_step = 3.024E+5
#         self.msh.initialize_time_step()
#         self.msh._temp_vector[:] = np.array([
#             -2.000000000000000,
#             -9.082587137736640,
#             -10.479659195340700,
#             -7.632980166438240,
#             -3.388514806411960,
#             0.178062121944726,
#             1.968571203878090,
#             2.056890541011660,
#             1.158018886579530,
#             0.100916691640023,
#             -0.547117547373778,
#             -0.607000082862531,
#             -0.210024714103347,
#         ])
#         self.msh._temp_rate_vector[:] = np.array([
#             0.00000000000000E+00,
#             -3.02825804238568E-10,
#             -3.46835310360251E-10,
#             -2.53743555523472E-10,
#             -1.11766930468156E-10,
#             6.15360310271584E-12,
#             6.53410260679122E-11,
#             6.81130155364171E-11,
#             3.83042339601586E-11,
#             3.32417750613871E-12,
#             -1.81465252840407E-11,
#             -2.01483749999432E-11,
#             -6.80692796927300E-12,
#         ])
#         self.msh.update_boundary_conditions(self.msh._t1)
#         self.msh.update_nodes()
#         self.msh.update_integration_points()
#         self.msh.update_global_matrices_and_vectors()
#         self.msh.iterative_correction_step()
#
#     def test_temperature_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
#             0.00000000000000E+00,
#             2.47498817526917E-07,
#             2.85763254553079E-08,
#             1.32984448853519E-07,
#             -2.87329006419087E-08,
#             -2.65614768071763E-08,
#             -2.42586956207760E-08,
#             -9.41748664381412E-09,
#             -9.95379039448084E-10,
#             1.30275115736807E-09,
#             5.39880674132476E-09,
#             7.56091046420124E-09,
#             -1.38388620278036E-08,
#         ])
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     actual_temp_rate_nodes,
#                                     ))
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     self.msh._temp_rate_vector,
#                                     ))
#
#     def test_temperature_rate_distribution_nodes(self):
#         expected_temp_rate_vector = np.array([
#             0.00000000000000E+00,
#             2.47498817526917E-07,
#             2.85763254553079E-08,
#             1.32984448853519E-07,
#             -2.87329006419087E-08,
#             -2.65614768071763E-08,
#             -2.42586956207760E-08,
#             -9.41748664381412E-09,
#             -9.95379039448084E-10,
#             1.30275115736807E-09,
#             5.39880674132476E-09,
#             7.56091046420124E-09,
#             -1.38388620278036E-08,
#         ])
#         actual_temp_rate_nodes = np.array([
#             nd.temp_rate for nd in self.msh.nodes
#         ])
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     actual_temp_rate_nodes,
#                                     atol=1e-12, rtol=1e-10,
#                                     ))
#         self.assertTrue(np.allclose(expected_temp_rate_vector,
#                                     self.msh._temp_rate_vector,
#                                     atol=1e-12, rtol=1e-10,
#                                     ))
#
#     def test_global_heat_flow_matrix_0(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.3754127694384990, -0.4791349492121190,
#                 0.1366244795670250, -0.0329022997934044,],
#             [-0.4791349492121190, 1.0974938412970900,
#              -0.7555601286108480, 0.1372012365258810,],
#             [0.1366244795670250, -0.7555601286108480,
#                 1.1003657454416000, -0.4814300963977770,],
#             [-0.0329022997934044, 0.1372012365258810,
#              -0.4814300963977770, 0.3771311596653000,],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.3742278648957170, -0.4704043073588020,
#                 0.1245486051912600, -0.0283721627281755,],
#             [-0.4704043073588020, 1.0458021981448800,
#              -0.6891083023287820, 0.1137104115427070,],
#             [0.1245486051912600, -0.6891083023287820,
#                 0.9190461742111960, -0.3544864770736730,],
#             [-0.0283721627281755, 0.1137104115427070,
#              -0.3544864770736730, 0.2691482282591420,],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.2668216256972590, -0.3407384274106880,
#                 0.0973538364030538, -0.0234370346896240,],
#             [-0.3407384274106880, 0.7788306912244300,
#              -0.5354461002167960, 0.0973538364030537,],
#             [0.0973538364030538, -0.5354461002167960,
#                 0.7788306912244310, -0.3407384274106880,],
#             [-0.0234370346896240, 0.0973538364030537,
#              -0.3407384274106880, 0.2668216256972590,],
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.3292202434390340, -0.4133999362112650,
#                 0.1112498768967810, -0.0270701841245502,],
#             [-0.4133999362112650, 0.9868437905990680,
#              -0.6971611631648800, 0.1237173087770760,],
#             [0.1112498768967810, -0.6971611631648800,
#                 1.0421402435851100, -0.4562289573170110,],
#             [-0.0270701841245502, 0.1237173087770760,
#              -0.4562289573170110, 0.3595818326644850,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix_0,
#         ))
#
#     def test_global_heat_flow_matrix(self):
#         expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_H[0:4, 0:4] += np.array([
#             [0.3753879492274280, -0.4791042561238620,
#              0.1366167073707450, -0.0329004004743101,],
#             [-0.4791042561238620, 1.0974433825255700,
#              -0.7555345038239140, 0.1371953774222090,],
#             [0.1366167073707450, -0.7555345038239140,
#                 1.1003399547735400, -0.4814221583203750,],
#             [-0.0329004004743101, 0.1371953774222090,
#                 -0.4814221583203750, 0.3771271813724750,],
#         ])
#         expected_H[3:7, 3:7] += np.array([
#             [0.3742218246395010, -0.4703963271434800,
#              0.1245458642680330, -0.0283713617640532,],
#             [-0.4703963271434800, 1.0458236191503300,
#              -0.6891377855047830, 0.1137104934979320,],
#             [0.1245458642680330, -0.6891377855047830,
#                 0.9190792299556210, -0.3544873087188710,],
#             [-0.0283713617640532, 0.1137104934979320,
#                 -0.3544873087188710, 0.2691481769849920,],
#         ])
#         expected_H[6:10, 6:10] += np.array([
#             [0.2668216256972600, -0.3407384274106900,
#              0.0973538364030547, -0.0234370346896243,],
#             [-0.3407384274106900, 0.7788306912244320,
#                 -0.5354461002167970, 0.0973538364030541,],
#             [0.0973538364030547, -0.5354461002167970,
#                 0.7788306912244330, -0.3407384274106900,],
#             [-0.0234370346896243, 0.0973538364030541,
#                 -0.3407384274106900, 0.2668216256972600,],
#
#         ])
#         expected_H[9:13, 9:13] += np.array([
#             [0.3289962995618880, -0.4130762153818610,
#              0.1111334919996520, -0.0270535761796790,],
#             [-0.4130762153818610, 0.9863568971153020,
#              -0.6970058899187200, 0.1237252081852800,],
#             [0.1111334919996520, -0.6970058899187200,
#                 1.0421898250233900, -0.4563174271043250,],
#             [-0.0270535761796790, 0.1237252081852800,
#              -0.4563174271043250, 0.3596457950987240,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_H, self.msh._heat_flow_matrix,
#         ))
#
#     def test_global_heat_storage_matrix_0(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [3.74901860610780E+06, 2.89593744482105E+06,
#              -1.05358398557321E+06, 5.54643193641815E+05,],
#             [2.89593744482105E+06, 1.88936621803465E+07,
#              -2.36630842102390E+06, -1.04792564897764E+06,],
#             [-1.05358398557321E+06, -2.36630842102390E+06,
#                 1.88757400422500E+07, 2.88462077162992E+06,],
#             [5.54643193641815E+05, -1.04792564897764E+06,
#                 2.88462077162992E+06, 3.73110189762532E+06,],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [3.74548978664947E+06, 2.83529217357612E+06,
#              -8.63799618191817E+05, 6.29634934688451E+05,],
#             [2.83529217357612E+06, 1.92531430053028E+07,
#              -3.32307881157745E+06, -1.41571111930997E+06,],
#             [-8.63799618191817E+05, -3.32307881157745E+06,
#                 2.30168957941050E+07, 3.93911517581243E+06,],
#             [6.29634934688451E+05, -1.41571111930997E+06,
#                 3.93911517581243E+06, 4.82119852843649E+06,],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329806E+06, 3.74856646825397E+06,
#              -1.36311507936508E+06, 7.19421847442681E+05,],
#             [3.74856646825397E+06, 2.45360714285714E+07,
#              -3.06700892857143E+06, -1.36311507936508E+06,],
#             [-1.36311507936508E+06, -3.06700892857143E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06,],
#             [7.19421847442681E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329806E+06,],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [4.26389587430359E+06, 3.18874298722127E+06,
#              -1.17518017569044E+06, 5.93504826831539E+05,],
#             [3.18874298722127E+06, 1.95951789440892E+07,
#              -2.51514790752116E+06, -1.07640022713129E+06,],
#             [-1.17518017569044E+06, -2.51514790752116E+06,
#                 1.94626651593996E+07, 2.99118309010297E+06,],
#             [5.93504826831539E+05, -1.07640022713129E+06,
#                 2.99118309010297E+06, 3.89099624195036E+06,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix_0,
#         ))
#
#     def test_global_heat_storage_matrix(self):
#         expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
#         expected_C[0:4, 0:4] += np.array([
#             [4.74285500589966E+06, 3.49343955102212E+06, -
#                 1.29807894103605E+06, 6.28942612193995E+05,],
#             [3.49343955102212E+06, 2.04775242196110E+07, -
#                 2.69184500685009E+06, -1.08352197936191E+06,],
#             [-1.29807894103605E+06, -2.69184500685009E+06,
#                 1.99795374233527E+07, 3.06432562767385E+06,],
#             [6.28942612193995E+05, -1.08352197936191E+06,
#                 3.06432562767385E+06, 4.00294343313938E+06,],
#         ])
#         expected_C[3:7, 3:7] += np.array([
#             [4.17010491590658E+06, 2.80148200739218E+06, -
#                 1.53913095787304E+06, 7.18402142957111E+05,],
#             [2.80148200739218E+06, 2.63128141962097E+07,
#                 8.30339725061684E+05, -1.83206863961090E+06,],
#             [-1.53913095787304E+06, 8.30339725061684E+05,
#                 2.78795160638767E+07, 3.38735737086790E+06,],
#             [7.18402142957111E+05, -1.83206863961090E+06,
#                 3.38735737086790E+06, 4.88471894957745E+06,],
#         ])
#         expected_C[6:10, 6:10] += np.array([
#             [4.84663139329808E+06, 3.74856646825397E+06, -
#                 1.36311507936508E+06, 7.19421847442682E+05,],
#             [3.74856646825397E+06, 2.45360714285714E+07, -
#                 3.06700892857144E+06, -1.36311507936508E+06,],
#             [-1.36311507936508E+06, -3.06700892857144E+06,
#                 2.45360714285714E+07, 3.74856646825397E+06,],
#             [7.19421847442682E+05, -1.36311507936508E+06,
#                 3.74856646825397E+06, 4.84663139329808E+06,],
#         ])
#         expected_C[9:13, 9:13] += np.array([
#             [1.05184703160211E+09, 5.27408427089753E+08, -
#                 2.39636893587137E+08, 5.53787629545943E+07,],
#             [5.27408427089753E+08, 3.86393746848379E+08, -
#                 1.44403919746439E+08, 1.44997802492022E+07,],
#             [-2.39636893587137E+08, -1.44403919746439E+08,
#                 1.69498436030225E+08, 1.91350794170755E+07,],
#             [5.53787629545943E+07, 1.44997802492022E+07,
#                 1.91350794170755E+07, 5.11239567880632E+07,],
#         ])
#         self.assertTrue(np.allclose(
#             expected_C, self.msh._heat_storage_matrix,
#         ))
#
#     def test_global_flux_vector_0(self):
#         expected_flux_vector_0 = np.array([
#             1.89850411575195E-08,
#             9.32804654257637E-09,
#             -4.31099786843920E-09,
#             -1.42748247257048E-06,
#             1.28559187259822E-05,
#             1.28558305047007E-05,
#             -1.42842430458346E-06,
#             -0.00000000000000E+00,
#             -0.00000000000000E+00,
#             1.10932858948307E-04,
#             7.12664676651715E-05,
#             -1.57821060328194E-05,
#             6.53002632662146E-02,
#         ])
#         self.assertTrue(np.allclose(
#             expected_flux_vector_0,
#             self.msh._heat_flux_vector_0,
#         ))
#
#     def test_global_flux_vector(self):
#         expected_flux_vector = np.array([
#             5.46755066490460E-09,
#             2.68640804931463E-09,
#             -1.24153532934089E-09,
#             -7.60449037120510E-07,
#             6.84653850755319E-06,
#             6.84649869388256E-06,
#             -7.60721487239478E-07,
#             -0.00000000000000E+00,
#             -0.00000000000000E+00,
#             5.41352133532762E-05,
#             3.30157898019445E-05,
#             -7.78264191890823E-06,
#             6.53078337911065E-02,
#         ])
#         self.assertTrue(np.allclose(
#             expected_flux_vector,
#             self.msh._heat_flux_vector,
#         ))
#
#     def test_global_residual_vector(self):
#         expected_Psi = np.array([
#             -2.36065087465956E-01,
#             -2.18385250792819E-05,
#             1.12974581959087E-06,
#             -1.08848129651701E-05,
#             -5.90483595852911E-06,
#             -9.44720627751167E-06,
#             5.79182251906826E-06,
#             -8.43658542483070E-07,
#             1.56954993024943E-07,
#             3.88509427186182E-07,
#             -7.78294034245843E-07,
#             -3.56285036477028E-07,
#             -1.66211459323978E-06,
#         ])
#         self.assertTrue(np.allclose(
#             expected_Psi, self.msh._residual_heat_flux_vector,
#             atol=1e-7,
#         ))
#
#     def test_temperature_increment_vector(self):
#         expected_dT = np.array([
#             0.00000000000000E+00,
#             -2.15400636773106E-05,
#             8.90839395904079E-07,
#             -1.04552863107220E-05,
#             -6.00010207255151E-06,
#             -9.35470456152675E-06,
#             5.59486659083009E-06,
#             -7.99158862030743E-07,
#             1.45760487476165E-07,
#             3.88096232875535E-07,
#             -7.78052157469524E-07,
#             -3.57816149462257E-07,
#             -1.65956446958961E-06,
#         ])
#         self.assertTrue(np.allclose(
#             expected_dT, self.msh._delta_temp_vector,
#             rtol=1e-3, atol=1e-7,
#         ))
#
#     def test_iteration_variables(self):
#         expected_eps_a = 1.62910190189313E-06
#         self.assertEqual(self.msh._iter, 1)
#         self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
#
#
if __name__ == "__main__":
    unittest.main()
