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
            void_ratio_tr=0.0,
            void_ratio_sep=1.6,
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
            void_ratio_tr=0.0,
            void_ratio_sep=1.6,
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
            2.66666666667,
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
        expected_void_ratio_vector_0 = np.array([
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
        ])
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
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts_0 = np.array([
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
        ])
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
                                    expected_void_ratio_int_pts_0))
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
            7.69716904600309E-09,
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

    def test_loc_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array([
            0.00000000000000E+00,
            2.24341159206153E+50,
            1.54021482997682E+04,
            1.55938230814525E+04,
            1.57341387601751E+04,
            1.59258135418595E+04,
            1.60661292205821E+04,
            1.62578040022664E+04,
        ])
        actual_sig_int_pts = np.array([
            ip.loc_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sig_int_pts,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            7.27181296175159E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
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
            -6.26996013389023E+52,
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
        self.assertTrue(np.allclose(
            expected_dsigde_int_pts,
            actual_dsigde_int_pts,
        ))

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array([
            -2.09299601559626E+02,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
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
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
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
        expected = 0.00587428488533737
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
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [1.00416479969625E+02, -1.00416479969625E+02,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-1.00416479969625E+02, 1.84756492052618E+02,
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
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [1.95272432649798E+05, 5.85438655571886E+04, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [5.85438655571886E+04, 5.59519989272211E+04, 8.52448467413248E+03,
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
        expected_Phi = np.array([
            -8.33935003943986E-02,
            -2.24222554544176E-02,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
            -0.00000000000000E+00,
        ])
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
        expected_K = np.array([
            [2.59527068320373E-09, -2.59527068320373E-09, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [-1.81079078550582E-10, 1.81079078550582E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
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
            expected_M, self.msh._mass_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [2.16333267482736E-03, 1.04135499405632E-03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.04135499405632E-03, 3.98922104937246E-03, 9.93566873987276E-04,
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
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))


class TestGlobalCorrectionLinearOneStep(unittest.TestCase):
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
            void_ratio_tr=0.0,
            void_ratio_sep=1.6,
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
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points_primary()
        self.msh.calculate_deformed_coords()
        self.msh.update_total_stress_distribution()
        self.msh.update_integration_points_secondary()
        self.msh.update_pore_pressure_distribution()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

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
            5.00000000000000,
            -4.93317614973010,
            -5.01741240743522,
            -4.99519948417723,
            -5.00233408103877,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_nodes,
                                    expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            2.90087288711227,
            -2.83404903684237,
            -4.95097736555187,
            -4.99961119161345,
            -5.01271826441747,
            -4.99989362719498,
            -4.99670720189871,
            -5.00082636331728,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array([
            2.66666666666667E+00,
            1.78196934053065E-02,
            -4.64330864939271E-03,
            1.28013755273978E-03,
            -6.22421610338364E-04,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_rate_nodes,
                                    expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            2.10689943656328E+00,
            5.77586923508701E-01,
            1.30727025195007E-02,
            1.03682236413034E-04,
            -3.39153717799219E-03,
            2.83660813392684E-05,
            8.78079493676889E-04,
            -2.20363551275469E-04,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -397.327045989204000,
            -397.327045989204000,
            -3.369450308204880,
            -3.369450308204880,
            0.888516930319867,
            0.888516930319867,
            -0.285383874461701,
            -0.285383874461701,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.0000000000000000,
            0.0496820848820785,
            0.0367146086279011,
            0.0365201051556731,
            0.0364681812180862,
            0.0365189840806239,
            0.0365316376352820,
            0.0365152824496168,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array([
            0.5850368186299210,
            0.0352702137092612,
            0.0271231173175534,
            0.0269848371163786,
            0.0269479074751672,
            0.0269839734580570,
            0.0269929632213637,
            0.0269813700097701,
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
            0.07063183236557720,
            0.00382581347617737,
            0.00284415542839772,
            0.00295207291043913,
            0.00281342243969949,
            0.00267109985044893,
            0.00281641795367762,
            0.00280665399191934,
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
            0.97218153631600,
            2.08130949882047,
            2.10807966891142,
            2.10849700564701,
            2.10860845686762,
            2.10849939082597,
            2.10847223060951,
            2.10850734474485,
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
            3.40245711886556E+06,
            2.07647074315400E+06,
            2.04625811734248E+06,
            2.04587890817645E+06,
            2.04577769627469E+06,
            2.04587728014699E+06,
            2.04590202020966E+06,
            2.04586982070706E+06,
        ])
        actual_vol_heat_cap_int_pts = np.array([
            ip.vol_heat_cap
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_vol_heat_cap_int_pts,
                                    expected_vol_heat_cap_int_pts,
                                    atol=1e-30))

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
        ])
        expected_void_ratio_vector = np.array([
            1.03011911113263,
            2.82703611599852,
            2.83079796876963,
            2.82977200892296,
            2.83011399553852,
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
        expected_void_ratio_int_pts_0 = np.array([
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
        ])
        expected_void_ratio_int_pts = np.array([
            1.40985235533021,
            2.44730287180095,
            2.82783108902905,
            2.83000299573910,
            2.83058115794312,
            2.82998881974947,
            2.82984427919847,
            2.83004172526302,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        actual_void_ratio_0_int_pts = np.array([
            ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_0_int_pts,
                                    expected_void_ratio_int_pts_0))
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.01475592022872E-09,
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
            7.66085854080689E-09,
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
            1.5057776376192E+04,
            1.5341057441242E+04,
            1.5672986178896E+04,
            1.6004992898353E+04,
            1.6336980105333E+04,
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
            1.51176407091358E+04,
            1.52811931082985E+04,
            1.54112022370508E+04,
            1.56028413830873E+04,
            1.57431474541990E+04,
            1.59348316230500E+04,
            1.60751500501842E+04,
            1.62668229535020E+04,
        ])
        actual_sig_int_pts = np.array([
            ip.tot_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sig_int_pts,
        ))

    def test_loc_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array([
            0.00000000000000E+00,
            2.14956087529361E+50,
            3.30037464730506E+04,
            1.55773351714842E+04,
            1.28113521802748E+04,
            1.59888166591600E+04,
            1.69740749026231E+04,
            1.60199789767812E+04,
        ])
        actual_sig_int_pts = np.array([
            ip.loc_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sig_int_pts,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            7.31964996413560E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
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
            -5.96856751743033E+52,
            -1.15968891191472E+07,
            -5.50087631467452E+06,
            -4.53017524055802E+06,
            -5.64634705448757E+06,
            -5.99233356552434E+06,
            -5.65789089590299E+06,
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
            -2.10676461166913E+02,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
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
            6.90906392008258E+03,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
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
            -3.16319698815031E-09,
            -5.55158941022770E-12,
            -4.50581503423868E-18,
            9.05406097817899E-19,
            -6.55055386134534E-18,
            -5.61682467288580E-19,
            -9.62682098458961E-20,
            1.97732654422452E-18,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-22, rtol=1e-13,
        ))

    def test_calculate_settlement(self):
        expected = 0.00588953885752812
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            0.00588953885752812,
            0.02500558072153320,
            0.04999851180759110,
            0.07500037204810220,
            0.10000000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
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
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [1.00494054452936E+02, -1.00494054452936E+02,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-1.00494054452936E+02, 1.84873314392263E+02,
             -8.43792599393273E+01, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -8.43792599393273E+01,
                1.68708867733541E+02, -8.43296077942136E+01,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -8.43296077942136E+01, 1.68671709760049E+02,
             -8.43421019658351E+01,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -8.43421019658351E+01,
             8.43421019658351E+01,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
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
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [1.94951686029744E+05, 5.84981131941464E+04, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [5.84981131941464E+04, 6.33050574710577E+04, 1.21905606485839E+04,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 1.21905606485839E+04, 4.86383692546801E+04,
                1.19924474362742E+04, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 1.19924474362742E+04,
                4.80007897384674E+04, 1.20803035002078E+04,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.20803035002078E+04, 2.41497966228996E+04,],
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
        expected_Phi = np.array([
            -8.28125812118903E-02,
            -2.22838397377318E-02,
            2.04468140211467E-13,
            9.04366543318248E-14,
            2.30818624914595E-14,
        ])
        self.assertTrue(np.allclose(
            expected_Phi, self.msh._heat_flux_vector,
            atol=1e-15, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [2.58819760425867E-09, -2.58819760425867E-09, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [-1.84806482036117E-10, 1.84806482036117E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
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
            expected_M, self.msh._mass_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [2.16333914131213E-03, 1.04137912730602E-03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.04137912730602E-03, 3.98934695481997E-03, 9.93576547128223E-04,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 9.93576547128223E-04, 3.97426115032726E-03,
                9.93564428240257E-04, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 9.93564428240257E-04,
                3.97426925819881E-03, 9.93567353572479E-04,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.93567353572479E-04, 1.98713332032972E-03,],
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
            -1.63005986745585E-06,
            -6.20996441165935E-06,
            -2.49068548726979E-09,
            5.93396976225382E-10,
            -1.49851154002932E-10,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-14, rtol=1e-8))

    def test_residual_heat_flux_vector(self):
        expected_Psi = np.array([
            -1.00299643256199E+01,
            6.77823603977139E-02,
            -1.82490970301538E-02,
            5.21402772290110E-03,
            -2.60701386145055E-03,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
            atol=1e-9, rtol=1e-8,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.000000000000000E+00,
            6.682385026989890E-02,
            -1.741240743522260E-02,
            4.800515822774490E-03,
            -2.334081038769090E-03,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
            atol=1e-10, rtol=1e-8,
        ))

    def test_residual_water_flux_vector(self):
        expected_Psi = np.array([
            1.79995502891963E+00,
            -2.96388646972966E-03,
            7.97969434157987E-04,
            -2.27991266902283E-04,
            1.13995633451142E-04,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-9, rtol=1e-8,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.000000000000000E+00,
            -2.963884001476730E-03,
            7.979687696283530E-04,
            -2.279910770366730E-04,
            1.139955385183370E-04,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            atol=1e-11, rtol=1e-8,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 6.20768442838417E-03
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a, delta=1e-10)
        self.assertEqual(self.msh._iter, 1)


class TestGlobalCorrectionLinearIterative(unittest.TestCase):
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
            void_ratio_tr=0.0,
            void_ratio_sep=1.6,
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
        self.msh.implicit_error_tolerance = 1e-4
        self.msh.initialize_global_system(0.0)
        self.msh.initialize_time_step()
        self.msh.iterative_correction_step()

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
            5.00000000000000,
            -4.94288663201352,
            -5.01504720195889,
            -4.99580782707204,
            -5.00205581585805,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_nodes,
                                    expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            2.8988208207507100,
            -2.8417074527642300,
            -4.9581359547447900,
            -4.9997978792276200,
            -5.0109814436504400,
            -4.9998735853804800,
            -4.9971281824613000,
            -5.0007354604687900,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array([
            2.66666666667E+00,
            1.52302314631E-02,
            -4.01258718904E-03,
            1.11791278079E-03,
            -5.48217562146E-04,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_rate_nodes,
                                    expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            2.10635221886686000,
            0.57554467926287200,
            0.01116374540138980,
            0.00005389887263449,
            -0.00292838497345164,
            0.00003371056520437,
            0.00076581801032079,
            -0.00019612279167771,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            -397.71546528054100000,
            -397.71546528054100000,
            -2.88642279781453000,
            -2.88642279781453000,
            0.76957499547378900,
            0.76957499547378900,
            -0.24991955144022900,
            -0.24991955144022900,
        ])
        actual_temp_gradient_int_pts = np.array([
            ip.temp_gradient for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_gradient_int_pts,
                                    expected_temp_gradient_int_pts))

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array([
            1.000000000000000,
            0.049609579364599,
            0.036685795369044,
            0.036519364123332,
            0.036475049693243,
            0.036519063629644,
            0.036529965177596,
            0.036515643157591,
        ])
        actual_deg_sat_water_int_pts = np.array([
            ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_deg_sat_water_int_pts,
                                    expected_deg_sat_water_int_pts))

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array([
            0.5850308349213140,
            0.0352181993298558,
            0.0271015296699256,
            0.0269842884643744,
            0.0269530617526990,
            0.0269840311477929,
            0.0269917066390633,
            0.0269816419348008,
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
            0.07064942454823320,
            0.00381528908800095,
            0.00281483415025699,
            0.00296426089204178,
            0.00278902974411304,
            0.00270397617646927,
            0.00279172102053797,
            0.00278382178171495,
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
            0.97218919425101,
            2.08145676998618,
            2.10814114328409,
            2.10849858926907,
            2.10859379907870,
            2.10849922044490,
            2.10847579887437,
            2.10850657542380,
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
            3.40244556073401E+06,
            2.07635418515680E+06,
            2.04621118802055E+06,
            2.04587764545300E+06,
            2.04578882338173E+06,
            2.04587742583443E+06,
            2.04589932419534E+06,
            2.04587039455450E+06,
        ])
        actual_vol_heat_cap_int_pts = np.array([
            ip.vol_heat_cap
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_vol_heat_cap_int_pts,
                                    expected_vol_heat_cap_int_pts,
                                    atol=1e-30))

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
        ])
        expected_void_ratio_vector = np.array([
            1.03011911113263,
            2.82687168093000,
            2.83084146850407,
            2.82975979844757,
            2.83012001569178,
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
        expected_void_ratio_int_pts_0 = np.array([
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
            2.83000000000000000,
        ])
        expected_void_ratio_int_pts = np.array([
            1.40981760611148000,
            2.44717318595114000,
            2.82771059575478000,
            2.83000255367929000,
            2.83061288472497000,
            2.82998838222667000,
            2.82983592130822000,
            2.83004389283113000,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        actual_void_ratio_0_int_pts = np.array([
            ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_0_int_pts,
                                    expected_void_ratio_int_pts_0))
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.01448974629494E-09,
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
            7.65884907152145E-09,
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
            1.5057784709296E+04,
            1.5341060353481E+04,
            1.5672985314020E+04,
            1.6004993005062E+04,
            1.6336980020337E+04,
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
            1.51176478966763E+04,
            1.52811971661010E+04,
            1.54112043510914E+04,
            1.56028413164091E+04,
            1.57431467946426E+04,
            1.59348315244388E+04,
            1.60751501163811E+04,
            1.62668229090176E+04,
        ])
        actual_sig_int_pts = np.array([
            ip.tot_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sig_int_pts,
        ))

    def test_loc_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array([
            0.00000000000000E+00,
            2.46576270619641E+50,
            3.44514138485866E+04,
            1.55797668096763E+04,
            1.26689143085986E+04,
            1.59912871212679E+04,
            1.70242745450539E+04,
            1.60077221101781E+04,
        ])
        actual_sig_int_pts = np.array([
            ip.loc_stress
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            expected_sig_int_pts,
            actual_sig_int_pts,
        ))

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array([
            7.32231313617113E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
            0.0000000000000E+00,
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
            -6.85306781736287E+52,
            -1.21144489489166E+07,
            -5.50183993415920E+06,
            -4.47901403195437E+06,
            -5.64720792012486E+06,
            -6.01031398591835E+06,
            -5.65350955257918E+06,
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
            -2.10753113419777E+02,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
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
            6.91009551890844E+03,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
            0.000000000000E+00,
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
            -3.16309057300454E-09,
            -5.38708074264096E-12,
            -3.59867286165033E-18,
            1.30824911520405E-18,
            -5.68270205266259E-18,
            -4.49156431416234E-19,
            -7.44518941610019E-20,
            1.72409628023870E-18,
        ])
        actual_water_flux_int_pts = np.array([
            ip.water_flux_rate
            for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_water_flux_int_pts,
            expected_water_flux_int_pts,
            atol=1e-22, rtol=1e-13,
        ))

    def test_calculate_settlement(self):
        expected = 0.00589038830748147
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            0.00589038830748147,
            0.02500589350350830,
            0.04999842989200070,
            0.07500039225150340,
            0.10000000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
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
            expected_H, self.msh._heat_flow_matrix_0,
        ))

    def test_global_heat_flow_matrix(self):
        expected_H = np.array([
            [1.00503359793848E+02, -1.00503359793848E+02,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-1.00503359793848E+02, 1.84886549576101E+02,
             -8.43831897822526E+01, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -8.43831897822526E+01,
                1.68711812380365E+02, -8.43286225981121E+01,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -8.43286225981121E+01, 1.68670916887743E+02,
             -8.43422942896312E+01,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -8.43422942896312E+01,
             8.43422942896312E+01,],
        ])
        self.assertTrue(np.allclose(
            expected_H, self.msh._heat_flow_matrix,
        ))

    def test_global_heat_storage_matrix_0(self):
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
            expected_C, self.msh._heat_storage_matrix_0,
        ))

    def test_global_heat_storage_matrix(self):
        expected_C = np.array([
            [1.94991265085943E+05, 5.85023156421955E+04, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [5.85023156421955E+04, 6.32147900601921E+04, 1.21796259191876E+04,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 1.21796259191876E+04, 4.86102192969331E+04,
                1.19978355891531E+04, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 1.19978355891531E+04,
                4.80120779152597E+04, 1.20502437776856E+04,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                1.20502437776856E+04, 2.40917313545577E+04,],
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
        expected_Phi = np.array([
            -8.28911380747473E-02,
            -2.23021882229809E-02,
            2.26140318388028E-13,
            6.68975183250999E-14,
            1.76511453602867E-14,
        ])
        self.assertTrue(np.allclose(
            expected_Phi, self.msh._heat_flux_vector,
            atol=1e-15, rtol=1e-8,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [2.58780640311249E-09, -2.58780640311249E-09, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [-1.85013066333885E-10, 1.85013066333885E-10, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix_0(self):
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
            expected_M, self.msh._mass_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_mass_matrix(self):
        expected_M = np.array([
            [2.16333819021251E-03, 1.04137557775393E-03, 0.00000000000000E+00,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [1.04137557775393E-03, 3.98932843366885E-03, 9.93575100279764E-04,
                0.00000000000000E+00, 0.00000000000000E+00,],
            [0.00000000000000E+00, 9.93575100279764E-04, 3.97426189292290E-03,
                9.93564768385044E-04, 0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00, 9.93564768385044E-04,
                3.97426906199635E-03, 9.93567289354967E-04,],
            [0.00000000000000E+00, 0.00000000000000E+00, 0.00000000000000E+00,
                9.93567289354967E-04, 1.98713336429415E-03,],
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
            -1.61970620775736E-06,
            -6.15164334951159E-06,
            -1.44160130656766E-09,
            3.61952149560319E-10,
            -9.11722623780379E-11,
        ])
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-14, rtol=1e-8))

    def test_residual_heat_flux_vector(self):
        expected_Psi = np.array([
            -1.00267459053042E+01,
            1.24641377626418E-04,
            -2.92317074200368E-05,
            7.55759674933489E-06,
            -3.48579176238581E-06,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_heat_flux_vector,
            atol=1e-9, rtol=1e-8,
        ))

    def test_temperature_increment_vector(self):
        expected_dT = np.array([
            0.000000000000000E+00,
            1.231622677044520E-04,
            -2.812931731557120E-05,
            7.052332792131510E-06,
            -3.165985376837170E-06,
        ])
        self.assertTrue(np.allclose(
            expected_dT, self.msh._delta_temp_vector,
            atol=1e-10, rtol=1e-8,
        ))

    def test_residual_water_flux_vector(self):
        expected_Psi = np.array([
            1.79998713456580E+00,
            3.13577668317627E-05,
            -7.86064538876201E-06,
            2.10289144438473E-06,
            -9.95219611804585E-07,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-9, rtol=1e-8,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.000000000000000E+00,
            3.135774071177510E-05,
            -7.860638356391370E-06,
            2.102889435140830E-06,
            -9.952186071820230E-07,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            atol=1e-11, rtol=1e-8,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.13406734711426E-05
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a, delta=1e-10)
        self.assertEqual(self.msh._iter, 3)


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


class TestInitializeGlobalSystemCubic(unittest.TestCase):
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
            void_ratio_tr=0.0,
            void_ratio_sep=1.6,
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
            order=3,
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
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
            -5.00000000000000,
        ])
        actual_temp_nodes = np.array([
            nd.temp for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_nodes,
                                    expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array([
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
            -5.0000000000000,
        ])
        actual_temp_int_pts = np.array([
            ip.temp for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_int_pts,
                                    expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array([
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
            0.00000000000E+00,
        ])
        actual_temp_rate_nodes = np.array([
            nd.temp_rate for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_temp_rate_nodes,
                                    expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array([
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
        ])
        actual_temp_rate_int_pts = np.array([
            ip.temp_rate for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_temp_rate_int_pts,
                                    expected_temp_rate_int_pts))

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array([
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
            0.00000000000,
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
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
            0.036518561878915,
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
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
            0.0269836893256734,
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
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
            0.00000000000000000,
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
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
            2.10850030207482,
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
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
            2.04587632179179E+06,
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
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
            2.83000000000000,
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
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
            2.8300000000000,
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
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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
            1.5110663486802E+04,
            1.5221326973605E+04,
            1.5331990460407E+04,
            1.5442653947209E+04,
            1.5553317434012E+04,
            1.5663980920814E+04,
            1.5774644407616E+04,
            1.5885307894418E+04,
            1.5995971381221E+04,
            1.6106634868023E+04,
            1.6217298354825E+04,
            1.6327961841628E+04,
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
            1.50155736980711E+04,
            1.50766118931150E+04,
            1.51659952302035E+04,
            1.52553785672920E+04,
            1.53164167623358E+04,
            1.53475641584781E+04,
            1.54086023535219E+04,
            1.54979856906104E+04,
            1.55873690276989E+04,
            1.56484072227427E+04,
            1.56795546188850E+04,
            1.57405928139288E+04,
            1.58299761510173E+04,
            1.59193594881058E+04,
            1.59803976831496E+04,
            1.60115450792919E+04,
            1.60725832743357E+04,
            1.61619666114242E+04,
            1.62513499485127E+04,
            1.63123881435566E+04,
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
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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
            -5.30271022784199E+06,
            -5.32426570535195E+06,
            -5.35583119497681E+06,
            -5.38739668460167E+06,
            -5.40895216211164E+06,
            -5.41995178286642E+06,
            -5.44150726037638E+06,
            -5.47307275000124E+06,
            -5.50463823962610E+06,
            -5.52619371713607E+06,
            -5.53719333789085E+06,
            -5.55874881540081E+06,
            -5.59031430502567E+06,
            -5.62187979465053E+06,
            -5.64343527216050E+06,
            -5.65443489291528E+06,
            -5.67599037042524E+06,
            -5.70755586005010E+06,
            -5.73912134967496E+06,
            -5.76067682718492E+06,
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
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
            0.00000000000000E+00,
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
            0.0,
            0.0,
            0.0,
            0.0,
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
            0.0,
            0.0,
            0.0,
            0.0,
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
            0.00833333333333333,
            0.01666666666666670,
            0.02500000000000000,
            0.03333333333333330,
            0.04166666666666670,
            0.05000000000000000,
            0.05833333333333330,
            0.06666666666666670,
            0.07500000000000000,
            0.08333333333333330,
            0.09166666666666670,
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
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array([
            [3.12058044707073E+02, -3.98506557092141E+02,
                1.13859016312040E+02, -2.74105039269726E+01,],
            [-3.98506557092141E+02, 9.10872130496322E+02, -
                6.26224589716221E+02, 1.13859016312040E+02,],
            [1.13859016312040E+02, -6.26224589716221E+02,
                9.10872130496322E+02, -3.98506557092141E+02,],
            [-2.74105039269726E+01, 1.13859016312040E+02, -
                3.98506557092141E+02, 6.24116089414146E+02,],
        ])
        expected_H[3:7, 3:7] = np.array([
            [6.24116089414146E+02, -3.98506557092141E+02,
                1.13859016312040E+02, -2.74105039269726E+01,],
            [-3.98506557092141E+02, 9.10872130496322E+02, -
                6.26224589716221E+02, 1.13859016312040E+02,],
            [1.13859016312040E+02, -6.26224589716221E+02,
                9.10872130496322E+02, -3.98506557092141E+02,],
            [-2.74105039269726E+01, 1.13859016312040E+02, -
                3.98506557092141E+02, 6.24116089414146E+02,],
        ])
        expected_H[6:10, 6:10] = np.array([
            [6.24116089414146E+02, -3.98506557092141E+02,
                1.13859016312040E+02, -2.74105039269726E+01,],
            [-3.98506557092141E+02, 9.10872130496322E+02, -
                6.26224589716221E+02, 1.13859016312040E+02,],
            [1.13859016312040E+02, -6.26224589716221E+02,
                9.10872130496322E+02, -3.98506557092141E+02,],
            [-2.74105039269726E+01, 1.13859016312040E+02, -
                3.98506557092141E+02, 6.24116089414146E+02,],
        ])
        expected_H[9:13, 9:13] = np.array([
            [6.24116089414146E+02, -3.98506557092141E+02,
                1.13859016312040E+02, -2.74105039269726E+01,],
            [-3.98506557092141E+02, 9.10872130496322E+02, -
                6.26224589716221E+02, 1.13859016312040E+02,],
            [1.13859016312040E+02, -6.26224589716221E+02,
                9.10872130496322E+02, -3.98506557092141E+02,],
            [-2.74105039269726E+01, 1.13859016312040E+02, -
                3.98506557092141E+02, 3.12058044707073E+02,],
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
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array([
            [3.89690727960341E+03, 3.01401422406826E+03, -
                1.09600517238846E+03, 5.78447174316131E+02,],
            [3.01401422406826E+03, 1.97280931029923E+04, -
                2.46601163787403E+03, -1.09600517238846E+03,],
            [-1.09600517238846E+03, -2.46601163787403E+03,
                1.97280931029923E+04, 3.01401422406826E+03,],
            [5.78447174316131E+02, -1.09600517238846E+03,
                3.01401422406826E+03, 7.79381455920682E+03,],
        ])
        expected_C[3:7, 3:7] = np.array([
            [7.79381455920682E+03, 3.01401422406826E+03, -
                1.09600517238846E+03, 5.78447174316131E+02,],
            [3.01401422406826E+03, 1.97280931029923E+04, -
                2.46601163787403E+03, -1.09600517238846E+03,],
            [-1.09600517238846E+03, -2.46601163787403E+03,
                1.97280931029923E+04, 3.01401422406826E+03,],
            [5.78447174316131E+02, -1.09600517238846E+03,
                3.01401422406826E+03, 7.79381455920682E+03,],
        ])
        expected_C[6:10, 6:10] = np.array([
            [7.79381455920682E+03, 3.01401422406826E+03, -
                1.09600517238846E+03, 5.78447174316131E+02,],
            [3.01401422406826E+03, 1.97280931029923E+04, -
                2.46601163787403E+03, -1.09600517238846E+03,],
            [-1.09600517238846E+03, -2.46601163787403E+03,
                1.97280931029923E+04, 3.01401422406826E+03,],
            [5.78447174316131E+02, -1.09600517238846E+03,
                3.01401422406826E+03, 7.79381455920682E+03,],
        ])
        expected_C[9:13, 9:13] = np.array([
            [7.79381455920682E+03, 3.01401422406826E+03, -
                1.09600517238846E+03, 5.78447174316131E+02,],
            [3.01401422406826E+03, 1.97280931029923E+04, -
                2.46601163787403E+03, -1.09600517238846E+03,],
            [-1.09600517238846E+03, -2.46601163787403E+03,
                1.97280931029923E+04, 3.01401422406826E+03,],
            [5.78447174316131E+02, -1.09600517238846E+03,
                3.01401422406826E+03, 3.89690727960341E+03,],
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
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array([
            [4.54201999537039E-04, 3.51296859016929E-04, -
                1.27744312369792E-04, 6.74206093062793E-05,],
            [3.51296859016929E-04, 2.29939762265626E-03, -
                2.87424702832033E-04, -1.27744312369793E-04,],
            [-1.27744312369792E-04, -2.87424702832033E-04,
                2.29939762265626E-03, 3.51296859016929E-04,],
            [6.74206093062793E-05, -1.27744312369793E-04,
                3.51296859016929E-04, 9.08403999074079E-04,],
        ])
        expected_M[3:7, 3:7] = np.array([
            [9.08403999074079E-04, 3.51296859016929E-04, -
                1.27744312369792E-04, 6.74206093062793E-05,],
            [3.51296859016929E-04, 2.29939762265626E-03, -
                2.87424702832033E-04, -1.27744312369793E-04,],
            [-1.27744312369792E-04, -2.87424702832033E-04,
                2.29939762265626E-03, 3.51296859016929E-04,],
            [6.74206093062793E-05, -1.27744312369793E-04,
                3.51296859016929E-04, 9.08403999074079E-04,],
        ])
        expected_M[6:10, 6:10] = np.array([
            [9.08403999074079E-04, 3.51296859016929E-04, -
                1.27744312369792E-04, 6.74206093062793E-05,],
            [3.51296859016929E-04, 2.29939762265626E-03, -
                2.87424702832033E-04, -1.27744312369793E-04,],
            [-1.27744312369792E-04, -2.87424702832033E-04,
                2.29939762265626E-03, 3.51296859016929E-04,],
            [6.74206093062793E-05, -1.27744312369793E-04,
                3.51296859016929E-04, 9.08403999074079E-04,],
        ])
        expected_M[9:13, 9:13] = np.array([
            [9.08403999074079E-04, 3.51296859016929E-04, -
                1.27744312369792E-04, 6.74206093062793E-05,],
            [3.51296859016929E-04, 2.29939762265626E-03, -
                2.87424702832033E-04, -1.27744312369793E-04,],
            [-1.27744312369792E-04, -2.87424702832033E-04,
                2.29939762265626E-03, 3.51296859016929E-04,],
            [6.74206093062793E-05, -1.27744312369793E-04,
                3.51296859016929E-04, 4.54201999537039E-04,],
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
