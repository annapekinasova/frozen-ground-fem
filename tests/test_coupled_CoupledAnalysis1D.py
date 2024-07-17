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
            msh.add_boundary(
                ThermalBoundary1D(
                    (msh.nodes[0],),
                    (ip,),
                )
            )

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
        self.assertAlmostEqual(self.msh.over_dt, 1.0 / 1.5)

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
        self.assertAlmostEqual(self.msh.nodes[1].z - self.msh.nodes[0].z, 1.0 / 3.0)
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
        t0 = (7.0 / 12.0) * per
        Tavg = 5.0
        Tamp = 20.0

        def f_T(t):
            return Tavg + Tamp * np.cos(om * (t - t0))

        self.f_T = f_T
        self.bnd0 = ThermalBoundary1D(
            (self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_function=f_T,
        )
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

        def f_e(t):
            return eavg + eamp * np.cos(om * (t - t0))

        self.f_e = f_e
        self.bnd0 = ConsolidationBoundary1D(
            (self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_function=f_e,
        )
        self.msh.add_boundary(self.bnd0)
        self.water_flux = 0.08
        self.bnd1 = ConsolidationBoundary1D(
            (self.msh.nodes[-1],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.water_flux,
            bnd_value=self.water_flux,
        )
        self.msh.add_boundary(self.bnd1)

    def test_initial_temp_heat_flux_vector(self):
        for tn, tn0 in zip(self.msh._temp_vector, self.msh._temp_vector_0):
            self.assertEqual(tn, 0.0)
            self.assertEqual(tn0, 0.0)
        for fx, fx0 in zip(self.msh._heat_flux_vector, self.msh._heat_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_initial_thrm_cond(self):
        expected_thrm_cond = 1.7755371996983
        for e in self.msh.elements:
            for ip in e.int_pts:
                self.assertAlmostEqual(ip.thrm_cond, expected_thrm_cond)

    def test_initial_void_ratio_water_flux_vector(self):
        for en, en0 in zip(self.msh._void_ratio_vector, self.msh._void_ratio_vector_0):
            self.assertEqual(en, 0.0)
            self.assertEqual(en0, 0.0)
        for fx, fx0 in zip(self.msh._water_flux_vector, self.msh._water_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_initial_porosity(self):
        expected_porosity = 0.6 / 1.6
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
        for fx, fx0 in zip(self.msh._heat_flux_vector, self.msh._heat_flux_vector_0):
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
        for fx, fx0 in zip(self.msh._heat_flux_vector, self.msh._heat_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_update_heat_flux_vector(self):
        t = 1.314e7
        self.msh.update_boundary_conditions(t)
        self.msh.update_heat_flux_vector()
        for k, (fx, fx0) in enumerate(
            zip(self.msh._heat_flux_vector, self.msh._heat_flux_vector_0)
        ):
            self.assertEqual(fx0, 0.0)
            if k == 5:
                self.assertAlmostEqual(fx, -self.fixed_flux)
            elif k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, -self.flux_geotherm)
            else:
                self.assertEqual(fx, 0.0)
        self.msh.update_boundary_conditions(t)
        self.msh.update_heat_flux_vector()
        for k, (fx, fx0) in enumerate(
            zip(self.msh._heat_flux_vector, self.msh._heat_flux_vector_0)
        ):
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
        self.assertAlmostEqual(self.msh.nodes[0].void_ratio, expected_void_ratio_0)
        self.assertAlmostEqual(self.msh.nodes[0].void_ratio, expected_void_ratio_1)
        self.assertAlmostEqual(self.msh._void_ratio_vector[0], expected_void_ratio_0)
        self.assertAlmostEqual(self.msh._void_ratio_vector[0], expected_void_ratio_1)
        for en in self.msh._void_ratio_vector[1:]:
            self.assertEqual(en, 0.0)
        for en0 in self.msh._void_ratio_vector_0:
            self.assertEqual(en0, 0.0)
        for fx, fx0 in zip(self.msh._water_flux_vector, self.msh._water_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)
        t = 18921600.0
        expected_void_ratio_2 = self.f_e(t)
        expected_void_ratio_3 = 0.599452189536827
        self.msh.update_boundary_conditions(t)
        self.assertAlmostEqual(self.msh.nodes[0].void_ratio, expected_void_ratio_2)
        self.assertAlmostEqual(self.msh.nodes[0].void_ratio, expected_void_ratio_3)
        self.assertAlmostEqual(self.msh._void_ratio_vector[0], expected_void_ratio_2)
        self.assertAlmostEqual(self.msh._void_ratio_vector[0], expected_void_ratio_3)
        for en in self.msh._void_ratio_vector[1:]:
            self.assertEqual(en, 0.0)
        for en0 in self.msh._void_ratio_vector_0:
            self.assertEqual(en0, 0.0)
        for fx, fx0 in zip(self.msh._water_flux_vector, self.msh._water_flux_vector_0):
            self.assertEqual(fx, 0.0)
            self.assertEqual(fx0, 0.0)

    def test_update_water_flux_vector(self):
        t = 6307200.0
        self.msh.update_boundary_conditions(t)
        self.msh.update_water_flux_vector()
        for k, (fx, fx0) in enumerate(
            zip(self.msh._water_flux_vector, self.msh._water_flux_vector_0)
        ):
            self.assertEqual(fx0, 0.0)
            if k == self.msh.num_nodes - 1:
                self.assertAlmostEqual(fx, -self.water_flux)
            else:
                self.assertEqual(fx, 0.0)
        self.msh.update_boundary_conditions(t)
        self.msh.update_water_flux_vector()
        for k, (fx, fx0) in enumerate(
            zip(self.msh._water_flux_vector, self.msh._water_flux_vector_0)
        ):
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
        self.msh._void_ratio_vector[:] = np.linspace(2.0, 22.0, self.msh.num_nodes)
        self.msh._void_ratio_vector_0[:] = np.linspace(1.0, 11.0, self.msh.num_nodes)
        self.msh.time_step = 0.25
        self.msh._temp_rate_vector[:] = (
            self.msh._temp_vector[:] - self.msh._temp_vector_0[:]
        ) / self.msh.dt
        self.msh.update_nodes()

    def test_initial_node_values(self):
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k + 1))
            self.assertAlmostEqual(nd.temp, 2.0 * (k + 1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k + 1))

    def test_repeat_update_nodes(self):
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k + 1))
            self.assertAlmostEqual(nd.temp, 2.0 * (k + 1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k + 1))

    def test_change_temp_vectors_update_nodes(self):
        self.msh._temp_vector[:] = np.linspace(4.0, 44.0, self.msh.num_nodes)
        self.msh._temp_vector_0[:] = np.linspace(2.0, 22.0, self.msh.num_nodes)
        self.msh._temp_rate_vector[:] = (
            self.msh._temp_vector[:] - self.msh._temp_vector_0[:]
        ) / self.msh.dt
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.temp, 2.0 * (k + 1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k + 1))
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.temp, 4.0 * (k + 1))
            self.assertAlmostEqual(nd.temp_rate, 8.0 * (k + 1))

    def test_change_void_ratio_vectors_update_nodes(self):
        self.msh._void_ratio_vector[:] = np.linspace(4.0, 44.0, self.msh.num_nodes)
        self.msh._void_ratio_vector_0[:] = np.linspace(2.0, 22.0, self.msh.num_nodes)
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 2.0 * (k + 1))
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
            self.assertAlmostEqual(nd.void_ratio, 4.0 * (k + 1))


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
        c0 = 8.68800000000000e06
        d0 = np.ones((self.msh.num_nodes,)) * 2.0 * c0
        d0[0] = c0
        d0[-1] = c0
        d1 = np.ones((self.msh.num_nodes - 1,)) * c0 * 0.5
        expected1 = np.diag(d0) + np.diag(d1, -1) + np.diag(d1, 1)
        self.msh.update_heat_storage_matrix()
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected0))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected1))

    def test_update_heat_flux_vector(self):
        expected0 = np.zeros(self.msh.num_nodes)
        expected1 = np.ones(self.msh.num_nodes) * 0.0022465125000000000
        expected1[0] = 0.0011232562500000000
        expected1[-1] = 0.0011232562500000000
        self.msh.update_heat_flux_vector()
        self.assertTrue(np.allclose(self.msh._heat_flux_vector_0, expected0))
        self.assertTrue(
            np.allclose(
                self.msh._heat_flux_vector,
                expected1,
                rtol=1e-13,
                atol=1e-16,
            )
        )

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
        k00 = 1.55999984566148e-09
        k11 = 7.82923225956888e-10
        d0 = np.ones((self.msh.num_nodes,)) * (k00 + k11)
        d0[0] = k00
        d0[-1] = k11
        dp1 = -np.ones((self.msh.num_nodes - 1,)) * k00
        dm1 = -np.ones((self.msh.num_nodes - 1,)) * k11
        expected1 = np.diag(d0) + np.diag(dm1, -1) + np.diag(dp1, 1)
        self.msh.update_stiffness_matrix()
        self.assertTrue(
            np.allclose(
                self.msh._stiffness_matrix_0,
                expected0,
                atol=1e-20,
                rtol=1e-16,
            )
        )
        self.assertTrue(
            np.allclose(
                self.msh._stiffness_matrix,
                expected1,
                atol=1e-20,
                rtol=1e-16,
            )
        )

    def test_update_mass_matrix(self):
        expected0 = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        m0 = 1.72280701754386
        d0 = np.ones((self.msh.num_nodes,)) * 2.0 * m0
        d0[0] = m0
        d0[-1] = m0
        d1 = np.ones((self.msh.num_nodes - 1,)) * m0 * 0.5
        expected1 = np.diag(d0) + np.diag(d1, -1) + np.diag(d1, 1)
        self.msh.update_mass_matrix()
        self.assertTrue(
            np.allclose(
                self.msh._mass_matrix_0,
                expected0,
                atol=1e-12,
                rtol=1e-10,
            )
        )
        self.assertTrue(
            np.allclose(
                self.msh._mass_matrix,
                expected1,
                atol=1e-12,
                rtol=1e-10,
            )
        )

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
            z_range=(0, 0.1), num_elements=4, generate=True, order=1
        )
        temp_bound = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=5.0,
        )
        self.msh.add_boundary(temp_bound)
        hyd_bound = HydraulicBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array(
            [
                2.830000000000000,
                2.830000000000000,
                2.830000000000000,
                2.830000000000000,
                2.830000000000000,
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array(
            [
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_int_pts)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5000000000000e04,
                1.5331990460407e04,
                1.5663980920814e04,
                1.5995971381221e04,
                1.6327961841628e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.50701578393613e04,
                1.52618326210456e04,
                1.54021482997682e04,
                1.55938230814525e04,
                1.57341387601751e04,
                1.59258135418595e04,
                1.60661292205821e04,
                1.62578040022664e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -5.32198647330003e06,
                -5.38967591665359e06,
                -5.43922802832446e06,
                -5.50691747167802e06,
                -5.55646958334889e06,
                -5.62415902670245e06,
                -5.67371113837331e06,
                -5.74140058172688e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.0
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
                0.00000000000000000,
                0.02500000000000000,
                0.05000000000000000,
                0.07500000000000000,
                0.10000000000000000,
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.array(
            [
                [
                    8.43400120829928e01,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    8.43400120829928e01,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.array(
            [
                [
                    1.70489693482649e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    1.70489693482649e04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.array(
            [
                [
                    1.98713374797455e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    1.98713374797455e-03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-18, rtol=1e-8
            )
        )


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
            z_range=(0, 0.1), num_elements=4, generate=True, order=1
        )
        temp_bound = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=5.0,
        )
        self.msh.add_boundary(temp_bound)
        hyd_bound = HydraulicBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
                5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                2.8867513459481300,
                -2.8867513459481300,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
                -5.0,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                2.66666666667,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                2.10313369225284000,
                0.56353297441383300,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -400.00000000000000000,
                -400.00000000000000000,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
                1.000000000000000,
                0.049189123034829,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
                0.036518561878915,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
                0.5851446432832020,
                0.0349299200049867,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
                0.0269836893256734,
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.07077197308170330,
                0.00376019673031735,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
                0.00000000000000000,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                0.97204355293632,
                2.08230418654856,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
                2.10850030207482,
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                3.40266539296583e06,
                2.07560330535707e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array(
            [
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
            ]
        )
        expected_void_ratio_vector = np.array(
            [
                1.03011911113263,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector_0, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts_0 = np.array(
            [
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
            ]
        )
        expected_void_ratio_int_pts = np.array(
            [
                1.41047869771790000,
                2.44964041341473000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_int_pts_0)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                1.01956560310148e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                7.69716904600309e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5057626734725e04,
                1.5340992813644e04,
                1.5672983274051e04,
                1.6004973734458e04,
                1.6336964194865e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.51175090332130e04,
                1.52811105151559e04,
                1.54111506530051e04,
                1.56028254346894e04,
                1.57431411134120e04,
                1.59348158950964e04,
                1.60751315738190e04,
                1.62668063555033e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_loc_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                0.00000000000000e00,
                2.24341159206153e50,
                1.54021482997682e04,
                1.55938230814525e04,
                1.57341387601751e04,
                1.59258135418595e04,
                1.60661292205821e04,
                1.62578040022664e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.loc_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                7.27181296175159e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                -6.26996013389023e52,
                -5.43922802832446e06,
                -5.50691747167802e06,
                -5.55646958334889e06,
                -5.62415902670245e06,
                -5.67371113837331e06,
                -5.74140058172688e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -2.09299601559626e02,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
                6.89049609873269e03,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                -3.16511899023414e-09,
                -4.50905622831981e-12,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.00587428488533737
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
                0.00587428488533737,
                0.02500000000000000,
                0.05000000000000000,
                0.07500000000000000,
                0.10000000000000000,
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array(
            [
                [
                    8.43400120829928e01,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    8.43400120829928e01,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.array(
            [
                [
                    1.00416479969625e02,
                    -1.00416479969625e02,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -1.00416479969625e02,
                    1.84756492052618e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    8.43400120829928e01,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array(
            [
                [
                    1.70489693482649e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    1.70489693482649e04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.array(
            [
                [
                    1.95272432649798e05,
                    5.85438655571886e04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    5.85438655571886e04,
                    5.59519989272211e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    1.70489693482649e04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array(
            [
                -8.33935003943986e-02,
                -2.24222554544176e-02,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.array(
            [
                [
                    2.59527068320373e-09,
                    -2.59527068320373e-09,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -1.81079078550582e-10,
                    1.81079078550582e-10,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.array(
            [
                [
                    1.98713374797455e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    1.98713374797455e-03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.array(
            [
                [
                    2.16333267482736e-03,
                    1.04135499405632e-03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.04135499405632e-03,
                    3.98922104937246e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    1.98713374797455e-03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.array(
            [
                -1.56524098050761e-06,
                -5.84157071020524e-06,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-18, rtol=1e-8
            )
        )


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
            z_range=(0, 0.1), num_elements=4, generate=True, order=1
        )
        temp_bound = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=5.0,
        )
        self.msh.add_boundary(temp_bound)
        hyd_bound = HydraulicBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
                5.00000000000000,
                -4.93317614973010,
                -5.01741240743522,
                -4.99519948417723,
                -5.00233408103877,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                2.90087288711227,
                -2.83404903684237,
                -4.95097736555187,
                -4.99961119161345,
                -5.01271826441747,
                -4.99989362719498,
                -4.99670720189871,
                -5.00082636331728,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                2.66666666666667e00,
                1.78196934053065e-02,
                -4.64330864939271e-03,
                1.28013755273978e-03,
                -6.22421610338364e-04,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                2.10689943656328e00,
                5.77586923508701e-01,
                1.30727025195007e-02,
                1.03682236413034e-04,
                -3.39153717799219e-03,
                2.83660813392684e-05,
                8.78079493676889e-04,
                -2.20363551275469e-04,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -397.327045989204000,
                -397.327045989204000,
                -3.369450308204880,
                -3.369450308204880,
                0.888516930319867,
                0.888516930319867,
                -0.285383874461701,
                -0.285383874461701,
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
                1.0000000000000000,
                0.0496820848820785,
                0.0367146086279011,
                0.0365201051556731,
                0.0364681812180862,
                0.0365189840806239,
                0.0365316376352820,
                0.0365152824496168,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
                0.5850368186299210,
                0.0352702137092612,
                0.0271231173175534,
                0.0269848371163786,
                0.0269479074751672,
                0.0269839734580570,
                0.0269929632213637,
                0.0269813700097701,
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.07063183236557720,
                0.00382581347617737,
                0.00284415542839772,
                0.00295207291043913,
                0.00281342243969949,
                0.00267109985044893,
                0.00281641795367762,
                0.00280665399191934,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                0.97218153631600,
                2.08130949882047,
                2.10807966891142,
                2.10849700564701,
                2.10860845686762,
                2.10849939082597,
                2.10847223060951,
                2.10850734474485,
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                3.40245711886556e06,
                2.07647074315400e06,
                2.04625811734248e06,
                2.04587890817645e06,
                2.04577769627469e06,
                2.04587728014699e06,
                2.04590202020966e06,
                2.04586982070706e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array(
            [
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
            ]
        )
        expected_void_ratio_vector = np.array(
            [
                1.03011911113263,
                2.82703611599852,
                2.83079796876963,
                2.82977200892296,
                2.83011399553852,
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector_0, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts_0 = np.array(
            [
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
            ]
        )
        expected_void_ratio_int_pts = np.array(
            [
                1.40985235533021,
                2.44730287180095,
                2.82783108902905,
                2.83000299573910,
                2.83058115794312,
                2.82998881974947,
                2.82984427919847,
                2.83004172526302,
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_int_pts_0)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                1.01475592022872e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                7.66085854080689e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5057776376192e04,
                1.5341057441242e04,
                1.5672986178896e04,
                1.6004992898353e04,
                1.6336980105333e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.51176407091358e04,
                1.52811931082985e04,
                1.54112022370508e04,
                1.56028413830873e04,
                1.57431474541990e04,
                1.59348316230500e04,
                1.60751500501842e04,
                1.62668229535020e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_loc_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                0.00000000000000e00,
                2.14956087529361e50,
                3.30037464730506e04,
                1.55773351714842e04,
                1.28113521802748e04,
                1.59888166591600e04,
                1.69740749026231e04,
                1.60199789767812e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.loc_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                7.31964996413560e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                -5.96856751743033e52,
                -1.15968891191472e07,
                -5.50087631467452e06,
                -4.53017524055802e06,
                -5.64634705448757e06,
                -5.99233356552434e06,
                -5.65789089590299e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -2.10676461166913e02,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
                6.90906392008258e03,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                -3.16319698815031e-09,
                -5.55158941022770e-12,
                -4.50581503423868e-18,
                9.05406097817899e-19,
                -6.55055386134534e-18,
                -5.61682467288580e-19,
                -9.62682098458961e-20,
                1.97732654422452e-18,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-22,
                rtol=1e-13,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.00588953885752812
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
                0.00588953885752812,
                0.02500558072153320,
                0.04999851180759110,
                0.07500037204810220,
                0.10000000000000000,
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array(
            [
                [
                    8.43400120829928e01,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    8.43400120829928e01,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.array(
            [
                [
                    1.00494054452936e02,
                    -1.00494054452936e02,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -1.00494054452936e02,
                    1.84873314392263e02,
                    -8.43792599393273e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    -8.43792599393273e01,
                    1.68708867733541e02,
                    -8.43296077942136e01,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43296077942136e01,
                    1.68671709760049e02,
                    -8.43421019658351e01,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43421019658351e01,
                    8.43421019658351e01,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array(
            [
                [
                    1.70489693482649e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    1.70489693482649e04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.array(
            [
                [
                    1.94951686029744e05,
                    5.84981131941464e04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    5.84981131941464e04,
                    6.33050574710577e04,
                    1.21905606485839e04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    1.21905606485839e04,
                    4.86383692546801e04,
                    1.19924474362742e04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    1.19924474362742e04,
                    4.80007897384674e04,
                    1.20803035002078e04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    1.20803035002078e04,
                    2.41497966228996e04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array(
            [
                -8.28125812118903e-02,
                -2.22838397377318e-02,
                2.04468140211467e-13,
                9.04366543318248e-14,
                2.30818624914595e-14,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.array(
            [
                [
                    2.58819760425867e-09,
                    -2.58819760425867e-09,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -1.84806482036117e-10,
                    1.84806482036117e-10,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.array(
            [
                [
                    1.98713374797455e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    1.98713374797455e-03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.array(
            [
                [
                    2.16333914131213e-03,
                    1.04137912730602e-03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.04137912730602e-03,
                    3.98934695481997e-03,
                    9.93576547128223e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    9.93576547128223e-04,
                    3.97426115032726e-03,
                    9.93564428240257e-04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93564428240257e-04,
                    3.97426925819881e-03,
                    9.93567353572479e-04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93567353572479e-04,
                    1.98713332032972e-03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.array(
            [
                -1.63005986745585e-06,
                -6.20996441165935e-06,
                -2.49068548726979e-09,
                5.93396976225382e-10,
                -1.49851154002932e-10,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-14, rtol=1e-8
            )
        )

    def test_residual_heat_flux_vector(self):
        expected_Psi = np.array(
            [
                -1.00299643256199e01,
                6.77823603977139e-02,
                -1.82490970301538e-02,
                5.21402772290110e-03,
                -2.60701386145055e-03,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.000000000000000e00,
                6.682385026989890e-02,
                -1.741240743522260e-02,
                4.800515822774490e-03,
                -2.334081038769090e-03,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
                atol=1e-10,
                rtol=1e-8,
            )
        )

    def test_residual_water_flux_vector(self):
        expected_Psi = np.array(
            [
                1.79995502891963e00,
                -2.96388646972966e-03,
                7.97969434157987e-04,
                -2.27991266902283e-04,
                1.13995633451142e-04,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_water_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_void_ratio_increment_vector(self):
        expected_de = np.array(
            [
                0.000000000000000e00,
                -2.963884001476730e-03,
                7.979687696283530e-04,
                -2.279910770366730e-04,
                1.139955385183370e-04,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_de,
                self.msh._delta_void_ratio_vector,
                atol=1e-11,
                rtol=1e-8,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 6.20768442838417e-03
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
            z_range=(0, 0.1), num_elements=4, generate=True, order=1
        )
        temp_bound = ThermalBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_value=5.0,
        )
        self.msh.add_boundary(temp_bound)
        hyd_bound = HydraulicBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
                5.00000000000000,
                -4.94288663201352,
                -5.01504720195889,
                -4.99580782707204,
                -5.00205581585805,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                2.8988208207507100,
                -2.8417074527642300,
                -4.9581359547447900,
                -4.9997978792276200,
                -5.0109814436504400,
                -4.9998735853804800,
                -4.9971281824613000,
                -5.0007354604687900,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                2.66666666667e00,
                1.52302314631e-02,
                -4.01258718904e-03,
                1.11791278079e-03,
                -5.48217562146e-04,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                2.10635221886686000,
                0.57554467926287200,
                0.01116374540138980,
                0.00005389887263449,
                -0.00292838497345164,
                0.00003371056520437,
                0.00076581801032079,
                -0.00019612279167771,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -397.71546528054100000,
                -397.71546528054100000,
                -2.88642279781453000,
                -2.88642279781453000,
                0.76957499547378900,
                0.76957499547378900,
                -0.24991955144022900,
                -0.24991955144022900,
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
                1.000000000000000,
                0.049609579364599,
                0.036685795369044,
                0.036519364123332,
                0.036475049693243,
                0.036519063629644,
                0.036529965177596,
                0.036515643157591,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
                0.5850308349213140,
                0.0352181993298558,
                0.0271015296699256,
                0.0269842884643744,
                0.0269530617526990,
                0.0269840311477929,
                0.0269917066390633,
                0.0269816419348008,
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.07064942454823320,
                0.00381528908800095,
                0.00281483415025699,
                0.00296426089204178,
                0.00278902974411304,
                0.00270397617646927,
                0.00279172102053797,
                0.00278382178171495,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                0.97218919425101,
                2.08145676998618,
                2.10814114328409,
                2.10849858926907,
                2.10859379907870,
                2.10849922044490,
                2.10847579887437,
                2.10850657542380,
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                3.40244556073401e06,
                2.07635418515680e06,
                2.04621118802055e06,
                2.04587764545300e06,
                2.04578882338173e06,
                2.04587742583443e06,
                2.04589932419534e06,
                2.04587039455450e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array(
            [
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
                2.83000000000000,
            ]
        )
        expected_void_ratio_vector = np.array(
            [
                1.03011911113263,
                2.82687168093000,
                2.83084146850407,
                2.82975979844757,
                2.83012001569178,
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector_0, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts_0 = np.array(
            [
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
                2.83000000000000000,
            ]
        )
        expected_void_ratio_int_pts = np.array(
            [
                1.40981760611148000,
                2.44717318595114000,
                2.82771059575478000,
                2.83000255367929000,
                2.83061288472497000,
                2.82998838222667000,
                2.82983592130822000,
                2.83004389283113000,
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_int_pts_0)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                1.01448974629494e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                7.65884907152145e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5057784709296e04,
                1.5341060353481e04,
                1.5672985314020e04,
                1.6004993005062e04,
                1.6336980020337e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.51176478966763e04,
                1.52811971661010e04,
                1.54112043510914e04,
                1.56028413164091e04,
                1.57431467946426e04,
                1.59348315244388e04,
                1.60751501163811e04,
                1.62668229090176e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_loc_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                0.00000000000000e00,
                2.46576270619641e50,
                3.44514138485866e04,
                1.55797668096763e04,
                1.26689143085986e04,
                1.59912871212679e04,
                1.70242745450539e04,
                1.60077221101781e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.loc_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                7.32231313617113e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
                0.0000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                -6.85306781736287e52,
                -1.21144489489166e07,
                -5.50183993415920e06,
                -4.47901403195437e06,
                -5.64720792012486e06,
                -6.01031398591835e06,
                -5.65350955257918e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -2.10753113419777e02,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
                6.91009551890844e03,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
                0.000000000000e00,
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                -3.16309057300454e-09,
                -5.38708074264096e-12,
                -3.59867286165033e-18,
                1.30824911520405e-18,
                -5.68270205266259e-18,
                -4.49156431416234e-19,
                -7.44518941610019e-20,
                1.72409628023870e-18,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-22,
                rtol=1e-13,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.00589038830748147
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
                0.00589038830748147,
                0.02500589350350830,
                0.04999842989200070,
                0.07500039225150340,
                0.10000000000000000,
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array(
            [
                [
                    8.43400120829928e01,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    1.68680024165986e02,
                    -8.43400120829928e01,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43400120829928e01,
                    8.43400120829928e01,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.array(
            [
                [
                    1.00503359793848e02,
                    -1.00503359793848e02,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -1.00503359793848e02,
                    1.84886549576101e02,
                    -8.43831897822526e01,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    -8.43831897822526e01,
                    1.68711812380365e02,
                    -8.43286225981121e01,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43286225981121e01,
                    1.68670916887743e02,
                    -8.43422942896312e01,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    -8.43422942896312e01,
                    8.43422942896312e01,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.array(
            [
                [
                    1.70489693482649e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    3.40979386965298e04,
                    8.52448467413248e03,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.52448467413248e03,
                    1.70489693482649e04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.array(
            [
                [
                    1.94991265085943e05,
                    5.85023156421955e04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    5.85023156421955e04,
                    6.32147900601921e04,
                    1.21796259191876e04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    1.21796259191876e04,
                    4.86102192969331e04,
                    1.19978355891531e04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    1.19978355891531e04,
                    4.80120779152597e04,
                    1.20502437776856e04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    1.20502437776856e04,
                    2.40917313545577e04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array(
            [
                -8.28911380747473e-02,
                -2.23021882229809e-02,
                2.26140318388028e-13,
                6.68975183250999e-14,
                1.76511453602867e-14,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.array(
            [
                [
                    2.58780640311249e-09,
                    -2.58780640311249e-09,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    -1.85013066333885e-10,
                    1.85013066333885e-10,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.array(
            [
                [
                    1.98713374797455e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    3.97426749594910e-03,
                    9.93566873987276e-04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93566873987276e-04,
                    1.98713374797455e-03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.array(
            [
                [
                    2.16333819021251e-03,
                    1.04137557775393e-03,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.04137557775393e-03,
                    3.98932843366885e-03,
                    9.93575100279764e-04,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    9.93575100279764e-04,
                    3.97426189292290e-03,
                    9.93564768385044e-04,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93564768385044e-04,
                    3.97426906199635e-03,
                    9.93567289354967e-04,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.93567289354967e-04,
                    1.98713336429415e-03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.array(
            [
                -1.61970620775736e-06,
                -6.15164334951159e-06,
                -1.44160130656766e-09,
                3.61952149560319e-10,
                -9.11722623780379e-11,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-14, rtol=1e-8
            )
        )

    def test_residual_heat_flux_vector(self):
        expected_Psi = np.array(
            [
                -1.00267459053042e01,
                1.24641377626418e-04,
                -2.92317074200368e-05,
                7.55759674933489e-06,
                -3.48579176238581e-06,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.000000000000000e00,
                1.231622677044520e-04,
                -2.812931731557120e-05,
                7.052332792131510e-06,
                -3.165985376837170e-06,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
                atol=1e-10,
                rtol=1e-8,
            )
        )

    def test_residual_water_flux_vector(self):
        expected_Psi = np.array(
            [
                1.79998713456580e00,
                3.13577668317627e-05,
                -7.86064538876201e-06,
                2.10289144438473e-06,
                -9.95219611804585e-07,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_water_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_void_ratio_increment_vector(self):
        expected_de = np.array(
            [
                0.000000000000000e00,
                3.135774071177510e-05,
                -7.860638356391370e-06,
                2.102889435140830e-06,
                -9.952186071820230e-07,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_de,
                self.msh._delta_void_ratio_vector,
                atol=1e-11,
                rtol=1e-8,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 1.13406734711426e-05
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
        c00 = 1.98582857142857e06
        c11 = 1.00532571428571e07
        c10 = 1.53591428571429e06
        c20 = -5.58514285714289e05
        c30 = 2.94771428571427e05
        c21 = -1.25665714285713e06
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
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected0))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected1))

    def test_update_heat_flux_vector(self):
        expected0 = np.zeros(self.msh.num_nodes)
        expected1 = np.ones(self.msh.num_nodes) * 0.0008424421875
        expected1[3::3] = 0.0005616281250
        expected1[0] = 0.0002808140625
        expected1[-1] = 0.0002808140625
        self.msh.update_heat_flux_vector()
        self.assertTrue(np.allclose(self.msh._heat_flux_vector_0, expected0))
        self.assertTrue(
            np.allclose(self.msh._heat_flux_vector, expected1, rtol=1e-13, atol=1e-16)
        )

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
        stiff_el = -np.array(
            [
                [
                    -4.72294599234629e-09,
                    6.08882284823793e-09,
                    -1.81459605925378e-09,
                    4.48719203362137e-10,
                ],
                [
                    4.98148866515888e-09,
                    -1.26517845867392e-08,
                    9.48489198083411e-09,
                    -1.81459605925378e-09,
                ],
                [
                    -1.34835008743102e-09,
                    7.91131182593230e-09,
                    -1.26517845867392e-08,
                    6.08882284823793e-09,
                ],
                [
                    3.12730794913833e-10,
                    -1.34835008743102e-09,
                    4.98148866515888e-09,
                    -3.94586937264169e-09,
                ],
            ]
        )
        for k in range(self.msh.num_elements):
            kmin = 3 * k
            kmax = kmin + 4
            expected1[kmin:kmax, kmin:kmax] += stiff_el
        self.msh.update_stiffness_matrix()
        self.assertTrue(
            np.allclose(
                self.msh._stiffness_matrix_0,
                expected0,
                atol=1e-20,
                rtol=1e-16,
            )
        )
        self.assertTrue(
            np.allclose(
                self.msh._stiffness_matrix,
                expected1,
                atol=1e-20,
                rtol=1e-16,
            )
        )

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
        self.assertTrue(
            np.allclose(
                self.msh._mass_matrix_0,
                expected0,
                atol=1e-12,
                rtol=1e-10,
            )
        )
        self.assertTrue(
            np.allclose(
                self.msh._mass_matrix,
                expected1,
                atol=1e-12,
                rtol=1e-10,
            )
        )

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
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
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
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
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
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
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
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
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
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
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
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector = np.array(
            [
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
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array(
            [
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
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_int_pts)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5000000000000e04,
                1.5110663486802e04,
                1.5221326973605e04,
                1.5331990460407e04,
                1.5442653947209e04,
                1.5553317434012e04,
                1.5663980920814e04,
                1.5774644407616e04,
                1.5885307894418e04,
                1.5995971381221e04,
                1.6106634868023e04,
                1.6217298354825e04,
                1.6327961841628e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.50155736980711e04,
                1.50766118931150e04,
                1.51659952302035e04,
                1.52553785672920e04,
                1.53164167623358e04,
                1.53475641584781e04,
                1.54086023535219e04,
                1.54979856906104e04,
                1.55873690276989e04,
                1.56484072227427e04,
                1.56795546188850e04,
                1.57405928139288e04,
                1.58299761510173e04,
                1.59193594881058e04,
                1.59803976831496e04,
                1.60115450792919e04,
                1.60725832743357e04,
                1.61619666114242e04,
                1.62513499485127e04,
                1.63123881435566e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -5.30271022784199e06,
                -5.32426570535195e06,
                -5.35583119497681e06,
                -5.38739668460167e06,
                -5.40895216211164e06,
                -5.41995178286642e06,
                -5.44150726037638e06,
                -5.47307275000124e06,
                -5.50463823962610e06,
                -5.52619371713607e06,
                -5.53719333789085e06,
                -5.55874881540081e06,
                -5.59031430502567e06,
                -5.62187979465053e06,
                -5.64343527216050e06,
                -5.65443489291528e06,
                -5.67599037042524e06,
                -5.70755586005010e06,
                -5.73912134967496e06,
                -5.76067682718492e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
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
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
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
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.0
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
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
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array(
            [
                [
                    3.12058044707073e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[3:7, 3:7] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[6:10, 6:10] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[9:13, 9:13] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    3.12058044707073e02,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array(
            [
                [
                    3.89690727960341e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[3:7, 3:7] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[6:10, 6:10] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[9:13, 9:13] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    3.89690727960341e03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array(
            [
                [
                    4.54201999537039e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[3:7, 3:7] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[6:10, 6:10] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[9:13, 9:13] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    4.54201999537039e-04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-18, rtol=1e-8
            )
        )


class TestInitializeTimeStepCubic(unittest.TestCase):
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
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 0.0)
        self.assertAlmostEqual(self.msh._t1, 3.75)

    def test_free_indices(self):
        expected_free_vec = [i for i in range(self.msh.num_nodes)][1:]
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
                5.00000000000000,
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
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                2.61335047284981000,
                -3.45235489226239000,
                -5.62500000000000000,
                -4.53571663107363000,
                -4.62527894951383000,
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
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                2.66666666667e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
                0.00000000000e00,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                2.03022679275995000,
                0.41270536206336900,
                -0.16666666666666700,
                0.12380889838036900,
                0.09992561346298070,
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
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -1874.13044414512000000,
                -826.05379629479600000,
                50.00000000000000000,
                143.19096189543300000,
                -243.00672145552500000,
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
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
                1.000000000000000,
                0.044646724785561,
                0.034253111943192,
                0.038503774143103,
                0.038097125437071,
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
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
                0.5934443054215600,
                0.0320752901884927,
                0.0255649251018659,
                0.0282263293087495,
                0.0279717907282567,
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
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.07440359118051350,
                0.00328990208243689,
                0.00227002275809187,
                0.00267646886846198,
                0.00263689857108717,
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
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                0.96148102294413,
                2.09144104991107,
                2.11348332448606,
                2.10422752104159,
                2.10509626519628,
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
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                3.41869702035229e06,
                2.06571964398055e06,
                2.03966461690623e06,
                2.05104242826674e06,
                2.05000353630872e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
                2.04587632179179e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array(
            [
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
            ]
        )
        expected_void_ratio_vector = np.array(
            [
                1.03011911113263,
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
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector_0, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_0_int_pts = np.array(
            [
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
            ]
        )
        expected_void_ratio_int_pts = np.array(
            [
                1.45968759836683000,
                2.55144231478341000,
                2.94249255555421000,
                2.74643452372505000,
                2.76255467425737000,
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
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_0_int_pts)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                1.47827704519679e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                1.11601924183129e-08,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5014406683681e04,
                1.5113450824417e04,
                1.5225567580442e04,
                1.5334778500411e04,
                1.5445441987213e04,
                1.5556105474015e04,
                1.5666768960818e04,
                1.5777432447620e04,
                1.5888095934422e04,
                1.5998759421224e04,
                1.6109422908027e04,
                1.6220086394829e04,
                1.6330749881631e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.50269560377622e04,
                1.50808403178020e04,
                1.51688737787277e04,
                1.52602235703229e04,
                1.53201836630644e04,
                1.53503521984818e04,
                1.54113903935256e04,
                1.55007737306141e04,
                1.55901570677026e04,
                1.56511952627464e04,
                1.56823426588887e04,
                1.57433808539325e04,
                1.58327641910210e04,
                1.59221475281095e04,
                1.59831857231533e04,
                1.60143331192956e04,
                1.60753713143394e04,
                1.61647546514279e04,
                1.62541379885164e04,
                1.63151761835603e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                4.34428753083649e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                -6.57607765407088e42,
                -2.41382227626305e-12,
                -8.42685801212584e18,
                -4.65533588369755e16,
                -5.41995178286642e06,
                -5.44150726037638e06,
                -5.47307275000124e06,
                -5.50463823962610e06,
                -5.52619371713607e06,
                -5.53719333789085e06,
                -5.55874881540081e06,
                -5.59031430502567e06,
                -5.62187979465053e06,
                -5.64343527216050e06,
                -5.65443489291528e06,
                -5.67599037042524e06,
                -5.70755586005010e06,
                -5.73912134967496e06,
                -5.76067682718492e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -1.25038646352300e02,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
                5.57737012356879e03,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                -1.08583734766074e-08,
                -9.05033230479496e-13,
                -3.55123694071925e-17,
                1.62838295160721e-15,
                -1.85482980836306e-15,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.00146857122133434
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
                0.00146857122133434,
                0.00833333333333334,
                0.01682984124681490,
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
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array(
            [
                [
                    3.12058044707073e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[3:7, 3:7] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[6:10, 6:10] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[9:13, 9:13] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    3.12058044707073e02,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array(
            [
                [
                    3.48664027375817e02,
                    -4.38525954412770e02,
                    1.18303395565072e02,
                    -2.84414685281186e01,
                ],
                [
                    -4.38525954412770e02,
                    9.54608208518547e02,
                    -6.34736771763405e02,
                    1.18654517657628e02,
                ],
                [
                    1.18303395565072e02,
                    -6.34736771763405e02,
                    9.30738310807756e02,
                    -4.14304934609422e02,
                ],
                [
                    -2.84414685281186e01,
                    1.18654517657628e02,
                    -4.14304934609422e02,
                    6.36149930186986e02,
                ],
            ]
        )
        expected_H[3:7, 3:7] = np.array(
            [
                [
                    6.36149930186986e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[6:10, 6:10] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[9:13, 9:13] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    3.12058044707073e02,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array(
            [
                [
                    3.89690727960341e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[3:7, 3:7] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[6:10, 6:10] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[9:13, 9:13] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    3.89690727960341e03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array(
            [
                [
                    4.52002240853881e04,
                    2.39630977568868e04,
                    -1.05661537248456e04,
                    2.76842077570419e03,
                ],
                [
                    2.39630977568868e04,
                    3.82078589602083e04,
                    -8.49237092158712e03,
                    -4.76344612225423e02,
                ],
                [
                    -1.05661537248456e04,
                    -8.49237092158712e03,
                    2.94237998605487e04,
                    3.78347953164655e03,
                ],
                [
                    2.76842077570419e03,
                    -4.76344612225423e02,
                    3.78347953164655e03,
                    9.42373481159300e03,
                ],
            ]
        )
        expected_C[3:7, 3:7] = np.array(
            [
                [
                    9.42373481159300e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[6:10, 6:10] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[9:13, 9:13] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    3.89690727960341e03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array(
            [
                -3.00366047323717e-01,
                -1.47600473091470e-01,
                6.82094005132756e-02,
                -1.47844911591501e-02,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
                -0.00000000000000e00,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array(
            [
                [
                    5.95441476472294e-09,
                    -8.86830749678387e-09,
                    3.68596428913000e-09,
                    -7.72071557069070e-10,
                ],
                [
                    -3.11933682107767e-09,
                    4.64583661172691e-09,
                    -1.93096460064881e-09,
                    4.04464809999574e-10,
                ],
                [
                    1.23015669228597e-09,
                    -1.83215450174071e-09,
                    7.61504499932386e-10,
                    -1.59506690477642e-10,
                ],
                [
                    -2.52866524227323e-10,
                    3.76610998914045e-10,
                    -1.56532088382611e-10,
                    3.27876136958889e-11,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array(
            [
                [
                    4.54201999537039e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[3:7, 3:7] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[6:10, 6:10] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[9:13, 9:13] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    4.54201999537039e-04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array(
            [
                [
                    4.93094042881713e-04,
                    3.70587005439364e-04,
                    -1.36584694911526e-04,
                    6.93453948016926e-05,
                ],
                [
                    3.70587005439364e-04,
                    2.30992749463599e-03,
                    -2.92252846848938e-04,
                    -1.26760773887934e-04,
                ],
                [
                    -1.36584694911526e-04,
                    -2.92252846848938e-04,
                    2.30167079772983e-03,
                    3.50939163392181e-04,
                ],
                [
                    6.93453948016926e-05,
                    -1.26760773887934e-04,
                    3.50939163392181e-04,
                    9.08569497065642e-04,
                ],
            ]
        )
        expected_M[3:7, 3:7] = np.array(
            [
                [
                    9.08569497065642e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[6:10, 6:10] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[9:13, 9:13] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    4.54201999537039e-04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.array(
            [
                -4.74113880478950e-07,
                -1.75664455255727e-06,
                3.64362360726964e-07,
                -4.72037810935308e-07,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-18, rtol=1e-8
            )
        )


class TestGlobalCorrectionCubicOneStep(unittest.TestCase):
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
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
                5.00000000000000,
                -4.58064305347386,
                -5.09259369878234,
                -4.80566118243967,
                -5.02013289549439,
                -4.99377774953049,
                -5.01551059728053,
                -4.99839296443377,
                -5.00049740270868,
                -4.99875428179049,
                -5.00013139862023,
                -4.99995013366397,
                -5.00019758038024,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                2.79350985635116000,
                -2.98251124947254000,
                -5.45334184924163000,
                -4.70543987432889000,
                -4.58445666724564000,
                -4.86123110592992000,
                -4.99320367296527000,
                -5.01900150159398000,
                -4.98192929089300000,
                -4.99871835495692000,
                -5.01107494452767000,
                -5.00054167909094000,
                -4.99848427657569000,
                -5.00144192988793000,
                -5.00009669270696000,
                -4.99911676817540000,
                -4.99996589674129000,
                -5.00011137052420000,
                -4.99988835399590000,
                -5.00006237554238000,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                2.66666666667e00,
                1.11828519074e-01,
                -2.46916530086e-02,
                5.18236846828e-02,
                -5.36877213184e-03,
                1.65926679187e-03,
                -4.13615927481e-03,
                4.28542817661e-04,
                -1.32640722314e-04,
                3.32191522536e-04,
                -3.50396320622e-05,
                1.32976896069e-05,
                -5.26881013961e-05,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                2.07826929502698000,
                0.53799700014066100,
                -0.12089115979776900,
                0.07854936684563340,
                0.11081155540116500,
                0.03700503841868800,
                0.00181235387593243,
                -0.00506706709172765,
                0.00481885576187126,
                0.00034177201148769,
                -0.00295331854071352,
                -0.00014444775757991,
                0.00040419291314960,
                -0.00038451463677718,
                -0.00002578472185409,
                0.00023552848655941,
                0.00000909420232786,
                -0.00002969880645241,
                0.00002977226776489,
                -0.00001663347796726,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -1741.61193271617000000,
                -821.99614139238800000,
                -20.08503120444660000,
                134.41344831667500000,
                -132.08827298017400000,
                -43.14004787916920000,
                -16.04239999098820000,
                4.60719177933043000,
                2.63871349862029000,
                -11.70195433090690000,
                3.44343255039109000,
                1.28022031909985000,
                -0.36788074456221100,
                -0.20994619054999900,
                0.93565943723484700,
                -0.28072726191793200,
                -0.10084016061352900,
                0.03168726204377850,
                0.00865252298432040,
                -0.09646408983621770,
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
                1.000000000000000,
                0.048327866859888,
                0.034835486316865,
                0.037743275744262,
                0.038280962976073,
                0.037081343248642,
                0.036545564728929,
                0.036443363829594,
                0.036590484142070,
                0.036523649742313,
                0.036474679839319,
                0.036516412127335,
                0.036524579200757,
                0.036512840109049,
                0.036518178110201,
                0.036522067910648,
                0.036518697235828,
                0.036518119855644,
                0.036519005010708,
                0.036518314312078,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
                0.5933409036663990,
                0.0347137941455160,
                0.0259983926596006,
                0.0276700304850956,
                0.0281049333503606,
                0.0273971641035589,
                0.0270037247601581,
                0.0269284930407589,
                0.0270364176827250,
                0.0269874890826638,
                0.0269515603071046,
                0.0269820902347432,
                0.0269880888155971,
                0.0269795141405410,
                0.0269834000408081,
                0.0269862422709633,
                0.0269837915274113,
                0.0269833679044109,
                0.0269840099015926,
                0.0269835118209834,
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.07267036608405440,
                0.00383154791709323,
                0.00217340770043845,
                0.00233005454441085,
                0.00269826017242343,
                0.00297959265767964,
                0.00294797975174945,
                0.00290483805406171,
                0.00291789086633871,
                0.00296474988215058,
                0.00290105458212178,
                0.00295210015848509,
                0.00290256774629881,
                0.00289555349912283,
                0.00299179611781682,
                0.00289045890194653,
                0.00299683202572657,
                0.00288605324259997,
                0.00287136043694659,
                0.00284574182773160,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                0.96161190808560,
                2.08386909830600,
                2.11222278039383,
                2.10583789102445,
                2.10470615678924,
                2.10729705788461,
                2.10844260973825,
                2.10866116401499,
                2.10834644401980,
                2.10848943940575,
                2.10859419730293,
                2.10850489385686,
                2.10848742524793,
                2.10851254914857,
                2.10850112058321,
                2.10849279599814,
                2.10850001343686,
                2.10850124854706,
                2.10849935270860,
                2.10850083296573,
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                3.41849728952202e06,
                2.07189843589171e06,
                2.04068492699884e06,
                2.04973762945771e06,
                2.05033158994784e06,
                2.04686180026389e06,
                2.04592197585812e06,
                2.04574415057103e06,
                2.04600325157033e06,
                2.04588471657155e06,
                2.04579851440537e06,
                2.04587272106436e06,
                2.04588704256472e06,
                2.04586605845371e06,
                2.04587571105607e06,
                2.04588265602292e06,
                2.04587653577673e06,
                2.04587551949074e06,
                2.04587713996668e06,
                2.04587585132210e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array(
            [
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
            ]
        )
        expected_void_ratio_vector = np.array(
            [
                1.03011911113263,
                2.82860362814449,
                2.83030219017640,
                2.82864319196977,
                2.83020739309145,
                2.82992420987324,
                2.83017238134810,
                2.82997363821678,
                2.83000967971728,
                2.82997775724541,
                2.83000350117433,
                2.82999835238855,
                2.83000556068865,
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector_0, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_0_int_pts = np.array(
            [
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
            ]
        )
        expected_void_ratio_int_pts = np.array(
            [
                1.45906217029424000,
                2.54984638864668000,
                2.94196187886160000,
                2.74688342963922000,
                2.76187613284318000,
                2.82906415640503000,
                2.83003329007169000,
                2.83014805333527000,
                2.82983353714076000,
                2.83001619266510000,
                2.83011887186063000,
                2.82999572897865000,
                2.82998123267581000,
                2.83002115207895000,
                2.82999770360935000,
                2.82998486870991000,
                2.83000087809921000,
                2.83000208525824000,
                2.82999724622774000,
                2.83000217846009000,
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_0_int_pts)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                1.47131359995378e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                1.11076221717147e-08,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5014449426337e04,
                1.5113476028161e04,
                1.5225585741593e04,
                1.5334795051351e04,
                1.5445453476614e04,
                1.5556118992751e04,
                1.5666780872216e04,
                1.5777445060385e04,
                1.5888108274891e04,
                1.5998771975971e04,
                1.6109435372849e04,
                1.6220098890878e04,
                1.6330762361581e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.50269954877711e04,
                1.50808695652533e04,
                1.51688944656309e04,
                1.52602408925064e04,
                1.53202003019101e04,
                1.53503672063938e04,
                1.54114021870632e04,
                1.55007860187946e04,
                1.55901709772620e04,
                1.56512080222119e04,
                1.56823547825786e04,
                1.57433934223862e04,
                1.58327766980807e04,
                1.59221598148903e04,
                1.59831981636064e04,
                1.60143456477484e04,
                1.60753837861055e04,
                1.61647671272493e04,
                1.62541504924742e04,
                1.63151886749502e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                4.37282420168772e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                -1.08583596188892e41,
                -5.91856127895118e-12,
                -1.21595631208941e19,
                -5.28973693373085e16,
                -7.40148242121397e06,
                -5.37422054411254e06,
                -5.20382625831659e06,
                -5.82652993204991e06,
                -5.49396766371034e06,
                -5.31531355577168e06,
                -5.56744793090401e06,
                -5.62661079891232e06,
                -5.58085776096722e06,
                -5.64806958757320e06,
                -5.68421501797755e06,
                -5.67421076576147e06,
                -5.70341918345798e06,
                -5.74463970437447e06,
                -5.75628339299024e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -1.25859997763622e02,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
                5.59237749521184e03,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                -1.08612342162270e-08,
                -6.11415768983904e-12,
                2.80596141810489e-17,
                7.17663208907426e-16,
                -1.20799268738528e-15,
                -1.04751860522187e-16,
                -8.90094253171621e-18,
                -3.36915779991035e-17,
                2.31169487646101e-18,
                -8.02371291610817e-19,
                -2.54441256571506e-17,
                -8.77500870088515e-18,
                -4.26797726991474e-20,
                1.48876702940155e-18,
                -5.95633331983979e-18,
                1.03233810240133e-20,
                9.58554565378461e-20,
                -2.02596451262750e-19,
                -5.33501961492381e-21,
                6.00287543301189e-19,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.00147292827080989
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
                0.00147292827080989,
                0.00833502512822456,
                0.01683036648801830,
                0.02500057167654190,
                0.03333320517382640,
                0.04166680098768840,
                0.04999992740615340,
                0.05833334957732680,
                0.06666664963848040,
                0.07500000907423080,
                0.08333333154089270,
                0.09166666857113480,
                0.10000000000000000,
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array(
            [
                [
                    3.12058044707073e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[3:7, 3:7] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[6:10, 6:10] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[9:13, 9:13] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    3.12058044707073e02,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array(
            [
                [
                    3.48551857427586e02,
                    -4.38664308836745e02,
                    1.18617959352344e02,
                    -2.85055079431851e01,
                ],
                [
                    -4.38664308836745e02,
                    9.54878072615016e02,
                    -6.34899807055506e02,
                    1.18686043277235e02,
                ],
                [
                    1.18617959352344e02,
                    -6.34899807055506e02,
                    9.30635630486192e02,
                    -4.14353782783030e02,
                ],
                [
                    -2.85055079431851e01,
                    1.18686043277235e02,
                    -4.14353782783030e02,
                    6.36209425205201e02,
                ],
            ]
        )
        expected_H[3:7, 3:7] = np.array(
            [
                [
                    6.36209425205201e02,
                    -3.98478223373076e02,
                    1.13850287547035e02,
                    -2.74082419301801e01,
                ],
                [
                    -3.98478223373076e02,
                    9.10831006786278e02,
                    -6.26205755062565e02,
                    1.13852971649363e02,
                ],
                [
                    1.13850287547035e02,
                    -6.26205755062565e02,
                    9.10856039877672e02,
                    -3.98500572362142e02,
                ],
                [
                    -2.74082419301801e01,
                    1.13852971649363e02,
                    -3.98500572362142e02,
                    6.24110412958808e02,
                ],
            ]
        )
        expected_H[6:10, 6:10] = np.array(
            [
                [
                    6.24110412958808e02,
                    -3.98500971469836e02,
                    1.13856465092936e02,
                    -2.74100639389494e01,
                ],
                [
                    -3.98500971469836e02,
                    9.10864503455536e02,
                    -6.26222393931813e02,
                    1.13858861946112e02,
                ],
                [
                    1.13856465092936e02,
                    -6.26222393931813e02,
                    9.10872617007105e02,
                    -3.98506688168228e02,
                ],
                [
                    -2.74100639389494e01,
                    1.13858861946112e02,
                    -3.98506688168228e02,
                    6.24116834896652e02,
                ],
            ]
        )
        expected_H[9:13, 9:13] = np.array(
            [
                [
                    6.24116834896652e02,
                    -3.98507945860702e02,
                    1.13859597237453e02,
                    -2.74105961123375e01,
                ],
                [
                    -3.98507945860702e02,
                    9.10874040576192e02,
                    -6.26225103532984e02,
                    1.13859008817494e02,
                ],
                [
                    1.13859597237453e02,
                    -6.26225103532984e02,
                    9.10871868681651e02,
                    -3.98506362386120e02,
                ],
                [
                    -2.74105961123375e01,
                    1.13859008817494e02,
                    -3.98506362386120e02,
                    3.12057949680963e02,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array(
            [
                [
                    3.89690727960341e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[3:7, 3:7] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[6:10, 6:10] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[9:13, 9:13] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    3.89690727960341e03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array(
            [
                [
                    4.43191512823289e04,
                    2.36981820128043e04,
                    -1.04228560749454e04,
                    2.72752667799082e03,
                ],
                [
                    2.36981820128043e04,
                    3.90029293105477e04,
                    -8.55740958874031e03,
                    -4.24779044556544e02,
                ],
                [
                    -1.04228560749454e04,
                    -8.55740958874031e03,
                    2.86880738707144e04,
                    3.70202795202652e03,
                ],
                [
                    2.72752667799082e03,
                    -4.24779044556544e02,
                    3.70202795202652e03,
                    1.11623327575310e04,
                ],
            ]
        )
        expected_C[3:7, 3:7] = np.array(
            [
                [
                    1.11623327575310e04,
                    4.34504137087101e03,
                    -1.57859291028286e03,
                    8.32013883836929e02,
                ],
                [
                    4.34504137087101e03,
                    2.83286569503940e04,
                    -3.56356665091589e03,
                    -1.57154179183068e03,
                ],
                [
                    -1.57859291028286e03,
                    -3.56356665091589e03,
                    2.82710488146534e04,
                    4.33093913396665e03,
                ],
                [
                    8.32013883836929e02,
                    -1.57154179183068e03,
                    4.33093913396665e03,
                    1.11847844690370e04,
                ],
            ]
        )
        expected_C[6:10, 6:10] = np.array(
            [
                [
                    1.11847844690370e04,
                    4.32565154978876e03,
                    -1.57068969983964e03,
                    8.30341858342614e02,
                ],
                [
                    4.32565154978876e03,
                    2.83231771696813e04,
                    -3.55281645032842e03,
                    -1.57371258764738e03,
                ],
                [
                    -1.57068969983964e03,
                    -3.55281645032842e03,
                    2.82262891058439e04,
                    4.33169732540424e03,
                ],
                [
                    8.30341858342614e02,
                    -1.57371258764738e03,
                    4.33169732540424e03,
                    1.11939182473388e04,
                ],
            ]
        )
        expected_C[9:13, 9:13] = np.array(
            [
                [
                    1.11939182473388e04,
                    4.33875339905279e03,
                    -1.57527170728184e03,
                    8.26460701311306e02,
                ],
                [
                    4.33875339905279e03,
                    2.83926657073714e04,
                    -3.56462995346732e03,
                    -1.54969645637448e03,
                ],
                [
                    -1.57527170728184e03,
                    -3.56462995346732e03,
                    2.81533764708163e04,
                    4.28760289723806e03,
                ],
                [
                    8.26460701311306e02,
                    -1.54969645637448e03,
                    4.28760289723806e03,
                    5.54428380775912e03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array(
            [
                -2.79290103673607e-01,
                -1.37357548454370e-01,
                6.34480778125953e-02,
                -1.37516584370449e-02,
                -2.21354685809020e-11,
                1.30181167751189e-11,
                -1.80114564012573e-12,
                6.89011230770680e-13,
                -2.24126575586310e-13,
                1.08147550668570e-13,
                2.50491651539409e-16,
                3.12862117696491e-16,
                5.49717497043505e-16,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array(
            [
                [
                    5.94778993558129e-09,
                    -8.85844069639067e-09,
                    3.68186331790065e-09,
                    -7.71212557091269e-10,
                ],
                [
                    -3.13517819888019e-09,
                    4.66943023344734e-09,
                    -1.94077089651127e-09,
                    4.06518861944119e-10,
                ],
                [
                    1.23703757204794e-09,
                    -1.84240265541974e-09,
                    7.65763974302678e-10,
                    -1.60398890930876e-10,
                ],
                [
                    -2.54329299095376e-10,
                    3.78789607197318e-10,
                    -1.57437590625859e-10,
                    3.29772825239172e-11,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array(
            [
                [
                    4.54201999537039e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[3:7, 3:7] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[6:10, 6:10] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[9:13, 9:13] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    4.54201999537039e-04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array(
            [
                [
                    4.93106606407249e-04,
                    3.70668389131976e-04,
                    -1.36612809358475e-04,
                    6.93490903675524e-05,
                ],
                [
                    3.70668389131976e-04,
                    2.31051693484695e-03,
                    -2.92328311245078e-04,
                    -1.26736712087998e-04,
                ],
                [
                    -1.36612809358475e-04,
                    -2.92328311245078e-04,
                    2.30161797775055e-03,
                    3.50916194591022e-04,
                ],
                [
                    6.93490903675524e-05,
                    -1.26736712087998e-04,
                    3.50916194591022e-04,
                    9.08598613506195e-04,
                ],
            ]
        )
        expected_M[3:7, 3:7] = np.array(
            [
                [
                    9.08598613506195e-04,
                    3.51308951039910e-04,
                    -1.27748674621162e-04,
                    6.74217876133794e-05,
                ],
                [
                    3.51308951039910e-04,
                    2.29940389124752e-03,
                    -2.87434805392762e-04,
                    -1.27743568740252e-04,
                ],
                [
                    -1.27748674621162e-04,
                    -2.87434805392762e-04,
                    2.29940612987324e-03,
                    3.51298739278092e-04,
                ],
                [
                    6.74217876133794e-05,
                    -1.27743568740252e-04,
                    3.51298739278092e-04,
                    9.08402685377218e-04,
                ],
            ]
        )
        expected_M[6:10, 6:10] = np.array(
            [
                [
                    9.08402685377218e-04,
                    3.51295914426545e-04,
                    -1.27743973659913e-04,
                    6.74205173960884e-05,
                ],
                [
                    3.51295914426545e-04,
                    2.29939713428068e-03,
                    -2.87423901320159e-04,
                    -1.27744371069125e-04,
                ],
                [
                    -1.27743973659913e-04,
                    -2.87423901320159e-04,
                    2.29939694975551e-03,
                    3.51296709244970e-04,
                ],
                [
                    6.74205173960884e-05,
                    -1.27744371069125e-04,
                    3.51296709244970e-04,
                    9.08404105019454e-04,
                ],
            ]
        )
        expected_M[9:13, 9:13] = np.array(
            [
                [
                    9.08404105019454e-04,
                    3.51296933579600e-04,
                    -1.27744339831523e-04,
                    6.74206160715568e-05,
                ],
                [
                    3.51296933579600e-04,
                    2.29939765744209e-03,
                    -2.87424761749663e-04,
                    -1.27744305532206e-04,
                ],
                [
                    -1.27744339831523e-04,
                    -2.87424761749663e-04,
                    2.29939767329788e-03,
                    3.51296864980965e-04,
                ],
                [
                    6.74206160715568e-05,
                    -1.27744305532206e-04,
                    3.51296864980965e-04,
                    4.54201991137872e-04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.array(
            [
                -6.35637111457762e-07,
                -3.29138294996811e-06,
                7.93886781303014e-07,
                -5.83294154838650e-07,
                -2.59930476232246e-08,
                1.17788090321145e-08,
                -5.17266314056848e-09,
                1.99505027478770e-09,
                -9.17895621047170e-10,
                4.06330855770363e-10,
                -1.65524749635941e-10,
                8.60565107688571e-11,
                -3.22785950809545e-11,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-18, rtol=1e-8
            )
        )

    def test_residual_heat_flux_vector(self):
        expected_Psi = np.array(
            [
                -1.04397436792134e01,
                4.64393784123848e-01,
                -1.35446062903610e-01,
                2.81402877433673e-01,
                -4.30134635057238e-02,
                1.57189221152992e-02,
                -3.57520049198517e-02,
                5.46745117053885e-03,
                -2.00757972668227e-03,
                4.61316192514244e-03,
                -7.26145858587236e-04,
                3.41715698158700e-04,
                -1.15329048128561e-03,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.000000000000000e00,
                4.193569465261340e-01,
                -9.259369878233970e-02,
                1.943388175603230e-01,
                -2.013289549438570e-02,
                6.222250469512280e-03,
                -1.551059728053060e-02,
                1.607035566226830e-03,
                -4.974027086765300e-04,
                1.245718209510860e-03,
                -1.313986202336430e-04,
                4.986633602600810e-05,
                -1.975803802350190e-04,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
                atol=1e-10,
                rtol=1e-8,
            )
        )

    def test_residual_water_flux_vector(self):
        expected_Psi = np.array(
            [
                1.79944279757781e00,
                -1.39638949606890e-03,
                3.02196599714599e-04,
                -1.35682072166694e-03,
                2.07395031378053e-04,
                -7.57908356972426e-05,
                1.72382960539653e-04,
                -2.63620298077366e-05,
                9.67980782002828e-06,
                -2.22429626502778e-05,
                3.50120708384002e-06,
                -1.64762686298354e-06,
                5.56074066256944e-06,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_water_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_void_ratio_increment_vector(self):
        expected_de = np.array(
            [
                0.000000000000000e00,
                -1.396371855507080e-03,
                3.021901764035320e-04,
                -1.356808030228990e-03,
                2.073930914451410e-04,
                -7.579012676446070e-05,
                1.723813481028630e-04,
                -2.636178322242110e-05,
                9.679717276982760e-06,
                -2.224275459391780e-05,
                3.501174334227810e-06,
                -1.647611451401320e-06,
                5.560688648479460e-06,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_de,
                self.msh._delta_void_ratio_vector,
                atol=1e-11,
                rtol=1e-8,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 2.63801493284172e-02
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a, delta=1e-10)
        self.assertEqual(self.msh._iter, 1)


class TestGlobalCorrectionCubicIterative(unittest.TestCase):
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
            nodes=(self.msh.nodes[0],),
            bnd_value=0.1,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_thrm[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_thrm[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_thrm[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec_cnsl[0]))
        self.assertTrue(
            np.all(expected_free_vec == self.msh._free_arr_cnsl[0].flatten())
        )
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr_cnsl[1]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_vec[0]))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_nodes = np.array(
            [
                5.00000000000000,
                -4.59336043823724,
                -5.09189016980125,
                -4.83500370228240,
                -5.01943985311486,
                -4.99377254177921,
                -5.01502121760800,
                -4.99823729250062,
                -5.00056703604783,
                -4.99862484548620,
                -5.00016475387809,
                -4.99993407918690,
                -5.00024775882627,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_nodes, expected_temp_nodes))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                2.78753149027896000,
                -2.99733112041194000,
                -5.45826573562900000,
                -4.70612026244036000,
                -4.60433441774759000,
                -4.88329390928973000,
                -4.99699945831122000,
                -5.01680541463476000,
                -4.98338078378341000,
                -4.99956316340016000,
                -5.01062724176644000,
                -5.00028019995579000,
                -4.99847455586512000,
                -5.00151017943026000,
                -5.00003277049020000,
                -4.99903535681406000,
                -4.99998698343686000,
                -5.00012605582953000,
                -4.99986514117356000,
                -5.00008395595302000,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_nodes = np.array(
            [
                2.66666666667e00,
                1.08437216470e-01,
                -2.45040452803e-02,
                4.39990127247e-02,
                -5.18396083063e-03,
                1.66065552555e-03,
                -4.00565802880e-03,
                4.70055333168e-04,
                -1.51209612755e-04,
                3.66707870347e-04,
                -4.39343674910e-05,
                1.75788834923e-05,
                -6.60690203380e-05,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(actual_temp_rate_nodes, expected_temp_rate_nodes))

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                2.07667506407439000,
                0.53404503455682200,
                -0.12220419616773300,
                0.07836793001590890,
                0.10551082193397800,
                0.03112162418940620,
                0.00080014445034629,
                -0.00448144390260325,
                0.00443179099109607,
                0.00011648975995821,
                -0.00283393113771774,
                -0.00007471998820577,
                0.00040678510263611,
                -0.00040271451473242,
                -0.00000873879738507,
                0.00025723818291689,
                0.00000347108350871,
                -0.00003361488787480,
                0.00003596235372228,
                -0.00002238825413807,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -1745.95635757254000000,
                -821.92238819607900000,
                -18.12649524972980000,
                132.80762983228200000,
                -139.25921733969400000,
                -37.43614214914010000,
                -13.60380436055170000,
                4.36517460694162000,
                2.21305611440800000,
                -10.81824597672170000,
                3.40639247703410000,
                1.23815239423931000,
                -0.39649723948248600,
                -0.20037494685313100,
                0.98552194667971100,
                -0.31737757915885800,
                -0.11078406278088000,
                0.03925565001066650,
                0.00806751629207270,
                -0.11736449839958100,
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_gradient_int_pts, expected_temp_gradient_int_pts)
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
                1.000000000000000,
                0.048198366112588,
                0.034818390961136,
                0.037740311977204,
                0.038191133883221,
                0.036990229145682,
                0.036530476545711,
                0.036452032464256,
                0.036584692359737,
                0.036520295804319,
                0.036476450877253,
                0.036517449810041,
                0.036524617800369,
                0.036512569348493,
                0.036518431813137,
                0.036522391124839,
                0.036518613541899,
                0.036518061571631,
                0.036519097146136,
                0.036518228661116,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_distribution(self):
        expected_vol_water_cont_int_pts = np.array(
            [
                0.5932500495532120,
                0.0346152965145001,
                0.0259844793346234,
                0.0276693218852949,
                0.0280387830775644,
                0.0273288989153381,
                0.0269925703740453,
                0.0269350366357119,
                0.0270319940550516,
                0.0269850138892345,
                0.0269529648970372,
                0.0269828569954122,
                0.0269881029635143,
                0.0269793291714689,
                0.0269835868958146,
                0.0269864708987725,
                0.0269837299142696,
                0.0269833262065366,
                0.0269840763393153,
                0.0269834497125950,
            ]
        )
        actual_vol_water_cont_int_pts = np.array(
            [ip.vol_water_cont for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_int_pts,
                expected_vol_water_cont_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.07271448737438800,
                0.00381071842010970,
                0.00218041610655991,
                0.00233303787908237,
                0.00266663010182653,
                0.00295793979186311,
                0.00295981502440837,
                0.00289506037302929,
                0.00290655881412038,
                0.00303217166698332,
                0.00289110093769945,
                0.00297048676872090,
                0.00289334610163643,
                0.00288717626342214,
                0.00312567368190168,
                0.00288352536950876,
                0.00311822681233537,
                0.00288062153220294,
                0.00286976872078942,
                0.00285403321276773,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                0.96172692522414,
                2.08413756331431,
                2.11225920789077,
                2.10584447824944,
                2.10489656987057,
                2.10749143714632,
                2.10847485870680,
                2.10864267750042,
                2.10835877835333,
                2.10849660945950,
                2.10859044211227,
                2.10850267579176,
                2.10848733821184,
                2.10851313268246,
                2.10850057809672,
                2.10849210191022,
                2.10850019240638,
                2.10850137356233,
                2.10849915524946,
                2.10850101641871,
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_thrm_cond_int_pts, expected_thrm_cond_int_pts, atol=1e-30
            )
        )

    def test_vol_heat_cap_distribution(self):
        expected_vol_heat_cap_int_pts = np.array(
            [
                3.41832179571698e06,
                2.07171410689456e06,
                2.04066558690832e06,
                2.04972079863794e06,
                2.05017997236275e06,
                2.04671323512722e06,
                2.04589612467034e06,
                2.04575786717713e06,
                2.04599451740763e06,
                2.04587893404750e06,
                2.04580074770793e06,
                2.04587450210258e06,
                2.04588722945674e06,
                2.04586546693455e06,
                2.04587615162508e06,
                2.04588329639208e06,
                2.04587639019939e06,
                2.04587540796020e06,
                2.04587731189395e06,
                2.04587569440347e06,
            ]
        )
        actual_vol_heat_cap_int_pts = np.array(
            [ip.vol_heat_cap for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_vol_heat_cap_int_pts, expected_vol_heat_cap_int_pts, atol=1e-30
            )
        )

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array(
            [
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
            ]
        )
        expected_void_ratio_vector = np.array(
            [
                1.03011911113263,
                2.82732769777332,
                2.83060266206765,
                2.82810849824174,
                2.83027647024316,
                2.82990074434278,
                2.83022752071687,
                2.82996632939256,
                2.83001219405090,
                2.82997182810954,
                2.83000432513294,
                2.82999798901623,
                2.83000684041998,
            ]
        )
        actual_void_ratio_nodes = np.array([nd.void_ratio for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, actual_void_ratio_nodes)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector, self.msh._void_ratio_vector)
        )
        self.assertTrue(
            np.allclose(expected_void_ratio_vector_0, self.msh._void_ratio_vector_0)
        )

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_0_int_pts = np.array(
            [
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
            ]
        )
        expected_void_ratio_int_pts = np.array(
            [
                1.45851289939080000,
                2.54841486782723000,
                2.94144660182465000,
                2.74742817069433000,
                2.76180203130300000,
                2.82868903802743000,
                2.83003102444347000,
                2.83020368201968000,
                2.82977575475112000,
                2.83001741632255000,
                2.83015746051716000,
                2.82999573448093000,
                2.82997546013529000,
                2.83002721745017000,
                2.82999745982161000,
                2.82998077364428000,
                2.83000096992057000,
                2.83000263492581000,
                2.82999658689906000,
                2.83000265221479000,
            ]
        )
        actual_void_ratio_int_pts = np.array(
            [ip.void_ratio for e in self.msh.elements for ip in e.int_pts]
        )
        actual_void_ratio_0_int_pts = np.array(
            [ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_0_int_pts, expected_void_ratio_0_int_pts)
        )
        self.assertTrue(
            np.allclose(actual_void_ratio_int_pts, expected_void_ratio_int_pts)
        )

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array(
            [
                1.46522513851929e-09,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_int_pts = np.array(
            [ip.hyd_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_int_pts,
                expected_hyd_cond_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_hyd_cond_grad_distribution(self):
        expected_hyd_cond_grad_int_pts = np.array(
            [
                1.10616575797857e-08,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_hyd_cond_grad_int_pts = np.array(
            [ip.hyd_cond_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_hyd_cond_grad_int_pts,
                expected_hyd_cond_grad_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_tot_stress_distribution_nodes(self):
        expected_sig_nodes = np.array(
            [
                1.5014479620178e04,
                1.5113483910882e04,
                1.5225583648097e04,
                1.5334798678723e04,
                1.5445454169474e04,
                1.5556120590663e04,
                1.5666781770705e04,
                1.5777446241166e04,
                1.5888109365526e04,
                1.5998773136132e04,
                1.6109436505259e04,
                1.6220100031655e04,
                1.6330763498077e04,
            ]
        )
        actual_sig_nodes = np.array([nd.tot_stress for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_sig_nodes,
                actual_sig_nodes,
            )
        )

    def test_tot_stress_distribution_int_pts(self):
        expected_sig_int_pts = np.array(
            [
                1.50270219219508e04,
                1.50808831554909e04,
                1.51688956082446e04,
                1.52602387313300e04,
                1.53202020491082e04,
                1.53503699846414e04,
                1.54114031211700e04,
                1.55007870244878e04,
                1.55901727836115e04,
                1.56512093201415e04,
                1.56823557632528e04,
                1.57433945806791e04,
                1.58327778470867e04,
                1.59221608852068e04,
                1.59831992843901e04,
                1.60143467999937e04,
                1.60753849209441e04,
                1.61647682623758e04,
                1.62541516355449e04,
                1.63151898146409e04,
            ]
        )
        actual_sig_int_pts = np.array(
            [ip.tot_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sig_int_pts,
            )
        )

    def test_eff_stress_distribution(self):
        expected_sig_int_pts = np.array(
            [
                4.39804057856260e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_sigp_int_pts = np.array(
            [ip.eff_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_sig_int_pts,
                actual_sigp_int_pts,
            )
        )

    def test_tot_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                0.00000000000000e00,
                -1.88258172696558e41,
                -7.02017314734135e-12,
                -1.01103461885862e19,
                -5.70256791894480e16,
                -8.46201300381934e06,
                -5.38058285047512e06,
                -5.10133829393024e06,
                -5.94737415898987e06,
                -5.49206543188994e06,
                -5.24306334510869e06,
                -5.56728817866143e06,
                -5.63808524023088e06,
                -5.56895321899750e06,
                -5.64851895464935e06,
                -5.69239285034752e06,
                -5.67403899313400e06,
                -5.70232071329682e06,
                -5.74596378579855e06,
                -5.75533308714600e06,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.tot_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_eff_stress_grad_distribution(self):
        expected_dsigde_int_pts = np.array(
            [
                -1.26585783432264e02,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_dsigde_int_pts = np.array(
            [ip.eff_stress_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                expected_dsigde_int_pts,
                actual_dsigde_int_pts,
            )
        )

    def test_pre_consol_stress_distribution(self):
        expected_ppc_int_pts = np.array(
            [
                5.60559074945287e03,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_ppc_int_pts = np.array(
            [ip.pre_consol_stress for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_ppc_int_pts,
                expected_ppc_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                -1.08636947045648e-08,
                -5.75846451652844e-12,
                2.48414478793271e-17,
                7.06771501030202e-16,
                -1.16668236601284e-15,
                -8.03062112339816e-17,
                -4.25215566010325e-18,
                -3.20613638663517e-17,
                1.87170831015227e-18,
                2.55016078239359e-18,
                -2.51773150519001e-17,
                -8.26585304665238e-18,
                -4.67194591004657e-20,
                1.42309590319089e-18,
                -5.97567929887524e-18,
                3.78166006749042e-21,
                1.35264234234654e-19,
                -2.52334581235649e-19,
                -4.54737507671192e-21,
                7.40051327938119e-19,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts,
                expected_water_flux_int_pts,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_calculate_settlement(self):
        expected = 0.00147600613434438
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array(
            [
                0.00147600613434438,
                0.00833572050968801,
                0.01682996070664900,
                0.02500082556066400,
                0.03333315740881080,
                0.04166685045445360,
                0.04999990165622240,
                0.05833335455029380,
                0.06666664449951950,
                0.07500001174039080,
                0.08333333106932270,
                0.09166666906390220,
                0.10000000000000000,
            ]
        )
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array(
            [
                [
                    3.12058044707073e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[3:7, 3:7] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[6:10, 6:10] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    6.24116089414146e02,
                ],
            ]
        )
        expected_H[9:13, 9:13] = np.array(
            [
                [
                    6.24116089414146e02,
                    -3.98506557092141e02,
                    1.13859016312040e02,
                    -2.74105039269726e01,
                ],
                [
                    -3.98506557092141e02,
                    9.10872130496322e02,
                    -6.26224589716221e02,
                    1.13859016312040e02,
                ],
                [
                    1.13859016312040e02,
                    -6.26224589716221e02,
                    9.10872130496322e02,
                    -3.98506557092141e02,
                ],
                [
                    -2.74105039269726e01,
                    1.13859016312040e02,
                    -3.98506557092141e02,
                    3.12058044707073e02,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] = np.array(
            [
                [
                    3.48781836546058e02,
                    -4.38905172104480e02,
                    1.18638999958430e02,
                    -2.85156644000068e01,
                ],
                [
                    -4.38905172104480e02,
                    9.55262794271496e02,
                    -6.35105767134820e02,
                    1.18748144967805e02,
                ],
                [
                    1.18638999958430e02,
                    -6.35105767134820e02,
                    9.30881330356296e02,
                    -4.14414563179905e02,
                ],
                [
                    -2.85156644000068e01,
                    1.18748144967805e02,
                    -4.14414563179905e02,
                    6.36282977995296e02,
                ],
            ]
        )
        expected_H[3:7, 3:7] = np.array(
            [
                [
                    6.36282977995296e02,
                    -3.98572992385527e02,
                    1.13887815917911e02,
                    -2.74157189155734e01,
                ],
                [
                    -3.98572992385527e02,
                    9.10963299218432e02,
                    -6.26252906720161e02,
                    1.13862599887255e02,
                ],
                [
                    1.13887815917911e02,
                    -6.26252906720161e02,
                    9.10872612417769e02,
                    -3.98507521615519e02,
                ],
                [
                    -2.74157189155734e01,
                    1.13862599887255e02,
                    -3.98507521615519e02,
                    6.24110303354203e02,
                ],
            ]
        )
        expected_H[6:10, 6:10] = np.array(
            [
                [
                    6.24110303354203e02,
                    -3.98493767037540e02,
                    1.13853600875775e02,
                    -2.74094965486020e01,
                ],
                [
                    -3.98493767037540e02,
                    9.10854463756617e02,
                    -6.26218847004048e02,
                    1.13858150284970e02,
                ],
                [
                    1.13853600875775e02,
                    -6.26218847004048e02,
                    9.10871430221844e02,
                    -3.98506184093572e02,
                ],
                [
                    -2.74094965486020e01,
                    1.13858150284970e02,
                    -3.98506184093572e02,
                    6.24116875065304e02,
                ],
            ]
        )
        expected_H[9:13, 9:13] = np.array(
            [
                [
                    6.24116875065304e02,
                    -3.98508535924215e02,
                    1.13859829844230e02,
                    -2.74106386281150e01,
                ],
                [
                    -3.98508535924215e02,
                    9.10874859330671e02,
                    -6.26225367323129e02,
                    1.13859043916674e02,
                ],
                [
                    1.13859829844230e02,
                    -6.26225367323129e02,
                    9.10871885528700e02,
                    -3.98506348049800e02,
                ],
                [
                    -2.74106386281150e01,
                    1.13859043916674e02,
                    -3.98506348049800e02,
                    3.12057942761241e02,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix_0(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array(
            [
                [
                    3.89690727960341e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[3:7, 3:7] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[6:10, 6:10] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    7.79381455920682e03,
                ],
            ]
        )
        expected_C[9:13, 9:13] = np.array(
            [
                [
                    7.79381455920682e03,
                    3.01401422406826e03,
                    -1.09600517238846e03,
                    5.78447174316131e02,
                ],
                [
                    3.01401422406826e03,
                    1.97280931029923e04,
                    -2.46601163787403e03,
                    -1.09600517238846e03,
                ],
                [
                    -1.09600517238846e03,
                    -2.46601163787403e03,
                    1.97280931029923e04,
                    3.01401422406826e03,
                ],
                [
                    5.78447174316131e02,
                    -1.09600517238846e03,
                    3.01401422406826e03,
                    3.89690727960341e03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] = np.array(
            [
                [
                    4.43409364026058e04,
                    2.37026354282405e04,
                    -1.04272010276310e04,
                    2.72763642069820e03,
                ],
                [
                    2.37026354282405e04,
                    3.89701284179230e04,
                    -8.54470011893541e03,
                    -4.23046179021659e02,
                ],
                [
                    -1.04272010276310e04,
                    -8.54470011893541e03,
                    2.86933950175783e04,
                    3.69432573102179e03,
                ],
                [
                    2.72763642069820e03,
                    -4.23046179021659e02,
                    3.69432573102179e03,
                    1.11346810388318e04,
                ],
            ]
        )
        expected_C[3:7, 3:7] = np.array(
            [
                [
                    1.11346810388318e04,
                    4.34340381499804e03,
                    -1.57619756900004e03,
                    8.33097629193998e02,
                ],
                [
                    4.34340381499804e03,
                    2.83430733122225e04,
                    -3.57302603099385e03,
                    -1.57723742387397e03,
                ],
                [
                    -1.57619756900004e03,
                    -3.57302603099385e03,
                    2.82510622342046e04,
                    4.34548352474590e03,
                ],
                [
                    8.33097629193998e02,
                    -1.57723742387397e03,
                    4.34548352474590e03,
                    1.12150223592120e04,
                ],
            ]
        )
        expected_C[6:10, 6:10] = np.array(
            [
                [
                    1.12150223592120e04,
                    4.32859868586274e03,
                    -1.56913477807286e03,
                    8.33573084771995e02,
                ],
                [
                    4.32859868586274e03,
                    2.83547525101554e04,
                    -3.57098738915109e03,
                    -1.58677953854261e03,
                ],
                [
                    -1.56913477807286e03,
                    -3.57098738915109e03,
                    2.82219785114869e04,
                    4.36388820680224e03,
                ],
                [
                    8.33573084771995e02,
                    -1.58677953854261e03,
                    4.36388820680224e03,
                    1.12649200195124e04,
                ],
            ]
        )
        expected_C[9:13, 9:13] = np.array(
            [
                [
                    1.12649200195124e04,
                    4.37301532545413e03,
                    -1.58246109433924e03,
                    8.28013034459200e02,
                ],
                [
                    4.37301532545413e03,
                    2.86286859810312e04,
                    -3.62427941032693e03,
                    -1.53955099233934e03,
                ],
                [
                    -1.58246109433924e03,
                    -3.62427941032693e03,
                    2.81606889518058e04,
                    4.28719512145432e03,
                ],
                [
                    8.28013034459200e02,
                    -1.53955099233934e03,
                    4.28719512145432e03,
                    5.54895519466056e03,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector_0(self):
        expected_Phi = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector_0,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array(
            [
                -2.80111500246032e-01,
                -1.37753484883726e-01,
                6.36329322852012e-02,
                -1.37917803372188e-02,
                -1.32067898068502e-11,
                9.20378069609712e-12,
                -6.30856265894091e-13,
                6.53564631108016e-13,
                -2.13256054226924e-13,
                1.09022353244805e-13,
                3.78699907150280e-16,
                4.76830797139368e-16,
                8.22999653747297e-16,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-6,
            )
        )

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array(
            [
                [
                    5.94203088797570e-09,
                    -8.84986336224903e-09,
                    3.67829829183976e-09,
                    -7.70465817566433e-10,
                ],
                [
                    -3.14908350609378e-09,
                    4.69014033596454e-09,
                    -1.94937870564851e-09,
                    4.08321875777741e-10,
                ],
                [
                    1.24307653330092e-09,
                    -1.85139688364682e-09,
                    7.69502275445856e-10,
                    -1.61181925099959e-10,
                ],
                [
                    -2.55613027492832e-10,
                    3.80701549616653e-10,
                    -1.58232257644690e-10,
                    3.31437355208682e-11,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_K,
                self.msh._stiffness_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix_0(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array(
            [
                [
                    4.54201999537039e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[3:7, 3:7] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[6:10, 6:10] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    9.08403999074079e-04,
                ],
            ]
        )
        expected_M[9:13, 9:13] = np.array(
            [
                [
                    9.08403999074079e-04,
                    3.51296859016929e-04,
                    -1.27744312369792e-04,
                    6.74206093062793e-05,
                ],
                [
                    3.51296859016929e-04,
                    2.29939762265626e-03,
                    -2.87424702832033e-04,
                    -1.27744312369793e-04,
                ],
                [
                    -1.27744312369792e-04,
                    -2.87424702832033e-04,
                    2.29939762265626e-03,
                    3.51296859016929e-04,
                ],
                [
                    6.74206093062793e-05,
                    -1.27744312369793e-04,
                    3.51296859016929e-04,
                    4.54201999537039e-04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_mass_matrix(self):
        expected_M = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_M[0:4, 0:4] = np.array(
            [
                [
                    4.93106149490779e-04,
                    3.70665591426933e-04,
                    -1.36612124184337e-04,
                    6.93487670448556e-05,
                ],
                [
                    3.70665591426933e-04,
                    2.31049595218586e-03,
                    -2.92324029174779e-04,
                    -1.26736655988384e-04,
                ],
                [
                    -1.36612124184337e-04,
                    -2.92324029174779e-04,
                    2.30161464448372e-03,
                    3.50914655035027e-04,
                ],
                [
                    6.93487670448556e-05,
                    -1.26736655988384e-04,
                    3.50914655035027e-04,
                    9.08591206911955e-04,
                ],
            ]
        )
        expected_M[3:7, 3:7] = np.array(
            [
                [
                    9.08591206911955e-04,
                    3.51306762121986e-04,
                    -1.27747853029511e-04,
                    6.74215846202641e-05,
                ],
                [
                    3.51306762121986e-04,
                    2.29940109143339e-03,
                    -2.87433168188893e-04,
                    -1.27743749730195e-04,
                ],
                [
                    -1.27747853029511e-04,
                    -2.87433168188893e-04,
                    2.29940534935140e-03,
                    3.51298555523355e-04,
                ],
                [
                    6.74215846202641e-05,
                    -1.27743749730195e-04,
                    3.51298555523355e-04,
                    9.08402597666534e-04,
                ],
            ]
        )
        expected_M[6:10, 6:10] = np.array(
            [
                [
                    9.08402597666534e-04,
                    3.51295973206610e-04,
                    -1.27743997257332e-04,
                    6.74205222159855e-05,
                ],
                [
                    3.51295973206610e-04,
                    2.29939731098759e-03,
                    -2.87423936075839e-04,
                    -1.27744363368629e-04,
                ],
                [
                    -1.27743997257332e-04,
                    -2.87423936075839e-04,
                    2.29939692522698e-03,
                    3.51296705429205e-04,
                ],
                [
                    6.74205222159855e-05,
                    -1.27744363368629e-04,
                    3.51296705429205e-04,
                    9.08404127592076e-04,
                ],
            ]
        )
        expected_M[9:13, 9:13] = np.array(
            [
                [
                    9.08404127592076e-04,
                    3.51296938316570e-04,
                    -1.27744341457428e-04,
                    6.74206165137023e-05,
                ],
                [
                    3.51296938316570e-04,
                    2.29939764527292e-03,
                    -2.87424766205142e-04,
                    -1.27744305153375e-04,
                ],
                [
                    -1.27744341457428e-04,
                    -2.87424766205142e-04,
                    2.29939768347813e-03,
                    3.51296865708465e-04,
                ],
                [
                    6.74206165137023e-05,
                    -1.27744305153375e-04,
                    3.51296865708465e-04,
                    4.54201987961274e-04,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_M,
                self.msh._mass_matrix,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._water_flux_vector_0,
                atol=1e-18,
                rtol=1e-8,
            )
        )

    def test_global_water_flux_vector(self):
        expected_flux_vector = np.array(
            [
                -6.27912575263612e-07,
                -3.23749067109424e-06,
                7.93834793181050e-07,
                -5.52802534700701e-07,
                -1.57500912307691e-08,
                7.44496099090243e-09,
                -3.38710652314809e-09,
                1.39017948682222e-09,
                -6.66126915084668e-10,
                3.05119726397311e-10,
                -1.32021543142700e-10,
                7.19771147843936e-11,
                -2.75232455299818e-11,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector, self.msh._water_flux_vector, atol=1e-18, rtol=1e-8
            )
        )

    def test_residual_heat_flux_vector(self):
        expected_Psi = np.array(
            [
                -1.03884932593656e01,
                7.72289733134868e-04,
                -1.85317264603774e-04,
                5.42579154097111e-04,
                -6.12973282016066e-05,
                2.09216072845882e-05,
                -5.38741354674435e-05,
                8.01052829395147e-06,
                -2.63800035882637e-06,
                6.55803533882696e-06,
                -1.08168589479284e-06,
                4.32376514419427e-07,
                -1.47662166505189e-06,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.000000000000000e00,
                7.017986076651120e-04,
                -1.145287768952290e-04,
                4.125206939319120e-04,
                -2.831588459382170e-05,
                8.088772250381380e-06,
                -2.609903007496110e-05,
                2.868777048090640e-06,
                -6.440809871879960e-07,
                2.103511277184400e-06,
                -3.052285750108720e-07,
                5.544206305175220e-08,
                -2.411512496318430e-07,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
                atol=1e-10,
                rtol=1e-8,
            )
        )

    def test_residual_water_flux_vector(self):
        expected_Psi = np.array(
            [
                1.79997690062171e00,
                5.30345399314693e-05,
                -6.87626037836043e-06,
                7.70470401148429e-05,
                -3.38990967129637e-06,
                6.95735562759552e-07,
                -2.69975302474217e-06,
                -8.05656478239353e-08,
                5.71974391585432e-08,
                -6.51616596802409e-08,
                3.80695523205310e-08,
                -1.89076622555655e-08,
                5.50938623607341e-08,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_water_flux_vector,
                atol=1e-9,
                rtol=1e-8,
            )
        )

    def test_void_ratio_increment_vector(self):
        expected_de = np.array(
            [
                0.000000000000000e00,
                5.303386490705350e-05,
                -6.876014443666460e-06,
                7.704655479817680e-05,
                -3.389835486987260e-06,
                6.957084530304870e-07,
                -2.699691364573810e-06,
                -8.057507731464470e-08,
                5.720090155346060e-08,
                -6.516961581851390e-08,
                3.807080467587610e-08,
                -1.890825159921080e-08,
                5.509585139559090e-08,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_de,
                self.msh._delta_void_ratio_vector,
                atol=1e-11,
                rtol=1e-8,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 4.59609128237106e-05
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a, delta=1e-10)
        self.assertEqual(self.msh._iter, 3)


if __name__ == "__main__":
    unittest.main()
