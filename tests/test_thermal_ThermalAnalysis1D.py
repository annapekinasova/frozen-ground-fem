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
            msh.add_boundary(
                ThermalBoundary1D(
                    (msh.nodes[0],),
                    (ip,),
                )
            )

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
        t0 = (7.0 / 12.0) * per
        Tavg = 5.0
        Tamp = 20.0

        def f(t):
            return Tavg + Tamp * np.cos(om * (t - t0))

        self.f = f
        self.bnd0 = ThermalBoundary1D(
            (self.msh.nodes[0],),
            bnd_type=ThermalBoundary1D.BoundaryType.temp,
            bnd_function=f,
        )
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
        self.msh.time_step = 3.024e05

    def test_initial_temp_heat_flux_vector(self):
        for tn, tn0 in zip(self.msh._temp_vector, self.msh._temp_vector_0):
            self.assertEqual(tn, 0.0)
            self.assertEqual(tn0, 0.0)
        for fx, fx0 in zip(self.msh._heat_flux_vector, self.msh._heat_flux_vector_0):
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
        for fx, fx0 in zip(self.msh._heat_flux_vector, self.msh._heat_flux_vector_0):
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
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected0))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected1))

    def test_update_heat_flux_vector(self):
        expected0 = np.zeros(self.msh.num_nodes)
        expected1 = np.ones(self.msh.num_nodes) * 0.0018217333333333300
        expected1[0] = 0.0009108666666666670
        expected1[-1] = 0.0009108666666666670
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
            z_range=(0, 100), num_elements=4, generate=True, order=1
        )
        self.msh._temp_vector_0[:] = np.array(
            [
                0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        self.msh._temp_vector[:] = np.array(
            [
                2,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        self.msh.time_step = 3.024e05
        self.msh.update_nodes()
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
                ip.vol_water_cont__0 = ip.porosity
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()
        self.msh.update_global_matrices_and_vectors()

    def test_temperature_distribution(self):
        expected_temp_int_pts = np.array(
            [
                1.5984827557301400,
                0.5015172442698560,
                -0.0901923788646684,
                -0.6098076211353320,
                -0.9479274057836310,
                -1.3520725942163700,
                -3.7189110867544700,
                -9.7810889132455400,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution(self):
        expected_temp_rate_int_pts = np.array(
            [
                5.21610538753183e-06,
                1.39765122622478e-06,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        print(actual_temp_rate_int_pts)
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -0.0760000,
                -0.0760000,
                -0.0360000,
                -0.0360000,
                -0.0280000,
                -0.0280000,
                -0.4200000,
                -0.4200000,
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
                1.000000000000000,
                0.314715929845879,
                0.113801777607921,
                0.089741864676250,
                0.074104172041942,
                0.042882888566470,
                0.025322726744343,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.00000000000000000,
                0.00000000000000000,
                1.96985868037600000,
                0.37676651903183400,
                0.24895666952857000,
                0.17754007257781200,
                0.06672422855669150,
                0.02583496685516250,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        print(actual_vol_water_cont_temp_gradient_int_pts)
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                1.45582271775933e-11,
                1.82156369719347e-12,
                3.66374860619567e-13,
                7.27534454355483e-14,
                8.43966555642625e-17,
                2.48452701591120e-27,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        print(actual_water_flux_int_pts)
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                1.94419643704324,
                1.94419643704324,
                2.48085630059944,
                2.66463955659925,
                2.68754164945741,
                2.70253225701445,
                2.73271219424962,
                2.74983450612514,
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

    def test_global_heat_flow_matrix(self):
        expected_H = np.array(
            [
                [
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675420790767840,
                    -0.0954281261856329,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954281261856329,
                    0.1953921854661540,
                    -0.0999640592805206,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999640592805206,
                    0.2016431146839040,
                    -0.1016790554033840,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790554033840,
                    0.1016790554033840,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.array(
            [
                [
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    4.75157069027845e09,
                    1.49253092352162e09,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    1.49253092352162e09,
                    1.87400276780934e09,
                    2.77995607535567e08,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    2.77995607535567e08,
                    6.55976597222048e08,
                    6.67084599390764e07,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    6.67084599390764e07,
                    8.85939210208664e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_heat_flux_vector(self):
        expected_Phi = np.array(
            [
                0.000000000000000e00,
                2.161787820306900e-05,
                8.652932548588540e-06,
                1.924174706879560e-07,
                3.790595411752880e-10,
            ]
        )
        print(self.msh._heat_flux_vector)
        self.assertTrue(
            np.allclose(
                expected_Phi,
                self.msh._heat_flux_vector,
                atol=1e-15,
                rtol=1e-6,
            )
        )


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
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix_0, expected0))
        self.assertTrue(np.allclose(self.msh._heat_storage_matrix, expected1))

    def test_update_heat_flux_vector(self):
        expected0 = np.zeros(self.msh.num_nodes)
        expected1 = np.ones(self.msh.num_nodes) * 0.00068315
        expected1[3::3] = 0.000455433333333333
        expected1[0] = 0.000227716666666667
        expected1[-1] = 0.000227716666666667
        self.msh.update_heat_flux_vector()
        self.assertTrue(np.allclose(self.msh._heat_flux_vector_0, expected0))
        self.assertTrue(
            np.allclose(self.msh._heat_flux_vector, expected1, rtol=1e-13, atol=1e-16)
        )


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
        self.msh._temp_vector[:] = np.array(
            [
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
            ]
        )
        self.msh._temp_rate_vector[:] = np.array(
            [
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
            ]
        )
        self.msh.time_step = 1e2
        self.msh.update_nodes()
        for e in self.msh.elements:
            for ip in e.int_pts:
                ip.material = self.mtl
                ip.void_ratio = 0.35
                ip.void_ratio_0 = 0.3
                ip.tot_stress = 1.2e5
                ip.vol_water_cont__0 = ip.porosity
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()

    def test_temperature_distribution(self):
        expected_temp_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution(self):
        expected_temp_rate_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        print(actual_temp_rate_int_pts)
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
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
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
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

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                1.8071264681e-15,
                5.0412534775e-23,
                1.5503621786e-28,
                -1.7186479754e-27,
                -3.0723442020e-24,
                -3.9524301404e-22,
                -5.4976497868e-18,
                -1.8328498927e-12,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                1.0956257621e-10,
                1.5160234947e-11,
                6.5849524657e-13,
                -6.1857084254e-12,
                -2.6451092292e-11,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )


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
            self.assertAlmostEqual(nd.temp, 2.0 * (k + 1))
            self.assertAlmostEqual(nd.temp_rate, 4.0 * (k + 1))

    def test_repeat_update_nodes(self):
        self.msh.update_nodes()
        for k, nd in enumerate(self.msh.nodes):
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
            z_range=(0, 100), num_elements=4, generate=True, order=1
        )
        initial_temp_vector = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        initial_temp_rate_vector = np.array(
            [
                0.05,
                0.02,
                0.01,
                -0.08,
                -0.05,
            ]
        )
        for nd, T0, dTdt0 in zip(
            self.msh.nodes,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_vector = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_vector, actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector, self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector, self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                0.0211324865405187,
                0.0788675134594813,
                -0.0901923788646684,
                -0.6098076211353320,
                -0.9479274057836310,
                -1.3520725942163700,
                -3.7189110867544700,
                -9.7810889132455400,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                0.05,
                0.02,
                0.01,
                -0.08,
                -0.05,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_rate_vector, actual_temp_rate_nodes))
        self.assertTrue(
            np.allclose(expected_temp_rate_vector, self.msh._temp_rate_vector)
        )

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                0.04366025403784440,
                0.02633974596215560,
                0.01788675134594810,
                0.01211324865405190,
                -0.00901923788646684,
                -0.06098076211353320,
                -0.07366025403784440,
                -0.05633974596215560,
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
                0.0040000,
                0.0040000,
                -0.0360000,
                -0.0360000,
                -0.0280000,
                -0.0280000,
                -0.4200000,
                -0.4200000,
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
                1.000000000000000,
                0.314715929845879,
                0.113801777607921,
                0.089741864676250,
                0.074104172041942,
                0.042882888566470,
                0.025322726744343,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
            ]
        )
        actual_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                ip.vol_water_cont_temp_gradient
                for e in self.msh.elements
                for ip in e.int_pts
            ]
        )
        print(actual_vol_water_cont_temp_gradient_int_pts)
        self.assertTrue(
            np.allclose(
                actual_vol_water_cont_temp_gradient_int_pts,
                expected_vol_water_cont_temp_gradient_int_pts,
            )
        )

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                0.0000000000e00,
                0.0000000000e00,
                -4.8910645169e-12,
                -5.5518497692e-13,
                8.3577053338e-13,
                1.7708810805e-13,
                2.0670411271e-16,
                6.0318175808e-27,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                1.94419643704324,
                1.94419643704324,
                2.48085630059944,
                2.66463955659925,
                2.68754164945741,
                2.70253225701445,
                2.73271219424962,
                2.74983450612514,
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

    def test_global_heat_flow_matrix(self):
        expected_H = np.array(
            [
                [
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675420790767840,
                    -0.0954281261856329,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954281261856329,
                    0.1953921854661540,
                    -0.0999640592805206,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999640592805206,
                    0.2016431146839040,
                    -0.1016790554033840,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790554033840,
                    0.1016790554033840,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.array(
            [
                [
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    3.89011555173310e07,
                    8.63025666982303e06,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.63025666982303e06,
                    3.34542102448499e07,
                    8.29817132739774e06,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.29817132739774e06,
                    3.29568618779144e07,
                    8.17817064124763e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.17817064124763e06,
                    1.63181792594571e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                -0.000000000000000e00,
                -7.240998977095220e-06,
                -1.693636195537800e-06,
                4.516088940272450e-07,
                6.874586265312840e-02,
            ]
        )
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector))


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
            z_range=(0, 100), num_elements=4, generate=True, order=1
        )
        initial_temp_vector = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        for nd, T0 in zip(self.msh.nodes, initial_temp_vector):
            nd.temp = T0
        for e in self.msh.elements:
            e.assign_material(self.mtl)
            for ip in e.int_pts:
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
        self.msh.time_step = 3.024e05
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5 + 3.024e5)

    def test_iteration_variables(self):
        self.assertEqual(self.msh._eps_a, 1.0)
        self.assertEqual(self.msh._iter, 0)

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        expected_temp_vector_1 = np.array(
            [
                2.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_vector_1, actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector_0, self.msh._temp_vector_0))
        self.assertTrue(np.allclose(expected_temp_vector_1, self.msh._temp_vector))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                1.5984827557301400,
                0.5015172442698560,
                -0.0901923788646684,
                -0.6098076211353320,
                -0.9479274057836310,
                -1.3520725942163700,
                -3.7189110867544700,
                -9.7810889132455400,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                6.61375661375661e-06,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        print(actual_temp_rate_nodes)
        self.assertTrue(np.allclose(expected_temp_rate_vector, actual_temp_rate_nodes))
        self.assertTrue(
            np.allclose(expected_temp_rate_vector, self.msh._temp_rate_vector)
        )

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                5.21610538753183e-06,
                1.39765122622478e-06,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
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
                -0.0760000,
                -0.0760000,
                -0.0360000,
                -0.0360000,
                -0.0280000,
                -0.0280000,
                -0.4200000,
                -0.4200000,
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
                1.000000000000000,
                0.314715929845879,
                0.113801777607921,
                0.089741864676250,
                0.074104172041942,
                0.042882888566470,
                0.025322726744343,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
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

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                1.45582271775933e-11,
                1.82156369719347e-12,
                3.66374860619567e-13,
                7.27534454355483e-14,
                8.43966555642625e-17,
                2.48452701591120e-27,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                1.94419643704324,
                1.94419643704324,
                2.48085630059944,
                2.66463955659925,
                2.68754164945741,
                2.70253225701445,
                2.73271219424962,
                2.74983450612514,
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

    def test_global_heat_flow_matrix(self):
        expected_H = np.array(
            [
                [
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675420790767840,
                    -0.0954281261856329,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954281261856329,
                    0.1953921854661540,
                    -0.0999640592805206,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999640592805206,
                    0.2016431146839040,
                    -0.1016790554033840,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790554033840,
                    0.1016790554033840,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.array(
            [
                [
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    3.89011555173310e07,
                    8.63025666982303e06,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.63025666982303e06,
                    3.34542102448499e07,
                    8.29817132739774e06,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.29817132739774e06,
                    3.29568618779144e07,
                    8.17817064124763e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.17817064124763e06,
                    1.63181792594571e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                0.000000000000000e00,
                2.161787820306900e-05,
                8.652932548588540e-06,
                1.924174706879560e-07,
                6.874586303218800e-02,
            ]
        )
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector_0))
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector))


class TestUpdateGlobalMatricesLinear(unittest.TestCase):
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
            z_range=(0, 100), num_elements=4, generate=True, order=1
        )
        initial_temp_vector = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        for nd, T0 in zip(self.msh.nodes, initial_temp_vector):
            nd.temp = T0
        for e in self.msh.elements:
            e.assign_material(self.mtl)
            for ip in e.int_pts:
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
        self.msh.time_step = 3.024e05
        self.msh.initialize_time_step()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        expected_temp_vector = np.array(
            [
                2.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_vector, actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector, self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector_0, self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
                1.5984827557301400,
                0.5015172442698550,
                -0.0901923788646683,
                -0.6098076211353320,
                -0.9479274057836310,
                -1.3520725942163700,
                -3.7189110867544600,
                -9.7810889132455400,
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                6.61375661375661e-06,
                0,
                0,
                0,
                0,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                actual_temp_rate_nodes,
                atol=1e-13,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                self.msh._temp_rate_vector,
                atol=1e-13,
            )
        )

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                5.21610538753183e-06,
                1.39765122622478e-06,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_temp_rate_int_pts,
                expected_temp_rate_int_pts,
                atol=1e-13,
            )
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -0.0760000,
                -0.0760000,
                -0.0360000,
                -0.0360000,
                -0.0280000,
                -0.0280000,
                -0.4200000,
                -0.4200000,
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
                1.000000000000000,
                0.314715929845879,
                0.113801777607921,
                0.089741864676250,
                0.074104172041942,
                0.042882888566470,
                0.025322726744343,
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
            [
                0.000000000000000000,
                0.000000000000000000,
                0.000000000000000000,
                0.000000000000000000,
                0.000000000000000000,
                0.000000000000000000,
                0.000000000000000000,
                0.000000000000000000,
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

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                0.00000000000000e00,
                0.00000000000000e00,
                1.45582271775933e-11,
                1.82156369719347e-12,
                3.66374860619567e-13,
                7.27534454355483e-14,
                8.43966555642625e-17,
                2.48452701591120e-27,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
                1.94419643704324,
                1.94419643704324,
                2.48085630059944,
                2.66463955659925,
                2.68754164945741,
                2.70253225701445,
                2.73271219424962,
                2.74983450612514,
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_thrm_cond_int_pts, expected_thrm_cond_int_pts)
        )

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array(
            [
                [
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675420790767840,
                    -0.0954281261856329,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954281261856329,
                    0.1953921854661540,
                    -0.0999640592805206,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999640592805206,
                    0.2016431146839040,
                    -0.1016790554033840,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790554033840,
                    0.1016790554033840,
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
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675420790767840,
                    -0.0954281261856329,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954281261856329,
                    0.1953921854661540,
                    -0.0999640592805206,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999640592805206,
                    0.2016431146839040,
                    -0.1016790554033840,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790554033840,
                    0.1016790554033840,
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
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    3.89011555173310e07,
                    8.63025666982303e06,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.63025666982303e06,
                    3.34542102448499e07,
                    8.29817132739774e06,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.29817132739774e06,
                    3.29568618779144e07,
                    8.17817064124763e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.17817064124763e06,
                    1.63181792594571e07,
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
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    3.89011555173310e07,
                    8.63025666982303e06,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.63025666982303e06,
                    3.34542102448499e07,
                    8.29817132739774e06,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.29817132739774e06,
                    3.29568618779144e07,
                    8.17817064124763e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.17817064124763e06,
                    1.63181792594571e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array(
            [
                -0.000000000000000e00,
                2.161787820306900e-05,
                8.652932548588540e-06,
                1.924174706879560e-07,
                6.874586303218800e-02,
            ]
        )
        self.assertTrue(
            np.allclose(expected_flux_vector_0, self.msh._heat_flux_vector_0)
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                0.000000000000000e00,
                2.161787820306900e-05,
                8.652932548588540e-06,
                1.924174706879560e-07,
                6.874586303218800e-02,
            ]
        )
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector))


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
            z_range=(0, 100), num_elements=4, generate=True, order=1
        )
        initial_temp_vector = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        initial_temp_rate_vector = np.array(
            [
                0.05,
                0.02,
                0.01,
                -0.08,
                -0.05,
            ]
        )
        for nd, T0, dTdt0 in zip(
            self.msh.nodes,
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
        self.msh.time_step = 3.024e05
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array(
            [
                2.0,
                0.6,
                -0.2,
                -0.8,
                -6,
            ]
        )
        self.msh._temp_rate_vector[:] = np.array(
            [
                0,
                500,
                600,
                700,
                6000,
            ]
        )
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()
        self.msh.update_global_matrices_and_vectors()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        expected_temp_vector = np.array(
            [
                2.0000000000000000,
                0.0988263131322971,
                -0.7971699198965330,
                -1.5126232123463600,
                -11.9736027887043000,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_vector,
                actual_temp_nodes,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_vector_0,
                self.msh._temp_vector_0,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_vector,
                self.msh._temp_vector,
            )
        )

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                6.61375661375661e-06,
                -3.88123964187482e-09,
                9.35873050088320e-09,
                -4.17434270713069e-08,
                8.72923653959047e-08,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                actual_temp_rate_nodes,
                atol=1e-12,
                rtol=1e-3,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                self.msh._temp_rate_vector,
                atol=1e-12,
                rtol=1e-3,
            )
        )

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array(
            [
                [
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675420790767840,
                    -0.0954281261856329,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954281261856329,
                    0.1953921854661540,
                    -0.0999640592805206,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999640592805206,
                    0.2016431146839040,
                    -0.1016790554033840,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790554033840,
                    0.1016790554033840,
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
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675482489237170,
                    -0.0954342960325658,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954342960325658,
                    0.1954036559952800,
                    -0.0999693599627141,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999693599627141,
                    0.2016484400499960,
                    -0.1016790800872820,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790800872820,
                    0.1016790800872820,
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
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    3.89011555173310e07,
                    8.63025666982303e06,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.63025666982303e06,
                    3.34542102448499e07,
                    8.29817132739774e06,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.29817132739774e06,
                    3.29568618779144e07,
                    8.17817064124763e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.17817064124763e06,
                    1.63181792594571e07,
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
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    1.14790024626751e09,
                    3.21101921370035e08,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    3.21101921370035e08,
                    2.06835155994880e08,
                    2.14800761090158e07,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    2.14800761090158e07,
                    5.70746434252292e07,
                    9.43514558536205e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.43514558536205e06,
                    1.74625389157076e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array(
            [
                -0.000000000000000e00,
                -7.240998977095210e-06,
                -1.693636195537800e-06,
                4.516088940272450e-07,
                6.874586265312840e-02,
            ]
        )
        self.assertTrue(
            np.allclose(expected_flux_vector_0, self.msh._heat_flux_vector_0)
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                0.000000000000000e00,
                2.143145921199510e-05,
                8.271535830408110e-06,
                2.236869459407050e-07,
                6.874523161026890e-02,
            ]
        )
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector))

    def test_global_residual_vector(self):
        expected_Psi = np.array(
            [
                -2.00010331808334e00,
                -5.01130348395403e-01,
                -5.97300376134295e-01,
                -7.10765507610895e-01,
                -5.97904836728674e00,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.00000000000000e00,
                -5.01173686867703e-01,
                -5.97169919896533e-01,
                -7.12623212346363e-01,
                -5.97360278870428e00,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 4.9481302578941e-01
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
            z_range=(0, 100), num_elements=4, generate=True, order=1
        )
        initial_temp_vector = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        initial_temp_rate_vector = np.array(
            [
                0.05,
                0.02,
                0.01,
                -0.08,
                -0.05,
            ]
        )
        for nd, T0, dTdt0 in zip(
            self.msh.nodes,
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
        self.msh.time_step = 3.024e05
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array(
            [
                2.0,
                0.6,
                -0.2,
                -0.8,
                -6,
            ]
        )
        self.msh._temp_rate_vector[:] = np.array(
            [
                0,
                500,
                600,
                700,
                6000,
            ]
        )
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()
        self.msh.update_global_matrices_and_vectors()
        self.msh.iterative_correction_step()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array(
            [
                0.0,
                0.1,
                -0.8,
                -1.5,
                -12,
            ]
        )
        expected_temp_vector = np.array(
            [
                2.0000000000000000,
                0.0986658199546282,
                -0.7965229034856440,
                -1.5139885335425800,
                -11.9724600868138000,
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_vector,
                actual_temp_nodes,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_vector_0,
                self.msh._temp_vector_0,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_vector,
                self.msh._temp_vector,
            )
        )

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                6.61375661375661e-06,
                -4.41197104950990e-09,
                1.14983350342468e-08,
                -4.62583781170062e-08,
                9.10711414888197e-08,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                actual_temp_rate_nodes,
                atol=1e-16,
                rtol=1e-10,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                self.msh._temp_rate_vector,
                atol=1e-16,
                rtol=1e-10,
            )
        )

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.array(
            [
                [
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675420790767840,
                    -0.0954281261856329,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954281261856329,
                    0.1953921854661540,
                    -0.0999640592805206,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999640592805206,
                    0.2016431146839040,
                    -0.1016790554033840,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016790554033840,
                    0.1016790554033840,
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
                    0.0721139528911510,
                    -0.0721139528911510,
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    -0.0721139528911510,
                    0.1675471042926980,
                    -0.0954331514015465,
                    0.0000000000000000,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    -0.0954331514015465,
                    0.1954028010121700,
                    -0.0999696496106236,
                    0.0000000000000000,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.0999696496106236,
                    0.2016488078173110,
                    -0.1016791582066870,
                ],
                [
                    0.0000000000000000,
                    0.0000000000000000,
                    0.0000000000000000,
                    -0.1016791582066870,
                    0.1016791582066870,
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
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    3.89011555173310e07,
                    8.63025666982303e06,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    8.63025666982303e06,
                    3.34542102448499e07,
                    8.29817132739774e06,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.29817132739774e06,
                    3.29568618779144e07,
                    8.17817064124763e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    8.17817064124763e06,
                    1.63181792594571e07,
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
                    2.12040123456790e07,
                    1.06020061728395e07,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    1.06020061728395e07,
                    1.14799363337692e09,
                    3.21136209426181e08,
                    0.00000000000000e00,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    3.21136209426181e08,
                    2.06883882586324e08,
                    2.14790000604973e07,
                    0.00000000000000e00,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    2.14790000604973e07,
                    5.70646987491469e07,
                    9.43497504608584e06,
                ],
                [
                    0.00000000000000e00,
                    0.00000000000000e00,
                    0.00000000000000e00,
                    9.43497504608584e06,
                    1.74625357643405e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array(
            [
                -0.000000000000000e00,
                -7.240998977095210e-06,
                -1.693636195537800e-06,
                4.516088940272450e-07,
                6.874586265312840e-02,
            ]
        )
        self.assertTrue(
            np.allclose(expected_flux_vector_0, self.msh._heat_flux_vector_0)
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                0.000000000000000e00,
                2.133070477819620e-05,
                8.181590920372290e-06,
                2.187351973225760e-07,
                6.874521027822230e-02,
            ]
        )
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector))

    def test_global_residual_vector(self):
        expected_Psi = np.array(
            [
                -2.00025900344127e00,
                -1.60722721278623e-04,
                6.47788749226980e-04,
                -1.36736823325139e-03,
                1.14601596360676e-03,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.00000000000000e00,
                -1.60493177668852e-04,
                6.47016410889105e-04,
                -1.36532119621939e-03,
                1.14270189049772e-03,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 1.5508312528406e-04
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
        initial_temp_vector = np.array(
            [
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
            ]
        )
        initial_temp_rate_vector = np.array(
            [
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
            ]
        )
        for nd, T0, dTdt0 in zip(
            self.msh.nodes,
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
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[0].flatten()))
        self.assertTrue(np.all(expected_free_vec == self.msh._free_arr[1]))

    def test_temperature_distribution_nodes(self):
        expected_temp_vector = np.array(
            [
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
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_vector, actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector, self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector, self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_temp_int_pts,
                expected_temp_int_pts,
            )
        )

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
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
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                actual_temp_rate_nodes,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                self.msh._temp_rate_vector,
            )
        )

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_temp_rate_int_pts,
                expected_temp_rate_int_pts,
            )
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_gradient_int_pts = np.array(
            [ip.temp_gradient for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_temp_gradient_int_pts,
                expected_temp_gradient_int_pts,
            )
        )

    def test_deg_sat_water_distribution(self):
        expected_deg_sat_water_int_pts = np.array(
            [
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
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_deg_sat_water_int_pts,
                expected_deg_sat_water_int_pts,
            )
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.zeros(
            self.msh.num_elements * 5
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

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                1.8071264681e-15,
                5.0412534775e-23,
                1.5503621786e-28,
                -1.7186479754e-27,
                -3.0723442020e-24,
                -3.9524301404e-22,
                -5.4976497868e-18,
                -1.8328498927e-12,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                0.0000000000e00,
                1.0956257621e-10,
                1.5160234947e-11,
                6.5849524657e-13,
                -6.1857084254e-12,
                -2.6451092292e-11,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
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

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3754127694384990,
                    -0.4791349492121190,
                    0.1366244795670250,
                    -0.0329022997934044,
                ],
                [
                    -0.4791349492121190,
                    1.0974938412970900,
                    -0.7555601286108480,
                    0.1372012365258810,
                ],
                [
                    0.1366244795670250,
                    -0.7555601286108480,
                    1.1003657454416000,
                    -0.4814300963977770,
                ],
                [
                    -0.0329022997934044,
                    0.1372012365258810,
                    -0.4814300963977770,
                    0.3771311596653000,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742278648957170,
                    -0.4704043073588020,
                    0.1245486051912600,
                    -0.0283721627281755,
                ],
                [
                    -0.4704043073588020,
                    1.0458021981448800,
                    -0.6891083023287820,
                    0.1137104115427070,
                ],
                [
                    0.1245486051912600,
                    -0.6891083023287820,
                    0.9190461742111960,
                    -0.3544864770736730,
                ],
                [
                    -0.0283721627281755,
                    0.1137104115427070,
                    -0.3544864770736730,
                    0.2691482282591420,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972590,
                    -0.3407384274106880,
                    0.0973538364030538,
                    -0.0234370346896240,
                ],
                [
                    -0.3407384274106880,
                    0.7788306912244300,
                    -0.5354461002167960,
                    0.0973538364030537,
                ],
                [
                    0.0973538364030538,
                    -0.5354461002167960,
                    0.7788306912244310,
                    -0.3407384274106880,
                ],
                [
                    -0.0234370346896240,
                    0.0973538364030537,
                    -0.3407384274106880,
                    0.2668216256972590,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3292202434390340,
                    -0.4133999362112650,
                    0.1112498768967810,
                    -0.0270701841245502,
                ],
                [
                    -0.4133999362112650,
                    0.9868437905990680,
                    -0.6971611631648800,
                    0.1237173087770760,
                ],
                [
                    0.1112498768967810,
                    -0.6971611631648800,
                    1.0421402435851100,
                    -0.4562289573170110,
                ],
                [
                    -0.0270701841245502,
                    0.1237173087770760,
                    -0.4562289573170110,
                    0.3595818326644850,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    3.74901860610780e06,
                    2.89593744482105e06,
                    -1.05358398557321e06,
                    5.54643193641815e05,
                ],
                [
                    2.89593744482105e06,
                    1.88936621803465e07,
                    -2.36630842102390e06,
                    -1.04792564897764e06,
                ],
                [
                    -1.05358398557321e06,
                    -2.36630842102390e06,
                    1.88757400422500e07,
                    2.88462077162992e06,
                ],
                [
                    5.54643193641815e05,
                    -1.04792564897764e06,
                    2.88462077162992e06,
                    3.73110189762532e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    3.74548978664947e06,
                    2.83529217357612e06,
                    -8.63799618191817e05,
                    6.29634934688451e05,
                ],
                [
                    2.83529217357612e06,
                    1.92531430053028e07,
                    -3.32307881157745e06,
                    -1.41571111930997e06,
                ],
                [
                    -8.63799618191817e05,
                    -3.32307881157745e06,
                    2.30168957941050e07,
                    3.93911517581243e06,
                ],
                [
                    6.29634934688451e05,
                    -1.41571111930997e06,
                    3.93911517581243e06,
                    4.82119852843649e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329806e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442681e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857143e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857143e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442681e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329806e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    4.26389587430359e06,
                    3.18874298722127e06,
                    -1.17518017569044e06,
                    5.93504826831539e05,
                ],
                [
                    3.18874298722127e06,
                    1.95951789440892e07,
                    -2.51514790752116e06,
                    -1.07640022713129e06,
                ],
                [
                    -1.17518017569044e06,
                    -2.51514790752116e06,
                    1.94626651593996e07,
                    2.99118309010297e06,
                ],
                [
                    5.93504826831539e05,
                    -1.07640022713129e06,
                    2.99118309010297e06,
                    3.89099624195036e06,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                1.89850411575195e-08,
                9.32804654257637e-09,
                -4.31099786843920e-09,
                -1.42748247257048e-06,
                1.28559187259822e-05,
                1.28558305047007e-05,
                -1.42842430458346e-06,
                -0.00000000000000e00,
                -0.00000000000000e00,
                1.10932858948307e-04,
                7.12664676651715e-05,
                -1.57821060328194e-05,
                6.53002632662146e-02,
            ]
        )
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector))


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
        initial_temp_vector = np.array(
            [
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
            ]
        )
        for nd, T0 in zip(self.msh.nodes, initial_temp_vector):
            nd.temp = T0
        for e in self.msh.elements:
            e.assign_material(self.mtl)
            for ip in e.int_pts:
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
        self.msh.time_step = 3.024e05
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5 + 3.024e5)

    def test_iteration_variables(self):
        self.assertEqual(self.msh._eps_a, 1.0)
        self.assertEqual(self.msh._iter, 0)

    def test_temperature_distribution_nodes(self):
        expected_temp_vector = np.array(
            [
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
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_vector, actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector, self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector, self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
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
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_rate_vector, actual_temp_rate_nodes))
        self.assertTrue(
            np.allclose(expected_temp_rate_vector, self.msh._temp_rate_vector)
        )

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
            [
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
                0.000000000000000,
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
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.zeros(
            self.msh.num_elements * 5
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

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                7.56791701724464e-16,
                2.05575577309513e-23,
                6.25866187204993e-29,
                -6.94813227511036e-28,
                -1.24976717946199e-24,
                -1.61597269270287e-22,
                -2.27855717449883e-18,
                -7.90484406717873e-13,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                5.44584145530179e-11,
                6.83027763154097e-12,
                2.92100583869216e-13,
                -2.76336220246488e-12,
                -1.20819543771439e-11,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
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

    def test_global_heat_flow_matrix(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3754127694384990,
                    -0.4791349492121190,
                    0.1366244795670250,
                    -0.0329022997934044,
                ],
                [
                    -0.4791349492121190,
                    1.0974938412970900,
                    -0.7555601286108480,
                    0.1372012365258810,
                ],
                [
                    0.1366244795670250,
                    -0.7555601286108480,
                    1.1003657454416000,
                    -0.4814300963977770,
                ],
                [
                    -0.0329022997934044,
                    0.1372012365258810,
                    -0.4814300963977770,
                    0.3771311596653000,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742278648957170,
                    -0.4704043073588020,
                    0.1245486051912600,
                    -0.0283721627281755,
                ],
                [
                    -0.4704043073588020,
                    1.0458021981448800,
                    -0.6891083023287820,
                    0.1137104115427070,
                ],
                [
                    0.1245486051912600,
                    -0.6891083023287820,
                    0.9190461742111960,
                    -0.3544864770736730,
                ],
                [
                    -0.0283721627281755,
                    0.1137104115427070,
                    -0.3544864770736730,
                    0.2691482282591420,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972590,
                    -0.3407384274106880,
                    0.0973538364030538,
                    -0.0234370346896240,
                ],
                [
                    -0.3407384274106880,
                    0.7788306912244300,
                    -0.5354461002167960,
                    0.0973538364030537,
                ],
                [
                    0.0973538364030538,
                    -0.5354461002167960,
                    0.7788306912244310,
                    -0.3407384274106880,
                ],
                [
                    -0.0234370346896240,
                    0.0973538364030537,
                    -0.3407384274106880,
                    0.2668216256972590,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3292202434390340,
                    -0.4133999362112650,
                    0.1112498768967810,
                    -0.0270701841245502,
                ],
                [
                    -0.4133999362112650,
                    0.9868437905990680,
                    -0.6971611631648800,
                    0.1237173087770760,
                ],
                [
                    0.1112498768967810,
                    -0.6971611631648800,
                    1.0421402435851100,
                    -0.4562289573170110,
                ],
                [
                    -0.0270701841245502,
                    0.1237173087770760,
                    -0.4562289573170110,
                    0.3595818326644850,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_H,
                self.msh._heat_flow_matrix_0,
            )
        )

    def test_global_heat_storage_matrix(self):
        expected_C = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    3.74901860610780e06,
                    2.89593744482105e06,
                    -1.05358398557321e06,
                    5.54643193641815e05,
                ],
                [
                    2.89593744482105e06,
                    1.88936621803465e07,
                    -2.36630842102390e06,
                    -1.04792564897764e06,
                ],
                [
                    -1.05358398557321e06,
                    -2.36630842102390e06,
                    1.88757400422500e07,
                    2.88462077162992e06,
                ],
                [
                    5.54643193641815e05,
                    -1.04792564897764e06,
                    2.88462077162992e06,
                    3.73110189762532e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    3.74548978664947e06,
                    2.83529217357612e06,
                    -8.63799618191817e05,
                    6.29634934688451e05,
                ],
                [
                    2.83529217357612e06,
                    1.92531430053028e07,
                    -3.32307881157745e06,
                    -1.41571111930997e06,
                ],
                [
                    -8.63799618191817e05,
                    -3.32307881157745e06,
                    2.30168957941050e07,
                    3.93911517581243e06,
                ],
                [
                    6.29634934688451e05,
                    -1.41571111930997e06,
                    3.93911517581243e06,
                    4.82119852843649e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329806e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442681e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857143e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857143e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442681e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329806e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    4.26389587430359e06,
                    3.18874298722127e06,
                    -1.17518017569044e06,
                    5.93504826831539e05,
                ],
                [
                    3.18874298722127e06,
                    1.95951789440892e07,
                    -2.51514790752116e06,
                    -1.07640022713129e06,
                ],
                [
                    -1.17518017569044e06,
                    -2.51514790752116e06,
                    1.94626651593996e07,
                    2.99118309010297e06,
                ],
                [
                    5.93504826831539e05,
                    -1.07640022713129e06,
                    2.99118309010297e06,
                    3.89099624195036e06,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix_0,
            )
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                7.95058998665675e-09,
                3.90641624893011e-09,
                -1.80536750828495e-09,
                -6.15666554867237e-07,
                5.54458994432774e-06,
                5.54455338012299e-06,
                -6.16060944960366e-07,
                -0.00000000000000e00,
                -0.00000000000000e00,
                5.49294982437498e-05,
                3.45212307107206e-05,
                -8.20929073888648e-06,
                6.52879669583503e-02,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._heat_flux_vector,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._heat_flux_vector_0,
            )
        )


class TestUpdateGlobalMatricesCubic(unittest.TestCase):
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
        initial_temp_vector = np.array(
            [
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
            ]
        )
        for nd, T0 in zip(self.msh.nodes, initial_temp_vector):
            nd.temp = T0
        for e in self.msh.elements:
            e.assign_material(self.mtl)
            for ip in e.int_pts:
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
        self.msh.time_step = 3.024e05
        self.msh.initialize_time_step()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array(
            [
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
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_vector_0, actual_temp_nodes))
        self.assertTrue(np.allclose(expected_temp_vector_0, self.msh._temp_vector))
        self.assertTrue(np.allclose(expected_temp_vector_0, self.msh._temp_vector_0))

    def test_temperature_distribution_int_pts(self):
        expected_temp_int_pts = np.array(
            [
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
            ]
        )
        actual_temp_int_pts = np.array(
            [ip.temp for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(np.allclose(actual_temp_int_pts, expected_temp_int_pts))

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
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
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(np.allclose(expected_temp_rate_vector, actual_temp_rate_nodes))
        self.assertTrue(
            np.allclose(expected_temp_rate_vector, self.msh._temp_rate_vector)
        )

    def test_temperature_rate_distribution_int_pts(self):
        expected_temp_rate_int_pts = np.array(
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
        actual_temp_rate_int_pts = np.array(
            [ip.temp_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_temp_rate_int_pts, expected_temp_rate_int_pts)
        )

    def test_temperature_gradient_distribution(self):
        expected_temp_gradient_int_pts = np.array(
            [
                -1.15093426984199000,
                -0.70037674599536100,
                -0.15129838219301400,
                0.26620714324994600,
                0.47571152668667600,
                0.52108465163989700,
                0.51343382134771900,
                0.43315319751339600,
                0.27077898886022600,
                0.11272541074530500,
                0.07267706952531840,
                -0.02454456350280680,
                -0.11231442240249500,
                -0.13519566470899900,
                -0.11353558171062700,
                -0.10632645781291500,
                -0.06254104067706060,
                -0.00664052813361892,
                0.03949323637822940,
                0.06538510258089550,
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
            ]
        )
        actual_deg_sat_water_int_pts = np.array(
            [ip.deg_sat_water for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_deg_sat_water_int_pts, expected_deg_sat_water_int_pts)
        )

    def test_vol_water_cont_temp_gradient_distribution(self):
        expected_vol_water_cont_temp_gradient_int_pts = np.array(
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

    def test_water_flux_distribution(self):
        expected_water_flux_int_pts = np.array(
            [
                7.56791701724464e-16,
                2.05575577309513e-23,
                6.25866187204993e-29,
                -6.94813227511036e-28,
                -1.24976717946199e-24,
                -1.61597269270287e-22,
                -2.27855717449883e-18,
                -7.90484406717873e-13,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                0.00000000000000e00,
                5.44584145530179e-11,
                6.83027763154097e-12,
                2.92100583869216e-13,
                -2.76336220246488e-12,
                -1.20819543771439e-11,
            ]
        )
        actual_water_flux_int_pts = np.array(
            [ip.water_flux_rate for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(
                actual_water_flux_int_pts, expected_water_flux_int_pts, atol=1e-30
            )
        )

    def test_thrm_cond_distribution(self):
        expected_thrm_cond_int_pts = np.array(
            [
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
            ]
        )
        actual_thrm_cond_int_pts = np.array(
            [ip.thrm_cond for e in self.msh.elements for ip in e.int_pts]
        )
        self.assertTrue(
            np.allclose(actual_thrm_cond_int_pts, expected_thrm_cond_int_pts)
        )

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3754127694384990,
                    -0.4791349492121190,
                    0.1366244795670250,
                    -0.0329022997934044,
                ],
                [
                    -0.4791349492121190,
                    1.0974938412970900,
                    -0.7555601286108480,
                    0.1372012365258810,
                ],
                [
                    0.1366244795670250,
                    -0.7555601286108480,
                    1.1003657454416000,
                    -0.4814300963977770,
                ],
                [
                    -0.0329022997934044,
                    0.1372012365258810,
                    -0.4814300963977770,
                    0.3771311596653000,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742278648957170,
                    -0.4704043073588020,
                    0.1245486051912600,
                    -0.0283721627281755,
                ],
                [
                    -0.4704043073588020,
                    1.0458021981448800,
                    -0.6891083023287820,
                    0.1137104115427070,
                ],
                [
                    0.1245486051912600,
                    -0.6891083023287820,
                    0.9190461742111960,
                    -0.3544864770736730,
                ],
                [
                    -0.0283721627281755,
                    0.1137104115427070,
                    -0.3544864770736730,
                    0.2691482282591420,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972590,
                    -0.3407384274106880,
                    0.0973538364030538,
                    -0.0234370346896240,
                ],
                [
                    -0.3407384274106880,
                    0.7788306912244300,
                    -0.5354461002167960,
                    0.0973538364030537,
                ],
                [
                    0.0973538364030538,
                    -0.5354461002167960,
                    0.7788306912244310,
                    -0.3407384274106880,
                ],
                [
                    -0.0234370346896240,
                    0.0973538364030537,
                    -0.3407384274106880,
                    0.2668216256972590,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3292202434390340,
                    -0.4133999362112650,
                    0.1112498768967810,
                    -0.0270701841245502,
                ],
                [
                    -0.4133999362112650,
                    0.9868437905990680,
                    -0.6971611631648800,
                    0.1237173087770760,
                ],
                [
                    0.1112498768967810,
                    -0.6971611631648800,
                    1.0421402435851100,
                    -0.4562289573170110,
                ],
                [
                    -0.0270701841245502,
                    0.1237173087770760,
                    -0.4562289573170110,
                    0.3595818326644850,
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
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3754127881031510,
                    -0.4791349725652340,
                    0.1366244859099620,
                    -0.0329023014478784,
                ],
                [
                    -0.4791349725652340,
                    1.0974938947368000,
                    -0.7555601661219740,
                    0.1372012439504100,
                ],
                [
                    0.1366244859099620,
                    -0.7555601661219740,
                    1.1003658020126000,
                    -0.4814301218005920,
                ],
                [
                    -0.0329023014478784,
                    0.1372012439504100,
                    -0.4814301218005920,
                    0.3771311792980600,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742278880876620,
                    -0.4704043350030930,
                    0.1245486104057020,
                    -0.0283721634902704,
                ],
                [
                    -0.4704043350030930,
                    1.0458022783717200,
                    -0.6891083590766130,
                    0.1137104157079890,
                ],
                [
                    0.1245486104057020,
                    -0.6891083590766130,
                    0.9190462296983460,
                    -0.3544864810274360,
                ],
                [
                    -0.0283721634902704,
                    0.1137104157079890,
                    -0.3544864810274360,
                    0.2691482288097170,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972590,
                    -0.3407384274106880,
                    0.0973538364030538,
                    -0.0234370346896240,
                ],
                [
                    -0.3407384274106880,
                    0.7788306912244300,
                    -0.5354461002167960,
                    0.0973538364030537,
                ],
                [
                    0.0973538364030538,
                    -0.5354461002167960,
                    0.7788306912244310,
                    -0.3407384274106880,
                ],
                [
                    -0.0234370346896240,
                    0.0973538364030537,
                    -0.3407384274106880,
                    0.2668216256972590,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3292204606981410,
                    -0.4134002391967840,
                    0.1112499890634780,
                    -0.0270702105648353,
                ],
                [
                    -0.4134002391967840,
                    0.9868443266035030,
                    -0.6971614651973810,
                    0.1237173777906620,
                ],
                [
                    0.1112499890634780,
                    -0.6971614651973810,
                    1.0421405856543300,
                    -0.4562291095204280,
                ],
                [
                    -0.0270702105648353,
                    0.1237173777906620,
                    -0.4562291095204280,
                    0.3595819422946010,
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
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    3.74901860610780e06,
                    2.89593744482105e06,
                    -1.05358398557321e06,
                    5.54643193641815e05,
                ],
                [
                    2.89593744482105e06,
                    1.88936621803465e07,
                    -2.36630842102390e06,
                    -1.04792564897764e06,
                ],
                [
                    -1.05358398557321e06,
                    -2.36630842102390e06,
                    1.88757400422500e07,
                    2.88462077162992e06,
                ],
                [
                    5.54643193641815e05,
                    -1.04792564897764e06,
                    2.88462077162992e06,
                    3.73110189762532e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    3.74548978664947e06,
                    2.83529217357612e06,
                    -8.63799618191817e05,
                    6.29634934688451e05,
                ],
                [
                    2.83529217357612e06,
                    1.92531430053028e07,
                    -3.32307881157745e06,
                    -1.41571111930997e06,
                ],
                [
                    -8.63799618191817e05,
                    -3.32307881157745e06,
                    2.30168957941050e07,
                    3.93911517581243e06,
                ],
                [
                    6.29634934688451e05,
                    -1.41571111930997e06,
                    3.93911517581243e06,
                    4.82119852843649e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329806e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442681e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857143e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857143e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442681e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329806e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    4.26389587430359e06,
                    3.18874298722127e06,
                    -1.17518017569044e06,
                    5.93504826831539e05,
                ],
                [
                    3.18874298722127e06,
                    1.95951789440892e07,
                    -2.51514790752116e06,
                    -1.07640022713129e06,
                ],
                [
                    -1.17518017569044e06,
                    -2.51514790752116e06,
                    1.94626651593996e07,
                    2.99118309010297e06,
                ],
                [
                    5.93504826831539e05,
                    -1.07640022713129e06,
                    2.99118309010297e06,
                    3.89099624195036e06,
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
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    3.74901860610781e06,
                    2.89593744482105e06,
                    -1.05358398557321e06,
                    5.54643193641817e05,
                ],
                [
                    2.89593744482105e06,
                    1.88936621803465e07,
                    -2.36630842102391e06,
                    -1.04792564897764e06,
                ],
                [
                    -1.05358398557321e06,
                    -2.36630842102391e06,
                    1.88757400422500e07,
                    2.88462077162992e06,
                ],
                [
                    5.54643193641817e05,
                    -1.04792564897764e06,
                    2.88462077162992e06,
                    3.73110189762534e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    3.74548978664949e06,
                    2.83529217357612e06,
                    -8.63799618191820e05,
                    6.29634934688452e05,
                ],
                [
                    2.83529217357612e06,
                    1.92531430053028e07,
                    -3.32307881157746e06,
                    -1.41571111930997e06,
                ],
                [
                    -8.63799618191820e05,
                    -3.32307881157746e06,
                    2.30168957941050e07,
                    3.93911517581242e06,
                ],
                [
                    6.29634934688452e05,
                    -1.41571111930997e06,
                    3.93911517581242e06,
                    4.82119852843651e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329808e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442682e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857144e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857144e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442682e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329808e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    4.26389587430361e06,
                    3.18874298722128e06,
                    -1.17518017569045e06,
                    5.93504826831540e05,
                ],
                [
                    3.18874298722128e06,
                    1.95951789440892e07,
                    -2.51514790752117e06,
                    -1.07640022713129e06,
                ],
                [
                    -1.17518017569045e06,
                    -2.51514790752117e06,
                    1.94626651593996e07,
                    2.99118309010297e06,
                ],
                [
                    5.93504826831540e05,
                    -1.07640022713129e06,
                    2.99118309010297e06,
                    3.89099624195037e06,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array(
            [
                7.95058998665675e-09,
                3.90641624893011e-09,
                -1.80536750828495e-09,
                -6.15666554867237e-07,
                5.54458994432774e-06,
                5.54455338012299e-06,
                -6.16060944960366e-07,
                -0.00000000000000e00,
                -0.00000000000000e00,
                5.49294982437498e-05,
                3.45212307107206e-05,
                -8.20929073888648e-06,
                6.52879669583503e-02,
            ]
        )
        self.assertTrue(
            np.allclose(expected_flux_vector_0, self.msh._heat_flux_vector_0)
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                7.95058998665675e-09,
                3.90641624893011e-09,
                -1.80536750828495e-09,
                -6.15666554867237e-07,
                5.54458994432774e-06,
                5.54455338012299e-06,
                -6.16060944960366e-07,
                -0.00000000000000e00,
                -0.00000000000000e00,
                5.49294982437498e-05,
                3.45212307107206e-05,
                -8.20929073888648e-06,
                6.52879669583503e-02,
            ]
        )
        self.assertTrue(np.allclose(expected_flux_vector, self.msh._heat_flux_vector))


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
        initial_temp_vector = np.array(
            [
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
            ]
        )
        initial_temp_rate_vector = np.array(
            [
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
            ]
        )
        for nd, T0, dTdt0 in zip(
            self.msh.nodes,
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
        self.msh.time_step = 3.024e05
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array(
            [
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
            ]
        )
        self.msh._temp_rate_vector[:] = np.array(
            [
                0.00000000000000e00,
                -9.15745232017429e-02,
                -1.04882997852940e-01,
                -7.67320511902980e-02,
                -3.37983197735703e-02,
                1.86084957826127e-03,
                1.97591262829366e-02,
                2.05973758982125e-02,
                1.15832003495520e-02,
                1.00523127785634e-03,
                -5.48750924589392e-03,
                -6.09286859998282e-03,
                -2.05841501790816e-03,
            ]
        )
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()
        self.msh.update_global_matrices_and_vectors()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

    def test_temperature_distribution_nodes(self):
        expected_temp_vector_0 = np.array(
            [
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
            ]
        )
        expected_temp_vector = np.array(
            [
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
            ]
        )
        actual_temp_nodes = np.array([nd.temp for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_vector,
                actual_temp_nodes,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_vector,
                self.msh._temp_vector,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_vector_0,
                self.msh._temp_vector_0,
            )
        )

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                0.00000000000000e00,
                2.47570047890236e-07,
                2.85733795580311e-08,
                1.33019023210993e-07,
                -2.87130590349577e-08,
                -2.65305419384922e-08,
                -2.42771971630921e-08,
                -9.41484392280149e-09,
                -9.95861051620030e-10,
                1.30146777036420e-09,
                5.40137966509952e-09,
                7.56209371863997e-09,
                -1.38333740500700e-08,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                actual_temp_rate_nodes,
                atol=1e-12,
                rtol=1e-10,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                self.msh._temp_rate_vector,
                atol=1e-12,
                rtol=1e-10,
            )
        )

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3754127694384990,
                    -0.4791349492121190,
                    0.1366244795670250,
                    -0.0329022997934044,
                ],
                [
                    -0.4791349492121190,
                    1.0974938412970900,
                    -0.7555601286108480,
                    0.1372012365258810,
                ],
                [
                    0.1366244795670250,
                    -0.7555601286108480,
                    1.1003657454416000,
                    -0.4814300963977770,
                ],
                [
                    -0.0329022997934044,
                    0.1372012365258810,
                    -0.4814300963977770,
                    0.3771311596653000,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742278648957170,
                    -0.4704043073588020,
                    0.1245486051912600,
                    -0.0283721627281755,
                ],
                [
                    -0.4704043073588020,
                    1.0458021981448800,
                    -0.6891083023287820,
                    0.1137104115427070,
                ],
                [
                    0.1245486051912600,
                    -0.6891083023287820,
                    0.9190461742111960,
                    -0.3544864770736730,
                ],
                [
                    -0.0283721627281755,
                    0.1137104115427070,
                    -0.3544864770736730,
                    0.2691482282591420,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972590,
                    -0.3407384274106880,
                    0.0973538364030538,
                    -0.0234370346896240,
                ],
                [
                    -0.3407384274106880,
                    0.7788306912244300,
                    -0.5354461002167960,
                    0.0973538364030537,
                ],
                [
                    0.0973538364030538,
                    -0.5354461002167960,
                    0.7788306912244310,
                    -0.3407384274106880,
                ],
                [
                    -0.0234370346896240,
                    0.0973538364030537,
                    -0.3407384274106880,
                    0.2668216256972590,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3292202434390340,
                    -0.4133999362112650,
                    0.1112498768967810,
                    -0.0270701841245502,
                ],
                [
                    -0.4133999362112650,
                    0.9868437905990680,
                    -0.6971611631648800,
                    0.1237173087770760,
                ],
                [
                    0.1112498768967810,
                    -0.6971611631648800,
                    1.0421402435851100,
                    -0.4562289573170110,
                ],
                [
                    -0.0270701841245502,
                    0.1237173087770760,
                    -0.4562289573170110,
                    0.3595818326644850,
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
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3753879416232020,
                    -0.4791042467337650,
                    0.1366167050145040,
                    -0.0329003999039413,
                ],
                [
                    -0.4791042467337650,
                    1.0974433676402300,
                    -0.7555344966562600,
                    0.1371953757497930,
                ],
                [
                    0.1366167050145040,
                    -0.7555344966562600,
                    1.1003399482612500,
                    -0.4814221566194900,
                ],
                [
                    -0.0329003999039413,
                    0.1371953757497930,
                    -0.4814221566194900,
                    0.3771271807736380,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742218218850550,
                    -0.4703963231115380,
                    0.1245458628648330,
                    -0.0283713616383489,
                ],
                [
                    -0.4703963231115380,
                    1.0458235893735000,
                    -0.6891377585117370,
                    0.1137104922497770,
                ],
                [
                    0.1245458628648330,
                    -0.6891377585117370,
                    0.9190792031511250,
                    -0.3544873075042210,
                ],
                [
                    -0.0283713616383489,
                    0.1137104922497770,
                    -0.3544873075042210,
                    0.2691481768927930,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972590,
                    -0.3407384274106880,
                    0.0973538364030538,
                    -0.0234370346896240,
                ],
                [
                    -0.3407384274106880,
                    0.7788306912244300,
                    -0.5354461002167960,
                    0.0973538364030537,
                ],
                [
                    0.0973538364030538,
                    -0.5354461002167960,
                    0.7788306912244310,
                    -0.3407384274106880,
                ],
                [
                    -0.0234370346896240,
                    0.0973538364030537,
                    -0.3407384274106880,
                    0.2668216256972590,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3289962966660710,
                    -0.4130762140338010,
                    0.1111334904406820,
                    -0.0270535730729519,
                ],
                [
                    -0.4130762140338010,
                    0.9863568813802830,
                    -0.6970058557736110,
                    0.1237251884271290,
                ],
                [
                    0.1111334904406820,
                    -0.6970058557736110,
                    1.0421897385013800,
                    -0.4563173731684510,
                ],
                [
                    -0.0270535730729519,
                    0.1237251884271290,
                    -0.4563173731684510,
                    0.3596457578142740,
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
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    3.74901860610780e06,
                    2.89593744482105e06,
                    -1.05358398557321e06,
                    5.54643193641815e05,
                ],
                [
                    2.89593744482105e06,
                    1.88936621803465e07,
                    -2.36630842102390e06,
                    -1.04792564897764e06,
                ],
                [
                    -1.05358398557321e06,
                    -2.36630842102390e06,
                    1.88757400422500e07,
                    2.88462077162992e06,
                ],
                [
                    5.54643193641815e05,
                    -1.04792564897764e06,
                    2.88462077162992e06,
                    3.73110189762532e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    3.74548978664947e06,
                    2.83529217357612e06,
                    -8.63799618191817e05,
                    6.29634934688451e05,
                ],
                [
                    2.83529217357612e06,
                    1.92531430053028e07,
                    -3.32307881157745e06,
                    -1.41571111930997e06,
                ],
                [
                    -8.63799618191817e05,
                    -3.32307881157745e06,
                    2.30168957941050e07,
                    3.93911517581243e06,
                ],
                [
                    6.29634934688451e05,
                    -1.41571111930997e06,
                    3.93911517581243e06,
                    4.82119852843649e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329806e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442681e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857143e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857143e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442681e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329806e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    4.26389587430359e06,
                    3.18874298722127e06,
                    -1.17518017569044e06,
                    5.93504826831539e05,
                ],
                [
                    3.18874298722127e06,
                    1.95951789440892e07,
                    -2.51514790752116e06,
                    -1.07640022713129e06,
                ],
                [
                    -1.17518017569044e06,
                    -2.51514790752116e06,
                    1.94626651593996e07,
                    2.99118309010297e06,
                ],
                [
                    5.93504826831539e05,
                    -1.07640022713129e06,
                    2.99118309010297e06,
                    3.89099624195036e06,
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
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    4.74285700814038e06,
                    3.49344088380004e06,
                    -1.29807950820942e06,
                    6.28942731006372e05,
                ],
                [
                    3.49344088380004e06,
                    2.04775275805259e07,
                    -2.69184560214359e06,
                    -1.08352184762866e06,
                ],
                [
                    -1.29807950820942e06,
                    -2.69184560214359e06,
                    1.99795376252816e07,
                    3.06432556263851e06,
                ],
                [
                    6.28942731006372e05,
                    -1.08352184762866e06,
                    3.06432556263851e06,
                    4.00294353743635e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    4.17010553124093e06,
                    2.80148011856716e06,
                    -1.53913344761798e06,
                    7.18402432622841e05,
                ],
                [
                    2.80148011856716e06,
                    2.63128374675488e07,
                    8.30360330006419e05,
                    -1.83207088751921e06,
                ],
                [
                    -1.53913344761798e06,
                    8.30360330006419e05,
                    2.78795373209754e07,
                    3.38735499836960e06,
                ],
                [
                    7.18402432622841e05,
                    -1.83207088751921e06,
                    3.38735499836960e06,
                    4.88471921523720e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329806e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442681e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857143e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857143e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442681e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329806e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    1.05184695178840e09,
                    5.27408401740155e08,
                    -2.39636875141802e08,
                    5.53787670808823e07,
                ],
                [
                    5.27408401740155e08,
                    3.86393873685784e08,
                    -1.44403954369652e08,
                    1.44997490061072e07,
                ],
                [
                    -2.39636875141802e08,
                    -1.44403954369652e08,
                    1.69498523604074e08,
                    1.91351534443363e07,
                ],
                [
                    5.53787670808823e07,
                    1.44997490061072e07,
                    1.91351534443363e07,
                    5.11240998577955e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array(
            [
                1.89850411575195e-08,
                9.32804654257637e-09,
                -4.31099786843920e-09,
                -1.42748247257048e-06,
                1.28559187259822e-05,
                1.28558305047007e-05,
                -1.42842430458346e-06,
                -0.00000000000000e00,
                -0.00000000000000e00,
                1.10932858948307e-04,
                7.12664676651715e-05,
                -1.57821060328194e-05,
                6.53002632662146e-02,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector_0,
                self.msh._heat_flux_vector_0,
            )
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                5.46746653845989e-09,
                2.68636671492506e-09,
                -1.24151622646639e-09,
                -7.60441975353691e-07,
                6.84647489228791e-06,
                6.84643509031610e-06,
                -7.60714420349861e-07,
                -0.00000000000000e00,
                -0.00000000000000e00,
                5.41348756985190e-05,
                3.30150985940221e-05,
                -7.78256844391883e-06,
                6.53078244526887e-02,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._heat_flux_vector,
            )
        )

    def test_global_residual_vector(self):
        expected_Psi = np.array(
            [
                -2.38461234636085e-01,
                7.59224243604616e-02,
                7.99943496297368e-03,
                4.17791714043375e-02,
                -8.96429211407198e-03,
                -7.86598279264764e-03,
                -7.63908219256820e-03,
                -2.82298628508657e-03,
                -3.20429239722832e-04,
                3.96237468132868e-04,
                1.63711247439950e-03,
                2.30096034332267e-03,
                -4.19458710959875e-03,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.00000000000000e00,
                7.49567570052084e-02,
                8.74547297620121e-03,
                4.03016846701951e-02,
                -8.64903073239771e-03,
                -8.02469673177830e-03,
                -7.36118354840201e-03,
                -2.86764617815346e-03,
                -3.12731582359419e-04,
                3.92558622480283e-04,
                1.63886471997195e-03,
                2.29287000911671e-03,
                -4.18115389772324e-03,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 5.22652045961174e-03
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
        initial_temp_vector = np.array(
            [
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
            ]
        )
        initial_temp_rate_vector = np.array(
            [
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
            ]
        )
        for nd, T0, dTdt0 in zip(
            self.msh.nodes,
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
        self.msh.time_step = 3.024e5
        self.msh.initialize_time_step()
        self.msh._temp_vector[:] = np.array(
            [
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
            ]
        )
        self.msh._temp_rate_vector[:] = np.array(
            [
                0.00000000000000e00,
                -3.02825804238568e-10,
                -3.46835310360251e-10,
                -2.53743555523472e-10,
                -1.11766930468156e-10,
                6.15360310271584e-12,
                6.53410260679122e-11,
                6.81130155364171e-11,
                3.83042339601586e-11,
                3.32417750613871e-12,
                -1.81465252840407e-11,
                -2.01483749999432e-11,
                -6.80692796927300e-12,
            ]
        )
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points_primary()
        self.msh.update_integration_points_secondary()
        self.msh.update_global_matrices_and_vectors()
        self.msh.iterative_correction_step()

    def test_temperature_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                0.00000000000000e00,
                2.47498817526917e-07,
                2.85763254553079e-08,
                1.32984448853519e-07,
                -2.87329006419087e-08,
                -2.65614768071763e-08,
                -2.42586956207760e-08,
                -9.41748664381412e-09,
                -9.95379039448084e-10,
                1.30275115736807e-09,
                5.39880674132476e-09,
                7.56091046420124e-09,
                -1.38388620278036e-08,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                actual_temp_rate_nodes,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                self.msh._temp_rate_vector,
            )
        )

    def test_temperature_rate_distribution_nodes(self):
        expected_temp_rate_vector = np.array(
            [
                0.00000000000000e00,
                2.47498817526917e-07,
                2.85763254553079e-08,
                1.32984448853519e-07,
                -2.87329006419087e-08,
                -2.65614768071763e-08,
                -2.42586956207760e-08,
                -9.41748664381412e-09,
                -9.95379039448084e-10,
                1.30275115736807e-09,
                5.39880674132476e-09,
                7.56091046420124e-09,
                -1.38388620278036e-08,
            ]
        )
        actual_temp_rate_nodes = np.array([nd.temp_rate for nd in self.msh.nodes])
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                actual_temp_rate_nodes,
                atol=1e-12,
                rtol=1e-10,
            )
        )
        self.assertTrue(
            np.allclose(
                expected_temp_rate_vector,
                self.msh._temp_rate_vector,
                atol=1e-12,
                rtol=1e-10,
            )
        )

    def test_global_heat_flow_matrix_0(self):
        expected_H = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3754127694384990,
                    -0.4791349492121190,
                    0.1366244795670250,
                    -0.0329022997934044,
                ],
                [
                    -0.4791349492121190,
                    1.0974938412970900,
                    -0.7555601286108480,
                    0.1372012365258810,
                ],
                [
                    0.1366244795670250,
                    -0.7555601286108480,
                    1.1003657454416000,
                    -0.4814300963977770,
                ],
                [
                    -0.0329022997934044,
                    0.1372012365258810,
                    -0.4814300963977770,
                    0.3771311596653000,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742278648957170,
                    -0.4704043073588020,
                    0.1245486051912600,
                    -0.0283721627281755,
                ],
                [
                    -0.4704043073588020,
                    1.0458021981448800,
                    -0.6891083023287820,
                    0.1137104115427070,
                ],
                [
                    0.1245486051912600,
                    -0.6891083023287820,
                    0.9190461742111960,
                    -0.3544864770736730,
                ],
                [
                    -0.0283721627281755,
                    0.1137104115427070,
                    -0.3544864770736730,
                    0.2691482282591420,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972590,
                    -0.3407384274106880,
                    0.0973538364030538,
                    -0.0234370346896240,
                ],
                [
                    -0.3407384274106880,
                    0.7788306912244300,
                    -0.5354461002167960,
                    0.0973538364030537,
                ],
                [
                    0.0973538364030538,
                    -0.5354461002167960,
                    0.7788306912244310,
                    -0.3407384274106880,
                ],
                [
                    -0.0234370346896240,
                    0.0973538364030537,
                    -0.3407384274106880,
                    0.2668216256972590,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3292202434390340,
                    -0.4133999362112650,
                    0.1112498768967810,
                    -0.0270701841245502,
                ],
                [
                    -0.4133999362112650,
                    0.9868437905990680,
                    -0.6971611631648800,
                    0.1237173087770760,
                ],
                [
                    0.1112498768967810,
                    -0.6971611631648800,
                    1.0421402435851100,
                    -0.4562289573170110,
                ],
                [
                    -0.0270701841245502,
                    0.1237173087770760,
                    -0.4562289573170110,
                    0.3595818326644850,
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
        expected_H[0:4, 0:4] += np.array(
            [
                [
                    0.3753879492274280,
                    -0.4791042561238620,
                    0.1366167073707450,
                    -0.0329004004743101,
                ],
                [
                    -0.4791042561238620,
                    1.0974433825255700,
                    -0.7555345038239140,
                    0.1371953774222090,
                ],
                [
                    0.1366167073707450,
                    -0.7555345038239140,
                    1.1003399547735400,
                    -0.4814221583203750,
                ],
                [
                    -0.0329004004743101,
                    0.1371953774222090,
                    -0.4814221583203750,
                    0.3771271813724750,
                ],
            ]
        )
        expected_H[3:7, 3:7] += np.array(
            [
                [
                    0.3742218246395010,
                    -0.4703963271434800,
                    0.1245458642680330,
                    -0.0283713617640532,
                ],
                [
                    -0.4703963271434800,
                    1.0458236191503300,
                    -0.6891377855047830,
                    0.1137104934979320,
                ],
                [
                    0.1245458642680330,
                    -0.6891377855047830,
                    0.9190792299556210,
                    -0.3544873087188710,
                ],
                [
                    -0.0283713617640532,
                    0.1137104934979320,
                    -0.3544873087188710,
                    0.2691481769849920,
                ],
            ]
        )
        expected_H[6:10, 6:10] += np.array(
            [
                [
                    0.2668216256972600,
                    -0.3407384274106900,
                    0.0973538364030547,
                    -0.0234370346896243,
                ],
                [
                    -0.3407384274106900,
                    0.7788306912244320,
                    -0.5354461002167970,
                    0.0973538364030541,
                ],
                [
                    0.0973538364030547,
                    -0.5354461002167970,
                    0.7788306912244330,
                    -0.3407384274106900,
                ],
                [
                    -0.0234370346896243,
                    0.0973538364030541,
                    -0.3407384274106900,
                    0.2668216256972600,
                ],
            ]
        )
        expected_H[9:13, 9:13] += np.array(
            [
                [
                    0.3289962995618880,
                    -0.4130762153818610,
                    0.1111334919996520,
                    -0.0270535761796790,
                ],
                [
                    -0.4130762153818610,
                    0.9863568971153020,
                    -0.6970058899187200,
                    0.1237252081852800,
                ],
                [
                    0.1111334919996520,
                    -0.6970058899187200,
                    1.0421898250233900,
                    -0.4563174271043250,
                ],
                [
                    -0.0270535761796790,
                    0.1237252081852800,
                    -0.4563174271043250,
                    0.3596457950987240,
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
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    3.74901860610780e06,
                    2.89593744482105e06,
                    -1.05358398557321e06,
                    5.54643193641815e05,
                ],
                [
                    2.89593744482105e06,
                    1.88936621803465e07,
                    -2.36630842102390e06,
                    -1.04792564897764e06,
                ],
                [
                    -1.05358398557321e06,
                    -2.36630842102390e06,
                    1.88757400422500e07,
                    2.88462077162992e06,
                ],
                [
                    5.54643193641815e05,
                    -1.04792564897764e06,
                    2.88462077162992e06,
                    3.73110189762532e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    3.74548978664947e06,
                    2.83529217357612e06,
                    -8.63799618191817e05,
                    6.29634934688451e05,
                ],
                [
                    2.83529217357612e06,
                    1.92531430053028e07,
                    -3.32307881157745e06,
                    -1.41571111930997e06,
                ],
                [
                    -8.63799618191817e05,
                    -3.32307881157745e06,
                    2.30168957941050e07,
                    3.93911517581243e06,
                ],
                [
                    6.29634934688451e05,
                    -1.41571111930997e06,
                    3.93911517581243e06,
                    4.82119852843649e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329806e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442681e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857143e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857143e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442681e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329806e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    4.26389587430359e06,
                    3.18874298722127e06,
                    -1.17518017569044e06,
                    5.93504826831539e05,
                ],
                [
                    3.18874298722127e06,
                    1.95951789440892e07,
                    -2.51514790752116e06,
                    -1.07640022713129e06,
                ],
                [
                    -1.17518017569044e06,
                    -2.51514790752116e06,
                    1.94626651593996e07,
                    2.99118309010297e06,
                ],
                [
                    5.93504826831539e05,
                    -1.07640022713129e06,
                    2.99118309010297e06,
                    3.89099624195036e06,
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
        expected_C[0:4, 0:4] += np.array(
            [
                [
                    4.74285500589966e06,
                    3.49343955102212e06,
                    -1.29807894103605e06,
                    6.28942612193995e05,
                ],
                [
                    3.49343955102212e06,
                    2.04775242196110e07,
                    -2.69184500685009e06,
                    -1.08352197936191e06,
                ],
                [
                    -1.29807894103605e06,
                    -2.69184500685009e06,
                    1.99795374233527e07,
                    3.06432562767385e06,
                ],
                [
                    6.28942612193995e05,
                    -1.08352197936191e06,
                    3.06432562767385e06,
                    4.00294343313938e06,
                ],
            ]
        )
        expected_C[3:7, 3:7] += np.array(
            [
                [
                    4.17010491590658e06,
                    2.80148200739218e06,
                    -1.53913095787304e06,
                    7.18402142957111e05,
                ],
                [
                    2.80148200739218e06,
                    2.63128141962097e07,
                    8.30339725061684e05,
                    -1.83206863961090e06,
                ],
                [
                    -1.53913095787304e06,
                    8.30339725061684e05,
                    2.78795160638767e07,
                    3.38735737086790e06,
                ],
                [
                    7.18402142957111e05,
                    -1.83206863961090e06,
                    3.38735737086790e06,
                    4.88471894957745e06,
                ],
            ]
        )
        expected_C[6:10, 6:10] += np.array(
            [
                [
                    4.84663139329808e06,
                    3.74856646825397e06,
                    -1.36311507936508e06,
                    7.19421847442682e05,
                ],
                [
                    3.74856646825397e06,
                    2.45360714285714e07,
                    -3.06700892857144e06,
                    -1.36311507936508e06,
                ],
                [
                    -1.36311507936508e06,
                    -3.06700892857144e06,
                    2.45360714285714e07,
                    3.74856646825397e06,
                ],
                [
                    7.19421847442682e05,
                    -1.36311507936508e06,
                    3.74856646825397e06,
                    4.84663139329808e06,
                ],
            ]
        )
        expected_C[9:13, 9:13] += np.array(
            [
                [
                    1.05184703160211e09,
                    5.27408427089753e08,
                    -2.39636893587137e08,
                    5.53787629545943e07,
                ],
                [
                    5.27408427089753e08,
                    3.86393746848379e08,
                    -1.44403919746439e08,
                    1.44997802492022e07,
                ],
                [
                    -2.39636893587137e08,
                    -1.44403919746439e08,
                    1.69498436030225e08,
                    1.91350794170755e07,
                ],
                [
                    5.53787629545943e07,
                    1.44997802492022e07,
                    1.91350794170755e07,
                    5.11239567880632e07,
                ],
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_C,
                self.msh._heat_storage_matrix,
            )
        )

    def test_global_flux_vector_0(self):
        expected_flux_vector_0 = np.array(
            [
                1.89850411575195e-08,
                9.32804654257637e-09,
                -4.31099786843920e-09,
                -1.42748247257048e-06,
                1.28559187259822e-05,
                1.28558305047007e-05,
                -1.42842430458346e-06,
                -0.00000000000000e00,
                -0.00000000000000e00,
                1.10932858948307e-04,
                7.12664676651715e-05,
                -1.57821060328194e-05,
                6.53002632662146e-02,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector_0,
                self.msh._heat_flux_vector_0,
            )
        )

    def test_global_flux_vector(self):
        expected_flux_vector = np.array(
            [
                5.46755066490460e-09,
                2.68640804931463e-09,
                -1.24153532934089e-09,
                -7.60449037120510e-07,
                6.84653850755319e-06,
                6.84649869388256e-06,
                -7.60721487239478e-07,
                -0.00000000000000e00,
                -0.00000000000000e00,
                5.41352133532762e-05,
                3.30157898019445e-05,
                -7.78264191890823e-06,
                6.53078337911065e-02,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_flux_vector,
                self.msh._heat_flux_vector,
            )
        )

    def test_global_residual_vector(self):
        expected_Psi = np.array(
            [
                -2.36065087465956e-01,
                -2.18385250792819e-05,
                1.12974581959087e-06,
                -1.08848129651701e-05,
                -5.90483595852911e-06,
                -9.44720627751167e-06,
                5.79182251906826e-06,
                -8.43658542483070e-07,
                1.56954993024943e-07,
                3.88509427186182e-07,
                -7.78294034245843e-07,
                -3.56285036477028e-07,
                -1.66211459323978e-06,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_Psi,
                self.msh._residual_heat_flux_vector,
                atol=1e-7,
            )
        )

    def test_temperature_increment_vector(self):
        expected_dT = np.array(
            [
                0.00000000000000e00,
                -2.15400636773106e-05,
                8.90839395904079e-07,
                -1.04552863107220e-05,
                -6.00010207255151e-06,
                -9.35470456152675e-06,
                5.59486659083009e-06,
                -7.99158862030743e-07,
                1.45760487476165e-07,
                3.88096232875535e-07,
                -7.78052157469524e-07,
                -3.57816149462257e-07,
                -1.65956446958961e-06,
            ]
        )
        self.assertTrue(
            np.allclose(
                expected_dT,
                self.msh._delta_temp_vector,
                rtol=1e-3,
                atol=1e-7,
            )
        )

    def test_iteration_variables(self):
        expected_eps_a = 1.62910190189313e-06
        self.assertEqual(self.msh._iter, 1)
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)


if __name__ == "__main__":
    unittest.main()
