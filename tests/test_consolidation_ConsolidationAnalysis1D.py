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
    HydraulicBoundary1D,
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
        hyd_bound = HydraulicBoundary1D(
            nodes=(self.msh.nodes[0],), bnd_value=100.0,
        )
        self.msh.add_boundary(hyd_bound)
        void_ratio_bound = ConsolidationBoundary1D(
            nodes=(self.msh.nodes[0],),
            bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
            bnd_value=0.6,
            bnd_value_1=self.mtl.eff_stress(0.6, 0.0)[0],
        )
        self.msh.add_boundary(void_ratio_bound)
        for nd in self.msh.nodes:
            nd.void_ratio_0 = 0.9
        for e in self.msh.elements:
            e.assign_material(self.mtl)
        self.msh._void_ratio_vector[:] = np.array([
            0.6,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        self.msh.update_nodes()
        self.msh.initialize_global_system(0.0)

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

    def test_calculate_settlement(self):
        expected = 20.13157894736840
        self.assertAlmostEqual(
            expected, self.msh.calculate_total_settlement()
        )

    def test_deformed_coords_nodes(self):
        expected_zdef_nodes = np.array([
            20.1315789473684,
            40.8552631578947,
            60.9868421052632,
            80.6578947368421,
            100.0000000000000,
        ])
        actual_zdef_nodes = np.array([
            nd.z_def for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(
            actual_zdef_nodes,
            expected_zdef_nodes,
        ))

    def test_total_stress_nodes(self):
        expected_sig_nodes = np.array([
            3.55161801637395E+05,
            7.64987459532132E+05,
            1.16900456479529E+06,
            1.56850390690055E+06,
            1.96477627532161E+06,
        ])
        actual_sig_nodes = np.array([
            nd.tot_stress for nd in self.msh.nodes
        ])
        print(actual_sig_nodes)
        self.assertTrue(np.allclose(
            actual_sig_nodes,
            expected_sig_nodes,
        ))

    def test_total_stress_int_pts(self):
        expected_sig_int_pts = np.array([
            4.41768153631593E+05,
            6.78381107537935E+05,
            8.50366319923262E+05,
            1.08362570440416E+06,
            1.25342870949515E+06,
            1.48407976220070E+06,
            1.65224611182093E+06,
            1.88103407040123E+06,
        ])
        actual_sig_int_pts = np.array([
            ip.tot_stress
            for e in self.msh.elements
            for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_sig_int_pts,
            expected_sig_int_pts,
        ))

    def test_pore_pressure_int_pts(self):
        expected_u_int_pts = np.array([
            2.74716868672114E+05,
            4.82758638439012E+05,
            6.33298685601398E+05,
            8.37333760981363E+05,
            9.86381922748051E+05,
            1.19049824805858E+06,
            1.34119864339427E+06,
            1.54970904704083E+06,
        ])
        actual_u_int_pts = np.array([
            ip.pore_pressure
            for e in self.msh.elements
            for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_u_int_pts,
            expected_u_int_pts,
        ))

    def test_excess_pore_pressure_int_pts(self):
        expected_ue_int_pts = np.array([
            3.42638730910717E+04,
            1.24930712967423E+05,
            1.90773839518160E+05,
            2.80787554433022E+05,
            3.47321002882403E+05,
            4.40024299503172E+05,
            5.09846636992317E+05,
            6.08807106074363E+05,
        ])
        actual_ue_int_pts = np.array([
            ip.exc_pore_pressure
            for e in self.msh.elements
            for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(
            actual_ue_int_pts,
            expected_ue_int_pts,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = -np.array([
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
            order=1,
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
            e.assign_material(self.mtl)
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
            0.8,
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
            0.747168783648703,
            0.602831216351297,
            0.541547005383793,
            0.518452994616208,
            0.503660254037844,
            0.486339745962156,
            0.475773502691896,
            0.464226497308104,
        ])
        expected_void_ratio_0_int_pts = 0.9 * np.ones(
            2 * self.msh.num_elements
        )
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        actual_void_ratio_0_int_pts = np.array([
            ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_0_int_pts,
                                    expected_void_ratio_0_int_pts))
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            3.40877688437546E-10,
            1.14646460334134E-10,
            7.21819834044123E-11,
            6.06332354537998E-11,
            5.42262977612564E-11,
            4.75796675742455E-11,
            4.39316973318227E-11,
            4.02641883365508E-11,
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
            2.57344224239529E-09,
            8.65518788622658E-10,
            5.44934947441782E-10,
            4.57748144576778E-10,
            4.09379229092870E-10,
            3.59200764872360E-10,
            3.31660561921905E-10,
            3.03972851951628E-10,
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
            7.04988931005544E+04,
            1.55248308168406E+05,
            2.17067634321865E+05,
            2.46291943422797E+05,
            2.67046786747095E+05,
            2.93581514142121E+05,
            3.11047468426655E+05,
            3.31325023360404E+05,
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
            -3.85581235928545E+05,
            -8.49103183138045E+05,
            -1.18721306166510E+06,
            -1.34705025522533E+06,
            -1.46056520260290E+06,
            -1.60569220437593E+06,
            -1.70121915442459E+06,
            -1.81212365730539E+06,
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
            7.04988931005544E+04,
            1.55248308168406E+05,
            2.17067634321865E+05,
            2.46291943422797E+05,
            2.67046786747095E+05,
            2.93581514142121E+05,
            3.11047468426655E+05,
            3.31325023360404E+05,
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
            -1.664630734847630E-10,
            3.186001754681700E-12,
            -5.769218208232320E-11,
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
            [1.46927921704707E-09, -1.46927921704707E-09,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [1.60178698481267E-11, 6.58064577771503E-10,
             -6.74082447619630E-10, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.95455548875149E-10,
                7.92450395857143E-10, -5.96994846981994E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22272641090901E-10, 7.72457873113186E-10,
             -5.50185232022285E-10,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -2.35477581693084E-10,
             2.35477581693084E-10,],
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
        expected_flux_vector[-1] = 2.0e-11
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5 + 2.5920E+06)

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
        expected_void_ratio_vector_0 = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))

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

    def test_global_stiffness_matrix_0(self):
        expected_K = np.array([
            [1.46927921704707E-09, -1.46927921704707E-09,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [1.60178698481267E-11, 6.58064577771503E-10,
             -6.74082447619630E-10, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.95455548875149E-10,
                7.92450395857143E-10, -5.96994846981994E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22272641090901E-10, 7.72457873113186E-10,
             -5.50185232022285E-10,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -2.35477581693084E-10,
             2.35477581693084E-10,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [7.99028679973533E-10, -7.99028679973533E-10,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-1.42998283395020E-10, 8.17080731014650E-10,
             -6.74082447619630E-10, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.95455548875149E-10,
                7.92450395857143E-10, -5.96994846981994E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22272641090901E-10, 7.72457873113186E-10,
             -5.50185232022285E-10,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -2.35477581693084E-10,
             2.35477581693084E-10,],
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
        self.assertTrue(np.allclose(
            expected_M, self.msh._mass_matrix_0,
        ))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = 2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))


class TestUpdateGlobalMatricesLinear(unittest.TestCase):
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
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
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))

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
            -8.12344162297375E-11,
            -6.33033162146255E-11,
            -5.76921820823232E-11,
            -4.72209351294693E-11,
            -4.54587359353543E-11,
            -3.92718341524608E-11,
            -3.97829354054352E-11,
            -3.62767785500602E-11,
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
            [1.46927921704707E-09, -1.46927921704707E-09,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [1.60178698481267E-11, 6.58064577771503E-10,
             -6.74082447619630E-10, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.95455548875149E-10,
                7.92450395857143E-10, -5.96994846981994E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22272641090901E-10, 7.72457873113186E-10,
             -5.50185232022285E-10,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -2.35477581693084E-10,
             2.35477581693084E-10,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [7.99028679973533E-10, -7.99028679973533E-10,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-1.42998283395020E-10, 8.17080731014650E-10,
             -6.74082447619630E-10, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.95455548875149E-10,
                7.92450395857143E-10, -5.96994846981994E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22272641090901E-10, 7.72457873113186E-10,
             -5.50185232022285E-10,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -2.35477581693084E-10,
             2.35477581693084E-10,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
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

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = 2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = 2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()
        self.msh.update_iteration_variables()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        expected_void_ratio_vector = np.array([
            0.600000000000000,
            0.550028476720522,
            0.509990640598832,
            0.479997039697723,
            0.460016081578176,
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(expected_void_ratio_vector_0,
                                    self.msh._void_ratio_vector_0))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    actual_void_ratio_nodes))
        self.assertTrue(np.allclose(expected_void_ratio_vector,
                                    self.msh._void_ratio_vector))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_int_pts = np.array([
            0.589439774568872,
            0.560588702151650,
            0.541567486390991,
            0.518451630928363,
            0.503652246925388,
            0.486335433371166,
            0.475774566412443,
            0.464238554863456,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.03622552066055E-10,
            8.33413640025354E-11,
            7.21931450838836E-11,
            6.06326112314324E-11,
            5.42230199148045E-11,
            4.75781185134807E-11,
            4.39320501273934E-11,
            4.02678536728603E-11,
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
            7.82294241591142E-10,
            6.29182237318126E-10,
            5.45019212086907E-10,
            4.57743432035409E-10,
            4.09354483124420E-10,
            3.59189070301130E-10,
            3.31663225337715E-10,
            3.04000523258996E-10,
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
            1.67022352957552E+05,
            1.95496055964745E+05,
            2.16939712662274E+05,
            2.46293780385706E+05,
            2.67058481913006E+05,
            2.93588438917519E+05,
            3.11037945437879E+05,
            3.31210058836345E+05,
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
            -4.80728975146062E+06,
            -5.62682880254440E+06,
            -6.24402685568204E+06,
            -1.34706030216930E+06,
            -1.46062916724586E+06,
            -1.60573007821110E+06,
            -8.95239170650944E+06,
            -9.53299180157811E+06,
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
            1.67051284959479E+05,
            1.95622469098922E+05,
            2.17067634321865E+05,
            2.46293780385706E+05,
            2.67058481913006E+05,
            2.93588438917519E+05,
            3.11047468426655E+05,
            3.31325023360404E+05,
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
            1.70216296054053E-11,
            3.08869478211344E-11,
            1.57716713193315E-11,
            -4.72046002747267E-11,
            -4.54583052263202E-11,
            -3.92729359023570E-11,
            -6.37653443009028E-12,
            -3.41910393288506E-12,
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
            [1.46927921704707E-09, -1.46927921704707E-09,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [1.60178698481267E-11, 6.58064577771503E-10,
             -6.74082447619630E-10, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.95455548875149E-10,
                7.92450395857143E-10, -5.96994846981994E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22272641090901E-10, 7.72457873113186E-10,
             -5.50185232022285E-10,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -2.35477581693084E-10,
             2.35477581693084E-10,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [2.72504385926954E-09, -2.72504385926954E-09,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-2.06895212001245E-09, 3.66042190928219E-09,
             -1.59146978926974E-09, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.11280811761984E-09,
                1.70978961489626E-09, -5.96981497276424E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22275614408773E-10, 2.43111587622748E-09,
             -2.20884026181870E-09,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -1.89411877297385E-09,
             1.89411877297385E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
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

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = 2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = 2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            1.99865413933747E-01,
            2.84846052037757E-05,
            -9.36279493372872E-06,
            -2.96123999409947E-06,
            1.60833719834228E-05,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000E+00,
            2.84767205216839E-05,
            -9.35940116823052E-06,
            -2.96030227735638E-06,
            1.60815781757823E-05,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 2.92296136212227E-05
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()
        self.msh.iterative_correction_step()

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_vector_0 = np.array([
            0.8,
            0.55,
            0.51,
            0.48,
            0.46,
        ])
        expected_void_ratio_vector = np.array([
            0.600000000000000,
            0.550040649129255,
            0.509990582977506,
            0.479997370203776,
            0.460025702147165,
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
            0.589442346901509,
            0.560598302227746,
            0.541577074290268,
            0.518454157816493,
            0.503652271325028,
            0.486335681856254,
            0.475776860139796,
            0.464246212211145,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            1.03624564406604E-10,
            8.33474044187805E-11,
            7.21983708623208E-11,
            6.06337679089101E-11,
            5.42230299029077E-11,
            4.75782077666920E-11,
            4.39328108789921E-11,
            4.02701815778508E-11,
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
            7.82309433674257E-10,
            6.29227839194853E-10,
            5.45058663888641E-10,
            4.57752164324974E-10,
            4.09354558529199E-10,
            3.59189744114618E-10,
            3.31668968600964E-10,
            3.04018097683025E-10,
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
            1.67009987466989E+05,
            1.95442045442292E+05,
            2.16879853821457E+05,
            2.46275868167593E+05,
            2.67058294363748E+05,
            2.93540770072724E+05,
            3.10993682294692E+05,
            3.31137069447185E+05,
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
            -4.80693384403266E+06,
            -5.62527425474608E+06,
            -6.24230397975018E+06,
            -7.08838928508584E+06,
            -7.68655559452977E+06,
            -8.44878251694309E+06,
            -8.95111771083854E+06,
            -9.53089099808526E+06,
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
            1.67051284959479E+05,
            1.95622469098922E+05,
            2.17067634321865E+05,
            2.46293780385706E+05,
            2.67058481913006E+05,
            2.93599130719879E+05,
            3.11053036259208E+05,
            3.31325023360404E+05,
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
            1.69833962174449E-11,
            3.08285232293992E-11,
            1.57753865319826E-11,
            2.39330022371739E-11,
            6.70997770370668E-12,
            1.16257681381967E-11,
            -6.40168397923104E-12,
            -3.44709279125614E-12,
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
            [1.46927921704707E-09, -1.46927921704707E-09,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [1.60178698481267E-11, 6.58064577771503E-10,
             -6.74082447619630E-10, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -1.95455548875149E-10,
                7.92450395857143E-10, -5.96994846981994E-10,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -2.22272641090901E-10, 7.72457873113186E-10,
             -5.50185232022285E-10,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -2.35477581693084E-10,
             2.35477581693084E-10,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.array([
            [2.72474047151696E-09, -2.72474047151696E-09,
             0.00000000000000E+00, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [-2.06862250884173E-09, 4.54792780410049E-09,
             -2.47930529525876E-09, 0.00000000000000E+00,
             0.00000000000000E+00,],
            [0.00000000000000E+00, -2.00062249814840E-09,
                4.31578276028432E-09, -2.31516026213592E-09,
             0.00000000000000E+00,],
            [0.00000000000000E+00, 0.00000000000000E+00,
             -1.94045404180519E-09, 4.14899700705799E-09,
             -2.20854296525281E-09,],
            [0.00000000000000E+00, 0.00000000000000E+00,
                0.00000000000000E+00, -1.89381084993773E-09,
             1.89381084993773E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
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

    def test_global_flux_vector_0(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = 2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector_0,
                                    atol=1e-18, rtol=1e-8))

    def test_global_flux_vector(self):
        expected_flux_vector = np.zeros(self.msh.num_nodes)
        expected_flux_vector[-1] = 2.0e-11
        self.assertTrue(np.allclose(expected_flux_vector,
                                    self.msh._water_flux_vector,
                                    atol=1e-18, rtol=1e-8))

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            1.99830913164837E-01,
            -9.70524228677597E-10,
            1.67879937417785E-10,
            9.53591748482750E-10,
            -1.23640863876790E-09,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-10, rtol=1e-8,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000E+00,
            -9.69495789110242E-10,
            1.67912724123035E-10,
            9.52160392336269E-10,
            -1.23446932624982E-09,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            atol=1e-14, rtol=1e-8,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.57812353037929E-09
        self.assertEqual(self.msh._iter, 5)
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a,
                               delta=1e-14)

    def test_calculate_settlement(self):
        expected = 20.13103350810370
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            20.1310335081037,
            40.8549851471119,
            60.9867695688670,
            80.6577429450596,
            100.0000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(expected, actual))


class TestInitializeIntegrationPointsCubic(unittest.TestCase):
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
            e.assign_material(self.mtl)
        self.msh.initialize_global_system(0.0)

    def test_void_ratio_distribution_nodes(self):
        expected_void_ratio_0_nodes = np.array([
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
        expected_void_ratio_nodes = np.array([
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
        actual_void_ratio_0_nodes = np.array([
            nd.void_ratio_0 for nd in self.msh.nodes
        ])
        actual_void_ratio_nodes = np.array([
            nd.void_ratio for nd in self.msh.nodes
        ])
        self.assertTrue(np.allclose(actual_void_ratio_0_nodes,
                                    expected_void_ratio_0_nodes))
        self.assertTrue(np.allclose(actual_void_ratio_nodes,
                                    expected_void_ratio_nodes))

    def test_void_ratio_distribution_int_pts(self):
        expected_void_ratio_0_int_pts = np.array([
            0.783745715060955,
            0.714698836928103,
            0.627327140376931,
            0.562625830645095,
            0.535113117201930,
            0.526348555862977,
            0.516904981852185,
            0.519829331413518,
            0.536033293625735,
            0.550919150923390,
            0.559000789171861,
            0.574407230147394,
            0.593902490791297,
            0.608339304393574,
            0.614478220550266,
            0.616433858525457,
            0.618541004706859,
            0.617888494120908,
            0.614272901437457,
            0.610951417714685,
        ])
        expected_void_ratio_int_pts = np.array([
            0.564605405564336,
            0.485143306872300,
            0.420257096108955,
            0.405867716936218,
            0.418920304310449,
            0.431085818448524,
            0.460255726081956,
            0.509407217025838,
            0.557357527450672,
            0.584326765554388,
            0.595666299923334,
            0.613396690730218,
            0.627874761329268,
            0.631085465808518,
            0.628173039897354,
            0.625458546779438,
            0.618949860617675,
            0.607982680571832,
            0.597283520127572,
            0.591265869710409,
        ])
        actual_void_ratio_0_int_pts = np.array([
            ip.void_ratio_0 for e in self.msh.elements for ip in e.int_pts
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_0_int_pts,
                                    expected_void_ratio_0_int_pts))
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
            [1.75973053787198E-09, -2.30012579221697E-09,
                7.04533370197087E-10, -1.64138115852098E-10,],
            [-1.59213591880079E-09, 3.81159727452406E-09, -
                2.74059958738664E-09, 5.21138231663364E-10,],
            [4.27905926569192E-10, -2.23565512579640E-09,
                3.37467740597141E-09, -1.56692820674420E-09,],
            [-9.52348424823319E-11, 3.75502889134307E-10, -
                1.26214863347869E-09, 2.26528300823914E-09,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [2.26528300823914E-09, -1.83605309846816E-09,
                7.40454657106645E-10, -1.87803980050906E-10,],
            [-1.45652110193580E-09, 4.53944671410106E-09, -
                3.93945857548988E-09, 8.56532963324615E-10,],
            [5.44362557557076E-10, -3.08043242328214E-09,
                7.05088438566790E-09, -4.51481451994284E-09,],
            [-1.06722513017259E-10, 5.09787707760499E-10, -
                3.59987343768192E-09, 6.77569446565902E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.77569446565902E-09, -4.81318466550500E-09,
                1.76001522428353E-09, -5.25716781498866E-10,],
            [-3.67644360509383E-09, 1.07093191546781E-08, -
                9.31245855882194E-09, 2.27958300923772E-09,],
            [1.26491812901207E-09, -7.43512509628434E-09,
                1.39050373566888E-08, -7.73483038941653E-09,],
            [-3.75396622353588E-10, 1.72079478841131E-09, -
                6.37646668050495E-09, 1.22941608598481E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.22941608598481E-08, -8.49031902641601E-09,
                1.58123119002383E-09, -3.54004509008713E-10,],
            [-7.19477267560949E-09, 1.23945827374776E-08, -
                6.53432234036163E-09, 1.33451227849347E-09,],
            [1.05233447833790E-09, -4.86743768888254E-09,
                6.74118845073356E-09, -2.92608524018892E-09,],
            [-2.08936498445707E-10, 8.70617741434581E-10, -
                1.86302253771705E-09, 1.20134129472817E-09,],
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
            e.assign_material(self.mtl)
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
            [1.75973053787198E-09, -2.30012579221697E-09,
                7.04533370197087E-10, -1.64138115852098E-10,],
            [-1.59213591880079E-09, 3.81159727452406E-09, -
                2.74059958738664E-09, 5.21138231663364E-10,],
            [4.27905926569192E-10, -2.23565512579640E-09,
                3.37467740597141E-09, -1.56692820674420E-09,],
            [-9.52348424823319E-11, 3.75502889134307E-10, -
                1.26214863347869E-09, 2.26528300823914E-09,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [2.26528300823914E-09, -1.83605309846816E-09,
                7.40454657106645E-10, -1.87803980050906E-10,],
            [-1.45652110193580E-09, 4.53944671410106E-09, -
                3.93945857548988E-09, 8.56532963324615E-10,],
            [5.44362557557076E-10, -3.08043242328214E-09,
                7.05088438566790E-09, -4.51481451994284E-09,],
            [-1.06722513017259E-10, 5.09787707760499E-10, -
                3.59987343768192E-09, 6.77569446565902E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.77569446565902E-09, -4.81318466550500E-09,
                1.76001522428353E-09, -5.25716781498866E-10,],
            [-3.67644360509383E-09, 1.07093191546781E-08, -
                9.31245855882194E-09, 2.27958300923772E-09,],
            [1.26491812901207E-09, -7.43512509628434E-09,
                1.39050373566888E-08, -7.73483038941653E-09,],
            [-3.75396622353588E-10, 1.72079478841131E-09, -
                6.37646668050495E-09, 1.22941608598481E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.22941608598481E-08, -8.49031902641601E-09,
                1.58123119002383E-09, -3.54004509008713E-10,],
            [-7.19477267560949E-09, 1.23945827374776E-08, -
                6.53432234036163E-09, 1.33451227849347E-09,],
            [1.05233447833790E-09, -4.86743768888254E-09,
                6.74118845073356E-09, -2.92608524018892E-09,],
            [-2.08936498445707E-10, 8.70617741434581E-10, -
                1.86302253771705E-09, 1.20134129472817E-09,],
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()

    def test_time_step_set(self):
        self.assertAlmostEqual(self.msh._t0, 1.5)
        self.assertAlmostEqual(self.msh._t1, 1.5 + 2.5920E+06)

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
            [1.75973053787198E-09, -2.30012579221697E-09,
                7.04533370197087E-10, -1.64138115852098E-10,],
            [-1.59213591880079E-09, 3.81159727452406E-09, -
                2.74059958738664E-09, 5.21138231663364E-10,],
            [4.27905926569192E-10, -2.23565512579640E-09,
                3.37467740597141E-09, -1.56692820674420E-09,],
            [-9.52348424823319E-11, 3.75502889134307E-10, -
                1.26214863347869E-09, 2.26528300823914E-09,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [2.26528300823914E-09, -1.83605309846816E-09,
                7.40454657106645E-10, -1.87803980050906E-10,],
            [-1.45652110193580E-09, 4.53944671410106E-09, -
                3.93945857548988E-09, 8.56532963324615E-10,],
            [5.44362557557076E-10, -3.08043242328214E-09,
                7.05088438566790E-09, -4.51481451994284E-09,],
            [-1.06722513017259E-10, 5.09787707760499E-10, -
                3.59987343768192E-09, 6.77569446565902E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.77569446565902E-09, -4.81318466550500E-09,
                1.76001522428353E-09, -5.25716781498866E-10,],
            [-3.67644360509383E-09, 1.07093191546781E-08, -
                9.31245855882194E-09, 2.27958300923772E-09,],
            [1.26491812901207E-09, -7.43512509628434E-09,
                1.39050373566888E-08, -7.73483038941653E-09,],
            [-3.75396622353588E-10, 1.72079478841131E-09, -
                6.37646668050495E-09, 1.22941608598481E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.22941608598481E-08, -8.49031902641601E-09,
                1.58123119002383E-09, -3.54004509008713E-10,],
            [-7.19477267560949E-09, 1.23945827374776E-08, -
                6.53432234036163E-09, 1.33451227849347E-09,],
            [1.05233447833790E-09, -4.86743768888254E-09,
                6.74118845073356E-09, -2.92608524018892E-09,],
            [-2.08936498445707E-10, 8.70617741434581E-10, -
                1.86302253771705E-09, 1.20134129472817E-09,],
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


class TestUpdateGlobalMatricesCubic(unittest.TestCase):
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()
        self.msh.update_boundary_conditions(self.msh._t1)
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.update_global_matrices_and_vectors()

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
            [1.75973053787198E-09, -2.30012579221697E-09,
                7.04533370197087E-10, -1.64138115852098E-10,],
            [-1.59213591880079E-09, 3.81159727452406E-09, -
                2.74059958738664E-09, 5.21138231663364E-10,],
            [4.27905926569192E-10, -2.23565512579640E-09,
                3.37467740597141E-09, -1.56692820674420E-09,],
            [-9.52348424823319E-11, 3.75502889134307E-10, -
                1.26214863347869E-09, 2.26528300823914E-09,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [2.26528300823914E-09, -1.83605309846816E-09,
                7.40454657106645E-10, -1.87803980050906E-10,],
            [-1.45652110193580E-09, 4.53944671410106E-09, -
                3.93945857548988E-09, 8.56532963324615E-10,],
            [5.44362557557076E-10, -3.08043242328214E-09,
                7.05088438566790E-09, -4.51481451994284E-09,],
            [-1.06722513017259E-10, 5.09787707760499E-10, -
                3.59987343768192E-09, 6.77569446565902E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.77569446565902E-09, -4.81318466550500E-09,
                1.76001522428353E-09, -5.25716781498866E-10,],
            [-3.67644360509383E-09, 1.07093191546781E-08, -
                9.31245855882194E-09, 2.27958300923772E-09,],
            [1.26491812901207E-09, -7.43512509628434E-09,
                1.39050373566888E-08, -7.73483038941653E-09,],
            [-3.75396622353588E-10, 1.72079478841131E-09, -
                6.37646668050495E-09, 1.22941608598481E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.22941608598481E-08, -8.49031902641601E-09,
                1.58123119002383E-09, -3.54004509008713E-10,],
            [-7.19477267560949E-09, 1.23945827374776E-08, -
                6.53432234036163E-09, 1.33451227849347E-09,],
            [1.05233447833790E-09, -4.86743768888254E-09,
                6.74118845073356E-09, -2.92608524018892E-09,],
            [-2.08936498445707E-10, 8.70617741434581E-10, -
                1.86302253771705E-09, 1.20134129472817E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [1.75973053787198E-09, -2.30012579221697E-09,
                7.04533370197087E-10, -1.64138115852098E-10,],
            [-1.59213591880079E-09, 3.81159727452406E-09, -
                2.74059958738664E-09, 5.21138231663364E-10,],
            [4.27905926569192E-10, -2.23565512579640E-09,
                3.37467740597141E-09, -1.56692820674420E-09,],
            [-9.52348424823319E-11, 3.75502889134307E-10, -
                1.26214863347869E-09, 2.26528300823914E-09,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [2.26528300823914E-09, -1.83605309846816E-09,
                7.40454657106645E-10, -1.87803980050906E-10,],
            [-1.45652110193580E-09, 4.53944671410106E-09, -
                3.93945857548988E-09, 8.56532963324615E-10,],
            [5.44362557557076E-10, -3.08043242328214E-09,
                7.05088438566790E-09, -4.51481451994284E-09,],
            [-1.06722513017259E-10, 5.09787707760499E-10, -
                3.59987343768192E-09, 6.77569446565902E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.77569446565902E-09, -4.81318466550500E-09,
                1.76001522428353E-09, -5.25716781498866E-10,],
            [-3.67644360509383E-09, 1.07093191546781E-08, -
                9.31245855882194E-09, 2.27958300923772E-09,],
            [1.26491812901207E-09, -7.43512509628434E-09,
                1.39050373566888E-08, -7.73483038941653E-09,],
            [-3.75396622353588E-10, 1.72079478841131E-09, -
                6.37646668050495E-09, 1.22941608598481E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.22941608598481E-08, -8.49031902641601E-09,
                1.58123119002383E-09, -3.54004509008713E-10,],
            [-7.19477267560949E-09, 1.23945827374776E-08, -
                6.53432234036163E-09, 1.33451227849347E-09,],
            [1.05233447833790E-09, -4.86743768888254E-09,
                6.74118845073356E-09, -2.92608524018892E-09,],
            [-2.08936498445707E-10, 8.70617741434581E-10, -
                1.86302253771705E-09, 1.20134129472817E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()
        self.msh.calculate_solution_vector_correction()
        self.msh.update_nodes()
        self.msh.update_integration_points()
        self.msh.calculate_deformed_coords()
        self.msh.update_total_stress_distribution()
        self.msh.update_pore_pressure_distribution()
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
            0.453854101158058,
            0.406149823856105,
            0.424774092725406,
            0.478507896381547,
            0.539921881433598,
            0.589923849978126,
            0.620404709626031,
            0.630998832481112,
            0.626836773248559,
            0.614903228636769,
            0.601185593778019,
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
            0.564664120858189,
            0.485300044815805,
            0.420328827025129,
            0.405842530973992,
            0.418971321091586,
            0.431156274622978,
            0.460269348234202,
            0.509448128602048,
            0.557409611047819,
            0.584292486613682,
            0.595608085226750,
            0.613381989932307,
            0.627866953483600,
            0.631051644673805,
            0.628122445500971,
            0.625426968048645,
            0.618965643877480,
            0.607997664280284,
            0.597273660895473,
            0.591259322841753,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            8.59453990902811E-11,
            4.72076679832879E-11,
            2.89062770618013E-11,
            2.59117120229139E-11,
            2.86115457833692E-11,
            3.13683726887991E-11,
            3.90791122636761E-11,
            5.66482623121018E-11,
            8.13649486181668E-11,
            9.96730918271895E-11,
            1.08562104460021E-10,
            1.24151894439739E-10,
            1.38498488778276E-10,
            1.41868718279444E-10,
            1.38765886145065E-10,
            1.35970621420536E-10,
            1.29497210380214E-10,
            1.19206469040929E-10,
            1.09935804097795E-10,
            1.05055795615955E-10,
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
            6.48841294284279E-10,
            3.56392369092889E-10,
            2.18226762808063E-10,
            1.95619415862020E-10,
            2.16001701010828E-10,
            2.36814253589214E-10,
            2.95026168346824E-10,
            4.27663751947089E-10,
            6.14261369771860E-10,
            7.52477952176765E-10,
            8.19585191454775E-10,
            9.37279676734142E-10,
            1.04558870709201E-09,
            1.07103211761449E-09,
            1.04760741263520E-09,
            1.02650467530512E-09,
            9.77633921986210E-10,
            8.99944388859346E-10,
            8.29955880989822E-10,
            7.93114455468618E-10,
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
            1.91025165784574E+05,
            2.94178805227970E+05,
            4.20530968818130E+05,
            4.55967083877153E+05,
            4.23869519785997E+05,
            3.96362434510200E+05,
            3.38466523766492E+05,
            2.58478972446999E+05,
            1.20917123704650E+05,
            7.89163061032461E+04,
            6.87932700302537E+04,
            5.90687506735607E+04,
            6.13309182067299E+04,
            7.83500500247808E+04,
            9.83578373567696E+04,
            1.11250684844545E+05,
            1.40735439825549E+05,
            1.50870150198523E+05,
            1.60039696985277E+05,
            1.65391642060803E+05,
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
            -5.4981462390284E+06,
            -8.4671466449090E+06,
            -1.2103854249287E+07,
            -2.4938313782221E+06,
            -1.2199945470422E+07,
            -1.1408227914075E+07,
            -9.7418496512780E+06,
            -7.4396228601110E+06,
            -3.4802745816256E+06,
            -2.2713938753436E+06,
            -1.9800294758747E+06,
            -1.7001353095340E+06,
            -1.7652457250307E+06,
            -2.2550957152800E+06,
            -2.8309661259604E+06,
            -3.2020521063554E+06,
            -4.0506915724784E+06,
            -4.3423919853112E+06,
            -8.7530883744794E+05,
            -9.0458035514254E+05,
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
            4.55967083877153E+05,
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
            1.60039696985277E+05,
            1.65391642060803E+05,
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
            1.04815988091594E-09,
            6.06747757538620E-10,
            1.96841191401410E-10,
            -3.69656749220223E-11,
            -2.07685505669191E-10,
            -2.55000012859971E-10,
            -3.20922685177856E-10,
            -3.82957746322230E-10,
            -2.69693855220735E-10,
            -2.14621893524031E-10,
            -2.08194248270051E-10,
            -1.88743482831372E-10,
            -1.66523634181567E-10,
            -1.31694095936685E-10,
            -9.62249487940891E-11,
            -7.86634564126383E-11,
            -4.59234528231849E-11,
            -3.00734515583833E-11,
            -9.56248620640415E-11,
            -9.45682780848628E-11,
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
        expected = 2.848640581821470
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.84864058182147,
            10.10143913239300,
            17.38569893576430,
            24.95563968579680,
            32.90444420745060,
            41.17562115994050,
            49.64596427257840,
            58.18131530076630,
            66.69024063095680,
            75.12466256547930,
            83.47573295131850,
            91.75916179824910,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [1.75973053787198E-09, -2.30012579221697E-09,
                7.04533370197087E-10, -1.64138115852098E-10,],
            [-1.59213591880079E-09, 3.81159727452406E-09, -
                2.74059958738664E-09, 5.21138231663364E-10,],
            [4.27905926569192E-10, -2.23565512579640E-09,
                3.37467740597141E-09, -1.56692820674420E-09,],
            [-9.52348424823319E-11, 3.75502889134307E-10, -
                1.26214863347869E-09, 2.26528300823914E-09,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [2.26528300823914E-09, -1.83605309846816E-09,
                7.40454657106645E-10, -1.87803980050906E-10,],
            [-1.45652110193580E-09, 4.53944671410106E-09, -
                3.93945857548988E-09, 8.56532963324615E-10,],
            [5.44362557557076E-10, -3.08043242328214E-09,
                7.05088438566790E-09, -4.51481451994284E-09,],
            [-1.06722513017259E-10, 5.09787707760499E-10, -
                3.59987343768192E-09, 6.77569446565902E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.77569446565902E-09, -4.81318466550500E-09,
                1.76001522428353E-09, -5.25716781498866E-10,],
            [-3.67644360509383E-09, 1.07093191546781E-08, -
                9.31245855882194E-09, 2.27958300923772E-09,],
            [1.26491812901207E-09, -7.43512509628434E-09,
                1.39050373566888E-08, -7.73483038941653E-09,],
            [-3.75396622353588E-10, 1.72079478841131E-09, -
                6.37646668050495E-09, 1.22941608598481E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.22941608598481E-08, -8.49031902641601E-09,
                1.58123119002383E-09, -3.54004509008713E-10,],
            [-7.19477267560949E-09, 1.23945827374776E-08, -
                6.53432234036163E-09, 1.33451227849347E-09,],
            [1.05233447833790E-09, -4.86743768888254E-09,
                6.74118845073356E-09, -2.92608524018892E-09,],
            [-2.08936498445707E-10, 8.70617741434581E-10, -
                1.86302253771705E-09, 1.20134129472817E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [7.99981820101025E-09, -1.01894673336391E-08,
                3.11045101279929E-09, -9.20801880170398E-10,],
            [-9.48105793182293E-09, 1.98194016828149E-08, -
                1.38180271154403E-08, 3.47968336444829E-09,],
            [2.83370606696355E-09, -1.33128004050324E-08,
                1.76207116837943E-08, -7.14161734572542E-09,],
            [-8.51870099780680E-10, 3.33397513498178E-09, -
                6.83678612605386E-09, 1.02540628675021E-08,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [1.02540628675021E-08, -7.38301782290062E-09,
                1.87195584242427E-09, -3.88319796172991E-10,],
            [-7.00336745737915E-09, 1.65340048249067E-08, -
                1.11038621247090E-08, 1.57322475718146E-09,],
            [1.67579091135996E-09, -1.02446010170194E-08,
                1.37016738715110E-08, -5.13286376585155E-09,],
            [-3.07239243145768E-10, 1.22651380631321E-09, -
                4.21794311533726E-09, 6.88063266274482E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.88063266274482E-09, -4.81782794554112E-09,
                1.76229653724152E-09, -5.26432702275400E-10,],
            [-3.68139378760827E-09, 1.07186338072850E-08, -
                9.31936131341694E-09, 2.28212129374020E-09,],
            [1.26734209278860E-09, -7.44223741828343E-09,
                1.39169858225610E-08, -7.74209049706613E-09,],
            [-3.76164268686044E-10, 1.72348040248750E-09, -
                6.38413357140016E-09, 1.23098887070785E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.23098887070785E-08, -8.70102844142895E-09,
                1.79013259657214E-09, -3.62175424622944E-10,],
            [-7.40559696830144E-09, 1.79743041809617E-08, -
                1.21102817331927E-08, 1.54157452053249E-09,],
            [1.26130162906953E-09, -1.04432868782076E-08,
                1.23145694925129E-08, -3.13258424337486E-09,],
            [-2.17124602357813E-10, 1.07767945016753E-09, -
                2.06957446970536E-09, 1.20901962189565E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
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

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -6.38732084880805E-04,
            1.44491831770773E-04,
            -5.52546380220487E-06,
            1.02848327905909E-04,
            1.52786827961623E-05,
            6.04706119078349E-05,
            -7.66930275427312E-05,
            -5.59761927534883E-06,
            -2.24319887355370E-05,
            -5.34059291429223E-05,
            2.27541911840281E-05,
            -1.85165624598571E-06,
            3.08471909922187E-05,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-20,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000E+00,
            1.44238195553753E-04,
            -5.34032404933294E-06,
            1.02386184761444E-04,
            1.54259361185959E-05,
            6.02208841306168E-05,
            -7.61500218736402E-05,
            -5.73047779667614E-06,
            -2.24951778322459E-05,
            -5.29553480946092E-05,
            2.25357095020074E-05,
            -1.78193315476814E-06,
            0.00000000000000E+00,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            rtol=1e-10, atol=1e-13,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 1.05454217509602E-04
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
            e.assign_material(self.mtl)
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
        self.msh.time_step = 2.5920E+06
        self.msh.implicit_error_tolerance = 1.0e-6
        self.msh.initialize_time_step()
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
            0.454182190588450,
            0.406143596425568,
            0.424968321955909,
            0.478485684097224,
            0.539870977244042,
            0.589934685555729,
            0.620403127398863,
            0.630999043744059,
            0.626838122252571,
            0.614888810659179,
            0.601178588481216,
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
            0.564795204624881,
            0.485653316529554,
            0.420497735073141,
            0.405785442459778,
            0.419060145355718,
            0.431305045448459,
            0.460289220131515,
            0.509394184034985,
            0.557372590760917,
            0.584292812512017,
            0.595615756892312,
            0.613382024799882,
            0.627865420779875,
            0.631052966120878,
            0.628124231138156,
            0.625423812802375,
            0.618952513463569,
            0.607985529375687,
            0.597269949654532,
            0.591259245472496,
        ])
        actual_void_ratio_int_pts = np.array([
            ip.void_ratio for e in self.msh.elements for ip in e.int_pts
        ])
        self.assertTrue(np.allclose(actual_void_ratio_int_pts,
                                    expected_void_ratio_int_pts))

    def test_hyd_cond_distribution(self):
        expected_hyd_cond_int_pts = np.array([
            8.60304937495058E-11,
            4.73337393684061E-11,
            2.89431608297679E-11,
            2.59005468073177E-11,
            2.86307384098431E-11,
            3.14036235328896E-11,
            3.90849754331900E-11,
            5.66251968732000E-11,
            8.13422116634722E-11,
            9.96733370588035E-11,
            1.08568392225591E-10,
            1.24151927120414E-10,
            1.38496886209842E-10,
            1.41870133598764E-10,
            1.38767756804426E-10,
            1.35967382584062E-10,
            1.29484374278382E-10,
            1.19195548801848E-10,
            1.09932723974701E-10,
            1.05055734253298E-10,
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
            6.49483712952556E-10,
            3.57344139886418E-10,
            2.18505215313943E-10,
            1.95535124521063E-10,
            2.16146594963662E-10,
            2.37080378402717E-10,
            2.95070432106563E-10,
            4.27489620354372E-10,
            6.14089718057303E-10,
            7.52479803542859E-10,
            8.19632660685172E-10,
            9.37279923455572E-10,
            1.04557660856680E-09,
            1.07104280251012E-09,
            1.04762153510195E-09,
            1.02648022384092E-09,
            9.77537016357585E-10,
            8.99861946958622E-10,
            8.29932627725821E-10,
            7.93113992213729E-10,
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
            1.90305805953289E+05,
            2.91202757654730E+05,
            4.18491491964397E+05,
            4.56087845025202E+05,
            4.22787252641294E+05,
            3.94668851548341E+05,
            3.38033546677329E+05,
            2.58795712516139E+05,
            1.21046033134360E+05,
            7.89155658632343E+04,
            6.87780815832405E+04,
            5.90686913939944E+04,
            6.13336238651076E+04,
            7.83470700918177E+04,
            9.83527824082839E+04,
            1.11260788566292E+05,
            1.40788637134153E+05,
            1.50922853914382E+05,
            1.60042941474382E+05,
            1.65391710053520E+05,
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
            -5.4774413987283E+06,
            -8.3814891101817E+06,
            -1.2045153386776E+07,
            -1.3127263413135E+07,
            -1.2168795317997E+07,
            -1.1359482678054E+07,
            -9.7293875688891E+06,
            -7.4487393721304E+06,
            -3.4839848932655E+06,
            -2.2713725695234E+06,
            -1.9795923172287E+06,
            -1.7001336033310E+06,
            -1.7653236001388E+06,
            -2.2550099459147E+06,
            -2.8308206328475E+06,
            -3.2023429148438E+06,
            -4.0522227141006E+06,
            -4.3439089201922E+06,
            -4.6064061409729E+06,
            -4.7603560759253E+06,
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
            4.56114549918221E+05,
            4.24492378919279E+05,
            3.97167030147011E+05,
            3.38655507912587E+05,
            2.58803432604326E+05,
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
            1.60042946444720E+05,
            1.65391712515329E+05,
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
            1.03951525326723E-09,
            6.01230620942697E-10,
            1.97832065153565E-10,
            -6.86038326495958E-11,
            -2.10291282667955E-10,
            -2.52906571951263E-10,
            -3.19795376506816E-10,
            -3.83083054440100E-10,
            -2.70014596295161E-10,
            -2.14823338575539E-10,
            -2.08130627579038E-10,
            -1.88723530143204E-10,
            -1.66530270962597E-10,
            -1.31707310051738E-10,
            -9.62181566736934E-11,
            -7.85038014410202E-11,
            -4.58337484846749E-11,
            -3.00934976062113E-11,
            -3.38841802655689E-11,
            -4.74183459440503E-11,
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
        expected = 2.846633862203060
        actual = self.msh.calculate_total_settlement()
        self.assertAlmostEqual(expected, actual)

    def test_calculate_deformed_coords(self):
        expected = np.array([
            2.84663386220306,
            10.10073102090010,
            17.38583246739120,
            24.95577341998600,
            32.90494046155440,
            41.17585365736900,
            49.64606912698890,
            58.18143524218130,
            66.69035402123050,
            75.12478347599940,
            83.47580520162140,
            91.75917398761720,
            100.00000000000000,
        ])
        actual = self.msh.calculate_deformed_coords()
        self.assertTrue(np.allclose(
            expected, actual,
        ))

    def test_global_stiffness_matrix_0(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [1.75973053787198E-09, -2.30012579221697E-09,
                7.04533370197087E-10, -1.64138115852098E-10,],
            [-1.59213591880079E-09, 3.81159727452406E-09, -
                2.74059958738664E-09, 5.21138231663364E-10,],
            [4.27905926569192E-10, -2.23565512579640E-09,
                3.37467740597141E-09, -1.56692820674420E-09,],
            [-9.52348424823319E-11, 3.75502889134307E-10, -
                1.26214863347869E-09, 2.26528300823914E-09,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [2.26528300823914E-09, -1.83605309846816E-09,
                7.40454657106645E-10, -1.87803980050906E-10,],
            [-1.45652110193580E-09, 4.53944671410106E-09, -
                3.93945857548988E-09, 8.56532963324615E-10,],
            [5.44362557557076E-10, -3.08043242328214E-09,
                7.05088438566790E-09, -4.51481451994284E-09,],
            [-1.06722513017259E-10, 5.09787707760499E-10, -
                3.59987343768192E-09, 6.77569446565902E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.77569446565902E-09, -4.81318466550500E-09,
                1.76001522428353E-09, -5.25716781498866E-10,],
            [-3.67644360509383E-09, 1.07093191546781E-08, -
                9.31245855882194E-09, 2.27958300923772E-09,],
            [1.26491812901207E-09, -7.43512509628434E-09,
                1.39050373566888E-08, -7.73483038941653E-09,],
            [-3.75396622353588E-10, 1.72079478841131E-09, -
                6.37646668050495E-09, 1.22941608598481E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.22941608598481E-08, -8.49031902641601E-09,
                1.58123119002383E-09, -3.54004509008713E-10,],
            [-7.19477267560949E-09, 1.23945827374776E-08, -
                6.53432234036163E-09, 1.33451227849347E-09,],
            [1.05233447833790E-09, -4.86743768888254E-09,
                6.74118845073356E-09, -2.92608524018892E-09,],
            [-2.08936498445707E-10, 8.70617741434581E-10, -
                1.86302253771705E-09, 1.20134129472817E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix_0,
            atol=1e-18, rtol=1e-8,
        ))

    def test_global_stiffness_matrix(self):
        expected_K = np.zeros((self.msh.num_nodes, self.msh.num_nodes))
        expected_K[0:4, 0:4] = np.array([
            [8.00708217847523E-09, -1.03351359196918E-08,
                3.02746232303349E-09, -6.99408581816904E-10,],
            [-9.62578380725186E-09, 2.05723258085146E-08, -
                1.34002214499697E-08, 2.45367944870699E-09,],
            [2.75045350674614E-09, -1.28943464378698E-08,
                1.77462000754598E-08, -7.60230714433607E-09,],
            [-6.30415876920846E-10, 2.30782018751834E-09, -
                7.29739268126147E-09, 1.15044486993808E-08,],
        ])
        expected_K[3:7, 3:7] = np.array([
            [1.15044486993808E-08, -7.36313693885748E-09,
                1.86546360590346E-09, -3.86786995762820E-10,],
            [-6.98326392843456E-09, 1.65097257191276E-08, -
                1.10966527230662E-08, 1.57019093237309E-09,],
            [1.66923472690997E-09, -1.02376051229550E-08,
                1.37003441007487E-08, -5.13197370470367E-09,],
            [-3.05689350891802E-10, 1.22349582604047E-09, -
                4.21713884414633E-09, 6.88093572628891E-09,],
        ])
        expected_K[6:10, 6:10] = np.array([
            [6.88093572628891E-09, -4.81728203426525E-09,
                1.76205152811555E-09, -5.26372851141560E-10,],
            [-3.68081436347180E-09, 1.07178071365933E-08, -
                9.31897740703071E-09, 2.28198463390925E-09,],
            [1.26708253009608E-09, -7.44185950401363E-09,
                1.39165939221040E-08, -7.74181694818643E-09,],
            [-3.76100431534178E-10, 1.72333960931627E-09, -
                6.38384567049662E-09, 1.24355519350824E-08,],
        ])
        expected_K[9:13, 9:13] = np.array([
            [1.24355519350824E-08, -9.28985140847478E-09,
                2.51780761190463E-09, -6.26901645797758E-10,],
            [-7.99448424579749E-09, 2.07455491924533E-08, -
                1.56054896931546E-08, 2.85442474649882E-09,],
            [1.98899638078167E-09, -1.39386166690269E-08,
                2.22038461109541E-08, -1.02542258227089E-08,],
            [-4.81853689532122E-10, 2.39054464347521E-09, -
                9.19123015551347E-09, 7.28253920157039E-09,],
        ])
        self.assertTrue(np.allclose(
            expected_K, self.msh._stiffness_matrix,
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

    def test_global_residual_vector(self):
        expected_Psi = np.array([
            -1.83064271312768E-03,
            9.14939915822553E-11,
            1.56825672301055E-10,
            -6.61695184376535E-10,
            3.95087502734061E-10,
            -4.51379552787420E-10,
            1.68976184688465E-10,
            2.96589291587741E-10,
            -1.88996054164447E-09,
            6.11572180846457E-09,
            -7.33493166326544E-09,
            -5.17839196664039E-09,
            9.11510988279886E-05,
        ])
        self.assertTrue(np.allclose(
            expected_Psi, self.msh._residual_water_flux_vector,
            atol=1e-20,
        ))

    def test_void_ratio_increment_vector(self):
        expected_de = np.array([
            0.00000000000000E+00,
            9.24303360177107E-11,
            1.54201869695999E-10,
            -6.53970357956554E-10,
            3.91532053344058E-10,
            -4.51603447576533E-10,
            1.87095974186305E-10,
            2.84802053686619E-10,
            -1.85781299813878E-09,
            6.00617458442423E-09,
            -7.29253980877351E-09,
            -5.17861058420076E-09,
            0.00000000000000E+00,
        ])
        self.assertTrue(np.allclose(
            expected_de, self.msh._delta_void_ratio_vector,
            rtol=1e-10, atol=1e-13,
        ))

    def test_iteration_variables(self):
        expected_eps_a = 5.46588164654307E-09
        self.assertAlmostEqual(self.msh._eps_a, expected_eps_a)
        self.assertEqual(self.msh._iter, 8)


if __name__ == "__main__":
    unittest.main()
