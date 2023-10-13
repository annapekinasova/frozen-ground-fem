import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Point1D,
    Node1D,
    IntegrationPoint1D,
    Element1D,
    Boundary1D,
    Mesh1D,
    shape_matrix_linear,
    gradient_matrix_linear,
)
from frozen_ground_fem.materials import (
    Material,
    NULL_MATERIAL,
)


class TestPoint1DDefaults(unittest.TestCase):
    def setUp(self):
        self.p = Point1D()

    def test_z_value(self):
        self.assertEqual(self.p.z, 0.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.zeros((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))


class TestPoint1DInitializers(unittest.TestCase):
    def setUp(self):
        self.p = Point1D(1.0)

    def test_z_value(self):
        self.assertEqual(self.p.z, 1.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.ones((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))


class TestPoint1DSetters(unittest.TestCase):
    def setUp(self):
        self.p = Point1D()

    def test_set_z_valid_float(self):
        self.p.z = 1.0
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int(self):
        self.p.z = 1
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int_type(self):
        self.p.z = 1
        self.assertIsInstance(self.p.z, float)

    def test_set_z_valid_str(self):
        self.p.z = "1.e5"
        self.assertEqual(self.p.z, 1.0e5)

    def test_set_z_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.z = "five"

    def test_set_coords_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.coords = 1.0


class TestNode1DDefaults(unittest.TestCase):
    def setUp(self):
        self.p = Node1D(0)

    def test_index_value(self):
        self.assertEqual(self.p.index, 0)

    def test_index_type(self):
        self.assertIsInstance(self.p.index, int)

    def test_z_value(self):
        self.assertEqual(self.p.z, 0.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.zeros((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))

    def test_temp_value(self):
        self.assertEqual(self.p.temp, 0.0)

    def test_temp_type(self):
        self.assertIsInstance(self.p.temp, float)


class TestNode1DInitializers(unittest.TestCase):
    def setUp(self):
        self.p = Node1D(0, 1.0, -5.0)

    def test_index_value(self):
        self.assertEqual(self.p.index, 0)

    def test_index_type(self):
        self.assertIsInstance(self.p.index, int)

    def test_z_value(self):
        self.assertEqual(self.p.z, 1.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.ones((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))

    def test_temp_value(self):
        self.assertEqual(self.p.temp, -5.0)

    def test_temp_type(self):
        self.assertIsInstance(self.p.temp, float)


class TestNode1DSetters(unittest.TestCase):
    def setUp(self):
        self.p = Node1D(0)

    def test_set_index_valid_int(self):
        self.p.index = 2
        self.assertEqual(self.p.index, 2)

    def test_set_index_invalid_int(self):
        with self.assertRaises(ValueError):
            self.p.index = -2

    def test_set_index_invalid_float(self):
        with self.assertRaises(TypeError):
            self.p.index = 2.1

    def test_set_index_valid_str(self):
        self.p.index = "2"
        self.assertEqual(self.p.index, 2)

    def test_set_index_invalid_str_float(self):
        with self.assertRaises(ValueError):
            self.p.index = "2.1"

    def test_set_index_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.index = "two"

    def test_set_z_valid_float(self):
        self.p.z = 1.0
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int(self):
        self.p.z = 1
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int_type(self):
        self.p.z = 1
        self.assertIsInstance(self.p.z, float)

    def test_set_z_valid_str(self):
        self.p.z = "1.e5"
        self.assertEqual(self.p.z, 1.0e5)

    def test_set_z_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.z = "five"

    def test_set_coords_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.coords = 1.0

    def test_set_temp_valid_float(self):
        self.p.temp = 1.0
        self.assertEqual(self.p.temp, 1.0)

    def test_set_temp_valid_int(self):
        self.p.temp = 1
        self.assertEqual(self.p.temp, 1.0)

    def test_set_temp_valid_int_type(self):
        self.p.temp = 1
        self.assertIsInstance(self.p.temp, float)

    def test_set_temp_valid_str(self):
        self.p.temp = "1.e5"
        self.assertEqual(self.p.temp, 1.0e5)

    def test_set_temp_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.temp = "five"


class TestIntegrationPoint1DDefaults(unittest.TestCase):
    def setUp(self):
        self.p = IntegrationPoint1D()

    def test_z_value(self):
        self.assertEqual(self.p.z, 0.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.zeros((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))

    def test_local_coord_value(self):
        self.assertEqual(self.p.local_coord, 0.0)

    def test_local_coord_type(self):
        self.assertIsInstance(self.p.local_coord, float)

    def test_weight_value(self):
        self.assertEqual(self.p.weight, 0.0)

    def test_weight_type(self):
        self.assertIsInstance(self.p.weight, float)

    def test_void_ratio_value(self):
        self.assertAlmostEqual(self.p.void_ratio, 0.0)
        self.assertAlmostEqual(self.p.porosity, 0.0)

    def test_void_ratio_type(self):
        self.assertIsInstance(self.p.void_ratio, float)
        self.assertIsInstance(self.p.porosity, float)

    def test_deg_sat_water_value(self):
        self.assertAlmostEqual(self.p.deg_sat_water, 1.0)
        self.assertAlmostEqual(self.p.deg_sat_ice, 0.0)
        self.assertAlmostEqual(self.p.vol_ice_cont, 0.0)

    def test_vol_ice_cont_type(self):
        self.assertIsInstance(self.p.deg_sat_water, float)
        self.assertIsInstance(self.p.deg_sat_ice, float)
        self.assertIsInstance(self.p.vol_ice_cont, float)

    def test_material_value(self):
        self.assertIs(self.p.material, NULL_MATERIAL)

    def test_material_type(self):
        self.assertIsInstance(self.p.material, Material)


class TestIntegrationPoint1DInitializers(unittest.TestCase):
    def setUp(self):
        self.m = Material(
            thrm_cond_solids=7.8,
            spec_grav_solids=2.5,
            spec_heat_cap_solids=7.41e5,
        )
        self.p = IntegrationPoint1D(
            coord=1.0,
            local_coord=-0.33,
            weight=1.0,
            void_ratio=0.5,
            deg_sat_water=0.2,
            material=self.m,
        )

    def test_z_value(self):
        self.assertEqual(self.p.z, 1.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.ones((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))

    def test_local_coord_value(self):
        self.assertEqual(self.p.local_coord, -0.33)

    def test_local_coord_type(self):
        self.assertIsInstance(self.p.local_coord, float)

    def test_weight_value(self):
        self.assertEqual(self.p.weight, 1.0)

    def test_weight_type(self):
        self.assertIsInstance(self.p.weight, float)

    def test_void_ratio_value(self):
        self.assertEqual(self.p.void_ratio, 0.5)
        expected_porosity = 0.5 / 1.5
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_void_ratio_type(self):
        self.assertIsInstance(self.p.void_ratio, float)
        self.assertIsInstance(self.p.porosity, float)

    def test_deg_sat_water_value(self):
        self.assertEqual(self.p.deg_sat_water, 0.2)
        expected_deg_sat_ice = 0.8
        self.assertAlmostEqual(self.p.deg_sat_ice, expected_deg_sat_ice)
        expected_vol_ice_cont = 0.5 * 0.8 / 1.5
        self.assertAlmostEqual(self.p.vol_ice_cont, expected_vol_ice_cont)

    def test_vol_ice_cont_type(self):
        self.assertIsInstance(self.p.deg_sat_water, float)
        self.assertIsInstance(self.p.deg_sat_ice, float)
        self.assertIsInstance(self.p.vol_ice_cont, float)

    def test_material_value(self):
        self.assertIs(self.p.material, self.m)

    def test_material_type(self):
        self.assertIsInstance(self.p.material, Material)

    def test_thrm_cond(self):
        expected = 4.682284029228440
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_vol_heat_cap(self):
        expected = 1235781866.66667
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)


class TestIntegrationPoint1DSetters(unittest.TestCase):
    def setUp(self):
        self.m = Material(
            thrm_cond_solids=7.8,
            spec_grav_solids=2.5,
            spec_heat_cap_solids=7.41e5,
        )
        self.p = IntegrationPoint1D(
            coord=1.0,
            local_coord=-0.33,
            weight=1.0,
            void_ratio=0.3,
            deg_sat_water=0.2,
            material=self.m,
        )

    def test_set_z_valid_float(self):
        self.p.z = 1.0
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int(self):
        self.p.z = 1
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int_type(self):
        self.p.z = 1
        self.assertIsInstance(self.p.z, float)

    def test_set_z_valid_str(self):
        self.p.z = "1.e5"
        self.assertEqual(self.p.z, 1.0e5)

    def test_set_z_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.z = "five"

    def test_set_coords_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.coords = 1.0

    def test_set_local_coord_valid_float(self):
        self.p.local_coord = 1.0
        self.assertEqual(self.p.local_coord, 1.0)

    def test_set_local_coord_valid_int(self):
        self.p.local_coord = 1
        self.assertEqual(self.p.local_coord, 1.0)

    def test_set_local_coord_valid_int_type(self):
        self.p.local_coord = 1
        self.assertIsInstance(self.p.local_coord, float)

    def test_set_local_coord_valid_str(self):
        self.p.local_coord = "1.e0"
        self.assertEqual(self.p.local_coord, 1.0)

    def test_set_local_coord_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.local_coord = "five"

    def test_set_weight_valid_float(self):
        self.p.weight = 2.0
        self.assertEqual(self.p.weight, 2.0)

    def test_set_weight_valid_int(self):
        self.p.weight = 2
        self.assertEqual(self.p.weight, 2.0)

    def test_set_weight_valid_int_type(self):
        self.p.weight = 2
        self.assertIsInstance(self.p.weight, float)

    def test_set_weight_valid_str(self):
        self.p.weight = "1.e0"
        self.assertEqual(self.p.weight, 1.0)

    def test_set_weight_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.weight = "five"

    def test_set_ratio_valid_float(self):
        self.p.void_ratio = 0.5
        self.assertAlmostEqual(self.p.void_ratio, 0.5)
        expected_porosity = 0.5 / 1.5
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_set_void_ratio_valid_float(self):
        self.p.void_ratio = 0.5
        self.assertEqual(self.p.void_ratio, 0.5)
        self.assertIsInstance(self.p.void_ratio, float)
        self.assertIsInstance(self.p.porosity, float)

    def test_set_void_ratio_valid_float_edge_0(self):
        self.p.void_ratio = 0.0
        self.assertEqual(self.p.void_ratio, 0.0)
        expected_porosity = 0.0 / 1.0
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_set_porosity_valid_float_edge_1(self):
        self.p.void_ratio = 1.0
        self.assertEqual(self.p.void_ratio, 1.0)
        expected_porosity = 1.0 / (1 + 1)
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_set_void_ratio_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio = -0.2

    def test_set_void_ratio_valid_int(self):
        self.p.void_ratio = 1
        self.assertEqual(self.p.void_ratio, 1.0)

    def test_set_void_ratio_valid_int_type(self):
        self.p.void_ratio = 1
        self.assertIsInstance(self.p.void_ratio, float)

    def test_set_void_ratio_valid_str(self):
        self.p.void_ratio = "1.e-1"
        self.assertEqual(self.p.void_ratio, 1.0e-1)

    def test_set_void_ratio_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio = "five"

    def test_set_deg_sat_water_valid_float(self):
        self.p.deg_sat_water = 0.2
        self.assertEqual(self.p.deg_sat_water, 0.2)

    def test_set_deg_sat_water_valid_float_edge_0(self):
        self.p.deg_sat_water = 0.0
        self.assertEqual(self.p.deg_sat_water, 0.0)

    def test_set_deg_sat_water_valid_float_edge_1(self):
        self.p.deg_sat_water = 1.0
        self.assertEqual(self.p.deg_sat_water, 1.0)

    def test_set_deg_sat_water_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = -0.2

    def test_set_deg_sat_water_invalid_float_positive(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = 1.1

    def test_set_deg_sat_water_valid_int(self):
        self.p.deg_sat_water = 0
        self.assertEqual(self.p.deg_sat_water, 0.0)

    def test_set_deg_sat_water_valid_int_type(self):
        self.p.deg_sat_water = 0
        self.assertIsInstance(self.p.deg_sat_water, float)

    def test_set_deg_sat_water_valid_str(self):
        self.p.deg_sat_water = "1.e-1"
        self.assertEqual(self.p.deg_sat_water, 1.0e-1)

    def test_set_deg_sat_water_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = "five"

    def test_set_material_valid(self):
        m = Material()
        self.p.material = m
        self.assertIs(self.p.material, m)

    def test_set_material_invalid(self):
        with self.assertRaises(TypeError):
            self.p.material = 1

    def test_set_thrm_cond_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.thrm_cond = 1.0e5

    def test_update_thrm_cond_void_ratio(self):
        self.p.void_ratio = 0.25
        expected = 5.74265192951243
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_update_thrm_cond_deg_sat_water(self):
        self.p.deg_sat_water = 0.05
        expected = 5.744855338606900
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_update_thrm_cond_material(self):
        self.p.material = Material(
            thrm_cond_solids=6.7,
            spec_grav_solids=2.8,
            spec_heat_cap_solids=6.43e5,
        )
        expected = 4.873817313136410
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_set_vol_heat_cap_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.vol_heat_cap = 1.0e5

    def test_update_vol_heat_cap_void_ratio(self):
        self.p.void_ratio = 0.25
        expected = 1482469120.0000
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)

    def test_update_vol_heat_cap_deg_sat_water(self):
        self.p.deg_sat_water = 0.05
        expected = 1425460880.769230
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)

    def test_update_vol_heat_cap_material(self):
        self.p.material = Material(
            thrm_cond_solids=6.7,
            spec_grav_solids=2.8,
            spec_heat_cap_solids=6.43e5,
        )
        expected = 1385464369.230770
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)


class TestElement1D(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.e = Element1D(self.nodes, order=1)

    def test_initialize_without_nodes(self):
        with self.assertRaises(TypeError):
            Element1D(order=1)

    def test_initialize_valid_nodes_value(self):
        self.assertEqual(self.e.nodes[1].z, 3.0)

    def test_initialize_valid_nodes_type(self):
        nodes = list(self.nodes)
        e = Element1D(nodes, order=1)
        self.assertIsInstance(e.nodes, tuple)

    def test_initialize_too_few_nodes(self):
        with self.assertRaises(ValueError):
            nodes = (Node1D(0),)
            Element1D(nodes, order=1)

    def test_initialize_too_many_nodes(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(3))
            Element1D(nodes, order=1)

    def test_initialize_invalid_nodes(self):
        with self.assertRaises(TypeError):
            nodes = tuple(k for k in range(2))
            Element1D(nodes, order=1)

    def test_jacobian_value(self):
        self.assertEqual(self.e.jacobian, 2.0)

    def test_set_jacobian(self):
        with self.assertRaises(AttributeError):
            self.e.jacobian = 5.0

    def test_int_pt_local_coords(self):
        expected = np.array([0.211324865405187, 0.788675134594813])
        actual = np.array([ip.local_coord for ip in self.e.int_pts])
        self.assertTrue(np.allclose(actual, expected))

    def test_int_pt_weights(self):
        expected = np.array([0.5, 0.5])
        actual = np.array([ip.weight for ip in self.e.int_pts])
        self.assertTrue(np.allclose(actual, expected))

    def test_int_pt_global_coords(self):
        expected = np.array([1.42264973081037, 2.57735026918963])
        actual = np.array([ip.z for ip in self.e.int_pts])
        self.assertTrue(np.allclose(actual, expected))

    def test_int_pt_type(self):
        self.assertIsInstance(self.e.int_pts, tuple)

    def test_set_int_pt(self):
        with self.assertRaises(AttributeError):
            self.e.int_pts = 3


class TestShapeMatrix(unittest.TestCase):
    def setUp(self):
        self.N = shape_matrix_linear(0.8)
        self.T_1D = np.array([5.0, 10.0])
        self.T_column = np.array([[5.0], [10.0]])

    def test_shape_matrix_linear_valid_float(self):
        expected = np.array([[0.2, 0.8]])
        self.assertTrue(np.allclose(self.N, expected))

    def test_shape_matrix_linear_shape(self):
        expected = (1, 2)
        self.assertEqual(self.N.shape, expected)

    def test_shape_matrix_linear_multiply_1D(self):
        expected = 9.0
        actual = self.N @ self.T_1D
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_shape_matrix_linear_multiply_column(self):
        expected = 9.0
        actual = self.N @ self.T_column
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_shape_matrix_linear_multiply_transpose(self):
        expected = np.array([[0.04, 0.16], [0.16, 0.64]])
        actual = self.N.T @ self.N
        self.assertTrue(np.allclose(expected, actual))

    def test_shape_matrix_linear_valid_str(self):
        expected = np.array([[0.2, 0.8]])
        self.assertTrue(np.allclose(shape_matrix_linear("8.e-1"), expected))

    def test_shape_matrix_linear_invalid_str(self):
        with self.assertRaises(ValueError):
            shape_matrix_linear("three")


class TestGradientMatrix(unittest.TestCase):
    def setUp(self):
        self.B = gradient_matrix_linear(0.8, 2.0)
        self.T_1D = np.array([5.0, 10.0])
        self.T_column = np.array([[5.0], [10.0]])

    def test_gradient_matrix_linear_valid_float(self):
        expected = np.array([[-0.5, 0.5]])
        self.assertTrue(np.allclose(self.B, expected))

    def test_gradient_matrix_linear_shape(self):
        expected = (1, 2)
        self.assertEqual(self.B.shape, expected)

    def test_gradient_matrix_linear_multiply_1D(self):
        expected = 2.5
        actual = self.B @ self.T_1D
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_gradient_matrix_linear_multiply_column(self):
        expected = 2.5
        actual = self.B @ self.T_column
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_gradient_matrix_linear_multiply_transpose(self):
        expected = np.array([[0.25, -0.25], [-0.25, 0.25]])
        actual = self.B.T @ self.B
        self.assertTrue(np.allclose(expected, actual))

    def test_gradient_matrix_linear_valid_str(self):
        expected = np.array([[-0.5, 0.5]])
        self.assertTrue(np.allclose(
            gradient_matrix_linear("8.e-1", "2.e0"), expected))

    def test_gradient_matrix_linear_invalid_str_arg0(self):
        with self.assertRaises(ValueError):
            gradient_matrix_linear("three", 2.0)

    def test_gradient_matrix_linear_invalid_str_arg1(self):
        with self.assertRaises(ValueError):
            gradient_matrix_linear(1.0, "three")


class TestBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 0.5),)
        self.e = Boundary1D(self.nodes)

    def test_initialize_without_nodes(self):
        with self.assertRaises(TypeError):
            Boundary1D()

    def test_initialize_valid_nodes_value(self):
        self.assertEqual(self.e.nodes[0].z, 0.5)

    def test_initialize_valid_nodes_type(self):
        nodes = list(self.nodes)
        e = Boundary1D(nodes)
        self.assertIsInstance(e.nodes, tuple)

    def test_initialize_too_many_nodes(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
            Boundary1D(nodes)

    def test_initialize_invalid_nodes(self):
        with self.assertRaises(TypeError):
            nodes = tuple(k for k in range(1))
            Boundary1D(nodes)


class TestMesh1D(unittest.TestCase):
    def setUp(self):
        pass

    def test_create_mesh_no_args(self):
        msh = Mesh1D(order=1)
        self.assertFalse(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 0)
        self.assertEqual(msh.num_elements, 0)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertTrue(np.isinf(msh.z_min))
        self.assertTrue(np.isinf(msh.z_max))
        self.assertTrue(msh.z_min < 0)
        self.assertTrue(msh.z_max > 0)
        self.assertEqual(msh.grid_size, 0.0)

    def test_create_mesh_z_range_generate(self):
        msh = Mesh1D(z_range=(100, -8), num_elements=9, generate=True, order=1)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 10)
        self.assertEqual(msh.num_elements, 9)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)
        self.assertEqual(msh.grid_size, 0.0)
        self.assertAlmostEqual(msh.nodes[1].z - msh.nodes[0].z, 12.0)

    def test_z_min_max_setters(self):
        msh = Mesh1D((100, -8), order=1)
        self.assertAlmostEqual(msh.z_min, -8.0)
        self.assertAlmostEqual(msh.z_max, 100.0)
        with self.assertRaises(ValueError):
            msh.z_min = "twelve"
        with self.assertRaises(ValueError):
            msh.z_min = 101.0
        msh.z_min = -7
        self.assertAlmostEqual(msh.z_min, -7.0)
        self.assertIsInstance(msh.z_min, float)
        with self.assertRaises(ValueError):
            msh.z_max = "twelve"
        with self.assertRaises(ValueError):
            msh.z_max = -8.0
        msh.z_max = 101
        self.assertAlmostEqual(msh.z_max, 101.0)
        self.assertIsInstance(msh.z_max, float)

    def test_grid_size_setter(self):
        msh = Mesh1D((100, -8), order=1)
        self.assertEqual(msh.grid_size, 0.0)
        msh.grid_size = 1
        self.assertAlmostEqual(msh.grid_size, 1.0)
        self.assertIsInstance(msh.grid_size, float)
        with self.assertRaises(ValueError):
            msh.grid_size = "twelve"
        with self.assertRaises(ValueError):
            msh.grid_size = -0.5
        msh.generate_mesh(order=1)
        self.assertEqual(msh.num_nodes, 109)
        self.assertEqual(msh.num_elements, 108)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.nodes[1].z - msh.nodes[0].z, 1.0)

    def test_generate_mesh(self):
        msh = Mesh1D(order=1, num_elements=9)
        self.assertFalse(msh.mesh_valid)
        with self.assertRaises(ValueError):
            msh.generate_mesh(order=1)
        with self.assertRaises(ValueError):
            Mesh1D(generate=True, order=1)
        msh.grid_size = np.inf
        msh.z_min = -8
        msh.z_max = 100
        with self.assertRaises(ValueError):
            msh.generate_mesh(order=1)
        msh.grid_size = 0
        msh.generate_mesh(num_elements=9, order=1)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 10)
        self.assertEqual(msh.num_elements, 9)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.elements[0].jacobian, 12.0)
        msh.grid_size = 1
        self.assertFalse(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 0)
        self.assertEqual(msh.num_elements, 0)
        self.assertEqual(msh.num_boundaries, 0)
        msh.generate_mesh(order=1)
        self.assertTrue(msh.mesh_valid)
        self.assertEqual(msh.num_nodes, 109)
        self.assertEqual(msh.num_elements, 108)
        self.assertEqual(msh.num_boundaries, 0)
        self.assertAlmostEqual(msh.elements[0].jacobian, 1.0)


if __name__ == "__main__":
    unittest.main()
