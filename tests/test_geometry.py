import unittest

import numpy as np

from frozen_ground_fem.geometry import (
        Point1D,
        Node1D,
        IntegrationPoint1D,
        Element1D,
        shape_matrix,
        gradient_matrix,
        )
from frozen_ground_fem.materials import (
        Material,
        NULL_MATERIAL,
        )


class TestPoint1DDefaults(unittest.TestCase):

    def setUp(self):
        self.p = Point1D()

    def test_z_value(self):
        self.assertEqual(self.p.z, 0.)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords,
                                       np.zeros((1, ))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1, ))


class TestPoint1DInitializers(unittest.TestCase):

    def setUp(self):
        self.p = Point1D(1.)

    def test_z_value(self):
        self.assertEqual(self.p.z, 1.)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords,
                                       np.ones((1, ))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1, ))


class TestPoint1DSetters(unittest.TestCase):

    def setUp(self):
        self.p = Point1D()

    def test_set_z_valid_float(self):
        self.p.z = 1.
        self.assertEqual(self.p.z, 1.)

    def test_set_z_valid_int(self):
        self.p.z = 1
        self.assertEqual(self.p.z, 1.)

    def test_set_z_valid_int_type(self):
        self.p.z = 1
        self.assertIsInstance(self.p.z, float)

    def test_set_z_valid_str(self):
        self.p.z = "1.e5"
        self.assertEqual(self.p.z, 1.e5)

    def test_set_z_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.z = "five"

    def test_set_coords_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.coords = 1.


class TestNode1DDefaults(unittest.TestCase):

    def setUp(self):
        self.p = Node1D(0)

    def test_index_value(self):
        self.assertEqual(self.p.index, 0)

    def test_index_type(self):
        self.assertIsInstance(self.p.index, int)

    def test_z_value(self):
        self.assertEqual(self.p.z, 0.)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords,
                                       np.zeros((1, ))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1, ))

    def test_temp_value(self):
        self.assertEqual(self.p.temp, 0.)

    def test_temp_type(self):
        self.assertIsInstance(self.p.temp, float)


class TestNode1DInitializers(unittest.TestCase):

    def setUp(self):
        self.p = Node1D(0, 1., -5.)

    def test_index_value(self):
        self.assertEqual(self.p.index, 0)

    def test_index_type(self):
        self.assertIsInstance(self.p.index, int)

    def test_z_value(self):
        self.assertEqual(self.p.z, 1.)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords,
                                       np.ones((1, ))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1, ))

    def test_temp_value(self):
        self.assertEqual(self.p.temp, -5.)

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
        self.p.index = '2'
        self.assertEqual(self.p.index, 2)

    def test_set_index_invalid_str_float(self):
        with self.assertRaises(ValueError):
            self.p.index = '2.1'

    def test_set_index_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.index = 'two'

    def test_set_z_valid_float(self):
        self.p.z = 1.
        self.assertEqual(self.p.z, 1.)

    def test_set_z_valid_int(self):
        self.p.z = 1
        self.assertEqual(self.p.z, 1.)

    def test_set_z_valid_int_type(self):
        self.p.z = 1
        self.assertIsInstance(self.p.z, float)

    def test_set_z_valid_str(self):
        self.p.z = "1.e5"
        self.assertEqual(self.p.z, 1.e5)

    def test_set_z_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.z = "five"

    def test_set_coords_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.coords = 1.

    def test_set_temp_valid_float(self):
        self.p.temp = 1.
        self.assertEqual(self.p.temp, 1.)

    def test_set_temp_valid_int(self):
        self.p.temp = 1
        self.assertEqual(self.p.temp, 1.)

    def test_set_temp_valid_int_type(self):
        self.p.temp = 1
        self.assertIsInstance(self.p.temp, float)

    def test_set_temp_valid_str(self):
        self.p.temp = "1.e5"
        self.assertEqual(self.p.temp, 1.e5)

    def test_set_temp_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.temp = "five"


class TestIntegrationPoint1DDefaults(unittest.TestCase):

    def setUp(self):
        self.p = IntegrationPoint1D()

    def test_z_value(self):
        self.assertEqual(self.p.z, 0.)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords,
                                       np.zeros((1, ))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1, ))

    def test_local_coord_value(self):
        self.assertEqual(self.p.local_coord, 0.)

    def test_local_coord_type(self):
        self.assertIsInstance(self.p.local_coord, float)

    def test_weight_value(self):
        self.assertEqual(self.p.weight, 0.)

    def test_weight_type(self):
        self.assertIsInstance(self.p.weight, float)

    def test_porosity_value(self):
        self.assertEqual(self.p.porosity, 0.)

    def test_porosity_type(self):
        self.assertIsInstance(self.p.porosity, float)

    def test_vol_ice_cont_value(self):
        self.assertEqual(self.p.vol_ice_cont, 0.)

    def test_vol_ice_cont_type(self):
        self.assertIsInstance(self.p.vol_ice_cont, float)

    def test_material_value(self):
        self.assertIs(self.p.material, NULL_MATERIAL)

    def test_material_type(self):
        self.assertIsInstance(self.p.material, Material)


class TestIntegrationPoint1DInitializers(unittest.TestCase):

    def setUp(self):
        self.m = Material(thrm_cond_solids=7.8,
                          dens_solids=2.5e3,
                          spec_heat_cap_solids=7.41e5)
        self.p = IntegrationPoint1D(1., -0.33, 1.0, 0.5, 0.2, self.m)

    def test_z_value(self):
        self.assertEqual(self.p.z, 1.)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords,
                                       np.ones((1, ))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1, ))

    def test_local_coord_value(self):
        self.assertEqual(self.p.local_coord, -0.33)

    def test_local_coord_type(self):
        self.assertIsInstance(self.p.local_coord, float)

    def test_weight_value(self):
        self.assertEqual(self.p.weight, 1.)

    def test_weight_type(self):
        self.assertIsInstance(self.p.weight, float)

    def test_porosity_value(self):
        self.assertEqual(self.p.porosity, 0.5)

    def test_porosity_type(self):
        self.assertIsInstance(self.p.porosity, float)

    def test_vol_ice_cont_value(self):
        self.assertEqual(self.p.vol_ice_cont, 0.2)

    def test_vol_ice_cont_type(self):
        self.assertIsInstance(self.p.vol_ice_cont, float)

    def test_material_value(self):
        self.assertIs(self.p.material, self.m)

    def test_material_type(self):
        self.assertIsInstance(self.p.material, Material)

    def test_thrm_cond(self):
        expected = 2.75721361713449
        self.assertAlmostEqual(self.p.thrm_cond, expected, delta=1e-8)

    def test_vol_heat_cap(self):
        expected = 9.278874e08
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, delta=1e-8)


class TestIntegrationPoint1DSetters(unittest.TestCase):

    def setUp(self):
        self.m = Material(thrm_cond_solids=7.8,
                          dens_solids=2.5e3,
                          spec_heat_cap_solids=7.41e5)
        self.p = IntegrationPoint1D(1., -0.33, 1.0, 0.3, 0.2, self.m)

    def test_set_z_valid_float(self):
        self.p.z = 1.
        self.assertEqual(self.p.z, 1.)

    def test_set_z_valid_int(self):
        self.p.z = 1
        self.assertEqual(self.p.z, 1.)

    def test_set_z_valid_int_type(self):
        self.p.z = 1
        self.assertIsInstance(self.p.z, float)

    def test_set_z_valid_str(self):
        self.p.z = "1.e5"
        self.assertEqual(self.p.z, 1.e5)

    def test_set_z_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.z = "five"

    def test_set_coords_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.coords = 1.

    def test_set_local_coord_valid_float(self):
        self.p.local_coord = 1.
        self.assertEqual(self.p.local_coord, 1.)

    def test_set_local_coord_valid_int(self):
        self.p.local_coord = 1
        self.assertEqual(self.p.local_coord, 1.)

    def test_set_local_coord_valid_int_type(self):
        self.p.local_coord = 1
        self.assertIsInstance(self.p.local_coord, float)

    def test_set_local_coord_valid_str(self):
        self.p.local_coord = "1.e0"
        self.assertEqual(self.p.local_coord, 1.)

    def test_set_local_coord_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.local_coord = "five"

    def test_set_weight_valid_float(self):
        self.p.weight = 2.
        self.assertEqual(self.p.weight, 2.)

    def test_set_weight_valid_int(self):
        self.p.weight = 2
        self.assertEqual(self.p.weight, 2.)

    def test_set_weight_valid_int_type(self):
        self.p.weight = 2
        self.assertIsInstance(self.p.weight, float)

    def test_set_weight_valid_str(self):
        self.p.weight = "1.e0"
        self.assertEqual(self.p.weight, 1.)

    def test_set_weight_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.weight = "five"

    def test_set_porosity_valid_float(self):
        self.p.porosity = 0.5
        self.assertEqual(self.p.porosity, 0.5)

    def test_set_porosity_valid_float_edge_0(self):
        self.p.porosity = 0.
        self.assertEqual(self.p.porosity, 0.)

    def test_set_porosity_valid_float_edge_1(self):
        self.p.porosity = 1.
        self.assertEqual(self.p.porosity, 1.)

    def test_set_porosity_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.porosity = -0.2

    def test_set_porosity_invalid_float_positive(self):
        with self.assertRaises(ValueError):
            self.p.porosity = 1.2

    def test_set_porosity_valid_int(self):
        self.p.porosity = 1
        self.assertEqual(self.p.porosity, 1.)

    def test_set_porosity_valid_int_type(self):
        self.p.porosity = 1
        self.assertIsInstance(self.p.porosity, float)

    def test_set_porosity_valid_str(self):
        self.p.porosity = "1.e-1"
        self.assertEqual(self.p.porosity, 1.e-1)

    def test_set_porosity_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.porosity = "five"

    def test_set_vol_ice_cont_valid_float(self):
        self.p.vol_ice_cont = 0.2
        self.assertEqual(self.p.vol_ice_cont, 0.2)

    def test_set_vol_ice_cont_valid_float_edge_0(self):
        self.p.vol_ice_cont = 0.
        self.assertEqual(self.p.vol_ice_cont, 0.)

    def test_set_vol_ice_cont_valid_float_edge_1(self):
        self.p.vol_ice_cont = self.p.porosity
        self.assertEqual(self.p.vol_ice_cont, self.p.porosity)

    def test_set_vol_ice_cont_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.vol_ice_cont = -0.2

    def test_set_vol_ice_cont_invalid_float_positive(self):
        with self.assertRaises(ValueError):
            self.p.vol_ice_cont = self.p.porosity + 0.1

    def test_set_vol_ice_cont_valid_int(self):
        self.p.vol_ice_cont = 0
        self.assertEqual(self.p.vol_ice_cont, 0.)

    def test_set_vol_ice_cont_valid_int_type(self):
        self.p.vol_ice_cont = 0
        self.assertIsInstance(self.p.vol_ice_cont, float)

    def test_set_vol_ice_cont_valid_str(self):
        self.p.vol_ice_cont = "1.e-1"
        self.assertEqual(self.p.vol_ice_cont, 1.e-1)

    def test_set_vol_ice_cont_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.vol_ice_cont = "five"

    def test_set_material_valid(self):
        m = Material()
        self.p.material = m
        self.assertIs(self.p.material, m)

    def test_set_material_invalid(self):
        with self.assertRaises(TypeError):
            self.p.material = 1

    def test_set_thrm_cond_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.thrm_cond = 1.e5

    def test_update_thrm_cond_porosity(self):
        self.p.porosity = 0.25
        expected = 5.31945288503591
        self.assertAlmostEqual(self.p.thrm_cond, expected, delta=1e-8)

    def test_update_thrm_cond_vol_ice_cont(self):
        self.p.vol_ice_cont = 0.05
        expected = 3.79674097529634
        self.assertAlmostEqual(self.p.thrm_cond, expected, delta=1e-8)

    def test_update_thrm_cond_material(self):
        self.p.material = Material(thrm_cond_solids=6.7,
                                   dens_solids=2.8e3,
                                   spec_heat_cap_solids=6.43e5)
        expected = 4.19347247030009
        self.assertAlmostEqual(self.p.thrm_cond, expected, delta=1e-8)

    def test_set_vol_heat_cap_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.vol_heat_cap = 1.e5

    def test_update_vol_heat_cap_porosity(self):
        self.p.porosity = 0.25
        expected = 1.389961400e9
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, delta=1e-8)

    def test_update_vol_heat_cap_vol_ice_cont(self):
        self.p.vol_ice_cont = 0.05
        expected = 1.297895050e9
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, delta=1e-8)

    def test_update_vol_heat_cap_material(self):
        self.p.material = Material(thrm_cond_solids=6.7,
                                   dens_solids=2.8e3,
                                   spec_heat_cap_solids=6.43e5)
        expected = 1.261076600e9
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, delta=1e-8)


class TestElement1D(unittest.TestCase):

    def setUp(self):
        self.nodes = tuple(Node1D(k, 2. * k + 1.) for k in range(2))
        self.e = Element1D(self.nodes)

    def test_initialize_without_nodes(self):
        with self.assertRaises(TypeError):
            Element1D()

    def test_initialize_valid_nodes_value(self):
        self.assertEqual(self.e.nodes[1].z, 3.)

    def test_initialize_valid_nodes_type(self):
        nodes = list(self.nodes)
        e = Element1D(nodes)
        self.assertIsInstance(e.nodes, tuple)

    def test_initialize_too_few_nodes(self):
        with self.assertRaises(ValueError):
            nodes = (Node1D(0), )
            Element1D(nodes)

    def test_initialize_too_many_nodes(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2. * k + 1.) for k in range(3))
            Element1D(nodes)

    def test_initialize_invalid_nodes(self):
        with self.assertRaises(TypeError):
            nodes = tuple(k for k in range(2))
            Element1D(nodes)

    def test_jacobian_value(self):
        self.assertEqual(self.e.jacobian, 2.)

    def test_set_jacobian(self):
        with self.assertRaises(AttributeError):
            self.e.jacobian = 5.

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
        self.N = shape_matrix(0.8)
        self.T_1D = np.array([5., 10.])
        self.T_column = np.array([[5.], [10.]])

    def test_shape_matrix_valid_float(self):
        expected = np.array([[0.2, 0.8]])
        self.assertTrue(np.allclose(self.N, expected))

    def test_shape_matrix_shape(self):
        expected = (1, 2)
        self.assertEqual(self.N.shape, expected)

    def test_shape_matrix_multiply_1D(self):
        expected = 9.
        actual = self.N @ self.T_1D
        self.assertAlmostEqual(expected, actual, delta=1.e-8)

    def test_shape_matrix_multiply_column(self):
        expected = 9.
        actual = self.N @ self.T_column
        self.assertAlmostEqual(expected, actual, delta=1.e-8)

    def test_shape_matrix_multiply_transpose(self):
        expected = np.array([[0.04, 0.16], [0.16, 0.64]])
        actual = self.N.T @ self.N
        self.assertTrue(np.allclose(expected, actual))

    def test_shape_matrix_valid_str(self):
        expected = np.array([[0.2, 0.8]])
        self.assertTrue(np.allclose(shape_matrix("8.e-1"), expected))

    def test_shape_matrix_invalid_str(self):
        with self.assertRaises(ValueError):
            shape_matrix("three")


class TestGradientMatrix(unittest.TestCase):

    def setUp(self):
        self.B = gradient_matrix(0.8, 2.0)
        self.T_1D = np.array([5., 10.])
        self.T_column = np.array([[5.], [10.]])

    def test_gradient_matrix_valid_float(self):
        expected = np.array([[-0.5, 0.5]])
        self.assertTrue(np.allclose(self.B, expected))

    def test_gradient_matrix_shape(self):
        expected = (1, 2)
        self.assertEqual(self.B.shape, expected)

    def test_gradient_matrix_multiply_1D(self):
        expected = 2.5
        actual = self.B @ self.T_1D
        self.assertAlmostEqual(expected, actual, delta=1.e-8)

    def test_gradient_matrix_multiply_column(self):
        expected = 2.5
        actual = self.B @ self.T_column
        self.assertAlmostEqual(expected, actual, delta=1.e-8)

    def test_gradient_matrix_multiply_transpose(self):
        expected = np.array([[0.25, -0.25], [-0.25, 0.25]])
        actual = self.B.T @ self.B
        self.assertTrue(np.allclose(expected, actual))

    def test_gradient_matrix_valid_str(self):
        expected = np.array([[-0.5, 0.5]])
        self.assertTrue(np.allclose(gradient_matrix("8.e-1", "2.e0"),
                                    expected))

    def test_gradient_matrix_invalid_str_arg0(self):
        with self.assertRaises(ValueError):
            gradient_matrix("three", 2.0)

    def test_gradient_matrix_invalid_str_arg1(self):
        with self.assertRaises(ValueError):
            gradient_matrix(1.0, "three")


if __name__ == "__main__":
    unittest.main()
