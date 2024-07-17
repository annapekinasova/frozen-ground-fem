import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Node1D,
)


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

    def test_temp_rate_value(self):
        self.assertEqual(self.p.temp_rate, 0.0)

    def test_temp_rate_type(self):
        self.assertIsInstance(self.p.temp_rate, float)

    def test_void_ratio_value(self):
        self.assertEqual(self.p.void_ratio, 0.0)

    def test_void_ratio_type(self):
        self.assertIsInstance(self.p.void_ratio, float)

    def test_void_ratio_0_value(self):
        self.assertEqual(self.p.void_ratio_0, 0.0)

    def test_void_ratio_0_type(self):
        self.assertIsInstance(self.p.void_ratio_0, float)


class TestNode1DInitializers(unittest.TestCase):
    def setUp(self):
        self.p = Node1D(
            0, 1.0, temp=-5.0, void_ratio=0.3, void_ratio_0=0.9, temp_rate=1.5
        )

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

    def test_temp_rate_value(self):
        self.assertEqual(self.p.temp_rate, 1.5)

    def test_temp_rate_type(self):
        self.assertIsInstance(self.p.temp_rate, float)

    def test_void_ratio_value(self):
        self.assertEqual(self.p.void_ratio, 0.3)

    def test_void_ratio_type(self):
        self.assertIsInstance(self.p.void_ratio, float)

    def test_void_ratio_0_value(self):
        self.assertEqual(self.p.void_ratio_0, 0.9)

    def test_void_ratio_0_type(self):
        self.assertIsInstance(self.p.void_ratio_0, float)


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

    def test_set_temp_rate_valid_float(self):
        self.p.temp_rate = 1.0
        self.assertEqual(self.p.temp_rate, 1.0)

    def test_set_temp_rate_valid_int(self):
        self.p.temp_rate = -1
        self.assertEqual(self.p.temp_rate, -1.0)

    def test_set_temp_rate_valid_int_type(self):
        self.p.temp_rate = 1
        self.assertIsInstance(self.p.temp_rate, float)

    def test_set_temp_rate_valid_str(self):
        self.p.temp_rate = "-1.e5"
        self.assertEqual(self.p.temp_rate, -1.0e5)

    def test_set_temp_rate_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.temp_rate = "five"

    def test_set_void_ratio_valid_float(self):
        self.p.void_ratio = 1.0
        self.assertEqual(self.p.void_ratio, 1.0)

    def test_set_void_ratio_invalid_float(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio = -0.1

    def test_set_void_ratio_valid_int(self):
        self.p.void_ratio = 1
        self.assertEqual(self.p.void_ratio, 1.0)

    def test_set_void_ratio_valid_int_type(self):
        self.p.void_ratio = 1
        self.assertIsInstance(self.p.void_ratio, float)

    def test_set_void_ratio_valid_str(self):
        self.p.void_ratio = "1.e-1"
        self.assertEqual(self.p.void_ratio, 0.1)

    def test_set_void_ratio_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio = "five"

    def test_set_void_ratio_0_valid_float(self):
        self.p.void_ratio_0 = 1.0
        self.assertEqual(self.p.void_ratio_0, 1.0)

    def test_set_void_ratio_0_invalid_float(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio_0 = -0.1

    def test_set_void_ratio_0_valid_int(self):
        self.p.void_ratio_0 = 1
        self.assertEqual(self.p.void_ratio_0, 1.0)

    def test_set_void_ratio_0_valid_int_type(self):
        self.p.void_ratio_0 = 1
        self.assertIsInstance(self.p.void_ratio_0, float)

    def test_set_void_ratio_0_valid_str(self):
        self.p.void_ratio_0 = "1.e-1"
        self.assertEqual(self.p.void_ratio_0, 0.1)

    def test_set_void_ratio_0_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio_0 = "five"


if __name__ == "__main__":
    unittest.main()
