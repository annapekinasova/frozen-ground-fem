import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Point1D,
    Node1D,
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
        self.p = Node1D()

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
        self.p = Node1D(1., -5.)

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
        self.p = Node1D()

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


if __name__ == "__main__":
    unittest.main()
