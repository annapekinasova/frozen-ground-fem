import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Point1D,
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


if __name__ == "__main__":
    unittest.main()
