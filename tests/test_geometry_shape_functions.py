import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    shape_matrix_linear,
    gradient_matrix_linear,
)


class TestShapeMatrixLinear(unittest.TestCase):
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


class TestGradientMatrixLinear(unittest.TestCase):
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


# TODO: TestShapeMatrixCubic, TestGradientMatrixCubic


if __name__ == "__main__":
    unittest.main()
