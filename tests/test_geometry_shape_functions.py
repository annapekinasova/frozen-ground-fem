import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    shape_matrix_linear,
    gradient_matrix_linear,
    shape_matrix_cubic,
    gradient_matrix_cubic,
)


class TestShapeMatrixLinearInvalid(unittest.TestCase):
    def test_shape_matrix_linear_invalid_str(self):
        with self.assertRaises(ValueError):
            shape_matrix_linear("three")

    def test_shape_matrix_linear_invalid_tuple(self):
        with self.assertRaises(TypeError):
            shape_matrix_linear(())


class TestGradientMatrixLinearInvalid(unittest.TestCase):
    def test_gradient_matrix_linear_invalid_str_arg0(self):
        with self.assertRaises(ValueError):
            gradient_matrix_linear("three", 2.0)

    def test_gradient_matrix_linear_invalid_str_arg1(self):
        with self.assertRaises(ValueError):
            gradient_matrix_linear(1.0, "three")

    def test_gradient_matrix_linear_invalid_tuple_arg0(self):
        with self.assertRaises(TypeError):
            gradient_matrix_linear((), 2.0)

    def test_gradient_matrix_linear_invalid_tuple_arg1(self):
        with self.assertRaises(TypeError):
            gradient_matrix_linear(1.0, ())


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
        self.assertTrue(np.allclose(gradient_matrix_linear("8.e-1", "2.e0"), expected))


class TestShapeMatrixCubicInvalid(unittest.TestCase):
    def test_shape_matrix_cubic_invalid_str(self):
        with self.assertRaises(ValueError):
            shape_matrix_cubic("three")

    def test_shape_matrix_cubic_invalid_tuple(self):
        with self.assertRaises(TypeError):
            shape_matrix_cubic(())


class TestGradientMatrixCubicInvalid(unittest.TestCase):
    def test_gradient_matrix_cubic_invalid_str_arg0(self):
        with self.assertRaises(ValueError):
            gradient_matrix_cubic("three", 2.0)

    def test_gradient_matrix_cubic_invalid_str_arg1(self):
        with self.assertRaises(ValueError):
            gradient_matrix_cubic(1.0, "three")

    def test_gradient_matrix_cubic_invalid_tuple_arg0(self):
        with self.assertRaises(TypeError):
            gradient_matrix_cubic((), 2.0)

    def test_gradient_matrix_cubic_invalid_tuple_arg1(self):
        with self.assertRaises(TypeError):
            gradient_matrix_cubic(1.0, ())


class TestShapeMatrixCubic(unittest.TestCase):
    def setUp(self):
        self.N = shape_matrix_cubic(0.8)
        self.T_1D = np.array([5.0, 10.0, 12.0, 13.0])
        self.T_column = np.reshape(self.T_1D, (4, 1))
        self.fct = np.array([0.8, 0.8 - 1.0 / 3.0, 0.8 - 2.0 / 3.0, -0.2])
        self.cef = np.array([-4.5, 13.5, -13.5, 4.5])

    def test_shape_matrix_cubic_valid_float(self):
        expected = np.array(
            [
                [
                    self.cef[0] * self.fct[1] * self.fct[2] * self.fct[3],
                    self.cef[1] * self.fct[0] * self.fct[2] * self.fct[3],
                    self.cef[2] * self.fct[0] * self.fct[1] * self.fct[3],
                    self.cef[3] * self.fct[0] * self.fct[1] * self.fct[2],
                ]
            ]
        )
        self.assertTrue(np.allclose(self.N, expected))

    def test_shape_matrix_cubic_shape(self):
        expected = (1, 4)
        self.assertEqual(self.N.shape, expected)

    def test_shape_matrix_cubic_multiply_1D(self):
        expected = 12.408
        actual = self.N @ self.T_1D
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_shape_matrix_cubic_multiply_column(self):
        expected = 12.408
        actual = self.N @ self.T_column
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_shape_matrix_cubic_multiply_transpose(self):
        expected = np.array(
            [
                [0.003136, -0.016128, 0.056448, 0.012544],
                [-0.016128, 0.082944, -0.290304, -0.064512],
                [0.056448, -0.290304, 1.016064, 0.225792],
                [0.012544, -0.064512, 0.225792, 0.050176],
            ]
        )
        actual = self.N.T @ self.N
        self.assertTrue(np.allclose(expected, actual))

    def test_shape_matrix_cubic_valid_str(self):
        expected = np.array(
            [
                [
                    self.cef[0] * self.fct[1] * self.fct[2] * self.fct[3],
                    self.cef[1] * self.fct[0] * self.fct[2] * self.fct[3],
                    self.cef[2] * self.fct[0] * self.fct[1] * self.fct[3],
                    self.cef[3] * self.fct[0] * self.fct[1] * self.fct[2],
                ]
            ]
        )
        self.assertTrue(np.allclose(shape_matrix_cubic("8.e-1"), expected))


class TestGradientMatrixCubic(unittest.TestCase):
    def setUp(self):
        self.B = gradient_matrix_cubic(0.8, 2.0)
        self.T_1D = np.array([5.0, 10.0, 12.0, 13.0])
        self.T_column = np.reshape(self.T_1D, (4, 1))
        self.fct = np.array([0.8, 0.8 - 1.0 / 3.0, 0.8 - 2.0 / 3.0, -0.2])
        self.cef = np.array([-4.5, 13.5, -13.5, 4.5])

    def test_gradient_matrix_cubic_valid_float(self):
        expected = 0.5 * np.array(
            [
                self.cef[0]
                * (
                    self.fct[2] * self.fct[3]
                    + self.fct[1] * self.fct[3]
                    + self.fct[1] * self.fct[2]
                ),
                self.cef[1]
                * (
                    self.fct[0] * self.fct[2]
                    + self.fct[0] * self.fct[3]
                    + self.fct[2] * self.fct[3]
                ),
                self.cef[2]
                * (
                    self.fct[0] * self.fct[1]
                    + self.fct[1] * self.fct[3]
                    + self.fct[0] * self.fct[3]
                ),
                self.cef[3]
                * (
                    self.fct[0] * self.fct[1]
                    + self.fct[0] * self.fct[2]
                    + self.fct[1] * self.fct[2]
                ),
            ]
        )
        self.assertTrue(np.allclose(self.B, expected))

    def test_gradient_matrix_cubic_shape(self):
        expected = (1, 4)
        self.assertEqual(self.B.shape, expected)

    def test_gradient_matrix_cubic_multiply_1D(self):
        expected = 1.39
        actual = self.B @ self.T_1D
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_gradient_matrix_cubic_multiply_column(self):
        expected = 1.39
        actual = self.B @ self.T_column
        self.assertAlmostEqual(expected, actual, delta=1.0e-8)

    def test_gradient_matrix_cubic_multiply_transpose(self):
        expected = np.array(
            [
                [0.0169, -0.0702, -0.1053, 0.1586],
                [-0.0702, 0.2916, 0.4374, -0.6588],
                [-0.1053, 0.4374, 0.6561, -0.9882],
                [0.1586, -0.6588, -0.9882, 1.4884],
            ]
        )
        actual = self.B.T @ self.B
        self.assertTrue(np.allclose(expected, actual))

    def test_gradient_matrix_cubic_valid_str(self):
        expected = 0.5 * np.array(
            [
                self.cef[0]
                * (
                    self.fct[2] * self.fct[3]
                    + self.fct[1] * self.fct[3]
                    + self.fct[1] * self.fct[2]
                ),
                self.cef[1]
                * (
                    self.fct[0] * self.fct[2]
                    + self.fct[0] * self.fct[3]
                    + self.fct[2] * self.fct[3]
                ),
                self.cef[2]
                * (
                    self.fct[0] * self.fct[1]
                    + self.fct[1] * self.fct[3]
                    + self.fct[0] * self.fct[3]
                ),
                self.cef[3]
                * (
                    self.fct[0] * self.fct[1]
                    + self.fct[0] * self.fct[2]
                    + self.fct[1] * self.fct[2]
                ),
            ]
        )
        self.assertTrue(np.allclose(gradient_matrix_cubic("8.e-1", "2.e0"), expected))


if __name__ == "__main__":
    unittest.main()
