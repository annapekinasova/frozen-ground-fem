import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Node1D,
    Element1D,
)


class TestElement1DInvalid(unittest.TestCase):
    def test_initialize_without_nodes(self):
        with self.assertRaises(TypeError):
            Element1D()

    def test_initialize_too_few_nodes_linear(self):
        with self.assertRaises(ValueError):
            nodes = (Node1D(0),)
            Element1D(nodes, order=1)

    def test_initialize_too_many_nodes_linear(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(3))
            Element1D(nodes, order=1)

    def test_initialize_too_few_nodes_cubic(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(3))
            Element1D(nodes, order=3)

    def test_initialize_too_many_nodes_cubic(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(5))
            Element1D(nodes, order=3)

    def test_initialize_invalid_nodes_linear(self):
        with self.assertRaises(TypeError):
            nodes = tuple(k for k in range(2))
            Element1D(nodes, order=1)

    def test_initialize_invalid_nodes_cubic(self):
        with self.assertRaises(TypeError):
            nodes = tuple(k for k in range(4))
            Element1D(nodes, order=3)

    def test_initialize_invalid_order(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(3))
            Element1D(nodes, order=2)

    def test_set_jacobian(self):
        with self.assertRaises(AttributeError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(4))
            e = Element1D(nodes)
            e.jacobian = 5.0

    def test_set_int_pt(self):
        with self.assertRaises(AttributeError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(4))
            e = Element1D(nodes)
            e.int_pts = 3


class TestElement1DLinear(unittest.TestCase):
    def setUp(self):
        self.nodes = list(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.e = Element1D(self.nodes, order=1)

    def test_initialize_valid_nodes_value(self):
        self.assertEqual(self.e.nodes[1].z, 3.0)

    def test_initialize_valid_nodes_type(self):
        self.assertIsInstance(self.e.nodes, tuple)

    def test_jacobian_value(self):
        self.assertEqual(self.e.jacobian, 2.0)

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


class TestElement1DCubic(unittest.TestCase):
    def setUp(self):
        self.nodes = list(Node1D(k, 2.0 * k + 1.0) for k in range(4))
        self.e = Element1D(self.nodes)

    def test_initialize_valid_nodes_value(self):
        self.assertEqual(self.e.nodes[3].z, 7.0)

    def test_initialize_valid_nodes_type(self):
        self.assertIsInstance(self.e.nodes, tuple)

    def test_jacobian_value(self):
        self.assertEqual(self.e.jacobian, 6.0)

    def test_int_pt_local_coords(self):
        expected = np.array(
            [
                0.046910077030668,
                0.230765344947159,
                0.500000000000000,
                0.769234655052842,
                0.953089922969332,
            ]
        )
        actual = np.array([ip.local_coord for ip in self.e.int_pts])
        self.assertTrue(np.allclose(actual, expected))

    def test_int_pt_weights(self):
        expected = np.array(
            [
                0.118463442528095,
                0.239314335249683,
                0.284444444444444,
                0.239314335249683,
                0.118463442528095,
            ]
        )
        actual = np.array([ip.weight for ip in self.e.int_pts])
        self.assertTrue(np.allclose(actual, expected))

    def test_int_pt_global_coords(self):
        expected = np.array(
            [
                1.281460462184010,
                2.384592069682950,
                4.000000000000000,
                5.615407930317050,
                6.718539537815990,
            ]
        )
        actual = np.array([ip.z for ip in self.e.int_pts])
        self.assertTrue(np.allclose(actual, expected))

    def test_int_pt_type(self):
        self.assertIsInstance(self.e.int_pts, tuple)


if __name__ == "__main__":
    unittest.main()
