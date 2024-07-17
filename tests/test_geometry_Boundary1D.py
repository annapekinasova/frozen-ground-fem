import unittest

from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
    Boundary1D,
)


class TestBoundary1DInvalid(unittest.TestCase):
    def test_initialize_without_nodes(self):
        with self.assertRaises(TypeError):
            Boundary1D()

    def test_initialize_too_many_nodes(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
            Boundary1D(nodes)

    def test_initialize_invalid_nodes(self):
        with self.assertRaises(TypeError):
            nodes = tuple(k for k in range(1))
            Boundary1D(nodes)

    def test_initialize_too_many_int_pts(self):
        with self.assertRaises(ValueError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(1))
            int_pts = tuple(IntegrationPoint1D(2.0 * k + 1.0) for k in range(2))
            Boundary1D(nodes, int_pts)

    def test_initialize_invalid_int_pts_integer(self):
        with self.assertRaises(TypeError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(1))
            int_pts = tuple(k for k in range(1))
            Boundary1D(nodes, int_pts)

    def test_initialize_invalid_int_pts_node(self):
        with self.assertRaises(TypeError):
            nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(1))
            int_pts = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(1))
            Boundary1D(nodes, int_pts)


class TestBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = list((Node1D(0, 0.5),))
        self.int_pts = list((IntegrationPoint1D(0.75),))
        self.e = Boundary1D(self.nodes, self.int_pts)

    def test_initialize_valid_nodes_value(self):
        self.assertEqual(self.e.nodes[0].z, 0.5)

    def test_initialize_valid_nodes_type(self):
        self.assertIsInstance(self.e.nodes, tuple)

    def test_initialize_valid_int_pts_value(self):
        self.assertEqual(self.e.int_pts[0].z, 0.75)

    def test_initialize_valid_int_pts_type(self):
        self.assertIsInstance(self.e.int_pts, tuple)


if __name__ == "__main__":
    unittest.main()
