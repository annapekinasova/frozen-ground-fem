import unittest

from frozen_ground_fem.geometry import (
    Node1D,
    Boundary1D,
)


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

# TODO: add tests for adding IntegrationPoint1D


if __name__ == "__main__":
    unittest.main()
