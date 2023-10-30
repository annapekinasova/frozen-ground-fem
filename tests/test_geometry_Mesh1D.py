import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Mesh1D,
)


# TODO: update tests for cubic case


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
