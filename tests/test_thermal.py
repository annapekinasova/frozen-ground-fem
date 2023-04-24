import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Node1D,
    Element1D,
)
from frozen_ground_fem.thermal import (
    ThermalElement1D,
)


class TestThermalElement1D(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.e = Element1D(self.nodes)
        self.thrm_e = ThermalElement1D(self.e)

    def test_invalid_no_parent(self):
        with self.assertRaises(TypeError):
            thrm_e = ThermalElement1D()

    def test_invalid_parent(self):
        with self.assertRaises(TypeError):
            thrm_e = ThermalElement1D(self.nodes)

    def test_jacobian_equal(self):
        self.assertAlmostEqual(self.e.jacobian, self.thrm_e.jacobian)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.thrm_e.nodes):
            self.assertIs(nd, e_nd)

    def test_int_pts_equal(self):
        for e_ip, te_ip in zip(self.e.int_pts, self.thrm_e.int_pts):
            self.assertIs(e_ip, te_ip)

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix(), np.zeros((2, 2)))
        )
