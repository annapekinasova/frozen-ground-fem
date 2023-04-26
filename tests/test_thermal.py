import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
)
from frozen_ground_fem.geometry import (
    Node1D,
    Element1D,
    BoundaryElement1D,
)
from frozen_ground_fem.thermal import (
    ThermalElement1D,
    ThermalBoundary1D,
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

    def test_heat_flow_matrix_uninitialized(self):
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix(), np.zeros((2, 2))))

    def test_heat_flow_matrix(self):
        m = Material(thrm_cond_solids=1e-5)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.porosity = 0.2
            ip.vol_ice_cont = 0.1
        lam = self.thrm_e.int_pts[0].thrm_cond
        jac = self.thrm_e.jacobian
        expected = lam / jac * np.array([[1.0, -1.0], [-1.0, 1.0]])
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix(), expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix(), np.zeros((2, 2)))
        )

    def test_heat_storage_matrix(self):
        m = Material(dens_solids=2.65e3, spec_heat_cap_solids=2.0e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.porosity = 0.3
            ip.vol_ice_cont = 0.2
        heat_cap = self.thrm_e.int_pts[0].vol_heat_cap
        jac = self.thrm_e.jacobian
        expected = heat_cap * jac / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(self.thrm_e.heat_storage_matrix(), expected))


class TestThermalBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.bnd_el = BoundaryElement1D(self.nodes)
        self.thrm_bnd = ThermalBoundary1D(self.bnd_el)

    def test_invalid_no_parent(self):
        with self.assertRaises(TypeError):
            ThermalElement1D()

    def test_invalid_parent(self):
        with self.assertRaises(TypeError):
            ThermalBoundary1D(self.nodes)

    def test_defaults(self):
        pass

    def test_nodes_equal(self):
        pass

    def test_assign_bnd_type_invalid(self):
        pass

    def test_assign_bnd_type_valid(self):
        pass

    def test_assign_bnd_value_invalid(self):
        pass

    def test_assign_bnd_value_valid(self):
        pass

    def test_update_nodes_bnd_type_temp(self):
        pass

    def test_update_nodes_bnd_type_non_temp(self):
        pass
