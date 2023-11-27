import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
)
from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
)
from frozen_ground_fem.thermal import (
    ThermalElement1D,
    ThermalBoundary1D,
)


class TestThermalElement1DLinear(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.thrm_e = ThermalElement1D(self.nodes, order=1)

    def test_jacobian_value(self):
        expected = self.nodes[-1].z - self.nodes[0].z
        self.assertAlmostEqual(self.thrm_e.jacobian, expected)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.thrm_e.nodes):
            self.assertIs(nd, e_nd)

    def test_heat_flow_matrix_uninitialized(self):
        self.assertTrue(np.allclose(
            self.thrm_e.heat_flow_matrix, np.zeros((2, 2))))

    def test_heat_flow_matrix(self):
        m = Material(thrm_cond_solids=1e-5)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.2
            ip.deg_sat_water = 0.1
        lam = self.thrm_e.int_pts[0].thrm_cond
        jac = self.thrm_e.jacobian
        expected = lam / jac * np.array([[1.0, -1.0], [-1.0, 1.0]])
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix, expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix, np.zeros((2, 2)))
        )

    def test_heat_storage_matrix(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=2.0e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.3
            ip.deg_sat_water = 0.2
        heat_cap = self.thrm_e.int_pts[0].vol_heat_cap
        jac = self.thrm_e.jacobian
        expected = heat_cap * jac / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(
            self.thrm_e.heat_storage_matrix, expected))


class TestThermalElement1DCubic(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(4))
        self.thrm_e = ThermalElement1D(self.nodes, order=3)

    def test_jacobian_value(self):
        expected = self.nodes[-1].z - self.nodes[0].z
        self.assertAlmostEqual(self.thrm_e.jacobian, expected)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.thrm_e.nodes):
            self.assertIs(nd, e_nd)

    def test_heat_flow_matrix_uninitialized(self):
        self.assertTrue(np.allclose(
            self.thrm_e.heat_flow_matrix, np.zeros((4, 4))))

    def test_heat_flow_matrix(self):
        m = Material(thrm_cond_solids=1e-5)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.2
            ip.deg_sat_water = 0.1
        lam = self.thrm_e.int_pts[0].thrm_cond
        jac = self.thrm_e.jacobian
        expected = (1/40) * lam / jac * np.array(
            [
                [148, -189, 54, -13],
                [-189, 432, -297, 54],
                [54, -297, 432, -189],
                [-13, 54, -189, 148],
            ]
        )
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix, expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix, np.zeros((4, 4)))
        )

    def test_heat_storage_matrix(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=2.0e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.3
            ip.deg_sat_water = 0.2
        heat_cap = self.thrm_e.int_pts[0].vol_heat_cap
        jac = self.thrm_e.jacobian
        expected = (1/1680) * heat_cap * jac * np.array(
            [
                [128,   99,  -36,   19],
                [99,  648,  -81,  -36],
                [-36,  -81,  648,   99],
                [19,  -36,   99,  128],
            ]
        )
        self.assertTrue(np.allclose(
            self.thrm_e.heat_storage_matrix, expected))


class TestThermalBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.int_pts = (IntegrationPoint1D(),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes, self.int_pts)

    def test_defaults(self):
        self.assertEqual(self.thrm_bnd.bnd_type,
                         ThermalBoundary1D.BoundaryType.temp)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 0.0)

    def test_nodes_equal(self):
        for nd, bnd_nd in zip(self.nodes, self.thrm_bnd.nodes):
            self.assertIs(nd, bnd_nd)

    def test_int_pts_equal(self):
        for ip, bnd_ip in zip(self.int_pts, self.thrm_bnd.int_pts):
            self.assertIs(ip, bnd_ip)

    def test_assign_bnd_type_invalid(self):
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = 0
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = "temp"
        with self.assertRaises(AttributeError):
            self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.disp

    def test_assign_bnd_type_valid(self):
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.heat_flux
        self.assertEqual(
            self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.heat_flux
        )
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp
        self.assertEqual(self.thrm_bnd.bnd_type,
                         ThermalBoundary1D.BoundaryType.temp)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
        self.assertEqual(
            self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.temp_grad
        )

    def test_assign_bnd_value_invalid(self):
        with self.assertRaises(ValueError):
            self.thrm_bnd.bnd_value = "temp"

    def test_assign_bnd_value_valid(self):
        self.thrm_bnd.bnd_value = 1.0
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1.0)
        self.thrm_bnd.bnd_value = -2
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, -2.0)
        self.thrm_bnd.bnd_value = "1e-5"
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1e-5)

    def test_update_nodes_bnd_type_temp(self):
        self.nodes[0].temp = 20.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp
        self.thrm_bnd.bnd_value = 5.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 5.0)
        self.thrm_bnd.bnd_value = 7.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 5.0)

    def test_update_nodes_bnd_type_non_temp(self):
        self.nodes[0].temp = 20.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.heat_flux
        self.thrm_bnd.bnd_value = 5.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
        self.thrm_bnd.bnd_value = 7.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
