import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
)

from frozen_ground_fem.geometry import (
    Node1D,
    Element1D,
    Boundary1D,
)

from frozen_ground_fem.coupled import (
    CoupledElement1D,
    CoupledBoundary1D,
)


class TestCoupledElement1D(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.e = Element1D(self.nodes)
        self.thrm_e = CoupledElement1D(self.e)

    def test_invalid_no_parent(self):
        with self.assertRaises(TypeError):
           CoupledElement1D()

    def test_invalid_parent(self):
        with self.assertRaises(TypeError):
            CoupledElement1D(self.nodes)

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
            ip.void_ratio = 0.2
            ip.deg_sat_water = 0.1
        lam = self.thrm_e.int_pts[0].thrm_cond
        jac = self.thrm_e.jacobian
        expected = lam / jac * np.array([[1.0, -1.0], [-1.0, 1.0]])
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix(), expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix(), np.zeros((2, 2)))
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
        self.assertTrue(np.allclose(self.thrm_e.heat_storage_matrix(), expected))
    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix(), np.zeros((2, 2)))
        )
####################################
    def test_stiffness_matrix(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=2.0e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.3
            ip.deg_sat_water = 0.2
        heat_cap = self.thrm_e.int_pts[0].vol_heat_cap
        jac = self.thrm_e.jacobian
        expected = heat_cap * jac / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(self.thrm_e.heat_storage_matrix(), expected))
    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix(), np.zeros((2, 2)))
        )

    def test_mass_matrix(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=2.0e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.3
            ip.deg_sat_water = 0.2
        heat_cap = self.thrm_e.int_pts[0].vol_heat_cap
        jac = self.thrm_e.jacobian
        expected = heat_cap * jac / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(self.thrm_e.heat_storage_matrix(), expected))

#############################33
class TestCoupledBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.bnd_el = Boundary1D(self.nodes)
        self.thrm_bnd = CoupledBoundary1D(self.bnd_el)

    def test_invalid_no_parent(self):
        with self.assertRaises(TypeError):
            CoupledElement1D()

    def test_invalid_parent(self):
        with self.assertRaises(TypeError):
            CoupledElement1D(self.nodes)

    def test_defaults(self):
        self.assertEqual(self.thrm_bnd.bnd_type, CoupledElement1D.BoundaryType.temp)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 0.0)

    def test_nodes_equal(self):
        for nd, bnd_nd in zip(self.nodes, self.thrm_bnd.nodes):
            self.assertIs(nd, bnd_nd)

    def test_assign_bnd_type_invalid(self):
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = 0
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = "temp"
        with self.assertRaises(AttributeError):
            self.thrm_bnd.bnd_type = CoupledElement1D.BoundaryType.disp

    def test_assign_bnd_type_valid(self):
        self.thrm_bnd.bnd_type = CoupledElement1D.BoundaryType.heat_flux
        self.assertEqual(
            self.thrm_bnd.bnd_type, CoupledElement1D.BoundaryType.heat_flux
        )
        self.thrm_bnd.bnd_type = CoupledElement1D.BoundaryType.temp
        self.assertEqual(self.thrm_bnd.bnd_type, CoupledElement1D.BoundaryType.temp)
        self.thrm_bnd.bnd_type = CoupledElement1D.BoundaryType.temp_grad
        self.assertEqual(
            self.thrm_bnd.bnd_type, CoupledElement1D.BoundaryType.temp_grad
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
        self.thrm_bnd.bnd_type = CoupledElement1D.BoundaryType.temp
        self.thrm_bnd.bnd_value = 5.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 5.0)
        self.thrm_bnd.bnd_value = 7.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 5.0)

    def test_update_nodes_bnd_type_non_temp(self):
        self.nodes[0].temp = 20.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = CoupledElement1D.BoundaryType.heat_flux
        self.thrm_bnd.bnd_value = 5.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = CoupledElement1D.BoundaryType.temp_grad
        self.thrm_bnd.bnd_value = 7.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
