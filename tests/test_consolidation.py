import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
)
from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
    Element1D,
    Boundary1D,
)
from frozen_ground_fem.consolidation import (
    ConsolidationElement1D,
    ConsolidationBoundary1D,
)


class TestConsolidationElement1DInvalid(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))

    def test_no_parent(self):
        with self.assertRaises(TypeError):
            ConsolidationElement1D()

    def test_invalid_parent(self):
        with self.assertRaises(TypeError):
            ConsolidationElement1D(self.nodes)


class TestConsolidationElement1DLinear(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.e = Element1D(self.nodes, order=1)
        self.consol_e = ConsolidationElement1D(self.e)

    def test_jacobian_equal(self):
        self.assertAlmostEqual(self.e.jacobian, self.consol_e.jacobian)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.consol_e.nodes):
            self.assertIs(nd, e_nd)

    def test_int_pts_equal(self):
        for e_ip, te_ip in zip(self.e.int_pts, self.consol_e.int_pts):
            self.assertIs(e_ip, te_ip)

    def test_stiffness_matrix_uninitialized(self):
        self.assertTrue(np.allclose(
            self.consol_e.stiffness_matrix, np.zeros((2, 2))))

    def test_stiffness_matrix(self):
        m = Material(spec_grav_solids=2.65, gam_w=9.801)
        for ip in self.consol_e.int_pts:
            ip.material = m
            ip.void_ratio_0 = 0.9
            ip.void_ratio = 0.3
            k = ip.hyd_cond
            ip.deg_sat_water = 0.1
            dsig_de = ip.eff_stress_gradient
            dk_de = ip.hyd_cond_gradient
            Gs = ip.material.spec_grav_solids
        e_ratio = (1.0 + ip.void_ratio_0) / (1.0 + ip.void_ratio)
        k_coef = dk_de * (Gs - 1.0) / (1.0 + e) - k * \
                (Gs - 1.0) / (1.0 + e) ** 2
        jac = self.consol_e.jacobian
        expected = [k * e_ratio * dsig_de / gam_w /
            jac * np.array([[1.0, -1.0], [-1.0, 1.0]])]
           + [k_coef / jac * np.array([[1.0, -1.0], [-1.0, 1.0]])]
        self.assertTrue(np.allclose(self.consol_e.stiffness_matrix, expected))

    def test_mass_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.consol_e.mass_matrix, np.zeros((2, 2)))
        )

    def test_mass_matrix_matrix(self):
        m = Material(spec_grav_ice=0.992)
        for ip in self.consol.int_pts:
            ip.material = m
            ip.void_ratio = 0.3
            ip.deg_sat_ice = 0.1
            ip.deg_sat_water = 0.2
        jac = self.consol_e.jacobian
        expected = jac * [(ip.deg_sat_water + spec_grav_ice * ip.deg_sat_ice)
                          / (1.0 + ip.void_ratio_0)] * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(
            self.consol_e.mass_matrix_matrix, expected))


class TestConsolidationElement1DCubic(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(4))
        self.e = Element1D(self.nodes, order=3)
        self.consol_e = ConsolidationElement1D(self.e)

    def test_jacobian_equal(self):
        self.assertAlmostEqual(self.e.jacobian, self.consol_e.jacobian)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.consol_e.nodes):
            self.assertIs(nd, e_nd)

    def test_int_pts_equal(self):
        for e_ip, te_ip in zip(self.e.int_pts, self.consol_e.int_pts):
            self.assertIs(e_ip, te_ip)

    def test_stiffness_matrix_uninitialized(self):
        self.assertTrue(np.allclose(
            self.consol_e.stiffness_matrix, np.zeros((4, 4))))

    def test_stiffness_matrix(self):
        m = Material(spec_grav_solids=2.65, gam_w=9.801)
        for ip in self.consol_e.int_pts:
            ip.material = m
            ip.void_ratio_0 = 0.9
            ip.void_ratio = 0.3
            k = ip.hyd_cond
            ip.deg_sat_water = 0.1
            dsig_de = ip.eff_stress_gradient
            dk_de = ip.hyd_cond_gradient
            Gs = ip.material.spec_grav_solids
        e_ratio = (1.0 + ip.void_ratio_0) / (1.0 + ip.void_ratio)
        k_coef = dk_de * (Gs - 1.0) / (1.0 + e) - k * \
                (Gs - 1.0) / (1.0 + e) ** 2
        jac = self.consol_e.jacobian
        expected = (1/40) * [k * e_ratio * dsig_de / gam_w / jac * np.array([
            [148, -189, 54, -13],
                [-189, 432, -297, 54],
                [54, -297, 432, -189],
                [-13, 54, -189, 148],
        ]])]
            + (1/1680) * [k_coef / jac * np.array([[
                [-840, 1197, -504, -147],
                   [-1197, 0, 1701, -504],
                   [504, -1701, 0, 1197],
                   [-147, 504, -1197, 840],
                ]])]

            self.assertTrue(np.allclose(
                self.consol_e.stiffness_matrix, expected))

        def test_mass_matrix_uninitialized(self):
           self.assertTrue(
           np.allclose(self.consol_e.mass_matrix, np.zeros((4, 4)))
        )

        def test_mass_matrix(self):
            m = Material(spec_grav_ice=0.992)
            for ip in self.consol_e.int_pts:
                ip.material= m
                ip.void_ratio = 0.3
                ip.deg_sat_ice = 0.1
                ip.deg_sat_water = 0.2
            jac = self.consol_e.jacobian
            expected = (1/1680) * jac * np.array(
           [
                [128,   99,  -36,   19],
                [99,  648,  -81,  -36],
                [-36,  -81,  648,   99],
                [19,  -36,   99,  128],
            ]
        )
            self.assertTrue(np.allclose(
            self.consol_e.mass_matrix, expected))


 class TestConsolidationBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.int_pts = (IntegrationPoint1D(),)
        self.bnd_el = Boundary1D(self.nodes, self.int_pts)
        self.consol_bnd = ConsolidationBoundary1D(self.bnd_el)

        def test_invalid_no_parent(self):
        with self.assertRaises(TypeError):
            ConsolidationElement1D()

        def test_invalid_parent(self):
        with self.assertRaises(TypeError):
            ConsolidationBoundary1D(self.nodes)

        def test_defaults(self):
        self.assertEqual(self.consol_bnd.bnd_type,
                        ConsolidationBoundary1D.BoundaryType.temp)
                         self.assertAlmostEqual(self.consol_bnd.bnd_value, 0.0)

                         def test_nodes_equal(self):
                         for nd, bnd_nd in zip(self.nodes, self.consol_bnd.nodes):
                         self.assertIs(nd, bnd_nd)

                         def test_int_pts_equal(self):
                         for ip, bnd_ip in zip(self.int_pts, self.consol_bnd.int_pts):
                         self.assertIs(ip, bnd_ip)

                         def test_assign_bnd_type_invalid(self):
                         with self.assertRaises(TypeError):
                         self.consol_bnd.bnd_type= 0
                         with self.assertRaises(TypeError):
                         self.consol_bnd.bnd_type= "temp"
                         with self.assertRaises(AttributeError):
                         self.consol_bnd.bnd_type= ConsolidationBoundary1D.BoundaryType.disp

                         def test_assign_bnd_type_valid(self):
                         self.consol_bnd.bnd_type= ConsolidationBoundary1D.BoundaryType.heat_flux
                         self.assertEqual(
            self.consol_bnd.bnd_type, ConsolidationBoundary1D.BoundaryType.heat_flux
        )
            self.consol_bnd.bnd_type = ConsolidationBoundary1D.BoundaryType.temp
            self.assertEqual(self.consol_bnd.bnd_type,
                        ConsolidationBoundary1D.BoundaryType.temp)
                         self.consol_bnd.bnd_type = ConsolidationBoundary1D.BoundaryType.temp_grad
                         self.assertEqual(
            self.consol_bnd.bnd_type, ConsolidationBoundary1D.BoundaryType.temp_grad
        )

        def test_assign_bnd_value_invalid(self):
        with self.assertRaises(ValueError):
        self.consol_bnd.bnd_value = "temp"

        def test_assign_bnd_value_valid(self):
        self.consol_bnd.bnd_value= 1.0
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1.0)
        self.consol_bnd.bnd_value= -2
        self.assertAlmostEqual(self.consol_bnd.bnd_value, -2.0)
        self.consol_bnd.bnd_value= "1e-5"
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1e-5)

        def test_update_nodes_bnd_type_temp(self):
        self.nodes[0].temp= 20.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 20.0)
        self.consol_bnd.bnd_type= ConsolidationBoundary1D.BoundaryType.temp
        self.consol_bnd.bnd_value= 5.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 20.0)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 5.0)
        self.consol_bnd.bnd_value= 7.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 5.0)

        def test_update_nodes_bnd_type_non_temp(self):
        self.nodes[0].temp= 20.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 20.0)
        self.consol_bnd.bnd_type= ConsolidationBoundary1D.BoundaryType.heat_flux
        self.consol_bnd.bnd_value= 5.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 20.0)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 20.0)
        self.consol_bnd.bnd_type= ConsolidationBoundary1D.BoundaryType.temp_grad
        self.consol_bnd.bnd_value= 7.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 20.0)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].temp, 20.0)
