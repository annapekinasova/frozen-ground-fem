import unittest

import numpy as np

from frozen_ground_fem.materials import (
    spec_grav_ice as Gi,
    unit_weight_water as gam_w,
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
        m = Material(
            spec_grav_solids=2.60,
            hyd_cond_index=0.305,
            hyd_cond_0=4.05e-4,
            void_ratio_0_hyd_cond=2.60,
            void_ratio_min=0.30,
            void_ratio_tr=2.60,
            void_ratio_0_comp=2.60,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
            eff_stress_0_comp=2.80e00,
        )
        for ip in self.consol_e.int_pts:
            ip.material = m
            ip.void_ratio_0 = 0.9
            ip.void_ratio = 0.3
            ip.deg_sat_water = 0.1
            ip.pre_consol_stress = 1.0
            k, dk_de = m.hyd_cond(ip.void_ratio, 1.0, False)
            ip.hyd_cond = k
            ip.hyd_cond_gradient = dk_de
            sig, dsig_de = m.eff_stress(ip.void_ratio, ip.pre_consol_stress)
            ip.eff_stress = sig
            ip.eff_stress_gradient = dsig_de
        jac = self.consol_e.jacobian
        Gs = self.consol_e.int_pts[0].material.spec_grav_solids
        e0 = self.consol_e.int_pts[0].void_ratio_0
        e = self.consol_e.int_pts[0].void_ratio
        ppc = self.consol_e.int_pts[0].pre_consol_stress
        k, dk_de = self.consol_e.int_pts[0].material.hyd_cond(e, 1.0, False)
        sig_p, dsig_de = self.consol_e.int_pts[0].material.eff_stress(e, ppc)
        e_ratio = (1.0 + e0) / (1.0 + e)
        coef_0 = k * e_ratio * dsig_de / gam_w / jac
        coef_1 = (dk_de * (Gs - 1.0) / (1.0 + e)
                  - k * (Gs - 1.0) / (1.0 + e) ** 2)
        stiff_0 = coef_0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
        stiff_1 = coef_1 * np.array([[-0.5, 0.5], [-0.5, 0.5]])
        expected = stiff_0 + stiff_1
        self.assertTrue(np.allclose(self.consol_e.stiffness_matrix, expected))

    def test_mass_matrix_uninitialized(self):
        jac = self.consol_e.jacobian
        expected = jac / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(
            np.allclose(self.consol_e.mass_matrix,
                        expected)
        )

    def test_mass_matrix(self):
        for ip in self.consol_e.int_pts:
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.2
        jac = self.consol_e.jacobian
        e0 = self.consol_e.int_pts[0].void_ratio_0
        Sw = self.consol_e.int_pts[0].deg_sat_water
        coef = (Sw + Gi * (1.0 - Sw)) / (1.0 + e0)
        expected = coef * jac / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(self.consol_e.mass_matrix, expected))


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
        m = Material(
            spec_grav_solids=2.60,
            hyd_cond_index=0.305,
            hyd_cond_0=4.05e-4,
            void_ratio_0_hyd_cond=2.60,
            void_ratio_min=0.30,
            void_ratio_tr=2.60,
            void_ratio_0_comp=2.60,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
            eff_stress_0_comp=2.80e00,
        )
        for ip in self.consol_e.int_pts:
            ip.material = m
            ip.void_ratio_0 = 0.9
            ip.void_ratio = 0.3
            ip.deg_sat_water = 0.1
            ip.pre_consol_stress = 1.0
            k, dk_de = m.hyd_cond(ip.void_ratio, 1.0, False)
            ip.hyd_cond = k
            ip.hyd_cond_gradient = dk_de
            sig, dsig_de = m.eff_stress(ip.void_ratio, ip.pre_consol_stress)
            ip.eff_stress = sig
            ip.eff_stress_gradient = dsig_de
        jac = self.consol_e.jacobian
        Gs = self.consol_e.int_pts[0].material.spec_grav_solids
        e0 = self.consol_e.int_pts[0].void_ratio_0
        e = self.consol_e.int_pts[0].void_ratio
        ppc = self.consol_e.int_pts[0].pre_consol_stress
        k, dk_de = self.consol_e.int_pts[0].material.hyd_cond(e, 1.0, False)
        sig_p, dsig_de = self.consol_e.int_pts[0].material.eff_stress(e, ppc)
        e_ratio = (1.0 + e0) / (1.0 + e)
        coef_0 = k * e_ratio * dsig_de / gam_w / jac
        coef_1 = (dk_de * (Gs - 1.0) / (1.0 + e)
                  - k * (Gs - 1.0) / (1.0 + e) ** 2)
        stiff_0 = coef_0 / 40.0 * np.array([[148, -189, 54, -13],
                                            [-189, 432, -297, 54],
                                            [54, -297, 432, -189],
                                            [-13, 54, -189, 148]])
        stiff_1 = coef_1 / 1680.0 * np.array([[-840, 1197, -504, 147],
                                              [-1197, 0, 1701, -504],
                                              [504, -1701, 0, 1197],
                                              [-147, 504, -1197, 840]])
        expected = stiff_0 + stiff_1
        self.assertTrue(np.allclose(self.consol_e.stiffness_matrix, expected))

    def test_mass_matrix_uninitialized(self):
        jac = self.consol_e.jacobian
        expected = jac / 1680.0 * np.array(
            [
                [128,   99,  -36,   19],
                [99,  648,  -81,  -36],
                [-36,  -81,  648,   99],
                [19,  -36,   99,  128],
            ]
        )
        self.assertTrue(
            np.allclose(self.consol_e.mass_matrix,
                        expected)
        )

    def test_mass_matrix(self):
        for ip in self.consol_e.int_pts:
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.2
        jac = self.consol_e.jacobian
        e0 = self.consol_e.int_pts[0].void_ratio_0
        Sw = self.consol_e.int_pts[0].deg_sat_water
        coef = (Sw + Gi * (1.0 - Sw)) / (1.0 + e0)
        expected = coef * jac / 1680.0 * np.array(
            [
                [128,   99,  -36,   19],
                [99,  648,  -81,  -36],
                [-36,  -81,  648,   99],
                [19,  -36,   99,  128],
            ]
        )
        self.assertTrue(np.allclose(self.consol_e.mass_matrix, expected))


class TestConsolidationBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.int_pts = (IntegrationPoint1D(),)
        # self.bnd_el = Boundary1D(self.nodes, self.int_pts)
        self.consol_bnd = ConsolidationBoundary1D(
            self.nodes, self.int_pts
        )

    # def test_invalid_no_parent(self):
    #     with self.assertRaises(TypeError):
    #         ConsolidationElement1D()

    # def test_invalid_parent(self):
    #     with self.assertRaises(TypeError):
    #         ConsolidationBoundary1D(self.nodes)

    def test_defaults(self):
        self.assertEqual(self.consol_bnd.bnd_type,
                         ConsolidationBoundary1D.BoundaryType.fixed_flux)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 0.0)
        self.assertIsNone(self.consol_bnd.bnd_function)

    def test_assign_bnd_type_invalid(self):
        cb1d = ConsolidationBoundary1D
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_type = 0
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_type = "temp"
        with self.assertRaises(AttributeError):
            self.consol_bnd.bnd_type = cb1d.BoundaryType.disp

    def test_assign_bnd_type_valid(self):
        cb1d = ConsolidationBoundary1D
        self.consol_bnd.bnd_type = cb1d.BoundaryType.void_ratio
        self.assertEqual(
            self.consol_bnd.bnd_type,
            ConsolidationBoundary1D.BoundaryType.void_ratio
        )
        self.consol_bnd.bnd_type = cb1d.BoundaryType.fixed_flux
        self.assertEqual(self.consol_bnd.bnd_type,
                         ConsolidationBoundary1D.BoundaryType.fixed_flux)
        self.consol_bnd.bnd_type = cb1d.BoundaryType.water_flux
        self.assertEqual(
            self.consol_bnd.bnd_type,
            ConsolidationBoundary1D.BoundaryType.water_flux
        )

    def test_assign_bnd_value_invalid(self):
        with self.assertRaises(ValueError):
            self.consol_bnd.bnd_value = "temp"

    def test_assign_bnd_value_valid(self):
        self.consol_bnd.bnd_value = 1.0
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1.0)
        self.consol_bnd.bnd_value = -2
        self.assertAlmostEqual(self.consol_bnd.bnd_value, -2.0)
        self.consol_bnd.bnd_value = "1e-5"
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1e-5)

    # TODO: test assign boundary function

    def test_update_nodes_bnd_type_void_ratio(self):
        cb1d = ConsolidationBoundary1D
        self.nodes[0].void_ratio = 0.6
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.6)
        self.consol_bnd.bnd_type = cb1d.BoundaryType.void_ratio
        self.consol_bnd.bnd_value = 0.3
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.6)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.3)
        self.consol_bnd.bnd_value = 0.4
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.3)

    def test_update_nodes_bnd_type_non_void_ratio(self):
        cb1d = ConsolidationBoundary1D
        self.nodes[0].void_ratio = 0.6
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.6)
        self.consol_bnd.bnd_type = cb1d.BoundaryType.fixed_flux
        self.consol_bnd.bnd_value = 5.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.6)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.6)
        self.consol_bnd.bnd_type = cb1d.BoundaryType.water_flux
        self.consol_bnd.bnd_value = 7.0
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.6)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.6)
