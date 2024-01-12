import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
    unit_weight_water as gam_w,
    spec_grav_ice as Gi,
)
from frozen_ground_fem.geometry import (
    Node1D,
)
from frozen_ground_fem.consolidation import (
    ConsolidationElement1D,
)


class TestConsolidationElement1DLinear(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.consol_e = ConsolidationElement1D(self.nodes, order=1)

    def test_jacobian_value(self):
        expected0 = self.nodes[-1].z - self.nodes[0].z
        expected1 = 2.0
        self.assertAlmostEqual(self.consol_e.jacobian, expected0)
        self.assertAlmostEqual(self.consol_e.jacobian, expected1)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.consol_e.nodes):
            self.assertIs(nd, e_nd)

    def test_stiffness_matrix_uninitialized(self):
        self.assertTrue(np.allclose(
            self.consol_e.stiffness_matrix, np.zeros((2, 2))))

    def test_stiffness_matrix_full(self):
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
        jac = 2.0
        Gs = 2.60
        e0 = 0.9
        e = 0.3
        Cku = 0.305
        k0 = 4.05e-4
        ek0 = 2.60
        k = k0 * 10 ** ((e - ek0) / Cku)
        dk_de = k * np.log(10) / Cku
        sig_p_0 = 2.80
        Ccu = 0.421
        ecu0 = 2.60
        sig_p = sig_p_0 * 10 ** (-(e - ecu0) / Ccu)
        dsig_de = -sig_p * np.log(10) / Ccu
        e_ratio = (1.0 + e0) / (1.0 + e)
        coef_0 = k * e_ratio * dsig_de / gam_w / jac
        coef_1 = (dk_de * (Gs - 1.0) / (1.0 + e)
                  - k * (Gs - 1.0) / (1.0 + e) ** 2)
        stiff_0 = coef_0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
        stiff_1 = coef_1 * np.array([[-0.5, 0.5], [-0.5, 0.5]])
        expected = stiff_0 + stiff_1
        self.assertTrue(np.allclose(self.consol_e.stiffness_matrix, expected))

    def test_mass_matrix_uninitialized(self):
        jac = 2.0
        expected = jac / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(
            np.allclose(self.consol_e.mass_matrix,
                        expected)
        )

    def test_mass_matrix_full(self):
        for ip in self.consol_e.int_pts:
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.2
        jac = 2.0
        e0 = 0.3
        Sw = 0.2
        coef = (Sw + Gi * (1.0 - Sw)) / (1.0 + e0)
        expected = coef * jac / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(self.consol_e.mass_matrix, expected))

    def test_update_integration_points_null_material(self):
        self.consol_e.nodes[0].void_ratio = 0.75
        self.consol_e.nodes[1].void_ratio = 0.65
        for ip in self.consol_e.int_pts:
            ip.void_ratio_0 = 0.9
        self.consol_e.update_integration_points()
        self.assertAlmostEqual(self.consol_e.int_pts[0].void_ratio,
                               0.728867513459481)
        self.assertAlmostEqual(self.consol_e.int_pts[1].void_ratio,
                               0.671132486540519)
        self.assertAlmostEqual(self.consol_e.int_pts[0].hyd_cond,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[1].hyd_cond,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[0].hyd_cond_gradient,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[1].hyd_cond_gradient,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[0].eff_stress,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[1].eff_stress,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[0].eff_stress_gradient,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[1].eff_stress_gradient,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[0].pre_consol_stress,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[1].pre_consol_stress,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[0].water_flux_rate,
                               0.0)
        self.assertAlmostEqual(self.consol_e.int_pts[1].water_flux_rate,
                               0.0)

    def test_update_integration_points_with_material(self):
        self.consol_e.nodes[0].void_ratio = 0.75
        self.consol_e.nodes[1].void_ratio = 0.65
        m = Material(
            spec_grav_solids=2.60,
            hyd_cond_index=0.305,
            hyd_cond_0=4.05e-4,
            hyd_cond_mult=0.8,
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
        self.consol_e.update_integration_points()
        self.assertAlmostEqual(self.consol_e.int_pts[0].void_ratio,
                               0.728867513459481)
        self.assertAlmostEqual(self.consol_e.int_pts[1].void_ratio,
                               0.671132486540519)
        self.assertAlmostEqual(self.consol_e.int_pts[0].hyd_cond,
                               2.968892083014210E-10, delta=1e-17)
        self.assertAlmostEqual(self.consol_e.int_pts[1].hyd_cond,
                               1.919991214136810E-10, delta=1e-17)
        self.assertAlmostEqual(self.consol_e.int_pts[0].hyd_cond_gradient,
                               2.241353001002160E-09, delta=1e-17)
        self.assertAlmostEqual(self.consol_e.int_pts[1].hyd_cond_gradient,
                               1.449489556836380E-09, delta=1e-17)
        self.assertAlmostEqual(self.consol_e.int_pts[0].eff_stress,
                               7.792077237928290E+04)
        self.assertAlmostEqual(self.consol_e.int_pts[1].eff_stress,
                               1.068540727404800E+05)
        self.assertAlmostEqual(self.consol_e.int_pts[0].eff_stress_gradient,
                               -4.261738929100220E+05)
        self.assertAlmostEqual(self.consol_e.int_pts[1].eff_stress_gradient,
                               -5.844194656007840E+05)
        self.assertAlmostEqual(self.consol_e.int_pts[0].pre_consol_stress,
                               7.792077237928290E+04)
        self.assertAlmostEqual(self.consol_e.int_pts[1].pre_consol_stress,
                               1.068540727404800E+05)
        self.assertAlmostEqual(self.consol_e.int_pts[0].water_flux_rate,
                               4.339596237668420E-10, delta=1e-17)
        self.assertAlmostEqual(self.consol_e.int_pts[1].water_flux_rate,
                               4.664043445100810E-10, delta=1e-17)


class TestConsolidationElement1DCubic(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(4))
        self.consol_e = ConsolidationElement1D(self.nodes, order=3)

    def test_jacobian_value(self):
        expected0 = self.nodes[-1].z - self.nodes[0].z
        expected1 = 6.0
        self.assertAlmostEqual(self.consol_e.jacobian, expected0)
        self.assertAlmostEqual(self.consol_e.jacobian, expected1)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.consol_e.nodes):
            self.assertIs(nd, e_nd)

    def test_stiffness_matrix_uninitialized(self):
        self.assertTrue(np.allclose(
            self.consol_e.stiffness_matrix, np.zeros((4, 4))))

    def test_stiffness_matrix_full(self):
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
        jac = 6.0
        Gs = 2.60
        e0 = 0.9
        e = 0.3
        Cku = 0.305
        k0 = 4.05e-4
        ek0 = 2.60
        k = k0 * 10 ** ((e - ek0) / Cku)
        dk_de = k * np.log(10) / Cku
        sig_p_0 = 2.80
        Ccu = 0.421
        ecu0 = 2.60
        sig_p = sig_p_0 * 10 ** (-(e - ecu0) / Ccu)
        dsig_de = -sig_p * np.log(10) / Ccu
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
        jac = 6.0
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

    def test_mass_matrix_full(self):
        for ip in self.consol_e.int_pts:
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.2
        jac = 6.0
        e0 = 0.3
        Sw = 0.2
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

    def test_update_integration_points_null_material(self):
        Te = np.array([
            -1.00,
            -0.10,
            1.10,
            2.00,
        ])
        dTdte = np.array([
            -0.5,
            -0.1,
            0.5,
            0.8,
        ])
        for T, dTdt, nd in zip(Te, dTdte, self.consol_e.nodes):
            nd.temp = T
            nd.temp_rate = dTdt
        self.consol_e.update_integration_points()
        expected_Tip = np.array([
            -0.913964840018686,
            -0.436743906025892,
            0.500000000000000,
            1.436743906025890,
            1.913964840018690,
        ])
        expected_dTdtip = np.array([
            -0.474536483402387,
            -0.267597978008154,
            0.206250000000000,
            0.647478693241513,
            0.794655768169027,
        ])
        expected_dTdZip = np.array([
            0.33535785429992,
            0.51464214570008,
            0.61250000000000,
            0.51464214570008,
            0.33535785429992,
        ])
        expected_Sw = np.array([
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
        ])
        expected_dSw_dT = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        expected_qw = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        for ip, eT, edTdt, edTdZ, eSw, edSw, eqw in zip(
            self.consol_e.int_pts,
            expected_Tip,
            expected_dTdtip,
            expected_dTdZip,
            expected_Sw,
            expected_dSw_dT,
            expected_qw,
        ):
            self.assertAlmostEqual(ip.temp, eT)
            self.assertAlmostEqual(ip.temp_rate, edTdt)
            self.assertAlmostEqual(ip.temp_gradient, edTdZ)
            self.assertAlmostEqual(ip.deg_sat_water, eSw)
            self.assertAlmostEqual(ip.deg_sat_water_temp_gradient, edSw)
            self.assertAlmostEqual(ip.water_flux_rate, eqw)

    def test_update_integration_points_with_material(self):
        m = Material(
            deg_sat_water_alpha=1.2e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=10.0e-6,
            seg_pot_0=2e-9,
        )
        for ip in self.consol_e.int_pts:
            ip.material = m
            ip.tot_stress = 120.0e3
        Te = np.array([
            -1.00,
            -0.10,
            1.10,
            2.00,
        ])
        dTdte = np.array([
            -0.5,
            -0.1,
            0.5,
            0.8,
        ])
        for T, dTdt, nd in zip(Te, dTdte, self.consol_e.nodes):
            nd.temp = T
            nd.temp_rate = dTdt
        self.consol_e.update_integration_points()
        expected_Tip = np.array([
            -0.913964840018686,
            -0.436743906025892,
            0.500000000000000,
            1.436743906025890,
            1.913964840018690,
        ])
        expected_dTdtip = np.array([
            -0.474536483402387,
            -0.267597978008154,
            0.206250000000000,
            0.647478693241513,
            0.794655768169027,
        ])
        expected_dTdZip = np.array([
            0.33535785429992,
            0.51464214570008,
            0.61250000000000,
            0.51464214570008,
            0.33535785429992,
        ])
        expected_Sw = np.array([
            0.0915235681884727,
            0.1361684964587000,
            1.00000000000000,
            1.00000000000000,
            1.00000000000000,
        ])
        expected_dSw_dT = np.array([
            0.0539532190585967,
            0.1674525178303510,
            0.00000000000000,
            0.00000000000000,
            0.00000000000000,
        ])
        expected_qw = np.array([
            -1.3562595181e-11,
            -1.3792048602e-10,
            0.0,
            0.0,
            0.0,
        ])
        for ip, eT, edTdt, edTdZ, eSw, edSw, eqw in zip(
            self.consol_e.int_pts,
            expected_Tip,
            expected_dTdtip,
            expected_dTdZip,
            expected_Sw,
            expected_dSw_dT,
            expected_qw,
        ):
            self.assertAlmostEqual(ip.temp, eT)
            self.assertAlmostEqual(ip.temp_rate, edTdt)
            self.assertAlmostEqual(ip.temp_gradient, edTdZ)
            self.assertAlmostEqual(ip.deg_sat_water, eSw)
            self.assertAlmostEqual(ip.deg_sat_water_temp_gradient, edSw)
            self.assertAlmostEqual(ip.water_flux_rate, eqw, delta=1e-18)


if __name__ == "__main__":
    unittest.main()
