import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
    vol_heat_cap_water as Cw,
    unit_weight_water as gam_w,
    spec_grav_ice as Gi,
)
from frozen_ground_fem.geometry import (
    Node1D,
)
from frozen_ground_fem.coupled import (
    CoupledElement1D,
)


class TestCoupledElement1DLinear(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.coup_e = CoupledElement1D(self.nodes, order=1)

    def test_jacobian_value(self):
        expected0 = self.nodes[-1].z - self.nodes[0].z
        expected1 = 2.0
        self.assertAlmostEqual(self.coup_e.jacobian, expected0)
        self.assertAlmostEqual(self.coup_e.jacobian, expected1)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.coup_e.nodes):
            self.assertIs(nd, e_nd)

    def test_heat_flow_matrix_uninitialized(self):
        self.assertTrue(np.allclose(self.coup_e.heat_flow_matrix, np.zeros((2, 2))))

    def test_heat_flow_matrix_conduction_only(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
        e_fact = 1.30 / 1.35
        lam = 2.0875447196636
        jac = 2.0
        expected = lam / jac * np.array([[1.0, -1.0], [-1.0, 1.0]]) * e_fact**2
        self.assertTrue(np.allclose(self.coup_e.heat_flow_matrix, expected))

    def test_heat_flux_vector_advection(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.water_flux_rate = -1.5e-8
            ip.temp_gradient = 0.5
        e_fact = 1.30 / 1.35
        qw = -1.5e-8
        dTdZ = 0.5
        jac = 2.0
        expected = -qw * Cw * dTdZ * e_fact * np.array([0.5, 0.5]) * jac
        self.assertTrue(np.allclose(self.coup_e.heat_flux_vector, expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(np.allclose(self.coup_e.heat_storage_matrix, np.zeros((2, 2))))

    def test_heat_storage_matrix_heat_capacity_only(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=7.41e2)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.water_flux_rate = -1.5e-8
        heat_cap = 2.42402962962963e6
        lat_heat = 0.0
        jac = 2.0
        expected = (
            (heat_cap + lat_heat) * jac * np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
        )
        self.assertTrue(np.allclose(self.coup_e.heat_storage_matrix, expected))

    def test_heat_storage_matrix_heat_capacity_and_latent_heat(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=7.41e2)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.vol_water_cont_temp_gradient = 2.5e-2
        heat_cap = 2.42402962962963e6
        lat_heat = 7.588262500e6
        jac = 2.0
        expected = (
            (heat_cap + lat_heat) * jac * np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
        )
        self.assertTrue(np.allclose(self.coup_e.heat_storage_matrix, expected))

    def test_stiffness_matrix_uninitialized(self):
        self.assertTrue(np.allclose(self.coup_e.stiffness_matrix, np.zeros((2, 2))))

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
        for ip in self.coup_e.int_pts:
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
        coef_1 = dk_de * (Gs - 1.0) / (1.0 + e) - k * (Gs - 1.0) / (1.0 + e) ** 2
        stiff_0 = coef_0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
        stiff_1 = coef_1 * np.array([[-0.5, 0.5], [-0.5, 0.5]])
        expected = stiff_0 + stiff_1
        self.assertTrue(np.allclose(self.coup_e.stiffness_matrix, expected))

    def test_mass_matrix_uninitialized(self):
        jac = 2.0
        expected = jac / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(self.coup_e.mass_matrix, expected))

    def test_mass_matrix_full(self):
        for ip in self.coup_e.int_pts:
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.2
        jac = 2.0
        e0 = 0.3
        Sw = 0.2
        coef = (Sw + Gi * (1.0 - Sw)) / (1.0 + e0)
        expected = coef * jac / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(self.coup_e.mass_matrix, expected))

    def test_update_integration_points_null_material(self):
        self.coup_e.nodes[0].temp = -1.0
        self.coup_e.nodes[1].temp = +2.0
        self.coup_e.nodes[0].temp_rate = 0.2
        self.coup_e.nodes[1].temp_rate = 0.4
        self.coup_e.nodes[0].void_ratio = 0.75
        self.coup_e.nodes[1].void_ratio = 0.65
        for ip in self.coup_e.int_pts:
            ip.void_ratio_0 = 0.9
        self.coup_e.update_integration_points_primary()
        self.coup_e.update_integration_points_secondary()
        self.assertAlmostEqual(self.coup_e.int_pts[0].temp, -0.366025403784439)
        self.assertAlmostEqual(self.coup_e.int_pts[1].temp, 1.366025403784440)
        self.assertAlmostEqual(self.coup_e.int_pts[0].temp_rate, 0.242264973081037)
        self.assertAlmostEqual(self.coup_e.int_pts[1].temp_rate, 0.357735026918963)
        self.assertAlmostEqual(self.coup_e.int_pts[0].temp_gradient, 1.5)
        self.assertAlmostEqual(self.coup_e.int_pts[1].temp_gradient, 1.5)
        self.assertAlmostEqual(self.coup_e.int_pts[0].deg_sat_water, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].deg_sat_water, 1.0)
        self.assertAlmostEqual(self.coup_e.int_pts[0].deg_sat_water_temp_gradient, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].deg_sat_water_temp_gradient, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[0].water_flux_rate, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].water_flux_rate, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[0].void_ratio, 0.728867513459481)
        self.assertAlmostEqual(self.coup_e.int_pts[1].void_ratio, 0.671132486540519)
        self.assertAlmostEqual(self.coup_e.int_pts[0].hyd_cond, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].hyd_cond, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[0].hyd_cond_gradient, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].hyd_cond_gradient, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[0].eff_stress, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].eff_stress, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[0].eff_stress_gradient, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].eff_stress_gradient, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[0].pre_consol_stress, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].pre_consol_stress, 0.0)
        # TODO: Test total stress

    def test_update_integration_points_with_material(self):
        m = Material(
            deg_sat_water_alpha=1.2e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=10.0e-6,
            seg_pot_0=2e-9,
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
            comp_index_frozen_a1=0.021,
            comp_index_frozen_a2=0.01,
            comp_index_frozen_a3=0.23,
        )
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.tot_stress = 120.0e3
            ip.void_ratio_0 = 0.9
        self.coup_e.nodes[0].temp = -1.0
        self.coup_e.nodes[1].temp = +2.0
        self.coup_e.nodes[0].temp_rate = 0.2
        self.coup_e.nodes[1].temp_rate = 0.4
        self.coup_e.nodes[0].void_ratio = 0.75
        self.coup_e.nodes[1].void_ratio = 0.65
        self.coup_e.update_integration_points_primary()
        self.coup_e.update_integration_points_secondary()
        self.assertAlmostEqual(self.coup_e.int_pts[0].temp, -0.366025403784439)
        self.assertAlmostEqual(self.coup_e.int_pts[1].temp, 1.366025403784440)
        self.assertAlmostEqual(self.coup_e.int_pts[0].temp_rate, 0.242264973081037)
        self.assertAlmostEqual(self.coup_e.int_pts[1].temp_rate, 0.357735026918963)
        self.assertAlmostEqual(self.coup_e.int_pts[0].temp_gradient, 1.5)
        self.assertAlmostEqual(self.coup_e.int_pts[1].temp_gradient, 1.5)
        self.assertAlmostEqual(
            self.coup_e.int_pts[0].deg_sat_water,
            0.149711781050801,
        )
        self.assertAlmostEqual(
            self.coup_e.int_pts[1].deg_sat_water,
            1.0,
        )
        self.assertAlmostEqual(
            self.coup_e.int_pts[0].deg_sat_water_temp_gradient,
            0.0,
        )
        self.assertAlmostEqual(
            self.coup_e.int_pts[1].deg_sat_water_temp_gradient,
            0.0,
        )
        self.assertAlmostEqual(self.coup_e.int_pts[0].void_ratio, 0.728867513459481)
        self.assertAlmostEqual(self.coup_e.int_pts[1].void_ratio, 0.671132486540519)
        self.assertAlmostEqual(self.coup_e.int_pts[0].hyd_cond, 0.0, delta=1e-17)
        self.assertAlmostEqual(
            self.coup_e.int_pts[1].hyd_cond, 1.919991214136810e-10, delta=1e-17
        )
        self.assertAlmostEqual(
            self.coup_e.int_pts[0].hyd_cond_gradient, 0.0, delta=1e-17
        )
        self.assertAlmostEqual(
            self.coup_e.int_pts[1].hyd_cond_gradient, 1.449489556836380e-09, delta=1e-17
        )
        self.assertAlmostEqual(self.coup_e.int_pts[0].eff_stress, 0.0)
        self.assertAlmostEqual(self.coup_e.int_pts[1].eff_stress, 1.068540727404800e05)
        self.assertAlmostEqual(self.coup_e.int_pts[0].eff_stress_gradient, 0.0)
        self.assertAlmostEqual(
            self.coup_e.int_pts[1].eff_stress_gradient, -5.844194656007840e05
        )
        self.assertAlmostEqual(self.coup_e.int_pts[0].pre_consol_stress, 0.0)
        self.assertAlmostEqual(
            self.coup_e.int_pts[1].pre_consol_stress, 1.068540727404800e05
        )
        self.assertAlmostEqual(
            self.coup_e.int_pts[0].water_flux_rate,
            1.25043538838904e-10,
            delta=1e-17,
        )
        self.assertAlmostEqual(
            self.coup_e.int_pts[1].water_flux_rate, 4.664043445100810e-10, delta=1e-17
        )
        # TODO: Test total stress

    def test_deformed_length(self):
        self.coup_e.nodes[0].void_ratio = 0.75
        self.coup_e.nodes[1].void_ratio = 0.65
        for ip in self.coup_e.int_pts:
            ip.void_ratio_0 = 0.9
        self.coup_e.update_integration_points_primary()
        self.coup_e.update_integration_points_secondary()
        expected = 1.7894736842105
        self.assertAlmostEqual(self.coup_e.deformed_length, expected)


class TestCoupledElement1DCubic(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(4))
        self.coup_e = CoupledElement1D(self.nodes, order=3)

    def test_jacobian_value(self):
        expected0 = self.nodes[-1].z - self.nodes[0].z
        expected1 = 6.0
        self.assertAlmostEqual(self.coup_e.jacobian, expected0)
        self.assertAlmostEqual(self.coup_e.jacobian, expected1)

    def test_nodes_equal(self):
        for nd, e_nd in zip(self.nodes, self.coup_e.nodes):
            self.assertIs(nd, e_nd)

    def test_heat_flow_matrix_uninitialized(self):
        self.assertTrue(np.allclose(self.coup_e.heat_flow_matrix, np.zeros((4, 4))))

    def test_heat_flow_matrix_conduction_only(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
        lam = 2.0875447196636
        jac = 6.0
        e_fact = 1.30 / 1.35
        expected = (
            (1 / 40)
            * lam
            / jac
            * np.array(
                [
                    [148, -189, 54, -13],
                    [-189, 432, -297, 54],
                    [54, -297, 432, -189],
                    [-13, 54, -189, 148],
                ]
            )
            * e_fact**2
        )
        self.assertTrue(np.allclose(self.coup_e.heat_flow_matrix, expected))

    def test_heat_flux_vector_advection(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.water_flux_rate = -1.5e-8
            ip.temp_gradient = 0.5
        jac = 6.0
        e_fact = 1.30 / 1.35
        qw = -1.5e-8
        dTdZ = 0.5
        expected = (
            -qw * Cw * dTdZ * jac * np.array([0.125, 0.375, 0.375, 0.125]) * e_fact
        )
        self.assertTrue(np.allclose(self.coup_e.heat_flux_vector, expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(np.allclose(self.coup_e.heat_storage_matrix, np.zeros((4, 4))))

    def test_heat_storage_matrix_heat_capacity_only(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=7.41e2)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.water_flux_rate = -1.5e-8
        heat_cap = 2.42402962962963e6
        lat_heat = 0.0
        jac = 6.0
        expected = (
            (1 / 1680)
            * (heat_cap + lat_heat)
            * jac
            * np.array(
                [
                    [128, 99, -36, 19],
                    [99, 648, -81, -36],
                    [-36, -81, 648, 99],
                    [19, -36, 99, 128],
                ]
            )
        )
        self.assertTrue(np.allclose(self.coup_e.heat_storage_matrix, expected))

    def test_heat_storage_matrix_heat_capacity_and_latent_heat(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=7.41e2)
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.vol_water_cont_temp_gradient = 2.5e-2
        heat_cap = 2.42402962962963e6
        lat_heat = 7.588262500e6
        jac = 6.0
        expected = (
            (1 / 1680)
            * (heat_cap + lat_heat)
            * jac
            * np.array(
                [
                    [128, 99, -36, 19],
                    [99, 648, -81, -36],
                    [-36, -81, 648, 99],
                    [19, -36, 99, 128],
                ]
            )
        )
        self.assertTrue(np.allclose(self.coup_e.heat_storage_matrix, expected))

    def test_stiffness_matrix_uninitialized(self):
        self.assertTrue(np.allclose(self.coup_e.stiffness_matrix, np.zeros((4, 4))))

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
        for ip in self.coup_e.int_pts:
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
        coef_1 = dk_de * (Gs - 1.0) / (1.0 + e) - k * (Gs - 1.0) / (1.0 + e) ** 2
        stiff_0 = (
            coef_0
            / 40.0
            * np.array(
                [
                    [148, -189, 54, -13],
                    [-189, 432, -297, 54],
                    [54, -297, 432, -189],
                    [-13, 54, -189, 148],
                ]
            )
        )
        stiff_1 = (
            coef_1
            / 1680.0
            * np.array(
                [
                    [-840, 1197, -504, 147],
                    [-1197, 0, 1701, -504],
                    [504, -1701, 0, 1197],
                    [-147, 504, -1197, 840],
                ]
            )
        )
        expected = stiff_0 + stiff_1
        self.assertTrue(np.allclose(self.coup_e.stiffness_matrix, expected))

    def test_mass_matrix_uninitialized(self):
        jac = 6.0
        expected = (
            jac
            / 1680.0
            * np.array(
                [
                    [128, 99, -36, 19],
                    [99, 648, -81, -36],
                    [-36, -81, 648, 99],
                    [19, -36, 99, 128],
                ]
            )
        )
        self.assertTrue(np.allclose(self.coup_e.mass_matrix, expected))

    def test_mass_matrix_full(self):
        for ip in self.coup_e.int_pts:
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.2
        jac = 6.0
        e0 = 0.3
        Sw = 0.2
        coef = (Sw + Gi * (1.0 - Sw)) / (1.0 + e0)
        expected = (
            coef
            * jac
            / 1680.0
            * np.array(
                [
                    [128, 99, -36, 19],
                    [99, 648, -81, -36],
                    [-36, -81, 648, 99],
                    [19, -36, 99, 128],
                ]
            )
        )
        self.assertTrue(np.allclose(self.coup_e.mass_matrix, expected))

    def test_update_integration_points_null_material(self):
        Te = np.array(
            [
                -1.00,
                -0.10,
                1.10,
                2.00,
            ]
        )
        dTdte = np.array(
            [
                -0.5,
                -0.1,
                0.5,
                0.8,
            ]
        )
        enode = np.array(
            [
                1.1,
                0.89,
                0.75,
                0.7,
            ]
        )
        for T, dTdt, e, nd in zip(Te, dTdte, enode, self.coup_e.nodes):
            nd.temp = T
            nd.temp_rate = dTdt
            nd.void_ratio = e
        for ip in self.coup_e.int_pts:
            ip.void_ratio_0 = 0.9
        self.coup_e.update_integration_points_primary()
        self.coup_e.update_integration_points_secondary()
        expected_Tip = np.array(
            [
                -0.913964840018686,
                -0.436743906025892,
                0.500000000000000,
                1.436743906025890,
                1.913964840018690,
            ]
        )
        expected_dTdtip = np.array(
            [
                -0.474536483402387,
                -0.267597978008154,
                0.206250000000000,
                0.647478693241513,
                0.794655768169027,
            ]
        )
        expected_dTdZip = np.array(
            [
                0.33535785429992,
                0.51464214570008,
                0.61250000000000,
                0.51464214570008,
                0.33535785429992,
            ]
        )
        expected_Sw = np.array(
            [
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
            ]
        )
        expected_dSw_dT = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        expected_qw = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        expected_e = np.array(
            [
                1.066963710411440,
                0.948090621196697,
                0.810000000000000,
                0.724100234429932,
                0.700845433961932,
            ]
        )
        for ip, eT, edTdt, edTdZ, eSw, edSw, eqw, ee in zip(
            self.coup_e.int_pts,
            expected_Tip,
            expected_dTdtip,
            expected_dTdZip,
            expected_Sw,
            expected_dSw_dT,
            expected_qw,
            expected_e,
        ):
            self.assertAlmostEqual(ip.temp, eT)
            self.assertAlmostEqual(ip.temp_rate, edTdt)
            self.assertAlmostEqual(ip.temp_gradient, edTdZ)
            self.assertAlmostEqual(ip.deg_sat_water, eSw)
            self.assertAlmostEqual(ip.deg_sat_water_temp_gradient, edSw)
            self.assertAlmostEqual(ip.water_flux_rate, eqw)
            self.assertAlmostEqual(ip.void_ratio, ee)
            self.assertEqual(ip.hyd_cond, 0.0)
            self.assertEqual(ip.hyd_cond_gradient, 0.0)
            self.assertEqual(ip.eff_stress, 0.0)
            self.assertEqual(ip.eff_stress_gradient, 0.0)
            self.assertEqual(ip.pre_consol_stress, 0.0)
        # TODO: Test total stress

    def test_update_integration_points_with_material(self):
        m = Material(
            deg_sat_water_alpha=1.2e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=10.0e-6,
            seg_pot_0=2e-9,
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
            comp_index_frozen_a1=0.021,
            comp_index_frozen_a2=0.01,
            comp_index_frozen_a3=0.23,
        )
        for ip in self.coup_e.int_pts:
            ip.material = m
            ip.tot_stress = 120.0e3
            ip.void_ratio_0 = 0.9
        Te = np.array(
            [
                -1.00,
                -0.10,
                1.10,
                2.00,
            ]
        )
        dTdte = np.array(
            [
                -0.5,
                -0.1,
                0.5,
                0.8,
            ]
        )
        ee = np.array(
            [
                1.1,
                0.89,
                0.75,
                0.7,
            ]
        )
        for T, dTdt, e, nd in zip(Te, dTdte, ee, self.coup_e.nodes):
            nd.temp = T
            nd.temp_rate = dTdt
            nd.void_ratio = e
        self.coup_e.update_integration_points_primary()
        self.coup_e.update_integration_points_secondary()
        expected_Tip = np.array(
            [
                -0.913964840018686,
                -0.436743906025892,
                0.500000000000000,
                1.436743906025890,
                1.913964840018690,
            ]
        )
        expected_dTdtip = np.array(
            [
                -0.474536483402387,
                -0.267597978008154,
                0.206250000000000,
                0.647478693241513,
                0.794655768169027,
            ]
        )
        expected_dTdZip = np.array(
            [
                0.33535785429992,
                0.51464214570008,
                0.61250000000000,
                0.51464214570008,
                0.33535785429992,
            ]
        )
        expected_Sw = np.array(
            [
                0.0915235681884727,
                0.1361684964587000,
                1.00000000000000,
                1.00000000000000,
                1.00000000000000,
            ]
        )
        expected_dSw_dT = np.array(
            [
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
                0.00000000000000,
            ]
        )
        expected_e_ip = np.array(
            [
                1.066963710411440,
                0.948090621196697,
                0.810000000000000,
                0.724100234429932,
                0.700845433961932,
            ]
        )
        expected_k_ip = np.array(
            [
                0.0,
                0.0,
                5.477754498683320e-10,
                2.863940483811450e-10,
                2.402806184667600e-10,
            ]
        )
        expected_dkde_ip = np.array(
            [
                0.0,
                0.0,
                4.135408475983370e-09,
                2.162120218113580e-09,
                1.813988754809670e-09,
            ]
        )
        expected_sigp_ip = np.array(
            [
                0.0,
                0.0,
                4.999648864537800e04,
                7.997918088269650e04,
                9.082679879495540e04,
            ]
        )
        expected_dsigde_ip = np.array(
            [
                0.0,
                0.0,
                -2.734469583299130e05,
                -4.374319944189340e05,
                -4.967611233958050e05,
            ]
        )
        expected_ppc_ip = np.array(
            [
                0.0,
                0.0,
                4.999648864537800e04,
                7.997918088269650e04,
                9.082679879495540e04,
            ]
        )
        expected_qw_ip = np.array(
            [
                -1.24670456065463e-11,
                -1.34515776932174e-10,
                6.444230380341150e-10,
                2.246276375886120e-10,
                -1.335017975631420e-10,
            ]
        )
        for ip, eT, edTdt, edTdZ, eSw, edSw, eqw, e, k, dkde, sigp, dsigde, ppc in zip(
            self.coup_e.int_pts,
            expected_Tip,
            expected_dTdtip,
            expected_dTdZip,
            expected_Sw,
            expected_dSw_dT,
            expected_qw_ip,
            expected_e_ip,
            expected_k_ip,
            expected_dkde_ip,
            expected_sigp_ip,
            expected_dsigde_ip,
            expected_ppc_ip,
        ):
            self.assertAlmostEqual(ip.temp, eT)
            self.assertAlmostEqual(ip.temp_rate, edTdt)
            self.assertAlmostEqual(ip.temp_gradient, edTdZ)
            self.assertAlmostEqual(ip.deg_sat_water, eSw)
            self.assertAlmostEqual(ip.deg_sat_water_temp_gradient, edSw)
            self.assertAlmostEqual(ip.water_flux_rate, eqw, delta=1e-18)
            self.assertAlmostEqual(ip.void_ratio, e)
            self.assertAlmostEqual(ip.hyd_cond, k, delta=1e-18)
            self.assertAlmostEqual(ip.hyd_cond_gradient, dkde, delta=1e-18)
            self.assertAlmostEqual(ip.eff_stress, sigp)
            self.assertAlmostEqual(ip.eff_stress_gradient, dsigde)
            self.assertAlmostEqual(ip.pre_consol_stress, ppc)
        # TODO: Test total stress

    def test_deformed_length(self):
        ee = np.array(
            [
                1.1,
                0.89,
                0.75,
                0.7,
            ]
        )
        for nd, e in zip(self.coup_e.nodes, ee):
            nd.void_ratio = e
        for ip in self.coup_e.int_pts:
            ip.void_ratio_0 = 0.9
        self.coup_e.update_integration_points_primary()
        self.coup_e.update_integration_points_secondary()
        expected = 5.8105263157895
        self.assertAlmostEqual(self.coup_e.deformed_length, expected)


if __name__ == "__main__":
    unittest.main()
