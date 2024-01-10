import unittest

import numpy as np

from frozen_ground_fem.materials import (
    Material,
    vol_heat_cap_water as Cw,
)
from frozen_ground_fem.geometry import (
    Node1D,
)
from frozen_ground_fem.thermal import (
    ThermalElement1D,
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

    def test_heat_flow_matrix_conduction_only(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
        e_fact = 1.30 / 1.35
        lam = 2.0875447196636
        jac = 2.0
        expected = (
            lam / jac * np.array([[1.0, -1.0], [-1.0, 1.0]]) * e_fact ** 2
        )
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix, expected))

    def test_heat_flow_matrix_conduction_advection(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.deg_sat_water_temp_gradient = 1.5e-2
            ip.water_flux_rate = -1.5e-8
        e_fact = 1.30 / 1.35
        lam = 2.0875447196636
        qw = -1.5e-8
        jac = 2.0
        expected = (
            lam / jac * np.array([[1.0, -1.0], [-1.0, 1.0]]) * e_fact ** 2
            + qw * Cw * e_fact * np.array([[-0.5, 0.5], [-0.5, 0.5]])
        )
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix, expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix, np.zeros((2, 2)))
        )

    def test_heat_storage_matrix_heat_capacity_only(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=7.41e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.water_flux_rate = -1.5e-8
        heat_cap = 2.42402962962963e6
        lat_heat = 0.0
        jac = 2.0
        expected = (
            (heat_cap + lat_heat) * jac * np.array([[2.0, 1.0], [1.0, 2.0]])
            / 6.0
        )
        self.assertTrue(np.allclose(
            self.thrm_e.heat_storage_matrix, expected))

    def test_heat_storage_matrix_heat_capacity_and_latent_heat(self):
        m = Material(spec_grav_solids=2.65, spec_heat_cap_solids=7.41e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.deg_sat_water_temp_gradient = 1.5e-2
            ip.water_flux_rate = -1.5e-8
        heat_cap = 2.42402962962963e6
        lat_heat = 1.18039638888889e6
        jac = 2.0
        expected = (
            (heat_cap + lat_heat) * jac * np.array([[2.0, 1.0], [1.0, 2.0]])
            / 6.0
        )
        self.assertTrue(np.allclose(
            self.thrm_e.heat_storage_matrix, expected))

    def test_update_integration_points_null_material(self):
        self.thrm_e.nodes[0].temp = -1.0
        self.thrm_e.nodes[1].temp = +2.0
        self.thrm_e.nodes[0].temp_rate = 0.2
        self.thrm_e.nodes[1].temp_rate = 0.4
        self.thrm_e.update_integration_points()
        self.assertAlmostEqual(self.thrm_e.int_pts[0].temp, -0.366025403784439)
        self.assertAlmostEqual(self.thrm_e.int_pts[1].temp, 1.366025403784440)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].temp_rate, 0.242264973081037)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].temp_rate, 0.357735026918963)
        self.assertAlmostEqual(self.thrm_e.int_pts[0].temp_gradient, 1.5)
        self.assertAlmostEqual(self.thrm_e.int_pts[1].temp_gradient, 1.5)
        self.assertAlmostEqual(self.thrm_e.int_pts[0].deg_sat_water, 0.0)
        self.assertAlmostEqual(self.thrm_e.int_pts[1].deg_sat_water, 1.0)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].deg_sat_water_temp_gradient, 0.0)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].deg_sat_water_temp_gradient, 0.0)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].water_flux_rate, 0.0
        )
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].water_flux_rate, 0.0
        )

    def test_update_integration_points_with_material(self):
        m = Material(
            deg_sat_water_alpha=1.2e4,
            deg_sat_water_beta=0.35,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=10.0e-6,
            seg_pot_0=2e-9,
        )
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.tot_stress = 120.0e3
        self.thrm_e.nodes[0].temp = -1.0
        self.thrm_e.nodes[1].temp = +2.0
        self.thrm_e.nodes[0].temp_rate = 0.2
        self.thrm_e.nodes[1].temp_rate = 0.4
        self.thrm_e.update_integration_points()
        self.assertAlmostEqual(self.thrm_e.int_pts[0].temp, -0.366025403784439)
        self.assertAlmostEqual(self.thrm_e.int_pts[1].temp, 1.366025403784440)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].temp_rate, 0.242264973081037)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].temp_rate, 0.357735026918963)
        self.assertAlmostEqual(self.thrm_e.int_pts[0].temp_gradient, 1.5)
        self.assertAlmostEqual(self.thrm_e.int_pts[1].temp_gradient, 1.5)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].deg_sat_water,
            0.149711781050801,
        )
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].deg_sat_water,
            1.0,
        )
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].deg_sat_water_temp_gradient,
            0.219419354111454,
        )
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].deg_sat_water_temp_gradient,
            0.0,
        )
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].water_flux_rate,
            1.1378090109e-10,
            delta=1e-17,
        )
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].water_flux_rate, 0.0,
            delta=1e-17,
        )


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

    def test_heat_flow_matrix_conduction_only(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
        lam = 2.0875447196636
        jac = 6.0
        e_fact = 1.30 / 1.35
        expected = (1/40) * lam / jac * np.array(
            [
                [148, -189, 54, -13],
                [-189, 432, -297, 54],
                [54, -297, 432, -189],
                [-13, 54, -189, 148],
            ]
        ) * e_fact ** 2
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix, expected))

    def test_heat_flow_matrix_conduction_advection(self):
        m = Material(thrm_cond_solids=3.0)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.water_flux_rate = -1.5e-8
        lam = 2.0875447196636
        jac = 6.0
        e_fact = 1.30 / 1.35
        qw = -1.5e-8
        expected = (1/40) * lam / jac * np.array(
            [
                [148, -189, 54, -13],
                [-189, 432, -297, 54],
                [54, -297, 432, -189],
                [-13, 54, -189, 148],
            ]
        ) * e_fact ** 2
        expected += (1/1680) * qw * Cw * np.array(
            [
                [-840, 1197, -504, 147],
                [-1197, 0, 1701, -504],
                [504, -1701, 0, 1197],
                [-147, 504, -1197, 840],
            ]
        ) * e_fact
        self.assertTrue(np.allclose(self.thrm_e.heat_flow_matrix, expected))

    def test_heat_storage_matrix_uninitialized(self):
        self.assertTrue(
            np.allclose(self.thrm_e.heat_storage_matrix, np.zeros((4, 4)))
        )

    def test_heat_storage_matrix_heat_capacity_only(self):
        m = Material(spec_grav_solids=2.65,
                     spec_heat_cap_solids=7.41e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.water_flux_rate = -1.5e-8
        heat_cap = 2.42402962962963e6
        lat_heat = 0.0
        jac = 6.0
        expected = (1/1680) * (heat_cap + lat_heat) * jac * np.array(
            [
                [128,   99,  -36,   19],
                [99,  648,  -81,  -36],
                [-36,  -81,  648,   99],
                [19,  -36,   99,  128],
            ]
        )
        self.assertTrue(np.allclose(
            self.thrm_e.heat_storage_matrix, expected))

    def test_heat_storage_matrix_heat_capacity_and_latent_heat(self):
        m = Material(spec_grav_solids=2.65,
                     spec_heat_cap_solids=7.41e2)
        for ip in self.thrm_e.int_pts:
            ip.material = m
            ip.void_ratio = 0.35
            ip.void_ratio_0 = 0.3
            ip.deg_sat_water = 0.8
            ip.deg_sat_water_temp_gradient = 1.5e-2
            ip.water_flux_rate = -1.5e-8
        heat_cap = 2.42402962962963e6
        lat_heat = 1.18039638888889e6
        jac = 6.0
        expected = (1/1680) * (heat_cap + lat_heat) * jac * np.array(
            [
                [128,   99,  -36,   19],
                [99,  648,  -81,  -36],
                [-36,  -81,  648,   99],
                [19,  -36,   99,  128],
            ]
        )
        self.assertTrue(np.allclose(
            self.thrm_e.heat_storage_matrix, expected))

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
        for T, dTdt, nd in zip(Te, dTdte, self.thrm_e.nodes):
            nd.temp = T
            nd.temp_rate = dTdt
        self.thrm_e.update_integration_points()
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
            self.thrm_e.int_pts,
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
        for ip in self.thrm_e.int_pts:
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
        for T, dTdt, nd in zip(Te, dTdte, self.thrm_e.nodes):
            nd.temp = T
            nd.temp_rate = dTdt
        self.thrm_e.update_integration_points()
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
            self.thrm_e.int_pts,
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
