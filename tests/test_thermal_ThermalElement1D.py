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

    def test_update_integration_points_null_material(self):
        self.thrm_e.nodes[0].temp = -1.0
        self.thrm_e.nodes[1].temp = +2.0
        self.thrm_e.update_integration_points()
        self.assertAlmostEqual(self.thrm_e.int_pts[0].temp, -0.366025403784439)
        self.assertAlmostEqual(self.thrm_e.int_pts[1].temp, 1.366025403784440)
        self.assertAlmostEqual(self.thrm_e.int_pts[0].deg_sat_water, 0.0)
        self.assertAlmostEqual(self.thrm_e.int_pts[1].deg_sat_water, 1.0)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].deg_sat_water_temp_gradient, 0.0)
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].deg_sat_water_temp_gradient, 0.0)
        # TODO: test temp_rate, temp_gradient, water_flux_rate

    def test_update_integration_points_with_material(self):
        m = Material(deg_sat_water_alpha=1.2e4, deg_sat_water_beta=0.35)
        for ip in self.thrm_e.int_pts:
            ip.material = m
        self.thrm_e.nodes[0].temp = -1.0
        self.thrm_e.nodes[1].temp = +2.0
        self.thrm_e.update_integration_points()
        self.assertAlmostEqual(
            self.thrm_e.int_pts[0].temp,
            -0.366025403784439,
        )
        self.assertAlmostEqual(
            self.thrm_e.int_pts[1].temp,
            1.366025403784440,
        )
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
        # TODO: test temp_rate, temp_gradient, water_flux_rate


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

    def test_update_integration_points_null_material(self):
        Te = np.array([
            -1.00,
            -0.10,
            1.10,
            2.00,
        ])
        for T, nd in zip(Te, self.thrm_e.nodes):
            nd.temp = T
        self.thrm_e.update_integration_points()
        expected_Tip = np.array([
            -0.913964840018686,
            -0.436743906025892,
            0.500000000000000,
            1.436743906025890,
            1.913964840018690,
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
        for ip, eT, eSw, edSw in zip(
            self.thrm_e.int_pts,
            expected_Tip,
            expected_Sw,
            expected_dSw_dT,
        ):
            self.assertAlmostEqual(ip.temp, eT)
            self.assertAlmostEqual(ip.deg_sat_water, eSw)
            self.assertAlmostEqual(ip.deg_sat_water_temp_gradient, edSw)
            # TODO: test temp_rate, temp_gradient, water_flux_rate

    def test_update_integration_points_with_material(self):
        m = Material(deg_sat_water_alpha=1.2e4, deg_sat_water_beta=0.35)
        for ip in self.thrm_e.int_pts:
            ip.material = m
        Te = np.array([
            -1.00,
            -0.10,
            1.10,
            2.00,
        ])
        for T, nd in zip(Te, self.thrm_e.nodes):
            nd.temp = T
        self.thrm_e.update_integration_points()
        expected_Tip = np.array([
            -0.913964840018686,
            -0.436743906025892,
            0.500000000000000,
            1.436743906025890,
            1.913964840018690,
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
        for ip, eT, eSw, edSw in zip(
            self.thrm_e.int_pts,
            expected_Tip,
            expected_Sw,
            expected_dSw_dT,
        ):
            self.assertAlmostEqual(ip.temp, eT)
            self.assertAlmostEqual(ip.deg_sat_water, eSw)
            self.assertAlmostEqual(ip.deg_sat_water_temp_gradient, edSw)
            # TODO: test temp_rate, temp_gradient, water_flux_rate


if __name__ == "__main__":
    unittest.main()
