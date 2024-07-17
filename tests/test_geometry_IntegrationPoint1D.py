import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    IntegrationPoint1D,
)
from frozen_ground_fem.materials import (
    Material,
    NULL_MATERIAL,
)


class TestIntegrationPoint1DDefaults(unittest.TestCase):
    def setUp(self):
        self.p = IntegrationPoint1D()

    def test_z_value(self):
        self.assertEqual(self.p.z, 0.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.zeros((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))

    def test_local_coord_value(self):
        self.assertEqual(self.p.local_coord, 0.0)

    def test_local_coord_type(self):
        self.assertIsInstance(self.p.local_coord, float)

    def test_weight_value(self):
        self.assertEqual(self.p.weight, 0.0)

    def test_weight_type(self):
        self.assertIsInstance(self.p.weight, float)

    def test_void_ratio_value(self):
        self.assertAlmostEqual(self.p.void_ratio, 0.0)
        self.assertAlmostEqual(self.p.porosity, 0.0)

    def test_void_ratio_type(self):
        self.assertIsInstance(self.p.void_ratio, float)
        self.assertIsInstance(self.p.porosity, float)

    def test_deg_sat_water_value(self):
        self.assertAlmostEqual(self.p.deg_sat_water, 1.0)
        self.assertAlmostEqual(self.p.deg_sat_ice, 0.0)
        self.assertAlmostEqual(self.p.vol_water_cont, 0.0)

    def test_deg_sat_water_type(self):
        self.assertIsInstance(self.p.deg_sat_water, float)
        self.assertIsInstance(self.p.deg_sat_ice, float)
        self.assertIsInstance(self.p.vol_water_cont, float)

    def test_deg_sat_water_temp_gradient_value(self):
        self.assertAlmostEqual(self.p.deg_sat_water_temp_gradient, 0.0)

    def test_deg_sat_water_temp_gradient_type(self):
        self.assertIsInstance(self.p.deg_sat_water_temp_gradient, float)

    def test_material_value(self):
        self.assertIs(self.p.material, NULL_MATERIAL)

    def test_material_type(self):
        self.assertIsInstance(self.p.material, Material)

    def test_void_ratio_0_value(self):
        self.assertAlmostEqual(self.p.void_ratio_0, 0.0)

    def test_void_ratio_0_type(self):
        self.assertIsInstance(self.p.void_ratio_0, float)

    def test_temp_value(self):
        self.assertAlmostEqual(self.p.temp, 0.0)

    def test_temp_type(self):
        self.assertIsInstance(self.p.temp, float)

    def test_temp_rate_value(self):
        self.assertAlmostEqual(self.p.temp_rate, 0.0)

    def test_temp_rate_type(self):
        self.assertIsInstance(self.p.temp_rate, float)

    def test_temp_gradient_value(self):
        self.assertAlmostEqual(self.p.temp_gradient, 0.0)

    def test_temp_gradient_type(self):
        self.assertIsInstance(self.p.temp_gradient, float)

    def test_hyd_cond_value(self):
        self.assertAlmostEqual(self.p.hyd_cond, 0.0)

    def test_hyd_cond_type(self):
        self.assertIsInstance(self.p.hyd_cond, float)

    def test_hyd_cond_gradient_value(self):
        self.assertAlmostEqual(self.p.hyd_cond_gradient, 0.0)

    def test_hyd_cond_gradient_type(self):
        self.assertIsInstance(self.p.hyd_cond_gradient, float)

    def test_water_flux_rate_value(self):
        self.assertEqual(self.p.water_flux_rate, 0.0)

    def test_water_flux_rate_type(self):
        self.assertIsInstance(self.p.water_flux_rate, float)

    def test_pre_consol_stress_value(self):
        self.assertEqual(self.p.pre_consol_stress, 0.0)

    def test_pre_consol_stress_type(self):
        self.assertIsInstance(self.p.pre_consol_stress, float)

    def test_eff_stress_value(self):
        self.assertEqual(self.p.eff_stress, 0.0)

    def test_eff_stress_type(self):
        self.assertIsInstance(self.p.eff_stress, float)

    def test_eff_stress_grad_value(self):
        self.assertEqual(self.p.eff_stress_gradient, 0.0)

    def test_eff_stress_grad_type(self):
        self.assertIsInstance(self.p.eff_stress_gradient, float)

    def test_void_ratio_0_ref_frozen_value(self):
        self.assertEqual(self.p.void_ratio_0_ref_frozen, 0.0)

    def test_void_ratio_0_ref_frozen_type(self):
        self.assertIsInstance(self.p.void_ratio_0_ref_frozen, float)

    def test_tot_stress_0_ref_frozen_value(self):
        self.assertEqual(self.p.tot_stress_0_ref_frozen, 0.0)

    def test_tot_stress_0_ref_frozen_type(self):
        self.assertIsInstance(self.p.tot_stress_0_ref_frozen, float)

    def test_tot_stress_value(self):
        self.assertEqual(self.p.tot_stress_0_ref_frozen, 0.0)

    def test_tot_stress_type(self):
        self.assertIsInstance(self.p.tot_stress_0_ref_frozen, float)

    def test_tot_stress_grad_value(self):
        self.assertEqual(self.p.tot_stress_gradient, 0.0)

    def test_tot_stress_grad_type(self):
        self.assertIsInstance(self.p.tot_stress_gradient, float)


class TestIntegrationPoint1DInitializers(unittest.TestCase):
    def setUp(self):
        self.m = Material(
            thrm_cond_solids=7.8,
            spec_grav_solids=2.5,
            spec_heat_cap_solids=7.41e5,
        )
        self.p = IntegrationPoint1D(
            coord=1.0,
            local_coord=-0.33,
            weight=1.0,
            void_ratio=0.5,
            void_ratio_0=0.3,
            deg_sat_water=0.2,
            deg_sat_water_temp_gradient=12.5,
            material=self.m,
            temp=0.5,
            temp_rate=0.25,
            temp_gradient=0.2,
            hyd_cond=0.1,
            hyd_cond_gradient=0.4,
            water_flux_rate=0.1,
            pre_consol_stress=120.0,
            eff_stress=100.0,
            eff_stress_gradient=-0.8,
            void_ratio_0_ref_frozen=0.1,
            tot_stress_0_ref_frozen=10.0,
            tot_stress=150.0,
            tot_stress_gradient=-0.5,
        )

    def test_z_value(self):
        self.assertEqual(self.p.z, 1.0)

    def test_z_type(self):
        self.assertIsInstance(self.p.z, float)

    def test_coords_value(self):
        self.assertTrue(np.array_equal(self.p.coords, np.ones((1,))))

    def test_coords_type(self):
        self.assertIsInstance(self.p.coords, np.ndarray)

    def test_coords_shape(self):
        self.assertEqual(self.p.coords.shape, (1,))

    def test_local_coord_value(self):
        self.assertEqual(self.p.local_coord, -0.33)

    def test_local_coord_type(self):
        self.assertIsInstance(self.p.local_coord, float)

    def test_weight_value(self):
        self.assertEqual(self.p.weight, 1.0)

    def test_weight_type(self):
        self.assertIsInstance(self.p.weight, float)

    def test_void_ratio_value(self):
        self.assertEqual(self.p.void_ratio, 0.5)
        expected_porosity = 0.5 / 1.5
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_void_ratio_type(self):
        self.assertIsInstance(self.p.void_ratio, float)
        self.assertIsInstance(self.p.porosity, float)

    def test_deg_sat_water_value(self):
        self.assertEqual(self.p.deg_sat_water, 0.2)
        expected_deg_sat_ice = 0.8
        self.assertAlmostEqual(self.p.deg_sat_ice, expected_deg_sat_ice)
        expected_vol_water_cont = 0.5 * 0.2 / 1.5
        self.assertAlmostEqual(self.p.vol_water_cont, expected_vol_water_cont)

    def test_deg_sat_water_type(self):
        self.assertIsInstance(self.p.deg_sat_water, float)
        self.assertIsInstance(self.p.deg_sat_ice, float)
        self.assertIsInstance(self.p.vol_water_cont, float)

    def test_deg_sat_water_temp_gradient_value(self):
        self.assertEqual(self.p.deg_sat_water_temp_gradient, 12.5)

    def test_deg_sat_water_temp_gradient_type(self):
        self.assertIsInstance(self.p.deg_sat_water_temp_gradient, float)

    def test_material_value(self):
        self.assertIs(self.p.material, self.m)

    def test_material_type(self):
        self.assertIsInstance(self.p.material, Material)

    def test_thrm_cond(self):
        expected = 4.682284029228440
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_vol_heat_cap(self):
        expected = 1235781866.66667
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)

    def test_void_ratio_0_value(self):
        self.assertEqual(self.p.void_ratio_0, 0.3)

    def test_void_ratio_0_type(self):
        self.assertIsInstance(self.p.void_ratio, float)

    def test_temp_value(self):
        self.assertEqual(self.p.temp, 0.5)

    def test_temp_type(self):
        self.assertIsInstance(self.p.temp, float)

    def test_temp_rate_value(self):
        self.assertEqual(self.p.temp_rate, 0.25)

    def test_temp_rate_type(self):
        self.assertIsInstance(self.p.temp_rate, float)

    def test_temp_gradient_value(self):
        self.assertEqual(self.p.temp_gradient, 0.2)

    def test_temp_gradient_type(self):
        self.assertIsInstance(self.p.temp_gradient, float)

    def test_hyd_cond_value(self):
        self.assertEqual(self.p.hyd_cond, 0.1)

    def test_hyd_cond_type(self):
        self.assertIsInstance(self.p.hyd_cond, float)

    def test_hyd_cond_gradient_value(self):
        self.assertEqual(self.p.hyd_cond_gradient, 0.4)

    def test_hyd_cond_gradient_type(self):
        self.assertIsInstance(self.p.hyd_cond_gradient, float)

    def test_water_flux_rate_value(self):
        self.assertAlmostEqual(self.p.water_flux_rate, 0.1)

    def test_water_flux_rate_type(self):
        self.assertIsInstance(self.p.water_flux_rate, float)

    def test_pre_consol_stress_value(self):
        self.assertAlmostEqual(self.p.pre_consol_stress, 120.0)

    def test_pre_consol_stress_type(self):
        self.assertIsInstance(self.p.pre_consol_stress, float)

    def test_eff_stress_value(self):
        self.assertAlmostEqual(self.p.eff_stress, 100.0)

    def test_eff_stress_type(self):
        self.assertIsInstance(self.p.eff_stress, float)

    def test_eff_stress_grad_value(self):
        self.assertAlmostEqual(self.p.eff_stress_gradient, -0.8)

    def test_eff_stress_grad_type(self):
        self.assertIsInstance(self.p.eff_stress_gradient, float)

    def test_void_ratio_0_ref_frozen_value(self):
        self.assertAlmostEqual(self.p.void_ratio_0_ref_frozen, 0.1)

    def test_void_ratio_0_ref_frozen_type(self):
        self.assertIsInstance(self.p.void_ratio_0_ref_frozen, float)

    def test_tot_stress_0_ref_frozen_value(self):
        self.assertAlmostEqual(self.p.tot_stress_0_ref_frozen, 10.0)

    def test_tot_stress_0_ref_frozen_type(self):
        self.assertIsInstance(self.p.tot_stress_0_ref_frozen, float)

    def test_tot_stress_value(self):
        self.assertAlmostEqual(self.p.tot_stress, 150.0)

    def test_tot_stress_type(self):
        self.assertIsInstance(self.p.tot_stress, float)

    def test_tot_stress_grad_value(self):
        self.assertAlmostEqual(self.p.tot_stress_gradient, -0.5)

    def test_tot_stress_grad_type(self):
        self.assertIsInstance(self.p.tot_stress_gradient, float)


class TestIntegrationPoint1DSetters(unittest.TestCase):
    def setUp(self):
        self.m = Material(
            thrm_cond_solids=7.8,
            spec_grav_solids=2.5,
            spec_heat_cap_solids=7.41e5,
        )
        self.p = IntegrationPoint1D(
            coord=1.0,
            local_coord=-0.33,
            weight=1.0,
            void_ratio=0.3,
            void_ratio_0=0.4,
            deg_sat_water=0.2,
            deg_sat_water_temp_gradient=12.5,
            material=self.m,
            temp=1.5,
            temp_rate=0.5,
            temp_gradient=-0.2,
            hyd_cond=2.5,
            hyd_cond_gradient=0.7,
            water_flux_rate=0.1,
            pre_consol_stress=120.0,
            eff_stress=100.0,
            eff_stress_gradient=-0.8,
            void_ratio_0_ref_frozen=0.1,
            tot_stress_0_ref_frozen=10.0,
            tot_stress=150.0,
            tot_stress_gradient=-0.5,
        )

    def test_set_z_valid_float(self):
        self.p.z = 1.0
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int(self):
        self.p.z = 1
        self.assertEqual(self.p.z, 1.0)

    def test_set_z_valid_int_type(self):
        self.p.z = 1
        self.assertIsInstance(self.p.z, float)

    def test_set_z_valid_str(self):
        self.p.z = "1.e5"
        self.assertEqual(self.p.z, 1.0e5)

    def test_set_z_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.z = "five"

    def test_set_coords_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.coords = 1.0

    def test_set_local_coord_valid_float(self):
        self.p.local_coord = 1.0
        self.assertEqual(self.p.local_coord, 1.0)

    def test_set_local_coord_valid_int(self):
        self.p.local_coord = 1
        self.assertEqual(self.p.local_coord, 1.0)

    def test_set_local_coord_valid_int_type(self):
        self.p.local_coord = 1
        self.assertIsInstance(self.p.local_coord, float)

    def test_set_local_coord_valid_str(self):
        self.p.local_coord = "1.e0"
        self.assertEqual(self.p.local_coord, 1.0)

    def test_set_local_coord_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.local_coord = "five"

    def test_set_weight_valid_float(self):
        self.p.weight = 2.0
        self.assertEqual(self.p.weight, 2.0)

    def test_set_weight_valid_int(self):
        self.p.weight = 2
        self.assertEqual(self.p.weight, 2.0)

    def test_set_weight_valid_int_type(self):
        self.p.weight = 2
        self.assertIsInstance(self.p.weight, float)

    def test_set_weight_valid_str(self):
        self.p.weight = "1.e0"
        self.assertEqual(self.p.weight, 1.0)

    def test_set_weight_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.weight = "five"

    def test_set_void_ratio_valid_float(self):
        self.p.void_ratio = 0.5
        self.assertAlmostEqual(self.p.void_ratio, 0.5)
        expected_porosity = 0.5 / 1.5
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_set_void_ratio_valid_float_type(self):
        self.p.void_ratio = 0.5
        self.assertIsInstance(self.p.void_ratio, float)
        self.assertIsInstance(self.p.porosity, float)

    def test_set_void_ratio_valid_float_edge_0(self):
        self.p.void_ratio = 0.0
        self.assertEqual(self.p.void_ratio, 0.0)
        expected_porosity = 0.0 / 1.0
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_set_porosity_valid_float_edge_1(self):
        self.p.void_ratio = 1.0
        self.assertEqual(self.p.void_ratio, 1.0)
        expected_porosity = 1.0 / (1 + 1)
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_set_void_ratio_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio = -0.2

    def test_set_void_ratio_valid_int(self):
        self.p.void_ratio = 1
        self.assertEqual(self.p.void_ratio, 1.0)

    def test_set_void_ratio_valid_int_type(self):
        self.p.void_ratio = 1
        self.assertIsInstance(self.p.void_ratio, float)

    def test_set_void_ratio_valid_str(self):
        self.p.void_ratio = "1.e-1"
        self.assertEqual(self.p.void_ratio, 1.0e-1)

    def test_set_void_ratio_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio = "five"

    def test_set_deg_sat_water_valid_float(self):
        self.p.deg_sat_water = 0.2
        self.assertEqual(self.p.deg_sat_water, 0.2)
        ee = self.p.void_ratio
        expected_vol_water_cont = (ee / (1 + ee)) * 0.2
        self.assertAlmostEqual(self.p.vol_water_cont, expected_vol_water_cont)

    def test_set_deg_sat_water_valid_float_edge_0(self):
        self.p.deg_sat_water = 0.0
        self.assertEqual(self.p.deg_sat_water, 0.0)
        expected_vol_water_cont = 0.0
        self.assertAlmostEqual(self.p.vol_water_cont, expected_vol_water_cont)

    def test_set_deg_sat_water_valid_float_edge_1(self):
        self.p.deg_sat_water = 1.0
        self.assertEqual(self.p.deg_sat_water, 1.0)
        ee = self.p.void_ratio
        expected_vol_water_cont = (ee / (1 + ee)) * 1.0
        self.assertAlmostEqual(self.p.vol_water_cont, expected_vol_water_cont)

    def test_set_deg_sat_water_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = -0.2

    def test_set_deg_sat_water_invalid_float_positive(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = 1.1

    def test_set_deg_sat_water_valid_int(self):
        self.p.deg_sat_water = 0
        self.assertEqual(self.p.deg_sat_water, 0.0)
        ee = self.p.void_ratio
        expected_vol_water_cont = (ee / (1 + ee)) * 0.0
        self.assertAlmostEqual(self.p.vol_water_cont, expected_vol_water_cont)

    def test_set_deg_sat_water_valid_int_type(self):
        self.p.deg_sat_water = 0
        self.assertIsInstance(self.p.deg_sat_water, float)

    def test_set_deg_sat_water_valid_str(self):
        self.p.deg_sat_water = "1.e-1"
        self.assertEqual(self.p.deg_sat_water, 1.0e-1)
        ee = self.p.void_ratio
        expected_vol_water_cont = (ee / (1 + ee)) * 0.1
        self.assertAlmostEqual(self.p.vol_water_cont, expected_vol_water_cont)

    def test_set_deg_sat_water_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = "five"

    def test_set_deg_sat_water_temp_gradient_valid_float(self):
        self.p.deg_sat_water_temp_gradient = 0.2
        self.assertEqual(self.p.deg_sat_water_temp_gradient, 0.2)

    def test_set_deg_sat_water_temp_gradient_valid_float_edge_0(self):
        self.p.deg_sat_water_temp_gradient = 0.0
        self.assertEqual(self.p.deg_sat_water_temp_gradient, 0.0)

    def test_set_deg_sat_water_temp_gradient_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water_temp_gradient = -0.2

    def test_set_deg_sat_water_temp_gradient_valid_int(self):
        self.p.deg_sat_water_temp_gradient = 0
        self.assertEqual(self.p.deg_sat_water_temp_gradient, 0.0)

    def test_set_deg_sat_water_temp_gradient_valid_int_type(self):
        self.p.deg_sat_water_temp_gradient = 0
        self.assertIsInstance(self.p.deg_sat_water_temp_gradient, float)

    def test_set_deg_sat_water_temp_gradient_valid_str(self):
        self.p.deg_sat_water_temp_gradient = "1.e-1"
        self.assertEqual(self.p.deg_sat_water_temp_gradient, 1.0e-1)

    def test_set_deg_sat_water_temp_gradient_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water_temp_gradient = "five"

    def test_set_material_valid(self):
        m = Material()
        self.p.material = m
        self.assertIs(self.p.material, m)

    def test_set_material_invalid(self):
        with self.assertRaises(TypeError):
            self.p.material = 1

    def test_set_thrm_cond_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.thrm_cond = 1.0e5

    def test_update_thrm_cond_void_ratio(self):
        self.p.void_ratio = 0.25
        expected = 5.74265192951243
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_update_thrm_cond_deg_sat_water(self):
        self.p.deg_sat_water = 0.05
        expected = 5.744855338606900
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_update_thrm_cond_material(self):
        self.p.material = Material(
            thrm_cond_solids=6.7,
            spec_grav_solids=2.8,
            spec_heat_cap_solids=6.43e5,
        )
        expected = 4.873817313136410
        self.assertAlmostEqual(self.p.thrm_cond, expected)

    def test_set_vol_heat_cap_invalid(self):
        with self.assertRaises(AttributeError):
            self.p.vol_heat_cap = 1.0e5

    def test_update_vol_heat_cap_void_ratio(self):
        self.p.void_ratio = 0.25
        expected = 1482469120.0000
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)

    def test_update_vol_heat_cap_deg_sat_water(self):
        self.p.deg_sat_water = 0.05
        expected = 1425460880.769230
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)

    def test_update_vol_heat_cap_material(self):
        self.p.material = Material(
            thrm_cond_solids=6.7,
            spec_grav_solids=2.8,
            spec_heat_cap_solids=6.43e5,
        )
        expected = 1385464369.230770
        self.assertAlmostEqual(self.p.vol_heat_cap, expected, places=4)

    def test_set_void_ratio_0_valid_float(self):
        self.p.void_ratio_0 = 0.3
        self.assertAlmostEqual(self.p.void_ratio_0, 0.3)

    def test_set_void_ratio_0_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio_0 = -0.2

    def test_set_void_ratio_0_valid_int(self):
        self.p.void_ratio_0 = 1
        self.assertEqual(self.p.void_ratio_0, 1.0)

    def test_set_void_ratio_0_valid_int_type(self):
        self.p.void_ratio_0 = 1
        self.assertIsInstance(self.p.void_ratio_0, float)

    def test_set_void_ratio_0_valid_str(self):
        self.p.void_ratio_0 = "1.e-1"
        self.assertEqual(self.p.void_ratio_0, 1.0e-1)

    def test_set_void_ratio_0_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio_0 = "five"

    def test_set_temp_valid_float(self):
        self.p.temp = 0.5
        self.assertAlmostEqual(self.p.temp, 0.5)

    def test_set_temp_valid_int(self):
        self.p.temp = 1
        self.assertEqual(self.p.temp, 1.0)

    def test_set_temp_valid_int_type(self):
        self.p.temp = 1
        self.assertIsInstance(self.p.temp, float)

    def test_set_temp_valid_str(self):
        self.p.temp = "1.e-1"
        self.assertEqual(self.p.temp, 1.0e-1)

    def test_set_temp_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.temp = "five"

    def test_set_temp_rate_valid_float(self):
        self.p.temp_rate = 0.2
        self.assertAlmostEqual(self.p.temp_rate, 0.2)

    def test_set_temp_rate_valid_int(self):
        self.p.temp_rate = 1
        self.assertEqual(self.p.temp_rate, 1.0)

    def test_set_temp_rate_valid_int_type(self):
        self.p.temp_rate = 1
        self.assertIsInstance(self.p.temp_rate, float)

    def test_set_temp_rate_valid_str(self):
        self.p.temp_rate = "1.e-1"
        self.assertEqual(self.p.temp_rate, 1.0e-1)

    def test_set_temp_rate_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.temp_rate = "five"

    def test_set_temp_gradient_valid_float(self):
        self.p.temp_gradient = 0.5
        self.assertAlmostEqual(self.p.temp_gradient, 0.5)

    def test_set_temp_gradient_valid_int(self):
        self.p.temp_gradient = 1
        self.assertEqual(self.p.temp_gradient, 1.0)

    def test_set_temp_gradient_valid_int_type(self):
        self.p.temp_gradient = 1
        self.assertIsInstance(self.p.temp_gradient, float)

    def test_set_temp_gradient_valid_str(self):
        self.p.temp_gradient = "1.e-1"
        self.assertEqual(self.p.temp_gradient, 1.0e-1)

    def test_set_temp_gradient_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.temp_gradient = "five"

    def test_set_hyd_cond_valid_float(self):
        self.p.hyd_cond = 0.5
        self.assertAlmostEqual(self.p.hyd_cond, 0.5)

    def test_set_hyd_cond_invalid_float(self):
        with self.assertRaises(ValueError):
            self.p.hyd_cond = -0.5

    def test_set_hyd_cond_valid_int(self):
        self.p.hyd_cond = 1
        self.assertEqual(self.p.hyd_cond, 1.0)

    def test_set_hyd_cond_valid_int_type(self):
        self.p.hyd_cond = 1
        self.assertIsInstance(self.p.hyd_cond, float)

    def test_set_hyd_cond_valid_str(self):
        self.p.hyd_cond = "1.e-1"
        self.assertEqual(self.p.hyd_cond, 1.0e-1)

    def test_set_hyd_cond_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.hyd_cond = "five"

    def test_set_hyd_cond_gradient_valid_float(self):
        self.p.hyd_cond_gradient = 0.5
        self.assertAlmostEqual(self.p.hyd_cond_gradient, 0.5)

    def test_set_hyd_cond_gradient_invalid_float(self):
        with self.assertRaises(ValueError):
            self.p.hyd_cond_gradient = -0.5

    def test_set_hyd_cond_gradient_valid_int(self):
        self.p.hyd_cond_gradient = 1
        self.assertEqual(self.p.hyd_cond_gradient, 1.0)

    def test_set_hyd_cond_gradient_valid_int_type(self):
        self.p.hyd_cond_gradient = 1
        self.assertIsInstance(self.p.hyd_cond_gradient, float)

    def test_set_hyd_cond_gradient_valid_str(self):
        self.p.hyd_cond_gradient = "1.e-1"
        self.assertEqual(self.p.hyd_cond_gradient, 1.0e-1)

    def test_set_hyd_cond_gradient_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.hyd_cond_gradient = "five"

    def test_set_water_flux_rate_valid_float(self):
        self.p.water_flux_rate = 1.0
        self.assertEqual(self.p.water_flux_rate, 1.0)

    def test_set_water_flux_rate_valid_int(self):
        self.p.water_flux_rate = 1
        self.assertEqual(self.p.water_flux_rate, 1.0)

    def test_set_water_flux_rate_valid_int_type(self):
        self.p.water_flux_rate = 1
        self.assertIsInstance(self.p.water_flux_rate, float)

    def test_set_water_flux_rate_valid_str(self):
        self.p.water_flux_rate = "1.e5"
        self.assertEqual(self.p.water_flux_rate, 1.0e5)

    def test_set_water_flux_rate_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.water_flux_rate = "five"

    def test_set_pre_consol_stress_valid_float(self):
        self.p.pre_consol_stress = 1.0
        self.assertEqual(self.p.pre_consol_stress, 1.0)

    def test_set_pre_consol_stress_valid_int(self):
        self.p.pre_consol_stress = 1
        self.assertEqual(self.p.pre_consol_stress, 1.0)

    def test_set_pre_consol_stress_valid_int_type(self):
        self.p.pre_consol_stress = 1
        self.assertIsInstance(self.p.pre_consol_stress, float)

    def test_set_pre_consol_stress_valid_str(self):
        self.p.pre_consol_stress = "1.e5"
        self.assertEqual(self.p.pre_consol_stress, 1.0e5)

    def test_set_pre_consol_stress_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.pre_consol_stress = "five"

    def test_set_eff_stress_valid_float(self):
        self.p.eff_stress = 1.0
        self.assertEqual(self.p.eff_stress, 1.0)

    def test_set_eff_stress_valid_int(self):
        self.p.eff_stress = 1
        self.assertEqual(self.p.eff_stress, 1.0)

    def test_set_eff_stress_valid_int_type(self):
        self.p.eff_stress = 1
        self.assertIsInstance(self.p.eff_stress, float)

    def test_set_eff_stress_valid_str(self):
        self.p.eff_stress = "1.e5"
        self.assertEqual(self.p.eff_stress, 1.0e5)

    def test_set_eff_stress_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.eff_stress = "five"

    def test_set_eff_stress_gradient_valid_float(self):
        self.p.eff_stress_gradient = -1.0
        self.assertEqual(self.p.eff_stress_gradient, -1.0)

    def test_set_eff_stress_gradient_invalid_float(self):
        with self.assertRaises(ValueError):
            self.p.eff_stress_gradient = 1.0

    def test_set_eff_stress_gradient_valid_int(self):
        self.p.eff_stress_gradient = -1
        self.assertEqual(self.p.eff_stress_gradient, -1.0)

    def test_set_eff_stress_gradient_valid_int_type(self):
        self.p.eff_stress_gradient = -1
        self.assertIsInstance(self.p.eff_stress_gradient, float)

    def test_set_eff_stress_gradient_valid_str(self):
        self.p.eff_stress_gradient = "-1.e5"
        self.assertEqual(self.p.eff_stress_gradient, -1.0e5)

    def test_set_eff_stress_gradient_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.eff_stress_gradient = "five"

    def test_set_void_ratio_0_ref_frozen_valid_float(self):
        self.p.void_ratio_0_ref_frozen = 1.0
        self.assertEqual(self.p.void_ratio_0_ref_frozen, 1.0)

    def test_set_void_ratio_0_ref_frozen_invalid_float(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio_0_ref_frozen = -1.0

    def test_set_void_ratio_0_ref_frozen_valid_int(self):
        self.p.void_ratio_0_ref_frozen = 1
        self.assertEqual(self.p.void_ratio_0_ref_frozen, 1.0)

    def test_set_void_ratio_0_ref_frozen_valid_int_type(self):
        self.p.void_ratio_0_ref_frozen = 1
        self.assertIsInstance(self.p.void_ratio_0_ref_frozen, float)

    def test_set_void_ratio_0_ref_frozen_valid_str(self):
        self.p.void_ratio_0_ref_frozen = "2.e-1"
        self.assertEqual(self.p.void_ratio_0_ref_frozen, 0.2)

    def test_set_void_ratio_0_ref_frozen_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.void_ratio_0_ref_frozen = "five"

    def test_set_tot_stress_0_ref_frozen_valid_float(self):
        self.p.tot_stress_0_ref_frozen = 1.0
        self.assertEqual(self.p.tot_stress_0_ref_frozen, 1.0)

    def test_set_tot_stress_0_ref_frozen_valid_int(self):
        self.p.tot_stress_0_ref_frozen = 1
        self.assertEqual(self.p.tot_stress_0_ref_frozen, 1.0)

    def test_set_tot_stress_0_ref_frozen_valid_int_type(self):
        self.p.tot_stress_0_ref_frozen = 1
        self.assertIsInstance(self.p.tot_stress_0_ref_frozen, float)

    def test_set_tot_stress_0_ref_frozen_valid_str(self):
        self.p.tot_stress_0_ref_frozen = "2.e-1"
        self.assertEqual(self.p.tot_stress_0_ref_frozen, 0.2)

    def test_set_tot_stress_0_ref_frozen_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.tot_stress_0_ref_frozen = "five"

    def test_set_tot_stress_valid_float(self):
        self.p.tot_stress = 1.0
        self.assertEqual(self.p.tot_stress, 1.0)

    def test_set_tot_stress_valid_int(self):
        self.p.tot_stress = 1
        self.assertEqual(self.p.tot_stress, 1.0)

    def test_set_tot_stress_valid_int_type(self):
        self.p.tot_stress = 1
        self.assertIsInstance(self.p.tot_stress, float)

    def test_set_tot_stress_valid_str(self):
        self.p.tot_stress = "1.e5"
        self.assertEqual(self.p.tot_stress, 1.0e5)

    def test_set_tot_stress_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.tot_stress = "five"

    def test_set_tot_stress_gradient_valid_float(self):
        self.p.tot_stress_gradient = -1.0
        self.assertEqual(self.p.tot_stress_gradient, -1.0)

    def test_set_tot_stress_gradient_invalid_float(self):
        with self.assertRaises(ValueError):
            self.p.tot_stress_gradient = 1.0

    def test_set_tot_stress_gradient_valid_int(self):
        self.p.tot_stress_gradient = -1
        self.assertEqual(self.p.tot_stress_gradient, -1.0)

    def test_set_tot_stress_gradient_valid_int_type(self):
        self.p.tot_stress_gradient = -1
        self.assertIsInstance(self.p.tot_stress_gradient, float)

    def test_set_tot_stress_gradient_valid_str(self):
        self.p.tot_stress_gradient = "-1.e5"
        self.assertEqual(self.p.tot_stress_gradient, -1.0e5)

    def test_set_tot_stress_gradient_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.tot_stress_gradient = "five"


if __name__ == "__main__":
    unittest.main()
