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
        self.assertAlmostEqual(self.p.vol_ice_cont, 0.0)

    def test_vol_ice_cont_type(self):
        self.assertIsInstance(self.p.deg_sat_water, float)
        self.assertIsInstance(self.p.deg_sat_ice, float)
        self.assertIsInstance(self.p.vol_ice_cont, float)

    def test_material_value(self):
        self.assertIs(self.p.material, NULL_MATERIAL)

    def test_material_type(self):
        self.assertIsInstance(self.p.material, Material)


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
            deg_sat_water=0.2,
            material=self.m,
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
        expected_vol_ice_cont = 0.5 * 0.8 / 1.5
        self.assertAlmostEqual(self.p.vol_ice_cont, expected_vol_ice_cont)

    def test_vol_ice_cont_type(self):
        self.assertIsInstance(self.p.deg_sat_water, float)
        self.assertIsInstance(self.p.deg_sat_ice, float)
        self.assertIsInstance(self.p.vol_ice_cont, float)

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
            deg_sat_water=0.2,
            material=self.m,
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

    def test_set_ratio_valid_float(self):
        self.p.void_ratio = 0.5
        self.assertAlmostEqual(self.p.void_ratio, 0.5)
        expected_porosity = 0.5 / 1.5
        self.assertAlmostEqual(self.p.porosity, expected_porosity)

    def test_set_void_ratio_valid_float(self):
        self.p.void_ratio = 0.5
        self.assertEqual(self.p.void_ratio, 0.5)
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

    def test_set_deg_sat_water_valid_float_edge_0(self):
        self.p.deg_sat_water = 0.0
        self.assertEqual(self.p.deg_sat_water, 0.0)

    def test_set_deg_sat_water_valid_float_edge_1(self):
        self.p.deg_sat_water = 1.0
        self.assertEqual(self.p.deg_sat_water, 1.0)

    def test_set_deg_sat_water_invalid_float_negative(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = -0.2

    def test_set_deg_sat_water_invalid_float_positive(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = 1.1

    def test_set_deg_sat_water_valid_int(self):
        self.p.deg_sat_water = 0
        self.assertEqual(self.p.deg_sat_water, 0.0)

    def test_set_deg_sat_water_valid_int_type(self):
        self.p.deg_sat_water = 0
        self.assertIsInstance(self.p.deg_sat_water, float)

    def test_set_deg_sat_water_valid_str(self):
        self.p.deg_sat_water = "1.e-1"
        self.assertEqual(self.p.deg_sat_water, 1.0e-1)

    def test_set_deg_sat_water_invalid_str(self):
        with self.assertRaises(ValueError):
            self.p.deg_sat_water = "five"

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


if __name__ == "__main__":
    unittest.main()
