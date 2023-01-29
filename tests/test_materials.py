import unittest

from frozen_ground_fem.materials import (
    vol_heat_cap_water,
    vol_heat_cap_ice,
    thrm_cond_water,
    thrm_cond_ice,
    Material,
    NULL_MATERIAL,
    )


class TestConstants(unittest.TestCase):

    def test_vol_heat_cap_water(self):
        self.assertEqual(vol_heat_cap_water, 4204000.)

    def test_vol_heat_cap_ice(self):
        self.assertEqual(vol_heat_cap_ice, 1881000.)

    def test_thrm_cond_water(self):
        self.assertEqual(thrm_cond_water, 5.63e-1)

    def test_thrm_cond_ice(self):
        self.assertEqual(thrm_cond_ice, 2.22e0)


class TestMaterialDefaults(unittest.TestCase):

    def setUp(self):
        self.m = Material()

    def test_thrm_cond_solids(self):
        self.assertEqual(self.m.thrm_cond_solids, 0.)

    def test_dens_solids(self):
        self.assertEqual(self.m.dens_solids, 0.)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(self.m.spec_heat_cap_solids, 0.)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(self.m.vol_heat_cap_solids, 0.)


class TestNullMaterial(unittest.TestCase):

    def test_thrm_cond_solids(self):
        self.assertEqual(NULL_MATERIAL.thrm_cond_solids, 0.)

    def test_dens_solids(self):
        self.assertEqual(NULL_MATERIAL.dens_solids, 0.)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(NULL_MATERIAL.spec_heat_cap_solids, 0.)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(NULL_MATERIAL.vol_heat_cap_solids, 0.)


class TestMaterialInitializers(unittest.TestCase):

    def setUp(self):
        self.m = Material(thrm_cond_solids=7.8,
                          dens_solids=2.5e3,
                          spec_heat_cap_solids=7.41e5)

    def test_thrm_cond_solids(self):
        self.assertEqual(self.m.thrm_cond_solids, 7.8)

    def test_dens_solids(self):
        self.assertEqual(self.m.dens_solids, 2.5e3)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(self.m.spec_heat_cap_solids, 7.41e5)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(self.m.vol_heat_cap_solids, 1.8525e9)


class TestMaterialThrmCondSolidsSetter(unittest.TestCase):

    def setUp(self):
        self.m = Material()

    def test_set_thrm_cond_solids_valid_float(self):
        self.m.thrm_cond_solids = 1.2
        self.assertEqual(self.m.thrm_cond_solids, 1.2)

    def test_set_thrm_cond_solids_valid_int(self):
        self.m.thrm_cond_solids = 12
        self.assertEqual(self.m.thrm_cond_solids, 12.0)

    def test_set_thrm_cond_solids_valid_int_type(self):
        self.m.thrm_cond_solids = 12
        self.assertIsInstance(self.m.thrm_cond_solids, float)

    def test_set_thrm_cond_solids_valid_str(self):
        self.m.thrm_cond_solids = '1.2e1'
        self.assertEqual(self.m.thrm_cond_solids, 12.0)

    def test_set_thrm_cond_solids_valid_str_type(self):
        self.m.thrm_cond_solids = '1.2e1'
        self.assertIsInstance(self.m.thrm_cond_solids, float)

    def test_set_thrm_cond_solids_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.thrm_cond_solids = (12., 1.8)

    def test_set_thrm_cond_solids_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.thrm_cond_solids = -12.0

    def test_set_thrm_cond_solids_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.thrm_cond_solids = 'twelve'


class TestMaterialDensSolidsSetter(unittest.TestCase):

    def setUp(self):
        self.m = Material()

    def test_set_dens_solids_valid_float(self):
        self.m.dens_solids = 1.2
        self.assertEqual(self.m.dens_solids, 1.2)

    def test_set_dens_solids_valid_int(self):
        self.m.dens_solids = 12
        self.assertEqual(self.m.dens_solids, 12.0)

    def test_set_dens_solids_valid_int_type(self):
        self.m.dens_solids = 12
        self.assertIsInstance(self.m.dens_solids, float)

    def test_set_dens_solids_valid_str(self):
        self.m.dens_solids = '1.2e1'
        self.assertEqual(self.m.dens_solids, 12.0)

    def test_set_dens_solids_valid_str_type(self):
        self.m.dens_solids = '1.2e1'
        self.assertIsInstance(self.m.dens_solids, float)

    def test_set_dens_solids_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.dens_solids = (12., 1.8)

    def test_set_dens_solids_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.dens_solids = -12.0

    def test_set_dens_solids_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.dens_solids = 'twelve'


class TestMaterialSpecHeatCapSolidsSetter(unittest.TestCase):

    def setUp(self):
        self.m = Material()

    def test_set_spec_heat_cap_solids_valid_float(self):
        self.m.spec_heat_cap_solids = 1.2
        self.assertEqual(self.m.spec_heat_cap_solids, 1.2)

    def test_set_spec_heat_cap_solids_valid_int(self):
        self.m.spec_heat_cap_solids = 12
        self.assertEqual(self.m.spec_heat_cap_solids, 12.0)

    def test_set_spec_heat_cap_solids_valid_int_type(self):
        self.m.spec_heat_cap_solids = 12
        self.assertIsInstance(self.m.spec_heat_cap_solids, float)

    def test_set_spec_heat_cap_solids_valid_str(self):
        self.m.spec_heat_cap_solids = '1.2e1'
        self.assertEqual(self.m.spec_heat_cap_solids, 12.0)

    def test_set_spec_heat_cap_solids_valid_str_type(self):
        self.m.spec_heat_cap_solids = '1.2e1'
        self.assertIsInstance(self.m.spec_heat_cap_solids, float)

    def test_set_spec_heat_cap_solids_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.spec_heat_cap_solids = (12., 1.8)

    def test_set_spec_heat_cap_solids_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.spec_heat_cap_solids = -12.0

    def test_set_spec_heat_cap_solids_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.spec_heat_cap_solids = 'twelve'


class TestMaterialVolHeatCapSolidsSetter(unittest.TestCase):

    def setUp(self):
        self.m = Material(thrm_cond_solids=7.8,
                          dens_solids=2.5e3,
                          spec_heat_cap_solids=7.41e5)

    def test_set_spec_heat_cap_solids(self):
        self.m.spec_heat_cap_solids = 1.2
        expected = 1.2 * 2.5e3
        self.assertEqual(self.m.vol_heat_cap_solids, expected)

    def test_set_dens_solids(self):
        self.m.dens_solids = 3.e3
        expected = 7.41e5 * 3.e3
        self.assertEqual(self.m.vol_heat_cap_solids, expected)

    def test_vol_heat_cap_solids(self):
        with self.assertRaises(AttributeError):
            self.m.vol_heat_cap_solids = 5.


if __name__ == "__main__":
    unittest.main()
