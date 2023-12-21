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
        self.assertEqual(vol_heat_cap_water, 4204000.0)

    def test_vol_heat_cap_ice(self):
        self.assertEqual(vol_heat_cap_ice, 1881000.0)

    def test_thrm_cond_water(self):
        self.assertEqual(thrm_cond_water, 5.63e-1)

    def test_thrm_cond_ice(self):
        self.assertEqual(thrm_cond_ice, 2.22e0)

# TODO: Default tests
# deg_sat_water()
# hyd_cond()
# water_flux()
# eff_stress()
# comp_index_frozen()
# tot_stress()


class TestMaterialDefaults(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_thrm_cond_solids(self):
        self.assertEqual(self.m.thrm_cond_solids, 0.0)

    def test_dens_solids(self):
        self.assertEqual(self.m.dens_solids, 0.0)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(self.m.spec_heat_cap_solids, 0.0)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(self.m.vol_heat_cap_solids, 0.0)

    def test_deg_sat_water_alpha(self):
        self.assertEqual(self.m.deg_sat_water_alpha, 0.0)

    def test_deg_sat_water_beta(self):
        self.assertEqual(self.m.deg_sat_water_beta, 0.0)

    def test_hyd_cond_index(self):
        self.assertEqual(self.m.hyd_cond_index, 0.0)

    def test_hyd_cond_mult(self):
        self.assertEqual(self.m.hyd_cond_mult, 1.0)

    def test_hyd_cond_0(self):
        self.assertEqual(self.m.hyd_cond_0, 0.0)

    def test_void_ratio_0_hyd_cond(self):
        self.assertEqual(self.m.void_ratio_0_hyd_cond, 0.0)

    def test_void_ratio_min(self):
        self.assertEqual(self.m.void_ratio_min, 0.0)

    def test_void_ratio_sep(self):
        self.assertEqual(self.m.void_ratio_sep, 0.0)

    def test_void_ratio_lim(self):
        self.assertEqual(self.m.void_ratio_lim, 0.0)

    def test_void_ratio_tr(self):
        self.assertEqual(self.m.void_ratio_tr, 0.0)

    def test_water_flux_b1(self):
        self.assertEqual(self.m.water_flux_b1, 0.0)

    def test_water_flux_b2(self):
        self.assertEqual(self.m.water_flux_b2, 0.0)

    def test_water_flux_b3(self):
        self.assertEqual(self.m.water_flux_b3, 0.0)

    def test_temp_rate_ref(self):
        self.assertEqual(self.m.temp_rate_ref, 0.0)

    def test_seg_pot_0(self):
        self.assertEqual(self.m.seg_pot_0, 0.0)

    def test_void_ratio_0_comp(self):
        self.assertEqual(self.m.void_ratio_0_comp, 0.0)

    def test_eff_stress_0_comp(self):
        self.assertEqual(self.m.eff_stress_0_comp, 0.0)

    def test_comp_index_unfrozen(self):
        self.assertEqual(self.m.comp_index_unfrozen, 0.0)

    def test_rebound_index_unfozen(self):
        self.assertEqual(self.m.comp_index_unfrozen, 0.0)

    def test_comp_index_frozen_a1(self):
        self.assertEqual(self.m.comp_index_frozen_a1, 0.0)

    def test_comp_index_frozen_a2(self):
        self.assertEqual(self.m.comp_index_frozen_a2, 0.0)

    def test_comp_index_frozen_a3(self):
        self.assertEqual(self.m.comp_index_frozen_a3, 0.0)


# TODO: Null Material tests
# deg_sat_water()
# hyd_cond()
# water_flux()
# eff_stress()
# comp_index_frozen()
# tot_stress()


class TestNullMaterial(unittest.TestCase):
    def test_thrm_cond_solids(self):
        self.assertEqual(NULL_MATERIAL.thrm_cond_solids, 0.0)

    def test_dens_solids(self):
        self.assertEqual(NULL_MATERIAL.dens_solids, 0.0)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(NULL_MATERIAL.spec_heat_cap_solids, 0.0)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(NULL_MATERIAL.vol_heat_cap_solids, 0.0)

    def test_deg_sat_water_alpha(self):
        self.assertEqual(NULL_MATERIAL.deg_sat_water_alpha, 0.0)

    def test_deg_sat_water_beta(self):
        self.assertEqual(NULL_MATERIAL.deg_sat_water_beta, 0.0)

    def test_hyd_cond_index(self):
        self.assertEqual(NULL_MATERIAL.hyd_cond_index, 0.0)

    def test_hyd_cond_mult(self):
        self.assertEqual(NULL_MATERIAL.hyd_cond_mult, 1.0)

    def test_hyd_cond_0(self):
        self.assertEqual(NULL_MATERIAL.hyd_cond_0, 0.0)

    def test_void_ratio_0_hyd_cond(self):
        self.assertEqual(NULL_MATERIAL.void_ratio_0_hyd_cond, 0.0)

    def test_void_ratio_min(self):
        self.assertEqual(NULL_MATERIAL.void_ratio_min, 0.0)

    def test_void_ratio_sep(self):
        self.assertEqual(NULL_MATERIAL.void_ratio_sep, 0.0)

    def test_void_ratio_lim(self):
        self.assertEqual(NULL_MATERIAL.void_ratio_lim, 0.0)

    def test_void_ratio_tr(self):
        self.assertEqual(NULL_MATERIAL.void_ratio_tr, 0.0)

    def test_water_flux_b1(self):
        self.assertEqual(NULL_MATERIAL.water_flux_b1, 0.0)

    def test_water_flux_b2(self):
        self.assertEqual(NULL_MATERIAL.water_flux_b2, 0.0)

    def test_water_flux_b3(self):
        self.assertEqual(NULL_MATERIAL.water_flux_b3, 0.0)

    def test_temp_rate_ref(self):
        self.assertEqual(NULL_MATERIAL.temp_rate_ref, 0.0)

    def test_seg_pot_0(self):
        self.assertEqual(NULL_MATERIAL.seg_pot_0, 0.0)

    def test_void_ratio_0_comp(self):
        self.assertEqual(NULL_MATERIAL.void_ratio_0_comp, 0.0)

    def test_eff_stress_0_comp(self):
        self.assertEqual(NULL_MATERIAL.eff_stress_0_comp, 0.0)

    def test_comp_index_unfrozen(self):
        self.assertEqual(NULL_MATERIAL.comp_index_unfrozen, 0.0)

    def test_rebound_index_unfozen(self):
        self.assertEqual(NULL_MATERIAL.comp_index_unfrozen, 0.0)

    def test_comp_index_frozen_a1(self):
        self.assertEqual(NULL_MATERIAL.comp_index_frozen_a1, 0.0)

    def test_comp_index_frozen_a2(self):
        self.assertEqual(NULL_MATERIAL.comp_index_frozen_a2, 0.0)

    def test_comp_index_frozen_a3(self):
        self.assertEqual(NULL_MATERIAL.comp_index_frozen_a3, 0.0)


# TODO: Initializer tests
# deg_sat_water()
# hyd_cond()
# water_flux()
# eff_stress()
# comp_index_frozen()
# tot_stress()


class TestMaterialInitializers(unittest.TestCase):
    def setUp(self):
        self.m = Material(
            thrm_cond_solids=7.8,
            spec_grav_solids=2.5,
            spec_heat_cap_solids=7.41e5,
            # deg_sat_water_alpha=,
            # deg_sat_water_beta=,
            # hyd_cond_index=,
            # hyd_cond_mult=,
            # hyd_cond_0=,
            # void_ratio_0_hyd_cond=,
            # void_ratio_min=,
            # void_ratio_sep=,
            # void_ratio_lim=,
            # void_ratio_tr=,
            # water_flux_b1=,
            # water_flux_b2=,
            # water_flux_b3=,
            # temp_rate_ref=,
            # seg_pot_0=,
            # void_ratio_0_comp=,
            # eff_stress_0_comp=,
            # comp_index_unfrozen=,
            # rebound_index_unfrozen=,
            # comp_index_frozen_a1=,
            # comp_index_frozen_a2=,
            # comp_index_frozen_a3=,
        )

    def test_thrm_cond_solids(self):
        self.assertEqual(self.m.thrm_cond_solids, 7.8)

    def test_dens_solids(self):
        self.assertEqual(self.m.dens_solids, 2.5e3)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(self.m.spec_heat_cap_solids, 7.41e5)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(self.m.vol_heat_cap_solids, 1.8525e9)

    # def test_deg_sat_water_alpha(self):
    #     self.assertEqual(self.m.deg_sat_water_alpha, 12.0)
    #
    # def test_deg_sat_water_beta(self):
    #     self.assertEqual(self.m.deg_sat_water_beta, 0.35)
    #
    # def test_hyd_cond_index(self):
    #     self.assertEqual(self.m.hyd_cond_index, 0.305)
    #
    # def test_hyd_cond_mult(self):
    #     self.assertEqual(self.m.hyd_cond_mult, 0.5)
    #
    # def test_hyd_cond_0(self):
    #     self.assertEqual(self.m.hyd_cond_0, 4.05e-4)
    #
    # def test_void_ratio_0_hyd_cond(self):
    #     self.assertEqual(self.m.void_ratio_0_hyd_cond, 2.6)
    #
    # def test_void_ratio_min(self):
    #     self.assertEqual(self.m.void_ratio_min, 0.3)
    #
    # def test_void_ratio_sep(self):
    #     self.assertEqual(self.m.void_ratio_sep, 1.6)
    #
    # def test_void_ratio_lim(self):
    #     self.assertEqual(self.m.void_ratio_lim, 0.28)
    #
    # def test_void_ratio_tr(self):
    #     self.assertEqual(self.m.void_ratio_tr, 0.5)
    #
    # def test_water_flux_b1(self):
    #     self.assertEqual(self.m.water_flux_b1, 0.08)
    #
    # def test_water_flux_b2(self):
    #     self.assertEqual(self.m.water_flux_b2, 4.0)
    #
    # def test_water_flux_b3(self):
    #     self.assertEqual(self.m.water_flux_b3, 10.0)
    #
    # def test_temp_rate_ref(self):
    #     self.assertEqual(self.m.temp_rate_ref, 10.0e-9)
    #
    # def test_seg_pot_0(self):
    #     self.assertEqual(self.m.seg_pot_0, 2.0e-9)
    #
    # def test_void_ratio_0_comp(self):
    #     self.assertEqual(self.m.void_ratio_0_comp, 2.6)
    #
    # def test_eff_stress_0_comp(self):
    #     self.assertEqual(self.m.eff_stress_0_comp, 0.0028)
    #
    # def test_comp_index_unfrozen(self):
    #     self.assertEqual(self.m.comp_index_unfrozen, 0.421)
    #
    # def test_rebound_index_unfozen(self):
    #     self.assertEqual(self.m.comp_index_unfrozen, 0.08)
    #
    # def test_comp_index_frozen_a1(self):
    #     self.assertEqual(self.m.comp_index_frozen_a1, 0.021)
    #
    # def test_comp_index_frozen_a2(self):
    #     self.assertEqual(self.m.comp_index_frozen_a2, 0.01)
    #
    # def test_comp_index_frozen_a3(self):
    #     self.assertEqual(self.m.comp_index_frozen_a3, 0.23)


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
        self.m.thrm_cond_solids = "1.2e1"
        self.assertEqual(self.m.thrm_cond_solids, 12.0)

    def test_set_thrm_cond_solids_valid_str_type(self):
        self.m.thrm_cond_solids = "1.2e1"
        self.assertIsInstance(self.m.thrm_cond_solids, float)

    def test_set_thrm_cond_solids_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.thrm_cond_solids = (12.0, 1.8)

    def test_set_thrm_cond_solids_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.thrm_cond_solids = -12.0

    def test_set_thrm_cond_solids_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.thrm_cond_solids = "twelve"


class TestMaterialDensSolidsSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_spec_grav_solids_valid_float(self):
        self.m.spec_grav_solids = 1.2
        self.assertEqual(self.m.spec_grav_solids, 1.2)
        self.assertEqual(self.m.dens_solids, 1.2e3)

    def test_set_spec_grav_solids_valid_int(self):
        self.m.spec_grav_solids = 12
        self.assertEqual(self.m.spec_grav_solids, 12.0)
        self.assertEqual(self.m.dens_solids, 12e3)
        self.assertIsInstance(self.m.spec_grav_solids, float)
        self.assertIsInstance(self.m.dens_solids, float)

    def test_set_spec_grav_solids_valid_str(self):
        self.m.spec_grav_solids = "1.2e1"
        self.assertEqual(self.m.spec_grav_solids, 12.0)
        self.assertEqual(self.m.dens_solids, 12.0e3)

    def test_set_spec_grav_solids_valid_str_type(self):
        self.m.spec_grav_solids = "1.2e1"
        self.assertIsInstance(self.m.spec_grav_solids, float)
        self.assertIsInstance(self.m.dens_solids, float)

    def test_set_spec_grav_solids_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.spec_grav_solids = (12.0, 1.8)

    def test_set_spec_grav_solids_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.spec_grav_solids = -12.0

    def test_set_spec_grav_solids_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.spec_grav_solids = "twelve"


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
        self.m.spec_heat_cap_solids = "1.2e1"
        self.assertEqual(self.m.spec_heat_cap_solids, 12.0)

    def test_set_spec_heat_cap_solids_valid_str_type(self):
        self.m.spec_heat_cap_solids = "1.2e1"
        self.assertIsInstance(self.m.spec_heat_cap_solids, float)

    def test_set_spec_heat_cap_solids_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.spec_heat_cap_solids = (12.0, 1.8)

    def test_set_spec_heat_cap_solids_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.spec_heat_cap_solids = -12.0

    def test_set_spec_heat_cap_solids_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.spec_heat_cap_solids = "twelve"


class TestMaterialVolHeatCapSolidsSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material(
            thrm_cond_solids=7.8,
            spec_grav_solids=2.5,
            spec_heat_cap_solids=7.41e5
        )

    def test_set_spec_heat_cap_solids(self):
        self.m.spec_heat_cap_solids = 1.2
        expected = 1.2 * 2.5e3
        self.assertEqual(self.m.vol_heat_cap_solids, expected)

    def test_set_dens_solids(self):
        self.m.spec_grav_solids = 3.0
        expected = 7.41e5 * 3.0e3
        self.assertEqual(self.m.vol_heat_cap_solids, expected)

    def test_vol_heat_cap_solids(self):
        with self.assertRaises(AttributeError):
            self.m.vol_heat_cap_solids = 5.0


# class TestMaterialDegSatWaterAlphaSetter(unittest.TestCase):
#     def setUp(self):
#         self.m = Material()
#
#     def test_set_deg_sat_water_alpha_valid_float(self):
#         self.m.deg_sat_water_alpha = 1.2
#         self.assertEqual(self.m.deg_sat_water_alpha, 1.2)
#
#     def test_set_deg_sat_water_alpha_valid_int(self):
#         self.m.deg_sat_water_alpha = 12
#         self.assertEqual(self.m.deg_sat_water_alpha, 12.0)
#
#     def test_set_deg_sat_water_alpha_valid_int_type(self):
#         self.m.deg_sat_water_alpha = 12
#         self.assertIsInstance(self.m.deg_sat_water_alpha, float)
#
#     def test_set_deg_sat_water_alpha_valid_str(self):
#         self.m.deg_sat_water_alpha = "1.2e1"
#         self.assertEqual(self.m.deg_sat_water_alpha, 12.0)
#
#     def test_set_deg_sat_water_alpha_valid_str_type(self):
#         self.m.deg_sat_water_alpha = "1.2e1"
#         self.assertIsInstance(self.m.deg_sat_water_alpha, float)
#
#     def test_set_deg_sat_water_alpha_invalid_type(self):
#         with self.assertRaises(TypeError):
#             self.m.deg_sat_water_alpha = (12.0, 1.8)
#
#     def test_set_deg_sat_water_alpha_invalid_value(self):
#         with self.assertRaises(ValueError):
#             self.m.deg_sat_water_alpha = -12.0
#
#     def test_set_deg_sat_water_alpha_invalid_str(self):
#         with self.assertRaises(ValueError):
#             self.m.deg_sat_water_alpha = "twelve"


# TODO: Setter tests
# deg_sat_water_alpha
# deg_sat_water_beta
# hyd_cond_index
# hyd_cond_mult
# hyd_cond_0
# void_ratio_0_hyd_cond
# void_ratio_min
# void_ratio_sep
# void_ratio_lim
# void_ratio_tr
# water_flux_b1
# water_flux_b2
# water_flux_b3
# temp_rate_ref
# seg_pot_0
# void_ratio_0_comp
# eff_stress_0_comp
# comp_index_unfrozen
# rebound_index_unfrozen
# comp_index_frozen_a1
# comp_index_frozen_a2
# comp_index_frozen_a3
# deg_sat_water()
# hyd_cond()
# water_flux()
# eff_stress()
# comp_index_frozen()
# tot_stress()


if __name__ == "__main__":
    unittest.main()
