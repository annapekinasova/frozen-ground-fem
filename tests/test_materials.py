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


class TestMaterialDefaults(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_thrm_cond_solids(self):
        self.assertEqual(self.m.thrm_cond_solids, 0.0)

    def test_dens_solids(self):
        self.assertEqual(self.m.dens_solids, 1.0e3)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(self.m.spec_heat_cap_solids, 0.0)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(self.m.vol_heat_cap_solids, 0.0)

    def test_deg_sat_water_alpha(self):
        self.assertEqual(self.m.deg_sat_water_alpha, 1.0)

    def test_deg_sat_water_beta(self):
        self.assertEqual(self.m.deg_sat_water_beta, 0.9)

    def test_hyd_cond_index(self):
        self.assertEqual(self.m.hyd_cond_index, 1.0)

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
        self.assertEqual(self.m.temp_rate_ref, 1.0e-9)

    def test_seg_pot_0(self):
        self.assertEqual(self.m.seg_pot_0, 0.0)

    def test_void_ratio_0_comp(self):
        self.assertEqual(self.m.void_ratio_0_comp, 0.0)

    def test_eff_stress_0_comp(self):
        self.assertEqual(self.m.eff_stress_0_comp, 0.0)

    def test_comp_index_unfrozen(self):
        self.assertEqual(self.m.comp_index_unfrozen, 1.0)

    def test_rebound_index_unfozen(self):
        self.assertEqual(self.m.comp_index_unfrozen, 1.0)

    def test_comp_index_frozen_a1(self):
        self.assertEqual(self.m.comp_index_frozen_a1, 0.0)

    def test_comp_index_frozen_a2(self):
        self.assertEqual(self.m.comp_index_frozen_a2, 0.0)

    def test_comp_index_frozen_a3(self):
        self.assertEqual(self.m.comp_index_frozen_a3, 0.0)

    def test_deg_sat_water(self):
        Sw, dSw_dT = self.m.deg_sat_water(-0.01)
        expected_Sw = 0.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = self.m.deg_sat_water(-50.0)
        expected_Sw = 0.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = self.m.deg_sat_water(0.01)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = self.m.deg_sat_water(50.0)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = self.m.deg_sat_water(0.0)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)

    def test_hyd_cond(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond(e=0.2, temp=-0.1, thawed=False)
        with self.assertRaises(ValueError):
            self.m.hyd_cond(e=-0.01, temp=0.1, thawed=False)
        k, dk_de = self.m.hyd_cond(e=0.2, temp=1.0, thawed=False)
        self.assertEqual(k, 0.0)
        self.assertEqual(dk_de, 0.0)

    def test_water_flux(self):
        with self.assertRaises(ValueError):
            self.m.water_flux(
                e=0.2, e0=0.3, temp=0.0, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
            )
        with self.assertRaises(ValueError):
            self.m.water_flux(
                e=0.2, e0=0.3, temp=1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
            )
        qw = self.m.water_flux(
            e=0.2, e0=0.3, temp=-1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
        )
        self.assertAlmostEqual(qw, 0.0)

    def test_eff_stress(self):
        sig_p, dsig_de = self.m.eff_stress(0.0, ppc=0.0)
        self.assertEqual(sig_p, 0.0)
        self.assertEqual(dsig_de, 0.0)

    def test_comp_index_frozen(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen(temp=0.0)

    def test_tot_stress(self):
        with self.assertRaises(ValueError):
            self.m.tot_stress(temp=0.0, e=0.350, e_f0=0.355, sig_f0=3e5)


class TestNullMaterial(unittest.TestCase):
    def test_thrm_cond_solids(self):
        self.assertEqual(NULL_MATERIAL.thrm_cond_solids, 0.0)

    def test_dens_solids(self):
        self.assertEqual(NULL_MATERIAL.dens_solids, 1.0e3)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(NULL_MATERIAL.spec_heat_cap_solids, 0.0)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(NULL_MATERIAL.vol_heat_cap_solids, 0.0)

    def test_deg_sat_water_alpha(self):
        self.assertEqual(NULL_MATERIAL.deg_sat_water_alpha, 1.0)

    def test_deg_sat_water_beta(self):
        self.assertEqual(NULL_MATERIAL.deg_sat_water_beta, 0.9)

    def test_hyd_cond_index(self):
        self.assertEqual(NULL_MATERIAL.hyd_cond_index, 1.0)

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
        self.assertEqual(NULL_MATERIAL.temp_rate_ref, 1.0e-9)

    def test_seg_pot_0(self):
        self.assertEqual(NULL_MATERIAL.seg_pot_0, 0.0)

    def test_void_ratio_0_comp(self):
        self.assertEqual(NULL_MATERIAL.void_ratio_0_comp, 0.0)

    def test_eff_stress_0_comp(self):
        self.assertEqual(NULL_MATERIAL.eff_stress_0_comp, 0.0)

    def test_comp_index_unfrozen(self):
        self.assertEqual(NULL_MATERIAL.comp_index_unfrozen, 1.0)

    def test_rebound_index_unfozen(self):
        self.assertEqual(NULL_MATERIAL.comp_index_unfrozen, 1.0)

    def test_comp_index_frozen_a1(self):
        self.assertEqual(NULL_MATERIAL.comp_index_frozen_a1, 0.0)

    def test_comp_index_frozen_a2(self):
        self.assertEqual(NULL_MATERIAL.comp_index_frozen_a2, 0.0)

    def test_comp_index_frozen_a3(self):
        self.assertEqual(NULL_MATERIAL.comp_index_frozen_a3, 0.0)

    def test_deg_sat_water(self):
        Sw, dSw_dT = NULL_MATERIAL.deg_sat_water(-0.01)
        expected_Sw = 0.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = NULL_MATERIAL.deg_sat_water(-50.0)
        expected_Sw = 0.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = NULL_MATERIAL.deg_sat_water(0.01)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = NULL_MATERIAL.deg_sat_water(50.0)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = NULL_MATERIAL.deg_sat_water(0.0)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)

    def test_hyd_cond(self):
        with self.assertRaises(ValueError):
            NULL_MATERIAL.hyd_cond(e=0.2, temp=-0.1, thawed=False)
        with self.assertRaises(ValueError):
            NULL_MATERIAL.hyd_cond(e=-0.01, temp=0.1, thawed=False)
        k, dk_de = NULL_MATERIAL.hyd_cond(e=0.2, temp=1.0, thawed=False)
        self.assertEqual(k, 0.0)
        self.assertEqual(dk_de, 0.0)

    def test_water_flux(self):
        with self.assertRaises(ValueError):
            NULL_MATERIAL.water_flux(
                e=0.2, e0=0.3, temp=0.0, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
            )
        with self.assertRaises(ValueError):
            NULL_MATERIAL.water_flux(
                e=0.2, e0=0.3, temp=1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
            )
        qw = NULL_MATERIAL.water_flux(
            e=0.2, e0=0.3, temp=-1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
        )
        self.assertAlmostEqual(qw, 0.0)

    def test_eff_stress(self):
        sig_p, dsig_de = NULL_MATERIAL.eff_stress(0.0, ppc=0.0)
        self.assertEqual(sig_p, 0.0)
        self.assertEqual(dsig_de, 0.0)

    def test_comp_index_frozen(self):
        with self.assertRaises(ValueError):
            NULL_MATERIAL.comp_index_frozen(temp=0.0)

    def test_tot_stress(self):
        with self.assertRaises(ValueError):
            NULL_MATERIAL.tot_stress(temp=0.0, e=0.350, e_f0=0.355, sig_f0=3e5)


class TestMaterialInitializers(unittest.TestCase):
    def setUp(self):
        self.m = Material(
            thrm_cond_solids=7.8,
            spec_grav_solids=2.5,
            spec_heat_cap_solids=7.41e5,
            deg_sat_water_alpha=12.0e3,
            deg_sat_water_beta=0.35,
            hyd_cond_index=0.305,
            hyd_cond_mult=0.5,
            hyd_cond_0=4.05e-4,
            void_ratio_0_hyd_cond=2.6,
            void_ratio_min=0.3,
            void_ratio_sep=1.6,
            void_ratio_lim=0.28,
            void_ratio_tr=0.5,
            water_flux_b1=0.08,
            water_flux_b2=4.0,
            water_flux_b3=10.0,
            temp_rate_ref=1e-9,
            seg_pot_0=2.0e-9,
            void_ratio_0_comp=2.6,
            eff_stress_0_comp=2.8e0,
            comp_index_unfrozen=0.421,
            rebound_index_unfrozen=0.08,
            comp_index_frozen_a1=0.021,
            comp_index_frozen_a2=0.01,
            comp_index_frozen_a3=0.23,
        )

    def test_thrm_cond_solids(self):
        self.assertEqual(self.m.thrm_cond_solids, 7.8)

    def test_dens_solids(self):
        self.assertEqual(self.m.dens_solids, 2.5e3)

    def test_spec_heat_cap_solids(self):
        self.assertEqual(self.m.spec_heat_cap_solids, 7.41e5)

    def test_vol_heat_cap_solids(self):
        self.assertEqual(self.m.vol_heat_cap_solids, 1.8525e9)

    def test_deg_sat_water_alpha(self):
        self.assertEqual(self.m.deg_sat_water_alpha, 12.0e3)

    def test_deg_sat_water_beta(self):
        self.assertEqual(self.m.deg_sat_water_beta, 0.35)

    def test_hyd_cond_index(self):
        self.assertEqual(self.m.hyd_cond_index, 0.305)

    def test_hyd_cond_mult(self):
        self.assertEqual(self.m.hyd_cond_mult, 0.5)

    def test_hyd_cond_0(self):
        self.assertEqual(self.m.hyd_cond_0, 4.05e-4)

    def test_void_ratio_0_hyd_cond(self):
        self.assertEqual(self.m.void_ratio_0_hyd_cond, 2.6)

    def test_void_ratio_min(self):
        self.assertEqual(self.m.void_ratio_min, 0.3)

    def test_void_ratio_sep(self):
        self.assertEqual(self.m.void_ratio_sep, 1.6)

    def test_void_ratio_lim(self):
        self.assertEqual(self.m.void_ratio_lim, 0.28)

    def test_void_ratio_tr(self):
        self.assertEqual(self.m.void_ratio_tr, 0.5)

    def test_water_flux_b1(self):
        self.assertEqual(self.m.water_flux_b1, 0.08)

    def test_water_flux_b2(self):
        self.assertEqual(self.m.water_flux_b2, 4.0)

    def test_water_flux_b3(self):
        self.assertEqual(self.m.water_flux_b3, 10.0)

    def test_temp_rate_ref(self):
        self.assertEqual(self.m.temp_rate_ref, 1.0e-9)

    def test_seg_pot_0(self):
        self.assertEqual(self.m.seg_pot_0, 2.0e-9)

    def test_void_ratio_0_comp(self):
        self.assertEqual(self.m.void_ratio_0_comp, 2.6)

    def test_eff_stress_0_comp(self):
        self.assertEqual(self.m.eff_stress_0_comp, 2.8e0)

    def test_comp_index_unfrozen(self):
        self.assertEqual(self.m.comp_index_unfrozen, 0.421)

    def test_rebound_index_unfozen(self):
        self.assertEqual(self.m.rebound_index_unfrozen, 0.08)

    def test_comp_index_frozen_a1(self):
        self.assertEqual(self.m.comp_index_frozen_a1, 0.021)

    def test_comp_index_frozen_a2(self):
        self.assertEqual(self.m.comp_index_frozen_a2, 0.01)

    def test_comp_index_frozen_a3(self):
        self.assertEqual(self.m.comp_index_frozen_a3, 0.23)

    def test_deg_sat_water(self):
        Sw, dSw_dT = self.m.deg_sat_water(-0.15)
        expected_Sw = 0.240990639551049
        expected_dSw_dT = 0.850491042320274
        self.assertAlmostEqual(Sw, expected_Sw, places=14)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT, places=14)
        Sw, dSw_dT = self.m.deg_sat_water(-50.0)
        expected_Sw = 0.0100688310213271
        expected_dSw_dT = 0.000120172321947552
        self.assertAlmostEqual(Sw, expected_Sw, places=14)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT, places=14)
        Sw, dSw_dT = self.m.deg_sat_water(1.0)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = self.m.deg_sat_water(50.0)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)
        Sw, dSw_dT = self.m.deg_sat_water(0.0)
        expected_Sw = 1.0
        expected_dSw_dT = 0.0
        self.assertAlmostEqual(Sw, expected_Sw)
        self.assertAlmostEqual(dSw_dT, expected_dSw_dT)

    def test_hyd_cond_method_invalid(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond(e=0.2, temp=-0.1, thawed=False)
        with self.assertRaises(ValueError):
            self.m.hyd_cond(e=-0.01, temp=0.1, thawed=False)

    def test_hyd_cond_method_0(self):
        K, dK_de = self.m.hyd_cond(e=0.4, temp=1.50, thawed=False)
        expected_K = 2.47936387494043e-11
        expected_dK_de = 1.87178567165436e-10
        self.assertAlmostEqual(K, expected_K, delta=1e-18)
        self.assertAlmostEqual(dK_de, expected_dK_de, delta=1e-18)

    def test_hyd_cond_method_1(self):
        K, dK_de = self.m.hyd_cond(e=0.6, temp=1.50, thawed=False)
        expected_K = 1.12221992165536e-10
        expected_dK_de = 8.47215364808056e-10
        self.assertAlmostEqual(K, expected_K, delta=1e-18)
        self.assertAlmostEqual(dK_de, expected_dK_de, delta=1e-18)

    def test_hyd_cond_method_2(self):
        K, dK_de = self.m.hyd_cond(e=0.4, temp=1.50, thawed=True)
        expected_K = 1.23968193747022e-11
        expected_dK_de = 9.3589283582718e-11
        self.assertAlmostEqual(K, expected_K, delta=1e-18)
        self.assertAlmostEqual(dK_de, expected_dK_de, delta=1e-18)

    def test_hyd_cond_method_3(self):
        K, dK_de = self.m.hyd_cond(e=0.6, temp=1.50, thawed=True)
        expected_K = 1.12221992165536e-10
        expected_dK_de = 8.47215364808056e-10
        self.assertAlmostEqual(K, expected_K, delta=1e-18)
        self.assertAlmostEqual(dK_de, expected_dK_de, delta=1e-18)

    def test_water_flux_method_invalid(self):
        with self.assertRaises(ValueError):
            self.m.water_flux(
                e=0.2, e0=0.3, temp=0.0, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
            )
        with self.assertRaises(ValueError):
            self.m.water_flux(
                e=0.2, e0=0.3, temp=1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=10.0
            )

    def test_water_flux_method_0(self):
        #   Check temp_rate > 0.0, temp_grad > 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=1e-3
        )
        expected_water_flux = 1.34121901234503e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_1(self):
        #   Check temp_rate > 0.0, temp_grad > 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=0.0
        )
        expected_water_flux = 1.35469848751556e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_2(self):
        #   Check temp_rate > 0.0, temp_grad > 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=0.05, sigma_1=-1e-3
        )
        expected_water_flux = 1.36831343366376e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_3(self):
        #   Check temp_rate > 0.0, temp_grad = 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=0.0, sigma_1=1e-3
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_4(self):
        #   Check temp_rate > 0.0, temp_grad = 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=0.0, sigma_1=0.0
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_5(self):
        #   Check temp_rate > 0.0, temp_grad = 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=0.0, sigma_1=-1e-3
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_6(self):
        #   Check temp_rate > 0.0, temp_grad < 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=-0.05, sigma_1=1e-3
        )
        expected_water_flux = -1.34121901234503e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_7(self):
        #   Check temp_rate > 0.0, temp_grad < 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=-0.05, sigma_1=0.0
        )
        expected_water_flux = -1.35469848751556e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_8(self):
        #   Check temp_rate > 0.0, temp_grad < 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.1, temp_grad=-0.05, sigma_1=-1e-3
        )
        expected_water_flux = -1.36831343366376e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_9(self):
        #   Check temp_rate = 0.0, temp_grad > 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=0.05, sigma_1=1e-3
        )
        expected_water_flux = -2.8316402081699e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_10(self):
        #   Check temp_rate = 0.0, temp_grad > 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=0.05, sigma_1=0.0
        )
        expected_water_flux = -2.86009866538426e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_11(self):
        #   Check temp_rate = 0.0, temp_grad > 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=0.05, sigma_1=-1e-3
        )
        expected_water_flux = -2.88884313484858e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_12(self):
        #   Check temp_rate = 0.0, temp_grad = 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=0.0, sigma_1=1e-3
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_13(self):
        #   Check temp_rate = 0.0, temp_grad = 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=0.0, sigma_1=0.0
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_14(self):
        #   Check temp_rate = 0.0, temp_grad = 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=0.0, sigma_1=-1e-3
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_15(self):
        #   Check temp_rate = 0.0, temp_grad < 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=-0.05, sigma_1=1e-3
        )
        expected_water_flux = 2.8316402081699e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_16(self):
        #   Check temp_rate = 0.0, temp_grad < 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=-0.05, sigma_1=0.0
        )
        expected_water_flux = 2.86009866538426e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_17(self):
        #   Check temp_rate = 0.0, temp_grad < 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=0.0, temp_grad=-0.05, sigma_1=-1e-3
        )
        expected_water_flux = 2.88884313484858e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_18(self):
        #   Check temp_rate < 0.0, temp_grad > 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=0.05, sigma_1=1e-3
        )
        expected_water_flux = -7.00449942868483e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_19(self):
        #   Check temp_rate < 0.0, temp_grad > 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=0.05, sigma_1=0.0
        )
        expected_water_flux = -7.07489581828408e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_20(self):
        #   Check temp_rate < 0.0, temp_grad > 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=0.05, sigma_1=-1e-3
        )
        expected_water_flux = -7.14599970336091e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_21(self):
        #   Check temp_rate < 0.0, temp_grad = 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=0.0, sigma_1=1e-3
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_22(self):
        #   Check temp_rate < 0.0, temp_grad = 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=0.0, sigma_1=0.0
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_23(self):
        #   Check temp_rate < 0.0, temp_grad = 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=0.0, sigma_1=-1e-3
        )
        expected_water_flux = 0.0
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_24(self):
        #   Check temp_rate < 0.0, temp_grad < 0.0, sigma_1 > 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=-0.05, sigma_1=1e-3
        )
        expected_water_flux = 7.00449942868483e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_25(self):
        #   Check temp_rate < 0.0, temp_grad < 0.0, sigma_1 = 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=-0.05, sigma_1=0.0
        )
        expected_water_flux = 7.07489581828408e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_water_flux_method_26(self):
        #   Check temp_rate < 0.0, temp_grad < 0.0, sigma_1 < 0.0
        water_flux = self.m.water_flux(
            e=0.3, e0=0.5, temp=-1.5, temp_rate=-0.1, temp_grad=-0.05, sigma_1=-1e-3
        )
        expected_water_flux = 7.14599970336091e-13
        self.assertAlmostEqual(water_flux, expected_water_flux, delta=1e-20)

    def test_eff_stress(self):
        #    e < e pc´
        sig_p, dsig_de = self.m.eff_stress(e=0.4, ppc=1e5)
        expected_sig_p = 470772.665017368
        expected_dsig_de = -2574807.88754886
        self.assertAlmostEqual(sig_p, expected_sig_p, places=8)
        self.assertAlmostEqual(dsig_de, expected_dsig_de, places=8)
        #    e > e pc´
        sig_p, dsig_de = self.m.eff_stress(e=0.9, ppc=1e5)
        expected_sig_p = 195.28511416096
        expected_dsig_de = -5620.75740938333
        self.assertAlmostEqual(sig_p, expected_sig_p, places=8)
        self.assertAlmostEqual(dsig_de, expected_dsig_de, places=8)

    def test_comp_index_frozen(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen(temp=0.0)
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen(temp=5.0)
        comp_index_frozen = self.m.comp_index_frozen(temp=-5.0)
        expected_comp_index_frozen = 0.00652018207187661
        self.assertAlmostEqual(comp_index_frozen, expected_comp_index_frozen, places=10)

    def test_tot_stress(self):
        with self.assertRaises(ValueError):
            self.m.tot_stress(temp=0.0, e=0.350, e_f0=0.355, sig_f0=3e5)
        with self.assertRaises(ValueError):
            self.m.tot_stress(temp=5.0, e=0.350, e_f0=0.355, sig_f0=3e5)
        # e < e_f0
        sig, dsig_de = self.m.tot_stress(temp=-5.0, e=0.350, e_f0=0.355, sig_f0=3e5)
        expected_sig = 1753763.41459215
        expected_dsig_de = -619336921.96972
        self.assertAlmostEqual(sig, expected_sig, places=5)
        self.assertAlmostEqual(dsig_de, expected_dsig_de, places=5)
        # e > e_f0
        sig, dsig_de = self.m.tot_stress(temp=-5.0, e=0.36, e_f0=0.355, sig_f0=3e5)
        expected_sig = 51318.2104559584
        expected_dsig_de = -18122890.6021962
        self.assertAlmostEqual(sig, expected_sig, places=5)
        self.assertAlmostEqual(dsig_de, expected_dsig_de, places=5)
        # e = e_f0
        sig, dsig_de = self.m.tot_stress(temp=-5.0, e=0.355, e_f0=0.355, sig_f0=3e5)
        expected_sig = 300000
        expected_dsig_de = -105944208.349292
        self.assertAlmostEqual(sig, expected_sig, places=5)
        self.assertAlmostEqual(dsig_de, expected_dsig_de, places=5)


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
            thrm_cond_solids=7.8, spec_grav_solids=2.5, spec_heat_cap_solids=7.41e5
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


class TestMaterialDegSatWaterAlphaSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_deg_sat_water_alpha_valid_float(self):
        self.m.deg_sat_water_alpha = 0.2
        self.assertEqual(self.m.deg_sat_water_alpha, 0.2)

    def test_set_deg_sat_water_alpha_valid_int(self):
        self.m.deg_sat_water_alpha = 12
        self.assertEqual(self.m.deg_sat_water_alpha, 12.0)

    def test_set_deg_sat_water_alpha_valid_int_type(self):
        self.m.deg_sat_water_alpha = 12
        self.assertIsInstance(self.m.deg_sat_water_alpha, float)

    def test_set_deg_sat_water_alpha_valid_str(self):
        self.m.deg_sat_water_alpha = "1.2e1"
        self.assertEqual(self.m.deg_sat_water_alpha, 12.0)

    def test_set_deg_sat_water_alpha_valid_str_type(self):
        self.m.deg_sat_water_alpha = "1.2e1"
        self.assertIsInstance(self.m.deg_sat_water_alpha, float)

    def test_set_deg_sat_water_alpha_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.deg_sat_water_alpha = (12.0, 1.8)

    def test_set_deg_sat_water_alpha_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.deg_sat_water_alpha = -12.0

    def test_set_deg_sat_water_alpha_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.deg_sat_water_alpha = "twelve"


class TestMaterialDegSatWaterBetaSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_deg_sat_water_beta_valid_float(self):
        self.m.deg_sat_water_beta = 0.2
        self.assertEqual(self.m.deg_sat_water_beta, 0.2)

    def test_set_deg_sat_water_beta_valid_int(self):
        self.m.deg_sat_water_beta = 12
        self.assertEqual(self.m.deg_sat_water_beta, 12.0)

    def test_set_deg_sat_water_beta_valid_int_type(self):
        self.m.deg_sat_water_beta = 12
        self.assertIsInstance(self.m.deg_sat_water_beta, float)

    def test_set_deg_sat_water_beta_valid_str(self):
        self.m.deg_sat_water_beta = "1.2e1"
        self.assertEqual(self.m.deg_sat_water_beta, 12.0)

    def test_set_deg_sat_water_beta_valid_str_type(self):
        self.m.deg_sat_water_beta = "1.2e1"
        self.assertIsInstance(self.m.deg_sat_water_beta, float)

    def test_set_deg_sat_water_beta_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.deg_sat_water_beta = (12.0, 1.8)

    def test_set_deg_sat_water_beta_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.deg_sat_water_beta = -12.0

    def test_set_deg_sat_water_beta_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.deg_sat_water_beta = "twelve"


class TestMaterialHydCondIndexSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_hyd_cond_index_valid_float(self):
        self.m.hyd_cond_index = 0.2
        self.assertEqual(self.m.hyd_cond_index, 0.2)

    def test_set_hyd_cond_index_valid_int(self):
        self.m.hyd_cond_index = 12
        self.assertEqual(self.m.hyd_cond_index, 12.0)

    def test_set_hyd_cond_index_valid_int_type(self):
        self.m.hyd_cond_index = 12
        self.assertIsInstance(self.m.hyd_cond_index, float)

    def test_set_hyd_cond_index_valid_str(self):
        self.m.hyd_cond_index = "1.2e1"
        self.assertEqual(self.m.hyd_cond_index, 12.0)

    def test_set_hyd_cond_index_valid_str_type(self):
        self.m.hyd_cond_index = "1.2e1"
        self.assertIsInstance(self.m.hyd_cond_index, float)

    def test_set_hyd_cond_index_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.hyd_cond_index = (12.0, 1.8)

    def test_set_hyd_cond_index_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond_index = -12.0

    def test_set_hyd_cond_index_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond_index = "twelve"


class TestMaterialHydCondMultSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_hyd_cond_mult_valid_float(self):
        self.m.hyd_cond_mult = 0.2
        self.assertEqual(self.m.hyd_cond_mult, 0.2)

    def test_set_hyd_cond_mult_valid_int(self):
        self.m.hyd_cond_mult = 12
        self.assertEqual(self.m.hyd_cond_mult, 12.0)

    def test_set_hyd_cond_mult_valid_int_type(self):
        self.m.hyd_cond_mult = 12
        self.assertIsInstance(self.m.hyd_cond_mult, float)

    def test_set_hyd_cond_mult_valid_str(self):
        self.m.hyd_cond_mult = "1.2e1"
        self.assertEqual(self.m.hyd_cond_mult, 12.0)

    def test_set_hyd_cond_mult_valid_str_type(self):
        self.m.hyd_cond_mult = "1.2e1"
        self.assertIsInstance(self.m.hyd_cond_mult, float)

    def test_set_hyd_cond_mult_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.hyd_cond_mult = (12.0, 1.8)

    def test_set_hyd_cond_mult_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond_mult = -12.0

    def test_set_hyd_cond_0_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond_mult = "twelve"


class TestMaterialHydCond0Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_hyd_cond_0_valid_float(self):
        self.m.hyd_cond_0 = 0.2
        self.assertEqual(self.m.hyd_cond_0, 0.2)

    def test_set_hyd_cond_0_valid_int(self):
        self.m.hyd_cond_0 = 12
        self.assertEqual(self.m.hyd_cond_0, 12.0)

    def test_set_hyd_cond_0_valid_int_type(self):
        self.m.hyd_cond_0 = 12
        self.assertIsInstance(self.m.hyd_cond_0, float)

    def test_set_hyd_cond_0_valid_str(self):
        self.m.hyd_cond_0 = "1.2e1"
        self.assertEqual(self.m.hyd_cond_0, 12.0)

    def test_set_hyd_cond_0_valid_str_type(self):
        self.m.hyd_cond_0 = "1.2e1"
        self.assertIsInstance(self.m.hyd_cond_0, float)

    def test_set_hyd_cond_0_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.hyd_cond_0 = (12.0, 1.8)

    def test_set_hyd_cond_0_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond_0 = -12.0

    def test_set_hyd_cond_0_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.hyd_cond_0 = "twelve"


class TestMaterialVoidRatio0HydCondSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_void_ratio_0_hyd_cond_valid_float(self):
        self.m.void_ratio_0_hyd_cond = 0.2
        self.assertEqual(self.m.void_ratio_0_hyd_cond, 0.2)

    def test_set_void_ratio_0_hyd_cond_valid_int(self):
        self.m.void_ratio_0_hyd_cond = 12
        self.assertEqual(self.m.void_ratio_0_hyd_cond, 12.0)

    def test_set_void_ratio_0_hyd_cond_valid_int_type(self):
        self.m.void_ratio_0_hyd_cond = 12
        self.assertIsInstance(self.m.void_ratio_0_hyd_cond, float)

    def test_set_void_ratio_0_hyd_cond_valid_str(self):
        self.m.void_ratio_0_hyd_cond = "1.2e1"
        self.assertEqual(self.m.void_ratio_0_hyd_cond, 12.0)

    def test_set_void_ratio_0_hyd_cond_valid_str_type(self):
        self.m.void_ratio_0_hyd_cond = "1.2e1"
        self.assertIsInstance(self.m.void_ratio_0_hyd_cond, float)

    def test_set_void_ratio_0_hyd_cond_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.void_ratio_0_hyd_cond = (12.0, 1.8)

    def test_set_void_ratio_0_hyd_cond_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_0_hyd_cond = -12.0

    def test_set_void_ratio_0_hyd_cond_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_0_hyd_cond = "twelve"


class TestMaterialVoidRatioMinSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_void_ratio_min_valid_float(self):
        self.m.void_ratio_min = 0.2
        self.assertEqual(self.m.void_ratio_min, 0.2)

    def test_set_void_ratio_min_valid_int(self):
        self.m.void_ratio_min = 12
        self.assertEqual(self.m.void_ratio_min, 12.0)

    def test_set_void_ratio_min_valid_int_type(self):
        self.m.void_ratio_min = 12
        self.assertIsInstance(self.m.void_ratio_min, float)

    def test_set_void_ratio_min_valid_str(self):
        self.m.void_ratio_min = "1.2e1"
        self.assertEqual(self.m.void_ratio_min, 12.0)

    def test_set_void_ratio_min_valid_str_type(self):
        self.m.void_ratio_min = "1.2e1"
        self.assertIsInstance(self.m.void_ratio_min, float)

    def test_set_void_ratio_min_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.void_ratio_min = (12.0, 1.8)

    def test_set_void_ratio_min_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_min = -12.0

    def test_set_void_ratio_min_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_min = "twelve"


class TestMaterialVoidRatioSepSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_void_ratio_sep_valid_float(self):
        self.m.void_ratio_sep = 0.2
        self.assertEqual(self.m.void_ratio_sep, 0.2)

    def test_set_void_ratio_sep_valid_int(self):
        self.m.void_ratio_sep = 12
        self.assertEqual(self.m.void_ratio_sep, 12.0)

    def test_set_void_ratio_sep_valid_int_type(self):
        self.m.void_ratio_sep = 12
        self.assertIsInstance(self.m.void_ratio_sep, float)

    def test_set_void_ratio_sep_valid_str(self):
        self.m.void_ratio_sep = "1.2e1"
        self.assertEqual(self.m.void_ratio_sep, 12.0)

    def test_set_void_ratio_sep_valid_str_type(self):
        self.m.void_ratio_sep = "1.2e1"
        self.assertIsInstance(self.m.void_ratio_sep, float)

    def test_set_void_ratio_sep_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.void_ratio_sep = (12.0, 1.8)

    def test_set_void_ratio_sep_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_sep = -12.0

    def test_set_void_ratio_sep_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_sep = "twelve"


class TestMaterialVoidRatioLimSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_void_ratio_lim_valid_float(self):
        self.m.void_ratio_lim = 0.2
        self.assertEqual(self.m.void_ratio_lim, 0.2)

    def test_set_void_ratio_lim_valid_int(self):
        self.m.void_ratio_lim = 12
        self.assertEqual(self.m.void_ratio_lim, 12.0)

    def test_set_void_ratio_lim_valid_int_type(self):
        self.m.void_ratio_lim = 12
        self.assertIsInstance(self.m.void_ratio_lim, float)

    def test_set_void_ratio_lim_valid_str(self):
        self.m.void_ratio_lim = "1.2e1"
        self.assertEqual(self.m.void_ratio_lim, 12.0)

    def test_set_void_ratio_lim_valid_str_type(self):
        self.m.void_ratio_lim = "1.2e1"
        self.assertIsInstance(self.m.void_ratio_lim, float)

    def test_set_void_ratio_lim_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.void_ratio_lim = (12.0, 1.8)

    def test_set_void_ratio_lim_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_lim = -12.0

    def test_set_void_ratio_lim_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_lim = "twelve"


class TestMaterialVoidRatioTrSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_void_ratio_tr_valid_float(self):
        self.m.void_ratio_tr = 0.2
        self.assertEqual(self.m.void_ratio_tr, 0.2)

    def test_set_void_ratio_tr_valid_int(self):
        self.m.void_ratio_tr = 12
        self.assertEqual(self.m.void_ratio_tr, 12.0)

    def test_set_void_ratio_tr_valid_int_type(self):
        self.m.void_ratio_tr = 12
        self.assertIsInstance(self.m.void_ratio_tr, float)

    def test_set_void_ratio_tr_valid_str(self):
        self.m.void_ratio_tr = "1.2e1"
        self.assertEqual(self.m.void_ratio_tr, 12.0)

    def test_set_void_ratio_tr_valid_str_type(self):
        self.m.void_ratio_tr = "1.2e1"
        self.assertIsInstance(self.m.void_ratio_tr, float)

    def test_set_void_ratio_tr_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.void_ratio_tr = (12.0, 1.8)

    def test_set_void_ratio_tr_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_tr = -12.0

    def test_set_void_ratio_tr_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_tr = "twelve"


class TestMaterialWaterFluxb1Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_water_flux_b1_valid_float(self):
        self.m.water_flux_b1 = 0.2
        self.assertEqual(self.m.water_flux_b1, 0.2)

    def test_set_water_flux_b1_valid_int(self):
        self.m.water_flux_b1 = 12
        self.assertEqual(self.m.water_flux_b1, 12.0)

    def test_set_water_flux_b1_valid_int_type(self):
        self.m.water_flux_b1 = 12
        self.assertIsInstance(self.m.water_flux_b1, float)

    def test_set_water_flux_b1_valid_str(self):
        self.m.water_flux_b1 = "1.2e1"
        self.assertEqual(self.m.water_flux_b1, 12.0)

    def test_set_water_flux_b1_valid_str_type(self):
        self.m.water_flux_b1 = "1.2e1"
        self.assertIsInstance(self.m.water_flux_b1, float)

    def test_set_water_flux_b1_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.water_flux_b1 = (12.0, 1.8)

    def test_set_water_flux_b1_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.water_flux_b1 = -12.0

    def test_set_water_flux_b1_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.water_flux_b1 = "twelve"


class TestMaterialWaterFluxb2Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_water_flux_b2_valid_float(self):
        self.m.water_flux_b2 = 0.2
        self.assertEqual(self.m.water_flux_b2, 0.2)

    def test_set_water_flux_b2_valid_int(self):
        self.m.water_flux_b2 = 12
        self.assertEqual(self.m.water_flux_b2, 12.0)

    def test_set_water_flux_b2_valid_int_type(self):
        self.m.water_flux_b2 = 12
        self.assertIsInstance(self.m.water_flux_b2, float)

    def test_set_water_flux_b2_valid_str(self):
        self.m.water_flux_b2 = "1.2e1"
        self.assertEqual(self.m.water_flux_b2, 12.0)

    def test_set_water_flux_b2_valid_str_type(self):
        self.m.water_flux_b2 = "1.2e1"
        self.assertIsInstance(self.m.water_flux_b2, float)

    def test_set_water_flux_b2_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.water_flux_b2 = (12.0, 1.8)

    def test_set_water_flux_b2_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.water_flux_b2 = -12.0

    def test_set_water_flux_b2_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.water_flux_b2 = "twelve"


class TestMaterialWaterFluxb3Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_water_flux_b3_valid_float(self):
        self.m.water_flux_b3 = 0.2
        self.assertEqual(self.m.water_flux_b3, 0.2)

    def test_set_water_flux_b3_valid_int(self):
        self.m.water_flux_b3 = 12
        self.assertEqual(self.m.water_flux_b3, 12.0)

    def test_set_water_flux_b3_valid_int_type(self):
        self.m.water_flux_b3 = 12
        self.assertIsInstance(self.m.water_flux_b3, float)

    def test_set_water_flux_b3_valid_str(self):
        self.m.water_flux_b3 = "1.2e1"
        self.assertEqual(self.m.water_flux_b3, 12.0)

    def test_set_water_flux_b3_valid_str_type(self):
        self.m.water_flux_b3 = "1.2e1"
        self.assertIsInstance(self.m.water_flux_b3, float)

    def test_set_water_flux_b3_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.water_flux_b3 = (12.0, 1.8)

    def test_set_water_flux_b3_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.water_flux_b3 = -12.0

    def test_set_water_flux_b3_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.water_flux_b3 = "twelve"


class TestMaterialTempRateRefSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_temp_rate_ref_valid_float(self):
        self.m.temp_rate_ref = 0.2
        self.assertEqual(self.m.temp_rate_ref, 0.2)

    def test_set_temp_rate_ref_valid_int(self):
        self.m.temp_rate_ref = 12
        self.assertEqual(self.m.temp_rate_ref, 12.0)

    def test_set_temp_rate_ref_valid_int_type(self):
        self.m.temp_rate_ref = 12
        self.assertIsInstance(self.m.temp_rate_ref, float)

    def test_set_temp_rate_ref_valid_str(self):
        self.m.temp_rate_ref = "1.2e1"
        self.assertEqual(self.m.temp_rate_ref, 12.0)

    def test_set_temp_rate_ref_valid_str_type(self):
        self.m.temp_rate_ref = "1.2e1"
        self.assertIsInstance(self.m.temp_rate_ref, float)

    def test_set_temp_rate_ref_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.temp_rate_ref = (12.0, 1.8)

    def test_set_temp_rate_ref_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.temp_rate_ref = -12.0

    def test_set_temp_rate_ref_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.temp_rate_ref = "twelve"


class TestMaterialSegPot0Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_seg_pot_0_valid_float(self):
        self.m.seg_pot_0 = 0.2
        self.assertEqual(self.m.seg_pot_0, 0.2)

    def test_set_seg_pot_0_valid_int(self):
        self.m.seg_pot_0 = 12
        self.assertEqual(self.m.seg_pot_0, 12.0)

    def test_set_seg_pot_0_valid_int_type(self):
        self.m.seg_pot_0 = 12
        self.assertIsInstance(self.m.seg_pot_0, float)

    def test_set_seg_pot_0_valid_str(self):
        self.m.seg_pot_0 = "1.2e1"
        self.assertEqual(self.m.seg_pot_0, 12.0)

    def test_set_seg_pot_0_valid_str_type(self):
        self.m.seg_pot_0 = "1.2e1"
        self.assertIsInstance(self.m.seg_pot_0, float)

    def test_set_seg_pot_0_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.seg_pot_0 = (12.0, 1.8)

    def test_set_seg_pot_0_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.seg_pot_0 = -12.0

    def test_set_seg_pot_0_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.seg_pot_0 = "twelve"


class TestMaterialVoidRatio0CompSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_void_ratio_0_comp_valid_float(self):
        self.m.void_ratio_0_comp = 0.2
        self.assertEqual(self.m.void_ratio_0_comp, 0.2)

    def test_set_void_ratio_0_comp_valid_int(self):
        self.m.void_ratio_0_comp = 12
        self.assertEqual(self.m.void_ratio_0_comp, 12.0)

    def test_set_void_ratio_0_comp_valid_int_type(self):
        self.m.void_ratio_0_comp = 12
        self.assertIsInstance(self.m.void_ratio_0_comp, float)

    def test_set_void_ratio_0_comp_valid_str(self):
        self.m.void_ratio_0_comp = "1.2e1"
        self.assertEqual(self.m.void_ratio_0_comp, 12.0)

    def test_set_void_ratio_0_comp_valid_str_type(self):
        self.m.void_ratio_0_comp = "1.2e1"
        self.assertIsInstance(self.m.void_ratio_0_comp, float)

    def test_set_void_ratio_0_comp_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.void_ratio_0_comp = (12.0, 1.8)

    def test_set_void_ratio_0_comp_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_0_comp = -12.0

    def test_set_void_ratio_0_comp_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.void_ratio_0_comp = "twelve"


class TestMaterialEffStress0Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_eff_stress_0_comp_valid_float(self):
        self.m.eff_stress_0_comp = 0.2
        self.assertEqual(self.m.eff_stress_0_comp, 0.2)

    def test_set_eff_stress_0_comp_valid_int(self):
        self.m.eff_stress_0_comp = 12
        self.assertEqual(self.m.eff_stress_0_comp, 12.0)

    def test_set_eff_stress_0_comp_valid_int_type(self):
        self.m.eff_stress_0_comp = 12
        self.assertIsInstance(self.m.eff_stress_0_comp, float)

    def test_set_eff_stress_0_comp_valid_str(self):
        self.m.eff_stress_0_comp = "1.2e1"
        self.assertEqual(self.m.eff_stress_0_comp, 12.0)

    def test_set_eff_stress_0_comp_valid_str_type(self):
        self.m.eff_stress_0_comp = "1.2e1"
        self.assertIsInstance(self.m.eff_stress_0_comp, float)

    def test_set_eff_stress_0_comp_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.eff_stress_0_comp = (12.0, 1.8)

    def test_set_eff_stress_0_comp_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.eff_stress_0_comp = -12.0

    def test_set_eff_stress_0_comp_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.eff_stress_0_comp = "twelve"


class TestMaterialCompIndexUnfrozenSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_comp_index_unfrozen_valid_float(self):
        self.m.comp_index_unfrozen = 0.2
        self.assertEqual(self.m.comp_index_unfrozen, 0.2)

    def test_set_comp_index_unfrozen_valid_int(self):
        self.m.comp_index_unfrozen = 12
        self.assertEqual(self.m.comp_index_unfrozen, 12.0)

    def test_set_comp_index_unfrozen_valid_int_type(self):
        self.m.comp_index_unfrozen = 12
        self.assertIsInstance(self.m.comp_index_unfrozen, float)

    def test_set_comp_index_unfrozen_valid_str(self):
        self.m.comp_index_unfrozen = "1.2e1"
        self.assertEqual(self.m.comp_index_unfrozen, 12.0)

    def test_set_comp_index_unfrozen_valid_str_type(self):
        self.m.comp_index_unfrozen = "1.2e1"
        self.assertIsInstance(self.m.comp_index_unfrozen, float)

    def test_set_comp_index_unfrozen_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.comp_index_unfrozen = (12.0, 1.8)

    def test_set_comp_index_unfrozen_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_unfrozen = -12.0

    def test_set_comp_index_unfrozen_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_unfrozen = "twelve"


class TestMaterialReboundIndexUnfrozenSetter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_rebound_index_unfrozen_valid_float(self):
        self.m.rebound_index_unfrozen = 0.2
        self.assertEqual(self.m.rebound_index_unfrozen, 0.2)

    def test_set_rebound_index_unfrozen_valid_int(self):
        self.m.rebound_index_unfrozen = 12
        self.assertEqual(self.m.rebound_index_unfrozen, 12.0)

    def test_set_rebound_index_unfrozen_valid_int_type(self):
        self.m.rebound_index_unfrozen = 12
        self.assertIsInstance(self.m.rebound_index_unfrozen, float)

    def test_set_rebound_index_unfrozen_valid_str(self):
        self.m.rebound_index_unfrozen = "1.2e1"
        self.assertEqual(self.m.rebound_index_unfrozen, 12.0)

    def test_set_rebound_index_unfrozen_valid_str_type(self):
        self.m.rebound_index_unfrozen = "1.2e1"
        self.assertIsInstance(self.m.rebound_index_unfrozen, float)

    def test_set_rebound_index_unfrozen_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.rebound_index_unfrozen = (12.0, 1.8)

    def test_set_rebound_index_unfrozen_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.rebound_index_unfrozen = -12.0

    def test_set_rebound_index_unfrozen_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.rebound_index_unfrozen = "twelve"


class TestMaterialCompIndexFrozena1Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_comp_index_frozen_a1_valid_float(self):
        self.m.comp_index_frozen_a1 = 0.2
        self.assertEqual(self.m.comp_index_frozen_a1, 0.2)

    def test_set_comp_index_frozen_a1_valid_int(self):
        self.m.comp_index_frozen_a1 = 12
        self.assertEqual(self.m.comp_index_frozen_a1, 12.0)

    def test_set_comp_index_frozen_a1_valid_int_type(self):
        self.m.comp_index_frozen_a1 = 12
        self.assertIsInstance(self.m.comp_index_frozen_a1, float)

    def test_set_comp_index_frozen_a1_valid_str(self):
        self.m.comp_index_frozen_a1 = "1.2e1"
        self.assertEqual(self.m.comp_index_frozen_a1, 12.0)

    def test_set_comp_index_frozen_a1_valid_str_type(self):
        self.m.comp_index_frozen_a1 = "1.2e1"
        self.assertIsInstance(self.m.comp_index_frozen_a1, float)

    def test_set_comp_index_frozen_a1_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.comp_index_frozen_a1 = (12.0, 1.8)

    def test_set_comp_index_frozen_a1_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen_a1 = -12.0

    def test_set_comp_index_frozen_a1_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen_a1 = "twelve"


class TestMaterialCompIndexFrozena2Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_comp_index_frozen_a2_valid_float(self):
        self.m.comp_index_frozen_a2 = 0.2
        self.assertEqual(self.m.comp_index_frozen_a2, 0.2)

    def test_set_comp_index_frozen_a2_valid_int(self):
        self.m.comp_index_frozen_a2 = 12
        self.assertEqual(self.m.comp_index_frozen_a2, 12.0)

    def test_set_comp_index_frozen_a2_valid_int_type(self):
        self.m.comp_index_frozen_a2 = 12
        self.assertIsInstance(self.m.comp_index_frozen_a2, float)

    def test_set_comp_index_frozen_a2_valid_str(self):
        self.m.comp_index_frozen_a2 = "1.2e1"
        self.assertEqual(self.m.comp_index_frozen_a2, 12.0)

    def test_set_comp_index_frozen_a2_valid_str_type(self):
        self.m.comp_index_frozen_a2 = "1.2e1"
        self.assertIsInstance(self.m.comp_index_frozen_a2, float)

    def test_set_comp_index_frozen_a2_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.comp_index_frozen_a2 = (12.0, 1.8)

    def test_set_comp_index_frozen_a2_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen_a2 = -12.0

    def test_set_comp_index_frozen_a2_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen_a2 = "twelve"


class TestMaterialCompIndexFrozena3Setter(unittest.TestCase):
    def setUp(self):
        self.m = Material()

    def test_set_comp_index_frozen_a3_valid_float(self):
        self.m.comp_index_frozen_a3 = 0.2
        self.assertEqual(self.m.comp_index_frozen_a3, 0.2)

    def test_set_comp_index_frozen_a3_valid_int(self):
        self.m.comp_index_frozen_a3 = 12
        self.assertEqual(self.m.comp_index_frozen_a3, 12.0)

    def test_set_comp_index_frozen_a3_valid_int_type(self):
        self.m.comp_index_frozen_a3 = 12
        self.assertIsInstance(self.m.comp_index_frozen_a3, float)

    def test_set_comp_index_frozen_a3_valid_str(self):
        self.m.comp_index_frozen_a3 = "1.2e1"
        self.assertEqual(self.m.comp_index_frozen_a3, 12.0)

    def test_set_comp_index_frozen_a3_valid_str_type(self):
        self.m.comp_index_frozen_a3 = "1.2e1"
        self.assertIsInstance(self.m.comp_index_frozen_a3, float)

    def test_set_comp_index_frozen_a3_invalid_type(self):
        with self.assertRaises(TypeError):
            self.m.comp_index_frozen_a3 = (12.0, 1.8)

    def test_set_comp_index_frozen_a3_invalid_value(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen_a3 = -12.0

    def test_set_comp_index_frozen_a3_invalid_str(self):
        with self.assertRaises(ValueError):
            self.m.comp_index_frozen_a3 = "twelve"


if __name__ == "__main__":
    unittest.main()
