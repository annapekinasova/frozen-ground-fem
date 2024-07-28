"""frozen_ground_fem.materials.py - A module for material constants and classes
for tracking material properties.

"""

__all__ = [
    "grav_acc",
    "dens_water",
    "unit_weight_water",
    "spec_grav_ice",
    "dens_ice",
    "vol_heat_cap_water",
    "vol_heat_cap_ice",
    "thrm_cond_water",
    "thrm_cond_ice",
    "latent_heat_fusion_water",
    "Material",
    "NULL_MATERIAL",
]

import numpy as np

"""
grav_acc : float
    The gravitational acceleration in SI units, m * s^{-2}
"""
grav_acc: float = 9.81

"""
dens_water : float
    The density of water in SI units, kg * m^{-3}
"""
dens_water: float = 1e3

"""
unit_weight_water : float
    The unit weight of water in SI units, N * m^{-3}
"""
unit_weight_water: float = grav_acc * dens_water

"""
spec_grav_ice : float
    The specific gravity of ice
"""
spec_grav_ice: float = 0.91

"""
dens_ice : float
    The density of ice in SI units, kg * m^{-3}
"""
dens_ice: float = spec_grav_ice * dens_water

"""
vol_heat_cap_water : float
    The volumetric heat capacity of water in SI units, J.m^{-3}.K^{-1}
"""
vol_heat_cap_water: float = 4.204e6

"""
vol_heat_cap_ice : float
    The volumetric heat capacity of ice in SI units, J.m^{-3}.K^{-1}
"""
vol_heat_cap_ice: float = 1.881e6

"""
thrm_cond_water : float
    The thermal conductivity of water in SI units, W·m{−1}·K{−1}
"""
thrm_cond_water: float = 0.563

"""
thrm_cond_ice : float
    The thermal conductivity of ice in SI units, W·m{−1}·K{−1}
"""
thrm_cond_ice: float = 2.22

"""
latent_heat_fusion_water : float
    The specific latent heat fusion water in SI units, J.kg^{-1}
"""
latent_heat_fusion_water: float = 333.55e3


# private constants (for convenience / efficiency)
_LOG_10: float = np.log(10)


class Material:
    """Class for storing the properties of the solids in porous medium.

    Attributes
    ----------
    thrm_cond_solids
    spec_grav_solids
    dens_solids
    spec_heat_cap_solids
    vol_heat_cap_solids
    deg_sat_water_alpha
    deg_sat_water_beta
    hyd_cond_index
    hyd_cond_mult
    hyd_cond_0
    void_ratio_0_hyd_cond
    void_ratio_min
    void_ratio_sep
    void_ratio_lim
    void_ratio_tr
    water_flux_b1
    water_flux_b2
    water_flux_b3
    temp_rate_ref
    seg_pot_0
    void_ratio_0_comp
    comp_index_unfrozen
    rebound_index_unfrozen
    eff_stress_0_comp
    residual_index
    comp_index_frozen_a1
    comp_index_frozen_a2
    comp_index_frozen_a3

    Methods
    -------
    deg_sat_water
    hyd_cond
    water_flux
    eff_stress
    res_stress
    comp_index_frozen
    tot_stress

    Parameters
    ----------
    thrm_cond_solids : float, optional, default=0.0
        The value to assign to thermal conductivity of solids.
        Cannot be negative.
    spec_grav_solids: float, optional, default=1.0
        The value to assign to specific gravity of solids.
        Cannot be negative.
    spec_heat_cap_solids: float, optional, default=0.0
        The value to assign to specific heat capacity of solids.
        Cannot be negative.
    deg_sat_water_alpha: float, optional, default=1.0
        The value to assign to alpha material constant[kPa]
        for calculation of degree of saturation of water.
        Cannot be negative.
    deg_sat_water_beta: float, optional, default=0.9
        The value to assign to beta material constant[]
        for calculation of degree of saturation of water.
        Cannot be negative.
    hyd_cond_index: float, optional, default=1.0
        The value to assign to hydraulic conductivity index constant
        of unfrozen soil. Cannot be negative.
    hyd_cond_mult: float, optional, default=1.0
        The value to assign to hydraulic conductivity multiplier constant[]
        for adjusting the hydraulic conductivity of
        thawed soil encoutered in freeze-thaw cycle.
        Cannot be negative.
    hyd_cond_0: float, optional, default=0.0
        The value to assign to reference hydraulic conductivity[m/s]
        with unfrozen reference void ratio.
        Cannot be negative.
    void_ratio_0_hyd_cond: float, optional, default=0.0
        The value to assign to reference unfrozen void ratio.
        Cannot be negative.
    void_ratio_min: float, optional, default=0.0
        The value to assign to minimum void ratio for consolidation curves.
        Cannot be negative.
    void_ratio_sep: float, optional, default=0.0
        The value to assign to separation void ratio for consolidation curves.
        Cannot be negative.
    void_ratio_lim: float, optional, default=0.0
        The value to assign to limit void ratio for consolidation curves.
        Cannot be negative.
    void_ratio_tr: float, optional, default=0.0
        The value to assign to thawed rebound void ratio for
        hydraulic conductivity curve. Cannot be negative.
    water_flux_b1: float, optional, default=0.0
        The value to assign to the b1 parameter for the water flux function
        for frozen soil. This value is unitless. Cannot be negative.
    water_flux_b2: float, optional, default=0.0
        The value to assign to the b2 parameter for the water flux function
        for frozen soil. This value has units of(deg C) ^ {-1}.
        Cannot be negative.
    water_flux_b3: float, optional, default=0.0
        The value to assign to the b3 parameter for the water flux function
        for frozen soil. This value has units of(MPa) ^ {-1}.
        Cannot be negative.
    temp_rate_ref: float, optional, default=1.0e-9
        The value to assign to the reference temperature rate
         for the water flux function. This value has units of
         (deg C) / s. Cannot be negative.
    seg_pot_0: float, optional, default=0.0
        The value to assign to the reference segregation potential
        for the water flux function. Cannot be negative.
    void_ratio_0_comp: float, optional, default=0.0
        The value to assign to reference unfrozen void ratio
        corresponding to compression (normal consolidation line).
        Cannot be negative.
    comp_index_unfrozen: float, optional, default=1.0
        The value to assign to compression index in unfrozen soil.
        Cannot be negative.
    rebound_index_unfrozen: float, optional, default=1.0
        The value to assign to rebound index in unfrozen soil.
        Cannot be negative.
    eff_stress_0_comp: float, optional, default=0.0
        The value to assign to effective stress for compression curve.
        Cannot be negative.
    comp_index_frozen_a1: float, optional, default=0.0
        The value to assign to material parameter a1(constant)
        for calculation of frozen compression or rebound index.
        Cannot be negative.
    comp_index_frozen_a2: float, optional, default=0.0
        The value to assign to material parameter a2(constant)
        for calculation of frozen compression or rebound index.
        Cannot be negative.
    comp_index_frozen_a3: float, optional, default=0.0
        The value to assign to material parameter a3(constant)
        for calculation of frozen compression or rebound index.
        SCannot be negative.

    Raises
    ------
    ValueError
        If thrm_cond_solids is not convertible to float.
        If thrm_cond_solids < 0.
        If spec_grav_solids is not convertible to float.
        If spec_grav_solids < 0.
        If spec_heat_cap_solids is not convertible to float.
        If spec_heat_cap_solids < 0.
        If deg_sat_water_alpha is not convertible to float.
        If deg_sat_water_alpha < 0.
        If deg_sat_water_beta is not convertible to float.
        If deg_sat_water_beta < 0.
        If hyd_cond_index is not convertible to float.
        If hyd_cond_index < 0.
        If hyd_cond_mult is not convertible to float.
        If hyd_cond_mult < 0.
        If hyd_cond_0 is not convertible to float.
        If hyd_cond_0 < 0.
        If void_ratio_0_hyd_cond is not convertible to float.
        If void_ratio_0_hyd_cond < 0.
        If void_ratio_min is not convertible to float.
        If void_ratio_min < 0.
        If void_ratio_sep is not convertible to float.
        If void_ratio_sep < 0.
        If void_ratio_lim is not convertible to float.
        If void_ratio_lim < 0.
        If void_ratio_tr is not convertible to float.
        If void_ratio_tr < 0.
        If water_flux_b1 is not convertible to float.
        If water_flux_b1 < 0.
        If water_flux_b2 is not convertible to float.
        If water_flux_b2 < 0.
        If water_flux_b3 is not convertible to float.
        If water_flux_b3 < 0.
        If temp_rate_ref is not convertible to float.
        If temp_rate_ref < 0.
        If seg_pot_0 is not convertible to float.
        If seg_pot_0 < 0.
        If void_ratio_0_comp is not convertible to float.
        If void_ratio_0_comp < 0.
        If comp_index_unfrozen not convertible to float.
        If comp_index_unfrozen < 0.
        If rebound_index_unfrozen is not convertible to float.
        If rebound_index_unfrozen < 0.
        If comp_index_frozen_a1 is not convertible to float.
        If comp_index_frozen_a1 < 0.
        If comp_index_frozen_a2 is not convertible to float.
        If comp_index_frozen_a2 < 0.
        If comp_index_frozen_a3 is not convertible to float.
        If comp_index_frozen_a3 < 0.

    """

    _thrm_cond_solids: float = 0.0
    _spec_grav_solids: float = 1.0
    _dens_solids: float = dens_water
    _spec_heat_cap_solids: float = 0.0
    _deg_sat_water_alpha: float = 1.0
    _deg_sat_water_beta: float = 0.9
    _hyd_cond_index: float = 1.0
    _hyd_cond_mult: float = 1.0
    _hyd_cond_0: float = 0.0
    _void_ratio_0_hyd_cond: float = 0.0
    _void_ratio_min: float = 0.0
    _void_ratio_sep: float = 0.0
    _void_ratio_lim: float = 0.0
    _void_ratio_tr: float = 0.0
    _water_flux_b1: float = 0.0
    _water_flux_b2: float = 0.0
    _water_flux_b3: float = 0.0
    _temp_rate_ref: float = 1.0e-9
    _seg_pot_0: float = 0.0
    _void_ratio_0_comp: float = 0.0
    _comp_index_unfrozen: float = 1.0
    _rebound_index_unfrozen: float = 1.0
    _eff_stress_0_comp: float = 0.0
    _residual_index: float = 0.0
    _comp_index_frozen_a1: float = 0.0
    _comp_index_frozen_a2: float = 0.0
    _comp_index_frozen_a3: float = 0.0

    def __init__(
        self,
        thrm_cond_solids: float = 0.0,
        spec_grav_solids: float = 1.0,
        spec_heat_cap_solids: float = 0.0,
        deg_sat_water_alpha: float = 1.0,
        deg_sat_water_beta: float = 0.9,
        hyd_cond_index: float = 1.0,
        hyd_cond_mult: float = 1.0,
        hyd_cond_0: float = 0.0,
        void_ratio_0_hyd_cond: float = 0.0,
        void_ratio_min: float = 0.0,
        void_ratio_sep: float = 0.0,
        void_ratio_lim: float = 0.0,
        void_ratio_tr: float = 0.0,
        water_flux_b1: float = 0.0,
        water_flux_b2: float = 0.0,
        water_flux_b3: float = 0.0,
        temp_rate_ref: float = 1.0e-9,
        seg_pot_0: float = 0.0,
        void_ratio_0_comp: float = 0.0,
        comp_index_unfrozen: float = 1.0,
        rebound_index_unfrozen: float = 1.0,
        eff_stress_0_comp: float = 0.0,
        comp_index_frozen_a1: float = 0.0,
        comp_index_frozen_a2: float = 0.0,
        comp_index_frozen_a3: float = 0.0,
    ):
        self.thrm_cond_solids = thrm_cond_solids
        self.spec_grav_solids = spec_grav_solids
        self.spec_heat_cap_solids = spec_heat_cap_solids
        self.deg_sat_water_alpha = deg_sat_water_alpha
        self.deg_sat_water_beta = deg_sat_water_beta
        self.hyd_cond_index = hyd_cond_index
        self.hyd_cond_mult = hyd_cond_mult
        self.hyd_cond_0 = hyd_cond_0
        self.void_ratio_0_hyd_cond = void_ratio_0_hyd_cond
        self.void_ratio_min = void_ratio_min
        self.void_ratio_sep = void_ratio_sep
        self.void_ratio_lim = void_ratio_lim
        self.void_ratio_tr = void_ratio_tr
        self.water_flux_b1 = water_flux_b1
        self.water_flux_b2 = water_flux_b2
        self.water_flux_b3 = water_flux_b3
        self.temp_rate_ref = temp_rate_ref
        self.seg_pot_0 = seg_pot_0
        self.void_ratio_0_comp = void_ratio_0_comp
        self.comp_index_unfrozen = comp_index_unfrozen
        self.rebound_index_unfrozen = rebound_index_unfrozen
        self.eff_stress_0_comp = eff_stress_0_comp
        self.comp_index_frozen_a1 = comp_index_frozen_a1
        self.comp_index_frozen_a2 = comp_index_frozen_a2
        self.comp_index_frozen_a3 = comp_index_frozen_a3

    @property
    def thrm_cond_solids(self) -> float:
        """Thermal conductivity of solids.

        Parameters
        ----------
        value: float
            Value to assign to the thermal conductivity of solids.

        Returns
        -------
        float
            Current value of thermal conductivity of solids.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._thrm_cond_solids

    @thrm_cond_solids.setter
    def thrm_cond_solids(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"thrm_cond_solids {value} is not positive")
        self._thrm_cond_solids = value

    @property
    def dens_solids(self) -> float:
        """Density of solids.

        Returns
        -------
        float
            Current value of density of solids.
        """
        return self._dens_solids

    @property
    def spec_grav_solids(self) -> float:
        """Specific gravity of solids.

        Parameters
        ----------
        value: float
            Value to assign to the specific gravity of solids.

        Returns
        -------
        float
            Current value of specific gravity of solids.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._spec_grav_solids

    @spec_grav_solids.setter
    def spec_grav_solids(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"spec_grav_solids {value} is not positive")
        self._spec_grav_solids = value
        self._dens_solids = value * dens_water
        self._update_vol_heat_cap_solids()

    @property
    def spec_heat_cap_solids(self) -> float:
        """Specific heat capacity of solids.

        Parameters
        ----------
        value: float
            Value to assign to the specific heat capacity of solids.

        Returns
        -------
        float
            Current value of specific heat capacity of solids.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._spec_heat_cap_solids

    @spec_heat_cap_solids.setter
    def spec_heat_cap_solids(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"spec_heat_cap_solids {value} is not positive")
        self._spec_heat_cap_solids = value
        self._update_vol_heat_cap_solids()

    @property
    def vol_heat_cap_solids(self) -> float:
        """Volumetric heat capacity of solids.

        Returns
        -------
        float
            Current value of volumetric heat capacity of solids.

        Notes
        -----
        This property cannot be set. It is calculated from density of solids
        and specific heat capacity of solids.
        """
        return self._vol_heat_cap_solids

    def _update_vol_heat_cap_solids(self) -> None:
        self._vol_heat_cap_solids = self.dens_solids * self.spec_heat_cap_solids

    @property
    def deg_sat_water_alpha(self) -> float:
        """Alpha material constant[kPa]
        for calculation of degree of saturation of water.

        Parameters
        ----------
        value: float
            Value to assign to the alpha constant.

        Returns
        -------
        float
            Current value of alpha.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.

        Notes
        -----
        The default values of alpha=1.0 and beta=0.9
        give step function behaviour.
        """
        return self._deg_sat_water_alpha

    @deg_sat_water_alpha.setter
    def deg_sat_water_alpha(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"deg_sat_water_alpha {value} is not positive")
        self._deg_sat_water_alpha = value

    @property
    def deg_sat_water_beta(self) -> float:
        """Beta material constant[]
        for calculation of degree of saturation of water.

        Parameters
        ----------
        value: float
            Value to assign to the beta constant.

        Returns
        -------
        float
            Current value of beta.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.

        Notes
        -----
        The default values of alpha=1.0 and beta=0.9
        give step function behaviour.
        """
        return self._deg_sat_water_beta

    @deg_sat_water_beta.setter
    def deg_sat_water_beta(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"deg_sat_water_beta {value} is not positive")
        self._deg_sat_water_beta = value

    @property
    def hyd_cond_index(self) -> float:
        """Hydraulic conductivity index constant of unfrozen soil.

        Parameters
        ----------
        value: float
            Value to assign to the hydraulic conductivity index.

        Returns
        -------
        float
            Current value of hydraulic conductivity index.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._hyd_cond_index

    @hyd_cond_index.setter
    def hyd_cond_index(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"hyd_cond_index {value} is not positive")
        self._hyd_cond_index = value

    @property
    def hyd_cond_mult(self) -> float:
        """Hydraulic conductivity multiplier constant[]
        for adjusting the hydraulic conductivity of
        thawed soil encoutered in freeze-thaw cycle.

        Parameters
        ----------
        value: float
            Value to assign to the hydraulic conductivity multiplier.

        Returns
        -------
        float
            Current value of hydraulic conductivity multiplier.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._hyd_cond_mult

    @hyd_cond_mult.setter
    def hyd_cond_mult(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"hyd_cond_mult {value} is not positive")
        self._hyd_cond_mult = value

    @property
    def hyd_cond_0(self) -> float:
        """Reference hydraulic conductivity[m/s]
        with unfrozen reference void ratio.

        Parameters
        ----------
        value: float
            Value to assign to the reference hydraulic conductivity.

        Returns
        -------
        float
            Current value of reference hydraulic conductivity.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._hyd_cond_0

    @hyd_cond_0.setter
    def hyd_cond_0(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"hyd_cond_0 {value} is not positive")
        self._hyd_cond_0 = value

    @property
    def void_ratio_0_hyd_cond(self) -> float:
        """Reference unfrozen void ratio.

        Parameters
        ----------
        value: float or int or str
            Value to assign to the reference unfrozen void ratio.

        Returns
        -------
        float
            Current value of reference unfrozen void ratio.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._void_ratio_0_hyd_cond

    @void_ratio_0_hyd_cond.setter
    def void_ratio_0_hyd_cond(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_0_hyd_cond {value} is not positive")
        self._void_ratio_0_hyd_cond = value

    @property
    def void_ratio_min(self) -> float:
        """Minimum void ratio for consolidation curves.

        Parameters
        ----------
        value: float
            Value to assign to the minimum void ratio.

        Returns
        -------
        float
            Current value of the minimum void ratio.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._void_ratio_min

    @void_ratio_min.setter
    def void_ratio_min(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_min {value} is not positive")
        self._void_ratio_min = value

    @property
    def void_ratio_sep(self) -> float:
        """Separation void ratio for consolidation curves.

        Parameters
        ----------
        value: float
            Value to assign to the separation void ratio.

        Returns
        -------
        float
            Current value of the separation void ratio.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._void_ratio_sep

    @void_ratio_sep.setter
    def void_ratio_sep(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_sep {value} is not positive")
        self._void_ratio_sep = value

    @property
    def void_ratio_lim(self) -> float:
        """Limit void ratio for consolidation curves.

        Parameters
        ----------
        value: float
            Value to assign to the limit void ratio.

        Returns
        -------
        float
            Current value of the limit void ratio.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._void_ratio_lim

    @void_ratio_lim.setter
    def void_ratio_lim(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_lim {value} is not positive")
        self._void_ratio_lim = value

    @property
    def void_ratio_tr(self) -> float:
        """Thawed rebound void ratio for hydraulic conductivity curve.

        Parameters
        ----------
        value: float
            Value to assign to the thawed rebound void ratio.

        Returns
        -------
        float
            Current value of the thawed rebound void ratio.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._void_ratio_tr

    @void_ratio_tr.setter
    def void_ratio_tr(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_tr {value} is not positive")
        self._void_ratio_tr = value

    @property
    def water_flux_b1(self) -> float:
        """The b1 parameter for the water flux function for frozen soil.
        This value is unitless.

        Parameters
        ----------
        value: float
            Value to assign to the b1 parameter.

        Returns
        -------
        float
            Current value of the b1 parameter.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._water_flux_b1

    @water_flux_b1.setter
    def water_flux_b1(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"water_flux_b1 {value} is not positive")
        self._water_flux_b1 = value

    @property
    def water_flux_b2(self) -> float:
        """The b2 parameter for the water flux function for frozen soil.
        This value has units of(deg C) ^ {-1}.

        Parameters
        ----------
        value: float
            Value to assign to the b2 parameter.

        Returns
        -------
        float
            Current value of the b2 parameter.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._water_flux_b2

    @water_flux_b2.setter
    def water_flux_b2(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"water_flux_b2 {value} is not positive")
        self._water_flux_b2 = value

    @property
    def water_flux_b3(self) -> float:
        """The b3 parameter for the water flux function for frozen soil.
        This value has units of(MPa) ^ {-1}.

        Parameters
        ----------
        value: float
            Value to assign to the b3 parameter.

        Returns
        -------
        float
            Current value of the b3 parameter.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._water_flux_b3

    @water_flux_b3.setter
    def water_flux_b3(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"water_flux_b3 {value} is not positive")
        self._water_flux_b3 = value

    @property
    def temp_rate_ref(self) -> float:
        """The reference temperature rate for the water flux function.

        Parameters
        ----------
        value: float
            Value to assign to the reference temperature rate.

        Returns
        -------
        float
            Current value of the reference temperature rate.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._temp_rate_ref

    @temp_rate_ref.setter
    def temp_rate_ref(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"temp_rate_ref {value} is not positive")
        self._temp_rate_ref = value

    @property
    def seg_pot_0(self) -> float:
        """The reference segregation potential for the water flux function.

        Parameters
        ----------
        value: float
            Value to assign to the reference segregation potential.

        Returns
        -------
        float
            Current value of the reference segregation potential.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._seg_pot_0

    @seg_pot_0.setter
    def seg_pot_0(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"seg_pot_0 {value} is not positive")
        self._seg_pot_0 = value

    @property
    def void_ratio_0_comp(self) -> float:
        """Reference unfrozen void ratio
        corresponding to compression
        (normal consolidation line).

        Parameters
        ----------
        value: float
            Value to assign to the reference unfrozen compression void ratio.

        Returns
        -------
        float
            Current value of reference unfrozen compression void ratio.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._void_ratio_0_comp

    @void_ratio_0_comp.setter
    def void_ratio_0_comp(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_0_comp {value} is not positive")
        self._void_ratio_0_comp = value

    @property
    def eff_stress_0_comp(self) -> float:
        """Effective stress for compression curve.

        Parameters
        ----------
        value: float
            Value to assign to the compression effective stress.

        Returns
        -------
        float
            Current value of the compression effective stress.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._eff_stress_0_comp

    @eff_stress_0_comp.setter
    def eff_stress_0_comp(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"eff_stress_0_comp {value} is not positive")
        self._eff_stress_0_comp = value

    @property
    def comp_index_unfrozen(self) -> float:
        """Compression index in unfrozen soil.

        Parameters
        ----------
        value: float
            Value to assign to the compression index.

        Returns
        -------
        float
            Current value of the compression index.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._comp_index_unfrozen

    @comp_index_unfrozen.setter
    def comp_index_unfrozen(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"comp_index_unfrozen {value} is not positive")
        self._comp_index_unfrozen = value

    @property
    def rebound_index_unfrozen(self) -> float:
        """Rebound index in unfrozen soil.

        Parameters
        ----------
        value: float
            Value to assign to the rebound index.

        Returns
        -------
        float
            Current value of the rebound index.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._rebound_index_unfrozen

    @rebound_index_unfrozen.setter
    def rebound_index_unfrozen(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"rebound_index_unfrozen {value} is not positive")
        self._rebound_index_unfrozen = value

    @property
    def residual_index(self) -> float:
        """Residual stress line index. Slope in the e-log p' space.

        Returns
        -------
        float
            Value of the residual stress index.
        """
        return self._residual_index

    @property
    def comp_index_frozen_a1(self) -> float:
        """Material parameter a1(constant)
           for calculation of frozen compression or rebound index.

        Parameters
        ----------
        value: float
            Value to assign to the material parameter a1

        Returns
        -------
        float
            Current value of the material parameter a1.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._comp_index_frozen_a1

    @comp_index_frozen_a1.setter
    def comp_index_frozen_a1(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"comp_index_frozen_a1 {value} is not positive")
        self._comp_index_frozen_a1 = value

    @property
    def comp_index_frozen_a2(self) -> float:
        """Material parameter a2(constant)
           for calculation of frozen compression or rebound index.

        Parameters
        ----------
        value: float
            Value to assign to the material parameter a2

        Returns
        -------
        float
            Current value of the material parameter a2.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._comp_index_frozen_a2

    @comp_index_frozen_a2.setter
    def comp_index_frozen_a2(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"comp_index_frozen_a2 {value} is not positive")
        self._comp_index_frozen_a2 = value

    @property
    def comp_index_frozen_a3(self) -> float:
        """Material parameter a3(constant)
           for calculation of frozen compression or rebound index.

        Parameters
        ----------
        value: float
            Value to assign to the material parameter a3

        Returns
        -------
        float
            Current value of the material parameter a3.

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.
        """
        return self._comp_index_frozen_a3

    @comp_index_frozen_a3.setter
    def comp_index_frozen_a3(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"comp_index_frozen_a3 {value} is not positive")
        self._comp_index_frozen_a3 = value

    def deg_sat_water(self, temp: float) -> tuple[float, float]:
        """The degree of saturation of water function.

        Parameters
        ----------
        temp: float
            Current temperature.

        Returns
        -------
        float
            The degree of saturation of water.
        float
            The derivative of degree of saturation of water
            with respect to temperature.
        """
        deg_sat_water = 1.0
        deg_sat_deriv = 0.0
        if temp > 0.0:
            return deg_sat_water, deg_sat_deriv
        rho_i = spec_grav_ice * dens_water
        temp_kelvin = temp + 273.15
        log_temp_ratio = np.log(temp_kelvin / 273.15)
        alpha = self.deg_sat_water_alpha
        beta = self.deg_sat_water_beta
        latent_heat_ratio = -latent_heat_fusion_water * rho_i / alpha
        beta_ratio_0 = 1.0 / (1.0 - beta)
        beta_ratio_1 = beta * beta_ratio_0
        beta_ratio_2 = (1 + beta) / beta
        deg_sat_base = (latent_heat_ratio * log_temp_ratio) ** beta_ratio_0
        deg_sat_water = (1.0 + deg_sat_base) ** (-beta)
        deg_sat_deriv = -beta_ratio_1 * latent_heat_ratio / temp_kelvin
        deg_sat_deriv *= (deg_sat_water**beta_ratio_2) * (deg_sat_base**beta)
        return deg_sat_water, deg_sat_deriv

    def hyd_cond(self, e: float, temp: float, thawed: bool) -> tuple[float, float]:
        """The hydraulic conductivity for unfrozen and thawed soil.

        Parameters
        ----------
        e: float
            Current void ratio.
        temp: float
            Current temperature.
        thawed: bool
            Flag for whether soil is thawed or unfrozen.

        Returns
        -------
        float
            The hydraulic conductivity.
        float
            The gradient of hydraulic conductivity
            with respect to void raito.

        Raises
        ------
        ValueError
            If temp < 0.0.
        """
        if temp < 0.0:
            raise ValueError(f"temp {temp} is negative.")
        if e < self.void_ratio_lim:
            raise ValueError(f"e {e} less than e_lim {self.void_ratio_lim}")
        eu0 = self.void_ratio_0_hyd_cond
        e_min = self.void_ratio_min
        e_tr = self.void_ratio_tr
        e_sep = self.void_ratio_sep
        m = 1.0
        if thawed and e >= e_min and e <= e_tr:
            m = self.hyd_cond_mult
        k0 = m * self.hyd_cond_0
        C_ku = self.hyd_cond_index
        if e > e_sep:
            k = k0 * 10 ** ((e_sep - eu0) / C_ku)
            dk_de = 0.0
        else:
            k = k0 * 10 ** ((e - eu0) / C_ku)
            dk_de = k * _LOG_10 / C_ku
        return k, dk_de

    def water_flux(
        self,
        e: float,
        e0: float,
        temp: float,
        temp_rate: float,
        temp_grad: float,
        sigma_1: float,
    ) -> float:
        """The water flux function for frozen soil.

        Parameters
        ----------
        e: float
            Current void ratio.
        e0: float
            Initial void ratio.
        temp: float
            Current temperature.
        temp_rate: float
            Current temperature time derivative.
        temp_grad: float
            Current temperature gradient (in Lagrangian coordinates).
        sigma_1: float
            Current local stress(overburden and void ratio correction).

        Returns
        -------
        float
            The water flux rate.

        Raises
        ------
        ValueError
            If the given temp >= 0.0 since this only applies for frozen soil.
        """
        if temp >= 0.0:
            raise ValueError(f"temp {temp} is >= Tf = 0.0")
        void_ratio_factor = (1.0 + e0) / (1.0 + e)
        temp_rate_ratio = np.abs(temp_rate / self.temp_rate_ref)
        temp_rate_factor = 1.0
        if temp_rate < 0.0:
            temp_rate_factor += self.water_flux_b1 * np.log(temp_rate_ratio)
        elif temp_rate > 0.0:
            temp_rate_factor -= self.water_flux_b1 * np.log(temp_rate_ratio)
        exp_factor = np.exp(
            self.water_flux_b2 * (temp - 0.0) - self.water_flux_b3 * sigma_1
        )
        water_flux = (
            -void_ratio_factor
            * temp_rate_factor
            * self.seg_pot_0
            * exp_factor
            * temp_grad
        )
        return water_flux

    def eff_stress(self, e: float, ppc: float) -> tuple[float, float]:
        """Calculate effective stress and
        gradient with respect to void ratio.

        Parameters
        ----------
        e: float
            Current void ratio.
        ppc: float
            Preconsolidation stress.

        Returns
        -------
        float
            Effective stress at the specified void ratio.
        float
            Gradient of effective stress
            with respect to void ratio.
        """
        # get normal consolidation line (NCL) parameters
        sig_cu0 = self.eff_stress_0_comp
        e_cu0 = self.void_ratio_0_comp
        Ccu = self.comp_index_unfrozen
        # check for separation
        if e > self.void_ratio_sep:
            sig_p = 1.0
            dsig_de = -0.1 * sig_p * _LOG_10 / Ccu
            return sig_p, dsig_de
        # check if current void ratio implies stress above preconsolidation
        sig_p_ncl = sig_cu0 * 10 ** ((e_cu0 - e) / Ccu)
        if sig_p_ncl >= ppc:
            # here, we are on the NCL, so calculate gradient and return
            dsig_de = -sig_p_ncl * _LOG_10 / Ccu
            return sig_p_ncl, dsig_de
        # here, we are on the unloading-reloading line (URL)
        # calculate parameters, effective stress, and gradient
        e_ru0 = e_cu0 - Ccu * np.log10(ppc / sig_cu0)
        Cru = self.rebound_index_unfrozen
        sig_p = ppc * 10 ** ((e_ru0 - e) / Cru)
        dsig_de = -sig_p * _LOG_10 / Cru
        return sig_p, dsig_de

    def res_stress(self, e: float) -> float:
        """Calculate post-thaw modified pre-consolidation stress
        on the normal consolidation line (NCL)
        consistent with the residual stress line (RSL).

        Inputs
        ------
        e : float
            The initial post-thaw void ratio.

        Returns
        -------
        float
            The modified pre-consolidation stress
            on the NCL for the given void ratio
            and corresponding residual stress.
        """
        # get consolidation curve parameters
        e_sep = self.void_ratio_sep
        e_cu0 = self.void_ratio_0_comp
        sig_cu0 = self.eff_stress_0_comp
        Ccu = self.comp_index_unfrozen
        Cru = self.rebound_index_unfrozen
        # check for separation
        # (if void ratio too large, set URL based on e_sep)
        if e > e_sep:
            e = e_sep
        # check for initialization of residual stress line index
        if not self.residual_index:
            e_min = self.void_ratio_min
            log_sig_max = np.log10(sig_cu0) + (e_cu0 - e_min) / Ccu
            self._residual_index = (e_sep - e_min) / log_sig_max
        # compute residual stress on the RSL
        Crsl = self.residual_index
        sig_p = 10 ** ((e_sep - e) / Crsl)
        # compute corrected pre-consolidation stress on the NCL
        ppc = 10 ** (
            ((e_cu0 - e) + (Ccu * np.log10(sig_cu0) - Cru * np.log10(sig_p)))
            / (Ccu - Cru)
        )
        return ppc

    def comp_index_frozen(self, temp: float) -> float:
        """Compression and rebound index
        for frozen soil.

        Parameters
        ----------
        temp: float
            The current temperature.

        Returns
        -------
        float
            The compression / rebound index of frozen soil.

        Raises
        ------
        ValueError
            If the given temp >= 0.0 since this only applies for frozen soil.
        """
        if temp >= 0.0:
            raise ValueError(f"temp {temp} is >= Tf = 0.0")
        return (
            self.comp_index_frozen_a1
            - self.comp_index_frozen_a2 * np.abs(temp) ** self.comp_index_frozen_a3
        )

    def tot_stress(
        self, temp: float, e: float, e_f0: float, sig_f0: float
    ) -> tuple[float, float]:
        """Calculate the total stress in frozen soil.

        Parameters
        ----------
        temp: float
            Current temperature.
        e: float
            Current void ratio.
        e_f0: float
            Reference void ratio for frozen soil.
        sig_f0: float
            Reference total stress for frozen soil.

        Returns
        -------
        float
            The total stress at the point.
        float
            The total stress gradient
            with respect to void ratio.

        Raises
        ------
        ValueError
            If temp >= 0.0.
        """
        if temp >= 0.0:
            raise ValueError(f"temp {temp} must be negative.")
        Cf = self.comp_index_frozen(temp)
        if Cf:
            sig = sig_f0 * 10 ** ((e_f0 - e) / Cf)
            dsig_de = -sig * _LOG_10 / Cf
        else:
            sig = sig_f0
            dsig_de = 0.0
        return sig, dsig_de


"""An instance of the material class with all parameters set to zero.
"""
NULL_MATERIAL = Material()
