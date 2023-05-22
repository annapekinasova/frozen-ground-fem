"""frozen_ground_fem.materials.py - A module for material constants and classes
for tracking material properties.

"""
import numpy as np

"""
grav_acc : float
    The gravitational acceleration in SI units, m * s^{-2}
"""
grav_acc = 9.81

"""
dens_water : float
    The density of water in SI units, kg * m^{-3}
"""
dens_water = 1e3

"""
unit_weight_water : float
    The unit weight of water in SI units, N * m^{-3}
"""
unit_weight_water = grav_acc * dens_water

"""
spec_grav_ice : float
    The specific gravity of ice
"""
spec_grav_ice = 0.91

"""
vol_heat_cap_water : float
    The volumetric heat capacity of water in SI units, J.m^{-3}.K^{-1}
"""
vol_heat_cap_water = 4.204e6

"""
vol_heat_cap_ice : float
    The volumetric heat capacity of ice in SI units, J.m^{-3}.K^{-1}
"""
vol_heat_cap_ice = 1.881e6

"""
thrm_cond_water : float
    The thermal conductivity of water in SI units, J.m^{-3}.K^{-1}
"""
thrm_cond_water = 0.563

"""
thrm_cond_ice : float
    The thermal conductivity of ice in SI units, J.m^{-3}.K^{-1}
"""
thrm_cond_ice = 2.22

"""
latent_heat_fusion_water : float
    The specific latent heat fusion water in SI units, J.kg^{-1}
"""
latent_heat_fusion_water = 333.55e3


class Material:
    """Class for storing the properties of the solids in porous medium.

    Attributes
    ----------
    thrm_cond_solids
    dens_solids
    spec_heat_cap_solids
    vol_heat_cap_solids
    """

    def __init__(
        self,
        thrm_cond_solids=0.0,
        spec_grav_solids=0.0,
        spec_heat_cap_solids=0.0,
        deg_sat_water_alpha=0.0,
        deg_sat_water_beta=0.0,
        hyd_cond_index=0.0,
        hyd_cond_mult=0.0,
        hyd_cond_0=0.0,
        void_ratio_min=0.0,
        void_ratio_sep=0.0,
        void_ratio_lim=0.0,
        void_ratio_tr=0.0,
        water_flux_b1=0.0,
        water_flux_b2=0.0,
        water_flux_b3=0.0,
        temp_rate_ref=0.0,
        seg_pot_0=0.0,
    ):
        self._thrm_cond_solids = 0.0
        self._spec_grav_solids = 0.0
        self._dens_solids = 0.0
        self._spec_heat_cap_solids = 0.0
        self._deg_sat_water_alpha = 0.0
        self._deg_sat_water_beta = 0.0
        self._hyd_cond_index = 0.0
        self._hyd_cond_mult = 0.0
        self._hyd_cond_0 = 0.0
        self._void_ratio_min = 0.0
        self._void_ratio_sep = 0.0
        self._void_ratio_lim = 0.0
        self._void_ratio_tr = 0.0
        self._water_flux_b1 = 0.0
        self._water_flux_b2 = 0.0
        self._water_flux_b3 = 0.0
        self._temp_rate_ref = 0.0
        self._seg_pot_0 = 0.0
        self.thrm_cond_solids = thrm_cond_solids
        self.spec_grav_solids = spec_grav_solids
        self.spec_heat_cap_solids = spec_heat_cap_solids
        self.deg_sat_water_alpha = deg_sat_water_alpha
        self.deg_sat_water_beta = deg_sat_water_beta
        self.hyd_cond_index = hyd_cond_index
        self.hyd_cond_mult = hyd_cond_mult
        self.hyd_cond_0 = hyd_cond_0
        self.void_ratio_min = void_ratio_min
        self.void_ratio_sep = void_ratio_sep
        self.void_ratio_lim = void_ratio_lim
        self.void_ratio_tr = void_ratio_tr
        self.water_flux_b1 = water_flux_b1
        self.water_flux_b2 = water_flux_b2
        self.water_flux_b3 = water_flux_b3
        self.temp_rate_ref = temp_rate_ref
        self.seg_pot_0 = seg_pot_0

    @property
    def thrm_cond_solids(self):
        """Thermal conductivity of solids.

        Parameters
        ----------
        value : float or int or str
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
    def thrm_cond_solids(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"thrm_cond_solids {value} is not positive")
        self._thrm_cond_solids = value

    @property
    def dens_solids(self):
        """Density of solids.

        Returns
        -------
        float
            Current value of density of solids.
        """
        return self._dens_solids

    @property
    def spec_grav_solids(self):
        """Specific gravity of solids.

        Parameters
        ----------
        value : float or int or str
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
    def spec_grav_solids(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"spec_grav_solids {value} is not positive")
        self._spec_grav_solids = value
        self._dens_solids = value * dens_water
        self._update_vol_heat_cap_solids()

    @property
    def spec_heat_cap_solids(self):
        """Specific heat capacity of solids.

        Parameters
        ----------
        value : float or int or str
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
    def spec_heat_cap_solids(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"spec_heat_cap_solids {value} is not positive")
        self._spec_heat_cap_solids = value
        self._update_vol_heat_cap_solids()

    @property
    def vol_heat_cap_solids(self):
        """Specific heat capacity of solids.

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

    def _update_vol_heat_cap_solids(self):
        self._vol_heat_cap_solids = self.dens_solids * self.spec_heat_cap_solids

    @property
    def deg_sat_water_alpha(self):
        """Alpha material constant [kPa]
        for calculation of degree of saturation of water.

        Parameters
        ----------
        value : float or int or str
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
        """
        return self._deg_sat_water_alpha

    @deg_sat_water_alpha.setter
    def deg_sat_water_alpha(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"deg_sat_water_alpha {value} is not positive")
        self._deg_sat_water_alpha = value

    @property
    def deg_sat_water_beta(self):
        """Beta material constant []
        for calculation of degree of saturation of water.

        Parameters
        ----------
        value : float or int or str
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
        """
        return self._deg_sat_water_beta

    @deg_sat_water_beta.setter
    def deg_sat_water_beta(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"deg_sat_water_beta {value} is not positive")
        self._deg_sat_water_beta = value

    @property
    def hyd_cond_index(self):
        """Hydraulic conductivity index constant of unfrozen soil.

        Parameters
        ----------
        value : float or int or str
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
    def hyd_cond_index(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"hyd_cond_index {value} is not positive")
        self._hyd_cond_index = value

    @property
    def hyd_cond_mult(self):
        """Hydraulic conductivity multiplier constant []
        for adjusting the hydraulic conductivity of
        thawed soil encoutered in freeze-thaw cycle.

        Parameters
        ----------
        value : float or int or str
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
    def hyd_cond_mult(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"hyd_cond_mult {value} is not positive")
        self._hyd_cond_mult = value

    @property
    def hyd_cond_0(self):
        """Reference hydraulic conductivity [m/s]
        with unfrozen reference void ratio.

        Parameters
        ----------
        value : float or int or str
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
    def hyd_cond_0(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"hyd_cond_0 {value} is not positive")
        self._hyd_cond_0 = value

    @property
    def void_ratio_min(self):
        """Minimum void ratio for consolidation curves.

        Parameters
        ----------
        value : float or int or str
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
    def void_ratio_min(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_min {value} is not positive")
        self._void_ratio_min = value

    @property
    def void_ratio_sep(self):
        """Separation void ratio for consolidation curves.

        Parameters
        ----------
        value : float or int or str
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
    def void_ratio_sep(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_sep {value} is not positive")
        self._void_ratio_sep = value

    @property
    def void_ratio_lim(self):
        """Limit void ratio for consolidation curves.

        Parameters
        ----------
        value : float or int or str
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
    def void_ratio_lim(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_lim {value} is not positive")
        self._void_ratio_lim = value

    @property
    def void_ratio_tr(self):
        """Thawed rebound void ratio for hydraulic conductivity curve.

        Parameters
        ----------
        value : float or int or str
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
    def void_ratio_tr(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_tr {value} is not positive")
        self._void_ratio_tr = value

    @property
    def water_flux_b1(self):
        """The b1 parameter for the water flux function for frozen soil.
        This value is unitless.

        Parameters
        ----------
        value : float or int or str
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
    def water_flux_b1(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"water_flux_b1 {value} is not positive")
        self._water_flux_b1 = value

    @property
    def water_flux_b2(self):
        """The b2 parameter for the water flux function for frozen soil.
        This value has units of (deg C)^{-1}.

        Parameters
        ----------
        value : float or int or str
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
    def water_flux_b2(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"water_flux_b2 {value} is not positive")
        self._water_flux_b2 = value

    @property
    def water_flux_b3(self):
        """The b3 parameter for the water flux function for frozen soil.
        This value has units of (MPa)^{-1}.

        Parameters
        ----------
        value : float or int or str
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
    def water_flux_b3(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"water_flux_b3 {value} is not positive")
        self._water_flux_b3 = value

    @property
    def temp_rate_ref(self):
        """The reference temperature rate for the water flux function.

        Parameters
        ----------
        value : float or int or str
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
    def temp_rate_ref(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"temp_rate_ref {value} is not positive")
        self._temp_rate_ref = value

    @property
    def seg_pot_0(self):
        """The reference segregation potential for the water flux function.

        Parameters
        ----------
        value : float or int or str
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
    def seg_pot_0(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"seg_pot_0 {value} is not positive")
        self._seg_pot_0 = value

    def deg_sat_water(self, temp):
        """The degree of saturation of water function.

        Parameters
        ----------
        temp : float
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

    def water_flux(self, e, e0, temp, temp_rate, temp_grad, sigma_1):
        """The water flux function for frozen soil.

        Parameters
        ----------
        e : float
            Current void ratio.
        e0 : float
            Initial void ratio.
        temp : float
            Current temperature.
        temp_rate : float
            Current temperature time derivative.
        temp_grad : float
            Current temperature gradient (in Lagrangian coordinates).
        sigma_1 : float
            Current local stress (overburden and void ratio correction).

        Returns
        -------
        float
            The water flux rate.

        Raises
        ------
        ValueError
            If the given temp > 0.0 since this only applies for frozen soil.
        """
        if temp > 0.0:
            raise ValueError(f"temp {temp} is above Tf = 0.0")
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

    # TODO: update this method for nonlinear large strain
    # currently it just returns the compression index,
    # which can be used like the modulus parameter in
    # Terzaghi consolidation
    def grad_sig_void_ratio(self, void_ratio, pre_consol_stress):
        raise NotImplementedError()


"""An instance of the material class with all parameters set to zero.
"""
NULL_MATERIAL = Material()
