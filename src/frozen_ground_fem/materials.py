"""frozen_ground_fem.materials.py - A module for material constants and classes
for tracking material properties.

"""

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
        self, thrm_cond_solids=0.0, spec_grav_solids=0.0, spec_heat_cap_solids=0.0
    ):
        self._thrm_cond_solids = 0.0
        self._spec_grav_solids = 0.0
        self._dens_solids = 0.0
        self._spec_heat_cap_solids = 0.0
        self.thrm_cond_solids = thrm_cond_solids
        self.spec_grav_solids = spec_grav_solids
        self.spec_heat_cap_solids = spec_heat_cap_solids

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


"""An instance of the material class with all parameters set to zero.
"""
NULL_MATERIAL = Material()
