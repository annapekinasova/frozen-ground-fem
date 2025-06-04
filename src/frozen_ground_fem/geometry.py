"""frozen_ground_fem.geometry.py - A module for classes
for finite element model geometry.

Functions
---------
shape_matrix_linear
shape_matrix_cubic
gradient_matrix_linear
gradient_matrix_cubic

Classes
-------
Point1D
Node1D
IntegrationPoint1D
Element1D
Boundary1D
Mesh1D
"""

__all__ = [
    "shape_matrix_linear",
    "shape_matrix_cubic",
    "gradient_matrix_linear",
    "gradient_matrix_cubic",
    "Point1D",
    "Node1D",
    "IntegrationPoint1D",
    "Boundary1D",
    "Element1D",
    "Mesh1D",
]

from typing import (
    Callable,
    ClassVar,
    Sequence,
    Any,
)

import numpy as np
import numpy.typing as npt

from . import (
    Material,
    thrm_cond_ice as lam_i,
    thrm_cond_water as lam_w,
    vol_heat_cap_ice as C_i,
    vol_heat_cap_water as C_w,
)
from .materials import (
    NULL_MATERIAL,
)


def shape_matrix_linear(s: float) -> npt.NDArray[np.floating]:
    """Calculates the shape (interpolation) function matrix
    for linear interpolation.

    Parameters
    ----------
    s : float
        The local coordinate. Should be between 0.0 and 1.0.

    Returns
    -------
    numpy.ndarray
        The shape function matrix.

    Raises
    ------
    ValueError
        If s cannot be converted to float.

    Notes
    -----
    Assumes linear interpolation of a single variable between two nodes.
    The resulting shape matrix N is:

        N = [[(1 - s), s]]
    """
    s = float(s)
    return np.array([[(1.0 - s), s]])


def shape_matrix_cubic(s: float) -> npt.NDArray[np.floating]:
    """Calculates the shape (interpolation) function matrix
    for cubic interpolation.

    Parameters
    ----------
    s : float
        The local coordinate. Should be between 0.0 and 1.0.

    Returns
    -------
    numpy.ndarray
        The shape function matrix.

    Raises
    ------
    ValueError
        If s cannot be converted to float.

    Notes
    -----
    Assumes cubic interpolation of a single variable between two nodes.
    The resulting shape matrix N is:

        N = [[-0.5 * ( 9 * s**3 - 18 * s**2 + 11 * s - 2),
               0.5 * (27 * s**3 - 45 * s**2 + 18 * s),
              -0.5 * (27 * s**3 - 36 * s**2 +  9 * s),
               0.5 * ( 9 * s**3 -  9 * s**2 +  2 * s)]]
    """
    s = float(s)
    s3 = s**3
    s2 = s**2
    return np.array(
        [
            [
                -0.5 * (9 * s3 - 18 * s2 + 11 * s - 2),
                0.5 * (27 * s3 - 45 * s2 + 18 * s),
                -0.5 * (27 * s3 - 36 * s2 + 9 * s),
                0.5 * (9 * s3 - 9 * s2 + 2 * s),
            ]
        ]
    )


def gradient_matrix_linear(s: float, dz: float) -> npt.NDArray[np.floating]:
    """Calculates the gradient of the shape (interpolation) function matrix
    for linear interpolation.

    Parameters
    ----------
    s : float
        The local coordinate. Should be between 0.0 and 1.0.
    dz : float
        The element scale parameter (Jacobian).

    Returns
    -------
    numpy.ndarray
        The gradient of the shape function matrix.

    Raises
    ------
    ValueError
        If s cannot be converted to float.
        If dz cannot be converted to float.

    Notes
    -----
    Assumes linear interpolation of a single variable between two nodes.
    The resulting gradient matrix B is:

        B = [[-1 , 1]] / dz
    """
    s = float(s)
    dz = float(dz)
    return np.array([[-1.0, 1.0]]) / dz


def gradient_matrix_cubic(s: float, dz: float) -> npt.NDArray[np.floating]:
    """Calculates the gradient of the shape (interpolation) function matrix
    for cubic interpolation.

    Parameters
    ----------
    s : float
        The local coordinate. Should be between 0.0 and 1.0.
    dz : float
        The element scale parameter (Jacobian).

    Returns
    -------
    numpy.ndarray
        The gradient of the shape function matrix.

    Raises
    ------
    ValueError
        If s cannot be converted to float.
        If dz cannot be converted to float.

    Notes
    -----
    Assumes cubic interpolation of a single variable between two nodes.
    The resulting gradient matrix B is:

        B = [[-0.5 * (27 * s**2 - 36 * s + 11),
               0.5 * (81 * s**2 - 90 * s + 18),
              -0.5 * (81 * s**2 - 72 * s +  9),
               0.5 * (27 * s**2 - 18 * s +  2)]] / dz
    """
    s = float(s)
    dz = float(dz)
    s2 = s**2
    return (
        np.array(
            [
                [
                    -0.5 * (27 * s2 - 36 * s + 11),
                    0.5 * (81 * s2 - 90 * s + 18),
                    -0.5 * (81 * s2 - 72 * s + 9),
                    0.5 * (27 * s2 - 18 * s + 2),
                ]
            ]
        )
        / dz
    )


class Point1D:
    """Class for storing the coordinates of a point.

    Attributes
    ----------
    coords
    z

    Parameters
    ----------
    value : float, optional, default=0.0
        The value to assign to the coordinate of the point.

    Raises
    ------
    ValueError
        If value cannot be converted to float.
    """

    _coords: npt.NDArray[np.floating]

    def __init__(self, value: float = 0.0):
        self._coords = np.zeros((1,))
        self.z = value

    @property
    def coords(self) -> npt.NDArray[np.floating]:
        """Coordinates of the point as an array.

        Returns
        -------
        numpy.ndarray, shape=(1,)
        """
        return self._coords

    @property
    def z(self) -> float:
        """Coordinate of the point.

        Parameters
        ----------
        float
            The value to assign to the coordinate.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be converted to float.
        """
        return self.coords[0]

    @z.setter
    def z(self, value: float) -> None:
        self.coords[0] = value


class Node1D(Point1D):
    """Class for storing the properties of a node.
    Inherits from :c:`Point1D`.

    Attributes
    ----------
    index
    coords
    z
    temp
    temp_rate
    void_ratio
    void_ratio_0
    tot_stress
    z_def
    deg_sat_water
    deg_sat_water__0

    Parameters
    ----------
    index: int
        The value to assign to the index of the node.
        Cannot be negative.
    coord: float, optional, default=0.0
        The value to assign to the coordinate of the node.
    temp: float, optional, default=0.0
        The value to assign to the temperature of the node.
    void_ratio: float, optional, default=0.0
        The value to assign to the void ratio of the node.
        Cannot be negative.
    void_ratio_0: float, optional, default=0.0
        The value to assign to the initial void ratio of the node.
        Cannot be negative.
    temp_rate: float, optional, default=0.0
        The value to assign to the rate of change (with time)
        of temperature of the node.
    tot_stress: float, optional, default=0.0
        The value to assign to the total stress
        of the node.

    Raises
    ------
    TypeError
        If index is a float.
    ValueError
        If index is a str not convertible to int.
        If index is negative.
        If z cannot be converted to float.
        If temp is not convertible to float.
        If void_ratio is not convertible to float.
        If void_ratio is negative.
        If temp_rate cannot be converted to float.
        If tot_stress cannot be converted to float.
        If z_def cannot be converted to float.
    """

    _index: int
    _temp: float = 0.0
    _void_ratio: float = 0.0
    _void_ratio_0: float = 0.0
    _temp_rate: float = 0.0
    _tot_stress: float = 0.0
    _z_def: float = 0.0
    _deg_sat_water: float = 1.0
    _deg_sat_water__0: float = 1.0

    def __init__(
        self,
        index: int,
        coord: float = 0.0,
        temp: float = 0.0,
        void_ratio: float = 0.0,
        void_ratio_0: float = 0.0,
        temp_rate: float = 0.0,
        tot_stress: float = 0.0,
        z_def: float = 0.0,
        deg_sat_water: float = 1.0,
        deg_sat_water__0: float = 1.0,
    ):
        self.index = index
        super().__init__(coord)
        self.temp = temp
        self.void_ratio = void_ratio
        self.void_ratio_0 = void_ratio_0
        self.temp_rate = temp_rate
        self.tot_stress = tot_stress
        self.z_def = coord
        self.deg_sat_water = deg_sat_water
        self.deg_sat_water__0 = deg_sat_water__0

    @property
    def temp(self) -> float:
        """Temperature of the node.

        Parameters
        ----------
        float
            Value to assign to the temperature of the :c:`Node1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._temp

    @temp.setter
    def temp(self, value: float) -> None:
        value = float(value)
        self._temp = value

    @property
    def temp_rate(self) -> float:
        """Rate of change of temperature (with time) of the node.

        Parameters
        ----------
        float
            Value to assign to the temperature rate of the :c:`Node1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._temp_rate

    @temp_rate.setter
    def temp_rate(self, value: float) -> None:
        value = float(value)
        self._temp_rate = value

    @property
    def index(self) -> int:
        """Index of the node.

        Parameters
        ----------
        int
            Value to assign to the index of the :c:`Node1D`.

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If value to assign is a float.
        ValueError
            If value to assign is a str not convertible to int.
            If value to assign is negative.
        """
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        if isinstance(value, float):
            raise TypeError(f"{value} is a float, must be int")
        value = int(value)
        if value < 0:
            raise ValueError(f"{value} is negative")
        self._index = value

    @property
    def void_ratio(self) -> float:
        """Void ratio of the node.

        Parameters
        ----------
        float
            Value to assign to the void ratio of the :c:`Node1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.
        """
        return self._void_ratio

    @void_ratio.setter
    def void_ratio(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio {value} is not positive")
        self._void_ratio = value

    @property
    def void_ratio_0(self) -> float:
        """Initial void ratio of the node.

        Parameters
        ----------
        float
            Value to assign to the initial void ratio of the :c:`Node1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.
        """
        return self._void_ratio_0

    @void_ratio_0.setter
    def void_ratio_0(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_0 {value} is not positive")
        self._void_ratio_0 = value

    @property
    def tot_stress(self) -> float:
        """Total (overburden) stress of the node.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._tot_stress

    @tot_stress.setter
    def tot_stress(self, value: float) -> None:
        self._tot_stress = float(value)

    @property
    def z_def(self) -> float:
        """Deformed coordinate of the node.

        Parameters
        ----------
        float
            Value to assign to the deformed coordinate of the :c:`Node1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._z_def

    @z_def.setter
    def z_def(self, value: float) -> None:
        value = float(value)
        self._z_def = value

    @property
    def deg_sat_water(self) -> float:
        """Degree of saturation of water of the node.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.0 or value > 1.0
        """
        return self._deg_sat_water

    @deg_sat_water.setter
    def deg_sat_water(self, value: float) -> None:
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"deg_sat_water value {value} not between 0.0 and 1.0")
        self._deg_sat_water = value

    @property
    def deg_sat_water__0(self) -> float:
        """Previous degree of saturation of water of the node
        (i.e. at the beginning of the time step).

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.0 or value > 1.0
        """
        return self._deg_sat_water__0

    @deg_sat_water__0.setter
    def deg_sat_water__0(self, value: float) -> None:
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"deg_sat_water__0 value {value} not between 0.0 and 1.0")
        self._deg_sat_water__0 = value


class IntegrationPoint1D(Point1D):
    """Class for storing the properties of an integration point.
    Inherits from :c:`Point1D`.

    Attributes
    ----------
    coords
    z
    local_coord
    weight
    void_ratio
    void_ratio__0
    void_ratio_0
    temp
    temp__0
    temp_rate
    temp_gradient
    porosity
    vol_water_cont
    vol_water_cont__0
    vol_water_cont_temp_gradient
    deg_sat_water
    deg_sat_ice
    deg_sat_water_temp_gradient
    material
    thrm_cond
    vol_heat_cap
    hyd_cond
    hyd_cond_gradient
    water_flux_rate
    pre_consol_stress
    eff_stress
    eff_stress__0
    eff_stress_gradient
    void_ratio_0_ref_frozen
    tot_stress_0_ref_frozen
    tot_stress
    loc_stress
    tot_stress_gradient
    pore_pressure
    exc_pore_pressure
    update_water_flux_rate

    Parameters
    ----------
    coord : float, optional, default=0.0
        The value to assign to the coordinate of the point.
        From Point1D Class.
    local_coord: float, optional, default=0.0
        Local coordinate of the integration point.
    weight: float, optional, default=0.0
        Quadrature weight of the integration point.
    void_ratio: float, optional, default=0.0
        Void ratio of the integration point.
        Should be not negative.
    void_ratio_0: float, optional, default=0.0
        Initial void ratio of the integration point.
        Should be not negative.
    temp: float, optional, default=0.0
        Temperature at the integration point.
    temp_rate: float, optional, default=0.0
        Temperature rate at the integration point.
    temp_gradient: float, optional, default=0.0
        Temperature gradient at the integration point.
    deg_sat_water: float, optional, default=1.0
        Degree of saturation of water of the integration point.
        Also sets degree of saturation of ice (assuming full saturation)
        and volumetric ice content.
        Value should be between 0.0 and 1.0.
    deg_sat_water_temp_gradient: float, optional, default=0.0
        Gradient of degree of saturation of water
        with respect to temperature, dSw/dT.
        Value should be >= 0.0.
    material: Material, optional, default=materials.NULL_MATERIAL
        Contains the properties of the solids.
    hyd_cond: float, optional, default=0.0
        Hydraulic conductivity of the integration point.
        Should be not negative.
    hyd_cond_gradient: float, optional, default=0.0
        Hydraulic conductivity gradient (with respect to void ratio)
        of the integration point. Should be not negative.
    water_flux_rate: float, optional, default=0.0
        Water flux rate of the integration point.
    pre_consol_stress: float, optional, default=0.0
        Preconsolidation stress of the integration point.
    eff_stress: float, optional, default=0.0
        Effective stress of the integration point.
    eff_stress_gradient: float, optional, default=0.0
        Effective stress gradient (with respect to void ratio)
        of the integration point.
        Should not be positive.
    void_ratio_0_ref_frozen: float, optional, default=0.0
        Reference void ratio for frozen void ratio - total stress curve.
        Should be not negative.
    tot_stress_0_ref_frozen: float, optional, default=0.0
        Reference total stress for frozen void ratio - total stress curve.
    tot_stress: float, optional, default=0.0
        Total (overburden) stress of the integration point.
    tot_stress_gradient: float, optional, default=0.0
        Total stress gradient (with respect to void ratio)
        of the integration point. Should be not positive.
    loc_stress: float, optional, default=0.0
        Local (overburden + void ratio increment) stress
        of the integration point.

    Raises
    ------
    TypeError
        If material is not an instance of
        :c:`frozen_ground_fem.materials.Material`.
    ValueError
        If coord is not convertible to float.
        If local_coord is not convertible to float.
        If weight is not convertible to float.
        If void_ratio is not convertible to float.
        If void_ratio is negative.
        If void_ratio_0 is not convertible to float.
        If void_ratio_0 is negative.
        If temp is not convertible to float.
        If temp_rate is not convertible to float.
        If temp_gradient is not convertible to float.
        If deg_sat_water is not convertible to float.
        If deg_sat_water <0.0 or >1.0.
        If deg_sat_water_temp_gradient is not convertible to float.
        If deg_sat_water_temp_gradient is negative.
        If hyd_cond is not convertible to float.
        If hyd_cond is negative.
        If hyd_cond_gradient is not convertible to float.
        If water_flux_rate is not convertible to float.
        If pre_consol_stress is not convertible to float.
        If eff_stress is not convertible to float.
        If eff_stress_gradient is not convertible to float.
        If eff_stress_gradient is positive.
        If void_ratio_0_ref_frozen is not convertible to float.
        If void_ratio_0_ref_frozen is negative.
        If tot_stress_0_ref_frozen is not convertible to float.
        If tot_stress is not convertible to float.
        If tot_stress_gradient is not convertible to float.
        If tot_stress_gradient is positive.
        If loc_stress is not convertible to float.
    """

    _local_coord: float = 0.0
    _weight: float = 0.0
    _void_ratio: float = 0.0
    _void_ratio__0: float = 0.0
    _porosity: float = 0.0
    _void_ratio_0: float = 0.0
    _temp: float = 0.0
    _temp__0: float = 0.0
    _temp_rate: float = 0.0
    _temp_gradient: float = 0.0
    _deg_sat_water: float = 1.0
    _deg_sat_ice: float = 0.0
    _deg_sat_water_temp_gradient: float = 0.0
    _vol_water_cont: float = 0.0
    _vol_water_cont__0: float = 0.0
    _vol_water_cont_temp_gradient: float = 0.0
    _hyd_cond: float = 0.0
    _hyd_cond_gradient: float = 0.0
    _water_flux_rate: float = 0.0
    _pre_consol_stress: float = 0.0
    _eff_stress: float = 0.0
    _eff_stress__0: float = 0.0
    _eff_stress_gradient: float = 0.0
    _void_ratio_0_ref_frozen: float = 0.0
    _tot_stress_0_ref_frozen: float = 0.0
    _tot_stress: float = 0.0
    _tot_stress_gradient: float = 0.0
    _loc_stress: float = 0.0
    _pore_pressure: float = 0.0
    _exc_pore_pressure: float = 0.0
    _material: Material
    _z_def: float = 0.0

    def __init__(
        self,
        coord: float = 0.0,
        local_coord: float = 0.0,
        weight: float = 0.0,
        void_ratio: float = 0.0,
        void_ratio_0: float = 0.0,
        temp: float = 0.0,
        temp_rate: float = 0.0,
        temp_gradient: float = 0.0,
        deg_sat_water: float = 1.0,
        deg_sat_water_temp_gradient: float = 0.0,
        vol_water_cont_temp_gradient: float = 0.0,
        material: Material = NULL_MATERIAL,
        hyd_cond: float = 0.0,
        hyd_cond_gradient: float = 0.0,
        water_flux_rate: float = 0.0,
        pre_consol_stress: float = 0.0,
        eff_stress: float = 0.0,
        eff_stress_gradient: float = 0.0,
        void_ratio_0_ref_frozen: float = 0.0,
        tot_stress_0_ref_frozen: float = 0.0,
        tot_stress: float = 0.0,
        tot_stress_gradient: float = 0.0,
        loc_stress: float = 0.0,
        pore_pressure: float = 0.0,
        exc_pore_pressure: float = 0.0,
    ):
        super().__init__(coord)
        self.local_coord = local_coord
        self.weight = weight
        self.void_ratio = void_ratio
        self.void_ratio_0 = void_ratio_0
        self.temp = temp
        self.temp_rate = temp_rate
        self.temp_gradient = temp_gradient
        self.deg_sat_water = deg_sat_water
        self.deg_sat_water_temp_gradient = deg_sat_water_temp_gradient
        self.vol_water_cont_temp_gradient = vol_water_cont_temp_gradient
        self.material = material
        self.hyd_cond = hyd_cond
        self.hyd_cond_gradient = hyd_cond_gradient
        self.water_flux_rate = water_flux_rate
        self.pre_consol_stress = pre_consol_stress
        self.eff_stress = eff_stress
        self.eff_stress_gradient = eff_stress_gradient
        self.void_ratio_0_ref_frozen = void_ratio_0_ref_frozen
        self.tot_stress_0_ref_frozen = tot_stress_0_ref_frozen
        self.tot_stress = tot_stress
        self.tot_stress_gradient = tot_stress_gradient
        self.loc_stress = loc_stress
        self.pore_pressure = pore_pressure
        self.exc_pore_pressure = exc_pore_pressure

    @property
    def local_coord(self) -> float:
        """Local coordinate of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._local_coord

    @local_coord.setter
    def local_coord(self, value: float) -> None:
        value = float(value)
        self._local_coord = value

    @property
    def weight(self) -> float:
        """Quadrature weight of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        value = float(value)
        self._weight = value

    @property
    def void_ratio(self) -> float:
        """Void ratio of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.

        Notes
        -----
        Also updates porosity and volumetric ice content.
        """
        return self._void_ratio

    @void_ratio.setter
    def void_ratio(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio {value} is not positive")
        self._void_ratio = value
        self._porosity = value / (1.0 + value)
        self._vol_water_cont = self.porosity * self.deg_sat_water

    @property
    def void_ratio_0(self) -> float:
        """Initial (reference) void ratio of the integration point
        (e.g. prior to settlement analysis).

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.
        """
        return self._void_ratio_0

    @void_ratio_0.setter
    def void_ratio_0(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_0 {value} is not positive")
        self._void_ratio_0 = value

    @property
    def temp(self) -> float:
        """Temperature at the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._temp

    @temp.setter
    def temp(self, value: float) -> None:
        self._temp = float(value)

    @property
    def temp__0(self) -> float:
        """Previous temperature at the integration point
        (e.g. at the beginning of a time step).

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._temp__0

    @temp__0.setter
    def temp__0(self, value: float) -> None:
        self._temp__0 = float(value)

    @property
    def temp_rate(self) -> float:
        """Temperature rate at the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._temp_rate

    @temp_rate.setter
    def temp_rate(self, value: float) -> None:
        self._temp_rate = float(value)

    @property
    def temp_gradient(self) -> float:
        """Temperature gradient at the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._temp_gradient

    @temp_gradient.setter
    def temp_gradient(self, value: float) -> None:
        self._temp_gradient = float(value)

    @property
    def porosity(self) -> float:
        """Porosity of the integration point.

        Returns
        -------
        float

        Notes
        -----
        Porosity is not intended to be updated directly.
        It is updated each time void ratio is set.
        """
        return self._porosity

    @property
    def vol_water_cont(self) -> float:
        """Volumetric water content of the integration point.

        Returns
        -------
        float

        Notes
        ------
        Volumetric water content is not intended to be set directly.
        It is updated when void ratio or
        degree of saturation of water are updated.
        """
        return self._vol_water_cont

    @property
    def vol_water_cont__0(self) -> float:
        """Previous volumetric water content of the integration point
        (e.g. at the beginning of a time step).

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.0.
        """
        return self._vol_water_cont__0

    @vol_water_cont__0.setter
    def vol_water_cont__0(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"vol_water_cont__0 value {value} " + "is negative.")
        self._vol_water_cont__0 = value

    @property
    def deg_sat_water(self) -> float:
        """Degree of saturation of water of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.0 or value > 1.0

        Notes
        -----
        Also updates degree of saturation of ice (assuming full saturation)
        and volumetric water content.
        """
        return self._deg_sat_water

    @deg_sat_water.setter
    def deg_sat_water(self, value: float) -> None:
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"deg_sat_water value {value} " + "not between 0.0 and 1.0"
            )
        self._deg_sat_water = value
        self._deg_sat_ice = 1.0 - value
        self._vol_water_cont = self.porosity * self._deg_sat_water

    @property
    def deg_sat_ice(self) -> float:
        """Degree of saturation of ice of the integration point.

        Returns
        -------
        float

        Notes
        ------
        Degree of saturation of ice is not intended to be set directly.
        It is updated when degree of saturation of water is updated,
        assuming fully saturated conditions, i.e.
            deg_sat_water + deg_sat_ice = 1.0
        """
        return self._deg_sat_ice

    @property
    def deg_sat_water_temp_gradient(self) -> float:
        """Gradient of degree of saturation of water
        with respect to temperature.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.
        """
        return self._deg_sat_water_temp_gradient

    @deg_sat_water_temp_gradient.setter
    def deg_sat_water_temp_gradient(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(
                f"deg_sat_water_temp_gradient value {value} " + "is negative"
            )
        self._deg_sat_water_temp_gradient = value

    @property
    def vol_water_cont_temp_gradient(self) -> float:
        """Gradient of degree of saturation of water
        with respect to temperature.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.
        """
        return self._vol_water_cont_temp_gradient

    @vol_water_cont_temp_gradient.setter
    def vol_water_cont_temp_gradient(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(
                f"vol_water_cont_temp_gradient value {value} " + "is negative"
            )
        self._vol_water_cont_temp_gradient = value

    @property
    def material(self) -> Material:
        """Contains the properties of the solids.

        Parameters
        ----------
        frozen_ground_fem.materials.Material

        Returns
        -------
        frozen_ground_fem.materials.Material

        Raises
        ------
        TypeError
            If value to assign is not an instance of
            :c:`frozen_ground_fem.materials.Material`.
        """
        return self._material

    @material.setter
    def material(self, value: Material) -> None:
        if not isinstance(value, Material):
            raise TypeError(f"{value} is not a Material object")
        self._material = value

    @property
    def thrm_cond(self) -> float:
        """Contains the bulk thermal conductivity of the integration point.

        Returns
        ------
        float

        Notes
        -----
        Calculated according to the geometric mean formula [1]_::

            lam = (lam_s ** (1 - por)) * (lam_i ** th_i) * (lam_w ** th_w)

        References
        ----------
        .. [1] Côté, J. and Konrad, J.-M. 2005. A generalized thermal
           conductivity model for soils and construction materials. Canadian
           Geotechnical Journal 42(2): 443-458, doi: 10.1139/t04-106.
        """
        lam_s = self.material.thrm_cond_solids
        por = self.porosity
        th_w = self.vol_water_cont
        th_i = por - th_w
        return (lam_s ** (1 - por)) * (lam_i**th_i) * (lam_w**th_w)

    @property
    def vol_heat_cap(self) -> float:
        """Contains the volumetric heat capacity of the integration point.

        Returns
        ------
        float

        Notes
        -----
        Calculated according to the volume averaging formula [1]_::

            C = (1 - por) * C_s + th_i * C_i + th_w * C_w

        References
        ----------
        .. [1] Andersland, O. and Ladanyi, B. 2004. Frozen Ground Engineering,
           2nd ed. Wiley: Hoboken, N.J.
        """
        C_s = self.material.vol_heat_cap_solids
        por = self.porosity
        th_w = self.vol_water_cont
        th_i = por - th_w
        return ((1 - por) * C_s) + (th_i * C_i) + (th_w * C_w)

    @property
    def hyd_cond(self) -> float:
        """Hydraulic conductivity of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
            If the value to assign is negative.
        """
        return self._hyd_cond

    @hyd_cond.setter
    def hyd_cond(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"value {value} for hyd_cond cannot be negative.")
        self._hyd_cond = value

    @property
    def hyd_cond_gradient(self) -> float:
        """Hydraulic conductivity gradient
        (with respect to void ratio)
        of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
            If the value to assign is negative.
        """
        return self._hyd_cond_gradient

    @hyd_cond_gradient.setter
    def hyd_cond_gradient(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"value {value} for hyd_cond_gradient cannot be negative.")
        self._hyd_cond_gradient = value

    @property
    def water_flux_rate(self) -> float:
        """Water flux rate of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._water_flux_rate

    @water_flux_rate.setter
    def water_flux_rate(self, value: float) -> None:
        self._water_flux_rate = float(value)

    @property
    def pre_consol_stress(self) -> float:
        """Preconsolidation stress of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._pre_consol_stress

    @pre_consol_stress.setter
    def pre_consol_stress(self, value: float) -> None:
        self._pre_consol_stress = float(value)

    @property
    def eff_stress(self) -> float:
        """Effective stress of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._eff_stress

    @eff_stress.setter
    def eff_stress(self, value: float) -> None:
        self._eff_stress = float(value)

    @property
    def eff_stress_gradient(self) -> float:
        """Effective stress gradient
        (with respect to void ratio)
        of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
            If the value to assign is positive.
        """
        return self._eff_stress_gradient

    @eff_stress_gradient.setter
    def eff_stress_gradient(self, value: float) -> None:
        value = float(value)
        if value > 0.0:
            raise ValueError(
                f"value {value} for eff_stress_gradient cannot be positive."
            )
        self._eff_stress_gradient = value

    @property
    def void_ratio_0_ref_frozen(self) -> float:
        """Reference void ratio for frozen void ratio - total stress curve.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
            If the value to assign is negative.
        """
        return self._void_ratio_0_ref_frozen

    @void_ratio_0_ref_frozen.setter
    def void_ratio_0_ref_frozen(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(
                f"value {value} for void_ratio_0_ref_frozen" + " cannot be negative."
            )
        self._void_ratio_0_ref_frozen = value

    @property
    def tot_stress_0_ref_frozen(self) -> float:
        """Reference total stress for frozen void ratio - total stress curve.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._tot_stress_0_ref_frozen

    @tot_stress_0_ref_frozen.setter
    def tot_stress_0_ref_frozen(self, value: float) -> None:
        self._tot_stress_0_ref_frozen = float(value)

    @property
    def tot_stress(self) -> float:
        """Total (overburden) stress of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._tot_stress

    @tot_stress.setter
    def tot_stress(self, value: float) -> None:
        self._tot_stress = float(value)

    @property
    def loc_stress(self) -> float:
        """Local (overburden + void ratio increment)
        stress of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._loc_stress

    @loc_stress.setter
    def loc_stress(self, value: float) -> None:
        self._loc_stress = float(value)

    @property
    def tot_stress_gradient(self) -> float:
        """Total stress gradient
        (with respect to void ratio)
        of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
            If the value to assign is positive.
        """
        return self._tot_stress_gradient

    @tot_stress_gradient.setter
    def tot_stress_gradient(self, value: float) -> None:
        value = float(value)
        # if value > 0.0:
        #     raise ValueError(
        #         f"value {value} for tot_stress_gradient cannot be positive."
        #     )
        self._tot_stress_gradient = value

    @property
    def pore_pressure(self) -> float:
        """Pore pressure of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._pore_pressure

    @pore_pressure.setter
    def pore_pressure(self, value: float) -> None:
        self._pore_pressure = float(value)

    @property
    def exc_pore_pressure(self) -> float:
        """Excess pore pressure of the integration point.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
        """
        return self._exc_pore_pressure

    @exc_pore_pressure.setter
    def exc_pore_pressure(self, value: float) -> None:
        self._exc_pore_pressure = float(value)

    @property
    def z_def(self) -> float:
        """Deformed coordinate of the integration point.

        Parameters
        ----------
        float
            Value to assign to the deformed coordinate
            of the :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._z_def

    @z_def.setter
    def z_def(self, value: float) -> None:
        value = float(value)
        self._z_def = value


class Element1D:
    """Class for organizing element level information.

    Attributes
    ----------
    order
    nodes
    jacobian
    int_pts

    Parameters
    ----------
    nodes : Sequence[Node1D]
        The tuple of :c:`Node1D` contained in the element.
    order : int, optional, default=3
        The order of interpolation used in the element.

    Raises
    ------
    TypeError:
        If nodes contains non-:c:`Node1D` objects.
    ValueError
        If len(nodes) is invalid for the order of interpolation.
        If order is not 1 or 3.
    """

    # _int_pt_coords_linear: ClassVar[tuple[float, float]] = (
    #     0.211324865405187,
    #     0.788675134594813,
    # )
    # _int_pt_weights_linear: ClassVar[tuple[float, float]] = (
    #     0.5,
    #     0.5,
    # )
    # _int_pt_coords_linear_deformed: ClassVar[tuple[tuple[float, float]]] = (
    #     (
    #         0.211324865405187,
    #         0.788675134594813,
    #     ),
    # )
    # _int_pt_weights_linear_deformed: ClassVar[tuple[tuple[float, float]]] = (
    #     (
    #         0.5,
    #         0.5,
    #     ),
    # )

    _int_pt_coords_linear: ClassVar[tuple[float, ...]] = (
        0.04691007703066802,
        0.2307653449471585,
        0.5,
        0.7692346550528415,
        0.9530899229693319,
    )
    _int_pt_weights_linear: ClassVar[tuple[float, ...]] = (
        0.11846344252809454,
        0.23931433524968324,
        0.28444444444444444,
        0.23931433524968324,
        0.11846344252809454,
    )
    _int_pt_coords_linear_deformed: ClassVar[tuple[tuple[float, ...]]] = (
        (
            0.04691007703066802,
            0.2307653449471585,
            0.5,
            0.7692346550528415,
            0.9530899229693319,
        ),
    )
    _int_pt_weights_linear_deformed: ClassVar[tuple[tuple[float, ...]]] = (
        (
            0.11846344252809454,
            0.23931433524968324,
            0.28444444444444444,
            0.23931433524968324,
            0.11846344252809454,
        ),
    )

    _int_pt_coords_cubic: ClassVar[tuple[float, ...]] = (
        0.04691007703066802,
        0.2307653449471585,
        0.5,
        0.7692346550528415,
        0.9530899229693319,
    )
    _int_pt_weights_cubic: ClassVar[tuple[float, ...]] = (
        0.11846344252809454,
        0.23931433524968324,
        0.28444444444444444,
        0.23931433524968324,
        0.11846344252809454,
    )
    _int_pt_coords_cubic_deformed: ClassVar[tuple[tuple[float, float], ...]] = (
        (
            0.070441621801729,
            0.262891711531604,
        ),
        (
            0.403774955135062,
            0.596225044864938,
        ),
        (
            0.737108288468396,
            0.929558378198271,
        ),
    )
    _int_pt_weights_cubic_deformed: ClassVar[tuple[tuple[float, float], ...]] = (
        (
            0.5,
            0.5,
        ),
        (
            0.5,
            0.5,
        ),
        (
            0.5,
            0.5,
        ),
    )

    _nodes: tuple[Node1D, ...]
    _int_pts: tuple[IntegrationPoint1D, ...]
    _int_pts_deformed: tuple[tuple[IntegrationPoint1D, ...], ...]
    _order: int = 3
    _shape_matrix: Callable[[float], npt.NDArray[np.floating]]
    _gradient_matrix: Callable[[float, float], npt.NDArray[np.floating]]

    def __init__(
        self,
        nodes: Sequence[Node1D],
        order: int = 3,
    ):
        # assign order parameter
        self.order = order
        # check for valid node list and assign to self
        if (nnod := len(nodes)) != self.order + 1:
            raise ValueError(
                f"len(nodes) is {nnod} not equal to order+1 = {self.order + 1}"
            )
        for nd in nodes:
            if not isinstance(nd, Node1D):
                raise TypeError("nodes contains invalid objects, not Node1D")
        self._nodes = tuple(nodes)
        # initialize integration points
        if self.order == 1:
            self._shape_matrix = shape_matrix_linear
            self._gradient_matrix = gradient_matrix_linear
            self._int_pts = tuple(
                IntegrationPoint1D(local_coord=xi, weight=wt)
                for (xi, wt) in zip(
                    Element1D._int_pt_coords_linear,
                    Element1D._int_pt_weights_linear,
                )
            )
            self._int_pts_deformed = tuple(
                tuple(
                    IntegrationPoint1D(local_coord=xi, weight=wt)
                    for (xi, wt) in zip(xx, ww)
                )
                for (xx, ww) in zip(
                    Element1D._int_pt_coords_linear_deformed,
                    Element1D._int_pt_weights_linear_deformed,
                )
            )
        elif self.order == 3:
            self._shape_matrix = shape_matrix_cubic
            self._gradient_matrix = gradient_matrix_cubic
            self._int_pts = tuple(
                IntegrationPoint1D(local_coord=xi, weight=wt)
                for (xi, wt) in zip(
                    Element1D._int_pt_coords_cubic,
                    Element1D._int_pt_weights_cubic,
                )
            )
            self._int_pts_deformed = tuple(
                tuple(
                    IntegrationPoint1D(local_coord=xi, weight=wt)
                    for (xi, wt) in zip(xx, ww)
                )
                for (xx, ww) in zip(
                    Element1D._int_pt_coords_cubic_deformed,
                    Element1D._int_pt_weights_cubic_deformed,
                )
            )
        z_e = np.array([[self.nodes[0].z, self.nodes[-1].z]]).T
        for ip in self.int_pts:
            N = shape_matrix_linear(ip.local_coord)
            ip.z = (N @ z_e)[0][0]

    @property
    def order(self) -> int:
        """The order of interpolation used in the element.

        Parameters
        ----------
        int

        Returns
        ------
        int

        Raises
        ------
        ValueError
            If the value to assign is not convertible to int.
            If the value to assign is not 1 or 3.
        """
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        value = int(value)
        if value not in [1, 3]:
            raise ValueError(f"order {value} not 1 or 3")
        self._order = value

    @property
    def num_nodes(self) -> int:
        """The number of nodes in the element.

        Returns
        -------
        int
        """
        return len(self.nodes)

    @property
    def nodes(self) -> tuple[Node1D, ...]:
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple[:c:`Node1D`]
        """
        return self._nodes

    @property
    def jacobian(self) -> float:
        """The length scale of the element (in Lagrangian coordinates).

        Returns
        -------
        float
        """
        return self.nodes[-1].z - self.nodes[0].z

    @property
    def int_pts(self) -> tuple[IntegrationPoint1D, ...]:
        """The tuple of :c:`IntegrationPoint1D` contained in the element.

        Returns
        ------
        tuple[:c:`IntegrationPoint1D`]
        """
        return self._int_pts

    def assign_material(self, m: Material) -> None:
        """Convenience method for assigning a material to
        all integration points in the element.
        """
        for ip in self.int_pts:
            ip.material = m
        for iipp in self._int_pts_deformed:
            for ip in iipp:
                ip.material = m

    def initialize_integration_points_primary(self) -> None:
        """Abstract base method for initializing values of
        primary solution variables
        (and any variables not affected by coupling)
        at the element integration points.
        """
        pass

    def initialize_integration_points_secondary(self) -> None:
        """Abstract base method for initializing values of
        secondary solution variables
        (i.e. variables affected by coupling)
        at the element integration points.
        """
        pass

    def update_integration_points_primary(self) -> None:
        """Abstract base method for updating values of
        primary solution variables (and, optionally,
        some variables not affected by coupling)
        at the element integration points.
        """
        pass

    def update_integration_points_secondary(self) -> None:
        """Abstract base method for updating values of
        secondary solution variables (esp. variables affected
        by coupling) at the element integration points.
        """
        pass


class Boundary1D:
    """Class for storing boundary condition geometry information.

    Attributes
    ----------
    nodes
    int_pts

    Parameters
    ----------
    nodes : Sequence[Node1D]
        The tuple of :c:`Node1D` contained in the element.
    int_pts : Sequence[IntegrationPoint1D], optional, default=()
        The tuple of :c:`IntegrationPoint1D` contained in the element.

    Raises
    ------
    TypeError:
        If nodes contains non-:c:`Node1D` objects.
        If int_pts contains non-:c:`IntegrationPoint1D` objects.
    ValueError
        If len(nodes) != 1.
        If len(int_pts) > 1.
    """

    _nodes: tuple[Node1D, ...]
    _int_pts: tuple[IntegrationPoint1D, ...]

    def __init__(
        self,
        nodes: Sequence[Node1D],
        int_pts: Sequence[IntegrationPoint1D] = (),
    ):
        # check for valid node list and assign to self
        if (nnod := len(nodes)) != 1:
            raise ValueError(f"len(nodes) is {nnod} not equal to 1")
        if not isinstance(nodes[0], Node1D):
            raise TypeError("nodes contains invalid objects, not Node1D")
        self._nodes = tuple(nodes)
        if len(int_pts) > 1:
            raise ValueError(f"len(int_pts) {len(int_pts)} > 1")
        for ip in int_pts:
            if not isinstance(ip, IntegrationPoint1D):
                raise TypeError(
                    "int_pts contains invalid objects, " + "not IntegrationPoint1D"
                )
        self._int_pts = tuple(int_pts)

    @property
    def nodes(self) -> tuple[Node1D, ...]:
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple[:c:`Node1D`]
        """
        return self._nodes

    @property
    def int_pts(self) -> tuple[IntegrationPoint1D, ...]:
        """The tuple of :c:`IntegrationPoint1D` contained in the element.

        Returns
        ------
        tuple[:c:`IntegrationPoint1D`]
        """
        return self._int_pts


class Mesh1D:
    """Class for generating, storing, and organizing global geometry
    information about the analysis mesh.

    Attributes
    ----------
    z_min
    z_max
    grid_size
    num_nodes
    nodes
    num_elements
    elements
    num_boundaries
    boundaries
    mesh_valid
    time_step
    dt
    over_dt
    implicit_factor
    alpha
    one_minus_alpha
    implicit_error_tolerance
    eps_s
    max_iterations

    Methods
    -------
    initialize_global_matrices_and_vectors
    generate_mesh
    add_boundary
    remove_boundary
    clear_boundaries
    initialize_global_system
    initialize_time_step
    store_converged_matrices
    update_boundary_vectors
    initialize_free_index_arrays
    initialize_integration_points
    update_nodes
    update_boundary_conditions
    initialize_solution_variable_vectors
    calculate_solution_vector_correction
    update_total_stress_distribution
    update_pore_pressure_distribution
    update_iteration_variables
    solve_to

    Parameters
    -----------
    z_range: array_like, optional, default=()
        The value to assign to range of z values from z_min to z_max.
    grid_size: float, optional, default=0.0
        The value to assign to specified grid size of the mesh.
        Cannot be negative.
    num_elements: int, optional, default=10
        The specified number of :c:`Element1D` in the mesh.
    order: int, optional, default=3
        The order of interpolation to be used.
    generate: bool, optional, default=False
        Flag for whether to generate a mesh using assigned properties.

    Raises
    ------
    ValueError
        If z_range values cannot be cast to float.
        If grid_size cannot be cast to float.
        If grid_size < 0.0.
    """

    _z_min: float = -np.inf
    _z_max: float = np.inf
    _mesh_valid: bool = False
    _grid_size: float = 0.0
    _hbw: int = 3
    _nodes: tuple[Node1D, ...]
    _elements: tuple[Any, ...]
    _boundaries: set[Any]
    _time_step: float = 0.0
    _inv_time_step: float = 0.0
    _implicit_factor: float = 0.5  # Crank-Nicolson
    _inv_implicit_factor: float = 0.5
    _implicit_error_tolerance: float = 1e-3
    _max_iterations: int = 100
    _free_vec: tuple[npt.NDArray, ...]
    _free_arr: tuple[npt.NDArray, ...]
    _t0: float
    _t1: float

    def __init__(
        self,
        z_range: npt.ArrayLike = (),
        grid_size: float = 0.0,
        num_elements: int = 10,
        order: int = 3,
        generate: bool = False,
    ):
        self._boundaries = set()
        self._nodes = ()
        self._elements = ()
        if z_range:
            self.z_min = np.min(z_range)
            self.z_max = np.max(z_range)
        self.grid_size = grid_size
        if generate:
            self.generate_mesh(num_elements, order)

    @property
    def z_min(self) -> float:
        """The minimum z value of the mesh.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to float.
            If the value to assign is >= z_max.
        """
        return self._z_min

    @z_min.setter
    def z_min(self, value: float) -> None:
        value = float(value)
        if value >= self.z_max:
            raise ValueError(f"{value} >= z_max := {self.z_max}")
        self._z_min = value
        self.mesh_valid = False

    @property
    def z_max(self) -> float:
        """The maximum z value of the mesh.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to float.
            If the value to assign is <= z_min.
        """
        return self._z_max

    @z_max.setter
    def z_max(self, value: float) -> None:
        value = float(value)
        if value <= self.z_min:
            raise ValueError(f"{value} <= z_min := {self.z_min}")
        self._z_max = value
        self.mesh_valid = False

    @property
    def grid_size(self) -> float:
        """The specified grid size of the mesh.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to float.
            If the value to assign is < 0.0.

        Notes
        -----
        This parameter is not the actual size of any element of the mesh.
        It is a suggested target value that will be recalculated so that
        an integer number of nodes between z_min and z_max is achieved.
        The actual element size will typically be smaller.
        If grid_size is set to 0.0, its value is ignored
        and the element size is calculated based on a specified (or default)
        number of elements.
        """
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError(f"{value} is negative")
        self._grid_size = value
        self.mesh_valid = False

    @property
    def num_nodes(self) -> int:
        """The number of :c:`Node1D` contained in the mesh.

        Returns
        ------
        int
        """
        return len(self.nodes)

    @property
    def nodes(self) -> tuple[Node1D, ...]:
        """The tuple of :c:`Node1D` contained in the mesh.

        Returns
        ------
        tuple[:c:`Node1D`]
        """
        return self._nodes

    @property
    def num_elements(self) -> int:
        """The number of :c:`Element1D` contained in the mesh.

        Returns
        ------
        int
        """
        return len(self.elements)

    @property
    def elements(self) -> tuple[Any, ...]:
        """The tuple of :c:`Element1D` contained in the mesh.

        Returns
        ------
        tuple[:c:`Element1D`]
        """
        return self._elements

    @property
    def num_boundaries(self) -> int:
        """The number of :c:`Boundary1D` contained in the mesh.

        Returns
        ------
        int
        """
        return len(self.boundaries)

    @property
    def boundaries(self) -> set[Any]:
        """The tuple of :c:`Boundary1D` contained in the mesh.

        Returns
        ------
        set[:c:`Boundary1D`]
        """
        return self._boundaries

    def add_boundary(self, new_boundary: Any) -> None:
        """Adds a boundary to the mesh.

        Parameters
        ----------
        new_boundary : :c:`Boundary1D`
            The boundary to add to the mesh.

        Raises
        ------
        TypeError
            If new_boundary is not an instance of :c:`Boundary1D`.
        ValueError
            If new_boundary contains a :c:`Node1D` not in the mesh.
            If new_boundary contains an :c:`IntegrationPoint1D`
                not in the mesh.
        """
        if not isinstance(new_boundary, Boundary1D):
            raise TypeError(
                f"type(new_boundary) {type(new_boundary)} invalid, "
                + "must be Boundary1D (or a subclass)"
            )
        for nd in new_boundary.nodes:
            if nd not in self.nodes:
                raise ValueError(f"new_boundary contains node {nd}" + " not in mesh")
        if new_boundary.int_pts:
            int_pts = tuple(ip for e in self.elements for ip in e.int_pts)
            for ip in new_boundary.int_pts:
                if ip not in int_pts:
                    raise ValueError(f"new_boundary contains int_pt {ip} not in mesh")
        self._boundaries.add(new_boundary)

    def remove_boundary(self, boundary: Boundary1D) -> None:
        """Remove an existing boundary from the mesh.

        Parameters
        ----------
        boundary : :c:`Boundary1D`
            The boundary to remove from the mesh.

        Raises
        ------
        ValueError
            If boundary is not in the mesh.
        """
        self._boundaries.remove(boundary)

    def clear_boundaries(self) -> None:
        """Clears existing :c:`Boundary1D` objects
        from the mesh.
        """
        self._boundaries.clear()

    @property
    def mesh_valid(self) -> bool:
        """Flag for valid mesh.

        Parameters
        ----------
        bool

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to bool.

        Notes
        -----
        When assigning to False also clears mesh information
        (e.g. nodes, elements).
        """
        return self._mesh_valid

    @mesh_valid.setter
    def mesh_valid(self, value: bool) -> None:
        value = bool(value)
        if value:
            self._mesh_valid = True
            self.initialize_global_matrices_and_vectors()
        else:
            self._nodes = ()
            self._elements = ()
            self.clear_boundaries()
            self._mesh_valid = False

    def generate_mesh(self, num_elements: int = 10, order: int = 3):
        """Generates a mesh using assigned mesh properties.

        Parameters
        ----------
        num_elements : int, optional, default=10
            Number of elements to be created in the generated mesh.
        order : int, optional, default=3
            The order of interpolation to be used.

        Raises
        ------
        ValueError
            If z_min or z_max are invalid (e.g. left as default +/-inf).
            If grid_size is invalid (e.g. set to inf).

        Notes
        -----
        If the grid_size parameter is set,
        the argument num_elements will be ignored
        and the number of elements will be calculated
        as the nearest integer number of elements:
            (z_max - z_min) // grid_size
        """
        self.mesh_valid = False
        num_elements = int(num_elements)
        if num_elements < 1:
            raise ValueError(f"num_elements {num_elements} not strictly positive")
        order = int(order)
        if order not in [1, 3]:
            raise ValueError(f"order {order} not 1 or 3")
        self._hbw = order
        num_elements_out = self._generate_nodes(num_elements * order + 1, order)
        if num_elements_out:
            num_elements = num_elements_out
        self._generate_elements(num_elements, order)
        self.mesh_valid = True

    def _generate_nodes(self, num_nodes: int, order: int) -> int:
        if np.isinf(self.z_min) or np.isinf(self.z_max):
            raise ValueError("cannot generate mesh, non-finite limits")
        if np.isinf(self.grid_size):
            raise ValueError("cannot generate mesh, non-finite grid size")
        if self.grid_size > 0.0:
            num_elements = int((self.z_max - self.z_min) // self.grid_size)
            num_nodes = num_elements * order + 1
        else:
            num_elements = 0
        z_nodes = np.linspace(self.z_min, self.z_max, num_nodes)
        self._nodes = tuple(Node1D(k, zk) for k, zk in enumerate(z_nodes))
        return num_elements

    def _generate_elements(self, num_elements: int, order: int):
        self._elements = tuple(
            Element1D(tuple(self.nodes[order * k + j] for j in range(order + 1)), order)
            for k in range(num_elements)
        )

    @property
    def time_step(self) -> float:
        """The time step for the transient analysis.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float.
            If the value to assign is negative.

        Notes
        -----
        Also computes and stores an inverse value
        1 / time_step
        available as the property over_dt
        for convenience in the simulation.
        """
        return self._time_step

    @time_step.setter
    def time_step(self, value: float) -> None:
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"invalid time_step {value}, must be positive")
        self._time_step = value
        self._inv_time_step = 1.0 / value

    @property
    def dt(self) -> float:
        """An alias for time_step."""
        return self._time_step

    @property
    def over_dt(self) -> float:
        """The value 1 / time_step.

        Returns
        -------
        float

        Notes
        -----
        This value is calculated and stored
        when time_step is set,
        so this property call just returns the value.
        """
        return self._inv_time_step

    @property
    def implicit_factor(self) -> float:
        """The implicit time stepping factor for the analysis.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to be assigned is not convertible to float
            If the value is < 0.0 or > 1.0

        Notes
        -----
        This parameter sets the weighting between
        vectors and matrices at the beginning and end of the time step
        in the implicit time stepping scheme.
        For example, a value of 0.0 would put no weight at the beginning
        implying a fully implicit scheme.
        A value of 1.0 would put no weight at the end
        implying a fully explicit scheme
        (in this case, the iterative correction will have no effect).
        A value of 0.5 puts equal weight at the beginning and end
        which is the well known Crank-Nicolson scheme.
        The default set by the __init__() method is 0.5.
        """
        return self._implicit_factor

    @implicit_factor.setter
    def implicit_factor(self, value: float) -> None:
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"invalid implicit_factor {value}, must be between 0.0 and 1.0"
            )
        self._implicit_factor = value
        self._inv_implicit_factor = 1.0 - value

    @property
    def alpha(self) -> float:
        """An alias for implicit_factor."""
        return self._implicit_factor

    @property
    def one_minus_alpha(self) -> float:
        """The value (1 - implicit_factor).

        Parameters
        ----------
        float

        Returns
        -------
        float

        Notes
        -----
        This value is calculated and stored
        when implicit_factor is set,
        so this property call just returns the value.
        """
        return self._inv_implicit_factor

    @property
    def implicit_error_tolerance(self) -> float:
        """The error tolerance for the iterative correction
        in the implicit time stepping scheme.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float
            If the value to assign is negative
        """
        return self._implicit_error_tolerance

    @implicit_error_tolerance.setter
    def implicit_error_tolerance(self, value: float) -> None:
        value = float(value)
        if value <= 0.0:
            raise ValueError(
                f"invalid implicit_error_tolerance {value}, must be positive"
            )
        self._implicit_error_tolerance = value

    @property
    def eps_s(self) -> float:
        """An alias for implicit_error_tolerance."""
        return self._implicit_error_tolerance

    @property
    def max_iterations(self) -> int:
        """The maximum number of iterations for iterative correction
        in the implicit time stepping scheme.

        Parameters
        ----------
        int

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If the value to be assigned is not an int.
        ValueError
            If the value to be assigned is negative.
        """
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"type(max_iterations) {type(value)}" + " invalid, must be int"
            )
        if value <= 0:
            raise ValueError(f"max_iterations {value}" + " invalid, must be positive")
        self._max_iterations = value

    def initialize_boundary_conditions(self) -> None:
        """Performs any necessary initialization of boundary conditions.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to initialize the appropriate boundary conditions
        based on the analysis requirements.
        At this abstract level, it does nothing.
        """
        pass

    def initialize_integration_points_primary(self) -> None:
        """Initializes the primary variables at integration points
        across the mesh.
        """
        for e in self.elements:
            e.initialize_integration_points_primary()

    def initialize_integration_points_secondary(self) -> None:
        """Initializes the secondary variables at integration points
        across the mesh.
        """
        for e in self.elements:
            e.initialize_integration_points_secondary()

    def initialize_global_matrices_and_vectors(self):
        """Initialize global matrices and in terms of
        memory allocation and initial values.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to initialize the appropriate matrices and vectors.
        At this abstract level, it does nothing.
        """
        pass

    def initialize_free_index_arrays(self) -> None:
        """Initialize arrays for free index variables,
        in terms of memory allocation and initial values.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to initialize arrays that store indices of free variables
        (variables that are not constrained by boundary conditions).
        At this abstract level, it does nothing.
        """
        pass

    def initialize_solution_variable_vectors(self) -> None:
        """Initialize solution variable vectors for the analysis,
        in terms of memory allocation and initial values.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to initialize vectors that store the solution variables.
        At this abstract level, it does nothing.
        """
        pass

    def initialize_system_state_variables(self) -> None:
        """Initialize system state variables for adaptive time stepping,
        in terms of memory allocation and initial values.
        These are any variables that are processed by
        save_system_state() and load_system_state()
        that can then be used to reinitialize other solution variables.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to initialize the variables that store the state of the system.
        At this abstract level, it does nothing.
        """
        pass

    def save_system_state(self) -> None:
        """Save the current state of the system
        so that it can be loaded later.
        Used in adaptive time stepping algorithms.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to save the current state of the system variables.
        At this abstract level, it does nothing.
        """
        pass

    def load_system_state(self, t0: float, t1: float, dt: float) -> None:
        """Load the system state for a given time range and time step.
        This is a simple implementation, and normally treated as an
        abstract method to be overridden.

        Parameters
        ----------
        t0 : float
          The initial time of the system state to load.
        t1 : float
         The final time of the system state to load.
        dt : float
         The time step to load.

        Notes
        -----
        This convenience method sets the internal time tracking variables
        based on the provided initial and final times, as well as the time step.
        It is meant to be called when restoring a previously saved state
        to continue the simulation from a specific point in time.
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to load the previous state of the system variables.
        """
        self._t0 = t0
        self._t1 = t1
        self.time_step = dt

    def initialize_global_system(self, t0: float) -> None:
        """Sets up the global system before the first time step.

        Parameters
        ----------
        t0 : float
            The value of time (in seconds)
            at the beginning of the first time step

        Notes
        -----
        This convenience method is meant to be called once
        at the beginning of the analysis.
        It assumes that initial conditions have already been assigned
        to the nodes in the mesh.
        It initializes variables tracking the time coordinate,
        updates the boundary conditions at the initial time,
        assigns initial void ratio values from the nodes to the global vector,
        updates the integration points in the parent mesh,
        then updates all global vectors and matrices.
        """
        t0 = float(t0)
        self._t0 = t0
        self._t1 = t0
        self.initialize_free_index_arrays()
        self.initialize_system_state_variables()
        self.initialize_boundary_conditions()
        self.initialize_solution_variable_vectors()
        self.initialize_integration_points_primary()
        self.calculate_deformed_coords()
        self.update_total_stress_distribution()
        self.initialize_integration_points_secondary()
        self.update_pore_pressure_distribution()
        self.update_global_matrices_and_vectors()

    def initialize_time_step(self) -> None:
        """Sets up the system at the beginning of a time step.

        Notes
        -----
        This convenience method is meant to be called once
        at the beginning of each time step.
        It increments time stepping variables,
        saves global vectors and matrices from the end
        of the previous time step,
        updates boundary conditions
        and global water flux vector at the end of
        the current time step,
        and initializes iterative correction parameters.
        """
        # update time coordinate
        self._t0 = self._t1
        self._t1 = self._t0 + self.dt
        # initialize iteration parameters
        self._eps_a = 1.0
        self._iter = 0
        # setup global matrices and vectors
        self.store_converged_matrices()
        self.update_boundary_conditions(self._t1)
        self.update_nodes()
        self.update_integration_points_primary()
        self.calculate_deformed_coords()
        self.update_total_stress_distribution()
        self.update_integration_points_secondary()
        self.update_pore_pressure_distribution()
        self.update_global_matrices_and_vectors()

    def store_converged_matrices(self) -> None:
        """Store the converged tangent matrices.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to store the tangent matrices that have converged
        after an iterative correction process.
        At this abstract level, it does nothing.
        """
        pass

    def update_boundary_conditions(self, t: float) -> None:
        """Update the boundary conditions for the current time step.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to update the boundary conditions based on the current time step.
        At this abstract level, it does nothing.
        """
        pass

    def update_nodes(self) -> None:
        """Update the nodes for the current time step,
        by transferring solution variable values from global vectors
        to Node1D objects.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to update the nodal values
        based on the current time step and solution state.
        At this abstract level, it does nothing.
        """
        pass

    def update_integration_points_primary(self) -> None:
        """Updates the primary variables at integration points
        across the mesh.
        This is a default implementation,
        but can be overridden if there are additional
        details or modifications for a specific type of analysis.
        """
        for e in self.elements:
            e.update_integration_points_primary()

    def update_integration_points_secondary(self) -> None:
        """Updates the secondary variables at integration points
        across the mesh.
        This is a default implementation,
        but can be overridden if there are additional
        details or modifications for a specific type of analysis.
        """
        for e in self.elements:
            e.update_integration_points_secondary()

    def update_global_matrices_and_vectors(self) -> None:
        """Update global vectors for the current time step,
        by performing integrations of element vectors.
        These are typically the forcing function or flux vectors
        in the finite element analysis.
        Update global matrices for the current time step,
        for the initial stiffness method,
        by performing integrations of element matrices.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to update the global vectors and matrices
        based on the current time step and solution state.
        At this abstract level, it does nothing.
        """
        pass

    def calculate_solution_vector_correction(self) -> None:
        """Calculate the correction for the solution vector,
        based on the current residuals and global matrices.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to calculate the correction for the solution vector
        based on the current residuals and global matrices.
        At this abstract level, it does nothing.
        """
        pass

    def update_void_ratio_phase_change(self) -> None:
        """Update the void ratio due to phase change.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to update the void ratio at integration points
        based on phase changes (e.g. freezing/thawing).
        At this abstract level, it does nothing.
        """
        pass

    def iterative_correction_step(self) -> None:
        """Performs iterative correction of the
        global void ratio vector for a single time step.

        Notes
        -----
        This convenience method performs an iterative correction loop
        based on the implicit_error_tolerance and max_iterations properties.
        It iteratively updates the global matrices
        and performs correction of the global void ratio vector.
        """
        while self._eps_a > self.eps_s and self._iter < self.max_iterations:
            self.calculate_solution_vector_correction()
            self.update_nodes()
            self.update_integration_points_primary()
            self.calculate_deformed_coords()
            self.update_total_stress_distribution()
            self.update_integration_points_secondary()
            self.update_pore_pressure_distribution()
            self.update_global_matrices_and_vectors()
            self.update_iteration_variables()
        self.update_void_ratio_phase_change()
        self.update_integration_points_primary()
        self.calculate_deformed_coords()
        self.update_total_stress_distribution()
        self.update_integration_points_secondary()
        self.update_pore_pressure_distribution()
        self.update_global_matrices_and_vectors()

    def update_iteration_variables(self) -> None:
        """Update variables required for the next iteration.

        Notes
        -----
        This is a simple implementation,
        but should be treated as an abstract method.
        This method increments the iteration counter and updates it
        to next integer for the next iteration
        in the iterative correction loop.
        Meant to be called at the end of each iteration
        within the iterative_correction_step method.
        Overrides may also want to calculate approximate relative error
        updating self._eps_a based on solution variable increments.
        """
        self._iter += 1

    def _check_tf(self, tf: float) -> float:
        """Validate the target final time for the simulation.

        Parameters
        ----------
        tf : float
         The target final time.

        Returns
        -------
        float
         The validated target final time.

        Raises
        ------
        ValueError
          If final time cannot be converted to float
          or if final time is less or equal to current simulation time.

        Notes
        -----
        This method ensures that the provided target final time tf is valid
        and greater than the current simulation time.
        It is meant to be called before starting the time integration process.
        """
        tf = float(tf)
        if tf <= self._t1:
            raise ValueError(
                f"Provided tf {tf} is <= current simulation time {self._t1}."
            )
        return tf

    def solve_to(
        self,
        tf: float,
    ) -> tuple[
        float,
        npt.NDArray,
        npt.NDArray,
    ]:
        """Performs time integration until
        specified final time tf.

        Inputs
        ------
        tf : float
            The target final time.
        adapt_dt : bool, optional, default=True
            Flag for adaptive time step correction.

        Returns
        -------
        float
            The time step at the second last step.
            Last step will typically be adjusted to
            reach the target tf, so that time step is
            not necessarily meaningful.
        numpy.ndnarray, shape=(nstep, )
            The array of time steps over the interval
            up to tf.
        numpy.ndnarray, shape=(nstep, )
            The array of (relative) errors at each time
            step over the interval up to tf.

        Raises
        ------
        ValueError
            If tf cannot be converted to float.
            If tf <= current simulation time.
        """
        tf = self._check_tf(tf)
        dt_list = []
        err_list = []
        done = False
        while not done and self._t1 < tf:
            # check if time step passes tf
            dt00 = self.time_step
            if self._t1 + self.time_step > tf:
                self.time_step = tf - self._t1
                done = True
            # take single time step
            self.initialize_time_step()
            self.iterative_correction_step()
            dt_list.append(dt00)
            err_list.append(0.0)
        # reset time step and return output values
        self.time_step = dt00
        return dt00, np.array(dt_list), np.array(err_list)

    def calculate_total_settlement(self) -> float:
        """Calculate the total settlement of the mesh.
        Abstract method.

        Notes
        -----
        This convenience method computes the total settlement
        by integrating the displacements of the nodes in the mesh.
        It is meant to be called at the end of the analysis
        to provide a summary measure of the deformation.
        Returns the total settlement as a float value.
        At this level, it is an abstract method and just returns 0.0.
        In analysis types where total settlement is relevant,
        it should be overridden with the actual calculation.
        """
        return 0.0

    def calculate_deformed_coords(self) -> npt.NDArray[np.floating]:
        """Calculate the deformed coordinates of the nodes in the mesh.
        Abstract method.

        Notes
        -----
        This convenience method computes the deformed coordinates of the nodes
        by applying the calculated displacements.
        It is meant to be called after updating the nodal displacements
        to obtain the current deformed shape of the mesh.
        Returns an array of the deformed coordinates.
        At this level, it is an abstract method and just returns
        the original coordinates of the nodes.
        In analysis types where settlement is relevant,
        it should be overridden with the actual calculation.
        """
        return np.array(nd.z for nd in self.nodes)

    def update_total_stress_distribution(self) -> None:
        """Update the total stress distribution in the mesh.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to update the total stress at each node and integration point
        based on the current state of the system.
        At this abstract level, it does nothing.
        """
        pass

    def update_pore_pressure_distribution(self) -> None:
        """Update the pore pressure distribution in the mesh.
        Abstract method.

        Notes
        -----
        This method is meant to be overridden
        by a specific type of Mesh/Analysis class
        to update the pore pressure at each node and integration point
        based on the current state of the system.
        At this abstract level, it does nothing.
        """
        pass
