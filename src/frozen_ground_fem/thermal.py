"""thermal.py
Module for implementing thermal physics using the finite element method.

Classes
-------
ThermalElement1D
ThermalBoundary1D
ThermalAnalysis1D
"""
from typing import (
    Callable,
    Sequence,
)
from enum import Enum

import numpy as np
import numpy.typing as npt

from .materials import (
    vol_heat_cap_water as Cw,
    dens_ice as rho_i,
    latent_heat_fusion_water as Lw,
)
from .geometry import (
    Node1D,
    IntegrationPoint1D,
    Element1D,
    Boundary1D,
    Mesh1D,
)


class ThermalElement1D(Element1D):
    """Class for computing element matrices for thermal physics.

    Attributes
    ----------
    nodes
    order
    jacobian
    int_pts
    heat_flow_matrix
    heat_storage_matrix

    Methods
    -------
    update_integration_points

    Parameters
    ----------
    nodes : Sequence[Node1D]
        The ordered :c:`Node1D` that define the element.
    order : int, optional, default=3
        The order of interpolation to be used in the element.

    Raises
    ------
    TypeError:
        If nodes contains non-:c:`Node1D` objects.
    ValueError
        If len(nodes) is invalid for the order of interpolation.
        If order is not 1 or 3.
    """

    @property
    def heat_flow_matrix(self) -> npt.NDArray[np.floating]:
        """The element heat flow (conduction) matrix.

        Returns
        -------
        numpy.ndarray
            Shape depends on order of interpolation.
            For order=1, shape=(2, 2).
            For order=3, shape=(4, 4).

        Notes
        -----
        Integrates B^T * lambda * B over the element
        where lambda is the thermal conductivity.
        """
        B = self._gradient_matrix(0, 1)
        H = np.zeros_like(B.T @ B)
        jac = self.jacobian
        for ip in self.int_pts:
            e = ip.void_ratio
            e0 = ip.void_ratio_0
            e_fact = (1+e0) / (1+e)
            B = self._gradient_matrix(ip.local_coord, jac)
            N = self._shape_matrix(ip.local_coord)
            H += ((B.T * e_fact * ip.thrm_cond
                   + N.T * ip.water_flux_rate * Cw)
                  @ (ip.weight * e_fact * B))
        H *= jac
        return H

    @property
    def heat_storage_matrix(self) -> npt.NDArray[np.floating]:
        """The element heat storage matrix.

        Returns
        -------
        numpy.ndarray
            Shape depends on order of interpolation.
            For order=1, shape=(2, 2).
            For order=3, shape=(4, 4).

        Notes
        -----
        Integrates N^T * C * N over the element
        where C is the volumetric heat capacity.
        """
        N = self._shape_matrix(0)
        C = np.zeros_like(N.T @ N)
        jac = self.jacobian
        for ip in self.int_pts:
            ee = ip.void_ratio
            e_fact = ee / (1.0 + ee)
            N = self._shape_matrix(ip.local_coord)
            C += (
                (ip.vol_heat_cap
                 + Lw * rho_i * e_fact * ip.deg_sat_water_temp_gradient)
                * ip.weight
            ) * N.T @ N
        C *= jac
        return C

    def update_integration_points(self) -> None:
        """Updates the properties of integration points
        in the element according to changes in temperature.

        Notes
        -----
        This convenience method loops over the integration points
        and interpolates temperatures from corresponding nodes
        and updates degree of saturation of water accordingly.
        """
        Te = np.array([nd.temp for nd in self.nodes])
        dTdte = np.array([nd.temp_rate for nd in self.nodes])
        for ip in self.int_pts:
            N = self._shape_matrix(ip.local_coord)
            B = self._gradient_matrix(ip.local_coord, self.jacobian)
            T = (N @ Te)[0]
            dTdZ = (B @ Te)[0]
            dTdt = (N @ dTdte)[0]
            Sw, dSw_dT = ip.material.deg_sat_water(T)
            ip.temp = T
            ip.temp_gradient = dTdZ
            ip.temp_rate = dTdt
            ip.deg_sat_water = Sw
            ip.deg_sat_water_temp_gradient = dSw_dT
            if T < 0.0:
                qw = ip.material.water_flux(
                    ip.void_ratio,
                    ip.void_ratio_0,
                    ip.temp,
                    ip.temp_rate,
                    ip.temp_gradient,
                    ip.tot_stress,
                )
            else:
                qw = 0.0
            ip.water_flux_rate = qw


class ThermalBoundary1D(Boundary1D):
    """Class for storing and updating boundary conditions for thermal physics.

    Attributes
    ----------
    BoundaryType : enum.Enum
        The set of possible boundary condition types
    nodes
    int_pts
    bnd_type
    bnd_value
    bnd_function

    Methods
    -------
    update_nodes
    update_value

    Parameters
    ----------
    nodes : Sequence[Node1D]
        The :c:`Node1D` to assign to the boundary condition.
    int_pts : Sequence[IntegrationPoint1D], optional, default=()
        The :c:`IntegrationPoint1D` to assign to the boundary condition.
    bnd_type : ThermalBoundary1D.BoundaryType, optional,
                default=BoundaryType.temp
        The type of boundary condition.
    bnd_value : float, optional, default=0.0
        The value of the boundary condition.
    bnd_function : callable or None, optional, default=None
        The function for the updates the boundary condition.

    Raises
    ------
    TypeError
        If nodes contains non-:c:`Node1D` objects.
        If int_pts contains non-:c:`IntegrationPoint1D` objects.
        If bnd_type is not a ThermalBoundary1D.BoundaryType.
        If bnd_function is not callable or None.
    ValueError
        If len(nodes) != 1.
        If len(int_pts) > 1.
        If bnd_value is not convertible to float.
    """
    BoundaryType = Enum("BoundaryType", ["temp", "heat_flux", "temp_grad"])

    _bnd_type: BoundaryType
    _bnd_value: float = 0.0
    _bnd_function: Callable | None

    def __init__(
        self,
        nodes: Sequence[Node1D],
        int_pts: Sequence[IntegrationPoint1D] = (),
        bnd_type=BoundaryType.temp,
        bnd_value: float = 0.0,
        bnd_function: Callable | None = None,
    ):
        super().__init__(nodes, int_pts)
        self.bnd_type = bnd_type
        self.bnd_value = bnd_value
        self.bnd_function = bnd_function

    @property
    def bnd_type(self) -> BoundaryType:
        """The type of boundary condition.

        Parameters
        ----------
        ThermalBoundary1D.BoundaryType

        Returns
        -------
        ThermalBoundary1D.BoundaryType

        Raises
        ------
        TypeError
            If the value to be assigned
            is not a ThermalBoundary1D.BoundaryType.
        """
        return self._bnd_type

    @bnd_type.setter
    def bnd_type(self, value: BoundaryType):
        if not isinstance(value, ThermalBoundary1D.BoundaryType):
            raise TypeError(f"{value} is not a ThermalBoundary1D.BoundaryType")
        self._bnd_type = value

    @property
    def bnd_value(self) -> float:
        """The value of the boundary condition.

        Parameters
        ----------
        float

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to be assigned is not convertible to float.
        """
        return self._bnd_value

    @bnd_value.setter
    def bnd_value(self, value: float) -> None:
        value = float(value)
        self._bnd_value = value

    @property
    def bnd_function(self) -> Callable[[float], float] | None:
        """The reference to the function
        that updates the boundary condition.

        Parameters
        ----------
        Callable or None

        Returns
        -------
        Callable or None

        Raises
        ------
        TypeError
            If the value to be assigned is not callable or None.

        Notes
        -----
        If a callable (i.e. function or class that implements __call__)
        reference is provided it should take one argument
        which is a time (in seconds).
        This function is called by the method update_value().
        """
        return self._bnd_function

    @bnd_function.setter
    def bnd_function(
            self,
            value: Callable[[float], float] | None) -> None:
        if not (callable(value) or value is None):
            raise TypeError(
                f"type(value) {type(value)} is not callable or None")
        self._bnd_function = value

    def update_nodes(self) -> None:
        """Update the boundary condition value at the nodes.

        Notes
        -----
        This method updates the temperature at each of the nodes
        in the ThermalBoundary1D
        only in the case that bnd_type == BoundaryType.temp.
        Otherwise, it does nothing.
        """
        if self.bnd_type == ThermalBoundary1D.BoundaryType.temp:
            for nd in self.nodes:
                nd.temp = self.bnd_value

    def update_value(self, time: float) -> None:
        """Update the value of the boundary conditions.

        Parameters
        ----------
        float

        Raises
        ------
        ValueError
            If time is not convertible to float.

        Notes
        -----
        This method uses the bnd_function callable property
        to update the bnd_value property.
        If bnd_function is None
        the time argument is ignored and nothing happens.
        """
        time = float(time)
        if self.bnd_function is not None:
            self.bnd_value = self.bnd_function(time)


class ThermalAnalysis1D(Mesh1D):
    """Class for simulating thermal physics
    on a mesh of :c:`ThermalElement1D`.

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
    generate_mesh
    add_boundary
    remove_boundary
    clear_boundaries
    update_thermal_boundary_conditions
    update_heat_flux_vector
    update_heat_flow_matrix
    update_heat_storage_matrix
    update_nodes
    update_integration_points
    initialize_global_system
    initialize_time_step
    update_weighted_matrices
    calculate_temperature_correction
    iterative_correction_step

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
    _elements: tuple[ThermalElement1D, ...]
    _boundaries: set[ThermalBoundary1D]
    _time_step: float = 0.0
    _inv_time_step: float = 0.0
    _implicit_factor: float = 0.5   # Crank-Nicolson
    _inv_implicit_factor: float = 0.5
    _implicit_error_tolerance: float = 1e-3
    _max_iterations: int = 100
    _free_vec: tuple[npt.NDArray, ...]
    _free_arr: tuple[npt.NDArray, ...]
    _temp_vector_0: npt.NDArray[np.floating]
    _temp_vector: npt.NDArray[np.floating]
    _heat_flux_vector_0: npt.NDArray[np.floating]
    _heat_flux_vector: npt.NDArray[np.floating]
    _heat_flow_matrix_0: npt.NDArray[np.floating]
    _heat_flow_matrix: npt.NDArray[np.floating]
    _heat_storage_matrix_0: npt.NDArray[np.floating]
    _heat_storage_matrix: npt.NDArray[np.floating]
    _weighted_heat_flux_vector: npt.NDArray[np.floating]
    _weighted_heat_flow_matrix: npt.NDArray[np.floating]
    _weighted_heat_storage_matrix: npt.NDArray[np.floating]
    _coef_matrix_0: npt.NDArray[np.floating]
    _coef_matrix_1: npt.NDArray[np.floating]
    _residual_heat_flux_vector: npt.NDArray[np.floating]
    _delta_temp_vector: npt.NDArray[np.floating]
    _temp_rate_vector: npt.NDArray[np.floating]

    @property
    def elements(self) -> tuple[ThermalElement1D, ...]:
        """The tuple of :c:`ThermalElement1D` contained in the mesh.

        Returns
        ------
        tuple[:c:`ThermalElement1D`]

        Notes
        -----
        Overrides :c:`frozen_ground_fem.geometry.Mesh1D`
        property method for more specific return value
        type hint.
        """
        return self._elements

    @property
    def boundaries(self) -> set[ThermalBoundary1D]:
        """The tuple of :c:`ThermalBoundary1D` contained in the mesh.

        Returns
        ------
        set[:c:`ThermalBoundary1D`]

        Notes
        -----
        Overrides :c:`frozen_ground_fem.geometry.Mesh1D`
        property method for more specific return value
        type hint.
        """
        return self._boundaries

    def _generate_elements(self, num_elements: int, order: int):
        """Generate the elements in the mesh.

        Notes
        -----
        Overrides Mesh1D._generate_elements()
        to generate ThermalElement1D objects.
        """
        self._elements = tuple(
            ThermalElement1D(tuple(self.nodes[order * k + j]
                                   for j in range(order + 1)),
                             order)
            for k in range(num_elements)
        )

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
        return super().mesh_valid

    @mesh_valid.setter
    def mesh_valid(self, value: bool) -> None:
        value = bool(value)
        if value:
            # TODO: check for mesh validity
            self._mesh_valid = True
            # initialize global vectors and matrices
            self._temp_vector_0 = np.zeros(self.num_nodes)
            self._temp_vector = np.zeros(self.num_nodes)
            self._heat_flux_vector_0 = np.zeros(self.num_nodes)
            self._heat_flux_vector = np.zeros(self.num_nodes)
            self._heat_flow_matrix_0 = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._heat_flow_matrix = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._heat_storage_matrix_0 = np.zeros(
                (self.num_nodes, self.num_nodes)
            )
            self._heat_storage_matrix = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._weighted_heat_flux_vector = np.zeros(self.num_nodes)
            self._weighted_heat_flow_matrix = np.zeros(
                (self.num_nodes, self.num_nodes)
            )
            self._weighted_heat_storage_matrix = np.zeros(
                (self.num_nodes, self.num_nodes)
            )
            self._coef_matrix_0 = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._coef_matrix_1 = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._residual_heat_flux_vector = np.zeros(self.num_nodes)
            self._delta_temp_vector = np.zeros(self.num_nodes)
            self._temp_rate_vector = np.zeros(self.num_nodes)
        else:
            self._nodes = ()
            self._elements = ()
            self.clear_boundaries()
            self._mesh_valid = False

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
    def time_step(self, value):
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
                f"type(max_iterations) {type(value)} invalid, " + "must be int"
            )
        if value <= 0:
            raise ValueError(f"max_iterations {value} invalid,"
                             + " must be positive")
        self._max_iterations = value

    def add_boundary(self, new_boundary: ThermalBoundary1D) -> None:
        """Adds a boundary to the mesh.

        Parameters
        ----------
        new_boundary : :c:`ThermalBoundary1D`
            The boundary to add to the mesh.

        Raises
        ------
        TypeError
            If new_boundary is not an instance of :c:`ThermalBoundary1D`.
        ValueError
            If new_boundary contains a :c:`Node1D` not in the mesh.
            If new_boundary contains an :c:`IntegrationPoint1D`
                not in the mesh.
        """
        if not isinstance(new_boundary, ThermalBoundary1D):
            raise TypeError(
                f"type(new_boundary) {type(new_boundary)} invalid, "
                + "must be ThermalBoundary1D"
            )
        for nd in new_boundary.nodes:
            if nd not in self.nodes:
                raise ValueError(f"new_boundary contains node {nd}"
                                 + " not in mesh")
        if new_boundary.int_pts:
            int_pts = tuple(ip for e in self.elements for ip in e.int_pts)
            for ip in new_boundary.int_pts:
                if ip not in int_pts:
                    raise ValueError(
                        f"new_boundary contains int_pt {ip} not in mesh"
                    )
        self._boundaries.add(new_boundary)

    def update_thermal_boundary_conditions(
            self,
            time: float) -> None:
        """Update the thermal boundary conditions in the ThermalAnalysis1D
        and in the parent Mesh1D.

        Parameters
        ----------
        time : float
            The time in seconds.
            Gets passed through to ThermalBoundary1D.update_value().

        Notes
        -----
        This convenience methods
        loops over all ThermalBoundary1D objects in boundaries
        and calls update_value() to update the boundary value
        and then calls update_nodes() to assign the new value
        to each boundary Node1D.
        For Dirichlet (temperature) boundary conditions,
        the value is then assigned to the global temperature vector
        in the ThermalAnalysis1D object.
        """
        for tb in self.boundaries:
            tb.update_value(time)
            tb.update_nodes()
            if tb.bnd_type == ThermalBoundary1D.BoundaryType.temp:
                for nd in tb.nodes:
                    self._temp_vector[nd.index] = nd.temp

    def update_heat_flux_vector(self) -> None:
        """Updates the global heat flux vector.

        Notes
        -----
        This convenience method clears the global heat flux vector
        then loops over the boundaries and
        assigns values for flux and gradient type boundaries
        to the global heat flux vector.
        """
        self._heat_flux_vector[:] = 0.0
        for be in self.boundaries:
            if be.bnd_type == ThermalBoundary1D.BoundaryType.heat_flux:
                self._heat_flux_vector[be.nodes[0].index] += be.bnd_value
            elif be.bnd_type == ThermalBoundary1D.BoundaryType.temp_grad:
                if not be.int_pts:
                    raise AttributeError(f"boundary {be} has no int_pts")
                self._heat_flux_vector[be.nodes[0].index] += (
                    -be.int_pts[0].thrm_cond * be.bnd_value
                )

    def update_heat_flow_matrix(self) -> None:
        """Updates the global heat flow matrix.

        Notes
        -----
        This convenience method first clears the global heat flow matrix
        then loops over the elements
        to get the element heat flow matrices
        and sums them into the global heat flow matrix
        respecting connectivity of global degrees of freedom.
        """
        self._heat_flow_matrix[:, :] = 0.0
        for e in self.elements:
            ind = np.array([nd.index for nd in e.nodes], dtype=int)
            He = e.heat_flow_matrix
            self._heat_flow_matrix[np.ix_(ind, ind)] += He

    def update_heat_storage_matrix(self) -> None:
        """Updates the global heat storage matrix.

        Notes
        -----
        This convenience method clears the global heat storage matrix
        then loops over the elements
        to get the element heat storage matrices
        and sums them into the global heat storage matrix
        respecting connectivity of global degrees of freedom.
        """
        self._heat_storage_matrix[:, :] = 0.0
        for e in self.elements:
            ind = np.array([nd.index for nd in e.nodes], dtype=int)
            Ce = e.heat_storage_matrix
            self._heat_storage_matrix[np.ix_(ind, ind)] += Ce

    def update_nodes(self) -> None:
        """Updates the temperature values at the nodes
        in the mesh.
        #
        # Notes
        # -----
        # This convenience method loops over nodes in the mesh
        # and assigns the temperature from the global temperature vector
        # in the ThermalAnalysis1D.
        """
        for nd in self.nodes:
            nd.temp = self._temp_vector[nd.index]
            nd.temp_rate = self._temp_rate_vector[nd.index]

    def update_integration_points(self) -> None:
        """Updates the properties of integration points
        in the mesh according to changes in temperature.
        """
        for e in self.elements:
            e.update_integration_points()

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
        updates the thermal boundary conditions at the initial time,
        assigns initial temperature values from the nodes to the global time
        vector,
        updates the integration points in the mesh,
        then updates all global vectors and matrices.
        """
        # initialize global time
        t0 = float(t0)
        self._t0 = t0
        self._t1 = t0
        # update nodes with boundary conditions first
        self.update_thermal_boundary_conditions(self._t0)
        # now get the temperatures from the nodes
        # (we assume that initial conditions have already been applied)
        for nd in self.nodes:
            self._temp_vector[nd.index] = nd.temp
            self._temp_vector_0[nd.index] = nd.temp
            self._temp_rate_vector[nd.index] = nd.temp_rate
        # now build the global matrices and vectors
        self.update_integration_points()
        self.update_heat_flux_vector()
        self.update_heat_flow_matrix()
        self.update_heat_storage_matrix()
        # create list of free node indices
        # that will be updated at each iteration
        # (i.e. are not fixed/Dirichlet boundary conditions)
        free_ind_list = [nd.index for nd in self.nodes]
        for tb in self.boundaries:
            if tb.bnd_type == ThermalBoundary1D.BoundaryType.temp:
                free_ind_list.remove(tb.nodes[0].index)
        free_ind = np.array(free_ind_list, dtype=int)
        self._free_vec = np.ix_(free_ind)
        self._free_arr = np.ix_(free_ind, free_ind)

    def initialize_time_step(self) -> None:
        """Sets up the system at the beginning of a time step.

        Notes
        -----
        This convenience method is meant to be called once
        at the beginning of each time step.
        It increments time stepping variables,
        saves global vectors and matrices from the end
        of the previous time step,
        updates thermal boundary conditions
        and global heat flux vector at the end of
        the current time step,
        and initializes iterative correction parameters.
        """
        # update time coordinate
        self._t0 = self._t1
        self._t1 = self._t0 + self.dt
        # store previous converged matrices and vectors
        self._temp_vector_0[:] = self._temp_vector[:]
        self._heat_flux_vector_0[:] = self._heat_flux_vector[:]
        self._heat_flow_matrix_0[:, :] = self._heat_flow_matrix[:, :]
        self._heat_storage_matrix_0[:, :] = self._heat_storage_matrix[:, :]
        # update boundary conditions
        self.update_thermal_boundary_conditions(self._t1)
        self.update_heat_flux_vector()
        self._weighted_heat_flux_vector[:] = (
            self.one_minus_alpha * self._heat_flux_vector_0
            + self.alpha * self._heat_flux_vector
        )
        # initialize iteration parameters
        self._eps_a = 1.0
        self._iter = 0

    def update_weighted_matrices(self) -> None:
        """Updates global weighted matrices
        according to implicit time stepping factor.

        Notes
        -----
        This convenience method updates
        the weighted heat flow matrix,
        the weighted heat storage matrix,
        and coefficient matrices
        using the implicit_factor property.
        """
        self._weighted_heat_flux_vector[:] = (
            self.one_minus_alpha * self._heat_flux_vector_0
            + self.alpha * self._heat_flux_vector
        )
        self._weighted_heat_flow_matrix[:, :] = (
            self.one_minus_alpha * self._heat_flow_matrix_0
            + self.alpha * self._heat_flow_matrix
        )
        self._weighted_heat_storage_matrix[:, :] = (
            self.one_minus_alpha * self._heat_storage_matrix_0
            + self.alpha * self._heat_storage_matrix
        )
        self._coef_matrix_0[:, :] = (
            self._weighted_heat_storage_matrix * self.over_dt
            - self.one_minus_alpha * self._weighted_heat_flow_matrix
        )
        self._coef_matrix_1[:, :] = (
            self._weighted_heat_storage_matrix * self.over_dt
            + self.alpha * self._weighted_heat_flow_matrix
        )

    def calculate_temperature_correction(self) -> None:
        """Performs a single iteration of temperature correction
        in the implicit time stepping scheme.

        Notes
        -----
        This convenience method
        updates the global residual heat flux vector,
        calculates the temperature correction
        using the global weighted matrices,
        applies the correction to the global temperature vector,
        then updates the nodes, integration points,
        and the global heat flux and heat storage matrices.
        """
        # update residual vector
        self._residual_heat_flux_vector[:] = (
            self._coef_matrix_0 @ self._temp_vector_0
            - self._coef_matrix_1 @ self._temp_vector
            - self._weighted_heat_flux_vector
        )
        # calculate temperature increment
        self._delta_temp_vector[self._free_vec] = np.linalg.solve(
            self._coef_matrix_1[self._free_arr],
            self._residual_heat_flux_vector[self._free_vec],
        )
        # increment temperature and iteration variables
        self._temp_vector[self._free_vec] += (
            self._delta_temp_vector[self._free_vec]
        )
        # update temp rate vector
        self._temp_rate_vector[:] = (
            self._temp_vector[:] - self._temp_vector_0[:]
        ) / self.dt
        self._eps_a = float(
            np.linalg.norm(self._delta_temp_vector) /
            np.linalg.norm(self._temp_vector)
        )
        self._iter += 1
        # update global system
        self.update_nodes()
        self.update_integration_points()
        self.update_heat_flux_vector()
        self.update_heat_flow_matrix()
        self.update_heat_storage_matrix()

    def iterative_correction_step(self) -> None:
        """Performs iterative correction of the
        global temperature vector for a single time step.

        Notes
        -----
        This convenience method performs an iterative correction loop
        based on the implicit_error_tolerance and max_iterations properties.
        It iteratively updates the global weighted matrices
        and performs correction of the global temperature vector.
        This method does not update the global heat flux vector
        since it assumes this is updated only once at the beginning
        of a new time step by initialize_time_step().
        """
        while self._eps_a > self.eps_s and self._iter < self.max_iterations:
            self.update_weighted_matrices()
            self.calculate_temperature_correction()

    def solve_to(
            self,
            tf: float,
            adapt_dt: bool = True,
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

        Notes
        -----
        By default, the method performs adaptive time step correction
        using the half-step algorithm. Correction is performed based
        on the error estimate, but steps are not repeated if error is
        exceeded for numerical efficiency. Target relative error is
        set based on the implicit_error_tolerance attribute.
        If adaptive correction is not performed, then error is not
        estimated and the error array that is returned is not meaningful.
        """
        tf = float(tf)
        if tf <= self._t1:
            raise ValueError(
                f"Provided tf {tf} is <= current "
                f"simulation time {self._t1}."
            )
        # flag to ensure analysis completes at tf
        # to within roundoff error
        done = False
        # simplified loop if not performing
        # adaptive correction
        if not adapt_dt:
            dt_list = []
            err_list = []
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
        # initialize vectors and matrices
        # for adaptive step size correction
        num_int_pt_per_element = len(self.elements[0].int_pts)
        temp_vector_0 = np.zeros_like(self._temp_vector)
        temp_vector_1 = np.zeros_like(self._temp_vector)
        temp_error = np.zeros_like(self._temp_vector)
        temp_rate = np.zeros_like(self._temp_vector)
        temp_scale = np.zeros_like(self._temp_vector)
        dt_list = []
        err_list = []
        while not done and self._t1 < tf:
            # check if time step passes tf
            dt00 = self.time_step
            if self._t1 + self.time_step > tf:
                self.time_step = tf - self._t1
                done = True
            # save system state before time step
            t0 = self._t1
            dt0 = self.time_step
            temp_vector_0[:] = self._temp_vector[:]
            # take single time step
            self.initialize_time_step()
            self.iterative_correction_step()
            # save the predictor result
            temp_vector_1[:] = self._temp_vector[:]
            # reset the system
            self._temp_vector[:] = temp_vector_0[:]
            self.update_nodes()
            self.update_integration_points()
            self.update_heat_flux_vector()
            self.update_heat_flow_matrix()
            self.update_heat_storage_matrix()
            self._t1 = t0
            # take two half steps
            self.time_step = 0.5 * dt0
            self.initialize_time_step()
            self.iterative_correction_step()
            self.initialize_time_step()
            self.iterative_correction_step()
            # compute truncation error correction
            temp_error[:] = (self._temp_vector[:]
                             - temp_vector_1[:]) / 3.0
            self._temp_vector[:] += temp_error[:]
            self.update_nodes()
            self.update_integration_points()
            self.update_heat_flux_vector()
            self.update_heat_flow_matrix()
            self.update_heat_storage_matrix()
            # update the time step
            temp_rate[:] = (self._temp_vector[:]
                            - temp_vector_0[:])
            temp_scale[:] = np.max(np.vstack([
                self._temp_vector[:],
                temp_rate,
            ]), axis=0)
            T_scale = float(np.linalg.norm(temp_scale))
            err_targ = self.eps_s * T_scale
            err_curr = float(np.linalg.norm(temp_error))
            # update the time step
            eps_a = err_curr / T_scale
            dt1 = dt0 * (err_targ / err_curr) ** 0.2
            self.time_step = dt1
            dt_list.append(dt0)
            err_list.append(eps_a)
        return dt00, np.array(dt_list), np.array(err_list)
