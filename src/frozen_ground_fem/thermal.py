"""thermal.py
Module for implementing thermal physics using the finite element method.

Classes
-------
ThermalElement1D
ThermalBoundary1D
ThermalAnalysis1D
"""

__all__ = [
    "ThermalBoundary1D",
    "ThermalElement1D",
    "ThermalAnalysis1D",
]

from typing import (
    Callable,
    Sequence,
)
from enum import Enum

import numpy as np
import numpy.typing as npt

from . import (
    vol_heat_cap_water as Cw,
    dens_ice as rho_i,
    latent_heat_fusion_water as Lw,
    Node1D,
    IntegrationPoint1D,
)
from .geometry import (
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
            e_fact = ((1 + e0) / (1 + e)) ** 2
            B = self._gradient_matrix(ip.local_coord, jac)
            H += (e_fact * ip.thrm_cond * ip.weight) * (B.T @ B)
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
            lat_heat = Lw * rho_i * ip.vol_water_cont_temp_gradient
            N = self._shape_matrix(ip.local_coord)
            C += ip.weight * (ip.vol_heat_cap + lat_heat) * (N.T @ N)
        C *= jac * 2.0
        return C

    @property
    def heat_flux_vector(self) -> npt.NDArray[np.floating]:
        """The element heat flux vector (for advection and latent heat).

        Returns
        -------
        numpy.ndarray
            Shape depends on order of interpolation.
            For order=1, shape=(2, ).
            For order=3, shape=(4, ).
        """
        N = self._shape_matrix(0.0)
        Q = np.zeros(self.order + 1)
        jac = self.jacobian
        for ip in self.int_pts:
            e = ip.void_ratio
            e0 = ip.void_ratio_0
            e_fact = (1 + e0) / (1 + e)
            qw = ip.water_flux_rate
            dTdZ = ip.temp_gradient
            N = self._shape_matrix(ip.local_coord).flatten()
            Q -= N * (e_fact * qw * Cw * dTdZ) * ip.weight
        Q *= jac
        return Q

    def initialize_integration_points_primary(self) -> None:
        """Initializes values of thermal primary solution variables
        (and variables not affected by coupling)
        at the element integration points.

        Notes
        -----
        Initializes
        temperature,
        temperature gradient,
        temperature rate,
        and degree of saturation of water.
        Temperature is a primary solution variable and degree of
        saturation is needed for total stress calculation, but is not
        affected by coupling (if present). Temperature gradient (in
        Lagrangian coordinates) and temperature rate are also not affected
        by coupling, so are safe to update here.
        Also initializes temp__0 and vol_water_cont__0 to same as
        temp and vol_water_cont, but allows opportunity for modifying
        these values before calling initialize_integration_points_secondary().
        """
        ee0 = np.array([nd.void_ratio_0 for nd in self.nodes])
        for ip in self.int_pts:
            N = self._shape_matrix(ip.local_coord)
            ip.void_ratio_0 = (N @ ee0)[0]
        for iipp in self._int_pts_deformed:
            for ip in iipp:
                N = self._shape_matrix(ip.local_coord)
                ip.void_ratio_0 = (N @ ee0)[0]
        ThermalElement1D.update_integration_points_primary(self)
        for ip in self.int_pts:
            ip.temp__0 = ip.temp
            ip.vol_water_cont__0 = ip.vol_water_cont
        for nd in self.nodes:
            nd.deg_sat_water__0 = nd.deg_sat_water

    def initialize_integration_points_secondary(self) -> None:
        """Initializes values of thermal secondary solution variables
        (variables affected by coupling)
        at the element integration points.

        Notes
        -----
        Initializes
        volumetric water content temperature gradient
        and water flux rate (for frozen points).
        Volumetric water content temperature gradient
        is affected by coupling because it depends on
        both degree of saturation and porosity (void ratio).
        Water flux rate for frozen points
        is affected by coupling because it depends on
        total stress and the void ratio correction factor
        for space derivatives.
        """
        ThermalElement1D.update_integration_points_secondary(self)

    def update_integration_points_primary(self) -> None:
        """Updates values of thermal primary solution variables
        (and variables not affected by coupling)
        at the element integration points.

        Notes
        -----
        Updates
        temperature,
        temperature gradient,
        temperature rate,
        and degree of saturation of water.
        Temperature is a primary solution variable and degree of
        saturation is needed for total stress calculation, but is not
        affected by coupling (if present). Temperature gradient (in
        Lagrangian coordinates) and temperature rate are also not affected
        by coupling, so are safe to update here.
        """
        self.update_deg_sat_water_nodes()
        ee = np.array([nd.void_ratio for nd in self.nodes])
        Te = np.array([nd.temp for nd in self.nodes])
        dTdte = np.array([nd.temp_rate for nd in self.nodes])
        jac = self.jacobian
        for ip in self.int_pts:
            N = self._shape_matrix(ip.local_coord)
            B = self._gradient_matrix(ip.local_coord, jac)
            T = (N @ Te)[0]
            ip.void_ratio = (N @ ee)[0]
            ip.temp = T
            ip.temp_gradient = (B @ Te)[0]
            ip.temp_rate = (N @ dTdte)[0]
            ip.deg_sat_water = ip.material.deg_sat_water(T)[0]
        for iipp in self._int_pts_deformed:
            for ip in iipp:
                N = self._shape_matrix(ip.local_coord)
                T = (N @ Te)[0]
                ip.void_ratio = (N @ ee)[0]
                ip.temp = T
                ip.deg_sat_water = ip.material.deg_sat_water(T)[0]

    def update_integration_points_secondary(self) -> None:
        """Updates values of thermal secondary solution variables
        (variables affected by coupling)
        at the element integration points.

        Notes
        -----
        Updates
        volumetric water content temperature gradient
        and water flux rate (for frozen points).
        Volumetric water content temperature gradient
        is affected by coupling because it depends on
        both degree of saturation and porosity (void ratio).
        Water flux rate for frozen points
        is affected by coupling because it depends on
        total stress and the void ratio correction factor
        for space derivatives.
        """
        for ip in self.int_pts:
            T = ip.temp
            T0 = ip.temp__0
            thw0 = ip.vol_water_cont__0
            thw1 = ip.vol_water_cont
            ip.vol_water_cont_temp_gradient = (
                np.abs((thw1 - thw0) / (T - T0)) if np.abs(T - T0) > 0.0 else 0.0
            )
            if T < 0.0:
                ip.water_flux_rate = ip.material.water_flux(
                    ip.void_ratio,
                    ip.void_ratio_0,
                    T,
                    ip.temp_rate,
                    ip.temp_gradient,
                    ip.tot_stress,
                )

    def update_deg_sat_water_nodes(self) -> None:
        """Updates the degree of saturation of water at the nodes,
        based on the current temperature and material parameters.
        """
        m = self.int_pts[0].material
        for nd in self.nodes:
            nd.deg_sat_water = m.deg_sat_water(nd.temp)[0]


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
    _bnd_function: Callable[[float], float] | None

    def __init__(
        self,
        nodes: Sequence[Node1D],
        int_pts: Sequence[IntegrationPoint1D] = (),
        bnd_type=BoundaryType.temp,
        bnd_value: float = 0.0,
        bnd_function: Callable[[float], float] | None = None,
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
        Callable[[float], float] or None

        Returns
        -------
        Callable[[float], float] or None

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
    def bnd_function(self, value: Callable[[float], float] | None) -> None:
        if not (callable(value) or value is None):
            raise TypeError(f"type(value) {type(value)} is not callable or None")
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
    initialize_solution_variable_vectors
    initialize_free_index_arrays
    initialize_global_matrices_and_vectors
    initialize_global_system
    initialize_time_step
    update_boundary_conditions
    update_boundary_vectors
    update_nodes
    update_integration_points
    update_heat_flux_vector
    update_heat_flow_matrix
    update_heat_storage_matrix
    update_global_matrices_and_vectors
    store_converged_matrices
    calculate_solution_vector_correction
    iterative_correction_step
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

    _elements: tuple[ThermalElement1D, ...]
    _boundaries: set[ThermalBoundary1D]
    _temp_vector_0: npt.NDArray[np.floating]
    _temp_vector: npt.NDArray[np.floating]
    _heat_flux_vector_0: npt.NDArray[np.floating]
    _heat_flux_vector: npt.NDArray[np.floating]
    _heat_flow_matrix_0: npt.NDArray[np.floating]
    _heat_flow_matrix: npt.NDArray[np.floating]
    _heat_storage_matrix_0: npt.NDArray[np.floating]
    _heat_storage_matrix: npt.NDArray[np.floating]
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
        """The set of :c:`ThermalBoundary1D` contained in the mesh.

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
                raise ValueError(f"new_boundary contains node {nd}" + " not in mesh")
        if new_boundary.int_pts:
            int_pts = tuple(ip for e in self.elements for ip in e.int_pts)
            for ip in new_boundary.int_pts:
                if ip not in int_pts:
                    raise ValueError(f"new_boundary contains int_pt {ip} not in mesh")
        self._boundaries.add(new_boundary)

    def _generate_elements(self, num_elements: int, order: int):
        """Generate the elements in the mesh.

        Notes
        -----
        Overrides Mesh1D._generate_elements()
        to generate ThermalElement1D objects.
        """
        self._elements = tuple(
            ThermalElement1D(
                tuple(self.nodes[order * k + j] for j in range(order + 1)), order
            )
            for k in range(num_elements)
        )

    def initialize_global_matrices_and_vectors(self):
        """Initializes the global matrices and vectors for thermal analysis."""
        self._temp_vector_0 = np.zeros(self.num_nodes)
        self._temp_vector = np.zeros(self.num_nodes)
        self._heat_flux_vector_0 = np.zeros(self.num_nodes)
        self._heat_flux_vector = np.zeros(self.num_nodes)
        self._heat_flow_matrix_0 = np.zeros((self.num_nodes, self.num_nodes))
        self._heat_flow_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self._heat_storage_matrix_0 = np.zeros((self.num_nodes, self.num_nodes))
        self._heat_storage_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self._residual_heat_flux_vector = np.zeros(self.num_nodes)
        self._delta_temp_vector = np.zeros(self.num_nodes)
        self._temp_rate_vector = np.zeros(self.num_nodes)

    def initialize_free_index_arrays(self) -> None:
        """Initializes the arrays of free node indices for thermal analysis.

        Notes
        -----
        This method creates a list of indices for nodes that will be updated
        at each iteration, excluding those with primary (Dirichlet) boundary conditions.
        It then converts this into open meshes (using numpy.ix_())
        to be used for updating vectors and matrices.
        """
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

    def initialize_solution_variable_vectors(self) -> None:
        """Initializes the solution variable vectors for thermal analysis.

        Notes
        -----
        This method initializes the global temperature vector,
        the previous time step temperature vector,
        and the temperature rate vector.
        It assigns the initial temperature and temperature rate values from the nodes
        to the corresponding positions in the global vectors.
        """
        for nd in self.nodes:
            self._temp_vector[nd.index] = nd.temp
            self._temp_vector_0[nd.index] = nd.temp
            self._temp_rate_vector[nd.index] = nd.temp_rate

    def store_converged_matrices(self) -> None:
        """Stores the converged solution vectors for thermal analysis.

        Notes
        -----
        This method updates the previous time step temperature vector
        and the previous time step residual heat flux vector
        with the current values.
        It also stores the degree of saturation of water for
        each node and the temperature and volumetric water content
        for each integration point in the elements.
        """
        self._temp_vector_0[:] = self._temp_vector[:]
        self._heat_flux_vector_0[:] = self._heat_flux_vector[:]
        self._heat_flow_matrix_0[:, :] = self._heat_flow_matrix[:, :]
        self._heat_storage_matrix_0[:, :] = self._heat_storage_matrix[:, :]
        for nd in self.nodes:
            nd.deg_sat_water__0 = nd.deg_sat_water
        for e in self.elements:
            for ip in e.int_pts:
                ip.temp__0 = ip.temp
                ip.vol_water_cont__0 = ip.vol_water_cont

    def update_boundary_conditions(self, time: float) -> None:
        """Update the thermal boundary conditions.

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

    def update_nodes(self) -> None:
        """Updates the temperature values at the nodes
        in the mesh.

        Notes
        -----
        This convenience method loops over nodes in the mesh
        and assigns the temperature from the global temperature vector.
        """
        self._temp_rate_vector[:] = (
            self._temp_vector[:] - self._temp_vector_0[:]
        ) * self.over_dt
        for nd in self.nodes:
            nd.temp = self._temp_vector[nd.index]
            nd.temp_rate = self._temp_rate_vector[nd.index]

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
        for e in self.elements:
            ind = np.array([nd.index for nd in e.nodes], dtype=int)
            Qe = e.heat_flux_vector
            self._heat_flux_vector[np.ix_(ind)] += Qe
        for be in self.boundaries:
            if be.bnd_type == ThermalBoundary1D.BoundaryType.heat_flux:
                self._heat_flux_vector[be.nodes[0].index] -= be.bnd_value
            elif be.bnd_type == ThermalBoundary1D.BoundaryType.temp_grad:
                if not be.int_pts:
                    raise AttributeError(f"boundary {be} has no int_pts")
                self._heat_flux_vector[be.nodes[0].index] -= (
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

    def update_global_matrices_and_vectors(self) -> None:
        """Updates the global vectors and matrices for thermal analysis."""
        self.update_heat_flux_vector()
        self.update_heat_flow_matrix()
        self.update_heat_storage_matrix()

    def calculate_solution_vector_correction(self) -> None:
        """Performs a single iteration of temperature correction
        in the implicit time stepping scheme.

        Notes
        -----
        This convenience method
        updates the global residual heat flux vector,
        calculates the temperature correction,
        and applies the correction to the global temperature vector.
        """
        # update the residual vector
        self._residual_heat_flux_vector[:] = (
            self.one_minus_alpha
            * self.dt
            * np.linalg.solve(
                self._heat_storage_matrix_0,
                self._heat_flux_vector_0
                - self._heat_flow_matrix_0 @ self._temp_vector_0,
            )
            + self.alpha
            * self.dt
            * np.linalg.solve(
                self._heat_storage_matrix,
                self._heat_flux_vector - self._heat_flow_matrix @ self._temp_vector,
            )
            - (self._temp_vector[:] - self._temp_vector_0[:])
        )
        # calculate temperature increment
        self._delta_temp_vector[self._free_vec] = np.linalg.solve(
            np.eye(self.num_nodes)[self._free_arr]
            + self.alpha
            * self.dt
            * np.linalg.solve(self._heat_storage_matrix, self._heat_flow_matrix)[
                self._free_arr
            ],
            self._residual_heat_flux_vector[self._free_vec],
        )
        # increment temperature and iteration variables
        self._temp_vector[self._free_vec] += self._delta_temp_vector[self._free_vec]

    def update_iteration_variables(self) -> None:
        """Updates the iteration variables
        (relative error and iteration counter)
        for thermal analysis.
        """
        self._eps_a = float(
            np.linalg.norm(self._delta_temp_vector) / np.linalg.norm(self._temp_vector)
        )
        self._iter += 1

    def initialize_system_state_variables(self) -> None:
        """Initializes the system state variables
        (vectors and matrices)
        for adaptive time step correction.
        """
        # initialize vectors and matrices
        # for adaptive step size correction
        num_int_pt_per_element = len(self.elements[0].int_pts)
        self._temp_vector_0_0 = np.zeros_like(self._temp_vector)
        self._temp_vector_0_1 = np.zeros_like(self._temp_vector)
        self._temp_vector_1 = np.zeros_like(self._temp_vector)
        self._temp_error = np.zeros_like(self._temp_vector)
        self._deg_sat_water_0_0 = np.zeros_like(self._temp_vector)
        self._deg_sat_water_0_1 = np.zeros_like(self._temp_vector)
        self._temp_scale = np.zeros_like(self._temp_vector)
        self._vol_water_cont__0 = np.zeros(
            (
                self.num_elements,
                num_int_pt_per_element,
            )
        )
        self._temp__0 = np.zeros(
            (
                self.num_elements,
                num_int_pt_per_element,
            )
        )

    def save_system_state(self) -> None:
        """Saves the current system state for thermal analysis."""
        self._temp_vector_0_0[:] = self._temp_vector_0[:]
        self._temp_vector_0_1[:] = self._temp_vector[:]
        self._deg_sat_water_0_0[:] = np.array(
            [nd.deg_sat_water__0 for nd in self.nodes]
        )
        self._deg_sat_water_0_1[:] = np.array([nd.deg_sat_water for nd in self.nodes])
        for ke, e in enumerate(self.elements):
            for jip, ip in enumerate(e.int_pts):
                self._temp__0[ke, jip] = ip.temp__0
                self._vol_water_cont__0[ke, jip] = ip.vol_water_cont__0

    def load_system_state(self, t0: float, t1: float, dt: float) -> None:
        """Loads the saved system state for thermal analysis."""
        self._t0 = t0
        self._t1 = t1
        self.time_step = dt
        self._temp_vector_0[:] = self._temp_vector_0_0[:]
        self._temp_vector[:] = self._temp_vector_0_1[:]
        for nd, Sw0, Sw1 in zip(
            self.nodes,
            self._deg_sat_water_0_0,
            self._deg_sat_water_0_1,
        ):
            nd.deg_sat_water__0 = Sw0
            nd.deg_sat_water = Sw1
        for e, T0e, thw0_e in zip(
            self.elements,
            self._temp__0,
            self._vol_water_cont__0,
        ):
            for ip, T0, thw0 in zip(
                e.int_pts,
                T0e,
                thw0_e,
            ):
                ip.temp__0 = T0
                ip.vol_water_cont__0 = thw0
        self.update_nodes()
        self.update_integration_points_primary()
        self.calculate_deformed_coords()
        self.update_total_stress_distribution()
        self.update_integration_points_secondary()
        self.update_pore_pressure_distribution()
        self.update_global_matrices_and_vectors()

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
        tf = self._check_tf(tf)
        if not adapt_dt:
            return Mesh1D.solve_to(self, tf)
        dt_list = []
        err_list = []
        done = False
        k_try = 0
        k_try_max = 3
        while not done and self._t1 < tf:
            # check if time step passes tf
            dt00 = self.time_step
            if self._t1 + self.time_step > tf:
                self.time_step = tf - self._t1
                done = True
            # save system state before time step
            t0 = self._t0
            t1 = self._t1
            dt0 = self.time_step
            self.save_system_state()
            try:
                # take single time step
                self.initialize_time_step()
                self.iterative_correction_step()
                self._temp_vector_1[:] = self._temp_vector[:]
                self.load_system_state(t0, t1, dt0)
                # take two half steps
                self.time_step = 0.5 * dt0
                self.initialize_time_step()
                self.iterative_correction_step()
                self.initialize_time_step()
                self.iterative_correction_step()
                # compute truncation error correction
                self._temp_error[:] = (
                    self._temp_vector[:] - self._temp_vector_1[:]
                ) / 3.0
                self._temp_vector[:] += self._temp_error[:]
                self.update_nodes()
                self.update_integration_points_primary()
                self.calculate_deformed_coords()
                self.update_total_stress_distribution()
                self.update_integration_points_secondary()
                self.update_pore_pressure_distribution()
                self.update_global_matrices_and_vectors()
                # update the time step
                self._temp_scale[:] = np.max(
                    np.vstack(
                        [
                            self._temp_vector[:],
                            self._temp_rate_vector[:] * self.time_step,
                        ]
                    ),
                    axis=0,
                )
                T_scale = float(np.linalg.norm(self._temp_scale))
                err_targ = self.eps_s * T_scale
                err_curr = float(np.linalg.norm(self._temp_error))
                # update the time step
                eps_a = err_curr / T_scale
                dt1 = dt0 * (err_targ / err_curr) ** 0.2
                self.time_step = dt1
                dt_list.append(dt0)
                err_list.append(eps_a)
                k_try = 0
            except Exception as exc:
                # check maximum attempts
                k_try += 1
                if k_try >= k_try_max:
                    raise exc
                # set time step smaller and try again
                self.load_system_state(t0, t1, dt0)
                self.time_step = 0.1 * self.time_step
                done = False
        return dt00, np.array(dt_list), np.array(err_list)
