"""consolidation.py
Module for implementing large strain consolidation physics
using the finite element method.

Classes
-------
ConsolidationElement1D
ConsolidationBoundary1D
ConsolidationAnalysis1D
"""
from typing import (
    Callable,
    Sequence,
)
from enum import Enum

import numpy as np
import numpy.typing as npt

from frozen_ground_fem.materials import (
    unit_weight_water as gam_w,
    spec_grav_ice,
)

from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
    Element1D,
    Boundary1D,
    Mesh1D,
)


class ConsolidationElement1D(Element1D):
    """Class for computing element matrices
    for large strain consolidation physics.

    Attributes
    ----------
    nodes
    order
    jacobian
    int_pts
    deformed_length
    stiffness_matrix
    mass_matrix

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
    def stiffness_matrix(self) -> npt.NDArray[np.floating]:
        """The element stiffness matrix.

        Returns
        -------
        numpy.ndarray
            Shape depends on order of interpolation.
            For order=1, shape=(2, 2).
            For order=3, shape=(4, 4).

        Notes
        -----
        Integrates
            B^T * (1+e0)/(1+e) * (k * (dsig'/de) / gam_w) * B
            + N^T * (d/de)(k * (Gs - 1) / (1+e)) * B
        over the element where
        e is the void ratio,
        e0 is the initial void ratio,
        k is the hydraulic conductivity,
        sig' is the effective stress,
        dsig'/de is the stress-strain coefficient from the consolidation curve,
        gam_w is the unit weight of water,
        Gs is the specific gravity of the solids.
        """
        B = self._gradient_matrix(0, 1)
        K = np.zeros_like(B.T @ B)
        jac = self.jacobian
        for ip in self.int_pts:
            e0 = ip.void_ratio_0
            e = ip.void_ratio
            e_ratio = (1.0 + e0) / (1.0 + e)
            dsig_de = ip.eff_stress_gradient
            Gs = ip.material.spec_grav_solids
            k = ip.hyd_cond
            dk_de = ip.hyd_cond_gradient
            k_coef = dk_de * (Gs - 1.0) / (1.0 + e) - k * \
                (Gs - 1.0) / (1.0 + e) ** 2
            B = self._gradient_matrix(ip.local_coord, jac)
            N = self._shape_matrix(ip.local_coord)
            K += (
                B.T @ (k * e_ratio * dsig_de / gam_w * B) + N.T @ (k_coef * B)
            ) * ip.weight
        K *= jac
        return K

    @property
    def mass_matrix(self) -> npt.NDArray[np.floating]:
        """The element mass matrix.

        Returns
        -------
        numpy.ndarray
            Shape depends on order of interpolation.
            For order=1, shape=(2, 2).
            For order=3, shape=(4, 4).

        Notes
        -----
        Integrates
            N^T * ((Sw + Gi * (1 - Sw)) / (1 + e0)) * N
        over the element where
        Sw is the degree of saturation of water,
        Gi is the specific gravity of ice,
        e0 is the initial void ratio.
        """
        N = self._shape_matrix(0)
        M = np.zeros_like(N.T @ N)
        jac = self.jacobian
        for ip in self.int_pts:
            N = self._shape_matrix(ip.local_coord)
            M += (
                N.T
                @ (
                    (ip.deg_sat_water + spec_grav_ice * ip.deg_sat_ice)
                    / (1.0 + ip.void_ratio_0)
                    * N
                )
                * ip.weight
            )
        M *= jac
        return M

    @property
    def deformed_length(self) -> float:
        """The deformed length of the element.

        Returns
        -------
        float

        Notes
        -----
        Integrates the ratio (1+e)/(1+e0)
        over the element.
        """
        L = 0.0
        for ip in self.int_pts:
            e = ip.void_ratio
            e0 = ip.void_ratio_0
            L += (1.0 + e) / (1.0 + e0) * ip.weight
        return L * self.jacobian

    def update_integration_points(self) -> None:
        """Updates the properties of integration points
        in the element according to changes in void ratio.
        """
        ee = np.array([nd.void_ratio for nd in self.nodes])
        for ip in self.int_pts:
            N = self._shape_matrix(ip.local_coord)
            B = self._gradient_matrix(ip.local_coord, self.jacobian)
            ep = (N @ ee)[0]
            de_dZ = (B @ ee)[0]
            ip.void_ratio = ep
            k, dk_de = ip.material.hyd_cond(ep, 1.0, False)
            ip.hyd_cond = k
            ip.hyd_cond_gradient = dk_de
            ppc = ip.pre_consol_stress
            sig, dsig_de = ip.material.eff_stress(ep, ppc)
            if sig > ppc:
                ip.pre_consol_stress = sig
            ip.eff_stress = sig
            ip.eff_stress_gradient = dsig_de
            e0 = ip.void_ratio_0
            Gs = ip.material.spec_grav_solids
            e_ratio = (1.0 + e0) / (1.0 + ep)
            ip.water_flux_rate = (
                -k
                / gam_w
                * e_ratio
                * ((Gs - 1.0) * gam_w / (1.0 + e0) - dsig_de * de_dZ)
            )


class ConsolidationBoundary1D(Boundary1D):
    """Class for storing and updating boundary conditions
    for consolidation physics.

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
    bnd_type : ConsolidationBoundary1D.BoundaryType, optional,
                default=BoundaryType.fixed_flux
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
        If bnd_type is not a ConsolidationBoundary1D.BoundaryType.
        If bnd_function is not callable or None.
    ValueError
        If len(nodes) != 1.
        If len(int_pts) > 1.
        If bnd_value is not convertible to float.
    """

    BoundaryType = Enum(
        "BoundaryType", ["void_ratio", "fixed_flux", "water_flux"]
    )

    _bnd_type: BoundaryType
    _bnd_value: float = 0.0
    _bnd_function: Callable | None

    def __init__(
        self,
        nodes: Sequence[Node1D],
        int_pts: Sequence[IntegrationPoint1D] = (),
        bnd_type=BoundaryType.fixed_flux,
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
        ConsolidationBoundary1D.BoundaryType

        Returns
        -------
        ConsolidationBoundary1D.BoundaryType

        Raises
        ------
        TypeError
            If the value to be assigned is not a
            ConsolidationBoundary1D.BoundaryType.
        """
        return self._bnd_type

    @bnd_type.setter
    def bnd_type(self, value: BoundaryType):
        if not isinstance(value, ConsolidationBoundary1D.BoundaryType):
            raise TypeError(
                f"{value} is not a ConsolidationBoundary1D.BoundaryType")
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
        the updates the boundary condition.

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
        This method updates the void_ratio at each of the nodes
        in the ConsolidationBoundary1D
        only in the case that bnd_type == BoundaryType.temp.
        Otherwise, it does nothing.
        """
        if self.bnd_type == ConsolidationBoundary1D.BoundaryType.void_ratio:
            for nd in self.nodes:
                nd.void_ratio = self.bnd_value

    def update_value(self, time: float) -> None:
        """Update the value of the boundary conditions.

        Parameters
        ----------
        float

        Raises
        ------
        ValueError
            It time is not convertible to float

        Notes
        -----
        This method uses the bnd_function callable property
        to update the bnd_value property
        If bnd_function is None
        the time argument is ignored and nothing happens.
        """
        time = float(time)
        if self.bnd_function is not None:
            self.bnd_value = self.bnd_function(time)


class ConsolidationAnalysis1D(Mesh1D):
    """Class for simulating consolidation physics
    on a mesh of :c:`ConsolidationElement1D`.

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
    update_consolidation_boundary_conditions
    update_water_flux_vector
    update_stiffness_matrix
    update_mass_matrix
    update_nodes
    update_integration_points
    initialize_global_system
    initialize_time_step
    update_weighted_matrices
    calculate_void_ratio_correction
    iterative_correction_step
    calculate_total_settlement
    calculate_deformed_coords

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
    _elements: tuple[ConsolidationElement1D, ...]
    _boundaries: set[ConsolidationBoundary1D]
    _time_step: float = 0.0
    _inv_time_step: float = 0.0
    _implicit_factor: float = 0.5   # Crank-Nicolson
    _inv_implicit_factor: float = 0.5
    _implicit_error_tolerance: float = 1e-3
    _max_iterations: int = 100
    _free_vec: tuple[npt.NDArray, ...]
    _free_arr: tuple[npt.NDArray, ...]
    _void_ratio_vector_0: npt.NDArray[np.floating]
    _void_ratio_vector: npt.NDArray[np.floating]
    _water_flux_vector_0: npt.NDArray[np.floating]
    _water_flux_vector: npt.NDArray[np.floating]
    _stiffness_matrix_0: npt.NDArray[np.floating]
    _stiffness_matrix: npt.NDArray[np.floating]
    _mass_matrix_0: npt.NDArray[np.floating]
    _mass_matrix: npt.NDArray[np.floating]
    _weighted_water_flux_vector: npt.NDArray[np.floating]
    _weighted_stiffness_matrix: npt.NDArray[np.floating]
    _weighted_mass_matrix: npt.NDArray[np.floating]
    _coef_matrix_0: npt.NDArray[np.floating]
    _coef_matrix_1: npt.NDArray[np.floating]
    _residual_water_flux_vector: npt.NDArray[np.floating]
    _delta_void_ratio_vector: npt.NDArray[np.floating]

    @property
    def elements(self) -> tuple[ConsolidationElement1D, ...]:
        """The tuple of :c:`ConsolidationElement1D` contained in the mesh.

        Returns
        ------
        tuple[:c:`ConsolidationElement1D`]

        Notes
        -----
        Overrides :c:`frozen_ground_fem.geometry.Mesh1D`
        property method for more specific return value
        type hint.
        """
        return self._elements

    @property
    def boundaries(self) -> set[ConsolidationBoundary1D]:
        """The tuple of :c:`ConsolidationBoundary1D` contained in the mesh.

        Returns
        ------
        set[:c:`ConsolidationBoundary1D`]

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
        to generate ConsolidationElement1D objects.
        """
        self._elements = tuple(
            ConsolidationElement1D(tuple(self.nodes[order * k + j]
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
            self._void_ratio_vector_0 = np.zeros(self.num_nodes)
            self._void_ratio_vector = np.zeros(self.num_nodes)
            self._water_flux_vector_0 = np.zeros(self.num_nodes)
            self._water_flux_vector = np.zeros(self.num_nodes)
            self._stiffness_matrix_0 = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._stiffness_matrix = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._mass_matrix_0 = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._mass_matrix = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._weighted_water_flux_vector = np.zeros(self.num_nodes)
            self._weighted_stiffness_matrix = np.zeros(
                (self.num_nodes, self.num_nodes)
            )
            self._weighted_mass_matrix = np.zeros(
                (self.num_nodes, self.num_nodes)
            )
            self._coef_matrix_0 = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._coef_matrix_1 = np.zeros(
                (self.num_nodes, self.num_nodes))
            self._residual_water_flux_vector = np.zeros(self.num_nodes)
            self._delta_void_ratio_vector = np.zeros(self.num_nodes)
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
            raise TypeError(f"type(max_iterations) {type(value)}"
                            + " invalid, must be int")
        if value <= 0:
            raise ValueError(f"max_iterations {value}"
                             + " invalid, must be positive")
        self._max_iterations = value

    def add_boundary(self, new_boundary: ConsolidationBoundary1D) -> None:
        """Adds a boundary to the mesh.

        Parameters
        ----------
        new_boundary : :c:`ConsolidationBoundary1D`
            The boundary to add to the mesh.

        Raises
        ------
        TypeError
            If new_boundary is not an instance of :c:`ConsolidationBoundary1D`.
        ValueError
            If new_boundary does not have parent Boundary1D
            in the parent mesh.
        """
        if not isinstance(new_boundary, ConsolidationBoundary1D):
            raise TypeError(
                f"type(new_boundary) {type(new_boundary)} invalid, "
                + "must be ConsolidationBoundary1D"
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

    def update_consolidation_boundary_conditions(
            self,
            time: float) -> None:
        """Update the boundary conditions in the ConsolidationAnalysis1D
        and in the parent Mesh1D.

        Parameters
        ----------
        time : float
            The time in seconds.
            Gets passed through to ConsolidationBoundary1D.update_value().

        Notes
        -----
        This convenience methods
        loops over all ConsolidationBoundary1D objects in boundaries
        and calls update_value() to update the boundary value
        and then calls update_nodes() to assign the new value
        to each boundary Node1D.
        For Dirichlet (void ratio) boundary conditions,
        the value is then assigned to the global temperature vector
        in the ConsolidationAnalysis1D object.
        """
        for tb in self.boundaries:
            tb.update_value(time)
            tb.update_nodes()
            if tb.bnd_type == ConsolidationBoundary1D.BoundaryType.void_ratio:
                for nd in tb.nodes:
                    self._void_ratio_vector[nd.index] = nd.void_ratio

    def update_water_flux_vector(self) -> None:
        """Updates the global water flux vector.

        Notes
        -----
        This convenience method clears the global water flux vector,
        then loops over elements integrating
        the element water flux vectors,
        then loops over the boundaries and
        assigns values for fixed flux and water flux type boundaries
        to the global water flux vector.
        """
        self._water_flux_vector[:] = 0.0
        for be in self.boundaries:
            if not (be.bnd_type
                    == ConsolidationBoundary1D.BoundaryType.void_ratio):
                self._water_flux_vector[be.nodes[0].index] += be.bnd_value

    def update_stiffness_matrix(self) -> None:
        """Updates the global stiffness matrix.

        Notes
        -----
        This convenience method first clears the global stiffness matrix
        then loops over the elements
        to get the element stiffness matrices
        and sums them into the global stiffness matrix
        respecting connectivity of global degrees of freedom.
        """
        self._stiffness_matrix[:, :] = 0.0
        for e in self.elements:
            ind = [nd.index for nd in e.nodes]
            Ke = e.stiffness_matrix
            self._stiffness_matrix[np.ix_(ind, ind)] += Ke

    def update_mass_matrix(self) -> None:
        """Updates the global mass matrix.

        Notes
        -----
        This convenience method clears the global mass matrix
        then loops over the elements
        to get the element mass matrices
        and sums them into the global mass matrix
        respecting connectivity of global degrees of freedom.
        """
        self._mass_matrix[:, :] = 0.0
        for e in self.elements:
            ind = [nd.index for nd in e.nodes]
            Me = e.mass_matrix
            self._mass_matrix[np.ix_(ind, ind)] += Me

    def update_nodes(self) -> None:
        """Updates the void ratio values at the nodes
        in the mesh.

        Notes
        -----
        This convenience method loops over nodes in the parent mesh
        and assigns the void ratio from the global void ratio vector
        in the ConsolidationAnalysis1D.
        """
        for nd in self.nodes:
            nd.void_ratio = self._void_ratio_vector[nd.index]

    def update_integration_points(self) -> None:
        """Updates the properties of integration points
        in the parent mesh according to changes in void ratio.
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
        updates the boundary conditions at the initial time,
        assigns initial void ratio values from the nodes to the global vector,
        updates the integration points in the parent mesh,
        then updates all global vectors and matrices.
        """
        # initialize global time
        t0 = float(t0)
        self._t0 = t0
        self._t1 = t0
        # update nodes with boundary conditions first
        self.update_consolidation_boundary_conditions(self._t0)
        # now get the void ratio from the nodes
        # (we assume that initial conditions have already been applied)
        for nd in self.nodes:
            self._void_ratio_vector[nd.index] = nd.void_ratio
        # now build the global matrices and vectors
        self.update_integration_points()
        self.update_water_flux_vector()
        self.update_stiffness_matrix()
        self.update_mass_matrix()
        self.update_weighted_matrices()
        # create list of free node indices
        # that will be updated at each iteration
        # (i.e. are not fixed/Dirichlet boundary conditions)
        free_ind_list = [nd.index for nd in self.nodes]
        for tb in self.boundaries:
            if tb.bnd_type == ConsolidationBoundary1D.BoundaryType.void_ratio:
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
        updates boundary conditions
        and global water flux vector at the end of
        the current time step,
        and initializes iterative correction parameters.
        """
        # update time coordinate
        self._t0 = self._t1
        self._t1 = self._t0 + self.dt
        # store previous converged matrices and vectors
        self._void_ratio_vector_0[:] = self._void_ratio_vector[:]
        self._water_flux_vector_0[:] = self._water_flux_vector[:]
        self._stiffness_matrix_0[:, :] = self._stiffness_matrix[:, :]
        self._mass_matrix_0[:, :] = self._mass_matrix[:, :]
        # update boundary conditions
        self.update_consolidation_boundary_conditions(self._t1)
        self.update_water_flux_vector()
        self.update_weighted_matrices()
        # initialize iteration parameters
        self._eps_a = 1.0
        self._iter = 0

    def update_weighted_matrices(self) -> None:
        """Updates global weighted matrices
        according to implicit time stepping factor.

        Notes
        -----
        This convenience method updates
        the weighted water flux vector,
        the weighted stiffness matrix,
        the weighted mass matrix,
        and coefficient matrices
        using the implicit_factor property.
        """
        self._weighted_water_flux_vector[:] = (
            self.one_minus_alpha * self._water_flux_vector_0
            + self.alpha * self._water_flux_vector
        )
        self._weighted_stiffness_matrix[:, :] = (
            self.one_minus_alpha * self._stiffness_matrix_0
            + self.alpha * self._stiffness_matrix
        )
        self._weighted_mass_matrix[:, :] = (
            self.one_minus_alpha * self._mass_matrix_0
            + self.alpha * self._mass_matrix
        )
        self._coef_matrix_0[:, :] = (
            self._weighted_mass_matrix * self.over_dt
            + self.one_minus_alpha * self._weighted_stiffness_matrix
        )
        self._coef_matrix_1[:, :] = (
            self._weighted_mass_matrix * self.over_dt
            - self.alpha * self._weighted_stiffness_matrix
        )

    def calculate_void_ratio_correction(self) -> None:
        """Performs a single iteration of void ratio correction
        in the implicit time stepping scheme.

        Notes
        -----
        This convenience method
        updates the global residual water flux vector,
        calculates the void ratio correction
        using the global weighted matrices,
        applies the correction to the global void ratio vector,
        then updates the nodes, integration points,
        and the global vectors and matrices.
        """
        # update residual vector
        self._residual_water_flux_vector[:] = (
            self._coef_matrix_0 @ self._void_ratio_vector_0
            - self._coef_matrix_1 @ self._void_ratio_vector
            - self._weighted_water_flux_vector
        )
        # calculate void ratio increment
        self._delta_void_ratio_vector[self._free_vec] = np.linalg.solve(
            self._coef_matrix_1[self._free_arr],
            self._residual_water_flux_vector[self._free_vec],
        )
        # increment void ratio and iteration variables
        self._void_ratio_vector[self._free_vec] += (
            self._delta_void_ratio_vector[self._free_vec]
        )
        self._eps_a = float(
            np.linalg.norm(self._delta_void_ratio_vector)
            / np.linalg.norm(self._void_ratio_vector)
        )
        self._iter += 1
        # update global system
        self.update_nodes()
        self.update_integration_points()
        self.update_water_flux_vector()
        self.update_stiffness_matrix()
        self.update_mass_matrix()

    def iterative_correction_step(self) -> None:
        """Performs iterative correction of the
        global void ratio vector for a single time step.

        Notes
        -----
        This convenience method performs an iterative correction loop
        based on the implicit_error_tolerance and max_iterations properties.
        It iteratively updates the global weighted matrices
        and performs correction of the global void ratio vector.
        """
        while self._eps_a > self.eps_s and self._iter < self.max_iterations:
            self.update_weighted_matrices()
            self.calculate_void_ratio_correction()

    def calculate_total_settlement(self) -> float:
        """Integrates volume change ratio
        to calculate total settlement.

        Returns
        -------
        float
            The total settlement result.

        Notes
        -----
        Positive values indicate net settlement,
        negative values indicate net heave.
        """
        s = 0.0
        for e in self.elements:
            s += e.jacobian - e.deformed_length
        return s

    def calculate_deformed_coords(self) -> npt.NDArray[np.floating]:
        """Integrates volume change ratio
        to calculate deformed coordinates of the nodes.

        Returns
        -------
        numpy.ndarray, shape = (mesh.num_nodes, )
            Vector of deformed coordinates
        """
        s = self.calculate_total_settlement()
        def_coords = np.array(self.num_nodes)
        def_coords[0] = self.nodes[0].z + s
        for k, e in enumerate(self.elements):
            dz = e.deformed_length
            def_coords[k + 1] = def_coords[k] + dz
        return def_coords
