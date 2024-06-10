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
    spec_grav_ice as Gi,
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
    calculate_deformed_coord_offsets
    calculate_total_stress_increments

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
                    (ip.deg_sat_water + Gi * ip.deg_sat_ice)
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

    def update_integration_points(
        self,
        update_water_flux: bool = True,
        update_res_stress: bool = False,
        TT0: npt.ArrayLike = (),
    ) -> None:
        """Updates the properties of integration points
        in the element according to changes in void ratio.
        """
        ee = np.array([nd.void_ratio for nd in self.nodes])
        ss = np.array([nd.tot_stress for nd in self.nodes])
        TT = np.array([nd.temp for nd in self.nodes])
        TT0 = np.array(TT0) if update_res_stress else np.zeros_like(TT)
        for ip in self.int_pts:
            N = self._shape_matrix(ip.local_coord)
            B = self._gradient_matrix(ip.local_coord, self.jacobian)
            # update void ratio interpolated from nodes
            ep = (N @ ee)[0]
            de_dZ = (B @ ee)[0]
            ip.void_ratio = ep
            # update total stress interpolated from nodes
            sp = (N @ ss)[0]
            ip.tot_stress = sp
            # get temperature state interpolated from nodes
            T = (N @ TT)[0]
            T0 = (N @ TT0)[0]
            # soil is frozen
            if T < 0.0:
                # set reference stress and void ratio for frozen state
                if T0 >= 0.0:
                    ip.void_ratio_0_ref_frozen = ep
                    ip.tot_stress_0_ref_frozen = sp
                # update local stress state
                # and total stress gradient (for stiffness matrix)
                e_f0 = ip.void_ratio_0_ref_frozen
                sig_f0 = ip.tot_stress_0_ref_frozen
                sig, dsig_de = ip.material.tot_stress(
                    T, ep, e_f0, sig_f0,
                )
                ip.loc_stress = sig
                ip.tot_stress_gradient = dsig_de
            # soil is unfrozen
            else:
                # update residual stress and pre-consolidation stress
                # if thawed on this step
                if update_res_stress and T0 < 0.0:
                    ppc = ip.material.res_stress(ep)
                    ip.pre_consol_stress = ppc
                # update effective stress
                ppc = ip.pre_consol_stress
                sig, dsig_de = ip.material.eff_stress(ep, ppc)
                if sig > ppc:
                    ip.pre_consol_stress = sig
                ip.eff_stress = sig
                ip.eff_stress_gradient = dsig_de
                # update hydraulic conductivity and water flux
                k, dk_de = ip.material.hyd_cond(ep, 1.0, False)
                ip.hyd_cond = k
                ip.hyd_cond_gradient = dk_de
            if update_water_flux:
                ip.update_water_flux_rate(de_dZ)
        for iipp in self._int_pts_deformed:
            for ip in iipp:
                N = self._shape_matrix(ip.local_coord)
                ep = (N @ ee)[0]
                ip.void_ratio = ep

    def calculate_deformed_coord_offsets(self) -> npt.NDArray[np.floating]:
        """Calculates the deformed coordinate offset distances
        relative to the first node of the element.
        """
        dc = np.zeros(len(self.nodes))
        for k, (nd0, iipp) in enumerate(
            zip(
                self.nodes[:-1],
                self._int_pts_deformed,
            )
        ):
            nd1 = self.nodes[k+1]
            jac = nd1.z - nd0.z
            ddcc = 0.0
            for ip in iipp:
                ee = ip.void_ratio
                ee0 = ip.void_ratio_0
                ddcc += (1.0 + ee) / (1.0 + ee0) * ip.weight
            dc[k+1] = dc[k] + ddcc * jac
        return dc

    def calculate_total_stress_increments(self) -> npt.NDArray[np.floating]:
        """Calculates the total stress increments
        relative to the first node of the element.
        """
        dsig = np.zeros(len(self.nodes))
        for k, (nd0, iipp) in enumerate(
            zip(
                self.nodes[:-1],
                self._int_pts_deformed,
            )
        ):
            nd1 = self.nodes[k+1]
            jac = nd1.z - nd0.z
            ddss = 0.0
            for ip in iipp:
                ee = ip.void_ratio
                ee0 = ip.void_ratio_0
                Sw = ip.deg_sat_water
                Si = ip.deg_sat_ice
                Gs = ip.material.spec_grav_solids
                ddss += (
                    (Gs + ee * (Sw + Gi * Si)) / (1 + ee0)
                ) * gam_w * ip.weight
            dsig[k+1] = dsig[k] + ddss * jac
        return dsig


class ConsolidationBoundary1D(Boundary1D):
    """Class for storing and updating hydromechanical boundary conditions
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
    bnd_value_1

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
                default=BoundaryType.water_flux
        The type of boundary condition.
    bnd_value : float, optional, default=0.0
        The value of the boundary condition.
    bnd_function : callable or None, optional, default=None
        The function for the updates the boundary condition.
    bnd_value_1 : float, optional, default=0.0
        A secondary value of the boundary condition.
        Used to store effective stress for void ratio boundary condition.

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
        "BoundaryType", ["void_ratio", "water_flux"]
    )

    _bnd_type: BoundaryType
    _bnd_value: float = 0.0
    _bnd_function: Callable | None
    _bnd_value_1: float = 0.0

    def __init__(
        self,
        nodes: Sequence[Node1D],
        int_pts: Sequence[IntegrationPoint1D] = (),
        bnd_type=BoundaryType.water_flux,
        bnd_value: float = 0.0,
        bnd_function: Callable | None = None,
        bnd_value_1: float = 0.0,
    ):
        super().__init__(nodes, int_pts)
        self.bnd_type = bnd_type
        self.bnd_value = bnd_value
        self.bnd_function = bnd_function
        self.bnd_value_1 = bnd_value_1

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

    @property
    def bnd_value_1(self) -> float:
        """The secondary value of the boundary condition.

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
        return self._bnd_value_1

    @bnd_value_1.setter
    def bnd_value_1(self, value: float) -> None:
        value = float(value)
        self._bnd_value_1 = value

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


class HydraulicBoundary1D(Boundary1D):
    """Class for storing and updating hydraulic boundary conditions
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
    bnd_type : HydraulicBoundary1D.BoundaryType, optional,
                default=BoundaryType.fixed_head
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
        If bnd_type is not a HydraulicBoundary1D.BoundaryType.
        If bnd_function is not callable or None.
    ValueError
        If len(nodes) != 1.
        If len(int_pts) > 1.
        If bnd_value is not convertible to float.
    """

    BoundaryType = Enum(
        "BoundaryType", ["fixed_head",]
    )

    _bnd_type: BoundaryType
    _bnd_value: float = 0.0
    _bnd_function: Callable | None

    def __init__(
        self,
        nodes: Sequence[Node1D],
        int_pts: Sequence[IntegrationPoint1D] = (),
        bnd_type=BoundaryType.fixed_head,
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
        HydraulicBoundary1D.BoundaryType

        Returns
        -------
        HydraulicBoundary1D.BoundaryType

        Raises
        ------
        TypeError
            If the value to be assigned is not a
            HydraulicBoundary1D.BoundaryType.
        """
        return self._bnd_type

    @bnd_type.setter
    def bnd_type(self, value: BoundaryType):
        if not isinstance(value, HydraulicBoundary1D.BoundaryType):
            raise TypeError(
                f"{value} is not a HydraulicBoundary1D.BoundaryType")
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

        Notes
        -----
        For HydraulicBoundary1D.BoundaryType.fixed_head,
        the value of the hydraulic head is taken relative to a datum
        at the (fixed) elevation of the bottom node.
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
        """Update the boundary condition value at the nodes, if necessary.
        """
        pass

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
    num_hyd_boundaries
    hyd_boundaries
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
    _generate_elements
    inialiaze_global_matrices_and_vectors
    add_boundary
    update_boundary_conditions
    update_water_flux_vector
    update_stiffness_matrix
    update_mass_matrix
    update_nodes
    initialize_integration_points
    initialize_solution_variable_vectors
    initialize_free_index_arrays
    store_converged_matrices
    update_boundary_vectors
    update_global_matrices_and_vectors
    update_integration_points
    update_weighted_matrices
    calculate_solution_vector_correction
    update_iteration_variables
    solve_to
    calculate_total_settlement
    calculate_deformed_coords
    update_total_stress_distribution
    calculate_degree_consolidation
    #initialize_time_step
    #iterative_correction_step
    #remove_boundary
    #clear_boundaries

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
    _boundaries: set[ConsolidationBoundary1D | HydraulicBoundary1D]
    _hyd_boundaries: tuple[HydraulicBoundary1D, ...] = ()
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
    def boundaries(self) -> set[ConsolidationBoundary1D | HydraulicBoundary1D]:
        """The set of :c:`ConsolidationBoundary1D` contained in the mesh.

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

    @property
    def num_hyd_boundaries(self) -> int:
        """The number of :c:`HydraulicBoundary1D` contained in the mesh.

        Returns
        ------
        int
        """
        return len(self.hyd_boundaries)

    @property
    def hyd_boundaries(self) -> tuple[HydraulicBoundary1D, ...]:
        """The tuple of :c:`HydraulicBoundary1D` contained in the mesh.

        Returns
        ------
        tuple[:c:`HydraulicBoundary1D`]
        """
        return self._hyd_boundaries

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

    def initialize_global_matrices_and_vectors(self):
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

    def add_boundary(
        self,
        new_boundary: ConsolidationBoundary1D | HydraulicBoundary1D,
    ) -> None:
        """Adds a boundary to the mesh.

        Parameters
        ----------
        new_boundary : :c:`ConsolidationBoundary1D` | :c:`HydraulicBoundary1D`
            The boundary to add to the mesh.

        Raises
        ------
        TypeError
            If new_boundary is not an instance of :c:`ConsolidationBoundary1D`
            or :c:`HydraulicBoundary1D`.
        ValueError
            If new_boundary contains a node not in the mesh.
            If new_boundary contains an integration point not in the mesh.
        """
        if not (
            isinstance(new_boundary, ConsolidationBoundary1D)
            or isinstance(new_boundary, HydraulicBoundary1D)
        ):
            raise TypeError(
                f"type(new_boundary) {type(new_boundary)} invalid, "
                + "must be ConsolidationBoundary1D or HydraulicBoundary1D"
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

    def update_boundary_conditions(
            self,
            time: float) -> None:
        """Update the boundary conditions in the ConsolidationAnalysis1D.

        Parameters
        ----------
        time : float
            The time in seconds.
            Gets passed through to the update_value()
            method of the boundary class.

        Notes
        -----
        This convenience methods
        loops over all Boundary1D objects in boundaries
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
                # self._water_flux_vector[be.nodes[0].index] -= be.bnd_value

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
            # self._stiffness_matrix[np.ix_(ind, ind)] -= Ke

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
        and assigns the void ratio from the global void ratio vector.
        """
        for nd in self.nodes:
            nd.void_ratio = self._void_ratio_vector[nd.index]

    def initialize_integration_points(self) -> None:
        # initialize initial void ratio
        # and preconsolidation stress
        for e in self.elements:
            ee0 = np.array([nd.void_ratio_0 for nd in e.nodes])
            for ip in e.int_pts:
                N = e._shape_matrix(ip.local_coord)
                e0 = (N @ ee0)[0]
                ip.void_ratio_0 = e0
                ppc = ip.pre_consol_stress
                ppc0, _ = ip.material.eff_stress(e0, ppc)
                if ppc0 > ppc:
                    ip.pre_consol_stress = ppc0
            for iipp in e._int_pts_deformed:
                for ip in iipp:
                    N = e._shape_matrix(ip.local_coord)
                    ip.void_ratio_0 = (N @ ee0)[0]
        # intialize global hydraulic boundaries
        # which will be used to update pore pressure
        # at the integration points
        hyd_boundaries: list[HydraulicBoundary1D] = []
        for b in self.boundaries:
            if isinstance(b, HydraulicBoundary1D):
                if not (
                    b.nodes[0].index == 0
                    or b.nodes[0].index == self.nodes[-1].index
                ):
                    raise ValueError(
                        "Boundary conditions include "
                        + f"node with index {b.nodes[0].index}. "
                        + "Can only be first node (index 0) or "
                        + f"last node (index {self.nodes[-1].index})."
                    )
                hyd_boundaries.append(b)
        # Check for invalid number of hydraulic boundaries
        # We only allow maximum of 2
        nhb = len(hyd_boundaries)
        if nhb > 2:
            raise ValueError("Too many hydraulic boundaries, must be <= 2.")
        # If there are 2 hydraulic boundaries,
        # make sure they are not duplicates (same node)
        # and that they are in the correct order
        if nhb == 2:
            b0 = hyd_boundaries[0]
            b1 = hyd_boundaries[1]
            if b0.nodes[0].index == b1.nodes[0].index:
                raise ValueError("Duplicate hydraulic boundaries (same node).")
            if b0.nodes[0].index > b1.nodes[0].index:
                hyd_boundaries[0] = b1
                hyd_boundaries[1] = b0
        # assign hydraulic boundaries to the analysis property
        self._hyd_boundaries = tuple(hyd_boundaries)

    def initialize_solution_variable_vectors(self) -> None:
        # now get the void ratio from the nodes
        for nd in self.nodes:
            self._void_ratio_vector[nd.index] = nd.void_ratio
            self._void_ratio_vector_0[nd.index] = nd.void_ratio

    def initialize_free_index_arrays(self) -> None:
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

    def store_converged_matrices(self) -> None:
        self._void_ratio_vector_0[:] = self._void_ratio_vector[:]
        self._water_flux_vector_0[:] = self._water_flux_vector[:]
        self._stiffness_matrix_0[:, :] = self._stiffness_matrix[:, :]
        self._mass_matrix_0[:, :] = self._mass_matrix[:, :]

    def update_boundary_vectors(self) -> None:
        self.update_water_flux_vector()

    def update_global_matrices_and_vectors(self) -> None:
        self.update_water_flux_vector()
        self.update_stiffness_matrix()
        self.update_mass_matrix()

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
        # self._coef_matrix_1[:, :] = (
        #     np.eye(self.num_nodes)
        #     + self.alpha * self.dt * np.linalg.solve(
        #         self._mass_matrix, self._stiffness_matrix,
        #     )
        # )

    def calculate_solution_vector_correction(self) -> None:
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
        # first update the global weighted matrices
        # (ensures coef_matrix_0 and coef_matrix_1)
        ConsolidationAnalysis1D.update_weighted_matrices(self)
        # update residual vector
        self._residual_water_flux_vector[:] = (
            self._coef_matrix_0 @ self._void_ratio_vector_0
            - self._coef_matrix_1 @ self._void_ratio_vector
            - self._weighted_water_flux_vector
        )
        # self._residual_water_flux_vector[:] = (
        #     self.one_minus_alpha * self.dt * np.linalg.solve(
        #         self._mass_matrix_0,
        #         self._water_flux_vector_0
        #         - self._stiffness_matrix_0 @ self._void_ratio_vector_0
        #     )
        #     + self.alpha * self.dt * np.linalg.solve(
        #         self._mass_matrix,
        #         self._water_flux_vector
        #         - self._stiffness_matrix @ self._void_ratio_vector
        #     )
        #     - (self._void_ratio_vector[:] - self._void_ratio_vector_0[:])
        # )
        # calculate void ratio increment
        self._delta_void_ratio_vector[self._free_vec] = np.linalg.solve(
            self._coef_matrix_1[self._free_arr],
            self._residual_water_flux_vector[self._free_vec],
        )
        # increment void ratio and iteration variables
        self._void_ratio_vector[self._free_vec] += (
            self._delta_void_ratio_vector[self._free_vec]
        )

    def update_iteration_variables(self) -> None:
        self._eps_a = float(
            np.linalg.norm(self._delta_void_ratio_vector)
            / np.linalg.norm(self._void_ratio_vector)
        )
        self._iter += 1

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
        # initialize vectors and matrices
        # for adaptive step size correction
        num_int_pt_per_element = len(self.elements[0].int_pts)
        void_ratio_vector_0 = np.zeros_like(self._void_ratio_vector)
        void_ratio_vector_1 = np.zeros_like(self._void_ratio_vector)
        void_ratio_error = np.zeros_like(self._void_ratio_vector)
        void_ratio_rate = np.zeros_like(self._void_ratio_vector)
        void_ratio_scale = np.zeros_like(self._void_ratio_vector)
        pre_consol_stress_0 = np.zeros((
            self.num_elements,
            num_int_pt_per_element,
        ))
        dt_list = []
        err_list = []
        done = False
        while not done and self._t1 < tf:
            # check if time step passes tf
            dt00 = self.time_step
            if self._t1 + self.time_step > tf:
                self.time_step = tf - self._t1
                done = True
            # save system state before time step
            t0 = self._t1
            dt0 = self.time_step
            void_ratio_vector_0[:] = self._void_ratio_vector[:]
            for ke, e in enumerate(self.elements):
                for jip, ip in enumerate(e.int_pts):
                    pre_consol_stress_0[ke, jip] = ip.pre_consol_stress
            # take single time step
            self.initialize_time_step()
            self.iterative_correction_step()
            # save the predictor result
            void_ratio_vector_1[:] = self._void_ratio_vector[:]
            # reset the system
            self._void_ratio_vector[:] = void_ratio_vector_0[:]
            for e, ppc_e in zip(self.elements, pre_consol_stress_0):
                for ip, ppc0 in zip(e.int_pts, ppc_e):
                    ip.pre_consol_stress = ppc0
            self.update_nodes()
            self.update_integration_points()
            self.update_global_matrices_and_vectors()
            self._t1 = t0
            # take two half steps
            self.time_step = 0.5 * dt0
            self.initialize_time_step()
            self.iterative_correction_step()
            self.initialize_time_step()
            self.iterative_correction_step()
            # compute truncation error correction
            void_ratio_error[:] = (self._void_ratio_vector[:]
                                   - void_ratio_vector_1[:]) / 3.0
            self._void_ratio_vector[:] += void_ratio_error[:]
            self.update_nodes()
            self.update_integration_points()
            self.update_global_matrices_and_vectors()
            # update the time step
            void_ratio_rate[:] = (self._void_ratio_vector[:]
                                  - void_ratio_vector_0[:])
            void_ratio_scale[:] = np.max(np.vstack([
                self._void_ratio_vector[:],
                void_ratio_rate,
            ]), axis=0)
            e_scale = float(np.linalg.norm(void_ratio_scale))
            err_targ = self.eps_s * e_scale
            err_curr = float(np.linalg.norm(void_ratio_error))
            # update the time step
            eps_a = err_curr / e_scale
            dt1 = dt0 * (err_targ / err_curr) ** 0.2
            self.time_step = dt1
            dt_list.append(dt0)
            err_list.append(eps_a)
        return dt00, np.array(dt_list), np.array(err_list)

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

        Notes
        -----
        Also assigns deformed coordinates to the nodes.
        """
        # initialize top node with settlement
        s = self.calculate_total_settlement()
        def_coords = np.zeros(self.num_nodes)
        def_coords[0] = self.nodes[0].z + s
        # integrate over elements to get deformed coords
        for k, e in enumerate(self.elements):
            kk0 = k * e.order
            kk1 = kk0 + e.order + 1
            ddcc = e.calculate_deformed_coord_offsets()
            def_coords[kk0:kk1] = def_coords[kk0] + ddcc
        # ensure bottom node stays fixed
        def_coords[-1] = self.nodes[-1].z
        # assign deformed coordinates to the nodes
        for k, zd in enumerate(def_coords):
            self.nodes[k].z_def = zd
        return def_coords

    def update_total_stress_distribution(self) -> None:
        """Integrates bulk unit weight to calculate
        total stress at the nodes.
        """
        # initialize total stress at surface
        sig = np.zeros(self.num_nodes)
        for b in self.boundaries:
            # find first node
            if not b.nodes[0].index:
                if (
                    isinstance(b, ConsolidationBoundary1D)
                    and b.bnd_type
                        == ConsolidationBoundary1D.BoundaryType.void_ratio
                ):
                    # surface total stress, effective stress component
                    sig[0] += b.bnd_value_1
                elif (
                    isinstance(b, HydraulicBoundary1D)
                    and b.bnd_type
                        == HydraulicBoundary1D.BoundaryType.fixed_head
                ):
                    # surface total stress, pore pressure component
                    h = b.bnd_value
                    z = self.nodes[-1].z - self.nodes[0].z_def
                    sig[0] += gam_w * (h - z)
        # integrate overburden stress over elements
        for k, e in enumerate(self.elements):
            kk0 = k * e.order
            kk1 = kk0 + e.order + 1
            ddss = e.calculate_total_stress_increments()
            sig[kk0:kk1] = sig[kk0] + ddss
        # assign total stresses to the nodes
        for k, ss in enumerate(sig):
            self.nodes[k].tot_stress = ss

    def update_pore_pressure_distribution(self) -> None:
        """Updates steady state and excess pore pressure distributions
        based on hydraulic boundary conditions and
        total and effective stress distributions.

        Notes
        -----
        If one HydraulicBoundary1D is provided,
        then total hydraulic head is assumed constant.
        If two HydraulicBoundary1D are provided,
        a steady state hydraulic gradient is calculated
        and used to determine total hydraulic head;
        in this case, one HydraulicBoundary1D should be at
        node 0 and the other at the last node.
        If zero HydraulicBoundary1D are provided,
        no ponding is assumed with fully saturated conditions,
        so total hydraulic head is assumed constant at the
        elevation head of node 0.
        """
        # initialize hydraulic boundary condition values
        # for interpolating hydraulic head
        nhb = self.num_hyd_boundaries
        z0 = self.nodes[0].z_def
        z1 = self.nodes[-1].z_def
        if not nhb:
            h0 = z1 - z0
            dhdz = 0.0
        elif nhb == 1:
            h0 = self.hyd_boundaries[0].bnd_value
            dhdz = 0.0
        else:
            h0 = self.hyd_boundaries[0].bnd_value
            h1 = self.hyd_boundaries[1].bnd_value
            dhdz = (h1 - h0) / (z1 - z0)
        # update pore pressure at the integration points
        for e in self.elements:
            zz = [nd.z_def for nd in e.nodes]
            for ip in e.int_pts:
                N = e._shape_matrix(ip.local_coord)
                z = (N @ zz)[0]
                zh = z1 - z
                h = h0 + dhdz * (z - z0)
                uh = (h - zh) * gam_w
                uu = ip.tot_stress - ip.eff_stress
                ue = uu - uh
                ip.pore_pressure = uu
                ip.exc_pore_pressure = ue

    def calculate_degree_consolidation(
        self,
        void_ratio_1: npt.ArrayLike,
    ) -> tuple[float, npt.NDArray[np.floating]]:
        """Compute average degree of consolidation and
        degree of consolidation profile given
        a final void ratio profile.

        Inputs
        ------
        void_ratio_1 : array_like, shape=(num_nodes, )
            The final void ratio profile at the nodes.

        Returns
        -------
        float
            The average degree of consolidation.
        numpy.ndarray, shape=(num_nodes, )
            The degree of consolidation at the nodes.

        Notes
        -----
        Degree of consolidation Uz(Z, t) is defined as

            Uz(Z, t) = (e0(Z) - e(Z, t)) / (e0(Z) - e1(Z))

        where Z are the original (Lagrangian coordinates),
        t is time,
        e0(Z) is the initial void ratio profile,
        e1(Z) is the final void ratio profile,
        and e(Z, t) is the current void ratio profile.
        Average degree of consolidation is calculated by
        integrating Uz over the entire profile and normalizing
        by the original height of the profile.
        """
        e0 = np.array([nd.void_ratio_0 for nd in self.nodes])
        e1 = np.array(void_ratio_1)
        et = np.array([nd.void_ratio for nd in self.nodes])
        Uz = np.array([
            (ee0 - eet) / (ee0 - ee1)
            for ee0, ee1, eet in zip(e0, e1, et)
        ])
        order = self.elements[0].order
        UU = 0.0
        H = 0.0
        for k, e in enumerate(self.elements):
            jac = e.jacobian
            H += jac
            kk0 = k * order
            kk1 = kk0 + order + 1
            Uze = Uz[kk0:kk1]
            for ip in e.int_pts:
                N = e._shape_matrix(ip.local_coord)
                Uzi = (N @ Uze)[0]
                UU += Uzi * ip.weight * jac
        UU /= H
        return UU, Uz
