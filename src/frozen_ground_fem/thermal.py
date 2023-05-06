"""thermal.py
Module for implementing thermal physics using the finite element method.
"""
from enum import Enum

import numpy as np

from frozen_ground_fem.geometry import (
    shape_matrix,
    gradient_matrix,
    Node1D,
    IntegrationPoint1D,
    Element1D,
    BoundaryElement1D,
    Mesh1D,
)


class ThermalElement1D(Element1D):
    """Class for computing element matrices for thermal physics.

    Parameters
    ----------
    parent : frozen_ground_fem.geometry.Element1D
        The parent element from the mesh

    Raises
    ------
    TypeError
        If parent initializer is not a
        :c:`frozen_ground_fem.geometry.Element1D`.
    """

    def __init__(self, parent: Element1D) -> None:
        if not isinstance(parent, Element1D):
            raise TypeError(f"type(parent): {type(parent)} is not Element1D")
        self._parent = parent

    @property
    def nodes(self) -> tuple[Node1D, ...]:
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple of :c:`Node1D`

        Notes
        -----
        This is a wrapper that references the nodes property
        of the parent Element1D.
        """
        return self._parent.nodes

    @property
    def jacobian(self) -> float:
        """The length scale of the element (in Lagrangian coordinates).

        Returns
        -------
        float

        Notes
        -----
        This is a wrapper that references the jacobian property
        of the parent Element1D.
        """
        return self._parent.jacobian

    @property
    def int_pts(self) -> tuple[IntegrationPoint1D]:
        """The tuple of :c:`IntegrationPoint1D` contained in the element.

        Returns
        ------
        tuple of :c:`IntegrationPoint1D`

        Notes
        -----
        This is a wrapper that references the int_pts property
        of the parent Element1D.
        """
        return self._parent.int_pts

    def heat_flow_matrix(self) -> np.ndarray:
        """The element heat flow (conduction) matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        Integrates B^T * lambda * B over the element
        where lambda is the thermal conductivity.
        """
        B = gradient_matrix(0, 1)
        H = np.zeros_like(B.T @ B)
        jac = self.jacobian
        for ip in self.int_pts:
            B = gradient_matrix(ip.local_coord, jac)
            H += B.T @ (ip.thrm_cond * B) * ip.weight
        H *= jac
        return H

    def heat_storage_matrix(self) -> np.ndarray:
        """The element heat storage matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        Integrates N^T * C * N over the element
        where C is the volumetric heat capacity.
        """
        N = shape_matrix(0)
        C = np.zeros_like(N.T @ N)
        jac = self.jacobian
        for ip in self.int_pts:
            N = shape_matrix(ip.local_coord)
            C += N.T @ (ip.vol_heat_cap * N) * ip.weight
        C *= jac
        return C


class ThermalBoundary1D(BoundaryElement1D):
    """Class for storing and updating boundary conditions for thermal physics.

    Parameters
    ----------
    parent : frozen_ground_fem.geometry.BoundaryElement1D
        The parent boundary element from the mesh
    bnd_type : ThermalBoundary1D.BoundaryType, optional
        The type of boundary condition
    bnd_value : float, optional
        The value of the boundary condition

    Attributes
    ----------
    BoundaryType : enum.Enum
        The set of possible boundary condition types

    Raises
    ------
    TypeError
        If parent initializer is not a
        :c:`frozen_ground_fem.geometry.BoundaryElement1D`.
    """

    BoundaryType = Enum("BoundaryType", ["temp", "heat_flux", "temp_grad"])

    def __init__(
        self,
        parent: BoundaryElement1D,
        bnd_type=BoundaryType.temp,
        bnd_value: float = 0.0,
        bnd_function=None,
    ) -> None:
        if not isinstance(parent, BoundaryElement1D):
            raise TypeError(f"type(parent): {type(parent)} is not BoundaryElement1D")
        self._parent = parent
        self.bnd_type = bnd_type
        self.bnd_value = bnd_value
        self.bnd_function = bnd_function

    @property
    def nodes(self) -> tuple[Node1D, ...]:
        """The tuple of :c:`Node1D` contained in the boundary element.

        Returns
        ------
        tuple of :c:`Node1D`

        Notes
        -----
        This is a wrapper that references the nodes property
        of the parent BoundaryElement1D.
        """
        return self._parent.nodes

    @property
    def int_pts(self) -> tuple[IntegrationPoint1D, ...] | None:
        """The tuple of :c:`IntegrationPoint1D` contained in the boundary element.

        Returns
        ------
        tuple of :c:`IntegrationPoint1D`

        Notes
        -----
        This is a wrapper that references the int_pts property
        of the parent BoundaryElement1D.
        """
        return self._parent.int_pts

    @property
    def bnd_type(self):
        """The type of boundary condition.

        Parameters
        ----------
        value : ThermalBoundary1D.BoundaryType
            The value to set the type of boundary condition.

        Returns
        -------
        ThermalBoundary1D.BoundaryType

        Raises
        ------
        TypeError
            If the value to be assigned is not a ThermalBoundary1D.BoundaryType
        """
        return self._bnd_type

    @bnd_type.setter
    def bnd_type(self, value):
        if not isinstance(value, ThermalBoundary1D.BoundaryType):
            raise TypeError(f"{value} is not a ThermalBoundary1D.BoundaryType")
        self._bnd_type = value

    @property
    def bnd_value(self):
        """The value of the boundary condition.

        Parameters
        ----------
        value : float
            The value to set for the boundary condition

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to be assigned is not convertible to float
        """
        return self._bnd_value

    @bnd_value.setter
    def bnd_value(self, value):
        value = float(value)
        self._bnd_value = value

    @property
    def bnd_function(self):
        """The reference to the function
        the updates the boundary condition.

        Parameters
        ----------
        value : callable or None
            The value to set for the boundary function

        Returns
        -------
        callable or None

        Raises
        ------
        TypeError
            If the value to be assigned is not callable or None

        Notes
        -----
        If a callable (i.e. function or class that implements __call__)
        reference is provided
        it should take one argument
        which is a time (in seconds).
        This function is called by the method update_value().
        """
        return self._bnd_function

    @bnd_function.setter
    def bnd_function(self, value):
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

    def update_value(self, time):
        """Update the value of the boundary conditions.

        Parameters
        ----------
        time: float
            The time in seconds

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


class ThermalAnalysis1D:
    def __init__(self, mesh: Mesh1D) -> None:
        # validate mesh on which the analysis is to be performed
        if not isinstance(mesh, Mesh1D):
            raise TypeError(f"mesh has type {type(mesh)}, not Mesh1D")
        if not mesh.mesh_valid:
            raise ValueError(
                f"mesh.mesh_valid is {mesh.mesh_valid}, need to generate mesh"
            )
        # assign the mesh and create thermal elements
        self._mesh = mesh
        self._thermal_elements = tuple(ThermalElement1D(e) for e in self.mesh.elements)
        self._thermal_boundaries = tuple(
            ThermalBoundary1D(be) for be in self.mesh.boundary_elements
        )
        # set default values for time stepping algorithm
        self.implicit_factor = 0.5  # (Crank-Nicolson)
        self.implicit_error_tolerance = 1e-3
        self.max_iterations = 100
        # initialize global vectors and matrices
        self._temp_vector_0 = np.zeros(self.mesh.num_nodes)
        self._temp_vector = np.zeros(self.mesh.num_nodes)
        self._heat_flux_vector_0 = np.zeros(self.mesh.num_nodes)
        self._heat_flux_vector = np.zeros(self.mesh.num_nodes)
        self._heat_flow_matrix_0 = np.zeros((self.mesh.num_nodes, self.mesh.num_nodes))
        self._heat_flow_matrix = np.zeros((self.mesh.num_nodes, self.mesh.num_nodes))
        self._heat_storage_matrix_0 = np.zeros(
            (self.mesh.num_nodes, self.mesh.num_nodes)
        )
        self._heat_storage_matrix = np.zeros((self.mesh.num_nodes, self.mesh.num_nodes))
        self._weighted_heat_flux_vector = np.zeros(self.mesh.num_nodes)
        self._weighted_heat_flow_matrix = np.zeros(
            (self.mesh.num_nodes, self.mesh.num_nodes)
        )
        self._weighted_heat_storage_matrix = np.zeros(
            (self.mesh.num_nodes, self.mesh.num_nodes)
        )
        self._coef_matrix_0 = np.zeros((self.mesh.num_nodes, self.mesh.num_nodes))
        self._coef_matrix_1 = np.zeros((self.mesh.num_nodes, self.mesh.num_nodes))
        self._residual_heat_flux_vector = np.zeros(self.mesh.num_nodes)
        self._delta_temp_vector = np.zeros(self.mesh.num_nodes)

    @property
    def mesh(self) -> Mesh1D:
        """A reference to the parent mesh object.

        Returns
        -------
        frozen_ground_fem.geometry.Mesh1D

        Notes
        -----
        This property is not intended to be set
        after creation of the ThermalAnalysis1D object.
        It is assigned during the __init__() method.
        Other methods and properties of ThermalAnalysis1D
        assume that the mesh object does not change
        (i.e. mesh is not regenerated,
        number of nodes and elements is fixed).
        Use caution with modifying mesh after creation of
        ThermalAnalysis1D,
        otherwise unexpected behaviour could occur.
        """
        return self._mesh

    @property
    def thermal_elements(self) -> tuple[ThermalElement1D, ...]:
        """A tuple of thermal elements contained in the mesh.

        Returns
        -------
        tuple[ThermalElement1D]

        Notes
        -----
        The tuple of ThermalElement1D is created
        during the __init__() method.
        It is assumed that the parent Element1D
        objects in the parent Mesh1D do not change.
        Therefore, the set of ThermalElement1D
        is immutable.
        """
        return self._thermal_elements

    @property
    def thermal_boundaries(self) -> tuple[ThermalBoundary1D, ...]:
        return self._thermal_boundaries

    @property
    def time_step(self):
        """The time step for the transient analysis.

        Parameters
        ----------
        value : float
            The value to assign to the time step.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign is not convertible to float
            If the value to assign is negative

        Notes
        -----
        Also computes and stores an inverse value
        1 / time_step
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
    def dt(self):
        """An alias for time_step."""
        return self._time_step

    @property
    def over_dt(self):
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
    def implicit_factor(self):
        """The implicit time stepping factor for the analysis.

        Parameters
        ----------
        value : float
            The value to assign to the implicit factor

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
    def implicit_factor(self, value):
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"invalid implicit_factor {value}, must be between 0.0 and 1.0"
            )
        self._implicit_factor = value
        self._inv_implicit_factor = 1.0 - value

    @property
    def alpha(self):
        """An alias for implicit_factor."""
        return self._implicit_factor

    @property
    def one_minus_alpha(self):
        """The value (1 - implicit_factor).

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
    def implicit_error_tolerance(self):
        """The error tolerance for the iterative correction
        in the implicit time stepping scheme.

        Parameters
        ----------
        value : float
            The value to assign for the error tolerance

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
    def implicit_error_tolerance(self, value):
        value = float(value)
        if value <= 0.0:
            raise ValueError(
                f"invalid implicit_error_tolerance {value}, must be positive"
            )
        self._implicit_error_tolerance = value

    @property
    def eps_s(self):
        """An alias for implicit_error_tolerance."""
        return self._implicit_error_tolerance

    @property
    def max_iterations(self):
        """The maximum number of iterations for iterative correction
        in the implicit time stepping scheme.

        Parameters
        ----------
        value : int
            The value to be assigned to the maximum number of iterations

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If the value to be assigned is not an int
        ValueError
            If the value to be assigned is negative
        """
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        if not isinstance(value, int):
            raise TypeError(f"type(max_iterations) {type(value)} invalid, must be int")
        if value <= 0:
            raise ValueError(f"max_iterations {value} invalid, must be positive")
        self._max_iterations = value

    def update_thermal_boundary_conditions(self, time) -> None:
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
        loops over all ThermalBoundary1D objects in thermal_boundaries
        and calls update_value() to update the boundary value
        and then calls update_nodes() to assign the new value
        to each boundary Node1D.
        For Dirichlet (temperature) boundary conditions,
        the value is then assigned to the global temperature vector
        in the ThermalAnalysis1D object.
        """
        for tb in self.thermal_boundaries:
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
        then loops over the thermal_boundaries and
        assigns values for flux and gradient type boundaries
        to the global heat flux vector.
        """
        self._heat_flux_vector[:] = 0.0
        for be in self.thermal_boundaries:
            if be.bnd_type == ThermalBoundary1D.BoundaryType.heat_flux:
                self._heat_flux_vector[be.nodes[0].index] += be.bnd_value
            elif be.bnd_type == ThermalBoundary1D.BoundaryType.temp_grad:
                self._heat_flux_vector[be.nodes[0].index] += (
                    -be.int_pts[0].thrm_cond * be.bnd_value
                )

    def update_heat_flow_matrix(self) -> None:
        """Updates the global heat flow matrix.

        Notes
        -----
        This convenience method first clears the global heat flow matrix
        then loops over the thermal_elements
        to get the element heat flow matrices
        and sums them into the global heat flow matrix
        respecting connectivity of global degrees of freedom.
        """
        self._heat_flow_matrix[:, :] = 0.0
        for e in self.thermal_elements:
            ind = [nd.index for nd in e.nodes]
            He = e.heat_flow_matrix()
            self._heat_flow_matrix[np.ix_(ind, ind)] += He

    def update_heat_storage_matrix(self) -> None:
        """Updates the global heat storage matrix.

        Notes
        -----
        This convenience method clears the global heat storage matrix
        then loops over the thermal_elements
        to get the element heat storage matrices
        and sums them into the global heat storage matrix
        respecting connectivity of global degrees of freedom.
        """
        self._heat_storage_matrix[:, :] = 0.0
        for e in self.thermal_elements:
            ind = [nd.index for nd in e.nodes]
            Ce = e.heat_storage_matrix()
            self._heat_storage_matrix[np.ix_(ind, ind)] += Ce

    def update_nodes(self) -> None:
        """Updates the temperature values at the nodes
        in the parent mesh.

        Notes
        -----
        This convenience method loops over nodes in the parent mesh
        and assigns the temperature from the global temperature vector
        in the ThermalAnalysis1D.
        """
        for nd in self.mesh.nodes:
            nd.temp = self._temp_vector[nd.index]

    def update_integration_points(self) -> None:
        """Updates the properties of integration points
        in the parent mesh according to changes in temperature.

        Notes
        -----
        This convenience method loops over integration points
        in the parent mesh,
        interpolates temperatures from corresponding nodes
        and updates volumetric ice content accordingly."""
        for e in self.mesh.elements:
            Te = np.array([nd.temp for nd in e.nodes])
            for ip in e.int_pts:
                N = shape_matrix(ip.local_coord)
                T = N @ Te
                if T <= 0.0:
                    ip.vol_ice_cont = ip.porosity
                else:
                    ip.vol_ice_cont = 0.0

    def initialize_global_system(self, t0) -> None:
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
        to the nodes in the parent mesh.
        It initializes variables tracking the time coordinate,
        updates the thermal boundary conditions at the initial time,
        assigns initial temperature values from the nodes to the global time vector,
        updates the integration points in the parent mesh,
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
        for nd in self.mesh.nodes:
            self._temp_vector[nd.index] = nd.temp
        # now build the global matrices and vectors
        self.update_integration_points()
        self.update_heat_flux_vector()
        self.update_heat_flow_matrix()
        self.update_heat_storage_matrix()
        # create list of free node indices
        # that will be updated at each iteration
        # (i.e. are not fixed/Dirichlet boundary conditions)
        free_ind = [nd.index for nd in self.mesh.nodes]
        for tb in self.thermal_boundaries:
            if tb.bnd_type == ThermalBoundary1D.BoundaryType.temp:
                free_ind.remove(tb.nodes[0].index)
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
        self._temp_vector[self._free_vec] += self._delta_temp_vector[self._free_vec]
        self._eps_a = np.linalg.norm(self._delta_temp_vector) / np.linalg.norm(
            self._temp_vector
        )
        self._iter += 1
        # update global system
        self.update_nodes()
        self.update_integration_points()
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
