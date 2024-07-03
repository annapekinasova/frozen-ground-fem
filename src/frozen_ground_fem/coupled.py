"""coupled.py
Module for coupled thermal and large strain consolidation physics
using the finite element method.
"""
import numpy as np
import numpy.typing as npt

from .geometry import (
    Mesh1D,
)
from .thermal import (
    ThermalElement1D,
    ThermalBoundary1D,
    ThermalAnalysis1D,
)
from .consolidation import (
    ConsolidationElement1D,
    ConsolidationBoundary1D,
    HydraulicBoundary1D,
    ConsolidationAnalysis1D,
)


class CoupledElement1D(ThermalElement1D, ConsolidationElement1D):
    """Class for computing element matrices
    for coupled thermal and large strain consolidation physics.

    Attributes
    ----------
    nodes
    order
    jacobian
    int_pts
    deformed_length
    heat_flow_matrix
    heat_storage_matrix
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

    def update_integration_points(
        self,
        update_water_flux: bool = True,
        update_res_stress: bool = True,
    ) -> None:
        """Updates the properties of integration points
        in the element according to changes in void ratio.
        """
        ThermalElement1D.update_integration_points(self, False)
        ConsolidationElement1D.update_integration_points(
            self, update_water_flux, update_res_stress,
        )


class CoupledAnalysis1D(ThermalAnalysis1D, ConsolidationAnalysis1D):
    """Class for simulating
    coupled thermal and large strain consolidation physics.

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
    update_boundary_conditions
    update_heat_flux_vector
    update_heat_flow_matrix
    update_heat_storage_matrix
    update_global_matrices_and_vectors
    update_nodes
    update_integration_points
    initialize_global_system
    initialize_time_step
    update_weighted_matrices
    calculate_solution_vector_correction
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
    _elements: tuple[CoupledElement1D, ...]
    _boundaries: set
    _free_vec_thrm: tuple[npt.NDArray, ...]
    _free_arr_thrm: tuple[npt.NDArray, ...]
    _free_vec_cnsl: tuple[npt.NDArray, ...]
    _free_arr_cnsl: tuple[npt.NDArray, ...]

    @property
    def elements(self) -> tuple[CoupledElement1D, ...]:
        """The tuple of :c:`CoupledElement1D` contained in the mesh.

        Returns
        ------
        tuple[:c:`CoupledElement1D`]

        Notes
        -----
        Overrides
        property method for more specific return value
        type hint.
        """
        return self._elements

    @property
    def boundaries(self) -> set:
        """The set of
        :c:`ThermalBoundary1D`
        and
        :c:`ConsolidationBoundary1D`
        contained in the mesh.

        Returns
        ------
        set[:c:`ThermalBoundary1D`, :c:`ConsolidationBoundary1D`]

        Notes
        -----
        Overrides
        property method for more specific return value
        type hint.
        """
        return self._boundaries

    def add_boundary(
        self,
        new_boundary:
            ThermalBoundary1D
            | ConsolidationBoundary1D
            | HydraulicBoundary1D,
    ) -> None:
        """Adds a boundary to the mesh.

        Parameters
        ----------
        new_boundary : :c:`ThermalBoundary1D` or :c:`ConsolidationBoundary1D`
            The boundary to add to the mesh.

        Raises
        ------
        TypeError
            If new_boundary is not an instance of
            :c:`ThermalBoundary1D`
            or
            :c:`ConsolidationBoundary1D`.
        ValueError
            If new_boundary contains a :c:`Node1D` not in the mesh.
            If new_boundary contains an :c:`IntegrationPoint1D`
                not in the mesh.
        """
        if not (
            isinstance(new_boundary, ThermalBoundary1D)
            or isinstance(new_boundary, ConsolidationBoundary1D)
            or isinstance(new_boundary, HydraulicBoundary1D)
        ):
            raise TypeError(
                f"type(new_boundary) {type(new_boundary)} invalid, "
                + "must be ThermalBoundary1D or ConsolidationBoundary1D"
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

    def _generate_elements(self, num_elements: int, order: int):
        """Generate the elements in the mesh.

        Notes
        -----
        Overrides Mesh1D._generate_elements()
        to generate CoupledElement1D objects.
        """
        self._elements = tuple(
            CoupledElement1D(tuple(self.nodes[order * k + j]
                                   for j in range(order + 1)),
                             order)
            for k in range(num_elements)
        )

    def initialize_integration_points(self) -> None:
        ThermalAnalysis1D.initialize_integration_points(self)
        ConsolidationAnalysis1D.initialize_integration_points(self)

    def initialize_global_matrices_and_vectors(self):
        ThermalAnalysis1D.initialize_global_matrices_and_vectors(self)
        ConsolidationAnalysis1D.initialize_global_matrices_and_vectors(self)

    def initialize_free_index_arrays(self) -> None:
        ThermalAnalysis1D.initialize_free_index_arrays(self)
        self._free_vec_thrm = tuple(self._free_vec)
        self._free_arr_thrm = tuple(self._free_arr)
        ConsolidationAnalysis1D.initialize_free_index_arrays(self)
        self._free_vec_cnsl = tuple(self._free_vec)
        self._free_arr_cnsl = tuple(self._free_arr)

    def initialize_solution_variable_vectors(self) -> None:
        ThermalAnalysis1D.initialize_solution_variable_vectors(self)
        ConsolidationAnalysis1D.initialize_solution_variable_vectors(self)

    def store_converged_matrices(self) -> None:
        ThermalAnalysis1D.store_converged_matrices(self)
        ConsolidationAnalysis1D.store_converged_matrices(self)

    def update_boundary_conditions(
            self,
            time: float) -> None:
        """Update the thermal and consolidation boundary conditions.

        Parameters
        ----------
        time : float
            The time in seconds.
            Gets passed through to the update_value() method
            for each Boundary1D object.

        Notes
        -----
        This convenience methods
        loops over all ThermalBoundary1D and ConsolidationBoundary1D
        objects in boundaries
        and calls update_value() to update the boundary value
        and then calls update_nodes() to assign the new value
        to each boundary Node1D.
        For Dirichlet boundary conditions,
        the value is then assigned to
        the appropriate global solution variable vector.
        """
        for tb in self.boundaries:
            tb.update_value(time)
            tb.update_nodes()
            if tb.bnd_type == ThermalBoundary1D.BoundaryType.temp:
                for nd in tb.nodes:
                    self._temp_vector[nd.index] = nd.temp
            elif (
                tb.bnd_type == ConsolidationBoundary1D.BoundaryType.void_ratio
            ):
                for nd in tb.nodes:
                    self._void_ratio_vector[nd.index] = nd.void_ratio

    def update_nodes(self) -> None:
        """Updates the temperature and void ratio values at the nodes
        in the mesh.

        Notes
        -----
        This convenience method loops over nodes in the mesh
        and assigns the temperature from the global temperature vector
        and the void ratio from the global void ratio vector.
        """
        for nd in self.nodes:
            nd.temp = self._temp_vector[nd.index]
            nd.temp_rate = self._temp_rate_vector[nd.index]
            nd.void_ratio = self._void_ratio_vector[nd.index]

    def update_global_matrices_and_vectors(self) -> None:
        ThermalAnalysis1D.update_global_matrices_and_vectors(self)
        ConsolidationAnalysis1D.update_global_matrices_and_vectors(self)

    def update_boundary_vectors(self) -> None:
        ThermalAnalysis1D.update_boundary_vectors(self)
        ConsolidationAnalysis1D.update_boundary_vectors(self)

    def calculate_solution_vector_correction(self) -> None:
        self._free_vec = self._free_vec_thrm
        self._free_arr = self._free_arr_thrm
        ThermalAnalysis1D.calculate_solution_vector_correction(self)
        self._free_vec = self._free_vec_cnsl
        self._free_arr = self._free_arr_cnsl
        ConsolidationAnalysis1D.calculate_solution_vector_correction(self)

    def update_iteration_variables(self) -> None:
        eps_a_thrm = float(
            np.linalg.norm(self._delta_temp_vector) /
            np.linalg.norm(self._temp_vector)
        )
        eps_a_cnsl = float(
            np.linalg.norm(self._delta_void_ratio_vector)
            / np.linalg.norm(self._void_ratio_vector)
        )
        self._eps_a = np.max([eps_a_thrm, eps_a_cnsl])
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
        temp_vector_0 = np.zeros_like(self._temp_vector)
        temp_vector_1 = np.zeros_like(self._temp_vector)
        temp_error = np.zeros_like(self._temp_vector)
        temp_rate_0 = np.zeros_like(self._temp_vector)
        temp_scale = np.zeros_like(self._temp_vector)
        vol_water_cont__0 = np.zeros((
            self.num_elements,
            num_int_pt_per_element,
        ))
        temp__0 = np.zeros((
            self.num_elements,
            num_int_pt_per_element,
        ))
        void_ratio_vector_0 = np.zeros_like(self._void_ratio_vector)
        void_ratio_vector_1 = np.zeros_like(self._void_ratio_vector)
        void_ratio_error = np.zeros_like(self._void_ratio_vector)
        void_ratio_rate = np.zeros_like(self._void_ratio_vector)
        void_ratio_scale = np.zeros_like(self._void_ratio_vector)
        pre_consol_stress__0 = np.zeros((
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
            temp_vector_0[:] = self._temp_vector[:]
            temp_rate_0[:] = self._temp_rate_vector[:]
            void_ratio_vector_0[:] = self._void_ratio_vector[:]
            for ke, e in enumerate(self.elements):
                for jip, ip in enumerate(e.int_pts):
                    temp__0[ke, jip] = ip.temp
                    vol_water_cont__0[ke, jip] = ip.vol_water_cont
                    pre_consol_stress__0[ke, jip] = ip.pre_consol_stress
            # take single time step
            self.initialize_time_step()
            self.iterative_correction_step()
            # save the predictor result
            temp_vector_1[:] = self._temp_vector[:]
            void_ratio_vector_1[:] = self._void_ratio_vector[:]
            # reset the system
            self._temp_vector[:] = temp_vector_0[:]
            self._temp_rate_vector[:] = temp_rate_0[:]
            self._void_ratio_vector[:] = void_ratio_vector_0[:]
            for e, T0e, thw0_e, ppc0_e in zip(
                self.elements,
                temp__0,
                vol_water_cont__0,
                pre_consol_stress__0,
            ):
                for ip, T0, thw0, ppc0 in zip(
                    e.int_pts,
                    T0e,
                    thw0_e,
                    ppc0_e,
                ):
                    ip.temp__0 = T0
                    ip.vol_water_cont__0 = thw0
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
            temp_error[:] = (self._temp_vector[:]
                             - temp_vector_1[:]) / 3.0
            self._temp_vector[:] += temp_error[:]
            self._temp_rate_vector[:] = (
                self.over_dt * (self._temp_vector[:] - self._temp_vector_0[:])
            )
            void_ratio_error[:] = (self._void_ratio_vector[:]
                                   - void_ratio_vector_1[:]) / 3.0
            self._void_ratio_vector[:] += void_ratio_error[:]
            self.update_nodes()
            self.update_integration_points()
            self.update_global_matrices_and_vectors()
            # update the time step
            temp_scale[:] = np.max(np.vstack([
                self._temp_vector[:],
                self._temp_rate_vector[:] * self.time_step,
            ]), axis=0)
            T_scale = float(np.linalg.norm(temp_scale))
            void_ratio_rate[:] = (self._void_ratio_vector[:]
                                  - void_ratio_vector_0[:])
            void_ratio_scale[:] = np.max(np.vstack([
                self._void_ratio_vector[:],
                void_ratio_rate,
            ]), axis=0)
            e_scale = float(np.linalg.norm(void_ratio_scale))
            err_targ_thrm = self.eps_s * T_scale
            err_curr_thrm = float(np.linalg.norm(temp_error))
            err_targ_cnsl = self.eps_s * e_scale
            err_curr_cnsl = float(np.linalg.norm(void_ratio_error))
            # update the time step
            eps_a = np.max([
                err_curr_thrm / T_scale,
                err_curr_cnsl / e_scale,
            ])
            dt1_thrm = dt0 * (err_targ_thrm / err_curr_thrm) ** 0.2
            dt1_cnsl = dt0 * (err_targ_cnsl / err_curr_cnsl) ** 0.2
            dt1 = np.min([dt1_thrm, dt1_cnsl])
            self.time_step = dt1
            dt_list.append(dt0)
            err_list.append(eps_a)
        return dt00, np.array(dt_list), np.array(err_list)
