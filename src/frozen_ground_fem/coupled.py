"""coupled.py
Module for coupled thermal and large strain consolidation physics
using the finite element method.
"""

__all__ = [
    "ThermalBoundary1D",
    "HydraulicBoundary1D",
    "ConsolidationBoundary1D",
    "CoupledElement1D",
    "CoupledAnalysis1D",
]

import numpy as np
import numpy.typing as npt

from . import (
    unit_weight_water as gam_w,
    spec_grav_ice as Gi,
    ThermalElement1D,
    ThermalBoundary1D,
    ThermalAnalysis1D,
    ConsolidationElement1D,
    ConsolidationBoundary1D,
    HydraulicBoundary1D,
    ConsolidationAnalysis1D,
)
from .geometry import (
    Mesh1D,
)

_ONE_MINUS_Gi = 1.0 - Gi


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

    def initialize_integration_points_primary(self) -> None:
        """Initializes primary variables
        at the integration points for coupled thermal and
        large strain consolidation physics. Calls the parent methods
        from ThermalElement1D and ConsolidationElement1D.
        """
        ThermalElement1D.initialize_integration_points_primary(self)
        ConsolidationElement1D.initialize_integration_points_primary(self)

    def initialize_integration_points_secondary(self) -> None:
        """Initializes secondary variables at the integration points
        for coupled thermal and large strain consolidation physics.

        Notes
        -----
        This method initializes the secondary variables at the integration points
        by updating
        the void ratio, pre-consolidation stress, and other parameters based
        on the temperature. It also sets reference stresses and void ratios
        for frozen states and resets effective and pre-consolidation stresses
        as needed. Finally, it calls `update_integration_points_secondary`
        to update the integration points.
        """
        for ip in self.int_pts:
            e0 = ip.void_ratio_0
            ppc = ip.pre_consol_stress
            ppc0, _ = ip.material.eff_stress(e0, ppc)
            if ppc0 > ppc:
                ip.pre_consol_stress = ppc0
            if ip.temp < 0.0:
                ip.void_ratio_0_ref_frozen = ip.void_ratio
                ip.tot_stress_0_ref_frozen = ip.tot_stress
                ip.eff_stress = 0.0
                ip.eff_stress_gradient = 0.0
                ip.pre_consol_stress = 0.0
            else:
                ip.void_ratio_0_ref_frozen = 0.0
                ip.tot_stress_0_ref_frozen = 0.0
                ip.loc_stress = 0.0
                ip.tot_stress_gradient = 0.0
        self.update_integration_points_secondary()

    def update_integration_points_primary(self) -> None:
        """Updates primary variables (temperature and void ratio)
        at the integration points for coupled thermal and
        large strain consolidation physics.
        Calls the parent methods from ThermalElement1D and
        ConsolidationElement1D.
        """
        ThermalElement1D.update_integration_points_primary(self)
        ConsolidationElement1D.update_integration_points_primary(self)

    def update_integration_points_secondary(self) -> None:
        """Updates secondary variables at the integration points
        for coupled thermal and large strain consolidation physics.

        Notes
        -----
        This method updates the secondary variables
        at the integration points by computing
        the gradient matrix, updating local stress states, void ratios,
        pre-consolidation stresses, and other parameters based on the
        temperature and void ratio. It handles both frozen and unfrozen
        states and updates hydraulic conductivity and water flux as needed.
        """
        ThermalElement1D.update_integration_points_secondary(self)
        ee = np.array([nd.void_ratio for nd in self.nodes])
        jac = self.jacobian
        for ip in self.int_pts:
            B = self._gradient_matrix(ip.local_coord, jac)
            e0 = ip.void_ratio_0
            ep = ip.void_ratio
            de_dZ = (B @ ee)[0]
            T0 = ip.temp__0
            T = ip.temp
            # soil is frozen
            if T < 0.0:
                # set reference stress and void ratio for frozen state
                if T0 >= 0.0:
                    ip.void_ratio_0_ref_frozen = ep
                    ip.tot_stress_0_ref_frozen = ip.tot_stress
                    # reset effective stress and pre-consolidation stress
                    ip.eff_stress = 0.0
                    ip.eff_stress_gradient = 0.0
                    ip.pre_consol_stress = 0.0
                # update local stress state
                # and total stress gradient (for stiffness matrix)
                e_f0 = ip.void_ratio_0_ref_frozen
                sig_f0 = ip.tot_stress_0_ref_frozen
                sig, dsig_de = ip.material.tot_stress(
                    T,
                    ep,
                    e_f0,
                    sig_f0,
                )
                ip.loc_stress = sig
                ip.tot_stress_gradient = dsig_de
            # soil is unfrozen
            else:
                # update residual stress and pre-consolidation stress
                # if thawed on this step
                if T0 < 0.0:
                    ppc = ip.material.res_stress(ep)
                    ip.pre_consol_stress = ppc
                    # reset total stress parameters
                    ip.void_ratio_0_ref_frozen = 0.0
                    ip.tot_stress_0_ref_frozen = 0.0
                    ip.loc_stress = 0.0
                    ip.tot_stress_gradient = 0.0
                # update effective stress
                ppc = ip.pre_consol_stress
                sig, dsig_de = ip.material.eff_stress(ep, ppc)
                if sig > ppc:
                    ip.pre_consol_stress = sig
                ip.eff_stress = sig
                ip.eff_stress_gradient = dsig_de
                # update hydraulic conductivity and water flux
                hyd_cond, dk_de = ip.material.hyd_cond(ep, 1.0, True)
                ip.hyd_cond = hyd_cond
                ip.hyd_cond_gradient = dk_de
                if not hyd_cond:
                    self.water_flux_rate = 0.0
                else:
                    Gs = ip.material.spec_grav_solids
                    e_ratio = (1.0 + e0) / (1.0 + ep)
                    ip.water_flux_rate = (
                        -hyd_cond
                        / gam_w
                        * e_ratio
                        * ((Gs - 1.0) * gam_w / (1.0 + e0) - dsig_de * de_dZ)
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
        """
        return self._boundaries

    def add_boundary(
        self,
        new_boundary: ThermalBoundary1D | ConsolidationBoundary1D | HydraulicBoundary1D,
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
        to generate CoupledElement1D objects.
        """
        self._elements = tuple(
            CoupledElement1D(
                tuple(self.nodes[order * k + j] for j in range(order + 1)), order
            )
            for k in range(num_elements)
        )

    def initialize_global_matrices_and_vectors(self):
        """Initializes global matrices and vectors for coupled thermal and
        large strain consolidation physics.
        """
        ThermalAnalysis1D.initialize_global_matrices_and_vectors(self)
        ConsolidationAnalysis1D.initialize_global_matrices_and_vectors(self)

    def initialize_free_index_arrays(self) -> None:
        """Initializes free index arrays for coupled thermal and
        large strain consolidation physics.

        Notes
        -----
        This method initializes the free index arrays by calling the
        corresponding methods in ThermalAnalysis1D and
        ConsolidationAnalysis1D classes. It stores the free vectors and
        arrays for thermal and consolidation analyses separately.
        """
        ThermalAnalysis1D.initialize_free_index_arrays(self)
        self._free_vec_thrm = tuple(self._free_vec)
        self._free_arr_thrm = tuple(self._free_arr)
        ConsolidationAnalysis1D.initialize_free_index_arrays(self)
        self._free_vec_cnsl = tuple(self._free_vec)
        self._free_arr_cnsl = tuple(self._free_arr)

    def initialize_solution_variable_vectors(self) -> None:
        """Initializes solution variable vectors for coupled thermal and
        large strain consolidation physics.
        """
        ThermalAnalysis1D.initialize_solution_variable_vectors(self)
        ConsolidationAnalysis1D.initialize_solution_variable_vectors(self)

    def store_converged_matrices(self) -> None:
        """Stores converged solution vectors for coupled thermal and
        large strain consolidation physics.
        """
        ThermalAnalysis1D.store_converged_matrices(self)
        ConsolidationAnalysis1D.store_converged_matrices(self)

    def update_boundary_conditions(self, time: float) -> None:
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
            elif tb.bnd_type == ConsolidationBoundary1D.BoundaryType.void_ratio:
                for nd in tb.nodes:
                    self._void_ratio_vector[nd.index] = nd.void_ratio

    def update_nodes(self) -> None:
        """Updates the temperature and void ratio values at the nodes
        in the mesh.
        """
        self._temp_rate_vector[:] = (
            self._temp_vector[:] - self._temp_vector_0[:]
        ) * self.over_dt
        for nd in self.nodes:
            nd.temp = self._temp_vector[nd.index]
            nd.temp_rate = self._temp_rate_vector[nd.index]
            nd.void_ratio = self._void_ratio_vector[nd.index]

    def update_global_matrices_and_vectors(self) -> None:
        """Updates global initial matrices for coupled thermal and
        large strain consolidation physics.
        """
        ThermalAnalysis1D.update_global_matrices_and_vectors(self)
        ConsolidationAnalysis1D.update_global_matrices_and_vectors(self)

    def calculate_solution_vector_correction(self) -> None:
        """Calculates the solution vector correction for coupled thermal and
        large strain consolidation physics.
        """
        # need to set _free_vec and _free_arr
        # because these are the variables used by
        # the calculate_solution_vector_correction() methods
        self._free_vec = self._free_vec_thrm
        self._free_arr = self._free_arr_thrm
        ThermalAnalysis1D.calculate_solution_vector_correction(self)
        self._free_vec = self._free_vec_cnsl
        self._free_arr = self._free_arr_cnsl
        ConsolidationAnalysis1D.calculate_solution_vector_correction(self)

    def update_void_ratio_phase_change(self) -> None:
        """Updates the void ratio based on phase change for coupled thermal
        and large strain consolidation physics.

        Notes
        -----
        This method updates the void ratio for each node in the mesh based
        on the degree of saturation of water. It calculates the change in
        void ratio due to phase change and updates the global void ratio
        vector accordingly.
        """
        for nd in self.nodes:
            dSw = nd.deg_sat_water - nd.deg_sat_water__0
            de = -_ONE_MINUS_Gi * nd.void_ratio_0 * dSw
            nd.void_ratio += de
        for nd in self.nodes:
            self._void_ratio_vector[nd.index] = nd.void_ratio

    def update_iteration_variables(self) -> None:
        """Updates iteration variables for coupled thermal and
        large strain consolidation physics.
        """
        eps_a_thrm = float(
            np.linalg.norm(self._delta_temp_vector) / np.linalg.norm(self._temp_vector)
        )
        eps_a_cnsl = float(
            np.linalg.norm(self._delta_void_ratio_vector)
            / np.linalg.norm(self._void_ratio_vector)
        )
        self._eps_a = np.max([eps_a_thrm, eps_a_cnsl])
        self._iter += 1

    def initialize_system_state_variables(self):
        """Initializes system state variables for coupled thermal and
        large strain consolidation physics.
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
        self._void_ratio_vector_0_1 = np.zeros_like(self._void_ratio_vector)
        self._void_ratio_vector_1 = np.zeros_like(self._void_ratio_vector)
        self._void_ratio_error = np.zeros_like(self._void_ratio_vector)
        self._void_ratio_rate = np.zeros_like(self._void_ratio_vector)
        self._void_ratio_scale = np.zeros_like(self._void_ratio_vector)
        self._pre_consol_stress__0 = np.zeros(
            (
                self.num_elements,
                num_int_pt_per_element,
            )
        )

    def save_system_state(self):
        """Saves the current system state variables for coupled thermal and
        large strain consolidation physics.
        """
        self._temp_vector_0_0[:] = self._temp_vector_0[:]
        self._temp_vector_0_1[:] = self._temp_vector[:]
        self._deg_sat_water_0_0[:] = np.array(
            [nd.deg_sat_water__0 for nd in self.nodes]
        )
        self._deg_sat_water_0_1[:] = np.array([nd.deg_sat_water for nd in self.nodes])
        self._void_ratio_vector_0_1[:] = self._void_ratio_vector[:]
        for ke, e in enumerate(self.elements):
            for jip, ip in enumerate(e.int_pts):
                self._temp__0[ke, jip] = ip.temp__0
                self._vol_water_cont__0[ke, jip] = ip.vol_water_cont__0
                self._pre_consol_stress__0[ke, jip] = ip.pre_consol_stress

    def load_system_state(self, t0: float, t1: float, dt: float):
        """Loads the saved system state variables for coupled thermal and
        large strain consolidation physics.

        Parameters
        ----------
        t0 : float
            The initial time (at the beginning of the time step).
        t1 : float
            The final time (at the end of the time step).
        dt : float
            The time step.

        Notes
        -----
        This method loads the previously saved state of the system including
        temperature, degree of saturation, void ratio, volumetric water
        content, and pre-consolidation stress for each node and integration
        point in the mesh. It updates nodes and integration points with
        loaded values and prepares the system for the next time step.
        """
        self._t0 = t0
        self._t1 = t1
        self.time_step = dt
        self._temp_vector_0[:] = self._temp_vector_0_0[:]
        self._temp_vector[:] = self._temp_vector_0_1[:]
        self._void_ratio_vector[:] = self._void_ratio_vector_0_1[:]
        for nd, Sw0, Sw1 in zip(
            self.nodes,
            self._deg_sat_water_0_0,
            self._deg_sat_water_0_1,
        ):
            nd.deg_sat_water__0 = Sw0
            nd.deg_sat_water = Sw1
        for e, T0e, thw0_e, ppc0_e in zip(
            self.elements,
            self._temp__0,
            self._vol_water_cont__0,
            self._pre_consol_stress__0,
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
                self._void_ratio_vector_1[:] = self._void_ratio_vector[:]
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
                self._void_ratio_error[:] = (
                    self._void_ratio_vector[:] - self._void_ratio_vector_1[:]
                ) / 3.0
                self._void_ratio_vector[:] += self._void_ratio_error[:]
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
                            np.abs(self._temp_vector[:]),
                            2.0 * np.abs(self._temp_rate_vector[:]) * self.time_step,
                        ]
                    ),
                    axis=0,
                )
                T_scale = float(np.linalg.norm(self._temp_scale))
                self._void_ratio_rate[:] = (
                    self._void_ratio_vector[:] - self._void_ratio_vector_0_1[:]
                )
                self._void_ratio_scale[:] = np.max(
                    np.vstack(
                        [
                            self._void_ratio_vector[:],
                            np.abs(self._void_ratio_rate[:]),
                        ]
                    ),
                    axis=0,
                )
                e_scale = float(np.linalg.norm(self._void_ratio_scale))
                err_targ_thrm = self.eps_s * T_scale
                err_curr_thrm = float(np.linalg.norm(self._temp_error))
                err_targ_cnsl = self.eps_s * e_scale
                err_curr_cnsl = float(np.linalg.norm(self._void_ratio_error))
                # update the time step
                eps_a = np.max(
                    [
                        err_curr_thrm / T_scale,
                        err_curr_cnsl / e_scale,
                    ]
                )
                dt1_thrm = dt0 * (err_targ_thrm / err_curr_thrm) ** 0.2
                dt1_cnsl = dt0 * (err_targ_cnsl / err_curr_cnsl) ** 0.2
                dt1 = np.min([dt1_thrm, dt1_cnsl])
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
