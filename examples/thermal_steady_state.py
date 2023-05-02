import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.geometry import (
    BoundaryElement1D,
    Mesh1D,
)

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
)


def main():
    # define mesh
    mesh = Mesh1D()
    mesh.z_min = 0.0
    mesh.z_max = 100.0
    mesh.generate_mesh(num_nodes=20)

    # define material properties
    # and initialize integration point porosity
    mtl = Material(thrm_cond_solids=7.0, dens_solids=2.65e3, spec_heat_cap_solids=741)
    por = 0.3
    for e in mesh.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.porosity = por

    # create geometric boundaries
    # and assign them to the mesh
    upper_boundary = BoundaryElement1D(
        (mesh.nodes[0],),
        (mesh.elements[0].int_pts[0],),
    )
    lower_boundary = BoundaryElement1D(
        (mesh.nodes[-1],),
        (mesh.elements[-1].int_pts[-1],),
    )
    # TODO: modify Mesh1D so that this can be done with method calls
    # mesh.add_boundary(upper_boundary)
    # mesh.add_boundary(lower_boundary)
    mesh._boundary_elements = (
        upper_boundary,
        lower_boundary,
    )

    # create thermal analysis object
    thermal_analysis = ThermalAnalysis1D(mesh)
    ta = thermal_analysis

    # set initial temperature conditions
    T0 = 5.0
    for nd in mesh.nodes:
        nd.temp = T0

    # create thermal boundary conditions
    temp_boundary = ThermalBoundary1D(upper_boundary)
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = 20.0
    grad_boundary = ThermalBoundary1D(lower_boundary)
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 0.2

    # assign thermal boundaries to the analysis
    thermal_analysis._thermal_boundaries = (
        temp_boundary,
        grad_boundary,
    )

    # initialize global matrices and vectors
    for tb in thermal_analysis._thermal_boundaries:
        tb.update_nodes()
    for nd in mesh.nodes:
        thermal_analysis._temp_vector[nd.index] = nd.temp
    thermal_analysis.update_integration_points()
    thermal_analysis.update_heat_flux_vector()
    thermal_analysis.update_heat_flow_matrix()
    thermal_analysis.update_heat_storage_matrix()

    # initialize plot
    z_vec = np.array([nd.z for nd in mesh.nodes])
    plt.figure(figsize=(6, 9))

    # **********************************************
    # TIME STEPPING ALGORITHM
    # **********************************************

    dt = 8.64e5 * 7  # one week, in seconds
    over_dt = 1.0 / dt
    t0 = 0.0
    alpha = 0.5  # 0.5 --> Crank-Nicolson method
    eps_s = 1e-3  # error tolerance, stopping criterion
    max_iter = 100  # maximum number of iterations

    # create list of free node indices
    # that will be updated at each iteration
    # (i.e. are not fixed/Dirichlet boundary conditions)
    free_ind = [nd.index for nd in mesh.nodes]
    for tb in ta._thermal_boundaries:
        if tb.bnd_type == ThermalBoundary1D.BoundaryType.temp:
            free_ind.remove(tb.nodes[0].index)
    free_vec = np.ix_(free_ind)
    free_arr = np.ix_(free_ind, free_ind)

    for k in range(1000):
        # generate temperature distribution plot
        if not k % 100:
            plt.plot(ta._temp_vector, z_vec, label=f"t={t0 / 8.64e5} days")

        # update time coordinate
        t1 = t0 + dt

        # store previous converged matrices and vectors
        ta._temp_vector_0[:] = ta._temp_vector[:]
        ta._heat_flux_vector_0[:] = ta._heat_flux_vector[:]
        ta._heat_flow_matrix_0[:, :] = ta._heat_flow_matrix[:, :]
        ta._heat_storage_matrix_0[:, :] = ta._heat_storage_matrix[:, :]

        # update boundary conditions
        ta.update_thermal_boundary_conditions()
        ta.update_heat_flux_vector()
        ta._weighted_heat_flux_vector[:] = (
            1.0 - alpha
        ) * ta._heat_flux_vector_0 + alpha * ta._heat_flux_vector

        # iterative correction loop
        eps_a = 1.0
        j = 0
        while eps_a > eps_s and j < max_iter:
            # calculate weighted matrices
            ta._weighted_heat_flow_matrix[:, :] = (
                1.0 - alpha
            ) * ta._heat_flow_matrix_0 + alpha * ta._heat_flow_matrix
            ta._weighted_heat_storage_matrix[:, :] = (
                1.0 - alpha
            ) * ta._heat_storage_matrix_0 + alpha * ta._heat_storage_matrix
            ta._coef_matrix_0[:, :] = (
                ta._weighted_heat_storage_matrix * over_dt
                - (1.0 - alpha) * ta._weighted_heat_flow_matrix
            )
            ta._coef_matrix_1[:, :] = (
                ta._weighted_heat_storage_matrix * over_dt
                + alpha * ta._weighted_heat_flow_matrix
            )

            # calculate residual heat flux vector
            ta._residual_heat_flux_vector[:] = (
                ta._coef_matrix_0 @ ta._temp_vector_0
                - ta._coef_matrix_1 @ ta._temp_vector
                - ta._weighted_heat_flux_vector
            )

            # calculate temperature correction
            ta._delta_temp_vector[free_vec] = np.linalg.solve(
                ta._coef_matrix_1[free_arr],
                ta._residual_heat_flux_vector[free_vec],
            )
            ta._temp_vector[free_vec] += ta._delta_temp_vector[free_vec]

            # update error and iteration counter
            eps_a = np.linalg.norm(ta._delta_temp_vector) / np.linalg.norm(
                ta._temp_vector
            )
            j += 1

            # update global matrices
            ta.update_nodes()
            ta.update_integration_points()
            ta.update_heat_flow_matrix()
            ta.update_heat_storage_matrix()

        # increment time step
        t0 = t1

    # finalize plot labels
    plt.ylim(mesh.z_max, mesh.z_min)
    plt.legend()
    plt.xlabel("temperature, T [deg C]")
    plt.ylabel("depth, z [m]")
    plt.savefig("examples/thermal_steady_state.png")


if __name__ == "__main__":
    main()
