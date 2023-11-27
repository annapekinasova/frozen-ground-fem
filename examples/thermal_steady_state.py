import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
)


def main():
    # create thermal analysis object
    # define mesh with 20 elements
    # and cubic interpolation
    ta = ThermalAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 100.0
    ta.generate_mesh(num_elements=20)

    # define material properties
    # and initialize integration points
    mtl = Material(
        thrm_cond_solids=7.0, spec_grav_solids=2.65, spec_heat_cap_solids=741
    )
    void_ratio = 0.3
    for e in ta.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.void_ratio = void_ratio

    # set initial temperature conditions
    T0 = 5.0
    for nd in ta.nodes:
        nd.temp = T0

    # create thermal boundary conditions
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = -10.0
    grad_boundary = ThermalBoundary1D(
        (ta.nodes[-1],),
        (ta.elements[-1].int_pts[-1],),
    )
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 0.2

    # assign thermal boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)

    # **********************************************
    # TIME STEPPING ALGORITHM
    # **********************************************

    # initialize plot
    z_vec = np.array([nd.z for nd in ta.nodes])
    plt.figure(figsize=(6, 9))

    # initialize global matrices and vectors
    ta.time_step = 8.64e4 * 7  # one week, in seconds
    ta.initialize_global_system(t0=0.0)
    plt.plot(
        ta._temp_vector,
        z_vec,
        "-r",
        label="initial conditions",
        linewidth=2.0,
    )

    for k in range(1500):
        # generate temperature distribution plot
        if k and not k % 105:
            plt.plot(ta._temp_vector, z_vec, "--b", linewidth=0.5)
        ta.initialize_time_step()
        ta.iterative_correction_step()

    # finalize plot labels
    plt.plot(
        ta._temp_vector,
        z_vec,
        "-b",
        label="steady state",
        linewidth=2.0,
    )
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("temperature, T [deg C]")
    plt.ylabel("depth, z [m]")
    plt.savefig("examples/thermal_steady_state.png")


if __name__ == "__main__":
    main()
