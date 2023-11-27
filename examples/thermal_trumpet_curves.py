import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.geometry import (
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
    mesh.z_max = 50.0
    mesh.generate_mesh(num_elements=20)

    # define material properties
    # and initialize integration point porosity
    mtl = Material(
        thrm_cond_solids=7.0, spec_grav_solids=2.65, spec_heat_cap_solids=741
    )
    void_ratio = 0.3
    for e in mesh.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.void_ratio = void_ratio

    # create thermal analysis object
    thermal_analysis = ThermalAnalysis1D(mesh)
    ta = thermal_analysis

    # set initial temperature conditions
    T0 = -5.0
    for nd in mesh.nodes:
        nd.temp = T0

    # define temperature boundary curve
    t_max = 195.0 * 8.64e4
    omega = 2.0 * np.pi / 365 / 8.64e4
    T_mean = -10.0
    T_amp = 25.0

    def air_temp(t):
        return T_amp * np.cos(omega * (t - t_max)) + T_mean

    # save a plot of the air temperature boundary
    plt.figure(figsize=(6, 4))
    t = np.linspace(0, 365 * 8.64e4, 100)
    plt.plot(t / 8.64e4, air_temp(t), "-k")
    plt.xlabel("time [days]")
    plt.ylabel("air temp [deg C]")
    plt.savefig("examples/thermal_trumpet_boundary.png")

    # create thermal boundary conditions
    temp_boundary = ThermalBoundary1D(
        (mesh.nodes[0],),
        (mesh.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_function = air_temp
    grad_boundary = ThermalBoundary1D(
        (mesh.nodes[-1],),
        (mesh.elements[-1].int_pts[-1],),
    )
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 0.03

    # assign thermal boundaries to the analysis
    thermal_analysis.add_boundary(temp_boundary)
    thermal_analysis.add_boundary(grad_boundary)

    # **********************************************
    # TIME STEPPING ALGORITHM
    # **********************************************

    plt.rc("font", size=8)

    # initialize plot
    z_vec = np.array([nd.z for nd in mesh.nodes])
    plt.figure(figsize=(3.7, 3.7))

    # initialize global matrices and vectors
    ta.time_step = 365 * 8.64e4 / 52  # ~one week, in seconds
    thermal_analysis.initialize_global_system(t0=0.0)

    temp_curve = np.zeros((mesh.num_nodes, 52))
    for k in range(1500):
        ta.initialize_time_step()
        ta.iterative_correction_step()
        temp_curve[:, k % 52] = ta._temp_vector
        if not (k % 260):
            print(f"t = {ta._t0 / 8.64e4 / 365: 0.5f} years")
            plt.plot(temp_curve[:, 0], z_vec, "--b", linewidth=0.5)
        elif not (k % 260 - 25):
            print(f"t = {ta._t1 / 8.64e4 / 365: 0.5f} years")
            plt.plot(temp_curve[:, 25], z_vec, "--r", linewidth=0.5)

    # generate converged temperature distribution plot
    plt.plot(temp_curve[:, 0], z_vec, "-b",
             linewidth=2, label="temp dist, jan 1")
    plt.plot(temp_curve[:, 25], z_vec, "-r",
             linewidth=2, label="temp dist, jun 1")
    plt.ylim(mesh.z_max, mesh.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig("examples/thermal_temp_dist_curves.svg")

    plt.figure(figsize=(3.7, 3.7))
    temp_min_curve = np.amin(temp_curve, axis=1)
    temp_max_curve = np.amax(temp_curve, axis=1)
    plt.plot(temp_curve[:, 0], z_vec, "--b",
             linewidth=1, label="temp dist, jan")
    plt.plot(temp_curve[:, 13], z_vec, ":b",
             linewidth=1, label="temp dist, apr")
    plt.plot(temp_curve[:, 26], z_vec, "--r",
             linewidth=1, label="temp dist, jul")
    plt.plot(temp_curve[:, 39], z_vec, ":r",
             linewidth=1, label="temp dist, oct")
    plt.plot(temp_min_curve, z_vec, "-b", linewidth=2, label="annual minimum")
    plt.plot(temp_max_curve, z_vec, "-r", linewidth=2, label="annual maximum")
    plt.ylim(mesh.z_max, mesh.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig("examples/thermal_trumpet_curves.svg")


if __name__ == "__main__":
    main()
