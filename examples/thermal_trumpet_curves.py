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
    ta.z_max = 50.0
    ta.generate_mesh(num_elements=20)

    # define plotting time increments
    s_per_day = 8.64e4
    s_per_wk = s_per_day * 7.0
    s_per_yr = s_per_day * 365.0
    t_plot_targ = np.linspace(0.0, 30.0, 61) * s_per_yr

    # define material properties
    # and initialize integration points
    mtl = Material(
        thrm_cond_solids=7.0,
        spec_grav_solids=2.65,
        spec_heat_cap_solids=741.0,
    )
    void_ratio = 0.3
    for e in ta.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.void_ratio = void_ratio
            ip.void_ratio_0 = void_ratio

    # set initial temperature conditions
    T0 = -5.0
    for nd in ta.nodes:
        nd.temp = T0

    # define temperature boundary curve
    t_max = 195.0 * s_per_day
    omega = 2.0 * np.pi / s_per_yr
    T_mean = -10.0
    T_amp = 25.0

    def air_temp(t):
        return T_amp * np.cos(omega * (t - t_max)) + T_mean

    # save a plot of the air temperature boundary
    plt.figure(figsize=(6, 4))
    t = np.linspace(0, s_per_yr, 100)
    plt.plot(t / s_per_day, air_temp(t), "-k")
    plt.xlabel("time [days]")
    plt.ylabel("air temp [deg C]")
    plt.savefig("examples/thermal_trumpet_boundary.svg")

    # create thermal boundary conditions
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_function = air_temp
    grad_boundary = ThermalBoundary1D(
        (ta.nodes[-1],),
        (ta.elements[-1].int_pts[-1],),
    )
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 0.03

    # assign thermal boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)

    # **********************************************
    # TIME STEPPING ALGORITHM
    # **********************************************

    plt.rc("font", size=8)

    # initialize plot
    z_vec = np.array([nd.z for nd in ta.nodes])
    plt.figure(figsize=(3.7, 3.7))

    # initialize global matrices and vectors
    ta.time_step = 0.1  # set initial time step small for adaptive
    ta.initialize_global_system(t0=0.0)

    temp_curve = np.zeros((ta.num_nodes, 2))
    for k, tf in enumerate(t_plot_targ):
        if not k:
            temp_curve[:, 0] = ta._temp_vector[:]
            continue
        dt00 = ta.solve_to(tf)[0]
        dT = ta._temp_vector[:] - temp_curve[:, k % 2]
        eps_a = (
            np.linalg.norm(dT) / np.linalg.norm(ta._temp_vector[:])
        )
        dTmax = np.max(np.abs(dT))
        temp_curve[:, k % 2] = ta._temp_vector[:]
        print(
            f"t = {ta._t1 / s_per_yr: 0.5f} years, "
            + f"eps = {eps_a:0.4e}, "
            + f"dTmax = {dTmax: 0.4f} deg C, "
            + f"dt = {dt00 / s_per_wk:0.4e} wks"
        )
        if not k % 10:
            plt.plot(temp_curve[:, 1], z_vec, "--r", linewidth=0.5)
        elif not (k - 1) % 10:
            plt.plot(temp_curve[:, 0], z_vec, "--b", linewidth=0.5)

    # generate converged temperature distribution plot
    plt.plot(temp_curve[:, 0], z_vec, "-b",
             linewidth=2, label="temp dist, jan 1")
    plt.plot(temp_curve[:, 1], z_vec, "-r",
             linewidth=2, label="temp dist, jul 1")
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig("examples/thermal_temp_dist_curves.svg")

    # now run one annual cycle to obtain temperature envelopes
    print("running final cycle to obtain temp envelopes")
    t_plot_targ = np.linspace(30.0, 31.0, 53) * s_per_yr
    temp_curve = np.zeros((ta.num_nodes, 53))
    for k, tf in enumerate(t_plot_targ):
        if not k:
            temp_curve[:, 0] = ta._temp_vector[:]
            continue
        dt00 = ta.solve_to(tf)[0]
        temp_curve[:, k] = ta._temp_vector[:]
        Tmin = np.min(temp_curve[:, k])
        Tmax = np.max(temp_curve[:, k])
        Tmean = np.mean(temp_curve[:, k])
        print(
            f"wk = {k}, "
            + f"dt = {dt00 / s_per_wk:0.4e} wks, "
            + f"Tmin = {Tmin: 0.4f} deg C, "
            + f"Tmax = {Tmax: 0.4f} deg C, "
            + f"Tmean = {Tmean: 0.4f} deg C"
        )

    plt.figure(figsize=(3.7, 3.7))
    temp_min_curve = np.amin(temp_curve, axis=1)
    temp_max_curve = np.amax(temp_curve, axis=1)
    plt.plot(temp_curve[:, 0], z_vec, "--b",
             linewidth=1, label="temp dist, jan 1")
    plt.plot(temp_curve[:, 13], z_vec, ":b",
             linewidth=1, label="temp dist, apr 1")
    plt.plot(temp_curve[:, 26], z_vec, "--r",
             linewidth=1, label="temp dist, jul 1")
    plt.plot(temp_curve[:, 39], z_vec, ":r",
             linewidth=1, label="temp dist, oct 1")
    plt.plot(temp_min_curve, z_vec, "-b", linewidth=2, label="annual minimum")
    plt.plot(temp_max_curve, z_vec, "-r", linewidth=2, label="annual maximum")
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig("examples/thermal_trumpet_curves.svg")


if __name__ == "__main__":
    main()
