"""
Thermal Spin-up test example based on paper Ross et al.(2022)
Arctic Science 8: 362–394 (2022) dx.doi.org/10.1139/as-2021-0013
for cold and warm permafrost models
with climate BC for cold (MAAT = −13 °C) and warm (MAAT = −4 °C) permafrost
for fully saturated case with depth of 200 m
"""

import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
)

def main():
    # create thermal analysis object
    # define mesh with 50 elements
    # and cubic interpolation
    # depth of cold permafrost is 200 m
    # depth of warm permafrost is 100 m
    ta = ThermalAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 10.0
    # ta.z_max = 100.0
    ta.generate_mesh(num_elements=5)
    # ta.implicit_error_tolerance = 1e-6

    # define material properties for
    # cold permafrost
    # and initialize integration points
    mtl = Material(
        thrm_cond_solids=4.116,
        spec_grav_solids=2.65,
        spec_heat_cap_solids=1.865E+06,
    )
    void_ratio = 1. / 3.
    for e in ta.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.void_ratio = void_ratio
            ip.void_ratio_0 = void_ratio

    # set initial temperature conditions
    # with two scenarios T0 = 0.0 and  -9.0 deg C for cold permafrost
    # with two scenarios T0 = 0.0 and  -1.0 deg C for warm permafrost
    T0 = 0.0
    # T0 = -9.0
    # T0 = -1.0
    for nd in ta.nodes:
        nd.temp = T0
        nd.void_ratio_0 = void_ratio
        nd.void_ratio = void_ratio

    # define temperature boundary curve
    t_max = 212.0 * 8.64e4
    omega = 2.0 * np.pi / 365 / 8.64e4
    T_mean = -13.9
    T_amp = 25.4

    def air_temp(t):
        return T_amp * np.cos(omega * (t - t_max)) + T_mean

    # save a plot of the air temperature boundary
    plt.figure(figsize=(6, 4))
    t = np.linspace(0, 365 * 8.64e4, 100)
    plt.plot(t / 8.64e4, air_temp(t), "-k")
    plt.xlabel("time [days]")
    plt.ylabel("air temp [deg C]")
    plt.savefig("examples/spin_up_air_temp_boundary_cold.svg")

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
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.heat_flux
    grad_boundary.bnd_value = 0.05

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
    ta.time_step = 0.5 * 8.64e4     # 0.5 days in s
    ta.initialize_global_system(t0=0.0)

    # print(ta._heat_flux_vector)
    # print(ta.elements[-1].int_pts[-1].thrm_cond)
    # print(ta.nodes[-1].temp)

    n_per_yr = 365 * 2
    temp_curve = np.zeros((ta.num_nodes, n_per_yr))
    temp_curve_0 = np.zeros((ta.num_nodes, n_per_yr))
    t_end = 1500.0 * 365.0 * 8.64e4
    k = 0
    while ta._t0 < t_end:
        ta.initialize_time_step()
        ta.iterative_correction_step()
        kk = k % n_per_yr
        temp_curve_0[:, kk] = temp_curve[:, kk]
        temp_curve[:, kk] = ta._temp_vector[:]
        if not (k % 3650):
            dT = temp_curve[:, 0] - temp_curve_0[:, 0]
            eps_a = np.linalg.norm(dT) / np.linalg.norm(temp_curve[:, 0])
            Tgmin = np.min(temp_curve[:, 0])
            Tgmax = np.max(temp_curve[:, 0])
            Tgmean = np.mean(temp_curve[:, 0])
            dTmax = np.max(np.abs(dT))
            dTbase = np.abs(dT[-1])
            # print(temp_curve[::5, 0])
            # print(ta._temp_vector[::5])
            print(
                f"t = {ta._t0 / 8.64e4 / 365: 0.5f} years, "
                + f"Tg,min = {Tgmin: 0.5f} deg C, "
                + f"Tg,max = {Tgmax: 0.5f} deg C, "
                + f"Tg,mean = {Tgmean: 0.5f} deg C, "
                + f"dT,max = {dTmax: 0.5f} deg C, "
                + f"dT,base = {dTbase: 0.5f} deg C, "
                + f"eps_a = {eps_a: 0.4e}"
            )
            # plt.plot(temp_curve[:, 0], z_vec, "--b", linewidth=0.5)
        elif not (k % 3650 - 365):
            dT = temp_curve[:, -365] - temp_curve_0[:, -365]
            eps_a = np.linalg.norm(dT) / np.linalg.norm(temp_curve[:, -365])
            Tgmin = np.min(temp_curve[:, -365])
            Tgmax = np.max(temp_curve[:, -365])
            Tgmean = np.mean(temp_curve[:, -365])
            dTmax = np.max(np.abs(dT))
            dTbase = np.abs(dT[-1])
            # print(temp_curve[::5, -365])
            # print(ta._temp_vector[::5])
            print(
                f"t = {ta._t0 / 8.64e4 / 365: 0.5f} years, "
                + f"Tg,min = {Tgmin: 0.5f} deg C, "
                + f"Tg,max = {Tgmax: 0.5f} deg C, "
                + f"Tg,mean = {Tgmean: 0.5f} deg C, "
                + f"dT,max = {dTmax: 0.5f} deg C, "
                + f"dT,base = {dTbase: 0.5f} deg C, "
                + f"eps_a = {eps_a: 0.4e}"
            )
            # plt.plot(temp_curve[:, 25], z_vec, "--r", linewidth=0.5)
        k += 1

    # generate converged temperature distribution plot
    plt.plot(temp_curve[:, 0], z_vec, "-b",
             linewidth=2, label="temp dist, jan 1")
    plt.plot(temp_curve[:, -365], z_vec, "-r",
             linewidth=2, label="temp dist, jul 1")
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig("examples/spin_up_ground_temp_dist_cold.svg")

    plt.figure(figsize=(3.7, 3.7))
    temp_min_curve = np.amin(temp_curve, axis=1)
    temp_max_curve = np.amax(temp_curve, axis=1)
    plt.plot(temp_curve[:, 0], z_vec, "--b",
             linewidth=1, label="temp dist, jan")
    plt.plot(temp_curve[:, 182], z_vec, ":b",
             linewidth=1, label="temp dist, apr")
    plt.plot(temp_curve[:, 365], z_vec, "--r",
             linewidth=1, label="temp dist, jul")
    plt.plot(temp_curve[:, 548], z_vec, ":r",
             linewidth=1, label="temp dist, oct")
    plt.plot(temp_min_curve, z_vec, "-b", linewidth=2, label="annual minimum")
    plt.plot(temp_max_curve, z_vec, "-r", linewidth=2, label="annual maximum")
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig("examples/thermal_trumpet_curves.svg")


if __name__ == "__main__":
    main()
