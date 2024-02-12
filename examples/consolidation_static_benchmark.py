"""Benchmark for large strain consolidation.

This script runs benchmark LSC using geometry,
boundary conditions, and soil properties
from:
    Fox, P.J. and Pu, H. (2015). Benchmark Problems
        for Large Strain Consolidation, J. Geotech.
        Geoenviron. Eng., 141(11), 06015008,
        doi: 10.1061/(ASCE)GT.1943-5606.0001357.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
    unit_weight_water,
)
from frozen_ground_fem.consolidation import (
    ConsolidationAnalysis1D,
    ConsolidationBoundary1D,
)


def solve_consolidation_benchmark(
    Gs: float,
    H_layer: float,
    num_elements: int,
    dt_sim_0: float,
    t_max: float,
    qi: float,
    qf: float,
    ppc0: float = 0.0,
    tol: float = 1e-6,
    stabilize: bool = True,
    s_per_yr: float = 3.1536E+07,
):
    print(f"Gs = {Gs}")
    print(f"H_layer = {H_layer} m")
    print(f"num_elements = {num_elements}")
    print(f"dt_sim_0 = {dt_sim_0} s")
    print(f"t_max = {t_max} s = {t_max / s_per_yr} yr")
    print(f"tol = {tol:0.4e}")

    # define the material properties
    m = Material(
        spec_grav_solids=Gs,
        hyd_cond_index=1.30,
        hyd_cond_0=2.00e-9,
        void_ratio_0_hyd_cond=4.30,
        void_ratio_min=0.30,
        void_ratio_tr=4.30,
        void_ratio_0_comp=2.70,
        comp_index_unfrozen=1.00,
        rebound_index_unfrozen=0.100,
        eff_stress_0_comp=40.0e3,
    )

    # define plotting time increments
    t_plot_targ = np.hstack([
        0.0,
        np.linspace(0.01, 0.1, 10)[:-1],
        np.linspace(0.1, 1.0, 10)[:-1],
        np.linspace(1.0, 60.0, 60),
    ]) * s_per_yr
    n_plot_targ = len(t_plot_targ)
    dt_plot = np.max([0.5 * s_per_yr, dt_sim_0])
    t_plot_extra = t_max - t_plot_targ[-1]
    n_plot = (
        n_plot_targ +
        int(np.floor(t_plot_extra / dt_plot) + 1)
    )
    n_stab_max = 500

    # create consolidation analysis
    # and generate the mesh
    con_static = ConsolidationAnalysis1D(
        z_range=[0.0, H_layer],
        num_elements=num_elements,
        generate=True,
    )
    con_static.implicit_error_tolerance = tol

    # assign material properties to integration points
    for e in con_static.elements:
        for ip in e.int_pts:
            ip.material = m
            ip.pre_consol_stress = ppc0

    # initialize plotting arrays
    z_nod = np.array([nd.z for nd in con_static.nodes])
    z_int = np.array([ip.z for e in con_static.elements for ip in e.int_pts])
    e_nod = np.zeros((len(z_nod), n_plot + 1))
    e_nod_stab = np.zeros_like(z_nod)
    sig_p_int = np.zeros((len(z_int), n_plot + 1))
    hyd_cond_int = np.zeros((len(z_int), n_plot + 1))

    # initialize void ratio profile
    e0, sig_p_0_exp, hyd_cond_0_exp = calculate_static_profile(m, qi, z_nod)
    e1, sig_p_1_exp, hyd_cond_1_exp = calculate_static_profile(m, qf, z_nod)
    sig_p_0_exp *= 1.0e-3
    sig_p_1_exp *= 1.0e-3
    for k, nd in enumerate(con_static.nodes):
        nd.void_ratio = e0[k]
        nd.void_ratio_0 = e0[k]

    # create void ratio boundary conditions
    # and assign them to the mesh
    void_ratio_boundary_0 = ConsolidationBoundary1D(
        (con_static.nodes[0],),
        bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
        bnd_value=e0[0],
    )
    void_ratio_boundary_1 = ConsolidationBoundary1D(
        (con_static.nodes[-1],),
        bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
        bnd_value=e0[-1],
    )
    con_static.add_boundary(void_ratio_boundary_0)
    con_static.add_boundary(void_ratio_boundary_1)

    # start simulation time
    tic = time.perf_counter()

    # initialize global matrices and vectors
    print(f"initialize system, qi = {qi} Pa")
    con_static.time_step = dt_sim_0
    con_static.initialize_global_system(t0=0.0)
    e_nod[:, 0] = e0[:]
    sig_p_int[:, 0] = 1.0e-3 * np.array(
        [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
    )
    hyd_cond_int[:, 0] = np.array(
        [ip.hyd_cond for e in con_static.elements for ip in e.int_pts]
    )

    if stabilize:
        print(f"initial stabilization at qi = {qi} Pa")
        t_con_stab = [0.0]
        s_con_stab = [0.0]
        t_plot = dt_plot
        k_stab = 0
        eps_s = 1.0
        while eps_s > tol and k_stab < n_stab_max:
            dt00, dt_s, err_s = con_static.solve_to(t_plot)
            t_con_stab.append(con_static._t1)
            s_con_stab.append(con_static.calculate_total_settlement())
            eps_s = np.abs((s_con_stab[-1] - s_con_stab[-2]) / s_con_stab[-1])
            print(
                f"t = {con_static._t1 / s_per_yr:0.3f} yr, "
                + f"s_con = {s_con_stab[-1]:0.3f} m, "
                + f"eps_s =  {eps_s:0.4e}, "
                + f"dt = {dt00:0.4e} s"
            )
            e_nod_stab[:] = con_static._void_ratio_vector[:]
            k_stab += 1
            t_plot += dt_plot
        e_nod[:, 0] = e_nod_stab[:]
        sig_p_int[:, 0] = 1.0e-3 * np.array(
            [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
        )
        hyd_cond_int[:, 0] = np.array(
            [ip.hyd_cond for e in con_static.elements for ip in e.int_pts]
        )

    # update boundary conditions for surface load increment
    print(f"apply static load increment, dq = {qf - qi} Pa")
    void_ratio_boundary_0.bnd_value = e1[0]
    void_ratio_boundary_1.bnd_value = e1[-1]
    con_static._t1 = 0.0
    con_static.time_step = dt_sim_0
    for nd in con_static.nodes:
        nd.void_ratio_0 = nd.void_ratio
    con_static.initialize_global_system(0.0)

    t_con = [0.0]
    s_con = [0.0]
    k_plot = 0
    t_plot = 0.0
    eps_s = 1.0
    while (eps_s > tol and k_plot < n_plot) or t_plot < t_plot_targ[-1]:
        k_plot += 1
        if k_plot < n_plot_targ:
            t_plot = t_plot_targ[k_plot]
        else:
            t_plot += dt_plot
        dt00, dt_s, err_s = con_static.solve_to(t_plot)
        t_con.append(con_static._t1)
        s_con.append(con_static.calculate_total_settlement())
        eps_s = np.abs((s_con[-1] - s_con[-2]) / s_con[-1])
        print(
            f"t = {con_static._t1 / s_per_yr:0.3f} yr, "
            + f"s_con = {s_con[-1]:0.3f} m, "
            + f"eps_s =  {eps_s:0.4e}, "
            + f"dt = {dt00:0.4e} s"
        )
        e_nod[:, k_plot] = con_static._void_ratio_vector[:]
        sig_p_int[:, k_plot] = 1.0e-3 * np.array(
            [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
        )
        hyd_cond_int[:, k_plot] = np.array(
            [ip.hyd_cond for e in con_static.elements for ip in e.int_pts]
        )

    toc = time.perf_counter()
    run_time = toc - tic

    # convert settlement to arrays
    t_con = np.array(t_con)
    s_con = np.array(s_con)

    # calculate time to 50 percent settlement
    s_50 = 0.5 * s_con[-1]
    k_50 = 0
    for k, s in enumerate(s_con):
        if s > s_50:
            k_50 = k
            break
    s1 = s_con[k_50]
    s0 = s_con[k_50 - 1]
    t1 = t_con[k_50]
    t0 = t_con[k_50 - 1]
    t_50 = (np.sqrt(t0) + ((np.sqrt(t1) - np.sqrt(t0))
                           * (s_50 - s0) / (s1 - s0))) ** 2

    print(f"Run time = {run_time: 0.4f} s")
    print(f"Total settlement = {s_con[-1]} m")
    print(f"t_50 = {t_50} s = {t_50 / s_per_yr} yr")

    return (
        t_con, s_con,
        z_nod, z_int,
        e_nod, sig_p_int, hyd_cond_int,
        t_50, run_time,
    )


def main():
    # define simulation parameters
    s_per_yr = 365.0 * 86400.0
    H_layer = 10.0
    num_elements = 15
    dt_sim_0 = 1.0e-1
    # tol = 1e-5
    t_max = 80.0 * s_per_yr
    qi = 40.0e3
    qf = 440.0e3

    # set plotting parameters
    plt.rc("font", size=8)
    plt.rc(
        "lines",
        linewidth=0.5,
        color="black",
        markeredgewidth=0.5,
        markeredgecolor="black",
        markerfacecolor="none",
    )

    # expected output
    t_con_exp = np.array([
        0.00,
        0.05,
        0.10,
        0.50,
        1.00,
        2.00,
        3.00,
        4.00,
        5.00,
        10.00,
        20.00,
        40.00,
        60.00,
    ])
    s_con_exp_Gs_1 = np.array([
        0.000,
        0.215,
        0.304,
        0.679,
        0.961,
        1.358,
        1.662,
        1.910,
        2.113,
        2.642,
        2.806,
        2.815,
        2.815,
    ])
    s_con_exp_Gs_278 = np.array([
        0.000,
        0.188,
        0.265,
        0.592,
        0.835,
        1.178,
        1.439,
        1.651,
        1.826,
        2.296,
        2.462,
        2.473,
        2.473,
    ])
    z_exp = np.linspace(0.0, 10.0, 11)
    e_nod_exp_Gs_1 = np.zeros((len(z_exp), 5))
    e_nod_exp_Gs_1[:, 0] = 2.70
    e_nod_exp_Gs_1[:, 1] = [
        1.659,
        2.636,
        2.700,
        2.700,
        2.700,
        2.700,
        2.700,
        2.700,
        2.700,
        2.636,
        1.659,
    ]
    e_nod_exp_Gs_1[:, 2] = [
        1.659,
        1.912,
        2.167,
        2.383,
        2.523,
        2.571,
        2.523,
        2.383,
        2.167,
        1.912,
        1.659,
    ]
    e_nod_exp_Gs_1[:, 3] = [
        1.659,
        1.779,
        1.897,
        1.997,
        2.066,
        2.090,
        2.066,
        1.997,
        1.897,
        1.779,
        1.659,
    ]
    e_nod_exp_Gs_1[:, 4] = 1.659
    e_nod_exp_Gs_278 = np.zeros((len(z_exp), 5))
    e_nod_exp_Gs_278[:, 0] = [
        2.700,
        2.651,
        2.607,
        2.566,
        2.529,
        2.494,
        2.461,
        2.430,
        2.402,
        2.374,
        2.348,
    ]
    e_nod_exp_Gs_278[:, 1] = [
        1.659,
        2.585,
        2.603,
        2.563,
        2.527,
        2.493,
        2.461,
        2.431,
        2.403,
        2.323,
        1.612,
    ]
    e_nod_exp_Gs_278[:, 2] = [
        1.659,
        1.890,
        2.112,
        2.283,
        2.376,
        2.387,
        2.323,
        2.196,
        2.019,
        1.816,
        1.612,
    ]
    e_nod_exp_Gs_278[:, 3] = [
        1.659,
        1.765,
        1.866,
        1.947,
        1.998,
        2.008,
        1.978,
        1.912,
        1.821,
        1.718,
        1.612,
    ]
    e_nod_exp_Gs_1[:, 4] = [
        1.659,
        1.656,
        1.650,
        1.645,
        1.640,
        1.635,
        1.630,
        1.626,
        1.621,
        1.616,
        1.612,
    ]

    # solve neutral buoyancy case
    (
        t_con_Gs_1, s_con_Gs_1,
        z_nod_Gs_1, z_int_Gs_1,
        e_nod_Gs_1, sig_p_int_Gs_1, hyd_cond_int_Gs_1,
        t_50_Gs_1, run_time_Gs_1,
    ) = solve_consolidation_benchmark(
        Gs=1.0, H_layer=H_layer, num_elements=num_elements,
        dt_sim_0=dt_sim_0, t_max=t_max, qi=qi, qf=qf,
        stabilize=False,  # tol=tol,
    )
    t_con_Gs_1_yr = t_con_Gs_1 / s_per_yr

    # solve self weight case
    (
        t_con_Gs_278, s_con_Gs_278,
        z_nod_Gs_278, z_int_Gs_278,
        e_nod_Gs_278, sig_p_int_Gs_278, hyd_cond_int_Gs_278,
        t_50_Gs_278, run_time_Gs_278,
    ) = solve_consolidation_benchmark(
        Gs=2.78, H_layer=H_layer, num_elements=num_elements,
        dt_sim_0=dt_sim_0, t_max=t_max, qi=qi, qf=qf,
        stabilize=False,  # tol=tol,
    )
    t_con_Gs_278_yr = t_con_Gs_278 / s_per_yr

    plt.figure(figsize=(3.5, 4))
    plt.semilogx(
        t_con_Gs_1_yr[1:], s_con_Gs_1[1:], "--k", label="Gs=1.0",
    )
    plt.semilogx(
        t_con_Gs_278_yr[1:], s_con_Gs_278[1:], "-k", label="Gs=2.78",
    )
    plt.semilogx(
        t_con_exp[1:], s_con_exp_Gs_1[1:], label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.semilogx(
        t_con_exp[1:], s_con_exp_Gs_278[1:],  # label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.xlabel(r"Time, $t$ [$yr$]")
    plt.ylabel(r"Settlement, $s$ [$m$]")
    plt.xlim([0.01, 100])
    plt.ylim([3.0, 0.0])
    plt.legend()

    plt.savefig("examples/con_static_bench_settlement.svg")

    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(
        e_nod_Gs_1[:, 0], z_nod_Gs_1, "--k", label="Gs=1.0"
    )
    plt.plot(
        e_nod_Gs_278[:, 0], z_nod_Gs_278, "-k", label="Gs=2.78"
    )
    plt.plot(
        e_nod_exp_Gs_1[:, 0], z_exp, label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_exp_Gs_278[:, 0], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_1[:, 10], z_nod_Gs_1, "--k",
    )
    plt.plot(
        e_nod_exp_Gs_1[:, 1], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_1[::6, 10], z_nod_Gs_1[::6], label="t=0.1 yr",
        marker="o", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_1[:, 20], z_nod_Gs_1, "--k",
    )
    plt.plot(
        e_nod_exp_Gs_1[:, 2], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_1[::6, 20], z_nod_Gs_1[::6], label="t=2 yr",
        marker="s", linestyle="none",
    )
    plt.plot(
        e_nod_exp_Gs_1[:, 3], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_1[:, 23], z_nod_Gs_1, "--k",
    )
    plt.plot(
        e_nod_Gs_1[::6, 23], z_nod_Gs_1[::6], label="t=5 yr",
        marker="^", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_1[:, 78], z_nod_Gs_1, "--k",
    )
    plt.plot(
        e_nod_exp_Gs_1[:, 4], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_1[::6, 78], z_nod_Gs_1[::6], label="t=60 yr",
        marker="D", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[:, 10], z_nod_Gs_278, "-k",
    )
    plt.plot(
        e_nod_exp_Gs_278[:, 1], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[::6, 10], z_nod_Gs_278[::6],  # label="t=0.1 yr",
        marker="o", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[:, 20], z_nod_Gs_278, "-k",
    )
    plt.plot(
        e_nod_exp_Gs_278[:, 2], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[::6, 20], z_nod_Gs_278[::6],  # label="t=2 yr",
        marker="s", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[:, 23], z_nod_Gs_278, "-k",
    )
    plt.plot(
        e_nod_exp_Gs_278[:, 3], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[::6, 23], z_nod_Gs_278[::6],  # label="t=5 yr",
        marker="^", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[:, 78], z_nod_Gs_278, "-k",
    )
    plt.plot(
        e_nod_exp_Gs_278[:, 4], z_exp,
        marker="x", linestyle="none",
    )
    plt.plot(
        e_nod_Gs_278[::6, 78], z_nod_Gs_278[::6],  # label="t=60 yr",
        marker="D", linestyle="none",
    )
    fig.legend()
    plt.xlabel(r"Void Ratio, $e$")
    plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")
    plt.ylim((11, -1))
    plt.xlim((1.5, 3.0))

    plt.subplot(1, 3, 2)
    plt.plot(
        sig_p_int_Gs_1[:, 0], z_int_Gs_1, "--k", label="Gs=1.0"
    )
    plt.plot(
        sig_p_int_Gs_278[:, 0], z_int_Gs_278, "-k", label="Gs=2.78"
    )
    plt.plot(
        sig_p_int_Gs_1[:, 10], z_int_Gs_1, "--k",
    )
    plt.plot(
        sig_p_int_Gs_1[2::5, 10], z_int_Gs_1[2::5], label="t=0.1 yr",
        marker="o", linestyle="none",
    )
    plt.plot(
        sig_p_int_Gs_1[:, 20], z_int_Gs_1, "--k",
    )
    plt.plot(
        sig_p_int_Gs_1[2::5, 20], z_int_Gs_1[2::5], label="t=2 yr",
        marker="s", linestyle="none",
    )
    plt.plot(
        sig_p_int_Gs_1[:, 23], z_int_Gs_1, "--k",
    )
    plt.plot(
        sig_p_int_Gs_1[2::5, 23], z_int_Gs_1[2::5], label="t=5 yr",
        marker="^", linestyle="none",
    )
    plt.plot(
        sig_p_int_Gs_1[:, 78], z_int_Gs_1, "--k",
    )
    plt.plot(
        sig_p_int_Gs_1[2::5, 78], z_int_Gs_1[2::5], label="t=60 yr",
        marker="D", linestyle="none",
    )
    plt.plot(
        sig_p_int_Gs_278[:, 10], z_int_Gs_278, "-k",
    )
    plt.plot(
        sig_p_int_Gs_278[2::5, 10], z_int_Gs_278[2::5],  # label="t=0.1 yr",
        marker="o", linestyle="none",
    )
    plt.plot(
        sig_p_int_Gs_278[:, 20], z_int_Gs_278, "-k",
    )
    plt.plot(
        sig_p_int_Gs_278[2::5, 20], z_int_Gs_278[2::5],  # label="t=2 yr",
        marker="s", linestyle="none",
    )
    plt.plot(
        sig_p_int_Gs_278[:, 23], z_int_Gs_278, "-k",
    )
    plt.plot(
        sig_p_int_Gs_278[2::5, 23], z_int_Gs_278[2::5],  # label="t=5 yr",
        marker="^", linestyle="none",
    )
    plt.plot(
        sig_p_int_Gs_278[:, 78], z_int_Gs_278, "-k",
    )
    plt.plot(
        sig_p_int_Gs_278[2::5, 78], z_int_Gs_278[2::5],  # label="t=60 yr",
        marker="D", linestyle="none",
    )
    plt.xlabel(r"Eff Stress, $\sigma^\prime$ [$kPa$]")
    plt.ylim((11, -1))
    plt.xlim((0.0, 500.0))

    plt.subplot(1, 3, 3)
    plt.semilogx(
        hyd_cond_int_Gs_1[:, 0], z_int_Gs_1, "--k", label="Gs=1.0"
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[:, 0], z_int_Gs_278, "-k", label="Gs=2.78"
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[:, 10], z_int_Gs_1, "--k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[2::5, 10], z_int_Gs_1[2::5], label="t=0.1 yr",
        marker="o", linestyle="none",
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[:, 20], z_int_Gs_1, "--k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[2::5, 20], z_int_Gs_1[2::5], label="t=2 yr",
        marker="s", linestyle="none",
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[:, 23], z_int_Gs_1, "--k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[2::5, 23], z_int_Gs_1[2::5], label="t=5 yr",
        marker="^", linestyle="none",
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[:, 78], z_int_Gs_1, "--k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_1[2::5, 78], z_int_Gs_1[2::5], label="t=60 yr",
        marker="D", linestyle="none",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[:, 10], z_int_Gs_278, "-k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[2::5, 10], z_int_Gs_278[2::5],  # label="t=0.1 yr",
        marker="o", linestyle="none",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[:, 20], z_int_Gs_278, "-k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[2::5, 20], z_int_Gs_278[2::5],  # label="t=2 yr",
        marker="s", linestyle="none",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[:, 23], z_int_Gs_278, "-k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[2::5, 23], z_int_Gs_278[2::5],  # label="t=5 yr",
        marker="^", linestyle="none",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[:, 78], z_int_Gs_278, "-k",
    )
    plt.semilogx(
        hyd_cond_int_Gs_278[2::5, 78], z_int_Gs_278[2::5],  # label="t=60 yr",
        marker="D", linestyle="none",
    )
    plt.xlabel(r"Hyd Cond, $k$ [$m/s$]")
    plt.xlim((1e-11, 2e-10))
    plt.ylim((11, -1))

    plt.savefig("examples/con_static_bench_void_sig_profiles.svg")


def terzaghi_consolidation(z, t, cv, H, ui):
    Tv = cv * t / H**2
    Uz = np.zeros_like(z)
    U_avg = 0.0
    eps_a = 1.0
    eps_s = 1.0e-8
    m = 0
    while eps_a > eps_s:
        M = 0.5 * np.pi * (2 * m + 1)
        dUz = (2.0 / M) * np.sin(M * z / H) * np.exp(-Tv * M**2)
        dU = (2.0 / M**2) * np.exp(-Tv * M**2)
        Uz[:] += dUz[:]
        U_avg += dU
        eps_a = np.max([np.linalg.norm(dUz), dU])
        m += 1
    ue = ui * Uz
    Uz = 1.0 - Uz
    U_avg = 1.0 - U_avg
    return ue, U_avg, Uz


def calculate_static_profile(m, qs, z):
    nnod = len(z)
    Gs = m.spec_grav_solids
    Ccu = m.comp_index_unfrozen
    sig_cu0 = m.eff_stress_0_comp
    e_cu0 = m.void_ratio_0_comp
    gam_p = np.zeros(nnod)
    sig_p = np.zeros(nnod)
    sig_p[0] = qs
    e0 = np.ones(nnod)
    e1 = np.zeros(nnod)
    eps_s = 1e-8
    eps_a = 1.0
    while eps_a > eps_s:
        for k, e in enumerate(e0):
            gam_p[k] = (Gs - 1.0) / (1.0 + e) * unit_weight_water
            if k:
                dz = z[k] - z[k - 1]
                gam_p_avg = 0.5 * (gam_p[k - 1] + gam_p[k])
                sig_p[k] = sig_p[k - 1] + gam_p_avg * dz
            e1[k] = e_cu0 - Ccu * np.log10(sig_p[k] / sig_cu0)
        eps_a = np.linalg.norm(e1 - e0) / np.linalg.norm(e1)
        e0[:] = e1[:]
    hyd_cond = m.hyd_cond_0 * \
        10 ** ((e1 - m.void_ratio_0_hyd_cond) / m.hyd_cond_index)
    return e1, sig_p, hyd_cond


if __name__ == "__main__":
    main()
