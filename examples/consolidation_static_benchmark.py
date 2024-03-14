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
import numpy.typing as npt
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
    unit_weight_water,
)
from frozen_ground_fem.consolidation import (
    ConsolidationAnalysis1D,
    ConsolidationBoundary1D,
    HydraulicBoundary1D,
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

    # compute initial excess pore pressure
    ui = qf - qi
    ui_kPa = ui * 1.0e-3

    print(f"H_layer = {H_layer} m")
    print(f"qi = {qi} Pa = {qi*1e-3} kPa")
    print(f"qf = {qf} Pa = {qf*1e-3} kPa")
    print(f"ui = {ui} Pa = {ui_kPa} kPa")
    print(f"Gs = {m.spec_grav_solids}")
    print(f"Ck = {m.hyd_cond_index}")
    print(f"k0 = {m.hyd_cond_0} m/s")
    print(f"e0k = {m.void_ratio_0_hyd_cond}")
    print(f"Cc = {m.comp_index_unfrozen}")
    print(f"Cr = {m.rebound_index_unfrozen}")
    print(f"sig_p_0 = {m.eff_stress_0_comp} Pa"
          + f" = {m.eff_stress_0_comp*1e-3} kPa")
    print(f"e0sig = {m.void_ratio_0_comp}")
    print(f"ppc0 = {ppc0} Pa = {ppc0*1e-3} kPa")
    print(f"num_elements = {num_elements}")
    print(f"dt_sim_0 = {dt_sim_0} s = {dt_sim_0 / s_per_yr} yr")
    print(f"t_max = {t_max} s = {t_max / s_per_yr} yr")
    print(f"tol = {tol:0.4e}")

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
    n_stab_max = 50

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
    Uz_nod = np.zeros_like(e_nod)
    e_nod_stab = np.zeros_like(z_nod)
    sig_p_int = np.zeros((len(z_int), n_plot + 1))
    ue_norm_int = np.zeros_like(sig_p_int)
    hyd_cond_int = np.zeros((len(z_int), n_plot + 1))

    # initialize void ratio profile
    (
        e0, sig_p_0_exp, hyd_cond_0_exp,
        e1, sig_p_1_exp, hyd_cond_1_exp,
    ) = calculate_static_profile(
        z_nod, m, qi, (qf - qi), ppc0,
    )
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
        bnd_value_1=qi,
    )
    void_ratio_boundary_1 = ConsolidationBoundary1D(
        (con_static.nodes[-1],),
        bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
        bnd_value=e0[-1],
    )
    con_static.add_boundary(void_ratio_boundary_0)
    con_static.add_boundary(void_ratio_boundary_1)

    # create hydraulic boundary conditions
    # and assign them to the mesh
    hyd_boundary = HydraulicBoundary1D(
        (con_static.nodes[0], ),
        bnd_type=HydraulicBoundary1D.BoundaryType.fixed_head,
        bnd_value=H_layer,
    )
    con_static.add_boundary(hyd_boundary)

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
    ue_norm_int[:, 0] = 1.0
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
                + f"s = {s_con_stab[-1]:0.3f} m, "
                + f"eps =  {eps_s:0.4e}, "
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
    void_ratio_boundary_0.bnd_value_1 = qf
    void_ratio_boundary_1.bnd_value = e1[-1]
    con_static._t1 = 0.0
    con_static.time_step = dt_sim_0
    for nd in con_static.nodes:
        nd.void_ratio_0 = nd.void_ratio
    con_static.initialize_global_system(0.0)

    t_con = [0.0]
    s_con = [0.0]
    U_con = [0.0]
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
        UU, Uz = con_static.calculate_degree_consolidation(e1)
        U_con.append(UU)
        ue = np.array([
            ip.exc_pore_pressure
            for e in con_static.elements
            for ip in e.int_pts
        ]) * 1.0e-3
        u = np.array([
            ip.pore_pressure
            for e in con_static.elements
            for ip in e.int_pts
        ]) * 1.0e-3
        sig = np.array([
            ip.tot_stress
            for e in con_static.elements
            for ip in e.int_pts
        ]) * 1.0e-3
        sig_p = np.array([
            ip.eff_stress
            for e in con_static.elements
            for ip in e.int_pts
        ]) * 1.0e-3
        # print(sig)
        # print(u)
        # print(sig_p)
        # print(ue)
        eps_s = np.abs((s_con[-1] - s_con[-2]) / s_con[-1])
        print(
            f"t = {con_static._t1 / s_per_yr:0.3f} yr, "
            + f"s = {s_con[-1]:0.3f} m, "
            + f"U = {U_con[-1]:0.4f}, "
            + f"ue_max = {np.max(ue):0.4f} kPa, "
            + f"ue_mean = {np.mean(ue):0.4f} kPa, "
            + f"eps =  {eps_s:0.4e}, "
            + f"dt = {dt00:0.4e} s"
        )
        e_nod[:, k_plot] = con_static._void_ratio_vector[:]
        Uz_nod[:, k_plot] = Uz[:]
        sig_p_int[:, k_plot] = 1.0e-3 * np.array(
            [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
        )
        # ue_norm_int[:, k_plot] = 1.0 - (
        #     sig_p_int[:, k_plot] - sig_p_int[:, 0]
        # ) / ui_kPa
        ue_norm_int[:, k_plot] = ue[:] / ui_kPa
        hyd_cond_int[:, k_plot] = np.array(
            [ip.hyd_cond for e in con_static.elements for ip in e.int_pts]
        )

    toc = time.perf_counter()
    run_time = toc - tic

    # convert settlement to arrays
    t_con = np.array(t_con)
    s_con = np.array(s_con)
    U_con = np.array(U_con)

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
    print(f"Final degree of consolidation = {U_con[-1]:0.4f}")
    print(f"t_50 = {t_50} s = {t_50 / s_per_yr} yr")

    return (
        t_con, s_con, U_con,
        z_nod, z_int,
        e_nod, Uz_nod,
        sig_p_int, ue_norm_int, hyd_cond_int,
        t_50, run_time,
    )


def main():
    # define simulation parameters
    s_per_yr = 365.0 * 86400.0
    H_layer = 10.0
    num_elements = 10
    dt_sim_0 = 1.0e-1
    t_max = 80.0 * s_per_yr
    qi = 40.0e3
    qf = 440.0e3
    ppc0 = 200.52773e3
    tol = 1e-5
    stabilize = False

    # set plotting parameters
    plt.rc("font", size=8)
    plt.rc(
        "lines",
        linewidth=0.5,
        color="black",
        markeredgewidth=0.5,
        markeredgecolor="black",
        markerfacecolor="none",
        markersize=4,
    )

    # plot indices at 0.0, 0.1, 2, 5, and 60 years
    k_plot = [0, 10, 20, 23, 78]
    k_plot_labels = [
        "Initial",
        "t=0.1 yr",
        "t=2 yr",
        "t=5 yr",
        "t=60 yr",
    ]
    k_plot_lts = [
        "-",
        "o",
        "s",
        "^",
        "D",
    ]
    t_FP15 = np.array([
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
    z_FP15 = np.linspace(0.0, 10.0, 11)

    # load expected settlement
    s_FP15 = np.loadtxt(
        "examples/FoxPu2015_Settlement.csv",
        delimiter=",",
    )
    s_FP15_Gs_1 = s_FP15[:, 0]
    s_FP15_Gs_278 = s_FP15[:, 1]
    s_FP15_Gs_1_OC = s_FP15[:, 2]
    s_FP15_Gs_278_OC = s_FP15[:, 3]

    # load expected degree of consolidation
    U_FP15_Gs_1 = s_FP15[:, 4] * 1e-2
    U_FP15_Gs_278 = s_FP15[:, 5] * 1e-2
    U_FP15_Gs_1_OC = s_FP15[:, 6] * 1e-2
    U_FP15_Gs_278_OC = s_FP15[:, 7] * 1e-2

    # load expected void ratio profiles
    e_FP15 = np.loadtxt(
        "examples/FoxPu2015_VoidRatio.csv",
        delimiter=",",
    )
    e_FP15_Gs_1 = e_FP15[:, 0:10:2]
    e_FP15_Gs_278 = e_FP15[:, 1:10:2]
    e_FP15_Gs_1_OC = e_FP15[:, 10::2]
    e_FP15_Gs_278_OC = e_FP15[:, 11::2]

    # load expected (normalized) excess pore pressure profiles
    ue_FP15 = np.loadtxt(
        "examples/FoxPu2015_NormExcPorePress.csv",
        delimiter=",",
    )
    ue_FP15_Gs_1 = ue_FP15[:, 0:10:2]
    ue_FP15_Gs_278 = ue_FP15[:, 1:10:2]
    ue_FP15_Gs_1_OC = ue_FP15[:, 10::2]
    ue_FP15_Gs_278_OC = ue_FP15[:, 11::2]

    # solve neutral buoyancy case
    # normally consolidated
    (
        t_con_Gs_1, s_con_Gs_1, U_con_Gs_1,
        z_nod_Gs_1, z_int_Gs_1,
        e_nod_Gs_1, Uz_nod_Gs_1,
        sig_p_int_Gs_1, ue_norm_int_Gs_1, hyd_cond_int_Gs_1,
        t_50_Gs_1, run_time_Gs_1,
    ) = solve_consolidation_benchmark(
        Gs=1.0, H_layer=H_layer, num_elements=num_elements,
        dt_sim_0=dt_sim_0, t_max=t_max, qi=qi, qf=qf, ppc0=0.0,
        stabilize=stabilize, tol=tol,
    )
    t_con_Gs_1_yr = t_con_Gs_1 / s_per_yr

    # solve self weight case
    # normally consolidated
    (
        t_con_Gs_278, s_con_Gs_278, U_con_Gs_278,
        z_nod_Gs_278, z_int_Gs_278,
        e_nod_Gs_278, Uz_nod_Gs_278,
        sig_p_int_Gs_278, ue_norm_int_Gs_278, hyd_cond_int_Gs_278,
        t_50_Gs_278, run_time_Gs_278,
    ) = solve_consolidation_benchmark(
        Gs=2.78, H_layer=H_layer, num_elements=num_elements,
        dt_sim_0=dt_sim_0, t_max=t_max, qi=qi, qf=qf, ppc0=0.0,
        stabilize=stabilize, tol=tol,
    )
    t_con_Gs_278_yr = t_con_Gs_278 / s_per_yr

    # solve neutral buoyancy case
    # overconsolidated
    (
        t_con_Gs_1_OC, s_con_Gs_1_OC, U_con_Gs_1_OC,
        z_nod_Gs_1_OC, z_int_Gs_1_OC,
        e_nod_Gs_1_OC, Uz_nod_Gs_1_OC,
        sig_p_int_Gs_1_OC, ue_norm_int_Gs_1_OC, hyd_cond_int_Gs_1_OC,
        t_50_Gs_1_OC, run_time_Gs_1_OC,
    ) = solve_consolidation_benchmark(
        Gs=1.0, H_layer=H_layer, num_elements=num_elements,
        dt_sim_0=dt_sim_0, t_max=t_max, qi=qi, qf=qf, ppc0=ppc0,
        stabilize=stabilize, tol=tol,
    )
    t_con_Gs_1_OC_yr = t_con_Gs_1_OC / s_per_yr

    # solve self weight case
    # overconsolidated
    (
        t_con_Gs_278_OC, s_con_Gs_278_OC, U_con_Gs_278_OC,
        z_nod_Gs_278_OC, z_int_Gs_278_OC,
        e_nod_Gs_278_OC, Uz_nod_Gs_278_OC,
        sig_p_int_Gs_278_OC, ue_norm_int_Gs_278_OC, hyd_cond_int_Gs_278_OC,
        t_50_Gs_278_OC, run_time_Gs_278_OC,
    ) = solve_consolidation_benchmark(
        Gs=2.78, H_layer=H_layer, num_elements=num_elements,
        dt_sim_0=dt_sim_0, t_max=t_max, qi=qi, qf=qf, ppc0=ppc0,
        stabilize=stabilize, tol=tol,
    )
    t_con_Gs_278_OC_yr = t_con_Gs_278_OC / s_per_yr

    # settlement and average degree of consolidation
    plt.figure(figsize=(8.0, 10.0))

    plt.subplot(2, 2, 1)
    plt.semilogx(
        t_con_Gs_1_yr[1:], s_con_Gs_1[1:], "--k", label="Gs=1.0",
    )
    plt.semilogx(
        t_con_Gs_278_yr[1:], s_con_Gs_278[1:], "-k", label="Gs=2.78",
    )
    plt.semilogx(
        t_FP15[1:], s_FP15_Gs_1[1:], label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.semilogx(
        t_FP15[1:], s_FP15_Gs_278[1:],  # label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.xlabel(r"Time, $t$ [$yr$]")
    plt.ylabel(r"Settlement, $s$ [$m$]")
    plt.xlim([0.01, 100])
    plt.ylim([3.0, 0.0])
    plt.title("Normally consolidated")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.semilogx(
        t_con_Gs_1_OC_yr[1:], s_con_Gs_1_OC[1:], "--k", label="Gs=1.0",
    )
    plt.semilogx(
        t_con_Gs_278_OC_yr[1:], s_con_Gs_278_OC[1:], "-k", label="Gs=2.78",
    )
    plt.semilogx(
        t_FP15[1:], s_FP15_Gs_1_OC[1:], label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.semilogx(
        t_FP15[1:], s_FP15_Gs_278_OC[1:],  # label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.xlabel(r"Time, $t$ [$yr$]")
    plt.ylabel(r"Settlement, $s$ [$m$]")
    plt.xlim([0.01, 100])
    plt.ylim([1.5, 0.0])
    plt.title("Overconsolidated")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.semilogx(
        t_con_Gs_1_yr[1:], U_con_Gs_1[1:], "--k", label="Gs=1.0",
    )
    plt.semilogx(
        t_con_Gs_278_yr[1:], U_con_Gs_278[1:], "-k", label="Gs=2.78",
    )
    plt.semilogx(
        t_FP15[1:], U_FP15_Gs_1[1:], label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.semilogx(
        t_FP15[1:], U_FP15_Gs_278[1:],  # label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.xlabel(r"Time, $t$ [$yr$]")
    plt.ylabel(r"Average degree of consolidation, $U$")
    plt.xlim([0.01, 100])
    plt.ylim([0.0, 1.0])
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.semilogx(
        t_con_Gs_1_OC_yr[1:], U_con_Gs_1_OC[1:], "--k", label="Gs=1.0",
    )
    plt.semilogx(
        t_con_Gs_278_OC_yr[1:], U_con_Gs_278_OC[1:], "-k", label="Gs=2.78",
    )
    plt.semilogx(
        t_FP15[1:], U_FP15_Gs_1_OC[1:], label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.semilogx(
        t_FP15[1:], U_FP15_Gs_278_OC[1:],  # label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.xlabel(r"Time, $t$ [$yr$]")
    plt.ylabel(r"Average degree of consolidation, $U$")
    plt.xlim([0.01, 100])
    plt.ylim([0.0, 1.0])
    plt.legend()

    plt.savefig("examples/con_static_bench_settle.svg")

    # void ratio and normalized excess pore pressure profiles
    fig = plt.figure(figsize=(8.0, 10.0))

    plt.subplot(2, 2, 1)
    plt.plot(
        e_nod_Gs_1[:, 0], z_nod_Gs_1, "--k", label="Gs=1.0"
    )
    plt.plot(
        e_nod_Gs_278[:, 0], z_nod_Gs_278, "-k", label="Gs=2.78"
    )
    plt.plot(
        e_FP15_Gs_1[:, 0], z_FP15, label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.plot(
        e_FP15_Gs_278[:, 0], z_FP15,
        marker="x", linestyle="none",
    )
    for kk, kk_plot in enumerate(k_plot):
        if not kk:
            continue
        plt.plot(e_nod_Gs_1[:, kk_plot], z_nod_Gs_1, "--k")
        plt.plot(e_nod_Gs_278[:, kk_plot], z_nod_Gs_278, "-k")
        plt.plot(
            e_nod_Gs_1[::6, kk_plot], z_nod_Gs_1[::6],
            label=k_plot_labels[kk],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            e_nod_Gs_278[::6, kk_plot], z_nod_Gs_278[::6],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            e_FP15_Gs_1[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
        plt.plot(
            e_FP15_Gs_278[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
    plt.xlabel(r"Void Ratio, $e$")
    plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")
    fig.legend()
    plt.ylim((11, -1))
    plt.xlim((1.5, 3.0))

    plt.subplot(2, 2, 3)
    plt.plot(
        e_nod_Gs_1_OC[:, 0], z_nod_Gs_1_OC, "--k", label="Gs=1.0"
    )
    plt.plot(
        e_nod_Gs_278_OC[:, 0], z_nod_Gs_278_OC, "-k", label="Gs=2.78"
    )
    plt.plot(
        e_FP15_Gs_1_OC[:, 0], z_FP15, label="FoxPu2015",
        marker="x", linestyle="none",
    )
    plt.plot(
        e_FP15_Gs_278_OC[:, 0], z_FP15,
        marker="x", linestyle="none",
    )
    for kk, kk_plot in enumerate(k_plot):
        if not kk:
            continue
        plt.plot(e_nod_Gs_1_OC[:, kk_plot], z_nod_Gs_1_OC, "--k")
        plt.plot(e_nod_Gs_278_OC[:, kk_plot], z_nod_Gs_278_OC, "-k")
        plt.plot(
            e_nod_Gs_1_OC[::6, kk_plot], z_nod_Gs_1_OC[::6],
            label=k_plot_labels[kk],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            e_nod_Gs_278_OC[::6, 10], z_nod_Gs_278_OC[::6],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            e_FP15_Gs_1_OC[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
        plt.plot(
            e_FP15_Gs_278_OC[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
    plt.xlabel(r"Void Ratio, $e$")
    plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")
    plt.ylim((11, -1))
    plt.xlim((1.5, 2.2))

    plt.subplot(2, 2, 2)
    for kk, kk_plot in enumerate(k_plot[:-1]):
        if not kk:
            continue
        plt.plot(ue_norm_int_Gs_1[:, kk_plot], z_int_Gs_1, "--k")
        plt.plot(ue_norm_int_Gs_278[:, kk_plot], z_int_Gs_278, "-k")
        plt.plot(
            ue_norm_int_Gs_1[::6, kk_plot], z_int_Gs_1[::6],
            label=k_plot_labels[kk],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            ue_norm_int_Gs_278[::6, kk_plot], z_int_Gs_278[::6],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            ue_FP15_Gs_1[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
        plt.plot(
            ue_FP15_Gs_278[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
    plt.xlabel(r"Normalized excess pore pressure, $u_e/\Delta q$")
    plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")
    plt.ylim((11, -1))
    plt.xlim((-0.1, 1.1))

    plt.subplot(2, 2, 4)
    for kk, kk_plot in enumerate(k_plot[:-1]):
        if not kk:
            continue
        plt.plot(ue_norm_int_Gs_1_OC[:, kk_plot], z_int_Gs_1_OC, "--k")
        plt.plot(ue_norm_int_Gs_278_OC[:, kk_plot], z_int_Gs_278_OC, "-k")
        plt.plot(
            ue_norm_int_Gs_1_OC[::6, kk_plot], z_int_Gs_1_OC[::6],
            label=k_plot_labels[kk],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            ue_norm_int_Gs_278_OC[::6, kk_plot], z_int_Gs_278_OC[::6],
            marker=k_plot_lts[kk],
            linestyle="none",
        )
        plt.plot(
            ue_FP15_Gs_1_OC[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
        plt.plot(
            ue_FP15_Gs_278_OC[:, kk], z_FP15,
            marker="x", linestyle="none",
        )
    plt.xlabel(r"Normalized excess pore pressure, $u_e/\Delta q$")
    plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")
    plt.ylim((11, -1))
    plt.xlim((-0.1, 1.1))

    plt.savefig("examples/con_static_bench_profiles.svg")


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


def calculate_static_profile(
    z: npt.ArrayLike,
    m: Material,
    qs: float,
    dqs: float = 0.0,
    ppc: float = 0.0,
) -> tuple[npt.NDArray[np.floating]]:
    """Calculate initial and final equilibrium profiles of
    void ratio, effective stress, and hydraulic conductivity
    based on surface load, self weight, consolidation curve,
    and initial coordinates.

    Inputs
    ------
    z : array_like, shape=(nnod, )
        Initial coordinates of nodes.
    m : frozen_ground_fem.materials.Material
        Reference to material defining consolidation curve.
    qs : float
        Initial surface stress.
    dqs : float
        Increment to surface stress.
    ppc : float
        Pre-consolidation stress.

    Returns
    -------
    e0 : numpy.ndarray, shape=(nnod, )
        Initial equilibrium void ratio profile.
    sig_p_0 : numpy.ndarray, shape=(nnod, )
        Initial equilibrium effective stress profile.
    hyd_cond_0 : numpy.ndarray, shape=(nnod, )
        Initial equilibrium hydraulic conductivity profile.
    e1 : numpy.ndarray, shape=(nnod, )
        Final equilibrium void ratio profile.
    sig_p_1 : numpy.ndarray, shape=(nnod, )
        Final equilibrium effective stress profile.
    hyd_cond_1 : numpy.ndarray, shape=(nnod, )
        Final equilibrium hydraulic conductivity profile.
    """
    z = np.array(z)
    # get material properties
    Gs = m.spec_grav_solids
    Ccu = m.comp_index_unfrozen
    Cru = m.rebound_index_unfrozen
    sig_cu0 = m.eff_stress_0_comp
    e_cu0 = m.void_ratio_0_comp
    # calculate void ratio on NCL corresponding to ppc
    e_ppc = e_cu0 - Ccu * np.log10(ppc / sig_cu0) if ppc else e_cu0
    # initialize solution grid
    nnod = len(z)
    gam_p = np.zeros(nnod)
    sig_p_0 = np.zeros(nnod)
    sig_p_0[0] = qs
    e0 = np.ones(nnod)
    e1 = np.zeros(nnod)
    eps_s = 1e-8
    eps_a = 1.0
    # iterate to determine initial equilibrium profile
    # for effective stress and void ratio
    while eps_a > eps_s:
        for k, e in enumerate(e0):
            # buoyant unit weight
            gam_p[k] = (Gs - 1.0) / (1.0 + e) * unit_weight_water
            # update effective stress
            if k:
                dz = z[k] - z[k - 1]
                gam_p_avg = 0.5 * (gam_p[k - 1] + gam_p[k])
                sig_p_0[k] = sig_p_0[k - 1] + gam_p_avg * dz
            # update void ratio
            if sig_p_0[k] < ppc:
                e1[k] = e_ppc - Cru * np.log10(sig_p_0[k] / ppc)
            else:
                e1[k] = e_cu0 - Ccu * np.log10(sig_p_0[k] / sig_cu0)
        # check for convergence
        eps_a = np.linalg.norm(e1 - e0) / np.linalg.norm(e1)
        e0[:] = e1[:]
    # increment stresses from initial equilibrium
    # and update void ratio
    # (do not iterate here because coordinates will change
    # but mass of solids does not)
    sig_p_1 = sig_p_0 + dqs
    for k, e in enumerate(e1):
        if sig_p_1[k] < ppc:
            e1[k] = e_ppc - Cru * np.log10(sig_p_1[k] / ppc)
        else:
            e1[k] = e_cu0 - Ccu * np.log10(sig_p_1[k] / sig_cu0)
    # compute the corresponding hydraulic conductivity
    hyd_cond_0 = (m.hyd_cond_0 *
                  10 ** ((e0 - m.void_ratio_0_hyd_cond) / m.hyd_cond_index))
    hyd_cond_1 = (m.hyd_cond_0 *
                  10 ** ((e1 - m.void_ratio_0_hyd_cond) / m.hyd_cond_index))
    return (
        e0, sig_p_0, hyd_cond_0,
        e1, sig_p_1, hyd_cond_1,
    )


if __name__ == "__main__":
    main()
