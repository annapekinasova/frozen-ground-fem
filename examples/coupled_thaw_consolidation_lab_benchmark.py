"""Benchmark for large strain thaw consolidation.

This script runs benchmark LSC using geometry,
boundary conditions, and soil properties from:
    Dumais, S., and Konrad, J.-M. 2018.
        One-dimensional large-strain thaw consolidation using
        nonlinear effective stress – void ratio – hydraulic conductivity
        relationships, Canadian Geotechnical Journal, 55(3): 414–426,
        https://doi.org/10.1139/cgj-2017-0221.
    Morgenstern, N., and Nixon, J. 1971.
        One-dimensional consolidation of thawing soils,
        Canadian Geotechnical Journal, 8(4): 558–565,
        https://doi.org/10.1139/t71-057.
    Yu, F., Guo, P., Lai, Y., and Stolle, D. 2020.
        Frost heave and thaw consolidation modelling. Part 2:
        One-dimensional thermohydromechanical (THM) framework,
        Canadian Geotechnical Journal, 57(10), 1595-1610,
        https://doi.org/10.1139/cgj-2019-0306.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem import (
    Material,
    ThermalBoundary1D,
    HydraulicBoundary1D,
    ConsolidationBoundary1D,
    CoupledAnalysis1D,
)


def main():
    # define simulation parameters
    s_per_min = 60.0
    H_layer = 0.5
    num_elements_top = 25
    num_elements = int(np.ceil(num_elements_top * 1.5))
    num_elements_bot = num_elements - num_elements_top
    dt_sim_0 = 1.0e-5
    t_max = 1000.0 * s_per_min
    qi = 15.0e3
    tol = 1e-2
    tol_str = f"{tol:0.1e}"
    tol_str = "p".join(tol_str.split("."))
    fname = f"examples/thaw_consol_lab_{num_elements_top}_{tol_str}"
    # compute modified node locations
    z_mesh_nod = np.hstack(
        [
            np.linspace(0.0, 0.1, num_elements_top + 1)[:-1],
            np.linspace(0.1, 0.5, num_elements_bot + 1),
        ]
    )

    # set plotting parameters
    plt.rc("font", size=9)
    plt.rc(
        "lines",
        linewidth=0.5,
        color="black",
        markeredgewidth=0.5,
        markerfacecolor="none",
        markersize=4,
    )

    # define the material properties
    m = Material(
        spec_grav_solids=2.6,
        thrm_cond_solids=2.1,
        spec_heat_cap_solids=874.0,
        deg_sat_water_alpha=1.20e4,
        deg_sat_water_beta=0.35,
        water_flux_b1=0.08,
        water_flux_b2=4.0,
        water_flux_b3=1.0e-5,
        seg_pot_0=2.0e-9,
        hyd_cond_index=0.305,
        void_ratio_0_hyd_cond=2.6,
        hyd_cond_mult=1.0,
        hyd_cond_0=4.05e-4,
        void_ratio_lim=0.3,
        void_ratio_min=0.3,
        void_ratio_tr=1.6,
        void_ratio_sep=1.6,
        void_ratio_0_comp=2.6,
        eff_stress_0_comp=2.8,
        comp_index_unfrozen=0.421,
        rebound_index_unfrozen=0.08,
        comp_index_frozen_a1=0.021,
        comp_index_frozen_a2=0.01,
        comp_index_frozen_a3=0.23,
    )

    print(f"H_layer = {H_layer} m")
    print(f"qi = {qi} Pa = {qi*1e-3} kPa")
    print(f"Gs = {m.spec_grav_solids}")
    print(f"lam_s = {m.thrm_cond_solids} W/m/K")
    print(f"cs = {m.spec_heat_cap_solids} J/kg/K")
    print(
        f"deg_sat_water_alpha = {m.deg_sat_water_alpha} Pa"
        + f" = {m.deg_sat_water_alpha*1e-3} kPa"
    )
    print(f"deg_sat_water_beta = {m.deg_sat_water_beta}")
    print(f"water_flux_b1 = {m.water_flux_b1}")
    print(f"water_flux_b2 = {m.water_flux_b2} 1/K")
    print(f"water_flux_b3 = {m.water_flux_b3} 1/Pa" + f" = {m.water_flux_b3*1e6} 1/MPa")
    print(f"e_min = {m.void_ratio_min}")
    print(f"e_sep = {m.void_ratio_sep}")
    print(f"seg_pot_0 = {m.seg_pot_0} m^2/K/s")
    print(f"Ck = {m.hyd_cond_index}")
    print(f"k0 = {m.hyd_cond_0} m/s")
    print(f"m = {m.hyd_cond_mult} m/s")
    print(f"e0k = {m.void_ratio_0_hyd_cond}")
    print(f"Cc = {m.comp_index_unfrozen}")
    print(f"Cr = {m.rebound_index_unfrozen}")
    print(f"sig_p_0 = {m.eff_stress_0_comp} Pa" + f" = {m.eff_stress_0_comp*1e-3} kPa")
    print(f"e0sig = {m.void_ratio_0_comp}")
    print(f"comp_index_frozen_a1 = {m.comp_index_frozen_a1}")
    print(f"comp_index_frozen_a2 = {m.comp_index_frozen_a2}")
    print(f"comp_index_frozen_a3 = {m.comp_index_frozen_a3}")
    print(f"num_elements_top = {num_elements_top}")
    print(f"num_elements = {num_elements}")
    print(f"dt_sim_0 = {dt_sim_0} s = {dt_sim_0 / s_per_min} min")
    print(f"t_max = {t_max} s = {t_max / s_per_min} min")
    print(f"tol = {tol:0.4e}")

    # define plotting time increments
    t_plot_targ = (
        np.hstack(
            [
                0.0,
                np.linspace(0.01, 0.1, 10)[:-1],
                np.linspace(0.1, 1.0, 10)[:-1],
                np.linspace(1.0, 5.0, 5)[:-1],
                np.linspace(5.0, 100.0, 20)[:-1],
                np.linspace(100.0, 300.0, 21)[:-1],
                np.linspace(300.0, 348.0, 7),
                np.linspace(400.0, 750.0, 8),
            ]
        )
        * s_per_min
    )
    n_plot_targ = len(t_plot_targ)
    dt_plot = np.max(
        [
            np.min(
                [
                    np.max(np.diff(t_plot_targ)),
                    50.0 * s_per_min,
                ]
            ),
            dt_sim_0,
        ]
    )
    t_plot_extra = t_max - t_plot_targ[-1]
    n_plot = n_plot_targ + int(np.floor(t_plot_extra / dt_plot) + 1)

    # create coupled analysis and generate the mesh
    con_static = CoupledAnalysis1D(
        z_range=[0.0, H_layer],
        num_elements=num_elements,
        generate=True,
        order=1,
    )
    con_static.implicit_error_tolerance = tol
    # modify node locations in the mesh
    for nd, zn in zip(con_static.nodes, z_mesh_nod):
        nd.z = zn

    # initialize plotting arrays
    z_nod = np.array([nd.z for nd in con_static.nodes])
    z_int = np.array([ip.z for e in con_static.elements for ip in e.int_pts])
    e_nod = np.zeros((len(z_nod), n_plot + 1))
    T_nod = np.zeros_like(e_nod)
    zdef_nod = np.zeros_like(e_nod)
    sig_p_int = np.zeros((len(z_int), n_plot + 1))
    sig_int = np.zeros_like(sig_p_int)
    uu_int = np.zeros_like(sig_p_int)
    ue_int = np.zeros_like(sig_p_int)
    uh_int = np.zeros_like(sig_p_int)
    zdef_int = np.zeros_like(sig_p_int)

    # set initial conditions
    for nd in con_static.nodes:
        nd.temp = -5.0
        nd.void_ratio = 2.83
        nd.void_ratio_0 = 2.83

    # assign material properties to elements
    for e in con_static.elements:
        e.assign_material(m)

    # create boundary conditions
    # and assign them to the mesh
    temp_bound = ThermalBoundary1D(
        nodes=(con_static.nodes[0],),
        bnd_type=ThermalBoundary1D.BoundaryType.temp,
        bnd_value=5.0,
    )
    con_static.add_boundary(temp_bound)
    hyd_bound = HydraulicBoundary1D(
        nodes=(con_static.nodes[0],),
        bnd_value=H_layer,
    )
    con_static.add_boundary(hyd_bound)
    e_cu0 = m.void_ratio_0_comp
    Ccu = m.comp_index_unfrozen
    sig_cu0 = m.eff_stress_0_comp
    sig_p_ob = 1.50e4
    e_bnd = e_cu0 - Ccu * np.log10(sig_p_ob / sig_cu0)
    void_ratio_bound = ConsolidationBoundary1D(
        nodes=(con_static.nodes[0],),
        bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
        bnd_value=e_bnd,
        bnd_value_1=sig_p_ob,
    )
    con_static.add_boundary(void_ratio_bound)

    # start simulation time
    tic = time.perf_counter()

    # initialize global matrices and vectors
    print(f"initialize system, qi = {qi} Pa")
    con_static.time_step = dt_sim_0
    con_static.initialize_global_system(t0=0.0)
    T_nod[:, 0] = np.array([nd.temp for nd in con_static.nodes])
    e_nod[:, 0] = np.array([nd.void_ratio for nd in con_static.nodes])
    zdef_nod[:, 0] = np.array([nd.z_def for nd in con_static.nodes])
    sig_p_int[:, 0] = 1.0e-3 * np.array(
        [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
    )
    sig_int[:, 0] = 1.0e-3 * np.array(
        [ip.tot_stress for e in con_static.elements for ip in e.int_pts]
    )
    uu_int[:, 0] = 1.0e-3 * np.array(
        [ip.pore_pressure for e in con_static.elements for ip in e.int_pts]
    )
    ue_int[:, 0] = 1.0e-3 * np.array(
        [ip.exc_pore_pressure for e in con_static.elements for ip in e.int_pts]
    )
    uh_int[:, 0] = uu_int[:, 0] - ue_int[:, 0]
    zdef_int[:, 0] = np.array(
        [ip.z_def for e in con_static.elements for ip in e.int_pts]
    )

    t_con = [0.0]
    s_con = [0.0]
    Z_con = [0.0]
    dt_step = []
    err_step = []
    dt_seq = []
    err_seq = []
    k_plot = 0
    t_plot = 0.0
    eps_a = 1.0
    sim_time_0 = 0.0
    sim_time = 0.0
    run_time_0 = 0.0
    run_time = 0.0
    while (eps_a > tol and k_plot < n_plot) or t_plot < t_plot_targ[-1]:
        k_plot += 1
        if k_plot < n_plot_targ:
            t_plot = t_plot_targ[k_plot]
        else:
            t_plot += dt_plot
        dt_s, err_s = con_static.solve_to(t_plot)[1:]
        toc = time.perf_counter()
        sim_time_0 = sim_time
        sim_time = con_static._t1
        run_time_0 = run_time
        run_time = (toc - tic) / s_per_min
        rem_time = (
            (t_max - sim_time) / (sim_time - sim_time_0) * (run_time - run_time_0)
        )
        # save time step and error information
        dt_step.append(dt_s)
        err_step.append(err_s)
        dt_seq = np.hstack([dt_seq, dt_s])
        err_seq = np.hstack([err_seq, err_s])
        # get total settlement
        t_con.append(con_static._t1)
        s_con.append(con_static.calculate_total_settlement())
        # find thaw depth
        # first find the element containing T=0.0
        for k, e in enumerate(con_static.elements):
            if e.nodes[0].temp * e.nodes[-1].temp < 0.0:
                ee = e
                break
        # get temperatures and depths for this element
        jac = ee.jacobian
        Te = np.array([nd.temp for nd in ee.nodes])
        # perform Newton-Raphson to find T = 0.0
        s = 0.5
        eps_a_Z = 1.0
        eps_s_Z = 1.0e-8
        while eps_a_Z > eps_s_Z:
            N = ee._shape_matrix(s)
            B = ee._gradient_matrix(s, jac)
            ds = (N @ Te)[0] / (jac * B @ Te)[0]
            s -= ds
            eps_a_Z = np.abs(ds / s)
        # compute thaw depth (in deformed coords)
        zde = np.array([nd.z_def for nd in ee.nodes])
        N = ee._shape_matrix(s)
        Zte = (N @ zde)[0]
        Z_con.append(Zte)
        # get pore pressure profiles
        eps_a_scon = np.abs((s_con[-1] - s_con[-2]) / s_con[-1]) if s_con[-1] else 0.0
        eps_a_Zcon = np.abs((Z_con[-1] - Z_con[-2]) / Z_con[-1]) if Z_con[-1] else 0.0
        eps_a = np.max([eps_a_scon, eps_a_Zcon])
        print(
            f"t = {con_static._t1 / s_per_min:0.2f} min, "
            + f"s = {s_con[-1] * 1e2:0.2f} cm, "
            + f"Z = {Z_con[-1] * 1e2:0.2f} cm, "
            + f"dt_min = {np.min(dt_s):0.2e} s, "
            + f"dt_max = {np.max(dt_s):0.2e} s, "
            + f"run_time = {run_time:0.2f} min, "
            + f"rem_time = {rem_time:0.2f} min"
        )
        # save temp, void ratio, eff stress, pore pressure, and deformed coord profiles
        zdef_nod[:, k_plot] = np.array([nd.z_def for nd in con_static.nodes])
        T_nod[:, k_plot] = np.array([nd.temp for nd in con_static.nodes])
        e_nod[:, k_plot] = np.array([nd.void_ratio for nd in con_static.nodes])
        zdef_int[:, k_plot] = np.array(
            [ip.z_def for e in con_static.elements for ip in e.int_pts]
        )
        sig_p_int[:, k_plot] = 1.0e-3 * np.array(
            [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
        )
        sig_int[:, k_plot] = 1.0e-3 * np.array(
            [ip.tot_stress for e in con_static.elements for ip in e.int_pts]
        )
        uu_int[:, k_plot] = 1e-3 * np.array(
            [ip.pore_pressure for e in con_static.elements for ip in e.int_pts]
        )
        ue_int[:, k_plot] = 1e-3 * np.array(
            [ip.exc_pore_pressure for e in con_static.elements for ip in e.int_pts]
        )
        uh_int[:, k_plot] = uu_int[:, k_plot] - ue_int[:, k_plot]

    toc = time.perf_counter()
    run_time = (toc - tic) / s_per_min

    # convert settlement to arrays
    t_con = np.array(t_con)
    s_con = np.array(s_con)
    Z_con = np.array(Z_con)
    t_con_min = t_con / s_per_min
    s_con_cm = s_con * 1e2
    Z_con_cm = Z_con * 1e2
    zdef_nod_cm = zdef_nod * 1e2
    zdef_int_cm = zdef_int * 1e2

    # calculate time to 50 percent settlement
    s_tot = s_con[-1]
    s_50 = 0.5 * s_tot
    k_50 = 0
    for k, s in enumerate(s_con):
        if s > s_50:
            k_50 = k
            break
    s1 = s_con[k_50]
    s0 = s_con[k_50 - 1]
    t1 = t_con[k_50]
    t0 = t_con[k_50 - 1]
    t_50 = (np.sqrt(t0) + ((np.sqrt(t1) - np.sqrt(t0)) * (s_50 - s0) / (s1 - s0))) ** 2

    print(f"Run time = {run_time: 0.4f} min")
    print(f"Total settlement = {s_con[-1]} m = {s_con_cm[-1]} cm")
    print(f"Thaw depth = {Z_con[-1]} m = {Z_con_cm[-1]} cm")
    print(f"t_50 = {t_50} s = {t_50 / s_per_min} min")

    # settlement, thaw depth, exc pore press, and void ratio profiles
    plt.figure(figsize=(8.0, 8.0))

    plt.subplot(2, 2, 1)
    plt.plot(
        np.sqrt(t_con_min[1:]),
        s_con_cm[1:],
        "or",
        label="settlement",
    )
    plt.plot(
        np.sqrt(t_con_min[1:]),
        Z_con_cm[1:],
        "xb",
        label="thaw depth",
    )
    plt.ylabel(r"Depth, $s$ [$cm$]")
    plt.xlabel(r"Time, $t$ [$min^{0.5}$]")
    plt.xlim([0.0, 30])
    plt.ylim([10.0, 0.0])
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(
        e_nod[:, 23],
        zdef_nod_cm[:, 23],
        "ob",
        # label="1.4 min",
        label="5 min",
    )
    plt.plot(
        e_nod[:, 27],
        zdef_nod_cm[:, 27],
        "or",
        # label="1.8 min",
        label="25 min",
    )
    plt.plot(
        e_nod[:, 41],
        zdef_nod_cm[:, 41],
        "xb",
        # label="3.2 min",
        label="95 min",
    )
    plt.plot(
        e_nod[:, 68],
        zdef_nod_cm[:, 68],
        "xr",
        # label="5.9 min",
        label="348 min",
    )
    # plt.plot(
    #     e_nod[:, 108],
    #     zdef_nod_cm[:, 108],
    #     "--k",
    #     label="9.9 min",
    # )
    plt.plot(
        e_nod[:, 76],
        zdef_nod_cm[:, 76],
        "--k",
        label="750 min",
    )
    plt.xlim([1.0, 3.0])
    plt.ylim([10.0, 0.0])
    plt.xlabel(r"Void ratio, $e$")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(
        ue_int[:, 23],
        zdef_int_cm[:, 23],
        "ob",
        # label="1.4 min",
        label="5 min",
    )
    plt.plot(
        ue_int[:, 27],
        zdef_int_cm[:, 27],
        "or",
        # label="1.8 min",
        label="25 min",
    )
    plt.plot(
        ue_int[:, 41],
        zdef_int_cm[:, 41],
        "xb",
        # label="3.2 min",
        label="95 min",
    )
    plt.plot(
        ue_int[:, 68],
        zdef_int_cm[:, 68],
        "xr",
        # label="5.9 min",
        label="348 min",
    )
    # plt.plot(
    #     ue_int[:, 108],
    #     zdef_int_cm[:, 108],
    #     "--k",
    #     label="9.9 min",
    # )
    plt.plot(
        ue_int[:, 76],
        zdef_int_cm[:, 76],
        "--k",
        label="750 min",
    )
    plt.xlim([0.0, 16.0])
    plt.ylim([10.0, 0.0])
    plt.xlabel(r"Excess pore pressure, $u_e$ [$kPa$]")
    plt.ylabel(r"Depth, $s$ [$cm$]")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(
        T_nod[:, 23],
        zdef_nod_cm[:, 23],
        "ob",
        # label="1.4 min",
        label="5 min",
    )
    plt.plot(
        T_nod[:, 27],
        zdef_nod_cm[:, 27],
        "or",
        # label="1.8 min",
        label="25 min",
    )
    plt.plot(
        T_nod[:, 41],
        zdef_nod_cm[:, 41],
        "xb",
        # label="3.2 min",
        label="95 min",
    )
    plt.plot(
        T_nod[:, 68],
        zdef_nod_cm[:, 68],
        "xr",
        # label="5.9 min",
        label="348 min",
    )
    # plt.plot(
    #     T_nod[:, 108],
    #     zdef_nod_cm[:, 108],
    #     "--k",
    #     label="9.9 min",
    # )
    plt.plot(
        T_nod[:, 76],
        zdef_nod_cm[:, 76],
        "--k",
        label="750 min",
    )
    plt.xlim([6.0, -6.0])
    plt.ylim([10.0, 0.0])
    plt.xlabel(r"Temperature, $T$ [$deg C$]")
    plt.legend()

    with open(fname + "_dt_plot.out", mode="w") as f_dt_plot:
        for tt, ddtt in zip(t_con[1:], dt_step):
            dt_str = ""
            for dddttt in ddtt:
                dt_str += f" {dddttt:0.8e}"
            f_dt_plot.write(f"{tt:0.8e} {dt_str}\n")
    np.savetxt(fname + "_dt_err.out", np.vstack([dt_seq, err_seq]).T)
    np.savetxt(fname + "_t_s_Z.out", np.vstack([t_con, s_con, Z_con]).T)
    np.savetxt(fname + "_zdef_nod.out", zdef_nod.T)
    np.savetxt(fname + "_T_nod.out", T_nod.T)
    np.savetxt(fname + "_e_nod.out", e_nod.T)
    np.savetxt(fname + "_zdef_int.out", zdef_int.T)
    np.savetxt(fname + "_sigp_int.out", sig_p_int.T)
    np.savetxt(fname + "_sig_int.out", sig_int.T)
    np.savetxt(fname + "_ue_int.out", ue_int.T)
    np.savetxt(fname + "_uu_int.out", uu_int.T)
    np.savetxt(fname + "_uh_int.out", uh_int.T)
    plt.savefig(fname + ".svg")
    plt.savefig(fname + ".png")


if __name__ == "__main__":
    main()
