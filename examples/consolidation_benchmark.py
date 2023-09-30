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
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

import frozen_ground_fem.materials as mtl
import frozen_ground_fem.geometry as geom
import frozen_ground_fem.consolidation as consol

from frozen_ground_fem.materials import (
    unit_weight_water as gam_w,
)


def main():
    # load simulation parameters
    fname = "examples/benchmark_params.bat"
    sim_params = np.loadtxt(fname)
    H_layer_bat = sim_params[:, 0]
    num_elements_bat = np.array(sim_params[:, 1], dtype=int)
    dt_sim_bat = sim_params[:, 2]
    t_max_bat = sim_params[:, 3]
    t_50_bat = np.zeros_like(H_layer_bat)
    s_tot_bat = np.zeros_like(H_layer_bat)
    runtime_bat = np.zeros_like(H_layer_bat)
    qs0 = 4.0e4  # initial surface load, Pa
    qs1 = 4.4e5  # final surface load, Pa

    # set plotting parameters
    plt.rc("font", size=8)

    # define the material properties
    m = mtl.Material(
        spec_grav_solids=2.78,
        hyd_cond_index=1.30,
        hyd_cond_0=2.0e-8,
        void_ratio_0_hyd_cond=4.30,
        void_ratio_min=0.30,
        void_ratio_tr=4.30,
        void_ratio_0_comp=2.70,
        comp_index_unfrozen=1.0,
        rebound_index_unfrozen=0.10,
        eff_stress_0_comp=4.0e4,
    )

    t_con_bench = np.array([
        0.05,
        0.1,
        0.5,
        1,
        2,
        3,
        4,
        5,
        10,
        20,
        40,
        60,
    ])
    s_con_bench1_278 = np.array([
        1.880E-01,
        2.150E-01,
        5.920E-01,
        8.350E-01,
        1.178E+00,
        1.439E+00,
        1.651E+00,
        1.826E+00,
        2.296E+00,
        2.462E+00,
        2.473E+00,
        2.473E+00,
    ])

    # initialize .out file
    with open(fname + ".out", "w", encoding="utf-8") as fout:
        fout.write(
            f"batch run on {datetime.now()}\n"
            + "properties:\n"
            + f"Gs={m.spec_grav_solids:0.5f}\n"
            + f"Ck={m.hyd_cond_index}\n"
            + f"k0={m.hyd_cond_0}\n"
            + f"e0k={m.void_ratio_0_hyd_cond}\n"
            + f"emin={m.void_ratio_min}\n"
            + f"etr={m.void_ratio_tr}\n"
            + f"Cu={m.comp_index_unfrozen}\n"
            + f"Cr={m.rebound_index_unfrozen}\n"
            + f"sig_cu0={m.eff_stress_0_comp}\n"
            + f"e0u={m.void_ratio_0_comp}\n"
            + "\n"
            + "         H [m]    Ne          dt [s]       t_max [s]"
            + "      t_50 [min]      s_tot [mm]     runtime [s]\n"
        )

    for k_bat, (H_layer, num_elements, dt_sim, t_max) in enumerate(
        zip(H_layer_bat, num_elements_bat, dt_sim_bat, t_max_bat)
    ):
        # compute plotting time step
        dt_plot = 0.05 * 365 * 24 * 60 * 60  # in seconds
        n_plot = int(np.floor(t_max / dt_plot) + 1)
        k_plot_list = [2, 40, 100, 1200]
        k_plot_labels = ["t=0.1 yr", "t=2 yr", "t=5 yr", "Final"]
        k_plot_linetype = ["--b", ":b", "-.b", "--k"]
        k_plot_init = "-k"

        print(f"H_layer = {H_layer}")
        print(f"num_elements = {num_elements}")
        print(f"dt_sim = {dt_sim}")
        print(f"t_max = {t_max}")
        print(f"qs0 = {qs0}")
        print(f"qs1 = {qs1}")

        # generate the mesh
        mesh = geom.Mesh1D(
            z_range=[0.0, H_layer],
            num_nodes=num_elements + 1,
            generate=True,
        )

        # assign material properties to integration points
        for e in mesh.elements:
            for ip in e.int_pts:
                ip.material = m

        # create geometric boundaries
        # and assign them to the mesh
        upper_boundary = geom.Boundary1D(
            (mesh.nodes[0],),
            (mesh.elements[0].int_pts[0],),
        )
        lower_boundary = geom.Boundary1D(
            (mesh.nodes[-1],),
            (mesh.elements[-1].int_pts[-1],),
        )
        mesh.add_boundary(upper_boundary)
        mesh.add_boundary(lower_boundary)

        # create consolidation analysis
        con_static = consol.ConsolidationAnalysis1D(mesh)

        # initialize plotting arrays
        z_nod = np.array([nd.z for nd in mesh.nodes])
        z_int = np.array([ip.z for e in mesh.elements for ip in e.int_pts])
        e_nod = np.zeros((len(z_nod), n_plot + 1))
        sig_p_int = np.zeros((len(z_int), n_plot + 1))
        ue_int = np.zeros_like(sig_p_int)
        hyd_cond_int = np.zeros_like(sig_p_int)

        # initialize void ratio and effective stress profiles
        e0, sig_p_0_exp, hyd_cond_0_exp = calculate_static_profile(
            m, qs0, z_nod)
        e1, sig_p_1_exp, hyd_cond_1_exp = calculate_static_profile(
            m, qs1, z_nod)
        sig_p_0_exp *= 1.0e-3
        sig_p_1_exp *= 1.0e-3
        for k, nd in enumerate(mesh.nodes):
            nd.void_ratio = e0[k]

        # interpolate expected effective stress at integration points
        sig_p_1_spl = CubicSpline(z_nod, sig_p_1_exp)
        sig_p_1_exp_int = sig_p_1_spl(z_int)

        # update void ratio conditions at the integration points
        # (first interpolates void ratio profile from the nodes,
        # then assign initial void ratio to int pts)
        con_static.update_integration_points()
        for e in con_static.elements:
            for ip in e.int_pts:
                ip.void_ratio_0 = ip.void_ratio

        # create void ratio boundary conditions
        void_ratio_boundary_0 = consol.ConsolidationBoundary1D(upper_boundary)
        void_ratio_boundary_0.bnd_type = (
            consol.ConsolidationBoundary1D.BoundaryType.void_ratio
        )
        void_ratio_boundary_0.bnd_value = e0[0]
        void_ratio_boundary_1 = consol.ConsolidationBoundary1D(lower_boundary)
        void_ratio_boundary_1.bnd_type = (
            consol.ConsolidationBoundary1D.BoundaryType.void_ratio
        )
        void_ratio_boundary_1.bnd_value = e0[-1]
        con_static.add_boundary(void_ratio_boundary_0)
        con_static.add_boundary(void_ratio_boundary_1)

        # initialize global matrices and vectors
        con_static.time_step = dt_sim  # in seconds
        con_static.initialize_global_system(t0=0.0)

        tic = time.perf_counter()

        # stabilize void ratio profile
        # at initial surface load
        t_con_stab = [0.0]
        s_con_stab = [0.0]
        t_plot = dt_plot
        k_plot = 0
        k_stab_max = 10
        k_stab = 0
        tol_stab = 1E-8
        eps_s_stab = 2*tol_stab
        print("initial stabilization")
        while k_stab <= k_stab_max and eps_s_stab > tol_stab:
            while con_static._t1 < t_plot:
                con_static.initialize_time_step()
                con_static.iterative_correction_step()
                t_con_stab.append(con_static._t1)
                s_con_stab.append(con_static.calculate_total_settlement())
            eps_s_stab = np.abs(
                (s_con_stab[-1] - s_con_stab[-2])/s_con_stab[-1])
            print(
                f"t = {con_static._t1 / 60.0:0.3f} min, "
                + f"s_con = {s_con_stab[-1] * 1e3:0.3f} mm, "
                + f"eps_s =  {eps_s_stab:0.4e}"
            )
            t_plot += dt_plot
            k_stab += 1

        # save stabilized profiles
        e_nod[:, k_plot] = con_static._void_ratio_vector[:]
        sig_p_int[:, k_plot] = 1.0e-3 * np.array(
            [ip.eff_stress for e in mesh.elements for ip in e.int_pts]
        )
        hyd_cond_int[:, k_plot] = np.array(
            [ip.hyd_cond for e in mesh.elements for ip in e.int_pts]
        )

        # update boundary conditions,
        # reset time,
        # set initial void ratio profile,
        # and reset pre-consolidation stress profile
        print("apply static load increment")
        void_ratio_boundary_0.bnd_value = e1[0]
        void_ratio_boundary_1.bnd_value = e1[-1]
        con_static._t1 = 0.0
        for e in con_static.elements:
            for ip in e.int_pts:
                ip.void_ratio_0 = ip.void_ratio
                ip.pre_consol_stress = ip.eff_stress

        # run consolidation analysis
        t_con = [0.0]
        s_con = [0.0]
        k_plot += 1
        t_plot = dt_plot
        tol = 1E-8
        eps_s = 2*tol
        while k_plot <= n_plot and eps_s > tol:
            while con_static._t1 < t_plot:
                con_static.initialize_time_step()
                con_static.iterative_correction_step()
                t_con.append(con_static._t1)
                s_con.append(con_static.calculate_total_settlement())
            eps_s = np.abs((s_con[-1] - s_con[-2])/s_con[-1])
            print(
                f"t = {con_static._t1 / 60.0:0.3f} min, "
                + f"s_con = {s_con[-1] * 1e3:0.3f} mm, "
                + f"eps_s =  {eps_s:0.4e}"
            )
            e_nod[:, k_plot] = con_static._void_ratio_vector[:]
            sig_p_int[:, k_plot] = 1.0e-3 * np.array(
                [ip.eff_stress for e in mesh.elements for ip in e.int_pts]
            )
            ue_int[:, k_plot] = sig_p_1_exp_int - sig_p_int[:, k_plot]
            hyd_cond_int[:, k_plot] = np.array(
                [ip.hyd_cond for e in mesh.elements for ip in e.int_pts]
            )
            k_plot += 1
            t_plot += dt_plot

        toc = time.perf_counter()

        k_plot_list[-1] = k_plot - 1
        t_max_bat[k_bat] = t_con[-1]

        # convert settlement to arrays
        t_con_stab = np.array(t_con_stab) / 3600.0 / 24 / 365
        # s_con_stab = np.array(s_con_stab) * 1.0e03  # convert to mm
        t_con = np.array(t_con) / 60.0 / 60 / 24 / 365  # convert to yr
        # s_con = np.array(s_con) * 1.0e03  # convert to mm

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
        t_50_05 = np.sqrt(t0) + ((np.sqrt(t1) - np.sqrt(t0))
                                 * (s_50 - s0) / (s1 - s0))

        print(f"Run time = {toc - tic: 0.4f} s")
        runtime_bat[k_bat] = toc - tic
        print(f"Total settlement = {s_con[-1]} mm")
        s_tot_bat[k_bat] = s_con[-1]
        print(f"t_50_05 = {t_50_05} min^0.5")
        t_50_bat[k_bat] = t_50_05

        # save results to .out file
        with open(fname + ".out", "a", encoding="utf-8") as fout:
            fout.write(
                f"{H_layer_bat[k_bat]:.8e}"
                + f"  {num_elements_bat[k_bat]: 4d}"
                + f"  {dt_sim_bat[k_bat]:.8e}"
                + f"  {t_max_bat[k_bat]:.8e}"
                + f"  {t_50_bat[k_bat]:.8e}"
                + f"  {s_tot_bat[k_bat]:.8e}"
                + f"  {runtime_bat[k_bat]:.8e}\n"
            )

        plt.figure(figsize=(3.5, 4))
        plt.semilogx(t_con, s_con, k_plot_init, label="Gs=2.78")
        plt.semilogx(t_con_bench, s_con_bench1_278, "ok", label="FoxPu2015")
        plt.xlabel(r"Time, $t$ [$yr$]")
        plt.ylabel(r"Settlement, $s$ [$m$]")
        plt.legend()

        plt.savefig(
            "examples/bench_settle"
            + f"_H{H_layer:0.3g}m"
            + f"_Ne{num_elements:d}"
            + f"_dt{dt_sim:0.2e}s"
            + f"_qi{qs0:0.2e}Pa"
            + f"_qf{qs1:0.2e}Pa"
            + ".svg"
        )

        plt.figure(figsize=(8, 8))

        plt.subplot(2, 2, 1)
        plt.plot(e_nod[:, 0], z_nod, k_plot_init, label="Initial")
        for k_plot, k_label, k_line in zip(k_plot_list,
                                           k_plot_labels,
                                           k_plot_linetype):
            plt.plot(e_nod[:, k_plot], z_nod, k_line,
                     label=k_label)
        plt.ylim((np.max(z_nod), np.min(z_nod)))
        plt.legend()
        plt.xlabel(r"Void Ratio, $e$")
        plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")

        plt.subplot(2, 2, 2)
        plt.semilogx(hyd_cond_int[:, 0], z_int, k_plot_init, label="Initial")
        for k_plot, k_label, k_line in zip(k_plot_list,
                                           k_plot_labels,
                                           k_plot_linetype):
            plt.semilogx(hyd_cond_int[:, k_plot], z_int, k_line,
                         label=k_label)
        plt.ylim((np.max(z_nod), np.min(z_nod)))
        # plt.legend()
        plt.xlabel(r"Hyd Cond, $k$ [$m/s$]")

        plt.subplot(2, 2, 3)
        plt.plot(sig_p_int[:, 0], z_int, k_plot_init, label="Initial")
        for k_plot, k_label, k_line in zip(k_plot_list,
                                           k_plot_labels,
                                           k_plot_linetype):
            plt.plot(sig_p_int[:, k_plot], z_int, k_line,
                     label=k_label)
        plt.ylim((np.max(z_nod), np.min(z_nod)))
        # plt.legend()
        plt.xlabel(r"Eff Stress, $\sigma^\prime$ [$kPa$]")
        plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")

        plt.subplot(2, 2, 4)
        for k_plot, k_label, k_line in zip(k_plot_list,
                                           k_plot_labels,
                                           k_plot_linetype):
            plt.plot(ue_int[:, k_plot] / (qs1 - qs0) * 1e3, z_int, k_line,
                     label=k_label)
        # plt.legend()
        plt.xlabel(r"Norm Exc Pore Pres, $u_e/\Delta q$")

        plt.savefig(
            "examples/bench_void_sig"
            + f"_H{H_layer:0.3g}m"
            + f"_Ne{num_elements:d}"
            + f"_dt{dt_sim:0.2e}s"
            + f"_qi{qs0:0.2e}Pa"
            + f"_qf{qs1:0.2e}Pa"
            + ".svg"
        )


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
            gam_p[k] = (Gs - 1.0) / (1.0 + e) * gam_w
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
