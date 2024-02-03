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


def main():
    # define simulation parameters
    sim_params = np.loadtxt("examples/con_static_params.csv")
    H_layer = sim_params[0]
    num_elements = int(sim_params[1])
    dt_sim = sim_params[2]
    t_max = sim_params[3]

    print(f"H_layer = {H_layer}")
    print(f"num_elements = {num_elements}")
    print(f"dt_sim = {dt_sim}")
    print(f"t_max = {t_max}")

    # set plotting parameters
    plt.rc("font", size=8)
    dt_plot = np.max([60.0 * 100, dt_sim])  # in seconds
    n_plot = int(np.floor(t_max / dt_plot) + 1)
    n_plot_stab = 15
    # arrow_props = {
    #     "width": 0.5,
    #     "linewidth": 0.5,
    #     "headwidth": 3.0,
    #     "headlength": 5.5,
    #     "fill": False,
    # }

    # define the material properties
    m = Material(
        spec_grav_solids=2.60,
        hyd_cond_index=0.305,
        hyd_cond_0=4.05e-4,
        void_ratio_0_hyd_cond=2.60,
        void_ratio_min=0.30,
        void_ratio_tr=2.60,
        void_ratio_0_comp=2.60,
        comp_index_unfrozen=0.421,
        rebound_index_unfrozen=0.08,
        eff_stress_0_comp=2.80e00,
    )

    # create consolidation analysis
    # and generate the mesh
    con_static = ConsolidationAnalysis1D(
        z_range=[0.0, H_layer],
        num_elements=num_elements,
        generate=True,
    )
    con_static.implicit_error_tolerance = 1e-6

    # assign material properties to integration points
    for e in con_static.elements:
        for ip in e.int_pts:
            ip.material = m

    # initialize plotting arrays
    z_nod = np.array([nd.z for nd in con_static.nodes])
    z_int = np.array([ip.z for e in con_static.elements for ip in e.int_pts])
    e_nod = np.zeros((len(z_nod), n_plot + 1))
    e_nod_stab = np.zeros((len(z_nod), n_plot_stab + 1))
    sig_p_int = np.zeros((len(z_int), n_plot + 1))
    hyd_cond_int = np.zeros((len(z_int), n_plot + 1))

    # initialize void ratio profile
    e0, sig_p_0_exp, hyd_cond_0_exp = calculate_static_profile(m, 1.0e5, z_nod)
    e1, sig_p_1_exp, hyd_cond_1_exp = calculate_static_profile(
        m, 1.1e6, z_nod)
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

    # initialize global matrices and vectors
    con_static.time_step = dt_sim  # in seconds
    con_static.initialize_global_system(t0=0.0)

    tic = time.perf_counter()

    # stabilize at initial void ratio profile
    # (no settlement expected during this period)
    t_con_stab = [0.0]
    s_con_stab = [0.0]
    t_plot = dt_plot
    k_plot = 0
    tol_s = 1e-6
    eps_s = 1.0
    print("initial stabilization")
    while eps_s > tol_s and k_plot < n_plot_stab:
        while con_static._t1 < t_plot:
            con_static.initialize_time_step()
            con_static.iterative_correction_step()
            t_con_stab.append(con_static._t1)
            s_con_stab.append(con_static.calculate_total_settlement())
        eps_s = np.abs((s_con_stab[-1] - s_con_stab[-2]) / s_con_stab[-1])
        print(
            f"t = {con_static._t1 / 60.0:0.3f} min, "
            + f"s_con = {s_con_stab[-1] * 1e3:0.3f} mm, "
            + f"eps_s =  {eps_s:0.4e}"
        )
        e_nod_stab[:, k_plot] = con_static._void_ratio_vector[:]
        k_plot += 1
        t_plot += dt_plot
    e_nod[:, 0] = e_nod_stab[:, k_plot - 1]
    sig_p_int[:, 0] = 1.0e-3 * np.array(
        [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
    )
    hyd_cond_int[:, 0] = np.array(
        [ip.hyd_cond for e in con_static.elements for ip in e.int_pts]
    )

    # compute expected Terzaghi result
    sig_p_trz = np.zeros((len(z_nod), n_plot + 1))
    e_trz = np.zeros((len(z_nod), n_plot + 1))
    H = 0.5 * (con_static.nodes[-1].z - con_static.nodes[0].z)
    ui = sig_p_1_exp[0] - sig_p_0_exp[0]
    e0t = np.average(e_nod[:, 0])
    # e0t = np.average(e0)
    e1t = np.average(e1)
    de_trz = e0 - e1
    Gs = m.spec_grav_solids
    gam_w = unit_weight_water * 1e-3
    gam_b = (Gs - 1.0) / (1.0 + e0t) * gam_w
    sig_0 = m.eff_stress(e0t, sig_p_0_exp[0] * 1e3)[0] * 1e-3
    sig_1 = m.eff_stress(e1t, sig_p_0_exp[0] * 1e3)[0] * 1e-3
    dsig_de_avg = (sig_1 - sig_0) / (e0t - e1t)
    mv = 1.0 / (1.0 + e0t) / dsig_de_avg
    kt = np.average(hyd_cond_0_exp)
    cv = kt / mv / gam_w
    sig_p_1_trz = sig_p_1_exp[0] + gam_b * z_nod
    s_tot_trz = 1e3 * (con_static.nodes[-1].z -
                       con_static.nodes[0].z) / (1.0 + e0t) * (e0t - e1t)

    sig_p_trz[:, 0] = sig_p_1_trz[:] - ui
    e_trz[:, 0] = e0[:]

    # update boundary conditions
    # (this corresponds to an application of
    # a surface load increment)
    print("apply static load increment")
    void_ratio_boundary_0.bnd_value = e1[0]
    void_ratio_boundary_1.bnd_value = e1[-1]
    con_static._t1 = 0.0
    for nd in con_static.nodes:
        nd.void_ratio_0 = nd.void_ratio
    con_static.initialize_global_system(0.0)

    t_con = [0.0]
    s_con = [0.0]
    s_trz = [0.0]
    t_trz = [0.0]
    k_plot = 0
    t_plot = 0.0
    eps_s = 1.0
    while eps_s > tol_s and k_plot < n_plot:
        k_plot += 1
        t_plot += dt_plot
        while con_static._t1 < t_plot:
            con_static.initialize_time_step()
            con_static.iterative_correction_step()
            t_con.append(con_static._t1)
            s_con.append(con_static.calculate_total_settlement())
        eps_s = np.abs((s_con[-1] - s_con[-2]) / s_con[-1])
        print(
            f"t = {con_static._t1 / 60.0:0.3f} min, "
            + f"s_con = {s_con[-1] * 1e3:0.3f} mm, "
            + f"eps_s =  {eps_s:0.4e}"
        )
        e_nod[:, k_plot] = con_static._void_ratio_vector[:]
        sig_p_int[:, k_plot] = 1.0e-3 * np.array(
            [ip.eff_stress for e in con_static.elements for ip in e.int_pts]
        )
        hyd_cond_int[:, k_plot] = np.array(
            [ip.hyd_cond for e in con_static.elements for ip in e.int_pts]
        )
        ue, U_avg, Uz = terzaghi_consolidation(
            z_nod, con_static._t1, cv, H, ui
        )
        sig_p_trz[:, k_plot] = sig_p_1_trz[:] - ue
        e_trz[:, k_plot] = e0[:] - de_trz[:] * Uz[:]
        s_trz.append(s_tot_trz * U_avg)
        t_trz.append(t_con[-1])

    toc = time.perf_counter()

    # convert settlement to arrays
    t_con = np.array(t_con) / 60.0  # convert to min
    s_con = np.array(s_con) * 1.0e03  # convert to mm
    t_trz = np.array(t_trz) / 60.0
    s_trz = np.array(s_trz)

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
    print(f"Total settlement = {s_con[-1]} m")
    print(f"t_50_05 = {t_50_05} min^0.5")

    # k_lab = np.array(
    #     [
    #         np.floor(0.2 * mesh.num_nodes),
    #         np.floor(0.4 * mesh.num_nodes),
    #         np.floor(0.6 * mesh.num_nodes),
    #     ],
    #     dtype=int,
    # )
    # t_lab = np.array([dt_plot, 9 * dt_plot, 17 * dt_plot]) / 60.0
    # e_lab = np.array([1.35, 1.30, 1.30])
    # z_lab = np.array([1.25, 5.00, 10.0])

    # convert z to cm
    # z_nod *= 100.0
    # z_int *= 100.0

    plt.figure(figsize=(3.5, 4))
    plt.plot(np.sqrt(t_con), s_con, "-k", label="current")
    plt.plot(np.sqrt(t_trz), s_trz, "--k", label="Terzaghi")
    plt.xlabel(r"Root Time, $t^{0.5}$ [$min^{0.5}$]")
    plt.ylabel(r"Settlement, $s$ [$mm$]")
    # plt.ylim((31, -1))
    plt.legend()

    plt.savefig("examples/con_static_settlement.svg")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(e0, z_nod, "-k", label="init")
    plt.plot(e_nod[:, 0], z_nod, ":r", label="post-stab")
    plt.plot(e_trz[:, 1], z_nod, "-.b", label="Terzaghi", linewidth=0.5)
    # plt.annotate(
    #     text="      ",
    #     xy=(e_trz[k_lab[0], 1], z_nod[k_lab[0]]),
    #     xytext=(e_lab[0], z_lab[0]),
    #     arrowprops=arrow_props,
    # )
    plt.plot(e_nod[:, 1], z_nod, ":b", label="consol (act)")
    # plt.annotate(
    #     text=f"{t_lab[0]:0.0f} min",
    #     xy=(e_nod[k_lab[0], 1], z_nod[k_lab[0]]),
    #     xytext=(e_lab[0], z_lab[0]),
    #     arrowprops=arrow_props,
    # )
    for k_plot_plot in [2, 3, k_plot]:
        plt.plot(e_trz[:, k_plot_plot], z_nod, "-.b", linewidth=0.5)
        plt.plot(e_nod[:, k_plot_plot], z_nod, ":b")
        # if k_plot > 0:
        #     plt.annotate(
        #         text="   ",
        #         xy=(e_trz[k_lab[k_plot - 1], k_plot],
        #         z_nod[k_lab[k_plot - 1]]),
        #         xytext=(e_lab[k_plot - 1], z_lab[k_plot - 1]),
        #         arrowprops=arrow_props,
        #     )
        #     plt.annotate(
        #         text=f"{t_lab[k_plot - 1]:0.0f}",
        #         xy=(e_nod[k_lab[k_plot - 1], k_plot],
        #         z_nod[k_lab[k_plot - 1]]),
        #         xytext=(e_lab[k_plot - 1], z_lab[k_plot - 1]),
        #         arrowprops=arrow_props,
        # )
    plt.plot(e1, z_nod, "--k", label="final (exp)")
    # plt.ylim((20, 0))
    # plt.xlim((1.0, 1.8))
    plt.legend()
    plt.xlabel(r"Void Ratio, $e$")
    plt.ylabel(r"Depth (Lagrangian), $Z$ [$m$]")

    plt.subplot(1, 3, 2)
    plt.plot(sig_p_0_exp, z_nod, "-k", label="init")
    plt.plot(sig_p_int[:, 0], z_int, ":r", label="post-stab")
    plt.plot(sig_p_1_exp, z_nod, "--k", label="final (exp)")
    plt.plot(sig_p_int[:, 1], z_int, ":b", label="consol (act)")
    plt.plot(sig_p_trz[:, 1], z_nod, "-.b", label="Terzaghi", linewidth=0.5)
    for k_plot_plot in [2, 3, k_plot]:
        plt.plot(sig_p_int[:, k_plot_plot], z_int, ":b")
        plt.plot(sig_p_trz[:, k_plot_plot], z_nod, "-.b", linewidth=0.5)
    # plt.ylim((20, 0))
    # plt.legend()
    plt.xlabel(r"Eff Stress, $\sigma^\prime$ [$kPa$]")

    plt.subplot(1, 3, 3)
    plt.semilogx(hyd_cond_0_exp, z_nod, "-k", label="init")
    plt.semilogx(hyd_cond_int[:, 0], z_int, ":r", label="post-stab")
    plt.semilogx(hyd_cond_1_exp, z_nod, "--k", label="final (exp)")
    plt.semilogx(hyd_cond_int[:, 1], z_int, ":b", label="consol (act)")
    for k_plot_plot in [2, 3, k_plot]:
        plt.semilogx(hyd_cond_int[:, k_plot_plot], z_int, ":b")
    # plt.ylim((20, 0))
    # plt.legend()
    plt.xlabel(r"Hyd Cond, $k$ [$m/s$]")

    plt.savefig("examples/con_static_void_sig_profiles.svg")


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
