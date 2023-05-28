import numpy as np
import matplotlib.pyplot as plt

import frozen_ground_fem.materials as mtl
import frozen_ground_fem.geometry as geom
import frozen_ground_fem.consolidation as consol


def main():
    # define the material properties
    m = mtl.Material(
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

    # generate the mesh
    mesh = geom.Mesh1D(
        z_range=[0.0, 10.0],
        num_nodes=21,
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
    mesh.add_boundary(upper_boundary)

    # create consolidation analysis
    con_static = consol.ConsolidationAnalysis1D(mesh)

    # intitial void ratio profile
    # (calculated such that static conditions
    # should be preserved, all points on NCL)
    e0 = np.array(
        [
            2.788,
            1.343,
            1.192,
            1.107,
            1.048,
            1.002,
            0.965,
            0.933,
            0.906,
            0.882,
            0.861,
            0.841,
            0.824,
            0.807,
            0.792,
            0.778,
            0.765,
            0.753,
            0.741,
            0.730,
            0.720,
        ]
    )

    e_fin_exp = np.array(
        [
            0.557,
            0.554,
            0.551,
            0.548,
            0.544,
            0.541,
            0.538,
            0.535,
            0.531,
            0.528,
            0.525,
            0.522,
            0.518,
            0.515,
            0.512,
            0.509,
            0.506,
            0.503,
            0.500,
            0.497,
            0.494,
        ]
    )

    # approximate expected initial effective stress profile
    sig_p_0_exp = 1.0e-03 * np.array(
        [
            1.00e00,
            2.71e03,
            6.18e03,
            9.83e03,
            1.36e04,
            1.75e04,
            2.14e04,
            2.55e04,
            2.96e04,
            3.37e04,
            3.79e04,
            4.21e04,
            4.64e04,
            5.07e04,
            5.51e04,
            5.95e04,
            6.39e04,
            6.84e04,
            7.29e04,
            7.74e04,
            8.20e04,
        ]
    )

    # approximate expected final effective stress profile
    sig_p_fin_exp = 1.0e-03 * np.array(
        [
            2.00e05,
            2.03e05,
            2.06e05,
            2.10e05,
            2.14e05,
            2.17e05,
            2.21e05,
            2.25e05,
            2.30e05,
            2.34e05,
            2.38e05,
            2.42e05,
            2.46e05,
            2.51e05,
            2.55e05,
            2.59e05,
            2.64e05,
            2.68e05,
            2.73e05,
            2.77e05,
            2.82e05,
        ]
    )

    # expected initial hydraulic conductivity profile
    hyd_cond_0_exp = np.array(
        [
            1.68e-03,
            3.06e-08,
            9.82e-09,
            5.17e-09,
            3.30e-09,
            2.34e-09,
            1.76e-09,
            1.39e-09,
            1.13e-09,
            9.44e-10,
            8.03e-10,
            6.94e-10,
            6.07e-10,
            5.37e-10,
            4.79e-10,
            4.31e-10,
            3.90e-10,
            3.55e-10,
            3.26e-10,
            3.00e-10,
            2.77e-10,
        ]
    )

    # approximate expected final hydraulic conductivity
    hyd_cond_fin_exp = np.array(
        [
            8.08e-11,
            7.93e-11,
            7.75e-11,
            7.56e-11,
            7.38e-11,
            7.20e-11,
            7.02e-11,
            6.85e-11,
            6.68e-11,
            6.52e-11,
            6.36e-11,
            6.21e-11,
            6.06e-11,
            5.92e-11,
            5.78e-11,
            5.64e-11,
            5.51e-11,
            5.39e-11,
            5.26e-11,
            5.15e-11,
            5.03e-11,
        ]
    )

    # assign initial void ratio conditions at the nodes
    for k, nd in enumerate(mesh.nodes):
        nd.void_ratio = e0[k]

    # update void ratio conditions at the integration points
    # (first interpolates void ratio profile from the nodes,
    # then assign initial void ratio to int pts)
    con_static.update_integration_points()
    for e in con_static.elements:
        for ip in e.int_pts:
            ip.void_ratio_0 = ip.void_ratio

    # create void ratio boundary conditions
    void_ratio_boundary = consol.ConsolidationBoundary1D(upper_boundary)
    void_ratio_boundary.bnd_type = (
        consol.ConsolidationBoundary1D.BoundaryType.void_ratio
    )
    void_ratio_boundary.bnd_value = e0[0]
    con_static.add_boundary(void_ratio_boundary)

    plt.rc("font", size=8)

    # initialize plot
    z_vec = np.array([nd.z for nd in mesh.nodes])

    # initialize global matrices and vectors
    con_static.time_step = 365 * 8.64e4 / 52  # ~one week, in seconds
    con_static.initialize_global_system(t0=0.0)

    # stabilize at initial void ratio profile
    # (no settlement expected during this period)
    t_con = [0.0]
    s_con = [0.0]
    print("initial stabilization")
    for k in range(52):
        if not k % 15:
            print(f"t = {con_static._t1 / 8.64e4:0.3f} days")
        con_static.initialize_time_step()
        con_static.iterative_correction_step()
        t_con.append(con_static._t1)
        s_con.append(con_static.calculate_total_settlement())

    z_int = []
    sig_p_0_int = []
    qw_0_int = []
    hyd_cond_0_int = []
    for e in con_static.elements:
        for ip in e.int_pts:
            z_int.append(ip.z)
            sig_p_0_int.append(ip.eff_stress)
            qw_0_int.append(ip.water_flux_rate)
            hyd_cond_0_int.append(ip.hyd_cond)
    z_int = np.array(z_int)
    sig_p_0_int = np.array(sig_p_0_int) * 1.0e-03
    qw_0_int = np.array(qw_0_int)
    hyd_cond_0_int = np.array(hyd_cond_0_int)
    e0_act = con_static._void_ratio_vector.copy()

    # update boundary conditions
    # (this corresponds to an application of
    # a surface load of ~200 kPa)
    void_ratio_boundary.bnd_value = e_fin_exp[0]
    con_static.time_step *= 0.1

    print("apply static load of 200 kPa")
    for k in range(15 * 52):
        con_static.initialize_time_step()
        con_static.iterative_correction_step()
        t_con.append(con_static._t1)
        s_con.append(con_static.calculate_total_settlement())
        if not k % 52:
            print(f"t = {con_static._t1 / 8.64e4:0.3f} days")
            print(f"num_iter = {con_static._iter}")

    sig_p_f_int = []
    qw_f_int = []
    hyd_cond_f_int = []
    for e in con_static.elements:
        for ip in e.int_pts:
            sig_p_f_int.append(ip.eff_stress)
            qw_f_int.append(ip.water_flux_rate)
            hyd_cond_f_int.append(ip.hyd_cond)
    sig_p_f_int = np.array(sig_p_f_int) * 1.0e-03
    qw_f_int = np.array(qw_f_int)
    hyd_cond_f_int = np.array(hyd_cond_f_int)
    e_fin_act = con_static._void_ratio_vector.copy()

    # convert settlement to arrays
    t_con = np.array(t_con) / 8.64e4  # convert to days
    s_con = np.array(s_con) * 1.0e03  # conver to mm

    plt.figure(figsize=(4, 4))
    plt.plot(t_con, s_con)
    plt.xlabel("time [days]")
    plt.ylabel("settlement [mm]")

    plt.savefig("examples/con_static_settlement.png")

    plt.figure(figsize=(7, 4))

    plt.subplot(1, 3, 1)
    plt.plot(e0, z_vec, "-k", label="exp 0")
    plt.plot(e0_act, z_vec, "xk", label="act 0")
    plt.plot(e_fin_exp, z_vec, "--r", label="exp fin")
    plt.plot(e_fin_act, z_vec, "or", label="act fin")
    plt.ylim((10, 0))
    plt.legend()
    plt.xlabel("void ratio")
    plt.ylabel("depth [m]")

    plt.subplot(1, 3, 2)
    plt.plot(sig_p_0_exp, z_vec, "-k", label="exp 0")
    plt.plot(sig_p_0_int, z_int, "xk", label="act 0")
    plt.plot(sig_p_fin_exp, z_vec, "--r", label="exp fin")
    plt.plot(sig_p_f_int, z_int, "or", label="act fin")
    plt.ylim((10, 0))
    plt.legend()
    plt.xlabel("effective stress [kPa]")

    plt.subplot(1, 3, 3)
    plt.semilogx(qw_0_int, z_int, ":b", label="water flux 0")
    plt.semilogx(qw_f_int, z_int, ":k", label="water flux fin")
    plt.semilogx(hyd_cond_0_exp, z_vec, "-k", label="exp 0")
    plt.semilogx(hyd_cond_0_int, z_int, "xk", label="act 0")
    plt.semilogx(hyd_cond_fin_exp, z_vec, "--r", label="exp fin")
    plt.semilogx(hyd_cond_f_int, z_int, "or", label="act fin")
    plt.ylim((10, 0))
    plt.legend()
    plt.xlabel("hydraulic conductivity [m/s]")

    plt.savefig("examples/con_static_void_sig_profiles.png")


if __name__ == "__main__":
    main()
