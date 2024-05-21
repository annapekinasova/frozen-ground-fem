import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
    vol_heat_cap_water as Cw,
    vol_heat_cap_ice as Ci,
    thrm_cond_ice as lam_i,
    thrm_cond_water as lam_w,
)

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
)


def main():
    # create thermal analysis object
    ta = ThermalAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 10.0
    ta.generate_mesh(num_elements=25, order=3)
    # ta.implicit_error_tolerance = 1e-4

    # znod0 = np.linspace(0.0, 1.0, 101)
    # znod1 = np.linspace(1.0, 10.0, 181)[1:]
    # znod = np.hstack([znod0, znod1])
    # for z, nd in zip(znod, ta.nodes):
    #     nd.z = z

    # define material properties
    # and initialize integration points
    cs = 2.0e6 / 2.65e3
    mtl = Material(
        thrm_cond_solids=2.5,
        spec_grav_solids=2.65,
        spec_heat_cap_solids=cs,
    )
    void_ratio = 0.35 / (1.0 - 0.35)
    for e in ta.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.void_ratio = void_ratio
            ip.void_ratio_0 = void_ratio

    # set initial temperature conditions
    T0 = +5.0
    for nd in ta.nodes:
        nd.temp = T0

    # create thermal boundary conditions
    # fixed temp at top
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = -5.0
    # zero flux at bottom
    grad_boundary = ThermalBoundary1D(
        (ta.nodes[-1],),
        (ta.elements[-1].int_pts[-1],),
    )
    # grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    # grad_boundary.bnd_value = 0.0
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    grad_boundary.bnd_value = T0

    # assign thermal boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)

    # initialize plotting variables
    z_nod = np.array([nd.z for nd in ta.nodes])
    s_per_day = 3.6e3 * 24.0
    t_plot = np.linspace(0.0, 150.0, 31) * s_per_day

    # # initialize figure
    # plt.figure(figsize=(6, 9))
    # T_vec = np.array([nd.temp for nd in ta.nodes])
    # plt.plot(
    #     T_vec,
    #     z_nod,
    #     "--r",
    #     label="t=0 days",
    #     linewidth=2.0,
    # )

    # initialize global matrices and vectors
    ta.time_step = 0.001
    adapt_dt = True
    ta.initialize_global_system(t0=0.0)

    # np.set_printoptions(
    #     formatter={"float_kind": lambda x: f"{x: 0.3e}"}, linewidth=2000
    # )

    # print out parameters
    ip = ta.elements[0].int_pts[0]
    print(f"dz = {ta.elements[0].jacobian}")
    print(f"Cs = {ip.material.vol_heat_cap_solids}")
    print(f"Ci = {Ci}")
    print(f"Cw = {Cw}")
    print(f"lam_s = {ip.material.thrm_cond_solids}")
    print(f"lam_i = {lam_i}")
    print(f"lam_w = {lam_w}")
    print("top ip:")
    print(f"C = {ip.vol_heat_cap}")
    print(f"lam = {ip.thrm_cond}")
    ip = ta.elements[-1].int_pts[-1]
    print("bot ip:")
    print(f"C = {ip.vol_heat_cap}")
    print(f"lam = {ip.thrm_cond}")

    # solve to selected time points
    Zt_freeze = [0.0]
    T_freeze = np.zeros((3, len(z_nod)))
    t_Tplot = [10.0, 50.0, 100.0]
    k_Tplot = 0
    for tf in t_plot[1:]:
        ta.solve_to(tf, adapt_dt=adapt_dt)
        dthw_dT_max = 0.0
        for e in ta.elements:
            for ip in e.int_pts:
                if ip.vol_water_cont_temp_gradient > dthw_dT_max:
                    dthw_dT_max = ip.vol_water_cont_temp_gradient
        # save temp profile
        if (tf // s_per_day) in t_Tplot:
            T_freeze[k_Tplot, :] = ta._temp_vector[:]
            k_Tplot += 1
        # find thaw depth
        # first find the element containing T=0.0
        for k, e in enumerate(ta.elements):
            if e.nodes[0].temp * e.nodes[-1].temp < 0.0:
                ee = e
                break
        # get temperatures and depths for this element
        jac = ee.jacobian
        Te = np.array([nd.temp for nd in ee.nodes])
        ze = np.array([nd.z for nd in ee.nodes])
        # perform Newton-Raphson to find T = 0.0
        s = 0.5
        eps_a = 1.0
        eps_s = 1.0e-8
        while eps_a > eps_s:
            N = ee._shape_matrix(s)
            B = ee._gradient_matrix(s, jac)
            ds = (N @ Te)[0] / (jac * B @ Te)[0]
            s -= ds
            eps_a = np.abs(ds / s)
        # compute Z
        N = ee._shape_matrix(s)
        Zte = (N @ ze)[0]
        Zt_freeze.append(Zte)
        C1 = ta.elements[0].int_pts[0].vol_heat_cap
        C2 = ta.elements[-1].int_pts[-1].vol_heat_cap
        lam_1 = ta.elements[0].int_pts[0].thrm_cond
        lam_2 = ta.elements[-1].int_pts[-1].thrm_cond
        alpha_1 = lam_1 / C1
        alpha_2 = lam_2 / C2
        lam_21 = lam_2 / lam_1
        alpha_12 = alpha_1 / alpha_2
        print(
            f"t = {tf / s_per_day} days, "
            + f"Z = {Zte:0.4f} m, "
            + f"lam_21 = {lam_21:0.4f}, "
            + f"alpha_12 = {alpha_12:0.4f}, "
            + f"dt = {ta.time_step / s_per_day:0.4e} days, "
            + f"dthw_dT_max = {dthw_dT_max} 1/K"
        )

    # # finalize plot labels
    # plt.ylim(ta.z_max, ta.z_min)
    # plt.legend()
    # plt.xlabel("temperature, T [deg C]")
    # plt.ylabel("depth, z [m]")
    # plt.savefig("examples/thermal_freeze_thaw_benchmark_profiles.svg")

    # reset initial and boundary conditions
    T0 = -5.0
    for nd in ta.nodes:
        nd.temp = T0
    temp_boundary.bnd_value = +5.0
    grad_boundary.bnd_value = T0

    # initialize global matrices and vectors
    ta.time_step = 0.001
    ta.initialize_global_system(t0=0.0)

    # print out parameters
    print()
    ip = ta.elements[0].int_pts[0]
    print(f"dz = {ta.elements[0].jacobian}")
    print(f"Cs = {ip.material.vol_heat_cap_solids}")
    print(f"Ci = {Ci}")
    print(f"Cw = {Cw}")
    print(f"lam_s = {ip.material.thrm_cond_solids}")
    print(f"lam_i = {lam_i}")
    print(f"lam_w = {lam_w}")
    print("top ip:")
    print(f"C = {ip.vol_heat_cap}")
    print(f"lam = {ip.thrm_cond}")
    ip = ta.elements[-1].int_pts[-1]
    print("bot ip:")
    print(f"C = {ip.vol_heat_cap}")
    print(f"lam = {ip.thrm_cond}")

    # solve to selected time points
    Zt_thaw = [0.0]
    T_thaw = np.zeros((3, len(z_nod)))
    k_Tplot = 0
    for tf in t_plot[1:]:
        ta.solve_to(tf, adapt_dt=adapt_dt)
        dthw_dT_max = 0.0
        for e in ta.elements:
            for ip in e.int_pts:
                if ip.vol_water_cont_temp_gradient > dthw_dT_max:
                    dthw_dT_max = ip.vol_water_cont_temp_gradient
        # save temp profile
        if (tf // s_per_day) in t_Tplot:
            T_thaw[k_Tplot, :] = ta._temp_vector[:]
            k_Tplot += 1
        # find thaw depth
        # first find the element containing T=0.0
        for k, e in enumerate(ta.elements):
            if e.nodes[0].temp * e.nodes[-1].temp < 0.0:
                ee = e
                break
        # get temperatures and depths for this element
        jac = ee.jacobian
        Te = np.array([nd.temp for nd in ee.nodes])
        ze = np.array([nd.z for nd in ee.nodes])
        # perform Newton-Raphson to find T = 0.0
        s = 0.5
        eps_a = 1.0
        eps_s = 1.0e-8
        while eps_a > eps_s:
            N = ee._shape_matrix(s)
            B = ee._gradient_matrix(s, jac)
            ds = (N @ Te)[0] / (jac * B @ Te)[0]
            s -= ds
            eps_a = np.abs(ds / s)
        # compute Z
        N = ee._shape_matrix(s)
        Zte = (N @ ze)[0]
        Zt_thaw.append(Zte)
        C1 = ta.elements[0].int_pts[0].vol_heat_cap
        C2 = ta.elements[-1].int_pts[-1].vol_heat_cap
        lam_1 = ta.elements[0].int_pts[0].thrm_cond
        lam_2 = ta.elements[-1].int_pts[-1].thrm_cond
        alpha_1 = lam_1 / C1
        alpha_2 = lam_2 / C2
        lam_21 = lam_2 / lam_1
        alpha_12 = alpha_1 / alpha_2
        print(
            f"t = {tf / s_per_day} days, "
            + f"Z = {Zte:0.4f} m, "
            + f"lam_21 = {lam_21:0.4f}, "
            + f"alpha_12 = {alpha_12:0.4f}, "
            + f"dt = {ta.time_step / s_per_day:0.4e} days, "
            + f"dthw_dT_max = {dthw_dT_max} 1/K"
        )

    t_Neumann = np.array([
        0.00,
        1.00,
        2.00,
        3.00,
        4.00,
        5.00,
        10.00,
        15.00,
        20.00,
        25.00,
        30.00,
        35.00,
        40.00,
        45.00,
        50.00,
        55.00,
        60.00,
        65.00,
        70.00,
        75.00,
        80.00,
        85.00,
        90.00,
        95.00,
        100.00,
        105.00,
        110.00,
        115.00,
        120.00,
        125.00,
        130.00,
        135.00,
        140.00,
        145.00,
        150.00,
    ])
    Z_Neumann_freeze = np.array([
        0.0000,
        0.1294,
        0.1830,
        0.2241,
        0.2588,
        0.2893,
        0.4092,
        0.5011,
        0.5787,
        0.6470,
        0.7087,
        0.7655,
        0.8183,
        0.8680,
        0.9149,
        0.9596,
        1.0023,
        1.0432,
        1.0826,
        1.1206,
        1.1573,
        1.1929,
        1.2275,
        1.2611,
        1.2939,
        1.3259,
        1.3571,
        1.3876,
        1.4174,
        1.4466,
        1.4753,
        1.5034,
        1.5310,
        1.5581,
        1.5847,
    ])
    Z_Neumann_thaw = np.array([
        0.0000,
        0.0904,
        0.1278,
        0.1565,
        0.1807,
        0.2020,
        0.2857,
        0.3500,
        0.4041,
        0.4518,
        0.4949,
        0.5346,
        0.5715,
        0.6061,
        0.6389,
        0.6701,
        0.6999,
        0.7285,
        0.7560,
        0.7825,
        0.8082,
        0.8330,
        0.8572,
        0.8807,
        0.9036,
        0.9259,
        0.9477,
        0.9690,
        0.9898,
        1.0102,
        1.0302,
        1.0499,
        1.0691,
        1.0880,
        1.1066,
    ])

    plt.figure(figsize=(4.0, 4.0))
    plt.plot(t_Neumann, Z_Neumann_freeze, "--k", label="Neumann")
    plt.plot(t_Neumann, Z_Neumann_thaw, "--k")
    plt.plot(t_plot / s_per_day, Zt_freeze, "ob", label="freezing",
             markerfacecolor="none")
    plt.plot(t_plot / s_per_day, Zt_thaw, "sr", label="thawing",
             markerfacecolor="none")
    plt.ylim((4.0, 0.0))
    plt.legend()
    plt.xlabel("time, t [days]")
    plt.ylabel("freeze/thaw depth, Z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_front.svg")

    z_Neumann = np.array([
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
        1.40,
        1.50,
        1.60,
        1.70,
        1.80,
        1.90,
        2.00,
    ])

    T_Neumann_freeze = np.zeros((3, 21))
    T_Neumann_freeze[0, :] = [
        -5.0000,
        -3.7628,
        -2.5315,
        -1.3118,
        -0.1092,
        0.5023,
        1.0275,
        1.5196,
        1.9760,
        2.3945,
        2.7743,
        3.1152,
        3.4180,
        3.6839,
        3.9150,
        4.1136,
        4.2826,
        4.4247,
        4.5430,
        4.6404,
        4.7198,
    ]
    T_Neumann_freeze[1, :] = [
        -5.0000,
        -4.4464,
        -3.8933,
        -3.3412,
        -2.7907,
        -2.2423,
        -1.6965,
        -1.1538,
        -0.6146,
        -0.0795,
        0.2131,
        0.4587,
        0.6984,
        0.9321,
        1.1592,
        1.3796,
        1.5930,
        1.7991,
        1.9977,
        2.1888,
        2.3722,
    ]
    T_Neumann_freeze[2, :] = [
        -5.0000,
        -4.6085,
        -4.2172,
        -3.8262,
        -3.4359,
        -3.0462,
        -2.6575,
        -2.2698,
        -1.8835,
        -1.4986,
        -1.1153,
        -0.7339,
        -0.3544,
        0.0109,
        0.1881,
        0.3627,
        0.5345,
        0.7034,
        0.8691,
        1.0317,
        1.1910,
    ]

    T_Neumann_thaw = np.zeros((3, 21))
    T_Neumann_thaw[0, :] = [
        5.0000,
        3.2275,
        1.4741,
        -0.0454,
        -0.3609,
        -0.6705,
        -0.9729,
        -1.2668,
        -1.5512,
        -1.8250,
        -2.0874,
        -2.3377,
        -2.5753,
        -2.7999,
        -3.0111,
        -3.2087,
        -3.3928,
        -3.5636,
        -3.7211,
        -3.8658,
        -3.9981,
    ]
    T_Neumann_thaw[1, :] = [
        5.0000,
        4.2062,
        3.4141,
        2.6254,
        1.8418,
        1.0650,
        0.2965,
        -0.0869,
        -0.2283,
        -0.3687,
        -0.5079,
        -0.6458,
        -0.7823,
        -0.9172,
        -1.0505,
        -1.1820,
        -1.3117,
        -1.4394,
        -1.5651,
        -1.6886,
        -1.8100,
    ]
    T_Neumann_thaw[2, :] = [
        5.0000,
        4.4386,
        3.8778,
        3.3182,
        2.7604,
        2.2050,
        1.6526,
        1.1038,
        0.5592,
        0.0192,
        -0.0970,
        -0.1970,
        -0.2966,
        -0.3956,
        -0.4940,
        -0.5917,
        -0.6888,
        -0.7851,
        -0.8806,
        -0.9754,
        -1.0693,
    ]

    plt.figure(figsize=(4.0, 4.0))
    plt.plot(T_Neumann_freeze[0, :], z_Neumann, "--k", label="Neumann")
    plt.plot(T_Neumann_freeze[1, :], z_Neumann, "--k")
    plt.plot(T_Neumann_freeze[2, :], z_Neumann, "--k")
    plt.plot(T_freeze[0, :], z_nod, "ob", label="t=10 days",
             markerfacecolor="none")
    plt.plot(T_freeze[1, :], z_nod, "sb", label="50 days",
             markerfacecolor="none")
    plt.plot(T_freeze[2, :], z_nod, "^b", label="100 days",
             markerfacecolor="none")
    plt.ylim((6.0, 0.0))
    plt.legend()
    plt.xlabel("temperature, T [deg C]")
    plt.ylabel("depth, z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_profiles_freeze.svg")

    plt.figure(figsize=(4.0, 4.0))
    plt.plot(T_Neumann_thaw[0, :], z_Neumann, "--k", label="Neumann")
    plt.plot(T_Neumann_thaw[1, :], z_Neumann, "--k")
    plt.plot(T_Neumann_thaw[2, :], z_Neumann, "--k")
    plt.plot(T_thaw[0, :], z_nod, "or", label="t=10 days",
             markerfacecolor="none")
    plt.plot(T_thaw[1, :], z_nod, "sr", label="50 days",
             markerfacecolor="none")
    plt.plot(T_thaw[2, :], z_nod, "^r", label="100 days",
             markerfacecolor="none")
    plt.ylim((6.0, 0.0))
    plt.legend()
    plt.xlabel("temperature, T [deg C]")
    plt.ylabel("depth, z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_profiles_thaw.svg")


if __name__ == "__main__":
    main()
