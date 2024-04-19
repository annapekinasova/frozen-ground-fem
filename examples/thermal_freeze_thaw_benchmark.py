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
    ta.generate_mesh(num_elements=300, order=1)
    ta.implicit_error_tolerance = 1e-4

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
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 0.0
    # grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    # grad_boundary.bnd_value = T0

    # assign thermal boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)

    # initialize plotting variables
    z_nod = np.array([nd.z for nd in ta.nodes])
    s_per_day = 3.6e3 * 24.0
    t_plot = np.linspace(0.0, 150.0, 151) * s_per_day

    # initialize figure
    plt.figure(figsize=(6, 9))
    T_vec = np.array([nd.temp for nd in ta.nodes])
    plt.plot(
        T_vec,
        z_nod,
        "--r",
        label="t=0 days",
        linewidth=2.0,
    )

    # initialize global matrices and vectors
    ta.time_step = 0.001
    adapt_dt = True
    ta.initialize_global_system(t0=0.0)

    np.set_printoptions(
        formatter={"float_kind": lambda x: f"{x: 0.3e}"}, linewidth=2000
    )

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
    Zt = [0.0]
    for tf in t_plot[1:]:
        ta.solve_to(tf, adapt_dt=adapt_dt)
        # print(np.array(
        #     [ip.vol_ice_cont_0 for e in ta.elements for ip in e.int_pts]
        # ))
        # print(np.array(
        #     [ip.vol_ice_cont for e in ta.elements for ip in e.int_pts]
        # ))
        # print(np.array(
        #     [ip.vol_water_cont_temp_gradient
        #         for e in ta.elements for ip in e.int_pts]
        # ))
        # print(ta._temp_vector_0)
        # print(ta._temp_vector)
        # print(ta._heat_flow_matrix_0)
        # print(ta._heat_storage_matrix_0)
        # print(ta._heat_flow_matrix)
        # print(ta._heat_storage_matrix)
        # print(id(ta._temp_vector_0))
        # print(id(ta._temp_vector))
        # print(ta._temp_vector - ta._temp_vector_0)
        dthw_dT_max = 0.0
        for e in ta.elements:
            for ip in e.int_pts:
                if ip.vol_water_cont_temp_gradient > dthw_dT_max:
                    dthw_dT_max = ip.vol_water_cont_temp_gradient
        # plot temp profile
        if not (tf // s_per_day) % 15:
            plt.plot(
                ta._temp_vector, z_nod,
                "--b", linewidth=0.5, label=f"{tf / s_per_day}"
            )
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
        Zt.append(Zte)
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

    # finalize plot labels
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("temperature, T [deg C]")
    plt.ylabel("depth, z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_profiles.svg")

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
    if T0 < 0.0:
        Z_Neumann = np.array([
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
    else:
        Z_Neumann = np.array([
            0.0000,
            0.1187,
            0.1678,
            0.2055,
            0.2373,
            0.2653,
            0.3752,
            0.4595,
            0.5306,
            0.5933,
            0.6499,
            0.7019,
            0.7504,
            0.7959,
            0.8390,
            0.8799,
            0.9191,
            0.9566,
            0.9927,
            1.0275,
            1.0612,
            1.0939,
            1.1256,
            1.1565,
            1.1865,
            1.2158,
            1.2444,
            1.2724,
            1.2998,
            1.3266,
            1.3528,
            1.3786,
            1.4039,
            1.4287,
            1.4532,
        ])

    plt.figure(figsize=(7, 4))
    plt.plot(t_plot / s_per_day, Zt, "--k", label="frozen_ground_fem")
    plt.plot(t_Neumann, Z_Neumann, "-b", label="Neumann")
    plt.ylim((1.6, 0.0))
    plt.legend()
    plt.xlabel("time, t [days]")
    plt.ylabel("freeze/thaw depth, Z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_front.svg")


if __name__ == "__main__":
    main()
