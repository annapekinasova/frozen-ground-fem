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
    ta.generate_mesh(num_elements=100, order=1)
    ta.implicit_error_tolerance = 1e-5

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
    T0 = -5.0
    for nd in ta.nodes:
        nd.temp = T0

    # create thermal boundary conditions
    # fixed temp at top
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = +5.0
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
    ta.time_step = 0.1
    adapt_dt = True
    ta.initialize_global_system(t0=0.0)

    # print out parameters
    ip = ta.elements[0].int_pts[0]
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
    Z_Neumann = np.array([
        0.0000,
        0.1295,
        0.1831,
        0.2243,
        0.2590,
        0.2895,
        0.4094,
        0.5015,
        0.5790,
        0.6474,
        0.7092,
        0.7660,
        0.8189,
        0.8685,
        0.9155,
        0.9602,
        1.0029,
        1.0439,
        1.0833,
        1.1213,
        1.1581,
        1.1937,
        1.2283,
        1.2620,
        1.2948,
        1.3267,
        1.3580,
        1.3885,
        1.4183,
        1.4476,
        1.4763,
        1.5044,
        1.5320,
        1.5591,
        1.5857,
    ])

    plt.figure(figsize=(7, 4))
    plt.plot(t_plot / s_per_day, Zt, "--k", label="frozen_ground_fem")
    plt.plot(t_Neumann, Z_Neumann, "-b", label="Neumann")
    plt.ylim((4.0, 0.0))
    plt.legend()
    plt.xlabel("time, t [days]")
    plt.ylabel("freeze/thaw depth, Z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_front.svg")


if __name__ == "__main__":
    main()
