import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
)

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
)


def main():
    # create thermal analysis object
    # define mesh with 20 elements
    # and cubic interpolation
    ta = ThermalAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 30.0
    ta.generate_mesh(num_elements=30)
    ta.implicit_error_tolerance = 1e-6

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
    T0 = +3.8
    for nd in ta.nodes:
        nd.temp = T0

    # create thermal boundary conditions
    # fixed temp at top
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = -7.2
    # zero flux at bottom
    grad_boundary = ThermalBoundary1D(
        (ta.nodes[-1],),
        (ta.elements[-1].int_pts[-1],),
    )
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 0.0

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
    ta.initialize_global_system(t0=0.0)

    # print out parameters
    # ip = ta.elements[0].int_pts[0]
    # print(f"Gs = {ip.material.spec_grav_solids}")
    # print(f"Lw = {latent_heat_fusion_water} ")

    # solve to selected time points
    Zt = [0.0]
    for tf in t_plot[1:]:
        ta.solve_to(tf)
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
            + f"dt = {ta.time_step / s_per_day:0.4e} days"
        )

    # finalize plot labels
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("temperature, T [deg C]")
    plt.ylabel("depth, z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_profiles_freeze.svg")

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
        0.1431,
        0.2024,
        0.2478,
        0.2863,
        0.3200,
        0.4525,
        0.5542,
        0.6399,
        0.7155,
        0.7837,
        0.8465,
        0.9050,
        0.9599,
        1.0118,
        1.0612,
        1.1084,
        1.1536,
        1.1972,
        1.2392,
        1.2798,
        1.3192,
        1.3575,
        1.3947,
        1.4309,
        1.4662,
        1.5008,
        1.5345,
        1.5675,
        1.5998,
        1.6315,
        1.6626,
        1.6931,
        1.7230,
        1.7525,
    ])

    plt.figure(figsize=(7, 4))
    plt.plot(t_plot / s_per_day, Zt, "--k", label="frozen_ground_fem")
    plt.plot(t_Neumann, Z_Neumann, "-b", label="Neumann")
    plt.ylim((15, 0.0))
    plt.legend()
    plt.xlabel("time, t [days]")
    plt.ylabel("freeze/thaw depth, Z [m]")
    plt.savefig("examples/thermal_freeze_thaw_benchmark_freeze_front.svg")


if __name__ == "__main__":
    main()
