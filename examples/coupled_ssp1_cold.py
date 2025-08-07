import numpy as np
from scipy.interpolate import RegularGridInterpolator

from frozen_ground_fem import (
    Material,
    ThermalBoundary1D,
    ConsolidationBoundary1D,
    CoupledAnalysis1D,
)

s_per_day = 8.64e4
s_per_wk = s_per_day * 7.0
s_per_yr = s_per_day * 365.0


def main():
    # setup and generate mesh
    print("Generating mesh:")
    ta = CoupledAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 50.0
    print(f"z_min={ta.z_min:0.4f}, z_max={ta.z_max:0.4f}")
    dTdZ_G = 0.03
    msh_z_list = [0.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    msh_dz_list = [0.25, 0.5, 1.0, 2.5, 5.0]
    num_el_list = []
    z_msh_nod = []
    for k, (z0, dz) in enumerate(zip(msh_z_list[:-1], msh_dz_list)):
        z1 = msh_z_list[k + 1]
        num_el_list.append(int((z1 - z0) // dz))
        z_msh_nod = np.hstack(
            [z_msh_nod, np.linspace(z0, z1, num_el_list[-1] + 1)[:-1]]
        )
    z_msh_nod = np.hstack([z_msh_nod, msh_z_list[-1]])
    num_el = int(np.sum(num_el_list))
    order = 1
    ta.generate_mesh(num_elements=num_el, order=order)
    print(f"num_el={ta.num_elements}")
    print(f"order={order}")
    for nd, zn in zip(ta.nodes, z_msh_nod):
        nd.z = zn

    # define plotting time increments
    t_plot_targ = np.linspace(0.0, 1356.0, 1357) * s_per_wk

    # define analysis parameters
    dt_sim_0 = 0.05 * s_per_day
    adapt_dt = True
    qi = 15.0e3
    tol = 1e-4
    tol_str = f"{tol:0.1e}"
    tol_str = "p".join(tol_str.split("."))
    fname = "examples/" + f"coupled_ssp1_sat_cold_{ta.num_elements}_{tol_str}_"
    print(fname)

    # define material properties
    # and initialize integration points
    mtl = Material(
        spec_grav_solids=2.654,
        thrm_cond_solids=4.109,
        spec_heat_cap_solids=702.8,
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
        void_ratio_lim=0.01,
        void_ratio_min=0.01,
        void_ratio_tr=1.6,
        void_ratio_sep=1.6,
        void_ratio_0_comp=0.333333,
        eff_stress_0_comp=15.0e3,
        comp_index_unfrozen=0.05,
        rebound_index_unfrozen=0.005,
        comp_index_frozen_a1=0.021,
        comp_index_frozen_a2=0.01,
        comp_index_frozen_a3=0.23,
    )

    # assign material properties to elements
    for e in ta.elements:
        e.assign_material(mtl)

    # set initial conditions
    print()
    print("Loading initial conditions:")
    print(f"file = {fname + 'temp_void.in'}")
    temp_void_0 = np.loadtxt(fname + "temp_void.in")
    for k, nd in enumerate(ta.nodes):
        nd.temp = temp_void_0[k, 1]
        nd.void_ratio = temp_void_0[k, 2]
        nd.void_ratio_0 = temp_void_0[k, 3]
        print(
            f"{nd.index: 3d} {nd.z: 10.4f} {nd.temp: 10.4f} {nd.void_ratio: 10.4f} {nd.void_ratio_0: 10.4f}"
        )

    # # show plot of initial conditions
    # print()
    # print("Plotting initial conditions:")
    # z_plt = np.array([nd.z for nd in ta.nodes])
    # e_plt = np.array([nd.void_ratio for nd in ta.nodes])
    # e0_plt = np.array([nd.void_ratio_0 for nd in ta.nodes])
    # T_plt = np.array([nd.temp for nd in ta.nodes])
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.plot(T_plt, z_plt, "-b")
    # plt.subplot(1, 2, 2)
    # plt.plot(e0_plt, z_plt, "--k")
    # plt.plot(e_plt, z_plt, "-r")
    # plt.show()

    print()
    print("Loading Tair boundary data:")
    temp_air_bnd = np.loadtxt(fname + "temp.bnd")
    air_temp_spline = RegularGridInterpolator(
        (temp_air_bnd[:, 0],),
        temp_air_bnd[:, 1],
        method="linear",
    )
    t_min = np.min(temp_air_bnd[:, 0])
    t_max = np.max(temp_air_bnd[:, 0])
    temp_min = np.min(temp_air_bnd[:, 1])
    temp_max = np.max(temp_air_bnd[:, 1])
    print(f"t_min = {t_min} s = {t_min / s_per_day} days = {t_min / s_per_yr} yrs")
    print(f"t_max = {t_max} s = {t_max / s_per_day} days = {t_max / s_per_yr} yrs")
    print(f"temp_min = {temp_min} deg C")
    print(f"temp_max = {temp_max} deg C")

    def air_temp(t):
        return air_temp_spline([t])

    # print()
    # print("Plotting sample of Tair boundary function:")
    # t_plt = np.linspace(0.0, 365.0, 5 * 365 + 1) * s_per_day
    # T_bnd_plt = np.array([air_temp(t) for t in t_plt])
    # plt.figure()
    # plt.plot(t_plt / s_per_day, T_bnd_plt, "-r")
    # plt.show()

    # create boundary conditions
    print()
    print("Applying boundary conditions:")
    e_cu0 = mtl.void_ratio_0_comp
    Ccu = mtl.comp_index_unfrozen
    sig_cu0 = mtl.eff_stress_0_comp
    sig_p_ob = qi
    e_bnd = e_cu0 - Ccu * np.log10(sig_p_ob / sig_cu0)
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_function = air_temp
    print(f"temp_boundary @ z = {temp_boundary.nodes[0].z}")
    print(f"temp_boundary.bnd_type: {temp_boundary.bnd_type}")
    print(f"temp_boundary.bnd_function: {temp_boundary.bnd_function}")
    grad_boundary = ThermalBoundary1D(
        (ta.nodes[-1],),
        (ta.elements[-1].int_pts[-1],),
    )
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = dTdZ_G
    print(f"grad_boundary @ z = {grad_boundary.nodes[0].z}")
    print(f"grad_boundary.bnd_type: {grad_boundary.bnd_type}")
    print(f"grad_boundary.bnd_value: {grad_boundary.bnd_value}")
    void_ratio_bound = ConsolidationBoundary1D(
        nodes=(ta.nodes[0],),
        bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
        bnd_value=e_bnd,
        bnd_value_1=sig_p_ob,
    )
    print(f"void_ratio_bound @ z = {void_ratio_bound.nodes[0].z}")
    print(f"void_ratio_bound.bnd_type: {void_ratio_bound.bnd_type}")
    print(f"void_ratio_bound.bnd_value: {void_ratio_bound.bnd_value}")
    print(f"void_ratio_bound.bnd_value_1: {void_ratio_bound.bnd_value_1}")

    # assign boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)
    ta.add_boundary(void_ratio_bound)

    # initialize output variables
    print()
    print("Initializing output data:")
    z_vec = np.array([nd.z for nd in ta.nodes])
    settle_out = np.zeros_like(t_plot_targ)
    zdef_nd_out = np.zeros((ta.num_nodes, len(t_plot_targ)))
    void_nd_out = np.zeros((ta.num_nodes, len(t_plot_targ)))
    temp_nd_out = np.zeros((ta.num_nodes, len(t_plot_targ)))

    # initialize/test output files
    generate_output(
        fname,
        z_vec,
        t_plot_targ,
        temp_nd_out,
        void_nd_out,
        zdef_nd_out,
        settle_out,
    )

    # initialize global matrices and vectors
    print()
    print("Initializing time stepping:")
    ta.time_step = dt_sim_0  # set initial time step small for adaptive
    ta.implicit_error_tolerance = tol
    print(f"dt={ta.time_step / s_per_day:0.5e} days")
    print(f"adapt_dt={adapt_dt}")
    print(f"tol={ta.implicit_error_tolerance:0.4e}")
    ta.initialize_global_system(t0=0.0)

    # begin spinup cycles
    print()
    print("*** Begin historical cycles ***")
    for k, tf in enumerate(t_plot_targ):
        # save initial state to output variables
        if not k:
            temp_nd_out[:, 0] = ta._temp_vector[:]
            void_nd_out[:, 0] = ta._void_ratio_vector[:]
            settle_out[0] = ta.calculate_total_settlement()
            for k_nd, nd in enumerate(ta.nodes):
                zdef_nd_out[k_nd, 0] = nd.z_def
            # print statistics and generate output data
            Tmax = np.max(temp_nd_out[:, k])
            Tmin = np.min(temp_nd_out[:, k])
            Tavg = np.mean(temp_nd_out[:, k])
            emax = np.max(void_nd_out[:, k])
            emin = np.min(void_nd_out[:, k])
            print(
                f"t = {ta._t1 / s_per_wk: 0.3f} wks, "
                + f"Tmin = {Tmin: 0.4f} deg C, "
                + f"Tmax = {Tmax: 0.4f} deg C, "
                + f"Tavg = {Tavg: 0.4f} deg C, "
                + f"emin = {emin: 0.4f}, "
                + f"emax = {emax: 0.4f}, "
                + f"dt = {ta.time_step / s_per_day:0.4e} days"
            )
            continue
        # perform adaptive time stepping to next target time
        dt00 = ta.solve_to(tf, adapt_dt=adapt_dt)[0]
        temp_nd_out[:, k] = ta._temp_vector[:]
        void_nd_out[:, k] = ta._void_ratio_vector[:]
        settle_out[k] = ta.calculate_total_settlement()
        for k_nd, nd in enumerate(ta.nodes):
            zdef_nd_out[k_nd, k] = nd.z_def
        # print statistics and generate output data
        Tmax = np.max(temp_nd_out[:, k])
        Tmin = np.min(temp_nd_out[:, k])
        Tavg = np.mean(temp_nd_out[:, k])
        emax = np.max(void_nd_out[:, k])
        emin = np.min(void_nd_out[:, k])
        print(
            f"t = {ta._t1 / s_per_wk: 0.3f} wks, "
            + f"Tmin = {Tmin: 0.4f} deg C, "
            + f"Tmax = {Tmax: 0.4f} deg C, "
            + f"Tavg = {Tavg: 0.4f} deg C, "
            + f"emin = {emin: 0.4f}, "
            + f"emax = {emax: 0.4f}, "
            + f"dt = {dt00 / s_per_day:0.4e} days"
        )
        generate_output(
            fname,
            z_vec,
            t_plot_targ,
            temp_nd_out,
            void_nd_out,
            zdef_nd_out,
            settle_out,
        )

    print()
    print("Finalizing output data files:")
    generate_output(
        fname,
        z_vec,
        t_plot_targ,
        temp_nd_out,
        void_nd_out,
        zdef_nd_out,
        settle_out,
    )


def generate_output(
    fname,
    z_vec,
    t_plot_targ,
    temp_nd_out,
    void_nd_out,
    zdef_nd_out,
    settle_out,
):
    np.savetxt(
        fname + "temp.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(z_vec).reshape((len(z_vec), 1))]),
                np.vstack([t_plot_targ / s_per_yr, temp_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    np.savetxt(
        fname + "void.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(z_vec).reshape((len(z_vec), 1))]),
                np.vstack([t_plot_targ / s_per_yr, void_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    np.savetxt(
        fname + "zdef.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(z_vec).reshape((len(z_vec), 1))]),
                np.vstack([t_plot_targ / s_per_yr, zdef_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    np.savetxt(
        fname + "settle.out",
        np.hstack(
            [
                np.vstack([t_plot_targ / s_per_yr, settle_out]),
            ]
        ),
        fmt="%.16e",
    )


if __name__ == "__main__":
    main()
