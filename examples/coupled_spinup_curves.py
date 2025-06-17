import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from frozen_ground_fem import (
    unit_weight_water,
    Material,
    ThermalBoundary1D,
    # HydraulicBoundary1D,
    ConsolidationBoundary1D,
    CoupledAnalysis1D,
)


def main():
    # setup and generate mesh
    print("Generating mesh:")
    ta = CoupledAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 50.0
    print(f"z_min={ta.z_min:0.4f}, z_max={ta.z_max:0.4f}")
    # H_layer = ta.z_max - ta.z_min
    dTdZ_G = 0.03
    k_cycle_list = [0, 5, 10, 25, 50, 100, 150, 200, 250]
    msh_z_list = [0.0, 0.5, 1.0, 2.0, 5.0, 25.0, 50.0]
    msh_dz_list = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    nd_z_list = [5.0, 10.0, 15.0, 20.0, 25.0, 50.0]
    nd_z_line_types = ["-r", "--r", "-b", "--b", "-k", "--k"]
    nd_z_line_width = [2.0, 2.0, 1.5, 1.5, 1.0, 1.0]
    num_el_list = []
    z_msh_nod = []
    for k, (z0, dz) in enumerate(zip(msh_z_list[:-1], msh_dz_list)):
        z1 = msh_z_list[k + 1]
        num_el_list.append(int((z1 - z0) // dz))
        z_msh_nod = np.hstack(
            [z_msh_nod, np.linspace(z0, z1, num_el_list[-1] + 1)[:-1]]
        )
    z_msh_nod = np.hstack([z_msh_nod, msh_z_list[-1]])
    nd_ind_list = [int(np.nonzero(z_msh_nod >= z)[0][0]) for z in nd_z_list]
    num_el = int(np.sum(num_el_list))
    print(f"num_el={num_el}")
    ta.generate_mesh(num_elements=num_el, order=1)
    for nd, zn in zip(ta.nodes, z_msh_nod):
        nd.z = zn

    # define plotting time increments
    s_per_day = 8.64e4
    s_per_yr = s_per_day * 365.0
    t_plot_targ = np.linspace(0.0, 250.0, 251) * s_per_yr

    # define analysis parameters
    dt_sim_0 = 0.5 * s_per_day
    adapt_dt = False
    qi = 15.0e3
    tol = 1e-4
    tol_str = f"{tol:0.1e}"
    tol_str = "p".join(tol_str.split("."))
    fname = "examples/" + f"coupled_spinup_sat_warm_{ta.num_elements}_{tol_str}_"
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
        void_ratio_lim=0.1,
        void_ratio_min=0.1,
        void_ratio_tr=1.6,
        void_ratio_sep=1.6,
        void_ratio_0_comp=0.333333,
        eff_stress_0_comp=1.5e4,
        comp_index_unfrozen=0.05,
        rebound_index_unfrozen=0.005,
        comp_index_frozen_a1=0.021,
        comp_index_frozen_a2=0.01,
        comp_index_frozen_a3=0.23,
    )

    # assign material properties to elements
    for e in ta.elements:
        e.assign_material(mtl)

    # calculate initial void ratio profile
    # e0 = calculate_static_profile(z_nod, mtl, qi)[0]
    # for zz, ee in zip(z_nod, e0):
    #     print(f"{zz: 0.4f}  {ee: 0.4f}")

    # set initial conditions
    print()
    print("Setting initial conditions:")
    T0 = -1.0
    print(f"T0={T0:0.4f}")
    e_cu0 = mtl.void_ratio_0_comp
    Ccu = mtl.comp_index_unfrozen
    sig_cu0 = mtl.eff_stress_0_comp
    sig_p_ob = qi
    e_bnd = e_cu0 - Ccu * np.log10(sig_p_ob / sig_cu0)
    print(f"sig_p_ob={sig_p_ob:0.4f}, e_bnd={e_bnd:0.4f}")
    print(f"dTdZ_G={dTdZ_G:0.4f}")
    for k, nd in enumerate(ta.nodes):
        nd.temp = T0 - dTdZ_G * (ta.z_max - nd.z)
        nd.void_ratio = e_bnd
        nd.void_ratio_0 = e_bnd
        # nd.void_ratio = e0[k]
        # nd.void_ratio_0 = e0[k]

    ## define temperature boundary curve
    # T_data = np.loadtxt(
    #     "examples/en_climate_daily_mean_QC_7103536_1994_P1D.csv", delimiter=","
    # )[:, 1]
    # t_data = np.arange(0, 365) * s_per_day
    #
    # def air_temp(t):
    #     return np.interp(t, t_data, T_data, period=365.0 * s_per_day)

    print()
    print("Defining Tair boundary function:")
    Tavg = -3.89
    Tamp = 19.1
    t_phs = 210 * s_per_day
    print(f"Tavg={Tavg:0.4f}, Tamp={Tamp:0.4f}, t_phs={t_phs / s_per_day:0.4f}")

    def air_temp(t):
        return Tavg + Tamp * np.cos((2 * np.pi / s_per_yr) * (t - t_phs))

    # save a plot of the air temperature boundary
    print()
    print("Generating boundary plot:")
    plt.rc("font", size=10)
    plt.figure(figsize=(6, 4))
    t = np.linspace(0, 2.5 * s_per_yr, 251)
    plt.plot(t / s_per_day, air_temp(t), "-k")
    plt.xlabel("time [days]")
    plt.ylabel("air temp [deg C]")
    print(fname + "boundary.svg")
    plt.savefig(fname + "boundary.svg")
    print(fname + "boundary.out")
    np.savetxt(
        fname + "boundary.out", np.vstack([t / s_per_day, air_temp(t)]).T, fmt="%.16e"
    )

    # create boundary conditions
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_function = air_temp
    grad_boundary = ThermalBoundary1D(
        (ta.nodes[-1],),
        (ta.elements[-1].int_pts[-1],),
    )
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = dTdZ_G
    # hyd_bound = HydraulicBoundary1D(
    #     nodes=(ta.nodes[0],),
    #     bnd_value=H_layer,
    # )
    void_ratio_bound = ConsolidationBoundary1D(
        nodes=(ta.nodes[0],),
        bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
        bnd_value=e_bnd,
        bnd_value_1=sig_p_ob,
    )

    # assign boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)
    # ta.add_boundary(hyd_bound)
    ta.add_boundary(void_ratio_bound)

    # initialize global matrices and vectors
    print()
    print("Initializing time stepping:")
    ta.time_step = dt_sim_0  # set initial time step small for adaptive
    ta.implicit_error_tolerance = tol
    print(f"dt={ta.time_step / s_per_day:0.5e} days")
    print(f"adapt_dt={adapt_dt}")
    print(f"tol={ta.implicit_error_tolerance:0.4e}")
    ta.initialize_global_system(t0=0.0)

    # initialize output variables
    z_vec = np.array([nd.z for nd in ta.nodes])
    temp_curve = np.zeros(ta.num_nodes)
    void_curve = np.zeros(ta.num_nodes)
    temp_nd_out = np.zeros((len(nd_z_list), len(t_plot_targ)))
    dT_nd_out = np.zeros((len(nd_z_list), len(t_plot_targ)))
    void_nd_out = np.zeros((len(nd_z_list), len(t_plot_targ)))
    de_nd_out = np.zeros((len(nd_z_list), len(t_plot_targ)))
    temp_prof_out = np.zeros((ta.num_nodes, len(k_cycle_list)))
    void_prof_out = np.zeros((ta.num_nodes, len(k_cycle_list)))
    temp_curve_annual = np.zeros((ta.num_nodes, 53))
    void_curve_annual = np.zeros((ta.num_nodes, 53))
    temp_min_curve = np.amin(temp_curve_annual, axis=1)
    temp_max_curve = np.amax(temp_curve_annual, axis=1)
    void_min_curve = np.amin(void_curve_annual, axis=1)
    void_max_curve = np.amax(void_curve_annual, axis=1)

    # initialize/test output files
    print()
    print("Test generating output files:")
    print(fname + "cycle_profiles_temp.out")
    np.savetxt(
        fname + "cycle_profiles_temp.out",
        np.hstack([z_vec.reshape((len(z_vec), 1)), temp_prof_out]),
        fmt="%.16e",
    )
    print(fname + "cycle_profiles_void.out")
    np.savetxt(
        fname + "cycle_profiles_void.out",
        np.hstack([z_vec.reshape((len(z_vec), 1)), void_prof_out]),
        fmt="%.16e",
    )
    print(fname + "convergence_temp.out")
    np.savetxt(
        fname + "convergence_temp.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, temp_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_dT.out")
    np.savetxt(
        fname + "convergence_dT.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, dT_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_void.out")
    np.savetxt(
        fname + "convergence_void.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, void_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_de.out")
    np.savetxt(
        fname + "convergence_de.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, de_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "temp_trumpet_curves.out")
    np.savetxt(
        fname + "temp_trumpet_curves.out",
        np.hstack(
            [
                np.array(z_vec).reshape((temp_curve_annual.shape[0], 1)),
                temp_curve_annual,
                temp_min_curve.reshape((temp_curve_annual.shape[0], 1)),
                temp_max_curve.reshape((temp_curve_annual.shape[0], 1)),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "void_trumpet_curves.out")
    np.savetxt(
        fname + "void_trumpet_curves.out",
        np.hstack(
            [
                np.array(z_vec).reshape((void_curve_annual.shape[0], 1)),
                void_curve_annual,
                void_min_curve.reshape((void_curve_annual.shape[0], 1)),
                void_max_curve.reshape((void_curve_annual.shape[0], 1)),
            ]
        ),
        fmt="%.16e",
    )

    # begin spinup cycles
    print()
    print("*** Begin spinup cycles ***")
    k_cycle = 0
    for k, tf in enumerate(t_plot_targ):
        # save initial state to output variables
        if not k:
            temp_curve[:] = ta._temp_vector[:]
            void_curve[:] = ta._void_ratio_vector[:]
            temp_prof_out[:, k_cycle] = temp_curve[:]
            void_prof_out[:, k_cycle] = void_curve[:]
            for nd_k, nd_ind in enumerate(nd_ind_list):
                temp_nd_out[nd_k, k] = temp_curve[nd_ind]
                void_nd_out[nd_k, k] = void_curve[nd_ind]
            k_cycle += 1
            continue
        # perform adaptive time stepping to next target time
        dt00 = ta.solve_to(tf, adapt_dt=adapt_dt)[0]
        dT = ta._temp_vector[:] - temp_curve[:]
        de = ta._void_ratio_vector[:] - void_curve[:]
        temp_curve[:] = ta._temp_vector[:]
        void_curve[:] = ta._void_ratio_vector[:]
        eps_a_T = np.linalg.norm(dT) / np.linalg.norm(temp_curve)
        eps_a_e = np.linalg.norm(de) / np.linalg.norm(void_curve)
        dTmax = np.max(np.abs(dT))
        demax = np.max(np.abs(de))
        print(
            f"t = {ta._t1 / s_per_yr: 0.1f} years, "
            + f"eps = {np.max([eps_a_T, eps_a_e]):0.4e}, "
            + f"dTmax = {dTmax: 0.4f} deg C, "
            + f"demax = {demax: 0.4f}, "
            + f"dt = {dt00 / s_per_day:0.4e} days"
        )
        for nd_k, nd_ind in enumerate(nd_ind_list):
            temp_nd_out[nd_k, k] = temp_curve[nd_ind]
            dT_nd_out[nd_k, k] = np.abs(dT[nd_ind])
            void_nd_out[nd_k, k] = void_curve[nd_ind]
            de_nd_out[nd_k, k] = np.abs(de[nd_ind])
        if k in k_cycle_list:
            temp_prof_out[:, k_cycle] = temp_curve[:]
            void_prof_out[:, k_cycle] = void_curve[:]
            k_cycle += 1
            print()
            print("Generating partial output:")
            print(fname + "convergence_temp.out")
            np.savetxt(
                fname + "convergence_temp.out",
                np.hstack(
                    [
                        np.vstack(
                            [0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]
                        ),
                        np.vstack([t_plot_targ / s_per_yr, temp_nd_out]),
                    ]
                ),
                fmt="%.16e",
            )
            print(fname + "convergence_dT.out")
            np.savetxt(
                fname + "convergence_dT.out",
                np.hstack(
                    [
                        np.vstack(
                            [0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]
                        ),
                        np.vstack([t_plot_targ / s_per_yr, dT_nd_out]),
                    ]
                ),
                fmt="%.16e",
            )
            print(fname + "convergence_void.out")
            np.savetxt(
                fname + "convergence_void.out",
                np.hstack(
                    [
                        np.vstack(
                            [0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]
                        ),
                        np.vstack([t_plot_targ / s_per_yr, void_nd_out]),
                    ]
                ),
                fmt="%.16e",
            )
            print(fname + "convergence_de.out")
            np.savetxt(
                fname + "convergence_de.out",
                np.hstack(
                    [
                        np.vstack(
                            [0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]
                        ),
                        np.vstack([t_plot_targ / s_per_yr, de_nd_out]),
                    ]
                ),
                fmt="%.16e",
            )
            print(fname + "cycle_profiles_temp.out")
            np.savetxt(
                fname + "cycle_profiles_temp.out",
                np.hstack([z_vec.reshape((len(z_vec), 1)), temp_prof_out]),
                fmt="%.16e",
            )
            print(fname + "cycle_profiles_void.out")
            np.savetxt(
                fname + "cycle_profiles_void.out",
                np.hstack([z_vec.reshape((len(z_vec), 1)), void_prof_out]),
                fmt="%.16e",
            )
            if k != k_cycle_list[-1]:
                print()
                print("*** Continue spinup cycles ***")

    # generate temperature and void ratio distribution plots
    print()
    print("Generating complete output:")
    plt.figure(figsize=(8.0, 3.7))
    plt.subplot(1, 3, 1)
    for ind_cycle, k_cycle in enumerate(k_cycle_list):
        plt.plot(temp_prof_out[:, ind_cycle], z_vec, "--k", linewidth=0.5)
    plt.plot(temp_prof_out[:, -1], z_vec, "-k", linewidth=2.0)
    plt.ylim(ta.z_max, ta.z_min)
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.subplot(1, 3, 2)
    for ind_cycle, k_cycle in enumerate(k_cycle_list):
        plt.plot(void_prof_out[:, ind_cycle], z_vec, "--k", linewidth=0.5)
    plt.plot(void_prof_out[:, -1], z_vec, "-k", linewidth=2.0)
    plt.ylim(ta.z_max, ta.z_min)
    plt.xlabel("Void ratio, e [-]")
    # plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    print(fname + "cycle_profiles.svg")
    plt.savefig(fname + "cycle_profiles.svg")
    print(fname + "cycle_profiles_temp.out")
    np.savetxt(
        fname + "cycle_profiles_temp.out",
        np.hstack([z_vec.reshape((len(z_vec), 1)), temp_prof_out]),
        fmt="%.16e",
    )
    print(fname + "cycle_profiles_void.out")
    np.savetxt(
        fname + "cycle_profiles_void.out",
        np.hstack([z_vec.reshape((len(z_vec), 1)), void_prof_out]),
        fmt="%.16e",
    )

    # convergence plots
    fig = plt.figure(figsize=(16.0, 6.0))
    plt.subplot(2, 3, 1)
    for nd_k, temp_nd in enumerate(temp_nd_out):
        plt.plot(
            t_plot_targ / s_per_yr,
            temp_nd,
            nd_z_line_types[nd_k],
            linewidth=nd_z_line_width[nd_k],
            label=f"z={nd_z_list[nd_k]:0.1f}m",
        )
    plt.ylabel("Temperature, T [deg C]")
    plt.ylim((-16.0, 2.0))
    fig.legend(loc="upper right")
    plt.subplot(2, 3, 2)
    for nd_k, dT_nd in enumerate(dT_nd_out):
        plt.plot(
            t_plot_targ / s_per_yr,
            dT_nd,
            nd_z_line_types[nd_k],
            linewidth=nd_z_line_width[nd_k],
        )
    plt.ylabel("Abs Inter Cycle dT [deg C]")
    plt.ylim((0.0, 0.5))
    plt.subplot(2, 3, 4)
    for nd_k, void_nd in enumerate(void_nd_out):
        plt.plot(
            t_plot_targ / s_per_yr,
            void_nd,
            nd_z_line_types[nd_k],
            linewidth=nd_z_line_width[nd_k],
        )
    plt.ylabel("Void Ratio, e [-]")
    plt.xlabel("Number of cycles [yrs]")
    plt.ylim((0.2, 1.0))
    plt.subplot(2, 3, 5)
    for nd_k, de_nd in enumerate(de_nd_out):
        plt.plot(
            t_plot_targ / s_per_yr,
            de_nd,
            nd_z_line_types[nd_k],
            linewidth=nd_z_line_width[nd_k],
        )
    plt.ylabel("Abs Inter Cycle de [-]")
    plt.xlabel("Number of cycles [yrs]")
    plt.ylim((0.0, 0.05))
    print(fname + "convergence.svg")
    plt.savefig(fname + "convergence.svg")
    print(fname + "convergence_temp.out")
    np.savetxt(
        fname + "convergence_temp.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, temp_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_dT.out")
    np.savetxt(
        fname + "convergence_dT.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, dT_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_void.out")
    np.savetxt(
        fname + "convergence_void.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, void_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_de.out")
    np.savetxt(
        fname + "convergence_de.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(nd_z_list).reshape((len(nd_z_list), 1))]),
                np.vstack([t_plot_targ / s_per_yr, de_nd_out]),
            ]
        ),
        fmt="%.16e",
    )

    # now run one annual cycle to obtain temperature envelopes
    print()
    print("*** Run final cycle to obtain annual envelopes ***")
    tf = t_plot_targ[-1] / s_per_yr
    t_plot_targ = np.linspace(tf, tf + 1.0, 53) * s_per_yr
    # temp_curve_annual = np.zeros((ta.num_nodes, 53))
    # void_curve_annual = np.zeros((ta.num_nodes, 53))
    for k, tf in enumerate(t_plot_targ):
        if not k:
            temp_curve_annual[:, 0] = ta._temp_vector[:]
            void_curve_annual[:, 0] = ta._void_ratio_vector[:]
            continue
        dt00 = ta.solve_to(tf, adapt_dt=adapt_dt)[0]
        temp_curve_annual[:, k] = ta._temp_vector[:]
        void_curve_annual[:, k] = ta._void_ratio_vector[:]
        Tmin = np.min(temp_curve_annual[:, k])
        Tmax = np.max(temp_curve_annual[:, k])
        Tmean = np.mean(temp_curve_annual[:, k])
        print(
            f"wk = {k}, "
            + f"dt = {dt00 / s_per_day:0.4e} days, "
            + f"Tmin = {Tmin: 0.4f} deg C, "
            + f"Tmax = {Tmax: 0.4f} deg C, "
            + f"Tmean = {Tmean: 0.4f} deg C"
        )

    print()
    print("Generating annual envelope output:")
    fig = plt.figure(figsize=(8.0, 3.7))
    plt.subplot(1, 3, 1)
    temp_min_curve = np.amin(temp_curve_annual, axis=1)
    temp_max_curve = np.amax(temp_curve_annual, axis=1)
    for k in range(53):
        plt.plot(temp_curve_annual[:, k], z_vec, "-r", linewidth=0.5)
    plt.plot(temp_min_curve, z_vec, "-b", linewidth=2, label="annual minimum")
    plt.plot(temp_max_curve, z_vec, "-r", linewidth=2, label="annual maximum")
    plt.ylim(ta.z_max, ta.z_min)
    fig.legend(loc="upper right")
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.subplot(1, 3, 2)
    void_min_curve = np.amin(void_curve_annual, axis=1)
    void_max_curve = np.amax(void_curve_annual, axis=1)
    for k in range(53):
        plt.plot(void_curve_annual[:, k], z_vec, "-k", linewidth=0.5)
    plt.plot(void_min_curve, z_vec, "-b", linewidth=2, label="annual minimum")
    plt.plot(void_max_curve, z_vec, "-r", linewidth=2, label="annual maximum")
    plt.ylim(ta.z_max, ta.z_min)
    # plt.legend()
    plt.xlabel("Void ratio, e [-]")
    # plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    print(fname + "temp_void_trumpet_curves.svg")
    plt.savefig(fname + "temp_void_trumpet_curves.svg")
    print(fname + "temp_trumpet_curves.out")
    np.savetxt(
        fname + "temp_trumpet_curves.out",
        np.hstack(
            [
                np.array(z_vec).reshape((temp_curve_annual.shape[0], 1)),
                temp_curve_annual,
                temp_min_curve.reshape((temp_curve_annual.shape[0], 1)),
                temp_max_curve.reshape((temp_curve_annual.shape[0], 1)),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "void_trumpet_curves.out")
    np.savetxt(
        fname + "void_trumpet_curves.out",
        np.hstack(
            [
                np.array(z_vec).reshape((void_curve_annual.shape[0], 1)),
                void_curve_annual,
                void_min_curve.reshape((void_curve_annual.shape[0], 1)),
                void_max_curve.reshape((void_curve_annual.shape[0], 1)),
            ]
        ),
        fmt="%.16e",
    )


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
    hyd_cond_0 = m.hyd_cond_0 * 10 ** (
        (e0 - m.void_ratio_0_hyd_cond) / m.hyd_cond_index
    )
    hyd_cond_1 = m.hyd_cond_0 * 10 ** (
        (e1 - m.void_ratio_0_hyd_cond) / m.hyd_cond_index
    )
    return (
        e0,
        sig_p_0,
        hyd_cond_0,
        e1,
        sig_p_1,
        hyd_cond_1,
    )


if __name__ == "__main__":
    main()
