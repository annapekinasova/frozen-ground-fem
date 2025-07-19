import numpy as np
import numpy.typing as npt

from frozen_ground_fem import (
    unit_weight_water,
    Material,
    ThermalBoundary1D,
    ConsolidationBoundary1D,
    CoupledAnalysis1D,
)

s_per_day = 8.64e4
s_per_yr = s_per_day * 365.0


def main():
    # setup and generate mesh
    print("Generating mesh:")
    ta = CoupledAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 50.0
    print(f"z_min={ta.z_min:0.4f}, z_max={ta.z_max:0.4f}")
    dTdZ_G = 0.03
    k_cycle_list = (
        np.array([0, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250], dtype=int)
        * 8
    )
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
    t_plot_targ = np.linspace(0.0, 250.0, 8 * 250 + 1) * s_per_yr

    # define analysis parameters
    dt_sim_0 = 0.05 * s_per_day
    adapt_dt = True
    qi = 15.0e3
    tol = 1e-4
    tol_str = f"{tol:0.1e}"
    tol_str = "p".join(tol_str.split("."))
    fname = "examples/" + f"coupled_spinup_sat_cold_{ta.num_elements}_{tol_str}_"
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

    # calculate initial void ratio profile
    # e0 = calculate_static_profile(z_nod, mtl, qi)[0]
    # for zz, ee in zip(z_nod, e0):
    #     print(f"{zz: 0.4f}  {ee: 0.4f}")

    # set initial conditions
    print()
    print("Setting initial conditions:")
    T0 = -9.0
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

    print()
    print("Defining Tair boundary function:")
    Tavg = -13.76
    Tamp = 22.6
    t_phs = 210 * s_per_day
    print(f"Tavg={Tavg:0.4f}, Tamp={Tamp:0.4f}, t_phs={t_phs / s_per_day:0.4f}")

    def air_temp(t):
        return Tavg + Tamp * np.cos((2 * np.pi / s_per_yr) * (t - t_phs))

    # save a plot of the air temperature boundary
    print()
    print("Generating air temperature boundary output:")
    t = np.linspace(0, 2.5 * s_per_yr, 251)
    print(fname + "boundary.out")
    np.savetxt(
        fname + "boundary.out", np.vstack([t / s_per_day, air_temp(t)]).T, fmt="%.16e"
    )

    # create boundary conditions
    print()
    print("Applying boundary conditions:")
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
    void_curve = np.zeros((ta.num_nodes, 8))
    temp_curve = np.zeros((ta.num_nodes, 8))
    settle_out = np.zeros_like(t_plot_targ)
    zdef_nd_out = np.zeros((ta.num_nodes, len(t_plot_targ)))
    void_nd_out = np.zeros((ta.num_nodes, len(t_plot_targ)))
    temp_nd_out = np.zeros((ta.num_nodes, len(t_plot_targ)))
    settle_annual = np.zeros(52)
    zdef_curve_annual = np.zeros((ta.num_nodes, 52))
    void_curve_annual = np.zeros((ta.num_nodes, 52))
    temp_curve_annual = np.zeros((ta.num_nodes, 52))

    # initialize/test output files
    generate_output_convergence(
        fname,
        z_vec,
        t_plot_targ,
        temp_nd_out,
        void_nd_out,
        zdef_nd_out,
        settle_out,
    )
    generate_output_annual(
        fname,
        z_vec,
        temp_curve_annual,
        void_curve_annual,
        zdef_curve_annual,
        settle_annual,
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
    print("*** Begin spinup cycles ***")
    k_cycle = 0
    for k, tf in enumerate(t_plot_targ):
        # save initial state to output variables
        if not k:
            temp_curve[:, 0] = ta._temp_vector[:]
            void_curve[:, 0] = ta._void_ratio_vector[:]
            temp_nd_out[:, 0] = temp_curve[:, 0]
            void_nd_out[:, 0] = void_curve[:, 0]
            settle_out[0] = ta.calculate_total_settlement()
            for k_nd, nd in enumerate(ta.nodes):
                zdef_nd_out[k_nd, 0] = nd.z_def
            k_cycle += 1
            continue
        # check target time is valid (necessary after generating annual envelopes)
        if tf <= ta._t1:
            continue
        # perform adaptive time stepping to next target time
        dt00 = ta.solve_to(tf, adapt_dt=adapt_dt)[0]
        # save temp, void, zdef profiles and total settlement
        if k < 8:
            dT = temp_curve[:, k]
            de = void_curve[:, k]
        else:
            dT = ta._temp_vector[:] - temp_curve[:, k % 8]
            de = ta._void_ratio_vector[:] - void_curve[:, k % 8]
        temp_curve[:, k % 8] = temp_nd_out[:, k] = ta._temp_vector[:]
        void_curve[:, k % 8] = void_nd_out[:, k] = ta._void_ratio_vector[:]
        settle_out[k] = ta.calculate_total_settlement()
        for k_nd, nd in enumerate(ta.nodes):
            zdef_nd_out[k_nd, k] = nd.z_def
        # compute error variables
        eps_a_T = np.linalg.norm(dT) / np.linalg.norm(temp_curve[:, k % 8])
        eps_a_e = np.linalg.norm(de) / np.linalg.norm(void_curve[:, k % 8])
        dTmax = np.max(np.abs(dT))
        demax = np.max(np.abs(de))
        print(
            f"t = {ta._t1 / s_per_yr: 0.3f} years, "
            + f"eps = {np.max([eps_a_T, eps_a_e]):0.4e}, "
            + f"dTmax = {dTmax: 0.4f} deg C, "
            + f"demax = {demax: 0.4f}, "
            + f"emax = {np.max(void_curve[:, k % 8]): 0.4f}, "
            + f"emin = {np.min(void_curve[:, k % 8]): 0.4f}, "
            + f"dt = {dt00 / s_per_day:0.4e} days"
        )

        # each complete annual cycle, generate output data
        if not k % 8:
            generate_output_convergence(
                fname,
                z_vec,
                t_plot_targ,
                temp_nd_out,
                void_nd_out,
                zdef_nd_out,
                settle_out,
            )
            if k not in k_cycle_list:
                print()

        # at target cycles, generate annual envelopes at weekly intervals
        if k in k_cycle_list:
            print()
            print("Generating annual envelopes:")
            t_targ_env = (
                np.linspace(tf / s_per_yr, tf / s_per_yr + 1.0, 53)[:-1] * s_per_yr
            )
            for k_ann, tf_ann in enumerate(t_targ_env):
                if not k_ann:
                    temp_curve_annual[:, 0] = ta._temp_vector[:]
                    void_curve_annual[:, 0] = ta._void_ratio_vector[:]
                    settle_annual[0] = ta.calculate_total_settlement()
                    for k_nd, nd in enumerate(ta.nodes):
                        zdef_curve_annual[k_nd, 0] = nd.z_def
                    Tmin = np.min(temp_curve_annual[:, k_ann])
                    Tmax = np.max(temp_curve_annual[:, k_ann])
                    Tmean = np.mean(temp_curve_annual[:, k_ann])
                    print(
                        f"wk = {k_ann}, "
                        + f"t = {ta._t1 / s_per_yr:0.4f} years, "
                        + f"dt = {dt00 / s_per_day:0.4e} days, "
                        + f"Tmin = {Tmin: 0.4f} deg C, "
                        + f"Tmax = {Tmax: 0.4f} deg C, "
                        + f"Tmean = {Tmean: 0.4f} deg C, "
                        + f"s_tot = {settle_annual[0] * 100.0: 0.4f} cm"
                    )
                    continue
                dt00 = ta.solve_to(tf_ann, adapt_dt=adapt_dt)[0]
                temp_curve_annual[:, k_ann] = ta._temp_vector[:]
                void_curve_annual[:, k_ann] = ta._void_ratio_vector[:]
                settle_annual[k_ann] = ta.calculate_total_settlement()
                for k_nd, nd in enumerate(ta.nodes):
                    zdef_curve_annual[k_nd, k_ann] = nd.z_def
                Tmin = np.min(temp_curve_annual[:, k_ann])
                Tmax = np.max(temp_curve_annual[:, k_ann])
                Tmean = np.mean(temp_curve_annual[:, k_ann])
                print(
                    f"wk = {k_ann}, "
                    + f"t = {ta._t1 / s_per_yr:0.4f} years, "
                    + f"dt = {dt00 / s_per_day:0.4e} days, "
                    + f"Tmin = {Tmin: 0.4f} deg C, "
                    + f"Tmax = {Tmax: 0.4f} deg C, "
                    + f"Tmean = {Tmean: 0.4f} deg C, "
                    + f"s_tot = {settle_annual[k_ann] * 100.0: 0.4f} cm"
                )

            generate_output_annual(
                fname,
                z_vec,
                temp_curve_annual,
                void_curve_annual,
                zdef_curve_annual,
                settle_annual,
            )

            if k != k_cycle_list[-1]:
                print()
                print("*** Continue spinup cycles ***")
            else:
                print()
                print("*** Spinup cycles complete ***")

    print()
    print("Finalizing output data files:")
    generate_output_convergence(
        fname,
        z_vec,
        t_plot_targ,
        temp_nd_out,
        void_nd_out,
        zdef_nd_out,
        settle_out,
    )
    generate_output_annual(
        fname,
        z_vec,
        temp_curve_annual,
        void_curve_annual,
        zdef_curve_annual,
        settle_annual,
    )


def generate_output_annual(
    fname,
    z_vec,
    temp_curve_annual,
    void_curve_annual,
    zdef_curve_annual,
    settle_annual,
):
    print()
    print("Generating annual envelope data files:")
    print(fname + "annual_curves_temp.out")
    np.savetxt(
        fname + "annual_curves_temp.out",
        np.hstack(
            [
                np.array(z_vec).reshape((temp_curve_annual.shape[0], 1)),
                temp_curve_annual,
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "annual_curves_void.out")
    np.savetxt(
        fname + "annual_curves_void.out",
        np.hstack(
            [
                np.array(z_vec).reshape((void_curve_annual.shape[0], 1)),
                void_curve_annual,
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "annual_curves_settle_zdef.out")
    np.savetxt(
        fname + "annual_curves_settle_zdef.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(z_vec).reshape((len(z_vec), 1))]),
                np.vstack([settle_annual, zdef_curve_annual]),
            ]
        ),
        fmt="%.16e",
    )


def generate_output_convergence(
    fname,
    z_vec,
    t_plot_targ,
    temp_nd_out,
    void_nd_out,
    zdef_nd_out,
    settle_out,
):
    print()
    print("Generating convergence output data files:")
    print(fname + "convergence_temp.out")
    np.savetxt(
        fname + "convergence_temp.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(z_vec).reshape((len(z_vec), 1))]),
                np.vstack([t_plot_targ / s_per_yr, temp_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_void.out")
    np.savetxt(
        fname + "convergence_void.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(z_vec).reshape((len(z_vec), 1))]),
                np.vstack([t_plot_targ / s_per_yr, void_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_zdef.out")
    np.savetxt(
        fname + "convergence_zdef.out",
        np.hstack(
            [
                np.vstack([0.0, np.array(z_vec).reshape((len(z_vec), 1))]),
                np.vstack([t_plot_targ / s_per_yr, zdef_nd_out]),
            ]
        ),
        fmt="%.16e",
    )
    print(fname + "convergence_settle.out")
    np.savetxt(
        fname + "convergence_settle.out",
        np.hstack(
            [
                np.vstack([t_plot_targ / s_per_yr, settle_out]),
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
