import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from frozen_ground_fem import (
    unit_weight_water,
    Material,
    ThermalBoundary1D,
    HydraulicBoundary1D,
    ConsolidationBoundary1D,
    CoupledAnalysis1D,
)


def main():
    # create analysis object
    ta = CoupledAnalysis1D()
    ta.z_min = 0.0
    ta.z_max = 100.0
    H_layer = ta.z_max - ta.z_min
    ta.generate_mesh(num_elements=80, order=1)
    # ta.generate_mesh(num_elements=315, order=1)

    # compute modified node locations
    num_el_list = [25, 20, 15, 10, 10]
    # num_el_list = [200, 80, 15, 10, 10]
    z_mesh_nod = np.hstack(
        [
            np.linspace(0.0, 2.0, num_el_list[0] + 1)[:-1],
            np.linspace(2.0, 10.0, num_el_list[1] + 1)[:-1],
            np.linspace(10.0, 25.0, num_el_list[2] + 1)[:-1],
            np.linspace(25.0, 50.0, num_el_list[3] + 1)[:-1],
            np.linspace(50.0, 100.0, num_el_list[4] + 1),
        ]
    )
    # modify node locations in the mesh
    for nd, zn in zip(ta.nodes, z_mesh_nod):
        nd.z = zn
    z_nod = np.array([nd.z for nd in ta.nodes])

    # define analysis parameters
    dt_sim_0 = 1.0e-5
    qi = 15.0e3
    tol = 1e-3
    tol_str = f"{tol:0.1e}"
    tol_str = "p".join(tol_str.split("."))
    fname = "examples/" + f"coupled_spinup_{ta.num_elements}_{tol_str}"

    # define plotting time increments
    s_per_day = 8.64e4
    s_per_wk = s_per_day * 7.0
    s_per_yr = s_per_day * 365.0
    t_plot_targ = np.linspace(0.0, 30.0, 61) * s_per_yr

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
    T0 = -0.5
    e_cu0 = mtl.void_ratio_0_comp
    Ccu = mtl.comp_index_unfrozen
    sig_cu0 = mtl.eff_stress_0_comp
    sig_p_ob = qi
    e_bnd = e_cu0 - Ccu * np.log10(sig_p_ob / sig_cu0)
    for k, nd in enumerate(ta.nodes):
        nd.temp = T0
        nd.void_ratio = e_bnd
        nd.void_ratio_0 = e_bnd
        # nd.void_ratio = e0[k]
        # nd.void_ratio_0 = e0[k]

    # define temperature boundary curve
    t_max = 195.0 * s_per_day
    omega = 2.0 * np.pi / s_per_yr
    T_mean = -10.0
    T_amp = 25.0

    def air_temp(t):
        return T_amp * np.cos(omega * (t - t_max)) + T_mean

    # save a plot of the air temperature boundary
    plt.figure(figsize=(6, 4))
    t = np.linspace(0, s_per_yr, 100)
    plt.plot(t / s_per_day, air_temp(t), "-k")
    plt.xlabel("time [days]")
    plt.ylabel("air temp [deg C]")
    plt.savefig(fname + "_boundary.png")

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
    grad_boundary.bnd_value = 0.03
    hyd_bound = HydraulicBoundary1D(
        nodes=(ta.nodes[0],),
        bnd_value=H_layer,
    )
    void_ratio_bound = ConsolidationBoundary1D(
        nodes=(ta.nodes[0],),
        bnd_type=ConsolidationBoundary1D.BoundaryType.void_ratio,
        bnd_value=e_bnd,
        bnd_value_1=sig_p_ob,
    )

    # assign boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)
    ta.add_boundary(hyd_bound)
    ta.add_boundary(void_ratio_bound)

    # **********************************************
    # TIME STEPPING ALGORITHM
    # **********************************************

    plt.rc("font", size=8)

    # initialize plot
    z_vec = np.array([nd.z for nd in ta.nodes])
    plt.figure(figsize=(4.0, 3.7))

    # initialize global matrices and vectors
    ta.time_step = dt_sim_0  # set initial time step small for adaptive
    ta.initialize_global_system(t0=0.0)

    temp_curve = np.zeros((ta.num_nodes, 2))
    void_curve = np.zeros((ta.num_nodes, 2))
    for k, tf in enumerate(t_plot_targ):
        if not k:
            temp_curve[:, 0] = ta._temp_vector[:]
            void_curve[:, 0] = ta._void_ratio_vector[:]
            continue
        dt00 = ta.solve_to(tf)[0]
        dT = ta._temp_vector[:] - temp_curve[:, k % 2]
        de = ta._void_ratio_vector[:] - void_curve[:, k % 2]
        eps_a_T = np.linalg.norm(dT) / np.linalg.norm(ta._temp_vector[:])
        eps_a_e = np.linalg.norm(de) / np.linalg.norm(ta._void_ratio_vector[:])
        dTmax = np.max(np.abs(dT))
        demax = np.max(np.abs(de))
        temp_curve[:, k % 2] = ta._temp_vector[:]
        void_curve[:, k % 2] = ta._void_ratio_vector[:]
        print(
            f"t = {ta._t1 / s_per_yr: 0.5f} years, "
            + f"eps = {np.max([eps_a_T, eps_a_e]):0.4e}, "
            + f"dTmax = {dTmax: 0.4f} deg C, "
            + f"demax = {demax: 0.4f}, "
            + f"dt = {dt00 / s_per_wk:0.4e} wks"
        )
        if not k % 10:
            plt.subplot(1, 2, 1)
            plt.plot(temp_curve[:, 1], z_vec, "--r", linewidth=0.5)
            plt.subplot(1, 2, 2)
            plt.plot(void_curve[:, 1], z_vec, "--r", linewidth=0.5)
        elif not (k - 1) % 10:
            plt.subplot(1, 2, 1)
            plt.plot(temp_curve[:, 0], z_vec, "--b", linewidth=0.5)
            plt.subplot(1, 2, 2)
            plt.plot(void_curve[:, 0], z_vec, "--b", linewidth=0.5)

    # generate converged temperature distribution plot
    plt.subplot(1, 2, 1)
    plt.plot(temp_curve[:, 0], z_vec, "-b", linewidth=2, label="temp dist, jan 1")
    plt.plot(temp_curve[:, 1], z_vec, "-r", linewidth=2, label="temp dist, jul 1")
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.subplot(1, 2, 2)
    plt.plot(void_curve[:, 0], z_vec, "-b", linewidth=2, label="void ratio dist, jan 1")
    plt.plot(void_curve[:, 1], z_vec, "-r", linewidth=2, label="void ratio dist, jul 1")
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("Void ratio, e [-]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig(fname + "_conv_dist.svg")

    # now run one annual cycle to obtain temperature envelopes
    print("running final cycle to obtain temp envelopes")
    t_plot_targ = np.linspace(30.0, 31.0, 53) * s_per_yr
    temp_curve = np.zeros((ta.num_nodes, 53))
    for k, tf in enumerate(t_plot_targ):
        if not k:
            temp_curve[:, 0] = ta._temp_vector[:]
            continue
        dt00 = ta.solve_to(tf)[0]
        temp_curve[:, k] = ta._temp_vector[:]
        Tmin = np.min(temp_curve[:, k])
        Tmax = np.max(temp_curve[:, k])
        Tmean = np.mean(temp_curve[:, k])
        print(
            f"wk = {k}, "
            + f"dt = {dt00 / s_per_wk:0.4e} wks, "
            + f"Tmin = {Tmin: 0.4f} deg C, "
            + f"Tmax = {Tmax: 0.4f} deg C, "
            + f"Tmean = {Tmean: 0.4f} deg C"
        )

    plt.figure(figsize=(3.7, 3.7))
    temp_min_curve = np.amin(temp_curve, axis=1)
    temp_max_curve = np.amax(temp_curve, axis=1)
    plt.plot(temp_curve[:, 0], z_vec, "--b", linewidth=1, label="temp dist, jan 1")
    plt.plot(temp_curve[:, 13], z_vec, ":b", linewidth=1, label="temp dist, apr 1")
    plt.plot(temp_curve[:, 26], z_vec, "--r", linewidth=1, label="temp dist, jul 1")
    plt.plot(temp_curve[:, 39], z_vec, ":r", linewidth=1, label="temp dist, oct 1")
    plt.plot(temp_min_curve, z_vec, "-b", linewidth=2, label="annual minimum")
    plt.plot(temp_max_curve, z_vec, "-r", linewidth=2, label="annual maximum")
    plt.ylim(ta.z_max, ta.z_min)
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")
    plt.ylabel("Depth (Lagrangian coordinate), Z [m]")
    plt.savefig("examples/thermal_trumpet_curves.svg")


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
