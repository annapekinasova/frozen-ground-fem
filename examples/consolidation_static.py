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
        z_range=[0.0, 0.2],
        num_nodes=201,
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
    lower_boundary = geom.Boundary1D(
        (mesh.nodes[-1],),
        (mesh.elements[-1].int_pts[-1],),
    )
    mesh.add_boundary(upper_boundary)
    mesh.add_boundary(lower_boundary)

    # create consolidation analysis
    con_static = consol.ConsolidationAnalysis1D(mesh)

    # load expected output
    exp_data = np.loadtxt(fname="examples/consolidation_static_data.csv",
                          delimiter=",",
                          skiprows=1)
    z_exp = exp_data[:, 0]
    e0 = exp_data[:, 1]
    sig_p_0_exp = 1.0e-03 * exp_data[:, 2]
    hyd_cond_0_exp = exp_data[:, 3]

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
    void_ratio_boundary_0 = consol.ConsolidationBoundary1D(upper_boundary)
    void_ratio_boundary_0.bnd_type = (
        consol.ConsolidationBoundary1D.BoundaryType.void_ratio
    )
    void_ratio_boundary_0.bnd_value = e0[0]
    void_ratio_boundary_1 = consol.ConsolidationBoundary1D(lower_boundary)
    void_ratio_boundary_1.bnd_type = (
        consol.ConsolidationBoundary1D.BoundaryType.void_ratio
    )
    void_ratio_boundary_1.bnd_value = e0[-1]
    con_static.add_boundary(void_ratio_boundary_0)
    con_static.add_boundary(void_ratio_boundary_1)

    # set plotting defaults
    plt.rc("font", size=8)
    ms = 1.0    # marker size

    # initialize plot
    z_vec = np.array([nd.z for nd in mesh.nodes])

    # initialize global matrices and vectors
    # con_static.time_step = 365 * 8.64e4 / 52  # ~one week, in seconds
    con_static.time_step = 0.5  # in seconds
    con_static.initialize_global_system(t0=0.0)

    plt.figure(figsize=(7, 9))

    plt.subplot(4, 2, 1)
    plt.imshow(con_static._stiffness_matrix)
    plt.colorbar()
    plt.title("Global Stiffness Matrix")

    plt.subplot(4, 2, 2)
    plt.imshow(con_static._mass_matrix)
    plt.colorbar()
    plt.title("Global Mass Matrix")

    plt.subplot(4, 2, 3)
    plt.imshow(con_static._coef_matrix_0)
    plt.colorbar()
    plt.title("Coefficient Matrix 0")

    plt.subplot(4, 2, 4)
    plt.imshow(con_static._coef_matrix_1)
    plt.colorbar()
    plt.title("Coefficient Matrix 1")

    plt.subplot2grid((4, 2), (2, 0), colspan=2)
    # plt.subplot(4, 2, [5, 6])
    plt.imshow(con_static._water_flux_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Water Flux Vector")

    plt.subplot2grid((4, 2), (3, 0), colspan=2)
    # plt.subplot(4, 2, [7, 8])
    plt.imshow(con_static._void_ratio_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Void Ratio Field")

    plt.savefig("examples/consolidation_static_global_matrices.svg")

    # stabilize at initial void ratio profile
    # (no settlement expected during this period)
    t_con = [0.0]
    s_con = [0.0]
    k = 0
    print("initial stabilization")
    while con_static._t1 < 64.0 * 60.0:
        if not k % 120:
            print(f"t = {con_static._t1 / 60.0:0.3f} min")
        con_static.initialize_time_step()
        con_static.iterative_correction_step()
        t_con.append(con_static._t1)
        s_con.append(con_static.calculate_total_settlement())
        k += 1

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

    # # update boundary conditions
    # # (this corresponds to an application of
    # # a surface load of ~200 kPa)
    # void_ratio_boundary.bnd_value = e_fin_exp[0]
    #
    # print("apply static load of 200 kPa")
    # for k in range(2000):
    #     con_static.initialize_time_step()
    #     con_static.iterative_correction_step()
    #     t_con.append(con_static._t1)
    #     s_con.append(con_static.calculate_total_settlement())
    #     if not k % 100:
    #         print(f"t = {con_static._t1 / 60.0:0.3f} min")

    plt.figure(figsize=(7, 9))

    plt.subplot(4, 2, 1)
    plt.imshow(con_static._stiffness_matrix)
    plt.colorbar()
    plt.title("Global Stiffness Matrix")

    plt.subplot(4, 2, 2)
    plt.imshow(con_static._mass_matrix)
    plt.colorbar()
    plt.title("Global Mass Matrix")

    plt.subplot(4, 2, 3)
    plt.imshow(con_static._coef_matrix_0)
    plt.colorbar()
    plt.title("Coefficient Matrix 0")

    plt.subplot(4, 2, 4)
    plt.imshow(con_static._coef_matrix_1)
    plt.colorbar()
    plt.title("Coefficient Matrix 1")

    plt.subplot2grid((4, 2), (2, 0), colspan=2)
    # plt.subplot(4, 2, [5, 6])
    plt.imshow(con_static._water_flux_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Water Flux Vector")

    plt.subplot2grid((4, 2), (3, 0), colspan=2)
    # plt.subplot(4, 2, [7, 8])
    plt.imshow(con_static._void_ratio_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Void Ratio Field")

    plt.savefig("examples/consolidation_static_global_matrices_final.png")

    # sig_p_f_int = []
    # qw_f_int = []
    # hyd_cond_f_int = []
    # for e in con_static.elements:
    #     for ip in e.int_pts:
    #         sig_p_f_int.append(ip.eff_stress)
    #         qw_f_int.append(ip.water_flux_rate)
    #         hyd_cond_f_int.append(ip.hyd_cond)
    # sig_p_f_int = np.array(sig_p_f_int) * 1.0e-03
    # qw_f_int = np.array(qw_f_int)
    # hyd_cond_f_int = np.array(hyd_cond_f_int)
    # e_fin_act = con_static._void_ratio_vector.copy()
    
    # convert settlement to arrays
    t_con = np.array(t_con) / 60.0  # convert to min
    s_con = np.array(s_con) * 1.0e03  # conver to mm

    plt.figure(figsize=(4, 4))
    plt.plot(t_con, s_con, "-k")
    plt.xlabel("Time [min]")
    plt.ylabel("Settlement [mm]")

    plt.savefig("examples/con_static_settlement.svg")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(e0, z_exp * 100, "-k", label="exp")
    plt.plot(e0_act, z_vec * 100, "--r", label="act")
    # plt.plot(e_fin_exp, z_exp, "--r", label="exp fin")
    # plt.plot(e_fin_act, z_vec, "or", label="act fin")
    plt.ylim((20, 0))
    plt.legend()
    plt.xlabel("Void Ratio, e")
    plt.ylabel("Depth (Lagrangian coordinate) [cm]")

    plt.subplot(1, 3, 2)
    plt.plot(sig_p_0_exp, z_exp * 100, "-k", label="exp")
    plt.plot(sig_p_0_int, z_int * 100, "--r", label="act")
    # plt.plot(sig_p_fin_exp, z_exp, "--r", label="exp fin")
    # plt.plot(sig_p_f_int, z_int, "or", label="act fin")
    plt.ylim((20, 0))
    plt.legend()
    plt.xlabel("Eff Stress [kPa]")

    plt.subplot(1, 3, 3)
    plt.semilogx(qw_0_int, z_int * 100, ":b", label="water flux")
    # plt.semilogx(qw_f_int, z_int, ":k", label="water flux fin")
    plt.semilogx(hyd_cond_0_exp, z_exp * 100, "-k", label="exp")
    plt.semilogx(hyd_cond_0_int, z_int * 100, "--r", label="act")
    # plt.semilogx(hyd_cond_fin_exp, z_exp, "--r", label="exp fin")
    # plt.semilogx(hyd_cond_f_int, z_int, "or", label="act fin")
    plt.ylim((20, 0))
    plt.legend()
    plt.xlabel("Hyd Cond, k [m/s]")

    plt.savefig("examples/con_static_void_sig_profiles.svg")


if __name__ == "__main__":
    main()
