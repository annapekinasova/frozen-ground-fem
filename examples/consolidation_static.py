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
    exp_data_0 = np.loadtxt(
        fname="examples/con_static_data_0.csv", delimiter=",", skiprows=1
    )
    e0 = exp_data_0[:, 1]
    sig_p_0_exp = 1.0e-03 * exp_data_0[:, 2]
    hyd_cond_0_exp = exp_data_0[:, 3]
    exp_data_1 = np.loadtxt(
        fname="examples/con_static_data_1.csv", delimiter=",", skiprows=1
    )
    e1 = exp_data_1[:, 1]
    sig_p_1_exp = 1.0e-03 * exp_data_1[:, 2]
    hyd_cond_1_exp = exp_data_1[:, 3]

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

    # set plotting parameters
    plt.rc("font", size=8)
    ms = 1.0  # marker size
    dt_plot = 60.0 * 20      # in seconds
    n_plot = 8

    # initialize plotting arrays
    z_nod = np.array([nd.z for nd in mesh.nodes])
    z_int = np.array([ip.z for e in mesh.elements for ip in e.int_pts])
    e_nod = np.zeros((len(z_nod), n_plot + 1))
    sig_p_int = np.zeros((len(z_int), n_plot + 1))
    hyd_cond_int = np.zeros((len(z_int), n_plot + 1))

    # initialize global matrices and vectors
    con_static.time_step = 120.0  # in seconds
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
    plt.imshow(con_static._water_flux_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Water Flux Vector")

    plt.subplot2grid((4, 2), (3, 0), colspan=2)
    plt.imshow(con_static._void_ratio_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Void Ratio Field")

    plt.savefig("examples/con_static_global_matrices_0.svg")

    # stabilize at initial void ratio profile
    # (no settlement expected during this period)
    t_con = [0.0]
    s_con = [0.0]
    t_plot = dt_plot
    k_plot = 0
    print("initial stabilization")
    while con_static._t1 < t_plot:
        con_static.initialize_time_step()
        con_static.iterative_correction_step()
        t_con.append(con_static._t1)
        s_con.append(con_static.calculate_total_settlement())

    print(f"t = {con_static._t1 / 60.0:0.3f} min")
    e_nod[:, k_plot] = con_static._void_ratio_vector[:]
    sig_p_int[:, k_plot] = 1.0e-3 * np.array([ip.eff_stress for e in mesh.elements for ip in e.int_pts])
    hyd_cond_int[:, k_plot] = np.array([ip.hyd_cond for e in mesh.elements for ip in e.int_pts])

    # update boundary conditions
    # (this corresponds to an application of
    # a surface load increment)
    print("apply static load increment")
    void_ratio_boundary_0.bnd_value = e1[0]
    void_ratio_boundary_1.bnd_value = e1[-1]

    k_plot += 1
    t_plot += dt_plot
    while con_static._t1 < t_plot:
        con_static.initialize_time_step()
        con_static.iterative_correction_step()
        t_con.append(con_static._t1)
        s_con.append(con_static.calculate_total_settlement())
    print(f"t = {con_static._t1 / 60.0:0.3f} min")
    e_nod[:, k_plot] = con_static._void_ratio_vector[:]
    sig_p_int[:, k_plot] = 1.0e-3 * np.array([ip.eff_stress for e in mesh.elements for ip in e.int_pts])
    hyd_cond_int[:, k_plot] = np.array([ip.hyd_cond for e in mesh.elements for ip in e.int_pts])

    while k_plot < n_plot:
        k_plot += 1
        t_plot += 8 * dt_plot
        while con_static._t1 < t_plot:
            con_static.initialize_time_step()
            con_static.iterative_correction_step()
            t_con.append(con_static._t1)
            s_con.append(con_static.calculate_total_settlement())
        print(f"t = {con_static._t1 / 60.0:0.3f} min")
        e_nod[:, k_plot] = con_static._void_ratio_vector[:]
        sig_p_int[:, k_plot] = 1.0e-3 * np.array([ip.eff_stress for e in mesh.elements for ip in e.int_pts])
        hyd_cond_int[:, k_plot] = np.array([ip.hyd_cond for e in mesh.elements for ip in e.int_pts])

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
    plt.imshow(con_static._water_flux_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Water Flux Vector")

    plt.subplot2grid((4, 2), (3, 0), colspan=2)
    plt.imshow(con_static._void_ratio_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Void Ratio Field")

    plt.savefig("examples/con_static_global_matrices_1.svg")

    # convert settlement to arrays
    t_con = np.array(t_con) / 60.0  # convert to min
    s_con = np.array(s_con) * 1.0e03  # convert to mm
    t_lab = np.array([dt_plot, 9*dt_plot, 17*dt_plot]) / 60.0
    e_lab = np.array([1.30, 1.20, 1.15])
    z_lab = np.array([1.25, 5.00, 10.0])

    # convert z to cm
    z_nod *= 100.0
    z_int *= 100.0

    plt.figure(figsize=(3.5, 4))
    plt.plot(np.sqrt(t_con), s_con, "-k")
    plt.xlabel(r"Root Time, $t^{0.5}$ [$min^{0.5}$]")
    plt.ylabel(r"Settlement, $s$ [$mm$]")
    plt.ylim((31, -1))

    plt.savefig("examples/con_static_settlement.svg")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.plot(e0, z_nod, "-k", label="init")
    plt.plot(e_nod[:, 0], z_nod, ":r", label="post-stab")
    plt.plot(e1, z_nod, "--k", label="final (exp)")
    plt.plot(e_nod[:, 1], z_nod, ":b", label="consol (act)")
    plt.text(e_lab[0], z_lab[0],
             f"t = {t_lab[0]:0.0f} min")
    for k_plot in [2, 3, -1]:
        plt.plot(e_nod[:, k_plot], z_nod, ":b")
        if k_plot > 0:
            plt.text(e_lab[k_plot - 1], z_lab[k_plot - 1],
                    f"{t_lab[k_plot - 1]:0.0f}")
    plt.ylim((20, 0))
    plt.xlim((1.0, 1.8))
    plt.legend()
    plt.xlabel(r"Void Ratio, $e$")
    plt.ylabel(r"Depth (Lagrangian), $Z$ [$cm$]")

    plt.subplot(1, 3, 2)
    plt.plot(sig_p_0_exp, z_nod, "-k", label="init")
    plt.plot(sig_p_int[:, 0], z_int, ":r", label="post-stab")
    plt.plot(sig_p_1_exp, z_nod, "--k", label="final (exp)")
    plt.plot(sig_p_int[:, 1], z_int, ":b", label="consol (act)")
    for k_plot in [2, 3, -1]:
        plt.plot(sig_p_int[:, k_plot], z_int, ":b")
    plt.ylim((20, 0))
    # plt.legend()
    plt.xlabel(r"Eff Stress, $\sigma^\prime$ [$kPa$]")

    plt.subplot(1, 3, 3)
    plt.semilogx(hyd_cond_0_exp, z_nod, "-k", label="init")
    plt.semilogx(hyd_cond_int[:, 0], z_int, ":r", label="post-stab")
    plt.semilogx(hyd_cond_1_exp, z_nod, "--k", label="final (exp)")
    plt.semilogx(hyd_cond_int[:, 1], z_int, ":b", label="consol (act)")
    for k_plot in [2, 3, -1]:
        plt.semilogx(hyd_cond_int[:, k_plot], z_int, ":b")
    plt.ylim((20, 0))
    # plt.legend()
    plt.xlabel(r"Hyd Cond, $k$ [$m/s$]")

    plt.savefig("examples/con_static_void_sig_profiles.svg")


if __name__ == "__main__":
    main()
