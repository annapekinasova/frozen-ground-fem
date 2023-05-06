import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.geometry import (
    Boundary1D,
    Mesh1D,
)

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
)


def main():
    # define mesh
    mesh = Mesh1D()
    mesh.z_min = 0.0
    mesh.z_max = 100.0
    mesh.generate_mesh(num_nodes=20)

    # define material properties
    # and initialize integration point porosity
    mtl = Material(thrm_cond_solids=7.0, dens_solids=2.65e3, spec_heat_cap_solids=741)
    por = 0.3
    for e in mesh.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.porosity = por

    # create geometric boundaries
    # and assign them to the mesh
    upper_boundary = Boundary1D(
        (mesh.nodes[0],),
        (mesh.elements[0].int_pts[0],),
    )
    lower_boundary = Boundary1D(
        (mesh.nodes[-1],),
        (mesh.elements[-1].int_pts[-1],),
    )
    mesh.add_boundary(upper_boundary)
    mesh.add_boundary(lower_boundary)

    # create thermal analysis object
    thermal_analysis = ThermalAnalysis1D(mesh)
    ta = thermal_analysis

    # set initial temperature conditions
    T0 = 5.0
    for nd in mesh.nodes:
        nd.temp = T0

    # create thermal boundary conditions
    temp_boundary = ThermalBoundary1D(upper_boundary)
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = -10.0
    grad_boundary = ThermalBoundary1D(lower_boundary)
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 0.2

    # assign thermal boundaries to the analysis
    thermal_analysis.add_boundary(temp_boundary)
    thermal_analysis.add_boundary(grad_boundary)

    # **********************************************
    # TIME STEPPING ALGORITHM
    # **********************************************

    # initialize plot
    z_vec = np.array([nd.z for nd in mesh.nodes])
    plt.figure(figsize=(6, 9))

    # initialize global matrices and vectors
    ta.time_step = 8.64e4 * 7  # one week, in seconds
    thermal_analysis.initialize_global_system(t0=0.0)

    for k in range(1500):
        # generate temperature distribution plot
        if not k % 105:
            plt.plot(
                ta._temp_vector, z_vec, label=f"t={ta._t1 / 8.64e4 / 365: 0.2f} years"
            )
        ta.initialize_time_step()
        ta.iterative_correction_step()

    # finalize plot labels
    plt.ylim(mesh.z_max, mesh.z_min)
    plt.legend()
    plt.xlabel("temperature, T [deg C]")
    plt.ylabel("depth, z [m]")
    plt.savefig("examples/thermal_steady_state.png")


if __name__ == "__main__":
    main()
