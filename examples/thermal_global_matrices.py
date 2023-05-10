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
    # define mesh with 10 nodes
    mesh = Mesh1D()
    mesh.z_min = -8.0
    mesh.z_max = 100.0
    mesh.generate_mesh(num_nodes=10)

    # define material properties
    # and initialize integration points
    mtl = Material(
        thrm_cond_solids=7.0, spec_grav_solids=2.65, spec_heat_cap_solids=741
    )
    void_ratio = 0.3
    deg_sat_water = 0.8
    for e in mesh.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.void_ratio = void_ratio
            ip.deg_sat_water = deg_sat_water

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

    # creating thermal analysis object
    # and initialize global matrices
    thermal_analysis = ThermalAnalysis1D(mesh)
    thermal_analysis.update_heat_flow_matrix()
    thermal_analysis.update_heat_storage_matrix()

    # create thermal boundary conditions
    temp_boundary = ThermalBoundary1D(upper_boundary)
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = -10.0
    grad_boundary = ThermalBoundary1D(lower_boundary)
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 25.0 / 1e3

    # assign thermal boundaries to the analysis
    thermal_analysis.add_boundary(temp_boundary)
    thermal_analysis.add_boundary(grad_boundary)

    # update boundary conditions
    thermal_analysis.update_thermal_boundary_conditions(time=0.0)
    thermal_analysis.update_heat_flux_vector()

    plt.figure(figsize=(7, 9))

    plt.subplot(4, 1, 1)
    plt.imshow(thermal_analysis._heat_flow_matrix)
    plt.colorbar()
    plt.title("Global Heat Flow Matrix")

    plt.subplot(4, 1, 2)
    plt.imshow(thermal_analysis._heat_storage_matrix)
    plt.colorbar()
    plt.title("Global Heat Storage Matrix")

    plt.subplot(4, 1, 3)
    plt.imshow(thermal_analysis._heat_flux_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Heat Flux Vector")

    plt.subplot(4, 1, 4)
    plt.imshow(thermal_analysis._temp_vector.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Temperature Field")

    plt.savefig("examples/thermal_global_matrices.png")


if __name__ == "__main__":
    main()
