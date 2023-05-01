import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.geometry import (
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
    mtl = Material(thrm_cond_solids=7.0, dens_solids=2.65e3, spec_heat_cap_solids=741)
    por = 0.3
    vol_ice_cont = 0.05
    for e in mesh.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.porosity = por
            ip.vol_ice_cont = vol_ice_cont

    # creating thermal analysis object
    # and initialize global matrices
    # (Note: A boundary element is created
    #        at the first and last nodes.)
    thermal_analysis = ThermalAnalysis1D(mesh)
    thermal_analysis.update_heat_flow_matrix()
    thermal_analysis.update_heat_storage_matrix()

    # assign boundary conditions
    thermal_analysis._thermal_boundaries[
        0
    ].bnd_type = ThermalBoundary1D.BoundaryType.temp
    thermal_analysis._thermal_boundaries[0].bnd_value = 20.0
    thermal_analysis._thermal_boundaries[
        1
    ].bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    thermal_analysis._thermal_boundaries[1]._bnd_value = 25.0 / 1e3
    for be in thermal_analysis._thermal_boundaries:
        be.update_nodes()
    temp_field = np.zeros(mesh.num_nodes)
    for nd in mesh.nodes:
        temp_field[nd.index] = nd.temp
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
    plt.imshow(temp_field.reshape((1, mesh.num_nodes)))
    plt.colorbar()
    plt.title("Global Temperature Field")

    plt.savefig("examples/global_matrices.png")


if __name__ == "__main__":
    main()
