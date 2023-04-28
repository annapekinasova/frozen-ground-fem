import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.geometry import (
    Mesh1D,
)

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
)


def main():
    mesh = Mesh1D()
    mesh.z_min = -8.0
    mesh.z_max = 100.0
    mesh.generate_mesh(num_nodes=10)

    mtl = Material(7.0, 2.65e3, 741e3)
    por = 0.3
    vol_ice_cont = 0.05
    for e in mesh.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.porosity = por
            ip.vol_ice_cont = vol_ice_cont

    thermal_analysis = ThermalAnalysis1D(mesh)
    thermal_analysis.update_heat_flow_matrix()
    thermal_analysis.update_heat_storage_matrix()

    plt.figure(figsize=(7, 9))

    plt.subplot(2, 1, 1)
    plt.imshow(thermal_analysis._heat_flow_matrix)
    plt.colorbar()
    plt.title("Global Heat Flow Matrix")

    plt.subplot(2, 1, 2)
    plt.imshow(thermal_analysis._heat_storage_matrix)
    plt.colorbar()
    plt.title("Global Heat Storage Matrix")

    plt.savefig("examples/global_matrices.png")


if __name__ == "__main__":
    main()
