import matplotlib.pyplot as plt

from frozen_ground_fem.materials import Material

from frozen_ground_fem.thermal import (
    ThermalAnalysis1D,
    ThermalBoundary1D,
)


def main():
    # create thermal analysis object
    # define mesh with 10 elements, cubic interpolation
    ta = ThermalAnalysis1D()
    ta.z_min = -8.0
    ta.z_max = 100.0
    ta.generate_mesh(num_elements=10)

    # define material properties
    # and initialize integration points
    mtl = Material(
        thrm_cond_solids=7.0, spec_grav_solids=2.65, spec_heat_cap_solids=741
    )
    void_ratio = 0.3
    deg_sat_water = 0.8
    for e in ta.elements:
        for ip in e.int_pts:
            ip.material = mtl
            ip.void_ratio = void_ratio
            ip.deg_sat_water = deg_sat_water

    # initialize global matrices
    ta.update_heat_flow_matrix()
    ta.update_heat_storage_matrix()

    # create thermal boundary conditions
    temp_boundary = ThermalBoundary1D(
        (ta.nodes[0],),
        (ta.elements[0].int_pts[0],),
    )
    temp_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp
    temp_boundary.bnd_value = -10.0
    grad_boundary = ThermalBoundary1D(
        (ta.nodes[-1],),
        (ta.elements[-1].int_pts[-1],),
    )
    grad_boundary.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
    grad_boundary.bnd_value = 25.0 / 1e3

    # assign thermal boundaries to the analysis
    ta.add_boundary(temp_boundary)
    ta.add_boundary(grad_boundary)

    # update boundary conditions
    ta.update_thermal_boundary_conditions(time=0.0)
    ta.update_heat_flux_vector()

    plt.figure(figsize=(7, 9))

    plt.subplot(4, 1, 1)
    plt.imshow(ta._heat_flow_matrix)
    plt.colorbar()
    plt.title("Global Heat Flow Matrix")

    plt.subplot(4, 1, 2)
    plt.imshow(ta._heat_storage_matrix)
    plt.colorbar()
    plt.title("Global Heat Storage Matrix")

    plt.subplot(4, 1, 3)
    plt.imshow(ta._heat_flux_vector.reshape((1, ta.num_nodes)))
    plt.colorbar()
    plt.title("Global Heat Flux Vector")

    plt.subplot(4, 1, 4)
    plt.imshow(ta._temp_vector.reshape((1, ta.num_nodes)))
    plt.colorbar()
    plt.title("Global Temperature Field")

    plt.savefig("examples/thermal_global_matrices.png")


if __name__ == "__main__":
    main()
