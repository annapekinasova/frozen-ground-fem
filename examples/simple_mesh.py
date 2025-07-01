"""A simple example script that generates and plots a mesh.
"""
import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.geometry import (
    Mesh1D,
)


def plot_mesh(mesh, fname):
    # initialize figure
    fig = plt.figure(figsize=(5, 8))
    # calculate mesh plotting coordinates
    total_depth = mesh.z_max - mesh.z_min
    z_nodes = np.array([nd.z for nd in mesh.nodes])
    z_elements = [0.5 * (e.nodes[0].z + e.nodes[-1].z) for e in mesh.elements]
    z_elements = np.array(z_elements)
    z_int_pts = [[p.z for p in e.int_pts] for e in mesh.elements]
    z_int_pts = np.array(z_int_pts).flatten()
    order = mesh.elements[0].order
    # plot element boundaries and connectivity
    for k, nd in enumerate(mesh.nodes):
        if not k % order:
            plt.plot([-0.1 * total_depth, 0.1 * total_depth], [nd.z, nd.z], "-k")
    ax = fig.axes[0]
    for k, e in enumerate(mesh.elements):
        plt.plot(
            [0, -0.05 * total_depth, 0],
            [e.nodes[0].z, z_elements[k], e.nodes[-1].z],
            "--g",
        )
    # plot node labels
    plt.plot(np.zeros_like(z_nodes), z_nodes, "ok", label="nodes", markersize=12)
    for k, nd in enumerate(mesh.nodes):
        ax.annotate(
            f"{k}",
            xy=(0, nd.z),
            xycoords="data",
            xytext=(-3, -3),
            textcoords="offset points",
            color="white",
            fontsize=7,
        )
    # plot element labels
    plt.plot(
        -0.05 * total_depth * np.ones_like(z_elements),
        z_elements,
        "sg",
        label="elements",
        markersize=12,
        markerfacecolor="white",
    )
    for k, e in enumerate(mesh.elements):
        ax.annotate(
            f"{k}",
            xy=(-0.05 * total_depth, z_elements[k]),
            xycoords="data",
            xytext=(-3, -3),
            textcoords="offset points",
            color="green",
            fontsize=8,
        )
    # plot integration points
    plt.plot(
        0.05 * total_depth * np.ones_like(z_int_pts), z_int_pts, "xr", label="int_pts"
    )
    # finalize figure and save to file
    plt.axis("equal")
    ax.invert_yaxis()
    plt.ylabel("z")
    plt.legend()
    plt.savefig(fname)


def main():
    # a mesh object can be created without arguments
    mesh = Mesh1D()
    print("-----------------------------")
    print("Created an empty mesh object:")
    print("-----------------------------")
    print(f"mesh_valid: {mesh.mesh_valid}")
    print(f"order: {mesh.elements[0].order if mesh.num_elements else None}")
    print(f"num_nodes: {mesh.num_nodes}")
    print(f"num_elements: {mesh.num_elements}")
    print(f"num_boundaries: {mesh.num_boundaries}")
    print()

    # to generate a mesh, you need to assign geometry parameters
    # at minimum, the minimum and maximum z coordinates
    # the num_elements argument to generate_mesh() is optional (default: 10)
    mesh.z_min = -8.0
    mesh.z_max = 100.0
    mesh.generate_mesh(num_elements=10)
    print("-----------------------------------------")
    print("Assigned parameters and generated a mesh:")
    print("-----------------------------------------")
    print(f"mesh_valid: {mesh.mesh_valid}")
    print(f"order: {mesh.elements[0].order}")
    print(f"num_nodes: {mesh.num_nodes}")
    print(f"num_elements: {mesh.num_elements}")
    print(f"num_boundaries: {mesh.num_boundaries}")
    print()
    plot_mesh(mesh, "examples/simple_mesh_10.svg")

    # you can also assign a grid_size parameter to the mesh
    # which will be used to calculate the number of nodes
    # note that the actual distance between nodes may be slightly different
    # because the grid size will be adjusted for an integer number of nodes
    mesh.grid_size = 13.5
    mesh.generate_mesh()
    print("-----------------------------------------")
    print("Assigned grid_size and generated a mesh:")
    print("-----------------------------------------")
    print(f"mesh_valid: {mesh.mesh_valid}")
    print(f"order: {mesh.elements[0].order}")
    print(f"grid_size (parameter): {mesh.grid_size}")
    print(f"grid_size (actual): {mesh.elements[0].jacobian}")
    print(f"num_nodes: {mesh.num_nodes}")
    print(f"num_elements: {mesh.num_elements}")
    print(f"num_boundaries: {mesh.num_boundaries}")
    print()
    plot_mesh(mesh, "examples/simple_mesh_grid_size.svg")


if __name__ == "__main__":
    main()
