from frozen_ground_fem.geometry import (
    shape_matrix,
    gradient_matrix,
    Point1D,
    Node1D,
    IntegrationPoint1D,
    Element1D,
    )

from frozen_ground_fem.materials import (
    Material,
    )


def main():
    print("successfully imported frozen_ground_fem")

    # testing Node1D
    nd = Node1D(1, 0.5, -5)
    print(f"created a Node1D with ID {id(nd)}")
    print(f"index: {nd.index}, z: {nd.z}, temp: {nd.temp}")
    nd.temp = -2
    print(f"index: {nd.index}, z: {nd.z}, temp: {nd.temp}")
    print(f"type(nd) is Node1D: {type(nd) is Node1D}")
    print(f"type(nd) is Point1D: {type(nd) is Point1D}")
    print(f"isinstance(nd, Node1D): {isinstance(nd, Node1D)}")
    print(f"isinstance(nd, Point1D): {isinstance(nd, Point1D)}")

    print("trying nd.index = 'three'")
    try:
        nd.index = 'three'
    except ValueError:
        print("raised ValueError as expected")

    print("trying nd.temp = 'three'")
    try:
        nd.temp = 'three'
    except ValueError:
        print("raised ValueError as expected")

    print(f"str(nd): {str(nd)}")
    print(nd)

    # testing IntegrationPoint1D
    ip = IntegrationPoint1D(0.5, 0.3, 0.05)
    print(f"created a IntegrationPoint1D with ID {id(ip)}")
    print(ip)
    ip.porosity = 0.5
    ip.vol_ice_cont = 0.08
    print(ip)
    print(f"type(ip) is Node1D: {type(ip) is Node1D}")
    print(f"type(ip) is Point1D: {type(ip) is Point1D}")
    print(f"type(ip) is IntegrationPoint1D: {type(ip) is IntegrationPoint1D}")
    print(f"isinstance(ip, Node1D): {isinstance(ip, Node1D)}")
    print(f"isinstance(ip, Point1D): {isinstance(ip, Point1D)}")
    print("isinstance(ip, IntegrationPoint1D): "
          + f"{isinstance(ip, IntegrationPoint1D)}")

    print("trying ip.porosity = 'three'")
    try:
        ip.porosity = 'three'
    except ValueError:
        print("raised ValueError as expected")

    print("trying ip.porosity = 1.2")
    try:
        ip.porosity = 1.2
    except ValueError:
        print("raised ValueError as expected")

    print("trying ip.porosity = -0.2")
    try:
        ip.porosity = -0.2
    except ValueError:
        print("raised ValueError as expected")

    print(ip)

    print("trying ip.vol_ice_cont = 'three'")
    try:
        ip.vol_ice_cont = 'three'
    except ValueError:
        print("raised ValueError as expected")

    print("trying ip.vol_ice_cont = -0.2")
    try:
        ip.vol_ice_cont = -0.2
    except ValueError:
        print("raised ValueError as expected")

    print("trying ip.vol_ice_cont = 0.6")
    try:
        ip.vol_ice_cont = 0.6
    except ValueError:
        print("raised ValueError as expected")

    # testing Material class
    m = Material(thrm_cond_solids=7.8,
                 dens_solids=2.5e3,
                 spec_heat_cap_solids=7.41e5)

    ip = IntegrationPoint1D(0.5, 0.3, 0.05)
    print('created IntegrationPoint1D with NULL_MATERIAL')
    print(f'ip.material.dens_solids: {ip.material.dens_solids}')

    ip = IntegrationPoint1D(0.5, 0.3, 0.05, m)
    print('created IntegrationPoint1D with a Material')
    print(f'ip.material.dens_solids: {ip.material.dens_solids}')
    print(f'ip.thrm_cond: {ip.thrm_cond}')
    print(f'ip.vol_heat_cap: {ip.vol_heat_cap}')

    # testing Element1D
    nodes = [Node1D(k, 2 * k) for k in range(10)]
    e0 = Element1D(nodes[2:4])
    print(f'e0.nodes[0].index: {e0.nodes[0].index}')
    print(f'e0.nodes[0].z: {e0.nodes[0].z}')
    print(f'e0.nodes[1].index: {e0.nodes[1].index}')
    print(f'e0.nodes[1].z: {e0.nodes[1].z}')

    print("trying Element1D(2)")
    try:
        Element1D(2)
    except TypeError:
        print("raised TypeError as expected")

    print("trying Element1D(Node1D(2))")
    try:
        Element1D(Node1D(2))
    except TypeError:
        print("raised TypeError as expected")

    print("trying Element1D(nodes[3:6])")
    try:
        Element1D(nodes[3:6])
    except ValueError:
        print("raised ValueError as expected")

    print("trying Element1D([2, 3])")
    try:
        Element1D([2, 3])
    except TypeError:
        print("raised TypeError as expected")

    # testing shape_matrix() and gradient_matrix()
    print(f'shape_matrix(0.1): {shape_matrix(0.1)}')
    print(f'gradient_matrix(0.1, 0.5): {gradient_matrix(0.1, 0.5)}')

    print("trying shape_matrix([2, 3])")
    try:
        shape_matrix([2, 3])
    except TypeError:
        print("raised TypeError as expected")

    print("trying shape_matrix('zero point one')")
    try:
        shape_matrix('zero point one')
    except ValueError:
        print("raised ValueError as expected")

    print("trying gradient_matrix(0.1, [2, 3])")
    try:
        gradient_matrix(0.1, [2, 3])
    except TypeError:
        print("raised TypeError as expected")

    print("trying gradient_matrix(0.1, 'zero point one')")
    try:
        gradient_matrix(0.1, 'zero point one')
    except ValueError:
        print("raised ValueError as expected")


if __name__ == "__main__":
    main()
