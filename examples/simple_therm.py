from frozen_ground_fem.geometry import (
    Point1D,
    Node1D,
    IntegrationPoint1D,
    )
from frozen_ground_fem.materials import (
    vol_heat_cap_water,
    vol_heat_cap_ice,
    thrm_cond_water,
    thrm_cond_ice,
    Material,
    )


def main():
    print("successfully imported frozen_ground_fem")

    # testing Point1D
    p = Point1D(3)
    print(f"created a Point1D with ID {id(p)}")
    print(f"coords: {p.coords}")
    print(f"z: {p.z}")
    p.z = 2
    print(f"z: {p.z}")
    print(f"type(p) is Node1D: {type(p) is Node1D}")
    print(f"type(p) is Point1D: {type(p) is Point1D}")
    print(f"isinstance(p, Node1D): {isinstance(p, Node1D)}")
    print(f"isinstance(p, Point1D): {isinstance(p, Point1D)}")

    point_list = [Point1D(k) for k in range(10)]
    for pp in point_list:
        print(pp.z)

    print("trying p = Point1D('three')")
    try:
        p = Point1D("three")
    except ValueError:
        print("raised ValueError as expected")

    # testing Node1D
    nd = Node1D(0.5, -5)
    print(f"created a Node1D with ID {id(nd)}")
    print(f"z: {nd.z}, temp: {nd.temp}")
    nd.temp = -2
    print(f"z: {nd.z}, temp: {nd.temp}")
    print(f"type(nd) is Node1D: {type(nd) is Node1D}")
    print(f"type(nd) is Point1D: {type(nd) is Point1D}")
    print(f"isinstance(nd, Node1D): {isinstance(nd, Node1D)}")
    print(f"isinstance(nd, Point1D): {isinstance(nd, Point1D)}")

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
        
    # testing materials module
    print(f"vol_heat_cap_ice: {vol_heat_cap_ice}")
    print(f"vol_heat_cap_water: {vol_heat_cap_water}")
    print(f"thrm_cond_water: {thrm_cond_water}")
    print(f"thrm_cond_ice: {thrm_cond_ice}")
    
    # testing Material class
    m = Material(thrm_cond_solids=7.8, 
                 dens_solids=2.5e3,
                 spec_heat_cap_solids=7.41e5)
    print(m)
    print(f'thrm_cond_solids: {m.thrm_cond_solids}')
    print(f'dens_solids: {m.dens_solids}')
    print(f'spec_heat_cap_solids: {m.spec_heat_cap_solids}')
    print(f'vol_heat_cap_solids: {m.vol_heat_cap_solids}')

    print("trying m.thrm_cond_solids = 'three'")
    try:
        m.thrm_cond_solids = 'three'
    except ValueError:
        print("raised ValueError as expected")

    print("trying m.thrm_cond_solids = -0.6")
    try:
        m.thrm_cond_solids = -0.6
    except ValueError:
        print("raised ValueError as expected")

    print("trying m.dens_solids = 'three'")
    try:
        m.dens_solids = 'three'
    except ValueError:
        print("raised ValueError as expected")

    print("trying m.dens_solids = -0.6")
    try:
        m.dens_solids = -0.6
    except ValueError:
        print("raised ValueError as expected")

    print("trying m.spec_heat_cap_solids = 'three'")
    try:
        m.spec_heat_cap_solids = 'three'
    except ValueError:
        print("raised ValueError as expected")

    print("trying m.spec_heat_cap_solids = -0.6")
    try:
        m.spec_heat_cap_solids = -0.6
    except ValueError:
        print("raised ValueError as expected")
        
    ip = IntegrationPoint1D(0.5, 0.3, 0.05)
    print('created IntegrationPoint1D with NULL_MATERIAL')
    print(f'ip.material.dens_solids: {ip.material.dens_solids}')
        
    ip = IntegrationPoint1D(0.5, 0.3, 0.05, m)
    print('created IntegrationPoint1D with a Material')
    print(f'ip.material.dens_solids: {ip.material.dens_solids}')
    print(f'ip.thrm_cond: {ip.thrm_cond}')
    print(f'ip.vol_heat_cap: {ip.vol_heat_cap}')

if __name__ == "__main__":
    main()
