from frozen_ground_fem.geometry import (
    Point1D,
    Node1D,
    )


def main():
    print("successfully imported frozen_ground_fem")

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
    

if __name__ == "__main__":
    main()
