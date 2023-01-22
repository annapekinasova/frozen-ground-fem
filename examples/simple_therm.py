from frozen_ground_fem.geometry import Point1D


def main():
    print("successfully imported frozen_ground_fem")

    p = Point1D(3)
    print(f"created a Point1D with ID {id(p)}")
    print(f"coords: {p.coords}")
    print(f"z: {p.z}")
    p.z = 2
    print(f"z: {p.z}")

    point_list = [Point1D(k) for k in range(10)]
    for pp in point_list:
        print(pp.z)

    print("trying p = Point1D('three')")
    try:
        p = Point1D("three")
    except ValueError:
        print("raised ValueError as expected")


if __name__ == "__main__":
    main()
