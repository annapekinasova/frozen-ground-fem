from frozen_ground_fem import (
    Node1D,
    Element1D,
)
from frozen_ground_fem.coupled import (
    CoupledElement1D,
)


def main():
    e = Element1D([Node1D(k, 2 * k) for k in range(4)])
    print(">>> e = Element1D([Node1D(k, 2 * k) for k in range(4)])")
    print(f"e={e}")
    print(f"id(e)={id(e)}")
    print(f"id(e.nodes)={id(e.nodes)}")
    for nd in e.nodes:
        print(f"index={nd.index}, z={nd.z}, id={id(nd)}")
    print()

    ce = CoupledElement1D(e.nodes)
    print(">>> ce = CoupledElement1D(e.nodes)")
    print(f"ce={ce}")
    print(f"id(ce)={id(ce)}")
    print(f"id(ce.nodes)={id(ce.nodes)}")
    for nd in ce.nodes:
        print(f"index={nd.index}, z={nd.z}, id={id(nd)}")


if __name__ == "__main__":
    main()
