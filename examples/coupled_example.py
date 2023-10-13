from frozen_ground_fem import (
    Node1D,
    Element1D,
)
from frozen_ground_fem.coupled import (
    CoupledElement1D,
)


def main():
    e = Element1D([Node1D(k, 2*k) for k in range(4)])
    print(f"Element1D, id={id(e)}")
    for nd in e.nodes:
        print(f"index={nd.index}, z={nd.z}, id={id(nd)}")
    print()

    ce = CoupledElement1D(e)
    print(f"CoupledElement1D, id={id(ce)}")
    print(f"parent id={id(ce._parent)}")
    for nd in ce.nodes:
        print(f"index={nd.index}, z={nd.z}, id={id(nd)}")


if __name__ == "__main__":
    main()
