from frozen_ground_fem.geometry import (
    shape_matrix,
    gradient_matrix,
    Element1D,
)


class ThermalElement1D(Element1D):
    def __init__(self, parent: Element1D):
        if not isinstance(parent, Element1D):
            raise TypeError(f"type(parent): {type(parent)} is not Element1D")
        self._parent = parent

    @property
    def nodes(self):
        return self._parent.nodes

    @property
    def jacobian(self):
        return self._parent.jacobian

    @property
    def int_pts(self):
        return self._parent.int_pts

    def heat_flow_matrix(self):
        B = gradient_matrix(0, 1)
        H = np.zeros_like(B.T @ B)
        jac = self.jacobian
        for ip in self.int_pts:
            B = gradient_matrix(ip.local_coord, jac)
            H += B.T @ (ip.thrm_cond * B) * ip.weight
        H *= jac
        return H

    def heat_storage_matrix(self):
        N = shape_matrix(0)
        C = np.zeros_like(N.T @ N)
        jac = self.jacobian
        for ip in self.int_pts:
            N = shape_matrix(ip.local_coord)
            C += N.T @ (ip.vol_heat_cap * N) * ip.weight
        C *= jac
        return C
