"""thermal.py
Module for implementing thermal physics using the finite element method.
"""
import numpy as np

from frozen_ground_fem.geometry import (
    shape_matrix,
    gradient_matrix,
    Element1D,
)


class ThermalElement1D(Element1D):
    """Class for computing element matrices for thermal physics.

    Attributes
    ----------
    nodes
    int_pts
    jacobian

    Raises
    ------
    TypeError
        If parent initializer is not a
        :c:`frozen_ground_fem.geometry.Element1D`.
    """

    def __init__(self, parent: Element1D):
        if not isinstance(parent, Element1D):
            raise TypeError(f"type(parent): {type(parent)} is not Element1D")
        self._parent = parent

    @property
    def nodes(self):
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple of :c:`Node1D`

        Notes
        -----
        This is a wrapper that references the nodes property
        of the parent Element1D.
        """
        return self._parent.nodes

    @property
    def jacobian(self):
        """The length scale of the element (in Lagrangian coordinates).

        Returns
        -------
        float

        Notes
        -----
        This is a wrapper that references the jacobian property
        of the parent Element1D.
        """
        return self._parent.jacobian

    @property
    def int_pts(self):
        """The tuple of :c:`IntegrationPoint1D` contained in the element.

        Returns
        ------
        tuple of :c:`IntegrationPoint1D`

        Notes
        -----
        This is a wrapper that references the int_pts property
        of the parent Element1D.
        """
        return self._parent.int_pts

    def heat_flow_matrix(self):
        """The element heat flow (conduction) matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        Integrates B^T * lambda * B over the element
        where lambda is the thermal conductivity.
        """
        B = gradient_matrix(0, 1)
        H = np.zeros_like(B.T @ B)
        jac = self.jacobian
        for ip in self.int_pts:
            B = gradient_matrix(ip.local_coord, jac)
            H += B.T @ (ip.thrm_cond * B) * ip.weight
        H *= jac
        return H

    def heat_storage_matrix(self):
        """The element heat storage matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        Integrates N^T * C * N over the element
        where C is the volumetric heat capacity.
        """
        N = shape_matrix(0)
        C = np.zeros_like(N.T @ N)
        jac = self.jacobian
        for ip in self.int_pts:
            N = shape_matrix(ip.local_coord)
            C += N.T @ (ip.vol_heat_cap * N) * ip.weight
        C *= jac
        return C
