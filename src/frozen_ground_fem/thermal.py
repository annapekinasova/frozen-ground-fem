"""thermal.py
Module for implementing thermal physics using the finite element method.
"""
from enum import Enum

import numpy as np

from frozen_ground_fem.geometry import (
    shape_matrix,
    gradient_matrix,
    Element1D,
    BoundaryElement1D,
)


class ThermalElement1D(Element1D):
    """Class for computing element matrices for thermal physics.

    Parameters
    ----------
    parent : frozen_ground_fem.geometry.Element1D
        The parent element from the mesh

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


class ThermalBoundary1D(BoundaryElement1D):
    """Class for storing and updating boundary conditions for thermal physics.

    Parameters
    ----------
    parent : frozen_ground_fem.geometry.BoundaryElement1D
        The parent boundary element from the mesh
    bnd_type : ThermalBoundary1D.BoundaryType, optional
        The type of boundary condition
    bnd_value : float, optional
        The value of the boundary condition

    Attributes
    ----------
    BoundaryType : enum.Enum
        The set of possible boundary condition types

    Raises
    ------
    TypeError
        If parent initializer is not a
        :c:`frozen_ground_fem.geometry.BoundaryElement1D`.
    """

    BoundaryType = Enum("BoundaryType", ["temp", "heat_flux", "temp_grad"])

    def __init__(self, parent, bnd_type=BoundaryType.temp, bnd_value=0.0):
        if not isinstance(parent, BoundaryElement1D):
            raise TypeError(f"type(parent): {type(parent)} is not BoundaryElement1D")
        self._parent = parent
        self.bnd_type = bnd_type
        self.bnd_value = bnd_value

    @property
    def nodes(self):
        """The tuple of :c:`Node1D` contained in the boundary element.

        Returns
        ------
        tuple of :c:`Node1D`

        Notes
        -----
        This is a wrapper that references the nodes property
        of the parent BoundaryElement1D.
        """
        return self._parent.nodes

    @property
    def bnd_type(self):
        """The type of boundary condition.

        Parameters
        ----------
        value : ThermalBoundary1D.BoundaryType
            The value to set the type of boundary condition.

        Returns
        -------
        ThermalBoundary1D.BoundaryType

        Raises
        ------
        TypeError
            If the value to be assigned is not a ThermalBoundary1D.BoundaryType
        """
        return self._bnd_type

    @bnd_type.setter
    def bnd_type(self, value):
        if not isinstance(value, ThermalBoundary1D.BoundaryType):
            raise TypeError(f"{value} is not a ThermalBoundary1D.BoundaryType")
        self._bnd_type = value

    @property
    def bnd_value(self):
        """The value of the boundary condition.

        Parameters
        ----------
        value : float
            The value to set for the boundary condition

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to be assigned is not convertible to float
        """
        return self._bnd_value

    @bnd_value.setter
    def bnd_value(self, value):
        value = float(value)
        self._bnd_value = value

    def update_nodes(self):
        """Update the boundary condition value at the nodes.

        Notes
        -----
        This method updates the temperature at each of the nodes
        in the ThermalBoundary1D
        only in the case that bnd_type == BoundaryType.temp.
        Otherwise, it does nothing.
        """
        if self.bnd_type == BoundaryType.temp:
            for nd in self.nodes:
                nd.temp = self.bnd_value
