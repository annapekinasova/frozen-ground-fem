"""coupled.py
Module for coupled thermal and large strain consolidation physics
using the finite element method.
"""
from typing import (
    Callable,
    Optional,
    override,
)
from enum import Enum

import numpy as np

from frozen_ground_fem.materials import (
    unit_weight_water as gam_w,
    spec_grav_ice,
)

from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
    Element1D,
    Boundary1D,
    Mesh1D,
)

from frozen_ground_fem.thermal import (
    ThermalElement1D,
    ThermalBoundary1D,
    ThermalAnalysis1D,
)

from frozen_ground_fem.consolidation import (
    ConsolidationElement1D,
    ConsolidationBoundary1D,
    ConsolidationAnalysis1D,
)

class CoupledElement1D(ThermalElement1D, ConsolidationElement1D):
    """Class for computing element matrices
    for coupled thermal and large strain consolidation physics,
    inheriting from ThermalElement1D and ConsolidationElement1D.

    Parameters
    ----------
    parent : frozen_ground_fem.geometry.ThermalElement1D,
    	     frozen_ground_fem.geometry.ConsolidationElement1D
        The parent element from the mesh

    Raises
    ------
    TypeError
        If parent initializer is not a
        :c:`frozen_ground_fem.geometry.ThermalElement1D`,
        :c:`frozen_ground_fem.geometry.ConsolidationElement1D`.
    """

    def __init__(self, parent: ThermalElement1D, ConsolidationElement1D) -> None:
        if not isinstance(parent, ThermalElement1D, ConsolidationElement1D):
            raise TypeError(f"type(parent): {type(parent)} is not ThermalElement1D, ConsolidationElement1D")
        self._parent = parent

    @override
    @property
    def nodes(self) -> tuple[ThermalElement1D, ConsolidationElement1D, ...]:
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple of :c:`Node1D`

        Notes
        -----
        This is a wrapper that references the nodes property
        of the parents ThermalElement1D and ConsolidationElement1D.
        """
        return self._parent.nodes

    @override
    @property
    def jacobian(self) -> float:
        """The length scale of the element (in Lagrangian coordinates).

        Returns
        -------
        float

        Notes
        -----
        This is a wrapper that references the jacobian property
        of the parents ThermalElement1D and ConsolidationElement1D.
        """
        return self._parent.jacobian

    @override
    @property
    def int_pts(self) -> tuple[ThermalElement1D, ConsolidationElement1D, ...]:
        """The tuple of :c:`IntegrationPoint1D` contained in the element.

        Returns
        ------
        tuple of :c:`IntegrationPoint1D`

        Notes
        -----
        This is a wrapper that references the int_pts property
        of the parents ThermalElement1D and ConsolidationElement1D.
        """
        return self._parent.int_pts
        
    @property
    def heat_flow_matrix(self) -> np.ndarray[ThermalElement1D, ...]:
        """An array of element heat flow (conduction) matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        This is a wrapper that references the element heat flow (conduction) matrix property
        of the parent ThermalElement1D.
        """
        return self._parent.heat_flow_matrix
           
    @property
    def heat_storage_matrix(self) -> np.ndarray[ThermalElement1D, ...]:
        """An array of element heat storage matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        This is a wrapper that references the element heat storage matrix property
        of the parent ThermalElement1D.
        """
        return self._parent.heat_storage_matrix
        
    @property
    def stiffness_matrix(self) -> np.ndarray[ConsolidationElement1D, ...]:
        """An array of element stiffness matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        This is a wrapper that references the element stiffness matrix property
        of the parent ConsolidationElement1D.
        """
        return self._parent.stiffness_matrix
                   
    @property
    def mass_matrix(self) -> np.ndarray[ConsolidationElement1D, ...]:
        """An array of element mass_matrix.

        Returns
        -------
        numpy.ndarray, shape=(2, 2)

        Notes
        -----
        This is a wrapper that references the element mass_matrix property
        of the parent ConsolidationElement1D.
        """
        return self._parent.mass_matrix
        
    @property
    def deformed_length(self) -> float[ConsolidationElement1D]:
        """The deformed length of the element.

        Returns
        -------
        float

        Notes
        -----
        This is a wrapper that references the deformed length of the element property
        of the parent ConsolidationElement1D.
        """
        return self._parent.deformed_length
        
class CoupledBoundary1D(ThermalBoundary1D, ConsolidationBoundary1D):
    """Class for inheriting and updating boundary conditions
    for thermal and consolidation physics.

    Parameters
    ----------
    parent : frozen_ground_fem.geometry.ThermalBoundary1D
             frozen_ground_fem.geometry.ConsolidationBoundary1D
        The parent boundary element from the mesh
    bnd_type : ThermalBoundary1D.BoundaryType, ConsolidationBoundary1D.BoundaryType, optional
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
        :c:`frozen_ground_fem.geometry.ThermalBoundary1D`,
        :c:`frozen_ground_fem.geometry.ConsolidationBoundary1D`.
    """

    BoundaryType = Enum(
        "BoundaryType", ["void_ratio", "fixed_flux", "water_flux"])
    def __init__(
        self,
        parent: ThermalBoundary1D, ConsolidationBoundary1D,
        bnd_type=BoundaryType.fixed_flux,
        bnd_value: float = 0.0,
        bnd_function=None,
    ) -> None:
        if not isinstance(parent, ThermalBoundary1D, ConsolidationBoundary1D):
            raise TypeError(f"type(parent): {type(parent)} is not ThermalBoundary1D, ConsolidationBoundary1D")
        self._parent = parent
        self.bnd_type = bnd_type
        self.bnd_value = bnd_value
        self.bnd_function = bnd_function

        
    @override
    @property
    def nodes(self) -> tuple[Node1D, ...]:
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple of :c:`Node1D`

        Notes
        -----
        This is a wrapper that references the nodes property
        of the parent ThermalBoundary1D, ConsolidationBoundary1D.
        """
        return self._parent.nodes


    @override
    @property
    def int_pts(self) -> Optional[tuple[IntegrationPoint1D, ...]]:
        """The tuple of :c:`IntegrationPoint1D` contained in
        the boundary element.

        Returns
        ------
        tuple of :c:`IntegrationPoint1D`

        Notes
        -----
        This is a wrapper that references the int_pts property
        of the parent Boundary1D.
        """
        return self._parent.int_pts

    
