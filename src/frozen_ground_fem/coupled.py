"""coupled.py
Module for coupled thermal and large strain consolidation physics
using the finite element method.
"""
from .thermal import (
    ThermalElement1D,
)

from .consolidation import (
    ConsolidationElement1D,
)


class CoupledElement1D(ThermalElement1D, ConsolidationElement1D):
    """Class for computing element matrices
    for coupled thermal and large strain consolidation physics,
    inheriting from ThermalElement1D and ConsolidationElement1D.

    Parameters
    ----------
    parent : :c:`frozen_ground_fem.geometry.Element1D`
        The parent element from the mesh

    Raises
    ------
    TypeError
        If parent initializer is not a
        :c:`frozen_ground_fem.geometry.Element1D`.
    """

    pass
