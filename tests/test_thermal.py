import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Node1D,
    Element1D,
)
from frozen_ground_fem.thermal import (
    ThermalElement1D,
)


class TestThermalElement1D(unittest.TestCase):
    def setUp(self):
        self.nodes = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
        self.e = Element1D(self.nodes)
        self.thrm_e = ThermalElement1D(self.e)

    def test_invalid_no_parent(self):
        with self.assertRaises(TypeError):
            thrm_e = ThermalElement1D()

    def test_invalid_parent(self):
        with self.assertRaises(TypeError):
            thrm_e = ThermalElement1D(self.nodes)
