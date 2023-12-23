import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
)
from frozen_ground_fem.thermal import (
    ThermalBoundary1D,
)


class TestThermalBoundary1D(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.int_pts = (IntegrationPoint1D(),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes, self.int_pts)

    def test_defaults(self):
        self.assertEqual(self.thrm_bnd.bnd_type,
                         ThermalBoundary1D.BoundaryType.temp)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 0.0)

    def test_nodes_equal(self):
        for nd, bnd_nd in zip(self.nodes, self.thrm_bnd.nodes):
            self.assertIs(nd, bnd_nd)

    def test_int_pts_equal(self):
        for ip, bnd_ip in zip(self.int_pts, self.thrm_bnd.int_pts):
            self.assertIs(ip, bnd_ip)

    def test_assign_bnd_type_invalid(self):
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = 0
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = "temp"
        with self.assertRaises(AttributeError):
            self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.disp

    def test_assign_bnd_type_valid(self):
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.heat_flux
        self.assertEqual(
            self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.heat_flux
        )
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp
        self.assertEqual(self.thrm_bnd.bnd_type,
                         ThermalBoundary1D.BoundaryType.temp)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
        self.assertEqual(
            self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.temp_grad
        )

    def test_assign_bnd_value_invalid(self):
        with self.assertRaises(ValueError):
            self.thrm_bnd.bnd_value = "temp"

    def test_assign_bnd_value_valid(self):
        self.thrm_bnd.bnd_value = 1.0
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1.0)
        self.thrm_bnd.bnd_value = -2
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, -2.0)
        self.thrm_bnd.bnd_value = "1e-5"
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1e-5)

    def test_assign_bnd_function_invalid(self):
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_function = 1.0
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_function = 2
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_function = "three"

    def test_assign_bnd_function_valid_function(self):
        per = 365.0 * 86400.0
        om = 2.0 * np.pi / per
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        def f(t): return Tavg + Tamp * np.cos(om * (t - t0))
        self.thrm_bnd.bnd_function = f
        self.assertIs(self.thrm_bnd.bnd_function, f)
        self.assertTrue(callable(self.thrm_bnd.bnd_function))
        self.thrm_bnd.bnd_function = None
        self.assertIsNone(self.thrm_bnd.bnd_function)

    def test_assign_bnd_function_valid_lambda(self):
        def bfunc(per, t0, Tavg, Tamp):
            om = 2.0 * np.pi / per
            return lambda t: Tavg + Tamp * np.cos(om * (t - t0))
        per = 365.0 * 86400.0
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        f = bfunc(per, t0, Tavg, Tamp)
        self.thrm_bnd.bnd_function = f
        self.assertIs(self.thrm_bnd.bnd_function, f)
        self.assertTrue(callable(self.thrm_bnd.bnd_function))

    def test_assign_bnd_function_valid_class(self):
        class BFunc:
            def __init__(self, per, t0, Tavg, Tamp):
                self.per = per
                self.t0 = t0
                self.Tavg = Tavg
                self.Tamp = Tamp
                self.om = 2.0 * np.pi / per

            def __call__(self, t):
                return (self.Tavg
                        + self.Tamp * np.cos(self.om * (t - self.t0)))

        per = 365.0 * 86400.0
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        f = BFunc(per, t0, Tavg, Tamp)
        self.thrm_bnd.bnd_function = f
        self.assertIs(self.thrm_bnd.bnd_function, f)
        self.assertTrue(callable(self.thrm_bnd.bnd_function))

    def test_update_nodes_bnd_type_temp(self):
        self.nodes[0].temp = 20.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp
        self.thrm_bnd.bnd_value = 5.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 5.0)
        self.thrm_bnd.bnd_value = 7.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 5.0)

    def test_update_nodes_bnd_type_non_temp(self):
        self.nodes[0].temp = 20.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.heat_flux
        self.thrm_bnd.bnd_value = 5.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
        self.thrm_bnd.bnd_value = 7.0
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)
        self.thrm_bnd.update_nodes()
        self.assertAlmostEqual(self.thrm_bnd.nodes[0].temp, 20.0)

    def test_update_value_none(self):
        self.assertIsNone(self.thrm_bnd.bnd_function)
        self.thrm_bnd.bnd_value = 1.5
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1.5)
        self.thrm_bnd.update_value(0.5)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1.5)
        self.thrm_bnd.update_value(-0.5)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1.5)

    def test_update_value_function(self):
        per = 365.0 * 86400.0
        om = 2.0 * np.pi / per
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        def f(t): return Tavg + Tamp * np.cos(om * (t - t0))
        self.thrm_bnd.bnd_function = f
        self.thrm_bnd.update_value(6307200.0)
        expected0 = f(6307200.0)
        expected1 = -9.86289650954788
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)
        self.thrm_bnd.update_value(18921600.0)
        expected0 = f(18921600.0)
        expected1 = 24.8904379073655
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)

    def test_update_value_lambda(self):
        def bfunc(per, t0, Tavg, Tamp):
            om = 2.0 * np.pi / per
            return lambda t: Tavg + Tamp * np.cos(om * (t - t0))
        per = 365.0 * 86400.0
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        f = bfunc(per, t0, Tavg, Tamp)
        self.thrm_bnd.bnd_function = f
        self.thrm_bnd.update_value(6307200.0)
        expected0 = f(6307200.0)
        expected1 = -9.86289650954788
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)
        self.thrm_bnd.update_value(18921600.0)
        expected0 = f(18921600.0)
        expected1 = 24.8904379073655
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)

    def test_update_value_class(self):
        class BFunc:
            def __init__(self, per, t0, Tavg, Tamp):
                self.per = per
                self.t0 = t0
                self.Tavg = Tavg
                self.Tamp = Tamp
                self.om = 2.0 * np.pi / per

            def __call__(self, t):
                return (self.Tavg
                        + self.Tamp * np.cos(self.om * (t - self.t0)))

        per = 365.0 * 86400.0
        t0 = (7.0/12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        f = BFunc(per, t0, Tavg, Tamp)
        self.thrm_bnd.bnd_function = f
        self.thrm_bnd.update_value(6307200.0)
        expected0 = f(6307200.0)
        expected1 = -9.86289650954788
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)
        self.thrm_bnd.update_value(18921600.0)
        expected0 = f(18921600.0)
        expected1 = 24.8904379073655
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)
