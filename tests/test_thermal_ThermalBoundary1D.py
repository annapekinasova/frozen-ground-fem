import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
)
from frozen_ground_fem.thermal import (
    ThermalBoundary1D,
)


class TestThermalBoundary1DDefaults(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes)

    def test_default_bnd_type(self):
        self.assertEqual(self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.temp)

    def test_default_bnd_value(self):
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 0.0)

    def test_default_bnd_value_type(self):
        self.assertIsInstance(self.thrm_bnd.bnd_value, float)

    def test_default_int_pts_value(self):
        self.assertEqual(self.thrm_bnd.int_pts, ())

    def test_default_int_pts_type(self):
        self.assertIsInstance(self.thrm_bnd.int_pts, tuple)

    def test_default_bnd_function_value(self):
        self.assertIsNone(self.thrm_bnd.bnd_function)


class TestThermalBoundary1DInvalid(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes)

    def test_initialize_no_nodes(self):
        with self.assertRaises(TypeError):
            ThermalBoundary1D()

    def test_initialize_too_few_nodes(self):
        with self.assertRaises(ValueError):
            ThermalBoundary1D(
                (),
            )

    def test_initialize_too_many_nodes(self):
        with self.assertRaises(ValueError):
            ThermalBoundary1D(tuple([Node1D(k, 2.0 * k + 1.0) for k in range(2)]))

    def test_initialize_invalid_nodes(self):
        with self.assertRaises(TypeError):
            ThermalBoundary1D((1,))

    def test_assign_nodes_not_allowed(self):
        with self.assertRaises(AttributeError):
            self.thrm_bnd.nodes = (Node1D(1, 3.0),)

    def test_assign_int_pts_not_allowed(self):
        with self.assertRaises(AttributeError):
            self.thrm_bnd.int_pts = (IntegrationPoint1D(),)

    def test_assign_bnd_type_invalid(self):
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = 0
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_type = "temp"
        with self.assertRaises(AttributeError):
            self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.disp

    def test_assign_bnd_value_invalid(self):
        with self.assertRaises(ValueError):
            self.thrm_bnd.bnd_value = "temp"
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_value = (Node1D(1, 3.0),)

    def test_assign_bnd_function_invalid(self):
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_function = 1.0
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_function = 2
        with self.assertRaises(TypeError):
            self.thrm_bnd.bnd_function = "three"


class TestThermalBoundary1DBasicSetters(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.int_pts = (IntegrationPoint1D(),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes, self.int_pts)

    def test_nodes_equal(self):
        for nd, bnd_nd in zip(self.nodes, self.thrm_bnd.nodes):
            self.assertIs(nd, bnd_nd)

    def test_int_pts_equal(self):
        for ip, bnd_ip in zip(self.int_pts, self.thrm_bnd.int_pts):
            self.assertIs(ip, bnd_ip)

    def test_assign_bnd_type_valid(self):
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.heat_flux
        self.assertEqual(
            self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.heat_flux
        )
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp
        self.assertEqual(self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.temp)
        self.thrm_bnd.bnd_type = ThermalBoundary1D.BoundaryType.temp_grad
        self.assertEqual(
            self.thrm_bnd.bnd_type, ThermalBoundary1D.BoundaryType.temp_grad
        )

    def test_assign_bnd_value_valid(self):
        self.thrm_bnd.bnd_value = 1.0
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1.0)
        self.thrm_bnd.bnd_value = -2
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, -2.0)
        self.thrm_bnd.bnd_value = "1e-5"
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, 1e-5)

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


class TestThermalBoundary1DBndFunction(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes)
        per = 365.0 * 86400.0
        om = 2.0 * np.pi / per
        t0 = (7.0 / 12.0) * per
        Tavg = 5.0
        Tamp = 20.0

        def f(t):
            return Tavg + Tamp * np.cos(om * (t - t0))

        self.f = f
        self.thrm_bnd.bnd_function = f

    def test_assign_bnd_function_valid_function(self):
        self.assertIs(self.thrm_bnd.bnd_function, self.f)
        self.assertTrue(callable(self.thrm_bnd.bnd_function))
        self.thrm_bnd.bnd_function = None
        self.assertIsNone(self.thrm_bnd.bnd_function)

    def test_update_value_function(self):
        self.thrm_bnd.update_value(6307200.0)
        expected0 = self.f(6307200.0)
        expected1 = -9.86289650954788
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)
        self.thrm_bnd.update_value(18921600.0)
        expected0 = self.f(18921600.0)
        expected1 = 24.8904379073655
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)


class TestThermalBoundary1DBndFunctionLambda(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes)

        def bfunc(per, t0, Tavg, Tamp):
            om = 2.0 * np.pi / per
            return lambda t: Tavg + Tamp * np.cos(om * (t - t0))

        per = 365.0 * 86400.0
        t0 = (7.0 / 12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        self.f = bfunc(per, t0, Tavg, Tamp)
        self.thrm_bnd.bnd_function = self.f

    def test_assign_bnd_function_valid_lambda(self):
        self.assertIs(self.thrm_bnd.bnd_function, self.f)
        self.assertTrue(callable(self.thrm_bnd.bnd_function))

    def test_update_value_lambda(self):
        self.thrm_bnd.update_value(6307200.0)
        expected0 = self.f(6307200.0)
        expected1 = -9.86289650954788
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)
        self.thrm_bnd.update_value(18921600.0)
        expected0 = self.f(18921600.0)
        expected1 = 24.8904379073655
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)


class TestThermalBoundary1DBndFunctionClass(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.thrm_bnd = ThermalBoundary1D(self.nodes)

        class BFunc:
            def __init__(self, per, t0, Tavg, Tamp):
                self.per = per
                self.t0 = t0
                self.Tavg = Tavg
                self.Tamp = Tamp
                self.om = 2.0 * np.pi / per

            def __call__(self, t):
                return self.Tavg + self.Tamp * np.cos(self.om * (t - self.t0))

        per = 365.0 * 86400.0
        t0 = (7.0 / 12.0) * per
        Tavg = 5.0
        Tamp = 20.0
        self.f = BFunc(per, t0, Tavg, Tamp)
        self.thrm_bnd.bnd_function = self.f

    def test_assign_bnd_function_valid_class(self):
        self.assertIs(self.thrm_bnd.bnd_function, self.f)
        self.assertTrue(callable(self.thrm_bnd.bnd_function))

    def test_update_value_class(self):
        self.thrm_bnd.update_value(6307200.0)
        expected0 = self.f(6307200.0)
        expected1 = -9.86289650954788
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)
        self.thrm_bnd.update_value(18921600.0)
        expected0 = self.f(18921600.0)
        expected1 = 24.8904379073655
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.thrm_bnd.bnd_value, expected1)


if __name__ == "__main__":
    unittest.main()
