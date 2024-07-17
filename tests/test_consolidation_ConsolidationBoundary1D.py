import unittest

import numpy as np

from frozen_ground_fem.geometry import (
    Node1D,
    IntegrationPoint1D,
)
from frozen_ground_fem.consolidation import (
    ConsolidationBoundary1D,
)


class TestConsolidationBoundary1DDefaults(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.consol_bnd = ConsolidationBoundary1D(self.nodes)

    def test_default_bnd_type(self):
        self.assertEqual(
            self.consol_bnd.bnd_type, ConsolidationBoundary1D.BoundaryType.water_flux
        )

    def test_default_bnd_value(self):
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 0.0)

    def test_default_bnd_value_type(self):
        self.assertIsInstance(self.consol_bnd.bnd_value, float)

    def test_default_int_pts_value(self):
        self.assertEqual(self.consol_bnd.int_pts, ())

    def test_default_int_pts_type(self):
        self.assertIsInstance(self.consol_bnd.int_pts, tuple)

    def test_default_bnd_function_value(self):
        self.assertIsNone(self.consol_bnd.bnd_function)


class TestConsolidationBoundary1DInvalid(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.consol_bnd = ConsolidationBoundary1D(self.nodes)

    def test_initialize_no_nodes(self):
        with self.assertRaises(TypeError):
            ConsolidationBoundary1D()

    def test_initialize_too_few_nodes(self):
        with self.assertRaises(ValueError):
            ConsolidationBoundary1D(
                (),
            )

    def test_initialize_too_many_nodes(self):
        with self.assertRaises(ValueError):
            ConsolidationBoundary1D(tuple([Node1D(k, 2.0 * k + 1.0) for k in range(2)]))

    def test_initialize_invalid_nodes(self):
        with self.assertRaises(TypeError):
            ConsolidationBoundary1D((1,))

    def test_assign_nodes_not_allowed(self):
        with self.assertRaises(AttributeError):
            self.consol_bnd.nodes = (Node1D(1, 3.0),)

    def test_assign_int_pts_not_allowed(self):
        with self.assertRaises(AttributeError):
            self.consol_bnd.int_pts = (IntegrationPoint1D(),)

    def test_assign_bnd_type_invalid(self):
        bt = ConsolidationBoundary1D.BoundaryType
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_type = 0
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_type = "void_ratio"
        with self.assertRaises(AttributeError):
            self.consol_bnd.bnd_type = bt.disp

    def test_assign_bnd_value_invalid(self):
        with self.assertRaises(ValueError):
            self.consol_bnd.bnd_value = "void_ratio"
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_value = (Node1D(1, 3.0),)

    def test_assign_bnd_function_invalid(self):
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_function = 1.0
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_function = 2
        with self.assertRaises(TypeError):
            self.consol_bnd.bnd_function = "three"


class TestConsolidationBoundary1DBasicSetters(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.int_pts = (IntegrationPoint1D(),)
        self.consol_bnd = ConsolidationBoundary1D(self.nodes, self.int_pts)

    def test_nodes_equal(self):
        for nd, bnd_nd in zip(self.nodes, self.consol_bnd.nodes):
            self.assertIs(nd, bnd_nd)

    def test_int_pts_equal(self):
        for ip, bnd_ip in zip(self.int_pts, self.consol_bnd.int_pts):
            self.assertIs(ip, bnd_ip)

    def test_assign_bnd_type_valid(self):
        bt = ConsolidationBoundary1D.BoundaryType
        self.consol_bnd.bnd_type = bt.water_flux
        self.assertEqual(self.consol_bnd.bnd_type, bt.water_flux)
        self.consol_bnd.bnd_type = bt.void_ratio
        self.assertEqual(self.consol_bnd.bnd_type, bt.void_ratio)

    def test_assign_bnd_value_valid(self):
        self.consol_bnd.bnd_value = 1.0
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1.0)
        self.consol_bnd.bnd_value = -2
        self.assertAlmostEqual(self.consol_bnd.bnd_value, -2.0)
        self.consol_bnd.bnd_value = "1e-5"
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1e-5)

    def test_update_nodes_bnd_type_void_ratio(self):
        bt = ConsolidationBoundary1D.BoundaryType
        self.nodes[0].void_ratio = 0.87
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.87)
        self.consol_bnd.bnd_type = bt.void_ratio
        self.consol_bnd.bnd_value = 0.5
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.87)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.5)
        self.consol_bnd.bnd_value = 0.55
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.5)

    def test_update_nodes_bnd_type_flux(self):
        bt = ConsolidationBoundary1D.BoundaryType
        self.nodes[0].void_ratio = 0.87
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.87)
        self.consol_bnd.bnd_type = bt.water_flux
        self.consol_bnd.bnd_value = 7.0e-8
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.87)
        self.consol_bnd.update_nodes()
        self.assertAlmostEqual(self.consol_bnd.nodes[0].void_ratio, 0.87)

    def test_update_value_none(self):
        self.assertIsNone(self.consol_bnd.bnd_function)
        self.consol_bnd.bnd_value = 1.5
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1.5)
        self.consol_bnd.update_value(0.5)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1.5)
        self.consol_bnd.update_value(-0.5)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, 1.5)


class TestConsolidationBoundary1DBndFunction(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.consol_bnd = ConsolidationBoundary1D(self.nodes)
        per = 365.0 * 86400.0
        om = 2.0 * np.pi / per
        t0 = (7.0 / 12.0) * per
        eavg = 0.5
        eamp = 0.1

        def f(t):
            return eavg + eamp * np.cos(om * (t - t0))

        self.f = f
        self.consol_bnd.bnd_function = f

    def test_assign_bnd_function_valid_function(self):
        self.assertIs(self.consol_bnd.bnd_function, self.f)
        self.assertTrue(callable(self.consol_bnd.bnd_function))
        self.consol_bnd.bnd_function = None
        self.assertIsNone(self.consol_bnd.bnd_function)

    def test_update_value_function(self):
        self.consol_bnd.update_value(6307200.0)
        expected0 = self.f(6307200.0)
        expected1 = 0.425685517452261
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected1)
        self.consol_bnd.update_value(18921600.0)
        expected0 = self.f(18921600.0)
        expected1 = 0.599452189536827
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected1)


class TestConsolidationBoundary1DBndFunctionLambda(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.consol_bnd = ConsolidationBoundary1D(self.nodes)

        def bfunc(per, t0, eavg, eamp):
            om = 2.0 * np.pi / per
            return lambda t: eavg + eamp * np.cos(om * (t - t0))

        per = 365.0 * 86400.0
        t0 = (7.0 / 12.0) * per
        eavg = 0.5
        eamp = 0.1
        self.f = bfunc(per, t0, eavg, eamp)
        self.consol_bnd.bnd_function = self.f

    def test_assign_bnd_function_valid_lambda(self):
        self.assertIs(self.consol_bnd.bnd_function, self.f)
        self.assertTrue(callable(self.consol_bnd.bnd_function))

    def test_update_value_lambda(self):
        self.consol_bnd.update_value(6307200.0)
        expected0 = self.f(6307200.0)
        expected1 = 0.425685517452261
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected1)
        self.consol_bnd.update_value(18921600.0)
        expected0 = self.f(18921600.0)
        expected1 = 0.599452189536827
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected1)


class TestConsolidationBoundary1DBndFunctionClass(unittest.TestCase):
    def setUp(self):
        self.nodes = (Node1D(0, 2.0),)
        self.consol_bnd = ConsolidationBoundary1D(self.nodes)

        class BFunc:
            def __init__(self, per, t0, eavg, eamp):
                self.per = per
                self.t0 = t0
                self.eavg = eavg
                self.eamp = eamp
                self.om = 2.0 * np.pi / per

            def __call__(self, t):
                return self.eavg + self.eamp * np.cos(self.om * (t - self.t0))

        per = 365.0 * 86400.0
        t0 = (7.0 / 12.0) * per
        eavg = 0.5
        eamp = 0.1
        self.f = BFunc(per, t0, eavg, eamp)
        self.consol_bnd.bnd_function = self.f

    def test_assign_bnd_function_valid_class(self):
        self.assertIs(self.consol_bnd.bnd_function, self.f)
        self.assertTrue(callable(self.consol_bnd.bnd_function))

    def test_update_value_class(self):
        self.consol_bnd.update_value(6307200.0)
        expected0 = self.f(6307200.0)
        expected1 = 0.425685517452261
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected1)
        self.consol_bnd.update_value(18921600.0)
        expected0 = self.f(18921600.0)
        expected1 = 0.599452189536827
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected0)
        self.assertAlmostEqual(self.consol_bnd.bnd_value, expected1)


if __name__ == "__main__":
    unittest.main()
