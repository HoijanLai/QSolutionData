"""Structural typing checks for the native MIS solver boundary."""

import unittest
from typing import is_typeddict

from lib import MisSolver
from lib.contracts import MisProblem, MisResult, Solver


class MisSolverProtocolTests(unittest.TestCase):
    def test_mis_solver_specializes_the_generic_protocol(self):
        self.assertIn(Solver, MisSolver.__mro__)

    def test_mis_solver_keeps_schema_specific_annotations(self):
        annotations = MisSolver.solve.__annotations__

        self.assertEqual(MisProblem, annotations['problem'])
        self.assertEqual(MisResult, annotations['return'])

    def test_mis_contract_types_are_field_aware(self):
        self.assertTrue(is_typeddict(MisProblem))
        self.assertTrue(is_typeddict(MisResult))


if __name__ == '__main__':
    unittest.main()
