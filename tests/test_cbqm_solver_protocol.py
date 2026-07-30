"""Structural typing checks for the native CBQM solver boundary."""

import unittest
from typing import is_typeddict

from lib import CbqmSolver
from lib.contracts import (
    CbqmProblem,
    CbqmResult,
    Solver,
    evaluate_cbqm_feasibility,
    evaluate_cbqm_objective,
    validate_cbqm_result,
)


class CbqmSolverProtocolTests(unittest.TestCase):
    def test_cbqm_solver_specializes_the_generic_protocol(self):
        self.assertIn(Solver, CbqmSolver.__mro__)

    def test_cbqm_solver_keeps_schema_specific_annotations(self):
        annotations = CbqmSolver.solve.__annotations__

        self.assertEqual(CbqmProblem, annotations['problem'])
        self.assertEqual(CbqmResult, annotations['return'])

    def test_cbqm_contract_types_are_field_aware_typed_dicts(self):
        self.assertTrue(is_typeddict(CbqmProblem))
        self.assertTrue(is_typeddict(CbqmResult))

    def test_public_runtime_operations_are_exported(self):
        self.assertTrue(callable(evaluate_cbqm_objective))
        self.assertTrue(callable(evaluate_cbqm_feasibility))
        self.assertTrue(callable(validate_cbqm_result))


if __name__ == '__main__':
    unittest.main()
