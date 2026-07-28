import unittest
from typing import is_typeddict

from lib import QuboSolver, Solver
from lib.contracts import (
    QuboProblem,
    QuboResult,
    validate_qubo,
    validate_qubo_result,
)


class SolverProtocolTests(unittest.TestCase):
    def test_qubo_solver_specializes_generic_solver(self):
        self.assertIn(Solver, QuboSolver.__mro__)

    def test_qubo_solver_keeps_schema_specific_annotations(self):
        annotations = QuboSolver.solve.__annotations__

        self.assertEqual(QuboProblem, annotations['problem'])
        self.assertEqual(QuboResult, annotations['return'])

    def test_qubo_contract_types_expose_field_aware_typed_dicts(self):
        self.assertTrue(is_typeddict(QuboProblem))
        self.assertTrue(is_typeddict(QuboResult))

    def test_public_runtime_validators_are_exported(self):
        self.assertTrue(callable(validate_qubo))
        self.assertTrue(callable(validate_qubo_result))


if __name__ == '__main__':
    unittest.main()
