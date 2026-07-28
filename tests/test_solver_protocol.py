import unittest

from lib import QuboSolver, Solver
from lib.contracts import QuboProblem, QuboResult


class SolverProtocolTests(unittest.TestCase):
    def test_qubo_solver_specializes_generic_solver(self):
        self.assertIn(Solver, QuboSolver.__mro__)

    def test_qubo_solver_keeps_schema_specific_annotations(self):
        annotations = QuboSolver.solve.__annotations__

        self.assertEqual(QuboProblem, annotations['problem'])
        self.assertEqual(QuboResult, annotations['return'])


if __name__ == '__main__':
    unittest.main()
