import unittest

from lib import solvers
from lib.solvers import cbqm, mis, qubo


class SolverPackageTests(unittest.TestCase):
    def test_exports_representation_specific_subpackages(self):
        self.assertEqual(['cbqm', 'mis', 'qubo'], solvers.__all__)
        self.assertIs(solvers.cbqm, cbqm)
        self.assertIs(solvers.mis, mis)
        self.assertIs(solvers.qubo, qubo)

    def test_subpackages_do_not_claim_unimplemented_solvers(self):
        self.assertEqual([], cbqm.__all__)
        self.assertEqual([], mis.__all__)
        self.assertEqual(
            ['BaseQuboSolver', 'ExactQuboSolver', 'QuboSolveOutcome'],
            qubo.__all__,
        )


if __name__ == '__main__':
    unittest.main()
