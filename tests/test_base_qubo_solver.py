import copy
import unittest

from lib.solvers.qubo import BaseQuboSolver, QuboSolveOutcome


class _ConcreteSolver(BaseQuboSolver):
    SOLVER_NAME = 'test-solver'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'test'

    def _run(self, problem, config):
        problem['terms'].clear()
        config.setdefault('nested', {})['mutated'] = True
        return QuboSolveOutcome(
            status='optimal',
            best_sample=[1, 0],
            termination_reason='search_exhausted',
            metrics={'iterations': 1},
        )


class _MissingCandidateSolver(BaseQuboSolver):
    SOLVER_NAME = 'missing-candidate'
    SOLVER_VERSION = '0.1.0'

    def _run(self, problem, config):
        return QuboSolveOutcome(status='optimal', best_sample=None)


class _NonJsonOutcomeSolver(BaseQuboSolver):
    SOLVER_NAME = 'non-json-outcome'
    SOLVER_VERSION = '0.1.0'

    def _run(self, problem, config):
        return QuboSolveOutcome(
            status='feasible',
            best_sample=[0, 0],
            metadata={'unsupported': object()},
        )


class BaseQuboSolverTests(unittest.TestCase):
    def setUp(self):
        self.problem = {
            'schema': 'qubo.v1',
            'problem_id': 'base-solver-example',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['x0', 'x1'],
            'offset': 4.0,
            'terms': [
                [0, 0, -6.0],
                [0, 1, 8.0],
                [1, 1, -5.0],
            ],
            'metadata': {},
        }

    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseQuboSolver()

    def test_builds_result_and_recomputes_canonical_energy(self):
        result = _ConcreteSolver().solve(self.problem)

        self.assertEqual('qubo-result.v1', result['schema'])
        self.assertEqual('optimal', result['status'])
        self.assertEqual([1, 0], result['best_sample'])
        self.assertEqual(-2.0, result['best_energy'])
        self.assertEqual({'iterations': 1}, result['metrics'])

    def test_defensively_copies_problem_and_config(self):
        original_problem = copy.deepcopy(self.problem)
        config = {'nested': {}}

        _ConcreteSolver().solve(self.problem, config)

        self.assertEqual(original_problem, self.problem)
        self.assertEqual({'nested': {}}, config)

    def test_requires_candidate_for_optimal_status(self):
        with self.assertRaisesRegex(ValueError, 'requires a best_sample'):
            _MissingCandidateSolver().solve(self.problem)

    def test_validates_the_fully_constructed_public_result(self):
        with self.assertRaisesRegex(TypeError, 'non-JSON value'):
            _NonJsonOutcomeSolver().solve(self.problem)


if __name__ == '__main__':
    unittest.main()
