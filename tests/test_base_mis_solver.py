"""Invariant-wrapper tests for native MIS solver implementations."""

import copy
import unittest

from lib.solvers.mis import BaseMisSolver, MisSolveOutcome


def _problem():
    return {
        'schema': 'mis.v1',
        'problem_id': 'base-mis',
        'objective': {'kind': 'maximum-cardinality'},
        'vertices': [
            {'index': 0, 'name': 'a'},
            {'index': 1, 'name': 'b'},
            {'index': 2, 'name': 'c'},
        ],
        'edges': [[0, 1], [1, 2]],
        'fixed_values': [],
        'metadata': {},
    }


class _ConcreteSolver(BaseMisSolver):
    SOLVER_NAME = 'test-mis'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'test'

    def _run(self, problem, config):
        problem['edges'].clear()
        config.setdefault('nested', {})['mutated'] = True
        return MisSolveOutcome(
            status='optimal',
            selected_vertices=[0, 2],
            termination_reason='search_exhausted',
            bounds={
                'incumbent_lower_bound': 2,
                'optimum_upper_bound': 2,
            },
            proof={
                'claim': 'optimality',
                'kind': 'test',
                'producer': 'test',
                'independently_verified': False,
            },
            trace=[
                {
                    'step': 0,
                    'time_seconds': 0,
                    'selected_vertices': [0, 1],
                },
            ],
        )


class _MissingCandidateSolver(BaseMisSolver):
    SOLVER_NAME = 'missing'
    SOLVER_VERSION = '1'

    def _run(self, problem, config):
        return MisSolveOutcome(status='optimal', selected_vertices=None)


class _NonIndependentSolver(BaseMisSolver):
    SOLVER_NAME = 'bad-set'
    SOLVER_VERSION = '1'

    def _run(self, problem, config):
        return MisSolveOutcome(status='feasible', selected_vertices=[0, 1])


class _TrustedTraceSolver(BaseMisSolver):
    SOLVER_NAME = 'bad-trace'
    SOLVER_VERSION = '1'

    def _run(self, problem, config):
        return MisSolveOutcome(
            status='feasible',
            selected_vertices=[0],
            trace=[
                {
                    'step': 0,
                    'time_seconds': 0,
                    'selected_vertices': [0],
                    'objective_value': 999,
                },
            ],
        )


class BaseMisSolverTests(unittest.TestCase):
    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseMisSolver()

    def test_recomputes_candidate_and_trace_semantics(self):
        result = _ConcreteSolver().solve(_problem())

        self.assertEqual('mis-result.v1', result['schema'])
        self.assertEqual(2, result['objective_value'])
        self.assertEqual(2, result['cardinality'])
        self.assertEqual(2, result['total_weight'])
        self.assertTrue(result['feasible'])
        self.assertEqual({
            'step': 0,
            'time_seconds': 0,
            'selected_vertices': [0, 1],
            'objective_value': 2,
            'cardinality': 2,
            'total_weight': 2,
            'feasible': False,
        }, result['trace'][0])

    def test_defensively_copies_problem_and_config(self):
        problem = _problem()
        original = copy.deepcopy(problem)
        config = {'nested': {}}

        _ConcreteSolver().solve(problem, config)

        self.assertEqual(original, problem)
        self.assertEqual({'nested': {}}, config)

    def test_requires_candidate_for_optimal_status(self):
        with self.assertRaisesRegex(ValueError, 'requires selected_vertices'):
            _MissingCandidateSolver().solve(_problem())

    def test_feasible_status_requires_an_independent_set(self):
        with self.assertRaisesRegex(ValueError, 'independent-set'):
            _NonIndependentSolver().solve(_problem())

    def test_kernel_cannot_supply_derived_trace_semantics(self):
        with self.assertRaisesRegex(ValueError, 'objective_value'):
            _TrustedTraceSolver().solve(_problem())


if __name__ == '__main__':
    unittest.main()
