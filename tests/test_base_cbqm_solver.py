"""Invariant-wrapper tests for native CBQM solver implementations."""

import copy
import unittest

from lib.solvers.cbqm import BaseCbqmSolver, CbqmSolveOutcome


def _problem():
    """Return a fresh constrained problem for every test."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'base-cbqm-solver',
        'variables': [
            {'index': 0, 'name': 'x', 'vartype': 'BINARY'},
            {'index': 1, 'name': 'y', 'vartype': 'BINARY'},
        ],
        'objective': {
            'sense': 'minimize',
            'offset': 0,
            'linear': [[0, -2], [1, -1]],
            'quadratic': [],
        },
        'constraints': [
            {
                'name': 'select_one',
                'family': 'selection',
                'linear': [[0, 1], [1, 1]],
                'lower_bound': 1,
                'upper_bound': 1,
            },
        ],
        'fixed_values': [],
        'metadata': {},
    }


class _ConcreteSolver(BaseCbqmSolver):
    SOLVER_NAME = 'test-cbqm-solver'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'test'

    def _run(self, problem, config):
        problem['objective']['linear'].clear()
        config.setdefault('nested', {})['mutated'] = True
        return CbqmSolveOutcome(
            status='optimal',
            best_sample=[1, 0],
            termination_reason='search_exhausted',
            bounds={'primal_bound': -2, 'dual_bound': -2},
            proof={
                'claim': 'optimality',
                'kind': 'test-proof',
                'producer': 'test',
                'independently_verified': False,
            },
            metrics={'iterations': 1},
            trace=[
                {
                    'step': 0,
                    'time_seconds': 0,
                    'sample': [0, 0],
                    'metadata': {'phase': 'initial'},
                },
            ],
        )


class _MissingCandidateSolver(BaseCbqmSolver):
    SOLVER_NAME = 'missing-candidate'
    SOLVER_VERSION = '0.1.0'

    def _run(self, problem, config):
        return CbqmSolveOutcome(status='optimal', best_sample=None)


class _InfeasibleCandidateSolver(BaseCbqmSolver):
    SOLVER_NAME = 'infeasible-candidate'
    SOLVER_VERSION = '0.1.0'

    def _run(self, problem, config):
        return CbqmSolveOutcome(status='feasible', best_sample=[0, 0])


class _NonJsonOutcomeSolver(BaseCbqmSolver):
    SOLVER_NAME = 'non-json-outcome'
    SOLVER_VERSION = '0.1.0'

    def _run(self, problem, config):
        return CbqmSolveOutcome(
            status='feasible',
            best_sample=[1, 0],
            metadata={'unsupported': object()},
        )


class _TrustedTraceSemanticsSolver(BaseCbqmSolver):
    SOLVER_NAME = 'bad-trace'
    SOLVER_VERSION = '0.1.0'

    def _run(self, problem, config):
        return CbqmSolveOutcome(
            status='feasible',
            best_sample=[1, 0],
            trace=[
                {
                    'step': 0,
                    'time_seconds': 0,
                    'sample': [1, 0],
                    'objective': 123,
                },
            ],
        )


class _InvalidSampleSolver(BaseCbqmSolver):
    SOLVER_NAME = 'invalid-sample'
    SOLVER_VERSION = '0.1.0'

    def _run(self, problem, config):
        return CbqmSolveOutcome(status='feasible', best_sample=[1])


class BaseCbqmSolverTests(unittest.TestCase):
    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseCbqmSolver()

    def test_recomputes_candidate_and_trace_from_original_cbqm(self):
        result = _ConcreteSolver().solve(_problem())

        self.assertEqual('cbqm-result.v1', result['schema'])
        self.assertEqual(-2, result['best_objective'])
        self.assertTrue(result['feasibility']['feasible'])
        self.assertEqual(
            {
                'step': 0,
                'time_seconds': 0,
                'sample': [0, 0],
                'objective': 0,
                'feasible': False,
                'metadata': {'phase': 'initial'},
            },
            result['trace'][0],
        )

    def test_defensively_copies_problem_and_nested_config(self):
        problem = _problem()
        original_problem = copy.deepcopy(problem)
        config = {'nested': {}}

        _ConcreteSolver().solve(problem, config)

        self.assertEqual(original_problem, problem)
        self.assertEqual({'nested': {}}, config)

    def test_requires_candidate_for_conclusive_positive_status(self):
        with self.assertRaisesRegex(ValueError, 'requires a best_sample'):
            _MissingCandidateSolver().solve(_problem())

    def test_rejects_feasible_status_for_constraint_violating_candidate(self):
        with self.assertRaisesRegex(ValueError, 'constraint-feasible'):
            _InfeasibleCandidateSolver().solve(_problem())

    def test_kernel_cannot_supply_trusted_trace_semantics(self):
        with self.assertRaisesRegex(ValueError, 'unknown fields: objective'):
            _TrustedTraceSemanticsSolver().solve(_problem())

    def test_invalid_kernel_sample_fails_at_the_contract_boundary(self):
        with self.assertRaisesRegex(ValueError, 'length'):
            _InvalidSampleSolver().solve(_problem())

    def test_validates_the_complete_public_result(self):
        with self.assertRaisesRegex(TypeError, 'non-JSON value'):
            _NonJsonOutcomeSolver().solve(_problem())


if __name__ == '__main__':
    unittest.main()
