"""Correctness and termination tests for native exact CBQM enumeration."""

import copy
import unittest

from lib.contracts import validate_cbqm_result
from lib.solvers.cbqm import ExactCbqmSolver
from tests.oracles.cbqm import enumerate_cbqm_feasible


class _ImmediatelyTimingOutExactSolver(ExactCbqmSolver):
    """Deterministic timeout seam independent of machine load."""

    def _deadline_reached(self, deadline):
        return True


def _selection_problem():
    """Two feasible assignments with a unique minimizing optimum."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'exact-cbqm-selection',
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


class ExactCbqmSolverTests(unittest.TestCase):
    def test_finds_and_describes_a_proven_global_optimum(self):
        problem = _selection_problem()
        result = ExactCbqmSolver().solve(problem)

        self.assertIsNone(validate_cbqm_result(problem, result))
        self.assertEqual('optimal', result['status'])
        self.assertEqual([1, 0], result['best_sample'])
        self.assertEqual(-2, result['best_objective'])
        self.assertTrue(result['feasibility']['feasible'])
        self.assertEqual({
            'primal_bound': -2,
            'dual_bound': -2,
            'absolute_gap': 0,
            'relative_gap': 0,
        }, result['bounds'])
        self.assertEqual('optimality', result['proof']['claim'])
        self.assertFalse(result['proof']['independently_verified'])
        self.assertEqual('search_exhausted', result['termination_reason'])
        self.assertEqual({
            'candidates_evaluated': 4,
            'feasible_candidates': 2,
            'total_candidates': 4,
            'search_space_exhausted': True,
        }, result['metrics'])

    def test_matches_independent_oracle_for_minimize_and_maximize(self):
        for sense in ('minimize', 'maximize'):
            with self.subTest(sense=sense):
                problem = _selection_problem()
                problem['objective']['sense'] = sense
                problem['objective']['offset'] = 0.25
                problem['objective']['quadratic'] = [[0, 1, 2.5]]
                expected = enumerate_cbqm_feasible(problem)[0]

                result = ExactCbqmSolver().solve(problem)

                self.assertEqual(expected['sample'], result['best_sample'])
                self.assertEqual(
                    expected['objective_exact'],
                    result['best_objective'],
                )

    def test_equal_objectives_choose_lexicographically_smallest_sample(self):
        problem = _selection_problem()
        problem['objective']['linear'] = []

        result = ExactCbqmSolver().solve(problem)

        self.assertEqual([0, 1], result['best_sample'])

    def test_fixed_values_reduce_enumerated_search_without_losing_proof(self):
        problem = _selection_problem()
        problem['fixed_values'] = [{'index': 0, 'value': 0}]

        result = ExactCbqmSolver().solve(problem)

        self.assertEqual([0, 1], result['best_sample'])
        self.assertEqual(2, result['metrics']['total_candidates'])
        self.assertEqual(2, result['metrics']['candidates_evaluated'])

    def test_proves_an_empty_feasible_set(self):
        problem = _selection_problem()
        problem['fixed_values'] = [
            {'index': 0, 'value': 0},
            {'index': 1, 'value': 0},
        ]

        result = ExactCbqmSolver().solve(problem)

        self.assertEqual('infeasible', result['status'])
        self.assertIsNone(result['best_sample'])
        self.assertIsNone(result['best_objective'])
        self.assertIsNone(result['feasibility'])
        self.assertEqual('infeasibility', result['proof']['claim'])
        self.assertEqual(0, result['metrics']['feasible_candidates'])
        self.assertEqual(1, result['metrics']['total_candidates'])

    def test_solves_a_zero_variable_feasible_model(self):
        problem = {
            'schema': 'cbqm.v1',
            'problem_id': 'empty-cbqm',
            'variables': [],
            'objective': {
                'sense': 'minimize',
                'offset': 7,
                'linear': [],
                'quadratic': [],
            },
            'constraints': [],
            'fixed_values': [],
            'metadata': {},
        }

        result = ExactCbqmSolver().solve(problem)

        self.assertEqual('optimal', result['status'])
        self.assertEqual([], result['best_sample'])
        self.assertEqual(7, result['best_objective'])
        self.assertEqual(1, result['metrics']['candidates_evaluated'])

    def test_timeout_without_a_feasible_incumbent_returns_null_triplet(self):
        result = _ImmediatelyTimingOutExactSolver().solve(
            _selection_problem(),
            {'timeout_seconds': 1},
        )

        self.assertEqual('timeout', result['status'])
        self.assertIsNone(result['best_sample'])
        self.assertIsNone(result['best_objective'])
        self.assertIsNone(result['feasibility'])
        self.assertNotIn('bounds', result)
        self.assertNotIn('proof', result)
        self.assertEqual(1, result['metrics']['candidates_evaluated'])

    def test_timeout_preserves_a_feasible_incumbent_and_primal_bound(self):
        problem = _selection_problem()
        problem['constraints'] = []

        result = _ImmediatelyTimingOutExactSolver().solve(
            problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('timeout', result['status'])
        self.assertEqual([0, 0], result['best_sample'])
        self.assertTrue(result['feasibility']['feasible'])
        self.assertEqual(
            result['best_objective'],
            result['bounds']['primal_bound'],
        )

    def test_completed_last_candidate_wins_over_expired_deadline(self):
        problem = _selection_problem()
        problem['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 1, 'value': 0},
        ]

        result = _ImmediatelyTimingOutExactSolver().solve(
            problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('optimal', result['status'])
        self.assertTrue(result['metrics']['search_space_exhausted'])

    def test_exact_arithmetic_preserves_a_one_unit_large_integer_difference(self):
        problem = _selection_problem()
        problem['variables'] = [
            {'index': 0, 'name': 'x', 'vartype': 'BINARY'},
        ]
        problem['objective'] = {
            'sense': 'minimize',
            'offset': 10_000_000_000_000_000,
            'linear': [[0, -1]],
            'quadratic': [],
        }
        problem['constraints'] = []

        result = ExactCbqmSolver().solve(problem)

        self.assertEqual([1], result['best_sample'])
        self.assertEqual(
            9_999_999_999_999_999,
            result['best_objective'],
        )
        self.assertIs(type(result['best_objective']), int)

    def test_rejected_large_activity_never_overflows_public_json_conversion(self):
        problem = _selection_problem()
        problem['objective']['linear'] = []
        problem['constraints'] = [
            {
                'name': 'large_upper_bound',
                'family': 'capacity',
                'linear': [[0, 1e308], [1, 1e308]],
                'upper_bound': 1e308,
            },
        ]

        result = ExactCbqmSolver().solve(problem)

        self.assertEqual('optimal', result['status'])
        self.assertEqual([0, 0], result['best_sample'])
        self.assertEqual(3, result['metrics']['feasible_candidates'])

    def test_rejects_unknown_invalid_and_oversized_configuration(self):
        solver = ExactCbqmSolver()
        problem = _selection_problem()

        with self.assertRaisesRegex(ValueError, 'Unknown exact CBQM'):
            solver.solve(problem, {'timeuot_seconds': 1})
        with self.assertRaisesRegex(ValueError, 'max_variables'):
            solver.solve(problem, {'max_variables': True})
        with self.assertRaisesRegex(ValueError, 'timeout_seconds'):
            solver.solve(problem, {'timeout_seconds': 0})
        with self.assertRaisesRegex(ValueError, 'exceeding max_variables=1'):
            solver.solve(problem, {'max_variables': 1})

    def test_does_not_mutate_problem_or_nested_config(self):
        problem = _selection_problem()
        original = copy.deepcopy(problem)
        config = {'max_variables': 2}

        ExactCbqmSolver().solve(problem, config)

        self.assertEqual(original, problem)
        self.assertEqual({'max_variables': 2}, config)


if __name__ == '__main__':
    unittest.main()
