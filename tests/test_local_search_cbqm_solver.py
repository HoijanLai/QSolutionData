"""Behavioral tests for the native feasibility-first CBQM heuristic."""

import copy
import unittest

from lib.contracts import validate_cbqm_result
from lib.solvers.cbqm import LocalSearchCbqmSolver
from problem.benchmarks import build_cbqm_mathematical_suite


class _ImmediatelyTimingOutLocalSearch(LocalSearchCbqmSolver):
    """Deterministic deadline seam independent of machine speed."""

    def _deadline_reached(self, deadline):
        return True


class _TimingOutDuringLocalImprovement(LocalSearchCbqmSolver):
    """Stop after repair has already produced a feasible incumbent."""

    def _best_improving_neighbor(
        self,
        problem,
        prepared,
        sample,
        max_pair_evaluations,
        deadline,
        metrics,
    ):
        return None, True, False


def _selection_problem():
    """Return a two-variable equality-constrained minimization model."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'local-search-selection',
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


def _paired_move_problem():
    """Make the only improving feasible move require two simultaneous flips."""
    problem = _selection_problem()
    problem['problem_id'] = 'local-search-paired-move'
    problem['objective']['linear'] = [[0, -1], [1, -1]]
    problem['constraints'] = [
        {
            'name': 'bits_are_equal',
            'family': 'linking',
            'linear': [[0, 1], [1, -1]],
            'lower_bound': 0,
            'upper_bound': 0,
        },
    ]
    return problem


class LocalSearchCbqmSolverTests(unittest.TestCase):
    def test_repairs_constraints_and_improves_the_original_objective(self):
        problem = _selection_problem()

        result = LocalSearchCbqmSolver().solve(
            problem,
            {'max_restarts': 0},
        )

        self.assertIsNone(validate_cbqm_result(problem, result))
        self.assertEqual('feasible', result['status'])
        self.assertEqual([1, 0], result['best_sample'])
        self.assertEqual(-2, result['best_objective'])
        self.assertTrue(result['feasibility']['feasible'])
        self.assertNotIn('proof', result)
        self.assertEqual(
            result['best_objective'],
            result['bounds']['primal_bound'],
        )

    def test_pair_moves_connect_feasible_points_separated_by_infeasibility(self):
        problem = _paired_move_problem()
        solver = LocalSearchCbqmSolver()

        singles_only = solver.solve(
            problem,
            {
                'max_restarts': 0,
                'max_pair_evaluations': 0,
            },
        )
        with_pairs = solver.solve(
            problem,
            {
                'max_restarts': 0,
                'max_pair_evaluations': 1,
            },
        )

        self.assertEqual([0, 0], singles_only['best_sample'])
        self.assertEqual(0, singles_only['best_objective'])
        self.assertEqual([1, 1], with_pairs['best_sample'])
        self.assertEqual(-2, with_pairs['best_objective'])
        self.assertGreater(with_pairs['metrics']['pair_neighbors_evaluated'], 0)

    def test_truncated_pair_scan_does_not_claim_a_local_optimum(self):
        problem = _paired_move_problem()

        result = LocalSearchCbqmSolver().solve(
            problem,
            {
                'max_restarts': 0,
                'max_pair_evaluations': 0,
                'include_trace': True,
            },
        )

        self.assertTrue(result['metrics']['pair_scan_truncated'])
        self.assertFalse(
            result['trace'][0]['metadata']['local_optimum_reached']
        )

    def test_supports_maximization_and_quadratic_objectives(self):
        problem = _selection_problem()
        problem['constraints'] = []
        problem['objective'] = {
            'sense': 'maximize',
            'offset': 0.25,
            'linear': [[0, 2], [1, 1]],
            'quadratic': [[0, 1, 4]],
        }

        result = LocalSearchCbqmSolver().solve(
            problem,
            {'max_restarts': 0},
        )

        self.assertEqual([1, 1], result['best_sample'])
        self.assertEqual(7.25, result['best_objective'])

    def test_never_changes_fixed_values_during_repair_or_local_search(self):
        problem = _selection_problem()
        problem['fixed_values'] = [{'index': 0, 'value': 0}]

        result = LocalSearchCbqmSolver().solve(problem)

        self.assertEqual([0, 1], result['best_sample'])
        self.assertTrue(result['feasibility']['feasible'])

    def test_failed_repair_is_unknown_and_never_false_infeasibility(self):
        problem = _selection_problem()
        problem['fixed_values'] = [
            {'index': 0, 'value': 0},
            {'index': 1, 'value': 0},
        ]

        result = LocalSearchCbqmSolver().solve(
            problem,
            {
                'max_restarts': 2,
                'max_repair_steps': 10,
            },
        )

        self.assertEqual('unknown', result['status'])
        self.assertIsNone(result['best_sample'])
        self.assertIsNone(result['best_objective'])
        self.assertIsNone(result['feasibility'])
        self.assertNotIn('proof', result)
        self.assertEqual(0, result['metrics']['feasible_starts'])

    def test_timeout_without_incumbent_returns_a_null_candidate_triplet(self):
        problem = _selection_problem()

        result = _ImmediatelyTimingOutLocalSearch().solve(
            problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('timeout', result['status'])
        self.assertEqual('timeout_reached', result['termination_reason'])
        self.assertIsNone(result['best_sample'])
        self.assertIsNone(result['best_objective'])
        self.assertIsNone(result['feasibility'])
        self.assertNotIn('bounds', result)

    def test_timeout_preserves_a_feasible_incumbent_and_primal_bound(self):
        problem = _selection_problem()
        problem['constraints'] = []

        result = _TimingOutDuringLocalImprovement().solve(
            problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('timeout', result['status'])
        self.assertIsNotNone(result['best_sample'])
        self.assertTrue(result['feasibility']['feasible'])
        self.assertEqual(
            result['best_objective'],
            result['bounds']['primal_bound'],
        )

    def test_seeded_runs_have_identical_search_decisions_and_accounting(self):
        problem = _selection_problem()
        config = {
            'seed': 41,
            'max_restarts': 5,
            'include_trace': False,
        }

        first = LocalSearchCbqmSolver().solve(problem, config)
        second = LocalSearchCbqmSolver().solve(problem, config)

        self.assertEqual(first['best_sample'], second['best_sample'])
        self.assertEqual(first['best_objective'], second['best_objective'])
        self.assertEqual(first['metrics'], second['metrics'])
        self.assertEqual(first['metadata'], second['metadata'])

    def test_optional_trace_is_canonicalized_by_the_base_solver(self):
        problem = _selection_problem()

        result = LocalSearchCbqmSolver().solve(
            problem,
            {
                'max_restarts': 0,
                'include_trace': True,
            },
        )

        self.assertEqual(1, len(result['trace']))
        observation = result['trace'][0]
        self.assertEqual(result['best_sample'], observation['sample'])
        self.assertEqual(result['best_objective'], observation['objective'])
        self.assertTrue(observation['feasible'])
        self.assertEqual(0, observation['step'])

    def test_zero_variable_feasible_model_returns_empty_witness(self):
        problem = {
            'schema': 'cbqm.v1',
            'problem_id': 'empty-local-search-cbqm',
            'variables': [],
            'objective': {
                'sense': 'minimize',
                'offset': 3,
                'linear': [],
                'quadratic': [],
            },
            'constraints': [],
            'fixed_values': [],
            'metadata': {},
        }

        result = LocalSearchCbqmSolver().solve(problem)

        self.assertEqual('feasible', result['status'])
        self.assertEqual([], result['best_sample'])
        self.assertEqual(3, result['best_objective'])

    def test_exact_arithmetic_preserves_large_integer_objective_order(self):
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

        result = LocalSearchCbqmSolver().solve(
            problem,
            {'max_restarts': 0},
        )

        self.assertEqual([1], result['best_sample'])
        self.assertEqual(
            9_999_999_999_999_999,
            result['best_objective'],
        )
        self.assertIs(type(result['best_objective']), int)

    def test_all_mathematical_cbqm_fixtures_yield_feasible_results(self):
        solver = LocalSearchCbqmSolver()
        for name, problem in build_cbqm_mathematical_suite().items():
            with self.subTest(name=name):
                result = solver.solve(
                    problem,
                    {
                        'seed': 0,
                        'max_restarts': 4,
                    },
                )

                self.assertEqual('feasible', result['status'])
                self.assertTrue(result['feasibility']['feasible'])

    def test_rejects_unknown_or_type_confused_configuration(self):
        solver = LocalSearchCbqmSolver()
        problem = _selection_problem()

        with self.assertRaisesRegex(ValueError, 'Unknown local-search CBQM'):
            solver.solve(problem, {'timeuot_seconds': 1})
        with self.assertRaisesRegex(ValueError, 'seed'):
            solver.solve(problem, {'seed': True})
        with self.assertRaisesRegex(ValueError, 'max_restarts'):
            solver.solve(problem, {'max_restarts': -1})
        with self.assertRaisesRegex(ValueError, 'max_repair_steps'):
            solver.solve(problem, {'max_repair_steps': 1.5})
        with self.assertRaisesRegex(ValueError, 'timeout_seconds'):
            solver.solve(problem, {'timeout_seconds': 0})
        with self.assertRaisesRegex(ValueError, 'include_trace'):
            solver.solve(problem, {'include_trace': 1})

    def test_does_not_mutate_problem_or_nested_configuration(self):
        problem = _selection_problem()
        original = copy.deepcopy(problem)
        config = {
            'seed': 9,
            'max_restarts': 1,
            'include_trace': True,
        }
        original_config = copy.deepcopy(config)

        LocalSearchCbqmSolver().solve(problem, config)

        self.assertEqual(original, problem)
        self.assertEqual(original_config, config)


if __name__ == '__main__':
    unittest.main()
