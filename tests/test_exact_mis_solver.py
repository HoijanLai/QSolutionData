"""Correctness, proof and termination tests for native exact MIS search."""

import copy
from fractions import Fraction
import unittest

from lib.contracts import validate_mis_result
from lib.solvers.mis import ExactMisSolver
from tests.oracles.mis import enumerate_mis


class _ImmediatelyTimingOutExactSolver(ExactMisSolver):
    """Deterministic timeout seam independent of machine speed."""

    def _deadline_reached(self, deadline):
        return True


def _cardinality_problem():
    """A four-vertex path with two tied maximum independent sets."""
    return {
        'schema': 'mis.v1',
        'problem_id': 'exact-mis-path',
        'objective': {'kind': 'maximum-cardinality'},
        'vertices': [
            {'index': 0, 'name': 'a'},
            {'index': 1, 'name': 'b'},
            {'index': 2, 'name': 'c'},
            {'index': 3, 'name': 'd'},
        ],
        'edges': [[0, 1], [1, 2], [2, 3]],
        'fixed_values': [],
        'metadata': {},
    }


def _weighted_problem():
    """A weighted path whose best cardinality and weight choices differ."""
    problem = _cardinality_problem()
    problem['problem_id'] = 'exact-mwis-path'
    problem['objective'] = {'kind': 'maximum-weight'}
    weights = [3, 8, 4, 3]
    problem['vertices'] = [
        {
            'index': index,
            'name': vertex['name'],
            'weight': weights[index],
        }
        for index, vertex in enumerate(problem['vertices'])
    ]
    return problem


class ExactMisSolverTests(unittest.TestCase):
    def test_finds_and_describes_a_proven_global_optimum(self):
        problem = _cardinality_problem()

        result = ExactMisSolver().solve(problem)

        self.assertIsNone(validate_mis_result(problem, result))
        self.assertEqual('optimal', result['status'])
        self.assertEqual([0, 2], result['selected_vertices'])
        self.assertEqual(2, result['objective_value'])
        self.assertEqual(2, result['cardinality'])
        self.assertEqual(2, result['total_weight'])
        self.assertTrue(result['feasible'])
        self.assertEqual({
            'incumbent_lower_bound': 2,
            'optimum_upper_bound': 2,
        }, result['bounds'])
        self.assertEqual('optimality', result['proof']['claim'])
        self.assertEqual('branch-and-reduce', result['proof']['kind'])
        self.assertFalse(result['proof']['independently_verified'])
        self.assertEqual(
            'search_tree_exhausted',
            result['termination_reason'],
        )
        self.assertTrue(result['metrics']['search_space_exhausted'])

    def test_matches_independent_oracle_for_cardinality_and_weight(self):
        for problem in (_cardinality_problem(), _weighted_problem()):
            with self.subTest(kind=problem['objective']['kind']):
                expected = enumerate_mis(problem)[0]

                result = ExactMisSolver().solve(problem)

                self.assertEqual(
                    expected['selected_vertices'],
                    result['selected_vertices'],
                )
                self.assertEqual(
                    expected['objective_exact'],
                    Fraction(result['objective_value']),
                )

    def test_negative_optional_weights_are_safely_excluded(self):
        problem = _weighted_problem()
        problem['vertices'] = [
            {'index': 0, 'name': 'a', 'weight': -1},
            {'index': 1, 'name': 'b', 'weight': -2},
            {'index': 2, 'name': 'c', 'weight': -3},
            {'index': 3, 'name': 'd', 'weight': -4},
        ]

        result = ExactMisSolver().solve(problem)

        self.assertEqual([], result['selected_vertices'])
        self.assertEqual(0, result['objective_value'])
        self.assertEqual(4, result['metrics']['reductions_applied'])

    def test_zero_weight_vertices_preserve_lexicographic_tie_breaking(self):
        problem = _weighted_problem()
        problem['vertices'] = [
            {'index': 0, 'name': 'zero', 'weight': 0},
            {'index': 1, 'name': 'low', 'weight': -1},
            {'index': 2, 'name': 'forced', 'weight': 5},
            {'index': 3, 'name': 'unused', 'weight': -1},
        ]
        problem['edges'] = []
        problem['fixed_values'] = [{'index': 2, 'value': 1}]

        result = ExactMisSolver().solve(problem)

        self.assertEqual([0, 2], result['selected_vertices'])
        self.assertEqual(
            enumerate_mis(problem)[0]['selected_vertices'],
            result['selected_vertices'],
        )

    def test_fixed_values_reduce_domain_and_preserve_forced_negative_weight(self):
        problem = _weighted_problem()
        problem['vertices'][0]['weight'] = -5
        problem['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 3, 'value': 0},
        ]

        result = ExactMisSolver().solve(problem)

        self.assertEqual([0, 2], result['selected_vertices'])
        self.assertEqual(-1, result['objective_value'])
        self.assertEqual(
            enumerate_mis(problem)[0]['selected_vertices'],
            result['selected_vertices'],
        )

    def test_adjacent_forced_in_vertices_prove_infeasibility(self):
        problem = _cardinality_problem()
        problem['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 1, 'value': 1},
        ]

        result = ExactMisSolver().solve(problem)

        self.assertEqual('infeasible', result['status'])
        self.assertIsNone(result['selected_vertices'])
        self.assertIsNone(result['objective_value'])
        self.assertIsNone(result['cardinality'])
        self.assertIsNone(result['total_weight'])
        self.assertIsNone(result['feasible'])
        self.assertEqual('infeasibility', result['proof']['claim'])
        self.assertEqual(
            'fixed-edge-conflict',
            result['proof']['kind'],
        )
        self.assertEqual(0, result['metrics']['nodes_expanded'])

    def test_solves_zero_vertex_and_edgeless_models_by_reduction(self):
        empty = {
            'schema': 'mis.v1',
            'problem_id': 'empty-mis',
            'objective': {'kind': 'maximum-cardinality'},
            'vertices': [],
            'edges': [],
            'fixed_values': [],
            'metadata': {},
        }
        edgeless = _cardinality_problem()
        edgeless['edges'] = []

        empty_result = ExactMisSolver().solve(empty)
        edgeless_result = ExactMisSolver().solve(edgeless)

        self.assertEqual([], empty_result['selected_vertices'])
        self.assertEqual(0, empty_result['objective_value'])
        self.assertEqual([0, 1, 2, 3], edgeless_result['selected_vertices'])
        self.assertEqual(4, edgeless_result['metrics']['reductions_applied'])
        self.assertEqual(1, edgeless_result['metrics']['nodes_expanded'])

    def test_complete_graph_tie_chooses_lowest_vertex_index(self):
        problem = _cardinality_problem()
        problem['edges'] = [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
        ]

        result = ExactMisSolver().solve(problem)

        self.assertEqual([0], result['selected_vertices'])

    def test_exact_arithmetic_preserves_one_unit_large_integer_difference(self):
        problem = _weighted_problem()
        problem['vertices'] = [
            {
                'index': 0,
                'name': 'large',
                'weight': 10_000_000_000_000_000,
            },
            {
                'index': 1,
                'name': 'larger',
                'weight': 10_000_000_000_000_001,
            },
        ]
        problem['edges'] = [[0, 1]]

        result = ExactMisSolver().solve(problem)

        self.assertEqual([1], result['selected_vertices'])
        self.assertEqual(
            10_000_000_000_000_001,
            result['objective_value'],
        )
        self.assertIs(type(result['objective_value']), int)

    def test_timeout_returns_feasible_incumbent_and_valid_global_bounds(self):
        problem = _cardinality_problem()

        result = _ImmediatelyTimingOutExactSolver().solve(
            problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('timeout', result['status'])
        self.assertEqual([], result['selected_vertices'])
        self.assertTrue(result['feasible'])
        self.assertEqual({
            'incumbent_lower_bound': 0,
            'optimum_upper_bound': 4,
        }, result['bounds'])
        self.assertFalse(result['metrics']['search_space_exhausted'])

    def test_fully_fixed_model_completes_before_deadline_check(self):
        problem = _cardinality_problem()
        problem['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 1, 'value': 0},
            {'index': 2, 'value': 1},
            {'index': 3, 'value': 0},
        ]

        result = _ImmediatelyTimingOutExactSolver().solve(
            problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('optimal', result['status'])
        self.assertEqual([0, 2], result['selected_vertices'])

    def test_rejects_unknown_invalid_and_oversized_configuration(self):
        solver = ExactMisSolver()
        problem = _cardinality_problem()

        with self.assertRaisesRegex(ValueError, 'Unknown exact MIS'):
            solver.solve(problem, {'timeuot_seconds': 1})
        with self.assertRaisesRegex(ValueError, 'max_vertices'):
            solver.solve(problem, {'max_vertices': True})
        with self.assertRaisesRegex(ValueError, 'timeout_seconds'):
            solver.solve(problem, {'timeout_seconds': 0})
        with self.assertRaisesRegex(ValueError, 'exceeding max_vertices=3'):
            solver.solve(problem, {'max_vertices': 3})

    def test_does_not_mutate_problem_or_nested_config(self):
        problem = _weighted_problem()
        original = copy.deepcopy(problem)
        config = {'max_vertices': 4}

        ExactMisSolver().solve(problem, config)

        self.assertEqual(original, problem)
        self.assertEqual({'max_vertices': 4}, config)


if __name__ == '__main__':
    unittest.main()
