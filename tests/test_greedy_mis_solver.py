"""Construction and local-improvement tests for the native MIS baseline."""

import copy
import unittest

from lib.contracts import validate_mis_result
from lib.solvers.mis import GreedyMisSolver


def _problem():
    return {
        'schema': 'mis.v1',
        'problem_id': 'greedy-mis',
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


class GreedyMisSolverTests(unittest.TestCase):
    def test_returns_a_deterministic_contract_valid_feasible_incumbent(self):
        problem = _problem()
        solver = GreedyMisSolver()

        first = solver.solve(problem)
        second = solver.solve(problem)

        self.assertIsNone(validate_mis_result(problem, first))
        self.assertEqual('feasible', first['status'])
        self.assertEqual([0, 2], first['selected_vertices'])
        self.assertEqual(2, first['objective_value'])
        self.assertTrue(first['feasible'])
        self.assertNotIn('proof', first)
        self.assertEqual(
            first['selected_vertices'],
            second['selected_vertices'],
        )
        self.assertEqual(first['metrics'], second['metrics'])

    def test_supports_weighted_objective_and_skips_optional_nonpositive_values(self):
        problem = _problem()
        problem['objective'] = {'kind': 'maximum-weight'}
        weights = [-5, 8, 4, 3]
        problem['vertices'] = [
            {
                'index': index,
                'name': vertex['name'],
                'weight': weights[index],
            }
            for index, vertex in enumerate(problem['vertices'])
        ]

        result = GreedyMisSolver().solve(problem)

        self.assertEqual([1, 3], result['selected_vertices'])
        self.assertEqual(11, result['objective_value'])

    def test_fixed_values_are_honored_even_for_negative_forced_weight(self):
        problem = _problem()
        problem['objective'] = {'kind': 'maximum-weight'}
        weights = [-5, 8, 4, 3]
        problem['vertices'] = [
            {
                'index': index,
                'name': vertex['name'],
                'weight': weights[index],
            }
            for index, vertex in enumerate(problem['vertices'])
        ]
        problem['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 3, 'value': 0},
        ]

        result = GreedyMisSolver().solve(problem)

        self.assertEqual([0, 2], result['selected_vertices'])
        self.assertEqual(-1, result['objective_value'])
        self.assertTrue(result['feasible'])

    def test_adjacent_forced_vertices_return_a_direct_infeasibility_proof(self):
        problem = _problem()
        problem['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 1, 'value': 1},
        ]

        result = GreedyMisSolver().solve(problem)

        self.assertEqual('infeasible', result['status'])
        self.assertIsNone(result['selected_vertices'])
        self.assertEqual('infeasibility', result['proof']['claim'])
        self.assertEqual(
            'fixed-edge-conflict',
            result['proof']['kind'],
        )

    def test_pair_exchange_improves_a_greedy_local_trap(self):
        problem = {
            'schema': 'mis.v1',
            'problem_id': 'greedy-local-trap',
            'objective': {'kind': 'maximum-cardinality'},
            'vertices': [
                {'index': index, 'name': f'v{index}'}
                for index in range(6)
            ],
            'edges': [
                [0, 2],
                [0, 4],
                [1, 3],
                [1, 5],
                [2, 5],
                [3, 4],
                [3, 5],
            ],
            'fixed_values': [],
            'metadata': {},
        }

        constructed = GreedyMisSolver().solve(
            problem,
            {'max_local_passes': 0},
        )
        improved = GreedyMisSolver().solve(problem)

        self.assertEqual([0, 1], constructed['selected_vertices'])
        self.assertEqual([1, 2, 4], improved['selected_vertices'])
        self.assertEqual(3, improved['objective_value'])
        self.assertEqual(1, improved['metrics']['improving_exchanges'])

    def test_pair_scan_limit_never_mislabels_a_partial_scan_as_local_optimum(self):
        problem = _problem()

        result = GreedyMisSolver().solve(
            problem,
            {'max_pair_evaluations': 0},
        )

        self.assertEqual(
            'local_search_limit_reached',
            result['termination_reason'],
        )
        self.assertFalse(result['metadata']['local_optimum_reached'])
        self.assertTrue(result['metrics']['pair_scan_truncated'])

    def test_zero_vertex_problem_returns_the_empty_feasible_set(self):
        problem = {
            'schema': 'mis.v1',
            'problem_id': 'empty-greedy-mis',
            'objective': {'kind': 'maximum-cardinality'},
            'vertices': [],
            'edges': [],
            'fixed_values': [],
            'metadata': {},
        }

        result = GreedyMisSolver().solve(problem)

        self.assertEqual('feasible', result['status'])
        self.assertEqual([], result['selected_vertices'])
        self.assertEqual(0, result['objective_value'])

    def test_rejects_unknown_and_invalid_configuration(self):
        solver = GreedyMisSolver()
        problem = _problem()

        with self.assertRaisesRegex(ValueError, 'Unknown greedy MIS'):
            solver.solve(problem, {'local_passes': 1})
        with self.assertRaisesRegex(ValueError, 'max_local_passes'):
            solver.solve(problem, {'max_local_passes': True})
        with self.assertRaisesRegex(ValueError, 'max_local_passes'):
            solver.solve(problem, {'max_local_passes': -1})
        with self.assertRaisesRegex(ValueError, 'max_pair_evaluations'):
            solver.solve(problem, {'max_pair_evaluations': True})

    def test_does_not_mutate_problem_or_configuration(self):
        problem = _problem()
        original = copy.deepcopy(problem)
        config = {'max_local_passes': 2}

        GreedyMisSolver().solve(problem, config)

        self.assertEqual(original, problem)
        self.assertEqual({'max_local_passes': 2}, config)


if __name__ == '__main__':
    unittest.main()
