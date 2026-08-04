"""Input and cross-semantic result tests for native MIS contracts."""

import copy
import json
import unittest
from pathlib import Path

from lib.contracts import (
    evaluate_mis_solution,
    validate_mis,
    validate_mis_result,
)


_ROOT = Path(__file__).resolve().parents[1]


def _load_example(filename):
    path = _ROOT / 'contracts' / 'examples' / filename
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


class MisContractTests(unittest.TestCase):
    def setUp(self):
        self.problem = _load_example('mis.v1.example.json')
        self.result = _load_example('mis-result.v1.example.json')

    def test_checked_in_problem_and_result_are_valid(self):
        self.assertIsNone(validate_mis(self.problem))
        self.assertIsNone(validate_mis_result(self.problem, self.result))

    def test_cardinality_witness_metrics_and_edge_feasibility_are_recomputed(self):
        self.assertEqual(
            {
                'objective_value': 2,
                'cardinality': 2,
                'total_weight': 2,
                'feasible': True,
            },
            evaluate_mis_solution(self.problem, [0, 2]),
        )
        self.assertFalse(
            evaluate_mis_solution(self.problem, [0, 1])['feasible']
        )

        wrong = copy.deepcopy(self.result)
        wrong['objective_value'] = 3
        with self.assertRaisesRegex(ValueError, 'objective_value'):
            validate_mis_result(self.problem, wrong)

    def test_weighted_objective_requires_every_weight_and_preserves_large_values(self):
        problem = copy.deepcopy(self.problem)
        problem['objective']['kind'] = 'maximum-weight'
        for index, vertex in enumerate(problem['vertices']):
            vertex['weight'] = (
                10_000_000_000_000_000
                if index == 0
                else -1
            )

        self.assertEqual(
            9_999_999_999_999_999,
            evaluate_mis_solution(problem, [0, 2])['total_weight'],
        )

        missing = copy.deepcopy(problem)
        del missing['vertices'][1]['weight']
        with self.assertRaisesRegex(ValueError, 'every vertex'):
            validate_mis(missing)

        cardinality_with_weight = copy.deepcopy(problem)
        cardinality_with_weight['objective']['kind'] = 'maximum-cardinality'
        with self.assertRaisesRegex(ValueError, 'must omit weight'):
            validate_mis(cardinality_with_weight)

    def test_edges_and_fixed_values_have_one_canonical_order(self):
        reversed_edge = copy.deepcopy(self.problem)
        reversed_edge['edges'][0] = [1, 0]
        with self.assertRaisesRegex(ValueError, 'u < v'):
            validate_mis(reversed_edge)

        duplicate_edge = copy.deepcopy(self.problem)
        duplicate_edge['edges'].insert(1, [0, 1])
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_mis(duplicate_edge)

        unordered_edges = copy.deepcopy(self.problem)
        unordered_edges['edges'][0], unordered_edges['edges'][1] = (
            unordered_edges['edges'][1],
            unordered_edges['edges'][0],
        )
        with self.assertRaisesRegex(ValueError, 'lexicographically'):
            validate_mis(unordered_edges)

        duplicate_fixed = copy.deepcopy(self.problem)
        duplicate_fixed['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 0, 'value': 0},
        ]
        with self.assertRaisesRegex(ValueError, 'unique increasing'):
            validate_mis(duplicate_fixed)

    def test_selected_vertices_are_sorted_unique_integer_indices(self):
        for selected, message in (
            ([2, 0], 'strictly increasing'),
            ([0, 0], 'strictly increasing'),
            ([True], 'outside the vertex range'),
            ([4], 'outside the vertex range'),
        ):
            with self.subTest(selected=selected):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    evaluate_mis_solution(self.problem, selected)

    def test_candidate_fields_are_atomic_and_status_aware(self):
        fields = (
            'selected_vertices',
            'objective_value',
            'cardinality',
            'total_weight',
            'feasible',
        )
        for field_name in fields:
            with self.subTest(field_name=field_name):
                partial = copy.deepcopy(self.result)
                partial['status'] = 'timeout'
                partial[field_name] = None
                partial.pop('bounds')
                partial.pop('proof')
                partial['trace'] = []
                with self.assertRaisesRegex(ValueError, 'all be null'):
                    validate_mis_result(self.problem, partial)

        no_candidate = copy.deepcopy(self.result)
        no_candidate['status'] = 'timeout'
        for field_name in fields:
            no_candidate[field_name] = None
        no_candidate.pop('bounds')
        no_candidate.pop('proof')
        no_candidate['trace'] = []
        self.assertIsNone(validate_mis_result(self.problem, no_candidate))

        no_candidate['status'] = 'optimal'
        with self.assertRaisesRegex(ValueError, 'requires'):
            validate_mis_result(self.problem, no_candidate)

    def test_infeasible_status_requires_conflicting_fixed_in_vertices(self):
        infeasible = copy.deepcopy(self.problem)
        infeasible['fixed_values'] = [
            {'index': 0, 'value': 1},
            {'index': 1, 'value': 1},
        ]
        result = copy.deepcopy(self.result)
        result['problem_id'] = infeasible['problem_id']
        result['status'] = 'infeasible'
        for field_name in (
            'selected_vertices',
            'objective_value',
            'cardinality',
            'total_weight',
            'feasible',
        ):
            result[field_name] = None
        result.pop('bounds')
        result['proof']['claim'] = 'infeasibility'
        result['trace'] = []

        self.assertIsNone(validate_mis_result(infeasible, result))

        with self.assertRaisesRegex(ValueError, 'empty independent set'):
            validate_mis_result(self.problem, result)

    def test_bounds_and_proof_follow_maximization_semantics(self):
        wrong_order = copy.deepcopy(self.result)
        wrong_order['status'] = 'feasible'
        wrong_order.pop('proof')
        wrong_order['bounds']['optimum_upper_bound'] = 1
        with self.assertRaisesRegex(ValueError, 'cannot exceed'):
            validate_mis_result(self.problem, wrong_order)

        wrong_incumbent = copy.deepcopy(self.result)
        wrong_incumbent['status'] = 'feasible'
        wrong_incumbent.pop('proof')
        wrong_incumbent['bounds'] = {
            'incumbent_lower_bound': 1,
            'optimum_upper_bound': 2,
        }
        with self.assertRaisesRegex(ValueError, 'incumbent_lower_bound'):
            validate_mis_result(self.problem, wrong_incumbent)

        wrong_claim = copy.deepcopy(self.result)
        wrong_claim['proof']['claim'] = 'bound'
        with self.assertRaisesRegex(ValueError, 'requires proof claim'):
            validate_mis_result(self.problem, wrong_claim)

    def test_trace_is_closed_atomic_and_recomputed(self):
        partial = copy.deepcopy(self.result)
        del partial['trace'][0]['total_weight']
        with self.assertRaisesRegex(ValueError, 'must appear together'):
            validate_mis_result(self.problem, partial)

        wrong = copy.deepcopy(self.result)
        wrong['trace'][0]['cardinality'] = 3
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            validate_mis_result(self.problem, wrong)

        extra = copy.deepcopy(self.result)
        extra['trace'][0]['message'] = 'not-in-v1'
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_mis_result(self.problem, extra)

    def test_closed_objects_reject_non_json_and_non_finite_values(self):
        extra = copy.deepcopy(self.problem)
        extra['objective']['sense'] = 'maximize'
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_mis(extra)

        non_json = copy.deepcopy(self.result)
        non_json['metadata'] = {'callback': lambda: None}
        with self.assertRaisesRegex(TypeError, 'non-JSON value'):
            validate_mis_result(self.problem, non_json)

        non_finite = copy.deepcopy(self.result)
        non_finite['runtime_seconds'] = float('inf')
        with self.assertRaisesRegex(TypeError, 'finite real number'):
            validate_mis_result(self.problem, non_finite)


if __name__ == '__main__':
    unittest.main()
