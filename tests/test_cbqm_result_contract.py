"""Cross-semantic tests for the native ``cbqm-result.v1`` contract."""

import copy
import json
import unittest
from pathlib import Path

from lib.contracts import (
    evaluate_cbqm_feasibility,
    evaluate_cbqm_objective,
    validate_cbqm_result,
)


_ROOT = Path(__file__).resolve().parents[1]


def _load_example(filename):
    """Load a checked-in JSON contract example as an independent document."""
    path = _ROOT / 'contracts' / 'examples' / filename
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _problem_with_all_constraint_shapes():
    """Build a tiny model whose witness violates fixed/lower/upper bounds."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'constraint-shapes',
        'variables': [
            {'index': 0, 'name': 'x', 'vartype': 'BINARY'},
            {'index': 1, 'name': 'y', 'vartype': 'BINARY'},
        ],
        'objective': {
            'sense': 'maximize',
            'offset': 10_000_000_000_000_000,
            'linear': [[0, -1], [1, 0.25]],
            'quadratic': [[0, 1, 2.5]],
        },
        'constraints': [
            {
                'name': 'need_two',
                'family': 'lower',
                'linear': [[0, 1], [1, 1]],
                'lower_bound': 2,
            },
            {
                'name': 'allow_none',
                'family': 'upper',
                'linear': [[0, 0.5], [1, 0.5]],
                'upper_bound': 0,
            },
        ],
        'fixed_values': [{'index': 0, 'value': 1}],
        'metadata': {},
    }


class CbqmResultContractTests(unittest.TestCase):
    def setUp(self):
        self.problem = _load_example('cbqm.v1.example.json')
        self.result = _load_example('cbqm-result.v1.example.json')

    def test_checked_in_example_is_cross_semantically_valid(self):
        self.assertIsNone(validate_cbqm_result(self.problem, self.result))

    def test_public_evaluators_return_original_objective_and_feasibility(self):
        self.assertEqual(-2, evaluate_cbqm_objective(self.problem, [1, 0]))
        self.assertEqual(
            {
                'feasible': True,
                'violated_count': 0,
                'max_violation': 0,
                'violations': [],
            },
            evaluate_cbqm_feasibility(self.problem, [1, 0]),
        )

        problem = _problem_with_all_constraint_shapes()
        self.assertEqual(
            9_999_999_999_999_999,
            evaluate_cbqm_objective(problem, [1, 0]),
        )
        self.assertEqual(
            {
                'feasible': False,
                'violated_count': 2,
                'max_violation': 1,
                'violations': [
                    {
                        'constraint_name': 'need_two',
                        'activity': 1,
                        'lower_bound': 2,
                        'magnitude': 1,
                    },
                    {
                        'constraint_name': 'allow_none',
                        'activity': 0.5,
                        'upper_bound': 0,
                        'magnitude': 0.5,
                    },
                ],
            },
            evaluate_cbqm_feasibility(problem, [1, 0]),
        )

        # Fixed-value failures are reported first, followed by explicit
        # constraints in source order.  That makes result diffs reproducible.
        violation = evaluate_cbqm_feasibility(problem, [0, 1])
        self.assertEqual(
            ['fixed:x', 'need_two', 'allow_none'],
            [
                item['constraint_name']
                for item in violation['violations']
            ],
        )

    def test_candidate_triplet_is_atomic_and_status_aware(self):
        for field_name in ('best_sample', 'best_objective', 'feasibility'):
            with self.subTest(field_name=field_name):
                partial = copy.deepcopy(self.result)
                partial['status'] = 'timeout'
                partial[field_name] = None
                with self.assertRaisesRegex(ValueError, 'all be'):
                    validate_cbqm_result(self.problem, partial)

        no_incumbent = copy.deepcopy(self.result)
        no_incumbent.update({
            'status': 'timeout',
            'best_sample': None,
            'best_objective': None,
            'feasibility': None,
        })
        no_incumbent.pop('bounds')
        no_incumbent.pop('proof')
        no_incumbent['trace'] = []
        self.assertIsNone(validate_cbqm_result(self.problem, no_incumbent))

        for status in ('optimal', 'feasible'):
            with self.subTest(status=status):
                missing = copy.deepcopy(no_incumbent)
                missing['status'] = status
                with self.assertRaisesRegex(ValueError, 'requires'):
                    validate_cbqm_result(self.problem, missing)

        infeasible = copy.deepcopy(no_incumbent)
        infeasible['status'] = 'infeasible'
        infeasible['proof'] = {
            'claim': 'infeasibility',
            'kind': 'exhaustive-enumeration',
            'producer': 'contract-test',
            'independently_verified': False,
        }
        self.assertIsNone(validate_cbqm_result(self.problem, infeasible))

        invalid_infeasible = copy.deepcopy(self.result)
        invalid_infeasible['status'] = 'infeasible'
        invalid_infeasible['proof']['claim'] = 'infeasibility'
        with self.assertRaisesRegex(ValueError, 'cannot include'):
            validate_cbqm_result(self.problem, invalid_infeasible)

    def test_timeout_may_report_an_infeasible_but_fully_verified_candidate(self):
        result = copy.deepcopy(self.result)
        result.update({
            'status': 'timeout',
            'best_sample': [0, 0],
            'best_objective': 0,
            'feasibility': {
                'feasible': False,
                'violated_count': 1,
                'max_violation': 1,
                'violations': [
                    {
                        'constraint_name': 'select_one',
                        'activity': 0,
                        'lower_bound': 1,
                        'upper_bound': 1,
                        'magnitude': 1,
                    },
                ],
            },
        })
        result.pop('bounds')
        result.pop('proof')
        result['trace'] = []

        self.assertIsNone(validate_cbqm_result(self.problem, result))

    def test_candidate_objective_and_feasibility_are_recomputed_exactly(self):
        wrong_objective = copy.deepcopy(self.result)
        wrong_objective['best_objective'] = -1.5
        with self.assertRaisesRegex(ValueError, 'recomputation'):
            validate_cbqm_result(self.problem, wrong_objective)

        rounded_large_integer = _problem_with_all_constraint_shapes()
        result = copy.deepcopy(self.result)
        result.update({
            'problem_id': rounded_large_integer['problem_id'],
            'status': 'timeout',
            'best_sample': [1, 0],
            'best_objective': 1e16,
            'feasibility': evaluate_cbqm_feasibility(
                rounded_large_integer,
                [1, 0],
            ),
            'trace': [],
        })
        result.pop('bounds')
        result.pop('proof')
        with self.assertRaisesRegex(ValueError, 'recomputation'):
            validate_cbqm_result(rounded_large_integer, result)

        wrong_violation = copy.deepcopy(result)
        wrong_violation['best_objective'] = 9_999_999_999_999_999
        wrong_violation['feasibility']['violations'][0]['magnitude'] = 0.5
        wrong_violation['feasibility']['max_violation'] = 0.5
        with self.assertRaisesRegex(
            ValueError,
            'does not match (canonical )?recomputation',
        ):
            validate_cbqm_result(rounded_large_integer, wrong_violation)

    def test_bounds_follow_objective_sense_and_candidate(self):
        wrong_primal = copy.deepcopy(self.result)
        wrong_primal['bounds']['primal_bound'] = -1
        wrong_primal['bounds']['dual_bound'] = -1
        with self.assertRaisesRegex(ValueError, 'primal_bound'):
            validate_cbqm_result(self.problem, wrong_primal)

        wrong_order = copy.deepcopy(self.result)
        wrong_order['status'] = 'feasible'
        wrong_order.pop('proof')
        wrong_order['bounds'] = {
            'primal_bound': -2,
            'dual_bound': -1,
        }
        with self.assertRaisesRegex(ValueError, 'objective sense'):
            validate_cbqm_result(self.problem, wrong_order)

        wrong_gap = copy.deepcopy(self.result)
        wrong_gap['bounds']['absolute_gap'] = 1
        with self.assertRaisesRegex(ValueError, 'canonical recomputation'):
            validate_cbqm_result(self.problem, wrong_gap)

    def test_proof_claim_must_agree_with_conclusive_status(self):
        wrong_optimality_claim = copy.deepcopy(self.result)
        wrong_optimality_claim['proof']['claim'] = 'bound'
        with self.assertRaisesRegex(ValueError, 'requires proof claim'):
            validate_cbqm_result(self.problem, wrong_optimality_claim)

        infeasible = copy.deepcopy(self.result)
        infeasible.update({
            'status': 'infeasible',
            'best_sample': None,
            'best_objective': None,
            'feasibility': None,
        })
        infeasible.pop('bounds')
        infeasible['trace'] = []
        with self.assertRaisesRegex(ValueError, 'infeasibility'):
            validate_cbqm_result(self.problem, infeasible)

    def test_trace_candidate_fields_travel_together_and_are_recomputed(self):
        partial = copy.deepcopy(self.result)
        del partial['trace'][0]['feasible']
        with self.assertRaisesRegex(ValueError, 'must appear together'):
            validate_cbqm_result(self.problem, partial)

        wrong_objective = copy.deepcopy(self.result)
        wrong_objective['trace'][0]['objective'] = 123
        with self.assertRaisesRegex(ValueError, 'recomputation'):
            validate_cbqm_result(self.problem, wrong_objective)

        wrong_feasibility = copy.deepcopy(self.result)
        wrong_feasibility['trace'][0]['feasible'] = False
        with self.assertRaisesRegex(ValueError, 'recomputation'):
            validate_cbqm_result(self.problem, wrong_feasibility)

    def test_result_objects_are_closed_finite_and_json_serializable(self):
        extra = copy.deepcopy(self.result)
        extra['artifact_id'] = 'not-in-v1'
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_cbqm_result(self.problem, extra)

        invalid_runtime = copy.deepcopy(self.result)
        invalid_runtime['runtime_seconds'] = float('inf')
        with self.assertRaisesRegex(TypeError, 'finite real number'):
            validate_cbqm_result(self.problem, invalid_runtime)

        invalid_metadata = copy.deepcopy(self.result)
        invalid_metadata['metadata'] = {'callback': lambda: None}
        with self.assertRaisesRegex(TypeError, 'non-JSON value'):
            validate_cbqm_result(self.problem, invalid_metadata)

        invalid_violation = copy.deepcopy(self.result)
        invalid_violation['status'] = 'timeout'
        invalid_violation['best_sample'] = [0, 0]
        invalid_violation['best_objective'] = 0
        invalid_violation['feasibility'] = {
            'feasible': False,
            'violated_count': 1,
            'max_violation': 0,
            'violations': [
                {
                    'constraint_name': 'select_one',
                    'activity': 0,
                    'lower_bound': 1,
                    'upper_bound': 1,
                    'magnitude': 0,
                },
            ],
        }
        invalid_violation.pop('bounds')
        invalid_violation.pop('proof')
        invalid_violation['trace'] = []
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            validate_cbqm_result(self.problem, invalid_violation)


if __name__ == '__main__':
    unittest.main()
