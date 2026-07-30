"""Schema-aligned semantic tests for ``binary-sample-set.v1``."""

import copy
import json
import unittest
from pathlib import Path

from lib.contracts import validate_binary_sample_set


_ROOT = Path(__file__).resolve().parents[1]


def _example():
    """Load a fresh copy of the checked-in sample-set example."""
    path = (
        _ROOT
        / 'contracts'
        / 'examples'
        / 'binary-sample-set.v1.example.json'
    )
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


class BinarySampleSetContractTests(unittest.TestCase):
    def test_checked_in_empirical_example_is_valid_and_source_bindable(self):
        sample_set = _example()

        self.assertIsNone(validate_binary_sample_set(sample_set))
        self.assertIsNone(
            validate_binary_sample_set(
                sample_set,
                expected_problem_id='two-asset-example',
                expected_variable_count=2,
            )
        )

    def test_exact_distribution_uses_integer_normalization_and_conserves_it(self):
        sample_set = _example()
        sample_set['distribution_kind'] = 'exact'
        sample_set['shots'] = 10
        sample_set['records'][0]['occurrences'] = 1
        sample_set['records'][1]['occurrences'] = 3
        sample_set['records'][2]['occurrences'] = 6

        self.assertIsNone(validate_binary_sample_set(sample_set))

        sample_set['shots'] = 9
        with self.assertRaisesRegex(ValueError, r'sum\(occurrences\)'):
            validate_binary_sample_set(sample_set)

    def test_deterministic_mode_is_one_non_shot_based_sample(self):
        sample_set = _example()
        sample_set.update({
            'distribution_kind': 'deterministic',
            'shots': None,
            'records': [
                {
                    'sample': [1, 0],
                    'occurrences': 1,
                },
            ],
            'seed': None,
        })

        self.assertIsNone(validate_binary_sample_set(sample_set))

        two_records = copy.deepcopy(sample_set)
        two_records['records'].append({
            'sample': [1, 1],
            'occurrences': 1,
        })
        with self.assertRaisesRegex(ValueError, 'exactly one record'):
            validate_binary_sample_set(two_records)

        counted = copy.deepcopy(sample_set)
        counted['shots'] = 1
        with self.assertRaisesRegex(ValueError, 'shots to be null'):
            validate_binary_sample_set(counted)

    def test_records_must_be_aggregated_and_strictly_lexicographic(self):
        duplicate = _example()
        duplicate['records'][1]['sample'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'distinct samples'):
            validate_binary_sample_set(duplicate)

        descending = _example()
        descending['records'][0], descending['records'][1] = (
            descending['records'][1],
            descending['records'][0],
        )
        with self.assertRaisesRegex(ValueError, 'lexicographic'):
            validate_binary_sample_set(descending)

    def test_samples_have_exact_width_and_genuine_binary_integers(self):
        wrong_width = _example()
        wrong_width['records'][0]['sample'] = [0]
        with self.assertRaisesRegex(ValueError, 'length'):
            validate_binary_sample_set(wrong_width)

        boolean_bit = _example()
        boolean_bit['records'][0]['sample'] = [False, 0]
        with self.assertRaisesRegex(ValueError, 'Python integer binary'):
            validate_binary_sample_set(boolean_bit)

    def test_shots_occurrences_and_seed_reject_boolean_lookalikes(self):
        for field, mutate, message in (
            (
                'shots',
                lambda value: value.update({'shots': True}),
                'non-negative integer',
            ),
            (
                'occurrences',
                lambda value: value['records'][0].update(
                    {'occurrences': True}
                ),
                'non-negative integer',
            ),
            (
                'seed',
                lambda value: value.update({'seed': True}),
                'integer or null',
            ),
        ):
            with self.subTest(field=field):
                invalid = _example()
                mutate(invalid)
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    message,
                ):
                    validate_binary_sample_set(invalid)

    def test_empty_or_zero_weight_distributions_are_rejected(self):
        empty = _example()
        empty['records'] = []
        empty['shots'] = 0
        with self.assertRaisesRegex(ValueError, 'must not be empty'):
            validate_binary_sample_set(empty)

        zero_occurrence = _example()
        zero_occurrence['records'][0]['occurrences'] = 0
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            validate_binary_sample_set(zero_occurrence)

    def test_header_can_be_cross_checked_against_the_source_problem(self):
        with self.assertRaisesRegex(ValueError, 'expected_problem_id'):
            validate_binary_sample_set(
                _example(),
                expected_problem_id='another-problem',
            )
        with self.assertRaisesRegex(ValueError, 'expected_variable_count'):
            validate_binary_sample_set(
                _example(),
                expected_variable_count=3,
            )
        with self.assertRaisesRegex(ValueError, 'non-empty string or None'):
            validate_binary_sample_set(
                _example(),
                expected_problem_id=7,
            )

    def test_objects_are_closed_finite_and_strictly_json_serializable(self):
        extra_record_field = _example()
        extra_record_field['records'][0]['objective'] = -99
        with self.assertRaisesRegex(ValueError, 'unknown fields: objective'):
            validate_binary_sample_set(extra_record_field)

        non_finite_runtime = _example()
        non_finite_runtime['runtime_seconds'] = float('nan')
        with self.assertRaisesRegex(TypeError, 'finite real number'):
            validate_binary_sample_set(non_finite_runtime)

        non_json_metadata = _example()
        non_json_metadata['metadata'] = {'values': {1, 2}}
        with self.assertRaisesRegex(TypeError, 'non-JSON value'):
            validate_binary_sample_set(non_json_metadata)

        unknown_kind = _example()
        unknown_kind['distribution_kind'] = 'quantum-ish'
        with self.assertRaisesRegex(ValueError, 'Unknown distribution_kind'):
            validate_binary_sample_set(unknown_kind)


if __name__ == '__main__':
    unittest.main()
