"""Focused checks for the public, schema-aligned CBQM validator."""

import copy
import unittest

from lib.contracts import validate_cbqm
from lib.contracts.validation import _validate_cbqm


def _cbqm():
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'validation-cbqm',
        'variables': [
            {
                'index': 0,
                'name': 'x',
                'vartype': 'BINARY',
                'metadata': {},
            },
        ],
        'objective': {
            'sense': 'minimize',
            'offset': 0.0,
            'linear': [[0, -1.0]],
            'quadratic': [],
        },
        'constraints': [
            {
                'name': 'select',
                'family': 'selection',
                'linear': [[0, 1.0]],
                'lower_bound': 1.0,
                'metadata': {},
            },
        ],
        'fixed_values': [],
        'metadata': {},
    }


class CbqmContractValidationTests(unittest.TestCase):
    def test_public_validator_and_compatibility_alias_are_identical(self):
        self.assertIs(_validate_cbqm, validate_cbqm)
        self.assertIsNone(validate_cbqm(_cbqm()))

    def test_rejects_explicit_null_optional_number(self):
        problem = _cbqm()
        problem['constraints'][0]['upper_bound'] = None

        with self.assertRaisesRegex(TypeError, 'finite real number'):
            validate_cbqm(problem)

    def test_rejects_unknown_nested_field_and_non_object_metadata(self):
        unknown = copy.deepcopy(_cbqm())
        unknown['objective']['future_field'] = 1
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            validate_cbqm(unknown)

        invalid_metadata = copy.deepcopy(_cbqm())
        invalid_metadata['variables'][0]['metadata'] = []
        with self.assertRaisesRegex(TypeError, 'must be an object'):
            validate_cbqm(invalid_metadata)

    def test_rejects_boolean_variable_index(self):
        problem = _cbqm()
        # Python considers ``False == 0``. The contract nevertheless requires
        # a genuine JSON integer and must not silently normalize a boolean.
        problem['variables'][0]['index'] = False

        with self.assertRaisesRegex(ValueError, 'outside the variable range'):
            validate_cbqm(problem)


if __name__ == '__main__':
    unittest.main()
