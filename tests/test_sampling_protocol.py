"""Structural typing tests for the representation-neutral Sampler protocol."""

import unittest
from typing import is_typeddict

from lib import Sampler
from lib.contracts import (
    BinarySampleRecord,
    BinarySampleSet,
    validate_binary_sample_set,
)


class SamplingProtocolTests(unittest.TestCase):
    def test_binary_sample_contract_types_are_field_aware(self):
        self.assertTrue(is_typeddict(BinarySampleRecord))
        self.assertTrue(is_typeddict(BinarySampleSet))

    def test_sampler_exposes_only_the_generic_sample_call_shape(self):
        annotations = Sampler.sample.__annotations__

        self.assertIn('problem', annotations)
        self.assertIn('config', annotations)
        self.assertIn('return', annotations)

    def test_runtime_validator_is_public(self):
        self.assertTrue(callable(validate_binary_sample_set))


if __name__ == '__main__':
    unittest.main()
