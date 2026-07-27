import unittest

import pandas as pd

from lib.portfolio.weight_encoding import DEFAULT_WEIGHT_LEVELS, weight_encoding


class WeightEncodingTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'code': ['A', 'B']})

    def test_builds_eight_variables_per_asset(self):
        result = weight_encoding(self.df)
        mapping = result['variable_mapping']

        self.assertEqual(16, result['variable_count'])
        self.assertEqual(16, len(mapping))
        self.assertEqual(DEFAULT_WEIGHT_LEVELS, result['weight_levels'])
        self.assertEqual('x_0_0', mapping.iloc[0]['variable_name'])
        self.assertEqual(0.15, mapping.iloc[-1]['weight_level'])
        self.assertEqual('B', result['weight_lookup']['x_1_7']['code'])

    def test_rejects_duplicate_codes(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            weight_encoding(pd.DataFrame({'code': ['A', 'A']}))

    def test_rejects_unsorted_levels(self):
        with self.assertRaisesRegex(ValueError, 'sorted'):
            weight_encoding(self.df, levels=(0, 0.04, 0.02))

    def test_rejects_levels_without_zero(self):
        with self.assertRaisesRegex(ValueError, 'include 0'):
            weight_encoding(self.df, levels=(0.02, 0.04))


if __name__ == '__main__':
    unittest.main()
