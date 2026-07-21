import unittest

import pandas as pd

from lib.client_profile import client_profile
from lib.portfolio_constraints import portfolio_constraints
from lib.weight_encoding import weight_encoding


class PortfolioConstraintsTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'code': ['A', 'B', 'C', 'D', 'E', 'F'],
            'fund_manager': ['m1', 'm1', 'm2', 'm2', 'm3', 'm4'],
            'investment_type_secondary': [
                'money_market',
                'hybrid_bond_primary',
                'hybrid_bond_primary',
                'equity_hybrid',
                'equity',
                'commodity',
            ],
            'asset_class': [
                'cash_money',
                'fixed_income',
                'fixed_income',
                'mixed',
                'equity',
                'alternative',
            ],
            'risk_level': ['R1', 'R3', 'R3', 'R3-R4', 'R4-R5', 'R5'],
        })
        self.mapping = weight_encoding(self.df)['variable_mapping']
        self.template = client_profile('conservative')

    def test_builds_all_document_constraint_families(self):
        result = portfolio_constraints(
            self.df,
            self.template,
            self.mapping,
            similarity_groups=[['E', 'F']],
        )
        counts = result['summary']['constraints_by_family']

        self.assertEqual(1, counts['budget'])
        self.assertEqual(6, counts['one_weight_level'])
        self.assertEqual(1, counts['holding_count'])
        self.assertEqual(6, counts['single_asset_cap'])
        self.assertEqual(5, counts['asset_class'])
        self.assertEqual(1, counts['r5_cap'])
        self.assertEqual(2, counts['manager_cap'])
        self.assertEqual(1, counts['investment_type_cap'])
        self.assertEqual(1, counts['similarity_group_cap'])

    def test_budget_and_one_level_constraints_have_expected_coefficients(self):
        result = portfolio_constraints(self.df, self.template, self.mapping)
        by_name = {item['name']: item for item in result['constraints']}

        budget = by_name['budget']
        self.assertEqual(1.0, budget['lower_bound'])
        self.assertEqual(0.02, budget['terms']['x_0_1'])
        self.assertNotIn('x_0_0', budget['terms'])

        one_level = by_name['one_weight_level::A']
        self.assertEqual(8, len(one_level['terms']))
        self.assertTrue(all(value == 1.0 for value in one_level['terms'].values()))

    def test_r5_constraint_excludes_r4_r5_assets(self):
        result = portfolio_constraints(self.df, self.template, self.mapping)
        r5 = next(item for item in result['constraints'] if item['name'] == 'r5_cap')
        variables = set(r5['terms'])

        self.assertTrue(any(name.startswith('x_5_') for name in variables))
        self.assertFalse(any(name.startswith('x_4_') for name in variables))

    def test_marks_levels_above_single_asset_cap_as_fixed(self):
        template = client_profile('conservative')
        template['single_asset_cap'] = 0.10
        result = portfolio_constraints(self.df, template, self.mapping)

        self.assertEqual(12, len(result['fixed_variables']))
        self.assertIn('x_0_6', result['fixed_variables'])
        self.assertIn('x_0_7', result['fixed_variables'])

    def test_rejects_unknown_similarity_code(self):
        with self.assertRaisesRegex(ValueError, 'unknown codes'):
            portfolio_constraints(
                self.df,
                self.template,
                self.mapping,
                similarity_groups=[['A', 'UNKNOWN']],
            )

    def test_rejects_duplicate_variable_names(self):
        broken = self.mapping.copy()
        broken.loc[1, 'variable_name'] = broken.loc[0, 'variable_name']
        with self.assertRaisesRegex(ValueError, 'Variable names must be unique'):
            portfolio_constraints(self.df, self.template, broken)


if __name__ == '__main__':
    unittest.main()
