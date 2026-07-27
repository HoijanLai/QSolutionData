import unittest

import pandas as pd

from lib.preprocessing.asset_classification import asset_classification


class AssetClassificationTests(unittest.TestCase):
    def test_classifies_chinese_and_english_categories(self):
        df = pd.DataFrame({
            'investment_type_secondary': [
                '货币市场型基金',
                'hybrid_bond_primary',
                'equity_hybrid',
                '增强指数型基金',
                'commodity',
            ]
        })
        result = asset_classification(df)

        self.assertEqual(
            ['cash_money', 'fixed_income', 'mixed', 'equity', 'alternative'],
            result['asset_class'].astype('string').tolist(),
        )
        self.assertEqual(
            ['R1', 'R3', 'R3-R4', 'R4-R5', 'R5'],
            result['risk_level'].astype('string').tolist(),
        )

    def test_unknown_category_raises_by_default(self):
        df = pd.DataFrame({'investment_type_secondary': ['new_category']})
        with self.assertRaisesRegex(ValueError, 'Unknown investment type'):
            asset_classification(df)

    def test_unknown_category_can_be_kept_as_unknown(self):
        df = pd.DataFrame({'investment_type_secondary': ['new_category']})
        result = asset_classification(df, unknown_policy='keep')
        self.assertEqual('unknown', result.loc[0, 'asset_class'])
        self.assertEqual('unknown', result.loc[0, 'risk_level'])


if __name__ == '__main__':
    unittest.main()
