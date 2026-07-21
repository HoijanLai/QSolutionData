import unittest

import numpy as np
import pandas as pd

from lib.asset_scoring import (
    RETURN_COLUMNS,
    RISK_COLUMNS,
    STABILITY_COLUMNS,
    asset_scoring,
)


class AssetScoringTests(unittest.TestCase):
    def setUp(self):
        columns = RETURN_COLUMNS + RISK_COLUMNS + STABILITY_COLUMNS
        self.df = pd.DataFrame({
            column: [index + 1.0, index + 2.0, np.nan]
            for index, column in enumerate(columns)
        })
        self.df.loc[2] = self.df.loc[1] + 1
        self.df.loc[0, '1y_max_drawdown'] = -0.10
        self.df.loc[1, '1y_max_drawdown'] = -0.20
        self.df.loc[2, '1y_max_drawdown'] = -0.30

    def test_adds_three_finite_scores_without_mutating_input(self):
        original = self.df.copy(deep=True)
        result, weights = asset_scoring(self.df)

        self.assertTrue(self.df.equals(original))
        self.assertEqual(
            {'return_score', 'risk_score', 'stability_score'},
            set(result.columns).difference(self.df.columns),
        )
        self.assertTrue(np.isfinite(result['return_score']).all())
        self.assertAlmostEqual(sum(weights['return'].values()), 1.0)

    def test_larger_absolute_drawdown_has_larger_risk_score(self):
        result, _ = asset_scoring(self.df)
        self.assertLess(result.loc[0, 'risk_score'], result.loc[2, 'risk_score'])

    def test_rejects_missing_column(self):
        with self.assertRaisesRegex(ValueError, 'Missing scoring columns'):
            asset_scoring(self.df.drop(columns=['6m_return']))

    def test_rejects_unknown_strategy(self):
        with self.assertRaises(NotImplementedError):
            asset_scoring(self.df, strategy='rank')


if __name__ == '__main__':
    unittest.main()
