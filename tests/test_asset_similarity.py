import unittest

import numpy as np
import pandas as pd

from lib.asset_similarity import SIMILARITY_COLUMNS, asset_similarity


class AssetSimilarityTests(unittest.TestCase):
    def setUp(self):
        base = np.arange(1, len(SIMILARITY_COLUMNS) + 1, dtype=float)
        self.df = pd.DataFrame([
            dict(code='A', **dict(zip(SIMILARITY_COLUMNS, base))),
            dict(code='B', **dict(zip(SIMILARITY_COLUMNS, base))),
            dict(code='C', **dict(zip(SIMILARITY_COLUMNS, base[::-1]))),
        ])

    def test_returns_symmetric_matrix_pairs_groups_and_clusters(self):
        result = asset_similarity(self.df, high_threshold=0.80, low_threshold=0.20)
        matrix = result['similarity_matrix']

        self.assertEqual((3, 3), matrix.shape)
        np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)
        np.testing.assert_allclose(np.diag(matrix), np.ones(3))
        self.assertEqual(['A', 'B', 'C'], result['style_clusters'].index.tolist())
        self.assertIn(['A', 'B'], result['high_similarity_groups'])

    def test_handles_constant_and_missing_features(self):
        self.df['3y_sortino'] = 1.0
        self.df.loc[1, '1y_return_std'] = np.nan
        result = asset_similarity(self.df)
        self.assertTrue(np.isfinite(result['similarity_matrix'].to_numpy()).all())

    def test_rejects_duplicate_codes(self):
        self.df.loc[1, 'code'] = 'A'
        with self.assertRaisesRegex(ValueError, 'unique'):
            asset_similarity(self.df)

    def test_rejects_invalid_threshold_order(self):
        with self.assertRaisesRegex(ValueError, 'low_threshold'):
            asset_similarity(self.df, high_threshold=0.5, low_threshold=0.5)


if __name__ == '__main__':
    unittest.main()
