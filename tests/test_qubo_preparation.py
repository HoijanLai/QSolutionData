import unittest

import numpy as np
import pandas as pd

from lib.preprocessing.asset_scoring import RETURN_COLUMNS, RISK_COLUMNS, STABILITY_COLUMNS
from lib.preprocessing.asset_similarity import SIMILARITY_COLUMNS
from lib.pipeline.qubo_preparation import prepare_qubo_inputs
from lib.portfolio.cbqm_builder import build_portfolio_cbqm


class QuboPreparationTests(unittest.TestCase):
    def setUp(self):
        codes = ['A', 'B', 'C', 'D', 'E', 'F']
        investment_types = [
            'money_market',
            'hybrid_bond_primary',
            'hybrid_bond_secondary',
            'equity_hybrid',
            'equity',
            'commodity',
        ]
        metric_columns = sorted(
            set(RETURN_COLUMNS + RISK_COLUMNS + STABILITY_COLUMNS + SIMILARITY_COLUMNS)
        )

        rows = []
        for row_index, (code, investment_type) in enumerate(
            zip(codes, investment_types)
        ):
            row = {
                'code': code,
                'investment_type_secondary': investment_type,
            }
            for column_index, column in enumerate(metric_columns):
                value = (row_index + 1) * (column_index + 1) / 100
                if column.endswith('max_drawdown'):
                    value = -value
                row[column] = value
            rows.append(row)
        self.df = pd.DataFrame(rows)

        self.manager_df = pd.DataFrame({
            'code': list(reversed(codes)),
            'fund_manager': ['m4', 'm3', 'm2', 'm2', 'm1', 'm1'],
            'security_name': [f'Fund {code}' for code in reversed(codes)],
        })

    def test_builds_complete_aligned_payload(self):
        original_df = self.df.copy(deep=True)
        original_managers = self.manager_df.copy(deep=True)
        payload = prepare_qubo_inputs(
            self.df,
            manager_df=self.manager_df,
            profile='conservative',
        )

        self.assertTrue(self.df.equals(original_df))
        self.assertTrue(self.manager_df.equals(original_managers))
        self.assertEqual(6, payload['metadata']['asset_count'])
        self.assertEqual(48, payload['metadata']['variable_count'])
        self.assertEqual((6, 6), payload['similarity_matrix'].shape)
        self.assertEqual(['A', 'B', 'C', 'D', 'E', 'F'], payload['assets']['code'].tolist())
        self.assertEqual('Fund A', payload['assets'].loc[0, 'security_name'])
        self.assertEqual('m1', payload['assets'].loc[0, 'fund_manager'])
        self.assertTrue(
            np.isfinite(
                payload['assets'][
                    ['return_score', 'risk_score', 'stability_score']
                ].to_numpy(dtype=float)
            ).all()
        )
        self.assertGreater(payload['constraint_summary']['constraint_count'], 0)

    def test_accepts_manager_column_inside_asset_data(self):
        df = self.df.copy()
        df['fund_manager'] = ['m1', 'm1', 'm2', 'm2', 'm3', 'm4']
        payload = prepare_qubo_inputs(df, profile='conservative')
        self.assertEqual(6, len(payload['assets']))

    def test_prepared_payload_builds_valid_cbqm(self):
        payload = prepare_qubo_inputs(
            self.df,
            manager_df=self.manager_df,
            profile='conservative',
        )
        objective_config = {
            'score_coefficients': {
                'return_score': -1.0,
                'risk_score': 1.0,
                'stability_score': -1.0,
            },
            'similarity_coefficient': 1.0,
            'similarity_transform': 'raw',
        }

        problem = build_portfolio_cbqm(
            payload,
            objective_config,
            problem_id='prepared-portfolio',
        )

        self.assertEqual('cbqm.v1', problem['schema'])
        self.assertEqual(48, len(problem['variables']))
        self.assertEqual(
            payload['constraint_summary']['constraint_count'],
            len(problem['constraints']),
        )
        self.assertGreater(len(problem['objective']['quadratic']), 0)

    def test_rejects_missing_manager_code(self):
        managers = self.manager_df.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, 'codes do not match'):
            prepare_qubo_inputs(
                self.df,
                manager_df=managers,
                profile='conservative',
            )

    def test_requires_manager_data(self):
        with self.assertRaisesRegex(ValueError, 'Provide manager_df'):
            prepare_qubo_inputs(self.df, profile='conservative')

    def test_rejects_unknown_constraint_strategy(self):
        with self.assertRaises(NotImplementedError):
            prepare_qubo_inputs(
                self.df,
                manager_df=self.manager_df,
                profile='conservative',
                constraint_strategy='qubo_penalty',
            )


if __name__ == '__main__':
    unittest.main()
