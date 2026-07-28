import copy
import json
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from lib.portfolio.cbqm_builder import build_portfolio_cbqm


class PortfolioCbqmBuilderTests(unittest.TestCase):
    def setUp(self):
        self.assets = pd.DataFrame({
            'code': ['A', 'B'],
            'security_name': ['Fund A', 'Fund B'],
            'fund_manager': ['manager_a', 'manager_b'],
            'investment_type_secondary': ['equity', 'bond'],
            'asset_class': ['equity', 'fixed_income'],
            'risk_level': ['R4', 'R2'],
            'return_score': [1.0, 2.0],
            'risk_score': [2.0, 1.0],
            'stability_score': [0.5, 1.0],
            'style_cluster': [0, 1],
        })
        self.mapping = pd.DataFrame([
            {
                'variable_index': 0,
                'variable_name': 'x_0_0',
                'asset_index': 0,
                'level_index': 0,
                'code': 'A',
                'weight_level': 0.0,
            },
            {
                'variable_index': 1,
                'variable_name': 'x_0_1',
                'asset_index': 0,
                'level_index': 1,
                'code': 'A',
                'weight_level': 0.5,
            },
            {
                'variable_index': 2,
                'variable_name': 'x_1_0',
                'asset_index': 1,
                'level_index': 0,
                'code': 'B',
                'weight_level': 0.0,
            },
            {
                'variable_index': 3,
                'variable_name': 'x_1_1',
                'asset_index': 1,
                'level_index': 1,
                'code': 'B',
                'weight_level': 0.5,
            },
        ])
        self.payload = {
            'assets': self.assets,
            'similarity_matrix': pd.DataFrame(
                [[1.0, 0.4], [0.4, 1.0]],
                index=['A', 'B'],
                columns=['A', 'B'],
            ),
            'variable_mapping': self.mapping,
            'constraints': [
                {
                    'name': 'budget',
                    'family': 'budget',
                    'type': 'equality',
                    'terms': {'x_0_1': 0.5, 'x_1_1': 0.5},
                    'lower_bound': 1.0,
                    'upper_bound': 1.0,
                }
            ],
            'fixed_variables': ['x_1_1'],
            'metadata': {
                'profile': 'test',
                'asset_count': 2,
                'variable_count': 4,
            },
        }
        self.objective_config = {
            'strategy': 'score_similarity',
            'sense': 'minimize',
            'offset': 0.0,
            'score_coefficients': {
                'return_score': -2.0,
                'risk_score': 1.0,
                'stability_score': -1.0,
            },
            'similarity_coefficient': 3.0,
            'similarity_transform': 'raw',
        }

    def test_builds_cbqm_with_expected_linear_and_quadratic_terms(self):
        problem = build_portfolio_cbqm(
            self.payload,
            self.objective_config,
            problem_id='portfolio-test',
        )

        self.assertEqual(problem['schema'], 'cbqm.v1')
        self.assertEqual(problem['problem_id'], 'portfolio-test')
        self.assertEqual(
            problem['objective']['linear'],
            [[1, -0.25], [3, -2.0]],
        )
        self.assertEqual(
            problem['objective']['quadratic'],
            [[1, 3, 0.30000000000000004]],
        )
        self.assertEqual(problem['fixed_values'], [{'index': 3, 'value': 0}])
        self.assertEqual(
            problem['constraints'][0]['linear'],
            [[1, 0.5], [3, 0.5]],
        )
        self.assertEqual(problem['variables'][1]['kind'], 'allocation_level')
        self.assertEqual(
            problem['variables'][1]['metadata']['weight_level'],
            0.5,
        )

    def test_does_not_mutate_payload_or_objective_config(self):
        original_payload = copy.deepcopy(self.payload)
        original_config = copy.deepcopy(self.objective_config)

        build_portfolio_cbqm(self.payload, self.objective_config)

        assert_frame_equal(self.payload['assets'], original_payload['assets'])
        assert_frame_equal(
            self.payload['variable_mapping'],
            original_payload['variable_mapping'],
        )
        assert_frame_equal(
            self.payload['similarity_matrix'],
            original_payload['similarity_matrix'],
        )
        self.assertEqual(self.payload['constraints'], original_payload['constraints'])
        self.assertEqual(self.objective_config, original_config)

    def test_positive_only_transform_omits_negative_similarity(self):
        self.payload['similarity_matrix'].loc['A', 'B'] = -0.4
        self.payload['similarity_matrix'].loc['B', 'A'] = -0.4
        self.objective_config['similarity_transform'] = 'positive_only'

        problem = build_portfolio_cbqm(self.payload, self.objective_config)

        self.assertEqual(problem['objective']['quadratic'], [])

    def test_absolute_transform_penalizes_negative_similarity_magnitude(self):
        self.payload['similarity_matrix'].loc['A', 'B'] = -0.4
        self.payload['similarity_matrix'].loc['B', 'A'] = -0.4
        self.objective_config['similarity_transform'] = 'absolute'

        problem = build_portfolio_cbqm(self.payload, self.objective_config)

        self.assertAlmostEqual(problem['objective']['quadratic'][0][2], 0.3)

    def test_requires_explicit_objective_config(self):
        with self.assertRaisesRegex(TypeError, 'explicit mapping'):
            build_portfolio_cbqm(self.payload, None)

    def test_rejects_misaligned_similarity_matrix(self):
        self.payload['similarity_matrix'] = self.payload[
            'similarity_matrix'
        ].reindex(index=['B', 'A'])

        with self.assertRaisesRegex(ValueError, 'index is not aligned'):
            build_portfolio_cbqm(self.payload, self.objective_config)

    def test_result_is_json_serializable(self):
        problem = build_portfolio_cbqm(self.payload, self.objective_config)

        serialized = json.dumps(problem, ensure_ascii=False)

        self.assertIn('portfolio-test', serialized)


if __name__ == '__main__':
    unittest.main()
