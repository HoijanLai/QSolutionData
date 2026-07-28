import itertools
import unittest

from lib.compilers.qubo_compiler import compile_qubo


def _evaluate_qubo(problem, sample):
    energy = problem['offset']
    for left, right, coefficient in problem['terms']:
        energy += coefficient * sample[left] * sample[right]
    return energy


def _minimum_energy_for_prefix(problem, prefix):
    slack_count = problem['num_variables'] - len(prefix)
    return min(
        _evaluate_qubo(problem, [*prefix, *slack])
        for slack in itertools.product((0, 1), repeat=slack_count)
    )


class QuboCompilerTests(unittest.TestCase):
    def setUp(self):
        self.problem = {
            'schema': 'cbqm.v1',
            'problem_id': 'compiler-example',
            'variables': [
                {'index': 0, 'name': 'x0', 'vartype': 'BINARY'},
                {'index': 1, 'name': 'x1', 'vartype': 'BINARY'},
            ],
            'objective': {
                'sense': 'minimize',
                'offset': 0.0,
                'linear': [[0, -2.0], [1, -1.0]],
                'quadratic': [],
            },
            'constraints': [
                {
                    'name': 'select_one',
                    'family': 'budget',
                    'linear': [[0, 1.0], [1, 1.0]],
                    'lower_bound': 1.0,
                    'upper_bound': 1.0,
                }
            ],
            'fixed_values': [],
            'metadata': {},
        }

    def test_compiles_equality_to_expected_qubo(self):
        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': 4.0},
        )

        self.assertEqual(qubo['schema'], 'qubo.v1')
        self.assertEqual(qubo['sense'], 'minimize')
        self.assertEqual(qubo['num_variables'], 2)
        self.assertEqual(qubo['offset'], 4.0)
        self.assertEqual(
            qubo['terms'],
            [[0, 0, -6.0], [0, 1, 8.0], [1, 1, -5.0]],
        )
        self.assertEqual(context['schema'], 'qubo-compilation-context.v1')
        self.assertEqual(context['constraints'][0]['penalty'], 4.0)
        self.assertEqual(
            context['constraints'][0]['encodings'][0]['status'],
            'encoded',
        )
        self.assertTrue(
            context['constraints'][0]['normalization']['exact_reconstruction']
        )
        self.assertEqual(
            context['constraints'][0]['normalization'][
                'max_abs_reconstruction_error'
            ],
            0.0,
        )
        self.assertTrue(context['equivalence']['exact_projection_certified'])
        self.assertTrue(context['equivalence']['arithmetic_exact'])
        self.assertEqual(context['dropped_qubo_terms'], [])

    def test_upper_bound_binary_slack_has_exact_minimum_penalty(self):
        self.problem['objective']['linear'] = []
        self.problem['constraints'][0].pop('lower_bound')
        penalty = 3.0

        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': penalty},
        )

        self.assertEqual(qubo['num_variables'], 3)
        self.assertEqual(len(context['slack_variables']), 1)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [0, 0]), 0.0)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [1, 0]), 0.0)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [0, 1]), 0.0)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [1, 1]), penalty)

    def test_lower_bound_binary_slack_has_exact_minimum_penalty(self):
        self.problem['objective']['linear'] = []
        self.problem['constraints'][0].pop('upper_bound')
        penalty = 3.0

        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': penalty},
        )

        self.assertEqual(qubo['num_variables'], 3)
        self.assertEqual(len(context['slack_variables']), 1)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [0, 0]), penalty)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [1, 0]), 0.0)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [0, 1]), 0.0)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [1, 1]), 0.0)

    def test_range_uses_independent_lower_and_upper_slack(self):
        self.problem['variables'].append(
            {'index': 2, 'name': 'x2', 'vartype': 'BINARY'}
        )
        self.problem['objective']['linear'] = []
        constraint = self.problem['constraints'][0]
        constraint['linear'].append([2, 1.0])
        constraint['upper_bound'] = 2.0
        penalty = 5.0

        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': penalty},
        )

        encodings = context['constraints'][0]['encodings']
        self.assertEqual([item['side'] for item in encodings], ['lower', 'upper'])
        self.assertEqual(_minimum_energy_for_prefix(qubo, [0, 0, 0]), penalty)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [1, 0, 0]), 0.0)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [1, 1, 0]), 0.0)
        self.assertEqual(_minimum_energy_for_prefix(qubo, [1, 1, 1]), penalty)

    def test_constraint_coefficients_use_reduced_integer_lattice(self):
        constraint = self.problem['constraints'][0]
        constraint['linear'] = [[0, 0.5], [1, 0.5]]

        _, context = compile_qubo(
            self.problem,
            {'default_penalty': 4.0},
        )

        compiled = context['constraints'][0]
        self.assertEqual(compiled['normalization']['lattice_unit'], 0.5)
        self.assertEqual(
            compiled['encodings'][0]['integer_terms'],
            [[0, 1], [1, 1]],
        )
        self.assertEqual(compiled['encodings'][0]['integer_rhs'], 2)

    def test_eliminates_fixed_values_from_objective_and_constraints(self):
        self.problem['fixed_values'] = [{'index': 0, 'value': 1}]
        self.problem['objective']['offset'] = 1.0
        self.problem['objective']['linear'] = [[0, 2.0], [1, 1.0]]
        self.problem['objective']['quadratic'] = [[0, 1, 3.0]]

        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': 5.0},
        )

        self.assertEqual(qubo['variable_names'], ['x1'])
        self.assertEqual(qubo['offset'], 3.0)
        self.assertEqual(qubo['terms'], [[0, 0, 9.0]])
        self.assertEqual(context['fixed_values'], [{'index': 0, 'value': 1}])
        self.assertEqual(context['free_variables'][0]['cbqm_index'], 1)

    def test_negates_maximization_objective_before_adding_penalties(self):
        self.problem['objective']['sense'] = 'maximize'
        self.problem['constraints'] = []

        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': 4.0},
        )

        self.assertEqual(qubo['terms'], [[0, 0, 2.0], [1, 1, 1.0]])
        self.assertEqual(context['objective_multiplier'], -1.0)

    def test_constraint_penalty_override_precedence_is_explicit(self):
        qubo, context = compile_qubo(
            self.problem,
            {
                'default_penalty': 2.0,
                'penalty_by_family': {'budget': 3.0},
                'penalty_by_constraint': {'select_one': 7.0},
            },
        )

        self.assertEqual(context['constraints'][0]['penalty'], 7.0)
        self.assertEqual(qubo['offset'], 7.0)

    def test_rejects_infeasible_constraint_after_fixed_substitution(self):
        self.problem['fixed_values'] = [
            {'index': 0, 'value': 0},
            {'index': 1, 'value': 0},
        ]

        with self.assertRaisesRegex(ValueError, 'infeasible'):
            compile_qubo(self.problem, {'default_penalty': 4.0})

    def test_rejects_constraint_rounding_beyond_tolerance(self):
        self.problem['constraints'][0]['linear'][0][1] = 1 / 3

        with self.assertRaisesRegex(ValueError, 'rounding tolerance'):
            compile_qubo(
                self.problem,
                {
                    'default_penalty': 4.0,
                    'constraint_precision': 2,
                    'rounding_tolerance': 1e-4,
                },
            )

    def test_accepted_constraint_rounding_disables_exact_projection(self):
        self.problem['constraints'][0]['linear'][0][1] = 1 / 3

        _, context = compile_qubo(
            self.problem,
            {
                'default_penalty': 4.0,
                'constraint_precision': 2,
                'rounding_tolerance': 0.01,
            },
        )

        normalization = context['constraints'][0]['normalization']
        self.assertFalse(normalization['exact_reconstruction'])
        self.assertGreater(normalization['max_abs_reconstruction_error'], 0.0)
        self.assertFalse(context['equivalence']['constraint_lattice_exact'])
        self.assertFalse(context['equivalence']['exact_projection_certified'])

    def test_dropping_nonzero_qubo_term_disables_exact_projection(self):
        self.problem['constraints'] = []
        self.problem['objective']['linear'] = [[0, 1e-13]]

        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': 4.0},
        )

        self.assertEqual(qubo['terms'], [])
        self.assertEqual(context['dropped_qubo_terms'], [[0, 0, 1e-13]])
        self.assertFalse(context['equivalence']['no_nonzero_terms_dropped'])
        self.assertFalse(context['equivalence']['objective_mapping_preserved'])
        self.assertFalse(context['equivalence']['exact_projection_certified'])

    def test_inexact_float_aggregation_disables_exact_projection(self):
        self.problem['constraints'][0]['linear'] = [
            [0, 1.0],
            [1, 3.0],
        ]

        _, context = compile_qubo(
            self.problem,
            {'default_penalty': 0.1},
        )

        self.assertFalse(context['arithmetic']['exact_accumulation'])
        self.assertFalse(context['equivalence']['arithmetic_exact'])
        self.assertFalse(context['equivalence']['objective_mapping_preserved'])
        self.assertFalse(context['equivalence']['exact_projection_certified'])

    def test_fixed_constraint_cancellation_uses_stable_exact_sum(self):
        self.problem = {
            'schema': 'cbqm.v1',
            'problem_id': 'fixed-cancellation',
            'variables': [
                {'index': 0, 'name': 'f_pos', 'vartype': 'BINARY'},
                {'index': 1, 'name': 'f_one', 'vartype': 'BINARY'},
                {'index': 2, 'name': 'f_neg', 'vartype': 'BINARY'},
                {'index': 3, 'name': 'x', 'vartype': 'BINARY'},
            ],
            'objective': {
                'sense': 'minimize',
                'offset': 0.0,
                'linear': [[3, 1.0]],
                'quadratic': [],
            },
            'constraints': [
                {
                    'name': 'stable-fixed-sum',
                    'family': 'numeric',
                    'linear': [
                        [0, 1e16],
                        [1, 1.0],
                        [2, -1e16],
                        [3, 1.0],
                    ],
                    'lower_bound': 0.5,
                },
            ],
            'fixed_values': [
                {'index': 0, 'value': 1},
                {'index': 1, 'value': 1},
                {'index': 2, 'value': 1},
            ],
            'metadata': {},
        }

        qubo, context = compile_qubo(
            self.problem,
            {'default_penalty': 4.0},
        )

        constraint = context['constraints'][0]
        self.assertEqual(1.0, constraint['fixed_contribution'])
        self.assertTrue(
            constraint['fixed_substitution']['exact_reconstruction']
        )
        self.assertEqual('redundant', constraint['encodings'][0]['status'])
        self.assertEqual([[0, 0, 1.0]], qubo['terms'])
        self.assertTrue(context['equivalence']['exact_projection_certified'])


if __name__ == '__main__':
    unittest.main()
