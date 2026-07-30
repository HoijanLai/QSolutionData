import unittest
from fractions import Fraction

from lib.contracts import validate_cbqm
from tests.oracles import (
    enumerate_cbqm_feasible,
    enumerate_qubo,
    evaluate_cbqm_objective,
    evaluate_qubo_energy,
    is_cbqm_feasible,
    public_json_number,
)


class SmallProblemOracleTests(unittest.TestCase):
    def test_qubo_uses_exact_arithmetic_and_lexicographic_ties(self):
        problem = {
            'num_variables': 2,
            'offset': 10_000_000_000_000_000,
            'terms': [[0, 0, -1]],
        }

        rows = enumerate_qubo(problem)

        self.assertEqual([1, 0], rows[0]['sample'])
        self.assertEqual(
            Fraction(9_999_999_999_999_999),
            rows[0]['energy_exact'],
        )
        self.assertEqual([1, 1], rows[1]['sample'])
        self.assertEqual(
            rows[0]['energy_exact'],
            evaluate_qubo_energy(problem, rows[0]['sample']),
        )

    def test_cbqm_checks_fixed_values_and_upper_bounds_independently(self):
        problem = self._cbqm_problem('minimize')
        validate_cbqm(problem)

        rows = enumerate_cbqm_feasible(problem)

        self.assertEqual([[1, 0]], [row['sample'] for row in rows])
        self.assertTrue(is_cbqm_feasible(problem, [1, 0]))
        self.assertFalse(is_cbqm_feasible(problem, [0, 0]))
        self.assertFalse(is_cbqm_feasible(problem, [1, 1]))
        self.assertEqual(
            Fraction(-3, 2),
            evaluate_cbqm_objective(problem, [1, 0]),
        )

    def test_cbqm_minimize_orders_distinct_objectives_then_ties(self):
        problem = self._unconstrained_cbqm_problem('minimize')
        validate_cbqm(problem)

        rows = enumerate_cbqm_feasible(problem)

        self.assertEqual(
            [[1, 0], [1, 1], [0, 0], [0, 1]],
            [row['sample'] for row in rows],
        )

    def test_cbqm_maximize_orders_distinct_objectives_then_ties(self):
        problem = self._unconstrained_cbqm_problem('maximize')
        validate_cbqm(problem)

        rows = enumerate_cbqm_feasible(problem)

        self.assertEqual(
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            [row['sample'] for row in rows],
        )

    def test_cbqm_equal_objectives_use_canonical_sample_order(self):
        problem = self._cbqm_problem('maximize')
        problem['fixed_values'] = []
        problem['constraints'] = []
        problem['objective']['linear'] = []

        rows = enumerate_cbqm_feasible(problem)

        self.assertEqual([0, 0], rows[0]['sample'])
        self.assertTrue(
            all(
                row['objective_exact'] == Fraction(1, 2)
                for row in rows
            )
        )

    def test_cbqm_can_have_an_empty_feasible_set(self):
        problem = self._cbqm_problem('minimize')
        problem['fixed_values'] = [{'index': 0, 'value': 0}]
        problem['constraints'] = [
            {
                'name': 'require_x0',
                'family': 'oracle-test',
                'linear': [[0, 1]],
                'lower_bound': 1,
                'metadata': {},
            }
        ]
        validate_cbqm(problem)

        self.assertEqual([], enumerate_cbqm_feasible(problem))

    def test_public_json_number_preserves_integral_values(self):
        self.assertEqual(3, public_json_number(Fraction(3)))
        self.assertIs(type(public_json_number(Fraction(3))), int)
        self.assertEqual(0.5, public_json_number(Fraction(1, 2)))
        with self.assertRaisesRegex(ValueError, 'finite JSON number'):
            public_json_number(Fraction(10**400, 3))

    def test_oracles_reject_non_binary_witnesses(self):
        qubo = {'num_variables': 1, 'offset': 0, 'terms': []}
        cbqm = self._cbqm_problem('minimize')

        with self.assertRaisesRegex(ValueError, 'integer 0/1'):
            evaluate_qubo_energy(qubo, [True])
        with self.assertRaisesRegex(ValueError, 'exactly 2'):
            is_cbqm_feasible(cbqm, [1])

    def _cbqm_problem(self, sense):
        return {
            'schema': 'cbqm.v1',
            'problem_id': f'oracle-{sense}',
            'variables': [
                {
                    'index': 0,
                    'name': 'x0',
                    'vartype': 'BINARY',
                    'kind': 'decision',
                    'metadata': {},
                },
                {
                    'index': 1,
                    'name': 'x1',
                    'vartype': 'BINARY',
                    'kind': 'decision',
                    'metadata': {},
                },
            ],
            'objective': {
                'sense': sense,
                'offset': 0.5,
                'linear': [[0, -2]],
                'quadratic': [],
            },
            'constraints': [
                {
                    'name': 'force_x1_zero',
                    'family': 'oracle-test',
                    'linear': [[1, 1]],
                    'upper_bound': 0,
                    'metadata': {},
                }
            ],
            'fixed_values': [{'index': 0, 'value': 1}],
            'metadata': {},
        }

    def _unconstrained_cbqm_problem(self, sense):
        problem = self._cbqm_problem(sense)
        problem['constraints'] = []
        problem['fixed_values'] = []
        return problem


if __name__ == '__main__':
    unittest.main()
