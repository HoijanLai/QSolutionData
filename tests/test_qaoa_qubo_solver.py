import copy
import json
import math
import unittest

import numpy as np

from lib.solvers.qubo import QaoaQuboSolver


class QaoaQuboSolverTests(unittest.TestCase):
    def setUp(self):
        self.problem = {
            'schema': 'qubo.v1',
            'problem_id': 'qaoa-solver-example',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['x0', 'x1'],
            'offset': 4.0,
            'terms': [
                [0, 0, -6.0],
                [0, 1, 8.0],
                [1, 1, -5.0],
            ],
            'metadata': {},
        }

    def test_returns_reproducible_feasible_result(self):
        config = {
            'layers': 1,
            'optimizer_iterations': 12,
            'restarts': 2,
            'shots': 256,
            'seed': 7,
        }

        first = QaoaQuboSolver().solve(self.problem, config)
        second = QaoaQuboSolver().solve(self.problem, config)

        self.assertEqual('feasible', first['status'])
        self.assertEqual(first['best_sample'], second['best_sample'])
        self.assertEqual(first['best_energy'], second['best_energy'])
        self.assertEqual(first['metadata'], second['metadata'])
        self.assertEqual(
            first['metrics']['expectation'],
            second['metrics']['expectation'],
        )
        self.assertEqual(2, len(first['best_sample']))
        self.assertTrue(set(first['best_sample']) <= {0, 1})

    def test_statevector_remains_normalised(self):
        solver = QaoaQuboSolver()
        energies = solver._build_cost_spectrum(self.problem)

        state = solver._prepare_qaoa_state(
            energies,
            variable_count=2,
            parameters=np.asarray([0.7, 0.3, 1.1, 0.2]),
        )

        self.assertAlmostEqual(1.0, float(np.sum(np.abs(state) ** 2)))

    def test_cost_spectrum_matches_canonical_qubo_order(self):
        spectrum = QaoaQuboSolver()._build_cost_spectrum(self.problem)

        np.testing.assert_allclose(spectrum, [4.0, -1.0, -2.0, 1.0])

    def test_zero_parameter_state_is_uniform(self):
        solver = QaoaQuboSolver()
        energies = solver._build_cost_spectrum(self.problem)

        state = solver._prepare_qaoa_state(
            energies,
            variable_count=2,
            parameters=np.asarray([0.0, 0.0]),
        )

        np.testing.assert_allclose(
            solver._measurement_probabilities(state),
            [0.25, 0.25, 0.25, 0.25],
        )

    def test_shotless_mode_selects_deterministically(self):
        problem = copy.deepcopy(self.problem)
        problem['terms'] = []
        problem['offset'] = 3.0

        result = QaoaQuboSolver().solve(
            problem,
            {
                'optimizer_iterations': 0,
                'restarts': 1,
                'shots': None,
                'initial_parameters': [0.0, 0.0],
            },
        )

        self.assertEqual([0, 0], result['best_sample'])
        self.assertEqual(3.0, result['best_energy'])
        self.assertTrue(
            all(type(value) is int for value in result['best_sample'])
        )
        json.dumps(result, allow_nan=False)

    def test_normalises_only_problem_independent_mixer_period(self):
        solver = QaoaQuboSolver()
        gamma = -7.25
        beta = 4.5 * math.pi

        normalised = solver._normalise_parameters(
            np.asarray([gamma, beta]),
        )

        self.assertEqual(gamma, normalised[0])
        self.assertAlmostEqual(0.5 * math.pi, normalised[1])

    def test_supports_zero_variable_problem(self):
        problem = copy.deepcopy(self.problem)
        problem['num_variables'] = 0
        problem['variable_names'] = []
        problem['terms'] = []

        result = QaoaQuboSolver().solve(
            problem,
            {
                'optimizer_iterations': 0,
                'restarts': 1,
                'shots': None,
                'initial_parameters': [0.0, 0.0],
            },
        )

        self.assertEqual([], result['best_sample'])
        self.assertEqual(4.0, result['best_energy'])

    def test_rejects_cost_spectrum_that_cannot_fit_binary64(self):
        problem = copy.deepcopy(self.problem)
        problem['offset'] = 1e308
        problem['terms'] = [[0, 0, 1e308]]

        with self.assertRaisesRegex(ValueError, 'finite binary64'):
            QaoaQuboSolver().solve(problem)

    def test_rejects_invalid_configuration_and_large_statevector(self):
        solver = QaoaQuboSolver()

        with self.assertRaisesRegex(ValueError, 'Unknown QAOA solver'):
            solver.solve(self.problem, {'layer': 1})
        with self.assertRaisesRegex(ValueError, 'layers'):
            solver.solve(self.problem, {'layers': 0})
        with self.assertRaisesRegex(ValueError, 'seed'):
            solver.solve(self.problem, {'seed': -1})
        with self.assertRaisesRegex(ValueError, 'initial_parameters'):
            solver.solve(
                self.problem,
                {'layers': 2, 'initial_parameters': [0.1, 0.2]},
            )
        with self.assertRaisesRegex(ValueError, 'exceeding max_variables=1'):
            solver.solve(self.problem, {'max_variables': 1})


if __name__ == '__main__':
    unittest.main()
