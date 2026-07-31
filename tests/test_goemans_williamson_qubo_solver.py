import copy
import json
import unittest

import numpy as np

from lib.contracts.validation import validate_qubo_result
from lib.solvers.qubo import (
    ExactQuboSolver,
    GoemansWilliamsonQuboSolver,
)
from problem.reproductions import build_qrbnbr_s1_instance


class GoemansWilliamsonQuboSolverTests(unittest.TestCase):
    def setUp(self):
        self.problem = build_qrbnbr_s1_instance(
            variable_count=6,
            edge_probability=0.5,
            seed=2718,
        )
        self.config = {
            'rounds': 32,
            'seed': 11,
            'sdp_solver': 'CLARABEL',
            'sdp_tolerance': 1e-7,
            'max_variables': 10,
        }

    def test_returns_contract_valid_feasible_cut_and_sdp_diagnostics(self):
        result = GoemansWilliamsonQuboSolver().solve(
            self.problem,
            self.config,
        )
        exact = ExactQuboSolver().solve(self.problem)

        self.assertEqual('feasible', result['status'])
        self.assertGreaterEqual(
            result['best_energy'],
            exact['best_energy'],
        )
        self.assertGreaterEqual(
            result['metrics']['sdp_relaxation_value'] + 1e-5,
            result['metrics']['cut_value'],
        )
        self.assertEqual(
            'not_proven',
            result['metadata']['optimality_basis'],
        )
        self.assertEqual('CLARABEL', result['metadata']['sdp_solver'])
        validate_qubo_result(self.problem, result)
        json.dumps(result, allow_nan=False)

    def test_triangle_has_sdp_value_nine_quarters_and_cut_two(self):
        triangle = build_qrbnbr_s1_instance(
            variable_count=3,
            edge_probability=1.0,
            seed=0,
        )

        result = GoemansWilliamsonQuboSolver().solve(
            triangle,
            self.config,
        )

        self.assertEqual(-2, result['best_energy'])
        self.assertAlmostEqual(
            2.25,
            result['metrics']['sdp_relaxation_value'],
            places=5,
        )
        self.assertEqual(2.0, result['metrics']['cut_value'])
        self.assertEqual('feasible', result['status'])

    def test_fixed_seed_repeats_algorithm_owned_results(self):
        first = GoemansWilliamsonQuboSolver().solve(
            self.problem,
            self.config,
        )
        second = GoemansWilliamsonQuboSolver().solve(
            self.problem,
            self.config,
        )

        self.assertEqual(first['best_sample'], second['best_sample'])
        self.assertEqual(first['best_energy'], second['best_energy'])
        self.assertEqual(first['metadata'], second['metadata'])
        for metric in (
            'cut_value',
            'sdp_relaxation_value',
            'rounds',
            'unique_samples',
            'cut_to_sdp_ratio',
            'sdp_iterations',
        ):
            self.assertEqual(first['metrics'][metric], second['metrics'][metric])

    def test_factor_projects_tiny_negative_eigenvalues_to_zero(self):
        solver = GoemansWilliamsonQuboSolver()
        matrix = np.asarray(
            [
                [1.0, 1.0 + 1e-12],
                [1.0 + 1e-12, 1.0],
            ]
        )

        vectors = solver._factor_psd_matrix(matrix)
        reconstructed = vectors @ vectors.T

        self.assertTrue(np.all(np.linalg.eigvalsh(reconstructed) >= -1e-12))

    def test_supports_zero_variable_problem(self):
        problem = build_qrbnbr_s1_instance(
            variable_count=0,
            edge_probability=0.5,
            seed=0,
        )

        result = GoemansWilliamsonQuboSolver().solve(problem, self.config)

        self.assertEqual([], result['best_sample'])
        self.assertEqual(0, result['best_energy'])
        self.assertEqual('trivial', result['metadata']['sdp_status'])

    def test_does_not_mutate_problem_or_config(self):
        problem_before = copy.deepcopy(self.problem)
        config = copy.deepcopy(self.config)
        config_before = copy.deepcopy(config)

        GoemansWilliamsonQuboSolver().solve(self.problem, config)

        self.assertEqual(problem_before, self.problem)
        self.assertEqual(config_before, config)

    def test_rejects_general_qubo_invalid_config_and_large_problem(self):
        solver = GoemansWilliamsonQuboSolver()
        general_qubo = {
            'schema': 'qubo.v1',
            'problem_id': 'not-maxcut',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['x', 'y'],
            'offset': 0,
            'terms': [[0, 0, -1], [0, 1, 4], [1, 1, -1]],
            'metadata': {},
        }

        with self.assertRaisesRegex(ValueError, 'only MaxCut QUBOs'):
            solver.solve(general_qubo)
        with self.assertRaisesRegex(ValueError, 'Unknown Goemans-Williamson'):
            solver.solve(self.problem, {'trial': 1})
        with self.assertRaisesRegex(ValueError, 'rounds'):
            solver.solve(self.problem, {'rounds': 0})
        with self.assertRaisesRegex(ValueError, 'seed'):
            solver.solve(self.problem, {'seed': -1})
        with self.assertRaisesRegex(ValueError, 'sdp_solver'):
            solver.solve(self.problem, {'sdp_solver': 'OSQP'})
        with self.assertRaisesRegex(ValueError, 'sdp_tolerance'):
            solver.solve(self.problem, {'sdp_tolerance': 0})
        with self.assertRaisesRegex(ValueError, 'exceeding max_variables=5'):
            solver.solve(self.problem, {'max_variables': 5})


if __name__ == '__main__':
    unittest.main()
