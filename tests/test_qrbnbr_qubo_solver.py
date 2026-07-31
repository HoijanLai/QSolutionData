import copy
import json
import unittest

import numpy as np

from lib.contracts.validation import validate_qubo_result
from lib.solvers.qubo import ExactQuboSolver, QrbnbrQuboSolver
from lib.solvers.qubo.qrbnbr import _ParityPartition
from problem.reproductions import build_qrbnbr_s1_instance


class QrbnbrQuboSolverTests(unittest.TestCase):
    def setUp(self):
        self.problem = build_qrbnbr_s1_instance(
            variable_count=7,
            edge_probability=0.45,
            seed=31415,
        )
        self.fast_config = {
            'brute_force_threshold': 2,
            'qaoa_grid_size': 3,
            'qaoa_refinement_steps': 1,
            'max_variables': 10,
        }

    def test_matches_exact_solver_and_reports_a_tree_certificate(self):
        exact = ExactQuboSolver().solve(self.problem)

        result = QrbnbrQuboSolver().solve(
            self.problem,
            self.fast_config,
        )

        self.assertEqual('optimal', result['status'])
        self.assertEqual(exact['best_energy'], result['best_energy'])
        self.assertEqual(
            'edge_parity_tree_exhausted',
            result['termination_reason'],
        )
        self.assertTrue(result['metrics']['tree_exhausted'])
        self.assertGreater(result['metrics']['qrr_subproblems'], 0)
        self.assertGreater(result['metrics']['exact_subproblems'], 0)
        self.assertEqual(
            'admissible_bound_and_exhausted_parity_tree',
            result['metadata']['optimality_basis'],
        )
        validate_qubo_result(self.problem, result)
        json.dumps(result, allow_nan=False)

    def test_all_paper_branching_rules_preserve_the_optimum(self):
        exact_energy = ExactQuboSolver().solve(
            self.problem,
        )['best_energy']

        for rule in ('r1', 'r2', 'r3'):
            with self.subTest(rule=rule):
                config = dict(self.fast_config, branching_rule=rule)
                result = QrbnbrQuboSolver().solve(self.problem, config)
                self.assertEqual('optimal', result['status'])
                self.assertEqual(exact_energy, result['best_energy'])

    def test_selective_composition_and_dfs_preserve_the_optimum(self):
        exact_energy = ExactQuboSolver().solve(
            self.problem,
        )['best_energy']
        config = dict(
            self.fast_config,
            branching_matrix='selective',
            selective_rank=2,
            traversal='dfs',
        )

        result = QrbnbrQuboSolver().solve(self.problem, config)

        self.assertEqual('optimal', result['status'])
        self.assertEqual(exact_energy, result['best_energy'])
        self.assertEqual('selective', result['metadata']['branching_matrix'])
        self.assertEqual('dfs', result['metadata']['traversal'])

    def test_node_limit_returns_a_candidate_without_claiming_optimality(self):
        config = dict(self.fast_config, max_nodes=1)

        result = QrbnbrQuboSolver().solve(self.problem, config)

        self.assertEqual('feasible', result['status'])
        self.assertEqual('node_limit_reached', result['termination_reason'])
        self.assertFalse(result['metrics']['tree_exhausted'])
        self.assertEqual('not_proven', result['metadata']['optimality_basis'])
        self.assertIsNotNone(result['best_sample'])
        validate_qubo_result(self.problem, result)

    def test_supports_weighted_maxcut_and_exact_trace_energy(self):
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'weighted-maxcut',
            'sense': 'minimize',
            'num_variables': 3,
            'variable_names': ['a', 'b', 'c'],
            'offset': 0.125,
            'terms': [
                [0, 0, -3.75],
                [0, 1, 3.0],
                [0, 2, 4.5],
                [1, 1, -2.25],
                [1, 2, 1.5],
                [2, 2, -3.0],
            ],
            'metadata': {},
        }

        result = QrbnbrQuboSolver().solve(
            problem,
            dict(self.fast_config, brute_force_threshold=1),
        )
        exact = ExactQuboSolver().solve(problem)

        self.assertEqual(exact['best_energy'], result['best_energy'])
        validate_qubo_result(problem, result)

    def test_large_integer_offset_never_enters_statevector_arithmetic(self):
        problem = copy.deepcopy(self.problem)
        problem['offset'] = 10 ** 400

        result = QrbnbrQuboSolver().solve(problem, self.fast_config)
        exact = ExactQuboSolver().solve(problem)

        self.assertEqual(exact['best_energy'], result['best_energy'])
        self.assertIsInstance(result['best_energy'], int)
        validate_qubo_result(problem, result)

    def test_rejects_a_general_qubo_instead_of_overclaiming_scope(self):
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
            QrbnbrQuboSolver().solve(general_qubo)

    def test_rejects_invalid_configuration_and_large_problem(self):
        solver = QrbnbrQuboSolver()

        with self.assertRaisesRegex(ValueError, 'Unknown Q-RBnBR'):
            solver.solve(self.problem, {'branch_rule': 'r1'})
        with self.assertRaisesRegex(ValueError, 'branching_rule'):
            solver.solve(self.problem, {'branching_rule': 'r4'})
        with self.assertRaisesRegex(ValueError, 'branching_matrix'):
            solver.solve(self.problem, {'branching_matrix': 'low-rank'})
        with self.assertRaisesRegex(ValueError, 'traversal'):
            solver.solve(self.problem, {'traversal': 'priority'})
        with self.assertRaisesRegex(ValueError, 'selective_rank'):
            solver.solve(self.problem, {'selective_rank': 0})
        with self.assertRaisesRegex(ValueError, 'timeout_seconds'):
            solver.solve(self.problem, {'timeout_seconds': 0})
        with self.assertRaisesRegex(ValueError, 'exceeding max_variables=6'):
            solver.solve(self.problem, {'max_variables': 6})

    def test_does_not_mutate_problem_or_config(self):
        problem_before = copy.deepcopy(self.problem)
        config = dict(self.fast_config)
        config_before = copy.deepcopy(config)

        QrbnbrQuboSolver().solve(self.problem, config)

        self.assertEqual(problem_before, self.problem)
        self.assertEqual(config_before, config)

    def test_parity_reduction_preserves_source_cut_exactly(self):
        solver = QrbnbrQuboSolver()
        model = solver._read_maxcut_model(self.problem)
        partition = _ParityPartition(model.variable_count)
        partition.join(0, 1, parity=1)
        partition.join(2, 3, parity=0)
        reduced = solver._reduce_model(model, partition)

        for basis_index in range(1 << reduced.variable_count):
            reduced_sample = solver._decode_basis_index(
                basis_index,
                reduced.variable_count,
            )
            source_sample = partition.expand(reduced_sample)
            self.assertAlmostEqual(
                solver._evaluate_source_cut(model, source_sample),
                solver._evaluate_reduced_cut(reduced, reduced_sample),
            )

    def test_spectral_bound_dominates_every_reduced_cut(self):
        solver = QrbnbrQuboSolver()
        model = solver._read_maxcut_model(self.problem)
        partition = _ParityPartition(model.variable_count)
        partition.join(0, 1, parity=1)
        partition.join(2, 3, parity=0)
        reduced = solver._reduce_model(model, partition)
        upper_bound = solver._maxcut_upper_bound(reduced)

        for basis_index in range(1 << reduced.variable_count):
            reduced_sample = solver._decode_basis_index(
                basis_index,
                reduced.variable_count,
            )
            self.assertLessEqual(
                solver._evaluate_reduced_cut(reduced, reduced_sample),
                upper_bound,
            )

    def test_uniform_state_has_zero_off_diagonal_correlations(self):
        solver = QrbnbrQuboSolver()
        probabilities = np.full(8, 1.0 / 8.0)

        correlation = solver._correlation_matrix(probabilities, 3)

        np.testing.assert_allclose(correlation, np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()
