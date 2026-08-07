"""Contract and paper-equation tests for the SCMF-QAOA solver."""

import copy
import itertools
import json
import math
import unittest

import numpy as np

from lib.solvers.qubo import QaoaQuboSolver, ScmfQaoaSolver
from problem.reproductions import build_scmf_gaussian_sk_instance
from tests.oracles import public_json_number
from tests.oracles.qubo import evaluate_qubo_energy


class ScmfQaoaSolverTests(unittest.TestCase):
    def setUp(self):
        self.problem = {
            'schema': 'qubo.v1',
            'problem_id': 'scmf-contract-example',
            'sense': 'minimize',
            'num_variables': 4,
            'variable_names': ['x0', 'x1', 'x2', 'x3'],
            'offset': 1.25,
            'terms': [
                [0, 0, -1.5],
                [0, 1, 0.75],
                [0, 3, -0.5],
                [1, 1, 0.25],
                [1, 2, -1.0],
                [2, 2, 0.8],
                [2, 3, 1.2],
                [3, 3, -0.4],
            ],
            'metadata': {},
        }

    def test_returns_reproducible_feasible_result_without_mutation(self):
        problem_snapshot = copy.deepcopy(self.problem)
        config = {
            'subproblem_count': 2,
            'optimizer_iterations': 3,
            'max_environment_sweeps': 20,
            'shots': 128,
            'seed': 11,
            'symmetry_breaking': 'none',
        }
        config_snapshot = copy.deepcopy(config)

        first = ScmfQaoaSolver().solve(self.problem, config)
        second = ScmfQaoaSolver().solve(self.problem, config)

        self.assertEqual('feasible', first['status'])
        self.assertEqual(first['best_sample'], second['best_sample'])
        self.assertEqual(first['best_energy'], second['best_energy'])
        self.assertEqual(first['metrics'], second['metrics'])
        self.assertEqual(first['metadata'], second['metadata'])
        self.assertEqual(problem_snapshot, self.problem)
        self.assertEqual(config_snapshot, config)
        self.assertEqual(
            public_json_number(
                evaluate_qubo_energy(
                    self.problem,
                    first['best_sample'],
                )
            ),
            first['best_energy'],
        )
        json.dumps(first, allow_nan=False)

    def test_qubo_to_ising_preserves_every_basis_energy(self):
        solver = ScmfQaoaSolver()
        model = solver._qubo_to_ising(self.problem)

        for sample in itertools.product((0, 1), repeat=4):
            spins = np.asarray([1 - 2 * bit for bit in sample])
            ising_energy = (
                model.constant
                + float(model.fields @ spins)
                + 0.5 * float(spins @ model.couplings @ spins)
            )
            with self.subTest(sample=sample):
                self.assertAlmostEqual(
                    float(evaluate_qubo_energy(self.problem, list(sample))),
                    ising_energy,
                    places=12,
                )

    def test_paper_p1_one_body_equation_matches_statevector(self):
        field_values = [0.4, -0.2]
        coupling = 0.7
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'paper-equation-eight',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['z0', 'z1'],
            'offset': sum(field_values) + coupling,
            'terms': [
                [0, 0, -2 * field_values[0] - 2 * coupling],
                [0, 1, 4 * coupling],
                [1, 1, -2 * field_values[1] - 2 * coupling],
            ],
            'metadata': {},
        }
        gamma = 0.31
        beta = 0.22
        solver = ScmfQaoaSolver()
        model = solver._qubo_to_ising(problem)
        state = solver._solve_subproblem(
            model,
            (0, 1),
            np.zeros(2),
            np.asarray([gamma, beta]),
        )

        expected = [
            -math.sin(2 * beta)
            * math.sin(2 * gamma * field)
            * math.cos(2 * gamma * coupling)
            for field in field_values
        ]
        np.testing.assert_allclose(state.one_body, expected, atol=1e-12)

    def test_single_partition_is_standard_qaoa_up_to_angle_convention(self):
        parameters = [0.37, 0.29]
        common = {
            'optimizer_iterations': 0,
            'shots': None,
            'initial_parameters': parameters,
            'symmetry_breaking': 'none',
        }
        scmf = ScmfQaoaSolver().solve(
            self.problem,
            {
                **common,
                'subproblem_count': 1,
                'max_environment_sweeps': 4,
            },
        )
        qaoa = QaoaQuboSolver().solve(
            self.problem,
            {
                'optimizer_iterations': 0,
                'restarts': 1,
                'shots': None,
                'initial_parameters': [-parameters[0], parameters[1]],
            },
        )

        self.assertAlmostEqual(
            qaoa['metrics']['expectation'],
            scmf['metrics']['expectation'],
            places=12,
        )
        self.assertEqual(qaoa['best_sample'], scmf['best_sample'])

    def test_zero_field_sk_uses_the_paper_symmetry_breaking_rule(self):
        problem = build_scmf_gaussian_sk_instance(
            spin_count=6,
            seed=29,
        )
        result = ScmfQaoaSolver().solve(
            problem,
            {
                'subproblem_count': 2,
                'optimizer_iterations': 0,
                'max_environment_sweeps': 40,
                'shots': 64,
                'seed': 5,
            },
        )

        self.assertEqual([[5, 1]], result['metadata']['fixed_spins'])
        self.assertEqual(0, result['best_sample'][5])
        self.assertEqual(5, result['metrics']['active_variables'])
        self.assertTrue(result['metrics']['environment_converged'])

    def test_auto_symmetry_breaking_keeps_real_field_at_large_scale(self):
        coupling = 1e12
        field = 0.5
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'scmf-large-coupling-with-field',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['x0', 'x1'],
            'offset': coupling + field,
            'terms': [
                [0, 0, -2.0 * coupling - 2.0 * field],
                [0, 1, 4.0 * coupling],
                [1, 1, -2.0 * coupling],
            ],
            'metadata': {},
        }
        solver = ScmfQaoaSolver()
        model = solver._qubo_to_ising(problem)

        self.assertAlmostEqual(field, model.fields[0])
        self.assertFalse(solver._has_zero_fields(model))
        active = solver._apply_symmetry_breaking(
            model,
            {'symmetry_breaking': 'auto'},
        )
        self.assertEqual((), active.fixed_spins)
        self.assertEqual((0, 1), active.original_indices)

    def test_environmentless_mode_is_an_explicit_baseline(self):
        problem = build_scmf_gaussian_sk_instance(
            spin_count=6,
            seed=31,
        )
        result = ScmfQaoaSolver().solve(
            problem,
            {
                'subproblem_count': 2,
                'optimizer_iterations': 0,
                'max_environment_sweeps': 0,
                'shots': None,
                'seed': 3,
            },
        )

        self.assertEqual(
            'independent_subproblems_completed',
            result['termination_reason'],
        )
        self.assertEqual(0, result['metrics']['environment_sweeps'])
        self.assertFalse(result['metrics']['environment_converged'])

    def test_explicit_symmetry_breaking_accepts_zero_variable_problem(self):
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'scmf-empty',
            'sense': 'minimize',
            'num_variables': 0,
            'variable_names': [],
            'offset': 1.5,
            'terms': [],
            'metadata': {},
        }

        for mode in ('fix-last-positive', 'fix-last-negative'):
            with self.subTest(mode=mode):
                result = ScmfQaoaSolver().solve(
                    problem,
                    {'symmetry_breaking': mode},
                )

                self.assertEqual('feasible', result['status'])
                self.assertEqual([], result['best_sample'])
                self.assertEqual(1.5, result['best_energy'])
                self.assertEqual(
                    'trivial_conditioned_problem',
                    result['termination_reason'],
                )

    def test_rejects_invalid_configuration_and_oversized_partition(self):
        solver = ScmfQaoaSolver()

        with self.assertRaisesRegex(ValueError, 'Unknown SCMF-QAOA'):
            solver.solve(self.problem, {'subproblems': 2})
        with self.assertRaisesRegex(ValueError, 'subproblem_count'):
            solver.solve(self.problem, {'subproblem_count': 5})
        with self.assertRaisesRegex(ValueError, 'environment_tolerance'):
            solver.solve(self.problem, {'environment_tolerance': 0})
        with self.assertRaisesRegex(ValueError, 'symmetry_breaking'):
            solver.solve(self.problem, {'symmetry_breaking': 'random'})
        with self.assertRaisesRegex(
            ValueError,
            'max_subproblem_variables=1',
        ):
            solver.solve(
                self.problem,
                {
                    'subproblem_count': 2,
                    'max_subproblem_variables': 1,
                    'symmetry_breaking': 'none',
                },
            )


if __name__ == '__main__':
    unittest.main()
