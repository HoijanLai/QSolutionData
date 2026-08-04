import copy
import itertools
import math
import unittest
from fractions import Fraction

import numpy as np

from lib.contracts import validate_qubo, validate_qubo_result
from lib.solvers.qubo.exact import ExactQuboSolver
from lib.solvers.qubo.simulated_annealing import (
    SimulatedAnnealingQuboSolver,
)
from problem.benchmarks import build_annealing_validation_suite
from tests.oracles import enumerate_qubo, evaluate_qubo_energy


class _ImmediatelyTimingOutAnnealingSolver(SimulatedAnnealingQuboSolver):
    """Make timeout behaviour deterministic without depending on wall time."""

    def _deadline_reached(self, deadline):
        return True


class _TimingOutAfterOneProposalAnnealingSolver(
    SimulatedAnnealingQuboSolver
):
    """Cross the deadline after one improving flip inside the first sweep."""

    def __init__(self):
        self._deadline_checks = 0

    def _deadline_reached(self, deadline):
        self._deadline_checks += 1
        return self._deadline_checks >= 3


class AnnealingValidationSuiteTests(unittest.TestCase):
    def test_suite_contains_fresh_valid_qubo_fixtures(self):
        first = build_annealing_validation_suite()
        second = build_annealing_validation_suite()

        self.assertEqual(
            {
                'zero_variable',
                'positive_bias',
                'negative_bias',
                'ferromagnetic_pair',
                'frustrated_triangle',
                'sparse_random',
                'large_offset_small_bias',
                'degenerate',
                'custom_schedule',
            },
            set(first),
        )
        for problem in first.values():
            validate_qubo(problem)

        first['sparse_random']['terms'].clear()
        self.assertNotEqual(
            first['sparse_random']['terms'],
            second['sparse_random']['terms'],
        )


class SimulatedAnnealingQuboSolverTests(unittest.TestCase):
    def setUp(self):
        self.suite = build_annealing_validation_suite()
        self.solver = SimulatedAnnealingQuboSolver()

    def test_returns_contract_valid_feasible_candidate_without_exact_claim(self):
        problem = self.suite['sparse_random']
        result = self.solver.solve(
            problem,
            {
                'num_reads': 4,
                'sweeps': 12,
                'seed': 17,
            },
        )

        validate_qubo_result(problem, result)
        exact = ExactQuboSolver().solve(problem)
        oracle = enumerate_qubo(problem)[0]

        self.assertEqual('feasible', result['status'])
        self.assertNotEqual('optimal', result['status'])
        self.assertIsNotNone(result['best_sample'])
        self.assertGreaterEqual(result['best_energy'], exact['best_energy'])
        self.assertGreaterEqual(
            evaluate_qubo_energy(problem, result['best_sample']),
            oracle['energy_exact'],
        )

    def test_fixed_seed_repeats_all_algorithm_owned_result_fields(self):
        problem = self.suite['frustrated_triangle']
        config = {
            'num_reads': 5,
            'sweeps': 9,
            'seed': 1946,
            'update_order': 'random',
        }

        first = self.solver.solve(problem, config)
        second = self.solver.solve(problem, config)

        for field_name in (
            'status',
            'best_sample',
            'best_energy',
            'termination_reason',
            'metrics',
            'metadata',
        ):
            self.assertEqual(first.get(field_name), second.get(field_name))

    def test_does_not_mutate_problem_or_nested_configuration(self):
        problem = self.suite['custom_schedule']
        config = {
            'num_reads': 2,
            'sweeps': 3,
            'beta_schedule_type': 'custom',
            'beta_schedule': [0.0, 0.5, 2.0],
            'initial_samples': [[0, 0], [1, 1]],
            'seed': 5,
        }
        expected_problem = copy.deepcopy(problem)
        expected_config = copy.deepcopy(config)

        self.solver.solve(problem, config)

        self.assertEqual(expected_problem, problem)
        self.assertEqual(expected_config, config)

    def test_flip_delta_matches_canonical_full_energy_difference(self):
        for fixture_name in ('sparse_random', 'large_offset_small_bias'):
            problem = self.suite[fixture_name]
            adjacency = self.solver._build_adjacency(problem)

            for bits in itertools.product(
                (0, 1),
                repeat=problem['num_variables'],
            ):
                sample = list(bits)
                before = evaluate_qubo_energy(problem, sample)
                for variable_index in range(problem['num_variables']):
                    flipped = sample.copy()
                    flipped[variable_index] ^= 1
                    after = evaluate_qubo_energy(problem, flipped)

                    delta = self.solver._delta_energy(
                        sample,
                        variable_index,
                        adjacency,
                    )

                    self.assertEqual(
                        after - before,
                        delta,
                        msg=(
                            f'{fixture_name}: sample={sample}, '
                            f'variable={variable_index}'
                        ),
                    )

    def test_resolves_linear_geometric_and_custom_beta_schedules(self):
        problem = self.suite['custom_schedule']
        cases = {
            'linear': {
                'num_reads': 1,
                'sweeps': 4,
                'beta_schedule_type': 'linear',
                'beta_start': 0.25,
                'beta_end': 2.0,
            },
            'geometric': {
                'num_reads': 1,
                'sweeps': 4,
                'beta_schedule_type': 'geometric',
                'beta_start': 0.25,
                'beta_end': 2.0,
            },
            'custom': {
                'num_reads': 1,
                'sweeps': 4,
                'beta_schedule_type': 'custom',
                'beta_schedule': [0.0, 0.1, 0.5, 2.0],
            },
        }

        schedules = {}
        for schedule_type, config in cases.items():
            with self.subTest(schedule_type=schedule_type):
                resolved = self.solver._resolve_config(config)
                schedule = self.solver._resolve_beta_schedule(
                    problem,
                    resolved,
                )
                result = self.solver.solve(problem, config)

                validate_qubo_result(problem, result)
                self.assertEqual('feasible', result['status'])
                self.assertEqual(config['sweeps'], len(schedule))
                self.assertTrue(np.all(np.diff(schedule) >= 0.0))
                schedules[schedule_type] = schedule

        np.testing.assert_allclose(
            schedules['linear'],
            np.linspace(0.25, 2.0, 4),
        )
        np.testing.assert_allclose(
            schedules['geometric'],
            np.geomspace(0.25, 2.0, 4),
        )
        np.testing.assert_allclose(
            schedules['custom'],
            [0.0, 0.1, 0.5, 2.0],
        )

    def test_rejects_unknown_malformed_and_inconsistent_configuration(self):
        problem = self.suite['custom_schedule']
        invalid_configs = (
            {'unknown_option': 1},
            {'num_reads': 0},
            {'num_reads': True},
            {'sweeps': 0},
            {'sweeps': 1.5},
            {'beta_schedule_type': 'logarithmic'},
            {'beta_start': -1.0},
            {'beta_end': math.inf},
            {'beta_start': 2.0, 'beta_end': 1.0},
            {
                'sweeps': 2,
                'beta_schedule_type': 'custom',
                'beta_schedule': None,
            },
            {
                'sweeps': 3,
                'beta_schedule_type': 'custom',
                'beta_schedule': [0.0, 1.0],
            },
            {
                'sweeps': 3,
                'beta_schedule_type': 'custom',
                'beta_schedule': [0.0, 2.0, 1.0],
            },
            {'seed': -1},
            {'seed': True},
            {'initial_samples': []},
            {'initial_samples': [[0, 2]]},
            {'update_order': 'reverse'},
            {'timeout_seconds': 0},
            {'timeout_seconds': 10**400},
            {'beta_start': 10**400},
            {
                'sweeps': 1,
                'beta_schedule_type': 'custom',
                'beta_schedule': [10**400],
            },
            {'trace_interval': 0},
        )

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    self.solver.solve(problem, config)

    def test_solves_zero_variable_problem_as_the_empty_sample(self):
        problem = self.suite['zero_variable']

        result = self.solver.solve(
            problem,
            {
                'num_reads': 2,
                'sweeps': 3,
                'seed': 11,
            },
        )

        self.assertEqual('feasible', result['status'])
        self.assertEqual([], result['best_sample'])
        self.assertEqual(problem['offset'], result['best_energy'])
        validate_qubo_result(problem, result)

    def test_large_integer_offset_preserves_the_small_improving_bias(self):
        problem = self.suite['large_offset_small_bias']

        result = self.solver.solve(
            problem,
            {
                'num_reads': 1,
                'sweeps': 1,
                'beta_schedule_type': 'custom',
                'beta_schedule': [1000.0],
                'initial_samples': [[1]],
                'update_order': 'sequential',
                'seed': 0,
            },
        )

        self.assertEqual([1], result['best_sample'])
        self.assertEqual(
            9_999_999_999_999_999,
            result['best_energy'],
        )
        self.assertIs(type(result['best_energy']), int)

    def test_zero_beta_accepts_even_an_unrepresentably_large_uphill_delta(self):
        accepted = self.solver._accept_flip(
            Fraction(10**400),
            beta=0.0,
            random_generator=np.random.default_rng(0),
        )

        self.assertTrue(accepted)

    def test_scales_exact_delta_before_binary64_metropolis_probability(self):
        accepted = self.solver._accept_flip(
            Fraction(10**310),
            beta=float(np.nextafter(0.0, 1.0)),
            random_generator=np.random.default_rng(0),
        )

        self.assertTrue(accepted)

    def test_timeout_seam_keeps_a_validated_initial_incumbent(self):
        problem = self.suite['custom_schedule']

        result = _ImmediatelyTimingOutAnnealingSolver().solve(
            problem,
            {
                'num_reads': 2,
                'sweeps': 5,
                'initial_samples': [[0, 1]],
                'timeout_seconds': 1.0,
                'seed': 0,
            },
        )

        self.assertEqual('timeout', result['status'])
        self.assertEqual([0, 1], result['best_sample'])
        self.assertNotEqual('optimal', result['status'])
        validate_qubo_result(problem, result)

    def test_timeout_mid_read_keeps_an_improved_incumbent(self):
        problem = self.suite['custom_schedule']

        result = _TimingOutAfterOneProposalAnnealingSolver().solve(
            problem,
            {
                'num_reads': 1,
                'sweeps': 1,
                'beta_schedule_type': 'custom',
                'beta_schedule': [1000.0],
                'initial_samples': [[0, 0]],
                'update_order': 'sequential',
                'timeout_seconds': 1.0,
                'seed': 0,
            },
        )

        self.assertEqual('timeout', result['status'])
        self.assertEqual([1, 0], result['best_sample'])
        self.assertEqual(0, result['best_energy'])
        self.assertEqual(1, result['metrics']['flips_proposed'])
        self.assertEqual(1, result['metrics']['flips_accepted'])
        validate_qubo_result(problem, result)

    def test_trace_uses_global_steps_and_canonical_observed_energies(self):
        problem = self.suite['custom_schedule']

        result = self.solver.solve(
            problem,
            {
                'num_reads': 2,
                'sweeps': 3,
                'trace_interval': 1,
                'seed': 4,
            },
        )

        validate_qubo_result(problem, result)
        self.assertEqual(
            [1, 2, 3, 4, 5, 6],
            [entry['step'] for entry in result['trace']],
        )
        for entry in result['trace']:
            self.assertEqual(
                evaluate_qubo_energy(problem, entry['sample']),
                Fraction(entry['energy']),
            )

    def test_initial_samples_cycle_and_update_order_changes_one_sweep(self):
        problem = self.suite['custom_schedule']
        initial_config = self.solver._resolve_config(
            {
                'num_reads': 3,
                'sweeps': 1,
                'initial_samples': [[1, 0], [0, 1]],
            }
        )
        initial_samples = self.solver._prepare_initial_samples(
            2,
            initial_config,
            np.random.default_rng(0),
        )

        self.assertEqual(
            [[1, 0], [0, 1], [1, 0]],
            [sample.tolist() for sample in initial_samples],
        )

        common = {
            'num_reads': 1,
            'sweeps': 1,
            'beta_schedule_type': 'custom',
            'beta_schedule': [1000.0],
            'initial_samples': [[0, 0]],
        }
        sequential = self.solver.solve(
            problem,
            {
                **common,
                'seed': 0,
                'update_order': 'sequential',
            },
        )
        random = self.solver.solve(
            problem,
            {
                **common,
                'seed': 3,
                'update_order': 'random',
            },
        )

        # Both improving first moves become local minima after the coupling
        # penalty activates; the chosen update order is therefore observable.
        self.assertEqual([1, 0], sequential['best_sample'])
        self.assertEqual([0, 1], random['best_sample'])


if __name__ == '__main__':
    unittest.main()
