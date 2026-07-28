import copy
import unittest

from lib.solvers.qubo import ExactQuboSolver


class _ImmediatelyTimingOutExactSolver(ExactQuboSolver):
    """Deterministic timeout seam; avoids relying on a busy or idle test host."""

    def _deadline_reached(self, deadline):
        return True


class ExactQuboSolverTests(unittest.TestCase):
    def setUp(self):
        self.problem = {
            'schema': 'qubo.v1',
            'problem_id': 'exact-solver-example',
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

    def test_finds_proven_global_optimum(self):
        result = ExactQuboSolver().solve(self.problem)

        self.assertEqual('optimal', result['status'])
        self.assertEqual([1, 0], result['best_sample'])
        self.assertEqual(-2.0, result['best_energy'])
        self.assertEqual('search_exhausted', result['termination_reason'])
        self.assertEqual(
            {
                'candidates_evaluated': 4,
                'total_candidates': 4,
                'search_space_exhausted': True,
            },
            result['metrics'],
        )

    def test_uses_lexicographically_smallest_equal_energy_sample(self):
        problem = copy.deepcopy(self.problem)
        problem['terms'] = []
        problem['offset'] = 7.0

        result = ExactQuboSolver().solve(problem)

        self.assertEqual([0, 0], result['best_sample'])
        self.assertEqual(7.0, result['best_energy'])

    def test_solves_zero_variable_problem(self):
        problem = copy.deepcopy(self.problem)
        problem['num_variables'] = 0
        problem['variable_names'] = []
        problem['terms'] = []

        result = ExactQuboSolver().solve(problem)

        self.assertEqual('optimal', result['status'])
        self.assertEqual([], result['best_sample'])
        self.assertEqual(4.0, result['best_energy'])
        self.assertEqual(1, result['metrics']['candidates_evaluated'])

    def test_rejects_problem_over_explicit_safety_limit(self):
        with self.assertRaisesRegex(ValueError, 'exceeding max_variables=1'):
            ExactQuboSolver().solve(
                self.problem,
                {'max_variables': 1},
            )

    def test_rejects_unknown_and_invalid_config(self):
        solver = ExactQuboSolver()

        with self.assertRaisesRegex(ValueError, 'Unknown exact solver'):
            solver.solve(self.problem, {'timeuot_seconds': 1})
        with self.assertRaisesRegex(ValueError, 'max_variables'):
            solver.solve(self.problem, {'max_variables': True})
        with self.assertRaisesRegex(ValueError, 'timeout_seconds'):
            solver.solve(self.problem, {'timeout_seconds': 0})

    def test_timeout_returns_best_candidate_seen_so_far(self):
        result = _ImmediatelyTimingOutExactSolver().solve(
            self.problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('timeout', result['status'])
        self.assertEqual([0, 0], result['best_sample'])
        self.assertEqual(4.0, result['best_energy'])
        self.assertEqual('timeout_reached', result['termination_reason'])
        self.assertEqual(1, result['metrics']['candidates_evaluated'])
        self.assertFalse(result['metrics']['search_space_exhausted'])

    def test_completed_search_wins_over_deadline_on_final_candidate(self):
        problem = copy.deepcopy(self.problem)
        problem['num_variables'] = 0
        problem['variable_names'] = []
        problem['terms'] = []

        result = _ImmediatelyTimingOutExactSolver().solve(
            problem,
            {'timeout_seconds': 1},
        )

        self.assertEqual('optimal', result['status'])
        self.assertTrue(result['metrics']['search_space_exhausted'])

    def test_large_integer_offset_does_not_hide_one_unit_improvement(self):
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'large-integer-resolution',
            'sense': 'minimize',
            'num_variables': 1,
            'variable_names': ['x'],
            'offset': 10_000_000_000_000_000,
            'terms': [[0, 0, -1]],
            'metadata': {},
        }

        result = ExactQuboSolver().solve(problem)

        self.assertEqual([1], result['best_sample'])
        self.assertEqual(9_999_999_999_999_999, result['best_energy'])
        self.assertIs(type(result['best_energy']), int)

    def test_finite_coefficients_do_not_overflow_during_exact_comparison(self):
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'large-finite-coefficients',
            'sense': 'minimize',
            'num_variables': 1,
            'variable_names': ['x'],
            'offset': 1e308,
            'terms': [[0, 0, 1e308]],
            'metadata': {},
        }

        result = ExactQuboSolver().solve(problem)

        self.assertEqual([0], result['best_sample'])
        self.assertEqual(int(1e308), result['best_energy'])


if __name__ == '__main__':
    unittest.main()
