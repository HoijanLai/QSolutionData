import copy
import json
import unittest

from lib.contracts.validation import validate_qubo_result
from lib.solvers.qubo import (
    ExactQuboSolver,
    GwBranchAndBoundQuboSolver,
)
from problem.reproductions import build_qrbnbr_s1_instance


class GwBranchAndBoundQuboSolverTests(unittest.TestCase):
    def setUp(self):
        self.problem = build_qrbnbr_s1_instance(
            variable_count=6,
            edge_probability=0.45,
            seed=1618,
        )
        self.config = {
            'brute_force_threshold': 3,
            'rounds': 8,
            'seed': 5,
            'sdp_solver': 'CLARABEL',
            'sdp_tolerance': 1e-7,
            'max_variables': 10,
        }

    def test_matches_exact_and_reports_classical_relaxation_work(self):
        exact = ExactQuboSolver().solve(self.problem)

        result = GwBranchAndBoundQuboSolver().solve(
            self.problem,
            self.config,
        )

        self.assertEqual('optimal', result['status'])
        self.assertEqual(exact['best_energy'], result['best_energy'])
        self.assertGreater(result['metrics']['gw_subproblems'], 0)
        self.assertEqual(
            result['metrics']['gw_subproblems'],
            result['metrics']['sdp_solves'],
        )
        self.assertNotIn('qrr_subproblems', result['metrics'])
        self.assertNotIn('root_qaoa_parameters', result['metadata'])
        self.assertEqual('sdp', result['metadata']['branching_matrix'])
        self.assertEqual(
            'admissible_bound_and_exhausted_parity_tree',
            result['metadata']['optimality_basis'],
        )
        validate_qubo_result(self.problem, result)
        json.dumps(result, allow_nan=False)

    def test_all_branching_rules_and_traversals_preserve_the_optimum(self):
        exact_energy = ExactQuboSolver().solve(
            self.problem,
        )['best_energy']

        for rule, traversal in (
            ('r1', 'bfs'),
            ('r2', 'dfs'),
            ('r3', 'bfs'),
        ):
            with self.subTest(rule=rule, traversal=traversal):
                config = dict(
                    self.config,
                    branching_rule=rule,
                    traversal=traversal,
                )
                result = GwBranchAndBoundQuboSolver().solve(
                    self.problem,
                    config,
                )
                self.assertEqual('optimal', result['status'])
                self.assertEqual(exact_energy, result['best_energy'])

    def test_node_limit_does_not_claim_optimality(self):
        result = GwBranchAndBoundQuboSolver().solve(
            self.problem,
            dict(self.config, brute_force_threshold=1, max_nodes=1),
        )

        self.assertEqual('feasible', result['status'])
        self.assertEqual('node_limit_reached', result['termination_reason'])
        self.assertFalse(result['metrics']['tree_exhausted'])
        self.assertEqual('not_proven', result['metadata']['optimality_basis'])

    def test_subproblem_seed_is_stable_and_depends_on_the_reduction(self):
        solver = GwBranchAndBoundQuboSolver()
        model = solver._read_maxcut_model(self.problem)
        search = solver._initialise_search(self.problem, model, self.config)
        root = search['frontier'][0]
        reduced = solver._reduce_model(model, root)
        first = solver._subproblem_seed(reduced, 7)
        second = solver._subproblem_seed(reduced, 7)

        self.assertEqual(first, second)
        self.assertNotEqual(first, solver._subproblem_seed(reduced, 8))

    def test_does_not_mutate_problem_or_config(self):
        problem_before = copy.deepcopy(self.problem)
        config = copy.deepcopy(self.config)
        config_before = copy.deepcopy(config)

        GwBranchAndBoundQuboSolver().solve(self.problem, config)

        self.assertEqual(problem_before, self.problem)
        self.assertEqual(config_before, config)

    def test_rejects_invalid_configuration(self):
        solver = GwBranchAndBoundQuboSolver()

        with self.assertRaisesRegex(ValueError, 'Unknown GW-BnB'):
            solver.solve(self.problem, {'qaoa_grid_size': 3})
        with self.assertRaisesRegex(ValueError, 'branching_rule'):
            solver.solve(self.problem, {'branching_rule': 'r4'})
        with self.assertRaisesRegex(ValueError, 'traversal'):
            solver.solve(self.problem, {'traversal': 'priority'})
        with self.assertRaisesRegex(ValueError, 'rounds'):
            solver.solve(self.problem, {'rounds': 0})
        with self.assertRaisesRegex(ValueError, 'timeout_seconds'):
            solver.solve(self.problem, {'timeout_seconds': 0})


if __name__ == '__main__':
    unittest.main()
