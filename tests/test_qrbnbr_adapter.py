import copy
import json
import math
import sys
import unittest

from lib.adapters.qrbnbr_maxcut import (
    QRBnBRMaxCutAdapter,
    QRBnBRSolverAdapter,
)


class _FakeGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_nodes_from(self, nodes):
        self.nodes.update(nodes)

    def add_edge(self, left, right, weight):
        self.edges[(min(left, right), max(left, right))] = weight


class _FakeMaxCutProblem:
    def __init__(self, graph, solve, name):
        self.graph = graph
        self.solve = solve
        self.name = name


class _FakeSolution:
    def __init__(self, solution, cost, elapsed_time=0.25, approx_ratio=None):
        self.z = solution
        self.cost = cost
        self.approx_ratio = approx_ratio
        self._steps = [
            {
                'solution': solution,
                'cost': cost,
                'time': elapsed_time,
                'approx_ratio': approx_ratio,
            }
        ]

    @property
    def data(self):
        return {'time': self._steps[-1]['time']}

    def __len__(self):
        return len(self._steps)

    def __getitem__(self, index):
        return self._steps[index]


class _FakeNativeSolver:
    def __init__(self):
        self.received_problem = None
        self.received_kwargs = None

    def solve(self, problem, **kwargs):
        self.received_problem = problem
        self.received_kwargs = kwargs
        return _FakeSolution([1, 0, 0], cost=6.0)


class QRBnBRMaxCutAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = QRBnBRMaxCutAdapter()
        self.qubo = {
            'schema': 'qubo.v1',
            'problem_id': 'two-asset-example',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['asset_a', 'asset_b'],
            'offset': 4.0,
            'terms': [[0, 0, -6.0], [0, 1, 8.0], [1, 1, -5.0]],
            'metadata': {},
        }

    def test_builds_expected_signed_maxcut_graph(self):
        problem, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        self.assertFalse(problem.solve)
        self.assertEqual(problem.name, 'two-asset-example')
        self.assertEqual(problem.graph.nodes, {0, 1, 2})
        self.assertEqual(
            problem.graph.edges,
            {(0, 1): 4.0, (0, 2): 2.0, (1, 2): 1.0},
        )
        self.assertEqual(context['anchor_node'], 2)
        self.assertEqual(
            context['energy_relation'],
            'qubo_energy = qubo_offset - cut_value',
        )

    def test_rejects_non_finite_accumulated_maxcut_weight(self):
        overflowing = {
            **self.qubo,
            'num_variables': 4,
            'variable_names': ['a', 'b', 'c', 'd'],
            'offset': 0.0,
            'terms': [
                [0, 1, 1.7e308],
                [0, 2, 1.7e308],
                [0, 3, 1.7e308],
            ],
        }

        with self.assertRaisesRegex(ValueError, 'overflowed'):
            self.adapter.from_qubo(
                overflowing,
                maxcut_problem_class=_FakeMaxCutProblem,
                graph_class=_FakeGraph,
            )

    def test_rejects_underflow_that_would_break_the_energy_relation(self):
        underflowing = {
            **self.qubo,
            'offset': 0.0,
            'terms': [[0, 1, 5e-324]],
        }

        with self.assertRaisesRegex(ValueError, 'lost numeric precision'):
            self.adapter.from_qubo(
                underflowing,
                maxcut_problem_class=_FakeMaxCutProblem,
                graph_class=_FakeGraph,
            )

    def test_rejects_possible_overflow_of_the_total_native_cut(self):
        overflowing_cut = {
            **self.qubo,
            'offset': 0.0,
            'terms': [[0, 0, -1e308], [1, 1, -1e308]],
        }

        with self.assertRaisesRegex(ValueError, 'cut accumulation may overflow'):
            self.adapter.from_qubo(
                overflowing_cut,
                maxcut_problem_class=_FakeMaxCutProblem,
                graph_class=_FakeGraph,
            )

    def test_cut_bound_reserves_sequential_rounding_headroom(self):
        maximum = sys.float_info.max
        maximum_ulp = math.ulp(maximum)
        large_weight = maximum - 4.0 * maximum_ulp
        small_weight = math.nextafter(0.5 * maximum_ulp, math.inf)
        near_overflow = {
            'schema': 'qubo.v1',
            'problem_id': 'sequential-cut-overflow',
            'sense': 'minimize',
            'num_variables': 6,
            'variable_names': [f'x{index}' for index in range(6)],
            'offset': 0.0,
            'terms': [
                [0, 0, -large_weight],
                *[
                    [index, index, -small_weight]
                    for index in range(1, 6)
                ],
            ],
            'metadata': {},
        }

        with self.assertRaisesRegex(ValueError, 'rounding headroom'):
            self.adapter.from_qubo(
                near_overflow,
                maxcut_problem_class=_FakeMaxCutProblem,
                graph_class=_FakeGraph,
            )

    def test_maxcut_energy_relation_holds_for_every_assignment(self):
        problem, _ = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        for first in (0, 1):
            for second in (0, 1):
                labels = [first, second, 0]
                cut_value = sum(
                    weight
                    for (left, right), weight in problem.graph.edges.items()
                    if labels[left] != labels[right]
                )
                energy = (
                    4.0
                    - 6.0 * first
                    - 5.0 * second
                    + 8.0 * first * second
                )
                self.assertAlmostEqual(energy, 4.0 - cut_value)

    def test_converts_native_solution_and_trace_to_canonical_result(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )
        native = _FakeSolution([1, 0, 0], cost=6.0, approx_ratio=0.9)

        result = self.adapter.to_result(
            native,
            context,
            solver_name='q-rbnbr-test',
            solver_version='1.2.3',
        )

        self.assertEqual(result['schema'], 'qubo-result.v1')
        self.assertEqual(result['best_sample'], [1, 0])
        self.assertEqual(result['best_energy'], -2.0)
        self.assertEqual(result['runtime_seconds'], 0.25)
        self.assertTrue(result['metrics']['energy_consistent'])
        self.assertEqual(result['trace'][0]['sample'], [1, 0])
        self.assertEqual(result['trace'][0]['energy'], -2.0)
        self.assertEqual(
            result['trace'][0]['metadata']['native_approximation_ratio'],
            0.9,
        )
        # Adapter boundaries must not leak native/numpy scalar types into the
        # versioned JSON result contract.
        json.dumps(result, allow_nan=False)

    def test_decodes_spin_solution_relative_to_anchor(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )
        native = _FakeSolution([-1, 1, 1], cost=6.0)

        result = self.adapter.to_result(native, context)

        self.assertEqual(result['best_sample'], [1, 0])

    def test_supports_result_without_candidate(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        result = self.adapter.to_result(
            None,
            context,
            status='timeout',
            runtime_seconds=5.0,
        )

        self.assertIsNone(result['best_sample'])
        self.assertIsNone(result['best_energy'])
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['runtime_seconds'], 5.0)

    def test_context_rejects_boolean_structural_indices(self):
        zero_variable = {
            'schema': 'qubo.v1',
            'problem_id': 'zero-variable-context',
            'sense': 'minimize',
            'num_variables': 0,
            'variable_names': [],
            'offset': 0,
            'terms': [],
            'metadata': {},
        }
        _, context = self.adapter.from_qubo(
            zero_variable,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        for field in ('num_variables', 'anchor_node'):
            with self.subTest(field=field):
                malformed = copy.deepcopy(context)
                malformed[field] = False
                with self.assertRaisesRegex(ValueError, 'inconsistent'):
                    self.adapter.to_result(
                        None,
                        malformed,
                        status='timeout',
                    )

    def test_infeasible_status_rejects_a_candidate(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        with self.assertRaisesRegex(
            ValueError,
            "infeasible.*cannot include",
        ):
            self.adapter.to_result(
                _FakeSolution([1, 0, 0], cost=6.0),
                context,
                status='infeasible',
            )

    def test_optimal_status_requires_independent_exhaustive_verification(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )
        native = _FakeSolution([1, 0, 0], cost=6.0)

        with self.assertRaisesRegex(
            ValueError,
            'independent exhaustive verification',
        ):
            self.adapter.to_result(native, context, status='optimal')

        result = self.adapter.to_result(
            native,
            context,
            status='optimal',
            verify_optimality=True,
        )
        self.assertEqual(result['status'], 'optimal')
        self.assertTrue(result['metadata']['optimality_verified'])
        self.assertEqual(
            'exhaustive-qubo-enumeration',
            result['metadata']['optimality_proof']['method'],
        )
        self.assertEqual(
            4,
            result['metadata']['optimality_proof']['assignments_checked'],
        )

    def test_independent_verifier_rejects_a_false_optimal_claim(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )
        # The native cost is internally consistent with x=(0, 0), but that
        # candidate is not globally optimal. The adapter must discover the
        # better assignment itself instead of trusting the native status.
        native = _FakeSolution([0, 0, 0], cost=0.0)

        with self.assertRaisesRegex(
            ValueError,
            'found a better QUBO assignment',
        ):
            self.adapter.to_result(
                native,
                context,
                status='optimal',
                verify_optimality=True,
            )

    def test_independent_verifier_keeps_large_integer_resolution(self):
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'large-integer-proof',
            'sense': 'minimize',
            'num_variables': 1,
            'variable_names': ['x'],
            'offset': 10_000_000_000_000_000,
            'terms': [[0, 0, -1]],
            'metadata': {},
        }
        _, context = self.adapter.from_qubo(
            problem,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        # x=0 has a cut value of zero and therefore looks internally
        # consistent, but x=1 is better by one exact integer unit.
        with self.assertRaisesRegex(
            ValueError,
            'found a better QUBO assignment',
        ):
            self.adapter.to_result(
                _FakeSolution([0, 0], cost=0.0),
                context,
                status='optimal',
                verify_optimality=True,
            )

    def test_optimal_proof_does_not_reject_normal_float_cut_summation(self):
        problem = {
            'schema': 'qubo.v1',
            'problem_id': 'rounded-native-cut',
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['x', 'y'],
            'offset': 0.0,
            'terms': [[0, 0, -0.1], [1, 1, -0.2]],
            'metadata': {},
        }
        _, context = self.adapter.from_qubo(
            problem,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        result = self.adapter.to_result(
            _FakeSolution([1, 1, 0], cost=0.1 + 0.2),
            context,
            status='optimal',
            verify_optimality=True,
        )

        self.assertEqual('optimal', result['status'])
        self.assertTrue(result['metrics']['energy_consistent'])
        self.assertTrue(result['metadata']['optimality_verified'])

    def test_independent_verifier_obeys_variable_safety_limit(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        with self.assertRaisesRegex(
            ValueError,
            'exceeding max_verification_variables=1',
        ):
            self.adapter.to_result(
                _FakeSolution([1, 0, 0], cost=6.0),
                context,
                status='optimal',
                verify_optimality=True,
                max_verification_variables=1,
            )

    def test_rejects_non_finite_native_ratio_and_trace_cost(self):
        _, context = self.adapter.from_qubo(
            self.qubo,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        with self.assertRaisesRegex(ValueError, 'approximation ratio'):
            self.adapter.to_result(
                _FakeSolution(
                    [1, 0, 0],
                    cost=6.0,
                    approx_ratio=float('inf'),
                ),
                context,
            )

        invalid_trace = _FakeSolution([1, 0, 0], cost=6.0)
        invalid_trace._steps[0]['cost'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'trace cost'):
            self.adapter.to_result(invalid_trace, context)

    def test_solver_wrapper_exposes_canonical_signature(self):
        native_solver = _FakeNativeSolver()
        solver = QRBnBRSolverAdapter(
            native_solver,
            solver_name='wrapped-q-rbnbr',
            solver_version='1.0.0',
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        result = solver.solve(
            self.qubo,
            config={'native_solve_kwargs': {'depth': 3}},
        )

        self.assertIsInstance(native_solver.received_problem, _FakeMaxCutProblem)
        self.assertEqual(native_solver.received_kwargs, {'depth': 3})
        self.assertEqual(result['best_sample'], [1, 0])
        self.assertEqual(result['best_energy'], -2.0)
        self.assertEqual(result['solver']['name'], 'wrapped-q-rbnbr')

    def test_solver_wrapper_cannot_claim_optimal_without_capability(self):
        native_solver = _FakeNativeSolver()
        with self.assertRaisesRegex(
            ValueError,
            'independent exhaustive verification',
        ):
            QRBnBRSolverAdapter(
                native_solver,
                result_status='optimal',
                maxcut_problem_class=_FakeMaxCutProblem,
                graph_class=_FakeGraph,
            )

        solver = QRBnBRSolverAdapter(
            native_solver,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )
        with self.assertRaisesRegex(
            ValueError,
            'independent exhaustive verification',
        ):
            solver.solve(self.qubo, config={'status': 'optimal'})
        self.assertIsNone(native_solver.received_problem)

    def test_solver_wrapper_allows_optimal_for_proving_backend(self):
        solver = QRBnBRSolverAdapter(
            _FakeNativeSolver(),
            result_status='optimal',
            proves_optimality=True,
            maxcut_problem_class=_FakeMaxCutProblem,
            graph_class=_FakeGraph,
        )

        result = solver.solve(self.qubo)

        self.assertEqual(result['status'], 'optimal')
        self.assertTrue(result['metadata']['optimality_verified'])
        self.assertEqual(
            'exhaustive-qubo-enumeration',
            result['metadata']['optimality_proof']['method'],
        )


if __name__ == '__main__':
    unittest.main()
