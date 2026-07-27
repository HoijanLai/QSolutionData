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


if __name__ == '__main__':
    unittest.main()
