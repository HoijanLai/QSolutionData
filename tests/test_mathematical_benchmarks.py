"""Contract, determinism and catalog tests for mathematical benchmarks."""

import copy
import unittest

from lib.contracts import validate_cbqm, validate_mis, validate_qubo
from problem import validate_problem_case
from problem.benchmarks import (
    audit_exact_benchmarks,
    build_cbqm_mathematical_suite,
    build_mathematical_benchmark_cases,
    build_mathematical_benchmark_references,
    build_mathematical_benchmark_suite,
    build_mis_mathematical_suite,
    build_qubo_mathematical_suite,
)
from tests.oracles.qubo import evaluate_qubo_energy


_EXPECTED_NAMES = {
    'qubo.ea-grid-4x4',
    'qubo.sk-15',
    'qubo.planted-consistency-18',
    'qubo.ferromagnetic-cycle-18',
    'qubo.frustrated-cycle-17',
    'cbqm.tsp-4',
    'cbqm.knapsack-16',
    'cbqm.exact-cover-8x12',
    'cbqm.graph-coloring-5x3',
    'mis.path-32',
    'mis.cycle-31',
    'mis.clique-36',
    'mis.complete-bipartite-12x12',
    'mis.grid-5x6',
    'mis.weighted-path-30',
    'mis.erdos-renyi-28-p022',
}


class MathematicalBenchmarkTests(unittest.TestCase):
    def test_catalog_contains_all_named_families_and_valid_contracts(self):
        suite = build_mathematical_benchmark_suite()

        self.assertEqual(_EXPECTED_NAMES, set(suite))
        self.assertEqual(5, len(build_qubo_mathematical_suite()))
        self.assertEqual(4, len(build_cbqm_mathematical_suite()))
        self.assertEqual(7, len(build_mis_mathematical_suite()))
        for name, problem in suite.items():
            with self.subTest(name=name):
                {
                    'qubo.v1': validate_qubo,
                    'cbqm.v1': validate_cbqm,
                    'mis.v1': validate_mis,
                }[problem['schema']](problem)

    def test_every_fixture_has_mathematical_provenance_and_safe_scale(self):
        for name, problem in build_mathematical_benchmark_suite().items():
            with self.subTest(name=name):
                benchmark = problem['metadata']['benchmark']
                self.assertEqual('mathematical', benchmark['domain'])
                self.assertTrue(benchmark['family'])
                self.assertTrue(benchmark['model'])
                self.assertEqual('1.0.0', benchmark['generator_version'])
                self.assertEqual('laptop-exact', benchmark['size_class'])
                self.assertEqual(
                    1800,
                    benchmark['intended_exact_budget_seconds'],
                )
                if problem['schema'] == 'qubo.v1':
                    self.assertLessEqual(problem['num_variables'], 24)
                elif problem['schema'] == 'cbqm.v1':
                    self.assertLessEqual(len(problem['variables']), 24)
                else:
                    self.assertLessEqual(len(problem['vertices']), 48)

    def test_builders_are_deterministic_and_return_fresh_documents(self):
        first = build_mathematical_benchmark_suite()
        pristine = copy.deepcopy(first)
        second = build_mathematical_benchmark_suite()

        self.assertEqual(pristine, second)
        first['qubo.sk-15']['terms'][0][2] += 100
        first['mis.path-32']['edges'].clear()

        self.assertEqual(pristine, second)
        self.assertEqual(pristine, build_mathematical_benchmark_suite())

    def test_planted_qubo_exposes_and_realizes_its_known_optimum(self):
        problem = build_mathematical_benchmark_suite()[
            'qubo.planted-consistency-18'
        ]
        parameters = problem['metadata']['benchmark']['parameters']

        energy = evaluate_qubo_energy(
            problem,
            parameters['planted_sample'],
        )

        self.assertEqual(parameters['planted_energy'], energy)
        self.assertTrue(parameters['unique_optimum_by_construction'])

    def test_spin_glass_qubo_preserves_the_declared_ising_hamiltonian(self):
        suite = build_qubo_mathematical_suite()
        for name in ('qubo.ea-grid-4x4', 'qubo.sk-15'):
            problem = suite[name]
            parameters = problem['metadata']['benchmark']['parameters']
            samples = [
                [0] * problem['num_variables'],
                [1] * problem['num_variables'],
                [
                    index % 2
                    for index in range(problem['num_variables'])
                ],
            ]

            for sample in samples:
                with self.subTest(name=name, sample=sample):
                    spins = [2 * bit - 1 for bit in sample]
                    ising_energy = -sum(
                        coupling * spins[left] * spins[right]
                        for left, right, coupling
                        in parameters['couplings']
                    ) - sum(
                        field * spins[index]
                        for index, field in enumerate(
                            parameters['fields']
                        )
                    )
                    self.assertEqual(
                        ising_energy,
                        evaluate_qubo_energy(problem, sample),
                    )

    def test_classical_mis_metadata_records_known_independence_numbers(self):
        suite = build_mis_mathematical_suite()
        expected = {
            'mis.path-32': 16,
            'mis.cycle-31': 15,
            'mis.clique-36': 1,
            'mis.complete-bipartite-12x12': 12,
            'mis.grid-5x6': 15,
        }

        for name, optimum in expected.items():
            with self.subTest(name=name):
                parameters = suite[name]['metadata']['benchmark'][
                    'parameters'
                ]
                self.assertEqual(
                    optimum,
                    parameters['known_independence_number'],
                )

    def test_catalog_wraps_every_problem_as_a_strict_native_case(self):
        cases = build_mathematical_benchmark_cases()
        references = build_mathematical_benchmark_references()

        self.assertEqual(_EXPECTED_NAMES, set(cases))
        self.assertEqual(_EXPECTED_NAMES, set(references))
        for name, case in cases.items():
            with self.subTest(name=name):
                report = validate_problem_case(case, strict=True)
                task = case.get_task('optimize')
                artifact = case.get_artifact(
                    task.canonical_artifact_id
                )
                self.assertTrue(report.fully_checked)
                self.assertEqual(
                    name,
                    case.metadata['benchmark_name'],
                )
                self.assertEqual(
                    artifact.representation,
                    artifact.payload['schema'],
                )
                self.assertTrue(task.best_known.exact)
                self.assertEqual(
                    references[name]['solution'],
                    task.best_known.solution,
                )
                self.assertEqual(
                    references[name]['objective_value'],
                    task.best_known.objective_value,
                )

    def test_exact_audit_runs_one_representative_per_contract(self):
        rows = audit_exact_benchmarks(
            [
                'qubo.ea-grid-4x4',
                'cbqm.exact-cover-8x12',
                'mis.erdos-renyi-28-p022',
            ],
            timeout_seconds=30,
        )

        self.assertEqual(3, len(rows))
        for row in rows:
            self.assertEqual('optimal', row['status'])
            self.assertTrue(row['within_intended_budget'])
            self.assertIsNotNone(row['solution'])
            reference = build_mathematical_benchmark_references()[
                row['benchmark_name']
            ]
            self.assertEqual(
                reference['objective_value'],
                row['objective_value'],
            )
            self.assertEqual(reference['solution'], row['solution'])

    def test_exact_audit_rejects_unknown_names_and_bad_controls(self):
        with self.assertRaisesRegex(KeyError, 'unknown'):
            audit_exact_benchmarks(['unknown'], timeout_seconds=1)
        with self.assertRaisesRegex(ValueError, 'timeout_seconds'):
            audit_exact_benchmarks([], timeout_seconds=True)
        with self.assertRaisesRegex(TypeError, 'sequence'):
            audit_exact_benchmarks('qubo.ea-grid-4x4')


if __name__ == '__main__':
    unittest.main()
