"""Problem-definition tests for the SCMF-QAOA paper reproduction."""

import itertools
import unittest

from lib.contracts import validate_qubo
from problem.reproductions import (
    build_scmf_gaussian_sk_instance,
    build_scmf_qaoa_reproduction_suite,
)
from tests.oracles.qubo import evaluate_qubo_energy


class ScmfQaoaReproductionProblemTests(unittest.TestCase):
    def test_gaussian_sk_instance_is_canonical_and_deterministic(self):
        first = build_scmf_gaussian_sk_instance(
            spin_count=6,
            seed=17,
        )
        second = build_scmf_gaussian_sk_instance(
            spin_count=6,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertIsNone(validate_qubo(first))
        self.assertEqual(6, first['num_variables'])
        self.assertEqual(
            15,
            len(first['metadata']['paper_ising']['couplings']),
        )
        self.assertEqual(
            'normal',
            first['metadata']['generator']['parameters'][
                'coupling_distribution'
            ],
        )
        self.assertEqual(
            'unscaled',
            first['metadata']['generator']['parameters']['normalization'],
        )

    def test_qubo_energy_preserves_the_paper_ising_convention(self):
        problem = build_scmf_gaussian_sk_instance(
            spin_count=4,
            seed=23,
        )
        paper_ising = problem['metadata']['paper_ising']

        for sample in itertools.product((0, 1), repeat=4):
            spins = [1 - 2 * bit for bit in sample]
            ising_energy = sum(
                field * spins[index]
                for index, field in enumerate(paper_ising['fields'])
            ) + sum(
                coupling * spins[left] * spins[right]
                for left, right, coupling in paper_ising['couplings']
            )
            with self.subTest(sample=sample):
                self.assertAlmostEqual(
                    ising_energy,
                    float(evaluate_qubo_energy(problem, list(sample))),
                    places=12,
                )

    def test_suite_separates_small_exact_and_decomposition_scales(self):
        suite = build_scmf_qaoa_reproduction_suite()

        self.assertEqual(
            {'scmf.sk-gaussian-8', 'scmf.sk-gaussian-16'},
            set(suite),
        )
        for problem in suite.values():
            self.assertIsNone(validate_qubo(problem))
            self.assertEqual(
                'Gaussian SK spin glass',
                problem['metadata']['reproduction']['experiment_family'],
            )

    def test_rejects_values_outside_the_reproduction_family(self):
        with self.assertRaisesRegex(ValueError, 'spin_count'):
            build_scmf_gaussian_sk_instance(spin_count=1)
        with self.assertRaisesRegex(ValueError, 'seed'):
            build_scmf_gaussian_sk_instance(seed=-1)
        with self.assertRaisesRegex(
            ValueError,
            'coupling_standard_deviation',
        ):
            build_scmf_gaussian_sk_instance(
                coupling_standard_deviation=0,
            )


if __name__ == '__main__':
    unittest.main()
