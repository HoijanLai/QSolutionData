"""Gaussian SK instances for the SCMF-QAOA paper reproduction.

The paper studies the Ising cost Hamiltonian

``C(z) = sum_i h_i z_i + sum_{i<j} W_ij z_i z_j``

with ``z_i in {-1, +1}``.  Its Sherrington--Kirkpatrick experiments use a
complete graph, zero local fields and normally distributed, unscaled
couplings.  This module preserves that native mathematical description in
metadata while publishing the actual solver input as canonical ``qubo.v1``.

Only problem construction lives here.  Self-consistency, decomposition and
QAOA belong to :mod:`lib.solvers.qubo.scmf_qaoa`.
"""

from __future__ import annotations

import math
import random


_PAPER_TITLE = 'Self-consistent mean-field quantum approximate optimization'
_PAPER_URL = 'https://arxiv.org/abs/2603.09838'
_PAPER_REVISION = 'arXiv:2603.09838v1'
_GENERATOR_VERSION = '0.1.0'


def build_scmf_gaussian_sk_instance(
    *,
    spin_count=8,
    seed=2603,
    coupling_standard_deviation=1.0,
):
    """Build one complete, zero-field Gaussian SK problem as ``qubo.v1``.

    The paper's energy grows as ``N**(3/2)``, so its Gaussian couplings are
    intentionally not divided by ``sqrt(N)``.  Keeping the scale explicit is
    important because it also determines useful QAOA phase angles.

    Args:
        spin_count: Number of Ising degrees of freedom.  The generous upper
            guard covers the paper's numerical scale while preventing an
            accidental quadratic metadata explosion.
        seed: Seed owned by a private standard-library random generator.
        coupling_standard_deviation: Standard deviation of every independent
            normal coupling ``W_ij``.

    Returns:
        A fresh JSON-serializable ``qubo.v1`` mapping.
    """
    _validate_settings(
        spin_count,
        seed,
        coupling_standard_deviation,
    )
    random_generator = random.Random(seed)
    couplings = [
        (
            left,
            right,
            random_generator.gauss(
                0.0,
                float(coupling_standard_deviation),
            ),
        )
        for left in range(spin_count)
        for right in range(left + 1, spin_count)
    ]
    fields = [0.0] * spin_count

    return _paper_ising_to_qubo(
        problem_id=(
            f'scmf-sk-gaussian-n{spin_count}-seed{seed}'
        ),
        couplings=couplings,
        fields=fields,
        metadata=_build_metadata(
            spin_count=spin_count,
            seed=seed,
            coupling_standard_deviation=(
                coupling_standard_deviation
            ),
            couplings=couplings,
            fields=fields,
        ),
    )


def build_scmf_qaoa_reproduction_suite():
    """Return exact-checkable and decomposition-oriented SCMF fixtures."""
    settings = (
        (8, 2603),
        (16, 9838),
    )
    return {
        f'scmf.sk-gaussian-{spin_count}': (
            build_scmf_gaussian_sk_instance(
                spin_count=spin_count,
                seed=seed,
            )
        )
        for spin_count, seed in settings
    }


def _validate_settings(
    spin_count,
    seed,
    coupling_standard_deviation,
):
    """Reject values outside the declared reproduction family."""
    if (
        not isinstance(spin_count, int)
        or isinstance(spin_count, bool)
        or not 2 <= spin_count <= 256
    ):
        raise ValueError('spin_count must be an integer in [2, 256].')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a non-negative integer.')
    if (
        not isinstance(coupling_standard_deviation, (int, float))
        or isinstance(coupling_standard_deviation, bool)
        or not math.isfinite(coupling_standard_deviation)
        or coupling_standard_deviation <= 0
    ):
        raise ValueError(
            'coupling_standard_deviation must be finite and positive.'
        )


def _paper_ising_to_qubo(*, problem_id, couplings, fields, metadata):
    """Apply ``z_i = 1 - 2*x_i`` to the paper's Ising convention.

    For one interaction,

    ``W_ij z_i z_j = W_ij - 2 W_ij x_i - 2 W_ij x_j
                         + 4 W_ij x_i x_j``.

    Keeping this conversion next to the problem definition makes its sign
    convention auditable without coupling the generator to solver internals.
    """
    variable_count = len(fields)
    diagonal = [-2.0 * float(field) for field in fields]
    offset = math.fsum(float(field) for field in fields)
    quadratic = []

    for left, right, coupling in couplings:
        weight = float(coupling)
        offset += weight
        diagonal[left] -= 2.0 * weight
        diagonal[right] -= 2.0 * weight
        quadratic.append([left, right, 4.0 * weight])

    terms = [
        [index, index, coefficient]
        for index, coefficient in enumerate(diagonal)
        if coefficient != 0
    ]
    terms.extend(quadratic)
    terms.sort(key=lambda term: (term[0], term[1]))

    return {
        'schema': 'qubo.v1',
        'problem_id': problem_id,
        'sense': 'minimize',
        'num_variables': variable_count,
        'variable_names': [
            f'spin_{index}'
            for index in range(variable_count)
        ],
        'offset': offset,
        'terms': terms,
        'metadata': metadata,
    }


def _build_metadata(
    *,
    spin_count,
    seed,
    coupling_standard_deviation,
    couplings,
    fields,
):
    """Record enough provenance to reconstruct the paper Hamiltonian."""
    return {
        'reproduction': {
            'algorithm': 'SCMF-QAOA',
            'paper_title': _PAPER_TITLE,
            'paper_url': _PAPER_URL,
            'paper_revision': _PAPER_REVISION,
            'implementation_style': (
                'independent-reimplementation-from-paper-formulas'
            ),
            'experiment_family': 'Gaussian SK spin glass',
            'claim_scope': (
                'small ideal-statevector validation of the '
                'self-consistent decomposition'
            ),
        },
        'generator': {
            'name': 'build_scmf_gaussian_sk_instance',
            'version': _GENERATOR_VERSION,
            'seed': seed,
            'parameters': {
                'model': 'sherrington-kirkpatrick',
                'spin_count': spin_count,
                'topology': 'complete-graph',
                'coupling_distribution': 'normal',
                'coupling_mean': 0.0,
                'coupling_standard_deviation': float(
                    coupling_standard_deviation
                ),
                'normalization': 'unscaled',
                'spin_encoding': 'z=1-2x',
            },
        },
        'paper_ising': {
            'convention': 'C=sum(h_i*z_i)+sum(W_ij*z_i*z_j)',
            'couplings': [
                [left, right, coupling]
                for left, right, coupling in couplings
            ],
            'fields': list(fields),
        },
    }


__all__ = [
    'build_scmf_gaussian_sk_instance',
    'build_scmf_qaoa_reproduction_suite',
]
