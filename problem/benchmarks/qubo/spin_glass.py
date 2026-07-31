"""Deterministic Edwards-Anderson and Sherrington-Kirkpatrick benchmarks."""

import random

from .._metadata import benchmark_metadata
from ._ising import _ising_to_qubo


def build_spin_glass_suite():
    """Return sparse-grid and dense mean-field spin-glass QUBOs."""
    return {
        'qubo.ea-grid-4x4': _build_edwards_anderson_grid(
            rows=4,
            columns=4,
            seed=1729,
        ),
        'qubo.sk-15': _build_sherrington_kirkpatrick(
            spin_count=15,
            seed=2718,
        ),
    }


def _build_edwards_anderson_grid(*, rows, columns, seed):
    """Build a nearest-neighbour ±J Ising model on a rectangular grid."""
    rng = random.Random(seed)
    couplings = []
    for row in range(rows):
        for column in range(columns):
            vertex = row * columns + column
            if column + 1 < columns:
                couplings.append(
                    (vertex, vertex + 1, _random_sign(rng))
                )
            if row + 1 < rows:
                couplings.append(
                    (vertex, vertex + columns, _random_sign(rng))
                )

    spin_count = rows * columns
    return _ising_to_qubo(
        problem_id='benchmark-ea-grid-4x4',
        variable_names=[
            f'spin_r{row}_c{column}'
            for row in range(rows)
            for column in range(columns)
        ],
        couplings=couplings,
        fields=[0] * spin_count,
        metadata=benchmark_metadata(
            family='spin-glass',
            model='edwards-anderson',
            generator='build_spin_glass_suite',
            seed=seed,
            size={
                'num_variables': spin_count,
                'interaction_count': len(couplings),
            },
            parameters={
                'topology': 'rectangular-grid',
                'rows': rows,
                'columns': columns,
                'coupling_distribution': 'uniform-plus-minus-one',
                'spin_encoding': 's=2x-1',
                'couplings': [
                    [left, right, coupling]
                    for left, right, coupling in couplings
                ],
                'fields': [0] * spin_count,
            },
        ),
    )


def _build_sherrington_kirkpatrick(*, spin_count, seed):
    """Build the dense complete-graph ±J mean-field spin-glass model."""
    rng = random.Random(seed)
    couplings = [
        (left, right, _random_sign(rng))
        for left in range(spin_count)
        for right in range(left + 1, spin_count)
    ]
    return _ising_to_qubo(
        problem_id='benchmark-sk-15',
        variable_names=[
            f'spin_{index}'
            for index in range(spin_count)
        ],
        couplings=couplings,
        fields=[0] * spin_count,
        metadata=benchmark_metadata(
            family='spin-glass',
            model='sherrington-kirkpatrick',
            generator='build_spin_glass_suite',
            seed=seed,
            size={
                'num_variables': spin_count,
                'interaction_count': len(couplings),
            },
            parameters={
                'topology': 'complete-graph',
                'coupling_distribution': 'uniform-plus-minus-one',
                'normalization': 'unscaled-integer',
                'spin_encoding': 's=2x-1',
                'couplings': [
                    [left, right, coupling]
                    for left, right, coupling in couplings
                ],
                'fields': [0] * spin_count,
            },
        ),
    )


def _random_sign(rng):
    """Draw a stable ±1 coupling from one private seeded RNG."""
    return 1 if rng.getrandbits(1) else -1


__all__ = ['build_spin_glass_suite']
