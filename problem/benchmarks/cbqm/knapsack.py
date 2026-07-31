"""Seeded 0/1 knapsack CBQM benchmark."""

import random

from .._metadata import benchmark_metadata
from ._common import _binary_variables, _linear_constraint


def build_knapsack_suite():
    """Return one finite-capacity value-maximization model."""
    return {
        'cbqm.knapsack-16': _build_knapsack(
            item_count=16,
            seed=161803,
        ),
    }


def _build_knapsack(*, item_count, seed):
    """Generate integer weights/values using one private deterministic RNG."""
    rng = random.Random(seed)
    weights = [2 + rng.randrange(11) for _ in range(item_count)]
    values = [3 + rng.randrange(18) for _ in range(item_count)]
    capacity = sum(weights) * 2 // 5

    return {
        'schema': 'cbqm.v1',
        'problem_id': 'benchmark-knapsack-16',
        'variables': _binary_variables(
            [f'take_item_{index}' for index in range(item_count)],
            kind='selection',
        ),
        'objective': {
            'sense': 'maximize',
            'offset': 0,
            'linear': [
                [index, value]
                for index, value in enumerate(values)
            ],
            'quadratic': [],
        },
        'constraints': [
            _linear_constraint(
                name='capacity',
                family='capacity',
                terms=list(enumerate(weights)),
                upper_bound=capacity,
            ),
        ],
        'fixed_values': [],
        'metadata': benchmark_metadata(
            family='packing',
            model='zero-one-knapsack',
            generator='build_knapsack_suite',
            seed=seed,
            size={
                'item_count': item_count,
                'num_variables': item_count,
                'constraint_count': 1,
            },
            parameters={
                'weights': weights,
                'values': values,
                'capacity': capacity,
            },
        ),
    }


__all__ = ['build_knapsack_suite']
