"""QUBOs with a reproducible planted unique optimum."""

import random

from .._metadata import benchmark_metadata


def build_planted_qubo_suite():
    """Return one interacting QUBO whose planted sample is globally optimal."""
    return {
        'qubo.planted-consistency-18': _build_planted_consistency_qubo(
            variable_count=18,
            seed=314159,
        ),
    }


def _build_planted_consistency_qubo(*, variable_count, seed):
    """Sum non-negative local penalties that all vanish at one target sample."""
    rng = random.Random(seed)
    planted_sample = [
        rng.getrandbits(1)
        for _ in range(variable_count)
    ]
    coefficients = {}
    offset = 0

    # Positive one-bit penalties make the planted assignment unique.
    for index, target in enumerate(planted_sample):
        weight = 1 + (index % 3)
        if target == 0:
            _add(coefficients, index, index, weight)
        else:
            offset += weight
            _add(coefficients, index, index, -weight)

    # Pair consistency penalties add a nontrivial interaction landscape while
    # retaining zero cost at the planted assignment.
    interaction_count = 0
    for left in range(variable_count):
        for right in range(left + 1, variable_count):
            if rng.random() >= 0.18:
                continue
            weight = 1 + rng.randrange(4)
            if planted_sample[left] == planted_sample[right]:
                # weight * (x_left - x_right)^2
                _add(coefficients, left, left, weight)
                _add(coefficients, right, right, weight)
                _add(coefficients, left, right, -2 * weight)
            else:
                # weight * (x_left + x_right - 1)^2
                offset += weight
                _add(coefficients, left, left, -weight)
                _add(coefficients, right, right, -weight)
                _add(coefficients, left, right, 2 * weight)
            interaction_count += 1

    return {
        'schema': 'qubo.v1',
        'problem_id': 'benchmark-planted-consistency-18',
        'sense': 'minimize',
        'num_variables': variable_count,
        'variable_names': [
            f'x_{index}'
            for index in range(variable_count)
        ],
        'offset': offset,
        'terms': [
            [left, right, coefficient]
            for (left, right), coefficient in sorted(coefficients.items())
            if coefficient != 0
        ],
        'metadata': benchmark_metadata(
            family='planted-solution',
            model='consistency-penalty-qubo',
            generator='build_planted_qubo_suite',
            seed=seed,
            size={
                'num_variables': variable_count,
                'interaction_count': interaction_count,
            },
            parameters={
                'planted_sample': planted_sample,
                'planted_energy': 0,
                'unique_optimum_by_construction': True,
                'interaction_density': 0.18,
            },
        ),
    }


def _add(coefficients, left, right, value):
    """Aggregate one canonical sparse coefficient."""
    key = (min(left, right), max(left, right))
    coefficients[key] = coefficients.get(key, 0) + value


__all__ = ['build_planted_qubo_suite']
