"""Protected non-negative MaxCut model shared by paper reproductions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class _MaxCutModel:
    """Validated source MaxCut represented in canonical variable order."""

    weights: np.ndarray

    @property
    def variable_count(self):
        """Return the number of source vertices."""
        return int(self.weights.shape[0])


def _read_nonnegative_maxcut_qubo(problem):
    """Recognize exactly ``offset - weighted_cut`` in sparse QUBO form.

    The check is exact over the JSON numbers' rational values.  Only the final
    edge matrix is converted to binary64, and that conversion must be lossless.
    This keeps a solver from claiming an optimum after silently rounding a
    large or unusually scaled coefficient.
    """
    variable_count = problem['num_variables']
    diagonal = [Fraction(0) for _ in range(variable_count)]
    exact_weights = {}

    for left, right, coefficient in problem['terms']:
        exact_coefficient = Fraction(coefficient)
        if left == right:
            diagonal[left] += exact_coefficient
            continue

        weight = exact_coefficient / 2
        if weight <= 0:
            raise ValueError(
                'MaxCut reproduction requires every edge weight to be '
                'strictly positive.'
            )
        exact_weights[left, right] = weight

    expected_diagonal = [Fraction(0) for _ in range(variable_count)]
    for (left, right), weight in exact_weights.items():
        expected_diagonal[left] -= weight
        expected_diagonal[right] -= weight
    if diagonal != expected_diagonal:
        raise ValueError(
            'MaxCut reproduction accepts only MaxCut QUBOs: each diagonal '
            'coefficient must equal the negative weighted degree implied by '
            'the positive quadratic terms.'
        )

    weights = np.zeros((variable_count, variable_count), dtype=float)
    for (left, right), exact_weight in exact_weights.items():
        weight = _lossless_float(exact_weight, left, right)
        weights[left, right] = weight
        weights[right, left] = weight

    absolute_weight_sum = math.fsum(
        abs(weight)
        for weight in weights[np.triu_indices(variable_count, k=1)]
    )
    if not math.isfinite(absolute_weight_sum):
        raise ValueError(
            'MaxCut edge accumulation may overflow binary64; rescale the '
            'QUBO coefficients.'
        )
    return _MaxCutModel(weights=weights)


def _lossless_float(exact_weight, left, right):
    """Convert one rational edge weight without overflow or rounding."""
    try:
        weight = float(exact_weight)
    except OverflowError as error:
        raise ValueError(
            'MaxCut edge weight is not representable as finite binary64 for '
            f'edge ({left}, {right}).'
        ) from error
    if (
        not math.isfinite(weight)
        or Fraction.from_float(weight) != exact_weight
    ):
        raise ValueError(
            'MaxCut edge weight is not losslessly representable as binary64 '
            f'for edge ({left}, {right}).'
        )
    return weight


def _evaluate_weighted_cut(weights, sample):
    """Evaluate an upper-triangular weighted cut."""
    cut = 0.0
    variable_count = weights.shape[0]
    for left in range(variable_count):
        for right in range(left + 1, variable_count):
            if sample[left] != sample[right]:
                cut += weights[left, right]
    return float(cut)


__all__ = [
    '_MaxCutModel',
    '_evaluate_weighted_cut',
    '_read_nonnegative_maxcut_qubo',
]
