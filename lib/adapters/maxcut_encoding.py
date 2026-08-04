"""Shared algebra for the signed MaxCut encoding used by QUBO adapters.

The encoding introduces one reference (``anchor``) node.  If a binary sample
is interpreted as the side of each variable node relative to that anchor, the
resulting signed cut satisfies

``qubo_energy(sample) = qubo_offset - cut_value(partition)``.

This module deliberately contains only the small algebraic kernel.  Callers
remain responsible for validating their complete ``qubo.v1`` payload and for
choosing the graph implementation in which the returned weights are stored.
Keeping the formula here prevents the persisted NetworkX representation and
the Q-RBnBR runtime adapter from silently drifting apart.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction


def build_signed_maxcut_edge_weights(qubo, anchor_node=None):
    """Return accumulated signed MaxCut weights for one canonical QUBO.

    Args:
        qubo: A validated ``qubo.v1``-like mapping.  Only ``num_variables`` and
            upper-triangular ``terms`` are read.
        anchor_node: Optional explicit reference-node identifier.  It defaults
            to ``qubo['num_variables']``, which is the canonical encoding used
            throughout this project.

    Returns:
        A dictionary mapping canonical undirected ``(left, right)`` pairs to
        finite built-in ``float`` weights. Every weight is checked to represent
        the exact QUBO-to-MaxCut algebra without overflow, underflow, or rounded
        accumulation. Zero-valued edges are omitted.
    """
    resolved_anchor = (
        qubo['num_variables']
        if anchor_node is None
        else anchor_node
    )
    edge_weights = {}

    for left, right, raw_coefficient in qubo['terms']:
        coefficient = Fraction(raw_coefficient)
        if left == right:
            _accumulate_edge(
                edge_weights,
                left,
                resolved_anchor,
                -coefficient,
            )
            continue

        half_coefficient = coefficient / 2
        _accumulate_edge(edge_weights, left, right, half_coefficient)
        _accumulate_edge(
            edge_weights,
            left,
            resolved_anchor,
            -half_coefficient,
        )
        _accumulate_edge(
            edge_weights,
            right,
            resolved_anchor,
            -half_coefficient,
        )

    # Algebraically cancelling contributions should not leave meaningless
    # zero-weight edges in either the runtime graph or persisted artifact.
    output = {
        edge: _lossless_float_weight(weight, edge)
        for edge, weight in edge_weights.items()
        if weight
    }
    _validate_cut_accumulation_bound(output)
    return output


def build_signed_maxcut_edges(qubo, anchor_node=None):
    """Return deterministically ordered ``[left, right, weight]`` triples."""
    edge_weights = build_signed_maxcut_edge_weights(qubo, anchor_node)
    return [
        [left, right, weight]
        for (left, right), weight in sorted(edge_weights.items())
    ]


def _accumulate_edge(edge_weights, left, right, weight):
    """Accumulate exactly so a small contribution cannot disappear."""
    edge = (min(left, right), max(left, right))
    edge_weights[edge] = edge_weights.get(edge, Fraction(0)) + weight


def _lossless_float_weight(weight, edge):
    """Return the backend float only when it preserves the exact edge weight."""
    try:
        output = float(weight)
    except OverflowError as error:
        raise ValueError(
            'Signed MaxCut edge accumulation overflowed or lost numeric '
            f'precision for edge {edge!r}. Rescale the QUBO coefficients '
            'before conversion.'
        ) from error
    if (
        not math.isfinite(output)
        or Fraction.from_float(output) != weight
    ):
        raise ValueError(
            'Signed MaxCut edge accumulation overflowed or lost numeric '
            f'precision for edge {edge!r}. Rescale the QUBO coefficients '
            'before conversion.'
        )
    return output


def _validate_cut_accumulation_bound(edge_weights):
    """Ensure every possible native float cut sum remains finite.

    A cut selects some subset of edges. Bounding that sum by the sum of all
    absolute weights is conservative, but it guarantees no summation order can
    overflow even when several individually finite edges cross together.
    """
    absolute_bound = sum(
        (
            Fraction.from_float(abs(weight))
            for weight in edge_weights.values()
        ),
        start=Fraction(0),
    )
    max_float = Fraction.from_float(sys.float_info.max)
    # Even when the exact total is just below ``max_float``, several ordinary
    # binary64 additions can round upward until a later addition overflows.
    # Reserve one maximum-magnitude ULP per addition. This is deliberately
    # conservative and makes the guarantee independent of backend edge order.
    rounding_headroom = (
        max(0, len(edge_weights) - 1)
        * Fraction.from_float(math.ulp(sys.float_info.max))
    )
    if (
        rounding_headroom >= max_float
        or absolute_bound > max_float - rounding_headroom
    ):
        raise ValueError(
            'Signed MaxCut cut accumulation may overflow finite binary64 '
            'after sequential-rounding headroom. Rescale the QUBO '
            'coefficients before conversion.'
        )
