"""Seeded maximum-weight independent-set benchmark."""

import random

from .._metadata import benchmark_metadata
from ._common import _make_mis, _path_edges


def build_weighted_mis_suite():
    """Return one weighted path with zero/negative values and hard endpoints."""
    vertex_count = 30
    seed = 57721
    rng = random.Random(seed)
    weights = [
        rng.randrange(-3, 11)
        for _ in range(vertex_count)
    ]
    edges = _path_edges(vertex_count)
    fixed_values = [
        {'index': 0, 'value': 1},
        {'index': 1, 'value': 0},
    ]
    return {
        'mis.weighted-path-30': _make_mis(
            problem_id='benchmark-mwis-weighted-path-30',
            vertex_count=vertex_count,
            edges=edges,
            weights=weights,
            fixed_values=fixed_values,
            metadata=benchmark_metadata(
                family='maximum-weight-independent-set',
                model='weighted-path',
                generator='build_weighted_mis_suite',
                seed=seed,
                size={
                    'vertex_count': vertex_count,
                    'edge_count': len(edges),
                    'fixed_value_count': len(fixed_values),
                },
                parameters={
                    'weight_distribution': (
                        'discrete-uniform-minus-3-through-10'
                    ),
                    'weights': weights,
                },
            ),
        ),
    }


__all__ = ['build_weighted_mis_suite']
