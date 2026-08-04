"""Fixed-seed Erdős-Rényi native MIS benchmark."""

import random

from .._metadata import benchmark_metadata
from ._common import _make_mis


def build_random_mis_suite():
    """Return one reproducible G(n,p) graph."""
    vertex_count = 28
    probability = 0.22
    seed = 424242
    rng = random.Random(seed)
    edges = [
        (left, right)
        for left in range(vertex_count)
        for right in range(left + 1, vertex_count)
        if rng.random() < probability
    ]
    return {
        'mis.erdos-renyi-28-p022': _make_mis(
            problem_id='benchmark-mis-erdos-renyi-28-p022',
            vertex_count=vertex_count,
            edges=edges,
            metadata=benchmark_metadata(
                family='maximum-independent-set',
                model='erdos-renyi',
                generator='build_random_mis_suite',
                seed=seed,
                size={
                    'vertex_count': vertex_count,
                    'edge_count': len(edges),
                },
                parameters={
                    'model': 'G(n,p)',
                    'edge_probability': probability,
                },
            ),
        ),
    }


__all__ = ['build_random_mis_suite']
