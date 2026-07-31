"""Checked-in exact reference witnesses for mathematical benchmarks.

The references were produced by the representation-native Exact solvers and
audited as a complete catalog on 2026-07-31.  They are intentionally separate
from the generators: solvers can consume fresh problem documents without
silently participating in construction of their expected answer.
"""

import copy


_EXACT_REFERENCES = {
    'qubo.ea-grid-4x4': {
        'solution': [
            0, 0, 0, 1,
            0, 1, 0, 0,
            0, 1, 1, 0,
            0, 0, 1, 0,
        ],
        'objective_value': -20,
    },
    'qubo.sk-15': {
        'solution': [
            0, 1, 0, 0, 1,
            0, 0, 1, 0, 1,
            0, 0, 1, 1, 1,
        ],
        'objective_value': -35,
    },
    'qubo.planted-consistency-18': {
        'solution': [
            0, 0, 0, 0, 0, 1,
            0, 1, 0, 1, 0, 0,
            1, 1, 1, 0, 0, 1,
        ],
        'objective_value': 0,
    },
    'qubo.ferromagnetic-cycle-18': {
        'solution': [0] * 18,
        'objective_value': 0,
    },
    'qubo.frustrated-cycle-17': {
        'solution': [
            0, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
        ],
        'objective_value': 1,
    },
    'cbqm.tsp-4': {
        'solution': [
            1, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0,
            0, 1, 0, 0,
        ],
        'objective_value': 14,
    },
    'cbqm.knapsack-16': {
        'solution': [
            1, 1, 1, 0,
            0, 0, 1, 1,
            0, 0, 1, 1,
            1, 0, 1, 0,
        ],
        'objective_value': 115,
    },
    'cbqm.exact-cover-8x12': {
        'solution': [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 1, 1,
        ],
        'objective_value': 2,
    },
    'cbqm.graph-coloring-5x3': {
        'solution': [
            1, 0, 0,
            0, 0, 1,
            0, 1, 0,
            1, 0, 0,
            0, 1, 0,
        ],
        'objective_value': 4,
    },
    'mis.path-32': {
        'solution': list(range(0, 32, 2)),
        'objective_value': 16,
    },
    'mis.cycle-31': {
        'solution': list(range(0, 30, 2)),
        'objective_value': 15,
    },
    'mis.clique-36': {
        'solution': [0],
        'objective_value': 1,
    },
    'mis.complete-bipartite-12x12': {
        'solution': list(range(12)),
        'objective_value': 12,
    },
    'mis.grid-5x6': {
        'solution': [
            0, 2, 4,
            7, 9, 11,
            12, 14, 16,
            19, 21, 23,
            24, 26, 28,
        ],
        'objective_value': 15,
    },
    'mis.weighted-path-30': {
        'solution': [
            0, 2, 4, 6, 8,
            10, 12, 14, 16, 18,
            21, 24, 26, 29,
        ],
        'objective_value': 69,
    },
    'mis.erdos-renyi-28-p022': {
        'solution': [
            5, 9, 11, 13,
            14, 15, 16, 19,
            22, 25, 26, 27,
        ],
        'objective_value': 12,
    },
}


def build_mathematical_benchmark_references():
    """Return fresh exact reference mappings keyed like the problem catalog."""
    return copy.deepcopy(_EXACT_REFERENCES)


__all__ = ['build_mathematical_benchmark_references']
