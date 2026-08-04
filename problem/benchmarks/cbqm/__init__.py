"""Mathematical CBQM benchmark families."""

from .exact_cover import build_exact_cover_suite
from .graph_coloring import build_graph_coloring_suite
from .knapsack import build_knapsack_suite
from .tsp import build_tsp_suite


def build_cbqm_mathematical_suite():
    """Build every deterministic mathematical CBQM fixture."""
    suite = {}
    for family in (
        build_tsp_suite(),
        build_knapsack_suite(),
        build_exact_cover_suite(),
        build_graph_coloring_suite(),
    ):
        overlap = set(suite) & set(family)
        if overlap:
            raise RuntimeError(
                f'Duplicate CBQM benchmark names: {sorted(overlap)}'
            )
        suite.update(family)
    return suite


__all__ = [
    'build_cbqm_mathematical_suite',
    'build_exact_cover_suite',
    'build_graph_coloring_suite',
    'build_knapsack_suite',
    'build_tsp_suite',
]
