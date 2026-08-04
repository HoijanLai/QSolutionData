"""Native mathematical MIS benchmark families."""

from .graph_families import build_mis_graph_family_suite
from .random_graph import build_random_mis_suite
from .weighted import build_weighted_mis_suite


def build_mis_mathematical_suite():
    """Build every deterministic native MIS fixture."""
    suite = {}
    for family in (
        build_mis_graph_family_suite(),
        build_weighted_mis_suite(),
        build_random_mis_suite(),
    ):
        overlap = set(suite) & set(family)
        if overlap:
            raise RuntimeError(
                f'Duplicate MIS benchmark names: {sorted(overlap)}'
            )
        suite.update(family)
    return suite


__all__ = [
    'build_mis_graph_family_suite',
    'build_mis_mathematical_suite',
    'build_random_mis_suite',
    'build_weighted_mis_suite',
]
