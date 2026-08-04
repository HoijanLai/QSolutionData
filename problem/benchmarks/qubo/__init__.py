"""Mathematical QUBO benchmark families."""

from .degeneracy import build_degenerate_qubo_suite
from .planted_qubo import build_planted_qubo_suite
from .spin_glass import build_spin_glass_suite


def build_qubo_mathematical_suite():
    """Build every deterministic mathematical QUBO fixture."""
    suite = {}
    for family in (
        build_spin_glass_suite(),
        build_planted_qubo_suite(),
        build_degenerate_qubo_suite(),
    ):
        overlap = set(suite) & set(family)
        if overlap:
            raise RuntimeError(
                f'Duplicate QUBO benchmark names: {sorted(overlap)}'
            )
        suite.update(family)
    return suite


__all__ = [
    'build_degenerate_qubo_suite',
    'build_planted_qubo_suite',
    'build_qubo_mathematical_suite',
    'build_spin_glass_suite',
]
