"""Solvers that natively consume ``qubo.v1`` problems."""

from .base import BaseQuboSolver, QuboSolveOutcome
from .exact import ExactQuboSolver

__all__ = [
    'BaseQuboSolver',
    'ExactQuboSolver',
    'QaoaQuboSolver',
    'QuboSolveOutcome',
]


def __getattr__(name):
    """Load the optional NumPy-backed solver only when it is requested.

    Exact enumeration and the shared solver contract are dependency-free.
    Keeping QAOA behind this package attribute means applications can import
    those lightweight pieces without importing—or requiring—NumPy.
    """
    if name == 'QaoaQuboSolver':
        from .qaoa import QaoaQuboSolver

        globals()[name] = QaoaQuboSolver
        return QaoaQuboSolver

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose lazy solver names to IDEs and interactive discovery."""
    return sorted(set(globals()) | set(__all__))
