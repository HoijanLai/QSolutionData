"""Solvers that natively consume ``qubo.v1`` problems."""

from .base import BaseQuboSolver, QuboSolveOutcome
from .exact import ExactQuboSolver

__all__ = [
    'AerMpsQaoaSolver',
    'BaseQuboSolver',
    'ExactQuboSolver',
    'GoemansWilliamsonQuboSolver',
    'GwBranchAndBoundQuboSolver',
    'QaoaQuboSolver',
    'QrbnbrQuboSolver',
    'QuboSolveOutcome',
    'ScmfQaoaSolver',
    'SimulatedAnnealingQuboSolver',
]


def __getattr__(name):
    """Load optional NumPy-backed solvers only when they are requested.

    Exact enumeration and the shared solver contract are dependency-free.
    Keeping QAOA and simulated annealing behind this package attribute means
    applications can import those lightweight pieces without importing—or
    requiring—NumPy.
    """
    if name == 'AerMpsQaoaSolver':
        from .aer_mps_qaoa import AerMpsQaoaSolver

        globals()[name] = AerMpsQaoaSolver
        return AerMpsQaoaSolver
    if name == 'QaoaQuboSolver':
        from .qaoa import QaoaQuboSolver

        globals()[name] = QaoaQuboSolver
        return QaoaQuboSolver
    if name == 'GoemansWilliamsonQuboSolver':
        from .goemans_williamson import GoemansWilliamsonQuboSolver

        globals()[name] = GoemansWilliamsonQuboSolver
        return GoemansWilliamsonQuboSolver
    if name == 'GwBranchAndBoundQuboSolver':
        from .gw_bnb import GwBranchAndBoundQuboSolver

        globals()[name] = GwBranchAndBoundQuboSolver
        return GwBranchAndBoundQuboSolver
    if name == 'QrbnbrQuboSolver':
        from .qrbnbr import QrbnbrQuboSolver

        globals()[name] = QrbnbrQuboSolver
        return QrbnbrQuboSolver
    if name == 'ScmfQaoaSolver':
        from .scmf_qaoa import ScmfQaoaSolver

        globals()[name] = ScmfQaoaSolver
        return ScmfQaoaSolver
    if name == 'SimulatedAnnealingQuboSolver':
        from .simulated_annealing import SimulatedAnnealingQuboSolver

        globals()[name] = SimulatedAnnealingQuboSolver
        return SimulatedAnnealingQuboSolver

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose lazy solver names to IDEs and interactive discovery."""
    return sorted(set(globals()) | set(__all__))
