"""Dependency-free structural interfaces for optimization solvers.

The JSON Schemas in the top-level ``contracts/schemas`` directory are
authoritative for each serializable payload shape. ``Solver`` fixes only the
representation-neutral Python call shape; representation-specific protocols
such as ``QuboSolver`` bind that shape to concrete input and output contracts.
No inheritance by implementations or third-party dependency is required.
"""

from typing import Any, Mapping, Protocol, TypeVar


QuboProblem = Mapping[str, Any]
SolverConfig = Mapping[str, Any]
QuboResult = Mapping[str, Any]


ProblemT_contra = TypeVar('ProblemT_contra', contravariant=True)
ResultT_co = TypeVar('ResultT_co', covariant=True)


class Solver(Protocol[ProblemT_contra, ResultT_co]):
    """Representation-neutral structural interface for optimization solvers.

    ``ProblemT_contra`` and ``ResultT_co`` must be bound by a concrete solver
    contract. This protocol intentionally assigns no schema or numerical
    semantics to either value; those belong to specializations such as
    :class:`QuboSolver` or future CBQM/MIS protocols.
    """

    def solve(
        self,
        problem: ProblemT_contra,
        config: SolverConfig | None = None,
    ) -> ResultT_co:
        """Solve one problem without mutating the problem or configuration."""


class QuboSolver(Solver[QuboProblem, QuboResult], Protocol):
    """Structural interface implemented by every canonical QUBO solver.

    Implementations do not need to inherit from this protocol. Any object with
    a compatible ``solve`` method can be used by the pipeline and type checker.
    """

    def solve(
        self,
        problem: QuboProblem,
        config: SolverConfig | None = None,
    ) -> QuboResult:
        """Solve a ``qubo.v1`` problem and return ``qubo-result.v1``.

        Args:
            problem: Read-only canonical QUBO mapping.
            config: Optional read-only, solver-owned configuration mapping.

        Returns:
            Mapping conforming to ``qubo-result.v1``. Its ``best_energy`` must
            include the QUBO offset and its sample must follow variable order.

        Notes:
            Implementations must not mutate either input. Normal algorithmic
            termination belongs in the result status; malformed input may raise
            ``ValueError`` or a more specific input exception.
        """
