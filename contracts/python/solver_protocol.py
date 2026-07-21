"""Dependency-free structural interface for QUBO solvers.

The JSON Schemas in ``contracts/schemas`` are authoritative for payload shape.
This module only fixes the Python call signature and intentionally introduces
no base-class or third-party dependency.
"""

from typing import Any, Mapping, Protocol


QuboProblem = Mapping[str, Any]
SolverConfig = Mapping[str, Any]
QuboResult = Mapping[str, Any]


class QuboSolver(Protocol):
    """A solver accepted by the QSolutionData solver wrapper."""

    def solve(
        self,
        problem: QuboProblem,
        config: SolverConfig | None = None,
    ) -> QuboResult:
        """Solve a ``qubo.v1`` problem and return ``qubo-result.v1``."""
