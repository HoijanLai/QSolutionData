"""Dependency-free structural types for native maximum-independent-set solvers."""

from typing import Any, NotRequired, Protocol, TypedDict

from .cbqm_protocol import ProofEvidence
from .solver_protocol import Solver, SolverConfig


class MisObjective(TypedDict):
    """Choose maximum cardinality or explicitly weighted MIS."""

    kind: str


class MisVertex(TypedDict):
    """One stable vertex in canonical index order."""

    index: int
    name: str
    weight: NotRequired[int | float]
    metadata: NotRequired[dict[str, Any]]


class MisFixedValue(TypedDict):
    """One hard vertex exclusion or inclusion."""

    index: int
    value: int


class MisProblem(TypedDict):
    """Static shape of one JSON-serializable ``mis.v1`` document."""

    schema: str
    problem_id: str
    objective: MisObjective
    vertices: list[MisVertex]
    edges: list[list[int]]
    fixed_values: NotRequired[list[MisFixedValue]]
    metadata: NotRequired[dict[str, Any]]


class MisSolverIdentity(TypedDict):
    """Stable implementation identity embedded in an MIS result."""

    name: str
    version: str
    backend: NotRequired[str]


class MisBounds(TypedDict, total=False):
    """Optional maximization lower/upper progress bounds."""

    incumbent_lower_bound: int | float
    optimum_upper_bound: int | float


class MisTraceEntry(TypedDict):
    """One optional progress observation."""

    step: int
    time_seconds: int | float
    selected_vertices: NotRequired[list[int]]
    objective_value: NotRequired[int | float]
    cardinality: NotRequired[int]
    total_weight: NotRequired[int | float]
    feasible: NotRequired[bool]
    metadata: NotRequired[dict[str, Any]]


class MisResult(TypedDict):
    """Static shape of one JSON-serializable ``mis-result.v1`` document."""

    schema: str
    problem_id: str
    solver: MisSolverIdentity
    status: str
    selected_vertices: list[int] | None
    objective_value: int | float | None
    cardinality: int | None
    total_weight: int | float | None
    feasible: bool | None
    bounds: NotRequired[MisBounds]
    proof: NotRequired[ProofEvidence]
    runtime_seconds: int | float
    termination_reason: NotRequired[str]
    metrics: NotRequired[dict[str, Any]]
    trace: NotRequired[list[MisTraceEntry]]
    metadata: NotRequired[dict[str, Any]]


class MisSolver(Solver[MisProblem, MisResult], Protocol):
    """Structural ``mis.v1 -> mis-result.v1`` solver interface."""

    def solve(
        self,
        problem: MisProblem,
        config: SolverConfig | None = None,
    ) -> MisResult:
        """Solve one MIS problem without mutating caller-owned inputs."""


__all__ = [
    'MisBounds',
    'MisFixedValue',
    'MisObjective',
    'MisProblem',
    'MisResult',
    'MisSolver',
    'MisSolverIdentity',
    'MisTraceEntry',
    'MisVertex',
]
