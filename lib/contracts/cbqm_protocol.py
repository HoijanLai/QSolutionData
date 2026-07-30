"""Dependency-free structural types for native ``cbqm.v1`` solvers."""

from typing import Any, NotRequired, Protocol, TypedDict

from .solver_protocol import Solver, SolverConfig


class CbqmVariable(TypedDict):
    """One stable binary variable in a canonical CBQM."""

    index: int
    name: str
    vartype: str
    kind: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class CbqmObjective(TypedDict):
    """Original constrained objective, before any penalty compilation."""

    sense: str
    offset: int | float
    linear: list[list[int | float]]
    quadratic: list[list[int | float]]


class CbqmConstraint(TypedDict):
    """One explicitly bounded linear constraint."""

    name: str
    family: str
    linear: list[list[int | float]]
    lower_bound: NotRequired[int | float]
    upper_bound: NotRequired[int | float]
    metadata: NotRequired[dict[str, Any]]


class CbqmFixedValue(TypedDict):
    """One hard binary assignment."""

    index: int
    value: int


class CbqmProblem(TypedDict):
    """Static shape of one JSON-serializable ``cbqm.v1`` document."""

    schema: str
    problem_id: str
    variables: list[CbqmVariable]
    objective: CbqmObjective
    constraints: list[CbqmConstraint]
    fixed_values: list[CbqmFixedValue]
    metadata: NotRequired[dict[str, Any]]


class CbqmSolverIdentity(TypedDict):
    """Stable identity embedded in a native CBQM result."""

    name: str
    version: str
    backend: NotRequired[str]


class CbqmConstraintViolation(TypedDict):
    """One exact fixed-value or linear-constraint violation."""

    constraint_name: str
    activity: int | float
    lower_bound: NotRequired[int | float]
    upper_bound: NotRequired[int | float]
    magnitude: int | float


class CbqmFeasibility(TypedDict):
    """Canonical feasibility summary recomputed from the source CBQM."""

    feasible: bool
    violated_count: int
    max_violation: int | float
    violations: list[CbqmConstraintViolation]


class CbqmBounds(TypedDict, total=False):
    """Optional solver-attested primal/dual progress."""

    primal_bound: int | float
    dual_bound: int | float
    absolute_gap: int | float
    relative_gap: int | float


class ProofEvidence(TypedDict):
    """Machine-readable claim metadata, not automatic persistent exactness."""

    claim: str
    kind: str
    producer: str
    independently_verified: bool
    details: NotRequired[dict[str, Any]]


class CbqmTraceEntry(TypedDict):
    """One optional progress observation."""

    step: int
    time_seconds: int | float
    sample: NotRequired[list[int]]
    objective: NotRequired[int | float]
    feasible: NotRequired[bool]
    metadata: NotRequired[dict[str, Any]]


class CbqmResult(TypedDict):
    """Static shape of one JSON-serializable ``cbqm-result.v1`` document."""

    schema: str
    problem_id: str
    solver: CbqmSolverIdentity
    status: str
    best_sample: list[int] | None
    best_objective: int | float | None
    feasibility: CbqmFeasibility | None
    bounds: NotRequired[CbqmBounds]
    proof: NotRequired[ProofEvidence]
    runtime_seconds: int | float
    termination_reason: NotRequired[str]
    metrics: NotRequired[dict[str, Any]]
    trace: NotRequired[list[CbqmTraceEntry]]
    metadata: NotRequired[dict[str, Any]]


class CbqmSolver(Solver[CbqmProblem, CbqmResult], Protocol):
    """Structural ``cbqm.v1 -> cbqm-result.v1`` solver interface."""

    def solve(
        self,
        problem: CbqmProblem,
        config: SolverConfig | None = None,
    ) -> CbqmResult:
        """Solve one CBQM without mutating the problem or configuration."""


__all__ = [
    'CbqmBounds',
    'CbqmConstraint',
    'CbqmConstraintViolation',
    'CbqmFeasibility',
    'CbqmFixedValue',
    'CbqmObjective',
    'CbqmProblem',
    'CbqmResult',
    'CbqmSolver',
    'CbqmSolverIdentity',
    'CbqmTraceEntry',
    'CbqmVariable',
    'ProofEvidence',
]
