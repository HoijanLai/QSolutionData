"""Template-method base class for canonical QUBO solvers."""

import copy
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, final

from ...contracts.validation import (
    _evaluate_qubo,
    validate_qubo,
    validate_qubo_result,
)


_RESULT_STATUSES = {
    'optimal',
    'feasible',
    'infeasible',
    'timeout',
    'error',
    'unknown',
}


@dataclass(frozen=True)
class QuboSolveOutcome:
    """Representation-independent output of a QUBO algorithm kernel.

    The algorithm reports a candidate and termination information, but not the
    canonical energy. ``BaseQuboSolver`` recomputes that energy from the source
    ``qubo.v1`` problem so backend-specific energy conventions cannot leak into
    the public result.
    """

    status: str
    best_sample: Sequence[int] | None
    termination_reason: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    trace: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseQuboSolver(ABC):
    """ABC implementing the invariant ``qubo.v1`` solver wrapper.

    Subclasses declare stable solver metadata and implement ``_run``. They may
    override ``_resolve_config`` to validate solver-specific options, but the
    public ``solve`` workflow remains fixed and contract-compatible.
    """

    SOLVER_NAME: ClassVar[str] = ''
    SOLVER_VERSION: ClassVar[str] = ''
    BACKEND: ClassVar[str] = 'local'

    @final
    def solve(self, problem, config=None):
        """Validate, execute, and return a canonical ``qubo-result.v1``."""
        validate_qubo(problem)
        resolved_config = self._resolve_config(config)
        algorithm_problem = copy.deepcopy(problem)

        started_at = time.perf_counter()
        outcome = self._run(algorithm_problem, resolved_config)
        runtime = time.perf_counter() - started_at

        result = self._build_result(problem, outcome, runtime)
        validate_qubo_result(problem, result)
        return result

    def _resolve_config(self, config):
        """Copy the solver-owned configuration before algorithm execution."""
        if config is None:
            return {}
        if not isinstance(config, Mapping):
            raise TypeError('config must be a mapping or None.')
        return copy.deepcopy(dict(config))

    @abstractmethod
    def _run(self, problem, config):
        """Execute solver meta-logic and return ``QuboSolveOutcome``."""
        raise NotImplementedError

    def _build_result(self, problem, outcome, runtime):
        """Validate an algorithm outcome and construct ``qubo-result.v1``."""
        self._validate_solver_metadata()
        self._validate_outcome(outcome)
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError('runtime_seconds must be finite and non-negative.')

        sample = None
        energy = None
        if outcome.best_sample is not None:
            sample = list(outcome.best_sample)
            energy = _evaluate_qubo(problem, sample)

        result = {
            'schema': 'qubo-result.v1',
            'problem_id': problem['problem_id'],
            'solver': {
                'name': self.SOLVER_NAME,
                'version': self.SOLVER_VERSION,
                'backend': self.BACKEND,
            },
            'status': outcome.status,
            'best_sample': sample,
            'best_energy': energy,
            'runtime_seconds': float(runtime),
            'metadata': copy.deepcopy(dict(outcome.metadata)),
        }
        if outcome.termination_reason is not None:
            result['termination_reason'] = outcome.termination_reason
        if outcome.metrics:
            result['metrics'] = copy.deepcopy(dict(outcome.metrics))
        if outcome.trace:
            result['trace'] = copy.deepcopy(list(outcome.trace))
        return result

    def _validate_solver_metadata(self):
        """Validate stable identity fields supplied by a concrete solver."""
        for field_name, value in (
            ('SOLVER_NAME', self.SOLVER_NAME),
            ('SOLVER_VERSION', self.SOLVER_VERSION),
            ('BACKEND', self.BACKEND),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f'{field_name} must be a non-empty string.')

    def _validate_outcome(self, outcome):
        """Validate generic outcome semantics before result construction."""
        if not isinstance(outcome, QuboSolveOutcome):
            raise TypeError('_run must return QuboSolveOutcome.')
        if outcome.status not in _RESULT_STATUSES:
            raise ValueError(f"Unknown solver status '{outcome.status}'.")
        if outcome.status in {'optimal', 'feasible'} and outcome.best_sample is None:
            raise ValueError(
                f"Status '{outcome.status}' requires a best_sample."
            )
        if outcome.status == 'infeasible' and outcome.best_sample is not None:
            raise ValueError("Status 'infeasible' cannot include a best_sample.")
        if (
            outcome.termination_reason is not None
            and not isinstance(outcome.termination_reason, str)
        ):
            raise TypeError('termination_reason must be a string or None.')
        for field_name, value in (
            ('metrics', outcome.metrics),
            ('metadata', outcome.metadata),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f'{field_name} must be a mapping.')
        if isinstance(outcome.trace, (str, bytes)) or not isinstance(
            outcome.trace,
            Sequence,
        ):
            raise TypeError('trace must be a sequence of mappings.')
        if any(not isinstance(item, Mapping) for item in outcome.trace):
            raise TypeError('Every trace item must be a mapping.')
