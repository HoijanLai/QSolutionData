"""Template-method base class for solvers that natively consume CBQM."""

import copy
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, final

from ...contracts.cbqm_validation import (
    _evaluate_cbqm_feasibility,
    _evaluate_cbqm_objective_exact,
    validate_cbqm_result,
)
from ...contracts.validation import (
    _fraction_to_json_number,
    _validate_binary_sample,
    validate_cbqm,
)


_RESULT_STATUSES = frozenset(
    {'optimal', 'feasible', 'infeasible', 'timeout', 'error', 'unknown'}
)
_INTERNAL_TRACE_REQUIRED_FIELDS = frozenset({'step', 'time_seconds'})
_INTERNAL_TRACE_OPTIONAL_FIELDS = frozenset(
    {'sample', 'metadata'}
)


@dataclass(frozen=True)
class CbqmSolveOutcome:
    """Untrusted algorithm-kernel output consumed by ``BaseCbqmSolver``.

    A kernel may nominate a sample and attach solver claims, bounds and
    bookkeeping.  It deliberately cannot nominate the public
    ``best_objective`` or ``feasibility`` fields.  The invariant wrapper derives
    both from the original ``cbqm.v1`` model.

    Internal trace entries follow the same rule: they may contain ``step``,
    ``time_seconds``, an optional ``sample`` and optional ``metadata``.  The
    wrapper adds the canonical objective and feasibility flag.
    """

    status: str
    best_sample: Sequence[int] | None
    termination_reason: str | None = None
    bounds: Mapping[str, Any] = field(default_factory=dict)
    proof: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    trace: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseCbqmSolver(ABC):
    """ABC owning the invariant ``cbqm.v1 -> cbqm-result.v1`` workflow.

    Subclasses implement only solver-specific configuration and ``_run``.
    Validation, defensive copying, canonical recomputation and final result
    validation stay here so every native CBQM solver has the same boundary.
    """

    SOLVER_NAME: ClassVar[str] = ''
    SOLVER_VERSION: ClassVar[str] = ''
    BACKEND: ClassVar[str] = 'local'

    @final
    def solve(self, problem, config=None):
        """Validate, execute and return one canonical CBQM result."""
        validate_cbqm(problem)
        resolved_config = self._resolve_config(config)
        algorithm_problem = copy.deepcopy(problem)

        started_at = time.perf_counter()
        outcome = self._run(algorithm_problem, resolved_config)
        runtime = time.perf_counter() - started_at

        result = self._build_result(problem, outcome, runtime)
        validate_cbqm_result(problem, result)
        return result

    def _resolve_config(self, config):
        """Own a deep configuration copy before handing it to the kernel."""
        if config is None:
            return {}
        if not isinstance(config, Mapping):
            raise TypeError('config must be a mapping or None.')
        return copy.deepcopy(dict(config))

    @abstractmethod
    def _run(self, problem, config):
        """Execute solver meta-logic and return ``CbqmSolveOutcome``."""
        raise NotImplementedError

    def _build_result(self, problem, outcome, runtime):
        """Turn an untrusted outcome into a closed public result document."""
        self._validate_solver_metadata()
        self._validate_outcome(outcome)
        self._validate_runtime(runtime)

        sample, objective, feasibility = self._canonical_candidate(
            problem,
            outcome.best_sample,
        )
        result = self._result_header(
            problem,
            outcome,
            runtime,
            sample,
            objective,
            feasibility,
        )
        self._attach_optional_result_fields(problem, result, outcome)
        return result

    def _canonical_candidate(self, problem, nominated_sample):
        """Recompute every public semantic attached to a nominated sample."""
        if nominated_sample is None:
            return None, None, None

        sample = list(nominated_sample)
        _validate_binary_sample(
            sample,
            len(problem['variables']),
            'CBQM outcome sample',
        )
        objective = _fraction_to_json_number(
            _evaluate_cbqm_objective_exact(problem, sample),
            'CBQM objective',
        )
        feasibility = _evaluate_cbqm_feasibility(problem, sample)
        return sample, objective, feasibility

    def _result_header(
        self,
        problem,
        outcome,
        runtime,
        sample,
        objective,
        feasibility,
    ):
        """Build the required fields in the same order as the wire contract."""
        return {
            'schema': 'cbqm-result.v1',
            'problem_id': problem['problem_id'],
            'solver': {
                'name': self.SOLVER_NAME,
                'version': self.SOLVER_VERSION,
                'backend': self.BACKEND,
            },
            'status': outcome.status,
            'best_sample': sample,
            'best_objective': objective,
            'feasibility': feasibility,
            'runtime_seconds': float(runtime),
            'metadata': copy.deepcopy(dict(outcome.metadata)),
        }

    def _attach_optional_result_fields(self, problem, result, outcome):
        """Copy optional claims and canonicalize every trace observation."""
        if outcome.bounds:
            result['bounds'] = copy.deepcopy(dict(outcome.bounds))
        if outcome.proof:
            result['proof'] = copy.deepcopy(dict(outcome.proof))
        if outcome.termination_reason is not None:
            result['termination_reason'] = outcome.termination_reason
        if outcome.metrics:
            result['metrics'] = copy.deepcopy(dict(outcome.metrics))
        if outcome.trace:
            result['trace'] = self._canonical_trace(problem, outcome.trace)

    def _canonical_trace(self, problem, trace):
        """Add original objective and feasibility to kernel observations."""
        public_trace = []
        for entry in trace:
            public_entry = {
                'step': entry['step'],
                'time_seconds': entry['time_seconds'],
            }
            if 'sample' in entry:
                sample, objective, feasibility = self._canonical_candidate(
                    problem,
                    entry['sample'],
                )
                public_entry.update({
                    'sample': sample,
                    'objective': objective,
                    'feasible': feasibility['feasible'],
                })
            if 'metadata' in entry:
                public_entry['metadata'] = copy.deepcopy(entry['metadata'])
            public_trace.append(public_entry)
        return public_trace

    def _validate_solver_metadata(self):
        """Validate identity constants supplied by a concrete implementation."""
        for field_name, value in (
            ('SOLVER_NAME', self.SOLVER_NAME),
            ('SOLVER_VERSION', self.SOLVER_VERSION),
            ('BACKEND', self.BACKEND),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f'{field_name} must be a non-empty string.')

    def _validate_outcome(self, outcome):
        """Reject malformed kernel output before canonical construction."""
        if not isinstance(outcome, CbqmSolveOutcome):
            raise TypeError('_run must return CbqmSolveOutcome.')
        if outcome.status not in _RESULT_STATUSES:
            raise ValueError(f"Unknown solver status '{outcome.status}'.")
        if (
            outcome.status in {'optimal', 'feasible'}
            and outcome.best_sample is None
        ):
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
            ('bounds', outcome.bounds),
            ('proof', outcome.proof),
            ('metrics', outcome.metrics),
            ('metadata', outcome.metadata),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f'{field_name} must be a mapping.')
        self._validate_internal_trace(outcome.trace)

    def _validate_internal_trace(self, trace):
        """Keep objective/feasibility out of the algorithm-owned trace."""
        if isinstance(trace, (str, bytes)) or not isinstance(trace, Sequence):
            raise TypeError('trace must be a sequence of mappings.')
        for position, entry in enumerate(trace):
            if not isinstance(entry, Mapping):
                raise TypeError('Every trace item must be a mapping.')
            keys = set(entry)
            missing = _INTERNAL_TRACE_REQUIRED_FIELDS - keys
            unknown = (
                keys
                - _INTERNAL_TRACE_REQUIRED_FIELDS
                - _INTERNAL_TRACE_OPTIONAL_FIELDS
            )
            if missing:
                names = ', '.join(sorted(missing))
                raise ValueError(
                    f'Internal trace {position} is missing fields: {names}.'
                )
            if unknown:
                names = ', '.join(sorted(str(name) for name in unknown))
                raise ValueError(
                    f'Internal trace {position} contains unknown fields: '
                    f'{names}.'
                )

    def _validate_runtime(self, runtime):
        """Defend the public duration field from an invalid clock value."""
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError('runtime_seconds must be finite and non-negative.')


__all__ = ['BaseCbqmSolver', 'CbqmSolveOutcome']
