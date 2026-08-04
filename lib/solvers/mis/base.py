"""Template-method base class for native maximum-independent-set solvers."""

import copy
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, final

from ...contracts.mis_validation import (
    _evaluate_mis_solution,
    _validate_selected_vertices,
    validate_mis,
    validate_mis_result,
)


_RESULT_STATUSES = frozenset(
    {'optimal', 'feasible', 'infeasible', 'timeout', 'error', 'unknown'}
)
_INTERNAL_TRACE_REQUIRED_FIELDS = frozenset({'step', 'time_seconds'})
_INTERNAL_TRACE_OPTIONAL_FIELDS = frozenset(
    {'selected_vertices', 'metadata'}
)


@dataclass(frozen=True)
class MisSolveOutcome:
    """Untrusted native MIS kernel output.

    The kernel nominates only a canonical vertex-index set plus claims and
    bookkeeping. ``BaseMisSolver`` derives objective, cardinality, total weight
    and independent-set feasibility from the source ``mis.v1`` graph.
    """

    status: str
    selected_vertices: Sequence[int] | None
    termination_reason: str | None = None
    bounds: Mapping[str, Any] = field(default_factory=dict)
    proof: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    trace: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseMisSolver(ABC):
    """ABC owning the invariant ``mis.v1 -> mis-result.v1`` workflow."""

    SOLVER_NAME: ClassVar[str] = ''
    SOLVER_VERSION: ClassVar[str] = ''
    BACKEND: ClassVar[str] = 'local'

    @final
    def solve(self, problem, config=None):
        """Validate, execute and return one canonical native MIS result."""
        validate_mis(problem)
        resolved_config = self._resolve_config(config)
        algorithm_problem = copy.deepcopy(problem)

        started_at = time.perf_counter()
        outcome = self._run(algorithm_problem, resolved_config)
        runtime = time.perf_counter() - started_at

        result = self._build_result(problem, outcome, runtime)
        validate_mis_result(problem, result)
        return result

    def _resolve_config(self, config):
        """Give the kernel an owned deep copy of its configuration."""
        if config is None:
            return {}
        if not isinstance(config, Mapping):
            raise TypeError('config must be a mapping or None.')
        return copy.deepcopy(dict(config))

    @abstractmethod
    def _run(self, problem, config):
        """Execute solver meta-logic and return ``MisSolveOutcome``."""
        raise NotImplementedError

    def _build_result(self, problem, outcome, runtime):
        """Convert an untrusted outcome into the closed public wire format."""
        self._validate_solver_metadata()
        self._validate_outcome(outcome)
        self._validate_runtime(runtime)

        selected, semantics = self._canonical_candidate(
            problem,
            outcome.selected_vertices,
        )
        result = self._result_header(
            problem,
            outcome,
            runtime,
            selected,
            semantics,
        )
        self._attach_optional_fields(problem, result, outcome)
        return result

    def _canonical_candidate(self, problem, nominated_vertices):
        """Validate the canonical index set and recompute all public metrics."""
        if nominated_vertices is None:
            return None, None
        selected = list(nominated_vertices)
        _validate_selected_vertices(
            selected,
            len(problem['vertices']),
            'MIS outcome selected_vertices',
        )
        return selected, _evaluate_mis_solution(problem, selected)

    def _result_header(
        self,
        problem,
        outcome,
        runtime,
        selected,
        semantics,
    ):
        """Build required fields in public contract order."""
        return {
            'schema': 'mis-result.v1',
            'problem_id': problem['problem_id'],
            'solver': {
                'name': self.SOLVER_NAME,
                'version': self.SOLVER_VERSION,
                'backend': self.BACKEND,
            },
            'status': outcome.status,
            'selected_vertices': selected,
            'objective_value': (
                None if semantics is None else semantics['objective_value']
            ),
            'cardinality': (
                None if semantics is None else semantics['cardinality']
            ),
            'total_weight': (
                None if semantics is None else semantics['total_weight']
            ),
            'feasible': (
                None if semantics is None else semantics['feasible']
            ),
            'runtime_seconds': float(runtime),
            'metadata': copy.deepcopy(dict(outcome.metadata)),
        }

    def _attach_optional_fields(self, problem, result, outcome):
        """Copy claims and canonicalize every trace witness."""
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
        """Add canonical metrics to internal sample-only observations."""
        public_trace = []
        for entry in trace:
            public_entry = {
                'step': entry['step'],
                'time_seconds': entry['time_seconds'],
            }
            if 'selected_vertices' in entry:
                selected, semantics = self._canonical_candidate(
                    problem,
                    entry['selected_vertices'],
                )
                public_entry.update({
                    'selected_vertices': selected,
                    **semantics,
                })
            if 'metadata' in entry:
                public_entry['metadata'] = copy.deepcopy(entry['metadata'])
            public_trace.append(public_entry)
        return public_trace

    def _validate_solver_metadata(self):
        """Validate identity constants declared by the concrete solver."""
        for field_name, value in (
            ('SOLVER_NAME', self.SOLVER_NAME),
            ('SOLVER_VERSION', self.SOLVER_VERSION),
            ('BACKEND', self.BACKEND),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f'{field_name} must be a non-empty string.')

    def _validate_outcome(self, outcome):
        """Reject malformed kernel output before result construction."""
        if not isinstance(outcome, MisSolveOutcome):
            raise TypeError('_run must return MisSolveOutcome.')
        if outcome.status not in _RESULT_STATUSES:
            raise ValueError(f"Unknown solver status '{outcome.status}'.")
        if (
            outcome.status in {'optimal', 'feasible'}
            and outcome.selected_vertices is None
        ):
            raise ValueError(
                f"Status '{outcome.status}' requires selected_vertices."
            )
        if (
            outcome.status == 'infeasible'
            and outcome.selected_vertices is not None
        ):
            raise ValueError(
                "Status 'infeasible' cannot include selected_vertices."
            )
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
        """Prevent kernels from supplying trusted derived trace values."""
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
        """Defend the result duration from an invalid clock value."""
        if not math.isfinite(runtime) or runtime < 0:
            raise ValueError('runtime_seconds must be finite and non-negative.')


__all__ = ['BaseMisSolver', 'MisSolveOutcome']
