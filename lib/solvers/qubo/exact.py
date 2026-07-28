"""Deterministic exhaustive-search solver for canonical QUBO problems.

The implementation deliberately favours readability over clever bit-level
optimisations.  Exact search is primarily a correctness oracle for small
problems: it gives heuristic and quantum solvers a known optimum to compare
against, and it catches mistakes in QUBO compilation and energy conventions.
"""

import math
import time

from .base import BaseQuboSolver, QuboSolveOutcome


class ExactQuboSolver(BaseQuboSolver):
    """Solve a small QUBO by evaluating every binary assignment.

    ``max_variables`` is a safety rail rather than an algorithmic restriction.
    Exhaustive search grows as ``2 ** num_variables``, so accidentally passing
    a large production problem can otherwise keep a process busy for an
    impractical amount of time.

    Supported configuration fields:

    ``max_variables``
        Largest accepted variable count. Defaults to ``24``. Set it higher
        only when the resulting search space is understood.
    ``timeout_seconds``
        Optional positive wall-clock limit. When reached, the best assignment
        visited so far is returned with status ``timeout``.
    """

    SOLVER_NAME = 'exact-enumeration'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'local-cpu'

    DEFAULT_MAX_VARIABLES = 24
    _ALLOWED_CONFIG_FIELDS = frozenset({'max_variables', 'timeout_seconds'})

    def _resolve_config(self, config):
        """Validate solver options while preserving the base class copy rule."""
        resolved = super()._resolve_config(config)
        self._reject_unknown_config_fields(resolved)

        max_variables = resolved.get(
            'max_variables',
            self.DEFAULT_MAX_VARIABLES,
        )
        timeout_seconds = resolved.get('timeout_seconds')

        self._validate_max_variables(max_variables)
        self._validate_timeout(timeout_seconds)

        return {
            'max_variables': max_variables,
            'timeout_seconds': timeout_seconds,
        }

    def _run(self, problem, config):
        """Enumerate the search space; orchestration intentionally reads plainly."""
        variable_count = problem['num_variables']
        self._guard_search_space(variable_count, config['max_variables'])

        total_candidates = 1 << variable_count  # 2 ** variable_count
        deadline = self._make_deadline(config['timeout_seconds'])
        best_sample = None
        best_energy = math.inf
        candidates_evaluated = 0

        for candidate_number in range(total_candidates):
            sample = self._decode_candidate(candidate_number, variable_count)
            energy = self._evaluate_candidate(problem, sample)
            candidates_evaluated += 1

            if energy < best_energy:
                best_sample, best_energy = sample, energy

            if (
                candidates_evaluated < total_candidates
                and self._deadline_reached(deadline)
            ):
                return self._timeout_outcome(
                    best_sample,
                    candidates_evaluated,
                    total_candidates,
                )

        return self._optimal_outcome(
            best_sample,
            candidates_evaluated,
            total_candidates,
        )

    def _reject_unknown_config_fields(self, config):
        """Keep misspelled configuration from silently changing an experiment."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'Unknown exact solver config fields: {names}')

    def _validate_max_variables(self, value):
        """Require a genuine non-negative integer, excluding Python booleans."""
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError('max_variables must be a non-negative integer.')

    def _validate_timeout(self, value):
        """Accept no deadline or a finite, strictly positive duration."""
        if value is None:
            return
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                'timeout_seconds must be a finite positive number or None.'
            )

    def _guard_search_space(self, variable_count, max_variables):
        """Fail early before constructing an unexpectedly enormous search."""
        if variable_count > max_variables:
            raise ValueError(
                f'Exact search received {variable_count} variables, exceeding '
                f'max_variables={max_variables}. Increase the limit explicitly '
                'only if exhaustive enumeration is intended.'
            )


    def _make_deadline(self, timeout_seconds):
        """Translate a relative timeout into one monotonic-clock deadline."""
        if timeout_seconds is None:
            return None
        return time.perf_counter() + timeout_seconds

    def _deadline_reached(self, deadline):
        """Isolate wall-clock interaction so timeout behaviour is testable."""
        return deadline is not None and time.perf_counter() >= deadline

    def _decode_candidate(self, candidate_number, variable_count):
        """Decode an integer into a stable, lexicographically ordered sample.

        The first variable is the most significant bit. Enumeration therefore
        starts at ``[0, ..., 0]`` and resolves equal-energy optima in favour of
        the lexicographically smallest sample.
        """
        return [
            (candidate_number >> shift) & 1
            for shift in range(variable_count - 1, -1, -1)
        ]

    def _evaluate_candidate(self, problem, sample):
        """Evaluate the sparse upper-triangular canonical QUBO expression."""
        energy = float(problem['offset'])
        for left, right, coefficient in problem['terms']:
            energy += coefficient * sample[left] * sample[right]
        return float(energy)

    def _optimal_outcome(
        self,
        best_sample,
        candidates_evaluated,
        total_candidates,
    ):
        """Describe a proof of optimality obtained by exhausting the space."""
        return QuboSolveOutcome(
            status='optimal',
            best_sample=best_sample,
            termination_reason='search_exhausted',
            metrics=self._search_metrics(
                candidates_evaluated,
                total_candidates,
                search_space_exhausted=True,
            ),
            metadata={'algorithm': 'exhaustive_enumeration'},
        )

    def _timeout_outcome(
        self,
        best_sample,
        candidates_evaluated,
        total_candidates,
    ):
        """Describe an interrupted search without claiming optimality."""
        return QuboSolveOutcome(
            status='timeout',
            best_sample=best_sample,
            termination_reason='timeout_reached',
            metrics=self._search_metrics(
                candidates_evaluated,
                total_candidates,
                search_space_exhausted=False,
            ),
            metadata={'algorithm': 'exhaustive_enumeration'},
        )

    def _search_metrics(
        self,
        candidates_evaluated,
        total_candidates,
        *,
        search_space_exhausted,
    ):
        """Build consistent progress metrics for both termination paths."""
        return {
            'candidates_evaluated': candidates_evaluated,
            'total_candidates': total_candidates,
            'search_space_exhausted': search_space_exhausted,
        }
