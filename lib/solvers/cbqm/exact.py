"""Deterministic exhaustive search for small native CBQM problems.

This solver is intentionally a correctness oracle, not a scale claim.  It
enumerates every assignment left free by ``fixed_values``, filters the original
constraints directly, and compares the original objective with exact rational
arithmetic.  No QUBO penalties or compilation conventions enter the proof.
"""

import math
import time

from ...contracts.cbqm_validation import (
    _evaluate_cbqm_objective_exact,
    _is_cbqm_feasible_exact,
)
from ...contracts.validation import _fraction_to_json_number
from .base import BaseCbqmSolver, CbqmSolveOutcome


class ExactCbqmSolver(BaseCbqmSolver):
    """Prove optimality or infeasibility by exhausting a small CBQM domain."""

    SOLVER_NAME = 'exact-cbqm-enumeration'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'local-cpu'

    DEFAULT_MAX_VARIABLES = 24
    _ALLOWED_CONFIG_FIELDS = frozenset({'max_variables', 'timeout_seconds'})

    def _resolve_config(self, config):
        """Resolve the two explicit safety controls for exhaustive search."""
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
        """Enumerate, filter, compare, and report—the proof in pseudocode."""
        self._guard_search_space(problem, config['max_variables'])

        search = self._prepare_search(problem)
        deadline = self._make_deadline(config['timeout_seconds'])
        incumbent = None
        candidates_evaluated = 0
        feasible_candidates = 0

        for candidate_number in range(search['total_candidates']):
            sample = self._decode_candidate(candidate_number, search)
            candidates_evaluated += 1

            if self._is_feasible(problem, sample):
                feasible_candidates += 1
                incumbent = self._keep_better(problem, sample, incumbent)

            if self._search_interrupted(
                deadline,
                candidates_evaluated,
                search['total_candidates'],
            ):
                return self._timeout_outcome(
                    incumbent,
                    candidates_evaluated,
                    feasible_candidates,
                    search['total_candidates'],
                )

        return self._conclusive_outcome(
            incumbent,
            candidates_evaluated,
            feasible_candidates,
            search['total_candidates'],
        )

    def _prepare_search(self, problem):
        """Separate hard-fixed positions from the bits that require search."""
        fixed = {
            item['index']: item['value']
            for item in problem['fixed_values']
        }
        free_indices = [
            index
            for index in range(len(problem['variables']))
            if index not in fixed
        ]
        return {
            'fixed': fixed,
            'free_indices': free_indices,
            'variable_count': len(problem['variables']),
            'total_candidates': 1 << len(free_indices),
        }

    def _decode_candidate(self, candidate_number, search):
        """Map free bits into a full sample in lexicographic order."""
        sample = [0] * search['variable_count']
        for index, value in search['fixed'].items():
            sample[index] = value
        for position, variable_index in enumerate(search['free_indices']):
            shift = len(search['free_indices']) - position - 1
            sample[variable_index] = (candidate_number >> shift) & 1
        return sample

    def _is_feasible(self, problem, sample):
        """Keep hard-domain logic behind one replaceable predicate."""
        return _is_cbqm_feasible_exact(problem, sample)

    def _keep_better(self, problem, sample, incumbent):
        """Retain the better original objective with deterministic ties."""
        objective = _evaluate_cbqm_objective_exact(problem, sample)
        if incumbent is None or self._is_better_candidate(
            problem['objective']['sense'],
            objective,
            sample,
            incumbent['objective'],
            incumbent['sample'],
        ):
            return {'sample': sample, 'objective': objective}
        return incumbent

    def _is_better_candidate(
        self,
        sense,
        objective,
        sample,
        incumbent_objective,
        incumbent_sample,
    ):
        """Compare objective first and use sample order only for exact ties."""
        if sense == 'minimize':
            return (
                objective < incumbent_objective
                or (
                    objective == incumbent_objective
                    and sample < incumbent_sample
                )
            )
        return (
            objective > incumbent_objective
            or (
                objective == incumbent_objective
                and sample < incumbent_sample
            )
        )

    def _search_interrupted(
        self,
        deadline,
        candidates_evaluated,
        total_candidates,
    ):
        """Let a completed final assignment win over an expired deadline."""
        return (
            candidates_evaluated < total_candidates
            and self._deadline_reached(deadline)
        )

    def _conclusive_outcome(
        self,
        incumbent,
        candidates_evaluated,
        feasible_candidates,
        total_candidates,
    ):
        """Distinguish a proved optimum from a proved-empty feasible set."""
        if incumbent is None:
            return self._infeasible_outcome(
                candidates_evaluated,
                feasible_candidates,
                total_candidates,
            )
        return self._optimal_outcome(
            incumbent,
            candidates_evaluated,
            feasible_candidates,
            total_candidates,
        )

    def _optimal_outcome(
        self,
        incumbent,
        candidates_evaluated,
        feasible_candidates,
        total_candidates,
    ):
        """Return equal primal/dual bounds after exhaustive comparison."""
        objective = _fraction_to_json_number(
            incumbent['objective'],
            'CBQM optimum',
        )
        return CbqmSolveOutcome(
            status='optimal',
            best_sample=incumbent['sample'],
            termination_reason='search_exhausted',
            bounds={
                'primal_bound': objective,
                'dual_bound': objective,
                'absolute_gap': 0,
                'relative_gap': 0,
            },
            proof=self._proof('optimality', candidates_evaluated),
            metrics=self._search_metrics(
                candidates_evaluated,
                feasible_candidates,
                total_candidates,
                search_space_exhausted=True,
            ),
            metadata={'algorithm': 'exhaustive_enumeration'},
        )

    def _infeasible_outcome(
        self,
        candidates_evaluated,
        feasible_candidates,
        total_candidates,
    ):
        """Report proved infeasibility only after the domain is exhausted."""
        return CbqmSolveOutcome(
            status='infeasible',
            best_sample=None,
            termination_reason='search_exhausted_without_feasible_candidate',
            proof=self._proof('infeasibility', candidates_evaluated),
            metrics=self._search_metrics(
                candidates_evaluated,
                feasible_candidates,
                total_candidates,
                search_space_exhausted=True,
            ),
            metadata={'algorithm': 'exhaustive_enumeration'},
        )

    def _timeout_outcome(
        self,
        incumbent,
        candidates_evaluated,
        feasible_candidates,
        total_candidates,
    ):
        """Preserve a feasible incumbent without making a proof claim."""
        return CbqmSolveOutcome(
            status='timeout',
            best_sample=(
                None if incumbent is None else incumbent['sample']
            ),
            termination_reason='timeout_reached',
            bounds=self._timeout_bounds(incumbent),
            metrics=self._search_metrics(
                candidates_evaluated,
                feasible_candidates,
                total_candidates,
                search_space_exhausted=False,
            ),
            metadata={'algorithm': 'exhaustive_enumeration'},
        )

    def _timeout_bounds(self, incumbent):
        """A feasible incumbent is a primal bound, never a dual proof."""
        if incumbent is None:
            return {}
        return {
            'primal_bound': _fraction_to_json_number(
                incumbent['objective'],
                'CBQM incumbent objective',
            )
        }

    def _proof(self, claim, candidates_evaluated):
        """Describe solver evidence without marking it independently verified."""
        return {
            'claim': claim,
            'kind': 'exhaustive-enumeration',
            'producer': f'{self.SOLVER_NAME}@{self.SOLVER_VERSION}',
            'independently_verified': False,
            'details': {
                'assignments_checked': candidates_evaluated,
                'search_space_exhausted': True,
            },
        }

    def _search_metrics(
        self,
        candidates_evaluated,
        feasible_candidates,
        total_candidates,
        *,
        search_space_exhausted,
    ):
        """Build identical progress accounting for all termination paths."""
        return {
            'candidates_evaluated': candidates_evaluated,
            'feasible_candidates': feasible_candidates,
            'total_candidates': total_candidates,
            'search_space_exhausted': search_space_exhausted,
        }

    def _reject_unknown_config_fields(self, config):
        """Make configuration typos fail instead of changing experiments."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'Unknown exact CBQM solver config fields: {names}')

    def _validate_max_variables(self, value):
        """Require a genuine non-negative integer, excluding booleans."""
        if type(value) is not int or value < 0:
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

    def _guard_search_space(self, problem, max_variables):
        """Fail before an accidentally enormous exhaustive search starts."""
        variable_count = len(problem['variables'])
        if variable_count > max_variables:
            raise ValueError(
                f'Exact CBQM search received {variable_count} variables, '
                f'exceeding max_variables={max_variables}. Increase the limit '
                'explicitly only if exhaustive enumeration is intended.'
            )

    def _make_deadline(self, timeout_seconds):
        """Translate one relative timeout to a monotonic-clock deadline."""
        if timeout_seconds is None:
            return None
        return time.perf_counter() + timeout_seconds

    def _deadline_reached(self, deadline):
        """Isolate wall-clock interaction for deterministic timeout tests."""
        return deadline is not None and time.perf_counter() >= deadline


__all__ = ['ExactCbqmSolver']
