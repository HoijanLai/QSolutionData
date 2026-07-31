"""Feasibility-first local search for native constrained BQMs.

The solver works on the original ``cbqm.v1`` feasible region.  It does not
compile constraints into penalty terms, so objective values never compete with
an arbitrary penalty scale.  Search is deliberately split into two phases:

1. repair a binary assignment until every original constraint is satisfied;
2. move only between feasible assignments while improving the true objective.

This is a heuristic baseline, not a proof procedure.  Failure to discover a
feasible assignment is therefore reported as ``unknown`` rather than
``infeasible``.  Likewise, a locally optimal incumbent is only ``feasible``.
"""

import math
import random
import time
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

from ...contracts.cbqm_validation import (
    _evaluate_cbqm_objective_exact,
    _is_cbqm_feasible_exact,
)
from ...contracts.validation import _fraction_to_json_number
from .base import BaseCbqmSolver, CbqmSolveOutcome


@dataclass(frozen=True)
class _PreparedConstraint:
    """One linear constraint converted once to exact internal arithmetic."""

    terms: tuple[tuple[int, Fraction], ...]
    lower_bound: Fraction | None
    upper_bound: Fraction | None


@dataclass(frozen=True)
class _PreparedCbqm:
    """Validated model information arranged for repeated neighborhood scans."""

    variable_count: int
    free_indices: tuple[int, ...]
    fixed_values: tuple[tuple[int, int], ...]
    constraints: tuple[_PreparedConstraint, ...]


@dataclass(frozen=True, order=True)
class _ViolationScore:
    """Lexicographic feasibility score; zero in every field means feasible.

    ``violated_count`` comes first so repair prefers satisfying an additional
    complete constraint.  Exact total and maximum violation then distinguish
    assignments with the same number of violated constraints.  The objective
    is intentionally absent: an attractive business objective must never make
    an infeasible point appear better than a less-infeasible point.
    """

    violated_count: int
    total_violation: Fraction
    max_violation: Fraction

    @property
    def feasible(self):
        """Return the semantic condition without duplicating tuple checks."""
        return self.violated_count == 0


@dataclass
class _SearchMetrics:
    """Mutable counters kept out of the pseudocode-level solver flow."""

    starts_attempted: int = 0
    feasible_starts: int = 0
    repair_steps: int = 0
    local_steps: int = 0
    single_neighbors_evaluated: int = 0
    pair_neighbors_evaluated: int = 0
    incumbent_updates: int = 0
    pair_scan_truncated: bool = False


@dataclass
class _SearchState:
    """Incumbent and optional observations shared across restart attempts."""

    incumbent_sample: list[int] | None = None
    incumbent_objective: Fraction | None = None
    trace: list[dict] = field(default_factory=list)


class LocalSearchCbqmSolver(BaseCbqmSolver):
    """Repair constraints first, then improve within the feasible region."""

    SOLVER_NAME = 'feasibility-first-cbqm-local-search'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'local-cpu'

    DEFAULT_SEED = 0
    DEFAULT_MAX_RESTARTS = 16
    DEFAULT_MAX_REPAIR_STEPS = 200
    DEFAULT_MAX_LOCAL_STEPS = 100
    DEFAULT_MAX_PAIR_EVALUATIONS = 2_000
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'seed',
            'max_restarts',
            'max_repair_steps',
            'max_local_steps',
            'max_pair_evaluations',
            'timeout_seconds',
            'include_trace',
        }
    )

    def _resolve_config(self, config):
        """Resolve a closed, type-sensitive experiment configuration."""
        resolved = super()._resolve_config(config)
        self._reject_unknown_config_fields(resolved)

        values = {
            'seed': resolved.get('seed', self.DEFAULT_SEED),
            'max_restarts': resolved.get(
                'max_restarts',
                self.DEFAULT_MAX_RESTARTS,
            ),
            'max_repair_steps': resolved.get(
                'max_repair_steps',
                self.DEFAULT_MAX_REPAIR_STEPS,
            ),
            'max_local_steps': resolved.get(
                'max_local_steps',
                self.DEFAULT_MAX_LOCAL_STEPS,
            ),
            'max_pair_evaluations': resolved.get(
                'max_pair_evaluations',
                self.DEFAULT_MAX_PAIR_EVALUATIONS,
            ),
            'timeout_seconds': resolved.get('timeout_seconds'),
            'include_trace': resolved.get('include_trace', False),
        }
        self._validate_seed(values['seed'])
        for field_name in (
            'max_restarts',
            'max_repair_steps',
            'max_local_steps',
            'max_pair_evaluations',
        ):
            self._validate_non_negative_integer(
                values[field_name],
                field_name,
            )
        self._validate_timeout(values['timeout_seconds'])
        self._validate_boolean(values['include_trace'], 'include_trace')
        return values

    def _run(self, problem, config):
        """Prepare, restart, repair, improve, retain: the whole algorithm."""
        prepared = self._prepare_problem(problem)
        deadline = self._make_deadline(config['timeout_seconds'])
        generator = random.Random(config['seed'])
        metrics = _SearchMetrics()
        state = _SearchState()
        started_at = time.perf_counter()

        for restart in range(config['max_restarts'] + 1):
            if self._deadline_reached(deadline):
                return self._timeout_outcome(state, metrics, config)

            metrics.starts_attempted += 1
            sample = self._initial_sample(prepared, restart, generator)
            sample, timed_out = self._repair_to_feasibility(
                prepared,
                sample,
                config,
                deadline,
                generator,
                metrics,
            )
            if timed_out:
                return self._timeout_outcome(state, metrics, config)
            if sample is None:
                continue

            metrics.feasible_starts += 1
            sample, local_optimum, timed_out = self._improve_feasibly(
                problem,
                prepared,
                sample,
                config,
                deadline,
                metrics,
            )
            self._consider_incumbent(
                problem,
                sample,
                restart,
                local_optimum,
                config,
                started_at,
                state,
                metrics,
            )
            if timed_out:
                return self._timeout_outcome(state, metrics, config)

        return self._completed_outcome(state, metrics, config)

    def _prepare_problem(self, problem):
        """Separate immutable fixed bits and pre-convert constraint numbers."""
        fixed = {
            item['index']: item['value']
            for item in problem['fixed_values']
        }
        free_indices = tuple(
            index
            for index in range(len(problem['variables']))
            if index not in fixed
        )
        constraints = tuple(
            self._prepare_constraint(constraint)
            for constraint in problem['constraints']
        )
        return _PreparedCbqm(
            variable_count=len(problem['variables']),
            free_indices=free_indices,
            fixed_values=tuple(sorted(fixed.items())),
            constraints=constraints,
        )

    def _prepare_constraint(self, constraint):
        """Convert one validated wire constraint into a compact exact record."""
        return _PreparedConstraint(
            terms=tuple(
                (index, Fraction(coefficient))
                for index, coefficient in constraint['linear']
            ),
            lower_bound=(
                Fraction(constraint['lower_bound'])
                if 'lower_bound' in constraint
                else None
            ),
            upper_bound=(
                Fraction(constraint['upper_bound'])
                if 'upper_bound' in constraint
                else None
            ),
        )

    def _initial_sample(self, prepared, restart, generator):
        """Use one canonical zero start, then seeded random restarts."""
        sample = [0] * prepared.variable_count
        if restart:
            for index in prepared.free_indices:
                sample[index] = generator.getrandbits(1)
        for index, value in prepared.fixed_values:
            sample[index] = value
        return sample

    def _repair_to_feasibility(
        self,
        prepared,
        initial_sample,
        config,
        deadline,
        generator,
        metrics,
    ):
        """Walk through least-violating one/two-bit neighbors.

        A repair move may be level or temporarily worse.  Requiring strict
        improvement would trap equality-constrained models whose feasible
        components are separated in the one-bit hypercube.  A per-start visited
        set prevents immediate cycles; seeded tie selection supplies controlled
        diversification while preserving reproducibility.
        """
        sample = initial_sample
        visited = {tuple(sample)}

        for _ in range(config['max_repair_steps']):
            if self._deadline_reached(deadline):
                return None, True
            if self._violation_score(prepared, sample).feasible:
                return sample, False

            neighbor, timed_out = self._best_repair_neighbor(
                prepared,
                sample,
                visited,
                config['max_pair_evaluations'],
                deadline,
                generator,
                metrics,
            )
            if timed_out:
                return None, True
            if neighbor is None:
                break

            sample = neighbor
            visited.add(tuple(sample))
            metrics.repair_steps += 1

        if self._violation_score(prepared, sample).feasible:
            return sample, False
        return None, False

    def _best_repair_neighbor(
        self,
        prepared,
        sample,
        visited,
        max_pair_evaluations,
        deadline,
        generator,
        metrics,
    ):
        """Choose the best unseen feasibility neighbor, with seen fallback."""
        best_unseen = None
        best_seen = None

        for indices in self._repair_moves(
            prepared,
            max_pair_evaluations,
            metrics,
        ):
            if self._deadline_reached(deadline):
                return None, True
            candidate = self._flipped_sample(sample, indices)
            score = self._violation_score(prepared, candidate)
            item = (score, candidate)
            if tuple(candidate) in visited:
                best_seen = self._keep_best_repair_items(best_seen, item)
            else:
                best_unseen = self._keep_best_repair_items(
                    best_unseen,
                    item,
                )

        pool = best_unseen if best_unseen is not None else best_seen
        if pool is None:
            return None, False
        candidates = sorted(item[1] for item in pool)
        return generator.choice(candidates), False

    def _repair_moves(
        self,
        prepared,
        max_pair_evaluations,
        metrics,
    ):
        """Yield all single flips and a bounded prefix of pair flips."""
        for index in prepared.free_indices:
            metrics.single_neighbors_evaluated += 1
            yield (index,)

        pairs_scanned = 0
        for left, right in combinations(prepared.free_indices, 2):
            if pairs_scanned >= max_pair_evaluations:
                metrics.pair_scan_truncated = True
                break
            pairs_scanned += 1
            metrics.pair_neighbors_evaluated += 1
            yield (left, right)

    def _keep_best_repair_items(self, incumbent, candidate):
        """Retain every exact tie so the seeded generator can diversify."""
        if incumbent is None or candidate[0] < incumbent[0][0]:
            return [candidate]
        if candidate[0] == incumbent[0][0]:
            incumbent.append(candidate)
        return incumbent

    def _violation_score(self, prepared, sample):
        """Aggregate exact lower/upper-bound violations for repair ordering."""
        violations = []
        for constraint in prepared.constraints:
            activity = sum(
                (
                    coefficient * sample[index]
                    for index, coefficient in constraint.terms
                ),
                start=Fraction(0),
            )
            magnitude = self._constraint_violation(constraint, activity)
            if magnitude:
                violations.append(magnitude)
        return _ViolationScore(
            violated_count=len(violations),
            total_violation=sum(violations, start=Fraction(0)),
            max_violation=max(violations, default=Fraction(0)),
        )

    def _constraint_violation(self, constraint, activity):
        """Return the exact distance outside one closed interval."""
        if (
            constraint.lower_bound is not None
            and activity < constraint.lower_bound
        ):
            return constraint.lower_bound - activity
        if (
            constraint.upper_bound is not None
            and activity > constraint.upper_bound
        ):
            return activity - constraint.upper_bound
        return Fraction(0)

    def _improve_feasibly(
        self,
        problem,
        prepared,
        initial_sample,
        config,
        deadline,
        metrics,
    ):
        """Repeatedly take the best feasible improving one/two-bit move."""
        sample = initial_sample
        for _ in range(config['max_local_steps']):
            if self._deadline_reached(deadline):
                return sample, False, True
            neighbor, timed_out, neighborhood_exhausted = (
                self._best_improving_neighbor(
                    problem,
                    prepared,
                    sample,
                    config['max_pair_evaluations'],
                    deadline,
                    metrics,
                )
            )
            if timed_out:
                return sample, False, True
            if neighbor is None:
                return sample, neighborhood_exhausted, False
            sample = neighbor
            metrics.local_steps += 1
        return sample, False, False

    def _best_improving_neighbor(
        self,
        problem,
        prepared,
        sample,
        max_pair_evaluations,
        deadline,
        metrics,
    ):
        """Scan feasible neighbors and retain the best original objective."""
        incumbent = {
            'sample': sample,
            'objective': _evaluate_cbqm_objective_exact(problem, sample),
        }
        best = incumbent
        neighborhood_exhausted = self._pair_scan_is_complete(
            prepared,
            max_pair_evaluations,
        )

        for indices in self._local_moves(
            prepared,
            max_pair_evaluations,
            metrics,
        ):
            if self._deadline_reached(deadline):
                return None, True, False
            candidate = self._flipped_sample(sample, indices)
            if not _is_cbqm_feasible_exact(problem, candidate):
                continue
            objective = _evaluate_cbqm_objective_exact(problem, candidate)
            if self._is_better_candidate(
                problem['objective']['sense'],
                objective,
                candidate,
                best['objective'],
                best['sample'],
            ):
                best = {'sample': candidate, 'objective': objective}

        if best is incumbent:
            return None, False, neighborhood_exhausted
        return best['sample'], False, neighborhood_exhausted

    def _pair_scan_is_complete(self, prepared, max_pair_evaluations):
        """Tell local-optimum claims apart from budget-truncated scans."""
        free_count = len(prepared.free_indices)
        total_pairs = free_count * (free_count - 1) // 2
        return total_pairs <= max_pair_evaluations

    def _local_moves(
        self,
        prepared,
        max_pair_evaluations,
        metrics,
    ):
        """Keep local-neighborhood accounting separate from repair accounting."""
        for index in prepared.free_indices:
            metrics.single_neighbors_evaluated += 1
            yield (index,)

        pairs_scanned = 0
        for left, right in combinations(prepared.free_indices, 2):
            if pairs_scanned >= max_pair_evaluations:
                metrics.pair_scan_truncated = True
                break
            pairs_scanned += 1
            metrics.pair_neighbors_evaluated += 1
            yield (left, right)

    def _flipped_sample(self, sample, indices):
        """Create one neighbor without ever mutating the current assignment."""
        candidate = list(sample)
        for index in indices:
            candidate[index] ^= 1
        return candidate

    def _is_better_candidate(
        self,
        sense,
        objective,
        sample,
        incumbent_objective,
        incumbent_sample,
    ):
        """Compare exact objective first, then canonical sample order."""
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

    def _consider_incumbent(
        self,
        problem,
        sample,
        restart,
        local_optimum,
        config,
        started_at,
        state,
        metrics,
    ):
        """Promote one feasible start and optionally record its observation."""
        objective = _evaluate_cbqm_objective_exact(problem, sample)
        if (
            state.incumbent_sample is not None
            and not self._is_better_candidate(
                problem['objective']['sense'],
                objective,
                sample,
                state.incumbent_objective,
                state.incumbent_sample,
            )
        ):
            return

        state.incumbent_sample = list(sample)
        state.incumbent_objective = objective
        metrics.incumbent_updates += 1
        if config['include_trace']:
            state.trace.append(
                {
                    'step': metrics.incumbent_updates - 1,
                    'time_seconds': time.perf_counter() - started_at,
                    'sample': list(sample),
                    'metadata': {
                        'restart': restart,
                        'local_optimum_reached': local_optimum,
                    },
                }
            )

    def _completed_outcome(self, state, metrics, config):
        """Return a feasible incumbent or an explicitly inconclusive result."""
        if state.incumbent_sample is None:
            return CbqmSolveOutcome(
                status='unknown',
                best_sample=None,
                termination_reason=(
                    'repair_budget_exhausted_without_feasible_candidate'
                ),
                metrics=self._public_metrics(metrics),
                metadata=self._metadata(config),
            )
        return CbqmSolveOutcome(
            status='feasible',
            best_sample=state.incumbent_sample,
            termination_reason='restart_budget_exhausted',
            bounds=self._primal_bound(state),
            metrics=self._public_metrics(metrics),
            trace=state.trace,
            metadata=self._metadata(config),
        )

    def _timeout_outcome(self, state, metrics, config):
        """Preserve only a genuinely feasible incumbent at a deadline."""
        return CbqmSolveOutcome(
            status='timeout',
            best_sample=state.incumbent_sample,
            termination_reason='timeout_reached',
            bounds=(
                {}
                if state.incumbent_sample is None
                else self._primal_bound(state)
            ),
            metrics=self._public_metrics(metrics),
            trace=state.trace,
            metadata=self._metadata(config),
        )

    def _primal_bound(self, state):
        """Expose the feasible objective as a primal bound, never a proof."""
        return {
            'primal_bound': _fraction_to_json_number(
                state.incumbent_objective,
                'CBQM local-search incumbent',
            )
        }

    def _public_metrics(self, metrics):
        """Freeze internal counters into one stable JSON metrics object."""
        return {
            'starts_attempted': metrics.starts_attempted,
            'feasible_starts': metrics.feasible_starts,
            'repair_steps': metrics.repair_steps,
            'local_steps': metrics.local_steps,
            'single_neighbors_evaluated': (
                metrics.single_neighbors_evaluated
            ),
            'pair_neighbors_evaluated': metrics.pair_neighbors_evaluated,
            'incumbent_updates': metrics.incumbent_updates,
            'pair_scan_truncated': metrics.pair_scan_truncated,
        }

    def _metadata(self, config):
        """Record reproducibility controls without copying the full problem."""
        return {
            'algorithm': 'feasibility_first_local_search',
            'seed': config['seed'],
            'max_restarts': config['max_restarts'],
            'max_repair_steps': config['max_repair_steps'],
            'max_local_steps': config['max_local_steps'],
            'max_pair_evaluations': config['max_pair_evaluations'],
        }

    def _reject_unknown_config_fields(self, config):
        """Fail fast on misspelled experiment controls."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(
                f'Unknown local-search CBQM solver config fields: {names}'
            )

    def _validate_seed(self, value):
        """Require the reproducibility seed to be an integer, not a boolean."""
        if type(value) is not int:
            raise ValueError('seed must be an integer.')

    def _validate_non_negative_integer(self, value, field_name):
        """Require genuine non-negative effort budgets."""
        if type(value) is not int or value < 0:
            raise ValueError(
                f'{field_name} must be a non-negative integer.'
            )

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

    def _validate_boolean(self, value, field_name):
        """Keep booleans distinct from Python's integer subtype."""
        if type(value) is not bool:
            raise ValueError(f'{field_name} must be a boolean.')

    def _make_deadline(self, timeout_seconds):
        """Translate a relative timeout to one monotonic deadline."""
        if timeout_seconds is None:
            return None
        return time.perf_counter() + timeout_seconds

    def _deadline_reached(self, deadline):
        """Isolate the clock so timeout behavior can be tested deterministically."""
        return deadline is not None and time.perf_counter() >= deadline


__all__ = ['LocalSearchCbqmSolver']
