"""Exact native MIS branch-and-reduce with deterministic witnesses.

The implementation is a correctness baseline for small and structurally easy
graphs.  It preserves graph neighborhoods directly, applies only reductions
whose safety is immediate, and uses an optimistic remaining-weight bound for
branch pruning.  It never compiles the graph to QUBO.
"""

import math
import time
from dataclasses import dataclass
from fractions import Fraction

from ...contracts.validation import _fraction_to_json_number
from .base import BaseMisSolver, MisSolveOutcome


class _SearchTimedOut(Exception):
    """Internal non-error signal used to unwind the recursive search."""


@dataclass
class _SearchState:
    """Mutable counters and incumbent shared by recursive branches."""

    adjacency: tuple[frozenset[int], ...]
    weights: tuple[Fraction, ...]
    deadline: float | None
    best_vertices: list[int]
    best_objective: Fraction
    root_upper_bound: Fraction
    nodes_expanded: int = 0
    branches_pruned: int = 0
    reductions_applied: int = 0


class ExactMisSolver(BaseMisSolver):
    """Prove maximum cardinality/weight through branch-and-reduce search."""

    SOLVER_NAME = 'exact-mis-branch-and-reduce'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'local-cpu'

    DEFAULT_MAX_VERTICES = 48
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {'max_vertices', 'timeout_seconds'}
    )

    def _resolve_config(self, config):
        """Validate explicit exponential-search safety controls."""
        resolved = super()._resolve_config(config)
        self._reject_unknown_config_fields(resolved)
        max_vertices = resolved.get(
            'max_vertices',
            self.DEFAULT_MAX_VERTICES,
        )
        timeout_seconds = resolved.get('timeout_seconds')
        self._validate_max_vertices(max_vertices)
        self._validate_timeout(timeout_seconds)
        return {
            'max_vertices': max_vertices,
            'timeout_seconds': timeout_seconds,
        }

    def _run(self, problem, config):
        """Prepare, reduce, branch, and report: the exact search in pseudocode."""
        self._guard_search_space(problem, config['max_vertices'])
        prepared = self._prepare_problem(problem)

        if prepared['fixed_conflict']:
            return self._infeasible_outcome()

        state = self._create_search_state(
            prepared,
            config['timeout_seconds'],
        )
        try:
            self._search(
                state,
                prepared['remaining'],
                prepared['forced_in'],
                prepared['forced_objective'],
            )
        except _SearchTimedOut:
            return self._timeout_outcome(state)
        return self._optimal_outcome(state)

    def _prepare_problem(self, problem):
        """Build adjacency, weights and the feasible domain after fixed values."""
        vertex_count = len(problem['vertices'])
        adjacency = [set() for _ in range(vertex_count)]
        for left, right in problem['edges']:
            adjacency[left].add(right)
            adjacency[right].add(left)

        weights = self._vertex_weights(problem)
        fixed = {
            item['index']: item['value']
            for item in problem.get('fixed_values', [])
        }
        forced_in = {
            index
            for index, value in fixed.items()
            if value == 1
        }
        fixed_out = {
            index
            for index, value in fixed.items()
            if value == 0
        }
        fixed_conflict = any(
            right in forced_in
            for left in forced_in
            for right in adjacency[left]
        )
        blocked_by_forced = set().union(
            *(adjacency[index] for index in forced_in)
        ) if forced_in else set()
        remaining = (
            set(range(vertex_count))
            - forced_in
            - fixed_out
            - blocked_by_forced
        )
        forced_objective = sum(
            (weights[index] for index in forced_in),
            start=Fraction(0),
        )
        return {
            'adjacency': tuple(frozenset(row) for row in adjacency),
            'weights': weights,
            'forced_in': set(forced_in),
            'forced_objective': forced_objective,
            'remaining': remaining,
            'fixed_conflict': fixed_conflict,
        }

    def _vertex_weights(self, problem):
        """Represent cardinality as unit weights and MWIS weights exactly."""
        if problem['objective']['kind'] == 'maximum-cardinality':
            return tuple(
                Fraction(1)
                for _ in problem['vertices']
            )
        return tuple(
            Fraction(vertex['weight'])
            for vertex in problem['vertices']
        )

    def _create_search_state(self, prepared, timeout_seconds):
        """Initialize a feasible incumbent and one globally valid upper bound."""
        incumbent = sorted(prepared['forced_in'])
        root_upper_bound = prepared['forced_objective'] + sum(
            (
                max(Fraction(0), prepared['weights'][index])
                for index in prepared['remaining']
            ),
            start=Fraction(0),
        )
        return _SearchState(
            adjacency=prepared['adjacency'],
            weights=prepared['weights'],
            deadline=self._make_deadline(timeout_seconds),
            best_vertices=incumbent,
            best_objective=prepared['forced_objective'],
            root_upper_bound=root_upper_bound,
        )

    def _search(self, state, remaining, selected, objective):
        """Apply safe reductions, prune by an upper bound, then branch."""
        if remaining and self._deadline_reached(state.deadline):
            raise _SearchTimedOut
        state.nodes_expanded += 1

        remaining, selected, objective = self._reduce(
            state,
            remaining,
            selected,
            objective,
        )
        self._consider_incumbent(state, selected, objective)

        upper_bound = self._upper_bound(state, remaining, objective)
        if upper_bound < state.best_objective:
            state.branches_pruned += 1
            return
        if not remaining:
            return

        branch_vertex = self._select_branch_vertex(state, remaining)
        self._search_including(
            state,
            branch_vertex,
            remaining,
            selected,
            objective,
        )
        self._search_excluding(
            state,
            branch_vertex,
            remaining,
            selected,
            objective,
        )

    def _reduce(self, state, remaining, selected, objective):
        """Remove negative optional vertices and include positive isolates.

        A zero-weight vertex is deliberately retained. Although excluding it
        cannot hurt the objective, selecting it can change the canonical
        lexicographic tie-break among equal-weight independent sets.
        """
        remaining = set(remaining)
        selected = set(selected)

        negative = {
            index
            for index in remaining
            if state.weights[index] < 0
        }
        if negative:
            remaining -= negative
            state.reductions_applied += len(negative)

        while True:
            isolated = [
                index
                for index in sorted(remaining)
                if state.weights[index] > 0
                if not (state.adjacency[index] & remaining)
            ]
            if not isolated:
                break
            for index in isolated:
                selected.add(index)
                objective += state.weights[index]
            remaining -= set(isolated)
            state.reductions_applied += len(isolated)
        return remaining, selected, objective

    def _upper_bound(self, state, remaining, objective):
        """Ignore all remaining conflicts to obtain a safe optimistic bound."""
        return objective + sum(
            (
                max(Fraction(0), state.weights[index])
                for index in remaining
            ),
            start=Fraction(0),
        )

    def _select_branch_vertex(self, state, remaining):
        """Prefer high-degree/high-weight vertices, then the lowest index."""
        return max(
            remaining,
            key=lambda index: (
                len(state.adjacency[index] & remaining),
                state.weights[index],
                -index,
            ),
        )

    def _search_including(
        self,
        state,
        vertex,
        remaining,
        selected,
        objective,
    ):
        """Explore the branch that selects ``vertex`` and removes neighbors."""
        self._search(
            state,
            (
                remaining
                - {vertex}
                - set(state.adjacency[vertex])
            ),
            selected | {vertex},
            objective + state.weights[vertex],
        )

    def _search_excluding(
        self,
        state,
        vertex,
        remaining,
        selected,
        objective,
    ):
        """Explore the complementary branch that excludes ``vertex``."""
        self._search(
            state,
            remaining - {vertex},
            selected,
            objective,
        )

    def _consider_incumbent(self, state, selected, objective):
        """Update by exact objective, then canonical index-set order."""
        candidate = sorted(selected)
        if (
            objective > state.best_objective
            or (
                objective == state.best_objective
                and candidate < state.best_vertices
            )
        ):
            state.best_vertices = candidate
            state.best_objective = objective

    def _optimal_outcome(self, state):
        """Return equal maximization bounds after the search tree is exhausted."""
        objective = self._public_objective(
            state.best_objective,
            'MIS optimum',
        )
        return MisSolveOutcome(
            status='optimal',
            selected_vertices=state.best_vertices,
            termination_reason='search_tree_exhausted',
            bounds={
                'incumbent_lower_bound': objective,
                'optimum_upper_bound': objective,
            },
            proof=self._proof('optimality', state, exhausted=True),
            metrics=self._metrics(state, exhausted=True),
            metadata={'algorithm': 'branch_and_reduce'},
        )

    def _infeasible_outcome(self):
        """A direct edge between two forced-in vertices proves infeasibility."""
        return MisSolveOutcome(
            status='infeasible',
            selected_vertices=None,
            termination_reason='conflicting_fixed_in_vertices',
            proof={
                'claim': 'infeasibility',
                'kind': 'fixed-edge-conflict',
                'producer': f'{self.SOLVER_NAME}@{self.SOLVER_VERSION}',
                'independently_verified': False,
                'details': {'search_space_exhausted': True},
            },
            metrics={
                'nodes_expanded': 0,
                'branches_pruned': 0,
                'reductions_applied': 0,
                'search_space_exhausted': True,
            },
            metadata={'algorithm': 'branch_and_reduce'},
        )

    def _timeout_outcome(self, state):
        """Return a feasible incumbent and conservative root upper bound."""
        return MisSolveOutcome(
            status='timeout',
            selected_vertices=state.best_vertices,
            termination_reason='timeout_reached',
            bounds={
                'incumbent_lower_bound': self._public_objective(
                    state.best_objective,
                    'MIS incumbent',
                ),
                'optimum_upper_bound': self._public_objective(
                    state.root_upper_bound,
                    'MIS root upper bound',
                ),
            },
            metrics=self._metrics(state, exhausted=False),
            metadata={'algorithm': 'branch_and_reduce'},
        )

    def _proof(self, claim, state, *, exhausted):
        """Describe solver evidence without marking independent verification."""
        return {
            'claim': claim,
            'kind': 'branch-and-reduce',
            'producer': f'{self.SOLVER_NAME}@{self.SOLVER_VERSION}',
            'independently_verified': False,
            'details': {
                'nodes_expanded': state.nodes_expanded,
                'branches_pruned': state.branches_pruned,
                'reductions_applied': state.reductions_applied,
                'search_space_exhausted': exhausted,
            },
        }

    def _metrics(self, state, *, exhausted):
        """Expose identical counters on exact and interrupted paths."""
        return {
            'nodes_expanded': state.nodes_expanded,
            'branches_pruned': state.branches_pruned,
            'reductions_applied': state.reductions_applied,
            'search_space_exhausted': exhausted,
        }

    def _public_objective(self, objective, label):
        """Convert an exact comparison value only at the result boundary."""
        return _fraction_to_json_number(objective, label)

    def _reject_unknown_config_fields(self, config):
        """Reject misspelled experiment controls."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'Unknown exact MIS solver config fields: {names}')

    def _validate_max_vertices(self, value):
        """Require a genuine non-negative integer safety limit."""
        if type(value) is not int or value < 0:
            raise ValueError('max_vertices must be a non-negative integer.')

    def _validate_timeout(self, value):
        """Accept no deadline or a finite positive duration."""
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

    def _guard_search_space(self, problem, max_vertices):
        """Fail before accidentally starting a large worst-case search."""
        vertex_count = len(problem['vertices'])
        if vertex_count > max_vertices:
            raise ValueError(
                f'Exact MIS search received {vertex_count} vertices, '
                f'exceeding max_vertices={max_vertices}. Increase the limit '
                'explicitly only if branch-and-reduce search is intended.'
            )

    def _make_deadline(self, timeout_seconds):
        """Translate a relative timeout into one monotonic deadline."""
        if timeout_seconds is None:
            return None
        return time.perf_counter() + timeout_seconds

    def _deadline_reached(self, deadline):
        """Isolate wall-clock interaction for deterministic timeout tests."""
        return deadline is not None and time.perf_counter() >= deadline


__all__ = ['ExactMisSolver']
