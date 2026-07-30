"""Deterministic native MIS greedy construction with local exchanges.

This solver is intended as a fast, dependency-free baseline.  It always
returns a graph-native independent-set witness and deliberately reports only
``feasible``: a good incumbent is not an optimality proof.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from ...contracts.validation import _fraction_to_json_number
from .base import BaseMisSolver, MisSolveOutcome


@dataclass(frozen=True)
class _PreparedMis:
    """Validated graph data arranged for constructive search."""

    adjacency: tuple[frozenset[int], ...]
    weights: tuple[Fraction, ...]
    forced_in: frozenset[int]
    fixed_out: frozenset[int]
    fixed_conflict: bool


@dataclass
class _LocalSearchMetrics:
    """Small mutable counter bundle kept away from the main algorithm."""

    greedy_additions: int = 0
    local_passes: int = 0
    single_insertions_evaluated: int = 0
    pair_insertions_evaluated: int = 0
    improving_exchanges: int = 0
    pair_scan_truncated: bool = False


@dataclass(frozen=True)
class _Exchange:
    """One improving replacement selected by deterministic tie-breaking."""

    selected: frozenset[int]
    objective: Fraction


class GreedyMisSolver(BaseMisSolver):
    """Build a maximal positive-value independent set, then improve it."""

    SOLVER_NAME = 'greedy-mis-local-exchange'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'local-cpu'

    DEFAULT_MAX_LOCAL_PASSES = 20
    DEFAULT_MAX_PAIR_EVALUATIONS = 50_000
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {'max_local_passes', 'max_pair_evaluations'}
    )

    def _resolve_config(self, config):
        """Validate the one explicit local-search effort control."""
        resolved = super()._resolve_config(config)
        self._reject_unknown_config_fields(resolved)
        max_local_passes = resolved.get(
            'max_local_passes',
            self.DEFAULT_MAX_LOCAL_PASSES,
        )
        max_pair_evaluations = resolved.get(
            'max_pair_evaluations',
            self.DEFAULT_MAX_PAIR_EVALUATIONS,
        )
        self._validate_non_negative_integer(
            max_local_passes,
            'max_local_passes',
        )
        self._validate_non_negative_integer(
            max_pair_evaluations,
            'max_pair_evaluations',
        )
        return {
            'max_local_passes': max_local_passes,
            'max_pair_evaluations': max_pair_evaluations,
        }

    def _run(self, problem, config):
        """Prepare, construct, improve, and report: the solver in pseudocode."""
        prepared = self._prepare_problem(problem)
        if prepared.fixed_conflict:
            return self._infeasible_outcome()

        metrics = _LocalSearchMetrics()
        selected = self._construct_greedily(prepared, metrics)
        selected, reached_local_optimum = self._improve_locally(
            prepared,
            selected,
            config['max_local_passes'],
            config['max_pair_evaluations'],
            metrics,
        )
        return self._feasible_outcome(
            prepared,
            selected,
            reached_local_optimum,
            metrics,
        )

    def _prepare_problem(self, problem):
        """Build exact vertex values, adjacency and hard-assignment sets."""
        vertex_count = len(problem['vertices'])
        adjacency = [set() for _ in range(vertex_count)]
        for left, right in problem['edges']:
            adjacency[left].add(right)
            adjacency[right].add(left)

        fixed = {
            item['index']: item['value']
            for item in problem.get('fixed_values', [])
        }
        forced_in = frozenset(
            index
            for index, value in fixed.items()
            if value == 1
        )
        fixed_out = frozenset(
            index
            for index, value in fixed.items()
            if value == 0
        )
        return _PreparedMis(
            adjacency=tuple(frozenset(row) for row in adjacency),
            weights=self._vertex_weights(problem),
            forced_in=forced_in,
            fixed_out=fixed_out,
            fixed_conflict=self._forced_vertices_conflict(
                adjacency,
                forced_in,
            ),
        )

    def _vertex_weights(self, problem):
        """Use unit cardinality values or exact JSON weight arithmetic."""
        if problem['objective']['kind'] == 'maximum-cardinality':
            return tuple(Fraction(1) for _ in problem['vertices'])
        return tuple(
            Fraction(vertex['weight'])
            for vertex in problem['vertices']
        )

    def _forced_vertices_conflict(self, adjacency, forced_in):
        """Recognize the only way fixed values can make MIS infeasible."""
        return any(
            neighbor in forced_in
            for vertex in forced_in
            for neighbor in adjacency[vertex]
        )

    def _construct_greedily(self, prepared, metrics):
        """Start with forced vertices and greedily complete the witness."""
        selected = set(prepared.forced_in)
        selected, additions = self._greedy_completion(prepared, selected)
        metrics.greedy_additions += additions
        return selected

    def _greedy_completion(self, prepared, selected):
        """Add positive-value vertices using an exact deterministic score."""
        selected = set(selected)
        additions = 0
        while True:
            available = self._available_positive_vertices(
                prepared,
                selected,
            )
            if not available:
                return selected, additions
            vertex = self._best_greedy_vertex(
                prepared,
                selected,
                available,
            )
            selected.add(vertex)
            additions += 1

    def _available_positive_vertices(self, prepared, selected):
        """Return optional positive vertices compatible with the incumbent."""
        return {
            vertex
            for vertex, weight in enumerate(prepared.weights)
            if weight > 0
            if vertex not in selected
            if vertex not in prepared.fixed_out
            if not (prepared.adjacency[vertex] & selected)
        }

    def _best_greedy_vertex(self, prepared, selected, available):
        """Prefer value per live neighborhood, then stable simple criteria."""
        return max(
            available,
            key=lambda vertex: self._greedy_score(
                prepared,
                vertex,
                available,
            ),
        )

    def _greedy_score(self, prepared, vertex, available):
        """Keep the heuristic score exact and make every tie deterministic."""
        live_degree = len(prepared.adjacency[vertex] & available)
        weight = prepared.weights[vertex]
        return (
            weight / (live_degree + 1),
            weight,
            -live_degree,
            -vertex,
        )

    def _improve_locally(
        self,
        prepared,
        selected,
        max_local_passes,
        max_pair_evaluations,
        metrics,
    ):
        """Apply best improving one- or two-vertex insertions."""
        for _ in range(max_local_passes):
            exchange = self._best_improving_exchange(
                prepared,
                selected,
                max_pair_evaluations,
                metrics,
            )
            exchange, pair_scan_exhausted = exchange
            if exchange is None:
                return selected, pair_scan_exhausted
            selected, additions = self._greedy_completion(
                prepared,
                set(exchange.selected),
            )
            metrics.greedy_additions += additions
            metrics.local_passes += 1
            metrics.improving_exchanges += 1
        return selected, False

    def _best_improving_exchange(
        self,
        prepared,
        selected,
        max_pair_evaluations,
        metrics,
    ):
        """Compare all bounded-neighborhood insertions by value then witness."""
        incumbent_objective = self._objective(prepared, selected)
        best = None
        insertable = [
            vertex
            for vertex, weight in enumerate(prepared.weights)
            if weight > 0
            if vertex not in selected
            if vertex not in prepared.fixed_out
        ]

        for vertex in insertable:
            metrics.single_insertions_evaluated += 1
            best = self._consider_exchange(
                prepared,
                selected,
                (vertex,),
                incumbent_objective,
                best,
            )

        pair_scan_exhausted = True
        pairs_this_pass = 0
        for left, right in combinations(insertable, 2):
            if pairs_this_pass >= max_pair_evaluations:
                pair_scan_exhausted = False
                metrics.pair_scan_truncated = True
                break
            pairs_this_pass += 1
            metrics.pair_insertions_evaluated += 1
            if right in prepared.adjacency[left]:
                continue
            best = self._consider_exchange(
                prepared,
                selected,
                (left, right),
                incumbent_objective,
                best,
            )
        return best, pair_scan_exhausted

    def _consider_exchange(
        self,
        prepared,
        selected,
        inserted,
        incumbent_objective,
        best,
    ):
        """Remove all insertion conflicts and retain a strict improvement."""
        conflicts = set().union(
            *(prepared.adjacency[vertex] & selected for vertex in inserted)
        )
        if conflicts & prepared.forced_in:
            return best

        candidate = (set(selected) - conflicts) | set(inserted)
        objective = self._objective(prepared, candidate)
        if objective <= incumbent_objective:
            return best
        exchange = _Exchange(
            selected=frozenset(candidate),
            objective=objective,
        )
        if self._exchange_is_better(exchange, best):
            return exchange
        return best

    def _exchange_is_better(self, candidate, incumbent):
        """Choose best objective, then canonical vertex-index-set order."""
        if incumbent is None:
            return True
        if candidate.objective != incumbent.objective:
            return candidate.objective > incumbent.objective
        return sorted(candidate.selected) < sorted(incumbent.selected)

    def _objective(self, prepared, selected):
        """Evaluate internal comparisons without binary64 rounding."""
        return sum(
            (prepared.weights[vertex] for vertex in selected),
            start=Fraction(0),
        )

    def _feasible_outcome(
        self,
        prepared,
        selected,
        reached_local_optimum,
        metrics,
    ):
        """Report a verified incumbent while making no optimality claim."""
        objective = _fraction_to_json_number(
            self._objective(prepared, selected),
            'greedy MIS incumbent',
        )
        termination_reason = (
            'local_optimum_reached'
            if reached_local_optimum
            else 'local_search_limit_reached'
        )
        return MisSolveOutcome(
            status='feasible',
            selected_vertices=sorted(selected),
            termination_reason=termination_reason,
            bounds={'incumbent_lower_bound': objective},
            metrics={
                'greedy_additions': metrics.greedy_additions,
                'local_passes': metrics.local_passes,
                'single_insertions_evaluated': (
                    metrics.single_insertions_evaluated
                ),
                'pair_insertions_evaluated': (
                    metrics.pair_insertions_evaluated
                ),
                'improving_exchanges': metrics.improving_exchanges,
                'pair_scan_truncated': metrics.pair_scan_truncated,
            },
            metadata={
                'algorithm': 'greedy_with_local_exchange',
                'local_optimum_reached': reached_local_optimum,
            },
        )

    def _infeasible_outcome(self):
        """Return the direct fixed-edge infeasibility certificate."""
        return MisSolveOutcome(
            status='infeasible',
            selected_vertices=None,
            termination_reason='conflicting_fixed_in_vertices',
            proof={
                'claim': 'infeasibility',
                'kind': 'fixed-edge-conflict',
                'producer': f'{self.SOLVER_NAME}@{self.SOLVER_VERSION}',
                'independently_verified': False,
            },
            metrics={
                'greedy_additions': 0,
                'local_passes': 0,
                'single_insertions_evaluated': 0,
                'pair_insertions_evaluated': 0,
                'improving_exchanges': 0,
                'pair_scan_truncated': False,
            },
            metadata={'algorithm': 'greedy_with_local_exchange'},
        )

    def _reject_unknown_config_fields(self, config):
        """Reject misspelled experiment controls instead of ignoring them."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(
                f'Unknown greedy MIS solver config fields: {names}'
            )

    def _validate_non_negative_integer(self, value, field_name):
        """Reject booleans and negative effort limits."""
        if type(value) is not int or value < 0:
            raise ValueError(
                f'{field_name} must be a non-negative integer.'
            )


__all__ = ['GreedyMisSolver']
