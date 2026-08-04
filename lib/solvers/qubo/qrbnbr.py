"""Independent Q-RBnBR reimplementation for weighted MaxCut QUBOs.

This module independently reimplements the algorithmic path described in the
``Quantum Relaxation Informed Branch-and-Bound Algorithm -- An Application
to Max-Cut``.  It intentionally does not import the research repository:

* a p=1 statevector QAOA relaxation produces pair correlations;
* quantum relaxation rounding (QRR) sign-rounds correlation eigenvectors;
* an edge-parity tree branches on ``z_u z_v = +1`` and ``-1``;
* a Laplacian eigenvalue bound prunes subproblems;
* sufficiently small leaves are closed by exhaustive enumeration.

The public entry point remains deliberately short.  Representation checks,
statevector mechanics, parity bookkeeping, rounding, branching, bounds and
result construction live behind protected methods so ``_run`` reads like the
paper's pseudocode.

Scope
-----
The paper studies MaxCut rather than arbitrary QUBO.  This solver therefore
accepts only canonical QUBOs of the exact form

``E(x) = offset - sum_((u,v) in E) w_uv [x_u != x_v]``

with finite ``w_uv > 0``.  General QUBOs can still be converted to a signed,
anchored MaxCut graph by :mod:`lib.adapters.qrbnbr_maxcut`, but that extension
is outside the paper-faithful reproduction claimed by this class.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ...contracts.validation import _evaluate_qubo
from ._maxcut import (
    _evaluate_weighted_cut,
    _read_nonnegative_maxcut_qubo,
)
from .base import BaseQuboSolver, QuboSolveOutcome
from .qaoa import QaoaQuboSolver


@dataclass(frozen=True)
class _ReducedMaxCut:
    """One parity-constrained MaxCut after exact variable substitution."""

    representatives: tuple[int, ...]
    weights: np.ndarray
    compensation: float

    @property
    def variable_count(self):
        """Return the number of independent parity components."""
        return len(self.representatives)


@dataclass(frozen=True)
class _Relaxation:
    """One provider's candidate, branching matrix and audit summary."""

    reduced_sample: tuple[int, ...]
    correlation: np.ndarray
    branching_matrix: np.ndarray
    work_units: int
    summary: dict


@dataclass
class _SearchStatistics:
    """Mutable counters kept separate from the tree-search decisions."""

    nodes_explored: int = 0
    nodes_pruned: int = 0
    branches_created: int = 0
    relaxation_subproblems: int = 0
    exact_subproblems: int = 0
    relaxation_work_units: int = 0
    max_frontier_size: int = 1


class _ParityPartition:
    """Union-find whose offsets encode equality or inequality of spins.

    ``offset[i] == 0`` means vertex ``i`` has the same binary value as its
    current representative; ``1`` means it has the opposite value.  Keeping
    this concern in a small object makes every search node just one partition.
    """

    def __init__(self, variable_count):
        self.parent = list(range(variable_count))
        self.rank = [0] * variable_count
        self.offset = [0] * variable_count

    def clone(self):
        """Return an independent search-node copy."""
        cloned = _ParityPartition(len(self.parent))
        cloned.parent = self.parent.copy()
        cloned.rank = self.rank.copy()
        cloned.offset = self.offset.copy()
        return cloned

    def find(self, vertex):
        """Find a representative and compress parity along the path."""
        parent = self.parent[vertex]
        if parent != vertex:
            representative = self.find(parent)
            self.offset[vertex] ^= self.offset[parent]
            self.parent[vertex] = representative
        return self.parent[vertex]

    def join(self, left, right, parity):
        """Impose ``x_left XOR x_right == parity``.

        Search only joins different representatives, but supporting an
        already-connected pair makes the invariant explicit and keeps this
        data structure independently testable.
        """
        left_root = self.find(left)
        right_root = self.find(right)
        left_offset = self.offset[left]
        right_offset = self.offset[right]

        if left_root == right_root:
            return (left_offset ^ right_offset) == parity

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        self.offset[right_root] = left_offset ^ right_offset ^ parity
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        self.compress()
        return True

    def compress(self):
        """Normalize every vertex to its final representative."""
        for vertex in range(len(self.parent)):
            self.find(vertex)

    def representatives(self):
        """Return representatives in deterministic source-variable order."""
        self.compress()
        return tuple(sorted(set(self.parent)))

    def expand(self, reduced_sample):
        """Lift component bits back to the source QUBO variable order."""
        representatives = self.representatives()
        component_index = {
            representative: index
            for index, representative in enumerate(representatives)
        }
        return [
            int(
                reduced_sample[component_index[self.parent[vertex]]]
                ^ self.offset[vertex]
            )
            for vertex in range(len(self.parent))
        ]


class QrbnbrQuboSolver(QaoaQuboSolver):
    """Reproduce QRR-informed parity branch-and-bound through ``qubo.v1``.

    Supported configuration fields:

    ``branching_rule``
        ``"r1"`` chooses the largest absolute matrix entry, ``"r2"`` chooses
        the pair whose matrix rows are closest to binary correlations, and
        ``"r3"`` chooses the smallest absolute entry.  Defaults to ``"r1"``.
    ``branching_matrix``
        ``"correlation"`` uses the p=1 QAOA correlation matrix directly.
        ``"selective"`` composes a matrix from the eigenvectors whose rounded
        cuts are best.  Defaults to ``"correlation"``.
    ``selective_rank``
        Number of eigenvectors retained by selective composition. Defaults to
        ``3`` and is clipped to the current subproblem size.
    ``traversal``
        ``"bfs"`` or ``"dfs"`` edge-parity-tree traversal. Defaults to BFS.
    ``brute_force_threshold``
        Subproblems at or below this component count are solved exhaustively.
        Defaults to ``8``.
    ``qaoa_grid_size``
        Number of deterministic gamma and beta grid points per dimension.
        Defaults to ``5``.
    ``qaoa_refinement_steps``
        Coordinate-search iterations after the grid start. Defaults to ``4``.
    ``max_variables``
        Statevector and tree-search safety rail. Defaults to ``18``.
    ``max_nodes``
        Optional positive search-node budget.
    ``timeout_seconds``
        Optional finite positive wall-clock budget.

    With no node/time limit, all surviving leaves are closed and the
    admissible parity-tree search returns ``optimal``.  QRR guides the search;
    it is not used as the optimality proof.
    """

    SOLVER_NAME = 'qrbnbr-maxcut-reproduction'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'numpy-statevector'

    DEFAULT_MAX_VARIABLES = 18
    DEFAULT_BRUTE_FORCE_THRESHOLD = 8
    DEFAULT_QAOA_GRID_SIZE = 5
    DEFAULT_QAOA_REFINEMENT_STEPS = 4

    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'branching_rule',
            'branching_matrix',
            'selective_rank',
            'traversal',
            'brute_force_threshold',
            'qaoa_grid_size',
            'qaoa_refinement_steps',
            'max_variables',
            'max_nodes',
            'timeout_seconds',
        }
    )
    _BOUND_TOLERANCE = 1e-10

    def _resolve_config(self, config):
        """Validate every experiment knob before allocating a statevector."""
        resolved = BaseQuboSolver._resolve_config(self, config)
        self._reject_unknown_qrbnbr_fields(resolved)

        output = {
            'branching_rule': resolved.get('branching_rule', 'r1'),
            'branching_matrix': resolved.get(
                'branching_matrix',
                'correlation',
            ),
            'selective_rank': resolved.get('selective_rank', 3),
            'traversal': resolved.get('traversal', 'bfs'),
            'brute_force_threshold': resolved.get(
                'brute_force_threshold',
                self.DEFAULT_BRUTE_FORCE_THRESHOLD,
            ),
            'qaoa_grid_size': resolved.get(
                'qaoa_grid_size',
                self.DEFAULT_QAOA_GRID_SIZE,
            ),
            'qaoa_refinement_steps': resolved.get(
                'qaoa_refinement_steps',
                self.DEFAULT_QAOA_REFINEMENT_STEPS,
            ),
            'max_variables': resolved.get(
                'max_variables',
                self.DEFAULT_MAX_VARIABLES,
            ),
            'max_nodes': resolved.get('max_nodes'),
            'timeout_seconds': resolved.get('timeout_seconds'),
        }
        self._validate_qrbnbr_config(output)
        return output

    def _run(self, problem, config):
        """Read like Algorithm 1: initialise, bound, relax, branch, finish."""
        model = self._read_maxcut_model(problem)
        self._guard_problem_size(model.variable_count, config)

        search = self._initialise_search(problem, model, config)
        while self._search_should_continue(search, config):
            node = self._take_next_node(search, config['traversal'])
            reduced = self._reduce_model(model, node)

            if self._can_prune(reduced, search['best_cut']):
                self._record_pruned_node(search)
                continue

            if self._is_exact_leaf(reduced, config):
                candidate = self._solve_exact_leaf(reduced, search)
                self._consider_candidate(
                    candidate,
                    node,
                    model,
                    search,
                )
                continue

            relaxation = self._solve_relaxation(
                reduced,
                config,
            )
            self._record_relaxation(relaxation, search)
            self._consider_candidate(
                relaxation.reduced_sample,
                node,
                model,
                search,
            )

            pair = self._choose_branching_pair(
                relaxation.branching_matrix,
                reduced,
                config['branching_rule'],
            )
            self._schedule_parity_branches(pair, node, search)

        return self._finish_search(search, config)

    def _reject_unknown_qrbnbr_fields(self, config):
        """Reject typos so experiment manifests cannot silently drift."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'Unknown Q-RBnBR solver config fields: {names}')

    def _validate_qrbnbr_config(self, config):
        """Validate categorical, integer and time-budget settings."""
        if config['branching_rule'] not in {'r1', 'r2', 'r3'}:
            raise ValueError("branching_rule must be 'r1', 'r2', or 'r3'.")
        if config['branching_matrix'] not in {'correlation', 'selective'}:
            raise ValueError(
                "branching_matrix must be 'correlation' or 'selective'."
            )
        if config['traversal'] not in {'bfs', 'dfs'}:
            raise ValueError("traversal must be 'bfs' or 'dfs'.")

        self._validate_positive_integer(
            config['selective_rank'],
            'selective_rank',
        )
        self._validate_non_negative_integer(
            config['brute_force_threshold'],
            'brute_force_threshold',
        )
        self._validate_positive_integer(
            config['qaoa_grid_size'],
            'qaoa_grid_size',
        )
        self._validate_non_negative_integer(
            config['qaoa_refinement_steps'],
            'qaoa_refinement_steps',
        )
        self._validate_non_negative_integer(
            config['max_variables'],
            'max_variables',
        )
        self._validate_optional_positive_integer(
            config['max_nodes'],
            'max_nodes',
        )
        self._validate_timeout(config['timeout_seconds'])

    def _validate_timeout(self, value):
        """Accept no timeout or one finite strictly positive duration."""
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

    def _guard_problem_size(self, variable_count, config):
        """Protect both the exponential statevector and parity tree."""
        if variable_count > config['max_variables']:
            raise ValueError(
                f'Q-RBnBR received {variable_count} variables, exceeding '
                f"max_variables={config['max_variables']}."
            )

    def _read_maxcut_model(self, problem):
        """Recognize exactly ``offset - weighted_cut`` in sparse QUBO form."""
        return _read_nonnegative_maxcut_qubo(problem)

    def _initialise_search(self, problem, model, config):
        """Create the root edge-parity node and all mutable search state."""
        root = _ParityPartition(model.variable_count)
        return {
            'problem': problem,
            'frontier': deque([root]),
            'best_sample': None,
            'best_cut': -math.inf,
            'trace': [],
            'statistics': _SearchStatistics(),
            'root_relaxation': None,
            'started_at': time.perf_counter(),
            'stopped_by': None,
            'config': config,
        }

    def _search_should_continue(self, search, config):
        """Stop only when the tree closes or an explicit budget is reached."""
        if not search['frontier']:
            return False
        if self._deadline_reached(search, config['timeout_seconds']):
            search['stopped_by'] = 'timeout_reached'
            return False
        max_nodes = config['max_nodes']
        if (
            max_nodes is not None
            and search['statistics'].nodes_explored >= max_nodes
        ):
            search['stopped_by'] = 'node_limit_reached'
            return False
        return True

    def _deadline_reached(self, search, timeout_seconds):
        """Keep monotonic-clock policy out of the tree-search pseudocode."""
        return (
            timeout_seconds is not None
            and time.perf_counter() - search['started_at'] >= timeout_seconds
        )

    def _take_next_node(self, search, traversal):
        """Pop from the appropriate frontier end for BFS or DFS."""
        statistics = search['statistics']
        statistics.nodes_explored += 1
        if traversal == 'bfs':
            return search['frontier'].popleft()
        return search['frontier'].pop()

    def _reduce_model(self, model, partition):
        """Substitute parity relations and aggregate a signed reduced graph.

        Opposite offsets can turn a positive source edge into a negative
        reduced coupling.  ``compensation`` preserves the exact source cut:

        ``source_cut = compensation + reduced_signed_cut``.
        """
        representatives = partition.representatives()
        reduced_index = {
            representative: index
            for index, representative in enumerate(representatives)
        }
        reduced_weights = np.zeros(
            (len(representatives), len(representatives)),
            dtype=float,
        )
        compensation = 0.0

        for left in range(model.variable_count):
            left_root = partition.parent[left]
            left_offset = partition.offset[left]
            for right in range(left + 1, model.variable_count):
                weight = model.weights[left, right]
                if weight == 0:
                    continue

                right_root = partition.parent[right]
                right_offset = partition.offset[right]
                parity_sign = 1.0 if left_offset == right_offset else -1.0
                if left_root == right_root:
                    if parity_sign < 0:
                        compensation += weight
                    continue

                reduced_left = reduced_index[left_root]
                reduced_right = reduced_index[right_root]
                reduced_weights[reduced_left, reduced_right] += (
                    weight * parity_sign
                )
                reduced_weights[reduced_right, reduced_left] = (
                    reduced_weights[reduced_left, reduced_right]
                )
                if parity_sign < 0:
                    compensation += weight

        return _ReducedMaxCut(
            representatives=representatives,
            weights=reduced_weights,
            compensation=float(compensation),
        )

    def _can_prune(self, reduced, incumbent_cut):
        """Apply the paper's Laplacian maximum-eigenvalue upper bound."""
        if incumbent_cut == -math.inf:
            return False
        upper_bound = self._maxcut_upper_bound(reduced)
        return upper_bound < incumbent_cut - self._BOUND_TOLERANCE

    def _maxcut_upper_bound(self, reduced):
        """Return ``c + n/4 lambda_max(L)`` with numerical padding.

        For binary spins ``z``, signed cut is ``z.T @ L @ z / 4`` and
        ``||z||² = n``.  Rayleigh's inequality therefore gives the bound.  A
        scale-aware margin prevents a tiny floating-point underestimate from
        turning an equality into an incorrect prune.
        """
        variable_count = reduced.variable_count
        if variable_count == 0:
            return reduced.compensation
        laplacian = self._laplacian(reduced.weights)
        largest_eigenvalue = float(np.linalg.eigvalsh(laplacian)[-1])
        matrix_scale = max(1.0, float(np.linalg.norm(laplacian, ord=np.inf)))
        numerical_margin = (
            np.finfo(float).eps
            * matrix_scale
            * max(16, 16 * variable_count)
        )
        return (
            reduced.compensation
            + variable_count
            * (largest_eigenvalue + numerical_margin)
            / 4.0
        )

    def _laplacian(self, weights):
        """Build the signed weighted Laplacian ``D - W``."""
        return np.diag(np.sum(weights, axis=1)) - weights

    def _record_pruned_node(self, search):
        """Increment one isolated pruning counter."""
        search['statistics'].nodes_pruned += 1

    def _is_exact_leaf(self, reduced, config):
        """Use exhaustive closure once the remaining component count is small."""
        return (
            reduced.variable_count <= 1
            or reduced.variable_count <= config['brute_force_threshold']
        )

    def _solve_exact_leaf(self, reduced, search):
        """Enumerate a reduced leaf with deterministic tie-breaking."""
        search['statistics'].exact_subproblems += 1
        best_sample = None
        best_cut = -math.inf

        for basis_index in range(1 << reduced.variable_count):
            sample = tuple(
                self._decode_basis_index(
                    basis_index,
                    reduced.variable_count,
                )
            )
            cut = self._evaluate_reduced_cut(reduced, sample)
            if self._is_better_cut(cut, sample, best_cut, best_sample):
                best_sample = sample
                best_cut = cut
        return best_sample

    def _solve_relaxation(self, reduced, config):
        """Run p=1 QAOA, compute correlations, and apply QRR rounding."""
        energies = self._build_reduced_cost_spectrum(reduced)
        parameters, expectation, evaluations = self._optimise_qaoa_angles(
            energies,
            reduced.variable_count,
            config,
        )
        state = self._prepare_qaoa_state(
            energies,
            reduced.variable_count,
            parameters,
        )
        probabilities = self._measurement_probabilities(state)
        correlation = self._correlation_matrix(
            probabilities,
            reduced.variable_count,
        )
        rounded_sample, eigensystem = self._quantum_relaxation_round(
            correlation,
            reduced,
        )
        branching_matrix = self._make_branching_matrix(
            correlation,
            eigensystem,
            reduced,
            config,
        )
        return _Relaxation(
            reduced_sample=tuple(rounded_sample),
            correlation=correlation,
            branching_matrix=branching_matrix,
            work_units=evaluations,
            summary={
                'gamma': float(parameters[0]),
                'beta': float(parameters[1]),
                'expectation': float(expectation),
                'correlation_frobenius_norm': float(
                    np.linalg.norm(correlation)
                ),
            },
        )

    def _build_reduced_cost_spectrum(self, reduced):
        """Represent QAOA's minimization Hamiltonian as negative cut value."""
        return np.asarray(
            [
                -self._evaluate_reduced_cut(
                    reduced,
                    self._decode_basis_index(
                        basis_index,
                        reduced.variable_count,
                    ),
                )
                for basis_index in range(1 << reduced.variable_count)
            ],
            dtype=float,
        )

    def _optimise_qaoa_angles(self, energies, variable_count, config):
        """Use a deterministic grid followed by inherited coordinate search."""
        grid_size = config['qaoa_grid_size']
        gamma_values = np.linspace(
            0.0,
            2.0 * math.pi,
            grid_size,
            endpoint=False,
        )
        beta_values = np.linspace(
            0.0,
            0.5 * math.pi,
            grid_size,
            endpoint=False,
        )
        best_parameters = np.asarray([0.0, 0.0])
        best_expectation = math.inf
        evaluations = 0

        for gamma in gamma_values:
            for beta in beta_values:
                parameters = np.asarray([gamma, beta], dtype=float)
                expectation = self._expectation(
                    energies,
                    variable_count,
                    parameters,
                )
                evaluations += 1
                if expectation < best_expectation:
                    best_parameters = parameters
                    best_expectation = expectation

        refined, expectation, refinement_evaluations = (
            self._coordinate_search(
                energies,
                variable_count,
                best_parameters,
                config['qaoa_refinement_steps'],
            )
        )
        return (
            refined,
            expectation,
            evaluations + refinement_evaluations,
        )

    def _correlation_matrix(self, probabilities, variable_count):
        """Compute every ``<Z_i Z_j>`` exactly from the statevector."""
        correlation = np.zeros((variable_count, variable_count), dtype=float)
        basis_indices = np.arange(len(probabilities), dtype=np.uint64)
        spin_vectors = []
        for variable_index in range(variable_count):
            shift = variable_count - variable_index - 1
            bits = ((basis_indices >> shift) & 1).astype(float)
            spin_vectors.append(1.0 - 2.0 * bits)

        for left in range(variable_count):
            for right in range(left + 1, variable_count):
                value = float(
                    np.dot(
                        probabilities,
                        spin_vectors[left] * spin_vectors[right],
                    )
                )
                correlation[left, right] = value
                correlation[right, left] = value
        return correlation

    def _quantum_relaxation_round(self, correlation, reduced):
        """Sign-round every correlation eigenvector and keep the best cut."""
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        candidates = []
        for eigen_index in range(reduced.variable_count):
            vector = eigenvectors[:, eigen_index]
            sample = tuple(int(value >= 0.0) for value in vector)
            cut = self._evaluate_reduced_cut(reduced, sample)
            candidates.append(
                {
                    'cut': cut,
                    'sample': sample,
                    'eigenvalue': float(eigenvalues[eigen_index]),
                    'eigenvector': vector.copy(),
                    'eigen_index': eigen_index,
                }
            )

        candidates.sort(key=lambda item: (-item['cut'], item['sample']))
        return candidates[0]['sample'], {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'ranked_candidates': candidates,
        }

    def _make_branching_matrix(
        self,
        correlation,
        eigensystem,
        reduced,
        config,
    ):
        """Select raw correlation or thesis selective composition."""
        if config['branching_matrix'] == 'correlation':
            return correlation

        rank = min(config['selective_rank'], reduced.variable_count)
        matrix = np.zeros_like(correlation)
        for candidate in eigensystem['ranked_candidates'][:rank]:
            vector = candidate['eigenvector']
            matrix += candidate['eigenvalue'] * np.outer(vector, vector)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def _record_relaxation(self, relaxation, search):
        """Accumulate provider-neutral relaxation statistics."""
        statistics = search['statistics']
        statistics.relaxation_subproblems += 1
        statistics.relaxation_work_units += relaxation.work_units
        if search['root_relaxation'] is None:
            search['root_relaxation'] = dict(relaxation.summary)

    def _consider_candidate(
        self,
        reduced_sample,
        partition,
        model,
        search,
    ):
        """Lift a candidate, evaluate it on the source graph, and update."""
        source_sample = tuple(partition.expand(reduced_sample))
        source_cut = self._evaluate_source_cut(model, source_sample)
        if not self._is_better_cut(
            source_cut,
            source_sample,
            search['best_cut'],
            search['best_sample'],
        ):
            return

        search['best_cut'] = source_cut
        search['best_sample'] = source_sample
        search['trace'].append(
            {
                'step': len(search['trace']),
                'time_seconds': (
                    time.perf_counter() - search['started_at']
                ),
                'energy': _evaluate_qubo(
                    search['problem'],
                    list(source_sample),
                ),
                'sample': list(source_sample),
                'metadata': {
                    'cut_value': source_cut,
                    'nodes_explored': search['statistics'].nodes_explored,
                },
            }
        )

    def _is_better_cut(self, cut, sample, best_cut, best_sample):
        """Maximize cut, resolving equal cuts by canonical sample order."""
        if cut > best_cut + self._BOUND_TOLERANCE:
            return True
        if cut < best_cut - self._BOUND_TOLERANCE:
            return False
        return best_sample is None or tuple(sample) < tuple(best_sample)

    def _choose_branching_pair(self, matrix, reduced, branching_rule):
        """Rank all distinct representative pairs according to R1/R2/R3."""
        pair_scores = []
        row_confidence = np.sum((1.0 - np.abs(matrix)) ** 2, axis=1)

        for left in range(reduced.variable_count):
            for right in range(left + 1, reduced.variable_count):
                magnitude = abs(float(matrix[left, right]))
                if branching_rule == 'r1':
                    score = -magnitude
                elif branching_rule == 'r2':
                    score = float(
                        row_confidence[left] + row_confidence[right]
                    )
                else:
                    score = magnitude
                pair_scores.append((score, left, right))

        _, reduced_left, reduced_right = min(pair_scores)
        return (
            reduced.representatives[reduced_left],
            reduced.representatives[reduced_right],
        )

    def _schedule_parity_branches(self, pair, partition, search):
        """Create the paper's same-spin and opposite-spin child nodes."""
        left, right = pair
        same = partition.clone()
        opposite = partition.clone()
        if not same.join(left, right, parity=0):
            raise RuntimeError('A fresh same-spin parity branch conflicted.')
        if not opposite.join(left, right, parity=1):
            raise RuntimeError('A fresh opposite-spin parity branch conflicted.')

        search['frontier'].append(same)
        search['frontier'].append(opposite)
        statistics = search['statistics']
        statistics.branches_created += 2
        statistics.max_frontier_size = max(
            statistics.max_frontier_size,
            len(search['frontier']),
        )

    def _evaluate_source_cut(self, model, sample):
        """Evaluate a source cut once using upper-triangular edge weights."""
        return _evaluate_weighted_cut(model.weights, sample)

    def _evaluate_reduced_cut(self, reduced, sample):
        """Evaluate compensation plus the signed reduced MaxCut objective."""
        cut = reduced.compensation
        for left in range(reduced.variable_count):
            for right in range(left + 1, reduced.variable_count):
                if sample[left] != sample[right]:
                    cut += reduced.weights[left, right]
        return float(cut)

    def _finish_search(self, search, config):
        """Translate tree exhaustion or a budget stop into one contract result."""
        statistics = search['statistics']
        stopped_by = search['stopped_by']
        if stopped_by is None:
            status = 'optimal'
            termination_reason = 'edge_parity_tree_exhausted'
        elif stopped_by == 'timeout_reached':
            status = 'timeout'
            termination_reason = stopped_by
        else:
            status = 'feasible' if search['best_sample'] is not None else 'unknown'
            termination_reason = stopped_by

        metrics = {
            'best_cut': (
                search['best_cut']
                if search['best_sample'] is not None
                else None
            ),
            'nodes_explored': statistics.nodes_explored,
            'nodes_pruned': statistics.nodes_pruned,
            'branches_created': statistics.branches_created,
            'exact_subproblems': statistics.exact_subproblems,
            'max_frontier_size': statistics.max_frontier_size,
            'tree_exhausted': stopped_by is None,
        }
        metrics.update(
            self._relaxation_result_metrics(
                statistics,
                search['root_relaxation'],
            )
        )
        metadata = {
            'reproduction_scope': 'nonnegative_weighted_maxcut',
            'paper_variant': 'admissible_laplacian_eigenvalue_bound',
            'diagonal_correction': 'u=0',
            'branching_rule': config['branching_rule'],
            'branching_matrix': config['branching_matrix'],
            'traversal': config['traversal'],
            'brute_force_threshold': config['brute_force_threshold'],
            'source_relation': 'qubo_energy = offset - cut_value',
            'optimality_basis': (
                'admissible_bound_and_exhausted_parity_tree'
                if status == 'optimal'
                else 'not_proven'
            ),
            'research_source_revision': (
                'ec72c202559655dc170f8bdf41f2936107ce94f8'
            ),
        }
        metadata.update(
            self._relaxation_result_metadata(
                config,
                search['root_relaxation'],
            )
        )
        return QuboSolveOutcome(
            status=status,
            best_sample=search['best_sample'],
            termination_reason=termination_reason,
            metrics=metrics,
            trace=search['trace'],
            metadata=metadata,
        )

    def _relaxation_result_metrics(self, statistics, root_relaxation):
        """Expose QRR work with stable, method-specific metric names."""
        metrics = {
            'qrr_subproblems': statistics.relaxation_subproblems,
            'optimizer_evaluations': statistics.relaxation_work_units,
        }
        if root_relaxation is not None:
            metrics['root_qaoa_expectation'] = root_relaxation[
                'expectation'
            ]
            metrics['root_correlation_frobenius_norm'] = root_relaxation[
                'correlation_frobenius_norm'
            ]
        return metrics

    def _relaxation_result_metadata(self, config, root_relaxation):
        """Describe the QAOA/QRR provider without polluting tree logic."""
        return {
            'algorithm': 'qrr_informed_edge_parity_branch_and_bound',
            'qaoa_layers': 1,
            'selective_rank': config['selective_rank'],
            'qaoa_grid_size': config['qaoa_grid_size'],
            'qaoa_refinement_steps': config['qaoa_refinement_steps'],
            'root_qaoa_parameters': (
                None
                if root_relaxation is None
                else {
                    'gamma': root_relaxation['gamma'],
                    'beta': root_relaxation['beta'],
                }
            ),
        }


__all__ = ['QrbnbrQuboSolver']
