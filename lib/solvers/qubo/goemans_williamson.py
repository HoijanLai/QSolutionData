"""Goemans--Williamson SDP relaxation and hyperplane rounding for MaxCut.

The solver deliberately accepts the same strict non-negative weighted MaxCut
subset of ``qubo.v1`` as the Q-RBnBR reproduction.  Its public algorithm reads
like the classical four-step method; CVXPY setup, PSD factorization, random
hyperplanes, tie-breaking and contract metadata remain protected details.

Finite random rounding is an approximation algorithm, so normal completion
returns ``feasible`` rather than ``optimal``.  The reported SDP value is useful
as a relaxation diagnostic but is not promoted to a repository-wide proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from ._maxcut import (
    _evaluate_weighted_cut,
    _read_nonnegative_maxcut_qubo,
)
from .base import BaseQuboSolver, QuboSolveOutcome


@dataclass(frozen=True)
class _GwSdpSolution:
    """Numerical SDP matrix and backend diagnostics."""

    matrix: np.ndarray
    objective: float
    solver: str
    status: str
    solve_time_seconds: float | None
    iterations: int | None


@dataclass(frozen=True)
class _GwRoundingResult:
    """Best hyperplane cut plus sampling statistics."""

    sample: tuple[int, ...]
    cut: float
    unique_samples: int


class GoemansWilliamsonQuboSolver(BaseQuboSolver):
    """Approximate non-negative weighted MaxCut through SDP rounding.

    Supported configuration fields:

    ``rounds``
        Number of independent Gaussian hyperplanes. Defaults to ``128``.
    ``seed``
        Non-negative NumPy random seed. Defaults to ``0``.
    ``sdp_solver``
        ``"CLARABEL"`` or ``"SCS"``. ``None`` selects the first installed
        solver in that order.
    ``sdp_tolerance``
        Positive numerical tolerance passed to the selected backend. Defaults
        to ``1e-7``.
    ``sdp_max_iterations``
        Positive backend iteration limit. Defaults to ``10_000``.
    ``max_variables``
        Safety rail for the dense ``n by n`` SDP variable. Defaults to ``100``.
    """

    SOLVER_NAME = 'goemans-williamson-maxcut'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'cvxpy-sdp'

    DEFAULT_ROUNDS = 128
    DEFAULT_SEED = 0
    DEFAULT_SDP_TOLERANCE = 1e-7
    DEFAULT_SDP_MAX_ITERATIONS = 10_000
    DEFAULT_MAX_VARIABLES = 100

    _SUPPORTED_SDP_SOLVERS = ('CLARABEL', 'SCS')
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'rounds',
            'seed',
            'sdp_solver',
            'sdp_tolerance',
            'sdp_max_iterations',
            'max_variables',
        }
    )

    def _resolve_config(self, config):
        """Validate all experiment choices before constructing an SDP."""
        resolved = super()._resolve_config(config)
        self._reject_unknown_config_fields(resolved)

        output = {
            'rounds': resolved.get('rounds', self.DEFAULT_ROUNDS),
            'seed': resolved.get('seed', self.DEFAULT_SEED),
            'sdp_solver': self._resolve_sdp_solver(
                resolved.get('sdp_solver')
            ),
            'sdp_tolerance': resolved.get(
                'sdp_tolerance',
                self.DEFAULT_SDP_TOLERANCE,
            ),
            'sdp_max_iterations': resolved.get(
                'sdp_max_iterations',
                self.DEFAULT_SDP_MAX_ITERATIONS,
            ),
            'max_variables': resolved.get(
                'max_variables',
                self.DEFAULT_MAX_VARIABLES,
            ),
        }
        self._validate_config_values(output)
        return output

    def _run(self, problem, config):
        """Relax, factor, round, and report—the GW algorithm in four lines."""
        model = _read_nonnegative_maxcut_qubo(problem)
        self._guard_problem_size(model.variable_count, config)

        sdp_solution = self._solve_sdp(model.weights, config)
        vectors = self._factor_psd_matrix(sdp_solution.matrix)
        rounding = self._round_hyperplanes(
            vectors,
            model.weights,
            config,
        )

        return self._build_outcome(rounding, sdp_solution, config)

    def _reject_unknown_config_fields(self, config):
        """Prevent misspelled experiment settings from being ignored."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(
                f'Unknown Goemans-Williamson solver config fields: {names}'
            )

    def _resolve_sdp_solver(self, requested_solver):
        """Choose a deterministic PSD-capable backend."""
        if requested_solver is not None:
            if not isinstance(requested_solver, str):
                raise TypeError('sdp_solver must be a string or None.')
            requested_solver = requested_solver.upper()
            if requested_solver not in self._SUPPORTED_SDP_SOLVERS:
                supported = ', '.join(self._SUPPORTED_SDP_SOLVERS)
                raise ValueError(
                    f'sdp_solver must be one of: {supported}.'
                )
            if requested_solver not in cp.installed_solvers():
                raise ValueError(
                    f"Requested SDP solver '{requested_solver}' is not "
                    'installed in the active Python environment.'
                )
            return requested_solver

        installed = set(cp.installed_solvers())
        for solver in self._SUPPORTED_SDP_SOLVERS:
            if solver in installed:
                return solver
        raise ValueError(
            'Goemans-Williamson requires CLARABEL or SCS, but neither is '
            'installed in the active Python environment.'
        )

    def _validate_config_values(self, config):
        """Validate integer and finite-real safety rails."""
        self._validate_positive_integer(config['rounds'], 'rounds')
        self._validate_non_negative_integer(config['seed'], 'seed')
        self._validate_positive_integer(
            config['sdp_max_iterations'],
            'sdp_max_iterations',
        )
        self._validate_non_negative_integer(
            config['max_variables'],
            'max_variables',
        )
        tolerance = config['sdp_tolerance']
        if (
            not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError('sdp_tolerance must be a finite positive number.')

    def _validate_positive_integer(self, value, name):
        """Require a genuine integer greater than zero."""
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer.')

    def _validate_non_negative_integer(self, value, name):
        """Require a genuine integer greater than or equal to zero."""
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer.')

    def _guard_problem_size(self, variable_count, config):
        """Fail before allocating an unexpectedly large dense SDP."""
        if variable_count > config['max_variables']:
            raise ValueError(
                f'Goemans-Williamson received {variable_count} variables, '
                f"exceeding max_variables={config['max_variables']}."
            )

    def _solve_sdp(self, weights, config):
        """Solve ``max trace(LX)/4`` subject to ``diag(X)=1, X>=0``."""
        variable_count = weights.shape[0]
        if variable_count == 0:
            return _GwSdpSolution(
                matrix=np.zeros((0, 0), dtype=float),
                objective=0.0,
                solver=config['sdp_solver'],
                status='trivial',
                solve_time_seconds=0.0,
                iterations=0,
            )

        laplacian = np.diag(np.sum(weights, axis=1)) - weights
        matrix_variable = cp.Variable(
            (variable_count, variable_count),
            symmetric=True,
        )
        objective = cp.Maximize(
            0.25 * cp.trace(laplacian @ matrix_variable)
        )
        constraints = [
            cp.diag(matrix_variable) == 1,
            matrix_variable >> 0,
        ]
        optimization = cp.Problem(objective, constraints)
        options = self._sdp_solver_options(config)
        value = optimization.solve(
            solver=config['sdp_solver'],
            warm_start=False,
            verbose=False,
            **options,
        )

        if optimization.status not in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }:
            raise RuntimeError(
                'Goemans-Williamson SDP did not produce a usable solution: '
                f'status={optimization.status!r}.'
            )
        if matrix_variable.value is None or value is None:
            raise RuntimeError(
                'Goemans-Williamson SDP returned no numerical matrix.'
            )

        matrix = np.asarray(matrix_variable.value, dtype=float)
        matrix = 0.5 * (matrix + matrix.T)
        if not np.all(np.isfinite(matrix)) or not math.isfinite(float(value)):
            raise RuntimeError(
                'Goemans-Williamson SDP returned non-finite values.'
            )
        return _GwSdpSolution(
            matrix=matrix,
            objective=float(value),
            solver=config['sdp_solver'],
            status=str(optimization.status),
            solve_time_seconds=self._optional_finite_float(
                optimization.solver_stats.solve_time
            ),
            iterations=self._optional_integer(
                optimization.solver_stats.num_iters
            ),
        )

    def _sdp_solver_options(self, config):
        """Translate neutral tolerance settings to one CVXPY backend."""
        tolerance = config['sdp_tolerance']
        iterations = config['sdp_max_iterations']
        if config['sdp_solver'] == 'CLARABEL':
            return {
                'tol_gap_abs': tolerance,
                'tol_gap_rel': tolerance,
                'tol_feas': tolerance,
                'max_iter': iterations,
            }
        return {
            'eps': tolerance,
            'max_iters': iterations,
        }

    def _factor_psd_matrix(self, matrix):
        """Return row vectors whose Gram matrix is the projected SDP matrix."""
        if matrix.size == 0:
            return np.zeros((0, 0), dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        threshold = np.finfo(float).eps * scale * matrix.shape[0] * 16
        positive = eigenvalues > threshold
        if not np.any(positive):
            return np.zeros((matrix.shape[0], 1), dtype=float)
        return (
            eigenvectors[:, positive]
            * np.sqrt(np.maximum(eigenvalues[positive], 0.0))
        )

    def _round_hyperplanes(self, vectors, weights, config):
        """Sample Gaussian hyperplanes and retain the best canonical cut."""
        variable_count = vectors.shape[0]
        if variable_count == 0:
            return _GwRoundingResult(
                sample=(),
                cut=0.0,
                unique_samples=1,
            )

        random_generator = np.random.default_rng(config['seed'])
        candidates = {tuple([0] * variable_count)}
        for _ in range(config['rounds']):
            direction = random_generator.normal(size=vectors.shape[1])
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                continue
            projections = vectors @ (direction / norm)
            sample = tuple(int(value >= 0.0) for value in projections)
            candidates.add(self._canonical_cut_orientation(sample))

        ranked = sorted(
            (
                (_evaluate_weighted_cut(weights, sample), sample)
                for sample in candidates
            ),
            key=lambda item: (-item[0], item[1]),
        )
        best_cut, best_sample = ranked[0]
        return _GwRoundingResult(
            sample=best_sample,
            cut=best_cut,
            unique_samples=len(candidates),
        )

    def _canonical_cut_orientation(self, sample):
        """Choose the lexicographically smaller global-spin orientation."""
        complement = tuple(1 - value for value in sample)
        return min(tuple(sample), complement)

    def _build_outcome(self, rounding, sdp_solution, config):
        """Build an approximate result without turning the SDP into a proof."""
        metrics = {
            'cut_value': rounding.cut,
            'sdp_relaxation_value': sdp_solution.objective,
            'rounds': config['rounds'],
            'unique_samples': rounding.unique_samples,
        }
        if sdp_solution.objective > 0:
            metrics['cut_to_sdp_ratio'] = (
                rounding.cut / sdp_solution.objective
            )
        if sdp_solution.solve_time_seconds is not None:
            metrics['sdp_solve_time_seconds'] = (
                sdp_solution.solve_time_seconds
            )
        if sdp_solution.iterations is not None:
            metrics['sdp_iterations'] = sdp_solution.iterations

        return QuboSolveOutcome(
            status='feasible',
            best_sample=rounding.sample,
            termination_reason='sdp_and_hyperplane_rounding_completed',
            metrics=metrics,
            metadata={
                'algorithm': 'goemans_williamson',
                'reproduction_scope': 'nonnegative_weighted_maxcut',
                'seed': config['seed'],
                'sdp_solver': sdp_solution.solver,
                'sdp_status': sdp_solution.status,
                'sdp_tolerance': config['sdp_tolerance'],
                'sdp_max_iterations': config['sdp_max_iterations'],
                'source_relation': 'qubo_energy = offset - cut_value',
                'optimality_basis': 'not_proven',
            },
        )

    def _optional_finite_float(self, value):
        """Normalize optional CVXPY statistics to JSON-native values."""
        if value is None:
            return None
        output = float(value)
        return output if math.isfinite(output) and output >= 0 else None

    def _optional_integer(self, value):
        """Normalize an optional iteration count."""
        if value is None:
            return None
        output = int(value)
        return output if output >= 0 else None


__all__ = ['GoemansWilliamsonQuboSolver']
