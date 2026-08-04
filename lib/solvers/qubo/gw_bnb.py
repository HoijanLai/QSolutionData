"""Goemans--Williamson-informed edge-parity branch-and-bound for MaxCut.

This is the classical control route used alongside QRR-BnB in the Q-RBnBR
study.  It reuses the exact same parity tree, variable elimination, admissible
bound, traversal, branching rules and exact leaf closure as
``QrbnbrQuboSolver``.  The only algorithmic substitution is the relaxation:
an SDP matrix supplies both random-hyperplane candidates and R1/R2/R3
branching information.

Signed couplings may appear after parity elimination.  The SDP remains a valid
relaxation and useful branching heuristic there, but the standalone GW
``0.878`` expectation theorem is not claimed for those reduced signed graphs.
Optimality still comes exclusively from the exhausted, admissibly pruned tree.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import BaseQuboSolver
from .goemans_williamson import GoemansWilliamsonQuboSolver
from .qrbnbr import QrbnbrQuboSolver, _Relaxation


class GwBranchAndBoundQuboSolver(QrbnbrQuboSolver):
    """Reproduce the SDP-informed classical parity-BnB baseline."""

    SOLVER_NAME = 'gw-parity-bnb-maxcut-reproduction'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'cvxpy-sdp'

    DEFAULT_ROUNDS = 16
    DEFAULT_SEED = 0
    DEFAULT_MAX_VARIABLES = 18

    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'branching_rule',
            'traversal',
            'brute_force_threshold',
            'rounds',
            'seed',
            'sdp_solver',
            'sdp_tolerance',
            'sdp_max_iterations',
            'max_variables',
            'max_nodes',
            'timeout_seconds',
        }
    )

    def _resolve_config(self, config):
        """Validate tree controls and the replaceable SDP provider."""
        resolved = BaseQuboSolver._resolve_config(self, config)
        self._reject_unknown_gw_bnb_fields(resolved)
        gw_kernel = GoemansWilliamsonQuboSolver()

        output = {
            'branching_rule': resolved.get('branching_rule', 'r1'),
            'branching_matrix': 'sdp',
            'traversal': resolved.get('traversal', 'bfs'),
            'brute_force_threshold': resolved.get(
                'brute_force_threshold',
                self.DEFAULT_BRUTE_FORCE_THRESHOLD,
            ),
            'rounds': resolved.get('rounds', self.DEFAULT_ROUNDS),
            'seed': resolved.get('seed', self.DEFAULT_SEED),
            'sdp_solver': gw_kernel._resolve_sdp_solver(
                resolved.get('sdp_solver')
            ),
            'sdp_tolerance': resolved.get(
                'sdp_tolerance',
                gw_kernel.DEFAULT_SDP_TOLERANCE,
            ),
            'sdp_max_iterations': resolved.get(
                'sdp_max_iterations',
                gw_kernel.DEFAULT_SDP_MAX_ITERATIONS,
            ),
            'max_variables': resolved.get(
                'max_variables',
                self.DEFAULT_MAX_VARIABLES,
            ),
            'max_nodes': resolved.get('max_nodes'),
            'timeout_seconds': resolved.get('timeout_seconds'),
        }
        self._validate_gw_bnb_config(output, gw_kernel)
        return output

    def _reject_unknown_gw_bnb_fields(self, config):
        """Reject route settings that do not belong to the classical solver."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'Unknown GW-BnB solver config fields: {names}')

    def _validate_gw_bnb_config(self, config, gw_kernel):
        """Validate shared tree controls and standalone GW controls."""
        if config['branching_rule'] not in {'r1', 'r2', 'r3'}:
            raise ValueError("branching_rule must be 'r1', 'r2', or 'r3'.")
        if config['traversal'] not in {'bfs', 'dfs'}:
            raise ValueError("traversal must be 'bfs' or 'dfs'.")

        self._validate_non_negative_integer(
            config['brute_force_threshold'],
            'brute_force_threshold',
        )
        self._validate_positive_integer(config['rounds'], 'rounds')
        self._validate_non_negative_integer(config['seed'], 'seed')
        self._validate_non_negative_integer(
            config['max_variables'],
            'max_variables',
        )
        self._validate_optional_positive_integer(
            config['max_nodes'],
            'max_nodes',
        )
        self._validate_timeout(config['timeout_seconds'])
        gw_kernel._validate_config_values(
            {
                'rounds': config['rounds'],
                'seed': config['seed'],
                'sdp_tolerance': config['sdp_tolerance'],
                'sdp_max_iterations': config['sdp_max_iterations'],
                'max_variables': config['max_variables'],
            }
        )

    def _solve_relaxation(self, reduced, config):
        """Solve the reduced SDP and round its vectors by random hyperplanes."""
        gw_kernel = GoemansWilliamsonQuboSolver()
        sdp_solution = gw_kernel._solve_sdp(reduced.weights, config)
        vectors = gw_kernel._factor_psd_matrix(sdp_solution.matrix)
        local_config = dict(config)
        local_config['seed'] = self._subproblem_seed(reduced, config['seed'])
        rounding = gw_kernel._round_hyperplanes(
            vectors,
            reduced.weights,
            local_config,
        )

        return _Relaxation(
            reduced_sample=rounding.sample,
            correlation=sdp_solution.matrix,
            branching_matrix=sdp_solution.matrix,
            work_units=1,
            summary={
                'sdp_relaxation_value': (
                    reduced.compensation + sdp_solution.objective
                ),
                'sdp_solver': sdp_solution.solver,
                'sdp_status': sdp_solution.status,
                'sdp_solve_time_seconds': (
                    sdp_solution.solve_time_seconds
                ),
                'sdp_iterations': sdp_solution.iterations,
                'sdp_matrix_frobenius_norm': float(
                    self._matrix_norm(sdp_solution.matrix)
                ),
                'rounding_unique_samples': rounding.unique_samples,
            },
        )

    def _subproblem_seed(self, reduced, base_seed):
        """Derive a stable independent seed from one reduced subproblem."""
        digest = hashlib.sha256()
        digest.update(str(base_seed).encode('ascii'))
        digest.update(repr(reduced.representatives).encode('ascii'))
        digest.update(reduced.weights.tobytes(order='C'))
        return int.from_bytes(digest.digest()[:8], 'big') % (2 ** 63)

    def _matrix_norm(self, matrix):
        """Keep NumPy interaction behind one easy-to-test seam."""
        return np.linalg.norm(matrix)

    def _relaxation_result_metrics(self, statistics, root_relaxation):
        """Expose SDP work without retaining QAOA-specific metric names."""
        metrics = {
            'gw_subproblems': statistics.relaxation_subproblems,
            'sdp_solves': statistics.relaxation_work_units,
        }
        if root_relaxation is None:
            return metrics

        metrics['root_sdp_relaxation_value'] = root_relaxation[
            'sdp_relaxation_value'
        ]
        metrics['root_sdp_matrix_frobenius_norm'] = root_relaxation[
            'sdp_matrix_frobenius_norm'
        ]
        metrics['root_rounding_unique_samples'] = root_relaxation[
            'rounding_unique_samples'
        ]
        if root_relaxation['sdp_solve_time_seconds'] is not None:
            metrics['root_sdp_solve_time_seconds'] = root_relaxation[
                'sdp_solve_time_seconds'
            ]
        if root_relaxation['sdp_iterations'] is not None:
            metrics['root_sdp_iterations'] = root_relaxation[
                'sdp_iterations'
            ]
        return metrics

    def _relaxation_result_metadata(self, config, root_relaxation):
        """Describe the classical provider attached to the shared tree."""
        return {
            'algorithm': (
                'goemans_williamson_informed_edge_parity_branch_and_bound'
            ),
            'rounds_per_subproblem': config['rounds'],
            'seed': config['seed'],
            'sdp_solver': config['sdp_solver'],
            'sdp_tolerance': config['sdp_tolerance'],
            'sdp_max_iterations': config['sdp_max_iterations'],
            'root_sdp_status': (
                None
                if root_relaxation is None
                else root_relaxation['sdp_status']
            ),
            'signed_subproblem_guarantee': (
                'no_standalone_gw_approximation_ratio_claim'
            ),
        }


__all__ = ['GwBranchAndBoundQuboSolver']
