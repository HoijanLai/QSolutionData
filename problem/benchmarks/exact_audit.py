"""Executable exact-solver timing audit for the mathematical benchmark suite."""

import math
import time
from collections.abc import Sequence

from .catalog import build_mathematical_benchmark_suite


def audit_exact_benchmarks(
    benchmark_names=None,
    *,
    timeout_seconds=30 * 60,
):
    """Solve selected fixtures exactly and return JSON-compatible timing rows."""
    names = _resolve_benchmark_names(benchmark_names)
    _validate_timeout(timeout_seconds)
    suite = build_mathematical_benchmark_suite()
    _require_known_names(names, suite)

    rows = []
    for name in names:
        problem = suite[name]
        solver, config = _exact_solver_for(
            problem,
            timeout_seconds,
        )
        started = time.perf_counter()
        result = solver.solve(problem, config)
        wall_seconds = time.perf_counter() - started
        rows.append(
            _audit_row(
                name,
                problem,
                result,
                wall_seconds,
            )
        )
    return rows


def _resolve_benchmark_names(benchmark_names):
    """Use catalog order by default and reject ambiguous string iterables."""
    if benchmark_names is None:
        return list(build_mathematical_benchmark_suite())
    if (
        isinstance(benchmark_names, (str, bytes))
        or not isinstance(benchmark_names, Sequence)
    ):
        raise TypeError(
            'benchmark_names must be a sequence of names or None.'
        )
    names = list(benchmark_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(
            'Every benchmark name must be a non-empty string.'
        )
    if len(names) != len(set(names)):
        raise ValueError('benchmark_names must not contain duplicates.')
    return names


def _validate_timeout(timeout_seconds):
    """Require one finite positive per-instance wall-clock budget."""
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError(
            'timeout_seconds must be a finite positive number.'
        )


def _require_known_names(names, suite):
    """Fail before running anything if one requested identity is unknown."""
    unknown = [name for name in names if name not in suite]
    if unknown:
        raise KeyError(f'Unknown mathematical benchmarks: {unknown}')


def _exact_solver_for(problem, timeout_seconds):
    """Select the representation-native Exact solver and safety config."""
    if problem['schema'] == 'qubo.v1':
        from lib.solvers.qubo import ExactQuboSolver

        return ExactQuboSolver(), {
            'max_variables': 24,
            'timeout_seconds': timeout_seconds,
        }
    if problem['schema'] == 'cbqm.v1':
        from lib.solvers.cbqm import ExactCbqmSolver

        return ExactCbqmSolver(), {
            'max_variables': 24,
            'timeout_seconds': timeout_seconds,
        }
    if problem['schema'] == 'mis.v1':
        from lib.solvers.mis import ExactMisSolver

        return ExactMisSolver(), {
            'max_vertices': 48,
            'timeout_seconds': timeout_seconds,
        }
    raise ValueError(
        f"No Exact solver is configured for '{problem['schema']}'."
    )


def _audit_row(name, problem, result, wall_seconds):
    """Normalize representation-specific result fields into one audit row."""
    objective, solution = _result_candidate(problem, result)
    return {
        'benchmark_name': name,
        'problem_id': problem['problem_id'],
        'schema': problem['schema'],
        'size': problem['metadata']['benchmark']['size'],
        'status': result['status'],
        'objective_value': objective,
        'solution': solution,
        'solver_runtime_seconds': result['runtime_seconds'],
        'wall_seconds': wall_seconds,
        'within_intended_budget': (
            result['status'] == 'optimal'
            and wall_seconds
            <= problem['metadata']['benchmark'][
                'intended_exact_budget_seconds'
            ]
        ),
    }


def _result_candidate(problem, result):
    """Read native candidate fields without changing their representation."""
    if problem['schema'] == 'qubo.v1':
        return result['best_energy'], result['best_sample']
    if problem['schema'] == 'cbqm.v1':
        return result['best_objective'], result['best_sample']
    if problem['schema'] == 'mis.v1':
        return result['objective_value'], result['selected_vertices']
    raise ValueError(
        f"Unsupported audit result schema for '{problem['schema']}'."
    )


__all__ = ['audit_exact_benchmarks']
