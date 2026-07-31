"""Small, deterministic problem fixtures used by solver validation.

Benchmark builders live outside solver modules so an algorithm cannot quietly
tailor the problem it is being asked to solve.  Only stable suite-level entry
points are exported from this package; individual fixture builders remain
implementation details of their respective modules.
"""

from .catalog import (
    build_mathematical_benchmark_cases,
    build_mathematical_benchmark_suite,
)
from .cbqm import build_cbqm_mathematical_suite
from .exact_audit import audit_exact_benchmarks
from .mis import build_mis_mathematical_suite
from .qubo import build_qubo_mathematical_suite
from .qubo_annealing import build_annealing_validation_suite
from .references import build_mathematical_benchmark_references


__all__ = [
    'audit_exact_benchmarks',
    'build_annealing_validation_suite',
    'build_cbqm_mathematical_suite',
    'build_mathematical_benchmark_cases',
    'build_mathematical_benchmark_references',
    'build_mathematical_benchmark_suite',
    'build_mis_mathematical_suite',
    'build_qubo_mathematical_suite',
]
