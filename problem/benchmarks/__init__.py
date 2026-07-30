"""Small, deterministic problem fixtures used by solver validation.

Benchmark builders live outside solver modules so an algorithm cannot quietly
tailor the problem it is being asked to solve.  Only stable suite-level entry
points are exported from this package; individual fixture builders remain
implementation details of their respective modules.
"""

from .qubo_annealing import build_annealing_validation_suite


__all__ = ['build_annealing_validation_suite']
