"""Independent small-instance oracles used by tests and validation notebooks.

These helpers deliberately do not import production evaluators or solvers.
They provide a second implementation against which contract and algorithm
results can be checked.
"""

from .cbqm import (
    enumerate_cbqm_feasible,
    evaluate_cbqm_objective,
    is_cbqm_feasible,
)
from .mis import enumerate_mis
from .numbers import public_json_number
from .qubo import enumerate_qubo, evaluate_qubo_energy

__all__ = [
    'enumerate_cbqm_feasible',
    'enumerate_mis',
    'enumerate_qubo',
    'evaluate_cbqm_objective',
    'evaluate_qubo_energy',
    'is_cbqm_feasible',
    'public_json_number',
]
