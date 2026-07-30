"""Deterministic QUBO fixtures for simulated-annealing validation.

The suite is deliberately tiny.  Its purpose is to exercise representation
and numerical edge cases in unit tests and notebooks, not to claim meaningful
performance measurements.  Every call constructs fresh JSON-native mappings,
so callers may safely annotate or transform a returned fixture without
changing a later experiment.
"""

from lib.contracts.solver_protocol import QuboProblem as _QuboProblem


__all__ = ['build_annealing_validation_suite']


def build_annealing_validation_suite() -> dict[str, _QuboProblem]:
    """Return named, deterministic ``qubo.v1`` fixtures for SA validation.

    The public workflow intentionally reads like a fixture checklist.  Details
    such as identifiers, coefficients, and descriptive metadata are isolated
    in protected builders below.
    """

    return {
        'zero_variable': _build_zero_variable_qubo(),
        'positive_bias': _build_single_bias_qubo(
            problem_id='annealing-positive-bias',
            coefficient=2,
        ),
        'negative_bias': _build_single_bias_qubo(
            problem_id='annealing-negative-bias',
            coefficient=-2,
        ),
        'ferromagnetic_pair': _build_ferromagnetic_pair_qubo(),
        'frustrated_triangle': _build_frustrated_triangle_qubo(),
        'sparse_random': _build_sparse_random_qubo(),
        'large_offset_small_bias': _build_large_offset_qubo(),
        'degenerate': _build_degenerate_qubo(),
        'custom_schedule': _build_custom_schedule_qubo(),
    }


def _build_zero_variable_qubo() -> _QuboProblem:
    """Exercise the unique empty binary assignment."""

    return _make_qubo(
        problem_id='annealing-zero-variable',
        variable_names=[],
        offset=3,
        terms=[],
        fixture='zero_variable',
        description='The empty sample is the only candidate.',
    )


def _build_single_bias_qubo(
    *,
    problem_id: str,
    coefficient: int,
) -> _QuboProblem:
    """Build the smallest problem with one possible Metropolis flip."""

    direction = 'positive' if coefficient > 0 else 'negative'
    return _make_qubo(
        problem_id=problem_id,
        variable_names=['x'],
        offset=0,
        terms=[[0, 0, coefficient]],
        fixture=f'{direction}_bias',
        description=f'One-variable QUBO with a {direction} linear bias.',
    )


def _build_ferromagnetic_pair_qubo() -> _QuboProblem:
    """Prefer aligned bits while retaining two degenerate ground states."""

    # x0 + x1 - 2*x0*x1 is zero for 00 and 11, and one otherwise.
    return _make_qubo(
        problem_id='annealing-ferromagnetic-pair',
        variable_names=['left', 'right'],
        offset=0,
        terms=[
            [0, 0, 1],
            [0, 1, -2],
            [1, 1, 1],
        ],
        fixture='ferromagnetic_pair',
        description='Aligned bits have lower energy than anti-aligned bits.',
    )


def _build_frustrated_triangle_qubo() -> _QuboProblem:
    """Encode negative MaxCut on a triangle, whose three edges cannot all cut."""

    return _make_qubo(
        problem_id='annealing-frustrated-triangle',
        variable_names=['a', 'b', 'c'],
        offset=0,
        terms=[
            [0, 0, -2],
            [0, 1, 2],
            [0, 2, 2],
            [1, 1, -2],
            [1, 2, 2],
            [2, 2, -2],
        ],
        fixture='frustrated_triangle',
        description='Negative MaxCut energy for a unit-weight triangle.',
    )


def _build_sparse_random_qubo() -> _QuboProblem:
    """Provide a fixed sparse signed instance without runtime randomness."""

    # These coefficients were fixed by hand rather than generated when the
    # suite is called.  Instance randomness and solver randomness therefore
    # remain separate, and the benchmark is stable across Python/NumPy versions.
    return _make_qubo(
        problem_id='annealing-sparse-random-seed-1729',
        variable_names=['r0', 'r1', 'r2', 'r3', 'r4'],
        offset=2,
        terms=[
            [0, 0, -3],
            [0, 2, 2],
            [0, 4, -1],
            [1, 1, 4],
            [1, 3, -3],
            [2, 2, -2],
            [2, 4, 3],
            [3, 3, 1],
            [4, 4, -2],
        ],
        fixture='sparse_random',
        description='A fixed sparse signed QUBO labelled with instance seed 1729.',
        extra_metadata={'instance_seed': 1729},
    )


def _build_large_offset_qubo() -> _QuboProblem:
    """Ensure a small improving bias survives a much larger integer offset."""

    return _make_qubo(
        problem_id='annealing-large-offset-small-bias',
        variable_names=['resolution_bit'],
        offset=10_000_000_000_000_000,
        terms=[[0, 0, -1]],
        fixture='large_offset_small_bias',
        description='A one-unit bias beneath an exactly represented large offset.',
    )


def _build_degenerate_qubo() -> _QuboProblem:
    """Make every assignment equally good for tie and reproducibility tests."""

    return _make_qubo(
        problem_id='annealing-degenerate',
        variable_names=['d0', 'd1', 'd2'],
        offset=7,
        terms=[],
        fixture='degenerate',
        description='All eight assignments have identical energy.',
    )


def _build_custom_schedule_qubo() -> _QuboProblem:
    """Offer a compact coupled instance for custom beta-schedule tests."""

    return _make_qubo(
        problem_id='annealing-custom-schedule',
        variable_names=['s0', 's1'],
        offset=1,
        terms=[
            [0, 0, -1],
            [0, 1, 3],
            [1, 1, -2],
        ],
        fixture='custom_schedule',
        description='A two-variable QUBO used with caller-supplied beta values.',
    )


def _make_qubo(
    *,
    problem_id: str,
    variable_names: list[str],
    offset: int,
    terms: list[list[int | float]],
    fixture: str,
    description: str,
    extra_metadata: dict | None = None,
) -> _QuboProblem:
    """Assemble one canonical mapping while keeping builders declarative."""

    metadata = {
        'benchmark_suite': 'qubo-annealing.v1',
        'fixture': fixture,
        'description': description,
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)

    return {
        'schema': 'qubo.v1',
        'problem_id': problem_id,
        'sense': 'minimize',
        'num_variables': len(variable_names),
        'variable_names': variable_names,
        'offset': offset,
        'terms': terms,
        'metadata': metadata,
    }
