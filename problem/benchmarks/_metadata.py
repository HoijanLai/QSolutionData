"""Shared metadata for deterministic mathematical benchmark generators."""

import copy


GENERATOR_VERSION = '1.0.0'
INTENDED_EXACT_BUDGET_SECONDS = 30 * 60


def benchmark_metadata(
    *,
    family,
    model,
    generator,
    size,
    parameters=None,
    seed=None,
):
    """Describe mathematical provenance without attaching business meaning."""
    benchmark = {
        'domain': 'mathematical',
        'family': family,
        'model': model,
        'generator': generator,
        'generator_version': GENERATOR_VERSION,
        'size_class': 'laptop-exact',
        'intended_exact_budget_seconds': INTENDED_EXACT_BUDGET_SECONDS,
        'size': copy.deepcopy(size),
        'parameters': copy.deepcopy(parameters or {}),
    }
    if seed is not None:
        benchmark['seed'] = seed
    return {'benchmark': benchmark}


__all__ = [
    'GENERATOR_VERSION',
    'INTENDED_EXACT_BUDGET_SECONDS',
    'benchmark_metadata',
]
