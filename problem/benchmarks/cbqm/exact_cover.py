"""Binary exact-cover CBQM benchmark."""

from .._metadata import benchmark_metadata
from ._common import _binary_variables, _linear_constraint


def build_exact_cover_suite():
    """Return a twelve-subset exact-cover minimization problem."""
    universe_size = 8
    subsets = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [0, 2, 4],
        [1, 3, 5],
        [2, 6],
        [3, 7],
        [0, 5, 6],
        [1, 4, 7],
        [0, 3, 4, 7],
        [1, 2, 5, 6],
    ]
    return {
        'cbqm.exact-cover-8x12': _build_exact_cover(
            universe_size,
            subsets,
        ),
    }


def _build_exact_cover(universe_size, subsets):
    """Require every universe element to occur in exactly one chosen subset."""
    constraints = [
        _linear_constraint(
            name=f'cover_element_{element}',
            family='exact-cover',
            terms=[
                (subset_index, 1)
                for subset_index, subset in enumerate(subsets)
                if element in subset
            ],
            lower_bound=1,
            upper_bound=1,
        )
        for element in range(universe_size)
    ]
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'benchmark-exact-cover-8x12',
        'variables': _binary_variables(
            [
                f'choose_subset_{index}'
                for index in range(len(subsets))
            ],
            kind='selection',
        ),
        'objective': {
            'sense': 'minimize',
            'offset': 0,
            'linear': [
                [index, 1]
                for index in range(len(subsets))
            ],
            'quadratic': [],
        },
        'constraints': constraints,
        'fixed_values': [],
        'metadata': benchmark_metadata(
            family='covering',
            model='exact-cover',
            generator='build_exact_cover_suite',
            size={
                'universe_size': universe_size,
                'subset_count': len(subsets),
                'num_variables': len(subsets),
                'constraint_count': len(constraints),
            },
            parameters={'subsets': subsets},
        ),
    }


__all__ = ['build_exact_cover_suite']
