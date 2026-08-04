"""One-hot travelling-salesperson CBQM benchmark."""

from .._metadata import benchmark_metadata
from ._common import _binary_variables, _linear_constraint


def build_tsp_suite():
    """Return a four-city cyclic TSP with rotational symmetry removed."""
    distances = [
        [0, 4, 7, 3],
        [4, 0, 2, 6],
        [7, 2, 0, 5],
        [3, 6, 5, 0],
    ]
    return {
        'cbqm.tsp-4': _build_tsp(distances),
    }


def _build_tsp(distances):
    """Encode city/position assignment and cyclic travel cost directly."""
    city_count = len(distances)
    variable_names = [
        f'city_{city}_at_position_{position}'
        for city in range(city_count)
        for position in range(city_count)
    ]
    constraints = []

    for city in range(city_count):
        constraints.append(
            _linear_constraint(
                name=f'visit_city_{city}_once',
                family='assignment',
                terms=[
                    (_variable_index(city, position, city_count), 1)
                    for position in range(city_count)
                ],
                lower_bound=1,
                upper_bound=1,
            )
        )
    for position in range(city_count):
        constraints.append(
            _linear_constraint(
                name=f'fill_position_{position}_once',
                family='assignment',
                terms=[
                    (_variable_index(city, position, city_count), 1)
                    for city in range(city_count)
                ],
                lower_bound=1,
                upper_bound=1,
            )
        )

    quadratic = {}
    for position in range(city_count):
        next_position = (position + 1) % city_count
        for departure in range(city_count):
            for arrival in range(city_count):
                if departure == arrival:
                    continue
                left = _variable_index(
                    departure,
                    position,
                    city_count,
                )
                right = _variable_index(
                    arrival,
                    next_position,
                    city_count,
                )
                key = (min(left, right), max(left, right))
                quadratic[key] = (
                    quadratic.get(key, 0)
                    + distances[departure][arrival]
                )

    variable_count = city_count * city_count
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'benchmark-tsp-4',
        'variables': _binary_variables(
            variable_names,
            kind='assignment',
        ),
        'objective': {
            'sense': 'minimize',
            'offset': 0,
            'linear': [],
            'quadratic': [
                [left, right, coefficient]
                for (left, right), coefficient in sorted(quadratic.items())
            ],
        },
        'constraints': constraints,
        # Fix the tour origin only; reversal symmetry remains intentionally.
        'fixed_values': [{'index': 0, 'value': 1}],
        'metadata': benchmark_metadata(
            family='routing',
            model='travelling-salesperson',
            generator='build_tsp_suite',
            size={
                'city_count': city_count,
                'num_variables': variable_count,
                'free_variable_count': variable_count - 1,
                'constraint_count': len(constraints),
            },
            parameters={
                'distance_matrix': distances,
                'encoding': 'city-by-position-one-hot',
                'tour': 'cyclic',
                'fixed_origin_city': 0,
            },
        ),
    }


def _variable_index(city, position, city_count):
    """Map a city/position pair to its stable binary-variable index."""
    return city * city_count + position


__all__ = ['build_tsp_suite']
