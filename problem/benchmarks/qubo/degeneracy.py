"""Small QUBOs exposing degeneracy and geometric frustration."""

from .._metadata import benchmark_metadata


def build_degenerate_qubo_suite():
    """Return ferromagnetic and frustrated cycle energy landscapes."""
    return {
        'qubo.ferromagnetic-cycle-18': _build_cycle_qubo(
            variable_count=18,
            prefer_equal=True,
        ),
        'qubo.frustrated-cycle-17': _build_cycle_qubo(
            variable_count=17,
            prefer_equal=False,
        ),
    }


def _build_cycle_qubo(*, variable_count, prefer_equal):
    """Penalize domain walls, or equal neighbours on an odd cycle."""
    coefficients = {}
    offset = 0
    for left in range(variable_count):
        right = (left + 1) % variable_count
        if prefer_equal:
            # (x_left - x_right)^2
            _add(coefficients, left, left, 1)
            _add(coefficients, right, right, 1)
            _add(coefficients, left, right, -2)
        else:
            # (x_left + x_right - 1)^2
            offset += 1
            _add(coefficients, left, left, -1)
            _add(coefficients, right, right, -1)
            _add(coefficients, left, right, 2)

    model = (
        'ferromagnetic-cycle'
        if prefer_equal
        else 'antiferromagnetic-odd-cycle'
    )
    parameters = {
        'topology': 'cycle',
        'coupling': 'prefer-equal' if prefer_equal else 'prefer-unequal',
        'known_optimum_energy': 0 if prefer_equal else 1,
        'degenerate': True,
    }
    if prefer_equal:
        parameters['known_ground_state_count'] = 2
    else:
        parameters['geometrically_frustrated'] = True

    return {
        'schema': 'qubo.v1',
        'problem_id': f'benchmark-{model}-{variable_count}',
        'sense': 'minimize',
        'num_variables': variable_count,
        'variable_names': [
            f'cycle_{index}'
            for index in range(variable_count)
        ],
        'offset': offset,
        'terms': [
            [left, right, coefficient]
            for (left, right), coefficient in sorted(coefficients.items())
            if coefficient != 0
        ],
        'metadata': benchmark_metadata(
            family='degenerate-energy-landscape',
            model=model,
            generator='build_degenerate_qubo_suite',
            size={
                'num_variables': variable_count,
                'interaction_count': variable_count,
            },
            parameters=parameters,
        ),
    }


def _add(coefficients, left, right, value):
    """Aggregate cycle endpoints into upper-triangular QUBO terms."""
    key = (min(left, right), max(left, right))
    coefficients[key] = coefficients.get(key, 0) + value


__all__ = ['build_degenerate_qubo_suite']
