"""Protected Ising-to-QUBO construction helpers used by benchmark families."""


def _ising_to_qubo(
    *,
    problem_id,
    variable_names,
    couplings,
    fields,
    metadata,
):
    """Map ``H(s) = -ΣJij si sj - Σhi si`` through ``si = 2xi - 1``."""
    offset = 0
    coefficients = {}

    for left, right, coupling in couplings:
        _add_coefficient(coefficients, left, left, 2 * coupling)
        _add_coefficient(coefficients, right, right, 2 * coupling)
        _add_coefficient(coefficients, left, right, -4 * coupling)
        offset -= coupling

    for index, field in enumerate(fields):
        _add_coefficient(coefficients, index, index, -2 * field)
        offset += field

    terms = [
        [left, right, coefficient]
        for (left, right), coefficient in sorted(coefficients.items())
        if coefficient != 0
    ]
    return {
        'schema': 'qubo.v1',
        'problem_id': problem_id,
        'sense': 'minimize',
        'num_variables': len(variable_names),
        'variable_names': list(variable_names),
        'offset': offset,
        'terms': terms,
        'metadata': metadata,
    }


def _add_coefficient(coefficients, left, right, value):
    """Aggregate one upper-triangular QUBO coefficient."""
    key = (min(left, right), max(left, right))
    coefficients[key] = coefficients.get(key, 0) + value


__all__ = ['_ising_to_qubo']
