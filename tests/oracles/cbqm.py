"""Independent exact arithmetic for small canonical CBQM problems."""

from fractions import Fraction
from itertools import product


def evaluate_cbqm_objective(problem, sample):
    """Evaluate the original CBQM objective with exact JSON-number arithmetic."""
    _validate_binary_sample(sample, len(problem['variables']))

    objective = problem['objective']
    value = Fraction(objective['offset'])
    for index, coefficient in objective['linear']:
        value += Fraction(coefficient) * sample[index]
    for left, right, coefficient in objective['quadratic']:
        value += Fraction(coefficient) * sample[left] * sample[right]
    return value


def is_cbqm_feasible(problem, sample):
    """Check fixed values and every explicit CBQM linear constraint."""
    _validate_binary_sample(sample, len(problem['variables']))

    for fixed in problem.get('fixed_values', []):
        if sample[fixed['index']] != fixed['value']:
            return False

    for constraint in problem['constraints']:
        activity = sum(
            (
                Fraction(coefficient) * sample[index]
                for index, coefficient in constraint['linear']
            ),
            start=Fraction(0),
        )
        lower = constraint.get('lower_bound')
        upper = constraint.get('upper_bound')
        if lower is not None and activity < Fraction(lower):
            return False
        if upper is not None and activity > Fraction(upper):
            return False

    return True


def enumerate_cbqm_feasible(problem):
    """Return feasible samples ordered by objective and canonical tie-break."""
    rows = []
    for bits in product((0, 1), repeat=len(problem['variables'])):
        if not is_cbqm_feasible(problem, bits):
            continue
        rows.append(
            {
                'sample': list(bits),
                'objective_exact': evaluate_cbqm_objective(problem, bits),
            }
        )

    sense = problem['objective']['sense']
    if sense == 'minimize':
        rows.sort(key=lambda row: (row['objective_exact'], row['sample']))
    elif sense == 'maximize':
        rows.sort(key=lambda row: (-row['objective_exact'], row['sample']))
    else:
        raise ValueError(f"Unknown CBQM objective sense '{sense}'.")
    return rows


def _validate_binary_sample(sample, variable_count):
    """Keep the oracle independent while still rejecting invalid witnesses."""
    if isinstance(sample, (str, bytes)) or len(sample) != variable_count:
        raise ValueError(
            f'sample must contain exactly {variable_count} binary values.'
        )
    if any(type(value) is not int or value not in (0, 1) for value in sample):
        raise ValueError('sample must contain only integer 0/1 values.')
