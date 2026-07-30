"""Independent exact arithmetic for small canonical QUBO problems."""

from fractions import Fraction
from itertools import product


def evaluate_qubo_energy(problem, sample):
    """Evaluate one QUBO sample without production contract helpers."""
    _validate_binary_sample(sample, problem['num_variables'])

    energy = Fraction(problem['offset'])
    for left, right, coefficient in problem['terms']:
        energy += Fraction(coefficient) * sample[left] * sample[right]
    return energy


def enumerate_qubo(problem):
    """Return every sample ordered by exact energy and canonical tie-break."""
    rows = [
        {
            'sample': list(bits),
            'energy_exact': evaluate_qubo_energy(problem, bits),
        }
        for bits in product((0, 1), repeat=problem['num_variables'])
    ]
    rows.sort(key=lambda row: (row['energy_exact'], row['sample']))
    return rows


def _validate_binary_sample(sample, variable_count):
    """Reject malformed witnesses before they can corrupt an oracle result."""
    if isinstance(sample, (str, bytes)) or len(sample) != variable_count:
        raise ValueError(
            f'sample must contain exactly {variable_count} binary values.'
        )
    if any(type(value) is not int or value not in (0, 1) for value in sample):
        raise ValueError('sample must contain only integer 0/1 values.')
