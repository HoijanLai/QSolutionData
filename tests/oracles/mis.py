"""Independent brute-force maximum independent set oracle for tiny graphs."""

from fractions import Fraction
from itertools import product


def enumerate_mis(problem):
    """Return feasible index sets ordered by exact objective and witness."""
    rows = []
    for bits in product((0, 1), repeat=len(problem['vertices'])):
        selected = [
            index
            for index, bit in enumerate(bits)
            if bit
        ]
        if not _is_feasible(problem, bits):
            continue
        rows.append({
            'selected_vertices': selected,
            'objective_exact': _objective(problem, selected),
        })
    rows.sort(
        key=lambda row: (
            -row['objective_exact'],
            row['selected_vertices'],
        )
    )
    return rows


def _is_feasible(problem, bits):
    """Check edges and fixed values without production helpers."""
    for left, right in problem['edges']:
        if bits[left] and bits[right]:
            return False
    for fixed in problem.get('fixed_values', []):
        if bits[fixed['index']] != fixed['value']:
            return False
    return True


def _objective(problem, selected):
    """Evaluate cardinality or explicit weight with exact JSON arithmetic."""
    if problem['objective']['kind'] == 'maximum-cardinality':
        return Fraction(len(selected))
    return sum(
        (
            Fraction(problem['vertices'][index]['weight'])
            for index in selected
        ),
        start=Fraction(0),
    )


__all__ = ['enumerate_mis']
