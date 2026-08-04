"""JSON-number conversion for independently computed exact values."""

import math
from fractions import Fraction


def public_json_number(exact_value):
    """Return the integral-or-float shape used by public JSON contracts."""
    if not isinstance(exact_value, Fraction):
        raise TypeError('exact_value must be a Fraction.')
    if exact_value.denominator == 1:
        return exact_value.numerator
    try:
        output = float(exact_value)
    except OverflowError as error:
        raise ValueError(
            'exact_value is not representable as a finite JSON number.'
        ) from error
    if not math.isfinite(output):
        raise ValueError(
            'exact_value is not representable as a finite JSON number.'
        )
    return output
