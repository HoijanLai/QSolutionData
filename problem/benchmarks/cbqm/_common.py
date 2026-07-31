"""Protected constructors shared by mathematical CBQM benchmark families."""


def _binary_variables(names, *, kind):
    """Create contiguous binary-variable declarations with stable names."""
    return [
        {
            'index': index,
            'name': name,
            'vartype': 'BINARY',
            'kind': kind,
        }
        for index, name in enumerate(names)
    ]


def _linear_constraint(
    *,
    name,
    family,
    terms,
    lower_bound=None,
    upper_bound=None,
):
    """Build one canonical sorted nonzero linear constraint."""
    constraint = {
        'name': name,
        'family': family,
        'linear': [
            [index, coefficient]
            for index, coefficient in sorted(terms)
            if coefficient != 0
        ],
    }
    if lower_bound is not None:
        constraint['lower_bound'] = lower_bound
    if upper_bound is not None:
        constraint['upper_bound'] = upper_bound
    return constraint


__all__ = ['_binary_variables', '_linear_constraint']
