# CBQM to QUBO compiler

`compile_qubo` converts `cbqm.v1` into `qubo.v1` and returns a compilation
context for restoring original variables later.

Input: a validated `cbqm.v1` mapping and explicit compiler configuration.

Output: `(qubo, compilation_context)`, where `qubo` is a QUBO-solver input and the
context records fixed/free mappings, slack variables, numerical lattices, and
penalty choices.

```python
from lib import compile_qubo

qubo, context = compile_qubo(
    cbqm,
    {
        'strategy': 'quadratic_penalty',
        'default_penalty': 20.0,
        'penalty_by_family': {
            'budget': 50.0,
            'one_weight_level': 50.0,
        },
        'penalty_by_constraint': {},
        'inequality_encoding': 'binary_slack',
        'constraint_precision': 8,
        'rounding_tolerance': 1e-9,
        'zero_tolerance': 1e-12,
        'fixed_value_strategy': 'eliminate',
    },
)
```

Only `default_penalty` is required. Every other field has the value shown
above as its default. Constraint-name overrides take precedence over family
overrides, which take precedence over the default.

## Numerical convention

Each linear constraint is converted to the smallest integer lattice supported
by `constraint_precision`. For example,

```text
0.02 x0 + 0.04 x1 = 1.00
```

becomes

```text
x0 + 2 x1 = 50
```

The configured penalty multiplies the squared residual in this integer
lattice. Therefore a residual of one lattice unit incurs exactly one penalty
unit. Values that cannot be represented within `rounding_tolerance` are
rejected instead of silently rounded.

Equalities use `penalty * (lhs - rhs)^2`. Upper and lower inequalities add
bounded binary slack variables before applying the same squared penalty. A
two-sided range is encoded as two inequalities.

Fixed CBQM variables are substituted exactly and removed from the QUBO. The
compilation context records free-variable mappings, fixed values, slack
variables, normalization units, penalty choices, and the original objective
sense.

Penalty sufficiency is intentionally not inferred. Selecting penalties remains
an explicit modelling and algorithm decision.
