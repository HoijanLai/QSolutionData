"""Dependency-free runtime validation for versioned optimization contracts.

The JSON Schemas under ``contracts/schemas`` define the wire format.  These
validators mirror the relevant schema rules in ordinary Python and then add
cross-field semantics that JSON Schema cannot conveniently express, such as
recomputing a reported QUBO energy from its sample.

Public functions validate values entering or leaving a solver boundary.
Private aliases are retained because the compiler and adapters historically
imported them directly.
"""

import json
import math
from collections.abc import Mapping
from fractions import Fraction


_QUBO_REQUIRED_FIELDS = frozenset(
    {
        'schema',
        'problem_id',
        'sense',
        'num_variables',
        'variable_names',
        'offset',
        'terms',
    }
)
_QUBO_OPTIONAL_FIELDS = frozenset({'metadata'})

_CBQM_REQUIRED_FIELDS = frozenset(
    {
        'schema',
        'problem_id',
        'variables',
        'objective',
        'constraints',
        'fixed_values',
    }
)
_CBQM_OPTIONAL_FIELDS = frozenset({'metadata'})
_CBQM_VARIABLE_REQUIRED_FIELDS = frozenset({'index', 'name', 'vartype'})
_CBQM_VARIABLE_OPTIONAL_FIELDS = frozenset({'kind', 'metadata'})
_CBQM_OBJECTIVE_REQUIRED_FIELDS = frozenset(
    {'sense', 'offset', 'linear', 'quadratic'}
)
_CBQM_CONSTRAINT_REQUIRED_FIELDS = frozenset({'name', 'family', 'linear'})
_CBQM_CONSTRAINT_OPTIONAL_FIELDS = frozenset(
    {'lower_bound', 'upper_bound', 'metadata'}
)
_CBQM_FIXED_REQUIRED_FIELDS = frozenset({'index', 'value'})

_RESULT_REQUIRED_FIELDS = frozenset(
    {
        'schema',
        'problem_id',
        'solver',
        'status',
        'best_sample',
        'best_energy',
        'runtime_seconds',
    }
)
_RESULT_OPTIONAL_FIELDS = frozenset(
    {
        'termination_reason',
        'metrics',
        'trace',
        'metadata',
    }
)
_RESULT_STATUSES = frozenset(
    {
        'optimal',
        'feasible',
        'infeasible',
        'timeout',
        'error',
        'unknown',
    }
)
_SOLVER_REQUIRED_FIELDS = frozenset({'name', 'version'})
_SOLVER_OPTIONAL_FIELDS = frozenset({'backend'})
_TRACE_REQUIRED_FIELDS = frozenset({'step', 'time_seconds', 'energy'})
_TRACE_OPTIONAL_FIELDS = frozenset({'sample', 'metadata'})


def validate_qubo(problem):
    """Validate a canonical ``qubo.v1`` document.

    Validation is intentionally strict at the wire boundary:

    - all schema-required fields must exist and unknown fields are rejected;
    - arrays must be actual JSON-style lists, not arbitrary iterables;
    - indices are genuine Python integers (booleans are not integers here);
    - every number is finite and the complete document is JSON serializable;
    - sparse terms remain upper triangular, nonzero, and unique.

    The function returns ``None`` on success and raises ``TypeError`` or
    ``ValueError`` with a field-oriented message on failure.
    """

    _validate_mapping(problem, 'problem')
    _validate_object_fields(
        problem,
        required=_QUBO_REQUIRED_FIELDS,
        optional=_QUBO_OPTIONAL_FIELDS,
        label='problem',
    )

    if problem['schema'] != 'qubo.v1':
        raise ValueError("problem schema must be 'qubo.v1'.")
    _validate_problem_id(problem)
    if problem['sense'] != 'minimize':
        raise ValueError("qubo.v1 sense must be 'minimize'.")

    variable_count = problem['num_variables']
    _validate_non_negative_integer(variable_count, 'num_variables')

    variable_names = problem['variable_names']
    _validate_string_list(variable_names, 'variable_names', allow_empty=True)
    if len(variable_names) != variable_count:
        raise ValueError('variable_names length must equal num_variables.')
    if len(variable_names) != len(set(variable_names)):
        raise ValueError('Variable names must be unique.')

    _validate_finite_number(problem['offset'], 'QUBO offset')
    _validate_sparse_quadratic_terms(
        problem['terms'],
        variable_count,
        'QUBO terms',
    )

    if 'metadata' in problem:
        _validate_json_object(problem['metadata'], "problem['metadata']")
    _validate_json_document(problem, 'problem')


def validate_qubo_result(problem, result):
    """Validate ``result`` against its source canonical QUBO.

    In addition to mirroring ``qubo-result.v1.schema.json``, this function
    checks the relationships that make the result trustworthy:

    - the result belongs to the supplied problem;
    - candidate presence agrees with status and energy presence;
    - samples use the problem's exact variable count and Python ``0``/``1``;
    - reported candidate and trace energies equal canonical recomputation;
    - nested solver, trace, metrics, and metadata objects reject invalid shape;
    - the complete result is finite and strictly JSON serializable.

    ``problem`` comes first to match the existing validation/evaluation style.
    The function returns ``None`` after successful validation.
    """

    validate_qubo(problem)
    _validate_mapping(result, 'result')
    _validate_object_fields(
        result,
        required=_RESULT_REQUIRED_FIELDS,
        optional=_RESULT_OPTIONAL_FIELDS,
        label='result',
    )

    if result['schema'] != 'qubo-result.v1':
        raise ValueError("result schema must be 'qubo-result.v1'.")
    _validate_result_problem_id(problem, result)
    _validate_solver_identity(result['solver'])

    status = result['status']
    if status not in _RESULT_STATUSES:
        raise ValueError(f"Unknown solver status '{status}'.")

    sample = result['best_sample']
    energy = result['best_energy']
    _validate_result_candidate(problem, status, sample, energy)
    _validate_non_negative_finite_number(
        result['runtime_seconds'],
        'runtime_seconds',
    )

    if 'termination_reason' in result and not isinstance(
        result['termination_reason'],
        str,
    ):
        raise TypeError('termination_reason must be a string.')
    if 'metrics' in result:
        _validate_json_object(result['metrics'], "result['metrics']")
    if 'trace' in result:
        _validate_result_trace(problem, result['trace'])
    if 'metadata' in result:
        _validate_json_object(result['metadata'], "result['metadata']")

    _validate_json_document(result, 'result')


def validate_cbqm(problem):
    """Validate the complete closed ``cbqm.v1`` JSON contract."""
    _validate_mapping(problem, 'problem')
    _validate_object_fields(
        problem,
        required=_CBQM_REQUIRED_FIELDS,
        optional=_CBQM_OPTIONAL_FIELDS,
        label='problem',
    )
    if problem['schema'] != 'cbqm.v1':
        raise ValueError("problem schema must be 'cbqm.v1'.")
    _validate_problem_id(problem)

    variables = problem['variables']
    if not isinstance(variables, list):
        raise TypeError("problem['variables'] must be a list.")

    variable_names = []
    for position, variable in enumerate(variables):
        _validate_mapping(variable, f'variable {position}')
        _validate_object_fields(
            variable,
            required=_CBQM_VARIABLE_REQUIRED_FIELDS,
            optional=_CBQM_VARIABLE_OPTIONAL_FIELDS,
            label=f'variable {position}',
        )
        # ``bool`` is a subclass of ``int`` in Python, so equality alone is
        # insufficient here: ``False == 0`` and ``True == 1``. Validate the
        # exact JSON integer type first, then enforce canonical list ordering.
        _validate_index(
            variable['index'],
            len(variables),
            f'Variable {position} index',
        )
        if variable['index'] != position:
            raise ValueError('Variable indices must match their list positions.')
        name = variable['name']
        if not isinstance(name, str) or not name:
            raise ValueError('Variable names must be non-empty strings.')
        if variable['vartype'] != 'BINARY':
            raise ValueError('cbqm.v1 adapters only accept BINARY variables.')
        if 'kind' in variable and not isinstance(variable['kind'], str):
            raise TypeError('Variable kind must be a string.')
        if 'metadata' in variable:
            _validate_json_object(
                variable['metadata'],
                f"variable {position}['metadata']",
            )
        variable_names.append(name)
    if len(variable_names) != len(set(variable_names)):
        raise ValueError('Variable names must be unique.')

    objective = problem['objective']
    _validate_mapping(objective, "problem['objective']")
    _validate_object_fields(
        objective,
        required=_CBQM_OBJECTIVE_REQUIRED_FIELDS,
        optional=frozenset(),
        label="problem['objective']",
    )
    if objective['sense'] not in {'minimize', 'maximize'}:
        raise ValueError("Objective sense must be 'minimize' or 'maximize'.")
    _validate_finite_number(objective['offset'], 'Objective offset')
    _validate_sparse_linear_terms(
        objective['linear'],
        len(variables),
        'Objective linear terms',
    )
    _validate_sparse_quadratic_terms(
        objective['quadratic'],
        len(variables),
        'Objective quadratic terms',
    )

    constraints = problem['constraints']
    if not isinstance(constraints, list):
        raise TypeError("problem['constraints'] must be a list.")
    constraint_names = []
    for position, constraint in enumerate(constraints):
        _validate_mapping(constraint, f'constraint {position}')
        _validate_object_fields(
            constraint,
            required=_CBQM_CONSTRAINT_REQUIRED_FIELDS,
            optional=_CBQM_CONSTRAINT_OPTIONAL_FIELDS,
            label=f'constraint {position}',
        )
        name = constraint['name']
        family = constraint['family']
        if not isinstance(name, str) or not name:
            raise ValueError('Constraint names must be non-empty strings.')
        if not isinstance(family, str) or not family:
            raise ValueError('Constraint families must be non-empty strings.')
        constraint_names.append(name)
        _validate_sparse_linear_terms(
            constraint['linear'],
            len(variables),
            f"Constraint '{name}' terms",
        )
        if 'lower_bound' not in constraint and 'upper_bound' not in constraint:
            raise ValueError(f"Constraint '{name}' must define at least one bound.")
        lower = constraint.get('lower_bound')
        upper = constraint.get('upper_bound')
        if 'lower_bound' in constraint:
            _validate_finite_number(lower, f"Constraint '{name}' lower bound")
        if 'upper_bound' in constraint:
            _validate_finite_number(upper, f"Constraint '{name}' upper bound")
        if (
            'lower_bound' in constraint
            and 'upper_bound' in constraint
            and lower > upper
        ):
            raise ValueError(f"Constraint '{name}' has inconsistent bounds.")
        if 'metadata' in constraint:
            _validate_json_object(
                constraint['metadata'],
                f"constraint {position}['metadata']",
            )
    if len(constraint_names) != len(set(constraint_names)):
        raise ValueError('Constraint names must be unique.')

    fixed_values = problem['fixed_values']
    if not isinstance(fixed_values, list):
        raise TypeError("problem['fixed_values'] must be a list.")
    fixed_indices = []
    for position, fixed_value in enumerate(fixed_values):
        label = f'fixed value {position}'
        _validate_mapping(fixed_value, label)
        _validate_object_fields(
            fixed_value,
            required=_CBQM_FIXED_REQUIRED_FIELDS,
            optional=frozenset(),
            label=label,
        )
        index = fixed_value['index']
        value = fixed_value['value']
        _validate_index(index, len(variables), 'Fixed variable index')
        if not _is_binary_python_integer(value):
            raise ValueError('Fixed variable values must be integer 0 or 1.')
        fixed_indices.append(index)
    if len(fixed_indices) != len(set(fixed_indices)):
        raise ValueError('A variable may only be fixed once.')
    if 'metadata' in problem:
        _validate_json_object(problem['metadata'], "problem['metadata']")
    _validate_json_document(problem, 'problem')


def _evaluate_qubo(problem, sample):
    """Evaluate a QUBO and return its canonical JSON-number representation.

    The polynomial is accumulated as an exact rational first.  This matters
    even for integer-looking models: converting a large offset to ``float``
    before adding a small bias can erase a genuine one-unit improvement and
    turn exhaustive search into a false proof of optimality.
    """

    exact_energy = _evaluate_qubo_exact(problem, sample)
    return _fraction_to_json_number(exact_energy, 'QUBO energy')


def _evaluate_qubo_exact(problem, sample):
    """Evaluate the JSON-number polynomial without floating-point summation.

    Python integers map to exact integers and Python floats map to their exact
    IEEE-754 values.  The returned :class:`Fraction` is for internal
    comparisons and certificates; it is deliberately not part of the JSON
    wire contract.
    """

    _validate_binary_sample(
        sample,
        problem['num_variables'],
        'Sample',
    )
    energy = Fraction(problem['offset'])
    for left, right, coefficient in problem['terms']:
        if sample[left] and sample[right]:
            energy += Fraction(coefficient)
    return energy


def _fraction_to_json_number(value, label):
    """Represent an exact rational as a finite built-in JSON number.

    Integral results remain integers, including values beyond binary64's exact
    range.  Non-integral results necessarily use a float because the public
    contracts intentionally expose ordinary JSON numbers rather than
    ``Fraction`` objects.
    """

    if value.denominator == 1:
        return value.numerator
    try:
        output = float(value)
    except OverflowError as error:
        raise ValueError(
            f'{label} is not representable as a finite JSON number.'
        ) from error
    if not math.isfinite(output):
        raise ValueError(
            f'{label} is not representable as a finite JSON number.'
        )
    return output


def _validate_result_problem_id(problem, result):
    """Require a result to identify exactly the problem it was evaluated on."""

    problem_id = result['problem_id']
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError('result problem_id must be a non-empty string.')
    if problem_id != problem['problem_id']:
        raise ValueError(
            'result problem_id must match the source problem_id.'
        )


def _validate_solver_identity(solver):
    """Validate the closed solver-identity object from the result schema."""

    _validate_mapping(solver, "result['solver']")
    _validate_object_fields(
        solver,
        required=_SOLVER_REQUIRED_FIELDS,
        optional=_SOLVER_OPTIONAL_FIELDS,
        label="result['solver']",
    )
    for field_name in ('name', 'version'):
        value = solver[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"result['solver']['{field_name}'] must be a non-empty string."
            )
    if 'backend' in solver and not isinstance(solver['backend'], str):
        raise TypeError("result['solver']['backend'] must be a string.")


def _validate_result_candidate(problem, status, sample, energy):
    """Check status/nullability rules and recompute a present candidate."""

    if (sample is None) != (energy is None):
        raise ValueError(
            'best_sample and best_energy must either both be null or both be present.'
        )
    if status in {'optimal', 'feasible'} and sample is None:
        raise ValueError(f"Status '{status}' requires a best_sample and best_energy.")
    if status == 'infeasible' and sample is not None:
        raise ValueError(
            "Status 'infeasible' cannot include a best_sample or best_energy."
        )
    if sample is None:
        return

    _validate_binary_sample(
        sample,
        problem['num_variables'],
        'best_sample',
    )
    _validate_finite_number(energy, 'best_energy')
    recomputed = _evaluate_qubo(problem, sample)
    _require_equal_energy(energy, recomputed, 'best_energy')


def _validate_result_trace(problem, trace):
    """Validate every closed ``trace`` entry and its optional candidate."""

    if not isinstance(trace, list):
        raise TypeError("result['trace'] must be a list.")

    for position, entry in enumerate(trace):
        label = f"result['trace'][{position}]"
        _validate_mapping(entry, label)
        _validate_object_fields(
            entry,
            required=_TRACE_REQUIRED_FIELDS,
            optional=_TRACE_OPTIONAL_FIELDS,
            label=label,
        )
        _validate_non_negative_integer(entry['step'], f"{label}['step']")
        _validate_non_negative_finite_number(
            entry['time_seconds'],
            f"{label}['time_seconds']",
        )
        _validate_finite_number(entry['energy'], f"{label}['energy']")

        if 'sample' in entry:
            _validate_binary_sample(
                entry['sample'],
                problem['num_variables'],
                f"{label}['sample']",
            )
            recomputed = _evaluate_qubo(problem, entry['sample'])
            _require_equal_energy(
                entry['energy'],
                recomputed,
                f"{label}['energy']",
            )
        if 'metadata' in entry:
            _validate_json_object(
                entry['metadata'],
                f"{label}['metadata']",
            )


def _require_equal_energy(reported, recomputed, label):
    """Require the exact canonical JSON number, allowing ``2`` versus ``2.0``.

    A relative tolerance is unsafe here: at a large magnitude it can accept an
    energy that is thousands of objective units away.  Both values already
    are validated finite JSON numbers, so rational comparison is deterministic
    and round-trip safe.
    """

    if Fraction(reported) != Fraction(recomputed):
        raise ValueError(
            f'{label} does not match the energy recomputed from its sample.'
        )


def _validate_sparse_linear_terms(terms, variable_count, label):
    """Validate JSON-array sparse linear terms."""

    if not isinstance(terms, list):
        raise TypeError(f'{label} must be a list.')
    seen = set()
    for term in terms:
        if not isinstance(term, list) or len(term) != 2:
            raise ValueError(f'{label} must contain [index, coefficient] pairs.')
        index, coefficient = term
        _validate_index(index, variable_count, f'{label} index')
        _validate_finite_number(coefficient, f'{label} coefficient')
        if coefficient == 0:
            raise ValueError(f'{label} must omit zero coefficients.')
        if index in seen:
            raise ValueError(f'{label} contains duplicate indices.')
        seen.add(index)


def _validate_sparse_quadratic_terms(terms, variable_count, label):
    """Validate JSON-array sparse upper-triangular quadratic terms."""

    if not isinstance(terms, list):
        raise TypeError(f'{label} must be a list.')
    seen = set()
    for term in terms:
        if not isinstance(term, list) or len(term) != 3:
            raise ValueError(
                f'{label} must contain [left, right, coefficient] triples.'
            )
        left, right, coefficient = term
        _validate_index(left, variable_count, f'{label} left index')
        _validate_index(right, variable_count, f'{label} right index')
        if left > right:
            raise ValueError(f'{label} must use upper-triangular indices.')
        _validate_finite_number(coefficient, f'{label} coefficient')
        if coefficient == 0:
            raise ValueError(f'{label} must omit zero coefficients.')
        pair = (left, right)
        if pair in seen:
            raise ValueError(f'{label} contains duplicate index pairs.')
        seen.add(pair)


def _validate_object_fields(value, *, required, optional, label):
    """Enforce JSON Schema ``required`` and ``additionalProperties: false``."""

    keys = set(value)
    missing = required - keys
    if missing:
        names = ', '.join(sorted(missing))
        raise ValueError(f'{label} is missing required fields: {names}.')

    unknown = keys - required - optional
    if unknown:
        names = ', '.join(sorted(str(name) for name in unknown))
        raise ValueError(f'{label} contains unknown fields: {names}.')


def _validate_binary_sample(sample, variable_count, label):
    """Validate a JSON array of genuine Python binary integers."""

    if not isinstance(sample, list):
        raise TypeError(f'{label} must be a list.')
    if len(sample) != variable_count:
        raise ValueError(f'{label} length must equal num_variables.')
    if any(not _is_binary_python_integer(value) for value in sample):
        raise ValueError(
            f'{label} must contain only Python integer binary values 0 or 1.'
        )


def _is_binary_python_integer(value):
    """Exclude booleans and NumPy integer scalars from the wire contract."""

    return type(value) is int and value in {0, 1}


def _validate_string_list(values, label, *, allow_empty):
    """Validate one JSON string array."""

    if not isinstance(values, list):
        raise TypeError(f'{label} must be a list.')
    if not allow_empty and not values:
        raise ValueError(f'{label} must not be empty.')
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f'{label} must contain non-empty strings.')


def _validate_json_object(value, label):
    """Validate an arbitrary object-valued extension field."""

    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be an object.')
    _validate_json_document(value, label)


def _validate_json_document(value, label):
    """Reject Python-only values, non-string object keys, NaN, and infinity."""

    _validate_json_value(value, label)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f'{label} must be JSON serializable.') from error


def _validate_json_value(value, label):
    """Walk a value so Python's permissive encoder cannot coerce object keys."""

    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f'{label} must not contain NaN or infinity.')
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f'{label}[{index}]')
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f'{label} object keys must be strings.')
            _validate_json_value(item, f"{label}['{key}']")
        return
    value_type = type(value).__name__
    raise TypeError(f'{label} contains a non-JSON value of type {value_type}.')


def _validate_mapping(value, label):
    """Require a mapping before field-oriented validation."""

    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping.')


def _validate_problem_id(problem):
    """Validate the common non-empty problem identifier."""

    problem_id = problem.get('problem_id')
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError('problem_id must be a non-empty string.')


def _validate_index(index, variable_count, label):
    """Validate one bounded Python integer variable index."""

    if (
        type(index) is not int
        or index < 0
        or index >= variable_count
    ):
        raise ValueError(f'{label} is outside the variable range.')


def _validate_non_negative_integer(value, label):
    """Validate a Python integer that is at least zero."""

    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a non-negative integer.')


def _validate_finite_number(value, label):
    """Validate a JSON-compatible finite real number, excluding booleans."""

    if type(value) is int:
        return
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f'{label} must be a finite real number.')


def _validate_non_negative_finite_number(value, label):
    """Validate a finite JSON number with a non-negative lower bound."""

    _validate_finite_number(value, label)
    if value < 0:
        raise ValueError(f'{label} must be non-negative.')


# Compatibility aliases for compiler and adapter modules that predate the
# public API.  Keep these assignments (rather than wrappers) so behaviour
# cannot drift between the old and new import paths.
_validate_cbqm = validate_cbqm
_validate_qubo = validate_qubo
