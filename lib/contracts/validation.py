import math
from collections.abc import Mapping


def _validate_cbqm(problem):
    """Validate cbqm."""
    _validate_mapping(problem, 'problem')
    if problem.get('schema') != 'cbqm.v1':
        raise ValueError("problem schema must be 'cbqm.v1'.")
    _validate_problem_id(problem)

    variables = problem.get('variables')
    if not isinstance(variables, list):
        raise TypeError("problem['variables'] must be a list.")

    variable_names = []
    for position, variable in enumerate(variables):
        _validate_mapping(variable, f'variable {position}')
        if variable.get('index') != position:
            raise ValueError('Variable indices must match their list positions.')
        name = variable.get('name')
        if not isinstance(name, str) or not name:
            raise ValueError('Variable names must be non-empty strings.')
        if variable.get('vartype') != 'BINARY':
            raise ValueError('cbqm.v1 adapters only accept BINARY variables.')
        variable_names.append(name)
    if len(variable_names) != len(set(variable_names)):
        raise ValueError('Variable names must be unique.')

    objective = problem.get('objective')
    _validate_mapping(objective, "problem['objective']")
    if objective.get('sense') not in {'minimize', 'maximize'}:
        raise ValueError("Objective sense must be 'minimize' or 'maximize'.")
    _validate_finite_number(objective.get('offset'), 'Objective offset')
    _validate_sparse_linear_terms(
        objective.get('linear'),
        len(variables),
        'Objective linear terms',
    )
    _validate_sparse_quadratic_terms(
        objective.get('quadratic'),
        len(variables),
        'Objective quadratic terms',
    )

    constraints = problem.get('constraints')
    if not isinstance(constraints, list):
        raise TypeError("problem['constraints'] must be a list.")
    constraint_names = []
    for position, constraint in enumerate(constraints):
        _validate_mapping(constraint, f'constraint {position}')
        name = constraint.get('name')
        family = constraint.get('family')
        if not isinstance(name, str) or not name:
            raise ValueError('Constraint names must be non-empty strings.')
        if not isinstance(family, str) or not family:
            raise ValueError('Constraint families must be non-empty strings.')
        constraint_names.append(name)
        _validate_sparse_linear_terms(
            constraint.get('linear'),
            len(variables),
            f"Constraint '{name}' terms",
        )
        lower = constraint.get('lower_bound')
        upper = constraint.get('upper_bound')
        if lower is None and upper is None:
            raise ValueError(f"Constraint '{name}' must define at least one bound.")
        if lower is not None:
            _validate_finite_number(lower, f"Constraint '{name}' lower bound")
        if upper is not None:
            _validate_finite_number(upper, f"Constraint '{name}' upper bound")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Constraint '{name}' has inconsistent bounds.")
    if len(constraint_names) != len(set(constraint_names)):
        raise ValueError('Constraint names must be unique.')

    fixed_values = problem.get('fixed_values')
    if not isinstance(fixed_values, list):
        raise TypeError("problem['fixed_values'] must be a list.")
    fixed_indices = []
    for fixed_value in fixed_values:
        _validate_mapping(fixed_value, 'fixed value')
        index = fixed_value.get('index')
        value = fixed_value.get('value')
        _validate_index(index, len(variables), 'Fixed variable index')
        if value not in {0, 1} or isinstance(value, bool):
            raise ValueError('Fixed variable values must be integer 0 or 1.')
        fixed_indices.append(index)
    if len(fixed_indices) != len(set(fixed_indices)):
        raise ValueError('A variable may only be fixed once.')


def _validate_qubo(problem):
    """Validate qubo."""
    _validate_mapping(problem, 'problem')
    if problem.get('schema') != 'qubo.v1':
        raise ValueError("problem schema must be 'qubo.v1'.")
    _validate_problem_id(problem)
    if problem.get('sense') != 'minimize':
        raise ValueError("qubo.v1 sense must be 'minimize'.")

    variable_count = problem.get('num_variables')
    if (
        not isinstance(variable_count, int)
        or isinstance(variable_count, bool)
        or variable_count < 0
    ):
        raise ValueError('num_variables must be a non-negative integer.')

    variable_names = problem.get('variable_names')
    if not isinstance(variable_names, list):
        raise TypeError('variable_names must be a list.')
    if len(variable_names) != variable_count:
        raise ValueError('variable_names length must equal num_variables.')
    if any(not isinstance(name, str) or not name for name in variable_names):
        raise ValueError('Variable names must be non-empty strings.')
    if len(variable_names) != len(set(variable_names)):
        raise ValueError('Variable names must be unique.')

    _validate_finite_number(problem.get('offset'), 'QUBO offset')
    _validate_sparse_quadratic_terms(
        problem.get('terms'),
        variable_count,
        'QUBO terms',
    )


def _evaluate_qubo(problem, sample):
    """Handle evaluate qubo."""
    if not isinstance(sample, (list, tuple)):
        sample = list(sample)
    if len(sample) != problem['num_variables']:
        raise ValueError('Sample length must equal num_variables.')
    if any(value not in {0, 1} for value in sample):
        raise ValueError('Samples must contain only binary values.')

    energy = float(problem['offset'])
    for left, right, coefficient in problem['terms']:
        energy += coefficient * sample[left] * sample[right]
    return float(energy)


def _validate_sparse_linear_terms(terms, variable_count, label):
    """Validate sparse linear terms."""
    if not isinstance(terms, list):
        raise TypeError(f'{label} must be a list.')
    seen = set()
    for term in terms:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
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
    """Validate sparse quadratic terms."""
    if not isinstance(terms, list):
        raise TypeError(f'{label} must be a list.')
    seen = set()
    for term in terms:
        if not isinstance(term, (list, tuple)) or len(term) != 3:
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


def _validate_mapping(value, label):
    """Validate mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping.')


def _validate_problem_id(problem):
    """Validate problem id."""
    problem_id = problem.get('problem_id')
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError('problem_id must be a non-empty string.')


def _validate_index(index, variable_count, label):
    """Validate index."""
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= variable_count
    ):
        raise ValueError(f'{label} is outside the variable range.')


def _validate_finite_number(value, label):
    """Validate finite number."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise TypeError(f'{label} must be a finite real number.')
