import copy
from collections.abc import Mapping

from ..contracts.validation import _validate_cbqm


_FIXED_PREFIX = '__cbqm_fixed__'
_LOWER_SUFFIX = '__cbqm_lower'
_UPPER_SUFFIX = '__cbqm_upper'


class QiskitQuadraticProgramAdapter:
    """Translate between ``cbqm.v1`` and Qiskit ``QuadraticProgram`` objects.

    Qiskit is an optional dependency and is imported lazily. Conversion context
    preserves metadata and enables split range constraints and encoded fixed
    values to be reconstructed on the return path.
    """

    def to_qiskit(self, problem, quadratic_program_class=None):
        """Convert a neutral constrained model to Qiskit representation.

        Args:
            problem: Mapping conforming to ``cbqm.v1``.
            quadratic_program_class: Optional compatible class used for
                dependency injection. When omitted, Qiskit's installed
                ``QuadraticProgram`` class is loaded.

        Returns:
            ``(quadratic_program, context)``. The context records original
            variable metadata, constraint families, range splits, fixed values,
            and model metadata for loss-aware reconstruction.

        Raises:
            ImportError: If Qiskit is unavailable and no class is injected.
            TypeError: If the input model has invalid container types.
            ValueError: If the model is invalid or generated constraint names
                collide.
        """
        _validate_cbqm(problem)
        program_class = _resolve_quadratic_program_class(quadratic_program_class)
        constraint_specs, context = _build_qiskit_constraint_specs(problem)

        quadratic_program = program_class(problem['problem_id'])
        variable_names = [variable['name'] for variable in problem['variables']]
        for variable_name in variable_names:
            quadratic_program.binary_var(name=variable_name)

        objective = problem['objective']
        linear = _linear_terms_to_name_dict(objective['linear'], variable_names)
        quadratic = _quadratic_terms_to_name_dict(
            objective['quadratic'],
            variable_names,
        )
        objective_builder = getattr(quadratic_program, objective['sense'])
        objective_builder(
            constant=float(objective['offset']),
            linear=linear,
            quadratic=quadratic,
        )

        for spec in constraint_specs:
            quadratic_program.linear_constraint(
                linear=_linear_terms_to_name_dict(spec['linear'], variable_names),
                sense=spec['sense'],
                rhs=float(spec['rhs']),
                name=spec['name'],
            )

        return quadratic_program, context

    def from_qiskit(self, quadratic_program, context=None):
        """Convert a Qiskit quadratic program into ``cbqm.v1``.

        Args:
            quadratic_program: Qiskit-compatible object containing only binary
                variables, a quadratic objective, and linear constraints.
            context: Optional context returned by :meth:`to_qiskit`. Supplying it
                restores original ranges, families, fixed values, and metadata.

        Returns:
            A validated ``cbqm.v1`` dictionary. Without context, each native
            linear constraint is imported independently with family
            ``qiskit_linear``.

        Raises:
            TypeError: If the object does not expose the required interface.
            ValueError: If variables, constraints, expressions, or context are
                unsupported or inconsistent.
        """
        _validate_qiskit_program(quadratic_program)
        if context is not None:
            _validate_qiskit_context(context, quadratic_program)

        variable_names = [variable.name for variable in quadratic_program.variables]
        variables = _build_cbqm_variables(quadratic_program, context)
        objective = _build_cbqm_objective(quadratic_program, variable_names)
        constraints, fixed_values = _build_cbqm_constraints(
            quadratic_program,
            variable_names,
            context,
        )

        problem_id = getattr(quadratic_program, 'name', '')
        if not problem_id and context is not None:
            problem_id = context['problem_id']
        if not problem_id:
            problem_id = 'qiskit-problem'

        metadata = {'source': 'qiskit_optimization'}
        if context is not None:
            metadata.update(copy.deepcopy(context.get('metadata', {})))

        problem = {
            'schema': 'cbqm.v1',
            'problem_id': problem_id,
            'variables': variables,
            'objective': objective,
            'constraints': constraints,
            'fixed_values': fixed_values,
            'metadata': metadata,
        }
        _validate_cbqm(problem)
        return problem


def _resolve_quadratic_program_class(quadratic_program_class):
    """Resolve quadratic program class."""
    if quadratic_program_class is not None:
        return quadratic_program_class
    try:
        from qiskit_optimization import QuadraticProgram
    except ImportError as error:
        raise ImportError(
            'Qiskit adapter requires qiskit-optimization. Install it or pass '
            'quadratic_program_class for dependency injection.'
        ) from error
    return QuadraticProgram


def _build_qiskit_constraint_specs(problem):
    """Build qiskit constraint specs."""
    specs = []
    constraint_context = []
    native_names = set()

    for constraint in problem['constraints']:
        lower = constraint.get('lower_bound')
        upper = constraint.get('upper_bound')
        generated = []

        if lower is not None and upper is not None and lower == upper:
            generated.append(
                _make_qiskit_constraint_spec(
                    constraint['name'],
                    constraint['linear'],
                    '==',
                    lower,
                    'equal',
                )
            )
        elif lower is not None and upper is not None:
            generated.extend(
                [
                    _make_qiskit_constraint_spec(
                        f"{constraint['name']}{_LOWER_SUFFIX}",
                        constraint['linear'],
                        '>=',
                        lower,
                        'lower',
                    ),
                    _make_qiskit_constraint_spec(
                        f"{constraint['name']}{_UPPER_SUFFIX}",
                        constraint['linear'],
                        '<=',
                        upper,
                        'upper',
                    ),
                ]
            )
        elif lower is not None:
            generated.append(
                _make_qiskit_constraint_spec(
                    constraint['name'],
                    constraint['linear'],
                    '>=',
                    lower,
                    'lower',
                )
            )
        else:
            generated.append(
                _make_qiskit_constraint_spec(
                    constraint['name'],
                    constraint['linear'],
                    '<=',
                    upper,
                    'upper',
                )
            )

        _register_native_names(generated, native_names)
        specs.extend(generated)
        constraint_context.append(
            {
                'name': constraint['name'],
                'family': constraint['family'],
                'metadata': copy.deepcopy(constraint.get('metadata', {})),
                'native_constraints': [
                    {'name': spec['name'], 'bound': spec['bound']}
                    for spec in generated
                ],
            }
        )

    fixed_context = []
    for fixed_value in problem['fixed_values']:
        native_name = f"{_FIXED_PREFIX}{fixed_value['index']}"
        spec = _make_qiskit_constraint_spec(
            native_name,
            [[fixed_value['index'], 1.0]],
            '==',
            fixed_value['value'],
            'fixed',
        )
        _register_native_names([spec], native_names)
        specs.append(spec)
        fixed_context.append(
            {'index': fixed_value['index'], 'native_name': native_name}
        )

    context = {
        'schema': 'qiskit-qp-context.v1',
        'problem_id': problem['problem_id'],
        'variables': [
            {
                'name': variable['name'],
                'kind': variable.get('kind'),
                'metadata': copy.deepcopy(variable.get('metadata', {})),
            }
            for variable in problem['variables']
        ],
        'constraints': constraint_context,
        'fixed_values': fixed_context,
        'metadata': copy.deepcopy(problem.get('metadata', {})),
    }
    return specs, context


def _make_qiskit_constraint_spec(name, linear, sense, rhs, bound):
    """Create qiskit constraint spec."""
    return {
        'name': name,
        'linear': copy.deepcopy(linear),
        'sense': sense,
        'rhs': rhs,
        'bound': bound,
    }


def _register_native_names(specs, native_names):
    """Register native names."""
    for spec in specs:
        if spec['name'] in native_names:
            raise ValueError(
                f"Qiskit constraint name collision for '{spec['name']}'."
            )
        native_names.add(spec['name'])


def _linear_terms_to_name_dict(terms, variable_names):
    """Handle linear terms to name dict."""
    return {variable_names[index]: coefficient for index, coefficient in terms}


def _quadratic_terms_to_name_dict(terms, variable_names):
    """Handle quadratic terms to name dict."""
    return {
        (variable_names[left], variable_names[right]): coefficient
        for left, right, coefficient in terms
    }


def _validate_qiskit_program(quadratic_program):
    """Validate qiskit program."""
    required = ('variables', 'objective', 'linear_constraints')
    missing = [name for name in required if not hasattr(quadratic_program, name)]
    if missing:
        raise TypeError(
            f'quadratic_program is missing required attributes: {missing}'
        )
    quadratic_constraints = getattr(quadratic_program, 'quadratic_constraints', [])
    if quadratic_constraints:
        raise ValueError('cbqm.v1 does not support quadratic constraints.')
    for variable in quadratic_program.variables:
        if _enum_name(variable.vartype) != 'BINARY':
            raise ValueError('cbqm.v1 only supports Qiskit binary variables.')


def _validate_qiskit_context(context, quadratic_program):
    """Validate qiskit context."""
    if not isinstance(context, Mapping):
        raise TypeError('Qiskit adapter context must be a mapping.')
    if context.get('schema') != 'qiskit-qp-context.v1':
        raise ValueError("Qiskit adapter context schema must be 'qiskit-qp-context.v1'.")
    current_names = [variable.name for variable in quadratic_program.variables]
    context_names = [variable['name'] for variable in context.get('variables', [])]
    if current_names != context_names:
        raise ValueError('Qiskit variables no longer match the adapter context.')


def _build_cbqm_variables(quadratic_program, context):
    """Build cbqm variables."""
    context_variables = context['variables'] if context is not None else None
    variables = []
    for index, variable in enumerate(quadratic_program.variables):
        output = {
            'index': index,
            'name': variable.name,
            'vartype': 'BINARY',
        }
        if context_variables is not None:
            source = context_variables[index]
            if source.get('kind') is not None:
                output['kind'] = source['kind']
            if source.get('metadata'):
                output['metadata'] = copy.deepcopy(source['metadata'])
        variables.append(output)
    return variables


def _build_cbqm_objective(quadratic_program, variable_names):
    """Build cbqm objective."""
    objective = quadratic_program.objective
    sense_name = _enum_name(objective.sense)
    if sense_name == 'MINIMIZE':
        sense = 'minimize'
    elif sense_name == 'MAXIMIZE':
        sense = 'maximize'
    else:
        raise ValueError(f"Unsupported Qiskit objective sense '{sense_name}'.")

    return {
        'sense': sense,
        'offset': float(objective.constant),
        'linear': _read_qiskit_linear_expression(
            objective.linear,
            variable_names,
        ),
        'quadratic': _read_qiskit_quadratic_expression(
            objective.quadratic,
            variable_names,
        ),
    }


def _build_cbqm_constraints(quadratic_program, variable_names, context):
    """Build cbqm constraints."""
    native_constraints = {
        constraint.name: constraint
        for constraint in quadratic_program.linear_constraints
    }
    consumed = set()
    constraints = []
    fixed_values = []

    if context is not None:
        for fixed_context in context.get('fixed_values', []):
            native_name = fixed_context['native_name']
            native = _get_native_constraint(native_constraints, native_name)
            fixed_values.append(
                _read_fixed_constraint(native, variable_names, fixed_context['index'])
            )
            consumed.add(native_name)

        for constraint_context in context.get('constraints', []):
            constraint, native_names = _restore_context_constraint(
                native_constraints,
                variable_names,
                constraint_context,
            )
            constraints.append(constraint)
            consumed.update(native_names)
    else:
        for name, native in native_constraints.items():
            if name.startswith(_FIXED_PREFIX):
                index_text = name[len(_FIXED_PREFIX):]
                if not index_text.isdigit():
                    raise ValueError(f"Invalid encoded fixed constraint '{name}'.")
                fixed_values.append(
                    _read_fixed_constraint(native, variable_names, int(index_text))
                )
                consumed.add(name)

    for name, native in native_constraints.items():
        if name in consumed:
            continue
        constraints.append(
            _convert_qiskit_constraint(native, variable_names, 'qiskit_linear')
        )

    fixed_values.sort(key=lambda item: item['index'])
    return constraints, fixed_values


def _restore_context_constraint(native_constraints, variable_names, context):
    """Restore context constraint."""
    lower = None
    upper = None
    reference_terms = None
    native_names = []

    for native_context in context['native_constraints']:
        native_name = native_context['name']
        native = _get_native_constraint(native_constraints, native_name)
        terms, sense, rhs = _read_qiskit_constraint(native, variable_names)
        if reference_terms is None:
            reference_terms = terms
        elif reference_terms != terms:
            raise ValueError(
                f"Split Qiskit constraint '{context['name']}' has mismatched terms."
            )

        bound = native_context['bound']
        if bound == 'equal':
            if sense != 'EQ':
                raise ValueError(f"Constraint '{native_name}' is no longer equality.")
            lower = rhs
            upper = rhs
        elif bound == 'lower':
            if sense != 'GE':
                raise ValueError(f"Constraint '{native_name}' is no longer a lower bound.")
            lower = rhs
        elif bound == 'upper':
            if sense != 'LE':
                raise ValueError(f"Constraint '{native_name}' is no longer an upper bound.")
            upper = rhs
        else:
            raise ValueError(f"Unknown Qiskit context bound '{bound}'.")
        native_names.append(native_name)

    output = {
        'name': context['name'],
        'family': context['family'],
        'linear': reference_terms or [],
    }
    if lower is not None:
        output['lower_bound'] = lower
    if upper is not None:
        output['upper_bound'] = upper
    if context.get('metadata'):
        output['metadata'] = copy.deepcopy(context['metadata'])
    return output, native_names


def _convert_qiskit_constraint(native, variable_names, family):
    """Convert qiskit constraint."""
    terms, sense, rhs = _read_qiskit_constraint(native, variable_names)
    output = {
        'name': native.name,
        'family': family,
        'linear': terms,
    }
    if sense == 'EQ':
        output['lower_bound'] = rhs
        output['upper_bound'] = rhs
    elif sense == 'GE':
        output['lower_bound'] = rhs
    elif sense == 'LE':
        output['upper_bound'] = rhs
    else:
        raise ValueError(f"Unsupported Qiskit constraint sense '{sense}'.")
    return output


def _read_fixed_constraint(native, variable_names, expected_index):
    """Read fixed constraint."""
    terms, sense, rhs = _read_qiskit_constraint(native, variable_names)
    if sense != 'EQ' or terms != [[expected_index, 1.0]] or rhs not in {0.0, 1.0}:
        raise ValueError(f"Encoded fixed constraint '{native.name}' was modified.")
    return {'index': expected_index, 'value': int(rhs)}


def _read_qiskit_constraint(native, variable_names):
    """Read qiskit constraint."""
    return (
        _read_qiskit_linear_expression(native.linear, variable_names),
        _enum_name(native.sense),
        float(native.rhs),
    )


def _read_qiskit_linear_expression(expression, variable_names):
    """Read qiskit linear expression."""
    coefficients = {}
    for key, coefficient in _expression_to_dict(expression).items():
        index = _qiskit_key_to_index(key, variable_names)
        coefficients[index] = coefficients.get(index, 0.0) + float(coefficient)
    return [
        [index, coefficient]
        for index, coefficient in sorted(coefficients.items())
        if coefficient != 0
    ]


def _read_qiskit_quadratic_expression(expression, variable_names):
    """Read qiskit quadratic expression."""
    coefficients = {}
    for key, coefficient in _expression_to_dict(expression).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError('Qiskit quadratic expression has an invalid key.')
        left = _qiskit_key_to_index(key[0], variable_names)
        right = _qiskit_key_to_index(key[1], variable_names)
        pair = (min(left, right), max(left, right))
        coefficients[pair] = coefficients.get(pair, 0.0) + float(coefficient)
    return [
        [left, right, coefficient]
        for (left, right), coefficient in sorted(coefficients.items())
        if coefficient != 0
    ]


def _expression_to_dict(expression):
    """Extract a coefficient dictionary from a Qiskit expression."""
    try:
        return expression.to_dict(use_name=False)
    except TypeError:
        return expression.to_dict()


def _qiskit_key_to_index(key, variable_names):
    """Resolve a Qiskit expression key to a neutral variable index."""
    if isinstance(key, int) and not isinstance(key, bool):
        if 0 <= key < len(variable_names):
            return key
        raise ValueError('Qiskit expression index is outside the variable range.')
    if isinstance(key, str):
        try:
            return variable_names.index(key)
        except ValueError as error:
            raise ValueError(f"Unknown Qiskit variable '{key}'.") from error
    raise ValueError(f"Unsupported Qiskit expression key '{key}'.")


def _get_native_constraint(native_constraints, name):
    """Return a named native constraint or raise a contextual error."""
    try:
        return native_constraints[name]
    except KeyError as error:
        raise ValueError(f"Qiskit constraint '{name}' is missing.") from error


def _enum_name(value):
    """Normalize a native enum-like value to an uppercase member name."""
    name = getattr(value, 'name', None)
    if name is not None:
        return str(name).upper()
    text = str(value).upper()
    return text.rsplit('.', 1)[-1]
