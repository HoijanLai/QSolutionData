import copy
import math
from collections.abc import Mapping
from fractions import Fraction
from functools import reduce
from math import gcd

from ..contracts.validation import _validate_cbqm, _validate_qubo


def compile_qubo(problem, config):
    """Compile ``cbqm.v1`` into the canonical minimization ``qubo.v1`` format.

    Args:
        problem: Mapping conforming to ``cbqm.v1`` with binary variables, a
            linear/quadratic objective, bounded linear constraints, and optional
            fixed values.
        config: Explicit compiler configuration. ``default_penalty`` is
            required. Optional fields control family/constraint overrides,
            integer-lattice precision, tolerances, inequality encoding, and
            fixed-value handling.

    Returns:
        ``(qubo, compilation_context)``. ``qubo`` conforms to ``qubo.v1`` and is
        ready for a solver. The context records original/free/fixed variables,
        generated slack variables, constraint normalization, penalties, and the
        objective-sense transformation needed for later decoding.

    Raises:
        TypeError: If model or configuration containers have invalid types.
        ValueError: If the model is invalid, configuration is inconsistent,
            numerical lattice conversion exceeds tolerance, or a constraint is
            provably infeasible after fixed-value substitution.
        NotImplementedError: If the requested compiler strategy is unavailable.

    Notes:
        Fixed variables are eliminated exactly. Equalities use squared
        penalties; inequalities use bounded binary slack. Penalty sufficiency
        is deliberately left to the caller rather than inferred.
    """
    _validate_cbqm(problem)
    resolved_config = _resolve_compiler_config(config)
    _validate_penalty_overrides(problem, resolved_config)

    if resolved_config['strategy'] == 'quadratic_penalty':
        state, context = _initialize_compilation(problem, resolved_config)
        for constraint_index, constraint in enumerate(problem['constraints']):
            constraint_context = _compile_constraint(
                state,
                constraint,
                constraint_index,
                context['fixed_values_by_index'],
                context['qubo_index_by_cbqm_index'],
                resolved_config,
            )
            context['constraints'].append(constraint_context)

        qubo = _build_qubo(problem, state, context, resolved_config)
        _finalize_context(context, qubo)
        _validate_qubo(qubo)
        return qubo, _public_context(context)

    raise NotImplementedError(
        f"Compiler strategy '{resolved_config['strategy']}' is not implemented."
    )


def _resolve_compiler_config(config):
    """Resolve compiler config."""
    if not isinstance(config, Mapping):
        raise TypeError('compiler config must be an explicit mapping.')

    allowed = {
        'strategy',
        'default_penalty',
        'penalty_by_family',
        'penalty_by_constraint',
        'inequality_encoding',
        'constraint_precision',
        'rounding_tolerance',
        'zero_tolerance',
        'fixed_value_strategy',
    }
    unknown = sorted(set(config).difference(allowed))
    if unknown:
        raise ValueError(f'Unknown compiler config fields: {unknown}')
    if 'default_penalty' not in config:
        raise ValueError("compiler config requires 'default_penalty'.")

    strategy = config.get('strategy', 'quadratic_penalty')
    inequality_encoding = config.get('inequality_encoding', 'binary_slack')
    fixed_value_strategy = config.get('fixed_value_strategy', 'eliminate')
    if not isinstance(strategy, str) or not strategy:
        raise ValueError('Compiler strategy must be a non-empty string.')
    if inequality_encoding != 'binary_slack':
        raise ValueError("inequality_encoding must be 'binary_slack'.")
    if fixed_value_strategy != 'eliminate':
        raise ValueError("fixed_value_strategy must be 'eliminate'.")

    default_penalty = _validate_penalty(
        config['default_penalty'],
        'default_penalty',
    )
    penalty_by_family = _validate_penalty_mapping(
        config.get('penalty_by_family', {}),
        'penalty_by_family',
    )
    penalty_by_constraint = _validate_penalty_mapping(
        config.get('penalty_by_constraint', {}),
        'penalty_by_constraint',
    )

    precision = config.get('constraint_precision', 8)
    if (
        not isinstance(precision, int)
        or isinstance(precision, bool)
        or not 0 <= precision <= 12
    ):
        raise ValueError('constraint_precision must be an integer from 0 to 12.')
    rounding_tolerance = config.get('rounding_tolerance', 1e-9)
    zero_tolerance = config.get('zero_tolerance', 1e-12)
    for name, value in (
        ('rounding_tolerance', rounding_tolerance),
        ('zero_tolerance', zero_tolerance),
    ):
        if not _is_finite_number(value) or value < 0:
            raise ValueError(f'{name} must be a finite non-negative number.')

    return {
        'strategy': strategy,
        'default_penalty': default_penalty,
        'penalty_by_family': penalty_by_family,
        'penalty_by_constraint': penalty_by_constraint,
        'inequality_encoding': inequality_encoding,
        'constraint_precision': precision,
        'rounding_tolerance': float(rounding_tolerance),
        'zero_tolerance': float(zero_tolerance),
        'fixed_value_strategy': fixed_value_strategy,
    }


def _validate_penalty(value, label):
    """Validate penalty."""
    if not _is_finite_number(value) or value <= 0:
        raise ValueError(f'{label} must be a finite positive number.')
    return float(value)


def _validate_penalty_mapping(mapping, label):
    """Validate penalty mapping."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f'{label} must be a mapping.')
    output = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f'{label} keys must be non-empty strings.')
        output[key] = _validate_penalty(value, f"{label}['{key}']")
    return output


def _validate_penalty_overrides(problem, config):
    """Validate penalty overrides."""
    constraint_names = {item['name'] for item in problem['constraints']}
    constraint_families = {item['family'] for item in problem['constraints']}
    unknown_names = sorted(
        set(config['penalty_by_constraint']).difference(constraint_names)
    )
    unknown_families = sorted(
        set(config['penalty_by_family']).difference(constraint_families)
    )
    if unknown_names:
        raise ValueError(f'Unknown constraint penalty overrides: {unknown_names}')
    if unknown_families:
        raise ValueError(f'Unknown family penalty overrides: {unknown_families}')


def _initialize_compilation(problem, config):
    """Initialize compilation."""
    fixed_values = {
        item['index']: item['value']
        for item in problem['fixed_values']
    }
    free_variables = [
        variable
        for variable in problem['variables']
        if variable['index'] not in fixed_values
    ]
    qubo_index_by_cbqm = {
        variable['index']: qubo_index
        for qubo_index, variable in enumerate(free_variables)
    }
    variable_names = [variable['name'] for variable in free_variables]
    if any(name.startswith('__slack__') for name in variable_names):
        raise ValueError(
            "CBQM variable names may not use the reserved '__slack__' prefix."
        )

    state = {
        'offset': 0.0,
        'offset_contributions': [],
        'exact_offset': Fraction(0),
        'terms': {},
        'term_contributions': {},
        'exact_terms': {},
        'variable_names': variable_names,
        'slack_variables': [],
    }
    _compile_objective(
        state,
        problem['objective'],
        fixed_values,
        qubo_index_by_cbqm,
    )

    context = {
        'schema': 'qubo-compilation-context.v1',
        'source_problem_id': problem['problem_id'],
        'source_variable_count': len(problem['variables']),
        'source_variable_names': [
            variable['name'] for variable in problem['variables']
        ],
        'objective_sense': problem['objective']['sense'],
        'objective_multiplier': (
            1.0 if problem['objective']['sense'] == 'minimize' else -1.0
        ),
        'fixed_values': copy.deepcopy(problem['fixed_values']),
        'fixed_values_by_index': fixed_values,
        'free_variables': [
            {
                'cbqm_index': variable['index'],
                'qubo_index': qubo_index_by_cbqm[variable['index']],
                'name': variable['name'],
            }
            for variable in free_variables
        ],
        'qubo_index_by_cbqm_index': qubo_index_by_cbqm,
        'slack_variables': state['slack_variables'],
        'constraints': [],
        'compiler_config': copy.deepcopy(config),
    }
    return state, context


def _compile_objective(state, objective, fixed_values, qubo_index_by_cbqm):
    """Compile objective."""
    multiplier = 1.0 if objective['sense'] == 'minimize' else -1.0
    _add_qubo_offset(
        state,
        multiplier * float(objective['offset']),
    )

    for cbqm_index, coefficient in objective['linear']:
        _compile_linear_monomial(
            state,
            cbqm_index,
            multiplier * coefficient,
            fixed_values,
            qubo_index_by_cbqm,
        )

    for left, right, coefficient in objective['quadratic']:
        _compile_quadratic_monomial(
            state,
            left,
            right,
            multiplier * coefficient,
            fixed_values,
            qubo_index_by_cbqm,
        )


def _compile_linear_monomial(
    state,
    cbqm_index,
    coefficient,
    fixed_values,
    qubo_index_by_cbqm,
):
    """Compile linear monomial."""
    if cbqm_index in fixed_values:
        _add_qubo_offset(
            state,
            coefficient * fixed_values[cbqm_index],
        )
        return
    qubo_index = qubo_index_by_cbqm[cbqm_index]
    _add_qubo_term(state, qubo_index, qubo_index, coefficient)


def _compile_quadratic_monomial(
    state,
    left,
    right,
    coefficient,
    fixed_values,
    qubo_index_by_cbqm,
):
    """Compile quadratic monomial."""
    if left == right:
        _compile_linear_monomial(
            state,
            left,
            coefficient,
            fixed_values,
            qubo_index_by_cbqm,
        )
        return

    left_fixed = left in fixed_values
    right_fixed = right in fixed_values
    if left_fixed and right_fixed:
        _add_qubo_offset(
            state,
            coefficient * fixed_values[left] * fixed_values[right],
        )
    elif left_fixed:
        if fixed_values[left]:
            qubo_right = qubo_index_by_cbqm[right]
            _add_qubo_term(
                state,
                qubo_right,
                qubo_right,
                coefficient,
            )
    elif right_fixed:
        if fixed_values[right]:
            qubo_left = qubo_index_by_cbqm[left]
            _add_qubo_term(
                state,
                qubo_left,
                qubo_left,
                coefficient,
            )
    else:
        _add_qubo_term(
            state,
            qubo_index_by_cbqm[left],
            qubo_index_by_cbqm[right],
            coefficient,
        )


def _compile_constraint(
    state,
    constraint,
    constraint_index,
    fixed_values,
    qubo_index_by_cbqm,
    config,
):
    """Compile constraint."""
    (
        terms,
        lower,
        upper,
        fixed_contribution,
        substitution,
    ) = _substitute_fixed_values(
        constraint,
        fixed_values,
        qubo_index_by_cbqm,
    )
    integer_terms, integer_lower, integer_upper, normalization = (
        _normalize_constraint_lattice(
            terms,
            lower,
            upper,
            config['constraint_precision'],
            config['rounding_tolerance'],
        )
    )
    penalty = _resolve_constraint_penalty(constraint, config)
    context = {
        'name': constraint['name'],
        'family': constraint['family'],
        'penalty': penalty,
        'fixed_contribution': fixed_contribution,
        'fixed_substitution': substitution,
        'normalization': normalization,
        'encodings': [],
    }

    if integer_lower is not None and integer_upper is not None:
        if integer_lower == integer_upper:
            context['encodings'].append(
                _compile_constraint_side(
                    state,
                    integer_terms,
                    integer_lower,
                    'equal',
                    penalty,
                    constraint_index,
                )
            )
            return context

        context['encodings'].append(
            _compile_constraint_side(
                state,
                integer_terms,
                integer_lower,
                'lower',
                penalty,
                constraint_index,
            )
        )
        context['encodings'].append(
            _compile_constraint_side(
                state,
                integer_terms,
                integer_upper,
                'upper',
                penalty,
                constraint_index,
            )
        )
        return context

    if integer_lower is not None:
        context['encodings'].append(
            _compile_constraint_side(
                state,
                integer_terms,
                integer_lower,
                'lower',
                penalty,
                constraint_index,
            )
        )
    if integer_upper is not None:
        context['encodings'].append(
            _compile_constraint_side(
                state,
                integer_terms,
                integer_upper,
                'upper',
                penalty,
                constraint_index,
            )
        )
    return context


def _substitute_fixed_values(constraint, fixed_values, qubo_index_by_cbqm):
    """Substitute fixed values."""
    fixed_contributions = []
    term_contributions = {}
    for cbqm_index, coefficient in constraint['linear']:
        if cbqm_index in fixed_values:
            fixed_contributions.append(
                float(coefficient) * fixed_values[cbqm_index]
            )
        else:
            qubo_index = qubo_index_by_cbqm[cbqm_index]
            term_contributions.setdefault(qubo_index, []).append(
                float(coefficient)
            )

    fixed_contribution = math.fsum(fixed_contributions)
    exact_fixed_contribution = sum(
        (
            Fraction.from_float(value)
            for value in fixed_contributions
        ),
        start=Fraction(0),
    )
    terms = {
        index: math.fsum(contributions)
        for index, contributions in term_contributions.items()
    }
    lower = constraint.get('lower_bound')
    upper = constraint.get('upper_bound')
    bounds_exact = True
    if lower is not None:
        lower, lower_exact = _subtract_fixed_contribution(
            lower,
            fixed_contribution,
            exact_fixed_contribution,
        )
        bounds_exact = bounds_exact and lower_exact
    if upper is not None:
        upper, upper_exact = _subtract_fixed_contribution(
            upper,
            fixed_contribution,
            exact_fixed_contribution,
        )
        bounds_exact = bounds_exact and upper_exact

    fixed_contribution_exact = (
        Fraction.from_float(float(fixed_contribution))
        == exact_fixed_contribution
    )
    substitution = {
        'fixed_contribution_exact': fixed_contribution_exact,
        'adjusted_bounds_exact': bounds_exact,
        'exact_reconstruction': (
            fixed_contribution_exact and bounds_exact
        ),
    }
    return (
        terms,
        lower,
        upper,
        float(fixed_contribution),
        substitution,
    )


def _subtract_fixed_contribution(
    bound,
    fixed_contribution,
    exact_fixed_contribution,
):
    """Subtract one bound and report whether the float retained exact algebra."""
    source_bound = float(bound)
    adjusted = source_bound - fixed_contribution
    exact_adjusted = (
        Fraction.from_float(source_bound)
        - exact_fixed_contribution
    )
    return (
        adjusted,
        Fraction.from_float(float(adjusted)) == exact_adjusted,
    )


def _normalize_constraint_lattice(
    terms,
    lower,
    upper,
    precision,
    rounding_tolerance,
):
    """Normalize constraint lattice."""
    decimal_scale = 10 ** precision
    scaled_terms = {
        index: _scale_lattice_value(
            coefficient,
            decimal_scale,
            rounding_tolerance,
        )
        for index, coefficient in terms.items()
    }
    scaled_lower = (
        None
        if lower is None
        else _scale_lattice_value(lower, decimal_scale, rounding_tolerance)
    )
    scaled_upper = (
        None
        if upper is None
        else _scale_lattice_value(upper, decimal_scale, rounding_tolerance)
    )
    exact_reconstruction, maximum_reconstruction_error = (
        _measure_lattice_reconstruction(
            terms,
            lower,
            upper,
            scaled_terms,
            scaled_lower,
            scaled_upper,
            decimal_scale,
        )
    )

    nonzero_values = [abs(value) for value in scaled_terms.values() if value]
    if scaled_lower:
        nonzero_values.append(abs(scaled_lower))
    if scaled_upper:
        nonzero_values.append(abs(scaled_upper))
    divisor = reduce(gcd, nonzero_values) if nonzero_values else 1

    integer_terms = {
        index: value // divisor
        for index, value in scaled_terms.items()
        if value
    }
    integer_lower = None if scaled_lower is None else scaled_lower // divisor
    integer_upper = None if scaled_upper is None else scaled_upper // divisor
    normalization = {
        'decimal_scale': decimal_scale,
        'integer_divisor': divisor,
        'lattice_unit': divisor / decimal_scale,
        'penalty_domain': 'integer_lattice',
        # ``rounding_tolerance`` answers whether an approximation is acceptable
        # for compilation.  Exact-result propagation needs the stricter fact:
        # dividing the stored integer back by the scale must reproduce every
        # source float exactly.  Keep both the verdict and its diagnostic error.
        'exact_reconstruction': exact_reconstruction,
        'max_abs_reconstruction_error': maximum_reconstruction_error,
    }
    return integer_terms, integer_lower, integer_upper, normalization


def _measure_lattice_reconstruction(
    terms,
    lower,
    upper,
    scaled_terms,
    scaled_lower,
    scaled_upper,
    decimal_scale,
):
    """Describe whether integer-lattice conversion changed any source value."""
    source_and_scaled_values = [
        (float(coefficient), scaled_terms[index])
        for index, coefficient in terms.items()
    ]
    if lower is not None:
        source_and_scaled_values.append((float(lower), scaled_lower))
    if upper is not None:
        source_and_scaled_values.append((float(upper), scaled_upper))

    reconstruction_errors = [
        abs(source - (scaled / decimal_scale))
        for source, scaled in source_and_scaled_values
    ]
    maximum_error = max(reconstruction_errors, default=0.0)
    return maximum_error == 0.0, float(maximum_error)


def _scale_lattice_value(value, decimal_scale, tolerance):
    """Convert one real constraint value to the configured integer scale."""
    scaled = round(float(value) * decimal_scale)
    restored = scaled / decimal_scale
    if abs(float(value) - restored) > tolerance:
        raise ValueError(
            f'Constraint value {value!r} exceeds the configured rounding tolerance.'
        )
    return int(scaled)


def _resolve_constraint_penalty(constraint, config):
    """Resolve constraint penalty."""
    if constraint['name'] in config['penalty_by_constraint']:
        return config['penalty_by_constraint'][constraint['name']]
    if constraint['family'] in config['penalty_by_family']:
        return config['penalty_by_family'][constraint['family']]
    return config['default_penalty']


def _compile_constraint_side(
    state,
    terms,
    rhs,
    side,
    penalty,
    constraint_index,
):
    """Compile constraint side."""
    minimum, maximum = _binary_linear_bounds(terms)
    encoding = {
        'side': side,
        'integer_rhs': rhs,
        'integer_terms': [
            [index, coefficient]
            for index, coefficient in sorted(terms.items())
        ],
        'slack_variables': [],
    }

    if side == 'equal':
        if rhs < minimum or rhs > maximum:
            raise ValueError('Equality constraint is infeasible after substitution.')
        if not terms and rhs == 0:
            encoding['status'] = 'redundant'
            return encoding
        equation_terms = dict(terms)
    elif side == 'upper':
        if rhs < minimum:
            raise ValueError('Upper-bound constraint is infeasible after substitution.')
        if rhs >= maximum:
            encoding['status'] = 'redundant'
            return encoding
        equation_terms = dict(terms)
        slack_range = rhs - minimum
        encoding['slack_variables'] = _add_slack_variables(
            state,
            equation_terms,
            slack_range,
            1,
            constraint_index,
            side,
        )
    elif side == 'lower':
        if rhs > maximum:
            raise ValueError('Lower-bound constraint is infeasible after substitution.')
        if rhs <= minimum:
            encoding['status'] = 'redundant'
            return encoding
        equation_terms = dict(terms)
        slack_range = maximum - rhs
        encoding['slack_variables'] = _add_slack_variables(
            state,
            equation_terms,
            slack_range,
            -1,
            constraint_index,
            side,
        )
    else:
        raise ValueError(f"Unknown constraint side '{side}'.")

    _add_squared_penalty(state, equation_terms, rhs, penalty)
    encoding['status'] = 'encoded'
    return encoding


def _binary_linear_bounds(terms):
    """Compute independent binary lower and upper bounds for a linear form."""
    minimum = sum(min(0, coefficient) for coefficient in terms.values())
    maximum = sum(max(0, coefficient) for coefficient in terms.values())
    return minimum, maximum


def _add_slack_variables(
    state,
    equation_terms,
    slack_range,
    sign,
    constraint_index,
    side,
):
    """Add slack variables."""
    slack_variables = []
    for bit_index, weight in enumerate(_bounded_binary_weights(slack_range)):
        qubo_index = len(state['variable_names'])
        name = f'__slack__{constraint_index}_{side}_{bit_index}'
        if name in state['variable_names']:
            raise ValueError(f"Generated slack variable name collision: '{name}'.")
        state['variable_names'].append(name)
        equation_terms[qubo_index] = sign * weight
        metadata = {
            'qubo_index': qubo_index,
            'name': name,
            'constraint_index': constraint_index,
            'side': side,
            'integer_weight': weight,
            'equation_sign': sign,
        }
        state['slack_variables'].append(metadata)
        slack_variables.append(copy.deepcopy(metadata))
    return slack_variables


def _bounded_binary_weights(maximum):
    """Construct bounded binary weights spanning a slack range."""
    if maximum < 0:
        raise ValueError('Slack range must be non-negative.')
    weights = []
    represented = 0
    next_weight = 1
    while represented + next_weight <= maximum:
        weights.append(next_weight)
        represented += next_weight
        next_weight *= 2
    remainder = maximum - represented
    if remainder:
        weights.append(remainder)
    return weights


def _add_squared_penalty(state, equation_terms, rhs, penalty):
    """Expand one squared linear residual into QUBO coefficients."""
    exact_penalty = Fraction.from_float(float(penalty))
    offset_multiplier = rhs * rhs
    _add_qubo_offset(
        state,
        penalty * offset_multiplier,
        exact_coefficient=exact_penalty * offset_multiplier,
    )
    ordered_terms = sorted(equation_terms.items())
    for index, coefficient in ordered_terms:
        diagonal_multiplier = (
            coefficient * coefficient
            - 2 * rhs * coefficient
        )
        diagonal = penalty * diagonal_multiplier
        _add_qubo_term(
            state,
            index,
            index,
            diagonal,
            exact_coefficient=exact_penalty * diagonal_multiplier,
        )
    for position, (left, left_coefficient) in enumerate(ordered_terms):
        for right, right_coefficient in ordered_terms[position + 1:]:
            pair_multiplier = 2 * left_coefficient * right_coefficient
            _add_qubo_term(
                state,
                left,
                right,
                penalty * pair_multiplier,
                exact_coefficient=exact_penalty * pair_multiplier,
            )


def _add_qubo_offset(state, coefficient, *, exact_coefficient=None):
    """Accumulate the offset with a rational reference for exactness proof."""
    value = float(coefficient)
    exact_value = (
        Fraction.from_float(value)
        if exact_coefficient is None
        else exact_coefficient
    )
    state['offset_contributions'].append(value)
    state['exact_offset'] += exact_value
    state['offset'] = math.fsum(state['offset_contributions'])


def _add_qubo_term(
    state,
    left,
    right,
    coefficient,
    *,
    exact_coefficient=None,
):
    """Accumulate one term and retain its exact rational reference."""
    pair = (min(left, right), max(left, right))
    value = float(coefficient)
    exact_value = (
        Fraction.from_float(value)
        if exact_coefficient is None
        else exact_coefficient
    )
    state['term_contributions'].setdefault(pair, []).append(value)
    state['exact_terms'][pair] = (
        state['exact_terms'].get(pair, Fraction(0))
        + exact_value
    )
    state['terms'][pair] = math.fsum(
        state['term_contributions'][pair]
    )


def _build_qubo(problem, state, context, config):
    """Build qubo."""
    context['arithmetic'] = _measure_qubo_arithmetic(state)
    kept_terms, dropped_terms = _partition_qubo_terms(
        state['terms'],
        config['zero_tolerance'],
    )
    # Retain the filtered terms in the compilation context.  A tiny coefficient
    # may be harmless for a heuristic run, but discarding any non-zero term
    # invalidates a proof that the compiled objective is exactly equivalent.
    context['dropped_qubo_terms'] = dropped_terms
    terms = [
        [left, right, float(coefficient)]
        for (left, right), coefficient in kept_terms
    ]
    return {
        'schema': 'qubo.v1',
        'problem_id': problem['problem_id'],
        'sense': 'minimize',
        'num_variables': len(state['variable_names']),
        'variable_names': copy.deepcopy(state['variable_names']),
        'offset': float(state['offset']),
        'terms': terms,
        'metadata': {
            'source_schema': 'cbqm.v1',
            'compiler': 'quadratic_penalty.v1',
            'original_variable_count': len(problem['variables']),
            'free_variable_count': len(context['free_variables']),
            'slack_variable_count': len(state['slack_variables']),
            'compiler_config': copy.deepcopy(config),
        },
    }


def _measure_qubo_arithmetic(state):
    """Report whether every emitted float equals its exact algebraic sum."""
    inexact_locations = []
    if Fraction.from_float(float(state['offset'])) != state['exact_offset']:
        inexact_locations.append('offset')

    for left, right in sorted(state['exact_terms']):
        coefficient = state['terms'][(left, right)]
        if (
            Fraction.from_float(float(coefficient))
            != state['exact_terms'][(left, right)]
        ):
            inexact_locations.append(f'term:{left},{right}')

    return {
        'exact_accumulation': not inexact_locations,
        'inexact_locations': inexact_locations,
        'summation': 'math.fsum-with-rational-reference',
    }


def _partition_qubo_terms(terms, zero_tolerance):
    """Split accumulated terms into emitted and non-zero tolerance drops."""
    kept = []
    dropped = []
    for (left, right), coefficient in sorted(terms.items()):
        item = [left, right, float(coefficient)]
        if abs(coefficient) > zero_tolerance:
            kept.append(((left, right), coefficient))
        elif coefficient != 0.0:
            dropped.append(item)
    return kept, dropped


def _finalize_context(context, qubo):
    """Finalize context."""
    context['qubo_variable_count'] = qubo['num_variables']
    context['qubo_variable_names'] = copy.deepcopy(qubo['variable_names'])
    context['equivalence'] = _build_equivalence_certificate(context)


def _build_equivalence_certificate(context):
    """Build the evidence gate used for safe exact-result propagation.

    This certificate deliberately does *not* claim that the configured finite
    penalties force every QUBO optimum to be feasible.  It only certifies the
    exact algebraic correspondence needed by a caller that separately verifies
    the projected sample's feasibility and zero-penalty energy relationship.
    """
    constraint_lattice_exact = all(
        constraint['normalization']['exact_reconstruction']
        and constraint['fixed_substitution']['exact_reconstruction']
        for constraint in context['constraints']
    )
    no_nonzero_terms_dropped = not context['dropped_qubo_terms']
    arithmetic_exact = context['arithmetic']['exact_accumulation']
    return {
        'schema': 'qubo-compilation-equivalence.v1',
        'constraint_lattice_exact': constraint_lattice_exact,
        'no_nonzero_terms_dropped': no_nonzero_terms_dropped,
        'arithmetic_exact': arithmetic_exact,
        'feasible_set_preserved': constraint_lattice_exact,
        'objective_mapping_preserved': (
            no_nonzero_terms_dropped and arithmetic_exact
        ),
        'exact_projection_certified': (
            constraint_lattice_exact
            and no_nonzero_terms_dropped
            and arithmetic_exact
        ),
        'scope': (
            'Certifies exact source-to-QUBO algebra only; callers must also '
            'verify global solver optimality, projected feasibility, and the '
            'zero-penalty objective-energy relationship.'
        ),
    }


def _public_context(context):
    """Remove internal lookup tables from the compilation context."""
    output = copy.deepcopy(context)
    output.pop('fixed_values_by_index')
    output.pop('qubo_index_by_cbqm_index')
    return output


def _is_finite_number(value):
    """Return whether a value is a finite, non-boolean real number."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
