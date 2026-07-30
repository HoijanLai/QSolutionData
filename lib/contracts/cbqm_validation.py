"""Runtime validation and canonical evaluation for ``cbqm-result.v1``."""

from fractions import Fraction

from .validation import (
    _fraction_to_json_number,
    _validate_binary_sample,
    _validate_finite_number,
    _validate_json_document,
    _validate_json_object,
    _validate_mapping,
    _validate_non_negative_finite_number,
    _validate_non_negative_integer,
    _validate_object_fields,
    _validate_solver_identity,
    validate_cbqm,
)


_RESULT_REQUIRED_FIELDS = frozenset(
    {
        'schema',
        'problem_id',
        'solver',
        'status',
        'best_sample',
        'best_objective',
        'feasibility',
        'runtime_seconds',
    }
)
_RESULT_OPTIONAL_FIELDS = frozenset(
    {
        'bounds',
        'proof',
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
_FEASIBILITY_REQUIRED_FIELDS = frozenset(
    {'feasible', 'violated_count', 'max_violation', 'violations'}
)
_VIOLATION_REQUIRED_FIELDS = frozenset(
    {'constraint_name', 'activity', 'magnitude'}
)
_VIOLATION_OPTIONAL_FIELDS = frozenset({'lower_bound', 'upper_bound'})
_BOUNDS_FIELDS = frozenset(
    {'primal_bound', 'dual_bound', 'absolute_gap', 'relative_gap'}
)
_PROOF_REQUIRED_FIELDS = frozenset(
    {'claim', 'kind', 'producer', 'independently_verified'}
)
_PROOF_OPTIONAL_FIELDS = frozenset({'details'})
_PROOF_CLAIMS = frozenset({'optimality', 'infeasibility', 'bound'})
_TRACE_REQUIRED_FIELDS = frozenset({'step', 'time_seconds'})
_TRACE_OPTIONAL_FIELDS = frozenset(
    {'sample', 'objective', 'feasible', 'metadata'}
)


def evaluate_cbqm_objective(problem, sample):
    """Return the original CBQM objective in canonical JSON-number form."""
    validate_cbqm(problem)
    _validate_binary_sample(sample, len(problem['variables']), 'sample')
    return _fraction_to_json_number(
        _evaluate_cbqm_objective_exact(problem, sample),
        'CBQM objective',
    )


def evaluate_cbqm_feasibility(problem, sample):
    """Return exact fixed-value and constraint violations for one sample."""
    validate_cbqm(problem)
    _validate_binary_sample(sample, len(problem['variables']), 'sample')
    return _evaluate_cbqm_feasibility(problem, sample)


def validate_cbqm_result(problem, result):
    """Validate a native CBQM result and recompute all candidate semantics."""
    validate_cbqm(problem)
    _validate_mapping(result, 'result')
    _validate_object_fields(
        result,
        required=_RESULT_REQUIRED_FIELDS,
        optional=_RESULT_OPTIONAL_FIELDS,
        label='result',
    )

    if result['schema'] != 'cbqm-result.v1':
        raise ValueError("result schema must be 'cbqm-result.v1'.")
    _validate_result_problem_id(problem, result)
    _validate_solver_identity(result['solver'])

    status = result['status']
    if status not in _RESULT_STATUSES:
        raise ValueError(f"Unknown solver status '{status}'.")

    _validate_result_candidate(
        problem,
        status,
        result['best_sample'],
        result['best_objective'],
        result['feasibility'],
    )
    _validate_non_negative_finite_number(
        result['runtime_seconds'],
        'runtime_seconds',
    )

    if 'bounds' in result:
        _validate_bounds(
            problem,
            result['bounds'],
            result['best_objective'],
            result['feasibility'],
            status,
        )
    if 'proof' in result:
        _validate_proof(result['proof'], status)
    if 'termination_reason' in result and not isinstance(
        result['termination_reason'],
        str,
    ):
        raise TypeError('termination_reason must be a string.')
    if 'metrics' in result:
        _validate_json_object(result['metrics'], "result['metrics']")
    if 'trace' in result:
        _validate_trace(problem, result['trace'])
    if 'metadata' in result:
        _validate_json_object(result['metadata'], "result['metadata']")

    _validate_json_document(result, 'result')


def _evaluate_cbqm_objective_exact(problem, sample):
    """Accumulate the original objective without binary64 rounding."""
    objective = problem['objective']
    value = Fraction(objective['offset'])
    for index, coefficient in objective['linear']:
        if sample[index]:
            value += Fraction(coefficient)
    for left, right, coefficient in objective['quadratic']:
        if sample[left] and sample[right]:
            value += Fraction(coefficient)
    return value


def _evaluate_cbqm_feasibility(problem, sample):
    """Evaluate a validated sample and build a deterministic violation list."""
    exact_violations = []

    for fixed in problem['fixed_values']:
        index = fixed['index']
        expected = Fraction(fixed['value'])
        activity = Fraction(sample[index])
        if activity == expected:
            continue
        exact_violations.append(
            {
                'constraint_name': (
                    f"fixed:{problem['variables'][index]['name']}"
                ),
                'activity': activity,
                'lower_bound': expected,
                'upper_bound': expected,
                'magnitude': abs(activity - expected),
            }
        )

    for constraint in problem['constraints']:
        activity = sum(
            (
                Fraction(coefficient) * sample[index]
                for index, coefficient in constraint['linear']
            ),
            start=Fraction(0),
        )
        lower = (
            Fraction(constraint['lower_bound'])
            if 'lower_bound' in constraint
            else None
        )
        upper = (
            Fraction(constraint['upper_bound'])
            if 'upper_bound' in constraint
            else None
        )
        magnitude = None
        if lower is not None and activity < lower:
            magnitude = lower - activity
        elif upper is not None and activity > upper:
            magnitude = activity - upper
        if magnitude is None:
            continue

        violation = {
            'constraint_name': constraint['name'],
            'activity': activity,
            'magnitude': magnitude,
        }
        if lower is not None:
            violation['lower_bound'] = lower
        if upper is not None:
            violation['upper_bound'] = upper
        exact_violations.append(violation)

    violations = [
        _public_violation(violation)
        for violation in exact_violations
    ]
    maximum = max(
        (
            violation['magnitude']
            for violation in exact_violations
        ),
        default=Fraction(0),
    )
    return {
        'feasible': not violations,
        'violated_count': len(violations),
        'max_violation': _fraction_to_json_number(
            maximum,
            'maximum CBQM violation',
        ),
        'violations': violations,
    }


def _public_violation(violation):
    """Convert one exact internal violation to finite JSON numbers."""
    output = {
        'constraint_name': violation['constraint_name'],
        'activity': _fraction_to_json_number(
            violation['activity'],
            'constraint activity',
        ),
        'magnitude': _fraction_to_json_number(
            violation['magnitude'],
            'constraint violation magnitude',
        ),
    }
    for field_name in ('lower_bound', 'upper_bound'):
        if field_name in violation:
            output[field_name] = _fraction_to_json_number(
                violation[field_name],
                f'constraint {field_name}',
            )
    return output


def _validate_result_problem_id(problem, result):
    """Require the result to identify its exact source problem."""
    problem_id = result['problem_id']
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError('result problem_id must be a non-empty string.')
    if problem_id != problem['problem_id']:
        raise ValueError(
            'result problem_id must match the source problem_id.'
        )


def _validate_result_candidate(
    problem,
    status,
    sample,
    objective,
    feasibility,
):
    """Validate candidate nullability and canonical objective/feasibility."""
    presence = (
        sample is not None,
        objective is not None,
        feasibility is not None,
    )
    if len(set(presence)) != 1:
        raise ValueError(
            'best_sample, best_objective, and feasibility must either all be '
            'null or all be present.'
        )
    if status in {'optimal', 'feasible'} and sample is None:
        raise ValueError(
            f"Status '{status}' requires a feasible candidate."
        )
    if status == 'infeasible' and sample is not None:
        raise ValueError(
            "Status 'infeasible' cannot include a candidate."
        )
    if sample is None:
        return

    _validate_binary_sample(
        sample,
        len(problem['variables']),
        'best_sample',
    )
    _validate_finite_number(objective, 'best_objective')
    recomputed_objective = _fraction_to_json_number(
        _evaluate_cbqm_objective_exact(problem, sample),
        'CBQM objective',
    )
    _require_equal_number(
        objective,
        recomputed_objective,
        'best_objective',
    )

    _validate_feasibility(feasibility)
    recomputed_feasibility = _evaluate_cbqm_feasibility(problem, sample)
    _require_equal_feasibility(feasibility, recomputed_feasibility)
    if status in {'optimal', 'feasible'} and not feasibility['feasible']:
        raise ValueError(
            f"Status '{status}' requires a constraint-feasible candidate."
        )


def _validate_feasibility(feasibility):
    """Validate the closed feasibility summary and its internal arithmetic."""
    _validate_mapping(feasibility, 'feasibility')
    _validate_object_fields(
        feasibility,
        required=_FEASIBILITY_REQUIRED_FIELDS,
        optional=frozenset(),
        label='feasibility',
    )
    if type(feasibility['feasible']) is not bool:
        raise TypeError("feasibility['feasible'] must be a boolean.")
    _validate_non_negative_integer(
        feasibility['violated_count'],
        "feasibility['violated_count']",
    )
    _validate_non_negative_finite_number(
        feasibility['max_violation'],
        "feasibility['max_violation']",
    )

    violations = feasibility['violations']
    if not isinstance(violations, list):
        raise TypeError("feasibility['violations'] must be a list.")
    for position, violation in enumerate(violations):
        _validate_violation(violation, position)

    if feasibility['violated_count'] != len(violations):
        raise ValueError(
            "feasibility['violated_count'] must equal violations length."
        )
    if feasibility['feasible'] != (len(violations) == 0):
        raise ValueError(
            "feasibility['feasible'] must agree with the violation list."
        )
    expected_maximum = max(
        (
            Fraction(violation['magnitude'])
            for violation in violations
        ),
        default=Fraction(0),
    )
    _require_equal_number(
        feasibility['max_violation'],
        _fraction_to_json_number(
            expected_maximum,
            'maximum CBQM violation',
        ),
        "feasibility['max_violation']",
    )


def _validate_violation(violation, position):
    """Validate one closed violation record."""
    label = f"feasibility['violations'][{position}]"
    _validate_mapping(violation, label)
    _validate_object_fields(
        violation,
        required=_VIOLATION_REQUIRED_FIELDS,
        optional=_VIOLATION_OPTIONAL_FIELDS,
        label=label,
    )
    name = violation['constraint_name']
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}['constraint_name'] must be non-empty.")
    _validate_finite_number(violation['activity'], f"{label}['activity']")
    _validate_finite_number(violation['magnitude'], f"{label}['magnitude']")
    if violation['magnitude'] <= 0:
        raise ValueError(f"{label}['magnitude'] must be positive.")
    if 'lower_bound' not in violation and 'upper_bound' not in violation:
        raise ValueError(f'{label} must include at least one bound.')
    for field_name in ('lower_bound', 'upper_bound'):
        if field_name in violation:
            _validate_finite_number(
                violation[field_name],
                f"{label}['{field_name}']",
            )
    if (
        'lower_bound' in violation
        and 'upper_bound' in violation
        and violation['lower_bound'] > violation['upper_bound']
    ):
        raise ValueError(f'{label} has inconsistent bounds.')


def _require_equal_feasibility(reported, recomputed):
    """Require the canonical violation order and exact numeric values."""
    for field_name in ('feasible', 'violated_count'):
        if reported[field_name] != recomputed[field_name]:
            raise ValueError(
                f"feasibility['{field_name}'] does not match recomputation."
            )
    _require_equal_number(
        reported['max_violation'],
        recomputed['max_violation'],
        "feasibility['max_violation']",
    )
    if len(reported['violations']) != len(recomputed['violations']):
        raise ValueError(
            "feasibility['violations'] does not match recomputation."
        )
    for position, (actual, expected) in enumerate(
        zip(reported['violations'], recomputed['violations'])
    ):
        if set(actual) != set(expected):
            raise ValueError(
                f'Violation {position} fields do not match recomputation.'
            )
        for field_name in actual:
            if field_name == 'constraint_name':
                equal = actual[field_name] == expected[field_name]
            else:
                equal = (
                    Fraction(actual[field_name])
                    == Fraction(expected[field_name])
                )
            if not equal:
                raise ValueError(
                    f'Violation {position} does not match recomputation.'
                )


def _validate_bounds(problem, bounds, objective, feasibility, status):
    """Validate optional solver-attested primal and dual progress."""
    _validate_mapping(bounds, 'bounds')
    _validate_object_fields(
        bounds,
        required=frozenset(),
        optional=_BOUNDS_FIELDS,
        label='bounds',
    )
    if not bounds:
        raise ValueError('bounds must contain at least one field.')
    for field_name, value in bounds.items():
        if field_name in {'absolute_gap', 'relative_gap'}:
            _validate_non_negative_finite_number(
                value,
                f"bounds['{field_name}']",
            )
        else:
            _validate_finite_number(value, f"bounds['{field_name}']")

    primal = bounds.get('primal_bound')
    dual = bounds.get('dual_bound')
    if primal is not None and dual is not None:
        sense = problem['objective']['sense']
        ordered = primal >= dual if sense == 'minimize' else primal <= dual
        if not ordered:
            raise ValueError(
                'primal_bound and dual_bound contradict objective sense.'
            )
        if 'absolute_gap' in bounds:
            _require_equal_number(
                bounds['absolute_gap'],
                _fraction_to_json_number(
                    abs(Fraction(primal) - Fraction(dual)),
                    'absolute bound gap',
                ),
                "bounds['absolute_gap']",
            )
        if status == 'optimal' and Fraction(primal) != Fraction(dual):
            raise ValueError(
                "Status 'optimal' requires equal reported primal/dual bounds."
            )

    if (
        primal is not None
        and objective is not None
        and feasibility is not None
        and feasibility['feasible']
    ):
        _require_equal_number(
            primal,
            objective,
            "bounds['primal_bound']",
        )


def _validate_proof(proof, status):
    """Validate proof metadata without turning it into independent evidence."""
    _validate_mapping(proof, 'proof')
    _validate_object_fields(
        proof,
        required=_PROOF_REQUIRED_FIELDS,
        optional=_PROOF_OPTIONAL_FIELDS,
        label='proof',
    )
    if proof['claim'] not in _PROOF_CLAIMS:
        raise ValueError(f"Unknown proof claim '{proof['claim']}'.")
    for field_name in ('kind', 'producer'):
        value = proof[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"proof['{field_name}'] must be a non-empty string."
            )
    if type(proof['independently_verified']) is not bool:
        raise TypeError(
            "proof['independently_verified'] must be a boolean."
        )
    if 'details' in proof:
        _validate_json_object(proof['details'], "proof['details']")

    expected_claim = {
        'optimal': 'optimality',
        'infeasible': 'infeasibility',
    }.get(status)
    if expected_claim is not None and proof['claim'] != expected_claim:
        raise ValueError(
            f"Status '{status}' requires proof claim '{expected_claim}'."
        )


def _validate_trace(problem, trace):
    """Validate optional candidate observations against the source CBQM."""
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

        candidate_fields = (
            'sample' in entry,
            'objective' in entry,
            'feasible' in entry,
        )
        if len(set(candidate_fields)) != 1:
            raise ValueError(
                f'{label} sample, objective, and feasible fields must appear '
                'together.'
            )
        if 'sample' in entry:
            _validate_binary_sample(
                entry['sample'],
                len(problem['variables']),
                f"{label}['sample']",
            )
            _validate_finite_number(
                entry['objective'],
                f"{label}['objective']",
            )
            if type(entry['feasible']) is not bool:
                raise TypeError(f"{label}['feasible'] must be a boolean.")
            recomputed_objective = _fraction_to_json_number(
                _evaluate_cbqm_objective_exact(problem, entry['sample']),
                'CBQM objective',
            )
            _require_equal_number(
                entry['objective'],
                recomputed_objective,
                f"{label}['objective']",
            )
            recomputed_feasible = _evaluate_cbqm_feasibility(
                problem,
                entry['sample'],
            )['feasible']
            if entry['feasible'] != recomputed_feasible:
                raise ValueError(
                    f"{label}['feasible'] does not match recomputation."
                )
        if 'metadata' in entry:
            _validate_json_object(entry['metadata'], f"{label}['metadata']")


def _require_equal_number(reported, recomputed, label):
    """Compare finite JSON numbers through their exact Python values."""
    if Fraction(reported) != Fraction(recomputed):
        raise ValueError(f'{label} does not match canonical recomputation.')


__all__ = [
    'evaluate_cbqm_feasibility',
    'evaluate_cbqm_objective',
    'validate_cbqm_result',
]
