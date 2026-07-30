"""Strict input, result and witness semantics for native MIS contracts."""

from fractions import Fraction

from .validation import (
    _fraction_to_json_number,
    _validate_finite_number,
    _validate_json_document,
    _validate_json_object,
    _validate_mapping,
    _validate_non_negative_finite_number,
    _validate_non_negative_integer,
    _validate_object_fields,
    _validate_solver_identity,
)


_MIS_REQUIRED_FIELDS = frozenset(
    {'schema', 'problem_id', 'objective', 'vertices', 'edges'}
)
_MIS_OPTIONAL_FIELDS = frozenset({'fixed_values', 'metadata'})
_OBJECTIVE_REQUIRED_FIELDS = frozenset({'kind'})
_OBJECTIVE_KINDS = frozenset(
    {'maximum-cardinality', 'maximum-weight'}
)
_VERTEX_REQUIRED_FIELDS = frozenset({'index', 'name'})
_VERTEX_OPTIONAL_FIELDS = frozenset({'weight', 'metadata'})
_FIXED_REQUIRED_FIELDS = frozenset({'index', 'value'})

_RESULT_REQUIRED_FIELDS = frozenset(
    {
        'schema',
        'problem_id',
        'solver',
        'status',
        'selected_vertices',
        'objective_value',
        'cardinality',
        'total_weight',
        'feasible',
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
    {'optimal', 'feasible', 'infeasible', 'timeout', 'error', 'unknown'}
)
_BOUNDS_FIELDS = frozenset(
    {'incumbent_lower_bound', 'optimum_upper_bound'}
)
_PROOF_REQUIRED_FIELDS = frozenset(
    {'claim', 'kind', 'producer', 'independently_verified'}
)
_PROOF_OPTIONAL_FIELDS = frozenset({'details'})
_PROOF_CLAIMS = frozenset({'optimality', 'infeasibility', 'bound'})
_TRACE_REQUIRED_FIELDS = frozenset({'step', 'time_seconds'})
_TRACE_CANDIDATE_FIELDS = frozenset(
    {
        'selected_vertices',
        'objective_value',
        'cardinality',
        'total_weight',
        'feasible',
    }
)
_TRACE_OPTIONAL_FIELDS = _TRACE_CANDIDATE_FIELDS | {'metadata'}


def validate_mis(problem):
    """Validate the closed canonical ``mis.v1`` graph contract."""
    _validate_mapping(problem, 'problem')
    _validate_object_fields(
        problem,
        required=_MIS_REQUIRED_FIELDS,
        optional=_MIS_OPTIONAL_FIELDS,
        label='problem',
    )
    if problem['schema'] != 'mis.v1':
        raise ValueError("problem schema must be 'mis.v1'.")
    _validate_problem_id(problem)
    objective_kind = _validate_mis_objective(problem['objective'])
    _validate_vertices(problem['vertices'], objective_kind)
    _validate_edges(problem['edges'], len(problem['vertices']))
    _validate_fixed_values(
        problem.get('fixed_values', []),
        len(problem['vertices']),
    )
    if 'metadata' in problem:
        _validate_json_object(problem['metadata'], "problem['metadata']")
    _validate_json_document(problem, 'problem')


def evaluate_mis_solution(problem, selected_vertices):
    """Return canonical objective/cardinality/weight/feasibility semantics."""
    validate_mis(problem)
    selected = _validate_selected_vertices(
        selected_vertices,
        len(problem['vertices']),
        'selected_vertices',
    )
    return _evaluate_mis_solution(problem, selected)


def validate_mis_result(problem, result):
    """Validate a native MIS result and recompute every candidate semantic."""
    validate_mis(problem)
    _validate_mapping(result, 'result')
    _validate_object_fields(
        result,
        required=_RESULT_REQUIRED_FIELDS,
        optional=_RESULT_OPTIONAL_FIELDS,
        label='result',
    )
    if result['schema'] != 'mis-result.v1':
        raise ValueError("result schema must be 'mis-result.v1'.")
    if result['problem_id'] != problem['problem_id']:
        raise ValueError(
            'result problem_id must match the source problem_id.'
        )
    _validate_solver_identity(result['solver'])

    status = result['status']
    if status not in _RESULT_STATUSES:
        raise ValueError(f"Unknown solver status '{status}'.")
    _validate_result_candidate(problem, result, status)
    _validate_non_negative_finite_number(
        result['runtime_seconds'],
        'runtime_seconds',
    )
    if 'bounds' in result:
        _validate_bounds(result['bounds'], result, status)
    if 'proof' in result:
        _validate_proof(result['proof'], status)
    if (
        'termination_reason' in result
        and not isinstance(result['termination_reason'], str)
    ):
        raise TypeError('termination_reason must be a string.')
    if 'metrics' in result:
        _validate_json_object(result['metrics'], "result['metrics']")
    if 'trace' in result:
        _validate_trace(problem, result['trace'])
    if 'metadata' in result:
        _validate_json_object(result['metadata'], "result['metadata']")
    _validate_json_document(result, 'result')


def _validate_problem_id(problem):
    """Require a stable non-empty source identifier."""
    problem_id = problem['problem_id']
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError('problem_id must be a non-empty string.')


def _validate_mis_objective(objective):
    """Validate the one-field objective selector."""
    _validate_mapping(objective, "problem['objective']")
    _validate_object_fields(
        objective,
        required=_OBJECTIVE_REQUIRED_FIELDS,
        optional=frozenset(),
        label="problem['objective']",
    )
    kind = objective['kind']
    if kind not in _OBJECTIVE_KINDS:
        raise ValueError(f"Unknown MIS objective kind '{kind}'.")
    return kind


def _validate_vertices(vertices, objective_kind):
    """Validate stable contiguous identities and weight policy."""
    if not isinstance(vertices, list):
        raise TypeError("problem['vertices'] must be a list.")
    names = []
    for position, vertex in enumerate(vertices):
        label = f"problem['vertices'][{position}]"
        _validate_mapping(vertex, label)
        _validate_object_fields(
            vertex,
            required=_VERTEX_REQUIRED_FIELDS,
            optional=_VERTEX_OPTIONAL_FIELDS,
            label=label,
        )
        if type(vertex['index']) is not int or vertex['index'] != position:
            raise ValueError(
                'Vertex indices must be contiguous and match list positions.'
            )
        name = vertex['name']
        if not isinstance(name, str) or not name:
            raise ValueError('Vertex names must be non-empty strings.')
        names.append(name)
        has_weight = 'weight' in vertex
        if objective_kind == 'maximum-weight' and not has_weight:
            raise ValueError(
                'maximum-weight requires every vertex to define weight.'
            )
        if objective_kind == 'maximum-cardinality' and has_weight:
            raise ValueError(
                'maximum-cardinality vertices must omit weight.'
            )
        if has_weight:
            _validate_finite_number(
                vertex['weight'],
                f"{label}['weight']",
            )
        if 'metadata' in vertex:
            _validate_json_object(
                vertex['metadata'],
                f"{label}['metadata']",
            )
    if len(names) != len(set(names)):
        raise ValueError('Vertex names must be unique.')


def _validate_edges(edges, vertex_count):
    """Require upper-triangular unique edges in canonical order."""
    if not isinstance(edges, list):
        raise TypeError("problem['edges'] must be a list.")
    previous = None
    for position, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError('MIS edges must be [u, v] pairs.')
        left, right = edge
        for index in edge:
            if (
                type(index) is not int
                or index < 0
                or index >= vertex_count
            ):
                raise ValueError('MIS edge index is outside the vertex range.')
        if left >= right:
            raise ValueError('MIS edges must satisfy u < v.')
        pair = (left, right)
        if previous is not None and pair <= previous:
            raise ValueError(
                'MIS edges must be unique and lexicographically ordered.'
            )
        previous = pair


def _validate_fixed_values(fixed_values, vertex_count):
    """Validate optional hard assignments in strictly increasing index order."""
    if not isinstance(fixed_values, list):
        raise TypeError("problem['fixed_values'] must be a list.")
    previous_index = None
    for position, fixed in enumerate(fixed_values):
        label = f"problem['fixed_values'][{position}]"
        _validate_mapping(fixed, label)
        _validate_object_fields(
            fixed,
            required=_FIXED_REQUIRED_FIELDS,
            optional=frozenset(),
            label=label,
        )
        index = fixed['index']
        if (
            type(index) is not int
            or index < 0
            or index >= vertex_count
        ):
            raise ValueError(
                'Fixed vertex index is outside the vertex range.'
            )
        if previous_index is not None and index <= previous_index:
            raise ValueError(
                'fixed_values must have unique increasing indices.'
            )
        previous_index = index
        if type(fixed['value']) is not int or fixed['value'] not in {0, 1}:
            raise ValueError('Fixed vertex values must be integer 0 or 1.')


def _validate_selected_vertices(selected, vertex_count, label):
    """Normalize a sorted vertex-index-set witness."""
    if not isinstance(selected, list):
        raise TypeError(f'{label} must be a list.')
    previous = None
    for index in selected:
        if (
            type(index) is not int
            or index < 0
            or index >= vertex_count
        ):
            raise ValueError(f'{label} index is outside the vertex range.')
        if previous is not None and index <= previous:
            raise ValueError(
                f'{label} must be unique and strictly increasing.'
            )
        previous = index
    return list(selected)


def _evaluate_mis_solution(problem, selected):
    """Evaluate an already validated canonical index set exactly."""
    selected_set = set(selected)
    feasible = all(
        not (left in selected_set and right in selected_set)
        for left, right in problem['edges']
    )
    for fixed in problem.get('fixed_values', []):
        if (fixed['index'] in selected_set) != bool(fixed['value']):
            feasible = False
            break

    cardinality = len(selected)
    if problem['objective']['kind'] == 'maximum-cardinality':
        total_weight_exact = Fraction(cardinality)
        objective_exact = Fraction(cardinality)
    else:
        total_weight_exact = sum(
            (
                Fraction(problem['vertices'][index]['weight'])
                for index in selected
            ),
            start=Fraction(0),
        )
        objective_exact = total_weight_exact
    return {
        'objective_value': _fraction_to_json_number(
            objective_exact,
            'MIS objective',
        ),
        'cardinality': cardinality,
        'total_weight': _fraction_to_json_number(
            total_weight_exact,
            'MIS total weight',
        ),
        'feasible': feasible,
    }


def _validate_result_candidate(problem, result, status):
    """Validate atomic nullability and canonical witness semantics."""
    candidate_fields = (
        result['selected_vertices'],
        result['objective_value'],
        result['cardinality'],
        result['total_weight'],
        result['feasible'],
    )
    presence = tuple(value is not None for value in candidate_fields)
    if len(set(presence)) != 1:
        raise ValueError(
            'MIS candidate fields must either all be null or all be present.'
        )
    selected = result['selected_vertices']
    if status in {'optimal', 'feasible'} and selected is None:
        raise ValueError(f"Status '{status}' requires a feasible candidate.")
    if status == 'infeasible' and selected is not None:
        raise ValueError("Status 'infeasible' cannot include a candidate.")
    if status == 'infeasible' and not _fixed_values_are_infeasible(problem):
        raise ValueError(
            "Status 'infeasible' requires conflicting fixed-in vertices; "
            'otherwise the empty independent set is feasible.'
        )
    if selected is None:
        return

    selected = _validate_selected_vertices(
        selected,
        len(problem['vertices']),
        'selected_vertices',
    )
    _validate_finite_number(result['objective_value'], 'objective_value')
    _validate_non_negative_integer(result['cardinality'], 'cardinality')
    _validate_finite_number(result['total_weight'], 'total_weight')
    if type(result['feasible']) is not bool:
        raise TypeError('feasible must be a boolean.')

    recomputed = _evaluate_mis_solution(problem, selected)
    _require_candidate_semantics(result, recomputed, 'result')
    if status in {'optimal', 'feasible'} and not result['feasible']:
        raise ValueError(
            f"Status '{status}' requires an independent-set candidate."
        )


def _fixed_values_are_infeasible(problem):
    """MIS feasibility is empty exactly when forced-in vertices conflict."""
    forced_in = {
        fixed['index']
        for fixed in problem.get('fixed_values', [])
        if fixed['value'] == 1
    }
    return any(
        left in forced_in and right in forced_in
        for left, right in problem['edges']
    )


def _require_candidate_semantics(reported, recomputed, label):
    """Compare all recomputable metrics without numerical tolerance."""
    if reported['cardinality'] != recomputed['cardinality']:
        raise ValueError(f'{label} cardinality does not match recomputation.')
    if reported['feasible'] != recomputed['feasible']:
        raise ValueError(f'{label} feasible does not match recomputation.')
    for field_name in ('objective_value', 'total_weight'):
        if Fraction(reported[field_name]) != Fraction(
            recomputed[field_name]
        ):
            raise ValueError(
                f'{label} {field_name} does not match recomputation.'
            )


def _validate_bounds(bounds, result, status):
    """Validate maximization lower/upper bounds and incumbent consistency."""
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
        _validate_finite_number(value, f"bounds['{field_name}']")

    lower = bounds.get('incumbent_lower_bound')
    upper = bounds.get('optimum_upper_bound')
    if lower is not None and upper is not None:
        if Fraction(lower) > Fraction(upper):
            raise ValueError('MIS lower bound cannot exceed upper bound.')
        if status == 'optimal' and Fraction(lower) != Fraction(upper):
            raise ValueError(
                "Status 'optimal' requires equal lower and upper bounds."
            )
    if (
        lower is not None
        and result['objective_value'] is not None
        and result['feasible']
        and Fraction(lower) != Fraction(result['objective_value'])
    ):
        raise ValueError(
            'incumbent_lower_bound must equal the reported feasible objective.'
        )


def _validate_proof(proof, status):
    """Validate proof metadata while retaining attestation semantics."""
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
    """Validate trace shape and recompute every present index-set witness."""
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
        present = _TRACE_CANDIDATE_FIELDS & set(entry)
        if present and present != _TRACE_CANDIDATE_FIELDS:
            raise ValueError(
                f'{label} candidate semantic fields must appear together.'
            )
        if present:
            selected = _validate_selected_vertices(
                entry['selected_vertices'],
                len(problem['vertices']),
                f"{label}['selected_vertices']",
            )
            _validate_finite_number(
                entry['objective_value'],
                f"{label}['objective_value']",
            )
            _validate_non_negative_integer(
                entry['cardinality'],
                f"{label}['cardinality']",
            )
            _validate_finite_number(
                entry['total_weight'],
                f"{label}['total_weight']",
            )
            if type(entry['feasible']) is not bool:
                raise TypeError(f"{label}['feasible'] must be a boolean.")
            _require_candidate_semantics(
                entry,
                _evaluate_mis_solution(problem, selected),
                label,
            )
        if 'metadata' in entry:
            _validate_json_object(entry['metadata'], f"{label}['metadata']")


__all__ = [
    'evaluate_mis_solution',
    'validate_mis',
    'validate_mis_result',
]
