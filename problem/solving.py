"""Case-aware execution bridge for canonical QUBO solvers.

Solvers intentionally remain small and representation-specific: a
``QuboSolver`` consumes a ``qubo.v1`` mapping and returns
``qubo-result.v1``.  This module owns the application-level questions around
that call:

* which artifact and task are being solved;
* whether the result really belongs to the selected artifact;
* how a compiled sample maps back to a canonical CBQM task;
* whether solver optimality is strong enough to mark that task exact; and
* whether the task's best-known solution should be updated.

Keeping those decisions here lets the public workflow read almost like
pseudocode while avoiding ProblemCase knowledge inside individual solvers.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .case_operations import (
    _validated_qubo_compilation,
    project_qubo_sample_to_cbqm,
)
from .problem_def import BestKnownSolution, ProblemCase
from .updater import (
    BestKnownUpdate,
    _evaluate_cbqm_objective_exact,
    evaluate_task_solution,
    update_best_known,
)

DEFAULT_EXACT_VERIFICATION_MAX_VARIABLES = 24


@dataclass(frozen=True)
class CaseSolveRecord:
    """Immutable provenance record for one solver invocation.

    ``raw_result`` keeps the validated solver-native result.  Canonical fields
    separately describe what the selected *task* means; for a compiled CBQM
    these are therefore the projected CBQM sample and original CBQM objective,
    not the penalized QUBO energy.
    """

    problem_id: str
    task_id: str
    artifact_id: str
    payload_sha256: str
    solver_config: Mapping[str, Any] | None
    raw_result: Mapping[str, Any]
    canonical_solution: Any | None
    canonical_objective_value: int | float | None
    exact_for_task: bool
    update: BestKnownUpdate | None
    case: ProblemCase

    def __post_init__(self):
        if not isinstance(self.case, ProblemCase):
            raise TypeError('case must be a ProblemCase.')
        if self.case.problem_id != self.problem_id:
            raise ValueError('record problem_id must match its resulting case.')
        if not isinstance(self.raw_result, Mapping):
            raise TypeError('raw_result must be a mapping.')
        if self.solver_config is not None and not isinstance(
            self.solver_config,
            Mapping,
        ):
            raise TypeError('solver_config must be a mapping or None.')
        if not isinstance(self.exact_for_task, bool):
            raise TypeError('exact_for_task must be a boolean.')
        if self.canonical_objective_value is not None:
            value = self.canonical_objective_value
            if (
                type(value) not in {int, float}
                or (type(value) is float and not math.isfinite(value))
            ):
                raise TypeError(
                    'canonical_objective_value must be finite or None.'
                )

        # Solver results have already passed their public contract.  Copy them
        # once more so a caller retaining the solver's dict cannot mutate this
        # execution record after construction.
        object.__setattr__(
            self,
            'raw_result',
            _freeze_json_snapshot(dict(self.raw_result)),
        )
        object.__setattr__(
            self,
            'canonical_solution',
            _freeze_json_snapshot(self.canonical_solution),
        )
        if self.solver_config is not None:
            object.__setattr__(
                self,
                'solver_config',
                _freeze_json_snapshot(dict(self.solver_config)),
            )

    @property
    def canonical_objective(self):
        """Short alias for callers that do not use best-known terminology."""
        return self.canonical_objective_value


@dataclass(frozen=True)
class _TaskRoute:
    """Validated interpretation route from selected QUBO to canonical task."""

    kind: str
    compilation_context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _CanonicalCandidate:
    """Internal candidate after canonical projection and evaluation."""

    solution: Any
    objective_value: int | float
    exact: bool
    exactness_metadata: Mapping[str, Any]


def solve_problem_task(
    case,
    task_id,
    artifact_id,
    solver,
    config=None,
    *,
    update_best=False,
    allow_exact_override=False,
    exact_verification_max_variables=(
        DEFAULT_EXACT_VERIFICATION_MAX_VARIABLES
    ),
) -> CaseSolveRecord:
    """Solve one explicit QUBO artifact in the context of one case task.

    ``artifact_id`` is deliberately required even if the case has only one
    QUBO.  Cases may later gain alternate penalties or encodings, and silently
    picking the first matching representation would make old experiments
    change meaning.

    ``exact_verification_max_variables`` bounds the independent exhaustive
    check used before a solver-reported optimum may lock the task as exact.
    Above the limit the candidate remains usable but is conservatively
    recorded as non-exact.
    """
    task, artifact, route = _resolve_execution_route(
        case,
        task_id,
        artifact_id,
    )
    _validate_execution_options(
        solver,
        config,
        update_best,
        allow_exact_override,
        exact_verification_max_variables,
    )
    normalized_config = _normalize_solver_config(config)

    result = _run_and_validate_solver(
        artifact.payload,
        solver,
        normalized_config,
    )
    candidate = _interpret_candidate(
        case,
        task,
        artifact,
        route,
        result,
        exact_verification_max_variables,
    )
    update, resulting_case = _apply_requested_update(
        case,
        task,
        artifact,
        result,
        candidate,
        normalized_config,
        update_best=update_best,
        allow_exact_override=allow_exact_override,
    )

    return CaseSolveRecord(
        problem_id=case.problem_id,
        task_id=task.task_id,
        artifact_id=artifact.artifact_id,
        payload_sha256=_payload_sha256(artifact.payload),
        solver_config=normalized_config,
        raw_result=result,
        canonical_solution=(
            None if candidate is None else candidate.solution
        ),
        canonical_objective_value=(
            None if candidate is None else candidate.objective_value
        ),
        exact_for_task=(
            False if candidate is None else candidate.exact
        ),
        update=update,
        case=resulting_case,
    )


def _resolve_execution_route(case, task_id, artifact_id):
    """Validate all representation choices before starting expensive work."""
    if not isinstance(case, ProblemCase):
        raise TypeError('case must be a ProblemCase.')
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('task_id must be a non-empty string.')
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError('artifact_id must be a non-empty string.')

    task = case.get_task(task_id)
    artifact = case.get_artifact(artifact_id)
    if artifact.representation != 'qubo.v1':
        raise ValueError(
            f"Solver artifact '{artifact_id}' must use representation "
            "'qubo.v1'."
        )

    if task.canonical_artifact_id == artifact.artifact_id:
        return task, artifact, _TaskRoute(kind='direct_qubo')

    canonical = case.get_artifact(task.canonical_artifact_id)
    if canonical.representation != 'cbqm.v1':
        raise ValueError(
            'The selected QUBO can only serve its own canonical task or a '
            'task on its direct compile_qubo CBQM parent.'
        )

    _, parent, context = _validated_qubo_compilation(
        case,
        artifact.artifact_id,
    )
    if parent.artifact_id != canonical.artifact_id:
        raise ValueError(
            f"QUBO artifact '{artifact.artifact_id}' is not compiled directly "
            f"from task '{task.task_id}' canonical artifact "
            f"'{canonical.artifact_id}'."
        )
    return task, artifact, _TaskRoute(
        kind='compiled_cbqm',
        compilation_context=context,
    )


def _validate_execution_options(
    solver,
    config,
    update_best,
    allow_exact_override,
    exact_verification_max_variables,
):
    """Keep API-shape errors separate from solver or model failures."""
    if not callable(getattr(solver, 'solve', None)):
        raise TypeError('solver must provide a callable solve method.')
    if config is not None and not isinstance(config, Mapping):
        raise TypeError('config must be a mapping or None.')
    if not isinstance(update_best, bool):
        raise TypeError('update_best must be a boolean.')
    if not isinstance(allow_exact_override, bool):
        raise TypeError('allow_exact_override must be a boolean.')
    if (
        type(exact_verification_max_variables) is not int
        or exact_verification_max_variables < 0
    ):
        raise ValueError(
            'exact_verification_max_variables must be a non-negative integer.'
        )


def _run_and_validate_solver(payload, solver, config):
    """Call only the native payload boundary, then distrust the returned dict."""
    # Keep basic ProblemCase reading independent from the solver package.  The
    # contract module is needed only when execution is actually requested.
    from lib.contracts import validate_qubo, validate_qubo_result

    # Validate before invoking arbitrary solver code.  A ProblemArtifact checks
    # its representation envelope, but it intentionally is not the detailed
    # mathematical schema authority.
    validate_qubo(payload)
    solver_payload = _mutable_json_copy(payload)
    solver_config = (
        None
        if config is None
        else _mutable_json_copy(config)
    )
    result = solver.solve(solver_payload, solver_config)
    validate_qubo_result(
        problem=payload,
        result=result,
    )
    return _mutable_json_copy(result)


def _interpret_candidate(
    case,
    task,
    artifact,
    route,
    result,
    exact_verification_max_variables,
):
    """Translate a solver candidate into the task's canonical vocabulary."""
    sample = result['best_sample']
    if sample is None:
        return None

    if route.kind == 'direct_qubo':
        solution = list(sample)
        objective = evaluate_task_solution(
            case,
            task.task_id,
            solution,
        )
        exact, evidence = _independently_verify_qubo_optimality(
            artifact.payload,
            sample,
            result['status'],
            exact_verification_max_variables,
        )
        evidence = {'route': 'direct_qubo', **evidence}
        return _CanonicalCandidate(
            solution=solution,
            objective_value=objective,
            exact=exact,
            exactness_metadata=evidence,
        )

    solution = project_qubo_sample_to_cbqm(
        case,
        artifact.artifact_id,
        sample,
    )
    objective = evaluate_task_solution(
        case,
        task.task_id,
        solution,
    )
    exact, evidence = _compiled_cbqm_exactness(
        result,
        route.compilation_context,
        source_problem=case.get_artifact(
            task.canonical_artifact_id
        ).payload,
        source_sample=solution,
        qubo_problem=artifact.payload,
        qubo_sample=sample,
        exact_verification_max_variables=(
            exact_verification_max_variables
        ),
    )
    return _CanonicalCandidate(
        solution=solution,
        objective_value=objective,
        exact=exact,
        exactness_metadata=evidence,
    )


def _compiled_cbqm_exactness(
    result,
    context,
    *,
    source_problem,
    source_sample,
    qubo_problem,
    qubo_sample,
    exact_verification_max_variables,
):
    """Require independently reproducible evidence for exact promotion.

    A persisted ``exact_projection_certified: true`` flag is not authority by
    itself.  We derive the certificate verdict again, reproduce the canonical
    compilation from its recorded resolved config, and verify that every
    encoded penalty equation has zero residual for this full QUBO sample.
    """
    equivalence = context.get('equivalence')
    certificate_consistent = _equivalence_certificate_is_consistent(
        context,
    )
    declared_certified = (
        isinstance(equivalence, Mapping)
        and equivalence.get('exact_projection_certified') is True
    )
    canonical_compilation_verified = (
        declared_certified
        and certificate_consistent
        and _canonical_compilation_matches(
            source_problem,
            qubo_problem,
            context,
        )
    )
    zero_penalty_verified = (
        canonical_compilation_verified
        and _compiled_penalties_are_zero(context, qubo_sample)
    )

    solver_optimality_verified, solver_evidence = (
        _independently_verify_qubo_optimality(
            qubo_problem,
            qubo_sample,
            result['status'],
            exact_verification_max_variables,
        )
    )
    multiplier = context['objective_multiplier']
    source_objective_exact = _evaluate_cbqm_objective_exact(
        source_problem,
        source_sample,
    )
    qubo_energy_exact = _evaluate_qubo_energy_exact(
        qubo_problem,
        qubo_sample,
    )
    energy_matches = (
        qubo_energy_exact
        == Fraction(multiplier) * source_objective_exact
    )

    evidence = {
        'route': 'compiled_cbqm',
        **solver_evidence,
        'compiler_exact_projection_certified': declared_certified,
        'compiler_certificate_consistent': certificate_consistent,
        'canonical_compilation_reproduced': canonical_compilation_verified,
        'zero_penalty_verified': zero_penalty_verified,
        'objective_energy_identity_verified': energy_matches,
        'objective_multiplier': float(multiplier),
    }
    exact = all(
        (
            solver_optimality_verified,
            declared_certified,
            certificate_consistent,
            canonical_compilation_verified,
            zero_penalty_verified,
            energy_matches,
        )
    )
    return exact, evidence


def _independently_verify_qubo_optimality(
    problem,
    sample,
    status,
    max_variables,
):
    """Turn a solver claim into exact evidence without trusting its metadata.

    A valid ``qubo-result.v1`` proves that the reported energy belongs to the
    sample; it does not prove no better sample exists.  Exact best-known locks
    are therefore granted only after this application layer has enumerated the
    source QUBO itself.  Oversized models still return useful incumbents, but
    they remain non-exact until a future independently verifiable certificate
    format is available.
    """
    evidence = {
        'solver_reported_optimal': status == 'optimal',
        'independent_qubo_optimality_verified': False,
        'optimality_verification_method': None,
        'optimality_verification_max_variables': max_variables,
        'optimality_assignments_checked': 0,
    }
    if status != 'optimal':
        evidence['optimality_verification_reason'] = (
            'solver_did_not_report_optimal'
        )
        return False, evidence

    variable_count = problem['num_variables']
    if variable_count > max_variables:
        evidence['optimality_verification_reason'] = (
            'variable_limit_exceeded'
        )
        return False, evidence

    candidate_energy = _evaluate_qubo_energy_exact(problem, sample)
    checked = 0
    for assignment in itertools.product((0, 1), repeat=variable_count):
        checked += 1
        if _evaluate_qubo_energy_exact(
            problem,
            list(assignment),
        ) < candidate_energy:
            evidence['optimality_assignments_checked'] = checked
            evidence['optimality_verification_reason'] = (
                'better_assignment_found'
            )
            return False, evidence

    evidence.update({
        'independent_qubo_optimality_verified': True,
        'optimality_verification_method': 'exhaustive-qubo-enumeration',
        'optimality_assignments_checked': checked,
        'optimality_verification_reason': 'search_space_exhausted',
    })
    return True, evidence


def _evaluate_qubo_energy_exact(problem, sample):
    """Load the shared exact evaluator only at the execution boundary."""
    from lib.contracts.validation import _evaluate_qubo_exact

    return _evaluate_qubo_exact(problem, sample)


def _equivalence_certificate_is_consistent(context):
    """Re-derive every boolean in the compiler's exactness certificate."""
    equivalence = context.get('equivalence')
    constraints = context.get('constraints')
    dropped_terms = context.get('dropped_qubo_terms')
    if (
        not isinstance(equivalence, Mapping)
        or equivalence.get('schema')
        != 'qubo-compilation-equivalence.v1'
        or not isinstance(constraints, list)
        or not isinstance(dropped_terms, list)
    ):
        return False

    try:
        lattice_exact = all(
            isinstance(constraint, Mapping)
            and isinstance(constraint.get('normalization'), Mapping)
            and constraint['normalization'].get('exact_reconstruction') is True
            and isinstance(constraint.get('fixed_substitution'), Mapping)
            and constraint['fixed_substitution'].get(
                'exact_reconstruction'
            ) is True
            for constraint in constraints
        )
    except (KeyError, TypeError):
        return False

    no_nonzero_terms_dropped = not dropped_terms
    arithmetic = context.get('arithmetic')
    arithmetic_exact = (
        isinstance(arithmetic, Mapping)
        and arithmetic.get('exact_accumulation') is True
    )
    expected = {
        'constraint_lattice_exact': lattice_exact,
        'no_nonzero_terms_dropped': no_nonzero_terms_dropped,
        'arithmetic_exact': arithmetic_exact,
        'feasible_set_preserved': lattice_exact,
        'objective_mapping_preserved': (
            no_nonzero_terms_dropped and arithmetic_exact
        ),
        'exact_projection_certified': (
            lattice_exact
            and no_nonzero_terms_dropped
            and arithmetic_exact
        ),
    }
    return all(
        equivalence.get(field) is value
        for field, value in expected.items()
    )


def _canonical_compilation_matches(source_problem, qubo_problem, context):
    """Re-run the trusted compiler and compare both payload and proof context."""
    from lib.compilers.qubo_compiler import compile_qubo

    config = context.get('compiler_config')
    if not isinstance(config, Mapping):
        return False
    try:
        reproduced_qubo, reproduced_context = compile_qubo(
            source_problem,
            dict(config),
        )
    except (TypeError, ValueError, NotImplementedError):
        return False
    return (
        reproduced_qubo == qubo_problem
        and reproduced_context == context
    )


def _compiled_penalties_are_zero(context, qubo_sample):
    """Verify every encoded integer equation, including all slack bits."""
    constraints = context.get('constraints')
    if not isinstance(constraints, list):
        return False

    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            return False
        encodings = constraint.get('encodings')
        if not isinstance(encodings, list):
            return False
        for encoding in encodings:
            if not _encoding_has_zero_penalty(encoding, qubo_sample):
                return False
    return True


def _encoding_has_zero_penalty(encoding, qubo_sample):
    """Evaluate one compiler-recorded squared-penalty residual as integers."""
    if not isinstance(encoding, Mapping):
        return False
    if encoding.get('status') == 'redundant':
        return True
    if encoding.get('status') != 'encoded':
        return False

    integer_terms = encoding.get('integer_terms')
    slack_variables = encoding.get('slack_variables')
    integer_rhs = encoding.get('integer_rhs')
    if (
        not isinstance(integer_terms, list)
        or not isinstance(slack_variables, list)
        or type(integer_rhs) is not int
    ):
        return False

    try:
        left_hand_side = sum(
            coefficient * qubo_sample[index]
            for index, coefficient in integer_terms
        )
        left_hand_side += sum(
            slack['equation_sign']
            * slack['integer_weight']
            * qubo_sample[slack['qubo_index']]
            for slack in slack_variables
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return left_hand_side == integer_rhs


def _apply_requested_update(
    case,
    task,
    artifact,
    result,
    candidate,
    solver_config,
    *,
    update_best,
    allow_exact_override,
):
    """Install a canonical candidate only when the caller requested mutation."""
    if candidate is None or not update_best:
        return None, case

    solver_identity = result['solver']
    best_known = BestKnownSolution(
        solution=candidate.solution,
        objective_value=candidate.objective_value,
        exact=candidate.exact,
        source=_solver_source(solver_identity),
        metadata={
            'artifact_id': artifact.artifact_id,
            'artifact_representation': artifact.representation,
            'payload_sha256': _payload_sha256(artifact.payload),
            'result_status': result['status'],
            'solver': copy.deepcopy(dict(solver_identity)),
            'solver_config': (
                None
                if solver_config is None
                else copy.deepcopy(dict(solver_config))
            ),
            'canonical_artifact_id': task.canonical_artifact_id,
            'canonical_payload_sha256': _payload_sha256(
                case.get_artifact(task.canonical_artifact_id).payload
            ),
            'exactness': copy.deepcopy(dict(candidate.exactness_metadata)),
        },
    )
    update = update_best_known(
        case,
        task.task_id,
        best_known,
        allow_exact_override=allow_exact_override,
    )
    return update, update.case


def _solver_source(solver_identity):
    """Build a concise human-readable source while metadata keeps full detail."""
    return (
        f"{solver_identity['name']}@"
        f"{solver_identity['version']}"
    )


def _normalize_solver_config(config):
    """Create the durable JSON snapshot used for execution and provenance."""
    if config is None:
        return None
    _validate_config_object_keys(config, 'config')
    try:
        serialized = json.dumps(
            dict(config),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
        normalized = json.loads(serialized)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            'case-aware solver config must be JSON-compatible so the run can '
            'be reproduced.'
        ) from error
    if not isinstance(normalized, dict):
        raise TypeError('case-aware solver config must serialize as an object.')
    return normalized


def _validate_config_object_keys(value, label):
    """Prevent JSON from silently coercing integer mapping keys to strings."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f'{label} object keys must be strings.')
            _validate_config_object_keys(item, f"{label}['{key}']")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_config_object_keys(item, f'{label}[{index}]')


def _mutable_json_copy(value):
    """Return a fully detached mutable JSON tree."""
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


class _FrozenDict(dict):
    """JSON-serializable dictionary that rejects post-record mutation."""

    def _immutable(self, *args, **kwargs):
        del args, kwargs
        raise TypeError('CaseSolveRecord snapshots are immutable.')

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, memo):
        del memo
        return self


class _FrozenList(list):
    """JSON-serializable list that rejects post-record mutation."""

    def _immutable(self, *args, **kwargs):
        del args, kwargs
        raise TypeError('CaseSolveRecord snapshots are immutable.')

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __deepcopy__(self, memo):
        del memo
        return self


def _freeze_json_snapshot(value):
    """Recursively freeze a detached JSON value without losing JSON shape."""
    if isinstance(value, Mapping):
        return _FrozenDict({
            key: _freeze_json_snapshot(item)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return _FrozenList(
            _freeze_json_snapshot(item)
            for item in value
        )
    return copy.deepcopy(value)


def _payload_sha256(payload):
    """Hash canonical JSON for reproducible artifact-level provenance."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()
