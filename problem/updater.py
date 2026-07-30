"""Task-scoped best-known update policies.

The updater never treats an artifact as "the problem" by itself.  It first
selects a task, then evaluates the candidate against that task's canonical
artifact.  This prevents, for example, an exact MaxCut result from locking an
MIS task that happens to share the same source graph.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from .problem_def import (
    BestKnownSolution,
    ProblemCase,
)
from .reader import load_problem_case, save_problem_case
from .registries import (
    _task_evaluator_for,
    register_task_evaluator,
)


@dataclass(frozen=True)
class BestKnownUpdate:
    """Result of applying the update policy to exactly one task."""

    case: ProblemCase
    task_id: str
    updated: bool
    reason: str
    previous: BestKnownSolution | None
    current: BestKnownSolution | None

    @property
    def problem(self):
        """Compatibility alias for code written against the first draft."""
        return self.case


def evaluate_task_solution(
    problem_case,
    task_id,
    solution,
    *,
    evaluator=None,
) -> int | float:
    """Evaluate one solution in the task's canonical interpretation.

    Solver runners should not evaluate a derived representation and silently
    treat that value as the business objective.  This public boundary resolves
    the task first, then delegates to the evaluator for its *canonical*
    artifact.  The same path is used by :func:`update_best_known`, so execution
    records and persisted incumbents cannot disagree about objective meaning.

    Args:
        problem_case: Case containing the task and its canonical artifact.
        task_id: Stable task identifier.
        solution: Solution expressed in ``task.solution_representation``.
        evaluator: Optional business-specific evaluator.  It receives
            ``(canonical_artifact, solution, task)``.

    Returns:
        A finite objective value in the task's declared sense.
    """
    if not isinstance(problem_case, ProblemCase):
        raise TypeError('problem_case must be a ProblemCase.')
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('task_id must be a non-empty string.')
    if evaluator is not None and not isinstance(evaluator, Callable):
        raise TypeError('evaluator must be callable or None.')

    task = problem_case.get_task(task_id)
    artifact = problem_case.get_artifact(task.canonical_artifact_id)
    return _evaluate_candidate(
        artifact,
        solution,
        task,
        evaluator,
    )


def update_best_known(
    problem_case,
    task_id,
    candidate,
    *,
    allow_exact_override=False,
    evaluator=None,
) -> BestKnownUpdate:
    """Validate and conditionally install a task's best-known solution.

    A custom evaluator receives ``(canonical_artifact, solution, task)`` and
    must return a finite objective.  Built-in evaluators cover ``qubo.v1`` and
    ``cbqm.v1`` binary-vector tasks.  Unknown business/graph tasks require an
    evaluator rather than trusting a caller-provided objective blindly.
    """
    task = _validate_update_inputs(
        problem_case,
        task_id,
        candidate,
        allow_exact_override,
        evaluator,
    )
    incumbent = task.best_known

    if incumbent is not None and incumbent.exact and not allow_exact_override:
        _validate_exact_incumbent_lock(
            problem_case,
            task,
            incumbent,
            evaluator,
        )
        return _unchanged(
            problem_case,
            task,
            'exact_solution_locked',
        )

    evaluated = evaluate_task_solution(
        problem_case,
        task.task_id,
        candidate.solution,
        evaluator=evaluator,
    )
    _validate_candidate_objective(candidate, evaluated)

    if incumbent is None:
        return _updated(problem_case, task, candidate, 'first_best_known')

    comparison = _compare_objectives(
        candidate.objective_value,
        incumbent.objective_value,
        task.sense,
    )
    if comparison > 0:
        return _updated(problem_case, task, candidate, 'objective_improved')

    if comparison == 0 and candidate.exact and not incumbent.exact:
        return _updated(problem_case, task, candidate, 'promoted_to_exact')

    if comparison < 0 and candidate.exact:
        raise ValueError(
            'An exact candidate cannot be worse than the current best-known '
            'objective; verify the task sense and candidate evidence.'
        )

    reason = 'objective_tied' if comparison == 0 else 'objective_not_improved'
    return _unchanged(problem_case, task, reason)


def update_best_known_file(
    path,
    task_id,
    candidate,
    *,
    allow_exact_override=False,
    evaluator=None,
) -> BestKnownUpdate:
    """Load, update and atomically persist one task when it improves."""
    problem_case = load_problem_case(path)
    update = update_best_known(
        problem_case,
        task_id,
        candidate,
        allow_exact_override=allow_exact_override,
        evaluator=evaluator,
    )
    if update.updated:
        save_problem_case(update.case, path)
    return update


def _validate_update_inputs(
    problem_case,
    task_id,
    candidate,
    allow_exact_override,
    evaluator,
):
    """Resolve the task after validating the small public API surface."""
    if not isinstance(problem_case, ProblemCase):
        raise TypeError('problem_case must be a ProblemCase.')
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('task_id must be a non-empty string.')
    if not isinstance(candidate, BestKnownSolution):
        raise TypeError('candidate must be a BestKnownSolution.')
    if not isinstance(allow_exact_override, bool):
        raise TypeError('allow_exact_override must be a boolean.')
    if evaluator is not None and not isinstance(evaluator, Callable):
        raise TypeError('evaluator must be callable or None.')
    return problem_case.get_task(task_id)


def _evaluate_candidate(artifact, solution, task, evaluator):
    """Dispatch objective calculation without polluting the update policy."""
    if evaluator is not None:
        value = evaluator(artifact, solution, task)
        _validate_finite_objective(value, 'custom evaluator result')
        return value

    registered = _task_evaluator_for(
        artifact.representation,
        task.solution_representation,
    )
    if registered is not None:
        value = registered(artifact, solution, task)
        _validate_finite_objective(value, 'registered evaluator result')
        return value

    raise NotImplementedError(
        'No built-in evaluator is registered for artifact/solution pair '
        f"'{artifact.representation}' / "
        f"'{task.solution_representation}'; pass evaluator= explicitly."
    )


def _evaluate_qubo_task(artifact, solution, task):
    """Adapt the binary-vector QUBO evaluator to the registry signature."""
    del task
    return _evaluate_qubo(artifact.payload, solution)


def _evaluate_qubo(problem, solution):
    """Evaluate a canonical QUBO vector, including its offset."""
    from lib.contracts import validate_qubo
    from lib.contracts.validation import _evaluate_qubo as evaluate_qubo

    validate_qubo(problem)
    sample = _validate_binary_sample(solution, problem['num_variables'])
    return evaluate_qubo(problem, sample)


def _evaluate_cbqm_task(artifact, solution, task):
    """Adapt the feasible CBQM evaluator to the registry signature."""
    del task
    return _evaluate_cbqm(artifact.payload, solution)


def _evaluate_cbqm(problem, solution):
    """Validate CBQM feasibility, then evaluate its original objective."""
    from lib.contracts import (
        evaluate_cbqm_feasibility,
        evaluate_cbqm_objective,
        validate_cbqm,
    )

    validate_cbqm(problem)
    sample = _validate_binary_sample(solution, len(problem['variables']))
    feasibility = evaluate_cbqm_feasibility(problem, sample)
    if not feasibility['feasible']:
        _raise_cbqm_violation(feasibility['violations'][0])
    return evaluate_cbqm_objective(problem, sample)


def _raise_cbqm_violation(violation):
    """Translate a canonical violation into the updater's stable diagnostic."""
    name = violation['constraint_name']
    if name.startswith('fixed:'):
        raise ValueError(
            f"Candidate violates fixed value for variable '{name[6:]}'."
        )
    if (
        'lower_bound' in violation
        and Fraction(violation['activity'])
        < Fraction(violation['lower_bound'])
    ):
        raise ValueError(
            f"Candidate violates lower bound of constraint '{name}'."
        )
    raise ValueError(
        f"Candidate violates upper bound of constraint '{name}'."
    )


def _evaluate_cbqm_objective_exact(problem, sample):
    """Evaluate a validated CBQM objective as an exact JSON-number rational."""
    from lib.contracts.cbqm_validation import (
        _evaluate_cbqm_objective_exact as evaluate_exact,
    )

    return evaluate_exact(problem, sample)


def _has_registered_task_evaluator(
    artifact_representation,
    solution_representation,
):
    """Report whether the registry can independently evaluate a task pair."""
    return (
        _task_evaluator_for(
            artifact_representation,
            solution_representation,
        )
        is not None
    )


def _validate_binary_sample(solution, variable_count):
    """Normalize a JSON vector and reject bool-as-int surprises."""
    if (
        not isinstance(variable_count, int)
        or isinstance(variable_count, bool)
        or variable_count < 0
    ):
        raise ValueError('Canonical artifact has an invalid variable count.')
    if (
        not isinstance(solution, Sequence)
        or isinstance(solution, (str, bytes, bytearray))
    ):
        raise TypeError('A binary-vector solution must be a sequence.')
    sample = list(solution)
    if len(sample) != variable_count:
        raise ValueError(
            'Candidate solution length must equal the canonical variable count.'
        )
    if any(type(value) is not int or value not in {0, 1} for value in sample):
        raise ValueError('Binary-vector solutions must contain integer 0 or 1.')
    return sample


def _validate_candidate_objective(candidate, evaluated):
    """Reject stale or fabricated objective values before persistence."""
    if Fraction(candidate.objective_value) != Fraction(evaluated):
        raise ValueError(
            'candidate objective_value does not match evaluation of its solution.'
        )


def _validate_exact_incumbent_lock(
    problem_case,
    task,
    incumbent,
    evaluator,
):
    """Refuse to let stale exact evidence silently lock a changed model."""
    artifact = problem_case.get_artifact(task.canonical_artifact_id)
    recorded_artifact_id = incumbent.metadata.get('canonical_artifact_id')
    if (
        recorded_artifact_id is not None
        and recorded_artifact_id != artifact.artifact_id
    ):
        raise ValueError(
            'Exact incumbent provenance names a different canonical artifact.'
        )

    recorded_hash = incumbent.metadata.get('canonical_payload_sha256')
    if (
        recorded_hash is not None
        and recorded_hash != _payload_sha256(artifact.payload)
    ):
        raise ValueError(
            'Exact incumbent is stale for the current canonical artifact payload.'
        )

    # External exact cases may predate artifact hashes.  Re-evaluate their
    # witness where a built-in or caller-supplied evaluator exists.  Unknown
    # business tasks retain their explicit external attestation.
    try:
        evaluated = evaluate_task_solution(
            problem_case,
            task.task_id,
            incumbent.solution,
            evaluator=evaluator,
        )
    except NotImplementedError:
        if evaluator is not None:
            raise
        return
    _validate_candidate_objective(incumbent, evaluated)


def _payload_sha256(payload):
    """Hash one canonical artifact payload for exact-lock provenance."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def _compare_objectives(candidate, incumbent, sense):
    """Return positive/tied/negative from the candidate's perspective."""
    candidate_exact = Fraction(candidate)
    incumbent_exact = Fraction(incumbent)
    if candidate_exact == incumbent_exact:
        return 0
    if sense == 'minimize':
        return 1 if candidate_exact < incumbent_exact else -1
    return 1 if candidate_exact > incumbent_exact else -1


def _updated(problem_case, task, candidate, reason):
    """Construct one immutable successful update."""
    updated_task = task.with_best_known(candidate)
    updated_case = problem_case.with_updated_task(updated_task)
    return BestKnownUpdate(
        case=updated_case,
        task_id=task.task_id,
        updated=True,
        reason=reason,
        previous=task.best_known,
        current=candidate,
    )


def _unchanged(problem_case, task, reason):
    """Construct a no-op result that still names the affected task."""
    return BestKnownUpdate(
        case=problem_case,
        task_id=task.task_id,
        updated=False,
        reason=reason,
        previous=task.best_known,
        current=task.best_known,
    )


def _validate_finite_objective(value, label):
    """Keep custom evaluator failures out of persisted state."""
    if type(value) is int:
        return
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f'{label} must be a finite real number.')


def _register_builtin_task_evaluators():
    """Install the two canonical binary-vector objective evaluators."""
    register_task_evaluator(
        'qubo.v1',
        'binary-vector.v1',
        _evaluate_qubo_task,
    )
    register_task_evaluator(
        'cbqm.v1',
        'binary-vector.v1',
        _evaluate_cbqm_task,
    )


_register_builtin_task_evaluators()
