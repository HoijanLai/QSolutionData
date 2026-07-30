"""Read-only integrity checks for complete problem cases and collections.

The persistence layer validates the ``problem-case.v1`` envelope and lineage.
This module deliberately builds on top of it: each known artifact is passed to
the representation's authoritative validator, and each built-in best-known
witness is re-evaluated against its canonical artifact. Re-evaluation checks
the stored witness; it does not rerun an exhaustive global-optimality proof.

The public workflow is intentionally short::

    report = validate_problem_path('problem/data')
    if report.warnings:
        ...

All representation-specific branching stays in protected helpers so callers
see a small, stable API rather than a second copy of the model contracts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from lib.contracts import validate_cbqm, validate_qubo

from .graph_codec import artifact_to_networkx
from .problem_def import (
    ProblemArtifact,
    ProblemCase,
    ProblemSet,
    TaskDefinition,
)
from .reader import DEFAULT_CASE_FILENAME, load_problem_case
from .registries import (
    _representation_validator_for,
    register_representation_validator,
)
from .transforms import (
    CBQM_FACTOR_REPRESENTATION,
    QUBO_INTERACTION_REPRESENTATION,
    QUBO_MAXCUT_REPRESENTATION,
    factor_graph_to_cbqm,
    interaction_graph_to_qubo,
    qubo_to_maxcut_graph,
)
from .updater import (
    _has_registered_task_evaluator,
    update_best_known,
)


class ProblemValidationError(ValueError):
    """Explain why a problem collection failed its read-only integrity audit."""


@dataclass(frozen=True)
class ProblemValidationReport:
    """Aggregate counts and conservative warnings from a successful audit."""

    case_count: int
    artifact_count: int
    validated_artifact_count: int
    task_count: int
    best_known_count: int
    validated_best_known_count: int
    representation_counts: tuple[tuple[str, int], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def fully_checked(self) -> bool:
        """Whether every artifact and incumbent had a built-in deep checker."""
        return not self.warnings

    def to_dict(self) -> dict:
        """Return a stable JSON-compatible summary for CLI and CI consumers."""
        return {
            'ok': True,
            'fully_checked': self.fully_checked,
            'case_count': self.case_count,
            'artifact_count': self.artifact_count,
            'validated_artifact_count': self.validated_artifact_count,
            'task_count': self.task_count,
            'best_known_count': self.best_known_count,
            'validated_best_known_count': (
                self.validated_best_known_count
            ),
            'representation_counts': dict(self.representation_counts),
            'warnings': list(self.warnings),
        }


def validate_problem_case(
    problem_case,
    *,
    strict=False,
) -> ProblemValidationReport:
    """Deep-check one already-loaded case without changing it.

    Unknown custom representations remain valid envelope artifacts, but they
    produce a warning because this package cannot verify their private
    semantics. ``strict=True`` promotes such warnings to a validation error.
    """
    _require_strict_flag(strict)
    if not isinstance(problem_case, ProblemCase):
        raise TypeError('problem_case must be a ProblemCase.')

    report = _validate_one_case(problem_case)
    _enforce_strict_mode(report, strict)
    return report


def validate_problem_set(
    problem_set,
    *,
    strict=False,
) -> ProblemValidationReport:
    """Deep-check every case in a duplicate-free :class:`ProblemSet`."""
    _require_strict_flag(strict)
    if not isinstance(problem_set, ProblemSet):
        raise TypeError('problem_set must be a ProblemSet.')
    if not problem_set:
        raise ProblemValidationError(
            'A problem set audit requires at least one case.'
        )

    report = _merge_reports(
        _validate_one_case(problem_case)
        for problem_case in problem_set
    )
    _enforce_strict_mode(report, strict)
    return report


def validate_problem_path(
    path,
    *,
    strict=False,
) -> ProblemValidationReport:
    """Load and deep-check one case file/directory or one collection directory.

    A directory containing ``case.json`` is one case. A directory whose
    immediate children contain ``case.json`` is a collection. If both layouts
    occur at once, the path is rejected as ambiguous rather than silently
    choosing one interpretation.
    """
    _require_strict_flag(strict)
    problem_path = Path(path)
    target = _classify_problem_path(problem_path)

    if target == 'case':
        problem_case = _load_case_with_context(problem_path)
        return validate_problem_case(problem_case, strict=strict)

    return _validate_collection_paths(
        _child_case_paths(problem_path),
        strict,
    )


def _validate_one_case(problem_case):
    """Keep the main audit loop close to pseudocode."""
    representation_counts = Counter()
    warnings = []
    validated_artifacts = 0

    for artifact in problem_case.artifacts:
        representation_counts[artifact.representation] += 1
        if _validate_artifact_with_context(problem_case, artifact):
            validated_artifacts += 1
        else:
            warnings.append(
                f"Case '{problem_case.problem_id}' artifact "
                f"'{artifact.artifact_id}' uses unchecked representation "
                f"'{artifact.representation}'."
            )

    best_known_count = 0
    validated_best_known = 0
    for task in problem_case.tasks:
        if task.best_known is None:
            continue
        best_known_count += 1
        if _validate_best_known_with_context(problem_case, task):
            validated_best_known += 1
        else:
            warnings.append(
                f"Case '{problem_case.problem_id}' task '{task.task_id}' "
                'best-known witness was not re-evaluated because no built-in '
                'evaluator covers its artifact/solution representation.'
            )

    return ProblemValidationReport(
        case_count=1,
        artifact_count=len(problem_case.artifacts),
        validated_artifact_count=validated_artifacts,
        task_count=len(problem_case.tasks),
        best_known_count=best_known_count,
        validated_best_known_count=validated_best_known,
        representation_counts=tuple(sorted(representation_counts.items())),
        warnings=tuple(warnings),
    )


def _validate_artifact_with_context(problem_case, artifact):
    """Run one authoritative artifact checker and add stable diagnostics."""
    try:
        return _validate_artifact(problem_case, artifact)
    except (
        TypeError,
        ValueError,
        OverflowError,
        nx.NetworkXException,
    ) as error:
        raise ProblemValidationError(
            f"Case '{problem_case.problem_id}' artifact "
            f"'{artifact.artifact_id}' ({artifact.representation}) failed "
            f'validation: {error}'
        ) from error


def _validate_artifact(problem_case, artifact):
    """Return whether this package owns a deep checker for ``artifact``."""
    validator = _representation_validator_for(artifact.representation)
    if validator is not None:
        validator(problem_case, artifact)
        return True

    if artifact.representation.startswith('networkx.'):
        artifact_to_networkx(artifact)
        # An unregistered project-specific ``networkx.*`` name may imply
        # semantics beyond node-link shape. Parsing it is useful diagnostics,
        # but it remains unchecked until that representation registers its
        # own authoritative validator.
        return False
    return False


def _validate_cbqm_artifact(problem_case, artifact):
    """Run the authoritative closed CBQM wire-contract validator."""
    del problem_case
    validate_cbqm(artifact.payload)


def _validate_qubo_artifact(problem_case, artifact):
    """Run the authoritative closed QUBO wire-contract validator."""
    del problem_case
    validate_qubo(artifact.payload)


def _validate_node_link_artifact(problem_case, artifact):
    """Validate the one generic NetworkX representation's promised shape."""
    del problem_case
    artifact_to_networkx(artifact)


def _validate_cbqm_factor_artifact(problem_case, artifact):
    """Reverse the factor graph and tie it to its canonical CBQM parent."""
    restored = factor_graph_to_cbqm(artifact)
    _validate_model_parent(
        problem_case,
        artifact,
        'cbqm.v1',
        restored,
    )


def _validate_qubo_interaction_artifact(problem_case, artifact):
    """Reverse the interaction graph and tie it to its QUBO parent."""
    restored = interaction_graph_to_qubo(artifact)
    _validate_model_parent(
        problem_case,
        artifact,
        'qubo.v1',
        restored,
    )


def _validate_qubo_maxcut_artifact(problem_case, artifact):
    """Rebuild the signed graph from its embedded canonical QUBO.

    Merely parsing node-link JSON would not prove that an edge still carries
    the correct signed MaxCut weight. Rebuilding gives this one-way transform a
    semantic checker without inventing a lossy reverse conversion.
    """
    graph = artifact_to_networkx(artifact)
    qubo = graph.graph.get('qubo_problem')
    if not isinstance(qubo, Mapping):
        raise TypeError(
            'QUBO MaxCut graph must embed a qubo_problem mapping.'
        )
    validate_qubo(qubo)
    _validate_model_parent(
        problem_case,
        artifact,
        'qubo.v1',
        qubo,
    )
    expected = qubo_to_maxcut_graph(qubo)
    if not _graphs_equal_type_sensitive(graph, expected):
        raise ValueError(
            'QUBO MaxCut graph does not equal the canonical graph rebuilt '
            'from its embedded qubo_problem.'
        )


def _validate_model_parent(
    problem_case,
    artifact,
    expected_representation,
    restored_payload,
):
    """Tie a known derived graph back to its declared direct parent model."""
    restored_problem_id = restored_payload.get('problem_id')
    if restored_problem_id != problem_case.problem_id:
        raise ValueError(
            f"Graph model problem_id {restored_problem_id!r} does not match "
            f"case problem_id '{problem_case.problem_id}'."
        )

    parent_id = artifact.parent_artifact_id
    if parent_id is None:
        return
    parent = problem_case.get_artifact(parent_id)
    if parent.representation != expected_representation:
        raise ValueError(
            f"Known graph representation requires a direct "
            f"'{expected_representation}' parent; got "
            f"'{parent.representation}'."
        )
    if not _json_values_equal(parent.payload, restored_payload):
        raise ValueError(
            f"Graph semantics do not reproduce parent artifact '{parent_id}'."
        )


def _graphs_equal_type_sensitive(actual, expected):
    """Compare graph semantics without Python's ``True == 1`` shortcut."""
    if (
        actual.is_directed() != expected.is_directed()
        or actual.is_multigraph() != expected.is_multigraph()
        or actual.number_of_nodes() != expected.number_of_nodes()
        or actual.number_of_edges() != expected.number_of_edges()
        or not _json_values_equal(actual.graph, expected.graph)
    ):
        return False

    for node_id, expected_attributes in expected.nodes(data=True):
        if node_id not in actual:
            return False
        if not _json_values_equal(
            actual.nodes[node_id],
            expected_attributes,
        ):
            return False

    # QUBO MaxCut is deliberately a simple undirected graph. Keeping this
    # helper generic enough to explain the assumption prevents accidental use
    # on a MultiGraph, where edge keys would need a separate identity check.
    if actual.is_multigraph() or expected.is_multigraph():
        return False
    for left, right, expected_attributes in expected.edges(data=True):
        if not actual.has_edge(left, right):
            return False
        if not _json_values_equal(
            actual.edges[left, right],
            expected_attributes,
        ):
            return False
    return True


def _json_values_equal(left, right):
    """Compare JSON-like values while preserving numeric type identity."""
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(
            _json_values_equal(left[key], right[key])
            for key in left
        )
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        return (
            len(left) == len(right)
            and all(
                _json_values_equal(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        )
    return type(left) is type(right) and left == right


def _validate_best_known_with_context(problem_case, task):
    """Check provenance/objective through the same public update boundary."""
    has_builtin_evaluator = _has_builtin_evaluator(problem_case, task)
    try:
        # Updating with the incumbent itself is intentionally a pure dry run.
        # The returned case is discarded; the updater still rechecks exact
        # provenance, feasibility and objective equality where supported.
        update_best_known(
            problem_case,
            task.task_id,
            task.best_known,
        )
    except NotImplementedError:
        return False
    except (TypeError, ValueError, OverflowError) as error:
        raise ProblemValidationError(
            f"Case '{problem_case.problem_id}' task '{task.task_id}' "
            f'best-known validation failed: {error}'
        ) from error
    return has_builtin_evaluator


def _has_builtin_evaluator(problem_case, task):
    """Ask the same evaluator registry used by the updater."""
    artifact = problem_case.get_artifact(task.canonical_artifact_id)
    return _has_registered_task_evaluator(
        artifact.representation,
        task.solution_representation,
    )


def _classify_problem_path(path):
    """Classify path layout while rejecting empty and ambiguous directories."""
    if path.is_file():
        return 'case'
    if not path.is_dir():
        raise FileNotFoundError(f'Problem path not found: {path}')

    root_case_exists = (path / DEFAULT_CASE_FILENAME).is_file()
    child_cases = _child_case_paths(path)
    if root_case_exists and child_cases:
        raise ProblemValidationError(
            f"Ambiguous problem path '{path}': it contains both "
            f"'{DEFAULT_CASE_FILENAME}' and child case directories."
        )
    if root_case_exists:
        return 'case'
    if child_cases:
        return 'set'
    raise ProblemValidationError(
        f"No '{DEFAULT_CASE_FILENAME}' files found under problem path: {path}"
    )


def _child_case_paths(path):
    """Return immediate child manifests in stable directory-name order."""
    return tuple(
        sorted(
            path.glob(f'*/{DEFAULT_CASE_FILENAME}'),
            key=lambda item: item.parent.name,
        )
    )


def _load_case_with_context(path):
    """Preserve the manifest path when a lower-level parser rejects content."""
    try:
        return load_problem_case(path)
    except (OSError, TypeError, ValueError) as error:
        raise ProblemValidationError(
            f"Could not load problem case '{path}': {error}"
        ) from error


def _validate_collection_paths(case_paths, strict):
    """Audit every discoverable case and aggregate one failure per case."""
    loaded_cases = []
    failures = []
    for case_path in case_paths:
        try:
            loaded_cases.append((case_path, _load_case_with_context(case_path)))
        except ProblemValidationError as error:
            failures.append(str(error))

    failures.extend(_duplicate_problem_id_failures(loaded_cases))

    reports = []
    for case_path, problem_case in loaded_cases:
        try:
            reports.append(_validate_one_case(problem_case))
        except ProblemValidationError as error:
            failures.append(
                f"Deep validation for '{case_path}' failed: {error}"
            )

    if failures:
        details = '\n'.join(f'- {failure}' for failure in failures)
        raise ProblemValidationError(
            f'Problem collection validation failed:\n{details}'
        )

    report = _merge_reports(reports)
    _enforce_strict_mode(report, strict)
    return report


def _duplicate_problem_id_failures(loaded_cases):
    """Report every duplicate ID with all contributing manifest paths."""
    paths_by_id = {}
    for case_path, problem_case in loaded_cases:
        paths_by_id.setdefault(problem_case.problem_id, []).append(case_path)

    failures = []
    for problem_id, paths in sorted(paths_by_id.items()):
        if len(paths) < 2:
            continue
        joined_paths = ', '.join(str(path) for path in paths)
        failures.append(
            f"Duplicate problem_id '{problem_id}' in: {joined_paths}"
        )
    return failures


def _merge_reports(reports):
    """Combine immutable case reports without weakening any warning."""
    case_count = 0
    artifact_count = 0
    validated_artifact_count = 0
    task_count = 0
    best_known_count = 0
    validated_best_known_count = 0
    representation_counts = Counter()
    warnings = []

    for report in reports:
        case_count += report.case_count
        artifact_count += report.artifact_count
        validated_artifact_count += report.validated_artifact_count
        task_count += report.task_count
        best_known_count += report.best_known_count
        validated_best_known_count += report.validated_best_known_count
        representation_counts.update(dict(report.representation_counts))
        warnings.extend(report.warnings)

    return ProblemValidationReport(
        case_count=case_count,
        artifact_count=artifact_count,
        validated_artifact_count=validated_artifact_count,
        task_count=task_count,
        best_known_count=best_known_count,
        validated_best_known_count=validated_best_known_count,
        representation_counts=tuple(sorted(representation_counts.items())),
        warnings=tuple(warnings),
    )


def _enforce_strict_mode(report, strict):
    """Promote conservative coverage warnings to one actionable error."""
    if not strict or not report.warnings:
        return
    details = '\n'.join(f'- {warning}' for warning in report.warnings)
    raise ProblemValidationError(
        f'Strict problem validation found unchecked content:\n{details}'
    )


def _require_strict_flag(strict):
    """Avoid treating truthy integers or strings as a policy decision."""
    if type(strict) is not bool:
        raise TypeError('strict must be a boolean.')


def _register_builtin_representation_validators():
    """Install authoritative validators without hard-coded dispatch branches."""
    registrations = (
        ('cbqm.v1', _validate_cbqm_artifact),
        ('qubo.v1', _validate_qubo_artifact),
        ('networkx.node-link.v1', _validate_node_link_artifact),
        (CBQM_FACTOR_REPRESENTATION, _validate_cbqm_factor_artifact),
        (
            QUBO_INTERACTION_REPRESENTATION,
            _validate_qubo_interaction_artifact,
        ),
        (QUBO_MAXCUT_REPRESENTATION, _validate_qubo_maxcut_artifact),
    )
    for representation, validator in registrations:
        register_representation_validator(representation, validator)


_register_builtin_representation_validators()
