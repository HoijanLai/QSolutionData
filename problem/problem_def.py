"""Core data structures for a representation-independent problem collection.

The classes in this module deliberately separate three concepts:

``ProblemCase``
    The stable identity and metadata of one business/problem instance.

``ProblemArtifact``
    One concrete representation of that instance, for example ``cbqm.v1``,
    ``qubo.v1`` or a NetworkX node-link graph.

``TaskDefinition``
    One optimization question asked of an artifact.  Best-known solutions live
    here because the same graph can legitimately be used for MaxCut, MIS and
    other tasks with different objectives and optima.

Keeping these concepts separate lets the main workflow read almost like
pseudocode: load a case, select an artifact, select a task, then solve or derive
another artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

PROBLEM_CASE_SCHEMA = 'problem-case.v1'
_SAFE_IDENTIFIER = re.compile(r'^[A-Za-z0-9._-]+$')
_KNOWN_MODEL_REPRESENTATIONS = {'cbqm.v1', 'qubo.v1'}


@dataclass(frozen=True)
class BestKnownSolution:
    """A task-scoped incumbent and the strength of its evidence.

    ``solution`` intentionally accepts any JSON-compatible value.  Binary
    vectors are common, but graph tasks may prefer a node set, a partition, or
    a richer structured witness.
    """

    solution: Any
    objective_value: int | float
    exact: bool = False
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_finite_number(
            self.objective_value,
            'best-known objective_value',
        )
        if not isinstance(self.exact, bool):
            raise TypeError('best-known exact must be a boolean.')
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source
        ):
            raise ValueError('best-known source must be a non-empty string or None.')

        object.__setattr__(self, 'solution', _json_copy(self.solution, 'solution'))
        object.__setattr__(
            self,
            'objective_value',
            self.objective_value,
        )
        object.__setattr__(
            self,
            'metadata',
            _json_mapping_copy(self.metadata, 'best-known metadata'),
        )

    @property
    def sample(self):
        """Return ``solution`` under the familiar binary-solver vocabulary."""
        return self.solution


@dataclass(frozen=True)
class TransformationRecord:
    """Describe how a derived artifact was produced from its parent."""

    name: str
    version: str
    lossless: bool
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_non_empty_string(self.name, 'transformation name')
        _validate_non_empty_string(self.version, 'transformation version')
        if not isinstance(self.lossless, bool):
            raise TypeError('transformation lossless must be a boolean.')
        object.__setattr__(
            self,
            'context',
            _json_mapping_copy(self.context, 'transformation context'),
        )


@dataclass(frozen=True)
class ProblemArtifact:
    """One serialized representation belonging to a :class:`ProblemCase`."""

    artifact_id: str
    representation: str
    payload: Mapping[str, Any]
    parent_artifact_id: str | None = None
    transformation: TransformationRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_identifier(self.artifact_id, 'artifact_id')
        _validate_non_empty_string(self.representation, 'representation')
        if self.parent_artifact_id is not None:
            _validate_identifier(self.parent_artifact_id, 'parent_artifact_id')
            if self.parent_artifact_id == self.artifact_id:
                raise ValueError('An artifact cannot be its own parent.')

        if self.parent_artifact_id is None and self.transformation is not None:
            raise ValueError(
                'A root artifact cannot have a transformation record without '
                'a parent artifact.'
            )
        if self.parent_artifact_id is not None and self.transformation is None:
            raise ValueError(
                'A derived artifact must record the transformation from its parent.'
            )
        if self.transformation is not None and not isinstance(
            self.transformation,
            TransformationRecord,
        ):
            raise TypeError('transformation must be a TransformationRecord or None.')

        normalized_payload = _json_mapping_copy(
            self.payload,
            'artifact payload',
        )
        _validate_known_representation(
            self.representation,
            normalized_payload,
        )
        object.__setattr__(self, 'payload', normalized_payload)
        object.__setattr__(
            self,
            'metadata',
            _json_mapping_copy(self.metadata, 'artifact metadata'),
        )


@dataclass(frozen=True)
class TaskDefinition:
    """An optimization task whose answer is interpreted on one artifact."""

    task_id: str
    canonical_artifact_id: str
    sense: str
    solution_representation: str = 'binary-vector.v1'
    task_type: str | None = None
    best_known: BestKnownSolution | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_non_empty_string(self.task_id, 'task_id')
        _validate_identifier(
            self.canonical_artifact_id,
            'canonical_artifact_id',
        )
        if self.sense not in {'minimize', 'maximize'}:
            raise ValueError("task sense must be 'minimize' or 'maximize'.")
        _validate_non_empty_string(
            self.solution_representation,
            'solution_representation',
        )
        if self.task_type is not None:
            _validate_non_empty_string(self.task_type, 'task_type')
        if self.best_known is not None and not isinstance(
            self.best_known,
            BestKnownSolution,
        ):
            raise TypeError('best_known must be a BestKnownSolution or None.')
        object.__setattr__(
            self,
            'metadata',
            _json_mapping_copy(self.metadata, 'task metadata'),
        )

    def with_best_known(
        self,
        best_known: BestKnownSolution | None,
    ) -> TaskDefinition:
        """Return a copy with only this task's incumbent changed."""
        return replace(self, best_known=best_known)


@dataclass(frozen=True)
class ProblemCase:
    """A business problem together with all of its known representations."""

    problem_id: str
    artifacts: tuple[ProblemArtifact, ...] = ()
    tasks: tuple[TaskDefinition, ...] = ()
    primary_artifact_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Existing cbqm.v1/qubo.v1 contracts allow any non-empty problem ID.
        # Only artifact IDs are filename stems and therefore use the stricter
        # path-safe identifier rule.
        _validate_non_empty_string(self.problem_id, 'problem_id')
        artifacts = _coerce_tuple(
            self.artifacts,
            ProblemArtifact,
            'artifacts',
        )
        tasks = _coerce_tuple(self.tasks, TaskDefinition, 'tasks')
        object.__setattr__(self, 'artifacts', artifacts)
        object.__setattr__(self, 'tasks', tasks)
        object.__setattr__(
            self,
            'metadata',
            _json_mapping_copy(self.metadata, 'case metadata'),
        )

        artifact_by_id = _index_unique(
            artifacts,
            lambda item: item.artifact_id,
            'artifact_id',
        )
        task_by_id = _index_unique(
            tasks,
            lambda item: item.task_id,
            'task_id',
        )

        self._validate_primary_artifact(artifact_by_id)
        self._validate_artifact_problem_ids()
        self._validate_lineage(artifact_by_id)
        self._validate_tasks(task_by_id, artifact_by_id)

    def get_artifact(self, artifact_id: str) -> ProblemArtifact:
        """Return one artifact by its stable ID."""
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise KeyError(
            f"Problem case '{self.problem_id}' has no artifact '{artifact_id}'."
        )

    def get_task(self, task_id: str) -> TaskDefinition:
        """Return one task by its stable ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(
            f"Problem case '{self.problem_id}' has no task '{task_id}'."
        )

    def artifacts_for(self, representation: str) -> tuple[ProblemArtifact, ...]:
        """Return all artifacts of a representation, preserving case order."""
        return tuple(
            artifact
            for artifact in self.artifacts
            if artifact.representation == representation
        )

    def with_artifact(
        self,
        artifact: ProblemArtifact,
        *,
        replace_existing: bool = False,
        make_primary: bool = False,
    ) -> ProblemCase:
        """Return a case containing ``artifact``.

        Representation is deliberately *not* used as a key.  For example, one
        CBQM may be compiled into several QUBOs with different penalty choices.
        """
        if not isinstance(artifact, ProblemArtifact):
            raise TypeError('artifact must be a ProblemArtifact.')
        if not isinstance(replace_existing, bool):
            raise TypeError('replace_existing must be a boolean.')
        if not isinstance(make_primary, bool):
            raise TypeError('make_primary must be a boolean.')

        existing_ids = {
            item.artifact_id
            for item in self.artifacts
        }
        if artifact.artifact_id in existing_ids and not replace_existing:
            raise ValueError(
                f"Artifact '{artifact.artifact_id}' already exists in this case."
            )
        if artifact.artifact_id in existing_ids:
            existing_artifact = self.get_artifact(artifact.artifact_id)
            if existing_artifact.payload != artifact.payload:
                locked_tasks = [
                    task.task_id
                    for task in self.tasks
                    if (
                        task.canonical_artifact_id == artifact.artifact_id
                        and task.best_known is not None
                    )
                ]
                if locked_tasks:
                    raise ValueError(
                        f"Cannot replace canonical artifact "
                        f"'{artifact.artifact_id}' with a different payload "
                        f"while tasks have best-known solutions: "
                        f"{locked_tasks}. Clear those incumbents first."
                    )

        updated_artifacts = tuple(
            artifact if item.artifact_id == artifact.artifact_id else item
            for item in self.artifacts
        )
        if artifact.artifact_id not in existing_ids:
            updated_artifacts = (*updated_artifacts, artifact)

        primary = (
            artifact.artifact_id
            if make_primary
            else self.primary_artifact_id
        )
        return replace(
            self,
            artifacts=updated_artifacts,
            primary_artifact_id=primary,
        )

    def with_task(
        self,
        task: TaskDefinition,
        *,
        replace_existing: bool = False,
    ) -> ProblemCase:
        """Return a case containing ``task`` without mutating other tasks."""
        if not isinstance(task, TaskDefinition):
            raise TypeError('task must be a TaskDefinition.')
        if not isinstance(replace_existing, bool):
            raise TypeError('replace_existing must be a boolean.')

        existing_ids = {item.task_id for item in self.tasks}
        if task.task_id in existing_ids and not replace_existing:
            raise ValueError(f"Task '{task.task_id}' already exists in this case.")
        updated_tasks = tuple(
            task if item.task_id == task.task_id else item
            for item in self.tasks
        )
        if task.task_id not in existing_ids:
            updated_tasks = (*updated_tasks, task)
        return replace(self, tasks=updated_tasks)

    def with_updated_task(self, task: TaskDefinition) -> ProblemCase:
        """Replace an existing task; useful for best-known updates."""
        self.get_task(task.task_id)
        return self.with_task(task, replace_existing=True)

    def _validate_primary_artifact(self, artifact_by_id):
        """Keep primary selection independent from insertion order."""
        if self.primary_artifact_id is None:
            return
        _validate_identifier(self.primary_artifact_id, 'primary_artifact_id')
        if self.primary_artifact_id not in artifact_by_id:
            raise ValueError(
                f"Primary artifact '{self.primary_artifact_id}' does not exist."
            )

    def _validate_artifact_problem_ids(self):
        """Prevent canonical model payloads from leaking across cases."""
        for artifact in self.artifacts:
            if artifact.representation not in _KNOWN_MODEL_REPRESENTATIONS:
                continue
            payload_problem_id = artifact.payload.get('problem_id')
            if payload_problem_id != self.problem_id:
                raise ValueError(
                    f"Artifact '{artifact.artifact_id}' problem_id "
                    f"{payload_problem_id!r} does not match case "
                    f"'{self.problem_id}'."
                )

    def _validate_lineage(self, artifact_by_id):
        """Validate the parent relation as a directed acyclic graph."""
        for artifact in self.artifacts:
            parent_id = artifact.parent_artifact_id
            if parent_id is not None and parent_id not in artifact_by_id:
                raise ValueError(
                    f"Artifact '{artifact.artifact_id}' references missing "
                    f"parent '{parent_id}'."
                )
            if parent_id is not None:
                _validate_artifact_integrity(
                    artifact_by_id[parent_id],
                    artifact,
                )

        visiting = set()
        visited = set()

        def visit(artifact_id):
            if artifact_id in visiting:
                raise ValueError('Artifact lineage must not contain a cycle.')
            if artifact_id in visited:
                return

            visiting.add(artifact_id)
            parent_id = artifact_by_id[artifact_id].parent_artifact_id
            if parent_id is not None:
                visit(parent_id)
            visiting.remove(artifact_id)
            visited.add(artifact_id)

        for artifact_id in artifact_by_id:
            visit(artifact_id)

    def _validate_tasks(self, task_by_id, artifact_by_id):
        """Make every task's canonical interpretation explicit."""
        del task_by_id  # Uniqueness was established by ``_index_unique``.
        for task in self.tasks:
            if task.canonical_artifact_id not in artifact_by_id:
                raise ValueError(
                    f"Task '{task.task_id}' references missing canonical "
                    f"artifact '{task.canonical_artifact_id}'."
                )
            artifact = artifact_by_id[task.canonical_artifact_id]
            canonical_sense = _canonical_artifact_sense(artifact)
            if canonical_sense is not None and task.sense != canonical_sense:
                raise ValueError(
                    f"Task '{task.task_id}' sense '{task.sense}' disagrees "
                    f"with canonical artifact '{artifact.artifact_id}' sense "
                    f"'{canonical_sense}'."
                )


@dataclass(frozen=True)
class ProblemSet:
    """A stable, duplicate-free collection of problem cases."""

    cases: tuple[ProblemCase, ...] = ()

    def __post_init__(self):
        cases = _coerce_tuple(self.cases, ProblemCase, 'cases')
        _index_unique(cases, lambda item: item.problem_id, 'problem_id')
        object.__setattr__(self, 'cases', cases)

    def __iter__(self):
        return iter(self.cases)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, key):
        """Support both stable-ID and ordinary positional lookup."""
        if isinstance(key, str):
            return self.get(key)
        return self.cases[key]

    def get(self, problem_id: str) -> ProblemCase:
        """Return one case by ID."""
        for case in self.cases:
            if case.problem_id == problem_id:
                return case
        raise KeyError(f"Problem set has no case '{problem_id}'.")


def _validate_known_representation(representation, payload):
    """Check only the stable envelope fields owned by this package.

    Detailed CBQM/QUBO mathematical validation remains in ``lib.contracts``.
    Repeating it here would create two schema authorities and would also make a
    lightweight data-reader import the portfolio stack.
    """
    if (
        representation in _KNOWN_MODEL_REPRESENTATIONS
        and payload.get('schema') != representation
    ):
        raise ValueError(
            f"Artifact representation '{representation}' requires payload "
            f"schema '{representation}'."
        )


def _canonical_artifact_sense(artifact):
    """Read objective sense only where the representation defines one."""
    if artifact.representation == 'qubo.v1':
        return artifact.payload.get('sense')
    if artifact.representation == 'cbqm.v1':
        objective = artifact.payload.get('objective')
        return objective.get('sense') if isinstance(objective, Mapping) else None
    return None


def _validate_artifact_integrity(parent, artifact):
    """Verify optional lineage hashes without making them mandatory metadata."""
    integrity = artifact.transformation.context.get('integrity')
    if integrity is None:
        return
    if not isinstance(integrity, Mapping):
        raise TypeError('transformation integrity must be a mapping.')

    expected_source = integrity.get('source_sha256')
    expected_target = integrity.get('target_sha256')
    if expected_source != _payload_sha256(parent.payload):
        raise ValueError(
            f"Artifact '{artifact.artifact_id}' source integrity hash mismatch."
        )
    if expected_target != _payload_sha256(artifact.payload):
        raise ValueError(
            f"Artifact '{artifact.artifact_id}' target integrity hash mismatch."
        )


def _payload_sha256(payload):
    """Hash the same canonical JSON form used by case transformations."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def _coerce_tuple(value, item_type, label):
    """Normalize public sequences while rejecting ambiguous strings/mappings."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise TypeError(f'{label} must be a sequence.')
    output = tuple(value)
    if any(not isinstance(item, item_type) for item in output):
        raise TypeError(f'Every item in {label} must be {item_type.__name__}.')
    return output


def _index_unique(items, key, label):
    """Build an index and explain duplicate stable identifiers."""
    output = {}
    for item in items:
        item_key = key(item)
        if item_key in output:
            raise ValueError(f"Duplicate {label} '{item_key}'.")
        output[item_key] = item
    return output


def _validate_identifier(value, label):
    """Validate identifiers that are also safe artifact filename stems."""
    _validate_non_empty_string(value, label)
    if not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f'{label} may contain only letters, digits, dot, underscore and dash.'
        )


def _validate_non_empty_string(value, label):
    """Validate a human-readable or machine-readable non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a non-empty string.')


def _validate_finite_number(value, label):
    """Reject booleans, NaN and infinity before they reach persisted JSON."""
    if type(value) is int:
        return
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f'{label} must be a finite real number.')


def _json_mapping_copy(value, label):
    """Validate a mapping and detach it from caller-owned mutable objects."""
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping.')
    copied = _json_copy(dict(value), label)
    if not isinstance(copied, dict):
        raise TypeError(f'{label} must serialize as a JSON object.')
    return copied


def _json_copy(value, label):
    """Normalize and deep-copy one JSON-compatible value.

    A JSON round-trip is intentional: it catches non-string object keys and
    non-finite numbers now, instead of failing halfway through an atomic save.
    """
    _validate_json_object_keys(value, label)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(serialized)
    except (TypeError, ValueError) as error:
        raise TypeError(f'{label} must be JSON-compatible: {error}') from error


def _validate_json_object_keys(value, label):
    """Reject keys ``json.dumps`` would silently coerce to strings."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f'{label} JSON object keys must be strings; got '
                    f'{key!r}.'
                )
            _validate_json_object_keys(item, f"{label}['{key}']")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_object_keys(item, f'{label}[{index}]')
