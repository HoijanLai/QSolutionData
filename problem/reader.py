"""Persistence for :mod:`problem` cases and their external artifact payloads.

Each case is a small manifest plus one JSON file per artifact::

    problem1/
      case.json
      artifacts/
        source-graph.json
        cbqm.json
        qubo-penalty-10.json

Keeping artifacts external makes lineage readable in code review and avoids
rewriting a large graph merely because one task's best-known solution changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .problem_def import (
    PROBLEM_CASE_SCHEMA,
    BestKnownSolution,
    ProblemArtifact,
    ProblemCase,
    ProblemSet,
    TaskDefinition,
    TransformationRecord,
)

DEFAULT_CASE_FILENAME = 'case.json'
DEFAULT_ARTIFACT_DIRECTORY = 'artifacts'

_CASE_REQUIRED_FIELDS = {'schema', 'problem_id', 'artifacts'}
_CASE_OPTIONAL_FIELDS = {
    'primary_artifact_id',
    'tasks',
    'metadata',
}
_ARTIFACT_REQUIRED_FIELDS = {
    'artifact_id',
    'representation',
    'path',
}
_ARTIFACT_OPTIONAL_FIELDS = {
    'parent_artifact_id',
    'transformation',
    'metadata',
}
_TRANSFORMATION_REQUIRED_FIELDS = {'name', 'version', 'lossless'}
_TRANSFORMATION_OPTIONAL_FIELDS = {'context'}
_TASK_REQUIRED_FIELDS = {
    'task_id',
    'canonical_artifact_id',
    'sense',
}
_TASK_OPTIONAL_FIELDS = {
    'task_type',
    'solution_representation',
    'best_known',
    'metadata',
}
_BEST_KNOWN_REQUIRED_FIELDS = {'objective_value'}
_BEST_KNOWN_OPTIONAL_FIELDS = {
    'solution',
    'sample',
    'exact',
    'source',
    'metadata',
}


def load_problem_case(path) -> ProblemCase:
    """Load one case from ``case.json`` or its containing directory."""
    case_path = _resolve_case_path(path)
    manifest = _read_json(case_path)
    return case_from_dict(manifest, base_directory=case_path.parent)


def load_problem_set(data_directory) -> ProblemSet:
    """Load every immediate ``*/case.json`` child in stable folder order."""
    root = Path(data_directory)
    if not root.is_dir():
        raise FileNotFoundError(f'Problem data directory not found: {root}')

    case_paths = sorted(
        root.glob(f'*/{DEFAULT_CASE_FILENAME}'),
        key=lambda item: item.parent.name,
    )
    return ProblemSet(tuple(load_problem_case(path) for path in case_paths))


def save_problem_case(problem_case, path) -> Path:
    """Atomically save artifact payloads followed by the case manifest.

    The manifest is replaced last.  A reader therefore sees either the former
    complete case or the newly completed case, never a manifest that points to
    artifact files which have not yet been written.
    """
    if not isinstance(problem_case, ProblemCase):
        raise TypeError('problem_case must be a ProblemCase.')

    case_path = _resolve_output_path(path)
    case_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_directory = case_path.parent / DEFAULT_ARTIFACT_DIRECTORY
    artifact_directory.mkdir(parents=True, exist_ok=True)
    for artifact in problem_case.artifacts:
        artifact_path = artifact_directory / f'{artifact.artifact_id}.json'
        _write_json_atomically(artifact_path, artifact.payload)

    manifest = case_to_dict(problem_case)
    _write_json_atomically(case_path, manifest)
    return case_path


def case_from_dict(payload, *, base_directory) -> ProblemCase:
    """Build a validated case from a manifest and its artifact directory."""
    if not isinstance(payload, dict):
        raise TypeError('Problem case manifest must be a JSON object.')
    _validate_object_fields(
        payload,
        required=_CASE_REQUIRED_FIELDS,
        optional=_CASE_OPTIONAL_FIELDS,
        label='case',
    )
    if payload.get('schema') != PROBLEM_CASE_SCHEMA:
        raise ValueError(
            f"Problem case schema must be '{PROBLEM_CASE_SCHEMA}'."
        )

    artifact_descriptors = payload.get('artifacts')
    if not isinstance(artifact_descriptors, list):
        raise TypeError("case['artifacts'] must be a list.")
    artifacts = tuple(
        _artifact_from_descriptor(item, base_directory)
        for item in artifact_descriptors
    )

    task_payloads = payload.get('tasks', [])
    if not isinstance(task_payloads, list):
        raise TypeError("case['tasks'] must be a list.")
    tasks = tuple(_task_from_dict(item) for item in task_payloads)

    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict):
        raise TypeError("case['metadata'] must be an object.")
    return ProblemCase(
        problem_id=payload.get('problem_id'),
        artifacts=artifacts,
        tasks=tasks,
        primary_artifact_id=payload.get('primary_artifact_id'),
        metadata=metadata,
    )


def case_to_dict(problem_case) -> dict:
    """Convert an in-memory case to its public manifest representation."""
    if not isinstance(problem_case, ProblemCase):
        raise TypeError('problem_case must be a ProblemCase.')

    return {
        'schema': PROBLEM_CASE_SCHEMA,
        'problem_id': problem_case.problem_id,
        'primary_artifact_id': problem_case.primary_artifact_id,
        'artifacts': [
            _artifact_to_descriptor(artifact)
            for artifact in problem_case.artifacts
        ],
        'tasks': [_task_to_dict(task) for task in problem_case.tasks],
        'metadata': dict(problem_case.metadata),
    }


def _artifact_from_descriptor(payload, base_directory):
    """Load one artifact descriptor and the payload it references."""
    if not isinstance(payload, dict):
        raise TypeError('Every artifact descriptor must be an object.')
    _validate_object_fields(
        payload,
        required=_ARTIFACT_REQUIRED_FIELDS,
        optional=_ARTIFACT_OPTIONAL_FIELDS,
        label='artifact descriptor',
    )
    relative_path = payload.get('path')
    artifact_path = _resolve_artifact_path(base_directory, relative_path)
    artifact_payload = _read_json(artifact_path)

    transformation_payload = payload.get('transformation')
    transformation = (
        None
        if transformation_payload is None
        else _transformation_from_dict(transformation_payload)
    )
    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict):
        raise TypeError("artifact['metadata'] must be an object.")

    return ProblemArtifact(
        artifact_id=payload.get('artifact_id'),
        representation=payload.get('representation'),
        payload=artifact_payload,
        parent_artifact_id=payload.get('parent_artifact_id'),
        transformation=transformation,
        metadata=metadata,
    )


def _artifact_to_descriptor(artifact):
    """Serialize lineage separately from the possibly-large payload."""
    return {
        'artifact_id': artifact.artifact_id,
        'representation': artifact.representation,
        'path': (
            f'{DEFAULT_ARTIFACT_DIRECTORY}/'
            f'{artifact.artifact_id}.json'
        ),
        'parent_artifact_id': artifact.parent_artifact_id,
        'transformation': _transformation_to_dict(artifact.transformation),
        'metadata': dict(artifact.metadata),
    }


def _transformation_from_dict(payload):
    """Parse one explicit transformation record."""
    if not isinstance(payload, dict):
        raise TypeError("artifact['transformation'] must be an object or null.")
    _validate_object_fields(
        payload,
        required=_TRANSFORMATION_REQUIRED_FIELDS,
        optional=_TRANSFORMATION_OPTIONAL_FIELDS,
        label='transformation',
    )
    context = payload.get('context', {})
    if not isinstance(context, dict):
        raise TypeError("transformation['context'] must be an object.")
    return TransformationRecord(
        name=payload.get('name'),
        version=payload.get('version'),
        lossless=payload.get('lossless'),
        context=context,
    )


def _transformation_to_dict(transformation):
    """Serialize an optional transformation without internal objects."""
    if transformation is None:
        return None
    return {
        'name': transformation.name,
        'version': transformation.version,
        'lossless': transformation.lossless,
        'context': dict(transformation.context),
    }


def _task_from_dict(payload):
    """Parse one task and its optional task-scoped incumbent."""
    if not isinstance(payload, dict):
        raise TypeError('Every task must be an object.')
    _validate_object_fields(
        payload,
        required=_TASK_REQUIRED_FIELDS,
        optional=_TASK_OPTIONAL_FIELDS,
        label='task',
    )
    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict):
        raise TypeError("task['metadata'] must be an object.")
    return TaskDefinition(
        task_id=payload.get('task_id'),
        canonical_artifact_id=payload.get('canonical_artifact_id'),
        task_type=payload.get('task_type'),
        sense=payload.get('sense'),
        solution_representation=payload.get(
            'solution_representation',
            'binary-vector.v1',
        ),
        best_known=_best_known_from_dict(payload.get('best_known')),
        metadata=metadata,
    )


def _task_to_dict(task):
    """Serialize a task with no case-level objective assumptions."""
    return {
        'task_id': task.task_id,
        'canonical_artifact_id': task.canonical_artifact_id,
        'task_type': task.task_type,
        'sense': task.sense,
        'solution_representation': task.solution_representation,
        'best_known': _best_known_to_dict(task.best_known),
        'metadata': dict(task.metadata),
    }


def _best_known_from_dict(payload):
    """Parse the optional, generic solution witness."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("task['best_known'] must be an object or null.")
    _validate_object_fields(
        payload,
        required=_BEST_KNOWN_REQUIRED_FIELDS,
        optional=_BEST_KNOWN_OPTIONAL_FIELDS,
        label='best_known',
    )
    if 'solution' not in payload and 'sample' not in payload:
        raise ValueError(
            "best_known must define 'solution' (or legacy 'sample')."
        )
    if 'solution' in payload and 'sample' in payload:
        raise ValueError(
            "best_known must not define both 'solution' and legacy 'sample'."
        )
    metadata = payload.get('metadata', {})
    if not isinstance(metadata, dict):
        raise TypeError("best_known['metadata'] must be an object.")

    # ``sample`` is accepted solely as an upgrade path from the first local
    # graph-problem draft.  New manifests always write the generic ``solution``.
    solution = payload.get('solution', payload.get('sample'))
    return BestKnownSolution(
        solution=solution,
        objective_value=payload.get('objective_value'),
        exact=payload.get('exact', False),
        source=payload.get('source'),
        metadata=metadata,
    )


def _best_known_to_dict(best_known):
    """Serialize a task incumbent without assuming a binary-vector shape."""
    if best_known is None:
        return None
    return {
        'solution': best_known.solution,
        'objective_value': best_known.objective_value,
        'exact': best_known.exact,
        'source': best_known.source,
        'metadata': dict(best_known.metadata),
    }


def _resolve_case_path(path):
    """Resolve directory shorthand without guessing alternate filenames."""
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / DEFAULT_CASE_FILENAME
    if not resolved.is_file():
        raise FileNotFoundError(f'Problem case file not found: {resolved}')
    return resolved


def _resolve_output_path(path):
    """Treat existing directories and suffix-less paths as case folders."""
    resolved = Path(path)
    if resolved.is_dir() or not resolved.suffix:
        return resolved / DEFAULT_CASE_FILENAME
    return resolved


def _resolve_artifact_path(base_directory, relative_path):
    """Resolve a manifest path while preventing absolute/traversal escapes."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact['path'] must be a non-empty relative path.")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError('Artifact paths must be relative to the case directory.')

    base = Path(base_directory).resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(
            'Artifact path must remain inside the case directory.'
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(f'Artifact payload not found: {resolved}')
    return resolved


def _read_json(path):
    """Read one UTF-8 JSON object with contextual parse errors."""
    try:
        with Path(path).open('r', encoding='utf-8') as stream:
            payload = json.load(
                stream,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_non_finite_json_constant,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid JSON in {path}: {error}') from error
    except ValueError as error:
        raise ValueError(f'Invalid JSON in {path}: {error}') from error
    if not isinstance(payload, dict):
        raise TypeError(f'JSON file must contain an object: {path}')
    return payload


def _unique_json_object(pairs):
    """Build one JSON object while rejecting duplicate property names."""
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f'Duplicate JSON object key {key!r}.')
        output[key] = value
    return output


def _reject_non_finite_json_constant(value):
    """Reject the non-standard NaN/Infinity extensions accepted by ``json``."""
    raise ValueError(f'Non-finite JSON constant {value!r} is not allowed.')


def _validate_object_fields(payload, *, required, optional, label):
    """Keep persisted envelopes closed and explain missing/unknown fields."""
    non_string_fields = [
        field
        for field in payload
        if not isinstance(field, str)
    ]
    if non_string_fields:
        rendered = ', '.join(repr(field) for field in non_string_fields)
        raise TypeError(
            f'{label} field names must be strings; got: {rendered}.'
        )
    fields = set(payload)
    missing = sorted(required - fields)
    if missing:
        raise ValueError(
            f'{label} is missing required fields: {", ".join(missing)}.'
        )
    unknown = sorted(fields - required - optional)
    if unknown:
        raise ValueError(
            f'{label} contains unknown fields: {", ".join(unknown)}.'
        )


def _write_json_atomically(path, payload):
    """Replace one JSON file only after its temporary write is complete."""
    path = Path(path)
    temporary_path = path.with_name(f'.{path.name}.tmp')
    try:
        with temporary_path.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write('\n')
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# Concise compatibility aliases: callers can say "problem" while the actual
# type remains explicitly named ``ProblemCase``.
load_problem = load_problem_case
save_problem = save_problem_case
problem_to_dict = case_to_dict
