"""High-level operations that add derived artifacts to a problem case.

The public functions intentionally keep orchestration short.  Validation,
context checks and artifact construction live in protected helpers, so a
typical workflow reads as:

``compile_case_qubo(...)``
``derive_qubo_interaction_graph(...)``
``derive_qubo_maxcut_graph(...)``
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence

from .graph_codec import artifact_from_networkx, graph_to_node_link
from .problem_def import (
    ProblemArtifact,
    ProblemCase,
    TransformationRecord,
)
from .transforms import (
    cbqm_to_factor_graph,
    qubo_to_interaction_graph,
    qubo_to_maxcut_graph,
)


def create_graph_case(
    problem_id,
    graph,
    *,
    artifact_id='graph',
    representation='networkx.node-link.v1',
    metadata=None,
    artifact_metadata=None,
) -> ProblemCase:
    """Create a task-free case from a native NetworkX graph.

    No objective is inferred.  A caller may later attach one or more tasks, or
    transform the graph into a business-specific model.
    """
    artifact = artifact_from_networkx(
        artifact_id,
        graph,
        representation=representation,
        metadata=artifact_metadata,
    )
    return ProblemCase(
        problem_id=problem_id,
        artifacts=(artifact,),
        primary_artifact_id=artifact_id,
        metadata={} if metadata is None else metadata,
    )


def compile_case_qubo(
    problem_case,
    source_artifact_id,
    config,
    *,
    target_artifact_id='qubo',
    compiler=None,
) -> ProblemCase:
    """Compile a ``cbqm.v1`` artifact and append its ``qubo.v1`` descendant.

    ``compiler`` is injectable for tests and alternative compatible compilers.
    The normal path lazily imports the repository's canonical compiler, keeping
    basic case reading independent from portfolio and pandas dependencies.
    """
    source = _require_artifact(
        problem_case,
        source_artifact_id,
        'cbqm.v1',
    )
    _validate_source_cbqm_contract(source.payload)
    resolved_compiler = (
        _load_default_qubo_compiler()
        if compiler is None
        else compiler
    )
    if not callable(resolved_compiler):
        raise TypeError('compiler must be callable.')
    if not isinstance(config, Mapping):
        raise TypeError('compiler config must be a mapping.')

    # A pluggable compiler is still an external boundary: give it detached
    # inputs so an implementation cannot mutate the case or the caller's
    # nested configuration while compiling.
    compiler_source = copy.deepcopy(dict(source.payload))
    compiler_config = copy.deepcopy(dict(config))
    qubo, compilation_context = resolved_compiler(
        compiler_source,
        compiler_config,
    )
    _validate_compiled_qubo_contract(qubo)
    _validate_compilation_result(
        source.payload,
        qubo,
        compilation_context,
    )

    transformation_context = {
        'requested_config': dict(config),
        'compilation': compilation_context,
        'model_equivalence': 'conditional',
        'exactness_note': (
            'An optimal derived QUBO result must be decoded and checked against '
            'the source CBQM before it can certify the canonical task.'
        ),
        'integrity': {
            'source_sha256': _payload_sha256(source.payload),
            'target_sha256': _payload_sha256(qubo),
        },
    }
    artifact = ProblemArtifact(
        artifact_id=target_artifact_id,
        representation='qubo.v1',
        payload=qubo,
        parent_artifact_id=source.artifact_id,
        transformation=TransformationRecord(
            name='compile_qubo',
            version=_compiler_version(qubo),
            lossless=False,
            context=transformation_context,
        ),
        metadata={
            'model_equivalence': 'conditional',
        },
    )
    return problem_case.with_artifact(artifact)


def derive_cbqm_factor_graph(
    problem_case,
    source_artifact_id,
    *,
    target_artifact_id='cbqm-factor-graph',
) -> ProblemCase:
    """Append a lossless NetworkX factor-graph view of a CBQM artifact."""
    source = _require_artifact(
        problem_case,
        source_artifact_id,
        'cbqm.v1',
    )
    graph = cbqm_to_factor_graph(source.payload)
    artifact = _derived_graph_artifact(
        target_artifact_id,
        'networkx.cbqm-factor.v1',
        graph,
        source,
        name='cbqm_to_factor_graph',
        lossless=True,
        context={
            'source_schema': 'cbqm.v1',
            'factor_graph_semantics': 'bipartite-variable-and-factor',
        },
    )
    return problem_case.with_artifact(artifact)


def derive_qubo_interaction_graph(
    problem_case,
    source_artifact_id,
    *,
    target_artifact_id='qubo-interaction-graph',
) -> ProblemCase:
    """Append a lossless variable-interaction view of a QUBO artifact."""
    source = _require_artifact(
        problem_case,
        source_artifact_id,
        'qubo.v1',
    )
    graph = qubo_to_interaction_graph(source.payload)
    artifact = _derived_graph_artifact(
        target_artifact_id,
        'networkx.qubo-interaction.v1',
        graph,
        source,
        name='qubo_to_interaction_graph',
        lossless=True,
        context={
            'source_schema': 'qubo.v1',
            'diagonal_storage': 'node.linear_bias',
            'off_diagonal_storage': 'edge.quadratic_bias',
        },
    )
    return problem_case.with_artifact(artifact)


def derive_qubo_maxcut_graph(
    problem_case,
    source_artifact_id,
    *,
    target_artifact_id='qubo-maxcut-graph',
) -> ProblemCase:
    """Append the signed, anchor-node MaxCut encoding used by Q-RBnBR."""
    source = _require_artifact(
        problem_case,
        source_artifact_id,
        'qubo.v1',
    )
    graph = qubo_to_maxcut_graph(source.payload)
    artifact = _derived_graph_artifact(
        target_artifact_id,
        'networkx.qubo-maxcut.v1',
        graph,
        source,
        name='qubo_to_maxcut_graph',
        lossless=True,
        context={
            'source_schema': 'qubo.v1',
            'anchor_node': source.payload['num_variables'],
            'energy_mapping': {
                'source_value': 'qubo_energy',
                'target_value': 'cut_value',
                'scale': -1.0,
                # The offset is metadata for the affine energy relation; it
                # never needs binary64 arithmetic while this artifact is
                # created.  Preserve the canonical JSON number exactly so a
                # large integer cannot be rounded (or overflow) merely because
                # the equivalent graph view was requested.
                'shift': copy.deepcopy(source.payload['offset']),
            },
        },
    )
    return problem_case.with_artifact(artifact)


def project_qubo_sample_to_cbqm(
    problem_case,
    qubo_artifact_id,
    sample,
) -> list[int]:
    """Project original/free variables back to the source CBQM order.

    Slack values are deliberately ignored.  This operation only decodes a
    candidate; callers must still use the CBQM task evaluator to establish
    feasibility and the original objective.
    """
    qubo_artifact, _, context = _validated_qubo_compilation(
        problem_case,
        qubo_artifact_id,
    )
    target_sample = _validate_binary_vector(
        sample,
        qubo_artifact.payload['num_variables'],
        'QUBO sample',
    )

    source_count = context['source_variable_count']
    source_sample = [None] * source_count
    for fixed in context['fixed_values']:
        source_sample[fixed['index']] = fixed['value']
    for free in context['free_variables']:
        source_sample[free['cbqm_index']] = target_sample[free['qubo_index']]

    if any(value is None for value in source_sample):
        raise ValueError(
            'Compilation context does not cover every source CBQM variable.'
        )
    return source_sample


def _validated_qubo_compilation(
    problem_case,
    qubo_artifact_id,
):
    """Return a QUBO, its direct CBQM parent and trusted compiler context.

    Persisted transformation metadata is input, not authority.  A hand-edited
    manifest could otherwise redirect variable indices while retaining the
    words ``compile_qubo``.  Every decode therefore rechecks representation,
    direct lineage, hashes, model identities and both index partitions.
    """
    qubo_artifact = _require_artifact(
        problem_case,
        qubo_artifact_id,
        'qubo.v1',
    )
    transformation = qubo_artifact.transformation
    if (
        transformation is None
        or transformation.name != 'compile_qubo'
    ):
        raise ValueError(
            'QUBO artifact does not carry a compile_qubo transformation.'
        )
    if transformation.lossless:
        raise ValueError(
            'A compile_qubo transformation cannot be marked lossless.'
        )

    parent_id = qubo_artifact.parent_artifact_id
    if parent_id is None:
        raise ValueError('Compiled QUBO artifact has no direct parent.')
    parent = problem_case.get_artifact(parent_id)
    if parent.representation != 'cbqm.v1':
        raise ValueError(
            'A compile_qubo transformation must have a direct cbqm.v1 parent.'
        )
    # Stored artifacts are untrusted JSON too. Validate both mathematical
    # contracts before any context indices are used for projection or before
    # an arbitrary solver receives the derived QUBO.
    _validate_source_cbqm_contract(parent.payload)
    _validate_compiled_qubo_contract(qubo_artifact.payload)

    outer_context = transformation.context
    context = outer_context.get('compilation')
    if not isinstance(context, Mapping):
        raise TypeError('Compilation transformation has no usable context.')
    _validate_compilation_integrity(
        parent,
        qubo_artifact,
        outer_context,
    )
    _validate_compilation_result(
        parent.payload,
        qubo_artifact.payload,
        context,
    )

    requested_config = outer_context.get('requested_config')
    _validate_requested_compiler_config(
        requested_config,
        context.get('compiler_config'),
    )
    return qubo_artifact, parent, context


def _derived_graph_artifact(
    artifact_id,
    representation,
    graph,
    source,
    *,
    name,
    lossless,
    context,
):
    """Create a graph artifact with consistent lineage and integrity data."""
    payload = graph_to_node_link(graph)
    enriched_context = {
        **context,
        'integrity': {
            'source_sha256': _payload_sha256(source.payload),
            'target_sha256': _payload_sha256(payload),
        },
    }
    return ProblemArtifact(
        artifact_id=artifact_id,
        representation=representation,
        payload=payload,
        parent_artifact_id=source.artifact_id,
        transformation=TransformationRecord(
            name=name,
            version='1',
            lossless=lossless,
            context=enriched_context,
        ),
    )


def _require_artifact(problem_case, artifact_id, representation):
    """Resolve one source and make the transformation contract obvious."""
    if not isinstance(problem_case, ProblemCase):
        raise TypeError('problem_case must be a ProblemCase.')
    artifact = problem_case.get_artifact(artifact_id)
    if artifact.representation != representation:
        raise ValueError(
            f"Artifact '{artifact_id}' must use representation "
            f"'{representation}', not '{artifact.representation}'."
        )
    return artifact


def _load_default_qubo_compiler():
    """Import the heavy application package only when compilation is requested."""
    from lib.compilers.qubo_compiler import compile_qubo

    return compile_qubo


def _validate_compiled_qubo_contract(qubo):
    """Apply the full public target contract to any injectable compiler."""
    from lib.contracts import validate_qubo

    validate_qubo(qubo)


def _validate_source_cbqm_contract(cbqm):
    """Reject a malformed source before invoking a pluggable compiler."""
    from lib.contracts import validate_cbqm

    validate_cbqm(cbqm)


def _validate_compilation_result(cbqm, qubo, context):
    """Detect a compiler context accidentally paired with another artifact."""
    if not isinstance(qubo, Mapping):
        raise TypeError('compiler must return a QUBO mapping.')
    if qubo.get('schema') != 'qubo.v1':
        raise ValueError("compiler target schema must be 'qubo.v1'.")
    if not isinstance(context, Mapping):
        raise TypeError('compiler context must be a mapping.')
    if context.get('schema') != 'qubo-compilation-context.v1':
        raise ValueError(
            "compiler context schema must be 'qubo-compilation-context.v1'."
        )
    if qubo.get('problem_id') != cbqm.get('problem_id'):
        raise ValueError('Compiled QUBO problem_id does not match its CBQM.')
    if context.get('source_problem_id') != cbqm.get('problem_id'):
        raise ValueError('Compilation context problem_id does not match CBQM.')

    variables = cbqm.get('variables')
    if not isinstance(variables, list):
        raise TypeError("CBQM field 'variables' must be a list.")
    for position, variable in enumerate(variables):
        if not isinstance(variable, Mapping):
            raise TypeError('Every CBQM variable must be a mapping.')
        if variable.get('index') != position:
            raise ValueError(
                'CBQM variable indices must match their list positions.'
            )
        if not isinstance(variable.get('name'), str) or not variable['name']:
            raise ValueError('CBQM variable names must be non-empty strings.')

    source_names = [
        variable.get('name')
        for variable in variables
    ]
    if context.get('source_variable_names') != source_names:
        raise ValueError('Compilation context source variable names mismatch.')
    if (
        type(context.get('source_variable_count')) is not int
        or context.get('source_variable_count') != len(source_names)
    ):
        raise ValueError('Compilation context source variable count mismatch.')
    if (
        type(context.get('qubo_variable_count')) is not int
        or context.get('qubo_variable_count') != qubo.get('num_variables')
    ):
        raise ValueError('Compilation context QUBO variable count mismatch.')
    if context.get('qubo_variable_names') != qubo.get('variable_names'):
        raise ValueError('Compilation context QUBO variable names mismatch.')

    objective = cbqm.get('objective')
    if not isinstance(objective, Mapping):
        raise TypeError("CBQM field 'objective' must be a mapping.")
    objective_sense = objective.get('sense')
    expected_multiplier = 1.0 if objective_sense == 'minimize' else -1.0
    if objective_sense not in {'minimize', 'maximize'}:
        raise ValueError('CBQM objective has an invalid sense.')
    if context.get('objective_sense') != objective_sense:
        raise ValueError('Compilation context objective sense mismatch.')
    objective_multiplier = context.get('objective_multiplier')
    if (
        type(objective_multiplier) not in {int, float}
        or objective_multiplier != expected_multiplier
    ):
        raise ValueError('Compilation context objective multiplier mismatch.')

    qubo_metadata = qubo.get('metadata', {})
    if not isinstance(qubo_metadata, Mapping):
        raise TypeError("QUBO field 'metadata' must be a mapping.")
    if context.get('compiler_config') != qubo_metadata.get('compiler_config'):
        raise ValueError('Compiler config differs between QUBO and context.')
    _validate_compilation_variable_coverage(cbqm, qubo, context)


def _validate_compilation_variable_coverage(cbqm, qubo, context):
    """Check that source and target index partitions are complete and unique."""
    variables = cbqm['variables']
    source_indices = set(range(len(variables)))
    fixed_values = _require_mapping_list(
        context.get('fixed_values'),
        'compilation fixed_values',
    )
    free_variables = _require_mapping_list(
        context.get('free_variables'),
        'compilation free_variables',
    )
    slack_variables = _require_mapping_list(
        context.get('slack_variables'),
        'compilation slack_variables',
    )

    fixed_indices = []
    for item in fixed_values:
        index = _validate_partition_index(
            item.get('index'),
            len(variables),
            'fixed CBQM index',
        )
        value = item.get('value')
        if type(value) is not int or value not in {0, 1}:
            raise ValueError('Compiled fixed values must be integer 0 or 1.')
        fixed_indices.append(index)

    if fixed_values != cbqm.get('fixed_values'):
        raise ValueError(
            'Compilation context fixed values do not match the source CBQM.'
        )

    qubo_count = qubo.get('num_variables')
    if (
        not isinstance(qubo_count, int)
        or isinstance(qubo_count, bool)
        or qubo_count < 0
    ):
        raise ValueError('Compiled QUBO has an invalid variable count.')

    free_source_indices = []
    free_target_indices = []
    for item in free_variables:
        cbqm_index = _validate_partition_index(
            item.get('cbqm_index'),
            len(variables),
            'free CBQM index',
        )
        qubo_index = _validate_partition_index(
            item.get('qubo_index'),
            qubo_count,
            'free QUBO index',
        )
        expected_name = variables[cbqm_index]['name']
        if item.get('name') != expected_name:
            raise ValueError(
                'Compilation free-variable name does not match its CBQM index.'
            )
        if qubo['variable_names'][qubo_index] != expected_name:
            raise ValueError(
                'Compilation free-variable mapping does not match QUBO names.'
            )
        free_source_indices.append(cbqm_index)
        free_target_indices.append(qubo_index)

    _require_exact_partition(
        [*fixed_indices, *free_source_indices],
        source_indices,
        'CBQM variables in compilation context',
    )

    slack_target_indices = []
    for item in slack_variables:
        qubo_index = _validate_partition_index(
            item.get('qubo_index'),
            qubo_count,
            'slack QUBO index',
        )
        name = item.get('name')
        if not isinstance(name, str) or not name.startswith('__slack__'):
            raise ValueError(
                'Compilation slack-variable names must use __slack__.'
            )
        if qubo['variable_names'][qubo_index] != name:
            raise ValueError(
                'Compilation slack-variable mapping does not match QUBO names.'
            )
        slack_target_indices.append(qubo_index)

    _require_exact_partition(
        [*free_target_indices, *slack_target_indices],
        set(range(qubo_count)),
        'QUBO variables in compilation context',
    )

    fixed_index_map = context.get('fixed_values_by_index')
    if fixed_index_map is not None:
        if not isinstance(fixed_index_map, Mapping):
            raise TypeError(
                'Compilation fixed_values_by_index must be a mapping.'
            )
        expected_map = {
            item['index']: item['value']
            for item in fixed_values
        }
        normalized_map = {
            int(index): value
            for index, value in fixed_index_map.items()
        }
        if normalized_map != expected_map:
            raise ValueError(
                'Compilation fixed-value lookup disagrees with fixed_values.'
            )

    qubo_index_map = context.get('qubo_index_by_cbqm_index')
    if qubo_index_map is not None:
        if not isinstance(qubo_index_map, Mapping):
            raise TypeError(
                'Compilation qubo_index_by_cbqm_index must be a mapping.'
            )
        expected_map = {
            item['cbqm_index']: item['qubo_index']
            for item in free_variables
        }
        normalized_map = {
            int(index): value
            for index, value in qubo_index_map.items()
        }
        if normalized_map != expected_map:
            raise ValueError(
                'Compilation variable lookup disagrees with free_variables.'
            )


def _validate_compilation_integrity(parent, qubo_artifact, outer_context):
    """Require source and target hashes on executable compiler lineage."""
    integrity = outer_context.get('integrity')
    if not isinstance(integrity, Mapping):
        raise ValueError(
            'Compiled QUBO transformation must record integrity hashes.'
        )
    if integrity.get('source_sha256') != _payload_sha256(parent.payload):
        raise ValueError('Compiled QUBO source integrity hash mismatch.')
    if integrity.get('target_sha256') != _payload_sha256(
        qubo_artifact.payload
    ):
        raise ValueError('Compiled QUBO target integrity hash mismatch.')


def _validate_requested_compiler_config(requested, resolved):
    """Check the caller's options against the compiler's resolved defaults.

    The canonical compiler records a fully resolved configuration in its
    context, while ``requested_config`` intentionally preserves only what the
    caller supplied.  Requiring whole-dictionary equality would therefore make
    every ordinary call such as ``{'default_penalty': 4.0}`` undecodable.
    """
    if not isinstance(requested, Mapping):
        raise TypeError('Requested compiler config must be a mapping.')
    if not isinstance(resolved, Mapping):
        raise TypeError('Resolved compiler config must be a mapping.')

    inconsistent = sorted(
        key
        for key, value in requested.items()
        if key not in resolved or resolved[key] != value
    )
    if inconsistent:
        raise ValueError(
            'Requested compiler config differs from its resolved context for '
            f'fields: {inconsistent}.'
        )


def _require_mapping_list(value, label):
    """Normalize a compiler list while keeping error messages local."""
    if not isinstance(value, list):
        raise TypeError(f'{label} must be a list.')
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError(f'Every item in {label} must be a mapping.')
    return value


def _validate_partition_index(value, upper_bound, label):
    """Reject bools and out-of-range indices before set partition checks."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value >= upper_bound
    ):
        raise ValueError(f'{label} is outside its variable range.')
    return value


def _require_exact_partition(indices, expected, label):
    """Require full coverage with no duplicate ownership."""
    if len(indices) != len(set(indices)) or set(indices) != expected:
        raise ValueError(f'{label} must form a complete, duplicate-free partition.')


def _validate_binary_vector(sample, expected_length, label):
    """Validate a solver sample before reading indices from it."""
    if (
        not isinstance(expected_length, int)
        or isinstance(expected_length, bool)
        or expected_length < 0
    ):
        raise ValueError(f'{label} context length is invalid.')
    if (
        not isinstance(sample, Sequence)
        or isinstance(sample, (str, bytes, bytearray))
    ):
        raise TypeError(f'{label} must be a sequence.')
    values = list(sample)
    if len(values) != expected_length:
        raise ValueError(f'{label} length does not match its QUBO artifact.')
    if any(type(value) is not int or value not in {0, 1} for value in values):
        raise ValueError(f'{label} must contain integer 0 or 1.')
    return values


def _compiler_version(qubo):
    """Reuse the compiler's own stable identifier where available."""
    metadata = qubo.get('metadata', {})
    if isinstance(metadata, Mapping):
        compiler = metadata.get('compiler')
        if isinstance(compiler, str) and compiler:
            return compiler
    return 'unknown'


def _payload_sha256(payload):
    """Hash canonical JSON so contexts cannot silently drift between models."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()
