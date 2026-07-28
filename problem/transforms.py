"""Lossless structural graph views and an exact QUBO-to-MaxCut encoding.

These functions transform model *representations*.  They do not create tasks
and do not move best-known solutions; that orchestration belongs to
``case_operations.py``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import networkx as nx

from lib.adapters.maxcut_encoding import build_signed_maxcut_edge_weights
from lib.contracts import (
    validate_cbqm as _validate_cbqm,
    validate_qubo as _validate_qubo,
)

from .graph_codec import graph_from_node_link
from .problem_def import ProblemArtifact

QUBO_INTERACTION_REPRESENTATION = 'networkx.qubo-interaction.v1'
CBQM_FACTOR_REPRESENTATION = 'networkx.cbqm-factor.v1'
QUBO_MAXCUT_REPRESENTATION = 'networkx.qubo-maxcut.v1'


def qubo_to_interaction_graph(qubo):
    """Map QUBO diagonal terms to nodes and pair terms to edges."""
    _validate_qubo(qubo)
    graph = _new_qubo_interaction_graph(qubo)
    _add_qubo_variable_nodes(graph, qubo)
    _add_qubo_biases(graph, qubo)
    return graph


def interaction_graph_to_qubo(graph_or_artifact):
    """Reconstruct canonical upper-triangular ``qubo.v1`` terms."""
    graph = _coerce_graph(
        graph_or_artifact,
        QUBO_INTERACTION_REPRESENTATION,
    )
    _validate_simple_undirected_graph(graph, 'QUBO interaction graph')
    _require_graph_schema(graph, QUBO_INTERACTION_REPRESENTATION)

    variable_count = _require_non_negative_int(
        graph.graph.get('num_variables'),
        'interaction graph num_variables',
    )
    node_order = _validate_index_node_order(
        graph.graph.get('node_order'),
        variable_count,
    )
    variable_names = _read_qubo_variable_nodes(
        graph,
        node_order,
    )
    terms = _read_qubo_graph_terms(graph, node_order)

    output = {
        'schema': 'qubo.v1',
        'problem_id': graph.graph.get('problem_id'),
        'sense': graph.graph.get('sense'),
        'num_variables': variable_count,
        'variable_names': variable_names,
        'offset': graph.graph.get('offset'),
        'terms': terms,
    }
    _restore_optional_field(
        output,
        'metadata',
        graph.graph.get('source_metadata'),
        graph.graph.get('source_metadata_present', False),
    )
    _merge_extra_fields(
        output,
        graph.graph.get('source_extra_fields', {}),
        {
            'schema',
            'problem_id',
            'sense',
            'num_variables',
            'variable_names',
            'offset',
            'terms',
            'metadata',
        },
        'interaction graph source_extra_fields',
    )
    _validate_qubo(output)
    return output


def cbqm_to_factor_graph(cbqm):
    """Create a bipartite graph containing every CBQM objective/constraint term."""
    _validate_cbqm(cbqm)
    graph = _new_cbqm_factor_graph(cbqm)

    _add_cbqm_variable_nodes(graph, cbqm)
    _add_cbqm_objective_factors(graph, cbqm['objective'])
    _add_cbqm_constraint_factors(graph, cbqm['constraints'])
    _annotate_fixed_values(graph, cbqm['fixed_values'])
    return graph


def factor_graph_to_cbqm(graph_or_artifact):
    """Reconstruct ``cbqm.v1`` from a factor graph's explicit identities."""
    graph = _coerce_graph(
        graph_or_artifact,
        CBQM_FACTOR_REPRESENTATION,
    )
    _validate_simple_undirected_graph(graph, 'CBQM factor graph')
    _require_graph_schema(graph, CBQM_FACTOR_REPRESENTATION)

    variable_nodes = graph.graph.get('variable_node_order')
    if not isinstance(variable_nodes, list):
        raise TypeError('factor graph variable_node_order must be a list.')
    _validate_factor_graph_topology(graph, variable_nodes)
    variables = _read_cbqm_variables(graph, variable_nodes)
    variable_count = len(variables)
    objective = _read_cbqm_objective(graph, variable_count)
    constraints = _read_cbqm_constraints(graph, variable_count)
    fixed_values = _read_cbqm_fixed_values(graph, len(variables))

    output = {
        'schema': 'cbqm.v1',
        'problem_id': graph.graph.get('problem_id'),
        'variables': variables,
        'objective': objective,
        'constraints': constraints,
        'fixed_values': fixed_values,
    }
    _restore_optional_field(
        output,
        'metadata',
        graph.graph.get('source_metadata'),
        graph.graph.get('source_metadata_present', False),
    )
    _merge_extra_fields(
        output,
        graph.graph.get('source_extra_fields', {}),
        {
            'schema',
            'problem_id',
            'variables',
            'objective',
            'constraints',
            'fixed_values',
            'metadata',
        },
        'factor graph source_extra_fields',
    )
    _validate_cbqm(output)
    return output


def qubo_to_maxcut_graph(qubo):
    """Encode QUBO as a signed MaxCut graph with one reference anchor.

    For a partition whose anchor label is zero, the exact relation is:

    ``qubo_energy(sample) = qubo_offset - cut_value(partition)``.
    """
    _validate_qubo(qubo)
    graph = _new_qubo_maxcut_graph(qubo)
    anchor_node = qubo['num_variables']

    _add_maxcut_nodes(graph, qubo, anchor_node)
    edge_weights = build_signed_maxcut_edge_weights(qubo, anchor_node)
    for (left, right), weight in sorted(edge_weights.items()):
        graph.add_edge(left, right, weight=weight)
    return graph


def _new_qubo_interaction_graph(qubo):
    """Initialize graph-level state needed for a reversible representation."""
    graph = nx.Graph()
    graph.graph.update({
        'representation_schema': QUBO_INTERACTION_REPRESENTATION,
        'problem_id': qubo['problem_id'],
        'sense': qubo['sense'],
        'offset': copy.deepcopy(qubo['offset']),
        'num_variables': qubo['num_variables'],
        'variable_names': copy.deepcopy(qubo['variable_names']),
        'node_order': list(range(qubo['num_variables'])),
        'source_metadata_present': 'metadata' in qubo,
        'source_metadata': copy.deepcopy(qubo.get('metadata')),
        'source_extra_fields': _extra_fields(
            qubo,
            {
                'schema',
                'problem_id',
                'sense',
                'num_variables',
                'variable_names',
                'offset',
                'terms',
                'metadata',
            },
        ),
    })
    return graph


def _add_qubo_variable_nodes(graph, qubo):
    """Add every variable before any terms so isolated variables survive."""
    for index, variable_name in enumerate(qubo['variable_names']):
        graph.add_node(
            index,
            node_type='binary_variable',
            qubo_index=index,
            variable_name=variable_name,
            linear_bias=0.0,
        )


def _add_qubo_biases(graph, qubo):
    """Keep diagonal and off-diagonal semantics visibly separate."""
    for left, right, coefficient in qubo['terms']:
        if left == right:
            graph.nodes[left]['linear_bias'] = copy.deepcopy(coefficient)
        else:
            graph.add_edge(
                left,
                right,
                quadratic_bias=copy.deepcopy(coefficient),
            )


def _read_qubo_variable_nodes(graph, node_order):
    """Validate stable index/name identity independently of insertion order."""
    variable_names = graph.graph.get('variable_names')
    if (
        not isinstance(variable_names, list)
        or len(variable_names) != len(node_order)
    ):
        raise ValueError('interaction graph variable_names are inconsistent.')
    actual_node_ids = list(graph.nodes)
    for node_id in actual_node_ids:
        _require_structural_index(
            node_id,
            len(node_order),
            'interaction graph node ID',
        )
    if actual_node_ids != node_order and set(actual_node_ids) != set(node_order):
        raise ValueError(
            'interaction graph nodes must exactly match its variable indices.'
        )

    output = []
    for index in node_order:
        attributes = graph.nodes[index]
        if attributes.get('node_type') != 'binary_variable':
            raise ValueError(f'Node {index!r} is not a binary variable.')
        qubo_index = _require_structural_index(
            attributes.get('qubo_index'),
            len(node_order),
            f'Node {index!r} qubo_index',
        )
        if qubo_index != index:
            raise ValueError(f'Node {index!r} has an inconsistent QUBO index.')
        if attributes.get('variable_name') != variable_names[index]:
            raise ValueError(f'Node {index!r} has an inconsistent variable name.')
        _validate_finite_number(
            attributes.get('linear_bias'),
            f'Node {index!r} linear_bias',
        )
        output.append(attributes['variable_name'])
    return output


def _read_qubo_graph_terms(graph, node_order):
    """Recreate sparse terms and omit explicit zero biases."""
    terms = []
    for index in node_order:
        coefficient = graph.nodes[index]['linear_bias']
        if coefficient != 0:
            terms.append([index, index, copy.deepcopy(coefficient)])

    for left, right, attributes in graph.edges(data=True):
        if left == right:
            raise ValueError(
                'QUBO interaction diagonals belong on nodes, not self-loops.'
            )
        coefficient = attributes.get('quadratic_bias')
        _validate_finite_number(
            coefficient,
            f'Edge ({left!r}, {right!r}) quadratic_bias',
        )
        if left not in node_order or right not in node_order:
            raise ValueError('QUBO interaction edge references an unknown node.')
        if coefficient != 0:
            lower, upper = sorted((left, right))
            terms.append([lower, upper, copy.deepcopy(coefficient)])
    return sorted(terms, key=lambda term: (term[0], term[1]))


def _new_cbqm_factor_graph(cbqm):
    """Initialize stable model-level fields before adding factor topology."""
    objective = cbqm['objective']
    graph = nx.Graph()
    graph.graph.update({
        'representation_schema': CBQM_FACTOR_REPRESENTATION,
        'problem_id': cbqm['problem_id'],
        'objective_sense': objective['sense'],
        'objective_offset': copy.deepcopy(objective['offset']),
        'objective_extra_fields': _extra_fields(
            objective,
            {'sense', 'offset', 'linear', 'quadratic'},
        ),
        'variable_node_order': [
            _variable_node_id(variable['index'])
            for variable in cbqm['variables']
        ],
        'constraint_count': len(cbqm['constraints']),
        'fixed_values': copy.deepcopy(cbqm['fixed_values']),
        'source_metadata_present': 'metadata' in cbqm,
        'source_metadata': copy.deepcopy(cbqm.get('metadata')),
        'source_extra_fields': _extra_fields(
            cbqm,
            {
                'schema',
                'problem_id',
                'variables',
                'objective',
                'constraints',
                'fixed_values',
                'metadata',
            },
        ),
    })
    return graph


def _add_cbqm_variable_nodes(graph, cbqm):
    """Namespace variable IDs so real names may collide with constraint names."""
    for variable in cbqm['variables']:
        index = variable['index']
        graph.add_node(
            _variable_node_id(index),
            node_type='variable',
            bipartite=0,
            variable_index=index,
            variable_name=variable['name'],
            vartype=variable['vartype'],
            kind_present='kind' in variable,
            kind=copy.deepcopy(variable.get('kind')),
            metadata_present='metadata' in variable,
            metadata=copy.deepcopy(variable.get('metadata')),
            extra_fields=_extra_fields(
                variable,
                {'index', 'name', 'vartype', 'kind', 'metadata'},
            ),
        )


def _add_cbqm_objective_factors(graph, objective):
    """Represent linear and quadratic monomials as true factor nodes."""
    for term_index, (variable_index, coefficient) in enumerate(
        objective['linear']
    ):
        factor_id = f'objective:linear:{term_index}'
        graph.add_node(
            factor_id,
            node_type='factor',
            bipartite=1,
            factor_type='objective_linear',
            term_index=term_index,
            variable_index=variable_index,
            coefficient=copy.deepcopy(coefficient),
        )
        graph.add_edge(
            factor_id,
            _variable_node_id(variable_index),
            incidence_type='objective_operand',
            coefficient=copy.deepcopy(coefficient),
            operand_positions=[0],
        )

    for term_index, (left, right, coefficient) in enumerate(
        objective['quadratic']
    ):
        factor_id = f'objective:quadratic:{term_index}'
        graph.add_node(
            factor_id,
            node_type='factor',
            bipartite=1,
            factor_type='objective_quadratic',
            term_index=term_index,
            left_index=left,
            right_index=right,
            coefficient=copy.deepcopy(coefficient),
        )
        positions_by_index = {left: [0]}
        positions_by_index.setdefault(right, []).append(1)
        for variable_index, positions in positions_by_index.items():
            graph.add_edge(
                factor_id,
                _variable_node_id(variable_index),
                incidence_type='objective_operand',
                coefficient=copy.deepcopy(coefficient),
                operand_positions=positions,
            )


def _add_cbqm_constraint_factors(graph, constraints):
    """Keep constraint identity even when two supports are identical or empty."""
    for constraint_index, constraint in enumerate(constraints):
        factor_id = _constraint_node_id(constraint_index)
        graph.add_node(
            factor_id,
            node_type='factor',
            bipartite=1,
            factor_type='linear_constraint',
            constraint_index=constraint_index,
            name=constraint['name'],
            family=constraint['family'],
            lower_bound_present='lower_bound' in constraint,
            lower_bound=copy.deepcopy(constraint.get('lower_bound')),
            upper_bound_present='upper_bound' in constraint,
            upper_bound=copy.deepcopy(constraint.get('upper_bound')),
            metadata_present='metadata' in constraint,
            metadata=copy.deepcopy(constraint.get('metadata')),
            extra_fields=_extra_fields(
                constraint,
                {
                    'name',
                    'family',
                    'linear',
                    'lower_bound',
                    'upper_bound',
                    'metadata',
                },
            ),
        )
        for term_index, (variable_index, coefficient) in enumerate(
            constraint['linear']
        ):
            graph.add_edge(
                factor_id,
                _variable_node_id(variable_index),
                incidence_type='constraint_term',
                term_index=term_index,
                coefficient=copy.deepcopy(coefficient),
            )


def _annotate_fixed_values(graph, fixed_values):
    """Expose fixed status on variable nodes without changing graph topology."""
    for fixed in fixed_values:
        node_id = _variable_node_id(fixed['index'])
        graph.nodes[node_id]['fixed_value'] = fixed['value']


def _read_cbqm_variables(graph, variable_nodes):
    """Restore variables in explicit index order, never graph insertion order."""
    expected_nodes = {
        node_id
        for node_id, attributes in graph.nodes(data=True)
        if attributes.get('node_type') == 'variable'
    }
    canonical_order = [
        _variable_node_id(index)
        for index in range(len(variable_nodes))
    ]
    if variable_nodes != canonical_order:
        raise ValueError(
            'factor graph variable_node_order must use canonical node IDs.'
        )
    if set(variable_nodes) != expected_nodes:
        raise ValueError('factor graph variable_node_order is inconsistent.')

    output = []
    for expected_index, node_id in enumerate(variable_nodes):
        attributes = graph.nodes[node_id]
        variable_index = _require_structural_index(
            attributes.get('variable_index'),
            len(variable_nodes),
            f'Variable node {node_id!r} variable_index',
        )
        if variable_index != expected_index:
            raise ValueError(f'Variable node {node_id!r} has wrong index.')
        variable = {
            'index': expected_index,
            'name': attributes.get('variable_name'),
            'vartype': attributes.get('vartype'),
        }
        _restore_optional_field(
            variable,
            'kind',
            attributes.get('kind'),
            attributes.get('kind_present', False),
        )
        _restore_optional_field(
            variable,
            'metadata',
            attributes.get('metadata'),
            attributes.get('metadata_present', False),
        )
        _merge_extra_fields(
            variable,
            attributes.get('extra_fields', {}),
            {'index', 'name', 'vartype', 'kind', 'metadata'},
            f'Variable node {node_id!r} extra_fields',
        )
        output.append(variable)
    return output


def _validate_factor_graph_topology(graph, variable_nodes):
    """Reject topology that has no meaning in the factor representation.

    Reverse transforms must be closed over the graph they consume.  Ignoring
    an isolated rogue node, a factor-to-factor edge, or a variable self-loop
    would produce a valid-looking CBQM that is not a faithful reconstruction
    of its stored artifact.
    """
    variable_ids = set(variable_nodes)
    factor_ids = set()
    known_factor_types = {
        'objective_linear',
        'objective_quadratic',
        'linear_constraint',
    }

    for node_id, attributes in graph.nodes(data=True):
        if node_id in variable_ids:
            if attributes.get('node_type') != 'variable':
                raise ValueError(
                    f'Canonical variable node {node_id!r} has invalid type.'
                )
            _require_bipartite_tag(
                attributes.get('bipartite'),
                0,
                f'Variable node {node_id!r}',
            )
            continue

        if (
            attributes.get('node_type') != 'factor'
            or attributes.get('factor_type') not in known_factor_types
        ):
            raise ValueError(
                f'Factor graph contains unknown node {node_id!r}.'
            )
        _require_bipartite_tag(
            attributes.get('bipartite'),
            1,
            f'Factor node {node_id!r}',
        )
        factor_ids.add(node_id)

    for left, right in graph.edges:
        if left == right:
            raise ValueError('Factor graph must not contain self-loops.')
        left_is_variable = left in variable_ids
        right_is_variable = right in variable_ids
        left_is_factor = left in factor_ids
        right_is_factor = right in factor_ids
        if not (
            (left_is_variable and right_is_factor)
            or (right_is_variable and left_is_factor)
        ):
            raise ValueError(
                'Factor graph edges must connect one canonical variable '
                'to one known factor.'
            )


def _read_cbqm_objective(graph, variable_count):
    """Read objective factors by their recorded term indices."""
    linear_factors = _ordered_factors(
        graph,
        'objective_linear',
        'term_index',
        node_id_for_index=lambda index: f'objective:linear:{index}',
    )
    quadratic_factors = _ordered_factors(
        graph,
        'objective_quadratic',
        'term_index',
        node_id_for_index=lambda index: f'objective:quadratic:{index}',
    )
    linear = []
    for factor_id, attributes in linear_factors:
        variable_index = _require_structural_index(
            attributes.get('variable_index'),
            variable_count,
            f'Objective factor {factor_id!r} variable_index',
        )
        coefficient = attributes.get('coefficient')
        _validate_finite_number(
            coefficient,
            f'Objective factor {factor_id!r} coefficient',
        )
        _validate_factor_incidence(
            graph,
            factor_id,
            attributes,
            _variable_node_id(variable_index),
            coefficient,
        )
        linear.append([variable_index, copy.deepcopy(coefficient)])

    quadratic = []
    for factor_id, attributes in quadratic_factors:
        left = _require_structural_index(
            attributes.get('left_index'),
            variable_count,
            f'Quadratic factor {factor_id!r} left_index',
        )
        right = _require_structural_index(
            attributes.get('right_index'),
            variable_count,
            f'Quadratic factor {factor_id!r} right_index',
        )
        if left > right:
            raise ValueError(
                f'Quadratic factor {factor_id!r} indices must be ordered.'
            )
        coefficient = attributes.get('coefficient')
        _validate_finite_number(
            coefficient,
            f'Quadratic factor {factor_id!r} coefficient',
        )
        _validate_quadratic_factor_incidence(
            graph,
            factor_id,
            left,
            right,
            coefficient,
        )
        quadratic.append([left, right, copy.deepcopy(coefficient)])

    objective = {
        'sense': graph.graph.get('objective_sense'),
        'offset': copy.deepcopy(graph.graph.get('objective_offset')),
        'linear': linear,
        'quadratic': quadratic,
    }
    _merge_extra_fields(
        objective,
        graph.graph.get('objective_extra_fields', {}),
        {'sense', 'offset', 'linear', 'quadratic'},
        'factor graph objective_extra_fields',
    )
    return objective


def _read_cbqm_constraints(graph, variable_count):
    """Read every constraint factor, including factors with no incidences."""
    factors = _ordered_factors(
        graph,
        'linear_constraint',
        'constraint_index',
        node_id_for_index=_constraint_node_id,
    )
    expected_count = _require_non_negative_int(
        graph.graph.get('constraint_count'),
        'factor graph constraint_count',
    )
    if expected_count != len(factors):
        raise ValueError('factor graph constraint count is inconsistent.')

    output = []
    for factor_id, attributes in factors:
        indexed_terms = []
        for neighbor, edge_attributes in graph[factor_id].items():
            if edge_attributes.get('incidence_type') != 'constraint_term':
                raise ValueError(
                    f'Constraint factor {factor_id!r} has invalid incidence.'
                )
            variable_index = _require_structural_index(
                graph.nodes[neighbor].get('variable_index'),
                variable_count,
                f'Constraint factor {factor_id!r} variable_index',
            )
            if (
                graph.nodes[neighbor].get('node_type') != 'variable'
                or neighbor != _variable_node_id(variable_index)
            ):
                raise ValueError(
                    f'Constraint factor {factor_id!r} references an '
                    'inconsistent variable node.'
                )
            term_index = _require_non_negative_int(
                edge_attributes.get('term_index'),
                f'Constraint factor {factor_id!r} edge term_index',
            )
            coefficient = edge_attributes.get('coefficient')
            _validate_finite_number(
                coefficient,
                f'Constraint factor {factor_id!r} edge coefficient',
            )
            indexed_terms.append((
                term_index,
                variable_index,
                copy.deepcopy(coefficient),
            ))
        indexed_terms.sort(key=lambda item: item[0])
        if [item[0] for item in indexed_terms] != list(
            range(len(indexed_terms))
        ):
            raise ValueError(
                f'Constraint factor {factor_id!r} has inconsistent term indices.'
            )

        constraint = {
            'name': attributes.get('name'),
            'family': attributes.get('family'),
            'linear': [
                [variable_index, coefficient]
                for _, variable_index, coefficient in indexed_terms
            ],
        }
        _restore_optional_field(
            constraint,
            'lower_bound',
            attributes.get('lower_bound'),
            attributes.get('lower_bound_present', False),
        )
        _restore_optional_field(
            constraint,
            'upper_bound',
            attributes.get('upper_bound'),
            attributes.get('upper_bound_present', False),
        )
        _restore_optional_field(
            constraint,
            'metadata',
            attributes.get('metadata'),
            attributes.get('metadata_present', False),
        )
        _merge_extra_fields(
            constraint,
            attributes.get('extra_fields', {}),
            {
                'name',
                'family',
                'linear',
                'lower_bound',
                'upper_bound',
                'metadata',
            },
            f'Constraint factor {factor_id!r} extra_fields',
        )
        output.append(constraint)
    return output


def _read_cbqm_fixed_values(graph, variable_count):
    """Restore and cross-check fixed values exposed on variable nodes."""
    fixed_values = copy.deepcopy(graph.graph.get('fixed_values'))
    if not isinstance(fixed_values, list):
        raise TypeError('factor graph fixed_values must be a list.')
    listed_indices = set()
    for fixed in fixed_values:
        _require_mapping(fixed, 'factor graph fixed value')
        index = _require_structural_index(
            fixed.get('index'),
            variable_count,
            'factor graph fixed value index',
        )
        value = fixed.get('value')
        _require_binary_integer(value, 'factor graph fixed value')
        node_value = graph.nodes[_variable_node_id(index)].get('fixed_value')
        _require_binary_integer(
            node_value,
            f'Variable node {_variable_node_id(index)!r} fixed_value',
        )
        if type(node_value) is not type(value) or node_value != value:
            raise ValueError('factor graph fixed value annotation mismatch.')
        listed_indices.add(index)

    for index in range(variable_count):
        node_id = _variable_node_id(index)
        has_annotation = 'fixed_value' in graph.nodes[node_id]
        if has_annotation != (index in listed_indices):
            raise ValueError(
                'factor graph fixed value list and node annotations mismatch.'
            )
    return fixed_values


def _ordered_factors(
    graph,
    factor_type,
    index_field,
    *,
    node_id_for_index,
):
    """Collect one factor family and require contiguous stable term indices."""
    factors = [
        (node_id, attributes)
        for node_id, attributes in graph.nodes(data=True)
        if attributes.get('factor_type') == factor_type
    ]
    indexed_factors = []
    for node_id, attributes in factors:
        index = _require_non_negative_int(
            attributes.get(index_field),
            f'{factor_type} factor {index_field}',
        )
        indexed_factors.append((index, node_id, attributes))

    indexed_factors.sort(key=lambda item: item[0])
    indices = [index for index, _, _ in indexed_factors]
    if indices != list(range(len(indices))):
        raise ValueError(f'{factor_type} factor indices must be contiguous.')
    for index, node_id, _ in indexed_factors:
        if node_id != node_id_for_index(index):
            raise ValueError(
                f'{factor_type} factor node ID does not match its '
                f'{index_field}.'
            )
    return [
        (node_id, attributes)
        for _, node_id, attributes in indexed_factors
    ]


def _validate_factor_incidence(
    graph,
    factor_id,
    attributes,
    variable_id,
    coefficient,
):
    """Ensure a linear objective factor has exactly its declared operand."""
    if set(graph.neighbors(factor_id)) != {variable_id}:
        raise ValueError(f'Objective factor {factor_id!r} incidence mismatch.')
    edge = graph.edges[factor_id, variable_id]
    _validate_finite_number(
        edge.get('coefficient'),
        f'Objective factor {factor_id!r} edge coefficient',
    )
    _require_exact_index_sequence(
        edge.get('operand_positions'),
        [0],
        f'Objective factor {factor_id!r} operand_positions',
    )
    if (
        edge.get('incidence_type') != 'objective_operand'
        or edge.get('coefficient') != coefficient
    ):
        raise ValueError(f'Objective factor {factor_id!r} coefficient mismatch.')


def _validate_quadratic_factor_incidence(
    graph,
    factor_id,
    left,
    right,
    coefficient,
):
    """Cross-check every duplicated quadratic incidence attribute."""
    expected = {_variable_node_id(left), _variable_node_id(right)}
    if set(graph.neighbors(factor_id)) != expected:
        raise ValueError(f'Quadratic factor {factor_id!r} incidence mismatch.')
    left_edge = graph.edges[
        factor_id,
        _variable_node_id(left),
    ]
    _validate_finite_number(
        left_edge.get('coefficient'),
        f'Quadratic factor {factor_id!r} left edge coefficient',
    )
    if (
        left_edge.get('incidence_type') != 'objective_operand'
        or left_edge.get('coefficient') != coefficient
    ):
        raise ValueError(
            f'Quadratic factor {factor_id!r} edge metadata mismatch.'
        )
    left_positions = left_edge.get('operand_positions')
    if left == right:
        _require_exact_index_sequence(
            left_positions,
            [0, 1],
            f'Diagonal factor {factor_id!r} operand_positions',
        )
        return
    right_edge = graph.edges[
        factor_id,
        _variable_node_id(right),
    ]
    _validate_finite_number(
        right_edge.get('coefficient'),
        f'Quadratic factor {factor_id!r} right edge coefficient',
    )
    if (
        right_edge.get('incidence_type') != 'objective_operand'
        or right_edge.get('coefficient') != coefficient
    ):
        raise ValueError(
            f'Quadratic factor {factor_id!r} edge metadata mismatch.'
        )
    right_positions = right_edge.get('operand_positions')
    _require_exact_index_sequence(
        left_positions,
        [0],
        f'Quadratic factor {factor_id!r} left operand_positions',
    )
    _require_exact_index_sequence(
        right_positions,
        [1],
        f'Quadratic factor {factor_id!r} right operand_positions',
    )


def _new_qubo_maxcut_graph(qubo):
    """Store both human-readable and machine-readable energy mappings."""
    graph = nx.Graph()
    graph.graph.update({
        'representation_schema': QUBO_MAXCUT_REPRESENTATION,
        'problem_id': qubo['problem_id'],
        'num_variables': qubo['num_variables'],
        'variable_names': copy.deepcopy(qubo['variable_names']),
        'node_order': list(range(qubo['num_variables'])),
        'anchor_node': qubo['num_variables'],
        'qubo_offset': copy.deepcopy(qubo['offset']),
        'qubo_problem': copy.deepcopy(dict(qubo)),
        'energy_relation': 'qubo_energy = qubo_offset - cut_value',
        'energy_mapping': {
            'source_value': 'qubo_energy',
            'target_value': 'cut_value',
            'scale': -1.0,
            'shift': copy.deepcopy(qubo['offset']),
        },
    })
    return graph


def _add_maxcut_nodes(graph, qubo, anchor_node):
    """Make the anchor visibly different from sample-bearing variable nodes."""
    for index, variable_name in enumerate(qubo['variable_names']):
        graph.add_node(
            index,
            node_type='binary_variable',
            node_role='variable',
            is_anchor=False,
            qubo_index=index,
            variable_name=variable_name,
        )
    graph.add_node(
        anchor_node,
        node_type='anchor',
        node_role='anchor',
        is_anchor=True,
    )


def _coerce_graph(graph_or_artifact, expected_representation):
    """Let reverse transforms consume either native graphs or stored artifacts."""
    if isinstance(graph_or_artifact, ProblemArtifact):
        if graph_or_artifact.representation != expected_representation:
            raise ValueError(
                f"Artifact representation must be '{expected_representation}'."
            )
        return graph_from_node_link(graph_or_artifact.payload)
    if not isinstance(graph_or_artifact, nx.Graph):
        raise TypeError('Expected a NetworkX graph or ProblemArtifact.')
    return graph_or_artifact


def _require_graph_schema(graph, expected):
    """Reject a structurally similar graph with different semantics."""
    if graph.graph.get('representation_schema') != expected:
        raise ValueError(
            f"Graph representation_schema must be '{expected}'."
        )


def _validate_simple_undirected_graph(graph, label):
    """These representations use one unambiguous incidence per node pair."""
    if graph.is_directed():
        raise ValueError(f'{label} must be undirected.')
    if graph.is_multigraph():
        raise ValueError(f'{label} must not be a multigraph.')


def _validate_index_node_order(node_order, variable_count):
    """Require a complete canonical index order."""
    if not isinstance(node_order, list):
        raise TypeError('interaction graph node_order must be a list.')
    for position, node_id in enumerate(node_order):
        _require_structural_index(
            node_id,
            variable_count,
            f'interaction graph node_order[{position}]',
        )
    expected = list(range(variable_count))
    if node_order != expected:
        raise ValueError(
            'interaction graph node_order must be canonical variable indices.'
        )
    return expected


def _require_structural_index(value, upper_bound, label):
    """Require a plain Python integer inside one declared index domain.

    Python deliberately makes ``bool`` a subclass of ``int`` and treats
    ``False == 0`` and ``True == 1``.  That convenience is dangerous in a
    persisted graph: equality/set checks could otherwise accept a boolean and
    the reverse transform would silently emit a different integer identity.
    """
    if type(value) is not int or not 0 <= value < upper_bound:
        raise ValueError(
            f'{label} must be an integer inside the declared index range.'
        )
    return value


def _require_exact_index_sequence(value, expected, label):
    """Compare position metadata without bool-as-int equality shortcuts."""
    if not isinstance(value, list):
        raise TypeError(f'{label} must be a list.')
    if any(type(item) is not int for item in value):
        raise ValueError(f'{label} must contain plain integer indices.')
    if value != expected:
        raise ValueError(f'{label} is inconsistent.')


def _require_binary_integer(value, label):
    """Validate binary annotations with exact Python integer identity."""
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f'{label} must be integer 0 or 1.')
    return value


def _require_bipartite_tag(value, expected, label):
    """Require the generated partition tag without accepting booleans."""
    _require_binary_integer(value, f'{label} bipartite tag')
    if value != expected:
        raise ValueError(
            f'{label} has an inconsistent bipartite partition tag.'
        )


def _variable_node_id(index):
    """Namespace variable identities from every factor identity."""
    return f'variable:{index}'


def _constraint_node_id(index):
    """Namespace constraint identity even if its name matches a variable."""
    return f'constraint:{index}'


def _extra_fields(mapping, known_fields):
    """Preserve forward-compatible fields without hiding core graph semantics."""
    return {
        key: copy.deepcopy(value)
        for key, value in mapping.items()
        if key not in known_fields
    }


def _merge_extra_fields(output, extra_fields, reserved_fields, label):
    """Merge extension data only when it cannot redefine core semantics."""
    if not isinstance(extra_fields, Mapping):
        raise TypeError(f'{label} must be a mapping.')
    if any(not isinstance(key, str) or not key for key in extra_fields):
        raise ValueError(f'{label} keys must be non-empty strings.')
    conflicts = sorted(set(extra_fields).intersection(reserved_fields))
    if conflicts:
        names = ', '.join(conflicts)
        raise ValueError(
            f'{label} must not override reserved fields: {names}.'
        )
    output.update(copy.deepcopy(dict(extra_fields)))


def _restore_optional_field(output, key, value, present):
    """Distinguish an absent optional field from an explicitly null one."""
    if type(present) is not bool:
        raise ValueError(
            f"Optional-field presence marker for '{key}' must be a boolean."
        )
    if present:
        output[key] = copy.deepcopy(value)


def _validate_qubo_legacy(problem):
    """Validate the complete subset needed by the QUBO graph transforms."""
    _require_mapping(problem, 'QUBO')
    if problem.get('schema') != 'qubo.v1':
        raise ValueError("QUBO schema must be 'qubo.v1'.")
    _require_non_empty_string(problem.get('problem_id'), 'QUBO problem_id')
    if problem.get('sense') != 'minimize':
        raise ValueError("qubo.v1 sense must be 'minimize'.")
    variable_count = _require_non_negative_int(
        problem.get('num_variables'),
        'QUBO num_variables',
    )
    names = problem.get('variable_names')
    if not isinstance(names, list) or len(names) != variable_count:
        raise ValueError(
            'QUBO variable_names length must equal num_variables.'
        )
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('QUBO variable names must be non-empty strings.')
    if len(set(names)) != len(names):
        raise ValueError('QUBO variable names must be unique.')
    _validate_finite_number(problem.get('offset'), 'QUBO offset')
    _validate_sparse_quadratic(
        problem.get('terms'),
        variable_count,
        'QUBO terms',
    )


def _validate_cbqm_legacy(problem):
    """Validate fields required for a faithful binary CBQM factor graph."""
    _require_mapping(problem, 'CBQM')
    if problem.get('schema') != 'cbqm.v1':
        raise ValueError("CBQM schema must be 'cbqm.v1'.")
    _require_non_empty_string(problem.get('problem_id'), 'CBQM problem_id')
    variables = problem.get('variables')
    if not isinstance(variables, list):
        raise TypeError('CBQM variables must be a list.')
    names = []
    for position, variable in enumerate(variables):
        _require_mapping(variable, f'CBQM variable {position}')
        if variable.get('index') != position:
            raise ValueError('CBQM variable indices must match list positions.')
        _require_non_empty_string(
            variable.get('name'),
            f'CBQM variable {position} name',
        )
        if variable.get('vartype') != 'BINARY':
            raise ValueError('CBQM factor graph supports BINARY variables.')
        names.append(variable['name'])
    if len(names) != len(set(names)):
        raise ValueError('CBQM variable names must be unique.')

    objective = problem.get('objective')
    _require_mapping(objective, 'CBQM objective')
    if objective.get('sense') not in {'minimize', 'maximize'}:
        raise ValueError('CBQM objective has invalid sense.')
    _validate_finite_number(objective.get('offset'), 'CBQM objective offset')
    _validate_sparse_linear(
        objective.get('linear'),
        len(variables),
        'CBQM objective linear terms',
    )
    _validate_sparse_quadratic(
        objective.get('quadratic'),
        len(variables),
        'CBQM objective quadratic terms',
    )

    constraints = problem.get('constraints')
    if not isinstance(constraints, list):
        raise TypeError('CBQM constraints must be a list.')
    constraint_names = []
    for index, constraint in enumerate(constraints):
        _require_mapping(constraint, f'CBQM constraint {index}')
        _require_non_empty_string(
            constraint.get('name'),
            f'CBQM constraint {index} name',
        )
        _require_non_empty_string(
            constraint.get('family'),
            f'CBQM constraint {index} family',
        )
        constraint_names.append(constraint['name'])
        _validate_sparse_linear(
            constraint.get('linear'),
            len(variables),
            f"CBQM constraint '{constraint['name']}' terms",
        )
        lower = constraint.get('lower_bound')
        upper = constraint.get('upper_bound')
        if lower is None and upper is None:
            raise ValueError('CBQM constraint must have at least one bound.')
        if lower is not None:
            _validate_finite_number(lower, 'CBQM constraint lower bound')
        if upper is not None:
            _validate_finite_number(upper, 'CBQM constraint upper bound')
        if lower is not None and upper is not None and lower > upper:
            raise ValueError('CBQM constraint bounds are inconsistent.')
    if len(constraint_names) != len(set(constraint_names)):
        raise ValueError('CBQM constraint names must be unique.')

    fixed_values = problem.get('fixed_values')
    if not isinstance(fixed_values, list):
        raise TypeError('CBQM fixed_values must be a list.')
    fixed_indices = []
    for fixed in fixed_values:
        _require_mapping(fixed, 'CBQM fixed value')
        index = fixed.get('index')
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(variables)
        ):
            raise ValueError('CBQM fixed value index is invalid.')
        value = fixed.get('value')
        if isinstance(value, bool) or value not in {0, 1}:
            raise ValueError('CBQM fixed values must be integer 0 or 1.')
        fixed_indices.append(index)
    if len(fixed_indices) != len(set(fixed_indices)):
        raise ValueError('CBQM variable may only be fixed once.')


def _validate_sparse_linear(terms, variable_count, label):
    """Validate unique sparse ``[index, coefficient]`` pairs."""
    if not isinstance(terms, list):
        raise TypeError(f'{label} must be a list.')
    seen = set()
    for term in terms:
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            raise ValueError(f'{label} must contain pairs.')
        index, coefficient = term
        _validate_variable_index(index, variable_count, label)
        _validate_finite_number(coefficient, f'{label} coefficient')
        if coefficient == 0:
            raise ValueError(f'{label} must omit zero coefficients.')
        if index in seen:
            raise ValueError(f'{label} contains duplicate variable indices.')
        seen.add(index)


def _validate_sparse_quadratic(terms, variable_count, label):
    """Validate unique upper-triangular sparse triples."""
    if not isinstance(terms, list):
        raise TypeError(f'{label} must be a list.')
    seen = set()
    for term in terms:
        if not isinstance(term, (list, tuple)) or len(term) != 3:
            raise ValueError(f'{label} must contain triples.')
        left, right, coefficient = term
        _validate_variable_index(left, variable_count, label)
        _validate_variable_index(right, variable_count, label)
        if left > right:
            raise ValueError(f'{label} must use upper-triangular indices.')
        _validate_finite_number(coefficient, f'{label} coefficient')
        if coefficient == 0:
            raise ValueError(f'{label} must omit zero coefficients.')
        pair = (left, right)
        if pair in seen:
            raise ValueError(f'{label} contains duplicate index pairs.')
        seen.add(pair)


def _validate_variable_index(index, variable_count, label):
    """Reject booleans and indices outside the declared variable range."""
    if (
        type(index) is not int
        or not 0 <= index < variable_count
    ):
        raise ValueError(f'{label} has an invalid variable index.')


def _require_mapping(value, label):
    """Require mapping input before looking up schema fields."""
    if not isinstance(value, Mapping):
        raise TypeError(f'{label} must be a mapping.')


def _require_non_empty_string(value, label):
    """Validate stable text identifiers."""
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a non-empty string.')
    return value


def _require_non_negative_int(value, label):
    """Validate sizes without accepting bool as integer."""
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a non-negative integer.')
    return value


def _validate_finite_number(value, label):
    """Keep NaN and infinity out of graph attributes and persisted JSON."""
    if type(value) is int:
        return
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f'{label} must be a finite real number.')
