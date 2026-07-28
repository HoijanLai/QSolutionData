"""NetworkX-compatible graph payload helpers.

The repository persists graphs as ordinary node-link dictionaries, rather than
pickled NetworkX objects.  This keeps artifacts language-neutral and diffable.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import networkx as nx

from .problem_def import ProblemArtifact


def graph_to_node_link(graph) -> dict:
    """Return current node-link JSON using the explicit ``edges`` spelling."""
    _validate_networkx_graph(graph)
    _validate_json_node_ids(graph)
    _validate_json_edge_keys(graph)
    _validate_node_link_attribute_collisions(graph)

    # NetworkX 3.4+ lets callers name the edge collection explicitly.  The
    # fallback keeps this package usable with the older API while normalizing
    # its historical ``links`` output to our stable on-disk spelling.
    try:
        payload = nx.node_link_data(graph, edges='edges')
    except TypeError:
        payload = nx.node_link_data(graph)
        if 'links' in payload and 'edges' not in payload:
            payload['edges'] = payload.pop('links')
    return copy.deepcopy(payload)


def graph_from_node_link(payload):
    """Build a NetworkX graph from current ``edges`` or legacy ``links`` JSON."""
    if not isinstance(payload, Mapping):
        raise TypeError('Node-link payload must be a mapping.')
    if 'edges' in payload and 'links' in payload:
        raise ValueError(
            "Node-link payload must not contain both 'edges' and 'links'."
        )
    edge_field = 'edges' if 'edges' in payload else 'links'
    if edge_field not in payload:
        raise ValueError("Node-link payload must contain 'edges' or 'links'.")

    detached = copy.deepcopy(dict(payload))
    _validate_node_link_payload(detached, edge_field)
    try:
        graph = nx.node_link_graph(detached, edges=edge_field)
    except TypeError:
        # Older NetworkX accepts only ``links`` through its public default.
        if edge_field == 'edges':
            detached['links'] = detached.pop('edges')
        graph = nx.node_link_graph(detached)
    _validate_json_node_ids(graph)
    return graph


def artifact_from_networkx(
    artifact_id,
    graph,
    *,
    representation='networkx.node-link.v1',
    metadata=None,
) -> ProblemArtifact:
    """Wrap a native graph as a root artifact without inventing a task.

    Root graphs receive an explicit node order and variable-name vector.  When
    a node does not already expose ``variable_name``, its stable string ID is
    used.  The caller's NetworkX object is never mutated.
    """
    if not isinstance(representation, str) or not representation.startswith(
        'networkx.'
    ):
        raise ValueError(
            "NetworkX artifact representation must start with 'networkx.'."
        )
    normalized_graph = _with_variable_identity(graph)
    return ProblemArtifact(
        artifact_id=artifact_id,
        representation=representation,
        payload=graph_to_node_link(normalized_graph),
        metadata={} if metadata is None else metadata,
    )


def artifact_to_networkx(artifact):
    """Decode one NetworkX artifact while leaving its payload untouched."""
    if not isinstance(artifact, ProblemArtifact):
        raise TypeError('artifact must be a ProblemArtifact.')
    if not artifact.representation.startswith('networkx.'):
        raise ValueError(
            f"Artifact '{artifact.artifact_id}' is not a NetworkX representation."
        )
    return graph_from_node_link(artifact.payload)


def _validate_networkx_graph(graph):
    """Accept all standard NetworkX graph variants through their base class."""
    if not isinstance(graph, nx.Graph):
        raise TypeError('graph must be a NetworkX graph.')


def _validate_json_node_ids(graph):
    """Keep IDs stable across JSON, where tuples would turn into lists."""
    for node_id in graph.nodes:
        if isinstance(node_id, bool) or not isinstance(node_id, (str, int)):
            raise TypeError(
                'Persisted NetworkX node IDs must be strings or integers; '
                f'got {node_id!r}.'
            )


def _validate_node_link_payload(payload, edge_field):
    """Reject identities NetworkX would otherwise merge while decoding."""
    directed = payload.get('directed')
    multigraph = payload.get('multigraph')
    if type(directed) is not bool or type(multigraph) is not bool:
        raise ValueError(
            "Node-link 'directed' and 'multigraph' must be booleans."
        )
    graph_metadata = payload.setdefault('graph', {})
    if not isinstance(graph_metadata, Mapping):
        raise TypeError("Node-link 'graph' must be a mapping when present.")

    nodes = payload.get('nodes')
    edges = payload.get(edge_field)
    if not isinstance(nodes, list):
        raise TypeError("Node-link 'nodes' must be a list.")
    if not isinstance(edges, list):
        raise TypeError(f"Node-link '{edge_field}' must be a list.")

    node_keys = set()
    for position, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise TypeError(f'Node-link node {position} must be a mapping.')
        identity = _node_identity_key(
            node.get('id'),
            f'Node-link node {position} id',
        )
        if identity in node_keys:
            raise ValueError(
                f'Node-link payload contains duplicate node id '
                f'{node.get("id")!r}.'
            )
        node_keys.add(identity)

    edge_keys = set()
    for position, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise TypeError(f'Node-link edge {position} must be a mapping.')
        source = _node_identity_key(
            edge.get('source'),
            f'Node-link edge {position} source',
        )
        target = _node_identity_key(
            edge.get('target'),
            f'Node-link edge {position} target',
        )
        if source not in node_keys or target not in node_keys:
            raise ValueError(
                f'Node-link edge {position} references an undeclared node.'
            )

        endpoints = (
            (source, target)
            if directed
            else frozenset((source, target))
        )
        edge_identity = endpoints
        if multigraph:
            if 'key' not in edge:
                raise ValueError(
                    f'Node-link multigraph edge {position} must define a key.'
                )
            edge_identity = (
                endpoints,
                _edge_key_identity(edge['key']),
            )
        if edge_identity in edge_keys:
            raise ValueError(
                f'Node-link payload contains duplicate edge at position '
                f'{position}.'
            )
        edge_keys.add(edge_identity)


def _node_identity_key(value, label):
    """Return a type-aware hash key for one persisted node identifier."""
    if isinstance(value, bool) or type(value) not in {str, int}:
        raise TypeError(
            f'{label} must be a string or integer, excluding booleans.'
        )
    return type(value), value


def _edge_key_identity(value):
    """Keep multigraph edge keys stable instead of accepting key collisions."""
    if isinstance(value, bool) or type(value) not in {str, int}:
        raise TypeError(
            'Persisted NetworkX multigraph edge keys must be strings or '
            'integers, excluding booleans.'
        )
    return type(value), value


def _validate_json_edge_keys(graph):
    """Keep multigraph keys stable through the JSON artifact round trip."""
    if not graph.is_multigraph():
        return
    for _, _, edge_key in graph.edges(keys=True):
        _edge_key_identity(edge_key)


def _validate_node_link_attribute_collisions(graph):
    """Reject attributes the node-link envelope would silently overwrite."""
    for node_id, attributes in graph.nodes(data=True):
        if 'id' in attributes:
            raise ValueError(
                f'Node {node_id!r} attributes contain reserved field: id.'
            )

    if graph.is_multigraph():
        edge_iterator = graph.edges(keys=True, data=True)
        for left, right, edge_key, attributes in edge_iterator:
            conflicts = sorted(
                set(attributes).intersection({'source', 'target', 'key'})
            )
            if conflicts:
                names = ', '.join(conflicts)
                raise ValueError(
                    f'Edge ({left!r}, {right!r}, {edge_key!r}) attributes '
                    f'contain reserved fields: {names}.'
                )
        return

    for left, right, attributes in graph.edges(data=True):
        conflicts = sorted(
            set(attributes).intersection({'source', 'target'})
        )
        if conflicts:
            names = ', '.join(conflicts)
            raise ValueError(
                f'Edge ({left!r}, {right!r}) attributes contain reserved '
                f'fields: {names}.'
            )


def _with_variable_identity(graph):
    """Copy a root graph and make sample/real-name interpretation explicit."""
    _validate_networkx_graph(graph)
    _validate_json_node_ids(graph)
    normalized = graph.copy()

    declared_order = normalized.graph.get('node_order')
    node_order = (
        list(normalized.nodes)
        if declared_order is None
        else list(declared_order)
    )
    node_order_keys = [
        _node_identity_key(node_id, 'Graph node_order entry')
        for node_id in node_order
    ]
    actual_node_keys = {
        _node_identity_key(node_id, 'Graph node ID')
        for node_id in normalized.nodes
    }
    if (
        len(node_order) != normalized.number_of_nodes()
        or len(set(node_order_keys)) != len(node_order_keys)
        or set(node_order_keys) != actual_node_keys
    ):
        raise ValueError(
            'Graph node_order must contain every node exactly once.'
        )

    variable_names = []
    for node_id in node_order:
        name = normalized.nodes[node_id].get('variable_name', str(node_id))
        if not isinstance(name, str) or not name:
            raise ValueError('Graph variable_name values must be non-empty strings.')
        normalized.nodes[node_id]['variable_name'] = name
        variable_names.append(name)
    if len(variable_names) != len(set(variable_names)):
        raise ValueError('Graph variable_name values must be unique.')

    normalized.graph['node_order'] = node_order
    normalized.graph['variable_names'] = variable_names
    return normalized
