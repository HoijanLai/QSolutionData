"""Deterministic small-graph instances for the Q-RBnBR S1 reproduction.

The thesis's first experiment family uses small random unweighted MaxCut
graphs and compares against exhaustive solutions.  These builders preserve
that mathematical abstraction without importing an external dataset or
embedding any business meaning.

Every returned value is a canonical ``qubo.v1``.  Its metadata also stores the
source graph in the repository's NetworkX-compatible node-link format, so a
reproduction can inspect graph statistics without reverse-engineering QUBO
coefficients.
"""

from __future__ import annotations

import random

import networkx as nx

from ..graph_codec import graph_to_node_link


_THESIS_URL = (
    'https://mediatum.ub.tum.de/doc/1782421/'
    'top3gde62rvzykino5palp4gm.pdf'
)
_SOURCE_REPOSITORY = 'https://github.com/HoijanLai/Q-RBnBR'
_SOURCE_REVISION = 'ec72c202559655dc170f8bdf41f2936107ce94f8'
_GENERATOR_VERSION = '0.1.0'


def build_qrbnbr_s1_instance(
    *,
    variable_count=10,
    edge_probability=0.5,
    seed=2025,
):
    """Build one connected unweighted Erdős-Rényi MaxCut QUBO.

    Args:
        variable_count: Number of graph vertices.  S1 uses fewer than twenty.
        edge_probability: Independent edge probability before the deterministic
            connectivity repair.
        seed: Seed for a private Python random generator.

    Returns:
        A JSON-serializable canonical ``qubo.v1`` mapping.
    """
    _validate_instance_settings(
        variable_count,
        edge_probability,
        seed,
    )
    graph = _build_connected_random_graph(
        variable_count,
        edge_probability,
        seed,
    )
    return _maxcut_graph_to_qubo(
        graph,
        problem_id=(
            f'qrbnbr-s1-er-n{variable_count}'
            f'-p{_probability_label(edge_probability)}-seed{seed}'
        ),
        edge_probability=edge_probability,
        seed=seed,
    )


def build_qrbnbr_s1_suite():
    """Return a fixed laptop-scale sequence for scaling observations."""
    settings = (
        (8, 0.50, 2025),
        (12, 0.40, 2026),
        (16, 0.35, 2027),
        (18, 0.30, 2028),
    )
    return {
        f'qrbnbr.s1-er-{variable_count}': build_qrbnbr_s1_instance(
            variable_count=variable_count,
            edge_probability=edge_probability,
            seed=seed,
        )
        for variable_count, edge_probability, seed in settings
    }


def _validate_instance_settings(variable_count, edge_probability, seed):
    """Reject settings that cannot describe the declared experiment family."""
    if (
        not isinstance(variable_count, int)
        or isinstance(variable_count, bool)
        or variable_count < 0
        or variable_count >= 20
    ):
        raise ValueError(
            'variable_count must be an integer in [0, 19] for S1.'
        )
    if (
        not isinstance(edge_probability, (int, float))
        or isinstance(edge_probability, bool)
        or not 0.0 <= edge_probability <= 1.0
    ):
        raise ValueError('edge_probability must be a number in [0, 1].')
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a non-negative integer.')


def _build_connected_random_graph(
    variable_count,
    edge_probability,
    seed,
):
    """Draw edges reproducibly, then connect components deterministically."""
    random_generator = random.Random(seed)
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            vertex,
            {'variable_name': f'x_{vertex}'},
        )
        for vertex in range(variable_count)
    )

    for left in range(variable_count):
        for right in range(left + 1, variable_count):
            if random_generator.random() < edge_probability:
                graph.add_edge(left, right, weight=1)

    components = [
        sorted(component)
        for component in nx.connected_components(graph)
    ]
    components.sort(key=lambda component: component[0])
    for left_component, right_component in zip(
        components,
        components[1:],
    ):
        graph.add_edge(
            left_component[0],
            right_component[0],
            weight=1,
        )
    return graph


def _maxcut_graph_to_qubo(
    graph,
    *,
    problem_id,
    edge_probability,
    seed,
):
    """Compile ``-cut`` while retaining the exact NetworkX source graph."""
    diagonal = [0] * graph.number_of_nodes()
    quadratic = []
    for left, right, attributes in sorted(graph.edges(data=True)):
        weight = attributes['weight']
        diagonal[left] -= weight
        diagonal[right] -= weight
        quadratic.append([left, right, 2 * weight])

    terms = [
        [vertex, vertex, coefficient]
        for vertex, coefficient in enumerate(diagonal)
        if coefficient != 0
    ]
    terms.extend(quadratic)
    terms.sort(key=lambda term: (term[0], term[1]))

    return {
        'schema': 'qubo.v1',
        'problem_id': problem_id,
        'sense': 'minimize',
        'num_variables': graph.number_of_nodes(),
        'variable_names': [
            graph.nodes[vertex]['variable_name']
            for vertex in graph.nodes
        ],
        'offset': 0,
        'terms': terms,
        'metadata': {
            'reproduction': {
                'algorithm': 'Q-RBnBR',
                'paper_title': (
                    'Quantum Relaxation Informed Branch-and-Bound '
                    'Algorithm -- An Application to Max-Cut'
                ),
                'paper_url': _THESIS_URL,
                'source_repository': _SOURCE_REPOSITORY,
                'source_revision': _SOURCE_REVISION,
                'implementation_style': (
                    'independent-reimplementation-from-paper-formulas'
                ),
                'experiment_family': 'S1',
                'claim_scope': (
                    'small random unweighted MaxCut with exhaustive reference'
                ),
            },
            'generator': {
                'name': 'build_qrbnbr_s1_instance',
                'version': _GENERATOR_VERSION,
                'seed': seed,
                'parameters': {
                    'model': 'erdos-renyi',
                    'variable_count': graph.number_of_nodes(),
                    'edge_probability': edge_probability,
                    'connectivity_policy': (
                        'connect-components-by-smallest-vertex'
                    ),
                },
            },
            'source_graph': graph_to_node_link(graph),
        },
    }


def _probability_label(edge_probability):
    """Create a path- and ID-friendly probability spelling."""
    return format(float(edge_probability), '.6g').replace('.', 'p')


__all__ = [
    'build_qrbnbr_s1_instance',
    'build_qrbnbr_s1_suite',
]
