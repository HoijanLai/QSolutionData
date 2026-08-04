"""One-hot graph-colouring CBQM benchmark."""

from .._metadata import benchmark_metadata
from ._common import _binary_variables, _linear_constraint


def build_graph_coloring_suite():
    """Return a three-colour model containing a forced triangle."""
    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
    ]
    return {
        'cbqm.graph-coloring-5x3': _build_graph_coloring(
            vertex_count=5,
            color_count=3,
            edges=edges,
        ),
    }


def _build_graph_coloring(*, vertex_count, color_count, edges):
    """Assign one colour per vertex and separate every adjacent pair."""
    constraints = []
    for vertex in range(vertex_count):
        constraints.append(
            _linear_constraint(
                name=f'color_vertex_{vertex}_once',
                family='assignment',
                terms=[
                    (
                        _variable_index(
                            vertex,
                            color,
                            color_count,
                        ),
                        1,
                    )
                    for color in range(color_count)
                ],
                lower_bound=1,
                upper_bound=1,
            )
        )
    for left, right in edges:
        for color in range(color_count):
            constraints.append(
                _linear_constraint(
                    name=f'edge_{left}_{right}_color_{color}',
                    family='conflict',
                    terms=[
                        (
                            _variable_index(
                                left,
                                color,
                                color_count,
                            ),
                            1,
                        ),
                        (
                            _variable_index(
                                right,
                                color,
                                color_count,
                            ),
                            1,
                        ),
                    ],
                    upper_bound=1,
                )
            )

    variable_count = vertex_count * color_count
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'benchmark-graph-coloring-5x3',
        'variables': _binary_variables(
            [
                f'vertex_{vertex}_color_{color}'
                for vertex in range(vertex_count)
                for color in range(color_count)
            ],
            kind='assignment',
        ),
        # Prefer lower colour labels solely to make equivalent colourings
        # deterministic; feasibility remains the graph-colouring question.
        'objective': {
            'sense': 'minimize',
            'offset': 0,
            'linear': [
                [
                    _variable_index(vertex, color, color_count),
                    color,
                ]
                for vertex in range(vertex_count)
                for color in range(1, color_count)
            ],
            'quadratic': [],
        },
        'constraints': constraints,
        'fixed_values': [{'index': 0, 'value': 1}],
        'metadata': benchmark_metadata(
            family='constraint-satisfaction',
            model='graph-coloring',
            generator='build_graph_coloring_suite',
            size={
                'vertex_count': vertex_count,
                'edge_count': len(edges),
                'color_count': color_count,
                'num_variables': variable_count,
                'free_variable_count': variable_count - 1,
                'constraint_count': len(constraints),
            },
            parameters={
                'edges': edges,
                'encoding': 'vertex-by-color-one-hot',
                'fixed_vertex': 0,
                'fixed_color': 0,
            },
        ),
    }


def _variable_index(vertex, color, color_count):
    """Map a vertex/colour pair to its stable variable index."""
    return vertex * color_count + color


__all__ = ['build_graph_coloring_suite']
