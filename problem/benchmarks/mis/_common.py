"""Protected constructors shared by native MIS benchmark families."""


def _make_mis(
    *,
    problem_id,
    vertex_count,
    edges,
    metadata,
    weights=None,
    fixed_values=None,
):
    """Create one canonical MIS document from a mathematical edge list."""
    canonical_edges = sorted({
        (min(left, right), max(left, right))
        for left, right in edges
        if left != right
    })
    if weights is not None and len(weights) != vertex_count:
        raise ValueError('weights length must equal vertex_count.')

    vertices = []
    for index in range(vertex_count):
        vertex = {'index': index, 'name': f'v_{index}'}
        if weights is not None:
            vertex['weight'] = weights[index]
        vertices.append(vertex)

    return {
        'schema': 'mis.v1',
        'problem_id': problem_id,
        'objective': {
            'kind': (
                'maximum-cardinality'
                if weights is None
                else 'maximum-weight'
            )
        },
        'vertices': vertices,
        'edges': [
            [left, right]
            for left, right in canonical_edges
        ],
        'fixed_values': list(fixed_values or []),
        'metadata': metadata,
    }


def _path_edges(vertex_count):
    """Return consecutive edges for a path."""
    return [
        (index, index + 1)
        for index in range(vertex_count - 1)
    ]


def _grid_edges(rows, columns):
    """Return horizontal and vertical edges of a rectangular grid."""
    edges = []
    for row in range(rows):
        for column in range(columns):
            vertex = row * columns + column
            if column + 1 < columns:
                edges.append((vertex, vertex + 1))
            if row + 1 < rows:
                edges.append((vertex, vertex + columns))
    return edges


__all__ = ['_grid_edges', '_make_mis', '_path_edges']
