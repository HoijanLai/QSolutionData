"""Classical graph-family MIS benchmarks with known structure."""

from .._metadata import benchmark_metadata
from ._common import _grid_edges, _make_mis, _path_edges


def build_mis_graph_family_suite():
    """Return path, cycle, clique, bipartite and grid instances."""
    return {
        'mis.path-32': _build_path(32),
        'mis.cycle-31': _build_cycle(31),
        'mis.clique-36': _build_clique(36),
        'mis.complete-bipartite-12x12': _build_complete_bipartite(
            12,
            12,
        ),
        'mis.grid-5x6': _build_grid(5, 6),
    }


def _build_path(vertex_count):
    """Build a path whose independence number is ceil(n / 2)."""
    edges = _path_edges(vertex_count)
    return _make_mis(
        problem_id=f'benchmark-mis-path-{vertex_count}',
        vertex_count=vertex_count,
        edges=edges,
        metadata=_metadata(
            model='path',
            vertex_count=vertex_count,
            edge_count=len(edges),
            parameters={
                'known_independence_number': (vertex_count + 1) // 2,
            },
        ),
    )


def _build_cycle(vertex_count):
    """Build an odd cycle with floor(n / 2) maximum cardinality."""
    edges = _path_edges(vertex_count) + [(0, vertex_count - 1)]
    return _make_mis(
        problem_id=f'benchmark-mis-cycle-{vertex_count}',
        vertex_count=vertex_count,
        edges=edges,
        metadata=_metadata(
            model='cycle',
            vertex_count=vertex_count,
            edge_count=len(edges),
            parameters={
                'known_independence_number': vertex_count // 2,
                'odd_cycle': vertex_count % 2 == 1,
            },
        ),
    )


def _build_clique(vertex_count):
    """Build a complete graph with independence number one."""
    edges = [
        (left, right)
        for left in range(vertex_count)
        for right in range(left + 1, vertex_count)
    ]
    return _make_mis(
        problem_id=f'benchmark-mis-clique-{vertex_count}',
        vertex_count=vertex_count,
        edges=edges,
        metadata=_metadata(
            model='complete-graph',
            vertex_count=vertex_count,
            edge_count=len(edges),
            parameters={'known_independence_number': 1},
        ),
    )


def _build_complete_bipartite(left_size, right_size):
    """Build K_(left,right), whose larger shore is an optimum."""
    vertex_count = left_size + right_size
    edges = [
        (left, left_size + right)
        for left in range(left_size)
        for right in range(right_size)
    ]
    return _make_mis(
        problem_id=(
            f'benchmark-mis-complete-bipartite-'
            f'{left_size}x{right_size}'
        ),
        vertex_count=vertex_count,
        edges=edges,
        metadata=_metadata(
            model='complete-bipartite',
            vertex_count=vertex_count,
            edge_count=len(edges),
            parameters={
                'left_size': left_size,
                'right_size': right_size,
                'known_independence_number': max(
                    left_size,
                    right_size,
                ),
            },
        ),
    )


def _build_grid(rows, columns):
    """Build a bipartite rectangular grid."""
    vertex_count = rows * columns
    edges = _grid_edges(rows, columns)
    return _make_mis(
        problem_id=f'benchmark-mis-grid-{rows}x{columns}',
        vertex_count=vertex_count,
        edges=edges,
        metadata=_metadata(
            model='rectangular-grid',
            vertex_count=vertex_count,
            edge_count=len(edges),
            parameters={
                'rows': rows,
                'columns': columns,
                'known_independence_number': (
                    vertex_count + 1
                ) // 2,
            },
        ),
    )


def _metadata(*, model, vertex_count, edge_count, parameters):
    """Attach consistent classification to every classical graph."""
    return benchmark_metadata(
        family='maximum-independent-set',
        model=model,
        generator='build_mis_graph_family_suite',
        size={
            'vertex_count': vertex_count,
            'edge_count': edge_count,
        },
        parameters=parameters,
    )


__all__ = ['build_mis_graph_family_suite']
