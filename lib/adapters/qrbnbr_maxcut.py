import copy
import math
import time
from collections.abc import Mapping

from ..contracts.validation import _evaluate_qubo, _validate_qubo


_RESULT_STATUSES = {
    'optimal',
    'feasible',
    'infeasible',
    'timeout',
    'error',
    'unknown',
}


class QRBnBRMaxCutAdapter:
    """Bridge canonical QUBO models and Q-RBnBR's MaxCut representation.

    The conversion adds one anchor node and preserves an exact energy mapping:
    ``qubo_energy = qubo_offset - cut_value``. Native result conversion restores
    the canonical sample order and recomputes energy from the source QUBO.
    """

    def from_qubo(
        self,
        problem,
        maxcut_problem_class=None,
        graph_class=None,
    ):
        """Convert ``qubo.v1`` into a Q-RBnBR ``MaxCutProblem``.

        Args:
            problem: Mapping conforming to canonical minimization ``qubo.v1``.
            maxcut_problem_class: Optional Q-RBnBR-compatible problem class for
                dependency injection.
            graph_class: Optional NetworkX-compatible graph class for dependency
                injection.

        Returns:
            ``(maxcut_problem, context)``. Context contains the source QUBO,
            anchor node, variable mapping, offset, and energy relation.

        Raises:
            ImportError: If optional dependencies are missing and no compatible
                classes are injected.
            TypeError: If the input has invalid container types.
            ValueError: If the QUBO violates the canonical contract.
        """
        _validate_qubo(problem)
        resolved_graph_class, resolved_problem_class = _resolve_qrbnbr_classes(
            graph_class,
            maxcut_problem_class,
        )

        anchor_node = problem['num_variables']
        edges = _build_maxcut_edges(problem, anchor_node)
        graph = resolved_graph_class()
        graph.add_nodes_from(range(problem['num_variables'] + 1))
        for left, right, weight in edges:
            graph.add_edge(left, right, weight=weight)

        maxcut_problem = resolved_problem_class(
            graph,
            solve=False,
            name=problem['problem_id'],
        )
        context = {
            'schema': 'qrbnbr-maxcut-context.v1',
            'problem_id': problem['problem_id'],
            'num_variables': problem['num_variables'],
            'variable_names': copy.deepcopy(problem['variable_names']),
            'anchor_node': anchor_node,
            'qubo_offset': float(problem['offset']),
            'qubo_problem': copy.deepcopy(problem),
            'energy_relation': 'qubo_energy = qubo_offset - cut_value',
        }
        return maxcut_problem, context

    def to_result(
        self,
        native_solution,
        context,
        solver_name='qrbnbr',
        solver_version='unknown',
        backend='MaxCutProblem',
        status='feasible',
        runtime_seconds=None,
        termination_reason=None,
    ):
        """Convert a native Q-RBnBR solution into ``qubo-result.v1``.

        Args:
            native_solution: Q-RBnBR-like solution exposing ``z`` and optional
                cost, ratio, runtime data, and breadcrumb access. Use ``None``
                when no candidate exists.
            context: Context returned by :meth:`from_qubo`.
            solver_name: Stable solver identifier stored in the result.
            solver_version: Solver implementation version.
            backend: Human-readable native backend name.
            status: Canonical result status.
            runtime_seconds: Optional explicit runtime override.
            termination_reason: Optional machine-readable or human-readable
                explanation for termination.

        Returns:
            A ``qubo-result.v1`` dictionary with decoded binary sample, canonical
            QUBO energy, optional native metrics, and optional trace.

        Raises:
            TypeError: If solution or context objects have invalid types.
            ValueError: If context, status, encoding, runtime, or native values
                are inconsistent.
        """
        _validate_qrbnbr_context(context)
        _validate_result_options(
            solver_name,
            solver_version,
            backend,
            status,
            runtime_seconds,
        )

        runtime = _resolve_runtime(native_solution, runtime_seconds)
        solver = {
            'name': solver_name,
            'version': solver_version,
            'backend': backend,
        }
        result = {
            'schema': 'qubo-result.v1',
            'problem_id': context['problem_id'],
            'solver': solver,
            'status': status,
            'best_sample': None,
            'best_energy': None,
            'runtime_seconds': runtime,
            'metadata': {
                'adapter': 'qrbnbr-maxcut.v1',
                'anchor_node': context['anchor_node'],
            },
        }
        if termination_reason is not None:
            result['termination_reason'] = termination_reason

        if native_solution is None:
            if status in {'optimal', 'feasible'}:
                raise ValueError(
                    f"Status '{status}' requires a candidate native solution."
                )
            return result

        sample = _decode_native_solution(native_solution.z, context)
        energy = _evaluate_qubo(context['qubo_problem'], sample)
        result['best_sample'] = sample
        result['best_energy'] = energy

        metrics = _build_native_metrics(native_solution, context, energy)
        if metrics:
            result['metrics'] = metrics
        trace = _build_native_trace(native_solution, context)
        if trace:
            result['trace'] = trace
        return result


class QRBnBRSolverAdapter:
    """Expose an existing Q-RBnBR solver through the canonical solver protocol.

    The wrapped native solver continues to operate on ``MaxCutProblem`` while
    callers interact only with ``solve(qubo.v1, config) -> qubo-result.v1``.
    """

    def __init__(
        self,
        native_solver,
        solver_name=None,
        solver_version='unknown',
        backend='MaxCutProblem',
        result_status='feasible',
        maxcut_problem_class=None,
        graph_class=None,
    ):
        """Configure a canonical wrapper around a native Q-RBnBR solver.

        Args:
            native_solver: Object with a callable ``solve`` method.
            solver_name: Optional result identifier; defaults to the native
                solver class name.
            solver_version: Version reported in canonical results.
            backend: Backend label reported in canonical results.
            result_status: Default status for returned native candidates.
            maxcut_problem_class: Optional injected problem class.
            graph_class: Optional injected graph class.

        Raises:
            TypeError: If ``native_solver`` has no callable solve method.
            ValueError: If result metadata or status is invalid.
        """
        if not callable(getattr(native_solver, 'solve', None)):
            raise TypeError('native_solver must provide a callable solve method.')
        self._native_solver = native_solver
        self._problem_adapter = QRBnBRMaxCutAdapter()
        self._solver_name = solver_name or type(native_solver).__name__
        self._solver_version = solver_version
        self._backend = backend
        self._result_status = result_status
        self._maxcut_problem_class = maxcut_problem_class
        self._graph_class = graph_class
        _validate_result_options(
            self._solver_name,
            self._solver_version,
            self._backend,
            self._result_status,
            None,
        )

    def solve(self, problem, config=None):
        """Solve a canonical QUBO through the wrapped native solver.

        Args:
            problem: Mapping conforming to ``qubo.v1``.
            config: Optional mapping with ``native_solve_kwargs``, a result
                ``status`` override, and ``termination_reason``.

        Returns:
            A canonical ``qubo-result.v1`` dictionary. Wall-clock runtime covers
            the native solve call, and energy is recomputed from the input QUBO.

        Raises:
            TypeError: If problem or config containers are invalid.
            ValueError: If QUBO, config, native result, or adapter context is
                inconsistent.
            ImportError: If required Q-RBnBR dependencies are unavailable.
        """
        resolved_config = _validate_solver_config(config)
        maxcut_problem, context = self._problem_adapter.from_qubo(
            problem,
            maxcut_problem_class=self._maxcut_problem_class,
            graph_class=self._graph_class,
        )

        started_at = time.perf_counter()
        native_solution = self._native_solver.solve(
            maxcut_problem,
            **resolved_config['native_solve_kwargs'],
        )
        runtime = time.perf_counter() - started_at
        status = resolved_config.get('status', self._result_status)
        if native_solution is None and status in {'optimal', 'feasible'}:
            status = 'unknown'

        return self._problem_adapter.to_result(
            native_solution,
            context,
            solver_name=self._solver_name,
            solver_version=self._solver_version,
            backend=self._backend,
            status=status,
            runtime_seconds=runtime,
            termination_reason=resolved_config.get('termination_reason'),
        )


def _resolve_qrbnbr_classes(graph_class, maxcut_problem_class):
    """Resolve qrbnbr classes."""
    if graph_class is None:
        try:
            import networkx as nx
        except ImportError as error:
            raise ImportError(
                'Q-RBnBR adapter requires networkx. Install Q-RBnBR dependencies '
                'or pass graph_class for dependency injection.'
            ) from error
        graph_class = nx.Graph

    if maxcut_problem_class is None:
        try:
            from rbnbr.problems.max_cut import MaxCutProblem
        except ImportError as error:
            raise ImportError(
                'Q-RBnBR adapter requires the rbnbr package. Install Q-RBnBR or '
                'pass maxcut_problem_class for dependency injection.'
            ) from error
        maxcut_problem_class = MaxCutProblem
    return graph_class, maxcut_problem_class


def _validate_solver_config(config):
    """Validate solver config."""
    if config is None:
        return {'native_solve_kwargs': {}}
    if not isinstance(config, Mapping):
        raise TypeError('solver config must be a mapping or None.')

    allowed = {'native_solve_kwargs', 'status', 'termination_reason'}
    unknown = sorted(set(config).difference(allowed))
    if unknown:
        raise ValueError(f'Unknown Q-RBnBR solver config fields: {unknown}')

    native_solve_kwargs = config.get('native_solve_kwargs', {})
    if not isinstance(native_solve_kwargs, Mapping):
        raise TypeError('native_solve_kwargs must be a mapping.')
    output = {
        'native_solve_kwargs': copy.deepcopy(dict(native_solve_kwargs)),
    }
    if 'status' in config:
        if config['status'] not in _RESULT_STATUSES:
            raise ValueError(f"Unknown solver result status '{config['status']}'.")
        output['status'] = config['status']
    if 'termination_reason' in config:
        reason = config['termination_reason']
        if reason is not None and not isinstance(reason, str):
            raise TypeError('termination_reason must be a string or None.')
        output['termination_reason'] = reason
    return output


def _build_maxcut_edges(problem, anchor_node):
    """Build maxcut edges."""
    edge_weights = {}
    for left, right, coefficient in problem['terms']:
        coefficient = float(coefficient)
        if left == right:
            _add_edge_weight(edge_weights, left, anchor_node, -coefficient)
            continue

        half_coefficient = coefficient / 2.0
        _add_edge_weight(edge_weights, left, right, half_coefficient)
        _add_edge_weight(edge_weights, left, anchor_node, -half_coefficient)
        _add_edge_weight(edge_weights, right, anchor_node, -half_coefficient)

    return [
        [left, right, weight]
        for (left, right), weight in sorted(edge_weights.items())
        if weight != 0
    ]


def _add_edge_weight(edge_weights, left, right, weight):
    """Accumulate one canonical undirected MaxCut edge weight."""
    edge = (min(left, right), max(left, right))
    edge_weights[edge] = edge_weights.get(edge, 0.0) + weight


def _validate_qrbnbr_context(context):
    """Validate qrbnbr context."""
    if not isinstance(context, Mapping):
        raise TypeError('Q-RBnBR adapter context must be a mapping.')
    if context.get('schema') != 'qrbnbr-maxcut-context.v1':
        raise ValueError(
            "Q-RBnBR adapter context schema must be 'qrbnbr-maxcut-context.v1'."
        )
    _validate_qubo(context.get('qubo_problem'))
    problem = context['qubo_problem']
    if context.get('problem_id') != problem['problem_id']:
        raise ValueError('Q-RBnBR context problem_id is inconsistent.')
    if context.get('num_variables') != problem['num_variables']:
        raise ValueError('Q-RBnBR context variable count is inconsistent.')
    if context.get('variable_names') != problem['variable_names']:
        raise ValueError('Q-RBnBR context variable names are inconsistent.')
    if context.get('anchor_node') != problem['num_variables']:
        raise ValueError('Q-RBnBR context anchor node is inconsistent.')
    if context.get('qubo_offset') != float(problem['offset']):
        raise ValueError('Q-RBnBR context offset is inconsistent.')


def _validate_result_options(
    solver_name,
    solver_version,
    backend,
    status,
    runtime_seconds,
):
    """Validate result options."""
    for label, value in (
        ('solver_name', solver_name),
        ('solver_version', solver_version),
        ('backend', backend),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f'{label} must be a non-empty string.')
    if status not in _RESULT_STATUSES:
        raise ValueError(f"Unknown solver result status '{status}'.")
    if runtime_seconds is not None:
        _validate_runtime(runtime_seconds)


def _resolve_runtime(native_solution, runtime_seconds):
    """Resolve runtime."""
    if runtime_seconds is not None:
        return float(runtime_seconds)
    if native_solution is None:
        return 0.0

    data = getattr(native_solution, 'data', None)
    native_runtime = data.get('time') if isinstance(data, Mapping) else None
    if native_runtime is None:
        return 0.0
    _validate_runtime(native_runtime)
    return float(native_runtime)


def _validate_runtime(runtime):
    """Validate runtime."""
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or not math.isfinite(runtime)
        or runtime < 0
    ):
        raise ValueError('runtime_seconds must be a finite non-negative number.')


def _decode_native_solution(native_values, context):
    """Decode native node labels relative to the anchor node."""
    try:
        values = list(native_values)
    except TypeError as error:
        raise TypeError('Q-RBnBR solution must be an iterable.') from error

    expected_length = context['num_variables'] + 1
    if len(values) != expected_length:
        raise ValueError(
            'Q-RBnBR solution length must include every variable and the anchor node.'
        )
    allowed = {0, 1, -1}
    if any(value not in allowed for value in values):
        raise ValueError('Q-RBnBR solution must use 0/1 or -1/1 encoding.')
    if -1 in values and any(value == 0 for value in values):
        raise ValueError('Q-RBnBR solution mixes binary and spin encodings.')

    anchor_value = values[context['anchor_node']]
    return [
        int(values[index] != anchor_value)
        for index in range(context['num_variables'])
    ]


def _build_native_metrics(native_solution, context, energy):
    """Build native metrics."""
    metrics = {}
    native_cost = getattr(native_solution, 'cost', None)
    if native_cost is not None:
        if not isinstance(native_cost, (int, float)) or not math.isfinite(native_cost):
            raise ValueError('Q-RBnBR solution cost must be finite.')
        native_cost = float(native_cost)
        expected_cost = context['qubo_offset'] - energy
        metrics['native_cut_value'] = native_cost
        metrics['energy_consistent'] = math.isclose(
            native_cost,
            expected_cost,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    approximation_ratio = getattr(native_solution, 'approx_ratio', None)
    if approximation_ratio is not None:
        metrics['native_approximation_ratio'] = float(approximation_ratio)
    return metrics


def _build_native_trace(native_solution, context):
    """Build native trace."""
    try:
        step_count = len(native_solution)
    except TypeError:
        return []

    trace = []
    for index in range(step_count):
        step = native_solution[index]
        if not isinstance(step, Mapping) or step.get('solution') is None:
            continue
        sample = _decode_native_solution(step['solution'], context)
        time_seconds = step.get('time')
        if time_seconds is None:
            time_seconds = 0.0
        _validate_runtime(time_seconds)
        output = {
            'step': index,
            'time_seconds': float(time_seconds),
            'energy': _evaluate_qubo(context['qubo_problem'], sample),
            'sample': sample,
        }
        if step.get('cost') is not None:
            output['metadata'] = {'native_cut_value': float(step['cost'])}
        trace.append(output)
    return trace
