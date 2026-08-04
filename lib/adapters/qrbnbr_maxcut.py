import copy
import itertools
import math
import time
from collections.abc import Mapping
from fractions import Fraction
from numbers import Real

from ..contracts.validation import (
    _evaluate_qubo,
    _evaluate_qubo_exact,
    _fraction_to_json_number,
    _validate_qubo,
    validate_qubo_result,
)
from .maxcut_encoding import build_signed_maxcut_edges


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
        edges = build_signed_maxcut_edges(problem, anchor_node)
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
            'qubo_offset': copy.deepcopy(problem['offset']),
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
        verify_optimality=False,
        max_verification_variables=24,
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
            verify_optimality: Whether the adapter must independently enumerate
                the source QUBO before emitting ``optimal``.
            max_verification_variables: Safety limit for that exhaustive proof.

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
            verify_optimality,
            max_verification_variables,
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
                'optimality_verified': False,
            },
        }
        if termination_reason is not None:
            result['termination_reason'] = termination_reason

        if native_solution is None:
            if status in {'optimal', 'feasible'}:
                raise ValueError(
                    f"Status '{status}' requires a candidate native solution."
                )
            return _validated_result(result, context)

        if status == 'infeasible':
            raise ValueError(
                "Status 'infeasible' cannot include a candidate native solution."
            )

        sample = _decode_native_solution(native_solution.z, context)
        energy = _evaluate_qubo(context['qubo_problem'], sample)
        result['best_sample'] = sample
        result['best_energy'] = energy

        metrics = _build_native_metrics(native_solution, context, sample)
        if metrics:
            result['metrics'] = metrics
        optimality_proof = _verify_native_optimality(
            status,
            sample,
            context,
            verify_optimality,
            max_verification_variables,
        )
        if optimality_proof is not None:
            result['metadata']['optimality_verified'] = True
            result['metadata']['optimality_proof'] = optimality_proof
        trace = _build_native_trace(native_solution, context)
        if trace:
            result['trace'] = trace
        return _validated_result(result, context)


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
        proves_optimality=False,
        optimality_verification_max_variables=24,
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
            proves_optimality: If true, every ``optimal`` native result is
                independently checked by exhaustive enumeration of the source
                QUBO. It is not merely trusted as a backend capability flag.
            optimality_verification_max_variables: Safety limit for the
                independent exhaustive check.

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
        if not isinstance(proves_optimality, bool):
            raise TypeError('proves_optimality must be a boolean.')
        _validate_verification_limit(optimality_verification_max_variables)
        self._proves_optimality = proves_optimality
        self._optimality_verification_max_variables = (
            optimality_verification_max_variables
        )
        _validate_result_options(
            self._solver_name,
            self._solver_version,
            self._backend,
            self._result_status,
            None,
            self._proves_optimality,
            self._optimality_verification_max_variables,
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
        resolved_config = _validate_solver_config(
            config,
            proves_optimality=self._proves_optimality,
        )
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
            verify_optimality=self._proves_optimality,
            max_verification_variables=(
                self._optimality_verification_max_variables
            ),
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


def _validate_solver_config(config, proves_optimality=False):
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
        _require_optimality_verification(
            config['status'],
            proves_optimality,
        )
        output['status'] = config['status']
    if 'termination_reason' in config:
        reason = config['termination_reason']
        if reason is not None and not isinstance(reason, str):
            raise TypeError('termination_reason must be a string or None.')
        output['termination_reason'] = reason
    return output


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
    if (
        type(context.get('num_variables')) is not int
        or context.get('num_variables') != problem['num_variables']
    ):
        raise ValueError('Q-RBnBR context variable count is inconsistent.')
    if context.get('variable_names') != problem['variable_names']:
        raise ValueError('Q-RBnBR context variable names are inconsistent.')
    if (
        type(context.get('anchor_node')) is not int
        or context.get('anchor_node') != problem['num_variables']
    ):
        raise ValueError('Q-RBnBR context anchor node is inconsistent.')
    if (
        type(context.get('qubo_offset')) is not type(problem['offset'])
        or context.get('qubo_offset') != problem['offset']
    ):
        raise ValueError('Q-RBnBR context offset is inconsistent.')


def _validate_result_options(
    solver_name,
    solver_version,
    backend,
    status,
    runtime_seconds,
    verify_optimality=False,
    max_verification_variables=24,
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
    if not isinstance(verify_optimality, bool):
        raise TypeError('verify_optimality must be a boolean.')
    _validate_verification_limit(max_verification_variables)
    _require_optimality_verification(status, verify_optimality)
    if runtime_seconds is not None:
        _validate_runtime(runtime_seconds)


def _require_optimality_verification(status, verify_optimality):
    """Prevent an optimal label unless an independent proof will run."""
    if status == 'optimal' and not verify_optimality:
        raise ValueError(
            "Status 'optimal' requires independent exhaustive verification."
        )


def _verify_native_optimality(
    status,
    sample,
    context,
    verify_optimality,
    max_verification_variables,
):
    """Independently prove that no source-QUBO assignment is better."""
    if status != 'optimal':
        return None

    if not verify_optimality:
        raise ValueError('Independent optimality verification was not enabled.')
    problem = context['qubo_problem']
    variable_count = problem['num_variables']
    if variable_count > max_verification_variables:
        raise ValueError(
            f'Optimality verification received {variable_count} variables, '
            f'exceeding max_verification_variables='
            f'{max_verification_variables}.'
        )

    candidate_energy = _evaluate_qubo_exact(problem, sample)
    assignments_checked = 0
    for assignment in itertools.product((0, 1), repeat=variable_count):
        assignments_checked += 1
        if _evaluate_qubo_exact(problem, list(assignment)) < candidate_energy:
            raise ValueError(
                'Independent exhaustive verification found a better QUBO '
                'assignment; the native result is not optimal.'
            )
    return {
        'verified': True,
        'method': 'exhaustive-qubo-enumeration',
        'assignments_checked': assignments_checked,
    }


def _validate_verification_limit(value):
    """Validate the safety rail used by independent optimality proof."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            'max_verification_variables must be a non-negative integer.'
        )


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
        not isinstance(runtime, Real)
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
    if any(
        isinstance(value, bool) or value not in allowed
        for value in values
    ):
        raise ValueError('Q-RBnBR solution must use 0/1 or -1/1 encoding.')
    if -1 in values and any(value == 0 for value in values):
        raise ValueError('Q-RBnBR solution mixes binary and spin encodings.')

    anchor_value = values[context['anchor_node']]
    return [
        int(values[index] != anchor_value)
        for index in range(context['num_variables'])
    ]


def _build_native_metrics(native_solution, context, sample):
    """Build native metrics."""
    metrics = {}
    native_cost = getattr(native_solution, 'cost', None)
    if native_cost is not None:
        native_cost = _finite_float(
            native_cost,
            'Q-RBnBR solution cost',
        )
        problem = context['qubo_problem']
        expected_cost = (
            Fraction(problem['offset'])
            - _evaluate_qubo_exact(problem, sample)
        )
        expected_cost_json = _fraction_to_json_number(
            expected_cost,
            'Q-RBnBR expected cut value',
        )
        metrics['native_cut_value'] = native_cost
        metrics['energy_consistent'] = (
            Fraction(native_cost) == Fraction(expected_cost_json)
        )

    approximation_ratio = getattr(native_solution, 'approx_ratio', None)
    if approximation_ratio is not None:
        metrics['native_approximation_ratio'] = _finite_float(
            approximation_ratio,
            'Q-RBnBR solution approximation ratio',
        )
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
        metadata = {}
        if step.get('cost') is not None:
            metadata['native_cut_value'] = _finite_float(
                step['cost'],
                'Q-RBnBR trace cost',
            )
        if step.get('approx_ratio') is not None:
            metadata['native_approximation_ratio'] = _finite_float(
                step['approx_ratio'],
                'Q-RBnBR trace approximation ratio',
            )
        if metadata:
            output['metadata'] = metadata
        trace.append(output)
    return trace


def _finite_float(value, label):
    """Return a JSON-native float while rejecting booleans, NaN, and infinity."""
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f'{label} must be a finite real number.')
    return float(value)


def _validated_result(result, context):
    """Apply the shared runtime contract before exposing an adapter result."""
    validate_qubo_result(
        problem=context['qubo_problem'],
        result=result,
    )
    return result
