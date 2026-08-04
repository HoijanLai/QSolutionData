"""Small explicit registries for representation-owned problem components.

The problem layer should not grow a new ``if representation == ...`` branch
every time the repository gains a native model.  These registries provide the
three extension seams that a representation actually needs:

* deep artifact validation;
* task-solution evaluation; and
* native solver execution/result interpretation.

Registration is intentionally explicit and duplicate-safe.  Silent replacement
would make an experiment depend on import order, so callers must opt in with
``replace=True`` when replacement is genuinely intended.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class NativeSolverRunner:
    """Representation-specific hooks used by the generic native case runner.

    ``run_and_validate``
        Calls a solver on a defensive payload/config copy and returns its
        validated native result mapping.
    ``candidate_from_result``
        Extracts a canonical task solution, or ``None`` when the result has no
        usable feasible incumbent.
    ``verify_exact``
        Independently evaluates a solver exactness claim within the caller's
        verification budget and returns ``(verified, evidence)``.
    ``route_kind``
        Stable provenance label stored with best-known exactness evidence.
    """

    run_and_validate: Callable
    candidate_from_result: Callable
    verify_exact: Callable
    route_kind: str

    def __post_init__(self):
        for field_name in (
            'run_and_validate',
            'candidate_from_result',
            'verify_exact',
        ):
            if not callable(getattr(self, field_name)):
                raise TypeError(f'{field_name} must be callable.')
        _validate_registry_name(self.route_kind, 'route_kind')


class _ComponentRegistry:
    """Duplicate-safe mapping with a deliberately tiny mutation surface."""

    def __init__(self, component_name):
        self._component_name = component_name
        self._components = {}

    def register(self, key, component, *, replace=False):
        """Register one callable component and return it for decorator use."""
        if not isinstance(key, Hashable):
            raise TypeError(f'{self._component_name} key must be hashable.')
        if not callable(component):
            raise TypeError(f'{self._component_name} must be callable.')
        _validate_replace_flag(replace)
        if key in self._components and not replace:
            raise ValueError(
                f'{self._component_name} is already registered for {key!r}.'
            )
        self._components[key] = component
        return component

    def resolve(self, key):
        """Return a registered component or ``None`` without side effects."""
        return self._components.get(key)

    def snapshot(self):
        """Expose an immutable diagnostic view, never the mutable dictionary."""
        return MappingProxyType(dict(self._components))


class _NativeRunnerRegistry:
    """Typed counterpart for the multi-hook native runner adapter."""

    def __init__(self):
        self._runners = {}

    def register(self, representation, runner, *, replace=False):
        _validate_registry_name(representation, 'representation')
        if not isinstance(runner, NativeSolverRunner):
            raise TypeError('runner must be a NativeSolverRunner.')
        _validate_replace_flag(replace)
        if representation in self._runners and not replace:
            raise ValueError(
                'native solver runner is already registered for '
                f'{representation!r}.'
            )
        self._runners[representation] = runner
        return runner

    def resolve(self, representation):
        return self._runners.get(representation)

    def snapshot(self):
        return MappingProxyType(dict(self._runners))


_REPRESENTATION_VALIDATORS = _ComponentRegistry(
    'representation validator'
)
_TASK_EVALUATORS = _ComponentRegistry('task evaluator')
_NATIVE_SOLVER_RUNNERS = _NativeRunnerRegistry()


def register_representation_validator(
    representation,
    validator,
    *,
    replace=False,
):
    """Register ``validator(problem_case, artifact)`` for one representation."""
    _validate_registry_name(representation, 'representation')
    return _REPRESENTATION_VALIDATORS.register(
        representation,
        validator,
        replace=replace,
    )


def register_task_evaluator(
    artifact_representation,
    solution_representation,
    evaluator,
    *,
    replace=False,
):
    """Register ``evaluator(artifact, solution, task)`` for one format pair."""
    _validate_registry_name(
        artifact_representation,
        'artifact_representation',
    )
    _validate_registry_name(
        solution_representation,
        'solution_representation',
    )
    return _TASK_EVALUATORS.register(
        (artifact_representation, solution_representation),
        evaluator,
        replace=replace,
    )


def register_native_solver_runner(
    representation,
    runner,
    *,
    replace=False,
):
    """Register one complete native solver adapter for a representation."""
    return _NATIVE_SOLVER_RUNNERS.register(
        representation,
        runner,
        replace=replace,
    )


def registered_problem_components():
    """Return immutable snapshots for diagnostics and reproducibility logs."""
    return {
        'representation_validators': (
            _REPRESENTATION_VALIDATORS.snapshot()
        ),
        'task_evaluators': _TASK_EVALUATORS.snapshot(),
        'native_solver_runners': _NATIVE_SOLVER_RUNNERS.snapshot(),
    }


def _representation_validator_for(representation):
    """Resolve one deep artifact validator."""
    return _REPRESENTATION_VALIDATORS.resolve(representation)


def _task_evaluator_for(
    artifact_representation,
    solution_representation,
):
    """Resolve one objective evaluator by both sides of its contract."""
    return _TASK_EVALUATORS.resolve(
        (artifact_representation, solution_representation)
    )


def _native_solver_runner_for(representation):
    """Resolve one native execution adapter."""
    return _NATIVE_SOLVER_RUNNERS.resolve(representation)


def _validate_registry_name(value, label):
    """Require stable non-empty string keys and provenance labels."""
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a non-empty string.')


def _validate_replace_flag(replace):
    """Avoid Python truthiness changing global registry state."""
    if type(replace) is not bool:
        raise TypeError('replace must be a boolean.')


__all__ = [
    'NativeSolverRunner',
    'register_native_solver_runner',
    'register_representation_validator',
    'register_task_evaluator',
    'registered_problem_components',
]
