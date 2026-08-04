"""Unified catalog for contract-native mathematical benchmark problems."""

from .cbqm import build_cbqm_mathematical_suite
from .mis import build_mis_mathematical_suite
from .qubo import build_qubo_mathematical_suite
from .references import build_mathematical_benchmark_references
from ..problem_def import (
    BestKnownSolution,
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
)


def build_mathematical_benchmark_suite():
    """Return all deterministic fixtures keyed by stable benchmark name."""
    suite = {}
    for representation_suite in (
        build_qubo_mathematical_suite(),
        build_cbqm_mathematical_suite(),
        build_mis_mathematical_suite(),
    ):
        _merge_without_duplicates(suite, representation_suite)
    return suite


def build_mathematical_benchmark_cases():
    """Wrap every fixture as one canonical-artifact ``ProblemCase``."""
    references = build_mathematical_benchmark_references()
    suite = build_mathematical_benchmark_suite()
    if set(references) != set(suite):
        raise RuntimeError(
            'Mathematical benchmark problems and exact references disagree.'
        )
    return {
        name: _problem_case_for(name, problem, references[name])
        for name, problem in suite.items()
    }


def _merge_without_duplicates(target, source):
    """Keep catalog identity independent from dictionary overwrite semantics."""
    overlap = set(target) & set(source)
    if overlap:
        raise RuntimeError(
            f'Duplicate mathematical benchmark names: {sorted(overlap)}'
        )
    target.update(source)


def _problem_case_for(benchmark_name, problem, reference):
    """Derive one task envelope from the problem's native representation."""
    representation = problem['schema']
    artifact_id = representation.removesuffix('.v1')
    sense, solution_representation = _task_semantics(problem)
    benchmark = problem['metadata']['benchmark']
    return ProblemCase(
        problem_id=problem['problem_id'],
        artifacts=(
            ProblemArtifact(
                artifact_id=artifact_id,
                representation=representation,
                payload=problem,
                metadata={'benchmark_name': benchmark_name},
            ),
        ),
        tasks=(
            TaskDefinition(
                task_id='optimize',
                canonical_artifact_id=artifact_id,
                sense=sense,
                solution_representation=solution_representation,
                task_type=benchmark['model'],
                best_known=BestKnownSolution(
                    solution=reference['solution'],
                    objective_value=reference['objective_value'],
                    exact=True,
                    source='mathematical-benchmark-exact-audit',
                    metadata={
                        'benchmark_name': benchmark_name,
                        'audited_on': '2026-07-31',
                        'method': _reference_method(representation),
                    },
                ),
                metadata={
                    'benchmark_name': benchmark_name,
                    'benchmark_family': benchmark['family'],
                },
            ),
        ),
        primary_artifact_id=artifact_id,
        metadata={
            'benchmark_name': benchmark_name,
            'benchmark_family': benchmark['family'],
        },
    )


def _reference_method(representation):
    """Name the proof method used to establish checked-in references."""
    return {
        'qubo.v1': 'exhaustive-qubo-enumeration',
        'cbqm.v1': 'exhaustive-cbqm-enumeration',
        'mis.v1': 'exact-mis-branch-and-reduce',
    }[representation]


def _task_semantics(problem):
    """Map a native contract to its canonical task vocabulary."""
    if problem['schema'] == 'qubo.v1':
        return 'minimize', 'binary-vector.v1'
    if problem['schema'] == 'cbqm.v1':
        return problem['objective']['sense'], 'binary-vector.v1'
    if problem['schema'] == 'mis.v1':
        return 'maximize', 'vertex-index-set.v1'
    raise ValueError(
        f"Unsupported benchmark schema '{problem['schema']}'."
    )


__all__ = [
    'build_mathematical_benchmark_cases',
    'build_mathematical_benchmark_suite',
]
