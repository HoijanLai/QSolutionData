"""End-to-end tests for registered direct native solver execution."""

import copy
import unittest

from lib.contracts import (
    evaluate_cbqm_feasibility,
    evaluate_cbqm_objective,
)
from lib.solvers.cbqm import ExactCbqmSolver
from problem import (
    NativeSolverRunner,
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
    register_native_solver_runner,
    register_task_evaluator,
    solve_native_problem_task,
)


def _cbqm(problem_id='native-cbqm'):
    """Return a two-variable select-one constrained model."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': problem_id,
        'variables': [
            {'index': 0, 'name': 'x', 'vartype': 'BINARY'},
            {'index': 1, 'name': 'y', 'vartype': 'BINARY'},
        ],
        'objective': {
            'sense': 'minimize',
            'offset': 0,
            'linear': [[0, -2], [1, -1]],
            'quadratic': [],
        },
        'constraints': [
            {
                'name': 'select_one',
                'family': 'selection',
                'linear': [[0, 1], [1, 1]],
                'lower_bound': 1,
                'upper_bound': 1,
            },
        ],
        'fixed_values': [],
        'metadata': {},
    }


def _cbqm_case(payload=None):
    """Wrap a CBQM as the canonical artifact of one binary-vector task."""
    problem = _cbqm() if payload is None else payload
    return ProblemCase(
        problem_id=problem['problem_id'],
        artifacts=(
            ProblemArtifact(
                artifact_id='cbqm',
                representation='cbqm.v1',
                payload=problem,
            ),
        ),
        tasks=(
            TaskDefinition(
                task_id='select-one',
                canonical_artifact_id='cbqm',
                sense=problem['objective']['sense'],
            ),
        ),
    )


class _FixedCbqmResultSolver:
    """Return one contract-valid nominated sample with a chosen status."""

    def __init__(self, sample, status):
        self.sample = sample
        self.status = status

    def solve(self, problem, config=None):
        del config
        sample = None if self.sample is None else list(self.sample)
        return {
            'schema': 'cbqm-result.v1',
            'problem_id': problem['problem_id'],
            'solver': {
                'name': 'fixed-cbqm-result',
                'version': '1',
                'backend': 'test',
            },
            'status': self.status,
            'best_sample': sample,
            'best_objective': (
                None
                if sample is None
                else evaluate_cbqm_objective(problem, sample)
            ),
            'feasibility': (
                None
                if sample is None
                else evaluate_cbqm_feasibility(problem, sample)
            ),
            'runtime_seconds': 0,
            'metadata': {},
        }


class _SideEffectSolver:
    def __init__(self):
        self.calls = 0

    def solve(self, problem, config=None):
        del problem, config
        self.calls += 1
        raise AssertionError('invalid input reached solver')


class NativeProblemSolvingTests(unittest.TestCase):
    def test_exact_cbqm_updates_only_after_independent_native_verification(self):
        case = _cbqm_case()

        record = solve_native_problem_task(
            case,
            'select-one',
            'cbqm',
            ExactCbqmSolver(),
            update_best=True,
        )

        self.assertEqual('cbqm-result.v1', record.raw_result['schema'])
        self.assertEqual([1, 0], record.canonical_solution)
        self.assertEqual(-2, record.canonical_objective_value)
        self.assertTrue(record.exact_for_task)
        self.assertTrue(record.update.current.exact)
        evidence = record.update.current.metadata['exactness']
        self.assertEqual('direct_cbqm', evidence['route'])
        self.assertTrue(
            evidence['independent_cbqm_optimality_verified']
        )

    def test_forged_optimal_status_does_not_promote_a_worse_cbqm_candidate(self):
        case = _cbqm_case()

        record = solve_native_problem_task(
            case,
            'select-one',
            'cbqm',
            _FixedCbqmResultSolver([0, 1], 'optimal'),
            update_best=True,
        )

        self.assertEqual('optimal', record.raw_result['status'])
        self.assertEqual([0, 1], record.canonical_solution)
        self.assertFalse(record.exact_for_task)
        self.assertFalse(record.update.current.exact)
        self.assertEqual(
            'better_assignment_found',
            record.update.current.metadata['exactness'][
                'optimality_verification_reason'
            ],
        )

    def test_verification_limit_keeps_exact_solver_claim_non_persistent(self):
        record = solve_native_problem_task(
            _cbqm_case(),
            'select-one',
            'cbqm',
            ExactCbqmSolver(),
            update_best=True,
            exact_verification_max_variables=1,
        )

        self.assertFalse(record.exact_for_task)
        self.assertEqual(
            'variable_limit_exceeded',
            record.update.current.metadata['exactness'][
                'optimality_verification_reason'
            ],
        )

    def test_infeasible_timeout_sample_remains_only_in_the_raw_result(self):
        record = solve_native_problem_task(
            _cbqm_case(),
            'select-one',
            'cbqm',
            _FixedCbqmResultSolver([0, 0], 'timeout'),
            update_best=True,
        )

        self.assertEqual([0, 0], record.raw_result['best_sample'])
        self.assertFalse(record.raw_result['feasibility']['feasible'])
        self.assertIsNone(record.canonical_solution)
        self.assertIsNone(record.update)

    def test_invalid_cbqm_is_rejected_before_solver_side_effect(self):
        problem = _cbqm('invalid-native-cbqm')
        problem['metadata'] = 'not-an-object'
        case = _cbqm_case(problem)
        solver = _SideEffectSolver()

        with self.assertRaisesRegex(TypeError, 'must be an object'):
            solve_native_problem_task(
                case,
                'select-one',
                'cbqm',
                solver,
            )
        self.assertEqual(0, solver.calls)

    def test_native_execution_requires_the_task_canonical_artifact(self):
        case = _cbqm_case()
        other = ProblemArtifact(
            artifact_id='other',
            representation='cbqm.v1',
            payload=_cbqm(),
        )
        case = case.with_artifact(other)

        with self.assertRaisesRegex(ValueError, 'canonical_artifact_id'):
            solve_native_problem_task(
                case,
                'select-one',
                'other',
                ExactCbqmSolver(),
            )

    def test_a_registered_custom_native_format_needs_no_dispatch_branch(self):
        representation = 'registry-native-score.v1'
        solution_representation = 'registry-native-solution.v1'

        def evaluate(artifact, solution, task):
            del task
            return artifact.payload['factor'] * solution['value']

        def run_and_validate(payload, solver, config):
            result = solver.solve(copy.deepcopy(payload), copy.deepcopy(config))
            if result.get('schema') != 'registry-native-result.v1':
                raise ValueError('wrong custom result schema')
            return copy.deepcopy(result)

        register_task_evaluator(
            representation,
            solution_representation,
            evaluate,
        )
        register_native_solver_runner(
            representation,
            NativeSolverRunner(
                run_and_validate=run_and_validate,
                candidate_from_result=lambda result: result['solution'],
                verify_exact=lambda *args: (
                    False,
                    {'verification_reason': 'custom_not_verified'},
                ),
                route_kind='direct_registry_native_score',
            ),
        )

        class Solver:
            def solve(self, problem, config=None):
                del problem, config
                return {
                    'schema': 'registry-native-result.v1',
                    'solver': {'name': 'custom', 'version': '1'},
                    'status': 'feasible',
                    'solution': {'value': 4},
                }

        case = ProblemCase(
            problem_id='custom-native-case',
            artifacts=(
                ProblemArtifact(
                    artifact_id='model',
                    representation=representation,
                    payload={'factor': 3},
                ),
            ),
            tasks=(
                TaskDefinition(
                    task_id='score',
                    canonical_artifact_id='model',
                    sense='maximize',
                    solution_representation=solution_representation,
                ),
            ),
        )

        record = solve_native_problem_task(
            case,
            'score',
            'model',
            Solver(),
            update_best=True,
        )

        self.assertEqual({'value': 4}, record.canonical_solution)
        self.assertEqual(12, record.canonical_objective_value)
        self.assertTrue(record.update.updated)
        self.assertFalse(record.exact_for_task)


if __name__ == '__main__':
    unittest.main()
