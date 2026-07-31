"""End-to-end tests for registered direct native solver execution."""

import copy
import unittest

from lib.contracts import (
    evaluate_cbqm_feasibility,
    evaluate_cbqm_objective,
    evaluate_mis_solution,
)
from lib.solvers.cbqm import ExactCbqmSolver, LocalSearchCbqmSolver
from lib.solvers.mis import ExactMisSolver, GreedyMisSolver
from problem import (
    BestKnownSolution,
    NativeSolverRunner,
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
    register_native_solver_runner,
    register_task_evaluator,
    solve_native_problem_task,
    validate_problem_case,
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


def _mis(problem_id='native-mis'):
    """Return a path graph with a canonical two-vertex MIS optimum."""
    return {
        'schema': 'mis.v1',
        'problem_id': problem_id,
        'objective': {'kind': 'maximum-cardinality'},
        'vertices': [
            {'index': 0, 'name': 'a'},
            {'index': 1, 'name': 'b'},
            {'index': 2, 'name': 'c'},
            {'index': 3, 'name': 'd'},
        ],
        'edges': [[0, 1], [1, 2], [2, 3]],
        'fixed_values': [],
        'metadata': {},
    }


def _mis_case(payload=None, best_known=None):
    """Wrap an MIS graph as one vertex-index-set maximization task."""
    problem = _mis() if payload is None else payload
    return ProblemCase(
        problem_id=problem['problem_id'],
        artifacts=(
            ProblemArtifact(
                artifact_id='mis',
                representation='mis.v1',
                payload=problem,
            ),
        ),
        tasks=(
            TaskDefinition(
                task_id='maximum-independent-set',
                canonical_artifact_id='mis',
                sense='maximize',
                solution_representation='vertex-index-set.v1',
                task_type='maximum-independent-set',
                best_known=best_known,
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


class _FixedMisResultSolver:
    """Return one contract-valid MIS witness with a caller-chosen status."""

    def __init__(self, selected_vertices, status):
        self.selected_vertices = selected_vertices
        self.status = status

    def solve(self, problem, config=None):
        del config
        selected = (
            None
            if self.selected_vertices is None
            else list(self.selected_vertices)
        )
        semantics = (
            None
            if selected is None
            else evaluate_mis_solution(problem, selected)
        )
        return {
            'schema': 'mis-result.v1',
            'problem_id': problem['problem_id'],
            'solver': {
                'name': 'fixed-mis-result',
                'version': '1',
                'backend': 'test',
            },
            'status': self.status,
            'selected_vertices': selected,
            'objective_value': (
                None if semantics is None else semantics['objective_value']
            ),
            'cardinality': (
                None if semantics is None else semantics['cardinality']
            ),
            'total_weight': (
                None if semantics is None else semantics['total_weight']
            ),
            'feasible': (
                None if semantics is None else semantics['feasible']
            ),
            'runtime_seconds': 0,
            'metadata': {},
        }


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

    def test_local_search_cbqm_updates_incumbent_without_exact_promotion(self):
        record = solve_native_problem_task(
            _cbqm_case(),
            'select-one',
            'cbqm',
            LocalSearchCbqmSolver(),
            {'seed': 0, 'max_restarts': 0},
            update_best=True,
        )

        self.assertEqual('feasible', record.raw_result['status'])
        self.assertEqual([1, 0], record.canonical_solution)
        self.assertEqual(-2, record.canonical_objective_value)
        self.assertFalse(record.exact_for_task)
        self.assertFalse(record.update.current.exact)
        evidence = record.update.current.metadata['exactness']
        self.assertEqual('direct_cbqm', evidence['route'])
        self.assertFalse(evidence['solver_reported_optimal'])
        self.assertEqual(
            'solver_did_not_report_optimal',
            evidence['optimality_verification_reason'],
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

    def test_exact_mis_updates_only_after_independent_native_verification(self):
        record = solve_native_problem_task(
            _mis_case(),
            'maximum-independent-set',
            'mis',
            ExactMisSolver(),
            update_best=True,
        )

        self.assertEqual('mis-result.v1', record.raw_result['schema'])
        self.assertEqual([0, 2], record.canonical_solution)
        self.assertEqual(2, record.canonical_objective_value)
        self.assertTrue(record.exact_for_task)
        self.assertTrue(record.update.current.exact)
        evidence = record.update.current.metadata['exactness']
        self.assertEqual('direct_mis', evidence['route'])
        self.assertTrue(evidence['independent_mis_optimality_verified'])
        self.assertEqual(
            'exhaustive-mis-enumeration',
            evidence['optimality_verification_method'],
        )

    def test_forged_optimal_status_cannot_lock_a_worse_mis_candidate(self):
        record = solve_native_problem_task(
            _mis_case(),
            'maximum-independent-set',
            'mis',
            _FixedMisResultSolver([0], 'optimal'),
            update_best=True,
        )

        self.assertEqual([0], record.canonical_solution)
        self.assertFalse(record.exact_for_task)
        self.assertFalse(record.update.current.exact)
        self.assertEqual(
            'better_assignment_found',
            record.update.current.metadata['exactness'][
                'optimality_verification_reason'
            ],
        )

    def test_greedy_mis_updates_a_non_exact_best_known_solution(self):
        record = solve_native_problem_task(
            _mis_case(),
            'maximum-independent-set',
            'mis',
            GreedyMisSolver(),
            update_best=True,
        )

        self.assertEqual('feasible', record.raw_result['status'])
        self.assertEqual([0, 2], record.canonical_solution)
        self.assertFalse(record.exact_for_task)
        self.assertTrue(record.update.updated)
        self.assertFalse(record.update.current.exact)
        self.assertEqual(
            'solver_did_not_report_optimal',
            record.update.current.metadata['exactness'][
                'optimality_verification_reason'
            ],
        )

    def test_mis_verification_limit_keeps_exact_claim_non_persistent(self):
        record = solve_native_problem_task(
            _mis_case(),
            'maximum-independent-set',
            'mis',
            ExactMisSolver(),
            update_best=True,
            exact_verification_max_variables=3,
        )

        self.assertFalse(record.exact_for_task)
        self.assertEqual(
            'vertex_limit_exceeded',
            record.update.current.metadata['exactness'][
                'optimality_verification_reason'
            ],
        )

    def test_mis_case_validation_deep_checks_artifact_and_best_known(self):
        case = _mis_case(
            best_known=BestKnownSolution(
                solution=[0, 2],
                objective_value=2,
                exact=False,
                source='test',
            ),
        )

        report = validate_problem_case(case, strict=True)

        self.assertTrue(report.fully_checked)
        self.assertEqual(1, report.validated_artifact_count)
        self.assertEqual(1, report.validated_best_known_count)

    def test_invalid_mis_is_rejected_before_solver_side_effect(self):
        problem = _mis('invalid-native-mis')
        problem['metadata'] = 'not-an-object'
        solver = _SideEffectSolver()

        with self.assertRaisesRegex(TypeError, 'must be an object'):
            solve_native_problem_task(
                _mis_case(problem),
                'maximum-independent-set',
                'mis',
                solver,
            )
        self.assertEqual(0, solver.calls)

    def test_mis_canonical_task_must_use_maximize_sense(self):
        problem = _mis('mis-sense')

        with self.assertRaisesRegex(ValueError, 'disagrees'):
            ProblemCase(
                problem_id=problem['problem_id'],
                artifacts=(
                    ProblemArtifact(
                        artifact_id='mis',
                        representation='mis.v1',
                        payload=problem,
                    ),
                ),
                tasks=(
                    TaskDefinition(
                        task_id='wrong-sense',
                        canonical_artifact_id='mis',
                        sense='minimize',
                        solution_representation='vertex-index-set.v1',
                    ),
                ),
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
