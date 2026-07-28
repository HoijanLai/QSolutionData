"""End-to-end tests for the ProblemCase-to-QUBO solver bridge."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from lib.solvers.qubo.exact import ExactQuboSolver

from problem.case_operations import (
    compile_case_qubo,
    project_qubo_sample_to_cbqm,
)
from problem.problem_def import (
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
    TransformationRecord,
)
from problem.reader import load_problem_case
from problem.solving import solve_problem_task


def _qubo_payload(problem_id='runner-direct'):
    return {
        'schema': 'qubo.v1',
        'problem_id': problem_id,
        'sense': 'minimize',
        'num_variables': 2,
        'variable_names': ['alpha', 'beta'],
        'offset': 0.5,
        'terms': [[0, 0, -2.0], [0, 1, 1.0], [1, 1, -1.0]],
        'metadata': {},
    }


def _cbqm_payload(problem_id='runner-compiled'):
    return {
        'schema': 'cbqm.v1',
        'problem_id': problem_id,
        'variables': [
            {
                'index': 0,
                'name': 'alpha',
                'vartype': 'BINARY',
                'kind': 'decision',
                'metadata': {},
            },
            {
                'index': 1,
                'name': 'beta',
                'vartype': 'BINARY',
                'kind': 'decision',
                'metadata': {},
            },
        ],
        'objective': {
            'sense': 'minimize',
            'offset': 0.0,
            'linear': [[0, -2.0], [1, -1.0]],
            'quadratic': [],
        },
        'constraints': [
            {
                'name': 'choose-one',
                'family': 'selection',
                'linear': [[0, 1.0], [1, 1.0]],
                'lower_bound': 1.0,
                'upper_bound': 1.0,
                'metadata': {},
            },
        ],
        'fixed_values': [],
        'metadata': {},
    }


def _penalty_compiler(certified, penalty):
    """Build a compatible injected compiler with controllable proof evidence."""

    def compile_fixture(cbqm, config):
        qubo = {
            'schema': 'qubo.v1',
            'problem_id': cbqm['problem_id'],
            'sense': 'minimize',
            'num_variables': 2,
            'variable_names': ['alpha', 'beta'],
            'offset': float(penalty),
            'terms': [
                [0, 0, -2.0 - penalty],
                [0, 1, 2.0 * penalty],
                [1, 1, -1.0 - penalty],
            ],
            'metadata': {
                'compiler': 'test-penalty.v1',
                'compiler_config': dict(config),
            },
        }
        context = {
            'schema': 'qubo-compilation-context.v1',
            'source_problem_id': cbqm['problem_id'],
            'source_variable_count': 2,
            'source_variable_names': ['alpha', 'beta'],
            'objective_sense': 'minimize',
            'objective_multiplier': 1.0,
            'fixed_values': [],
            'free_variables': [
                {'cbqm_index': 0, 'qubo_index': 0, 'name': 'alpha'},
                {'cbqm_index': 1, 'qubo_index': 1, 'name': 'beta'},
            ],
            'slack_variables': [],
            'constraints': [],
            'compiler_config': dict(config),
            'qubo_variable_count': 2,
            'qubo_variable_names': ['alpha', 'beta'],
            'equivalence': {
                'schema': 'qubo-compilation-equivalence.v1',
                'exact_projection_certified': certified,
            },
        }
        return qubo, context

    return compile_fixture


def _direct_case():
    artifact = ProblemArtifact(
        artifact_id='qubo',
        representation='qubo.v1',
        payload=_qubo_payload(),
    )
    return ProblemCase(
        problem_id='runner-direct',
        artifacts=(artifact,),
        tasks=(
            TaskDefinition(
                task_id='minimize-qubo',
                canonical_artifact_id='qubo',
                sense='minimize',
            ),
        ),
    )


def _compiled_case(*, certified=False, artifact_id='qubo-p4', penalty=4.0):
    cbqm = ProblemArtifact(
        artifact_id='cbqm',
        representation='cbqm.v1',
        payload=_cbqm_payload(),
    )
    case = ProblemCase(
        problem_id='runner-compiled',
        artifacts=(cbqm,),
        tasks=(
            TaskDefinition(
                task_id='choose-one',
                canonical_artifact_id='cbqm',
                sense='minimize',
            ),
        ),
    )
    return compile_case_qubo(
        case,
        'cbqm',
        {'default_penalty': penalty},
        target_artifact_id=artifact_id,
        compiler=_penalty_compiler(certified, penalty),
    )


def _canonically_compiled_case(cbqm=None, *, config=None):
    """Compile through the production compiler so exact evidence is reproducible."""
    source_payload = _cbqm_payload() if cbqm is None else cbqm
    source = ProblemArtifact(
        artifact_id='cbqm',
        representation='cbqm.v1',
        payload=source_payload,
    )
    case = ProblemCase(
        problem_id=source_payload['problem_id'],
        artifacts=(source,),
        tasks=(
            TaskDefinition(
                task_id='canonical-task',
                canonical_artifact_id='cbqm',
                sense=source_payload['objective']['sense'],
            ),
        ),
    )
    return compile_case_qubo(
        case,
        'cbqm',
        {'default_penalty': 4.0} if config is None else config,
        target_artifact_id='qubo-canonical',
    )


class _FixedResultSolver:
    """Small QAOA-like stub returning one sampled feasible candidate."""

    def __init__(self, sample, status='feasible', energy_delta=0.0):
        self.sample = sample
        self.status = status
        self.energy_delta = energy_delta
        self.seen_problem = None

    def solve(self, problem, config=None):
        del config
        self.seen_problem = copy.deepcopy(problem)
        if self.sample is None:
            energy = None
        else:
            energy = float(problem['offset']) + sum(
                coefficient * self.sample[left] * self.sample[right]
                for left, right, coefficient in problem['terms']
            )
            energy += self.energy_delta
        return {
            'schema': 'qubo-result.v1',
            'problem_id': problem['problem_id'],
            'solver': {
                'name': 'qaoa-like-fixture',
                'version': '1',
                'backend': 'test',
            },
            'status': self.status,
            'best_sample': (
                None if self.sample is None else list(self.sample)
            ),
            'best_energy': energy,
            'runtime_seconds': 0.001,
            'metadata': {},
        }


class _SideEffectSolver:
    """Record whether malformed input reached the solver boundary."""

    def __init__(self):
        self.calls = 0

    def solve(self, problem, config=None):
        del problem, config
        self.calls += 1
        raise AssertionError('Malformed input must be rejected before solve().')


class ProblemSolvingTests(unittest.TestCase):
    def test_persisted_compiled_example_runs_end_to_end(self):
        case_path = (
            Path(__file__).resolve().parents[1]
            / 'problem'
            / 'data'
            / 'problem2'
        )
        case = load_problem_case(case_path)

        record = solve_problem_task(
            case,
            'select-one',
            'qubo-penalty-4',
            ExactQuboSolver(),
        )

        self.assertEqual([1, 0], record.canonical_solution)
        self.assertEqual(-2.0, record.canonical_objective_value)
        self.assertTrue(record.exact_for_task)

    def test_direct_qubo_exact_result_updates_with_artifact_provenance(self):
        case = _direct_case()

        record = solve_problem_task(
            case,
            'minimize-qubo',
            'qubo',
            ExactQuboSolver(),
            update_best=True,
        )

        self.assertEqual('optimal', record.raw_result['status'])
        self.assertEqual(-1.5, record.canonical_objective_value)
        self.assertTrue(record.exact_for_task)
        self.assertTrue(record.update.updated)
        best = record.case.get_task('minimize-qubo').best_known
        self.assertTrue(best.exact)
        self.assertEqual('qubo', best.metadata['artifact_id'])
        self.assertEqual(record.payload_sha256, best.metadata['payload_sha256'])
        self.assertEqual(
            record.payload_sha256,
            best.metadata['canonical_payload_sha256'],
        )

    def test_forged_direct_optimal_status_is_not_promoted_to_exact(self):
        case = _direct_case()

        record = solve_problem_task(
            case,
            'minimize-qubo',
            'qubo',
            _FixedResultSolver([0, 0], status='optimal'),
            update_best=True,
        )

        self.assertEqual('optimal', record.raw_result['status'])
        self.assertFalse(record.exact_for_task)
        best = record.case.get_task('minimize-qubo').best_known
        self.assertFalse(best.exact)
        evidence = best.metadata['exactness']
        self.assertTrue(evidence['solver_reported_optimal'])
        self.assertFalse(
            evidence['independent_qubo_optimality_verified']
        )
        self.assertEqual(
            'better_assignment_found',
            evidence['optimality_verification_reason'],
        )

    def test_exact_verification_limit_keeps_large_claim_non_exact(self):
        case = _direct_case()

        record = solve_problem_task(
            case,
            'minimize-qubo',
            'qubo',
            ExactQuboSolver(),
            update_best=True,
            exact_verification_max_variables=1,
        )

        self.assertEqual('optimal', record.raw_result['status'])
        self.assertFalse(record.exact_for_task)
        self.assertEqual(
            'variable_limit_exceeded',
            record.update.current.metadata['exactness'][
                'optimality_verification_reason'
            ],
        )

    def test_exact_verification_limit_rejects_boolean(self):
        with self.assertRaisesRegex(
            ValueError,
            'exact_verification_max_variables',
        ):
            solve_problem_task(
                _direct_case(),
                'minimize-qubo',
                'qubo',
                ExactQuboSolver(),
                exact_verification_max_variables=True,
            )

    def test_record_snapshots_config_and_rejects_nested_mutation(self):
        case = _direct_case()
        config = {'experiment': {'seed': 7, 'labels': ['baseline']}}

        record = solve_problem_task(
            case,
            'minimize-qubo',
            'qubo',
            _FixedResultSolver([1, 0], status='optimal'),
            config=config,
            update_best=True,
        )

        self.assertEqual(config, record.solver_config)
        self.assertEqual(
            config,
            record.case.get_task(
                'minimize-qubo'
            ).best_known.metadata['solver_config'],
        )
        with self.assertRaisesRegex(TypeError, 'immutable'):
            record.raw_result['status'] = 'unknown'
        with self.assertRaisesRegex(TypeError, 'immutable'):
            record.raw_result['best_sample'][0] = 0
        with self.assertRaisesRegex(TypeError, 'immutable'):
            record.solver_config['experiment']['labels'].append('changed')

    def test_invalid_qubo_is_rejected_before_solver_side_effect(self):
        malformed = _qubo_payload('malformed-runner')
        malformed['metadata'] = 'not-an-object'
        artifact = ProblemArtifact(
            artifact_id='qubo',
            representation='qubo.v1',
            payload=malformed,
        )
        case = ProblemCase(
            problem_id='malformed-runner',
            artifacts=(artifact,),
            tasks=(
                TaskDefinition(
                    task_id='task',
                    canonical_artifact_id='qubo',
                    sense='minimize',
                ),
            ),
        )
        solver = _SideEffectSolver()

        with self.assertRaisesRegex(TypeError, 'must be an object'):
            solve_problem_task(case, 'task', 'qubo', solver)
        self.assertEqual(0, solver.calls)

    def test_qaoa_like_compiled_candidate_is_projected_and_not_exact(self):
        case = _compiled_case(certified=True)
        solver = _FixedResultSolver([0, 1], status='feasible')

        record = solve_problem_task(
            case,
            'choose-one',
            'qubo-p4',
            solver,
            update_best=True,
        )

        self.assertEqual([0, 1], record.canonical_solution)
        self.assertEqual(-1.0, record.canonical_objective_value)
        self.assertFalse(record.exact_for_task)
        self.assertFalse(
            record.case.get_task('choose-one').best_known.exact
        )

    def test_compiled_exact_requires_reproducible_equivalence_certificate(self):
        uncertified = _compiled_case(certified=False)
        merely_declared = _compiled_case(certified=True)
        certified = _canonically_compiled_case()

        weak_record = solve_problem_task(
            uncertified,
            'choose-one',
            'qubo-p4',
            ExactQuboSolver(),
        )
        declared_record = solve_problem_task(
            merely_declared,
            'choose-one',
            'qubo-p4',
            ExactQuboSolver(),
        )
        strong_record = solve_problem_task(
            certified,
            'canonical-task',
            'qubo-canonical',
            ExactQuboSolver(),
        )

        self.assertEqual([1, 0], weak_record.canonical_solution)
        self.assertEqual(-2.0, weak_record.canonical_objective_value)
        self.assertFalse(weak_record.exact_for_task)
        self.assertFalse(declared_record.exact_for_task)
        self.assertTrue(strong_record.exact_for_task)

    def test_forged_compiled_optimal_status_is_not_promoted_to_exact(self):
        case = _canonically_compiled_case()

        # [0, 1] satisfies the source equality and has zero penalty, but its
        # objective (-1) is worse than [1, 0] (-2). All compiler gates pass;
        # the independent global QUBO check must be the deciding gate.
        record = solve_problem_task(
            case,
            'canonical-task',
            'qubo-canonical',
            _FixedResultSolver([0, 1], status='optimal'),
            update_best=True,
        )

        self.assertEqual([0, 1], record.canonical_solution)
        self.assertFalse(record.exact_for_task)
        evidence = record.update.current.metadata['exactness']
        self.assertTrue(evidence['zero_penalty_verified'])
        self.assertTrue(evidence['objective_energy_identity_verified'])
        self.assertFalse(
            evidence['independent_qubo_optimality_verified']
        )

    def test_tiny_real_constraint_violation_is_not_promoted_or_updated(self):
        cbqm = {
            'schema': 'cbqm.v1',
            'problem_id': 'tiny-feasibility',
            'variables': [
                {'index': 0, 'name': 'x', 'vartype': 'BINARY'},
            ],
            'objective': {
                'sense': 'minimize',
                'offset': 0.0,
                'linear': [[0, 1.0]],
                'quadratic': [],
            },
            'constraints': [
                {
                    'name': 'must-select',
                    'family': 'selection',
                    'linear': [[0, 1e-12]],
                    'lower_bound': 1e-12,
                },
            ],
            'fixed_values': [],
            'metadata': {},
        }
        case = _canonically_compiled_case(
            cbqm,
            config={
                'default_penalty': 1e-20,
                'constraint_precision': 12,
                'zero_tolerance': 0.0,
            },
        )

        with self.assertRaisesRegex(ValueError, 'lower bound'):
            solve_problem_task(
                case,
                'canonical-task',
                'qubo-canonical',
                ExactQuboSolver(),
                update_best=True,
            )
        self.assertIsNone(
            case.get_task('canonical-task').best_known
        )

    def test_exact_promotion_requires_zero_penalty_slack_assignment(self):
        cbqm = {
            'schema': 'cbqm.v1',
            'problem_id': 'slack-proof',
            'variables': [
                {'index': 0, 'name': 'x', 'vartype': 'BINARY'},
                {'index': 1, 'name': 'y', 'vartype': 'BINARY'},
            ],
            'objective': {
                'sense': 'minimize',
                'offset': 0.0,
                'linear': [],
                'quadratic': [],
            },
            'constraints': [
                {
                    'name': 'at-most-one',
                    'family': 'selection',
                    'linear': [[0, 1.0], [1, 1.0]],
                    'upper_bound': 1.0,
                },
            ],
            'fixed_values': [],
            'metadata': {},
        }
        case = _canonically_compiled_case(
            cbqm,
            config={
                'default_penalty': 1e-20,
                'zero_tolerance': 0.0,
            },
        )

        # [x=0, y=0] is source-feasible, but slack=0 leaves the compiler's
        # equation x + y + slack = 1 with a non-zero residual.  The tiny
        # resulting energy is deliberately below the numeric identity
        # tolerance, so the structural zero-penalty check is the deciding gate.
        record = solve_problem_task(
            case,
            'canonical-task',
            'qubo-canonical',
            _FixedResultSolver([0, 0, 0], status='optimal'),
        )

        self.assertEqual([0, 0], record.canonical_solution)
        self.assertEqual(0.0, record.canonical_objective_value)
        self.assertFalse(record.exact_for_task)

    def test_canonical_maximization_uses_negative_objective_multiplier(self):
        cbqm = _cbqm_payload('runner-maximize')
        cbqm['objective'] = {
            'sense': 'maximize',
            'offset': 0.0,
            'linear': [[0, 2.0], [1, 1.0]],
            'quadratic': [],
        }
        case = _canonically_compiled_case(cbqm)

        record = solve_problem_task(
            case,
            'canonical-task',
            'qubo-canonical',
            ExactQuboSolver(),
            update_best=True,
        )

        self.assertEqual([1, 0], record.canonical_solution)
        self.assertEqual(2.0, record.canonical_objective_value)
        self.assertEqual(-2.0, record.raw_result['best_energy'])
        self.assertTrue(record.exact_for_task)
        self.assertTrue(
            record.case.get_task('canonical-task').best_known.exact
        )

    def test_invalid_solver_result_is_rejected_before_canonical_update(self):
        case = _direct_case()
        solver = _FixedResultSolver(
            [1, 0],
            status='feasible',
            energy_delta=3.0,
        )

        with self.assertRaisesRegex(ValueError, 'best_energy'):
            solve_problem_task(
                case,
                'minimize-qubo',
                'qubo',
                solver,
                update_best=True,
            )
        self.assertIsNone(
            case.get_task('minimize-qubo').best_known
        )

    def test_multiple_qubos_use_the_explicit_artifact_only(self):
        case = _compiled_case(
            certified=False,
            artifact_id='qubo-p4',
            penalty=4.0,
        )
        case = compile_case_qubo(
            case,
            'cbqm',
            {'default_penalty': 9.0},
            target_artifact_id='qubo-p9',
            compiler=_penalty_compiler(False, 9.0),
        )
        solver = _FixedResultSolver([1, 0])

        record = solve_problem_task(
            case,
            'choose-one',
            'qubo-p9',
            solver,
        )

        self.assertEqual('qubo-p9', record.artifact_id)
        self.assertEqual(9.0, solver.seen_problem['offset'])
        self.assertEqual(
            record.payload_sha256,
            _payload_hash(case.get_artifact('qubo-p9').payload),
        )

    def test_no_solver_candidate_does_not_request_best_known_update(self):
        case = _direct_case()
        solver = _FixedResultSolver(None, status='infeasible')

        record = solve_problem_task(
            case,
            'minimize-qubo',
            'qubo',
            solver,
            update_best=True,
        )

        self.assertIsNone(record.canonical_solution)
        self.assertIsNone(record.canonical_objective_value)
        self.assertIsNone(record.update)
        self.assertIs(record.case, case)

    def test_forged_index_context_is_rejected_before_projection(self):
        case = _compiled_case(certified=True)
        artifact = case.get_artifact('qubo-p4')
        outer = copy.deepcopy(dict(artifact.transformation.context))
        compilation = copy.deepcopy(dict(outer['compilation']))
        compilation['free_variables'][0]['name'] = 'beta'
        outer['compilation'] = compilation
        forged = ProblemArtifact(
            artifact_id=artifact.artifact_id,
            representation=artifact.representation,
            payload=artifact.payload,
            parent_artifact_id=artifact.parent_artifact_id,
            transformation=TransformationRecord(
                name='compile_qubo',
                version=artifact.transformation.version,
                lossless=False,
                context=outer,
            ),
        )
        forged_case = case.with_artifact(
            forged,
            replace_existing=True,
        )

        with self.assertRaisesRegex(ValueError, 'free-variable name'):
            project_qubo_sample_to_cbqm(
                forged_case,
                'qubo-p4',
                [1, 0],
            )

    def test_compilation_context_rejects_boolean_numeric_identity(self):
        case = _compiled_case(certified=True)
        artifact = case.get_artifact('qubo-p4')
        outer = copy.deepcopy(dict(artifact.transformation.context))
        compilation = copy.deepcopy(dict(outer['compilation']))
        # ``True == 1.0`` in Python, but a boolean is not an objective scale.
        compilation['objective_multiplier'] = True
        outer['compilation'] = compilation
        forged = ProblemArtifact(
            artifact_id=artifact.artifact_id,
            representation=artifact.representation,
            payload=artifact.payload,
            parent_artifact_id=artifact.parent_artifact_id,
            transformation=TransformationRecord(
                name='compile_qubo',
                version=artifact.transformation.version,
                lossless=False,
                context=outer,
            ),
        )
        forged_case = case.with_artifact(forged, replace_existing=True)

        with self.assertRaisesRegex(ValueError, 'objective multiplier'):
            project_qubo_sample_to_cbqm(
                forged_case,
                'qubo-p4',
                [1, 0],
            )

    def test_projection_rejects_float_binary_lookalikes(self):
        case = _compiled_case(certified=True)

        with self.assertRaisesRegex(ValueError, 'integer 0 or 1'):
            project_qubo_sample_to_cbqm(
                case,
                'qubo-p4',
                [1.0, 0],
            )


def _payload_hash(payload):
    """Keep the assertion independent from solving.py's protected helper."""
    import hashlib
    import json

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


if __name__ == '__main__':
    unittest.main()
