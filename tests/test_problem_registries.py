"""Extension-seam tests for problem representation component registries."""

import unittest

from problem import (
    BestKnownSolution,
    NativeSolverRunner,
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
    evaluate_task_solution,
    register_representation_validator,
    register_task_evaluator,
    registered_problem_components,
    validate_problem_case,
)


class ProblemRegistryTests(unittest.TestCase):
    def test_registered_validator_and_evaluator_remove_custom_warnings(self):
        representation = 'registry-scored-model.v1'
        solution_representation = 'registry-score.v1'
        calls = []

        def validate_scored_model(problem_case, artifact):
            calls.append((problem_case.problem_id, artifact.artifact_id))
            if type(artifact.payload.get('multiplier')) is not int:
                raise TypeError('multiplier must be an integer.')

        def evaluate_score(artifact, solution, task):
            del task
            return artifact.payload['multiplier'] * solution['value']

        register_representation_validator(
            representation,
            validate_scored_model,
        )
        register_task_evaluator(
            representation,
            solution_representation,
            evaluate_score,
        )

        case = ProblemCase(
            problem_id='registry-scored-case',
            artifacts=(
                ProblemArtifact(
                    artifact_id='model',
                    representation=representation,
                    payload={'multiplier': 3},
                ),
            ),
            tasks=(
                TaskDefinition(
                    task_id='score',
                    canonical_artifact_id='model',
                    sense='maximize',
                    solution_representation=solution_representation,
                    best_known=BestKnownSolution(
                        solution={'value': 4},
                        objective_value=12,
                    ),
                ),
            ),
        )

        self.assertEqual(
            12,
            evaluate_task_solution(case, 'score', {'value': 4}),
        )
        report = validate_problem_case(case, strict=True)
        self.assertTrue(report.fully_checked)
        self.assertEqual(1, report.validated_artifact_count)
        self.assertEqual(1, report.validated_best_known_count)
        self.assertEqual(
            [('registry-scored-case', 'model')],
            calls,
        )

    def test_duplicate_registration_requires_explicit_replacement(self):
        representation = 'registry-duplicate-test.v1'

        def first(problem_case, artifact):
            del problem_case, artifact

        def second(problem_case, artifact):
            del problem_case, artifact

        register_representation_validator(representation, first)
        with self.assertRaisesRegex(ValueError, 'already registered'):
            register_representation_validator(representation, second)

        register_representation_validator(
            representation,
            second,
            replace=True,
        )
        self.assertIs(
            second,
            registered_problem_components()[
                'representation_validators'
            ][representation],
        )

    def test_registry_snapshots_are_read_only_and_include_builtins(self):
        components = registered_problem_components()

        self.assertIn('cbqm.v1', components['representation_validators'])
        self.assertIn(
            ('qubo.v1', 'binary-vector.v1'),
            components['task_evaluators'],
        )
        self.assertIn(
            ('mis.v1', 'vertex-index-set.v1'),
            components['task_evaluators'],
        )
        self.assertIn('cbqm.v1', components['native_solver_runners'])
        self.assertIn('mis.v1', components['native_solver_runners'])
        with self.assertRaises(TypeError):
            components['task_evaluators']['new'] = lambda: None

    def test_native_runner_validates_all_hook_shapes(self):
        with self.assertRaisesRegex(TypeError, 'run_and_validate'):
            NativeSolverRunner(
                run_and_validate=None,
                candidate_from_result=lambda result: result,
                verify_exact=lambda *args: (False, {}),
                route_kind='test',
            )


if __name__ == '__main__':
    unittest.main()
