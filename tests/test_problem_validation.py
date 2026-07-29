"""Tests for repository-wide problem collection integrity checks."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from problem.__main__ import main
from problem.graph_codec import graph_to_node_link
from problem.problem_def import (
    BestKnownSolution,
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
    TransformationRecord,
)
from problem.reader import (
    case_from_dict,
    load_problem_case,
    save_problem_case,
)
from problem.transforms import (
    cbqm_to_factor_graph,
    qubo_to_interaction_graph,
    qubo_to_maxcut_graph,
)
from problem.validation import (
    ProblemValidationError,
    validate_problem_case,
    validate_problem_path,
)


def _qubo(problem_id='validation-case', *, linear_bias=-2.0):
    """Return a one-variable canonical QUBO suitable for tiny audit cases."""
    return {
        'schema': 'qubo.v1',
        'problem_id': problem_id,
        'sense': 'minimize',
        'num_variables': 1,
        'variable_names': ['x'],
        'offset': 1.0,
        'terms': [[0, 0, linear_bias]],
        'metadata': {},
    }


def _qubo_case(
    problem_id='validation-case',
    *,
    payload=None,
    best_known=None,
):
    """Build one direct-QUBO case with an optional persisted incumbent."""
    artifact = ProblemArtifact(
        artifact_id='qubo',
        representation='qubo.v1',
        payload=_qubo(problem_id) if payload is None else payload,
    )
    tasks = ()
    if best_known is not None:
        tasks = (
            TaskDefinition(
                task_id='energy',
                canonical_artifact_id='qubo',
                sense='minimize',
                best_known=best_known,
            ),
        )
    return ProblemCase(
        problem_id=problem_id,
        artifacts=(artifact,),
        tasks=tasks,
        primary_artifact_id='qubo',
    )


def _minimal_manifest(problem_id, **extra):
    """Return a task-free manifest that needs no artifact files."""
    manifest = {
        'schema': 'problem-case.v1',
        'problem_id': problem_id,
        'artifacts': [],
    }
    manifest.update(extra)
    return manifest


def _cbqm(problem_id='validation-case'):
    """Return the smallest nontrivial canonical binary CBQM."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': problem_id,
        'variables': [{
            'index': 0,
            'name': 'x',
            'vartype': 'BINARY',
            'kind': 'decision',
            'metadata': {},
        }],
        'objective': {
            'sense': 'minimize',
            'offset': 0,
            'linear': [[0, -1]],
            'quadratic': [],
        },
        'constraints': [],
        'fixed_values': [],
        'metadata': {},
    }


class ProblemValidationApiTests(unittest.TestCase):
    def test_repository_examples_are_fully_checked(self):
        project_root = Path(__file__).resolve().parents[1]

        report = validate_problem_path(project_root / 'problem' / 'data')

        self.assertTrue(report.fully_checked)
        self.assertEqual(2, report.case_count)
        self.assertEqual(3, report.artifact_count)
        self.assertEqual(3, report.validated_artifact_count)
        self.assertEqual(1, report.best_known_count)
        self.assertEqual(1, report.validated_best_known_count)

    def test_deep_qubo_contract_failure_has_case_and_artifact_context(self):
        invalid_qubo = _qubo()
        invalid_qubo['terms'] = [[0, 0, 0.0]]
        problem_case = _qubo_case(payload=invalid_qubo)

        with self.assertRaisesRegex(
            ProblemValidationError,
            r"validation-case.*'qubo'.*zero",
        ):
            validate_problem_case(problem_case)

    def test_rechecks_best_known_objective(self):
        problem_case = _qubo_case(
            best_known=BestKnownSolution(
                solution=[1],
                objective_value=99,
            ),
        )

        with self.assertRaisesRegex(
            ProblemValidationError,
            r"task 'energy'.*objective_value",
        ):
            validate_problem_case(problem_case)

    def test_custom_representation_warns_and_strict_mode_rejects_it(self):
        problem_case = ProblemCase(
            problem_id='custom-case',
            artifacts=(
                ProblemArtifact(
                    artifact_id='custom',
                    representation='custom.v1',
                    payload={'domain': 'private'},
                ),
            ),
        )

        report = validate_problem_case(problem_case)

        self.assertFalse(report.fully_checked)
        self.assertEqual(0, report.validated_artifact_count)
        self.assertIn('unchecked representation', report.warnings[0])
        with self.assertRaisesRegex(
            ProblemValidationError,
            'Strict problem validation',
        ):
            validate_problem_case(problem_case, strict=True)

    def test_unregistered_networkx_representation_is_structurally_checked_only(
        self,
    ):
        graph_artifact = ProblemArtifact(
            artifact_id='domain-graph',
            representation='networkx.private-domain.v1',
            payload={
                'directed': False,
                'multigraph': False,
                'graph': {},
                'nodes': [{'id': 'x'}],
                'edges': [],
            },
        )
        problem_case = ProblemCase(
            problem_id='private-networkx',
            artifacts=(graph_artifact,),
        )

        report = validate_problem_case(problem_case)

        self.assertEqual(0, report.validated_artifact_count)
        self.assertIn(
            'networkx.private-domain.v1',
            report.warnings[0],
        )

    def test_maxcut_checker_is_type_sensitive_and_tied_to_parent(self):
        source_qubo = _qubo('maxcut-case')
        parent = ProblemArtifact(
            artifact_id='qubo',
            representation='qubo.v1',
            payload=source_qubo,
        )

        correct_graph = qubo_to_maxcut_graph(source_qubo)
        correct_artifact = self._maxcut_artifact(correct_graph)
        correct_case = ProblemCase(
            problem_id='maxcut-case',
            artifacts=(parent, correct_artifact),
        )
        self.assertTrue(validate_problem_case(correct_case).fully_checked)

        boolean_weight = graph_to_node_link(correct_graph)
        boolean_weight['edges'][0]['weight'] = True
        corrupted_artifact = ProblemArtifact(
            artifact_id='maxcut',
            representation='networkx.qubo-maxcut.v1',
            payload=boolean_weight,
            parent_artifact_id='qubo',
            transformation=correct_artifact.transformation,
        )
        corrupted_case = ProblemCase(
            problem_id='maxcut-case',
            artifacts=(parent, corrupted_artifact),
        )
        with self.assertRaisesRegex(
            ProblemValidationError,
            'canonical graph rebuilt',
        ):
            validate_problem_case(corrupted_case)

        different_graph = qubo_to_maxcut_graph(
            _qubo('maxcut-case', linear_bias=-3.0)
        )
        mismatched_case = ProblemCase(
            problem_id='maxcut-case',
            artifacts=(parent, self._maxcut_artifact(different_graph)),
        )
        with self.assertRaisesRegex(
            ProblemValidationError,
            'do not reproduce parent',
        ):
            validate_problem_case(mismatched_case)

    def test_root_model_graphs_must_belong_to_their_case(self):
        foreign_qubo = _qubo('foreign')
        graph_cases = (
            (
                'networkx.qubo-interaction.v1',
                qubo_to_interaction_graph(foreign_qubo),
            ),
            (
                'networkx.qubo-maxcut.v1',
                qubo_to_maxcut_graph(foreign_qubo),
            ),
            (
                'networkx.cbqm-factor.v1',
                cbqm_to_factor_graph(_cbqm('foreign')),
            ),
        )

        for representation, graph in graph_cases:
            artifact = ProblemArtifact(
                artifact_id='graph',
                representation=representation,
                payload=graph_to_node_link(graph),
            )
            problem_case = ProblemCase(
                problem_id='local',
                artifacts=(artifact,),
            )

            with self.subTest(representation=representation):
                with self.assertRaisesRegex(
                    ProblemValidationError,
                    "does not match case problem_id 'local'",
                ):
                    validate_problem_case(problem_case)

    @staticmethod
    def _maxcut_artifact(graph):
        return ProblemArtifact(
            artifact_id='maxcut',
            representation='networkx.qubo-maxcut.v1',
            payload=graph_to_node_link(graph),
            parent_artifact_id='qubo',
            transformation=TransformationRecord(
                name='qubo_to_maxcut_graph',
                version='1',
                lossless=True,
            ),
        )


class ProblemValidationPathTests(unittest.TestCase):
    def test_single_case_file_and_directory_are_both_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            case_directory = Path(directory) / 'single case'
            manifest_path = save_problem_case(
                _qubo_case('single-path-case'),
                case_directory,
            )

            file_report = validate_problem_path(manifest_path)
            directory_report = validate_problem_path(case_directory)

        self.assertEqual(file_report, directory_report)
        self.assertEqual(1, file_report.case_count)

    def test_empty_and_ambiguous_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                ProblemValidationError,
                'No.*case.json',
            ):
                validate_problem_path(root)

            (root / 'case.json').write_text('{}', encoding='utf-8')
            child = root / 'child'
            child.mkdir()
            (child / 'case.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(
                ProblemValidationError,
                'Ambiguous problem path',
            ):
                validate_problem_path(root)

    def test_collection_aggregates_one_load_error_per_bad_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('10-first', '20-second'):
                case_directory = root / name
                case_directory.mkdir()
                manifest = _minimal_manifest(
                    name,
                    misspelled_metadata={},
                )
                (case_directory / 'case.json').write_text(
                    json.dumps(manifest),
                    encoding='utf-8',
                )

            with self.assertRaises(ProblemValidationError) as raised:
                validate_problem_path(root)

        message = str(raised.exception)
        self.assertIn('10-first', message)
        self.assertIn('20-second', message)

    def test_collection_rejects_duplicate_problem_ids_with_both_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_problem_case(
                ProblemCase(problem_id='duplicate'),
                root / 'left',
            )
            save_problem_case(
                ProblemCase(problem_id='duplicate'),
                root / 'right',
            )

            with self.assertRaisesRegex(
                ProblemValidationError,
                r"Duplicate problem_id 'duplicate'.*left.*right",
            ):
                validate_problem_path(root)


class StrictProblemJsonTests(unittest.TestCase):
    def test_reader_rejects_duplicate_keys_and_non_finite_constants(self):
        invalid_documents = {
            'duplicate': (
                '{"schema":"problem-case.v1",'
                '"problem_id":"first","problem_id":"second",'
                '"artifacts":[]}'
            ),
            'nan': (
                '{"schema":"problem-case.v1",'
                '"problem_id":"nan-case","artifacts":[],'
                '"ignored_before":NaN}'
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, document in invalid_documents.items():
                case_path = root / f'{label}.json'
                case_path.write_text(document, encoding='utf-8')
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, 'Invalid JSON'):
                        load_problem_case(case_path)

    def test_reader_closes_every_manifest_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'artifact.json').write_text('{}', encoding='utf-8')
            base_artifact = {
                'artifact_id': 'custom',
                'representation': 'custom.v1',
                'path': 'artifact.json',
            }
            base_task = {
                'task_id': 'task',
                'canonical_artifact_id': 'custom',
                'sense': 'minimize',
            }
            cases = {
                'case': _minimal_manifest('closed', typo=True),
                'artifact descriptor': {
                    **_minimal_manifest('closed'),
                    'artifacts': [{**base_artifact, 'typo': True}],
                },
                'transformation': {
                    **_minimal_manifest('closed'),
                    'artifacts': [{
                        **base_artifact,
                        'parent_artifact_id': 'parent',
                        'transformation': {
                            'name': 'fixture',
                            'version': '1',
                            'lossless': False,
                            'typo': True,
                        },
                    }],
                },
                'task': {
                    **_minimal_manifest('closed'),
                    'artifacts': [base_artifact],
                    'tasks': [{**base_task, 'typo': True}],
                },
                'best_known': {
                    **_minimal_manifest('closed'),
                    'artifacts': [base_artifact],
                    'tasks': [{
                        **base_task,
                        'best_known': {
                            'solution': {'nodes': []},
                            'objective_value': 0,
                            'typo': True,
                        },
                    }],
                },
            }

            for label, manifest in cases.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        ValueError,
                        f'{label} contains unknown fields',
                    ):
                        case_from_dict(manifest, base_directory=root)

    def test_direct_manifest_rejects_non_string_field_names(self):
        manifest = _minimal_manifest('non-string-field')
        manifest[1] = 'not JSON'

        with self.assertRaisesRegex(
            TypeError,
            'case field names must be strings',
        ):
            case_from_dict(manifest, base_directory=Path('.'))

    def test_best_known_cannot_define_solution_and_legacy_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'artifact.json').write_text('{}', encoding='utf-8')
            manifest = {
                **_minimal_manifest('ambiguous-witness'),
                'artifacts': [{
                    'artifact_id': 'custom',
                    'representation': 'custom.v1',
                    'path': 'artifact.json',
                }],
                'tasks': [{
                    'task_id': 'task',
                    'canonical_artifact_id': 'custom',
                    'sense': 'minimize',
                    'best_known': {
                        'solution': [],
                        'sample': [],
                        'objective_value': 0,
                    },
                }],
            }

            with self.assertRaisesRegex(
                ValueError,
                "must not define both 'solution'.*'sample'",
            ):
                case_from_dict(manifest, base_directory=root)


class ProblemValidationCliTests(unittest.TestCase):
    def test_json_cli_success_and_data_failure_exit_codes(self):
        project_root = Path(__file__).resolve().parents[1]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            success_code = main([
                'validate',
                str(project_root / 'problem' / 'data'),
                '--json',
            ])

        success = json.loads(stdout.getvalue())
        self.assertEqual(0, success_code)
        self.assertTrue(success['ok'])
        self.assertEqual(2, success['case_count'])

        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                failure_code = main([
                    'validate',
                    directory,
                    '--json',
                ])

        failure = json.loads(stderr.getvalue())
        self.assertEqual(1, failure_code)
        self.assertFalse(failure['ok'])
        self.assertEqual('ProblemValidationError', failure['exception_type'])


if __name__ == '__main__':
    unittest.main()
