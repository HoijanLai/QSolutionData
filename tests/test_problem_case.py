"""Tests for the representation-independent problem-case envelope.

These tests deliberately exercise the public workflow rather than internal
serialization helpers.  A case is allowed to start as a plain graph, acquire
several mathematical representations, and host multiple independent tasks.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx

from problem.case_operations import (
    compile_case_qubo,
    create_graph_case,
    derive_qubo_maxcut_graph,
    project_qubo_sample_to_cbqm,
)
from problem.problem_def import (
    BestKnownSolution,
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
    TransformationRecord,
)
from problem.reader import (
    load_problem_case,
    load_problem_set,
    save_problem_case,
)
from problem.updater import update_best_known


def _cbqm_payload(problem_id='integration-case'):
    """Return a compact CBQM with constraints, metadata and a fixed variable."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': problem_id,
        'variables': [
            {
                'index': 0,
                'name': 'budget',
                'vartype': 'BINARY',
                'kind': 'decision',
                'metadata': {'label': 'Budget variable'},
            },
            {
                'index': 1,
                'name': 'asset_beta',
                'vartype': 'BINARY',
                'kind': 'decision',
                'metadata': {'label': 'Beta'},
            },
            {
                'index': 2,
                'name': 'unused_free',
                'vartype': 'BINARY',
                'kind': 'auxiliary',
                'metadata': {},
            },
            {
                'index': 3,
                'name': 'fixed_unused',
                'vartype': 'BINARY',
                'kind': 'decision',
                'metadata': {'reason': 'external policy'},
            },
        ],
        'objective': {
            'sense': 'minimize',
            'offset': 0.25,
            'linear': [[0, -2.0], [1, 1.0]],
            'quadratic': [[0, 1, 3.0]],
        },
        'constraints': [
            {
                'name': 'budget',
                'family': 'allocation',
                'linear': [[0, 1.0], [1, 2.0]],
                'lower_bound': 1.0,
                'upper_bound': 2.0,
                'metadata': {'external_id': 'left'},
            },
            {
                'name': 'choose-one',
                'family': 'cardinality',
                'linear': [[0, 1.0], [1, 1.0]],
                'lower_bound': 1.0,
                'upper_bound': 1.0,
                'metadata': {'external_id': 'right'},
            },
            {
                'name': 'constant-check',
                'family': 'constant',
                'linear': [],
                'lower_bound': 0.0,
                'upper_bound': 0.0,
                'metadata': {'keep_even_without_edges': True},
            },
        ],
        'fixed_values': [{'index': 3, 'value': 1}],
        'metadata': {'business_name': 'integration fixture'},
    }


def _qubo_payload(problem_id='integration-case'):
    """Return a QUBO whose final variable is intentionally isolated."""
    return {
        'schema': 'qubo.v1',
        'problem_id': problem_id,
        'sense': 'minimize',
        'num_variables': 4,
        'variable_names': [
            'budget',
            'asset_beta',
            'unused_free',
            'isolated',
        ],
        'offset': 1.25,
        'terms': [
            [0, 0, -2.0],
            [0, 1, 3.5],
            [1, 1, 1.0],
            [2, 2, -4.0],
        ],
        'metadata': {'compiler': 'fixture-v1'},
    }


def _root_artifact(artifact_id, representation, payload):
    return ProblemArtifact(
        artifact_id=artifact_id,
        representation=representation,
        payload=payload,
    )


def _derived_artifact(artifact_id, representation, payload, parent):
    return ProblemArtifact(
        artifact_id=artifact_id,
        representation=representation,
        payload=payload,
        parent_artifact_id=parent,
        transformation=TransformationRecord(
            name='fixture_transform',
            version='1',
            lossless=False,
            context={'penalty': artifact_id},
        ),
    )


class ProblemCaseStructureTests(unittest.TestCase):
    def test_maxcut_derivation_preserves_large_integer_offset_exactly(self):
        # Offset is only the affine shift between QUBO energy and cut value.
        # Graph derivation must therefore retain arbitrary JSON integers rather
        # than forcing them through a lossy/overflowing float conversion.
        for offset in (10_000_000_000_000_001, 10**400):
            problem_id = f'large-offset-{len(str(offset))}'
            qubo = _qubo_payload(problem_id)
            qubo['offset'] = offset
            problem_case = ProblemCase(
                problem_id=problem_id,
                artifacts=(
                    _root_artifact('qubo', 'qubo.v1', qubo),
                ),
            )

            derived = derive_qubo_maxcut_graph(problem_case, 'qubo')
            artifact = derived.get_artifact('qubo-maxcut-graph')

            self.assertEqual(
                offset,
                artifact.transformation.context['energy_mapping']['shift'],
            )
            self.assertIs(
                int,
                type(
                    artifact.transformation.context[
                        'energy_mapping'
                    ]['shift']
                ),
            )

    def test_json_snapshots_reject_non_string_object_keys(self):
        with self.assertRaisesRegex(TypeError, 'object keys must be strings'):
            ProblemArtifact(
                artifact_id='bad-json-key',
                representation='custom.v1',
                payload={'metadata': {1: 'silently-coerced-before'}},
            )

        with self.assertRaisesRegex(TypeError, 'object keys must be strings'):
            BestKnownSolution(
                solution={'assignments': [{0: 1}]},
                objective_value=0,
            )

    def test_graph_native_case_may_have_no_task_or_problem_type(self):
        graph = nx.Graph(name='untyped business graph')
        graph.add_node('asset-alpha', variable_name='客户资产甲')
        graph.add_node('asset-beta', variable_name='客户资产乙')
        graph.add_edge('asset-alpha', 'asset-beta', relation='depends_on')

        problem_case = create_graph_case(
            'native-graph',
            graph,
            metadata={'owner': 'research'},
        )

        self.assertEqual((), problem_case.tasks)
        self.assertEqual('graph', problem_case.primary_artifact_id)
        self.assertEqual(
            'networkx.node-link.v1',
            problem_case.get_artifact('graph').representation,
        )
        graph_payload = problem_case.get_artifact('graph').payload
        self.assertEqual(
            ['asset-alpha', 'asset-beta'],
            graph_payload['graph']['node_order'],
        )
        self.assertEqual(
            ['客户资产甲', '客户资产乙'],
            graph_payload['graph']['variable_names'],
        )
        self.assertNotIn('problem_type', problem_case.metadata)
        self.assertNotIn('sense', problem_case.metadata)

        with self.assertRaises(KeyError):
            update_best_known(
                problem_case,
                'implicitly-guessed-task',
                BestKnownSolution(['asset-alpha'], 1.0),
            )

    def test_case_round_trip_preserves_multiple_and_duplicate_representations(self):
        cbqm = _root_artifact('cbqm', 'cbqm.v1', _cbqm_payload())
        qubo_a = _derived_artifact(
            'qubo-penalty-5',
            'qubo.v1',
            _qubo_payload(),
            'cbqm',
        )
        qubo_b_payload = _qubo_payload()
        qubo_b_payload['metadata'] = {
            'compiler': 'fixture-v1',
            'penalty': 20,
        }
        qubo_b = _derived_artifact(
            'qubo-penalty-20',
            'qubo.v1',
            qubo_b_payload,
            'cbqm',
        )
        task = TaskDefinition(
            task_id='allocation',
            canonical_artifact_id='cbqm',
            task_type='portfolio_selection',
            sense='minimize',
            best_known=BestKnownSolution(
                [0, 1, 0, 1],
                1.25,
                source='heuristic',
                metadata={'run': 7},
            ),
            metadata={'customer': 'example'},
        )
        problem_case = ProblemCase(
            problem_id='integration-case',
            artifacts=(cbqm, qubo_a, qubo_b),
            tasks=(task,),
            primary_artifact_id='cbqm',
            metadata={'title': 'One case, several views'},
        )

        with tempfile.TemporaryDirectory() as directory:
            case_directory = Path(directory) / 'case-a'
            manifest_path = save_problem_case(problem_case, case_directory)
            restored = load_problem_case(case_directory)

            self.assertEqual(case_directory / 'case.json', manifest_path)
            self.assertTrue(
                (case_directory / 'artifacts' / 'cbqm.json').is_file()
            )
            self.assertTrue(
                (case_directory / 'artifacts' / 'qubo-penalty-20.json').is_file()
            )

        self.assertEqual(problem_case, restored)
        self.assertEqual(
            ('qubo-penalty-5', 'qubo-penalty-20'),
            tuple(
                artifact.artifact_id
                for artifact in restored.artifacts_for('qubo.v1')
            ),
        )
        self.assertEqual(
            {'penalty': 'qubo-penalty-20'},
            dict(restored.get_artifact('qubo-penalty-20').transformation.context),
        )

    def test_load_problem_set_uses_stable_child_directory_order(self):
        case_a = ProblemCase(
            problem_id='case-a',
            artifacts=(
                _root_artifact('qubo', 'qubo.v1', _qubo_payload('case-a')),
            ),
            primary_artifact_id='qubo',
        )
        case_b = ProblemCase(
            problem_id='case-b',
            artifacts=(
                _root_artifact('qubo', 'qubo.v1', _qubo_payload('case-b')),
            ),
            primary_artifact_id='qubo',
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_problem_case(case_b, root / '20-b')
            save_problem_case(case_a, root / '10-a')
            problem_set = load_problem_set(root)

        self.assertEqual(
            ('case-a', 'case-b'),
            tuple(item.problem_id for item in problem_set),
        )
        self.assertIs(problem_set.get('case-b'), problem_set.cases[1])

    def test_payload_and_metadata_are_detached_from_caller_mutation(self):
        payload = _qubo_payload()
        metadata = {'tags': ['original']}
        artifact = ProblemArtifact(
            artifact_id='qubo',
            representation='qubo.v1',
            payload=payload,
            metadata=metadata,
        )

        payload['terms'][0][2] = 999
        metadata['tags'].append('mutated')

        self.assertEqual(-2.0, artifact.payload['terms'][0][2])
        self.assertEqual(['original'], artifact.metadata['tags'])

    def test_replacing_canonical_payload_requires_clearing_best_known(self):
        qubo = _root_artifact('qubo', 'qubo.v1', _qubo_payload())
        task = TaskDefinition(
            task_id='energy',
            canonical_artifact_id='qubo',
            sense='minimize',
            best_known=BestKnownSolution(
                [1, 0, 0, 0],
                -0.75,
                exact=True,
            ),
        )
        problem_case = ProblemCase(
            problem_id='integration-case',
            artifacts=(qubo,),
            tasks=(task,),
        )
        changed_payload = _qubo_payload()
        changed_payload['offset'] = 99.0
        replacement = _root_artifact(
            'qubo',
            'qubo.v1',
            changed_payload,
        )

        with self.assertRaisesRegex(ValueError, 'Clear those incumbents'):
            problem_case.with_artifact(
                replacement,
                replace_existing=True,
            )


class LineageValidationTests(unittest.TestCase):
    def test_rejects_missing_parent_and_cycle(self):
        missing_parent = _derived_artifact(
            'child',
            'networkx.fixture.v1',
            {'nodes': [], 'edges': []},
            'not-in-this-case',
        )
        with self.assertRaisesRegex(ValueError, 'missing parent'):
            ProblemCase(
                problem_id='lineage',
                artifacts=(missing_parent,),
            )

        transform = TransformationRecord(
            name='cycle',
            version='1',
            lossless=True,
        )
        artifact_a = ProblemArtifact(
            artifact_id='a',
            representation='networkx.fixture.v1',
            payload={},
            parent_artifact_id='b',
            transformation=transform,
        )
        artifact_b = ProblemArtifact(
            artifact_id='b',
            representation='networkx.fixture.v1',
            payload={},
            parent_artifact_id='a',
            transformation=transform,
        )
        with self.assertRaisesRegex(ValueError, 'cycle'):
            ProblemCase(
                problem_id='lineage',
                artifacts=(artifact_a, artifact_b),
            )

    def test_derived_artifact_requires_transformation_record(self):
        with self.assertRaisesRegex(ValueError, 'transformation'):
            ProblemArtifact(
                artifact_id='child',
                representation='networkx.fixture.v1',
                payload={},
                parent_artifact_id='parent',
            )

        with self.assertRaisesRegex(ValueError, 'own parent'):
            ProblemArtifact(
                artifact_id='same',
                representation='networkx.fixture.v1',
                payload={},
                parent_artifact_id='same',
                transformation=TransformationRecord(
                    name='invalid',
                    version='1',
                    lossless=True,
                ),
            )


class BestKnownTaskTests(unittest.TestCase):
    def _case_with_two_tasks(self):
        cbqm = _root_artifact('cbqm', 'cbqm.v1', _cbqm_payload())
        qubo = _root_artifact('qubo', 'qubo.v1', _qubo_payload())
        return ProblemCase(
            problem_id='integration-case',
            artifacts=(cbqm, qubo),
            tasks=(
                TaskDefinition(
                    task_id='compiled-energy',
                    canonical_artifact_id='qubo',
                    sense='minimize',
                    best_known=BestKnownSolution(
                        [0, 0, 0, 0],
                        1.25,
                        exact=True,
                        source='exact enumeration',
                    ),
                ),
                TaskDefinition(
                    task_id='business-objective',
                    canonical_artifact_id='cbqm',
                    sense='minimize',
                    best_known=BestKnownSolution(
                        [0, 1, 0, 1],
                        1.25,
                        source='heuristic',
                    ),
                ),
            ),
            primary_artifact_id='cbqm',
        )

    def test_exact_lock_is_scoped_to_selected_task(self):
        problem_case = self._case_with_two_tasks()

        locked = update_best_known(
            problem_case,
            'compiled-energy',
            BestKnownSolution([1, 0, 0, 0], -0.75),
        )
        improved = update_best_known(
            problem_case,
            'business-objective',
            BestKnownSolution([1, 0, 0, 1], -1.75),
        )

        self.assertFalse(locked.updated)
        self.assertEqual('exact_solution_locked', locked.reason)
        self.assertTrue(improved.updated)
        self.assertEqual('objective_improved', improved.reason)
        self.assertEqual(
            1.25,
            improved.case.get_task('compiled-energy').best_known.objective_value,
        )
        self.assertTrue(
            improved.case.get_task('compiled-energy').best_known.exact
        )
        self.assertEqual(
            -1.75,
            improved.case.get_task('business-objective').best_known.objective_value,
        )

    def test_cbqm_evaluator_rejects_infeasible_candidate(self):
        problem_case = self._case_with_two_tasks()

        with self.assertRaisesRegex(ValueError, 'violates'):
            update_best_known(
                problem_case,
                'business-objective',
                BestKnownSolution([0, 0, 0, 1], 0.25),
            )

    def test_qubo_evaluator_recomputes_candidate_energy(self):
        problem_case = self._case_with_two_tasks()

        with self.assertRaisesRegex(ValueError, 'does not match evaluation'):
            update_best_known(
                problem_case,
                'compiled-energy',
                BestKnownSolution([1, 0, 0, 0], -999.0),
                allow_exact_override=True,
            )

    def test_builtin_evaluator_validates_the_canonical_payload(self):
        malformed = {
            'schema': 'qubo.v1',
            'problem_id': 'malformed-updater',
            'sense': 'minimize',
            'num_variables': 1,
            'variable_names': ['x'],
            'offset': False,
            'terms': [[0, 0, True]],
            'metadata': {},
        }
        problem_case = ProblemCase(
            problem_id='malformed-updater',
            artifacts=(
                _root_artifact('qubo', 'qubo.v1', malformed),
            ),
            tasks=(
                TaskDefinition(
                    task_id='energy',
                    canonical_artifact_id='qubo',
                    sense='minimize',
                ),
            ),
        )

        with self.assertRaisesRegex(TypeError, 'finite real number'):
            update_best_known(
                problem_case,
                'energy',
                BestKnownSolution([1], 1),
            )

    def test_binary_vector_rejects_float_lookalikes(self):
        problem_case = self._case_with_two_tasks()

        with self.assertRaisesRegex(ValueError, 'integer 0 or 1'):
            update_best_known(
                problem_case,
                'compiled-energy',
                BestKnownSolution([1.0, 0, 0, 0], -0.75),
                allow_exact_override=True,
            )

    def test_large_exact_objective_cannot_be_locked_with_relative_error(self):
        objective = 1_000_000_000_000_000
        payload = {
            'schema': 'qubo.v1',
            'problem_id': 'large-objective-lock',
            'sense': 'minimize',
            'num_variables': 1,
            'variable_names': ['x'],
            'offset': objective,
            'terms': [],
            'metadata': {},
        }
        problem_case = ProblemCase(
            problem_id='large-objective-lock',
            artifacts=(
                _root_artifact('qubo', 'qubo.v1', payload),
            ),
            tasks=(
                TaskDefinition(
                    task_id='energy',
                    canonical_artifact_id='qubo',
                    sense='minimize',
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, 'does not match evaluation'):
            update_best_known(
                problem_case,
                'energy',
                BestKnownSolution(
                    [0],
                    objective - 500_000,
                    exact=True,
                ),
            )

        accepted = update_best_known(
            problem_case,
            'energy',
            BestKnownSolution([0], objective, exact=True),
        )
        self.assertTrue(accepted.updated)
        self.assertIs(
            type(accepted.current.objective_value),
            int,
        )

    def test_unknown_graph_task_requires_an_explicit_evaluator(self):
        graph = nx.path_graph(['a', 'b', 'c'])
        graph_case = create_graph_case('unknown-task', graph)
        graph_case = graph_case.with_task(
            TaskDefinition(
                task_id='custom-score',
                canonical_artifact_id='graph',
                task_type='business_specific',
                sense='maximize',
                solution_representation='node-set.v1',
            )
        )
        candidate = BestKnownSolution(['a', 'c'], 2.0)

        with self.assertRaisesRegex(NotImplementedError, 'built-in evaluator'):
            update_best_known(graph_case, 'custom-score', candidate)

        accepted = update_best_known(
            graph_case,
            'custom-score',
            candidate,
            evaluator=lambda artifact, solution, task: len(solution),
        )
        self.assertTrue(accepted.updated)
        self.assertEqual(2.0, accepted.current.objective_value)


class CaseOperationTests(unittest.TestCase):
    def test_injected_compiler_receives_fully_detached_inputs(self):
        source = _root_artifact('cbqm', 'cbqm.v1', _cbqm_payload())
        problem_case = ProblemCase(
            problem_id='integration-case',
            artifacts=(source,),
        )
        config = {
            'default_penalty': 7.0,
            'custom': {'labels': ['original']},
        }

        def mutating_compiler(cbqm, received_config):
            cbqm['variables'][0]['name'] = 'mutated'
            received_config['custom']['labels'].append('mutated')
            raise RuntimeError('stop after attempted mutation')

        with self.assertRaisesRegex(RuntimeError, 'attempted mutation'):
            compile_case_qubo(
                problem_case,
                'cbqm',
                config,
                compiler=mutating_compiler,
            )

        self.assertEqual(
            'budget',
            problem_case.get_artifact('cbqm').payload['variables'][0]['name'],
        )
        self.assertEqual(['original'], config['custom']['labels'])

    def test_injected_compiler_output_must_pass_full_qubo_contract(self):
        source = _root_artifact('cbqm', 'cbqm.v1', _cbqm_payload())
        problem_case = ProblemCase(
            problem_id='integration-case',
            artifacts=(source,),
        )

        def malformed_compiler(cbqm, config):
            return (
                {
                    'schema': 'qubo.v1',
                    'problem_id': cbqm['problem_id'],
                    'sense': 'minimize',
                    'num_variables': 0,
                    'variable_names': [],
                    'offset': 0.0,
                    'terms': [],
                    'metadata': 'not-an-object',
                },
                {'compiler_config': dict(config)},
            )

        with self.assertRaisesRegex(TypeError, 'must be an object'):
            compile_case_qubo(
                problem_case,
                'cbqm',
                {'default_penalty': 7.0},
                compiler=malformed_compiler,
            )

    def test_default_compiler_resolves_options_without_breaking_projection(self):
        source = _root_artifact('cbqm', 'cbqm.v1', _cbqm_payload())
        problem_case = ProblemCase(
            problem_id='integration-case',
            artifacts=(source,),
            primary_artifact_id='cbqm',
        )

        compiled = compile_case_qubo(
            problem_case,
            'cbqm',
            {'default_penalty': 7.0},
            target_artifact_id='qubo-defaults',
        )

        transformation = compiled.get_artifact(
            'qubo-defaults'
        ).transformation
        self.assertEqual(
            {'default_penalty': 7.0},
            transformation.context['requested_config'],
        )
        self.assertEqual(
            'quadratic_penalty',
            transformation.context['compilation']['compiler_config']['strategy'],
        )
        qubo_variable_count = compiled.get_artifact(
            'qubo-defaults'
        ).payload['num_variables']
        projected = project_qubo_sample_to_cbqm(
            compiled,
            'qubo-defaults',
            [1, 0, 1, *([0] * (qubo_variable_count - 3))],
        )
        self.assertEqual([1, 0, 1, 1], projected)

    def test_compile_with_injected_compiler_records_lineage_and_decodes_sample(self):
        source = _root_artifact('cbqm', 'cbqm.v1', _cbqm_payload())
        task = TaskDefinition(
            task_id='canonical-business-task',
            canonical_artifact_id='cbqm',
            sense='minimize',
            best_known=BestKnownSolution(
                [0, 1, 0, 1],
                1.25,
                exact=False,
            ),
        )
        problem_case = ProblemCase(
            problem_id='integration-case',
            artifacts=(source,),
            tasks=(task,),
            primary_artifact_id='cbqm',
        )
        config = {'default_penalty': 7.0}

        def fake_compiler(cbqm, received_config):
            self.assertEqual(_cbqm_payload(), cbqm)
            self.assertEqual(config, received_config)
            qubo = {
                'schema': 'qubo.v1',
                'problem_id': cbqm['problem_id'],
                'sense': 'minimize',
                'num_variables': 3,
                'variable_names': [
                    'budget',
                    'asset_beta',
                    'unused_free',
                ],
                'offset': 0.0,
                'terms': [[0, 0, -1.0]],
                'metadata': {
                    'compiler': 'fake-compiler-v1',
                    'compiler_config': dict(received_config),
                },
            }
            context = {
                'schema': 'qubo-compilation-context.v1',
                'source_problem_id': cbqm['problem_id'],
                'source_variable_count': 4,
                'source_variable_names': [
                    'budget',
                    'asset_beta',
                    'unused_free',
                    'fixed_unused',
                ],
                'objective_sense': 'minimize',
                'objective_multiplier': 1.0,
                'qubo_variable_count': 3,
                'qubo_variable_names': list(qubo['variable_names']),
                'compiler_config': dict(received_config),
                'fixed_values': [{'index': 3, 'value': 1}],
                'free_variables': [
                    {
                        'cbqm_index': 0,
                        'qubo_index': 0,
                        'name': 'budget',
                    },
                    {
                        'cbqm_index': 1,
                        'qubo_index': 1,
                        'name': 'asset_beta',
                    },
                    {
                        'cbqm_index': 2,
                        'qubo_index': 2,
                        'name': 'unused_free',
                    },
                ],
                'slack_variables': [],
            }
            return qubo, context

        compiled = compile_case_qubo(
            problem_case,
            'cbqm',
            config,
            target_artifact_id='qubo-penalty-7',
            compiler=fake_compiler,
        )

        derived = compiled.get_artifact('qubo-penalty-7')
        self.assertEqual('cbqm', derived.parent_artifact_id)
        self.assertEqual('compile_qubo', derived.transformation.name)
        self.assertFalse(derived.transformation.lossless)
        self.assertEqual(
            config,
            derived.transformation.context['requested_config'],
        )
        self.assertEqual(
            [1, 0, 1, 1],
            project_qubo_sample_to_cbqm(
                compiled,
                'qubo-penalty-7',
                [1, 0, 1],
            ),
        )

        # Merely adding an exact-solver-compatible QUBO must not alter or
        # promote the incumbent attached to the canonical CBQM task.
        self.assertEqual(
            problem_case.get_task('canonical-business-task').best_known,
            compiled.get_task('canonical-business-task').best_known,
        )
        self.assertEqual(1, len(problem_case.artifacts))
        self.assertEqual(2, len(compiled.artifacts))


if __name__ == '__main__':
    unittest.main()
