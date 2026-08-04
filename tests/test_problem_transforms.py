"""Semantic tests for NetworkX views of CBQM and QUBO artifacts."""

from __future__ import annotations

import copy
import itertools
import unittest

import networkx as nx

from problem.graph_codec import (
    artifact_from_networkx,
    graph_from_node_link,
    graph_to_node_link,
)
from problem.transforms import (
    cbqm_to_factor_graph,
    factor_graph_to_cbqm,
    interaction_graph_to_qubo,
    qubo_to_interaction_graph,
    qubo_to_maxcut_graph,
)


def _qubo_payload():
    """Include every storage case: offset, diagonals, pair and isolated node."""
    return {
        'schema': 'qubo.v1',
        'problem_id': 'transform-qubo',
        'sense': 'minimize',
        'num_variables': 4,
        'variable_names': ['alpha', 'beta', 'gamma', 'isolated'],
        'offset': 2.75,
        'terms': [
            [0, 0, -3.0],
            [0, 2, 2.5],
            [1, 1, 4.0],
            [1, 2, -1.25],
            [2, 2, -0.5],
        ],
        'metadata': {'source': 'transform-test'},
    }


def _qubo_energy(qubo, sample):
    return float(qubo['offset']) + sum(
        float(coefficient) * sample[left] * sample[right]
        for left, right, coefficient in qubo['terms']
    )


def _cut_value(graph, partition):
    return sum(
        float(attributes['weight'])
        for left, right, attributes in graph.edges(data=True)
        if partition[left] != partition[right]
    )


def _cbqm_payload():
    """Use colliding names and equal supports to expose lossy graph designs."""
    return {
        'schema': 'cbqm.v1',
        'problem_id': 'transform-cbqm',
        'variables': [
            {
                'index': 0,
                'name': 'budget',
                'vartype': 'BINARY',
                'kind': 'decision',
                'metadata': {'display': 'Budget'},
            },
            {
                'index': 1,
                'name': 'asset',
                'vartype': 'BINARY',
                'kind': 'decision',
                'metadata': {'display': 'Asset'},
            },
            {
                'index': 2,
                'name': 'fixed-isolated',
                'vartype': 'BINARY',
                'kind': 'auxiliary',
                'metadata': {'display': 'Fixed'},
            },
        ],
        'objective': {
            'sense': 'maximize',
            'offset': -0.75,
            'linear': [[0, 1.5], [1, -2.0]],
            'quadratic': [[0, 0, 0.25], [0, 1, 3.0]],
        },
        'constraints': [
            {
                'name': 'budget',
                'family': 'lower-policy',
                'linear': [[0, 1.0], [1, 2.0]],
                'lower_bound': 1.0,
                'metadata': {'external_id': 'constraint-a'},
            },
            {
                'name': 'same-support',
                'family': 'upper-policy',
                'linear': [[0, -4.0], [1, 5.0]],
                'upper_bound': 3.0,
                'metadata': {'external_id': 'constraint-b'},
            },
            {
                'name': 'constant',
                'family': 'constant-policy',
                'linear': [],
                'lower_bound': 0.0,
                'upper_bound': 0.0,
                'metadata': {'must_survive': True},
            },
        ],
        'fixed_values': [{'index': 2, 'value': 1}],
        'metadata': {'owner': 'tests'},
    }


class GraphCodecTests(unittest.TestCase):
    def test_node_link_codec_writes_edges_and_reads_legacy_links(self):
        graph = nx.Graph(case='codec')
        graph.add_node(0, variable_name='alpha')
        graph.add_node(1, variable_name='beta')
        graph.add_edge(0, 1, weight=2.0)

        payload = graph_to_node_link(graph)
        self.assertIn('edges', payload)
        self.assertNotIn('links', payload)

        legacy_payload = dict(payload)
        legacy_payload['links'] = legacy_payload.pop('edges')
        restored = graph_from_node_link(legacy_payload)

        self.assertEqual('codec', restored.graph['case'])
        self.assertEqual('alpha', restored.nodes[0]['variable_name'])
        self.assertEqual(2.0, restored.edges[0, 1]['weight'])

    def test_node_link_decode_rejects_duplicate_nodes_and_simple_edges(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=2.0)
        payload = graph_to_node_link(graph)

        duplicate_node = copy.deepcopy(payload)
        duplicate_node['nodes'].append(
            copy.deepcopy(duplicate_node['nodes'][0])
        )
        with self.assertRaisesRegex(ValueError, 'duplicate node'):
            graph_from_node_link(duplicate_node)

        duplicate_edge = copy.deepcopy(payload)
        duplicate_edge['edges'].append(
            copy.deepcopy(duplicate_edge['edges'][0])
        )
        with self.assertRaisesRegex(ValueError, 'duplicate edge'):
            graph_from_node_link(duplicate_edge)

    def test_graph_codec_rejects_boolean_node_id_and_declared_order(self):
        payload = graph_to_node_link(nx.empty_graph(1))
        payload['nodes'][0]['id'] = False
        with self.assertRaisesRegex(TypeError, 'excluding booleans'):
            graph_from_node_link(payload)

        graph = nx.empty_graph(1)
        graph.graph['node_order'] = [False]
        with self.assertRaisesRegex(TypeError, 'excluding booleans'):
            artifact_from_networkx('boolean-order', graph)

    def test_graph_codec_rejects_unstable_multigraph_edge_keys(self):
        graph = nx.MultiGraph()
        graph.add_edge(0, 1, key=('tuple', 1))
        with self.assertRaisesRegex(TypeError, 'edge keys'):
            artifact_from_networkx('tuple-edge-key', graph)

        payload = graph_to_node_link(nx.MultiGraph([(0, 1)]))
        missing_then_zero = copy.deepcopy(payload)
        del missing_then_zero['edges'][0]['key']
        missing_then_zero['edges'].append({
            'source': 0,
            'target': 1,
            'key': 0,
        })
        with self.assertRaisesRegex(ValueError, 'must define a key'):
            graph_from_node_link(missing_then_zero)

    def test_graph_codec_rejects_reserved_attribute_collisions(self):
        mutations = []

        node_collision = nx.empty_graph(1)
        node_collision.nodes[0]['id'] = 'shadow'
        mutations.append(node_collision)

        edge_collision = nx.Graph()
        edge_collision.add_edge(0, 1, source='shadow')
        mutations.append(edge_collision)

        multiedge_collision = nx.MultiGraph()
        multiedge_collision.add_edge(0, 1, key=0)
        multiedge_collision.edges[0, 1, 0]['key'] = 'shadow'
        mutations.append(multiedge_collision)

        for graph in mutations:
            with self.subTest(graph_type=type(graph).__name__):
                with self.assertRaisesRegex(ValueError, 'reserved field'):
                    graph_to_node_link(graph)

    def test_graph_codec_rejects_non_mapping_graph_metadata(self):
        payload = graph_to_node_link(nx.empty_graph(1))

        for invalid in (None, [], 'metadata', 1):
            with self.subTest(value=invalid):
                malformed = copy.deepcopy(payload)
                malformed['graph'] = invalid
                with self.assertRaisesRegex(TypeError, 'must be a mapping'):
                    graph_from_node_link(malformed)

    def test_graph_codec_rejects_unknown_top_level_fields(self):
        payload = graph_to_node_link(nx.Graph())
        payload['egde_typo'] = []

        with self.assertRaisesRegex(
            ValueError,
            'unknown top-level fields: egde_typo',
        ):
            graph_from_node_link(payload)


class QuboInteractionGraphTests(unittest.TestCase):
    def test_rejects_schema_invalid_metadata_and_unknown_fields(self):
        invalid_metadata = _qubo_payload()
        invalid_metadata['metadata'] = 'not-an-object'
        with self.assertRaisesRegex(TypeError, 'must be an object'):
            qubo_to_interaction_graph(invalid_metadata)

        unknown_field = _qubo_payload()
        unknown_field['future_field'] = 7
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            qubo_to_interaction_graph(unknown_field)

    def test_round_trip_preserves_every_sample_energy_and_variable_identity(self):
        qubo = _qubo_payload()

        graph = qubo_to_interaction_graph(qubo)
        restored = interaction_graph_to_qubo(graph)

        self.assertEqual({0, 1, 2, 3}, set(graph.nodes))
        self.assertEqual(0, graph.degree[3])
        self.assertEqual('isolated', graph.nodes[3]['variable_name'])
        self.assertEqual(-3.0, graph.nodes[0]['linear_bias'])
        self.assertEqual(2.5, graph.edges[0, 2]['quadratic_bias'])
        self.assertEqual(qubo['problem_id'], restored['problem_id'])
        self.assertEqual(qubo['sense'], restored['sense'])
        self.assertEqual(qubo['offset'], restored['offset'])
        self.assertEqual(qubo['variable_names'], restored['variable_names'])
        self.assertEqual(qubo['metadata'], restored['metadata'])

        for sample in itertools.product((0, 1), repeat=qubo['num_variables']):
            self.assertAlmostEqual(
                _qubo_energy(qubo, sample),
                _qubo_energy(restored, sample),
            )

        self.assertTrue(
            all(left <= right for left, right, _ in restored['terms'])
        )
        self.assertTrue(
            all(coefficient != 0 for _, _, coefficient in restored['terms'])
        )
        self.assertEqual(
            len(restored['terms']),
            len({(left, right) for left, right, _ in restored['terms']}),
        )

    def test_reverse_rejects_boolean_interaction_indices(self):
        """``False == 0`` must not silently normalize persisted identities."""
        node_order_graph = qubo_to_interaction_graph(_qubo_payload())
        node_order_graph.graph['node_order'][0] = False
        with self.assertRaisesRegex(ValueError, r'node_order\[0\].*integer'):
            interaction_graph_to_qubo(node_order_graph)

        node_attribute_graph = qubo_to_interaction_graph(_qubo_payload())
        node_attribute_graph.nodes[0]['qubo_index'] = False
        with self.assertRaisesRegex(ValueError, 'qubo_index.*integer'):
            interaction_graph_to_qubo(node_attribute_graph)

        node_id_graph = qubo_to_interaction_graph(_qubo_payload())
        node_attributes = copy.deepcopy(node_id_graph.nodes[0])
        incident_edges = [
            (
                neighbor,
                copy.deepcopy(node_id_graph.edges[0, neighbor]),
            )
            for neighbor in node_id_graph.neighbors(0)
        ]
        node_id_graph.remove_node(0)
        node_id_graph.add_node(False, **node_attributes)
        for neighbor, edge_attributes in incident_edges:
            node_id_graph.add_edge(False, neighbor, **edge_attributes)

        with self.assertRaisesRegex(ValueError, 'node ID.*integer'):
            interaction_graph_to_qubo(node_id_graph)

    def test_reverse_rejects_source_extras_that_override_qubo_core(self):
        graph = qubo_to_interaction_graph(_qubo_payload())
        graph.graph['source_extra_fields'] = {'offset': 999.0}

        with self.assertRaisesRegex(ValueError, 'reserved fields: offset'):
            interaction_graph_to_qubo(graph)

    def test_reverse_rejects_non_boolean_presence_marker(self):
        graph = qubo_to_interaction_graph(_qubo_payload())
        graph.graph['source_metadata_present'] = 1

        with self.assertRaisesRegex(ValueError, 'presence marker'):
            interaction_graph_to_qubo(graph)


class CbqmFactorGraphTests(unittest.TestCase):
    def test_rejects_explicit_null_bound_and_unknown_nested_fields(self):
        null_bound = _cbqm_payload()
        null_bound['constraints'][0]['upper_bound'] = None
        with self.assertRaisesRegex(TypeError, 'finite real number'):
            cbqm_to_factor_graph(null_bound)

        unknown_field = copy.deepcopy(_cbqm_payload())
        unknown_field['variables'][0]['future_field'] = 7
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            cbqm_to_factor_graph(unknown_field)

    def test_preserves_constraint_identity_empty_factor_and_fixed_variable(self):
        cbqm = _cbqm_payload()

        graph = cbqm_to_factor_graph(cbqm)
        restored = factor_graph_to_cbqm(graph)

        self.assertIn('variable:0', graph)
        self.assertIn('constraint:0', graph)
        self.assertNotEqual('variable:0', 'constraint:0')
        self.assertIn('constraint:1', graph)
        self.assertIn('constraint:2', graph)
        self.assertIn('variable:2', graph)
        self.assertEqual(0, graph.degree['constraint:2'])
        self.assertEqual(0, graph.degree['variable:2'])

        first = graph.nodes['constraint:0']
        second = graph.nodes['constraint:1']
        constant = graph.nodes['constraint:2']
        self.assertEqual('budget', first['name'])
        self.assertEqual('lower-policy', first['family'])
        self.assertEqual(1.0, first['lower_bound'])
        self.assertIsNone(first['upper_bound'])
        self.assertEqual('constraint-a', first['metadata']['external_id'])
        self.assertEqual('same-support', second['name'])
        self.assertEqual('upper-policy', second['family'])
        self.assertEqual(3.0, second['upper_bound'])
        self.assertTrue(constant['metadata']['must_survive'])

        self.assertEqual(
            1.0,
            graph.edges['constraint:0', 'variable:0']['coefficient'],
        )
        self.assertEqual(
            -4.0,
            graph.edges['constraint:1', 'variable:0']['coefficient'],
        )
        self.assertEqual(1, graph.nodes['variable:2']['fixed_value'])
        self.assertEqual(cbqm, restored)

    def test_rejects_contradictory_quadratic_edge_metadata(self):
        graph = cbqm_to_factor_graph(_cbqm_payload())
        factor_id = 'objective:quadratic:1'
        edge = graph.edges[factor_id, 'variable:0']
        edge['coefficient'] = 999.0
        edge['incidence_type'] = 'constraint_term'

        with self.assertRaisesRegex(ValueError, 'edge metadata mismatch'):
            factor_graph_to_cbqm(graph)

    def test_reverse_rejects_boolean_factor_indices(self):
        """Every structural factor index must be a plain Python ``int``."""
        mutations = (
            (
                'variable_index',
                lambda graph: graph.nodes['variable:0'].__setitem__(
                    'variable_index',
                    False,
                ),
            ),
            (
                'term_index',
                lambda graph: graph.nodes[
                    'objective:quadratic:0'
                ].__setitem__('term_index', False),
            ),
            (
                'left_index',
                lambda graph: graph.nodes[
                    'objective:quadratic:0'
                ].__setitem__('left_index', False),
            ),
            (
                'constraint_index',
                lambda graph: graph.nodes['constraint:0'].__setitem__(
                    'constraint_index',
                    False,
                ),
            ),
            (
                'edge term_index',
                lambda graph: graph.edges[
                    'constraint:0',
                    'variable:0',
                ].__setitem__('term_index', False),
            ),
        )

        for label, mutate in mutations:
            with self.subTest(label=label):
                graph = cbqm_to_factor_graph(_cbqm_payload())
                mutate(graph)
                with self.assertRaisesRegex(ValueError, 'integer'):
                    factor_graph_to_cbqm(graph)

    def test_reverse_rejects_factor_index_identity_normalization(self):
        graph = cbqm_to_factor_graph(_cbqm_payload())
        graph = nx.relabel_nodes(
            graph,
            {'constraint:0': 'constraint:renamed'},
            copy=True,
        )

        with self.assertRaisesRegex(ValueError, 'node ID'):
            factor_graph_to_cbqm(graph)

    def test_reverse_rejects_boolean_incidence_metadata_and_fixed_value(self):
        linear_positions = cbqm_to_factor_graph(_cbqm_payload())
        linear_positions.edges[
            'objective:linear:0',
            'variable:0',
        ]['operand_positions'] = [False]
        with self.assertRaisesRegex(ValueError, 'plain integer'):
            factor_graph_to_cbqm(linear_positions)

        diagonal_positions = cbqm_to_factor_graph(_cbqm_payload())
        diagonal_positions.edges[
            'objective:quadratic:0',
            'variable:0',
        ]['operand_positions'] = [False, True]
        with self.assertRaisesRegex(ValueError, 'plain integer'):
            factor_graph_to_cbqm(diagonal_positions)

        boolean_coefficient = cbqm_to_factor_graph(_cbqm_payload())
        boolean_coefficient.edges[
            'constraint:0',
            'variable:0',
        ]['coefficient'] = True
        with self.assertRaisesRegex(TypeError, 'finite real number'):
            factor_graph_to_cbqm(boolean_coefficient)

        boolean_fixed_annotation = cbqm_to_factor_graph(_cbqm_payload())
        boolean_fixed_annotation.nodes['variable:2']['fixed_value'] = True
        with self.assertRaisesRegex(ValueError, 'fixed_value.*integer'):
            factor_graph_to_cbqm(boolean_fixed_annotation)

    def test_reverse_requires_fixed_list_and_annotations_to_be_bijective(self):
        missing_list_entry = cbqm_to_factor_graph(_cbqm_payload())
        missing_list_entry.graph['fixed_values'] = []
        with self.assertRaisesRegex(ValueError, 'annotations mismatch'):
            factor_graph_to_cbqm(missing_list_entry)

        unexpected_annotation = cbqm_to_factor_graph(_cbqm_payload())
        unexpected_annotation.nodes['variable:0']['fixed_value'] = 1
        with self.assertRaisesRegex(ValueError, 'annotations mismatch'):
            factor_graph_to_cbqm(unexpected_annotation)

    def test_constraint_incidence_must_use_canonical_variable_node(self):
        graph = cbqm_to_factor_graph(_cbqm_payload())
        edge_attributes = copy.deepcopy(
            graph.edges['constraint:0', 'variable:0']
        )
        graph.remove_edge('constraint:0', 'variable:0')
        graph.add_node(
            'rogue-variable',
            node_type='rogue',
            variable_index=0,
        )
        graph.add_edge(
            'constraint:0',
            'rogue-variable',
            **edge_attributes,
        )

        with self.assertRaisesRegex(
            ValueError,
            'unknown node|inconsistent variable node',
        ):
            factor_graph_to_cbqm(graph)

    def test_reverse_rejects_unrepresented_factor_graph_topology(self):
        topology_mutations = (
            (
                'rogue node',
                lambda graph: graph.add_node('rogue'),
                'unknown node',
            ),
            (
                'factor-to-factor edge',
                lambda graph: graph.add_edge(
                    'objective:linear:0',
                    'constraint:0',
                ),
                'edges must connect',
            ),
            (
                'variable self-loop',
                lambda graph: graph.add_edge('variable:0', 'variable:0'),
                'self-loops',
            ),
        )

        for label, mutate, message in topology_mutations:
            with self.subTest(label=label):
                graph = cbqm_to_factor_graph(_cbqm_payload())
                mutate(graph)
                with self.assertRaisesRegex(ValueError, message):
                    factor_graph_to_cbqm(graph)

    def test_reverse_rejects_extras_that_override_cbqm_core(self):
        mutations = (
            (
                'source',
                lambda graph: graph.graph.__setitem__(
                    'source_extra_fields',
                    {'constraints': []},
                ),
                'source_extra_fields',
            ),
            (
                'objective',
                lambda graph: graph.graph.__setitem__(
                    'objective_extra_fields',
                    {'linear': []},
                ),
                'objective_extra_fields',
            ),
            (
                'variable',
                lambda graph: graph.nodes['variable:0'].__setitem__(
                    'extra_fields',
                    {'index': 2},
                ),
                'Variable node',
            ),
            (
                'constraint',
                lambda graph: graph.nodes['constraint:0'].__setitem__(
                    'extra_fields',
                    {'linear': []},
                ),
                'Constraint factor',
            ),
        )

        for label, mutate, message in mutations:
            with self.subTest(label=label):
                graph = cbqm_to_factor_graph(_cbqm_payload())
                mutate(graph)
                with self.assertRaisesRegex(
                    ValueError,
                    f'{message}.*reserved fields',
                ):
                    factor_graph_to_cbqm(graph)


class QuboMaxCutGraphTests(unittest.TestCase):
    def test_anchor_cut_value_has_documented_energy_relation(self):
        qubo = _qubo_payload()

        graph = qubo_to_maxcut_graph(qubo)
        anchor = qubo['num_variables']

        self.assertIn(anchor, graph)
        self.assertTrue(graph.nodes[anchor]['is_anchor'])
        for sample in itertools.product((0, 1), repeat=qubo['num_variables']):
            partition = {
                index: sample[index]
                for index in range(qubo['num_variables'])
            }
            partition[anchor] = 0
            cut_value = _cut_value(graph, partition)
            self.assertAlmostEqual(
                _qubo_energy(qubo, sample),
                float(qubo['offset']) - cut_value,
            )


if __name__ == '__main__':
    unittest.main()
