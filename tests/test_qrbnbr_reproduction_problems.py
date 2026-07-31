import unittest

import networkx as nx

from lib.contracts.validation import validate_qubo
from problem.graph_codec import graph_from_node_link
from problem.reproductions import (
    build_qrbnbr_s1_instance,
    build_qrbnbr_s1_suite,
)


class QrbnbrReproductionProblemTests(unittest.TestCase):
    def test_instance_is_canonical_deterministic_and_graph_backed(self):
        first = build_qrbnbr_s1_instance(
            variable_count=9,
            edge_probability=0.4,
            seed=17,
        )
        second = build_qrbnbr_s1_instance(
            variable_count=9,
            edge_probability=0.4,
            seed=17,
        )

        self.assertEqual(first, second)
        validate_qubo(first)
        graph = graph_from_node_link(
            first['metadata']['source_graph'],
        )
        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(9, graph.number_of_nodes())
        self.assertTrue(nx.is_connected(graph))
        self.assertTrue(
            all(attributes['weight'] == 1 for *_, attributes in graph.edges(data=True))
        )
        self.assertEqual(
            'S1',
            first['metadata']['reproduction']['experiment_family'],
        )

    def test_full_probability_builds_a_complete_graph_maxcut_qubo(self):
        problem = build_qrbnbr_s1_instance(
            variable_count=4,
            edge_probability=1.0,
            seed=0,
        )
        graph = graph_from_node_link(problem['metadata']['source_graph'])

        self.assertEqual(6, graph.number_of_edges())
        diagonal = {
            left: coefficient
            for left, right, coefficient in problem['terms']
            if left == right
        }
        self.assertEqual({0: -3, 1: -3, 2: -3, 3: -3}, diagonal)

    def test_suite_stays_inside_the_declared_s1_size_range(self):
        suite = build_qrbnbr_s1_suite()

        self.assertEqual(
            {
                'qrbnbr.s1-er-8',
                'qrbnbr.s1-er-12',
                'qrbnbr.s1-er-16',
                'qrbnbr.s1-er-18',
            },
            set(suite),
        )
        for problem in suite.values():
            validate_qubo(problem)
            self.assertLess(problem['num_variables'], 20)

    def test_rejects_settings_outside_the_reproduction_family(self):
        with self.assertRaisesRegex(ValueError, 'variable_count'):
            build_qrbnbr_s1_instance(variable_count=20)
        with self.assertRaisesRegex(ValueError, 'edge_probability'):
            build_qrbnbr_s1_instance(edge_probability=1.1)
        with self.assertRaisesRegex(ValueError, 'seed'):
            build_qrbnbr_s1_instance(seed=-1)


if __name__ == '__main__':
    unittest.main()
