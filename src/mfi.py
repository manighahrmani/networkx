"""
mfi.py: Module for finding the Minimum Fill-In (MFI) of a graph.

This module contains a function that calculates the minimum triangulation
of a given graph using a brute-force approach. It also includes test cases
for the function.
"""

import unittest
import itertools
import networkx as nx  # type: ignore


def minimum_triangulation_bruteforce(graph):
    """
    Find the minimum triangulation of a graph using a brute-force approach.

    Parameters:
    - graph (NetworkX graph): The input graph.

    Returns:
    - NetworkX graph: The triangulated graph.
    - int: The number of added edges for triangulation.
    """
    min_added_edges = float('inf')
    # best_ordering = None
    best_triangulated_graph = None

    # Generate all possible elimination orderings
    all_orderings = list(itertools.permutations(graph.nodes()))

    for ordering in all_orderings:
        graph_h = graph.copy()
        added_edges = 0

        for node in ordering:
            neighbors = list(graph_h.neighbors(node))

            # Add edges to make the neighborhood of the node a clique
            for i, vertex_u in enumerate(neighbors):
                for j, vertex_v in enumerate(neighbors):
                    if i < j and not graph_h.has_edge(vertex_u, vertex_v):
                        graph_h.add_edge(vertex_u, vertex_v)
                        added_edges += 1

            # Remove the node
            graph_h.remove_node(node)

        if added_edges < min_added_edges:
            min_added_edges = added_edges
            # best_ordering = ordering
            best_triangulated_graph = graph_h.copy()

    return best_triangulated_graph, min_added_edges


MFI_FUNCTION = minimum_triangulation_bruteforce


class TestMinimumTriangulation(unittest.TestCase):
    """Test cases for minimum triangulation functions."""

    def test_minimum_triangulation(self):
        """Test the minimum_triangulation_bruteforce function."""
        # 3x3 grid graph
        graph = nx.grid_2d_graph(3, 3)
        _, added_edges = MFI_FUNCTION(graph)
        self.assertEqual(added_edges, 5)

        # 3x4 grid graph
        graph = nx.grid_2d_graph(3, 4)
        _, added_edges = MFI_FUNCTION(graph)
        self.assertEqual(added_edges, 9)  # 5 + 4 * (4 - 3)

        # 2x2 grid graph
        graph = nx.grid_2d_graph(2, 2)
        _, added_edges = MFI_FUNCTION(graph)
        self.assertEqual(added_edges, 5)  # 4 * 2 - 3

        # 2x3 grid graph
        graph = nx.grid_2d_graph(2, 3)
        _, added_edges = MFI_FUNCTION(graph)
        self.assertEqual(added_edges, 9)  # 4 * 3 - 3


if __name__ == "__main__":
    unittest.main()
