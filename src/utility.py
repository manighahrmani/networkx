"""
Utility Module
--------------
This module contains utility functions for graph analysis.
It includes functions to check for cliques, separators, and to get vertex connectivity.
"""

from typing import List, Union, Tuple
import unittest
import networkx as nx # type: ignore

def get_vertex_connectivity(graph: nx.Graph) -> int:
    """
    Returns the vertex connectivity of a graph.
    
    Parameters:
        graph (nx.Graph): The input graph.
        
    Returns:
        int: The vertex connectivity of the graph.
    """
    return nx.node_connectivity(graph)

def is_clique(graph: nx.Graph, vertex_set: List[Union[int, Tuple[int, int]]]) -> bool:
    """
    Checks if a given vertex set forms a clique in the graph.
    
    Parameters:
        graph (nx.Graph): The input graph.
        vertex_set (List[Union[int, Tuple[int, int]]]): The set of vertices to check.
        
    Returns:
        bool: True if the vertex set forms a clique, False otherwise.
    """
    all_cliques = list(nx.find_cliques(graph.subgraph(vertex_set)))
    return len(all_cliques) == 1 and set(all_cliques[0]) == set(vertex_set)

def is_separator(graph: nx.Graph, vertex_set: List[Union[int, Tuple[int, int]]]) -> bool:
    """
    Checks if a given vertex set is a separator in the graph.
    
    Parameters:
        graph (nx.Graph): The input graph.
        vertex_set (List[Union[int, Tuple[int, int]]]): The set of vertices to check.
        
    Returns:
        bool: True if the vertex set is a separator, False otherwise.
    """
    original_components = nx.number_connected_components(graph)
    graph_removed = graph.copy()
    graph_removed.remove_nodes_from(vertex_set)
    new_components = nx.number_connected_components(graph_removed)
    return new_components > original_components

class TestGraphFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    def test_get_vertex_connectivity(self):
        """Test for get_vertex_connectivity function."""
        graph = nx.complete_graph(5)
        self.assertEqual(get_vertex_connectivity(graph), 4)

        graph = nx.path_graph(5)
        self.assertEqual(get_vertex_connectivity(graph), 1)

    def test_is_clique(self):
        """Test for is_clique function."""
        graph = nx.complete_graph(5)
        vertex_set = [0, 1, 2, 3]
        self.assertTrue(is_clique(graph, vertex_set))

        vertex_set = [0, 1, 4]
        self.assertTrue(is_clique(graph, vertex_set))

        vertex_set = [0, 2, 4]
        self.assertTrue(is_clique(graph, vertex_set))

    def test_is_separator(self):
        """Test for is_separator function."""
        graph = nx.grid_2d_graph(3, 3)

        vertex_set = [(1, 1)]
        self.assertFalse(is_separator(graph, vertex_set))

        vertex_set = [(0, 0)]
        self.assertFalse(is_separator(graph, vertex_set))

        vertex_set = [(1, 0), (1, 1), (1, 2)]
        self.assertTrue(is_separator(graph, vertex_set))

if __name__ == '__main__':
    unittest.main()
