"""
Utility Module
--------------
This module contains utility functions for graph analysis.
It includes the following functions:
- get_vertex_degree
- get_missing_edges_in_neighborhood
- is_simplicial
- is_almost_simplicial
- is_clique_minimal_separator
- is_minimal_separator
- get_vertex_connectivity
- is_clique
- is_separator
"""

from typing import List, Union, Tuple
import unittest
import networkx as nx  # type: ignore


def get_vertex_degree(graph: nx.Graph, vertex: Union[int, Tuple[int, int]]) -> int:
    """
    Returns the degree of a given vertex in the graph.

    Parameters:
        graph (nx.Graph): The input graph.
        vertex (Union[int, Tuple[int, int]]): The vertex whose degree is to be found.

    Returns:
        int: The degree of the vertex.
    """
    return graph.degree(vertex)


def get_missing_edges_in_neighborhood(
        graph: nx.Graph,
        vertex: Union[int, Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Returns the missing edges in the neighborhood of a given vertex.

    Parameters:
        graph (nx.Graph): The input graph.
        vertex (Union[int, Tuple[int, int]]): The vertex.

    Returns:
        List[Tuple[int, int]]: The list of missing edges.
    """
    neighbors = list(graph.neighbors(vertex))
    induced_subgraph = graph.subgraph(neighbors)
    missing_edges = [(u, v) for u in neighbors for v in neighbors if u <
                     v and not induced_subgraph.has_edge(u, v)]
    return missing_edges


def is_simplicial(graph: nx.Graph, vertex: Union[int, Tuple[int, int]]) -> bool:
    """
    Checks if a given vertex is simplicial in the graph.

    Parameters:
        graph (nx.Graph): The input graph.
        vertex (Union[int, Tuple[int, int]]): The vertex to check.

    Returns:
        bool: True if the vertex is simplicial, False otherwise.
    """
    neighbors = list(graph.neighbors(vertex))
    return is_clique(graph, neighbors)


def is_almost_simplicial(graph: nx.Graph, vertex: Union[int, Tuple[int, int]]) -> bool:
    """
    Checks if a given vertex is almost simplicial in the graph.

    Parameters:
        graph (nx.Graph): The input graph.
        vertex (Union[int, Tuple[int, int]]): The vertex to check.

    Returns:
        bool: True if the vertex is almost simplicial, False otherwise.
    """
    if is_simplicial(graph, vertex):
        return False

    neighbors = list(graph.neighbors(vertex))

    for vertex_u in neighbors:
        subset = [v for v in neighbors if v != vertex_u]
        if is_clique(graph, subset):
            return True
    return False


def is_clique_minimal_separator(
    graph: nx.Graph,
    vertex_set: List[Union[int, Tuple[int, int]]]
) -> bool:
    """
    Checks if a given vertex set is a clique minimal separator in the graph.

    Parameters:
        graph (nx.Graph): The input graph.
        vertex_set (List[Union[int, Tuple[int, int]]]): The set of vertices to check.

    Returns:
        bool: True if the vertex set is a clique minimal separator, False otherwise.
    """
    return is_clique(graph, vertex_set) and is_minimal_separator(graph, vertex_set)


def is_minimal_separator(graph: nx.Graph, vertex_set: List[Union[int, Tuple[int, int]]]) -> bool:
    """
    Checks if a given vertex set is a minimal separator in the graph.

    Parameters:
        graph (nx.Graph): The input graph.
        vertex_set (List[Union[int, Tuple[int, int]]]): The set of vertices to check.

    Returns:
        bool: True if the vertex set is a minimal separator, False otherwise.
    """
    # First, check if the vertex set is a separator
    if not is_separator(graph, vertex_set):
        return False

    # Then, check if it's minimal by trying to remove each vertex from the set
    for v in vertex_set:
        subset = [x for x in vertex_set if x != v]
        if is_separator(graph, subset):
            return False

    return True


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

    def test_is_minimal_separator(self):
        """Test for is_minimal_separator function."""
        graph = nx.grid_2d_graph(3, 3)

        # Not a separator, so not a minimal separator
        vertex_set = [(1, 1)]
        self.assertFalse(is_minimal_separator(graph, vertex_set))

        # A separator, but not minimal
        vertex_set = [(1, 0), (0, 1), (1, 2), (2, 1)]
        self.assertFalse(is_minimal_separator(graph, vertex_set))

        # A minimal separator
        vertex_set = [(1, 0), (0, 1)]
        self.assertTrue(is_minimal_separator(graph, vertex_set))

        # Test with a complete graph (should not have any separators)
        graph = nx.complete_graph(5)

        vertex_set = [0, 1]
        self.assertFalse(is_minimal_separator(graph, vertex_set))

    def test_is_clique_minimal_separator(self):
        """Test for is_clique_minimal_separator function."""
        graph = nx.grid_2d_graph(3, 3)

        # Not a separator, so not a clique minimal separator
        vertex_set = [(1, 1)]
        self.assertFalse(is_clique_minimal_separator(graph, vertex_set))

        # A separator and a clique, but not minimal
        vertex_set = [(1, 0), (0, 1), (1, 2), (2, 1)]
        self.assertFalse(is_clique_minimal_separator(graph, vertex_set))

        # A minimal separator but not a clique
        vertex_set = [(1, 0), (0, 1)]
        self.assertFalse(is_clique_minimal_separator(graph, vertex_set))

        # Test with a complete graph (should not have any separators)
        graph = nx.complete_graph(5)

        vertex_set = [0, 1]
        self.assertFalse(is_clique_minimal_separator(graph, vertex_set))

    def test_is_simplicial(self):
        """Test for is_simplicial function."""
        graph = nx.complete_graph(5)

        # In a complete graph, every vertex should be simplicial
        for vertex in graph.nodes:
            self.assertTrue(is_simplicial(graph, vertex))

        graph = nx.path_graph(5)

        # In a path graph, only the end vertices should be simplicial
        self.assertTrue(is_simplicial(graph, 0))
        self.assertTrue(is_simplicial(graph, 4))
        self.assertFalse(is_simplicial(graph, 1))
        self.assertFalse(is_simplicial(graph, 2))
        self.assertFalse(is_simplicial(graph, 3))

    def test_is_almost_simplicial(self):
        """Test for is_almost_simplicial function."""
        graph = nx.complete_graph(5)

        # In a complete graph, no vertex should be almost simplicial
        for vertex in graph.nodes:
            self.assertFalse(is_almost_simplicial(graph, vertex))

        graph = nx.path_graph(5)

        # In a path graph, only the middle vertices should be almost simplicial
        self.assertFalse(is_almost_simplicial(graph, 0))
        self.assertFalse(is_almost_simplicial(graph, 4))
        self.assertTrue(is_almost_simplicial(graph, 1))
        self.assertTrue(is_almost_simplicial(graph, 2))
        self.assertTrue(is_almost_simplicial(graph, 3))

    def test_get_vertex_degree(self):
        """Test for get_vertex_degree function."""
        graph = nx.complete_graph(5)
        for vertex in graph.nodes:
            self.assertEqual(get_vertex_degree(graph, vertex), 4)

        graph = nx.path_graph(5)
        self.assertEqual(get_vertex_degree(graph, 0), 1)
        self.assertEqual(get_vertex_degree(graph, 4), 1)
        self.assertEqual(get_vertex_degree(graph, 2), 2)

    def test_get_missing_edges_in_neighborhood(self):
        """Test for get_missing_edges function."""
        graph = nx.complete_graph(5)
        for vertex in graph.nodes:
            self.assertEqual(get_missing_edges_in_neighborhood(graph, vertex), [])

        graph = nx.path_graph(5)
        self.assertEqual(get_missing_edges_in_neighborhood(graph, 0), [])
        self.assertEqual(get_missing_edges_in_neighborhood(graph, 4), [])
        self.assertEqual(get_missing_edges_in_neighborhood(graph, 2), [(1, 3)])


if __name__ == '__main__':
    unittest.main()
