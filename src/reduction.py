"""
Grid reduction module

Includes implementations of the following functions:
- generate_grid_graph(num_rows: int, num_columns: int) -> nx.Graph
- is_clique(graph: nx.Graph, vertexset: Set[int]) -> bool
- is_separator(graph: nx.Graph, vertexset: Set[int]) -> bool
- is_minimal_separator(graph: nx.Graph, vertexset: Set[int]) -> bool
- is_clique_minimal_separator(graph: nx.Graph, vertexset: Set[int]) -> bool
- is_simplicial(graph: nx.Graph, vertex: int) -> bool
- is_almost_clique(graph: nx.Graph, vertexset: Set[int]) -> bool
- clique_minimal_separator_decomposition(graph: nx.Graph) -> List[Set[int]]
- is_almost_simplicial(graph: nx.Graph, vertex: int) -> bool
"""

from typing import List, Set
import unittest
import os
from typing import Set
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore


def generate_grid_graph(num_rows: int, num_columns: int) -> nx.Graph:
    """
    Generate a grid graph with custom vertex labels and save its edges to a text file.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.

    Returns:
    - nx.Graph: The generated graph with custom vertex labels.
    """

    # Generate the original grid graph
    grid = nx.grid_2d_graph(num_rows, num_columns)

    # Generate a mapping from old labels (tuples) to new labels (strings).
    # Add a leading '1' to each label to avoid leading zeros.
    mapping = {(r, c): f"1{r+1:02}{c+1:02}" for r in range(num_rows)
               for c in range(num_columns)}

    # Create a new graph with nodes relabeled
    relabeled_graph = nx.relabel_nodes(grid, mapping)

    with open(
        os.path.join('reduction', 'logs', 'original',
                     f'{num_rows}x{num_columns}.txt'),
        mode='w',
        encoding='utf8'
    ) as f:
        for edge in relabeled_graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    pos = {node: (int(node[3:5]) - 1, -(int(node[1:3]) - 1))
           for node in relabeled_graph.nodes()}

    # Plot and save the original graph using the positions
    plt.figure(figsize=(8, 6))
    nx.draw(relabeled_graph, pos, with_labels=True, font_weight='bold')
    plt.savefig(os.path.join('reduction', 'images', 'original',
                f'{num_rows}x{num_columns}_grid.png'))
    plt.close()

    return relabeled_graph


def is_clique(graph: nx.Graph, vertexset: Set[int]) -> bool:
    """
    Check if a set of vertices induces a complete subgraph (clique) in the graph.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[int]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set induces a complete subgraph, otherwise False.
    """
    for vertex in vertexset:
        neighbors = set(graph.neighbors(vertex))
        if not (vertexset - {vertex}).issubset(neighbors):
            return False
    return True


def is_separator(graph: nx.Graph, vertexset: Set[int]) -> bool:
    """
    Check if removing a set of vertices from the graph disconnects it.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[int]): The set of vertices to remove.

    Returns:
    - bool: True if the vertex set disconnects the graph, otherwise False.
    """
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(vertexset)
    number_of_connected_components = nx.number_connected_components(graph_copy)
    return number_of_connected_components > 1


def is_minimal_separator(graph: nx.Graph, vertexset: Set[int]) -> bool:
    """
    Check if a vertex set is a minimal separator in the graph.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[int]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set is a minimal separator, otherwise False.
    """
    if not is_separator(graph, vertexset):
        return False

    for vertex in vertexset:
        if is_separator(graph, vertexset - {vertex}):
            return False
    return True


def is_clique_minimal_separator(graph: nx.Graph, vertexset: Set[int]) -> bool:
    """
    Check if a vertex set is both a minimal separator and a clique.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[int]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set is a clique minimal separator, otherwise False.
    """
    return is_minimal_separator(graph, vertexset) and is_clique(graph, vertexset)


def is_simplicial(graph: nx.Graph, vertex: int) -> bool:
    """
    Check if a vertex is simplicial (its neighborhood induces a clique).

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertex (int): The vertex to check.

    Returns:
    - bool: True if the vertex is simplicial, otherwise False.
    """
    neighbors = set(graph.neighbors(vertex))
    return is_clique(graph, neighbors)


def is_almost_clique(graph: nx.Graph, vertexset: Set[int]) -> bool:
    """
    Check if a vertex set is almost a clique (becomes a clique if one vertex is removed).

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[int]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set is almost a clique, otherwise False.
    """
    for vertex in vertexset:
        if is_clique(graph, vertexset - {vertex}):
            return True
    return False


def clique_minimal_separator_decomposition(graph: nx.Graph) -> List[Set[int]]:
    """
    Decompose a graph into a set of atoms using clique minimal separators.

    Parameters:
    - graph (nx.Graph): The input graph.

    Returns:
    - List[Set[int]]: A list of atoms (maximal connected components) obtained after the decomposition.
    """
    atoms = []

    # Base case: if the graph is empty or a singleton, it's an atom
    if graph.number_of_nodes() <= 1:
        return [set(graph.nodes())]

    # Find a clique minimal separator using enumerate_all_cliques
    for vertexset in nx.enumerate_all_cliques(graph):
        if is_clique_minimal_separator(graph, set(vertexset)):
            break
    else:
        # If no clique minimal separator is found, the graph itself is an atom
        return [set(graph.nodes())]

    # Create a copy of the graph for finding connected components
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(vertexset)

    for component_nodes in nx.connected_components(graph_copy):
        # Union the component with the separator
        atom_nodes = set(component_nodes) | set(vertexset)

        # Create a subgraph induced by the atom nodes
        subgraph = graph.subgraph(atom_nodes).copy()

        # Recursively decompose the subgraphs
        atoms.extend(clique_minimal_separator_decomposition(subgraph))

    return atoms


def is_almost_simplicial(graph: nx.Graph, vertex: int) -> bool:
    """
    Check if a vertex is almost simplicial (its neighborhood is almost a clique).

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertex (int): The vertex to check.

    Returns:
    - bool: True if the vertex is almost simplicial, otherwise False.
    """
    neighbors = set(graph.neighbors(vertex))
    return is_almost_clique(graph, neighbors)


class TestReduction(unittest.TestCase):
    """
    Unit tests for the reduction module.
    """

    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])

    def test_generate_grid_graph(self):
        """
        Test the generation of a grid graph.
        """
        graph = generate_grid_graph(2, 2)
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 4)

    def test_is_clique(self):
        """
        Test the clique check function.
        """
        self.assertTrue(is_clique(self.graph, {1, 2, 3}))
        self.assertFalse(is_clique(self.graph, {1, 2, 4}))

    def test_is_separator(self):
        """
        Test the separator check function.
        """
        self.assertTrue(is_separator(self.graph, {3}))
        self.assertFalse(is_separator(self.graph, {1}))

    def test_is_minimal_separator(self):
        """
        Test the minimal separator check function.
        """
        circle = nx.Graph()
        circle.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        self.assertTrue(is_minimal_separator(circle, {1, 3}))
        circle.add_edge(2, 4)
        self.assertFalse(is_minimal_separator(circle, {1, 3}))

    def test_is_clique_minimal_separator(self):
        """
        Test the clique minimal separator check function.
        """
        circle = nx.Graph()
        circle.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        self.assertFalse(is_clique_minimal_separator(circle, {1, 3}))
        self.assertFalse(is_clique_minimal_separator(circle, {1}))
        circle.add_edge(1, 3)
        self.assertTrue(is_clique_minimal_separator(circle, {1, 3}))

    def test_is_simplicial(self):
        """
        Test the simplicial check function.
        """
        self.assertTrue(is_simplicial(self.graph, 1))
        self.assertFalse(is_simplicial(self.graph, 3))

    def test_is_almost_clique(self):
        """
        Test the almost clique check function.
        """
        # Create a new graph that is not almost a clique
        circle = nx.Graph()
        circle.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        self.assertFalse(is_almost_clique(circle, {1, 2, 3, 4}))
        circle.add_edge(1, 3)
        self.assertTrue(is_almost_clique(self.graph, {1, 2, 3, 4}))


if __name__ == "__main__":
    unittest.main()
