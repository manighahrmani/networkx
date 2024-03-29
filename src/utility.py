"""
Utility Module
--------------
This module contains utility functions for graph analysis.
It includes the following functions:
- save_grid_to_image
- write_graph_to_file
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

from typing import List, Union, Tuple, Dict, Optional
import os
import unittest
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore


def save_grid_to_image(
        num_rows: int,
        num_columns: int,
        grid: nx.Graph,
        path_to_graph_image: List[str],
        filename_end: str = "grid",
        node_colors: Optional[List[str]] = None,
) -> None:
    """
    Save the grid graph as an image.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - grid (nx.Graph): The grid graph.
    - path_to_graph_image (List[str]): The list of folders where the image will be saved.
    - filename_end (str): The name of the image file (default: "grid")
    - node_colors (Optional[List[str]]): The list of colors for the nodes defined in order of `.nodes()`.

    Returns:
    - None

    The function saves the grid graph as an image.
    The name of the image is in the format '{num_rows}x{num_columns}_{filename_end}.png'.
    The image is saved in the specified folders.
    """
    if node_colors is None:
        node_to_color: Dict[str, str] = {}
        for node in grid.nodes():
            node_to_color[node] = 'red'
        node_colors = [node_to_color[node] for node in grid.nodes()]

    for node in grid.nodes():
        if not isinstance(node, str):
            raise TypeError("Nodes must be strings")
        elif len(node) != 5 or node[0] != '1' or not node[1:].isnumeric():
            raise ValueError(
                "Nodes must be in the format '1nnmm'\
                      where n is the row number and m is the column number")

    # Create a dictionary of positions for the nodes
    pos: Dict[str, Tuple[int, int]] = {}
    for node in grid.nodes():
        row = int(node[1:3])
        column = int(node[3:5])
        pos[node] = (column - 1, -(row - 1))

    # Plot and save the original graph using the positions
    plt.figure(figsize=(8, 6))

    nx.draw(
        G=grid,
        pos=pos,
        with_labels=True,
        font_weight='bold',
        node_color=node_colors,
    )

    filename = f'{num_rows}x{num_columns}_{filename_end}.png'
    path = os.path.join(*path_to_graph_image, filename)
    plt.savefig(path)
    plt.close()


def write_graph_to_file(
        num_rows: int,
        num_columns: int,
        graph: nx.Graph,
        folders: List[str],
        filename: str = "",
) -> None:
    """
    Write the edges of the input graph to a text file.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - graph (nx.Graph): The graph with relabeled vertices.
    - folders (List[str]): The list of folders where the file will be saved.
    - filename (str): The name of the file (default: "")

    The function writes the edges of the relabeled graph to a text file.
    The file is saved in the specified folders.
    Each line in the file represents an edge and contains two vertex labels separated by a space.
    If no filename is provided, the filename is in the format '{num_rows}x{num_columns}.txt'.
    Else, the filename is the provided filename.
    """
    # Join the folders to form the path
    folder_path = os.path.join(*folders)

    # Use the number of rows and columns to form the filename if none is provided
    if filename == "":
        filename = f'{num_rows}x{num_columns}.txt'

    with open(
        os.path.join(folder_path, filename),
        mode='w',
        encoding='utf8'
    ) as f:
        for edge in graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")


def append_to_file(
        num_rows: int,
        num_columns: int,
        folders: List[str],
        content: str,
) -> None:
    """
    Append the content to the specified file.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - folders (List[str]): The list of folders where the file will be saved.
    - content (str): The content to append to the file.

    The function appends the content to the specified file.
    The file is saved in the specified folders.
    """
    # Join the folders to form the path
    folder_path = os.path.join(*folders)

    filename: str = f'{num_rows}x{num_columns}.txt'

    with open(
        os.path.join(folder_path, filename),
        mode='a',
        encoding='utf8'
    ) as f:
        f.write(content)


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
            self.assertEqual(
                get_missing_edges_in_neighborhood(graph, vertex), [])

        graph = nx.path_graph(5)
        self.assertEqual(get_missing_edges_in_neighborhood(graph, 0), [])
        self.assertEqual(get_missing_edges_in_neighborhood(graph, 4), [])
        self.assertEqual(get_missing_edges_in_neighborhood(graph, 2), [(1, 3)])


if __name__ == '__main__':
    unittest.main()
