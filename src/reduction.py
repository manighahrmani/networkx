"""
Grid reduction module

Includes implementations of the following functions:
- generate_grid_graph(num_rows: int, num_columns: int) -> nx.Graph
- is_clique(graph: nx.Graph, vertexset: Set[int]) -> bool
- is_separator(graph: nx.Graph, vertexset: Set[int]) -> bool
- is_minimal_separator(graph: nx.Graph, vertexset: Set[int]) -> bool
- get_missing_edges_in_neighborhood(graph: nx.Graph, vertex: int) -> Set[Tuple[int, int]]
- get_missing_edges(graph: nx.Graph, vertexset: Set[int]) -> Set[Tuple[int, int]]
- is_clique_minimal_separator(graph: nx.Graph, vertexset: Set[int]) -> bool
- is_simplicial(graph: nx.Graph, vertex: int) -> bool
- is_almost_clique(graph: nx.Graph, vertexset: Set[int]) -> bool
- clique_minimal_separator_decomposition(graph: nx.Graph) -> List[Set[int]]
- is_almost_simplicial(graph: nx.Graph, vertex: int) -> bool
"""

from typing import List, Set, Tuple
from itertools import combinations
import unittest
import os
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
    missing_edges: Set[Tuple[int, int]] = get_missing_edges(graph, vertexset)
    return len(missing_edges) == 0


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


def get_missing_edges_in_neighborhood(
        graph: nx.Graph,
        vertex: int
) -> Set[Tuple[int, int]]:
    """
    Get missing edges in the neighborhood of a vertex.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertex (int): The vertex whose neighborhood is considered.

    Returns:
    - Set[Tuple[int, int]]: A set of missing edges as tuples of two vertices.

    Description:
    Calculates the missing edges in the neighborhood of a vertex in the input graph.

    """
    neighbors: Set[int] = set(graph.neighbors(vertex))
    missing_edges: Set[Tuple[int, int]] = get_missing_edges(graph, neighbors)
    return missing_edges


def get_missing_edges(graph: nx.Graph, vertexset: Set[int]) -> Set[Tuple[int, int]]:
    """
    Get missing edges in a subgraph induced by a vertex set.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[int]): The set of vertices for which missing edges are to be calculated.

    Returns:
    - Set[Tuple[int, int]]: A set of missing edges as tuples of two vertices.

    Description:
    Calculates the missing edges in the subgraph induced by a vertex set in the input graph.

    """
    missing_edges: Set[Tuple[int, int]] = set()
    subgraph: nx.Graph = graph.subgraph(vertexset)
    complement_subgraph: nx.Graph = nx.complement(subgraph)
    missing_edges.update(complement_subgraph.edges)
    return missing_edges


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


def reduce_graph(G: nx.Graph) -> Tuple[Set[Tuple[int, int]], List[nx.Graph]]:
    """
    Reduce an input graph G to construct a minimum chordal triangulation.

    Parameters:
    - G (nx.Graph): The input graph.

    Returns:
    - Tuple[Set[Tuple[int, int]], List[nx.Graph]]:
        - Set of added edges F to make the graph chordal.
        - List of processed connected components (subgraphs).

    """
    # Initialize
    atoms = [G.copy()]  # Step 1: Let As = [G]
    processed = []  # Step 2: Let processed = []
    fill_edges = set()  # Step 3: Let F = []

    # Step 4: While As has at least 1 element
    while atoms:
        atom = atoms.pop(0)  # Step 4.1: Let A_i be an element from As

        # Step 4.2: For every vertex v in A_i
        for v in atom.nodes():
            # Neighborhood of v in A_i
            neighbours_of_v = set(atom.neighbors(v))

            # Step 4.2.1: For every non-clique vertex set S in N(v)
            for r in range(2, len(neighbours_of_v) + 1):  # Subsets with at least 2 vertices
                for combination in combinations(neighbours_of_v, r):
                    vertex_set: Set[int] = set(combination)
                    if not is_clique(atom, vertex_set):
                        if is_separator(atom, vertex_set):
                            missing_edges = get_missing_edges(atom, vertex_set)
                            if len(missing_edges) == 1:
                                # Add the missing edge to A_i and F
                                e = missing_edges.pop()
                                atom.add_edge(*e)
                                fill_edges.add(e)

        # Step 4.3: For every vertex v in A_i
        for v in list(atom.nodes()):  # Convert to list for stable iteration
            # Neighborhood of v in A_i
            neighbours_of_v = set(atom.neighbors(v))

            if is_simplicial(atom, v):
                # Step 4.3.1: If v is simplicial, eliminate it
                atom.remove_node(v)
            elif is_almost_simplicial(atom, v) and len(neighbours_of_v) == nx.node_connectivity(atom):
                # Step 4.3.2: Else if v is almost simplicial and size of N(v) == vertex connectivity of A_i
                missing_edges = get_missing_edges_in_neighborhood(atom, v)
                for e in missing_edges:
                    atom.add_edge(*e)
                    fill_edges.add(e)

                # Then remove v from A_i
                atom.remove_node(v)

        # Step 4.4: Let L_i be the list of vertex sets returned by clique_minimal_separator_decomposition with A_i
        L_i = clique_minimal_separator_decomposition(atom)

        # Step 4.5: Let G_i = []
        G_i = []

        # Step 4.6: For each vertex set L in L_i
        for L in L_i:
            # Let G be the graph induced on L
            G = atom.subgraph(L).copy()
            G_i.append(G)

        # Step 4.7: If G_i has one element, add it to processed
        if len(G_i) == 1:
            processed.append(G_i[0])
        else:
            # Otherwise, add all the elements of G_i to As
            atoms.extend(G_i)

    # Step 5: Return F and processed
    return fill_edges, processed


# Example usage
grid_graph = generate_grid_graph(3, 3)
added_edges, processed_components = reduce_graph(grid_graph)

# Show added edges
print("Added edges:", added_edges)

# Show processed graphs
for i, processed_component in enumerate(processed_components):
    print(
        f"Processed graph {i + 1}: Nodes = {list(processed_component.nodes)}, Edges = {list(processed_component.edges)}")
print(
    f"Total processed graphs: {len(processed_components)}, Total added edges: {len(added_edges)}")
# Plot the processed graphs
for i, processed_component in enumerate(processed_components):
    pos = {node: (int(node[3:5]) - 1, -(int(node[1:3]) - 1))
           for node in processed_component.nodes()}
    plt.figure(figsize=(8, 6))
    nx.draw(processed_component, pos, with_labels=True, font_weight='bold')
    plt.savefig(os.path.join('reduction', 'images', 'processed',
                f'{i + 1}_processed_graph.png'))
    plt.close()


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

    def test_get_missing_edges_in_neighborhood(self):
        """
        Test the get_missing_edges_in_neighborhood function.
        """
        circle = nx.Graph()
        circle.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        missing_edges = get_missing_edges_in_neighborhood(circle, 1)
        self.assertEqual(missing_edges, {(2, 4)})

        circle.add_edge(1, 3)
        missing_edges = get_missing_edges_in_neighborhood(circle, 2)
        self.assertEqual(missing_edges, set())

    def test_get_missing_edges(self):
        """
        Test the get_missing_edges function.
        """
        circle = nx.Graph()
        circle.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        missing_edges = get_missing_edges(circle, {1, 2, 3})
        self.assertEqual(missing_edges, {(1, 3)})
        missing_edges = get_missing_edges(circle, {1, 2, 3, 4})
        self.assertEqual(missing_edges, {(1, 3), (2, 4)})


if __name__ == "__main__":
    unittest.main()
