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
import networkx as nx  # type: ignore


def generate_grid_graph(num_rows: int, num_columns: int) -> nx.Graph:
    """
    Generate a grid graph with custom vertex labels.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.

    Returns:
    - nx.Graph: The generated graph with custom vertex labels.

    The vertex labels are of the form 1rrcc, where rr is the row number and cc is the column number.
    """

    # Generate the original grid graph
    grid = nx.grid_2d_graph(num_rows, num_columns)

    # Generate a mapping from old labels (tuples) to new labels (strings).
    # Add a leading '1' to each label to avoid leading zeros.
    # The new labels are of the form 1rrcc, where rr is the row number and cc is the column number.
    mapping = {(r, c): f"1{r+1:02}{c+1:02}" for r in range(num_rows)
               for c in range(num_columns)}

    # Create a new graph with nodes relabeled
    relabeled_graph = nx.relabel_nodes(grid, mapping)

    return relabeled_graph


def is_clique(graph: nx.Graph, vertexset: Set[str]) -> bool:
    """
    Check if a set of vertices induces a complete subgraph (clique) in the graph.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[str]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set induces a complete subgraph, otherwise False.
    """
    missing_edges: Set[Tuple[str, str]] = get_missing_edges(graph, vertexset)
    return len(missing_edges) == 0


def is_separator(graph: nx.Graph, vertexset: Set[str]) -> bool:
    """
    Check if removing a set of vertices from the graph disconnects it.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[str]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set disconnects the graph, otherwise False.
    """
    graph_copy = graph.copy()
    graph_copy.remove_nodes_from(vertexset)
    number_of_connected_components = nx.number_connected_components(graph_copy)
    return number_of_connected_components > 1


def is_minimal_separator(graph: nx.Graph, vertexset: Set[str]) -> bool:
    """
    Check if a vertex set is a minimal separator in the graph.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[str]): The set of vertices to check.

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
        vertex: str
) -> Set[Tuple[str, str]]:
    """
    Get missing edges in the neighborhood of a vertex.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertex (str): The vertex whose neighborhood is considered.

    Returns:
    - Set[Tuple[str, str]]: A set of missing edges as tuples of two vertices.

    Description:
    Calculates the missing edges in the neighborhood of a vertex in the input graph.

    """
    neighbors: Set[str] = set(graph.neighbors(vertex))
    missing_edges: Set[Tuple[str, str]] = get_missing_edges(
        graph=graph,
        vertexset=neighbors
    )
    return missing_edges


def get_missing_edges(graph: nx.Graph, vertexset: Set[str]) -> Set[Tuple[str, str]]:
    """
    Get missing edges in a subgraph induced by a vertex set.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[str]): The set of vertices whose subgraph is considered.

    Returns:
    - Set[Tuple[str, str]]: A set of missing edges as tuples of two vertices.

    Description:
    Calculates the missing edges in the subgraph induced by a vertex set in the input graph.

    """
    missing_edges: Set[Tuple[str, str]] = set()
    subgraph: nx.Graph = graph.subgraph(vertexset)
    complement_subgraph: nx.Graph = nx.complement(subgraph)
    missing_edges.update(complement_subgraph.edges)
    return missing_edges


def is_clique_minimal_separator(graph: nx.Graph, vertexset: Set[str]) -> bool:
    """
    Check if a vertex set is both a minimal separator and a clique.

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[str]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set is a clique minimal separator, otherwise False.
    """
    return is_minimal_separator(graph, vertexset) and is_clique(graph, vertexset)


def is_simplicial(graph: nx.Graph, vertex: str) -> bool:
    """
    Check if a vertex is simplicial (its neighborhood induces a clique).

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertex (str): The vertex to check.

    Returns:
    - bool: True if the vertex is simplicial, otherwise False.
    """
    neighbors: Set[str] = set(graph.neighbors(vertex))
    return is_clique(graph, neighbors)


def is_almost_clique(graph: nx.Graph, vertexset: Set[str]) -> bool:
    """
    Check if a vertex set is almost a clique (becomes a clique if one vertex is removed).

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertexset (Set[str]): The set of vertices to check.

    Returns:
    - bool: True if the vertex set is almost a clique, otherwise False.
    """
    for vertex in vertexset:
        if is_clique(graph, vertexset - {vertex}):
            return True
    return False


def clique_minimal_separator_decomposition(graph: nx.Graph) -> List[Set[str]]:
    """
    Decompose a graph into a set of atoms using clique minimal separators.

    Parameters:
    - graph (nx.Graph): The input graph.

    Returns:
    - List[Set[str]]: A list of atoms (sets of vertices).
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
        nodes: Set[str] = set(graph.nodes())
        return [nodes]

    # Create a copy of the graph for finding connected components
    graph_copy: nx.Graph = graph.copy()
    graph_copy.remove_nodes_from(vertexset)

    for component_nodes in nx.connected_components(graph_copy):
        # Union the component with the separator
        atom_nodes: Set[str] = set(component_nodes) | set(vertexset)

        # Create a subgraph induced by the atom nodes
        subgraph: nx.Graph = graph.subgraph(atom_nodes).copy()

        # Recursively decompose the subgraphs
        atoms.extend(clique_minimal_separator_decomposition(subgraph))

    return atoms


def is_almost_simplicial(graph: nx.Graph, vertex: str) -> bool:
    """
    Check if a vertex is almost simplicial (its neighborhood is almost a clique).

    Parameters:
    - graph (nx.Graph): The input graph.
    - vertex (str): The vertex to check.

    Returns:
    - bool: True if the vertex is almost simplicial, otherwise False.
    """
    neighbors: Set[str] = set(graph.neighbors(vertex))
    return is_almost_clique(graph, neighbors)


def reduce_graph(
        graph: nx.Graph,
        debug: bool = False
) -> Tuple[Set[Tuple[str, str]], List[nx.Graph], List[str]]:
    """
    Reduce an input graph G to construct a minimum chordal triangulation.

    Parameters:
    - graph (nx.Graph): The input graph.
    - debug (bool): If True, print debug information.

    Returns:
    - Tuple[Set[Tuple[str, str]], List[nx.Graph], List[str]]:
        - Set of added edges F to make the graph chordal.
        - List of processed components (subgraphs).
        - List of vertices in the order they were eliminated (elimination ordering).

    """
    # Initialize
    atoms: nx.Graph = [graph.copy()]
    processed: List[nx.Graph] = []
    fill_edges: Set[Tuple[str, str]] = set()
    elimination_order: List[str] = []

    while atoms:
        atom = atoms.pop(0)

        # Sort vertices by degree (number of neighbors), ascending
        vertices_by_degree: List[str] = sorted(
            atom.nodes(), key=lambda v: len(set(atom.neighbors(v))))

        for v in vertices_by_degree:
            # for v in list(atom.nodes()):  # Convert to list for stable iteration
            reason_for_elimination: str = ""
            neighbours_of_v: Set[str] = set(atom.neighbors(v))

            degree_v: int = len(neighbours_of_v)

            for r in range(2, degree_v + 1):  # Subsets with at least 2 vertices
                for combination in combinations(neighbours_of_v, r):
                    vertexset: Set[str] = set(combination)
                    if is_clique(
                            graph=atom,
                            vertexset=vertexset
                    ):
                        continue
                    if not is_minimal_separator(
                        graph=atom,
                        vertexset=vertexset
                    ):
                        continue

                    missing_edges: Set[
                        Tuple[str, str]
                    ] = get_missing_edges(
                        graph=atom,
                        vertexset=vertexset
                    )
                    if len(missing_edges) == 1:
                        e: Tuple[str, str] = missing_edges.pop()
                        atom.add_edge(*e)
                        fill_edges.add(e)
                        if debug:
                            print(
                                f"\t➕ Added edge {e} in the neighborhood of {v}")

            if debug:
                print(
                    f"\tℹ️ {v} has degree {degree_v}\
                          and the vertex connectivity is {nx.node_connectivity(atom)}")

            if is_simplicial(atom, v):
                atom.remove_node(v)
                elimination_order.append(v)
                reason_for_elimination = "simplicial"
            elif is_almost_simplicial(atom, v) and degree_v == nx.node_connectivity(atom):
                missing_edges = get_missing_edges_in_neighborhood(
                    graph=atom,
                    vertex=v
                )
                for e in missing_edges:
                    atom.add_edge(*e)
                    fill_edges.add(e)
                atom.remove_node(v)
                elimination_order.append(v)
                reason_for_elimination = "almost simplicial"

            if reason_for_elimination and debug:
                print(f"❌ Eliminated vertex {v} ({reason_for_elimination})")

        atoms_vertex_set_i: List[
            Set[str]
        ] = clique_minimal_separator_decomposition(atom)
        atoms_i: List[nx.Graph] = []

        for atom_vertex_set_i in atoms_vertex_set_i:
            graph = atom.subgraph(atom_vertex_set_i).copy()
            atoms_i.append(graph)

        if len(atoms_i) == 1:
            processed.append(atoms_i[0])
        else:
            atoms.extend(atoms_i)

    # Eliminate atoms that have 1 or 0 vertices
    processed = [G for G in processed if G.number_of_nodes() > 1]

    return fill_edges, processed, elimination_order


def reduce_grid(
        graph: nx.Graph,
        debug: bool = False
) -> Tuple[Set[Tuple[str, str]], nx.Graph, List[str], nx.Graph]:
    """
    Reduces a grid graph, prints information about the reduction process,
    and saves visualizations of the reduced and processed graphs.

    Parameters:
    - num_rows (int): The number of rows in the original grid.
    - num_columns (int): The number of columns in the original grid.
    - graph (nx.Graph): The original graph to be reduced.
    - debug (bool): If True, print debug information.

    Returns:
    - Tuple[Set[Tuple[str, str]], nx.Graph, List[str], nx.Graph:
        - Set of added edges F to make the graph chordal.
        - The reduced graph,
        - List of vertices in the order they were eliminated (elimination ordering).
        - The processed graph (the original graph with added edges).
    """
    added_edges: Set[Tuple[str, str]] = set()
    processed_components: List[nx.Graph] = []
    ordering: List[str] = []
    added_edges, processed_components, ordering = reduce_graph(graph)

    if debug:
        print(f"{len(ordering)} vertices eliminated in the following order: {ordering}")
        print(f"{len(added_edges)} edges added: {added_edges}")
        print(f"{len(processed_components)} processed components: ")

    if len(processed_components) > 1:
        raise ValueError("More than one processed component")

    reduced_graph: nx.Graph = processed_components[0]

    processed_graph: nx.Graph = graph.copy()
    processed_graph.add_edges_from(added_edges)

    return added_edges, reduced_graph, ordering, processed_graph


class TestReduction(unittest.TestCase):
    """
    Unit tests for the reduction module.
    """

    def test_generate_grid_graph(self) -> None:
        """
        Test the generation of a grid graph.
        """
        graph = generate_grid_graph(2, 2)

        # Check if the nodes are labeled correctly
        desired_node_labels: Set[str] = {
            "10101", "10102", "10201", "10202"
        }
        self.assertEqual(set(graph.nodes), desired_node_labels)

        desired_edges: Set[Tuple[str, str]] = {
            ("10101", "10102"),
            ("10101", "10201"),
            ("10102", "10202"),
            ("10201", "10202")
        }
        self.assertEqual(set(graph.edges), desired_edges)

    def test_is_clique(self) -> None:
        """
        Test the clique check function.
        """
        graph: nx.Graph = generate_grid_graph(2, 2)
        vertices: Set[str] = {"10101", "10102", "10201"}
        self.assertFalse(is_clique(graph, vertices))
        graph.add_edge("10102", "10201")
        self.assertTrue(is_clique(graph, vertices))

    def test_is_separator(self) -> None:
        """
        Test the separator check function.
        """
        graph: nx.Graph = generate_grid_graph(2, 2)
        vertices: Set[str] = {"10102", "10201"}
        self.assertTrue(is_separator(graph, vertices))
        graph.add_edge("10101", "10202")
        self.assertFalse(is_separator(graph, vertices))

    def test_is_minimal_separator(self) -> None:
        """
        Test the minimal separator check function.
        """
        graph: nx.Graph = generate_grid_graph(3, 3)
        # Not a minimal separator
        vertices: Set[str] = {"10102", "10202", "10302", "10201"}
        self.assertFalse(is_minimal_separator(graph, vertices))
        vertices = {"10102", "10202", "10302"}
        self.assertTrue(is_minimal_separator(graph, vertices))
        # Not a separator
        vertices = {"10102", "10202"}
        self.assertFalse(is_minimal_separator(graph, vertices))

    def test_is_clique_minimal_separator(self) -> None:
        """
        Test the clique minimal separator check function.
        """
        graph: nx.Graph = generate_grid_graph(3, 3)
        # Not minimal and not a clique
        vertices: Set[str] = {"10102", "10202", "10302", "10201"}
        self.assertFalse(is_clique_minimal_separator(graph, vertices))
        # minimal separator but not a clique
        vertices = {"10102", "10202", "10302"}
        self.assertFalse(is_clique_minimal_separator(graph, vertices))
        # Add edges to make it a clique
        graph.add_edge("10102", "10302")
        self.assertTrue(is_clique_minimal_separator(graph, vertices))

    def test_is_simplicial(self) -> None:
        """
        Test the simplicial check function.
        """
        graph: nx.Graph = generate_grid_graph(3, 3)
        self.assertFalse(is_simplicial(graph, "10101"))
        graph.add_edge("10102", "10201")
        self.assertTrue(is_simplicial(graph, "10101"))

    def test_is_almost_clique(self) -> None:
        """
        Test the almost clique check function.
        """
        graph: nx.Graph = generate_grid_graph(3, 3)
        # Not an almost clique
        vertices: Set[str] = {"10101", "10202", "10303"}
        self.assertFalse(is_almost_clique(graph, vertices))
        graph.add_edge("10101", "10202")
        self.assertTrue(is_almost_clique(graph, vertices))

    def test_get_missing_edges_in_neighborhood(self) -> None:
        """
        Test the get_missing_edges_in_neighborhood function.
        """
        graph: nx.Graph = generate_grid_graph(3, 3)
        missing_edges = get_missing_edges_in_neighborhood(graph, "10101")
        self.assertEqual(missing_edges, {("10201", "10102")})
        graph.add_edge("10201", "10102")
        missing_edges = get_missing_edges_in_neighborhood(graph, "10101")
        self.assertEqual(missing_edges, set())

    def test_get_missing_edges(self) -> None:
        """
        Test the get_missing_edges function.
        """
        graph: nx.Graph = generate_grid_graph(2, 2)
        vertices: Set[str] = {"10101", "10102", "10201"}
        missing_edges = get_missing_edges(graph, vertices)
        self.assertEqual(missing_edges, {("10102", "10201")})
        graph.add_edge("10102", "10201")
        missing_edges = get_missing_edges(graph, vertices)
        self.assertEqual(missing_edges, set())


if __name__ == "__main__":
    unittest.main()
