"""
Elimination orderings module
"""

import csv
from typing import List, Set, Tuple, Dict
import networkx as nx  # type: ignore
from src.reduction import get_missing_edges_in_neighborhood


def is_valid_path(
        graph: nx.Graph,
        start: str,
        end: str,
        vertex_position: int,
        position_dict: Dict[str, int]
) -> bool:
    """
    Check if there's a valid path from start to end such that all intermediate vertices
    are earlier in the ordering than the specified vertex position.
    """
    visited: Set[str] = set()
    stack: List[str] = [start]

    while stack:
        current: str = stack.pop()
        if current == end:
            return True

        visited.add(current)
        for neighbor in graph.neighbors(current):
            # Only consider neighbors not visited and coming earlier in the ordering than 'end'
            if neighbor not in visited and \
                    (neighbor == end or position_dict[neighbor] < vertex_position):
                stack.append(neighbor)

    return False


def compute_madj(
        vertex: str,
        ordering: List[str],
        graph: nx.Graph,
        position_dict: Dict[str, int]
) -> Set[str]:
    """
    Compute the madj of a vertex based on the current ordering and graph.
    """
    vertex_position: int = position_dict[vertex]
    madj: Set[str] = set()

    # Check direct neighbors that are later in the ordering
    for neighbor in graph.neighbors(vertex):
        if position_dict[neighbor] > vertex_position:
            madj.add(neighbor)

    # Check for non-neighbors that are later in the ordering
    for later_vertex in ordering[vertex_position + 1:]:
        if later_vertex not in graph[vertex]:
            if is_valid_path(graph, later_vertex, vertex, vertex_position, position_dict):
                madj.add(later_vertex)

    return madj


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


ELIMINATION_ORDERINGS = {
    "5x5": "10101 10105 10501 10505 10102 10104 10201 10205 10401 10405 10502 10504 10404 10402 10303 10204 10503 10403 10305 10304 10302 10301 10203 10202 10103",
    "5x6": "10101 10106 10501 10506 10102 10105 10201 10206 10401 10406 10502 10505 10503 10404 10402 10305 10303 10204 10405 10306 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x7": "10101 10107 10501 10507 10102 10106 10201 10207 10401 10407 10502 10506 10505 10503 10406 10404 10402 10305 10303 10206 10204 10105 10307 10306 10405 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x8": "10101 10108 10501 10508 10102 10107 10201 10208 10401 10408 10502 10507 10505 10105 10503 10406 10404 10402 10307 10305 10303 10206 10204 10506 10407 10308 10207 10306 10405 10106 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x9": "10101 10109 10501 10509 10102 10108 10201 10209 10401 10409 10502 10508 10505 10105 10507 10503 10408 10406 10404 10402 10307 10305 10303 10208 10206 10204 10107 10309 10308 10407 10506 10207 10306 10405 10106 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x10": "10101 10110 10501 10510 10102 10109 10201 10210 10401 10410 10502 10509 10507 10505 10107 10105 10503 10408 10406 10404 10402 10309 10307 10305 10303 10208 10206 10204 10508 10409 10310 10209 10308 10407 10506 10108 10207 10306 10405 10106 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x11": "10101 10111 10501 10511 10102 10110 10201 10211 10401 10411 10502 10510 10507 10505 10107 10105 10509 10503 10410 10408 10406 10404 10402 10309 10307 10305 10303 10210 10208 10206 10204 10109 10311 10310 10409 10508 10209 10308 10407 10506 10108 10207 10306 10405 10106 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x12": "10101 10112 10501 10512 10102 10111 10201 10212 10401 10412 10502 10511 10509 10507 10505 10109 10107 10105 10503 10410 10408 10406 10404 10402 10311 10309 10307 10305 10303 10210 10208 10206 10204 10510 10411 10312 10211 10310 10409 10508 10110 10209 10308 10407 10506 10108 10207 10306 10405 10106 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x13":
    "10101 10113 10501 10513 10102 10112 10201 10213 10401 10413 10502 10512 10509 10507 10505 10109 10107 10105 10511 10503 10412 10410 10408 10406 10404 10402 10311 10309 10307 10305 10303 10212 10210 10208 10206 10204 10111 10313 10312 10411 10510 10211 10310 10409 10508 10110 10209 10308 10407 10506 10108 10207 10306 10405 10106 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103",
    "5x14": "10101 10114 10501 10514 10102 10113 10201 10214 10401 10414 10502 10513 10511 10509 10507 10505 10111 10109 10107 10105 10503 10412 10410 10408 10406 10404 10402 10313 10311 10309 10307 10305 10303 10212 10210 10208 10206 10204 10512 10413 10314 10213 10312 10411 10510 10112 10211 10310 10409 10508 10110 10209 10308 10407 10506 10108 10207 10306 10405 10106 10205 10304 10504 10403 10302 10301 10203 10202 10104 10103"
}


def get_madj_for_ordering(
    graph_size: str,
    elimination_ordering: str
) -> Tuple[List[Tuple[str, Set[str]]], nx.Graph]:
    """
    Computes the madj list for each vertex in the given elimination ordering.

    Args:
      graph_size (str): The size of the grid graph in the format "num_rows x num_columns".
      elimination_ordering (str): The elimination ordering as a
      space-separated string of vertex names.

    Returns:
        Tuple[List[Tuple[str, Set[str]]], nx.Graph]: The madj list for each vertex in the ordering
        and the original input graph.
    """
    # Parse the graph size
    num_rows, num_columns = map(int, graph_size.split('x'))

    # Generate the grid graph
    graph = generate_grid_graph(num_rows, num_columns)

    # Parse the elimination ordering
    ordering = elimination_ordering.split()

    # Create a dictionary mapping vertices to their positions in the ordering
    position_dict: Dict[str, int] = {v: i for i, v in enumerate(ordering)}

    # Compute madj for each vertex in the ordering
    madj_list = [(vertex, compute_madj(
        vertex=vertex,
        ordering=ordering,
        graph=graph,
        position_dict=position_dict
    ))
        for vertex in ordering]

    return madj_list, graph


def extend_madj_list(
    madj_list: List[Tuple[str, Set[str]]],
    graph: nx.Graph
) -> List[Tuple[str, Set[str], Set[Tuple[str, str]]]]:
    """
    Extend madj_list with edges between vertices in the madj of each vertex.

    Parameters:
    - madj_list (List[Tuple[str, Set[str]]]): List of vertices and their madj in order.
    - graph (nx.Graph): The original input graph.

    Returns:
    - List[Tuple[str, Set[str], Set[Tuple[str, str]]]]: Extended madj_list with edges in madj.
    """
    extended_madj_list: List[Tuple[str, Set[str], Set[Tuple[str, str]]]] = []
    madj_dict: Dict[str, Set[str]] = {
        vertex: madj for vertex, madj in madj_list
    }
    position_dict: Dict[str, int] = {
        vertex: index for index, (vertex, _) in enumerate(madj_list)
    }

    for vertex, madj in madj_list:
        edges_in_madj: Set[Tuple[str, str]] = set()

        for u in madj:
            for w in madj:
                if u != w:
                    # Check if (u, w) was already an edge in the input graph
                    if graph.has_edge(u, w):
                        edge: Tuple[str, str] = (u, w)
                        if w < u:
                            edge = (w, u)
                        edges_in_madj.add(edge)
                    else:
                        # Check if u and w were both in the madj at an earlier step
                        for earlier_vertex, earlier_madj in madj_dict.items():
                            if position_dict[earlier_vertex] < position_dict[vertex] and \
                                    u in earlier_madj and w in earlier_madj:
                                edge = (u, w)
                                if w < u:
                                    edge = (w, u)
                                edges_in_madj.add(edge)
                                break  # No need to check further once a match is found

        extended_madj_list.append((vertex, madj, edges_in_madj))

    return extended_madj_list


def extend_madj_list_with_graph_operations(
    madj_list: List[Tuple[str, Set[str]]],
    graph: nx.Graph
) -> List[Tuple[str, Set[str], Set[Tuple[str, str]], int]]:
    """
    Extend madj_list with edges between vertices in the madj of each vertex
    and add the vertex connectivity of the graph at each step.

    Parameters:
    - madj_list (List[Tuple[str, Set[str]]]): List of vertices and their madj in order.
    - graph (nx.Graph): The original input graph.

    Returns:
    - List[Tuple[str, Set[str], Set[Tuple[str, str]], int]]:
      Extended madj_list with edges in madj and vertex connectivity.
    """
    graph_copy = graph.copy()
    extended_madj_list: List[Tuple[str, Set[str],
                                   Set[Tuple[str, str]], int]] = []

    for vertex, madj in madj_list:
        # Get missing edges in the neighborhood and add them to the graph
        missing_edges = get_missing_edges_in_neighborhood(graph_copy, vertex)
        graph_copy.add_edges_from(missing_edges)

        # Remove the vertex from the graph
        graph_copy.remove_node(vertex)

        # Calculate vertex connectivity
        vertex_connectivity = nx.node_connectivity(graph_copy)

        # Store the information in the extended madj list
        extended_madj_list.append(
            (vertex, madj, missing_edges, vertex_connectivity))

    return extended_madj_list


def main() -> None:
    """
    Main function for testing.

    Args:
      None

    Returns:
      None
    """
    with open('output.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Graph Size", "Order", "Vertex", "Madj", "Edges in Madj", "Size of Madj", "Vertex Connectivity"])

        first = True
        for graph_size, elimination_ordering in ELIMINATION_ORDERINGS.items():
            if first:
                first = False

            print(f"Processing {graph_size}...")

            _, col = graph_size.split('x', maxsplit=1)

            if int(col) >= 9:
                break

            madj_list, graph = get_madj_for_ordering(
                graph_size=graph_size,
                elimination_ordering=elimination_ordering
            )

            extended_madj_list = extend_madj_list_with_graph_operations(
                madj_list=madj_list,
                graph=graph
            )
            for step, (vertex, madj, edges, vertex_connectivity) in enumerate(extended_madj_list):
                print(f"{step}: {vertex} {madj} {edges} {vertex_connectivity}")

            order = 1
            for vertex, madj, edges, vertex_connectivity in extended_madj_list:
                if len(madj) >= 2 and len(madj) <= 4:
                    writer.writerow(
                        [graph_size, order, vertex, madj, edges, len(madj), vertex_connectivity])
                    file.flush()  # Flush the file buffer
                else:
                    break
                order += 1


if __name__ == "__main__":
    main()
