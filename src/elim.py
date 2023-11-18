"""
Elimination orderings module
"""

import csv
from typing import List, Set, Tuple
import networkx as nx  # type: ignore


def compute_madj(vertex: str, ordering: List[str], graph: nx.Graph) -> Set[str]:
    """
    Compute the madj of a vertex based on the current ordering and graph.

    Parameters:
    - vertex (str): The vertex whose madj is to be computed.
    - ordering (List[str]): The current ordering of the vertices.
    - graph (nx.Graph): The graph.

    Returns:
    - Set[str]: The madj of the vertex.
    """
    # Get the position of the vertex in the ordering
    position: int = ordering.index(vertex)

    # Initialize madj
    madj: Set[str] = set()

    # Iterate over all vertices that come after the current vertex in the ordering
    for later_vertex in ordering[position+1:]:
        # Check for a path where all intermediate vertices are earlier in the ordering
        path_exists = False
        for path in nx.all_simple_paths(graph, later_vertex, vertex):
            if all(ordering.index(p) < position for p in path[1:-1]):
                path_exists = True
                break

        if path_exists:
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
) -> List[Tuple[str, Set[str]]]:
    """
    Computes the madj list for each vertex in the given elimination ordering.

    Args:
      graph_size (str): The size of the grid graph in the format "num_rows x num_columns".
      elimination_ordering (str): The elimination ordering as a
      space-separated string of vertex names.

    Returns:
      List[Tuple[str, Set[str]]]: A list of tuples,
      where each tuple contains a vertex name and its corresponding madj set.
    """
    # Parse the graph size
    num_rows, num_columns = map(int, graph_size.split('x'))

    # Generate the grid graph
    graph = generate_grid_graph(num_rows, num_columns)

    # Parse the elimination ordering
    ordering = elimination_ordering.split()

    # Compute madj for each vertex in the ordering
    madj_list = [(vertex, compute_madj(vertex, ordering, graph))
                 for vertex in ordering]

    return madj_list


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
            ["Graph Size", "Order", "Vertex", "Madj", "Size of Madj"])

        first = True
        for graph_size, elimination_ordering in ELIMINATION_ORDERINGS.items():
            if not first:
                writer.writerow([])  # Add a blank line between sections
            else:
                first = False

            print(f"Processing {graph_size}...")

            row, col = graph_size.split('x', maxsplit=1)

            if int(col) <= 9:
                continue

            number_of_vertices = int(row) * int(col)
            madj_list = get_madj_for_ordering(graph_size, elimination_ordering)

            order = 1
            for vertex, madj in madj_list:
                if len(madj) >= 3 and order <= number_of_vertices - 5:
                    writer.writerow(
                        [graph_size, order, vertex, madj, len(madj)])
                    file.flush()  # Flush the file buffer
                order += 1


if __name__ == "__main__":
    main()
