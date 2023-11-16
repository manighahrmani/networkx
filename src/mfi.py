"""
Minimum Fill-In Module
"""

import csv
import subprocess
import os
from typing import List, Tuple, Set, Dict, Optional
import networkx as nx  # type: ignore
from config import SOLVER_PATH, ROWS, MAX_COLUMNS, CSV_FILENAME, MIN_COLUMNS
from utility import write_graph_to_file, save_grid_to_image, append_to_file
from reduction import reduce_grid, generate_grid_graph, get_missing_edges


def run_solver(
        num_rows: int,
        num_columns: int,
        run_with_cmd: bool = True
) -> List[Tuple[str, str]]:
    """
    Run an external solver to generate fill edges that triangulate the graph.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - run_with_cmd (bool): Whether to run the solver with the cmd command.

    Returns:
    - List[Tuple[str, str]]: A list of fill edges as tuples.

    This function runs an external solver script that reads the graph from a text file,
    triangulates the graph, and then writes the fill edges to an output text file.
    The function reads this output file and returns the fill edges as a list of tuples.
    """

    num_added_chords: Optional[int] = None
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, mode='r', newline='', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            _ = next(csv_reader)
            for row in csv_reader:
                if int(row[0]) == num_columns:
                    num_added_chords = int(row[2])
                    break

    # Prepare the command to run the solver
    os_type: str = os.name
    script_filename: str = "run_solver.bat" if os_type == "nt" else "run_solver.sh"
    cmd: str = os.path.join(SOLVER_PATH, script_filename)

    if run_with_cmd:
        # Add the parameters to the command
        if num_added_chords is not None:
            cmd += f' -k={num_added_chords}'
        cmd += ' -pmcprogress -info'

    print(f"For {num_rows}x{num_columns} grid, running command: {cmd}")

    # Run the solver
    # subprocess.run(cmd, shell=True, cwd=SOLVER_PATH, check=True)
    result: subprocess.CompletedProcess = subprocess.run(
        cmd,
        shell=True,
        cwd=SOLVER_PATH,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Display the stdout and stderr
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Read the output file to get the fill edges
    fill_edges: List[Tuple[str, str]] = []
    with open(os.path.join(SOLVER_PATH, "output.txt"), mode="r", encoding="utf-8") as file:
        lines: List[str] = file.readlines()
        for line in lines:
            # Remove any leading/trailing white spaces and split the vertices
            vertices: List[str] = line.strip().split(" ")

            # Add the edge as a tuple to the fill_edges list
            if len(vertices) == 2:
                edge: Tuple[str, str] = vertices[0], vertices[1]
                fill_edges.append(edge)

    return fill_edges


def generate_triangulated_grid_graph(
        num_rows: int,
        num_columns: int,
        reduce: bool = True,
) -> Tuple[nx.Graph, List[Tuple[str, str]], nx.Graph, List[str]]:
    """
    Generate a {num_rows}x{num_columns} grid graph and triangulates it.
    Given that `reduce` is True, reduces the grid graph before triangulation.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - reduce (bool): Whether to reduce the grid graph before triangulation.

    Returns:
    - Tuple[nx.Graph, List[Tuple[str, str]], nx.Graph, List[str]]:
        * nx.Graph: The original grid graph.
        * List[Tuple[str, str]]: The fill edges added to triangulate the graph.
        * nx.Graph: The triangulated graph.
        * List[str]: The elimination ordering of the vertices.

    Raises:
    - RuntimeError: If the graph is not triangulated.
    - RuntimeError: If the fill-in does not match the expected formula.
    """

    os.makedirs(os.path.join('images', 'original'), exist_ok=True)
    os.makedirs(os.path.join('images', 'triangulated'), exist_ok=True)

    grid: nx.Graph = generate_grid_graph(num_rows, num_columns)

    elimination_ordering: List[str] = []
    reduction_elimination: List[str] = []

    reduced_grid: nx.Graph = grid.copy()

    chords_added_in_reduction: Set[Tuple[str, str]] = set()
    chords: List[Tuple[str, str]] = []
    if reduce:
        chords_added_in_reduction, reduced_grid, reduction_elimination, _ = reduce_grid(
            graph=grid
        )

    # Add the chords added in reduction to the list of chords
    chords += list(chords_added_in_reduction)
    elimination_ordering += reduction_elimination

    # This must be here because the solver expects `reduced_graph` to be in the solver folder
    write_graph_to_file(
        num_rows=num_rows,
        num_columns=num_columns,
        graph=reduced_grid,
        folders=[SOLVER_PATH],
        filename="graph.txt"
    )

    chords_after_reduction: List[
        Tuple[str, str]
    ] = run_solver(
        num_columns=num_columns,
        num_rows=num_rows
    )
    triangulated_reduced_grid: nx.Graph = reduced_grid.copy()
    triangulated_reduced_grid.add_edges_from(chords_after_reduction)
    after_reduction_elimination: List[str] = maximum_cardinality_search(
        graph=triangulated_reduced_grid
    )

    elimination_ordering += after_reduction_elimination
    chords += chords_after_reduction

    # Create the triangulated graph
    grid_triangulated: nx.Graph = grid.copy()
    grid_triangulated.add_edges_from(chords)

    # # Check if the graph is truly chordal (triangulated)
    if not nx.is_chordal(grid_triangulated):
        raise RuntimeError("The graph is not triangulated!")

    # Check if the fill-in matches the expected formula
    if not check_fill_in(
        num_columns=num_columns,
        num_rows=num_rows,
        fill_in=len(chords)
    ):
        raise RuntimeError("The fill-in does not match the expected formula!")

    # if not check_elimination_ordering(
    #     graph=grid_triangulated,
    #     ordering=elimination_ordering
    # ):
    #     raise RuntimeError("The elimination ordering is not valid!")

    return grid, chords, grid_triangulated, elimination_ordering


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


def check_fill_in(num_rows: int, num_columns: int, fill_in: int) -> bool:
    """
    Check if the fill-in matches the expected formula based on the number of rows and columns.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - fill_in (int): The number of fill-in edges added to triangulate the graph.

    Returns:
    - bool: True if the fill-in matches the expected formula, False otherwise.

    The function checks the fill-in against the formula for mfi of a grid graph.
    The formula varies depending on the number of rows:
        * For a 3-row grid, the mfi is 5 + 4 * (n - 3) for n >= 3.
        * For a 4-row grid, the mfi is:
            * 18 + 8 * (n - 4) if n is even
            * 25 + 8 * (n - 5) if n is odd
    """
    if num_rows > 4 or num_rows < 3:
        return True
    elif num_rows == 3:
        # Check that the fill-in is 5 + 4 * (n - 3) for n >= 3
        return fill_in == 5 + 4 * (num_columns - 3)
    else:
        if num_columns % 2 == 0:
            return fill_in == 18 + 8 * (num_columns - 4)
        else:
            return fill_in == 25 + 8 * (num_columns - 5)


def maximum_cardinality_search(graph: nx.Graph) -> List[str]:
    """
    Perform a Maximum Cardinality Search (MCS) on a given graph to find an elimination ordering.

    Parameters:
    - graph (networkx.Graph): The input graph, assumed to be chordal.

    Returns:
    - List[str]: A list representing the elimination ordering of the vertices.

    Note:
    This function assumes that the input graph G is chordal. Using it on a non-chordal graph
    may not produce a valid elimination ordering.
    """

    # Initialize
    visited: Set[str] = set()
    label: Dict[str, int] = {}
    order: List[str] = []

    # Initialize all vertices with label 0
    for node in graph.nodes():
        label[node] = 0

    # Main loop to find the elimination ordering
    while len(visited) < len(graph):
        # Select a node with maximum label
        max_label_node: str = max((node for node in graph.nodes() if node not in visited),
                                  key=lambda node: label[node])

        visited.add(max_label_node)
        order.append(max_label_node)

        # Update labels of neighbors
        for neighbor in graph.neighbors(max_label_node):
            if neighbor not in visited:
                label[neighbor] += 1

    return order[::-1]  # Reverse to get elimination ordering


def check_elimination_ordering(
        graph: nx.Graph,
        ordering: List[str]
) -> bool:
    """
    Check if the given ordering is a valid elimination ordering of the given graph.

    Parameters:
    - graph (nx.Graph): The input graph.
    - ordering (List[str]): The ordering to check.

    Returns:
    - bool: True if the ordering is valid (triangulates the graph), False otherwise.

    """
    graph_copy: nx.Graph = graph.copy()
    for node in ordering:
        # Get the madj of the node
        madj: Set[str] = compute_madj(node, ordering, graph)

        madj_missing_edges: Set[Tuple[str, str]] = get_missing_edges(
            graph=graph,
            vertexset=madj
        )

        graph_copy.add_edges_from(madj_missing_edges)

    if not nx.is_chordal(graph_copy):
        return False
    return True


def get_all_maximum_cliques(graph: nx.Graph) -> List[List[str]]:
    """
    Get all maximum cliques of a given graph.

    Parameters:
    - graph (nx.Graph): The input graph.

    Returns:
    - List[List[str]]: A list of all maximum cliques of the graph.
    """
    # Find all cliques
    cliques: List[List[str]] = list(nx.find_cliques(graph))

    # Find the maximum clique size
    max_clique_size: int = max([len(clique) for clique in cliques])

    # Find all cliques of the maximum size
    maximum_cliques: List[List[str]] = [
        clique for clique in cliques if len(clique) == max_clique_size]

    return maximum_cliques


def run_experiments() -> None:
    """
    Run experiments to generate triangulated grid graphs and collect data.

    Parameters:
    - None

    Returns:
    - None

    # TODO: Update this docstring
    """
    existing_data: Dict[int, Tuple[int, int]] = {}

    # Read existing data from the CSV file if it exists
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, mode='r', newline='', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            _ = next(csv_reader)
            for row in csv_reader:
                existing_data[int(row[0])] = (int(row[2]), int(row[3]))

    for column in range(MIN_COLUMNS, MAX_COLUMNS + 1):
        print(f"Running experiment for {ROWS}x{column} grid...")

        # Generate the triangulated grid
        grid, chords, triangulated_grid, elimination_ordering = generate_triangulated_grid_graph(
            num_rows=ROWS, num_columns=column, reduce=True)

        # Write the input graph to the input/logs folder
        write_graph_to_file(
            num_columns=column,
            num_rows=ROWS,
            graph=grid,
            folders=["input", "logs"],
        )
        # Save the grid to an input/images folder
        save_grid_to_image(
            num_columns=column,
            num_rows=ROWS,
            grid=grid,
            path_to_graph_image=["input", "images"],
        )

        # First write the grid to the output/logs folder
        write_graph_to_file(
            num_columns=column,
            num_rows=ROWS,
            graph=triangulated_grid,
            folders=["output", "logs"],
        )
        # Then append the chords to the output/logs folder
        chords_str: str = "=" * 20 + "\n"
        chords_str += "\n".join(
            [f"{chord[0]} {chord[1]}" for chord in chords]) + "\n"
        append_to_file(
            content=chords_str,
            folders=["output", "logs"],
            num_columns=column,
            num_rows=ROWS,
        )
        # Then append the maximum cliques to the output/logs folder
        maximum_cliques: List[List[str]] = get_all_maximum_cliques(
            graph=triangulated_grid
        )
        maximum_cliques_str: str = "=" * 20 + "\n"
        for maximum_clique in maximum_cliques:
            maximum_cliques_str += " ".join(maximum_clique) + "\n"
        append_to_file(
            content=maximum_cliques_str,
            folders=["output", "logs"],
            num_columns=column,
            num_rows=ROWS,
        )
        # Lastly, append the elimination ordering to the output/logs folder
        elimination_ordering_str: str = "=" * 20 + "\n"
        elimination_ordering_str += " ".join(elimination_ordering) + "\n"
        append_to_file(
            content=elimination_ordering_str,
            folders=["output", "logs"],
            num_columns=column,
            num_rows=ROWS,
        )
        # Save the triangulated grid to an output/images folder
        save_grid_to_image(
            num_columns=column,
            num_rows=ROWS,
            grid=triangulated_grid,
            path_to_graph_image=["output", "images"],
        )

        # Calculate the treewidth and number of added chords
        treewidth: int = len(maximum_cliques[0]) - 1
        num_added_chords: int = len(chords)

        # Update the existing data if needed
        if column not in existing_data or existing_data[column] != (num_added_chords, treewidth):
            existing_data[column] = (num_added_chords, treewidth)

    # Write the updated data back to the CSV file
    with open(CSV_FILENAME, mode='w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['Columns', 'Rows', 'Num_Added_Chords', 'Treewidth'])
        for column, (num_added_chords, treewidth) in sorted(existing_data.items()):
            csv_writer.writerow([column, ROWS, num_added_chords, treewidth])


run_experiments()
