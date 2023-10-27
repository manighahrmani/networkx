"""
Minimum Fill-In Module
"""

import csv
import subprocess
import os
from typing import List, Tuple, Set, Dict, Optional
import networkx as nx  # type: ignore
from config import SOLVER_PATH, ROWS, MAX_COLUMNS, CSV_FILENAME
from utility import write_graph_to_file, save_grid_to_image
from reduction import reduce_grid, generate_grid_graph


def run_solver(
        num_rows: int,
        num_columns: int
) -> List[Tuple[str, str]]:
    """
    Run an external solver to generate fill edges that triangulate the graph.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.

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
) -> Tuple[nx.Graph, List[Tuple[str, str]], nx.Graph]:
    """
    Generate a {num_rows}x{num_columns} grid graph and triangulates it.
    Given that `reduce` is True, reduces the grid graph before triangulation.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - reduce (bool): Whether to reduce the grid graph before triangulation.

    Returns:
    - Tuple[nx.Graph, List[Tuple[int, int]], nx.Graph, List[List[int]]]:
        * nx.Graph: The original grid graph.
        * List[Tuple[int, int]]: The fill edges added to triangulate the graph.
        * nx.Graph: The triangulated graph.
        * List[List[int]]: The maximum cliques in the triangulated graph.

    Raises:
    - RuntimeError: If the graph is not triangulated.
    - RuntimeError: If the fill-in does not match the expected formula.
    """

    os.makedirs(os.path.join('images', 'original'), exist_ok=True)
    os.makedirs(os.path.join('images', 'triangulated'), exist_ok=True)

    grid: nx.Graph = generate_grid_graph(num_rows, num_columns)

    # TODO: Let the caller do this
    # path_to_graph_image: List[str] = ["images", "original"]
    # # Get node positions for the original graph
    # # This is so that the vertices of grid graph and triangulated graph have the same positions
    # save_grid_to_image(num_rows, num_columns, grid, path_to_graph_image)

    chords: List[Tuple[str, str]] = []
    if reduce:
        chords_added_in_reduction: Set[Tuple[str, str]] = set()
        chords_added_in_reduction, _, _, _ = reduce_grid(
            graph=grid
        )

    chords += list(chords_added_in_reduction)

    # TODO: Let the caller do this
    # # Write the input graph to the solver folder and to the logs folder
    # write_graph_to_file(
    #     num_rows=num_rows,
    #     num_columns=num_columns,
    #     graph=reduced_grid,
    #     folders=[SOLVER_PATH],
    #     filename="graph"
    # )
    # write_graph_to_file(
    #     num_columns=num_columns,
    #     num_rows=num_rows,
    #     graph=reduced_grid,
    #     folders=["logs"],
    # )

    chords_after_reduction: List[
        Tuple[str, str]
    ] = run_solver(
        num_columns=num_columns,
        num_rows=num_rows
    )

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

    # TODO: Let the caller do this
    # # Find all cliques
    # cliques: List[List[int]] = list(nx.find_cliques(grid_triangulated))

    # # Find the maximum clique size
    # max_clique_size = max([len(clique) for clique in cliques])

    # # Find all cliques of the maximum size
    # maximum_cliques: List[List[int]] = [
    #     clique for clique in cliques if len(clique) == max_clique_size]

    # # Write the chords and maximum cliques to the text file in logs
    # with open(
    #     os.path.join("logs", f'{num_rows}x{num_columns}.txt'),
    #     mode='a',
    #     encoding='utf8'
    # ) as f:
    #     f.write("====================\n")
    #     for chord in chords:
    #         f.write(f"{chord[0]} {chord[1]}\n")
    #     f.write("====================\n")
    #     for clique in maximum_cliques:
    #         for node in clique:
    #             f.write(f"{node} ")
    #         f.write("\n")

    # TODO: Let the caller do this
    # path_to_graph_image = ["images", "triangulated"]
    # # Get node positions for the original graph
    # # This is so that the vertices of grid graph and triangulated graph have the same positions
    # save_grid_to_image(
    #     num_columns=num_columns,
    #     num_rows=num_rows,
    #     grid=grid_triangulated,
    #     path_to_graph_image=path_to_graph_image,
    #     filename_end="triangulated"
    # )

    return grid, chords, grid_triangulated


def compute_madj(
        vertex: str,
        ordering: List[str],
        graph: nx.Graph
) -> Set[str]:
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
        # Check if there's a path from later_vertex to vertex that only goes through vertices
        # earlier in the ordering than both vertex and later_vertex
        if nx.has_path(graph, later_vertex, vertex):
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


def run_experiments() -> None:
    """
    Run experiments to generate triangulated grid graphs and collect data.

    Parameters:
    - ROWS (int): The constant number of rows for the grid.
    - MAX_COLUMNS (int): The maximum number of columns to iterate through.

    Returns:
    - None

    This function iterates through a range of columns from `ROWS` to `MAX_COLUMNS`,
    generating triangulated grid graphs for each. It calculates the treewidth and
    number of added chords for each grid, saving these data points to a CSV file.
    Additionally, it writes the added chords and largest clique to a text file for each grid.
    """
    existing_data = {}

    # Read existing data from the CSV file if it exists
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, mode='r', newline='', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            _ = next(csv_reader)
            for row in csv_reader:
                existing_data[int(row[0])] = (int(row[2]), int(row[3]))

    for column in range(ROWS, MAX_COLUMNS + 1):
        print(f"Running experiment for {ROWS}x{column} grid...")

        # Generate the triangulated grid
        _, chords, triangulated_grid, maximum_cliques = generate_triangulated_grid_graph(
            num_rows=ROWS, num_columns=column, reduce=True)

        # Calculate the elimination ordering
        elimination_ordering = maximum_cardinality_search(triangulated_grid)

        with open(
            os.path.join("logs", f'{ROWS}x{column}.txt'),
            mode='a',
            encoding='utf8'
        ) as f:
            f.write("====================\n")
            for node in elimination_ordering:
                f.write(f"{node} ")
            f.write("\n")

        # Calculate the treewidth and number of added chords
        treewidth = len(maximum_cliques[0]) - 1
        num_added_chords = len(chords)

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
