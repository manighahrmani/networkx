"""
Minimum Fill-In Module
---------------------
This module contains functions for generating a grid graph,
running a minimum fill-in solver, and creating a triangulated graph based on
the solver's output.
It includes the following functions:
- generate_grid_graph
- run_solver
- generate_and_solve
"""

import csv
from typing import List, Tuple
import subprocess
import os
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from config import SOLVER_PATH, ROWS, MAX_COLUMNS


def generate_grid_graph(num_rows: int, num_columns: int) -> nx.Graph:
    """
    Generates a 2D grid graph with 'm' rows and 'n' columns.
    Writes the graph to a text file in a specific format.

    Parameters:
        num_rows (int): Number of rows.
        num_columns (int): Number of columns.

    Returns:
        nx.Graph: The generated NetworkX graph.
    """

    # Generate the grid graph
    graph = nx.grid_2d_graph(num_rows, num_columns)

    output_path = os.path.join(SOLVER_PATH, "graph.txt")

    # Open the file to write the edges
    with open(output_path, mode='w', encoding="utf-8") as file:
        for edge in graph.edges():
            vertex_u, vertex_v = (edge[0][0] + 1) * 10 + edge[0][1] + \
                1, (edge[1][0] + 1) * 10 + edge[1][1] + 1
            file.write(f"{vertex_u} {vertex_v}\n")

    return graph


def run_solver() -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Executes a batch file to run a minimum fill-in solver on Windows
    or a shell script on Unix-like systems.
    Reads the solver's output and returns the fill edges.

    Returns:
        List[Tuple[Tuple[int, int], Tuple[int, int]]]: List of fill edges.
    """

    # Determine the operating system
    os_type = os.name

    # Determine the script filename based on the operating system
    script_filename = "run_solver.bat" if os_type == "nt" else "run_solver.sh"

    # Execute the appropriate script
    subprocess.run(os.path.join(SOLVER_PATH, script_filename),
                   shell=True, cwd=SOLVER_PATH, check=True)

    # Read the output and count lines
    fill_edges = []
    with open(os.path.join(SOLVER_PATH, "output.txt"), mode="r", encoding="utf-8") as file:
        lines = file.readlines()

        for line in lines:
            vertex_1, vertex_2 = map(int, line.split())
            row_1, col_1 = divmod(vertex_1 - 11, 10)
            row_2, col_2 = divmod(vertex_2 - 11, 10)
            fill_edges.append(((row_1, col_1), (row_2, col_2)))

    return fill_edges


def generate_and_solve(
    num_rows: int,
    num_columns: int
) -> Tuple[nx.Graph, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """
    Calls `generate_grid_graph` to generate a graph and write it to a text file.
    Then, calls `run_solver` to find and return the fill edges for graph triangulation.

    Parameters:
        num_rows (int): Number of rows.
        num_columns (int): Number of columns.

    Returns:
        Tuple[nx.Graph, List[Tuple[int, int]]]: Original graph and list of fill edges.
    """

    # Generate grid graph and save to graph.txt
    graph: nx.Graph = generate_grid_graph(num_rows, num_columns)

    # Run the solver and get fill edges
    fill_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    fill_edges = run_solver()

    return graph, fill_edges


def generate_triangulated_grid_graph(
        num_rows: int,
        num_columns: int
) -> Tuple[
        nx.Graph,
        List[Tuple[Tuple[int, int], Tuple[int, int]]],
        nx.Graph,
        List[Tuple[int, int]]
]:
    """
    Generates a triangulated grid graph, plots and saves it.
    Highlights the largest clique in the triangulated graph.

    Parameters:
        num_rows (int): The number of rows in the grid graph.
        num_columns (int): The number of columns in the grid graph.

    Returns:
        grid (nx.Graph): The original grid graph.
        chords (List[Tuple[Tuple[int, int], Tuple[int, int]]]): The fill edges.
        graph_triangulated (nx.Graph): The triangulated graph.
        largest_clique (List[Tuple[int, int]]): The largest clique in the triangulated graph.

    Raises:
        RuntimeError: If the graph is not triangulated.
    """
    # Generate the grid graph and its fill edges
    grid: nx.Graph = None
    chords: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    grid, chords = generate_and_solve(
        num_rows=num_rows, num_columns=num_columns)

    # Get node positions for the original graph
    # You can use other layout functions if you prefer
    pos = nx.spring_layout(grid)

    # Plot and save the original graph using the positions
    plt.figure(figsize=(8, 6))
    nx.draw(grid, pos, with_labels=True, font_weight='bold')
    plt.savefig(f'{num_rows}x{num_columns}_grid.png')

    # Create the triangulated graph
    grid_triangulated: nx.Graph = grid.copy()
    grid_triangulated.add_edges_from(chords)

    # Check if the graph is truly chordal (triangulated)
    if not nx.is_chordal(grid_triangulated):
        raise RuntimeError("The graph is not triangulated!")

    # Find all cliques
    cliques: List[List[Tuple[int, int]]] = []
    cliques = list(nx.find_cliques(grid_triangulated))

    # Find the largest clique
    largest_clique = max(cliques, key=len)

    # Create a set of colors, using a different one for the largest clique
    node_colors = [
        'blue' if node in largest_clique else 'red' for node in grid_triangulated.nodes()]

    # Plot and save the triangulated graph using the same positions
    plt.figure(figsize=(8, 6))
    nx.draw(grid_triangulated, pos, with_labels=True,
            font_weight='bold', node_color=node_colors)
    plt.savefig(f'{num_rows}x{num_columns}_triangulated.png')

    return grid, chords, grid_triangulated, largest_clique


def run_experiments() -> None:
    """
    Runs the experiments for the report.
    """
    # Initialize the CSV file and write header
    with open('grid_data.csv', mode='a', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)

    for column in range(ROWS, MAX_COLUMNS + 1):
        _, chords, _, largest_clique = generate_triangulated_grid_graph(
            num_rows=ROWS, num_columns=column)

        # Calculate the treewidth and number of added chords
        treewidth = len(largest_clique) - 1
        num_added_chords = len(chords)

        # Append the data to the CSV file
        with open('grid_data.csv', mode='a', newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([column, ROWS, num_added_chords, treewidth])


run_experiments()
