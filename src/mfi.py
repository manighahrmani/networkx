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

from typing import List, Tuple
import subprocess
import os
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


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

    output_path = "C:\\Users\\manig\\Downloads\\PACE2017-min-fill\\graph.txt"

    # Open the file to write the edges
    with open(output_path, mode='w', encoding="utf-8") as file:
        for edge in graph.edges():
            vertex_u, vertex_v = (edge[0][0] + 1) * 10 + edge[0][1] + \
                1, (edge[1][0] + 1) * 10 + edge[1][1] + 1
            file.write(f"{vertex_u} {vertex_v}\n")

    return graph


def run_solver() -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Executes a batch file to run a minimum fill-in solver.
    Reads the solver's output and returns the fill edges.

    Returns:
        List[Tuple[Tuple[int, int], Tuple[int, int]]]: List of fill edges.
    """

    # Define the path where run_solver.bat and graph.txt are located
    solver_path = "C:\\Users\\manig\\Downloads\\PACE2017-min-fill"

    # Execute the batch file
    subprocess.run(os.path.join(solver_path, "run_solver.bat"),
                   shell=True, cwd=solver_path, check=True)

    # Read the output and count lines
    fill_edges = []
    with open(os.path.join(solver_path, "output.txt"), mode="r", encoding="utf-8") as file:
        lines = file.readlines()
        num_lines = len(lines)

        for line in lines:
            vertex_1, vertex_2 = map(int, line.split())
            row_1, col_1 = divmod(vertex_1 - 11, 10)
            row_2, col_2 = divmod(vertex_2 - 11, 10)
            fill_edges.append(((row_1, col_1), (row_2, col_2)))

    print(f"Number of added edges: {num_lines}")

    return fill_edges


def generate_and_solve(num_rows: int, num_columns: int) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
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
    graph = generate_grid_graph(num_rows, num_columns)

    # Run the solver and get fill edges
    fill_edges = run_solver()

    return graph, fill_edges


# Example usage
if __name__ == "__main__":
    rows, columns = 5, 5
    grid, chords = generate_and_solve(rows, columns)
    print("The edges of the original graph:", grid.edges())

    # Plot and save the original graph
    plt.figure(figsize=(8, 6))
    nx.draw(grid, with_labels=True, font_weight='bold')
    plt.savefig('3x4_grid.png')

    print("Fill edges:", chords)

    # Create the triangulated graph
    graph_triangulated = grid.copy()
    graph_triangulated.add_edges_from(chords)
    # Check if the graph is triangulated (chordal)
    print("Is the graph triangulated?", nx.is_chordal(graph_triangulated))

    # Plot and save the triangulated graph
    plt.figure(figsize=(8, 6))
    nx.draw(graph_triangulated, with_labels=True, font_weight='bold')
    plt.savefig('3x4_grid_triangulated.png')
