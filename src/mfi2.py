"""
Minimum Fill-In Module
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
    Generate a grid graph with custom vertex labels and save its edges to a text file.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.

    Returns:
    - nx.Graph: The generated graph with custom vertex labels.

    The vertices of the generated graph are labeled as strings, starting with '1' 
    followed by two digits for the row number and two digits for the column number.
    For example, the vertex in the first row and first column is labeled '10101'.

    The edges of the graph are saved in a text file, and an image of the grid is also saved.
    """

    # Generate the original grid graph
    graph = nx.grid_2d_graph(num_rows, num_columns)

    # Generate a mapping from old labels (tuples) to new labels (strings).
    # Add a leading '1' to each label to avoid leading zeros.
    mapping = {(r, c): f"1{r+1:02}{c+1:02}" for r in range(num_rows)
               for c in range(num_columns)}

    # Create a new graph with nodes relabeled
    relabeled_graph = nx.relabel_nodes(graph, mapping)

    # Generate the output path and write the edges to the file
    output_path = os.path.join(SOLVER_PATH, "graph.txt")
    with open(output_path, mode='w', encoding='utf8') as f:
        for edge in relabeled_graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")
    # Also save it to the logs folder
    with open(
        os.path.join("logs", f'{num_rows}x{num_columns}.txt'),
        mode='w',
        encoding='utf8'
    ) as f:
        for edge in relabeled_graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    # Generate the positions for the nodes.
    # In this case, we can simply reverse the labels back to tuples for positioning.
    # Reversed to match typical Cartesian coordinates
    pos = {mapping[key]: (key[1], -key[0]) for key in mapping.keys()}

    # Plot and save the original graph using the positions
    plt.figure(figsize=(8, 6))
    nx.draw(relabeled_graph, pos, with_labels=True, font_weight='bold')
    plt.savefig(f'{num_rows}x{num_columns}_grid.png')

    return relabeled_graph


def run_solver():
    """
    Run an external solver to generate fill edges that triangulate the graph.

    Returns:
    - List[Tuple[str, str]]: The list of fill edges added to triangulate the graph.

    This function runs an external solver script that reads the graph from a text file,
    triangulates the graph, and then writes the fill edges to an output text file.
    The function reads this output file and returns the fill edges as a list of tuples.
    """

    os_type = os.name
    script_filename = "run_solver.bat" if os_type == "nt" else "run_solver.sh"

    subprocess.run(os.path.join(SOLVER_PATH, script_filename),
                   shell=True, cwd=SOLVER_PATH, check=True)

    fill_edges = []
    with open(os.path.join(SOLVER_PATH, "output.txt"), mode="r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            # Remove any leading/trailing white spaces and split the vertices
            vertices = line.strip().split(" ")

            # Add the edge as a tuple to the fill_edges list
            if len(vertices) == 2:
                fill_edges.append((vertices[0], vertices[1]))

    return fill_edges


edges = run_solver()
print(edges)
