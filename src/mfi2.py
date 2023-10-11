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

    The edges of the graph are saved in a text file.
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


def generate_triangulated_grid_graph(num_rows: int, num_columns: int):
    """
    Generate a grid graph, triangulate it, and visualize the original and triangulated graphs.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.

    Returns:
    - Tuple[nx.Graph, List[Tuple[str, str]], nx.Graph, List[str]]:
        * The original grid graph.
        * The fill edges added to triangulate the graph.
        * The triangulated graph.
        * The largest clique in the triangulated graph.

    The function saves images of both the original and triangulated graphs.
    The vertices in the largest clique of the triangulated graph are colored differently.
    """

    os.makedirs(os.path.join('images', 'original'), exist_ok=True)
    os.makedirs(os.path.join('images', 'triangulated'), exist_ok=True)

    # Generate the grid graph (`generate_grid_graph`) and find the fill edges (`run_solver`)
    grid = generate_grid_graph(num_rows, num_columns)
    chords = run_solver()

    # Get node positions for the original graph
    # This is so that the vertices of grid graph and triangulated graph have the same positions
    pos = {node: (int(node[3:5]) - 1, -(int(node[1:3]) - 1))
           for node in grid.nodes()}

    # Plot and save the original graph using the positions
    plt.figure(figsize=(8, 6))
    nx.draw(grid, pos, with_labels=True, font_weight='bold')
    plt.savefig(os.path.join('images', 'original',
                f'{num_rows}x{num_columns}_grid.png'))

    # Create the triangulated graph
    grid_triangulated: nx.Graph = grid.copy()
    grid_triangulated.add_edges_from(chords)  # This should work

    # Check if the graph is truly chordal (triangulated)
    if not nx.is_chordal(grid_triangulated):
        raise RuntimeError("The graph is not triangulated!")

    # Find all cliques
    cliques: List[List[str]] = list(nx.find_cliques(grid_triangulated))

    # Find the largest clique
    largest_clique = max(cliques, key=len)

    # Create a set of colors, using a different one for the largest clique
    node_colors = [
        'blue' if node in largest_clique else 'red' for node in grid_triangulated.nodes()]

    # Plot and save the triangulated graph using the same positions
    plt.figure(figsize=(8, 6))
    nx.draw(grid_triangulated, pos, with_labels=True,
            font_weight='bold', node_color=node_colors)
    plt.savefig(os.path.join('images', 'triangulated',
                f'{num_rows}x{num_columns}_triangulated.png'))

    return grid, chords, grid_triangulated, largest_clique


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

    # Initialize the CSV file and write header
    with open('grid_data.csv', mode='w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['Columns', 'Rows', 'Num_Added_Chords', 'Treewidth'])

    for column in range(ROWS, MAX_COLUMNS + 1):
        print(f"Running experiment for {ROWS}x{column} grid...")

        # Generate the triangulated grid
        _, chords, _, largest_clique = generate_triangulated_grid_graph(
            num_rows=ROWS, num_columns=column)

        # Write the added chords and largest clique to a text file
        with open(os.path.join("logs", f"{ROWS}x{column}.txt"), mode='w', encoding="utf-8") as file:
            file.write("Added Chords:\n")
            for chord in chords:
                file.write(f"{chord[0]} {chord[1]}\n")
            file.write("=" * 20 + "\n")
            file.write("Largest Clique:\n")
            for node in largest_clique:
                file.write(f"{node}\n")

        # Calculate the treewidth and number of added chords
        treewidth = len(largest_clique) - 1
        num_added_chords = len(chords)

        # Append the data to the CSV file
        with open('grid_data.csv', mode='a', newline='', encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([column, ROWS, num_added_chords, treewidth])


run_experiments()
