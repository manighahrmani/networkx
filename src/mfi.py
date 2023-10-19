"""
Minimum Fill-In Module
"""

import csv
import subprocess
import os
from typing import List, Tuple, Set, Dict, Any
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from config import SOLVER_PATH, ROWS, MAX_COLUMNS
from src.utility import write_input_graph_to_file, write_input_graph_to_solver_folder
from src.reduction import reduce_grid


def generate_grid_graph(num_rows: int, num_columns: int) -> nx.Graph:
    """
    Generate a grid graph with custom vertex labels.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.

    Returns:
    - nx.Graph: The generated graph with custom vertex labels.

    The vertices of the generated graph are labeled as strings, starting with '1' 
    followed by two digits for the row number and two digits for the column number.
    For example, the vertex in the first row and first column is labeled '10101'.
    """

    # Generate the original grid graph
    graph = nx.grid_2d_graph(num_rows, num_columns)

    # Generate a mapping from old labels (tuples) to new labels (strings).
    # Add a leading '1' to each label to avoid leading zeros.
    mapping = {(r, c): f"1{r+1:02}{c+1:02}" for r in range(num_rows)
               for c in range(num_columns)}

    # Create a new graph with nodes relabeled
    relabeled_graph = nx.relabel_nodes(graph, mapping)

    return relabeled_graph


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
    - List[Tuple[str, str]]: The list of fill edges added to triangulate the graph.

    This function runs an external solver script that reads the graph from a text file,
    triangulates the graph, and then writes the fill edges to an output text file.
    The function reads this output file and returns the fill edges as a list of tuples.
    """

    # Read the number of added edges from the CSV file if it exists
    csv_filename = f'{num_rows}_grid_data.csv'
    num_added_chords = None
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r', newline='', encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            _ = next(csv_reader)
            for row in csv_reader:
                if int(row[0]) == num_columns:
                    num_added_chords = int(row[2])
                    break

    # Prepare the command to run the solver
    os_type = os.name
    script_filename = "run_solver.bat" if os_type == "nt" else "run_solver.sh"
    cmd = os.path.join(SOLVER_PATH, script_filename)

    # Add the parameters to the command
    if num_added_chords is not None:
        cmd += f' -k={num_added_chords}'
    cmd += ' -pmcprogress -info'

    print(f"For {num_rows}x{num_columns} grid, running command: {cmd}")

    # Run the solver
    # subprocess.run(cmd, shell=True, cwd=SOLVER_PATH, check=True)
    result = subprocess.run(cmd, shell=True, cwd=SOLVER_PATH, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Display the stdout and stderr
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Read the output file to get the fill edges
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


def generate_triangulated_grid_graph(
        num_rows: int,
        num_columns: int,
        reduce: bool = True
) -> Tuple[nx.Graph, List[Tuple[str, str]], nx.Graph, List[List[str]]]:
    """
    Generate a grid graph, triangulate it, and visualize the original and triangulated graphs.
    Given that `reduce` is True, the function also reduces the grid graph before triangulating it.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - reduce (bool): Whether to reduce the grid graph before triangulating it.

    Returns:
    - Tuple[nx.Graph, List[Tuple[str, str]], nx.Graph, List[List[str]]]:
        * The original grid graph.
        * The fill edges added to triangulate the graph.
        * The triangulated graph.f
        * The list of all maximal cliques in the triangulated graph.

    The function saves images of both the original and triangulated graphs.
    The vertices in the maximal cliques are colored differently from the rest of the graph.

    Raises:
    - RuntimeError: If the graph is not triangulated.
    - RuntimeError: If the fill-in does not match the expected formula.
    """

    os.makedirs(os.path.join('images', 'original'), exist_ok=True)
    os.makedirs(os.path.join('images', 'triangulated'), exist_ok=True)

    grid = generate_grid_graph(num_rows, num_columns)

    chords: List[Tuple[str, str]] = []
    if reduce:
        chords, grid, elimination_ordering, grid_with_reduction_edges = reduce_grid(
            num_columns=num_columns,
            num_rows=num_rows,
            graph=grid
        )

    # Write the input graph to the solver folder and to the logs folder
    write_input_graph_to_solver_folder(grid)
    folder_name: str = "logs"
    write_input_graph_to_file(num_rows, num_columns,
                              grid, [folder_name])

    chords_after_reduction: List[Tuple[str, str]
                                 ] = run_solver(num_rows, num_columns)
    chords += chords_after_reduction

    path_to_graph_image: List[str] = ["images", "original"]
    # Get node positions for the original graph
    # This is so that the vertices of grid graph and triangulated graph have the same positions
    pos = save_grid_to_image(num_rows, num_columns, grid, path_to_graph_image)

    # Create the triangulated graph
    grid_triangulated: nx.Graph = grid.copy()
    grid_triangulated.add_edges_from(chords)  # This should work

    # Check if the graph is truly chordal (triangulated)
    if not nx.is_chordal(grid_triangulated):
        raise RuntimeError("The graph is not triangulated!")

    # Check if the fill-in matches the expected formula
    if not check_fill_in(num_rows, num_columns, len(chords)):
        raise RuntimeError("The fill-in does not match the expected formula!")

    # Find all cliques
    cliques: List[List[str]] = list(nx.find_cliques(grid_triangulated))

    # Find the maximum clique size
    max_clique_size = max([len(clique) for clique in cliques])

    # Find all cliques of the maximum size
    maximum_cliques: List[List[str]] = [
        clique for clique in cliques if len(clique) == max_clique_size]

    # Write the chords and maximum cliques to a text file
    with open(
        os.path.join("logs", f'{num_rows}x{num_columns}.txt'),
        mode='a',
        encoding='utf8'
    ) as f:
        f.write("====================\n")
        for chord in chords:
            f.write(f"{chord[0]} {chord[1]}\n")
        f.write("====================\n")
        for clique in maximum_cliques:
            for node in clique:
                f.write(f"{node} ")
            f.write("\n")

    # Create a dictionary to store color for each node
    node_color_dict = {}

    # Create a list of unique colors (one for each maximal clique)
    unique_colors = [
        'green', 'blue', 'purple', 'brown',
        'orange', 'pink', 'yellow', 'black',
        'cyan', 'magenta', 'lime', 'gray',
        'olive', 'maroon', 'navy', 'teal',
        'silver', 'white'
    ]

    if len(maximum_cliques) > len(unique_colors):
        # Reuse colors if there are not enough unique colors
        unique_colors = unique_colors * 10  # Duplicate the list 10 times

    # Assign a unique color to each maximal clique
    for i, max_clique in enumerate(maximum_cliques):
        for node in max_clique:
            node_color_dict[node] = unique_colors[i]

    # Generate list of node colors based on the populated node_color_dict
    node_colors = [node_color_dict.get(node, 'gray')
                   for node in grid_triangulated.nodes()]

    # Plot and save the triangulated graph using the same positions
    plt.figure(figsize=(8, 6))
    nx.draw(grid_triangulated, pos, with_labels=True,
            font_weight='bold', node_color=node_colors)
    plt.savefig(os.path.join('images', 'triangulated',
                f'{num_rows}x{num_columns}_triangulated.png'))
    plt.close()

    return grid, chords, grid_triangulated, maximum_cliques


def save_grid_to_image(num_rows: int, num_columns: int, grid: nx.Graph, path_to_graph_image: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Save the grid graph as an image.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.
    - grid (nx.Graph): The grid graph.
    - path_to_graph_image (List[str]): The list of folders where the image will be saved.

    Returns:
    - Dict[str, Tuple[int, int]]: The positions of the nodes in the grid.

    The function saves the grid graph as an image. The name of the image is in the format '{num_rows}x{num_columns}_grid.png'.
    The image is saved in the specified folders.
    """
    pos = {node: (int(node[3:5]) - 1, -(int(node[1:3]) - 1))
        for node in grid.nodes()}

    # Plot and save the original graph using the positions
    plt.figure(figsize=(8, 6))
    nx.draw(grid, pos, with_labels=True, font_weight='bold')
    path = os.path.join(*path_to_graph_image)
    filename = f'{num_rows}x{num_columns}_grid.png'
    plt.savefig(os.path.join(path, filename))
    plt.close()
    return pos


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


def maximum_cardinality_search(graph: nx.Graph) -> List[Any]:
    """
    Perform a Maximum Cardinality Search (MCS) on a given graph to find an elimination ordering.

    Parameters:
    - graph (networkx.Graph): The input graph, assumed to be chordal.

    Returns:
    - List[Any]: A list representing the elimination ordering of the vertices.

    Note:
    This function assumes that the input graph G is chordal. Using it on a non-chordal graph
    may not produce a valid elimination ordering.
    """

    # Initialize
    visited: Set[Any] = set()
    label: Dict[Any, int] = {}
    order: List[Any] = []

    # Initialize all vertices with label 0
    for node in graph.nodes():
        label[node] = 0

    # Main loop to find the elimination ordering
    while len(visited) < len(graph):
        # Select a node with maximum label
        max_label_node = max((node for node in graph.nodes() if node not in visited),
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

    csv_filename = f'{ROWS}_grid_data.csv'
    existing_data = {}

    # Read existing data from the CSV file if it exists
    if os.path.exists(csv_filename):
        with open(csv_filename, mode='r', newline='', encoding="utf-8") as csvfile:
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
    with open(csv_filename, mode='w', newline='', encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ['Columns', 'Rows', 'Num_Added_Chords', 'Treewidth'])
        for column, (num_added_chords, treewidth) in sorted(existing_data.items()):
            csv_writer.writerow([column, ROWS, num_added_chords, treewidth])


run_experiments()
