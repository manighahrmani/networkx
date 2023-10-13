"""
Grid reduction module
"""

import os
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def generate_grid_graph(num_rows: int, num_columns: int) -> nx.Graph:
    """
    Generate a grid graph with custom vertex labels and save its edges to a text file.

    Parameters:
    - num_rows (int): The number of rows in the grid.
    - num_columns (int): The number of columns in the grid.

    Returns:
    - nx.Graph: The generated graph with custom vertex labels.
    """

    # Generate the original grid graph
    grid = nx.grid_2d_graph(num_rows, num_columns)

    # Generate a mapping from old labels (tuples) to new labels (strings).
    # Add a leading '1' to each label to avoid leading zeros.
    mapping = {(r, c): f"1{r+1:02}{c+1:02}" for r in range(num_rows)
               for c in range(num_columns)}

    # Create a new graph with nodes relabeled
    relabeled_graph = nx.relabel_nodes(grid, mapping)

    with open(
        os.path.join('reduction', 'logs', 'original',
                     f'{num_rows}x{num_columns}.txt'),
        mode='w',
        encoding='utf8'
    ) as f:
        for edge in relabeled_graph.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    pos = {node: (int(node[3:5]) - 1, -(int(node[1:3]) - 1))
           for node in relabeled_graph.nodes()}

    # Plot and save the original graph using the positions
    plt.figure(figsize=(8, 6))
    nx.draw(relabeled_graph, pos, with_labels=True, font_weight='bold')
    plt.savefig(os.path.join('reduction', 'images', 'original',
                f'{num_rows}x{num_columns}_grid.png'))
    plt.close()

    return relabeled_graph


generate_grid_graph(3, 3)
