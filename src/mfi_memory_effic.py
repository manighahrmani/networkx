import itertools
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore


def minimum_triangulation_memory_efficient(graph):
    min_added_edges = float('inf')
    best_added_edges = None

    # Generate all possible elimination orderings
    all_orderings = list(itertools.permutations(graph.nodes()))

    for ordering in all_orderings:
        added_edges = set()
        graph_h = graph.copy()

        for node in ordering:
            neighbors = list(graph_h.neighbors(node))

            # Identify edges to make the neighborhood of the node a clique
            for i, vertex_u in enumerate(neighbors):
                for j, vertex_v in enumerate(neighbors):
                    if i < j and not graph_h.has_edge(vertex_u, vertex_v):
                        added_edges.add(
                            (min(vertex_u, vertex_v), max(vertex_u, vertex_v)))
                        graph_h.add_edge(vertex_u, vertex_v)

            # Remove the node
            graph_h.remove_node(node)

        if len(added_edges) < min_added_edges:
            min_added_edges = len(added_edges)
            best_added_edges = added_edges

    return best_added_edges, min_added_edges


if __name__ == "__main__":
    # G = nx.Graph()
    # G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    G = nx.grid_2d_graph(3, 4)

    plt.figure()
    nx.draw(G, with_labels=True, font_weight='bold',
            node_color='skyblue', font_size=18, node_size=700)
    plt.savefig("original_graph.png")

    chords, num_added_edges = minimum_triangulation_memory_efficient(G)
    print(f"Number of added edges: {num_added_edges}")

    # Add the edges to the original graph to make it triangulated
    G.add_edges_from(chords)

    print("Edges in the triangulated graph:", G.edges())

    plt.figure()
    nx.draw(G, with_labels=True, font_weight='bold',
            node_color='lightgreen', font_size=18, node_size=700)
    plt.savefig("triangulated_graph.png")
