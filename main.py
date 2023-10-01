import unittest
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore


def save_grid_graph(m, n, filename):
    G = nx.grid_2d_graph(m, n)
    pos = {(x, y): (y, -x) for x, y in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            font_weight='bold', node_size=700, font_size=18)
    plt.savefig(filename)


def get_vertex_connectivity(G):
    return nx.node_connectivity(G)


def is_clique(G, vertex_set):
    all_cliques = list(nx.find_cliques(G.subgraph(vertex_set)))
    return len(all_cliques) == 1 and set(all_cliques[0]) == set(vertex_set)


def is_separator(G, vertex_set):
    original_components = nx.number_connected_components(G)
    G_removed = G.copy()
    G_removed.remove_nodes_from(vertex_set)
    new_components = nx.number_connected_components(G_removed)
    return new_components > original_components


class TestGraphFunctions(unittest.TestCase):

    def test_get_vertex_connectivity(self):
        G = nx.complete_graph(5)
        self.assertEqual(get_vertex_connectivity(G), 4)

        G = nx.path_graph(5)
        self.assertEqual(get_vertex_connectivity(G), 1)

    def test_is_clique(self):
        G = nx.complete_graph(5)
        vertex_set = [0, 1, 2, 3]
        self.assertTrue(is_clique(G, vertex_set))

        vertex_set = [0, 1, 4]
        self.assertTrue(is_clique(G, vertex_set))

        vertex_set = [0, 2, 4]
        self.assertTrue(is_clique(G, vertex_set))  # Corrected to True

    def test_is_separator(self):
        G = nx.grid_2d_graph(3, 3)

        # Not a separator
        vertex_set = [(1, 1)]
        self.assertFalse(is_separator(G, vertex_set))

        # Not a separator
        vertex_set = [(0, 0)]
        self.assertFalse(is_separator(G, vertex_set))

        # Should be a separator: removing an entire row
        vertex_set = [(1, 0), (1, 1), (1, 2)]
        self.assertTrue(is_separator(G, vertex_set))


if __name__ == '__main__':
    unittest.main()
