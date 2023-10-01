import matplotlib.pyplot as plt
import networkx as nx

def save_grid_graph(m, n, filename):
    G = nx.grid_2d_graph(m, n)
    pos = {(x, y): (y, -x) for x, y in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=700, font_size=18)
    plt.savefig(filename)

# Example usage
save_grid_graph(5, 5, "5x5_grid.png")
