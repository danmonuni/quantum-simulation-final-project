# graph drawing function

import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(G):
    
    '''function to draw the square 9 neel lattice must be generalised'''
    
    plt.figure(figsize=(8, 8))

    color_map = {tuple([0, 1]): "red", tuple([1, 0]): "blue"}
    node_colors = [color_map[tuple(G.nodes[node]["sublattice_encoding"])] for node in G.nodes]

    nx.draw(
        G,
        pos = nx.kamada_kawai_layout(G),
        with_labels=True,
        node_color=node_colors ,
        node_size=500,
        font_size=10,
        edge_color="gray"
    )

    plt.title("Lattice Graph")
    plt.show()

# graph printing function

def print_graph(G_pyg):
    print(G_pyg)
    print(G_pyg.x)
    print(G_pyg.edge_index)
    print(G_pyg.edge_attr)