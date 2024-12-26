# a set of functions to generate the needed graphs

import networkx as nx
import numpy as np
import pytorch_geometric as pyg

def square_9():
    #generate the lattice and equip it with a sublattice encoding (gotta write it custom)
    #square N = 36
    #encoding = Neel

    G = nx.grid_2d_graph(3, 3, periodic=True)

    for edge in G.edges:
        G.edges[edge]["edge_init"] = np.array([0,0,0])

    for nodes in G.nodes:
        G.nodes[nodes]["spin_init"] = 0

        if (nodes[0]+ nodes[1]+2) % 2 == 0:
            G.nodes[nodes]["sublattice_encoding"] = np.array([1,0])

        else:
            G.nodes[nodes]["sublattice_encoding"] = np.array([0,1])
    

    G_pyg = pyg.utils.from_networkx(G, "all", "all")

    return G, G_pyg 
