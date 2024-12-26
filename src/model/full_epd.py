import torch
import pytorch_geometric as pyg
from gn_decoder import GraphDecoder
from gn_core import GraphCore
from gn_decoder import GraphEncoder

class FullEPD(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_core_layers):
        super().__init__()

        self.num_core_layers = num_core_layers

        # Encoder
        self.encoder = GraphEncoder(input_dim, feature_dim)

        # Core (stack of GNCore layers)
        self.core = GraphCore(feature_dim, hidden_dim)

        # Decoder
        self.decoder = GraphDecoder(feature_dim, hidden_dim)

        #Pooling and Linear
        self.pool = pyg.nn.global_mean_pool
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        """
        Args:
            x: Input node features [N, F_node].
            edge_index: Graph connectivity [2, E].
            edge_attr: Input edge features [E, F_edge].
        Returns:
            Decoded node and edge features.
        """
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        edge_attr = edge_attr.type(torch.float)
        x = x.type(torch.float)

        # encode node and edge features
        x, edge_attr = self.encoder(x, edge_index, edge_attr, None, None)[0:2]

        # apply core layers
        for _ in range(self.num_core_layers - 1):
            x, edge_attr = self.core(x, edge_index, edge_attr, None, None)[0:2]

        # decode the final features
        x, edge_attr = self.decoder(x, edge_index, edge_attr, None, None)[0:2]

        graph = pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return graph