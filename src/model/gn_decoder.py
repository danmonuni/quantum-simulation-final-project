import torch
import pytorch_geometric as pyg

# Edge update model
class EdgeDecoderModel(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
            )

    def forward(self, src, dst, edge_attr, u, batch):
        return self.edge_mlp(edge_attr)

# Node update model
class NodeDecoderModel(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.node_mlp = torch.nn.Sequential(
          torch.nn.Linear(feature_dim, hidden_dim),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_dim, hidden_dim),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_dim, hidden_dim),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.node_mlp(x)



# MetaLayer decoder
class GraphDecoder(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.decoder = pyg.nn.models.MetaLayer(
            EdgeDecoderModel(feature_dim, hidden_dim),
            NodeDecoderModel(feature_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.decoder( x, edge_index, edge_attr)