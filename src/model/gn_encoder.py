import torch
import pytorch_geometric as pyg

# edge update model
class EdgeEncoderModel(torch.nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, feature_dim),
            )

    def forward(self, src, dst, edge_attr, u, batch):
        return self.edge_mlp(edge_attr)

# node update model
class NodeEncoderModel(torch.nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.node_mlp = torch.nn.Sequential(
          torch.nn.Linear(input_dim, 16),
          torch.nn.ReLU(),
          torch.nn.Linear(16, 32),
          torch.nn.ReLU(),
          torch.nn.Linear(32, 64),
          torch.nn.ReLU(),
          torch.nn.Linear(64, feature_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.node_mlp(x)


# metaLayer encoder
class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = pyg.nn.models.MetaLayer(
            EdgeEncoderModel(input_dim, feature_dim),
            NodeEncoderModel(input_dim, feature_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.encoder(x, edge_index, edge_attr)
    

