import torch
import pytorch_geometric as pyg


# edge update model
class EdgeCoreModel(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(3 * feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, src, dst, edge_attr, u, batch):
        """
        Forward pass for the edge update model.
        Args:
            src (Tensor): Features of source nodes [E, F_x].
            dst (Tensor): Features of target nodes [E, F_x].
            edge_attr (Tensor): Current edge features [E, F_e].
        Returns:
            Tensor: Updated edge features [E, F_out].
        """
        # Concatenate inputs: src, dst, edge_attr, and global features (mapped to edges)
        edge_inputs = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_mlp(edge_inputs)

# node update model
class NodeCoreModel(torch.nn.Module):
    def __init__(self, feature_dimension, hidden_dim):
        super().__init__()
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * feature_dimension, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dimension),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        Forward pass for the node update model.
        Args:
            x (Tensor): Current node features [N, F_x].
            edge_index (Tensor): Graph connectivity [2, E].
            edge_attr (Tensor): Updated edge features [E, F_e].
        Returns:
            Tensor: Updated node features [N, F_out].
        """
        # Aggregate edge features for each node
        row, col = edge_index  # row: source nodes, col: target nodes
        aggregated_edges = pyg.utils.scatter(edge_attr, col, dim=0, reduce="mean", dim_size=x.size(0))

        # Concatenate current node features, aggregated edge features, and global features
        node_inputs = torch.cat([x, aggregated_edges], dim=1)
        return self.node_mlp(node_inputs)



# metalayer encoder
class GraphCore(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.encoder = pyg.nn.models.MetaLayer(
            EdgeCoreModel(feature_dim, hidden_dim),
            NodeCoreModel(feature_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        return  x + self.encoder(x, edge_index, edge_attr)[0], edge_attr + self.encoder(x, edge_index, edge_attr)[1]