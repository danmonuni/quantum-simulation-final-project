import torch
from full_epd import FullEPD

# Define a Graph Neural Network Model
class QuantumGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_core_layers, target):
        super(QuantumGNN, self).__init__()

        self.target = target

        # Graph Convolutional Layers
        self.epd_model = FullEPD(input_dim=input_dim, feature_dim=feature_dim, hidden_dim=hidden_dim, num_core_layers=num_core_layers)

        # Linear layer for final output (log complex amplitude or phase estimate)
        self.final_layer = torch.nn.Linear(feature_dim * 2, 1)

        if self.target == "phase":
            self.final_activation = torch.nn.Sigmoid()
        elif self.target == "log_amp":
            self.final_activation = torch.ReLU()

    def forward(self, graph):

        # Pass through the GCN layers
        graph = self.epd_model(graph)

        # Pool node features (global mean pooling)
        node_pooled = torch.sum(graph.edge_attr,dim=0)

        # Pool edge features (mean pooling across edges)
        edge_pooled = torch.sum(graph.x,dim=0)

        # Concatenate node and edge pooled features into a 128-dimensional vector
        combined_features = torch.cat((node_pooled, edge_pooled), dim=-1)

        # Pass through the fully connected layer to get the final estimate
        output = self.final_activation(self.final_layer(combined_features))

        if self.target == "phase":
            output = output * 3.141592653589793

        return output