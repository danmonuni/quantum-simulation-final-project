import torch
from full_epd import FullEPD


# define a graph neural network model
class ClassicalGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_core_layers, target):
        super(ClassicalGNN, self).__init__()

        self.target = target

        #graph convolutional layers
        self.epd_model = FullEPD(input_dim=input_dim, feature_dim=feature_dim, hidden_dim=hidden_dim, num_core_layers=num_core_layers)

        #linear layer for final output (log complex amplitude or phase estimate)
        self.final_layer = torch.nn.Linear(feature_dim * 2, 1)

        #final activation to achieve a probability
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, graph):

        # pass through the graph EPD structure
        graph = self.epd_model(graph)

        # pool node features (sum mean pooling)
        node_pooled = torch.sum(graph.edge_attr, dim=0)

        # pool edge features (sum pooling across edges)
        edge_pooled = torch.sum(graph.x, dim=0)

        # Concatenate node and edge pooled features into a 128-dimensional vector
        combined_features = torch.cat((node_pooled, edge_pooled), dim=-1)

        # Pass through the fully connected layer to get the final estimate
        output = self.final_activation(self.final_layer(combined_features))

        return output