import torch
from tqdm.auto import tqdm

from src.evaluation import H, mcmc
from src.lattice_generation import square_9
from src.model.full_classical_gnn import GNN
from src.visualisation import draw_graph, print_graph

G,G_pyg = square_9()

draw_graph(G)
print_graph(G_pyg)


gnn = GNN(input_dim=3, feature_dim=64, hidden_dim=128, num_core_layers=6)
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005, weight_decay=5e-4)

torch.set_printoptions(profile="full")


#Training Loop
num_epochs = 20
n_mc_points = 512

for epoch in tqdm(range(num_epochs)):

    gnn.eval()

    # Forward pass
    mc_points = mcmc(G_pyg = G_pyg, gnn = gnn, burn_in = 100, n_mc_points = n_mc_points, spacing = 10)

    gnn.train()
    optimizer.zero_grad()

    energy_distribution = torch.zeros((n_mc_points, 2))

    for i in range(mc_points.shape[0]):
      G_pyg.x[:,0] = mc_points[i,:]
      energy_distribution[i,0] = gnn(G_pyg)
      energy_distribution[i,1] = H(mc_points[i,:]) #+ 45

    print(energy_distribution.T)

    # Compute loss
    loss = torch.matmul(energy_distribution[:,0].T, energy_distribution[:,1]) / torch.matmul(energy_distribution[:,0].T, torch.ones(energy_distribution.shape[0]))

    print(loss)

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

