
import itertools

import torch
from tqdm.auto import tqdm


#classical hamiltonian
def H(x, G_pyg):

    J = 1
    interactions = torch.zeros((G_pyg.num_nodes, G_pyg.num_nodes))
    interactions[G_pyg.edge_index[0], G_pyg.edge_index[1]] = J

    M = 1
    external_field = torch.zeros(G_pyg.num_nodes)

    external_field[:] = M

    x = x.type(torch.float32)

    return -torch.matmul(external_field, x) - torch.matmul(torch.matmul(x.T, interactions), x)


#mcmc

#proposal function
def g(x):

    x_ = x.clone()

    idx1 = torch.randint(0, len(x_), (1,))

    # opposite_indices = torch.where(x_ == -x_[idx1])[0]

    # idx2 = opposite_indices[torch.randint(0, len(opposite_indices), (1,))].item()

    x_[idx1] =  - x[idx1]
    return x_

#evaluation function
def unnorm_prob(x, G_pyg, gnn):
    G_pyg.x[:,0] = x
    return gnn(G_pyg).item()

# 1 step
def mcmc_step(x, G_pyg, gnn):
    x_ = g(x)

    a_1 = unnorm_prob(x_, G_pyg, gnn)
    a_2 = unnorm_prob(x, G_pyg, gnn)

    if a_2 == 0:
        acceptance_ratio = 1
    else:
        acceptance_ratio = a_1/a_2

    if torch.rand(1) < acceptance_ratio:
        x = x_

    return x_, x


#full mcmc
def mcmc(G_pyg, gnn, burn_in = 1000, n_mc_points = 1000, spacing = 10):

    mc_points = torch.zeros((n_mc_points, G_pyg.num_nodes))


    x = torch.randint(0, 2, (G_pyg.num_nodes,)) * 2 - 1

    for i in range(burn_in):
        x = mcmc_step(x, G_pyg, gnn)[1]

    for i in tqdm(range(n_mc_points)):
        for j in range(spacing):
            x = mcmc_step(x, G_pyg, gnn)[1]

        mc_points[i,:] = x


    return mc_points


#exhaustive sampling instead of mcmc
def exhaustive(G_pyg , n_mc_points = 512):

    mc_points = torch.zeros((n_mc_points, G_pyg.num_nodes))

    for i, x in enumerate(itertools.product((-1,1), repeat=G_pyg.num_nodes)):
        mc_points[i,:] = torch.tensor(x)

    return mc_points