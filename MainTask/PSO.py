import numpy as np
import torch
from tqdm import tqdm

def PSO(x, optim_func, n_features, max_iter=30, n_particles=30, w=0.6, c1=2, c2=2, device='cpu', use_tqdm=True, batch_size=10000, batch_threshold=20000):
    '''
    Particle swarm optimization
    
    :param x:            Tensor[#n classes/functions, #k features]
    :param optim_func:   Function to optimise - Sammon or Kruskal
    :param n_features:   How many features do we want
    :param max_iter:     Max num of iterations
    :param n_particles:  Number of particles
    :param w:            Inertia weight
    :param c1:           Cognitive weight
    :param c2:           Social weight
    :param device:       Device to compute (cpu or cuda)
    :param use_tqdm:     Flag to show progress bar
    :return:             Index of best features and error during training
    '''
    
    assert torch.is_tensor(x), "x must be a tensor"
    assert len(x.shape) == 2, "x must be a 2dim tensor"
    
    error_log = []
    
    x = x.to(device)
    
    dim = x.shape[1]
    
    particles = torch.rand(size=(n_particles, dim), dtype=torch.float, device=device)
    
    # Velicities init as 0 in
    # https://github.com/Pixelatory/PSO-GPU/blob/main/PSO.py
    velocities = torch.zeros(size=(n_particles, dim), dtype=torch.float, device=device)     
    
    pbest_pos = torch.rand(size=(n_particles, dim), dtype=torch.float, device=device)
    pbest_val = float('inf') * torch.ones(size=(n_particles, 1), dtype=torch.float, device=device)
    
    gbest_val = torch.tensor(float('inf'), dtype=torch.float, device=device)
    gbest_pos = torch.zeros(size=(1, dim), dtype=torch.float, device=device)
    
    use_batch = x.shape[0] >= batch_threshold
    
    if not use_batch:
        tril_index = torch.tril_indices(x.shape[0], x.shape[0], offset=-1, device=device)
        tril_index = tril_index[0] * x.shape[0] + tril_index[1]

        d_y = torch.cdist(x, x)
        d_y = d_y.view(-1)[tril_index]
    
    iterator = range(max_iter)
    if use_tqdm:
        iterator = tqdm(iterator)
    
    for _ in iterator:
        # Calculate error
        best_features_idx = torch.topk(particles, n_features, dim=1)[1]
        #x_stacked_indexed = batched_index_select(x_stacked, 2, best_features_idx)
        error = float('inf') * torch.ones(size=(n_particles, 1), dtype=torch.float, device=device)
        for idx in range(n_particles):
            if not use_batch:
                d_x = torch.cdist(x[:, best_features_idx[idx]], x[:, best_features_idx[idx]])
                d_x = d_x.view(-1)[tril_index]

                error[idx, 0] = optim_func(d_y, d_x)
                
            else:
                
                error[idx, 0] = optim_func(x, x[:, best_features_idx[idx]], batched_input=True, batch_size=batch_size)
        #error = optim_func(x_stacked_indexed, x)
        
        # Update personal best
        pbest_pos = torch.where(error < pbest_val, particles, pbest_pos)
        pbest_val = torch.minimum(error, pbest_val)

        # Update global best
        best_sol = torch.min(pbest_val, dim=0)
        gbest_pos = pbest_pos[best_sol[1]].detach().clone()
        gbest_val = best_sol[0]
        
        # Velocity equation
        r1 = torch.rand(size=(n_particles, dim), dtype=torch.float, device=device)
        r2 = torch.rand(size=(n_particles, dim), dtype=torch.float, device=device)
        inertia = w * velocities
        cognitive = c1 * r1 * (pbest_pos - particles)
        social = c2 * r2 * (gbest_pos - particles)
        
        # velocity update and constraint
        velocities = inertia + cognitive + social

        particles += velocities
        
        error_log.append(gbest_val.cpu().item())
    
    return torch.topk(gbest_pos, n_features, dim=1)[1].squeeze().cpu(), error_log