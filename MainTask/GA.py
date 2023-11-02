import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from tqdm import tqdm
import itertools


def GA(x, optim_func, n_features, max_iter=30, population_size=30, n_best=6, p_c_threshold=0.5, p_m_threshold=0.5, device='cpu', use_tqdm=True, batch_size=10000, batch_threshold=20000, cache=None):
    
    
    assert torch.is_tensor(x), "x must be a tensor"
    assert len(x.shape) == 2, "x must be a 2dim tensor"
    
    x = x.to(device)
    
    dim = x.shape[1]
    
    error_log = []
    
    best_error = float('inf') * torch.ones(1, dtype=torch.float, device=device)[0]
    best_in_population = torch.zeros(dim, dtype=torch.float, device=device)
    
    population = torch.rand(size=(population_size, dim), dtype=torch.float, device=device)
    
    combs = list(itertools.combinations(range(n_best), 2))[:population_size]
    
    
    use_batch = x.shape[0] >= batch_threshold
    
    if not use_batch:
        d_x = torch.cdist(x, x)
    
    iterator = range(max_iter)
    if use_tqdm:
        iterator = tqdm(iterator)
    
    for _ in iterator:
        best_features_idx = torch.topk(population, n_features, dim=1)[1]
        error = float('inf') * torch.ones(size=(population_size, 1), dtype=torch.float, device=device)
        
        for idx in range(population_size):
            feat_list = best_features_idx[idx].cpu().tolist()
            bitmask = 0
            for feat_idx in feat_list:
                bitmask = bitmask | 2**feat_idx
            
            if cache is not None and bitmask in cache:
                error[idx, 0] = torch.tensor(cache[bitmask]).double().to(device)
                continue
            
            y = x[:, best_features_idx[idx]]
            if not use_batch:
                err_val = optim_func(x, y, d_x=d_x)                
            else:
                err_val = optim_func(x, y, batched_input=True, batch_size=batch_size)
            
            if cache is not None:
                cache[bitmask] = err_val.cpu().item()
            error[idx, 0] = err_val
        
        min_error_idx = error.argmin()
        if error[min_error_idx] < best_error:
            best_error = error[min_error_idx]
            best_in_population = population[min_error_idx]
        
        error_log.append(best_error.cpu().item())
        
        parents_idx = torch.topk(error, n_best, dim=0)[1].squeeze(1)
        parents = population[parents_idx, :]
        
        
        new_population = torch.zeros(size=(population_size, dim), dtype=torch.float, device=device)
        
        for idx, comb in enumerate(combs):
            childs = parents[[comb[0], comb[1]]]
            
            p_c = torch.rand(1)[0]
            p_m = torch.rand(2)
            if p_c < p_c_threshold:
                cut_idx = torch.randint(0, dim, (1,), device=device)
                childs[0, :cut_idx], childs[1, :cut_idx] = childs[1, :cut_idx], childs[0, :cut_idx]
                
            if p_m[0] < p_m_threshold:
                childs[0] = childs[0] + torch.rand(dim, device=device)
            
            if p_m[1] < p_m_threshold:
                childs[1] = childs[1] + torch.rand(dim, device=device)
            
            new_population[2*idx:2*idx+2] = childs
        
        population = new_population.clone()
        
        
    return torch.topk(best_in_population, n_features)[1].squeeze().cpu(), error_log