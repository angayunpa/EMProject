import torch

def sammon_error(d_x, d_y, smooth=1e-8):
    '''
    :param d_x:     Original data distance must be lower-triangular using torch.tril() -> Tensor[#n classes/functions, #n classes/functions]
    :param d_y:     Input data lower-triangular matrix of distances -> Tensor[#n classes/functions, #n classes/functions]
    :param device:  Device cpu or cuda
    :param smooth:  Smooth to avoid div by 0
    '''
    
    #d_x = torch.tril(torch.cdist(x, x))
    #d_y = torch.cdist(y, y)
    if len(d_x.shape) == 3:
        return 1 / torch.sum(d_x, dim=(-2,-1)) * torch.sum(torch.square(d_x - d_y) / (d_x + smooth), dim=(-2,-1))
    
    return 1 / torch.sum(d_x) * torch.sum(torch.square(d_x - d_y) / (d_x + smooth))



# https://discuss.pytorch.org/t/batched-index-select/9115/11

def batched_index_select(inp, dim, index):
    for ii in range(1, len(inp.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(inp, dim, index)