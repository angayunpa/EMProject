import torch
import numpy as np

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


def kruskal_stress_error(d_x, d_y, smooth=1e-8, k_neighbors=3):
    '''
    :param d_x:         Original data distance must be lower-triangular using torch.tril() -> Tensor[#n classes/functions, #n classes/functions]
    :param d_y:         Input data lower-triangular matrix of distances -> Tensor[#n classes/functions, #n classes/functions]
    :param smooth:      Smooth to avoid div by 0
    :param k_neighbors: The number of nearest neighbors to consider
    '''
    d_x_thresh = torch.where(d_x <= d_x.topk(k_neighbors, largest=False).values[:, -1:], d_x, 0.)
    d_x_geodesic = dijkstra(d_x_thresh)
    d_y_thresh = torch.where(d_y <= d_y.topk(k_neighbors, largest=False).values[:, -1:], d_y, 0.)
    d_y_geodesic = dijkstra(d_y_thresh)
    return torch.sqrt(torch.sum(d_x_geodesic - d_y_geodesic) / torch.sum(d_x_geodesic + smooth))

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


class priorityQ_torch(object):
    """Priority Q implelmentation in PyTorch

    Args:
        object ([torch.Tensor]): [The Queue to work on]
    """

    def __init__(self, val):
        self.q = torch.tensor([[val, 0]])

    def push(self, x):
        """Pushes x to q based on weightvalue in x. Maintains ascending order

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]
            x ([torch.Tensor]): [[index, weight] tensor to be inserted]

        Returns:
            [torch.Tensor]: [The queue tensor after correct insertion]
        """
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if self.isEmpty():
            self.q = x
            self.q = torch.unsqueeze(self.q, dim=0)
            return
        idx = torch.searchsorted(self.q.T[1], x[1])
        self.q = torch.vstack([self.q[0:idx], x, self.q[idx:]]).contiguous()

    def top(self):
        """Returns the top element from the queue

        Returns:
            [torch.Tensor]: [top element]
        """
        return self.q[0]

    def pop(self):
        """pops(without return) the highest priority element with the minimum weight

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [torch.Tensor]: [highest priority element]
        """
        self.q = self.q[1:]

    def isEmpty(self):
        """Checks is the priority queue is empty

        Args:
            q ([torch.Tensor]): [The tensor queue arranged in ascending order of weight value]

        Returns:
            [Bool] : [Returns True is empty]
        """
        return self.q.shape[0] == 0


def dijkstra(adj):
    n = adj.shape[0]
    distance_matrix = torch.zeros([n, n])
    for i in range(n):
        d = np.inf * torch.ones(n)
        d[i] = 0
        q = priorityQ_torch(i)
        while not q.isEmpty():
            v, d_v = q.top()  # point and distance
            v = v.int()
            q.pop()
            if d_v != d[v]:
                continue
            for j, py in enumerate(adj[v]):
                if py == 0 and j != v:
                    continue
                else:
                    to = j
                    if d[v] + py < d[to]:
                        d[to] = d[v] + py
                        q.push(torch.Tensor([to, d[to]]))
        distance_matrix[i] = d
    return distance_matrix