import torch
from torch import Tensor
from torch import nn
from torch_scatter import scatter


class GNN(nn.Module):
    """
    Reimplementaion of the Message-Passing class in torch-geometric to allow more flexibility.
    """
    def __init__(self):
        super(GNN, self).__init__()
        
    def forward(self, *input):
        raise NotImplementedError

    def propagate(self, x: Tensor, es: Tensor, f_e: Tensor=None, agg: str='mean') -> Tensor:
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            x: [node, ..., dim], node embeddings 
        """
        msg, idx, size = self.message(x, es, f_e)
        x = self.aggregate(msg, idx, size, agg)                          
        return x

    def aggregate(self, msg: Tensor, idx: Tensor, size: int, agg: str='mean') -> Tensor:
        """
        Args:
            msg: [E, ..., dim * 2]
            idx: [E]
            size: number of nodes
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            aggregated node embeddings
        """
        assert agg in {'add', 'mean', 'max'}
        return scatter(msg, idx, dim_size=size, dim=0, reduce=agg)

    def node2edge(self, x_i: Tensor, x_o: Tensor, f_e: Tensor) -> Tensor:
        """
        Args:
            x_i: [E, ..., dim], embeddings of incoming nodes
            x_o: [E, ..., dim], embeddings of outcoming nodes
            f_e: [E, ..., dim * 2], edge embeddings

        Return:
            edge embeddings
        """
        return torch.cat([x_i, x_o], dim=-1)

    def message(self, x: Tensor, es: Tensor, f_e: Tensor=None, option: str='o2i'):
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            option: default: 'o2i'
                'o2i': collecting incoming edge embeddings
                'i2o': collecting outcoming edge embeddings

        Return:
            mgs: [E, ..., dim * 2], edge embeddings
            col: [E], indices of 
            size: number of nodes
        """
        if option == 'i2o':
            row, col = es
        if option == 'o2i':
            col, row = es
        else:
            raise ValueError('i2o or o2i')
        x_i, x_o = x[row], x[col]
        msg = self.node2edge(x_i, x_o, f_e)
        return msg, col, len(x)

    def update(self, x):
        return x
