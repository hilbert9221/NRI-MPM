from models.base import MLP, SelfAtt
from torch import nn
import torch
from torch import Tensor
import math
from models.gnn import GNN
import numpy as np


class CNN(nn.Module):
    """
    CNN from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float=0.):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension
        """
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(do_prob),
            nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                         dilation=1, return_indices=False,
                         ceil_mode=False),
            nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid)
        )
        self.out = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.att = nn.Conv1d(n_hid, 1, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: [batch * E, dim, step], raw edge representations at each step

        Return:
            edge_prob: [batch * E, dim], edge representations over all steps with the step-dimension reduced
        """
        x = self.cnn(inputs)
        pred = self.out(x)
        attention = self.att(x).softmax(2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
      

class GNNENC(GNN):
    """
    Encoder of NRI. A combination of MLPEncoder and CNNEncoder from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int,
                 do_prob: float=0., factor: bool=True,
                 reducer: str='mlp'):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension, i.e., number of edge types
            do_prob: rate of dropout, default: 0
            factor: using a factor graph or not, default: True
            reducer: using an MLP or an CNN to reduce edge representations over multiple steps
        """
        super(GNNENC, self).__init__()
        self.factor = factor
        assert reducer in {'mlp', 'cnn'}
        self.reducer = reducer
        if self.reducer == 'mlp':
            self.emb = MLP(n_in, n_hid, n_hid, do_prob)
            self.n2e_i = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        else:
            self.cnn = CNN(2 * n_in, n_hid, n_hid, do_prob)
            self.n2e_i = MLP(n_hid, n_hid, n_hid, do_prob)
        self.e2n = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.n2e_o = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        else:
            self.n2e_o = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def reduce_mlp(self, x: Tensor, es: Tensor):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        x = x.view(x.size(0), x.size(1), -1)
        x = self.emb(x)
        z, col, size = self.message(x, es)
        return z, col, size

    def reduce_cnn(self, x, es):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        # z: [E, batch, step, dim * 2]
        z, col, size = self.message(x, es)
        z = z.transpose(3, 2).contiguous()
        # z: [E * batch, dim * 2, step]
        z = z.view(-1, z.size(2), z.size(3))
        z = self.cnn(z)
        z = z.view(len(col), x.size(1), -1)
        return z, col, size

    def forward(self, x: Tensor, es: Tensor) -> Tensor:
        """
        Given the historical node states, output the K-dimension edge representations ready for relation prediction.

        Args:
            x: [batch, step, node, dim], node representations
            es: [2, E], edge list

        Return:
            z: [E, batch, dim], edge representations
        """
        x = x.permute(2, 0, 1, -1).contiguous()
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        z, col, size = getattr(self, 'reduce_{}'.format(self.reducer))(x, es)
        z = self.n2e_i(z)
        z_skip = z
        if self.factor:
            h = self.aggregate(z, col, size)
            h = self.e2n(h)
            z, _, __ = self.message(h, es)
            # skip connection
            z = torch.cat((z, z_skip), dim=2)
            z = self.n2e_o(z)
        else:
            z = self.e2n(z)
            z = torch.cat((z, z_skip), dim=2)
            z = self.n2e_o(z)
        z = self.fc_out(z)
        return z


class RNNENC(GNN):
    """
    Encoder using the relation interaction mechanism implemented by RNNs.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int,
                 do_prob: float=0., factor: bool=True,
                 reducer: str='mlp', option: str='inter'):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension, i.e., number of edge types
            do_prob: rate of dropout, default: 0
            factor: using a factor graph or not, default: True
            reducer: using an MLP or an CNN to reduce edge representations over multiple steps
            option: default: 'both'
                'intra': using the intra-edge interaction operation
                'inter': using the inter-edge interaction operation
                'both': using both operations
        """
        super(RNNENC, self).__init__()
        self.factor = factor
        self.option = option
        assert reducer in {'mlp', 'cnn'}
        self.reducer = reducer
        if self.reducer == 'mlp':
            self.emb = MLP(n_in, n_hid, n_hid, do_prob)
            self.n2e_i = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        else:
            self.cnn = CNN(2 * n_in, n_hid, n_hid, do_prob)
            self.n2e_i = MLP(n_hid, n_hid, n_hid, do_prob)
        self.e2n = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.n2e_o = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        else:
            self.n2e_o = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        # rnns for both intra-edge and inter-edge operations
        self.intra_gru = nn.GRUCell(n_hid, n_hid)
        self.inter_gru = nn.GRUCell(n_hid, n_hid)
        if option == 'both':
            self.fc_out = nn.Linear(n_hid * 2, n_out)
        else:
            self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def reduce_mlp(self, x: Tensor, es: Tensor):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        x = x.view(x.size(0), x.size(1), -1)
        x = self.emb(x)
        z, col, size = self.message(x, es)
        return z, col, size

    def reduce_cnn(self, x, es):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        # z: [E, batch, step, dim * 2]
        z, col, size = self.message(x, es)
        z = z.transpose(3, 2).contiguous()
        # z: [E * batch, dim * 2, step]
        z = z.view(-1, z.size(2), z.size(3))
        z = self.cnn(z)
        z = z.view(len(col), x.size(1), -1)
        return z, col, size

    def intra_es(self, z: Tensor, size: Tensor) -> Tensor:
        """
        Args:
            z: [E, batch, dim]
            es: [2, E]

        Return:
            zs: [size, size - 1, batch, dim]
        """
        E, batch, dim = z.shape
        # zz: [size, size - 1, batch, dim]
        zz = z.view(E//(size-1), size-1, batch, dim)
        zz = zz.transpose(0, 1).contiguous().view(size-1, -1, dim)
        orig = np.arange(zz.shape[1]).reshape((1, -1)).repeat(size-1, axis=0)
        std = np.arange(size-1).reshape((-1, 1)).repeat(E*batch//(size-1), axis=-1)
        np.random.seed()
        # permute edge indices
        index = np.apply_along_axis(np.random.permutation, 0, std)
        inv_index = index.copy()
        # inversed indexing, satifying zz[index, orig][inv_index, orig] = zz
        inv_index[index, orig] = std
        # permute edge embeddings
        zz = zz[index, orig]
        hidden = torch.zeros(*zz.shape[1:])
        zs = []
        if z.is_cuda:
            hidden = hidden.cuda(z.device)
        for i in range(size-1):
            hidden = self.intra_gru(zz[i], hidden)
            zs.append(hidden)
        zs = torch.stack(zs)
        # restore the order of edge embeddings
        zs = zs[inv_index, orig]
        zs = zs.view(size-1, E//(size-1), batch, dim)
        zs = zs.transpose(0, 1).contiguous()
        return zs

    def inter_es(self, zs: Tensor, size: Tensor) -> Tensor:
        """
        Args:
            zs: [size, size - 1, batch, dim]
            es: [2, E]

        Return:
            hs: [size, size - 1, batch, dim]
        """
        # mean pooling to get the overall embedding of all incoming edges
        h = zs.mean(1)
        batch = h.shape[1]
        orig = np.arange(batch).reshape((1, -1)).repeat(size, axis=0)
        std = np.arange(size).reshape((-1, 1)).repeat(batch, axis=-1)
        np.random.seed()
        # permute edge indices
        index = np.apply_along_axis(np.random.permutation, 0, std)
        inv_index = index.copy()
        # inversed indexing, satifying h[index, orig][inv_index, orig] = h
        inv_index[index, orig] = std
        # permute edge embeddings
        h = h[index, orig]
        hidden = torch.zeros(*h.shape[1:])
        hs = []
        if h.is_cuda:
            hidden = hidden.cuda(h.device)
        for i in range(size):
            hidden = self.inter_gru(h[i], hidden)
            hs.append(hidden)
        hs = torch.stack(hs)
        # restore the order of edge embeddings
        hs = hs[inv_index, orig]
        hs = hs.unsqueeze(1).repeat(1, size-1, 1, 1)
        return hs

    def forward(self, x, es):
        """
        Given the historical node states, output the K-dimension edge representations ready for relation prediction.

        Args:
            x: [batch, step, node, dim], node representations
            es: [2, E], edge list

        Return:
            z: [E, batch, dim], edge representations
        """
        x = x.permute(2, 0, 1, -1).contiguous()
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        z, col, size = getattr(self, 'reduce_{}'.format(self.reducer))(x, es)
        z = self.n2e_i(z)
        z_skip = z
        if self.factor:
            h = self.aggregate(z, col, size)
            h = self.e2n(h)
            z, _, __ = self.message(h, es)
            # skip connection
            z = torch.cat((z, z_skip), dim=2)
            z = self.n2e_o(z)
        else:
            z = self.e2n(z)
            z = torch.cat((z, z_skip), dim=2)
            z = self.n2e_o(z)
        E, batch, dim = z.shape
        if self.option == 'both':
            zs = self.intra_es(z, size)
            hs = self.inter_es(zs, size)
            zs = torch.cat([zs, hs], dim=-1)
            z = zs.view(-1, batch, dim * 2)
        elif self.option == 'intra':
            z = self.intra_es(z, size)
            z = z.view(-1, batch, dim)
        else:
            z = z.view(E//(size-1), size-1, batch, dim)
            z = self.inter_es(z, size)
            z = z.view(-1, batch, dim)
        z = self.fc_out(z)
        return z


class AttENC(GNN):
    """
    Encoder using the relation interaction mechanism implemented by self-attention.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int,
                 do_prob: float=0., factor: bool=True,
                 reducer: str='mlp', option: str='both'):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension, i.e., number of edge types
            do_prob: rate of dropout, default: 0
            factor: using a factor graph or not, default: True
            reducer: using an MLP or an CNN to reduce edge representations over multiple steps
            option: default: 'both'
                'intra': using the intra-edge interaction operation
                'inter': using the inter-edge interaction operation
                'both': using both operations
        """
        super(AttENC, self).__init__()
        self.factor = factor
        self.option = option
        assert reducer in {'mlp', 'cnn'}
        self.reducer = reducer
        if self.reducer == 'mlp':
            self.emb = MLP(n_in, n_hid, n_hid, do_prob)
            self.n2e_i = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        else:
            self.cnn = CNN(2 * n_in, n_hid, n_hid, do_prob)
            self.n2e_i = MLP(n_hid, n_hid, n_hid, do_prob)
        self.e2n = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.n2e_o = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        else:
            self.n2e_o = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        # rnns for both intra-edge and inter-edge operations
        self.intra_att = SelfAtt(n_hid, n_hid)
        self.inter_att = SelfAtt(n_hid, n_hid)
        if option == 'both':
            self.fc_out = nn.Linear(n_hid * 2, n_out)
        else:
            self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def reduce_mlp(self, x: Tensor, es: Tensor):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        x = x.view(x.size(0), x.size(1), -1)
        x = self.emb(x)
        z, col, size = self.message(x, es)
        return z, col, size

    def reduce_cnn(self, x, es):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        # z: [E, batch, step, dim * 2]
        z, col, size = self.message(x, es)
        z = z.transpose(3, 2).contiguous()
        # z: [E * batch, dim * 2, step]
        z = z.view(-1, z.size(2), z.size(3))
        z = self.cnn(z)
        z = z.view(len(col), x.size(1), -1)
        return z, col, size

    def intra_es(self, z: Tensor, size: Tensor) -> Tensor:
        """
        Args:
            z: [E, batch, dim]
            es: [2, E]

        Return:
            zs: [size, size - 1, batch, dim]
        """
        E, batch, dim = z.shape
        # zz: [size, size - 1, batch, dim]
        zz = z.view(E//(size-1), size-1, batch, dim)
        # zz: [size, batch, size - 1, dim]
        zz = zz.transpose(1, 2).contiguous()
        zs = self.intra_att(zz)
        zs = zs.transpose(1, 2).contiguous()
        return zs

    def inter_es(self, zs: Tensor, size: Tensor) -> Tensor:
        """
        Args:
            zs: [size, size - 1, batch, dim]
            es: [2, E]

        Return:
            hs: [size, size - 1, batch, dim]
        """
        # mean pooling to get the overall embedding of all incoming edges
        h = zs.mean(1)
        h = h.transpose(0, 1).contiguous()
        hs = self.inter_att(h)
        hs = hs.transpose(0, 1).contiguous()
        hs = hs.unsqueeze(1).repeat(1, size-1, 1, 1)
        return hs

    def forward(self, x, es):
        """
        Given the historical node states, output the K-dimension edge representations ready for relation prediction.

        Args:
            x: [batch, step, node, dim], node representations
            es: [2, E], edge list

        Return:
            z: [E, batch, dim], edge representations
        """
        x = x.permute(2, 0, 1, -1).contiguous()
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        z, col, size = getattr(self, 'reduce_{}'.format(self.reducer))(x, es)
        z = self.n2e_i(z)
        z_skip = z
        if self.factor:
            h = self.aggregate(z, col, size)
            h = self.e2n(h)
            z, _, __ = self.message(h, es)
            # skip connection
            z = torch.cat((z, z_skip), dim=2)
            z = self.n2e_o(z)
        else:
            z = self.e2n(z)
            z = torch.cat((z, z_skip), dim=2)
            z = self.n2e_o(z)
        E, batch, dim = z.shape
        if self.option == 'both':
            zs = self.intra_es(z, size)
            hs = self.inter_es(zs, size)
            zs = torch.cat([zs, hs], dim=-1)
            z = zs.view(-1, batch, dim * 2)
        elif self.option == 'intra':
            z = self.intra_es(z, size)
            z = z.view(-1, batch, dim)
        else:
            z = z.view(E//(size-1), size-1, batch, dim)
            z = self.inter_es(z, size)
            z = z.view(-1, batch, dim)
        z = self.fc_out(z)
        return z
