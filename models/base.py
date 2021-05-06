from torch import nn
import torch.nn.functional as F
from utils.general import prod
from utils.torch_extension import my_bn
from torch import Tensor
import math


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float=0.):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension
            do_prob: rate of dropout
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.dropout_prob = do_prob
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs: Tensor) -> Tensor:
        """
        Batch normzalization for multi-dimensional tensor.
        """
        x = inputs.view(prod(inputs.shape[:-1]), -1)
        x = self.bn(x)
        return x.view(*inputs.shape[:-1], -1)

    def forward(self, inputs: Tensor) -> Tensor:
        x = F.elu(self.fc1(inputs))
        x = my_bn(x, self.bn2)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class LinAct(nn.Module):
    """A linear layer with a non-linear activation function."""
    def __init__(self, n_in: int, n_out: int, do_prob: float=0, act=None):
        """
        Args:
            n_in: input dimension
            n_out: output dimension
            do_prob: rate of dropout
        """
        super(LinAct, self).__init__()
        if act == None:
            act = nn.ReLU()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_out),
            act,
            nn.Dropout(do_prob)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SelfAtt(nn.Module):
    """Self-attention."""
    def __init__(self, n_in: int, n_out: int):
        """
        Args:
            n_in: input dimension
            n_hid: output dimension
        """
        super(SelfAtt, self).__init__()
        self.query, self.key, self.value = nn.ModuleList([
            nn.Sequential(nn.Linear(n_in, n_out), nn.Tanh())
            for _ in range(3)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., size, dim]

        Return:
            out: [..., size, dim]
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # scaled dot product
        alpha = (query @ key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        att = alpha.softmax(-1)
        out = att @ value
        return out
