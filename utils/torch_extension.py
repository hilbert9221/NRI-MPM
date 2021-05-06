import torch
from torch import Tensor
from itertools import combinations
from utils.general import prod
import numpy as np


def edge_accuracy(preds: Tensor, target: Tensor) -> float:
    """
    Compute the accuracy of edge prediction (relation reconstruction).

    Args:
        preds: [E, batch, K], probability distribution of K types of relation for each edge
        target: [batch, E], ground truth relations

    Return:
         accuracy of edge prediction (relation reconstruction)
    """
    _, preds = preds.max(-1)
    correct = (preds.t() == target).sum().float()
    return correct / (target.size(0) * target.size(1))


def asym_rate(x: Tensor, size: int) -> float:
    """
    Given an edge list of a graph, compute the rate of asymmetry.

    Args:
        x: [batch, E], edge indicator
        size: number of nodes of the graph

    Return:
        rate of asymmetry
    """
    # get the edge indicator of a transposed adjacency matrix
    idx = transpose_id(size)
    x_t = x[:, idx]
    rate = (x != x_t).sum().float() / (x.shape[0] * x.shape[1])
    return rate


def transpose_id(size: int) -> Tensor:
    """
    Return the edge list corresponding to a transposed adjacency matrix.
    """
    idx = torch.arange(size * (size - 1))
    ii = idx // (size - 1)
    jj = idx % (size - 1)
    jj = jj * (jj < ii).long() + (jj + 1) * (jj >= ii).long()
    index = jj * (size - 1) + ii * (ii < jj).long() + (ii - 1) * (ii > jj).long()
    return index


def transpose(x: Tensor, size: int) -> Tensor:
    """
    Transpose the edge features x.
    """
    index = transpose_id(size)
    return x[index]


def sym_hard(x: Tensor, size: int) -> Tensor:
    """
    Given the edge features x, set x(e_ji) = x(e_ij) to impose hard symmetric constraints.
    """
    i, j = np.array(list(combinations(range(size), 2))).T
    idx_s = j * (size - 1) + i * (i < j) + (i - 1) * (i > j)
    idx_t = i * (size - 1) + j * (j < i) + (j - 1) * (j > i)
    x[idx_t] = x[idx_s]
    return x


def my_bn(x: Tensor, bn: torch.nn.BatchNorm1d) -> Tensor:
    """
    Applying BatchNorm1d to a multi-dimesional tensor x.
    """
    shape = x.shape
    z = x.view(prod(shape[:-1]), shape[-1])
    z = bn(z)
    z = z.view(*shape)
    return z


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape)
    return - (- (U.relu() + eps).log().clamp(max=0.) + eps).log()


def sample_gumbel_max(logits, eps=1e-10, one_hot=False):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda(logits.device)
    y = logits + gumbel_noise
    ms, index = y.max(-1, keepdim=True)
    es = (y >= ms) if one_hot else index
    return es


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda(logits.device)
    y = logits + gumbel_noise
    return (y / tau).softmax(-1)
