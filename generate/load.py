"""
Re-implementation of data preprocessing in NRI.
"""
import torch
import numpy as np
from itertools import permutations


def load_nri(data: dict, size: int):
    """
    Load Springs / Charged dataset.

    Args:
        data: train / val / test
        size: number of nodes, used for generating the edge list
    
    Return:
        data: min-max normalized data
        es: edge list
        max_min: maximum and minimum values of each input dimension
    """
    # edge list of a fully-connected graph
    es = np.array(list(permutations(range(size), 2))).T
    # convert the original data to torch.Tensor
    data = {key: preprocess(value, es) for key, value in data.items()}
    # for spring and charged
    # return maximum and minimum values of each input dimension in order to map normalized data back to the original space
    data, max_min = loc_vel(data)
    return data, es, max_min


def load_kuramoto(data: dict, size: int):
    """
    Load Kuramoto dataset.

    Args:
        data: train / val / test
        size: number of nodes, used for generating the edge list
    
    Return:
        data: min-max normalized data
        es: edge list
        max_min: maximum and minimum values of each input dimension
    """
    # edge list of a fully-connected graph
    es = np.array(list(permutations(range(size), 2))).T
    # convert the original data to torch.Tensor
    data = {key: preprocess(value, es) for key, value in data.items()}
    # return maximum and minimum values of each input dimension in order to map normalized data back to the original space
    data, max_min = dim_norm(data)
    # selected features, same as in NRI
    for key, value in data.items():
        data[key] = (value[0], value[1][:, :, :, [0, 1, 3]])
    max_min = [m[:, :, :, [0, 1, 3]] for m in max_min]
    return data, es, max_min


def dim_norm(data: dict):
    """
    Normalize node states in each dimension separately.

    Args:
        data: train / val / test, each in the form of [adj, state]

    Return:
        data: normalized data
        min_max: maximum and minimum values over all dimensions
    """
    # state: [batch, steps, node, dim]
    _, state = data['train']
    # maximum values over all dimensions
    M = max_except(state)
    # minimum values over all dimensions
    m = -max_except(-state)
    for key, value in data.items():
        try:
            data[key][1] = normalize(value[1], M, m)
        except:
            data[key] = (value[0], normalize(value[1], M, m))
    return data, (M, m)


def max_except(x: np.ndarray) -> np.ndarray:
    """
    Return the maximum values of x for each dimension over all samples.
    """
    shape = x.shape
    x = x.reshape((-1, shape[-1]))
    x, _ = x.max(0)
    size = [1] * len(shape[:-1]) + [shape[-1]]
    x = x.view(*size)
    return x


def loc_vel(data: dict):
    """
    Normalize Springs / Charged dataset (2-D dyamical systems). The dimension of the input feature is 4, 2 for location and 2 for velocity.

    Args:
        data: train / val / test

    Return:
        data: normalized data
        min_max: minimum and maximum values of each dimension of features in the training set
    """
    _, state = data['train']
    loc = state[:, :, :, :2]
    vel = state[:, :, :, 2:]
    loc_max = loc.max()
    loc_min = loc.min()
    vel_max = vel.max()
    vel_min = vel.min()
    for key, value in data.items():
        # normalize the location
        data[key][1][:, :, :, :2] = normalize(value[1][:, :, :, :2], loc_max, loc_min)
        # normalize the velocity
        data[key][1][:, :, :, 2:] = normalize(value[1][:, :, :, 2:], vel_max, vel_min)
    return data, (loc_max, loc_min, vel_max, vel_min)


def normalize(x: np.ndarray, up: float, down: float, a: float=-1, b: float=1) -> np.ndarray:
    """Scale the data x bounded in [down, up] to [a, b]."""
    return (x - down) / (up - down) * (b - a) + a


def preprocess(data: list, es: np.ndarray):
    """
    Convert the original data to torch.Tensor and organize them in the batch form.

    Args:
        data: [[adj, states], ...], all samples, each contains an adjacency matrix and the node states
        es: edge list

    Return:
        adj: adjacency matrices in the batch form
        states: node states in the batch form
    """
    # data: [[adj, states]]
    adj, state = [np.stack(i, axis=0) for i in zip(*data)]
    # scale the adjacency matrix to {0, 1}, only effective for Charged dataset since the elements take values in {-1, 1}
    adj = (adj + 1) / 2
    row, col = es
    # adjacency matrix in the form of edge list
    adj = adj[:, row, col]
    # organize the data in the batch form
    adj = torch.LongTensor(adj)
    state = torch.FloatTensor(state)
    return adj, state
