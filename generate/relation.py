import numpy as np


def rand_adj(size: int) -> np.ndarray:
    """
    Generate a random symmetric adjacency matrix. Each element is sampled from a Bernoulli distribution, Bern(0.5).

    Users may define various schemes to generate random adjacency matrix.

    Args:
        size: number of nodes
    
    Return:
        adj: adjacency matrix
    """
    adj = np.random.choice([0, 1], size=(size, size), p=[0.5, 0.5])
    # upper triangular matrix
    adj = np.triu(adj)
    np.fill_diagonal(adj, 0)
    adj = adj + adj.T
    return adj
