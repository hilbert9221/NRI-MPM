"""
Re-implementation of data generation for Springs and Charged dataset in NRI.
"""
import numpy as np
from generate.relation import rand_adj


class Simulator:
    """
    A base simulator for continuous dynamical systems.
    """
    def __init__(self, size: int, epoch: int, interval: int, var: float=0):
        """
        Args:
            size: number of nodes
            epoch: simulation steps
            interval: down-sampling factor
            var: variance of the noise, default: 0
        """
        self.size = size
        self.epoch = epoch
        self.interval = interval
        self.var = var

    def set_epoch(self, epoch: int):
        """Reset the simulation steps"""
        self.epoch = epoch

    def init_state(self) -> np.ndarray:
        """Initialize node states"""
        return np.zeros((self.size, 1))

    def update_state(self, state: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Args:
            state: [size, dim], historical states
            adj: [size, size], adjacency matrix

        Return:
            state: updated node states
        """
        return state

    def noise(self, state: np.ndarray) -> np.ndarray:
        """Add a Gaussian noise."""
        return np.random.randn(*state.shape) * self.var

    def init_adj(self) -> np.ndarray:
        """Initialize the adjacency matrix."""
        return rand_adj(self.size)

    def generate(self):
        """
        Simulate a time series for a dynamical with (size) nodes over (self.epoch // self.interval) time steps.

        Retrun:
            adj: [size, size]
            states: [epoch, size, dim]
        """
        # NOTE: set random seeds every time to avoid repetition in multi-processing generation
        np.random.seed()
        # initialize node states and the adjacency matrix
        adj = self.init_adj()
        state = self.init_state()
        states = np.zeros((self.epoch, *state.shape))
        for i in range(self.epoch):
            # sequential generation, with node states saved under a down-samling factor self.interval
            for _ in range(self.interval):
                state = self.update_state(state, adj)
            states[i] = state
        # add noise
        states += self.noise(states)
        return adj, states
