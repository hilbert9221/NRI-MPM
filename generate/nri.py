import numpy as np
from generate.dynamics import Simulator


def distance_pairwise(xs: np.ndarray, ys: np.ndarray, scale: float=1) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance between two set of vectors. The trick is,
    \|x - y\|_2^2 = \|x\|^2_2 + \|y\|^2_2 - 2 x^T y,

    where \|x\|^2_2 can be \|y\|^2_2 pre-computed and reused. x^T y can be computed via matrix product.
    """
    xs /= scale
    ys /= scale
    x_sq = (xs ** 2).sum(axis=1, keepdims=True)
    y_sq = (ys ** 2).sum(axis=1, keepdims=True)
    xy = xs @ ys.T
    return np.sqrt(np.abs(x_sq - 2 * xy + y_sq.T)) * scale


class Base(object):
    def __init__(self):
        self.init_hyper()

    def init_hyper(self):
        self.box_size = 5
        self.loc_std = 0.5
        self.vel_norm = 0.5
        self.interaction_strength = 0.1
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _clamp(self, loc, vel):
        """
        Clamp the values of the location and velocity to avoid extremely large values.

        Args:
            loc: [2, size], location at one time stamp
            vel: [2, size], velocity at one time stamp
            
        Return: 
            location and velocity after hiting walls and returning after
            elastically colliding with walls
        """
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _energy(self, loc, vel, edges):
        # diag(edges) = 0
        K = 0.5 * (vel ** 2).sum()
        dist = distance_pairwise(vel, vel) ** 2
        U = (0.5 * self.interaction_strength * dist * edges / 2).sum()
        return U + K


class BaseSim(Base, Simulator):
    """A base simulator for Springs / Charged dataset."""
    def __init__(self, size: int, epoch: int, interval: int, var: float=0):
        """
        Args:
            size: number of nodes
            epoch: simulation steps
            interval: down-sampling factor
            var: variance of the noise, default: 0
        """
        Simulator.__init__(size, epoch, interval, var=var)
        Base.__init__(self)

    def force(self, state):
        return 0

    def init_state(self):
        """Initialize the location and velocity of nodes."""
        loc = np.random.randn(self.size, 2) * self.loc_std
        vel = np.random.randn(self.size, 2)
        v_norm = np.sqrt((vel ** 2).sum(axis=1)).reshape(-1, 1)
        # normalize the velocity
        vel = vel * self.vel_norm / v_norm
        loc, vel = self._clamp(loc, vel)
        state = np.hstack((loc, vel))
        return state

    def update_state(self, state: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Given the node states, compute the forces between node pairs, and update the node states.
        """
        n = self.size
        force = self.force(state) * adj
        loc, vel = state[:, :2], state[:, 2:]
        r = np.concatenate([np.subtract.outer(loc[:, i],
                            loc[:, i]).reshape(1, n, n)
                            for i in range(loc.shape[-1])])
        F = (force.reshape(1, n, n) * r).sum(axis=-1).T
        # clamp the value of the force
        F[F > self._max_F] = self._max_F
        F[F < -self._max_F] = -self._max_F
        # update the velocity
        vel += self._delta_T * F
        # update the location
        loc += self._delta_T * vel
        loc, vel = self._clamp(loc, vel)
        state = np.hstack((loc, vel))
        return state


class Spring(BaseSim):
    """Simulator for Springs dataset."""
    def __init__(self, size: int, epoch: int, interval: int, var: float=0):
        """
        Args:
            size: number of nodes
            epoch: simulation steps
            interval: down-sampling factor
            var: variance of the noise, default: 0
        """
        Simulator.__init__(self, size, epoch, interval, var=var)
        Base.__init__(self)

    def force(self, state):
        return - self.interaction_strength


class Charged(BaseSim):
    """Simulator for Charged dataset."""
    def __init__(self, size: int, epoch: int, interval: int, var: float=0):
        """
        Args:
            size: number of nodes
            epoch: simulation steps
            interval: down-sampling factor
            var: variance of the noise, default: 0
        """
        Simulator.__init__(self, size, epoch, interval, var=var)
        self.init_hyper()

    def init_hyper(self):
        self.box_size = 5
        self.loc_std = 1
        self.vel_norm = 0.5
        self.interaction_strength = 1
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def init_adj(self):
        """
        Initialize the adjacency matrix among charged particles. Each element takes value from {-1, 1}, representation repel or attraction, respectively.
        """
        charges = np.random.choice([1, -1], size=(self.size, 1))
        adj = charges @ charges.T
        np.fill_diagonal(adj, 0)
        return adj

    def force(self, state: np.ndarray) -> np.ndarray:
        """
        Forces between two charged particles.
        
        Args:
            state: node states

        Return:
            force: forces between node pairs
        """
        with np.errstate(divide='ignore'):
            loc = state[:, :2]
            dist = distance_pairwise(loc, loc) ** 3
            np.fill_diagonal(dist, 1)
            force = self.interaction_strength / dist
            return force
