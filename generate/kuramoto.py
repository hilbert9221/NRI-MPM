"""
Re-implementation of Kuramoto simulator in NRI.
"""
import numpy as np
from generate.dynamics import Simulator
from scipy.integrate._ode import ode


class Kuramoto(Simulator):
    """Simulator for Kuramoto dataset."""
    def __init__(self, size: int, epoch: int, interval: int, var: float=0):
        """
        Args:
            size: number of nodes
            epoch: simulation steps
            interval: down-sampling factor
            var: variance of the noise, default: 0
        """
        super().__init__(size, epoch, interval, var=var)

    def set_trunc(self, steps):
        self.trunc = steps

    def init_adj(self):
        """
        Initialize the adjacency matrix.

        Return:
            adj: [1, size, size]
        """
        adj = super().init_adj()
        return adj[None, :]

    def init_state(self):
        """
        Initialize variables for integration.

        Return:
            dt: time interval
            t: time steps
            y: phase
            w: intrinsic frequency
        """
        dt = 0.01
        # abandon the last state
        t = [i * dt for i in range((self.epoch + 1) * self.interval)]
        # intrinsic frequencies
        w = np.random.uniform(1, 10, self.size)
        # initial phase
        y = np.random.uniform(0, 2 * np.pi, self.size)
        return dt, t, y, w
        
    def kuramoto_ODE(self, t: list, y: np.ndarray, arg: tuple) -> np.ndarray:
        """
        General Kuramoto ODE of m'th harmonic order.

        Args:
            t: time steps
            y: phase
            arg: (w, k)
                w: iterable frequency
                k: 3D coupling matrix, unless 1st order
            
        Return:
            delta_phase: increment in phase 
        """
        w, k = arg
        yt = y[:, None]
        dy = y-yt
        # NOTE: deep copy
        delta_phase = w.copy()
        for m, _k in enumerate(k):
            delta_phase += np.sum(_k*np.sin((m+1)*dy), axis=1)

        return delta_phase

    def kuramoto_ODE_jac(self, t: list, y: np.ndarray, arg: tuple) -> np.ndarray:
        """
        Kuramoto's Jacobian passed for ODE solver.

        Args:
            t: time steps
            y: phase
            arg: (w, k)
                w: iterable frequency
                k: 3D coupling matrix, unless 1st order
            
        Return:
            jacobian_phase: jacobian w.r.t. phase 
        """
        w, k = arg
        yt = y[:, None]
        dy = y-yt

        jacobian_phase = [m*k[m-1]*np.cos(m*dy) for m in range(1, 1+k.shape[0])]
        jacobian_phase = np.sum(jacobian_phase, axis=0)

        for i in range(self.size):
            jacobian_phase[i, i] = -np.sum(jacobian_phase[:, i])

        return jacobian_phase

    def solve(self, t: list, y: np.ndarray, w: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Solves Kuramoto ODE for time series `t` with initial parameters passed when initiated object.

        General Kuramoto ODE of m'th harmonic order.

        Args:
            t: time steps
            y: phase
            w: iterable frequency
            k: 3D coupling matrix, unless 1st order
            
        Return:
            phase: output of the integrator
        """
        dt = t[1]-t[0]

        kODE = ode(self.kuramoto_ODE, jac=self.kuramoto_ODE_jac)
        # 5th-order Runge-Kutta methods
        kODE.set_integrator("dopri5")

        # Set parameters into model
        kODE.set_initial_value(y.copy(), t[0])
        kODE.set_f_params((w.copy(), k.copy()))
        kODE.set_jac_params((w.copy(), k.copy()))

        phase = np.empty((self.size, len(t)))

        # Run ODE integrator
        for idx, _t in enumerate(t[1:]):
            phase[:, idx] = kODE.y
            kODE.integrate(_t)

        phase[:, -1] = kODE.y

        return phase

    def min_max(self, x: np.ndarray, axis: float=0) -> np.ndarray:
        """
        Dimension-wise min-max normalization.
        """
        m = x.min(axis, keepdims=True)
        M = x.max(axis, keepdims=True)
        h = (x - m) / (M - m)
        h = h * 2 - 1
        return h
    
    def generate(self):
        """
        Simulate a time series for a dynamical with (size) nodes over (self.epoch // self.interval) time steps.

        Return:
            adj: [size, size]
            states: [epoch, size, dim]
        """
        # NOTE: set random seeds every time to avoid repetition in multi-processing generation
        np.random.seed()
        k = self.init_adj()
        dt, t, y, w = self.init_state()
        omega = w.copy()
        sol = self.solve(t, y, w, k)
        # sol: [step, node]
        sol = sol.T
        delta = (sol[1:] - sol[:-1]) / dt
        omegas = np.tile(omega[:, None], self.epoch).T
        # down sampling
        sol = sol[:-1][::self.interval]
        delta = delta[::self.interval]
        amplitude = np.sin(sol)

        # post-processing
        sol = self.min_max(sol)
        delta = self.min_max(delta)
        omegas = (omegas - 1) * 2 / (10 - 1) - 1
        isolated = np.where(k[0].sum(1) == 0)[0]
        delta[:, isolated] = 0

        sol = sol[:self.epoch]
        delta = delta[:self.epoch]
        amplitude = amplitude[:self.epoch]

        states = np.array([delta, amplitude, sol, omegas])
        states = states.transpose((1, 2, 0))
        if hasattr(self, 'trunc'):
            states = states[:self.trunc]
        adj = k.squeeze(0)
        return adj, states
