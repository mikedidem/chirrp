#!/usr/bin/env python
import numpy as np


class Problem(object):
    def __init__(self,
                 sigma=25.0,
                 domain=(-500, 500, -500, 500, 0.0, 30.0),
                 xi=(201.67, -98.33)):
        self.sigma = sigma
        self.domain = domain
        self.xi = xi
        self.mu = 0.1
        self.K = 33.33

    def __repr__(self):
        return f'well problem with point source'

    def f(self, x, Q=-4.e4):
        """
        Source term for the pumping well.

        Parameters
        ----------
        x : ndarray, shape (N, 3)
            Collocation points [x, y, t].
        Q : float
            Pumping rate in m^3/day (negative = extraction).
            Range used for training: Q_MIN to Q_MAX (see options.py).
        """
        f = Q * np.exp(-((x[:, [0]] - self.xi[0])**2 +
                         (x[:, [1]] - self.xi[1])**2) / (2 * self.sigma**2)) / \
            (2 * np.pi * self.sigma**2)
        return f

    def bc(self, x, mode=0):
        """Boundary / initial condition."""
        if mode == 0:   # initial condition
            return 90 * np.ones_like(x[:, [0]])
        elif mode == 1: # Dirichlet BC
            return 90 * np.ones_like(x[:, [0]])
        elif mode == 2: # Neumann BC
            return 0.0 * np.ones_like(x[:, [0]])
