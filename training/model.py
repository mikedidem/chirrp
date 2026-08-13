#!/usr/bin/env python
"""
model.py  —  Parameterized PINN for unconfined groundwater flow.

Architecture follows the paper: Q_norm is concatenated directly as a 4th
input (x, y, t, Q_norm), no separate parameter encoder.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from problem import Problem
from options import Options


class Net(nn.Module):
    """
    Sin-residual network with Q concatenated as 4th input.

    layers = [4, hidden, hidden, ..., hidden, 1]
    Input : (x_norm, y_norm, t_norm, Q_norm)  — all in [-1, 1]
    Output: hydraulic head h
    """

    def __init__(self, args, stage=1):
        super().__init__()
        self.args = args
        self.tau = args.tau
        self.layers = args.layers           # e.g. [4, 50, 50, 50, 50, 50, 1]
        self.scale = args.scale
        self.device = args.device
        self.constraint = args.constraint
        self.problem = args.problem
        self.stage = stage
        self.fcs = []
        self.params = []

        # Hidden layers with learnable sin-activation scale
        for i in range(len(self.layers) - 2):
            fc = nn.Linear(self.layers[i], self.layers[i + 1])
            setattr(self, f'fc{i + 1}', fc)
            self._init_weights(fc)
            self.fcs.append(fc)

            param = nn.Parameter(torch.randn(self.layers[i + 1]))
            setattr(self, f'param{i + 1}', param)
            self.params.append(param)

        # Output layer (linear, no sin)
        fc = nn.Linear(self.layers[-2], self.layers[-1])
        setattr(self, f'fc{len(self.layers) - 1}', fc)
        self._init_weights(fc)
        self.fcs.append(fc)

    def _init_weights(self, layer):
        init.xavier_normal_(layer.weight)
        init.constant_(layer.bias, 0.01)

    def forward(self, xyt, Q_norm, hstar=None):
        """
        Parameters
        ----------
        xyt    : (N, 3) tensor  —  [x, y, t]
        Q_norm : float or (N, 1) tensor  —  normalized pumping rate in [-1, 1]
        hstar  : (N, 1) tensor  —  Stage 1 solution at tau (Stage 2 only)
        """
        N = xyt.shape[0]

        # Expand Q_norm to (N, 1)
        if isinstance(Q_norm, (int, float)):
            q = torch.full((N, 1), float(Q_norm), dtype=torch.float32,
                           device=xyt.device)
        elif isinstance(Q_norm, torch.Tensor) and Q_norm.dim() == 0:
            q = Q_norm.view(1, 1).expand(N, 1)
        else:
            q = Q_norm

        # Normalize spatial-temporal coordinates to [-1, 1]
        xmin, xmax, ymin, ymax, tmin, tmax = self.problem.domain
        if self.stage == 1:
            tmax = self.tau
        elif self.stage == 2:
            tmin = self.tau

        lb = torch.tensor([xmin, ymin, tmin], dtype=torch.float32,
                          device=xyt.device)
        ub = torch.tensor([xmax, ymax, tmax], dtype=torch.float32,
                          device=xyt.device)
        X_norm = 2.0 * (xyt - lb) / (ub - lb) - 1.0

        # Concatenate normalized coordinates with Q_norm  →  (N, 4)
        X = torch.cat([X_norm, q], dim=-1)

        # First hidden layer
        X = self.fcs[0](X)
        X = torch.mul(self.params[0], X) * self.scale
        X = torch.sin(X)

        # Residual hidden layers
        for i in range(1, len(self.fcs) - 1):
            res = self.fcs[i](X)
            res = torch.mul(self.params[i], res) * self.scale
            res = torch.sin(res)
            X = X + res

        # Output layer (no activation)
        u = self.fcs[-1](X)

        # Apply boundary constraint
        if self.constraint == 'SOFT':
            h = u

        elif self.constraint == 'HARD':
            hstar_val = 90.0 if hstar is None else hstar
            y = xyt[:, [1]]
            t = xyt[:, [2]]
            d = ((t - tmin) * (ymax - y) * (y - ymin)) / \
                ((tmax - tmin) * (ymax - ymin) ** 2)
            h = hstar_val + d * u

        return h


# ---------------------------------------------------------------------------
# Neumann boundary wrapper
# ---------------------------------------------------------------------------

class Net_Neumann(nn.Module):
    """Computes ∂h/∂x on the Neumann (no-flow) boundary."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.stage = net.stage

    def forward(self, xyt_bdy2, Q_norm, hstar_x_bdy2=None):
        xyt_bdy2.requires_grad_(True)
        h = self.net(xyt_bdy2, Q_norm)

        w = torch.ones_like(xyt_bdy2[:, [0]])
        h_x_bdy2 = torch.autograd.grad(
            h, xyt_bdy2, w, create_graph=True)[0][:, [0]]

        xyt_bdy2.detach_()

        if self.stage == 1:
            return h_x_bdy2
        elif self.stage == 2:
            return hstar_x_bdy2 + h_x_bdy2


# ---------------------------------------------------------------------------
# PDE residual wrapper
# ---------------------------------------------------------------------------

class Net_PDE(nn.Module):
    """Evaluates the unconfined groundwater (Boussinesq) PDE residual."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.problem = net.problem
        self.stage = net.stage
        self.device = net.device

    def forward(self, xyt, Q_norm, Q_scalar, hstar_diff=None, out_diff=False):
        """
        Parameters
        ----------
        xyt        : (N, 3)  interior collocation points
        Q_norm     : float   normalized pumping rate in [-1, 1]
        Q_scalar   : float   actual pumping rate in m^3/day (for problem.f)
        hstar_diff : (N, 1)  diffusion term from Stage 1 (Stage 2 only)
        out_diff   : bool    if True, return only the diffusion term
        """
        xyt.requires_grad_(True)
        h = self.net(xyt, Q_norm)

        w = torch.ones_like(xyt[:, [0]])

        h_grad = torch.autograd.grad(h, xyt, w, create_graph=True)[0]
        h_x, h_y = h_grad[:, [0]], h_grad[:, [1]]

        h_xx = torch.autograd.grad(
            h_x, xyt, w, create_graph=True)[0][:, [0]]
        h_yy = torch.autograd.grad(
            h_y, xyt, w, create_graph=True)[0][:, [1]]
        h_t = h_grad[:, [2]]

        xyt.detach_()

        f = self.problem.f(xyt.cpu().numpy(), Q=Q_scalar)
        f = torch.from_numpy(f).float().to(self.device)

        mu = self.problem.mu
        K = self.problem.K

        diffusion = -K * (h * (h_xx + h_yy) + h_x * h_x + h_y * h_y)

        if hstar_diff is not None:
            return mu * h_t + diffusion - f + hstar_diff
        else:
            if out_diff:
                return diffusion
            return mu * h_t + diffusion - f


# ---------------------------------------------------------------------------
# Full PINN
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.constraint = net.constraint
        self.stage = net.stage
        self.net = net
        self.net_Neumann = Net_Neumann(net)
        self.net_PDE = Net_PDE(net)

    def forward(self, xyt, xyt_bdy2, Q_norm, Q_scalar,
                xy0=None, xyt_bdy1=None,
                hstar=None, hstar_diff=None, hstar_x_bdy2=None,
                out_diff=False):

        h_x_bdy2 = self.net_Neumann(xyt_bdy2, Q_norm,
                                    hstar_x_bdy2=hstar_x_bdy2)
        res = self.net_PDE(xyt, Q_norm, Q_scalar,
                           hstar_diff=hstar_diff, out_diff=out_diff)

        if self.constraint == 'HARD':
            return res, h_x_bdy2

        elif self.constraint == 'SOFT':
            h0 = self.net(xy0, Q_norm)
            h_bdy1 = self.net(xyt_bdy1, Q_norm)
            return res, h0, h_bdy1, h_x_bdy2


if __name__ == '__main__':
    args = Options().parse()
    args.problem = Problem(sigma=args.sigma)
    net = Net(args)
    print(net)
    total = sum(p.numel() for p in net.parameters())
    print(f'Total parameters: {total:,}')
