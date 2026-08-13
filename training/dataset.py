#!/usr/bin/env python
from options import Options
from utils import tile
from pyDOE import lhs
from sampler import Sampler
from problem import Problem
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class Trainset(object):
    """Generate DataSet for Training

    Params
    ------
    problem  (class Problem)
        used for determine some properties of the problem

    filename (class str) mesh filename
        used for generate interior and boundary points (nodes) """

    def __init__(self, problem, stage=1, tau=1.0,
                 spatial_strategy='LR',
                 temporal_strategy='UNIFORM',
                 nx=100, ny=100, nt=50, n=None,
                 ratio=None, filename=None):
        """
        Obtain the interior points and boundary points from specified file using Dataprocessing

        Members:
        -------
        self.xyt:      (n, 2) ndarray

        self.xy0:      (n0, 2) ndarray
        self.u0:       (n0, 1) ndarray

        self.xyt_bdy1: (n_bdy1, 2) ndarray
        self.u_bdy1:   (n_bdy1, 2) ndarray

        self.xyt_bdy2: (n_bdy2, 2) ndarray
        self.u_bdy2:   (n_bdy2, 2) ndarray
        """
        self.problem = problem
        self.spatial_strategy = spatial_strategy
        self.temporal_strategy = temporal_strategy
        self.n, self.nx, self.ny, self.nt = n, nx, ny, nt
        self.ratio = ratio
        self.filename = filename

        self.sampler = Sampler(problem, stage=stage, tau=tau)
        self.xyt, self.xy0, self.xyt_bdy1, self.xyt_bdy2 = self.sampler(spatial_strategy=spatial_strategy,
                                                                        temporal_strategy=temporal_strategy,
                                                                        nx=nx, ny=ny, nt=nt, n=n,
                                                                        ratio=ratio, filename=filename)
        self.xyt = torch.from_numpy(self.xyt).float()

        self.u0 = problem.bc(self.xy0, mode=0)
        self.u0 = torch.from_numpy(self.u0).float()
        self.xy0 = torch.from_numpy(self.xy0).float()

        self.u_bdy1 = problem.bc(self.xyt_bdy1, mode=1)
        self.u_bdy1 = torch.from_numpy(self.u_bdy1).float()
        self.xyt_bdy1 = torch.from_numpy(self.xyt_bdy1).float()

        self.u_bdy2 = problem.bc(self.xyt_bdy2, mode=2)
        self.u_bdy2 = torch.from_numpy(self.u_bdy2).float()
        self.xyt_bdy2 = torch.from_numpy(self.xyt_bdy2).float()

    def spatial(self):
        xy, xy_bdy1, xy_bdy2 = self.sampler.spatial(strategy=self.spatial_strategy,
                                                    nx=self.nx,
                                                    ny=self.ny,
                                                    n=self.n,
                                                    filename=self.filename)
        return xy, xy_bdy1, xy_bdy2

    def temporal(self, t):
        """generation of temporal sampling points"""
        if isinstance(t, int):
            _, _, _, _, t_min, t_max = self.problem.domain
            result = np.linspace(t_min, t_max, t+1)[1:]
        elif isinstance(t, (float, list, tuple, np.ndarray)):
            result = t

        return result

    def spatial_temporal(self, xy, t):
        """Generation of spatial_temporal sampling points"""
        X, T = np.meshgrid(xy[:, [0]], t)
        Y, T = np.meshgrid(xy[:, [1]], t)
        return np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T

    def __repr__(self):
        res = f'''************** Trainset *****************\n''' + \
            f'''Spatial INFO: {self.spatial_strategy}, n={self.n}, nx={self.nx}, ny={self.ny}, file={self.filename}\n''' + \
            f'''Temporal INFO: {self.temporal_strategy}, nt={self.nt}, ratio={self.ratio}\n''' + \
            f'''n_xyt      = {self.xyt.shape[0]}\n''' + \
            f'''n_xy0      = {self.xy0.shape[0]}\n''' + \
            f'''n_xyt_bdy1 = {self.xyt_bdy1.shape[0]}\n''' + \
            f'''n_xyt_bdy2 = {self.xyt_bdy2.shape[0]}\n''' + \
            f'''****************************************'''
        return res

    def __call__(self, mode=None):
        if mode == 0:
            return self.xy0, self.u0
        elif mode == 1:
            return self.xyt_bdy1, self.u_bdy1
        elif mode == 2:
            return self.xyt_bdy2, self.u_bdy2
        elif mode is None:
            return self.xyt


class Validset(object):
    """
    Generate DataSet for Training

    Params:
    ------
    problem (class Problem)
        used for determine the solution domain

    nx, ny  (class int)
        the number of partitions in x and y direction

    t: (int, float, list, tuple, ndarray)
       int: number of temporal points to get uniform points
       (float, list, tuple, ndarray): given temporal points

    """

    def __init__(self, problem, nx, ny, t):
        """Obtain the sample points for validation

        Params
        ======
        size:
        """
        self.problem = problem
        self.nx = nx
        self.ny = ny
        self.t = t

    def spatial(self, grid=False):
        """generation of spatial sampling points"""
        x_min, x_max, y_min, y_max, _, _ = self.problem.domain

        x = np.linspace(x_min, x_max, self.nx)
        y = np.linspace(y_min, y_max, self.ny)
        grid_x, grid_y = np.meshgrid(x, y)
        x_, y_ = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
        xy = np.c_[x_, y_]

        if grid:
            return grid_x, grid_y
        return xy

    def temporal(self):
        """generation of temporal sampling points"""
        if isinstance(self.t, int):
            _, _, _, _, t_min, t_max = self.problem.domain
            t = np.linspace(t_min, t_max, self.t+1)[1:]
        elif isinstance(self.t, (float, list, tuple, np.ndarray)):
            t = self.t

        return t

    def spatial_temporal(self):
        """generation of spatial-temporal sampling points"""
        x_min, x_max, y_min, y_max, t_min, t_max = self.problem.domain

        x = np.linspace(x_min, x_max, self.nx)
        y = np.linspace(y_min, y_max, self.ny)
        X, Y = np.meshgrid(x, y)
        xy = np.vstack([X.ravel(), Y.ravel()]).T

        if isinstance(self.t, int):
            _, _, _, _, t_min, t_max = self.problem.domain
            t = np.linspace(t_min, t_max, self.t+1)[1:]
        elif isinstance(self.t, (float, list, tuple, np.ndarray)):
            t = self.t

        X, T = np.meshgrid(xy[:, [0]], t)
        Y, T = np.meshgrid(xy[:, [1]], t)
        return np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T

    def __call__(self):
        xyt = self.spatial_temporal()
        return torch.from_numpy(xyt).float()

    def __repr__(self):
        res = f'''*********** Validset *************\n''' + \
            f'''Spatial INFO: nx={self.nx}, ny={self.ny}\n''' + \
            f'''Temporal INFO: t = {self.temporal()}\n''' + \
            f'''*********************************\n'''
        return res


if __name__ == '__main__':
    args = Options().parse()
    problem = Problem()

    # Generate Train Dataset

    trainset = Trainset(problem,
                        spatial_strategy='LR', filename='./data/well.mat')
    print(trainset)
    xy, xy_bdy1, xy_bdy2 = trainset.spatial()
    print(xy.shape, xy_bdy1.shape, xy_bdy2.shape)
    tau = [1.0, 2.0]
    t = trainset.temporal(tau)
    print(t)
    xytau = trainset.spatial_temporal(xy, t)
    xytau_bdy1 = trainset.spatial_temporal(xy_bdy1, t)
    xytau_bdy2 = trainset.spatial_temporal(xy_bdy2, t)
    print(xytau.shape, xytau_bdy1.shape, xytau_bdy2.shape)
    print(xytau)
    # trainset = Trainset(problem,
    #                     spatial_strategy='UNIFORM', nx=10, ny=10,
    #                     temporal_strategy='UNIFORM', nt=50)

    # trainset = Trainset(problem,
    #                     spatial_strategy='LHS', n=50, nx=10, ny=10,
    #                     temporal_strategy='LHS', nt=50)

    # print(trainset)

    # print('Trainset')
    # epochs = 2
    # for epoch in range(epochs):
    #     print(f"Epoch: {epoch+1}/{epochs}")
    #     xyt = trainset()
    #     print(xyt.shape)

    #     xy0, u0 = trainset(0)
    #     print(xy0.shape, u0.shape)

    #     xyt_bdy1, u_bdy1 = trainset(1)
    #     print(xyt_bdy1.shape, u_bdy1.shape)

    #     xyt_bdy2, u_bdy2 = trainset(2)
    #     print(xyt_bdy2.shape, u_bdy2.shape)

    # Generate Validate Dataset
    print('\nValidset')
    validset = Validset(problem, 5, 5, [1, 2])
    xyt = validset.spatial_temporal()
    print(xyt, xyt.shape)

    # # Generate Test Dataset
    # print('\nTestset')
    # test_set = Testset()
    # x3 = test_set()
    # x4 = x3[0:14, :]

    # print('x', x4)
