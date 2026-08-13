#!/usr/bin/env python
import argparse
import torch


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda', action='store_true', default=False)
        parser.add_argument('--cuda_index', type=int, default=0)
        parser.add_argument('--seed', type=int, default=200)
        parser.add_argument('--scale', type=float, default=1.0,
                            help='Scale in adaptive activation function')
        parser.add_argument('--hidden_layers', type=int, default=5)
        parser.add_argument('--hidden_neurons', type=int, default=50)
        parser.add_argument('--stage', type=int, default=1)
        parser.add_argument('--tau', type=float, default=1.0,
                            help='Time watershed between stage 1 and stage 2')
        parser.add_argument('--constraint', type=str, default='HARD',
                            help='HARD or SOFT')
        parser.add_argument('--spatial_strategy', type=str, default='LR')
        parser.add_argument('--temporal_strategy', type=str, default='LHS')
        parser.add_argument('--temporal_strategy_prev', type=str, default='UNIFORM')
        parser.add_argument('--n', type=int, default=None)
        parser.add_argument('--nx', type=int, default=None)
        parser.add_argument('--ny', type=int, default=None)
        parser.add_argument('--nt', type=int, default=None)
        parser.add_argument('--nt_prev', type=int, default=None)
        parser.add_argument('--ratio', type=float, default=None)
        parser.add_argument('--filename', type=str, default=None)
        parser.add_argument('--sigma', type=float, default=30.0,
                            help='Sigma in Gaussian well approximation')
        parser.add_argument('--lam', type=float, default=100,
                            help='Weight on boundary/IC losses')
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--epochs_Adam', type=int, default=2000)
        parser.add_argument('--resume', type=str, default=None)

        # --- Parameterization ---
        parser.add_argument('--Q_min', type=float, default=-50000.0,
                            help='Minimum pumping rate (m^3/day)')
        parser.add_argument('--Q_max', type=float, default=-25000.0,
                            help='Maximum pumping rate (m^3/day)')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.device = torch.device(
            f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')

        # 4 inputs (x, y, t, Q_norm) → hidden layers → 1 output
        args.layers = [4] + args.hidden_layers * [args.hidden_neurons] + [1]

        return args
