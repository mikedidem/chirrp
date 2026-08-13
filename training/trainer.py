#!/usr/bin/env python
"""
trainer.py  —  Parameterized PINN trainer.

Key difference from pure_pinns/trainer.py
------------------------------------------
Each training step samples a random Q from [Q_min, Q_max], normalizes it
to [-1, 1], and passes it through the network.  No MODFLOW data is used —
only physics (PDE + BC + IC losses).
"""
from options import Options
from utils import save_checkpoints, show_surface, show_contours_2x2, mae, mse, rrmse
from model import Net, Net_Neumann, Net_PDE, PINN
from dataset import Trainset, Validset
from sampler import Sampler
from problem import Problem

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

import time
import os
import shutil


def normalize_Q(Q_scalar, Q_min, Q_max):
    """Map Q_scalar ∈ [Q_min, Q_max] to Q_norm ∈ [-1, 1]."""
    return 2.0 * (Q_scalar - Q_min) / (Q_max - Q_min) - 1.0


def sample_Q(Q_min, Q_max):
    """Uniform sample from training Q range."""
    return float(np.random.uniform(Q_min, Q_max))


class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.cuda_index = args.cuda_index
        self.problem = Problem(sigma=args.sigma)
        self.constraint = args.constraint
        self.stage = args.stage
        self.tau = args.tau
        self.Q_min = args.Q_min
        self.Q_max = args.Q_max

        # Model name
        name = (f"{args.constraint}_{args.hidden_layers}x{args.hidden_neurons}"
                f"_tau:{self.tau:.0f}_sigma:{args.sigma:.0f}"
                f"_S:{args.spatial_strategy}")
        self.model_name = f"Stage{self.stage}_{name}_T:{args.temporal_strategy}_nt:{args.nt}"
        if self.stage > 1:
            self.model_name_prev = (f"Stage{self.stage - 1}_{name}"
                                    f"_T:{args.temporal_strategy_prev}_nt:{args.nt_prev}")

        # Networks
        self.net = Net(self.args, stage=self.stage)
        self.net_neumann = Net_Neumann(self.net)
        self.net_pde = Net_PDE(self.net)
        self.pinn = PINN(self.net)

        if self.stage > 1:
            self.net_prev = Net(self.args, stage=self.stage - 1)
            self.net_neumann_prev = Net_Neumann(self.net_prev)
            self.net_pde_prev = Net_PDE(self.net_prev)
            self.pinn_prev = PINN(self.net_prev)

            if self.device != torch.device('cpu'):
                self.net_prev.to(self.device)
                self.net_pde_prev.to(self.device)
                self.pinn_prev.to(self.device)

            self.net_prev.eval()
            self.net_pde_prev.eval()
            self.pinn_prev.eval()

            best_model = torch.load(
                f'checkpoints/{self.model_name_prev}/best_model.pth.tar',
                map_location=self.device)
            self.pinn_prev.load_state_dict(best_model['state_dict'])

        # Criterion
        self.criterion = nn.MSELoss()

        # Resume
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'Resuming training, loading {args.resume} ...')
                self.pinn.load_state_dict(
                    torch.load(args.resume, map_location=self.device)['state_dict'])
            else:
                print('Resume path not found:', args.resume)

        # Trainset and Validset
        self.trainset = Trainset(self.problem, stage=self.stage, tau=self.tau,
                                 spatial_strategy=args.spatial_strategy,
                                 temporal_strategy=args.temporal_strategy,
                                 n=args.n, nx=args.nx, ny=args.ny, nt=args.nt,
                                 ratio=args.ratio, filename='./data/well.mat')

        self.validsize = (100, 100)
        if self.stage == 1:
            time_stamps = [0.25, 0.5, 0.75, 1]
        elif self.stage == 2:
            time_stamps = [20, 25, 29, 30]
        self.validset = Validset(
            self.problem, self.validsize[0], self.validsize[1], time_stamps)

        # For Stage 2: precompute spatial points at tau (Q-independent geometry)
        if self.stage > 1:
            xy, _, xy_bdy2 = self.trainset.spatial()
            xytau = self.trainset.spatial_temporal(xy, self.tau)
            xytau_bdy2 = self.trainset.spatial_temporal(xy_bdy2, self.tau)
            self.xytau = torch.from_numpy(xytau).float()
            self.xytau_bdy2 = torch.from_numpy(xytau_bdy2).float()

            xytau_valid = Validset(
                self.problem, self.validsize[0], self.validsize[1], self.tau)()
            self.xytau_valid = xytau_valid

            if self.device != torch.device('cpu'):
                self.xytau = self.xytau.to(self.device)
                self.xytau_bdy2 = self.xytau_bdy2.to(self.device)
                self.xytau_valid = self.xytau_valid.to(self.device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_hstar(self, xytau, Q_norm, Q_scalar, nt_repeat):
        """Compute hstar, hstar_diff, hstar_x_bdy2 from Stage 1 for a given Q."""
        with torch.no_grad():
            hstar = self.net_prev(xytau, Q_norm).detach()

        # hstar_diff needs autograd enabled (computes h_xx, h_yy via second-order grad)
        hstar_diff = self.net_pde_prev(
            xytau.clone(), Q_norm, Q_scalar, out_diff=True).detach()

        hstar = hstar.repeat(nt_repeat, 1)
        hstar_diff = hstar_diff.repeat(nt_repeat, 1)
        return hstar, hstar_diff

    def _get_hstar_x_bdy2(self, xytau_bdy2, Q_norm, nt_repeat):
        # needs autograd enabled — computes ∂h/∂x via torch.autograd.grad
        hstar_x_bdy2 = self.net_neumann_prev(
            xytau_bdy2.clone(), Q_norm).detach()
        return hstar_x_bdy2.repeat(nt_repeat, 1)

    def train_info(self, optimizer, epoch, train_loss, valid_loss, tt):
        result = f'{optimizer:5s} '
        result += f'{epoch + 1:5d}/{self.epochs_Adam:5d} '
        result += f'train_loss: {train_loss:.4e} '
        result += f'valid_loss: {valid_loss:.4e} '
        result += f'time: {time.time() - tt:5.2f} '
        if optimizer == 'Adam':
            result += f'lr: {self.lr_scheduler.get_last_lr()[0]:.2e}'
        print(result)

    # ------------------------------------------------------------------
    # Main train entry
    # ------------------------------------------------------------------

    def train(self):
        self.epochs_Adam = self.args.epochs_Adam
        self.lam = self.args.lam

        self.writer = SummaryWriter(comment=f'_{self.model_name}')

        self.lr = self.args.lr
        self.optimizer_Adam = optim.Adam(
            [p for p in self.pinn.parameters() if p.requires_grad],
            lr=self.lr)
        self.lr_scheduler = StepLR(self.optimizer_Adam, step_size=2000, gamma=0.1)

        print(f'{self.trainset}')
        print(f'Q range: [{self.Q_min}, {self.Q_max}] m^3/day')

        # Collocation points
        xyt = self.trainset()
        xyt_bdy2, hx_bdy2 = self.trainset(mode=2)

        if self.constraint == 'SOFT':
            xy0, u0 = self.trainset(mode=0)
            xyt_bdy1, h_bdy1 = self.trainset(mode=1)

        best_loss = 1.0e10
        tt = time.time()
        self.pinn.train()
        step = 0

        # GPU transfer
        if self.device != torch.device('cpu'):
            xyt = xyt.to(self.device)
            xyt_bdy2 = xyt_bdy2.to(self.device)
            hx_bdy2 = hx_bdy2.to(self.device)
            if self.constraint == 'SOFT':
                xy0 = xy0.to(self.device)
                u0 = u0.to(self.device)
                xyt_bdy1 = xyt_bdy1.to(self.device)
                h_bdy1 = h_bdy1.to(self.device)
            self.net.to(self.device)
            self.net_pde.to(self.device)
            self.pinn.to(self.device)

        self.xyt = xyt
        self.xyt_bdy2 = xyt_bdy2
        self.hx_bdy2 = hx_bdy2
        if self.constraint == 'SOFT':
            self.xy0 = xy0
            self.u0 = u0
            self.xyt_bdy1 = xyt_bdy1
            self.h_bdy1 = h_bdy1

        # nt_repeat for Stage 2 hstar replication
        if self.stage > 1:
            self.nt_repeat = self.args.nt - 1

        # Adam
        for epoch in range(self.epochs_Adam):
            train_loss = self.train_Adam(epoch)

            if (epoch + 1) % 100 == 0:
                step += 1
                valid_loss = self.validate(step)
                self.train_info('Adam', epoch, train_loss, valid_loss, tt)
                tt = time.time()
                self.pinn.train()

                is_best = valid_loss < best_loss
                if is_best:
                    best_loss = valid_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.pinn.state_dict(),
                    'best_loss': best_loss,
                }
                save_checkpoints(state, is_best, save_dir=self.model_name)

        self.writer.close()
        print('Training finished successfully!\n')

    # ------------------------------------------------------------------
    # Adam step
    # ------------------------------------------------------------------

    def train_Adam(self, epoch):
        self.optimizer_Adam.zero_grad()

        # Sample Q for this step
        Q_scalar = sample_Q(self.Q_min, self.Q_max)
        Q_norm = normalize_Q(Q_scalar, self.Q_min, self.Q_max)

        if self.constraint == 'HARD':
            if self.stage == 1:
                res, hx_bdy2_pred = self.pinn(
                    self.xyt, self.xyt_bdy2, Q_norm, Q_scalar)
            else:
                hstar, hstar_diff = self._get_hstar(
                    self.xytau, Q_norm, Q_scalar, self.nt_repeat)
                hstar_x_bdy2 = self._get_hstar_x_bdy2(
                    self.xytau_bdy2, Q_norm, self.nt_repeat)
                if self.device != torch.device('cpu'):
                    hstar = hstar.to(self.device)
                    hstar_diff = hstar_diff.to(self.device)
                    hstar_x_bdy2 = hstar_x_bdy2.to(self.device)
                res, hx_bdy2_pred = self.pinn(
                    self.xyt, self.xyt_bdy2, Q_norm, Q_scalar,
                    hstar=hstar, hstar_diff=hstar_diff,
                    hstar_x_bdy2=hstar_x_bdy2)

            loss = self.criterion(res, torch.zeros_like(res))
            loss_bdy2 = self.criterion(hx_bdy2_pred, self.hx_bdy2)
            loss_total = loss + self.lam * loss_bdy2

        elif self.constraint == 'SOFT':
            res, u0_pred, h_bdy1_pred, hx_bdy2_pred = self.pinn(
                self.xyt, self.xyt_bdy2, Q_norm, Q_scalar,
                xy0=self.xy0, xyt_bdy1=self.xyt_bdy1)

            loss = self.criterion(res, torch.zeros_like(res))
            loss0 = self.criterion(u0_pred, self.u0)
            loss_bdy1 = self.criterion(h_bdy1_pred, self.h_bdy1)
            loss_bdy2 = self.criterion(hx_bdy2_pred, self.hx_bdy2)
            loss_total = loss + self.lam * (loss0 + loss_bdy1 + loss_bdy2)

        loss_total.backward()
        self.optimizer_Adam.step()
        self.lr_scheduler.step()

        self.writer.add_scalar('train_loss', loss_total.item(), epoch)
        return loss_total.item()

    # ------------------------------------------------------------------
    # Validation  (uses mid-range Q for monitoring)
    # ------------------------------------------------------------------

    def validate(self, step):
        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()

        # Use mid-range Q for validation monitoring
        Q_scalar_val = (self.Q_min + self.Q_max) / 2.0
        Q_norm_val = normalize_Q(Q_scalar_val, self.Q_min, self.Q_max)  # = 0.0

        xyt_valid = self.validset()
        if self.device != torch.device('cpu'):
            xyt_valid = xyt_valid.to(self.device)

        if self.stage == 1:
            res = self.net_pde(xyt_valid.clone(), Q_norm_val, Q_scalar_val)
        elif self.stage == 2:
            n_valid = xyt_valid.shape[0]
            hstar_valid = self.net_prev(
                self.xytau_valid.repeat(4, 1), Q_norm_val).detach()
            hstar_valid_diff = self.net_pde_prev(
                self.xytau_valid.clone().repeat(4, 1),
                Q_norm_val, Q_scalar_val, out_diff=True).detach()
            res = self.net_pde(xyt_valid.clone(), Q_norm_val, Q_scalar_val,
                               hstar_diff=hstar_valid_diff)

        loss = self.criterion(res, torch.zeros_like(res))
        valid_loss = loss.item()
        self.writer.add_scalar('valid_loss', valid_loss, step)

        # Plots
        if self.stage == 1:
            h_valid = self.net(xyt_valid, Q_norm_val)
        elif self.stage == 2:
            h_valid = self.net(xyt_valid, Q_norm_val, hstar_valid)

        fig = show_surface(xyt_valid, h_valid, stage=self.stage)
        self.writer.add_figure(tag='3D surface', figure=fig, global_step=step)
        fig = show_contours_2x2(self.validset, h_valid, stage=self.stage,
                                well_type='unconfined_single_well')
        self.writer.add_figure(tag='contour', figure=fig, global_step=step)

        return valid_loss

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self, Q_test=None):
        """
        Run inference for a specific Q value and compare against MODFLOW.

        Parameters
        ----------
        Q_test : float, optional
            Pumping rate to test.  Defaults to the original Q = -40,000 m^3/day.
        """
        if Q_test is None:
            Q_test = -4.0e4

        print(f'\n=== Testing at Q = {Q_test:.0f} m^3/day ===')
        Q_norm_test = normalize_Q(Q_test, self.Q_min, self.Q_max)

        self.net.eval()
        self.net_pde.eval()
        self.net_neumann.eval()
        self.pinn.eval()

        if self.device != torch.device('cpu'):
            self.net.to(self.device)
            self.pinn.to(self.device)

        best_model = torch.load(
            f'checkpoints/{self.model_name}/best_model.pth.tar',
            map_location=self.device)
        self.pinn.load_state_dict(best_model['state_dict'])

        # Validation grid
        xyt_valid = self.validset()
        if self.device != torch.device('cpu'):
            xyt_valid = xyt_valid.to(self.device)

        if self.stage == 1:
            h_valid = self.net(xyt_valid, Q_norm_test)
        elif self.stage == 2:
            hstar_valid = self.net_prev(
                self.xytau_valid.repeat(4, 1), Q_norm_test).detach()
            h_valid = self.net(xyt_valid, Q_norm_test, hstar_valid)

        fig = show_surface(xyt_valid, h_valid, self.stage)
        fig = show_contours_2x2(self.validset, h_valid, self.stage,
                                well_type='unconfined_single_well')
        plt.show()

        if self.stage == 2:
            test_times = [26, 27, 28, 29, 30]
            all_mae, all_rmse, all_rrmse, all_r2 = [], [], [], []
            all_preds, all_targets = [], []

            print(f'\n=== Extrapolation test (t=26-30) for Q={Q_test:.0f} ===')
            for t_val in test_times:
                try:
                    df = pd.read_csv(f'./modflow/Q{int(Q_test)}/t{t_val}.txt',
                                     sep=r'\s*,\s*|\s+', engine='python',
                                     header=None, names=['x', 'y', 'h'])
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    if len(df) == 0:
                        continue

                    xy_shift = df[['x', 'y']].values - 500
                    t_arr = np.full((len(df), 1), float(t_val))
                    xyt_np = np.hstack([xy_shift, t_arr])
                    xytau_np = np.hstack([xy_shift,
                                          self.tau * np.ones_like(t_arr)])

                    xyt_t = torch.from_numpy(xyt_np.astype(np.float32))
                    xytau_t = torch.from_numpy(xytau_np.astype(np.float32))
                    if self.device != torch.device('cpu'):
                        xyt_t = xyt_t.to(self.device)
                        xytau_t = xytau_t.to(self.device)

                    hstar = self.net_prev(xytau_t, Q_norm_test).detach()
                    h_pred = self.net(xyt_t, Q_norm_test, hstar).detach().cpu().numpy()
                    h_true = df[['h']].values

                    t_mae = mae(h_true, h_pred)
                    t_rmse = float(np.sqrt(np.mean((h_true - h_pred) ** 2)))
                    t_rrmse = rrmse(h_true, h_pred) * 100
                    ss_res = np.sum((h_true - h_pred) ** 2)
                    ss_tot = np.sum((h_true - np.mean(h_true)) ** 2)
                    t_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                    all_mae.append(t_mae)
                    all_rmse.append(t_rmse)
                    all_rrmse.append(t_rrmse)
                    all_r2.append(t_r2)
                    all_preds.append(h_pred)
                    all_targets.append(h_true)
                    print(f'  t={t_val}: MAE={t_mae:.3f}m  RMSE={t_rmse:.3f}m  '
                          f'RRMSE={t_rrmse:.2f}%  R²={t_r2:.4f}')
                except Exception as e:
                    print(f'  t={t_val}: skipped — {e}')

            if all_mae:
                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)
                ss_res_tot = np.sum((all_targets - all_preds) ** 2)
                ss_tot_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
                r2_overall = 1 - ss_res_tot / ss_tot_tot if ss_tot_tot > 0 else 0.0

                print(f'\n--- SUMMARY (t=26-30, Q={Q_test:.0f}) ---')
                print(f'Avg MAE:   {np.mean(all_mae):.3f} m')
                print(f'Avg RMSE:  {np.mean(all_rmse):.3f} m')
                print(f'Avg RRMSE: {np.mean(all_rrmse):.2f} %')
                print(f'Overall R²:{r2_overall:.4f}')

        print('Testing finished.\n')


if __name__ == '__main__':
    args = Options().parse()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.problem = Problem(sigma=args.sigma)
    print('=== Parameterized PINN ===')
    print(f'Stage {args.stage} | tau={args.tau} | constraint={args.constraint}')
    print(f'Q range: [{args.Q_min}, {args.Q_max}] m^3/day')
    print(f'layers={args.layers}')

    trainer = Trainer(args)
    trainer.train()
