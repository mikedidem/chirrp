#!/usr/bin/env python
"""
sensitivity_test.py
-------------------
Sensitivity analysis for the parameterized GW-PINN.
Computes Δh = h(Q) - h(Q_baseline) across the domain at t = 30 days,
saves PNG plots to sensitivity_plots/, and prints a summary table.

Usage (from param_pinn directory):
    python sensitivity_test.py
"""

import sys
import os
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # must be before pyplot import — works in subprocess
import matplotlib.pyplot as plt

sys.argv = ['sensitivity_test']

from options import Options
from problem import Problem
from model import Net, PINN

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def normalize_Q(Q, Q_min, Q_max):
    return 2.0 * (Q - Q_min) / (Q_max - Q_min) - 1.0


# ------------------------------------------------------------------
# Configuration  (must match training)
# ------------------------------------------------------------------
base_args = Options().parse()
base_args.hidden_layers          = 5
base_args.hidden_neurons         = 50
base_args.Q_min                  = -50000.0
base_args.Q_max                  = -25000.0
base_args.sigma                  = 30.0
base_args.tau                    = 1.0
base_args.constraint             = 'HARD'
base_args.stage                  = 2
base_args.spatial_strategy       = 'LR'
base_args.temporal_strategy      = 'LHS'
base_args.temporal_strategy_prev = 'LHS'
base_args.nt                     = 100
base_args.nt_prev                = 100
base_args.lam                    = 100
base_args.lr                     = 0.001
base_args.epochs_Adam            = 0
base_args.resume                 = None
base_args.layers = [4] + base_args.hidden_layers * [base_args.hidden_neurons] + [1]
base_args.problem = Problem(sigma=base_args.sigma)
torch.manual_seed(base_args.seed)

device  = base_args.device
Q_min   = base_args.Q_min
Q_max   = base_args.Q_max
tau     = base_args.tau

# Checkpoint paths (same naming as Trainer)
name_stem = (f"{base_args.constraint}_{base_args.hidden_layers}x{base_args.hidden_neurons}"
             f"_tau:{tau:.0f}_sigma:{base_args.sigma:.0f}"
             f"_S:{base_args.spatial_strategy}")
ckpt_s1 = (f"checkpoints/Stage1_{name_stem}"
           f"_T:{base_args.temporal_strategy_prev}_nt:{base_args.nt_prev}"
           f"/best_model.pth.tar")
ckpt_s2 = (f"checkpoints/Stage2_{name_stem}"
           f"_T:{base_args.temporal_strategy}_nt:{base_args.nt}"
           f"/best_model.pth.tar")

print(f"Stage 1 checkpoint: {ckpt_s1}")
print(f"Stage 2 checkpoint: {ckpt_s2}")

# ------------------------------------------------------------------
# Load networks
# ------------------------------------------------------------------
args1 = copy.deepcopy(base_args)
args1.stage = 1
net1  = Net(args1, stage=1)
pinn1 = PINN(net1)
pinn1.load_state_dict(
    torch.load(ckpt_s1, map_location=device)['state_dict'])
net1.to(device)
net1.eval()
print("Stage 1 loaded.")

args2 = copy.deepcopy(base_args)
args2.stage = 2
net2  = Net(args2, stage=2)
pinn2 = PINN(net2)
pinn2.load_state_dict(
    torch.load(ckpt_s2, map_location=device)['state_dict'])
net2.to(device)
net2.eval()
print("Stage 2 loaded.")

# ------------------------------------------------------------------
# Evaluation grid  (100×100, t = 30 days)
# ------------------------------------------------------------------
nx, ny  = 100, 100
x_lin   = np.linspace(-500, 500, nx)
y_lin   = np.linspace(-500, 500, ny)
XX, YY  = np.meshgrid(x_lin, y_lin)        # both (ny, nx)
xy_flat = np.column_stack([XX.ravel(), YY.ravel()])   # (N, 2)
N       = xy_flat.shape[0]

t_eval = 30.0

xytau_np = np.hstack([xy_flat, np.full((N, 1), tau)])
xyt_np   = np.hstack([xy_flat, np.full((N, 1), t_eval)])

xytau_t = torch.from_numpy(xytau_np.astype(np.float32))
xyt_t   = torch.from_numpy(xyt_np.astype(np.float32))
if device != torch.device('cpu'):
    xytau_t = xytau_t.to(device)
    xyt_t   = xyt_t.to(device)


def predict_h(Q_scalar):
    """Return h(x, y, t=30) for a given Q as a (ny, nx) numpy array."""
    Q_norm = normalize_Q(Q_scalar, Q_min, Q_max)
    with torch.no_grad():
        hstar = net1(xytau_t, Q_norm)        # Stage 1 at tau=1
        h     = net2(xyt_t,   Q_norm, hstar) # Stage 2 at t=30
    return h.cpu().numpy().reshape(ny, nx)


# ------------------------------------------------------------------
# Sensitivity analysis
# ------------------------------------------------------------------
Q_baseline  = -40000.0
pct_changes = [-25, -10, -5, 0, +5, +10, +25]
Q_values    = {pct: Q_baseline * (1.0 + pct / 100.0) for pct in pct_changes}

print('\n' + '=' * 70)
print('Sensitivity Analysis — Parameterized GW-PINN')
print(f'Baseline Q = {Q_baseline:,.0f} m\u00b3/day   |   evaluated at t = {t_eval:.0f} days')
print(f'Training range: [{Q_min:,.0f}, {Q_max:,.0f}] m\u00b3/day')
print('=' * 70)

h_base = predict_h(Q_baseline)   # (ny, nx)

results = {}
print(f'\n{"% Change":>10}  {"Q (m\u00b3/d)":>14}  {"max |\u0394h| (m)":>14}  '
      f'{"mean \u0394h (m)":>13}  {"max drawdown (m)":>18}')
print('-' * 75)

for pct in pct_changes:
    Q_test  = Q_values[pct]
    h_pred  = predict_h(Q_test)
    dh      = h_pred - h_base

    max_abs_dh   = float(np.max(np.abs(dh)))
    mean_dh      = float(np.mean(dh))
    max_drawdown = float(90.0 - np.min(h_pred))

    results[pct] = {
        'h': h_pred, 'dh': dh, 'Q': Q_test,
        'max_abs_dh': max_abs_dh,
        'mean_dh': mean_dh,
        'max_drawdown': max_drawdown,
    }

    marker = '  <- baseline' if pct == 0 else ''
    print(f'{pct:>+10d}%  {Q_test:>14,.0f}  {max_abs_dh:>14.4f}  '
          f'{mean_dh:>13.4f}  {max_drawdown:>18.4f}{marker}')

print('=' * 70)

# ------------------------------------------------------------------
# Plot 1: Δh spatial maps  (2 × 4 grid; baseline head in slot 0)
# ------------------------------------------------------------------
os.makedirs('sensitivity_plots', exist_ok=True)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()
axes[-1].set_visible(False)   # hide unused 8th slot

vmax = max(results[p]['max_abs_dh'] for p in pct_changes if p != 0)
vmax = max(vmax, 0.01)

well_x, well_y = 201.67, -98.33   # well position in PINN coordinates

for k, pct in enumerate(pct_changes):
    ax = axes[k]
    if pct == 0:
        im = ax.contourf(XX, YY, results[0]['h'], 20, cmap='viridis')
        ax.set_title(f'Baseline h  (Q = {Q_baseline:,.0f} m\u00b3/d)')
        plt.colorbar(im, ax=ax, label='h (m)')
    else:
        im = ax.contourf(XX, YY, results[pct]['dh'], 20,
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'\u0394h  ({pct:+d}%,  Q = {results[pct]["Q"]:,.0f})')
        plt.colorbar(im, ax=ax, label='\u0394h (m)')
    ax.plot(well_x, well_y, 'r*', markersize=8)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

fig.suptitle('\u0394h = h(Q) \u2212 h(Q_baseline)   at t = 30 d', fontsize=14)
plt.tight_layout()
out1 = 'sensitivity_plots/sensitivity_dh_maps.png'
plt.savefig(out1, dpi=120, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out1}')

# ------------------------------------------------------------------
# Plot 2: Summary bar + line charts
# ------------------------------------------------------------------
pcts_nonzero  = [p for p in pct_changes if p != 0]
max_abs_dhs   = [results[p]['max_abs_dh']   for p in pcts_nonzero]
max_drawdowns = [results[p]['max_drawdown'] for p in pct_changes]
pct_labels    = [f'{p:+d}%' for p in pct_changes]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar([f'{p:+d}%' for p in pcts_nonzero], max_abs_dhs, color='steelblue')
ax1.set_xlabel('Q change (%)')
ax1.set_ylabel('Max |\u0394h| (m)')
ax1.set_title('Maximum Absolute Head Difference vs Baseline')
ax1.grid(axis='y', alpha=0.4)

ax2.plot(pct_labels, max_drawdowns, 'o-', color='firebrick')
ax2.axvline(x='+0%', color='k', linestyle='--', alpha=0.5, label='baseline')
ax2.set_xlabel('Q change (%)')
ax2.set_ylabel('Max drawdown (m)')
ax2.set_title('Maximum Drawdown (relative to h\u2080 = 90 m)')
ax2.legend()
ax2.grid(alpha=0.4)

plt.tight_layout()
out2 = 'sensitivity_plots/sensitivity_summary.png'
plt.savefig(out2, dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved: {out2}')

print('\n' + '=' * 70)
print('Sensitivity analysis complete.')
print('=' * 70)
