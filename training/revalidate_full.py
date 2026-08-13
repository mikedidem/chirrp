"""Full space-time revalidation of the two-stage PINN surrogate against MODFLOW-2005.

For every pumping rate with stored MODFLOW output (10 rates), compares the PINN
prediction at the exact MODFLOW cell centers across ALL 33 stored timesteps
(t = 0.25 ... 30 d), and runs the finite-volume mass-balance diagnostic
(identical to train_colab.ipynb cell 26) at every rate.

Outputs:
  paper_figures/revalidation_per_rate.csv       - pooled space-time metrics per rate
  paper_figures/revalidation_per_timestep.csv   - RMSE/MAE per (rate, timestep)
  paper_figures/revalidation_massbalance.csv    - mass-balance summary per rate
  paper_figures/revalidation_table.md           - manuscript-ready markdown
"""
import os, sys, glob, re, copy, time
import numpy as np
import torch

CODE = r"G:\My Drive\para_pinns\param_pinn"
OUT  = os.path.join(CODE, "evaluation")
os.chdir(CODE)
sys.path.insert(0, CODE)
sys.argv = ["train"]

from options import Options
from problem import Problem
from model import Net, PINN

t0 = time.time()

# ---- config: must match training (from train_colab.ipynb cell 22) ----
base_args = Options().parse()
base_args.hidden_layers          = 5
base_args.hidden_neurons         = 50
base_args.Q_min                  = -50000.0
base_args.Q_max                  = -25000.0
base_args.sigma                  = 30.0
base_args.tau                    = 1.0
base_args.constraint             = "HARD"
base_args.stage                  = 2
base_args.spatial_strategy       = "LR"
base_args.temporal_strategy      = "LHS"
base_args.temporal_strategy_prev = "LHS"
base_args.nt                     = 100
base_args.nt_prev                = 100
base_args.lam                    = 100
base_args.lr                     = 0.001
base_args.epochs_Adam            = 0
base_args.resume                 = None
base_args.layers = [4] + base_args.hidden_layers * [base_args.hidden_neurons] + [1]
base_args.problem = Problem(sigma=base_args.sigma)
base_args.device = torch.device("cpu")
device = base_args.device
TAU, QMIN, QMAX = base_args.tau, base_args.Q_min, base_args.Q_max

stem = (f"{base_args.constraint}_{base_args.hidden_layers}x{base_args.hidden_neurons}"
        f"_tau:{TAU:.0f}_sigma:{base_args.sigma:.0f}_S:{base_args.spatial_strategy}")
ck1 = f"checkpoints/Stage1_{stem}_T:{base_args.temporal_strategy_prev}_nt:{base_args.nt_prev}/best_model.pth.tar"
ck2 = f"checkpoints/Stage2_{stem}_T:{base_args.temporal_strategy}_nt:{base_args.nt}/best_model.pth.tar"
if not os.path.exists(ck1):  # Windows: ':' in Colab folder names became ' '
    ck1, ck2 = ck1.replace(":", " "), ck2.replace(":", " ")
assert os.path.exists(ck1) and os.path.exists(ck2), (ck1, ck2)

a1 = copy.deepcopy(base_args); a1.stage = 1
net1 = Net(a1, stage=1)
PINN(net1).load_state_dict(torch.load(ck1, map_location=device)["state_dict"])
net1.to(device).eval()
a2 = copy.deepcopy(base_args); a2.stage = 2
net2 = Net(a2, stage=2)
PINN(net2).load_state_dict(torch.load(ck2, map_location=device)["state_dict"])
net2.to(device).eval()
print("checkpoints loaded", flush=True)

def qnorm(Q):
    return 2.0 * (Q - QMIN) / (QMAX - QMIN) - 1.0

@torch.no_grad()
def predict(xy, t_val, Q):
    """xy: (N,2) PINN-domain coords; returns (N,) head."""
    N = xy.shape[0]
    xyt = torch.from_numpy(
        np.hstack([xy, np.full((N, 1), t_val)]).astype(np.float32)).to(device)
    xytau = torch.cat(
        [xyt[:, :2], torch.full((N, 1), TAU, dtype=torch.float32, device=device)], dim=1)
    qn = qnorm(Q)
    hstar = net1(xytau, qn)
    h = net2(xyt, qn, hstar)
    return h.cpu().numpy().ravel()

def parse_time(f):
    m = re.search(r"t([0-9]+(?:\.[0-9]+)?)\.txt$", os.path.basename(f))
    return float(m.group(1)) if m else None

# ---- 1. accuracy: all rates x all timesteps ----
BASE = 40000.0
qdirs = sorted(glob.glob(os.path.join(CODE, "modflow", "Q-*")),
               key=lambda d: float(d.rsplit("Q-", 1)[1]))
per_rate, per_ts = [], []
for qdir in qdirs:
    Qabs = float(qdir.rsplit("Q-", 1)[1])
    Q = -Qabs
    pct = (Qabs - BASE) / BASE * 100.0
    files = sorted([f for f in glob.glob(os.path.join(qdir, "t*.txt"))
                    if parse_time(f) is not None], key=parse_time)
    diffs, trues = [], []
    for f in files:
        tv = parse_time(f)
        d = np.loadtxt(f)
        xy = d[:, :2] - 500.0          # 0..1000 -> -500..500
        h_true = d[:, 2]
        h_pred = predict(xy, tv, Q)
        diff = h_pred - h_true
        diffs.append(diff); trues.append(h_true)
        per_ts.append(dict(Q=Q, pct=pct, t=tv,
                           rmse=float(np.sqrt(np.mean(diff**2))),
                           mae=float(np.mean(np.abs(diff))),
                           max_abs=float(np.max(np.abs(diff)))))
    diff = np.concatenate(diffs); true = np.concatenate(trues)
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((true - true.mean())**2))
    rmse = float(np.sqrt(np.mean(diff**2)))
    rng = float(true.max() - true.min())
    per_rate.append(dict(
        Q=Q, pct=pct, n_timesteps=len(files), n_points=diff.size,
        rmse=rmse, mae=float(np.mean(np.abs(diff))),
        bias=float(np.mean(diff)), r2=1.0 - ss_res / ss_tot,
        rrmse_pct=100.0 * rmse / rng, max_abs=float(np.max(np.abs(diff))),
        head_range=rng))
    r = per_rate[-1]
    print(f"Q={Q:8.0f} ({pct:+6.1f}%) nt={r['n_timesteps']} "
          f"RMSE={r['rmse']:.3f} MAE={r['mae']:.3f} bias={r['bias']:+.3f} "
          f"R2={r['r2']:.4f} RRMSE={r['rrmse_pct']:.2f}% max={r['max_abs']:.2f}",
          flush=True)

# ---- 2. mass balance per rate (identical to notebook cell 26) ----
K, Sy = 33.33, 0.10
nmb = 300
xm = np.linspace(-500, 500, nmb); ym = np.linspace(-500, 500, nmb)
dxm, dym = xm[1] - xm[0], ym[1] - ym[0]
gx, gy = np.meshgrid(xm, ym)
xym = np.vstack([gx.ravel(), gy.ravel()]).T
cell_area = dxm * dym
t_seq = np.linspace(1.0, 30.0, 60)

def boundary_flux(head):
    hb_s, hi_s = head[0, :], head[1, :]
    q_s = np.sum(K * 0.5 * (hb_s + hi_s) * (hb_s - hi_s) / dym * dxm)
    hb_n, hi_n = head[-1, :], head[-2, :]
    q_n = np.sum(K * 0.5 * (hb_n + hi_n) * (hb_n - hi_n) / dym * dxm)
    return q_s + q_n

mb_rows = []
for r in per_rate:
    Q = r["Q"]
    res = []
    for tv in t_seq:
        h_t = predict(xym, tv, Q).reshape(nmb, nmb)
        h_p = predict(xym, tv - 0.01, Q).reshape(nmb, nmb)
        dS = np.sum(Sy * (h_t - h_p) / 0.01) * cell_area
        res.append(dS - (boundary_flux(h_t) + Q))
    res = np.abs(np.array(res))
    mb_rows.append(dict(Q=Q, pct=r["pct"],
                        mb_mean=float(res.mean()), mb_max=float(res.max()),
                        mb_mean_pct=float(100 * res.mean() / abs(Q)),
                        mb_max_pct=float(100 * res.max() / abs(Q))))
    m = mb_rows[-1]
    print(f"MB Q={Q:8.0f}: mean {m['mb_mean']:8.1f} m3/d ({m['mb_mean_pct']:5.2f}%) "
          f"max {m['mb_max']:8.1f} ({m['mb_max_pct']:5.2f}%)", flush=True)

# ---- 3. write outputs ----
import csv
os.makedirs(OUT, exist_ok=True)
def wcsv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
wcsv(os.path.join(OUT, "revalidation_per_rate.csv"), per_rate)
wcsv(os.path.join(OUT, "revalidation_per_timestep.csv"), per_ts)
wcsv(os.path.join(OUT, "revalidation_massbalance.csv"), mb_rows)

mean = lambda k, rows: float(np.mean([r[k] for r in rows]))
lines = ["| Q (m3/day) | change (%) | RMSE (m) | MAE (m) | bias (m) | R2 | RRMSE (%) | max abs err (m) | MB mean (% of Q) |",
         "|---|---|---|---|---|---|---|---|---|"]
for r, m in zip(per_rate, mb_rows):
    lines.append(f"| {r['Q']:,.0f} | {r['pct']:+.1f} | {r['rmse']:.3f} | {r['mae']:.3f} | "
                 f"{r['bias']:+.3f} | {r['r2']:.4f} | {r['rrmse_pct']:.2f} | "
                 f"{r['max_abs']:.2f} | {m['mb_mean_pct']:.2f} |")
lines.append(f"| **Mean** |  | **{mean('rmse', per_rate):.3f}** | **{mean('mae', per_rate):.3f}** | "
             f"{mean('bias', per_rate):+.3f} | **{mean('r2', per_rate):.4f}** | "
             f"{mean('rrmse_pct', per_rate):.2f} |  | {mean('mb_mean_pct', mb_rows):.2f} |")
open(os.path.join(OUT, "revalidation_table.md"), "w").write(
    "# Full space-time revalidation (33 timesteps x 90,000 cells per rate)\n\n" +
    "\n".join(lines) + "\n")
print("\nPOOLED MEANS: RMSE %.3f  MAE %.3f  R2 %.4f  MB %.2f%%" %
      (mean('rmse', per_rate), mean('mae', per_rate), mean('r2', per_rate),
       mean('mb_mean_pct', mb_rows)), flush=True)
print("done in %.1f s" % (time.time() - t0))
