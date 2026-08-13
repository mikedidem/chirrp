"""
Network architecture for the parameterized groundwater PINN.

Ported from the research code (param_pinn/model.py). The class structure and
attribute names (fc1..fcN, param1..paramN) must stay exactly as trained so the
shipped checkpoints load without key remapping.

Input  : (x, y, t, Q_norm) — coordinates are normalized to [-1, 1] inside
         forward() from the problem domain; Q_norm arrives pre-normalized.
Output : hydraulic head h in metres (physical units, no denormalization).
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from .config import PinnConfig


class Net(nn.Module):
    """Sin-residual MLP with a learnable activation scale per hidden layer.

    With the HARD constraint the Dirichlet boundary (y = ±500 m, h = 90 m) and
    the initial/stage-handoff condition are enforced exactly by construction:

        d(y, t) = (t - tmin)(ymax - y)(y - ymin) / ((tmax - tmin)(ymax - ymin)²)
        h       = h* + d · u(x, y, t, Q)

    where h* is 90 m in Stage 1 and the Stage-1 solution at t = tau in Stage 2.
    """

    def __init__(self, config: PinnConfig, stage: int = 1):
        super().__init__()
        self.config = config
        self.stage = stage
        self.tau = config.tau
        self.scale = config.scale
        self.constraint = config.constraint
        layers = config.layers

        self.fcs = []
        self.params = []

        # Hidden layers with learnable sin-activation scale
        for i in range(len(layers) - 2):
            fc = nn.Linear(layers[i], layers[i + 1])
            setattr(self, f"fc{i + 1}", fc)
            self._init_weights(fc)
            self.fcs.append(fc)

            param = nn.Parameter(torch.randn(layers[i + 1]))
            setattr(self, f"param{i + 1}", param)
            self.params.append(param)

        # Output layer (linear, no sin)
        fc = nn.Linear(layers[-2], layers[-1])
        setattr(self, f"fc{len(layers) - 1}", fc)
        self._init_weights(fc)
        self.fcs.append(fc)

    @staticmethod
    def _init_weights(layer: nn.Linear) -> None:
        init.xavier_normal_(layer.weight)
        init.constant_(layer.bias, 0.01)

    def forward(self, xyt: torch.Tensor, q_norm, hstar=None) -> torch.Tensor:
        """
        Parameters
        ----------
        xyt    : (N, 3) tensor — [x, y, t] in physical coordinates
        q_norm : float or (N, 1) tensor — normalized pumping rate in [-1, 1]
        hstar  : (N, 1) tensor — Stage-1 solution at t = tau (Stage 2 only)
        """
        n = xyt.shape[0]

        if isinstance(q_norm, (int, float)):
            q = torch.full((n, 1), float(q_norm), dtype=torch.float32,
                           device=xyt.device)
        elif isinstance(q_norm, torch.Tensor) and q_norm.dim() == 0:
            q = q_norm.view(1, 1).expand(n, 1)
        else:
            q = q_norm

        xmin, xmax, ymin, ymax, tmin, tmax = self.config.domain
        if self.stage == 1:
            tmax = self.tau
        elif self.stage == 2:
            tmin = self.tau

        lb = torch.tensor([xmin, ymin, tmin], dtype=torch.float32,
                          device=xyt.device)
        ub = torch.tensor([xmax, ymax, tmax], dtype=torch.float32,
                          device=xyt.device)
        x_norm = 2.0 * (xyt - lb) / (ub - lb) - 1.0

        x = torch.cat([x_norm, q], dim=-1)

        x = self.fcs[0](x)
        x = torch.mul(self.params[0], x) * self.scale
        x = torch.sin(x)

        for i in range(1, len(self.fcs) - 1):
            res = self.fcs[i](x)
            res = torch.mul(self.params[i], res) * self.scale
            res = torch.sin(res)
            x = x + res

        u = self.fcs[-1](x)

        if self.constraint == "SOFT":
            return u

        # HARD constraint
        hstar_val = self.config.initial_head if hstar is None else hstar
        y = xyt[:, [1]]
        t = xyt[:, [2]]
        d = ((t - tmin) * (ymax - y) * (y - ymin)) / \
            ((tmax - tmin) * (ymax - ymin) ** 2)
        return hstar_val + d * u


def load_stage(checkpoint_path: str, config: PinnConfig, stage: int) -> Net:
    """Build a Net for the given stage and load trained weights.

    The training code saved ``PINN.state_dict()`` where the PINN wrapper
    registered the same Net under ``net``, ``net_Neumann.net`` and
    ``net_PDE.net``; we keep only the ``net.`` entries.
    """
    net = Net(config, stage=stage)
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    state = checkpoint["state_dict"]
    net_state = {k[len("net."):]: v for k, v in state.items()
                 if k.startswith("net.")}
    net.load_state_dict(net_state, strict=True)
    net.eval()
    return net
