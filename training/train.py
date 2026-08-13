#!/usr/bin/env python
import torch
from options import Options
from trainer import Trainer
from problem import Problem

args = Options().parse()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

args.problem = Problem(sigma=args.sigma)
print('=== Parameterized PINN — Unconfined Aquifer (Single Well) ===')
print(f'Stage {args.stage} | tau={args.tau} | constraint={args.constraint}')
print(f'Q range: [{args.Q_min}, {args.Q_max}] m^3/day')
print(f'layers={args.layers}')
print('=' * 60)

trainer = Trainer(args)
trainer.train()
