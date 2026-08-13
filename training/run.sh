#!/bin/bash
# Stage 1  (t = 0 → 1 day)
python train.py \
  --stage 1 \
  --constraint HARD \
  --tau 1 \
  --sigma 30 \
  --hidden_layers 5 \
  --hidden_neurons 50 \
  --Q_min -50000 \
  --Q_max -25000 \
  --spatial_strategy LR \
  --temporal_strategy LHS \
  --nt 100 \
  --lam 100 \
  --lr 0.001 \
  --epochs_Adam 5000

# Stage 2  (t = 1 → 30 days)
python train.py \
  --stage 2 \
  --constraint HARD \
  --tau 1 \
  --sigma 30 \
  --hidden_layers 5 \
  --hidden_neurons 50 \
  --Q_min -50000 \
  --Q_max -25000 \
  --spatial_strategy LR \
  --temporal_strategy LHS \
  --temporal_strategy_prev LHS \
  --nt 100 \
  --nt_prev 100 \
  --lam 100 \
  --lr 0.001 \
  --epochs_Adam 5000
