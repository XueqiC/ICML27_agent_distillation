# LoRA stability probe (2026-10-21) — choice of the shared optimiser setting

**Decision:** learning rate 1e-4, linear decay, no warmup; every other AD recipe setting unchanged (LoRA r64 / alpha 128 / dropout 0.05,
AdamW 0.9/0.999/1e-8, weight decay 0, gradient clip 1.0, batch 8 trajectories, turn packing, shared-history scoring).
Applied uniformly to trajectory SFT, AD (FTP data + SAG) and SmartAD. The paper recipe (lr 2e-4, 2 epochs) stays as a separate column.
Pre-registered fallback: if a 1e-4 ladder rung fails the divergence check, that method x K ladder is rerun at 5e-5.
Selection used training loss only; no dev or test outcome was looked at.

## Setup
Trajectory SFT on the K100 view (stratified_nested_v2, seed 0), 4 epochs = 52 updates (13 updates per epoch), LONI QB-D, 2x A100 80GB,
tree 20cb697. Only lr and warmup vary. Loss = token-mean NLL of the supervised agent-output tokens, averaged over each complete epoch.
Gradient norm = pre-clip global L2 norm of the LoRA gradient.

| run | epoch 1 | epoch 2 | epoch 3 | epoch 4 | max grad norm | median grad norm | loss spikes* |
|---|---|---|---|---|---|---|---|
| lr 2e-4, no warmup (paper; LONI tsft-k100-e4) | 0.3315 | 0.2553 | 1.0955 | 1.1431 | 1106.8 | 32.2 | 2 (updates 32, 34) |
| lr 2e-4, 10% warmup | 0.4445 | 0.3920 | 0.2160 | 0.1863 | 171.7 | 8.3 | 1 (update 15) |
| **lr 1e-4, no warmup** | 0.3513 | 0.2267 | 0.1913 | **0.1711** | 94.5 | 5.3 | 0 |
| lr 5e-5, no warmup | 0.3714 | 0.2295 | 0.2013 | 0.1881 | 33.8 | 6.1 | 0 |

*spike = an update whose loss exceeds 3x the median of the previous 8 updates (and > 0.1).

lr 1e-4 and 5e-5 were both spike-free; 1e-4 reached the lowest final loss, so it needs fewer doubling rungs to converge. 5e-5 has the
smallest gradient norms and is the fallback.

## The paper recipe across the old ladder (lr 2e-4, no warmup; trajectory SFT, LONI)
Divergence check (see below) fails on 6 of 19 rungs: K30 32/64 epochs, K100 2/4/8 epochs, K500 8 epochs. K500 8 epochs diverged inside its
first epoch (epoch means 1.84 4.47 4.35 4.34 3.97 1.51 0.99 vs 0.13 for 4 epochs), which the within-rung check alone cannot detect.

## Divergence check added to the convergence rule
A rung (one training run with a given epoch count) is valid iff
1. within the rung: L_last <= max(1.1 * L_min, L_min + 0.02), L_min = lowest complete-epoch mean of the same run, and
2. across rungs: L_last(2E epochs) <= max(1.1 * L_last(E epochs), L_last(E epochs) + 0.02).
Converged iff both adjacent rungs are valid and L_last(E) - L_last(2E) < 0.01 nats/token.
