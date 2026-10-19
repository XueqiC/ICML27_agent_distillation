# Chain-of-Agents

Paper: https://arxiv.org/abs/2508.13167. Official source is the local read-only
`third_party/Agent_Foundation_Models` at `9a6594d`. See [DEVIATIONS.md](DEVIATIONS.md)
for the selected recipes and every ALFWorld/Gemma adaptation.

Use a persistent output outside seed storage and `reports/runs`. All invocations
share `--config`, `--snapshot`, `--verified`, `--output`, and `--gpus` identity.
`--verified` accepts the registered nested support views and full-TRAIN collection.

Execute separate jobs, in order:

1. `--mode prepare` (CPU, optional).
2. `--through teacher` (CPU): three multi-agent executions per support task.
3. `--through data` (CPU): filters, resampling decisions and masked SFT sources.
4. `--through train-sft` (GPU): full CoA SFT; publishes `sft`.
5. `--through query-filter` (GPU): eight SFT executions per task, discard 8/8 wins.
6. `--through explore-0`, then `--through train-rl-0` (separate GPU jobs).
   Repeat `explore-r` / `train-rl-r` until `result.json` records completion, at
   most r=59. Each update publishes `rl-{r+1}`. An empty final mixed batch stops
   without a fabricated update; inspect the exploration plan and result.
7. `--mode evaluate` (GPU): gate and dev140 for every entry in `models.json`.
8. `--mode teacher-evaluate` (CPU): separately capped multi-agent teacher dev140.

`scripts/hpg_coa.slurm` uses `COA_THROUGH`, `COA_MODE`, `COA_OUT`, `VERIFIED`,
`NGPUS` and the existing HPG environment. It requests no GPU by default; supply
GPU resources only for GPU stages. This port does not submit any jobs itself.

Artifacts include original per-call responses, complete public agent states,
execution/replay records, all rejected trajectories, Alpaca-format exports,
filter decisions, immutable stage receipts, update exposures, optimizer resume
state, checkpoint manifests and separate TRAIN/teacher-dev disclosures.

Offline tests:

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/tf5-env/bin/python -m pytest -q tests/test_coa.py tests/test_coa_torch.py
```
