# HAD

Paper implementation of [arXiv:2610.02858v1](https://arxiv.org/html/2610.02858v1).
The complete registration is in [DEVIATIONS.md](DEVIATIONS.md). Beta and the
gradient-matching ratio are both 0.5, as specified in the paper.

Set `HAD_OUT`, `VERIFIED` and the existing HPG configuration. Keep `NGPUS`
unchanged across the cell, including CPU stages (it identifies the training world
size; it does not allocate GPUs). Use separate jobs in this dependency order:

```bash
# CPU allocation: prepare validates the verified collection and freezes identity.
HAD_MODE=prepare HAD_THROUGH=explore-0 bash scripts/hpg_had.slurm

# Submit each GPU stage with --partition=hpg-b200 --gpus=$NGPUS.
HAD_THROUGH=explore-0  # student inference and real environment execution
# Next job: CPU partition, no GPU resource request.
HAD_THROUGH=teacher-0  # paired capped teacher queries
# Next job: GPU partition/allocation.
HAD_THROUGH=train-0    # full FT; publishes round checkpoint and exact resume state
```

Repeat `explore-1`, `teacher-1`, `train-1`, etc. Inspect `iteration-r.json` and
stop when `result.json` reports 250 updates. `--through` executes only the named
stage, after verifying all completed prerequisites. It never launches an omitted
CPU stage from a GPU job. The wrapper deliberately has no default GPU request.
Use the scheduler's `afterok` dependencies when submitting chains; no jobs are
submitted by this implementation. A hard teacher-cap or interrupted teacher
failure is preserved and stops the cell without claiming completion.

Evaluate all published checkpoints in a separate GPU job:

```bash
HAD_MODE=evaluate HAD_THROUGH=train-0 bash scripts/hpg_had.slurm
```

`models.json` contains `iter-N-had` and `iter-N-h0` for each standalone checkpoint.
Each produces its own `eval-iter-N-*` stage/result directory. The existing
`scripts/hpg_eval_adapters.slurm` also accepts these checkpoints through
`MERGED_MODEL_PATH` / `FULL_CHECKPOINT_NAME`; set `VERIFIED` for its mandatory
HF/vLLM gate and `HAD_EVAL_HARNESS=had` or `h0` (default h0). The gate runs HF
before starting vLLM, then verifies on that server before either evaluation.

Teacher queries and their durable 30,000-attempt counter live in `HAD_OUT/teacher`.
Never delete/reset that directory to resume. Stages retain failed attempts;
`disclosure.json` includes their compute/exposure alongside successful rounds.
Do not put outputs under seed storage. No task outside the verified support
collection is used for rollout or labelling. Dev140 is evaluation-only.

Offline checks:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_had.py tests/test_had_torch.py
bash -n scripts/hpg_had.slurm scripts/eval_adapters.sh
```
