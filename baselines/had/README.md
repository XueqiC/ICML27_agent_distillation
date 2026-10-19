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

## Co-scheduled cell

Submit a pair of long-lived workers from the repository on the cluster:

```bash
export HAD_OUT=/shared/had-cell VERIFIED=/shared/verified-support
bash scripts/ops_had_cell.sh --account=YOUR_ACCOUNT --qos=YOUR_QOS
```

The GPU job requests hpg-b200, 4 GPUs, 8 CPUs, 256G and 48 hours. The CPU job
requests hpg-milan, zero GPUs, 2 CPUs, 8G and 48 hours. Account/QOS arguments
are forwarded unchanged to both jobs and their replacements. Both workers use
the same configuration and training world size (`NGPUS=4`). Teacher calls only
run in the CPU worker and retain the existing persistent 30,000-attempt cap.

The workers coordinate through the existing hash-bound `stages/*/complete.json`
files and advisory filesystem locks. They poll with backoff (1, 2, 4, ... up to
30 seconds). Locks exclude duplicate workers and per-stage/evaluation writers;
they are released by the OS after process death. No coordination service or
socket is introduced. Existing local student serving is unchanged.

Every wait is recorded before sleeping in `scheduling/{gpu,teacher}.jsonl`.
After 1,200 seconds of idle waiting the GPU worker returns exit 75 at a stage
boundary. The launcher submits one replacement, dependent on the current job
ending, then exits successfully to release the allocation. This requires
`sbatch` on compute nodes. A two-minute walltime warning similarly submits a
replacement for either role. Explicit cancellation and ordinary stage failures
do not automatically resubmit. A retained capped-teacher halt also stops the
GPU worker, preventing an endless replacement loop. Submission IDs and SLURM output remain under
`scheduling/`. `HAD_GPU_WAIT_SECONDS` may shorten, but never increase, the idle
limit; `HAD_POLL_SECONDS` and `HAD_POLL_MAX_SECONDS` control backoff.

Resume a killed member without duplicating its still-running peer:

```bash
bash scripts/ops_had_cell.sh --resume-role gpu --account=YOUR_ACCOUNT --qos=YOUR_QOS
# Or --resume-role teacher
```

Completed stages are validated and reused; interrupted attempts remain visible
and incomplete stages restart from their last committed input. Training resumes
the preceding round's exact masters/Adam state. An interrupted update may have
additional failed-attempt exposure, as in the per-stage implementation. An
interrupted live teacher request with an unknown outcome still halts the capped
collector: resubmission cannot recover an unrecorded remote response. Never reset
the counter or conceal unknown usage. This is not a guarantee of uninterrupted
progress after an arbitrary kill during a teacher transport call.

This mode preserves the serial algorithm's exact policy dependencies:
`explore-r -> teacher-r -> train-r -> explore-(r+1)`. The two-version allowance
permits older-policy data, but collecting r+1 before train-r would change its
policy input, trajectories and labels relative to the existing mode. Exact
artifact equivalence therefore precludes that lookahead. The scheduling audit
logs are additional; real timestamps, process IDs and latency naturally vary.
The offline equivalence test freezes attempt timestamps and compares every
existing artifact byte for byte with deterministic stages.

Both modes must run from the same frozen code tree. The existing identity binds
code/content hashes, so upgrading code inside a previously frozen cell fails
validation; this mode does not bypass that check.

Offline checks:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_had.py tests/test_had_torch.py
CUDA_VISIBLE_DEVICES='' /home/xueqi/tf5-env/bin/python -m pytest -q tests/test_had_coschedule.py tests/test_had_coschedule_launchers.py
bash -n scripts/ops_had_cell.sh scripts/hpg_had.slurm scripts/eval_adapters.sh
```
