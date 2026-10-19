# GAD on verified support histories

Implementation: [DEVIATIONS.md](DEVIATIONS.md), [NOTICE.md](NOTICE.md).
This port uses full text fine-tuning and a faithful port of the pinned verl
GRPO/BT equations, with the shared FSDP2 runtime and vLLM serving.

Use the same `--snapshot`, `--verified`, `--output`, `--gpus` and config for the
whole cell. `--verified` accepts the registered nested K views or the full-TRAIN
collection. There is no outside task list or teacher-response import option.

```bash
export PYTHONPATH="$PWD/src:$PWD"
python -m baselines.gad.pipeline --config configs/base.yaml \
  --snapshot "$MODEL_PATH" --verified "$VERIFIED" --output "$GAD_OUT" \
  --gpus 4 --mode prepare
```

Run these nine stages in order, **as separate jobs**:

| Round | GPU: real greedy H0 rollouts | CPU: capped teacher labels | GPU: training |
|---|---|---|---|
| 0 | `explore-0` | `teacher-0` | `train-0`: 10 D-only steps, then one CE+BT epoch |
| 1 | `explore-1` | `teacher-1` | `train-1`: one GRPO+BT epoch |
| 2 | `explore-2` | `teacher-2` | `train-2`: one GRPO+BT epoch |

Set `GAD_THROUGH` to one table entry and submit `scripts/hpg_gad.slurm`.
For example, GPU stages use `sbatch --partition=hpg-b200 --gpus=4`; teacher
stages use a site-appropriate CPU partition with **no GPU allocation**.
Keep `NGPUS=4` in CPU jobs too: it records the cell's training world size.
Supply `GAD_OUT`, `VERIFIED` and the usual `configs/hpg.env` runtime settings.
The wrapper never submits jobs itself, and no cluster run has been performed.
For a future one-GPU K20 s0 smoke, use a separate cell output and `NGPUS=1`
throughout. Startup overhead, disk usage and B200 memory fit require profiling.

Each `--through` runs only that selected stage. Completed prerequisites are
hash-validated; missing ones cause an error. Resuming a train stage reuses its
completed nested sample/critic/actor steps, including fp32 masters and Adam
moments. Do not delete partial attempts or reset the teacher counter. A failed
teacher transport reservation intentionally halts the cell rather than spending
a second call at that history.

For evaluation, set `GAD_MODE=evaluate` (retain a valid `GAD_THROUGH`, unused in
this mode) or invoke the pipeline with `--mode evaluate`. Every entry in
`models.json` passes the common HF/vLLM equivalence check before dev140. No
checkpoint is selected by dev performance. `disclosure.json` counts all
exploration/replay attempts, capped calls, group sampling and optimizer exposure.

CPU verification:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xueqi/tf5-env/bin/python \
  -m pytest tests/test_gad.py tests/test_gad_torch.py tests/test_gad_integration.py -q
```

GPU behavior and the real serving stack remain untested in this implementation
task. Passing CPU tests is not a scientific finding or a claim of faithful
reproduction on the original chat benchmark.
