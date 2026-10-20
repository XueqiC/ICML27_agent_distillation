# SAD shared-history execution path — 2026-10-20

SAD defaults to `shared_history` in `baselines.common.fsdp_train` and
`baselines.sad.pipeline`. Select `--scoring per_turn` (or `SAD_SCORING=per_turn`
in `scripts/hpg_sad.slurm`) to retain separate turn forwards. SFT, STeCa and HAD
keep their existing per-turn paths.

The packer accepts target labels that are either their original token or -100,
while still requiring a masked prompt, target supervision and consistent native
history. It preserves the original SAD encoder's Thought sections, Action lines
and turn-end labels, including holes in the reply mask. Masked reply text stays
in subsequent history. SAD forwards/backpropagates each packed trajectory once,
sums supervised token CE, then averages across the actual trajectory batch.

The 655,000,000 budget, curriculum, learning-rate schedule and checkpoint
thresholds use the sum of the original per-turn input lengths. Repeated masked
history counts logically under both paths. Rank padding counts only physically.
`physical_policy_forward_tokens` records model input positions, including
zero-weight rank padding; SAD has no extra scoring recomputation or reference
forward. As in the existing trainer, decoder activation-checkpoint recomputation
is not counted as another model-input forward. Checkpoint state, update ledgers,
final state and disclosure retain the scoring choice and separate counters.

The GPU acceptance tool accepts existing preparation without changing it:

```bash
source configs/hpg.env
"$TRAIN_PYTHON" -m tools.validate_shared_history_gpu \
  --config configs/base.yaml --snapshot "$MODEL_PATH" \
  --sad-prepared /blue/.../reports/runs/baselines/sad-k20s0 \
  --num-rows 4 --output reports/engineering/sad_shared_history_gpu.json
```

`--sad-prepared` also accepts a directory underneath that run, such as its
`stages/prepare/attempt-000/output`. The loader locates the prepared root's
`identity.json` and `tasks.json`, checks their config/task/parent/collection/split
bindings, loads checked original episodes, and calls the actual SAD encoder and
stable curriculum. The first N curriculum trajectories are selected without
filtering, retries or truncation. Historical identities without a scoring field
are disclosed as per-turn. A new training run requires a new output identity;
the existing per-turn run and its exposure remain intact.

For later execution on hpg, `scripts/hpg_validate_sad_shared_history.slurm`
requests one B200 for one hour:

```bash
SAD_PREPARED=/blue/.../reports/runs/baselines/sad-k20s0 \
  sbatch --export=ALL scripts/hpg_validate_sad_shared_history.slurm
```

Optional variables: `NUM_ROWS` (default 4), `OUTPUT`, `CONFIG`, `HPG_CONFIG`.
The wrapper defaults to `$PROJECT_ROOT/reports/runs/baselines/sad-k20s0`.

The report compares per-turn SDPA, per-turn Flex and shared-history Flex:
span-token/turn/trajectory log-prob sums, SAD loss differences, and gradients of
the **negative SAD span-token sum** for every tensor in the last two decoder
layers and final norm. It uses the existing acceptance limits: packing-only
side-sum relative difference ≤1e-3 and per-tensor gradient relative L2 ≤2e-2;
all paths must produce finite scores and selected gradients. Kernel-only
differences and token MAE remain diagnostics. No optimizer updates occur.

CPU tests exercise real tiny Gemma-4 text layers with full and sliding attention,
the production SAD span encoder, every parameter gradient, checkpointing,
logical budgets/checkpoint crossings, physical forward counts, and the GPU
tool's comparison/encoding path. The loss tolerance is 1e-6 relative/absolute.
Every parameter gradient must satisfy absolute error ≤1e-6, or **both**
relative L2 ≤1e-6 and maximum absolute error ≤1e-6 times that tensor's maximum
reference magnitude, tightening the existing shared-history tests' 1e-5 criteria.
This is tensor-relative fp32 agreement, not bitwise or elementwise absolute
1e-6 agreement: shared-prefix gradient accumulation changes summation order,
especially for near-zero elements. CPU evidence does not establish bf16 GPU/FSDP
acceptance or a speedup. No GPU, cluster or teacher/API run was performed here.

Validation completed with:

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/tf5-env/bin/python -m pytest -q
```

Result: **1,842 passed, 5 skipped, 2 failed**, 84 warnings, 1,463.98 seconds.
Full output: `logs/pytest_sadpack_cpu.txt`. All SAD/shared-history tests passed;
the new SAD test module also passed independently (11 tests).

The two failures reproduce independently and involve unchanged files:

- `tests/test_coa.py::test_schedule_full_support_and_original_recipe`:
  `official_sources()` expects AFM revision
  `9a6594d4661fa994f189074ecfdfd259eea87c6d`, but
  `git -C third_party/Agent_Foundation_Models rev-parse --show-toplevel HEAD`
  resolves to this workspace and its parent repository revision. No Git metadata
  was modified to bypass the check.
- `tests/test_theory_independent_quick.py::TestIndependentTheoryQuick::test_quick_has_no_failed_claims`:
  its fixed interpreter `theory/.venv/bin/python` is absent, so the subprocess
  cannot start. No scientific check was suppressed or reclassified.

The five skips cover absent ALFWorld, local probe artifacts, and a real seed
collection. `git diff --check` and both SAD shell wrappers' `bash -n` checks pass.
