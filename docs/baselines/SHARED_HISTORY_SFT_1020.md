# Shared-history SFT / SmartAD implementation — 2026-10-20

Status: shared-history scoring and replicated LoRA training support the original
Kang/AD-FTP/SmartAD **LoRA dropout 0.05**, retained by the user's explicit decision.
Nonzero LoRA dropout no longer blocks shared-history or multi-rank training.
No dropout-zero production option or default has been added. Equivalence is
recorded as `"dropout_equivalence": "expected_gradient"` at nonzero dropout and
`"exact"` at dropout zero. GPU acceptance and performance remain unmeasured here.

## Scoring and objective

`baselines/common/fsdp_train.py` accepts opt-in `--scoring shared_history` for
turn-packed `sft` and `smartad`. Both still default to `per_turn`; STeCa and HAD
remain per-turn-only. Full-parameter SFT also supports the new path. SAD,
DPO/IPR, and WKM defaults remain unchanged.

The shared path calls the existing `pack_shared_history`, exact visibility masks,
position IDs, selected-token scorer, and large-head FlexAttention settings. Each
trajectory requires one policy forward and one backward. It retains the native
history replies as masked context and isolated scored copies with Gemma OFF's
empty thought marker. Targets include native `<turn|>` ID 106. No tokenizer,
conversation encoder, visible-field whitelist, or truncation limit changed:
individual exact-prefix contexts exceeding 32768 tokens are refused.

For a global batch of B trajectories, let n_i count supervised reply tokens:

- SFT: `L = (1/B) sum_i [sum_t NLL_it / n_i]`. The old path uses one turn per
  microstep weighted by `n_turn/n_i`; the new path uses one trajectory weighted
  by 1. It does not average turns equally.
- SmartAD: `L = sum_it w_it NLL_it / sum_it w_it`, with reason/action/final weights
  1/1.5/2 and the original reason weight on the turn-end token. Its denominator
  spans the actual global batch, including a final partial batch. It is not a
  mean of trajectory means. Shared microsteps use the same batch denominator;
  weights are taken only from supervised positions, in label order.

With dropout > 0, per-turn scoring draws an independent LoRA-input dropout mask
collection for each turn forward; shared-history scoring draws one collection
per trajectory forward, reusing the stochastic history activations across turns.
Each collection contains elementwise masks at the individual LoRA modules; it
is not one scalar mask shared by every token. The visible causal graph for each
turn has the same marginal mask distribution under either layout.

At fixed parameters, write the loss as `L = sum_t a_t L_t`, where the original
token/trajectory weights `a_t` do not depend on dropout. Linearity gives
`E[grad L] = sum_t a_t E[grad L_t]` under either layout. Thus both optimize the
same dropout-regularized objective with the same expected gradient. Sharing
changes correlations between turn gradients and hence their variance. Individual
losses, gradients, and training runs are not bitwise reproducible across the two
scorings; equal expected gradients also do not imply equal Adam updates after
nonlinear clipping and optimizer transformations. Exact numerical equivalence is
established at dropout zero by the existing 1e-6 loss/gradient tests. That label
does not claim bitwise equality of different floating-point accumulation orders.
The separate zero-attention-dropout requirement remains for FlexAttention;
LoRA-input dropout is supported at 0.05.

## Parallel training, identity, and accounting

Kang, AD-FTP, and SmartAD accept `--gpus {1,2,4,8}` (default 1) and launch
`python -m torch.distributed.run --standalone --nproc_per_node N -m
baselines.common.fsdp_train`, matching SAD's launcher structure.

The LoRA branch replicates the frozen bf16 base and fp32 adapters. Rank zero
checks/encodes the verified data and publishes provenance/checkpoints. Adapter
initialization is broadcast. `rank_slots` distributes the existing microsteps;
missing slots execute a real duplicate forward/backward with zero loss weight.
Each active loss is scaled by `world_size / actual_batch_size` and its original
microstep weight. Accumulated adapter gradients are averaged once per update,
**before** global gradient clipping and AdamW. The optimizer, scheduler, sampler,
b', and logical trajectory exposure are unchanged. The full-parameter FSDP path
continues to use its existing FSDP reductions.

One rank versus N ranks likewise preserves the expected accumulated gradient
before clipping, while changing dropout draws and gradient variance. Bitwise
equivalence across world sizes is not claimed with nonzero dropout. Every rank
is seeded with `seed + rank`, including a reset immediately before training so
preparation cannot shift its dropout stream. Adapter initialization is still
broadcast from rank zero, and all ranks use the common `seed` for global batch
sampling. This makes RNG streams reproducible run to run for a fixed
configuration and execution environment. Training provenance records
`rng_seed_policy` and the explicit `rank_seeds` list; training state records its
rank's `rng_seed` as well as `dropout_equivalence`.

Pipeline identities include `gpus`, effective `recipe.scoring`,
`recipe.dropout_equivalence`, and `recipe.rng_seed_policy`; the three AD recipes
retain `recipe.dropout: 0.05`. Training provenance records the actual model's
`dropout_equivalence` under `config.baseline_training`, which also contributes to
the exported checkpoint's identity hash. Stage keys include the pipeline
identity, so changed scoring, rank count, or equivalence policy cannot reuse a
completed stage.
Slurm wrappers expose `NGPUS`, `KANG_SCORING`, `ADFTP_SCORING`, and
`SMARTAD_SCORING`; all three scoring defaults remain `per_turn`. Set the Slurm
GPU allocation and `NGPUS` together. SmartAD selection/evaluation keep their
single-GPU allocation requirement. CPU collection remains a separate stage.

Ledgers retain logical tokens and trajectories separately from packed physical
forward tokens and zero-weight padding. Shared SFT/SmartAD report trajectory
microsteps and no reference or recomputation pass.

## CPU exactness evidence

`tests/test_sft_shared_history_torch.py` uses the existing SAD tiny Gemma fixture,
real SFT/SmartAD encoders, CPU fp32, and zero dropout. LoRA uses Kang's r64/a128
text targets. Nonzero synthetic B matrices exercise A gradients as well as B;
these fixtures are not trained foundation checkpoints.

The tests cover both objectives, full and LoRA parameters, eager/SDPA masks,
non-reentrant checkpointing, unequal multi-turn trajectories, singleton final
batches, and zero-weight rank slots. Loss and every participating gradient use
the SAD 1e-6 criterion: absolute maximum error <=1e-6, or both tensor-relative L2
and tensor-relative maximum error <=1e-6. Tokens and masks retain the existing
bitwise checks. Shared-prefix gradient addition order need not be bitwise equal.

A separate CPU test, `test_dropout_mean_gradients_agree`, uses dropout 0.05,
256 deterministic samples per scoring path, short unequal turns, and the real
tiny Gemma/Kang LoRA SFT computation. Both A and B gradients participate; B is
initialized with standard deviation 0.001 for this diagnostic. Disjoint seeds
provide independent samples across paths. The test checks every gradient element
within six estimated standard errors plus 1e-7, and each tensor's mean-error L2
within four standard-error L2 norms plus 1e-7. An aggregate standard-error bound
of 10% of the mean-gradient norm prevents a vacuously wide tolerance. This is a
statistical gradient check, separate from the unchanged dropout-zero 1e-6 tests.

The standalone statistical test passed in **12.88 s** (20.88 s including pytest
startup). Mean-gradient relative L2 difference was **0.02795**, versus estimated
standard-error relative L2 **0.03223**. Four additional tests exercise real CPU
training with dropout 0.05, check `seed + rank`, and verify identical parameters
on repeated runs despite different ambient RNG states. Collectives in those
rank-local tests are faked; the real Gloo test remains separate and unchanged.

The three-trajectory CPU diagnostic is saved in
`reports/engineering/sft_shared_history_cpu_1020.json`:

| Objective / parameters | Loss relative difference | Max gradient absolute error | Max tensor gradient relative L2 |
|---|---:|---:|---:|
| SFT / full | 8.39e-8 | 2.09e-7 | 3.46e-7 |
| SFT / Kang LoRA | 3.37e-8 | 1.19e-7 | 2.69e-7 |
| SmartAD / full | 5.03e-8 | 2.38e-7 | 5.23e-7 |
| SmartAD / Kang LoRA | 6.74e-8 | 9.69e-8 | 3.51e-7 |

These synthetic rows require 3682 per-turn physical forward positions versus
2133 shared positions. This ratio is not a measured production-data speedup.

A real 17-trajectory epoch exercises b'=2 and its final singleton, identical
sampler/LR/loss accounting, actual forward-token hooks, and one shared forward
per trajectory. Nine successive Adam updates agree within 3e-6 in parameter
space (one SmartAD element differed by 1.69e-6 with a 1e-6 update threshold);
the stricter loss/gradient tests pass. This distinction is reported rather than
claiming bitwise optimizer trajectories.

A separate test spawns one process and two Gloo processes, checks both objectives
and scorings, and compares every adapter gradient/update, exact inter-rank
parameter equality, and final zero-weight padding. FileStore rendezvous uses
only local files; Gloo uses loopback communication. In this restricted sandbox,
Gloo initialization fails with `Operation not permitted` in
`gloo/transport/tcp/device.cc`. The actual two-process test was attempted and is
not reported as passing. Run it in an environment that permits loopback IPC:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/tf5-env/bin/python -m pytest -q \
  tests/test_sft_shared_history_torch.py::test_two_process_gloo_matches_one_process_lora
```

All unit tests prohibit external network/API access and use synthetic/fake data.
No model download, teacher call, real environment interaction, commit, or push
was performed.

## One-B200 acceptance

The existing tool is `tools/validate_shared_history_gpu.py`, also used by
`scripts/hpg_validate_sad_shared_history.slurm`. It now accepts a verified
training collection and `--objective sft|smartad`, with Kang LoRA r64/a128 and
explicitly disclosed **diagnostic dropout 0**. It uses four real checked receipts
by default; requesting more than available fails instead of undersampling.
SmartAD uses its real segment encoder on the same collection; this numerical
diagnostic does not perform or claim SmartAD candidate selection.

For each row, it compares:

1. Per-turn packing with SDPA.
2. The same per-turn packing with FlexAttention (kernel-only control).
3. Shared-history packing with FlexAttention.

The report records actual normalized loss relative difference, every trainable
LoRA tensor's relative gradient L2 (and its maximum), token MAE, forward counts,
physical tokens, wall time, peak memory, input/code hashes, student identity,
template hash, dependency versions, and no training updates/teacher calls.
Packing-only acceptance requires relative score and loss differences <=1e-3 and
every gradient tensor's relative L2 <=2e-2, with all paths finite and present.
Kernel-only and combined differences and the 0.05 nats/token MAE reference are
reported separately. Zero initial B produces zero A gradients; CPU tests also
exercise nonzero B. Cold compilation is included in timings, so these are not
steady-state throughput measurements.

Run both objectives, each in its own process on the same one-B200 allocation:

```bash
sbatch --gpus=1 \
  --export=ALL,VERIFIED=/path/to/verified,OUTPUT_PREFIX=/path/to/sft_shared_history_gpu \
  scripts/hpg_validate_sft_shared_history.slurm
```

Or, after loading the cluster configuration:

```bash
source configs/hpg.env
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for objective in sft smartad; do
  "$TRAIN_PYTHON" -m tools.validate_shared_history_gpu \
    --config configs/base.yaml --snapshot "$MODEL_PATH" \
    --verified "$VERIFIED" --objective "$objective" --num-rows 4 \
    --output "reports/engineering/${objective}_shared_history_gpu_1020.json"
done
```

No GPU acceptance result or B200 speedup was measured in this workspace. CPU
SDPA/eager masks validate mathematical sharing; only the cluster command tests
actual CUDA Flex kernels and bf16 numerics.

## Original validation commands and results (before this follow-up)

Interpreter: `/home/xueqi/tf5-env/bin/python`; PyTorch 2.14.0+cu130,
Transformers 5.17.0.dev0, PEFT 0.18.0. This is the available local test stack, not
a claim that it matches the pinned production package versions. The worktree's
pre-existing `.venv` symlink has no torch and was left unchanged.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/tf5-env/bin/python -m pytest -q \
  tests/test_sft_shared_history_torch.py -k 'not two_process'
# Initial focused run: 40 passed, 1 deselected, 47.82 s.
# Two dropout-guard tests also passed then; this follow-up replaces them.

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/tf5-env/bin/python -m pytest -q \
  tests/test_sft_shared_history_options.py tests/test_validate_shared_history_gpu.py
# Final run: 49 passed, 18.19 s.
```

The affected suite includes every test importing `baselines.common`, `kang`,
`adftp`, `smartad`, `sad`, or `shared_history`, plus the new option tests. Its
41-file list is saved in `reports/engineering/shsft_affected_tests_1020.txt`.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/tf5-env/bin/python -m pytest -q \
  $(cat reports/engineering/shsft_affected_tests_1020.txt) \
  -k 'not two_process_gloo'
```

Initial result: **842 passed, 38 failed, 1 skipped, 1 deselected**, 733.55 s.
Thirty-seven failures were missing local upstream sources: this worktree had no
`third_party/` directory. The other failure was the Kang legacy launcher/identity
fixture, updated for torchrun and explicit default scoring. No scientific
checks or source pins were relaxed.

During validation, `third_party` temporarily linked to the existing main
checkout's upstream sources at
`/home/xueqi/hq/projects/ICML27_agent_distillation/third_party`. CoA and GAD's
pinned-source checks passed. No repository was copied or fetched, and the link
was removed after validation. A fresh worktree needs these pinned dependencies
available before reproducing the broader suite.

All 38 initial failures, the new numerical/option tests, existing Kang CPU
adapter tests, and SAD shared-history regression tests were then run together:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/tf5-env/bin/python -m pytest -q \
  $(cat reports/engineering/shsft_initial_failed_tests_1020.txt) \
  tests/test_sft_shared_history_torch.py tests/test_sft_shared_history_options.py \
  tests/test_stage3b_torch.py tests/test_sad_shared_history_torch.py \
  -k 'not two_process_gloo'
```

Result: **138 passed, 1 deselected**, 101.11 s. This included the two original
fail-closed dropout checks, since replaced by stochastic-training checks.
All 38 initial failures cleared. The one existing
skip requires an unavailable real seed collection. The separately attempted
Gloo test remains **1 failed**, 25.39 s, solely at sandbox-denied socket creation;
its two-rank numerical assertions have not run here.

Logs: `logs/pytest_shsft_affected_initial.txt`, `logs/pytest_shsft_rerun.txt`, and
`logs/pytest_shsft_gloo.txt`; final validator/options output is in
`logs/pytest_shsft_options.txt`. `git diff --check`, Python compilation, and `bash -n`
for all four changed/new SFT Slurm wrappers passed.

## Dropout-0.05 follow-up validation

The affected regression command for this follow-up is:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/tf5-env/bin/python -m pytest -q \
  tests/test_sft_shared_history_torch.py tests/test_sft_shared_history_options.py \
  tests/test_baseline_training_options.py tests/test_stage3b_torch.py \
  tests/test_sad_shared_history_torch.py tests/test_shared_history_torch.py \
  tests/test_validate_shared_history_gpu.py tests/test_smartad_torch.py \
  tests/test_stage3a_torch.py tests/test_stage3c_torch.py tests/test_wkm_steca_torch.py \
  -k 'not two_process_gloo' --durations=10
```

The existing pinned ETO/IPR sources are needed for their upstream-loss checks.
As in the original validation, `third_party` temporarily links to the main
checkout's existing sources and is removed afterward; no network fetch is used.
The Gloo test is intentionally deselected and left unchanged for the user to run
outside this sandbox.

Initial affected result: **232 passed, 3 failed, 1 skipped, 1 deselected**, 118.02 s.
The metadata helper initially assumed every model had a Hugging Face config;
the existing SmartAD toy model exposed that assumption, now fixed. The other
two failures were the absent ETO/IPR sources. The earlier statistical fixture
with B standard deviation 0.01 failed the precision bound; the final diagnostic
uses 0.001 while retaining nonzero A/B gradients and the statistical tolerances.
These initial results are retained in `logs/pytest_shsft_dropout_affected_1020.txt`,
`logs/pytest_shsft_dropout_1020.txt`, and `logs/pytest_shsft_dropout_final_1020.txt`.
The final small-adapter standalone result is in
`logs/pytest_shsft_dropout_small_adapter_1020.txt` with metrics in its adjacent XML.

Final affected result: **235 passed, 1 skipped, 1 deselected**, **117.20 s**.
The skip needs a real seed collection absent from this worktree; the deselection
is the unchanged Gloo test. The 256-sample gradient test took **12.01 s** in this
run and reproduced the metrics above. The suite also checks both equivalence
labels in training provenance and exported checkpoint identities, the unchanged
0.05 recipe in all three pipeline identities, and deterministic rank-local runs.
No GPU, network, or OpenAI API was used. The temporary dependency link was
removed. Python compilation and `git diff --check` passed; no commit was made.

Final log: `logs/pytest_shsft_dropout_affected_final_1020.txt`; test timings and
gradient statistics: `logs/pytest_shsft_dropout_affected_final_1020.xml`.
