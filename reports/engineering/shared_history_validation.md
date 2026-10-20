# Shared-history scoring CPU validation — 2026-10-19

Branch: `wt/dpopack`. Engineering equivalence only; no experiments, teacher calls,
GPU execution or cluster jobs. The original SFT execution path remains unchanged.

Runtime: Python 3.11.7, Transformers **5.14.1**, torch **2.13.0+cu130**, CPU fp32,
`CUDA_VISIBLE_DEVICES=''`, Hugging Face offline mode. Cached packages were read
without installing or changing the linked project environment. Pytest's existing
fixture blocks DNS, socket traffic and the OpenAI transport. All models used for
numerical checks are tiny random Gemma4Unified models, with sliding window 7 and
both sliding and full layers. Real local tokenizer tests use the pinned Gemma4
template; no model weights are downloaded.

Commands and outcomes (overlapping suites, not unique-test totals):

```text
python -m pytest -q tests/test_shared_history_torch.py tests/test_baseline_packing.py
33 passed (21 new acceptance tests, 12 existing packing tests)

python -m pytest -q tests/test_stage3a.py tests/test_stage3a_torch.py \
  tests/test_stage3c.py tests/test_stage3c_torch.py tests/test_baseline_packing.py \
  tests/test_baseline_training_options.py tests/test_empty_visible_turns.py \
  tests/test_baseline_dose_controls_torch.py
165 passed, 1 skipped (real seed collection not installed)

python -m pytest -q tests/test_stage2b_torch.py tests/test_baseline_k_ladder.py \
  tests/test_checkpoint_export_torch.py tests/test_baseline_training_options.py \
  tests/test_baseline_packing.py
65 passed

CUDA_VISIBLE_DEVICES='' python -m tools.benchmark_shared_history \
  --output reports/engineering/shared_history_cpu.json
```

The acceptance tests check per-turn log-probabilities, ETO sigmoid and IPR
sigmoid-plus-chosen-summed-NLL, every parameter gradient, checkpointing, unequal
turn counts, zero-weight rank slots, actual forward-token accounting, optimizer
updates, empty visible turns, pinned tokenizer paths, block coverage and full-block
validity. Gradient comparison requires relative L2 and relative maximum error
<=1e-5 for **each parameter tensor**, with 1e-10 absolute tolerance at zero.
Elementwise relative error near cancellation zeros is not a stable fp32 metric.
Losses and turn scores use rtol 1e-5. No tolerance masks a missing gradient.

The HF FlexAttention interface is exercised using torch's CPU math forward path.
Torch 2.13 explicitly rejects CPU FlexAttention backward, and its compiled CPU
forward's vector ISA probe also fails in this local runtime. Therefore gradient
acceptance uses the identical visibility predicate through CPU SDPA/eager, and
GPU fused kernel execution/backward remains untested. This does not trigger a
dense GPU fallback: the production path accepts only FlexAttention BlockMasks.
The torch 2.11/2.13 `from_kv_blocks` APIs were inspected locally; the cached 2.11
wheel cannot be imported with this host's CUDA runtime libraries, so numerical
results here apply to torch 2.13 only.

The [benchmark JSON](shared_history_cpu.json) contains effective tiny-model config,
seeds, source hashes, dependency versions, repeated wall times and measured root
forward token counts. Its dense CPU timing backend is disclosed explicitly.
The long-mask check builds BlockMasks for 30,829 packed tokens: 1,866,304 bytes
of integer block metadata and 493,264 bytes of position/branch vectors. These
are shared per attention type; no token-square tensor is constructed in this
path. GPU/FSDP peak memory, compilation and throughput have not been measured.

PyTorch interface reference:
[BlockMask.from_kv_blocks](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask.from_kv_blocks).

## Large-head Flex kernel follow-up — 2026-10-19

The reported hpg B200 acceptance failure occurs in
`pair_0/shared_history/no_grad_sums`: the default Triton tiles for Gemma4 global
attention (`head_dim=512`) require 102400 bytes of shared memory against a 101376
byte hardware limit. This diagnosis and the rai GPU reproduction were supplied
by the user; no GPU or cluster was run for this follow-up.

Shared-history scoring now passes a dedicated options kwarg through the HF
Gemma4 model and decoder path. An attention-module pre-hook forwards it as
`kernel_options` only when `head_dim > 256`. Smaller heads retain torch defaults.
The hook remains installed during non-reentrant checkpoint recomputation; it
does not replace HF's attention function or mutate a global attention registry.

The default large-head forward tiles are `BLOCK_M=BLOCK_N=32`. When installed
torch's `FlexKernelOptions` declares separate backward tiles, all four
`bwd_BLOCK_M1`, `bwd_BLOCK_N1`, `bwd_BLOCK_M2`, `bwd_BLOCK_N2` also default to 32.
The BlockMask size stays 128; tile sizes must be divisors of 128 and at least 16.
Backward divisibility constraints are checked before a kernel launch. Overrides
are restricted to tiles and `num_warps`/`num_stages` (including `fwd_`/`bwd_`
prefixes for the latter), preserving the visibility predicate.

Both `baselines.common.fsdp_train` and `tools.validate_shared_history_gpu` accept
`--flex-kernel-options` as a JSON object of overrides, for example:

```text
--flex-kernel-options '{"BLOCK_M":32,"BLOCK_N":32,"bwd_num_stages":1}'
```

Explicit backward tiles override the defaults derived from the forward tiles.
Resolved options, the head-dimension threshold and mask block size are recorded
under `flex_attention` in training state/results, every update ledger row,
training provenance and pipeline disclosure. The GPU acceptance report records
them in settings, provenance and the shared-history path. Runtime failures now
include full `error_message=str(error)` and `error_repr=repr(error)` alongside
the exception type and frames, preserving resource errors and completed phases.

CPU regression tests exercise a tiny real Gemma4 with 256/512-dimensional heads
and mock only HF's compiled Flex boundary with SDPA. They check default and
overridden options, 128-token BlockMask compatibility, checkpoint recomputation,
ledger/state agreement, and full multiline failure messages. These checks do
not establish GPU compilation, bf16 equivalence, or FSDP acceptance.

Follow-up verification with `/home/xueqi/tf5-env/bin/python`,
`CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`:

```text
python -m pytest -q tests/test_shared_history_torch.py tests/test_validate_shared_history_gpu.py
46 passed

python -m pytest -q
1376 passed, 5 skipped, 1 failed (946.68 s)
```

The full-suite failure is the unchanged
`tests/test_theory_independent_quick.py::TestIndependentTheoryQuick::test_quick_has_no_failed_claims`:
its hard-coded `theory/.venv/bin/python` does not exist in this worktree, so the
subprocess cannot start (`FileNotFoundError`). The test and theory checker are
unchanged from `31e4875`; no scientific assertion was relaxed or suppressed.
Skips are for missing ALFWorld, local probe artifacts, and a real seed collection.
All changed-path tests, including the simulated-rank CLI/provenance checks,
passed in the full run. `git diff --check` also passed.

## Matched Flex control — 2026-10-19

The user reports that the 32x32 hpg run executes but fails the old SDPA-versus-
packed-Flex thresholds: side-sum relative difference 0.0037 and gradient relative
L2 0.06–0.15. Signed per-turn differences also occur on each side's first turn.
Kernel numerics are a hypothesis to test; these observations do not establish
packing equivalence. No GPU was run for this follow-up.

`tools.validate_shared_history_gpu` now measures three paths on the same model:

- `per_turn`: the existing SDPA scorer, one forward per supervised turn.
- `per_turn_flex`: the same input IDs and labels, one forward per turn, sequential
  positions and a single unbranched causal history. Its full/sliding BlockMasks
  use the same `attention_masks` builder, block metadata and kernel-option routing
  as `shared_history`. Defaults and `--flex-kernel-options` apply to both paths,
  including non-reentrant checkpoint recomputation.
- `shared_history`: the existing packed Flex scorer, one forward per side.

Pair reports and aggregate summaries contain `comparisons.kernel_only`
(`per_turn` vs `per_turn_flex`), `comparisons.packing_only` (`per_turn_flex` vs
`shared_history`), and `comparisons.combined` (the original comparison). Each
reports side sums, per-turn signed/absolute/relative differences, gradient
objective differences, and every selected tensor's gradient relative L2. The
reference path supplies the relative denominator.

Acceptance uses only the **packing-only magnitudes**: every side-sum relative
difference must be <=1e-3 and every gradient tensor's relative L2 <=2e-2. All
three paths must still produce finite scores and present, finite selected
gradients. The kernel-only magnitude is diagnostic.

The per-turn diagnostic pass also collects the actual supervised-token
log-probabilities, retaining causal target alignment and 128-token fp32
log-softmax chunks. Each comparison reports `mean(abs(actual - reference))`
per turn, per side, and pooled by token count across pairs. This is not the
absolute difference of turn sums divided by token count: opposite token errors
cannot cancel. Reports compare these means with the project's HF/vLLM gate
base tolerance of **0.05 nats/token** (`tools/check_lora_serving.py`);
this is a diagnostic reference, not an additional packing acceptance threshold
or a reproduction of the serving gate's noise-adjusted rule.

CPU verification uses dense SDPA mask oracles; the large-head routing tests
mock only the compiled Flex boundary. Tests cover both causal/sliding masks,
default/overridden tiles, checkpoint recomputation, unchanged input/forward-token
counts, token-error cancellation, token-weighted aggregation, threshold
boundaries, diagnostic-only kernel differences, and nonfinite/missing results.

```text
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/tf5-env/bin/python -m pytest -q \
  tests/test_validate_shared_history_gpu.py tests/test_shared_history_torch.py
54 passed (25.57 s)
```

GPU bf16 acceptance remains pending the next hpg run.
