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
