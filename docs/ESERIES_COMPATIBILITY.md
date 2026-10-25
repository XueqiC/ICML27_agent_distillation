# Gemma-4 E-series compatibility

This change adds the `gemma4` family without changing the default student or the
`gemma4_unified` training recipe. Configure the requested repository, pinned
revision and local snapshot explicitly. E4B was checked locally at
`ee0ef6023621cff504d758262d4e04895a5af4a2`; E2B is supported through the same family
but its snapshot was unavailable for direct validation. No production checkpoint weights were
loaded or trained in these checks: model tests use randomly initialized tiny configs.

## Implementation

- The tokenizer determines the OFF prefix. Only native model-turn prefixes with
  the legacy empty-thought prefill or no prefill are accepted. Token boundaries,
  reply transformations, special-token rejection and complete-context limits
  remain checked. Reports describe the actual prefix and zero history mismatch
  for E-series.
- Both text attention classes receive Flex kernel options. HF 5.14.1 forwards
  the supplied full/sliding mask dictionary and positions to every E-series text
  layer, including layers reusing earlier K/V tensors in the same forward. The
  implementation retains HF's shared-KV `UserDict` and configured sliding window.
- The loader dispatches by local `model_type`. E-series FSDP selects
  `Gemma4TextDecoderLayer` and verifies the count against `num_hidden_layers`.
  Full-parameter training includes text per-layer embeddings/projections; audio,
  vision and multimodal projections remain frozen and excluded from E-series
  root FSDP parameters. The existing replicated LoRA training strategy remains.
- LoRA uses the unchanged q/k/v/o/gate/up/down projection recipe, restricted to
  text decoder attention/MLPs. Shared-KV layers have no K/V adapters. Per-layer
  input gates/projections are outside this LoRA recipe. Full E-series target
  paths are retained after PEFT target minimization so provenance, reload and
  serving conversion record exactly the selected modules.
- SmartAD selection, dose-control validation and HF serving-reference scoring
  use these common loaders/encoders. Their own implementations and semantics
  did not require changes; tests exercise them with E-series models/adapters.
- With E-series `attention_k_eq_v=False`, value-LoRA derivation is a no-op.
  Optional serving conversion receives the snapshot family explicitly and
  records `model_type: gemma4` only for E-series. Original adapter files and
  tensor payload bytes are preserved.

## vLLM source verification

The local uv cache contains vLLM **0.27.1**. Source inspected under
`vllm/model_executor/models/`:

- `registry.py` maps `Gemma4ForConditionalGeneration` to `gemma4_mm`.
- `gemma4_mm.py:1013` maps `model.language_model.` to
  `language_model.model.`. Its packed-module mapping keeps q/k/v projections as
  the constituent names for `qkv_proj`, and gate/up for `gate_up_proj`.
- `gemma4.py` implements KV sharing and the packed projection loaders.

SHA-256 of the inspected files:

| File | SHA-256 |
| --- | --- |
| `gemma4_mm.py` | `027f4c7df0754d00a987f5c900d8fbb7d8e215ecbd0d0f5806d400e8a266cd96` |
| `gemma4.py` | `b2bf27dfb914d3afa62f4a126f3f7aff08845fb4be46f2469659097d7ff19100` |
| `lora/utils.py` | `46a10458705853012a906277ce099781e809ffaeb69371c6de8fce0392052643` |

The E-series conversion is a separate tested branch, even though the current
text-prefix rename matches the unified family. The unified conversion records,
config bytes, tensor headers and payloads match the reference implementation.

## CPU verification

Interpreter: `/home/xueqi/hq/projects/tc-alignment/.venv/bin/python`.
Versions: torch `2.13.0+cu130`, transformers `5.14.1`, PEFT `0.20.0`.
All tests run with `CUDA_VISIBLE_DEVICES=''`, offline Hugging Face settings,
and the repository's network/OpenAI transport blocking fixtures.

`tests/test_eseries_torch.py`: **52 passed**. Coverage includes:

- Real E4B and 12B templates, turn/episode packing, Unicode, whitespace, empty
  history turns, control-token rejection, truncation and token-boundary guards.
- Canonical byte equality against source loaded directly from
  `c6b15b25f92bc1a92709ba89fcbe3b3f8aa9e8e6`: 12B examples, labels, masks,
  provenance fields, mask reports, packed layouts, snapshot metadata and loader
  arguments; unchanged PEFT recipe and FSDP calls; adapter conversion and K=V
  derivation artifact bytes. Source hashes and commit identities naturally
  identify this new code revision rather than the reference revision.
- Tiny models from both real configs; E-series retains two KV-shared layers,
  per-layer inputs and both attention types with a seven-token sliding window.
  Full and LoRA FP32 loss/gradient comparisons cover eager/SDPA, checkpointing,
  unequal contexts, padding, label holes, and real tokenizer evaluation prefixes.
  Per-parameter relative L2 and max gradient bounds are `2e-5` plus `2e-6`
  absolute. The 256/512-head Flex routing test uses `1e-4` relative for FP32
  reduction-order differences; only its compiled kernel boundary is replaced
  by CPU SDPA. It checks all text layers and checkpoint recomputation.
- Real tiny audio/vision towers remain excluded from training, decoder wrapping,
  and Flex hooks. FSDP calls are intercepted without a process group.
- E-series local save/load, nonzero PEFT save/reload, HF base/adapter reference
  scoring, SmartAD candidate NLL, dose-control validation, pure serving rename,
  and absence of invented K/V adapters.

Full suites were run in four CPU batches each, using an untouched `git archive`
of the reference commit in `/tmp` (no other worktree was changed). The XML
comparison accounts for every reference test by class and test name.

| Run | Passed | Skipped | Failed |
| --- | ---: | ---: | ---: |
| Reference `c6b15b2` | 2,331 | 6 | 57 |
| Updated initial full run | 2,379 | 6 | 61 |
| Focused follow-up after startup fixture fix | 132 | 0 | 0 |
| Updated unique outcomes after follow-up | 2,383 | 6 | 57 |

The four initial new failures were the existing startup/handoff test fixture
omitting `config.json`. The fixture now supplies its model family and checks
wrap provenance. The provenance lookup uses a pure class-name mapping before
model imports/allocation. The follow-up reran all 80 startup/hash tests and all
52 compatibility tests. No reference test is missing, no reference pass became
a failure/skip, and all existing tests have exactly the same final outcome.

The 57 common failures are retained and disclosed:

- 53 require absent pinned third-party sources/prompts (LEAP, ETO, IPR,
  AgentGym, Agent Foundation Models, LMOps/GAD, and verl-gad).
- One requires the absent `theory/.venv/bin/python`.
- One probe test has an outdated encoder stub missing
  `empty_visible_turns_masked`.
- One process-pipeline byte-comparison test fails in both revisions.
- One two-process Gloo test is blocked by the sandbox socket restriction
  (`Operation not permitted`).

Six existing optional local-data/integration tests skip in both revisions.
These results do not imply that the full suite is green. `git diff --check`
also passes.

## Remaining hardware checks

No GPU validation was performed. Before production use, run the existing
HF/vLLM equivalence gate with real E-series weights and a nonzero LoRA, including
KV-shared layers, and run real FSDP/Flex forward/backward/export checks. CPU
kernel substitution does not establish CUDA Flex compilation or vLLM execution
correctness, memory usage, throughput, or real-weight FSDP behavior.

## Changed files

| File | Change |
| --- | --- |
| `train/dataset.py` | Native OFF prefix detection and accurate mask reports |
| `train/modeling.py` | Family dispatch, text decoder selection, E-series LoRA paths |
| `baselines/common/shared_history.py` | No-prefill packing and E-series text attention hooks |
| `baselines/common/fsdp_train.py` | Text-only E-series FSDP and decoder provenance |
| `harness/lora_adapters.py` | E-series K=V no-op and explicit naming conversion |
| `harness/serve_student.py` | Pass snapshot family to adapter conversion |
| `tests/test_eseries_torch.py` | 52 compatibility and reference regression tests |
| `tests/test_hash_parallel.py` | Include family metadata in the startup fixture and assert wrap provenance |
| `docs/ESERIES_COMPATIBILITY.md` | Implementation, verification evidence and GPU limits |
