# Stage 2e engineering validation — 2026-10-04

Implemented unmerged LoRA serving, request-model selection, and rai/HPG adapter
evaluation launchers. Each checkpoint retains its original files. Serving
provenance includes adapter config/weight hashes, training provenance binding,
supervised tokens, passes, packing, effective memory utilization, selected port,
and installed serving-source hashes. The existing merged-model path remains.

The equivalence gate uses eight deterministically selected verified TRAIN turns
and the existing ReAct/Gemma OFF rendering. It compares teacher-reply token
logprobs, including the native turn closure, under HF+PEFT bf16/eager, adapter
vLLM, and base vLLM. Each prompt and the pooled tokens must satisfy mean error
≤0.05 nats and positive adapter effect ≥5× that error. Reports bind the original
checkpoint and server run; missing, failed, or mismatched reports block adapter
evaluation. HF references run before server startup in the launchers, avoiding
simultaneous full-model allocations. No merged 24 GB checkpoint is produced.

Read-only inspection of the installed serving environment confirmed vLLM
0.27.1, Transformers 5.14.1, and the registered Gemma4 reasoning parser. Its
LoRA loader checks safetensor module suffixes and applies the Gemma HF-to-vLLM
prefix mapper, so native loading is the default. This is source evidence only.
Explicit `targets`/`gemma4` conversion fallbacks write under the new server
output, record config/key changes, and compare complete tensor payload bytes.
They do not automatically retry unrelated server failures.

Offline CPU validation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_stage2b_torch.py tests/test_lora_serving_torch.py
```

The project suite passed **173 tests**, with three expected skips (two torch
modules unavailable in the project venv and the optional real ALFWorld test).
The separate torch run passed **13 tests**, including the existing SFT/merge
checks; one PEFT warning concerns the synthetic model's absent source config.
Shell syntax checks and `git diff --check` passed. Tests cover name and identity
rejection, cross-host snapshot relocation, serving command/provenance fields,
byte-identical synthetic and real PEFT conversion, causal logprob alignment,
exact OFF prompts from fake verified evidence, report thresholds/exit status,
request-model routing, bound-port selection, listener ownership, and both
launchers' process order, cleanup, output refusal, and failed-check gating.
HTTP, environment, serving-process, and socket behavior use fakes.

No real GPU inference, vLLM server, network/OpenAI call, Slurm job, commit, or
push was performed. Existing `reports/runs/` and `data/verified/` outputs were
not changed, including the active `seedsft-k20-s0-rai` training run. Prompts,
parser, decoding, caps, template, training, and merge implementation are
unchanged. Real HF/vLLM equivalence, adapter loading, memory fit, and dev
evaluation on rai and HPG remain unmeasured and must pass the new gate before
any adapter evaluation is trusted. No scientific finding is claimed.

## Amendment: measured kernel noise floor — 2026-10-04

This engineering amendment supersedes the fixed equivalence threshold above.
Read-only inspection of the main checkout's
`reports/runs/eval/k20s0-rai-dev140/{k20s0-p03,k20s0-p05}/check/report.json`
confirmed that p03 passed schema 1, while p05 failed on a 34-token prompt:
mean adapter error 0.0563177 nats, maximum 0.6339703. Across 236 tokens,
p05's mean error was 0.0308351 and mean adapter effect was 0.8412603.
Those reports contain no HF base logprobs, so they cannot establish the
kernel noise floor or be re-judged under this amendment. No real inference
was rerun here, and no amended pass is claimed for either checkpoint.

HF reference generation now scores the same eight fixed verified TRAIN
turns twice on the same loaded PEFT model: adapter enabled (`hf_logprobs`)
and inside `model.disable_adapter()` (`hf_base_logprobs`). Both arrays are
stored in schema 2 `reference.json`; no second model is loaded. Token
selection, OFF rendering, native turn closure, and provenance/identity
bindings retain their existing definitions.

Schema 2 check reports record `mean_abs_diff` and `max_abs_diff` in nats
for each prompt and for all reply tokens pooled together:

- `adapter_error`: adapter vLLM versus HF+PEFT.
- `base_noise`: base vLLM versus HF with the adapter disabled.
- `effect`: adapter vLLM versus base vLLM.

Every prompt must satisfy
`adapter_error <= max(0.05, 1.5 * base_noise)`, using its own means.
The pooled means must satisfy the same inequality and
`effect >= 5 * adapter_error`. Pooling weights each token equally, rather
than averaging prompt means. The effect requirement applies only to the
pooled tokens; maximum differences are diagnostic. Each summary also
records its effective `mean_abs_diff_tolerance_nats`.

`validate_report` requires schema 2, preserves failed-report and
adapter/server/endpoint mismatch rejection, and recomputes both prompt and
pooled gates from the four stored logprob arrays. Stored summaries or
threshold overrides cannot bypass the rule. Schema 1, unversioned, and
unknown-version reports are refused. Missing, misaligned, or nonfinite HF
base references require regeneration before any serving comparison.
Check mode still exits 0 on pass and 1 on failure; numerical failures retain
`report.json`, while operational/input failures retain sanitized
`failure.json` and provenance.

Offline validation used the two requested commands:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_lora_serving_torch.py tests/test_stage2b_torch.py
```

The full project suite produced **223 passed, 3 skipped, 1 failed**.
The sole failure is
`tests/test_suite.py::test_process_suite_with_fake_server_and_environment[True]`:
its synthetic passing report omits both the schema version and HF base
measurements. The new validator correctly refuses it. This file is outside
the authorized four-file scope and was left unchanged; its fixture needs
schema 2 and measured synthetic HF base logprobs during integration.

The separate torch run passed **14 tests**, with the existing PEFT warning
about the synthetic model's absent source config. It verifies that disabling
LoRA on the same tiny CPU model exactly reproduces scores taken before
adapter attachment and restores the adapter afterward. Other regressions
cover a single model load in `--hf-only`, both saved HF references, the
observed 0.0563 error passing with base noise 0.038 and failing with 0.01,
236-token weighted pooling, independent prompt/pooled error failures,
pooled effect failure, inclusive thresholds, schema rejection, provenance
bindings, and CLI exit statuses. Synthetic noise values are test inputs,
not measurements of the real p05 kernel noise floor.

The focused checker suite (`tests/test_lora_serving.py`) passed **76 tests**
after the final validator regression update. `git diff --check` passed, and
the original report text was verified unchanged before this appended section.

Only the checker, its two test files, and this appended amendment were
changed. No GPUs, serving processes, network/OpenAI calls, submitted jobs,
commits, or pushes were used. Real run artifacts remained read-only.
