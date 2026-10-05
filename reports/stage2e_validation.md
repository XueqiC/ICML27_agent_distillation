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

## Amendment 2: effect-relative prompt gate and launcher continuation — 2026-10-04

This amendment supersedes the schema 2 per-prompt rule. The engineering
gate detects missing or misapplied adapters by comparing serving error
with the adapter's measured effect, while allowing isolated kernel-numerics
outliers. The eight fixed verified TRAIN turns, scoring, token selection,
model identities, and server/provenance bindings retain their definitions.

Read-only inspection and recomputation used the main checkout's
`/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/eval/ladder-s0-rai-dev140/`
reports, specifically `{k20s0-p20,k20s0-p10}/check/report.json` and the recorded
`server/provenance.json`. Both reports contain all four token measurement
arrays and 236 scored tokens. The following means are pooled over tokens;
outlier counts use strict greater-than comparisons.

| Adapter | Adapter error | Base noise | Effect | Effect/error | Adapter counts >0.25 / >0.5 | Base counts >0.25 / >0.5 | Largest prompt error/effect |
|---|---:|---:|---:|---:|---:|---:|---:|
| k20s0-p20 | 0.031197812 | 0.046081653 | 1.307455978 | 41.9086 | 6 / 3 | 10 / 2 | 0.0700332 |
| k20s0-p10 | 0.024269774 | 0.046081653 | 1.124531027 | 46.3346 | 4 / 1 | 10 / 2 | 0.0370356 |

The p20 schema 2 failures were zero-based prompts 2 and 5: mean errors
0.105254490 and 0.070331889 over 28 and 34 tokens, with maximum differences
2.719193935 and 1.021809667 nats. Their effects were 1.502923611 and
1.327441089 nats. The low pooled error, large effect, and similar adapter/base
outlier frequencies motivate this numerical gate change. Recomputing the
stored arrays under schema 3 passes both adapters, including validation
against their recorded server identity. This is an offline reclassification;
no new model inference or environment evaluation was performed. The original
p20 failed report and p10 passing report remain unchanged. Their SHA256s are:

- p20: `a2749a1e92bd3891aa3d21f7d63b0951cc96aa5cd2bba6cb5dc6bf5c7c400933`.
- p10: `cd74c8dc87ba002313df3ad7256edf6afe5babce5383d1a4e4d8b4dde106b099`.

Schema 3 requires all three inclusive inequalities:

1. Pooled `adapter_error <= max(0.05, 1.5 * base_noise)`.
2. Pooled `effect >= 5 * adapter_error`.
3. Every prompt's `adapter_error <= 0.1 * effect`, using that prompt's means.

Reports preserve per-prompt and pooled mean/max absolute differences for
`adapter_error`, `base_noise`, and `effect`. Each adapter-error/base-noise
summary now includes `tokens_abs_diff_gt_0_25` and `tokens_abs_diff_gt_0_5`;
their aggregate values are the per-run counts. Maximum differences and
outlier counts are diagnostics. `mean_abs_diff_tolerance_nats` records
0.1 times the prompt effect for prompt rows and the noise-based tolerance
for the aggregate. Pooling still weights each token equally.

`validate_report` accepts only schema 3 and recomputes gates from stored
measurements; edited summaries or threshold metadata cannot bypass them.
Schema 1 and unconverted schema 2 reports are refused. Complete schema 2
measurements may be explicitly recomputed with
`{**old_report, **summarize_prompts(old_report["prompts"])}`. This refreshes
schema, statistics, pass flags, counts, thresholds, and rule text while
preserving adapter/server binding metadata. Missing measurements require
regeneration. New HF references also use schema 3; complete schema 2 HF
references remain subject to the existing identity/code provenance checks.

The launcher now retains each failed check's report or sanitized failure
artifact, writes `<OUT>/<adapter>/check/FAILED` with the failed stage and
exit status, skips that adapter's evaluation, and continues in adapter
order. It exits 1 after the remaining checks/evaluations when any check
failed. HF reference failures receive the same marker and skip behavior;
their failure artifacts remain under `<adapter>/hf/`. `run_work` explicitly
returns the child's status so Bash conditional calls cannot mask failures.
Base evaluation, successful-run output layout, and process cleanup retain
their behavior. `INCLUDE_BASE=0` already supports adapter-only runs on rai
and HPG, so no additional option or change to `INCLUDE_BASE` was needed.

Final offline CPU validation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_lora_serving_torch.py tests/test_stage2b_torch.py tests/test_stage3a_torch.py
```

The full project suite passed **269 tests, 4 skipped**: three torch modules
are unavailable in the project venv, and real ALFWorld integration is optional.
The separate torch command initially produced **35 passed, 1 skipped, 1 failed**
because this worktree lacks `third_party/ETO/fastchat/train/dpo_trainer.py`,
which an existing stage 3a test reads for its verbatim/formula comparison.
The local main checkout has that source at the documented ETO pin
`a2fc5da38f8d00cfaf3f9b6370d586eebaf72904`; its bytes were checked against
`git show` at that commit (source SHA256
`ea04bab546bb84994c0c651b1da0165382c43eb0f5c9c6309fc39a9cb5071682`).

The torch command then passed **36 tests, 1 skipped** from temporary directory
`/tmp/gate2-torch-21tj7ym6`. That directory links this worktree's code, tests,
configuration, and data, plus only the missing ETO source file from the local
checkout. It leaves repository files unchanged and requires no download.
The remaining skip is the absent optional real seed collection; the existing
PEFT warning concerns the synthetic model's absent source config.

Regression coverage includes the p20-shaped 236-token outlier case, strict
count cutoffs, inclusive gate boundaries, missing adapters on one or all
prompts, independent pooled error/effect failures, stale-summary rejection,
complete schema 2 recomputation, and missing-measurement/schema rejection.
Fake launcher tests cover first/last/all check failures, operational and HF
reference failures, retained reports/markers, later adapter evaluation,
base inclusion/exclusion, order, output refusal, and cleanup on rai/HPG.
`bash -n scripts/eval_adapters.sh` and `git diff --check` passed.

Only `tools/check_lora_serving.py`, `scripts/eval_adapters.sh`, their two
non-torch test files, and this appended amendment changed. The existing
`tests/test_suite.py` fixture already provides complete measurements through
`summarize_prompts`, so it needed no edit. Earlier report text is unchanged.
No GPU, real server, network/OpenAI call, submitted job, commit, or push was
used. Real run artifacts remained read-only; no scientific finding is claimed.
