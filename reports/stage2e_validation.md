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

## Amendment 3: kernel equivalence and adapter application — 2026-10-05

This amendment supersedes schema 3's effect-relative precision requirements.
The gate answers two separate questions: whether adapter vLLM matches HF+PEFT
within measured kernel noise, and whether the adapter has a visible effect.
The eight fixed verified TRAIN turns, token selection, scoring, model identities,
and server/provenance bindings retain their definitions.

Schema 4 requires all four inclusive inequalities, using token-weighted pooled
means and strict greater-than comparisons for outlier counts:

1. Pooled `adapter_error <= max(0.05, 1.5 * base_noise)`.
2. Pooled adapter-error `count(abs_diff > 0.5) <= 2 * base_noise_count(abs_diff > 0.5) + 2`.
3. Pooled `effect >= 2 * base_noise`.
4. Every prompt's `effect >= base_noise`, using that prompt's means.

The first two conditions check numerical equivalence; the last two check
adapter application. There is no per-prompt error ceiling or requirement to
scale precision with adapter effect. Reports retain every prompt's and the
aggregate's mean/max absolute differences for `adapter_error`, `base_noise`,
and `effect`, plus adapter/base counts above 0.25 and 0.5 nats. The aggregate
records its error ceiling and outlier allowance; those two tolerance fields
are null on prompt rows because they apply only to pooled measurements.
`min_effect_nats` records the applicable effect floor on every row. New reports
and HF references carry `schema_version: 4`.

`validate_report` requires schema 4 and recomputes every gate from all four
stored token arrays. Edited summaries, pass flags, counts, or threshold
metadata cannot bypass a failing measurement. Failed-report, adapter/server,
and endpoint checks remain enforced. Complete schema 2/3 reports can be
explicitly recomputed in memory with
`{**old_report, **summarize_prompts(old_report["prompts"])}` before validation.
This refreshes the schema, summaries, flags, counts, thresholds, and rule while
preserving identity metadata. Unconverted schema 2/3 reports, schema 1, and
missing/invalid measurements are refused.

Read-only recomputation covered every `report.json` under both
`/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/eval/ladder-s0-rai-dev140/`
and `ladder-s0-rai-dev140-b/`: thirteen reports, each with eight prompts and 236
scored tokens. Four k5 reports appeared during validation and were included in
the final recomputation. Names below are relative to the common `eval/` directory;
each source is `<name>/check/report.json`. Means are in nats. Outlier counts
are pooled adapter error / base noise above 0.5 nats.

| Name | Pooled error | Base noise | Effect | Outliers >0.5 (adapter / base) | Pass |
|---|---:|---:|---:|---:|:---:|
| ladder-s0-rai-dev140/k20s0-p10 | 0.024269774 | 0.046081653 | 1.124531027 | 1 / 2 | PASS |
| ladder-s0-rai-dev140/k20s0-p20 | 0.031197812 | 0.046081653 | 1.307455978 | 3 / 2 | PASS |
| ladder-s0-rai-dev140-b/k10s0-p03 | 0.049910179 | 0.046081653 | 0.143896720 | 5 / 2 | PASS |
| ladder-s0-rai-dev140-b/k10s0-p05 | 0.046671770 | 0.046081653 | 0.336831716 | 4 / 2 | PASS |
| ladder-s0-rai-dev140-b/k10s0-p10 | 0.028305977 | 0.046081653 | 0.918445327 | 1 / 2 | PASS |
| ladder-s0-rai-dev140-b/k10s0-p20 | 0.022092803 | 0.046081653 | 1.276593256 | 2 / 2 | PASS |
| ladder-s0-rai-dev140-b/k20s0-p03 | 0.039622564 | 0.046081653 | 0.470432301 | 3 / 2 | PASS |
| ladder-s0-rai-dev140-b/k20s0-p05 | 0.031740395 | 0.046081653 | 0.841183242 | 2 / 2 | PASS |
| ladder-s0-rai-dev140-b/k20s0-p20 | 0.030638280 | 0.046081653 | 1.288269865 | 6 / 2 | PASS |
| ladder-s0-rai-dev140-b/k5s0-p03 | 0.047745035 | 0.046081653 | 0.056921754 | 4 / 2 | FAIL |
| ladder-s0-rai-dev140-b/k5s0-p05 | 0.067385323 | 0.046081653 | 0.056914867 | 6 / 2 | FAIL |
| ladder-s0-rai-dev140-b/k5s0-p10 | 0.037103002 | 0.046081653 | 0.219798714 | 2 / 2 | PASS |
| ladder-s0-rai-dev140-b/k5s0-p20 | 0.057812333 | 0.046081653 | 0.676436713 | 4 / 2 | PASS |

Eleven recomputed reports pass `validate_report` against their recorded
`server/provenance.json`; the other two are refused. A separate calculation
directly from the four arrays confirmed each gate's result for every report.
Both failures were also rejected after identity checks when supplying a stale
passing flag, demonstrating rejection from measurements. The common pooled error
ceiling is 0.069122479 nats, effect floor 0.092163306 nats, and outlier allowance
6 tokens. Among passing reports, the smallest prompt effect/noise ratio is
1.296093 on `-b/k10s0-p03`.
`-b/k20s0-p20` passes exactly at the six-outlier boundary.

`-b/k5s0-p03` and `-b/k5s0-p05` pass both numerical-equivalence conditions but
fail the pooled effect floor and the requirement for visible effect on every
prompt. Their failing zero-based prompt indices are respectively `[0, 2, 3, 7]`
and `[1, 7]`; their minimum prompt effect/noise ratios are 0.155958 and 0.459634.
These negative application checks remain failures under schema 4.

The light-adapter failures illustrate why the two questions need separate
criteria: `-b/k10s0-p03`, `-b/k10s0-p05`, and `-b/k20s0-p03` failed schema 3
but pass all four schema 4 requirements. On zero-based prompt 1 of
`-b/k20s0-p03`, the maximum adapter error is 1.824928522 nats, on the same
prompt whose maximum base noise is 0.812375307 nats; its mean error is
0.095291296, base noise 0.062088057, and effect 0.412420542. The aggregate
noise and outlier checks accommodate this isolated large error without
requiring small adapter effects to imply proportionally smaller kernel error.

These are offline reclassifications of existing measurements. The original
schema 2 failure at `ladder-s0-rai-dev140/k20s0-p20` and all seven schema 3
failures remain unchanged on disk. Source SHA256s were checked before and
after recomputation; the inputs are recorded below for reproducibility.

| Name | Original report SHA256 |
|---|---|
| ladder-s0-rai-dev140/k20s0-p10 | `cd74c8dc87ba002313df3ad7256edf6afe5babce5383d1a4e4d8b4dde106b099` |
| ladder-s0-rai-dev140/k20s0-p20 | `a2749a1e92bd3891aa3d21f7d63b0951cc96aa5cd2bba6cb5dc6bf5c7c400933` |
| ladder-s0-rai-dev140-b/k10s0-p03 | `c446c14f0eb80d0527de057c4884c0fabc156d862ca4a57fbc31c23eba6a1621` |
| ladder-s0-rai-dev140-b/k10s0-p05 | `ce516ca7a701cfcecce0c5f32eef2e1dbac9d5d7830b261426569ab9a6946e7c` |
| ladder-s0-rai-dev140-b/k10s0-p10 | `fec89c04e0a6800f064651649b7f9fe9bed4b56f7f96ed294e2d7f4246f9ffa5` |
| ladder-s0-rai-dev140-b/k10s0-p20 | `919432e9db8898f89573fd884d0097d43933db58f07db6f4679392cbfa64e43f` |
| ladder-s0-rai-dev140-b/k20s0-p03 | `8afbaea39ebc5bbe36506ee9852fa8451db2bcea3fcb2efdcbd6086845ac2fc3` |
| ladder-s0-rai-dev140-b/k20s0-p05 | `7a19816bd8ac4c35c61a64f093dc383749aeb7e62ba513ea31e49509ae094124` |
| ladder-s0-rai-dev140-b/k20s0-p20 | `2eb91ca608259424b14e826d1d6ceedcf7a044b030aa2681dfe1c053eee9db6f` |
| ladder-s0-rai-dev140-b/k5s0-p03 | `7152e970357b84acb192d4c26e9ce612fa898522ff0902c10c032d4e0f669603` |
| ladder-s0-rai-dev140-b/k5s0-p05 | `30267252f222c4c8a52dc05c0f626c9e7333d7155a3a4339ddeb495c5865c863` |
| ladder-s0-rai-dev140-b/k5s0-p10 | `c1268b1e86065a603d7ad256cda060b3057cac58f200d79d38e545324507d0c8` |
| ladder-s0-rai-dev140-b/k5s0-p20 | `a4fa24a90923c374a9e5c94a47e8363ad0bcb940630f4b27bafeacb3c9928184` |

The schema 4 checker source SHA256 used for recomputation is
`cbbc677a00a86faa2253dc90cf40d0bbcd6a9995fffe709650db832793e2681e`.

Offline CPU validation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_lora_serving_torch.py tests/test_stage2b_torch.py
```

The full project suite passed **387 tests, 4 skipped**: three torch modules are
unavailable in the project venv, and real ALFWorld integration is optional.
The separate torch command passed **14 tests**, with the existing PEFT warning
about the synthetic model's absent source config. Regression coverage includes
light-adapter cases with 236 weighted tokens, isolated large token errors,
strict outlier cutoffs and inclusive allowances, independent failures of all
four gates, missing adapters, stale-summary/threshold rejection, complete
schema 2/3 recomputation, missing measurements, schema rejection, and CLI
reports/exit statuses. `git diff --check` passed.

Only `tools/check_lora_serving.py`, `tests/test_lora_serving.py`, and this
appended amendment changed. The existing suite fixture and torch tests needed
no edits. Earlier report text is unchanged. No GPU, real server, network/API
call, submitted job, commit, or push was used. No new inference or environment
evaluation was performed, and no scientific finding is claimed.
