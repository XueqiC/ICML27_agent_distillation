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
