# Kang Agent Distillation SFT + SAG port registration — 2026-10-05

Official pin: `8884b80ea3d22e53a2e1e4b9fd324600d13e0430`
([source](https://github.com/Nardien/agent-distillation/tree/8884b80ea3d22e53a2e1e4b9fd324600d13e0430)).
Paths below are relative to `third_party/agent-distillation/`.
This port runs the released non-FTP SFT baseline and test-time SAG under the
fixed harness. It does not claim an FTP reproduction or learning results.

| Official file:line | Replacement/change | Reason class |
| --- | --- | --- |
| `scripts/training/train_agent.sh:4-7`; `exps_research/finetune_sft.py:119-121` | Configured pinned Gemma 4, K environment-verified teacher seeds. No upstream model/data/adapter or model-name fallback. No retriever, judge, new teacher call or outside task pool. | model-origin contract X9 / fixed few-shot scope |
| `README.md:141-145`; `exps_research/build_prefix_memory.py:31-45`; `src/smolagents/agents.py:1455-1459,1488-1492` | **No first-thought prefix.** Partial prefix helpers exist, but the required end-to-end generation artifacts/workflow are not released. Do not fill in missing components or emulate prefill. | fixed user missing-code rule |
| `exps_research/train_utils/preprocess.py:148-172,176-213`; `exps_research/train_utils/utils.py:213-239` | Default `packing=turn`: each assistant reply and native 106 is supervised after its exact evaluation generation prefix, including Gemma's empty thought marker. Full native history stays masked; no template edit or extra planning sequences. The official one-sequence trajectory template has no history/generation asymmetry, so turn packing preserves its intended prefix equivalence on Gemma. `--packing episode` remains an explicit comparison option in the shared trainer. | engineering: Gemma thought-marker prefix equivalence |
| `exps_research/finetune_sft.py:137-146` | Preserve LoRA r64, alpha128, dropout .05, bias none. Enumerate full q/k/v/o/gate/up/down module names **under .language_model.**; exclude multimodal towers and lm_head. Save native PEFT safetensors adapter. | architecture adaptation X2/X6 |
| `scripts/training/train_agent.sh:5`; `exps_research/scripts_train/finetune_sft_agent.sh:11-21` | Preserve lr 2e-4 and two epochs. Batch 8 trajectories shrinks by X5; single-GPU accumulation uses one turn per forward, all turns of the selected trajectories per update; no quantization. Each turn's mean CE is weighted by its assistant-token share of the trajectory, then divided by the actual trajectory batch. Identical trajectory permutations, update sets, optimizer/scheduler steps and target exposure under both packings. | dose X5 / engineering: Gemma thought-marker prefix equivalence |
| `exps_research/finetune_sft.py:201-220`; `pyproject.toml:25-40,80-83` | Explicit `linear`, warmup 0, wd 0, AdamW .9/.999/1e-8, clip 1. Scheduler/warmup are omitted in official SFTConfig and inherited from the library. TRL is pinned to 0.19.1; **Transformers is not exactly pinned** (`>=4.0.0`). See verification note below. | make implicit defaults explicit; dependency limitation |
| `exps_research/scripts_train/finetune_sft_agent.sh:11,20-21`; `exps_research/finetune_sft.py:201-219` | Installed Stage 3a runtime, SDPA, bf16 frozen base/compute, fp32 LoRA masters, checkpointing on one B200. Max length 32768, reject overlength instead of truncating; no FSDP for this single-GPU LoRA branch. Full-FT branch retained. | engineering X1/X6/X7 / no truncation |
| `exps_research/finetune_sft.py:201-220`; `scripts/inference/run_agent_student.sh:123` | Explicit training seed 0 (official Trainer default 42); common seed-0 server and batch-invariant serving. | fixed user seed / serving protocol |
| `src/smolagents/agents.py:1435-1453,1493-1496`; `scripts/inference/run_agent_student.sh:114-128` | ALFWorld uses the shared ReAct prompt/parser/caps/template. Candidate integrity means the unchanged parser extracts an admissible action; Thought remains optional under X3. No candidate execution/forking; vote on exact parsed action strings instead of executor logs. | requested SAG adaptation / protocol X3 |
| `src/smolagents/agents.py:1461-1474,1501-1515`; `scripts/inference/run_agent_student.sh:10-13,122` | Preserve N=8, T=.4, top_p=.8, max1024. Stable largest group; earliest group on tie; its first sample. All singletons: first passing. None passing: sample 0, even if parser-invalid or inadmissible; normal harness semantics then apply. | unchanged vote/fallback rule |
| `src/smolagents/agents.py:1495`; `scripts/inference/run_agent_student.sh:119-123` | Shared server stop/template behavior; no CodeAgent-specific stop strings or separate SAG prompt. Greedy default request stays byte-identical. Record every candidate raw text/channel, parse/admissibility result, group and selection in step metadata/blobs. Usage totals cover all eight candidates, not just the winner. | protocol X3 / accounting |
| `scripts/inference/run_agent_student.sh:126-128`; `exps_research/finetune_sft.py:219` | Existing `serve_student --lora-adapter` schema, artifact/provenance hashes, and mandatory HF+PEFT/vLLM equivalence gate before evaluation. Same independent dev140 suite, no online update/selection. | serving integrity / engineering |
| `scripts/training/train_agent.sh:9-16`; `exps_research/finetune_sft.py:196,201-220` | Hash-bound pipeline stages, preserved failed attempts and training exposure; separate training and SAG evaluation jobs/output trees. No checkpoint overwrite or optimizer resume inside an interrupted stage. | provenance / engineering |

Scheduler verification boundary: `pyproject.toml:29` pins **TRL 0.19.1**,
`:82` specifies only **Transformers >=4.0.0**, and `finetune_sft.py:201-220`
sets neither scheduler nor warmup. Thus there is no uniquely pinned Transformers
version to cite from this repository. The explicit port recipe is the requested
TRL/Transformers linear decay with zero warmup. Offline inspection and a test
confirm installed Transformers **5.14.1** `training_args.py:767-778` defaults to
`linear`/0. Its `get_linear_schedule_with_warmup` is compared numerically at
every step. Historical TRL 0.19.1 dependency source is not installed locally;
no network retrieval or invented historical source/version verification is claimed.

All tests are CPU/fake based. Adapter schema compatibility and tiny-model
save/reload pass; a real r64 Gemma/vLLM equivalence report remains required on
the target GPU before any adapter evaluation. No upstream files were modified.

2026-10-05 engineering diagnosis supplied by the user from read-only HPG
inspection: the K20/s0 episode-packed run had one parser error at step 0 in
140/140 dev140 episodes; subsequent replies were well formed. This is evidence
for repairing the Gemma boundary mismatch, not a new learning result from this
checkout. No retraining or evaluation was run here. The recipe, effective config,
update ledger and adapter metadata record packing; logical and physical forward
tokens disclose repeated history compute, including incomplete attempts.
