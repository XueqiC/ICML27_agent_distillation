# AgentGym BC / AgentEvol port registration — 2026-10-05

Official pin: `3ef9235d23e68e7c2920c5422ad957dc8ced5c6c`
([source](https://github.com/WooooDyy/AgentGym/tree/3ef9235d23e68e7c2920c5422ad957dc8ced5c6c)).
Paths below are relative to `third_party/AgentGym/`. `bc` means
`agentenv/agentenv/trainer/bc_trainer.py`; `bc.sh` means
`agentenv/examples/behavioral_cloning/train_behavioral_clone_alfworld.sh`;
`evol.sh` means `agentenv/examples/agentevol/train_agentevol_multi_task.sh`.
This is an engineering port under the fixed few-shot black-box protocol.
No performance or full experimental reproduction is claimed.

| Official file:line | Replacement/change | Reason class |
| --- | --- | --- |
| `bc.sh:8,12`; `evol.sh:11,17` | Configured pinned Gemma 4, local-only Stage 3a loader and installed torch/transformers/vLLM stack. No upstream checkpoint, dataset, or unspecified Accelerate config is loaded. | engineering X1/X2 |
| `bc:118-139`; `agentenv/agentenv/envs/alfworld.py:227-243` | Existing Gemma OFF episode template/masks: every actual assistant turn plus native 106 supervised, observation/system/header tokens masked. No fabricated acknowledgement or first-thought prefix. Preserve Stage 3a's documented non-final history serialization. | protocol harmonisation X2/X3 |
| `agentenv/agentenv/envs/alfworld.py:565-591`; `bc.sh:43-48` | Shared harness and unchanged prompt/parser; admissible list every turn, 40 turns, 1024 output tokens, thinking OFF, unchanged termination/parser-feedback behavior. Frozen dev140 evaluation is a separate step. | protocol harmonisation X3 |
| `bc.sh:11`; `evol.sh:189,201,218` | D0 is exactly K checked executed seed trajectories; exploration is exactly one rollout per seed TRAIN task per round. No outside exploration pool and no new teacher calls. K, unique instances, task groups, source lineage, and expanded exposures are separate. | fixed user scope overriding specification option |
| `agentenv-alfworld/agentenv_alfworld/env_wrapper.py:30-72,117` | Frozen manifest trial-directory IDs with TRAIN authorization, not official integer mappings or valid_train evaluation. | engineering X8 / protocol X3 |
| `bc.sh:3,6,15-23`; `bc:285-321` | Keep lr 1e-5, linear, **integer floor** warmup `int(.1*total_steps)`, wd 0, clip 1, one epoch. Global batch alone shrinks from 32 via X5. Steps use actual complete/partial batches. Explicit integer warmup overrides the shared default ceil rule, including zero. | few-shot dose X5 |
| `bc:295-316,359-378`; `evol.sh:20-22` | Stage 3a full text-parameter FSDP2 on four GPUs, fp32 masters / bf16 compute, AdamW .9/.999/1e-8, SDPA, checkpointing. Frozen multimodal towers exported from the base. Microbatch one turn, accumulating every turn of the selected trajectories; episode-mean CE then actual global-batch mean; partial/rank padding never adds logical exposure. | engineering X6/X7 and exact X5 realization |
| `bc:135-139,295-316`; `bc.sh`; `evol.sh` | BC and all AgentEvol SFT stages default to recorded `packing=turn`, using `train.dataset` turn rows: each supervised reply follows the exact Gemma OFF evaluation generation prefix, with the empty thought marker masked and complete native history preserved. X5 N and batches still count trajectories, including repeated instances; per-turn assistant-token weights preserve trajectory-mean CE and identical update sets/step counts. Config, checkpoint and update ledger record packing and logical/physical forward tokens. Episode packing remains an explicit trainer option. | engineering: Gemma thought-marker prefix equivalence |
| `bc:135-139`; `bc.sh:18`; `agentenv/agentenv/controller/task.py:61-64` | 32768-token maximum with explicit overlength failure, no truncation or fixed 4096 padding. Serving keeps the shared 1024 output cap. | fixed no-truncation protocol |
| `bc.sh:24`; `evol.sh:30,154,205` | Training seed 0 and shared seed-0 serving; deterministic epoch permutations. `BATCH_INVARIANT=1` exports `VLLM_BATCH_INVARIANT=1` in all Stage 3b exploration/evaluation launch paths. Greedy request body unchanged. | fixed seed / serving protocol |
| `evol.sh:89-105,183-211,235-242`; `agentenv/utils/agentevol_filter.py:40-72` | Preserve four iterations 0..3, **each from base**, reward strictly >.99, first successful task per iteration, Fm **prepended** to the entire Dm. No cross-round dedup. D4 is produced but never trained. Empty Fm preserves Dm and runs the next round. | unchanged algorithm; explicit empty-round receipt |
| `agentenv/utils/agentevol_filter.py:32-38`; `evol.sh:183-242` | Stable frozen task order, hash-checked execution and actual reset/replay before eligibility. Unlike upstream's skip-invalid JSON reader, incomplete/unknown exploration stops that stage and remains recorded; retry is a new attempt. Known failed rollouts are rejected by reward and retained. | evidence integrity / engineering |
| `evol.sh:143-160` | Separate dev140 evaluation through existing full-checkpoint launcher. No evaluation-driven collection, early stopping or checkpoint selection. All completed iterates are available. | protocol harmonisation X3 |
| `bc:278`; `evol.sh:91-106,235-242` | Stage 3a standalone checkpoint export; hash-bound stage resumption. Preserve failed attempts, requests, raw outputs, tokens, replay transitions, filter decisions, dataset chain, per-iteration Dm/Fm/exposure and all-attempt disclosure. Resume restarts an interrupted stage from its beginning. | provenance / engineering |

Implementation: `baselines/agentgym/data.py`, `pipeline.py`, shared trainer and
schedule. Data is reconstructed from checked source receipts before training;
only allowlisted visible conversations enter tokenization. Sources in
`example_sources.json` retain parent seed and source episode IDs for every
repeated exposure. No arbitrary event/audit record becomes model input.

No upstream source was modified. Offline tests execute the isolated official
filter on synthetic records and compare its output, test empty and accumulating
rounds, and verify all four training commands restart from base. Real four-GPU
FSDP execution, memory fit and serving remain hardware acceptance checks.
