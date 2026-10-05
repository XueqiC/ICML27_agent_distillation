# IPR port registration — 2026-10-05

Official pin: `93335a3a818b46c1d0e081cd76288cf1cbf7dee4`
([source](https://github.com/WeiminXiong/IPR/tree/93335a3a818b46c1d0e081cd76288cf1cbf7dee4)).
File references below are relative to `third_party/IPR/`. This is an engineering
port under the fixed few-shot black-box protocol. No learning result or faithful
end-to-end experimental reproduction is claimed. Upstream files remain unchanged.

| Official file:line | Replacement/change | Reason class |
| --- | --- | --- |
| `run_pipeline.sh:3,39-61,263-291`; `requirements.txt:1-3` | Configured pinned local Gemma 4 and current installed torch/transformers/vLLM stack; no upstream models, downloaded demonstration zip, verifier or reward model. | X1/X2/X4 |
| `generate_response.py:208,221-247` | Exactly K verified, executed TRAIN seed trajectories and their prefixes; no game without an expert. Manifest IDs replace upstream game-to-integer mapping. K, unique instances and groups remain distinct counts. | X4/X8, unchanged expert-task scope |
| `fastchat/train/train_dpo.py:174-234`; `fastchat/train/train.py` | SFT defaults to recorded `packing=turn`: each seed reply follows its exact Gemma OFF evaluation prefix, including the masked empty thought marker. X5 samples trajectories and accumulates all their turns with assistant-token weights; trajectory-mean CE, update sets and step counts are preserved. Preferences also score each continuation reply at its exact generation prefix; full native history and all shared expert-prefix replies stay masked. Summed continuation log-probs, chosen summed NLL, reference and pair dose are unchanged. Record packing and logical/physical forward tokens. | engineering: Gemma thought-marker prefix equivalence |
| `eval_agent/configs/task/alfworld.json:11`; `generate_response.py:252-276` | Shared harness in every stage: admissible commands, parser-feedback turns, actual termination, 40 total turns including replay. Exploration/evaluation stay greedy, OFF, 1024 output tokens. | X3 (20→40; exploration 512→1024) |
| `monte_carlo_sample_alfworld.py:280-397` (commented-out expert block) | **Un-commenting registered:** reimplement the disabled expert-MC routine as a one-time SFT-model stage. Five real reset/replay continuations after each expert action (prefix lengths 1..T), T=1/top_p=1/max512. A terminal prefix is still reset/replayed five times and its real terminal outcome used, with zero subsequent calls. | requested activation of official disabled code; terminal safety |
| `generate_response.py:244-276`; `monte_carlo_sample_alfworld.py:237-257` | Exploration j=0..T−1 replays j expert actions and continues with current-iterate greedy actions. Agent MC resets and replays those j expert actions **plus the first agent action**, then samples only from fixed SFT. Expert MC is keyed by actions replayed: comparison for branch j uses expert prefix j+1. No discount. | unchanged algorithm; explicit indexing |
| `eval_agent/configs/model/fastchat_explore.json:6-7`; `run_pipeline.sh:90,167-227` | Five MC samples, T=1/top_p=1/max512; top_k=-1 explicitly avoids inheriting Gemma's top-k. Per-rollout seed hashes (sampling seed, seed episode ID, task, kind, iteration, j, sample); request seed hashes (rollout seed, absolute turn). Spawned CPU environment workers share one SFT vLLM server. | seeded sampling / engine adaptation |
| **Defect A:** `monte_carlo_sample_alfworld.py:219,237-239` | **fixed: cut = full prompt length**. ALFWorld and WebShop files both have MD5 **`b5c550d3e82393e9e92086409aeb1352`**. Released `[:2]+[10:]` executes the ICL example's last action and drops the task observation; its ALFWorld prompt has 12 messages. Our structured real episode prefix never includes ICL, so no numeric cut or example action is replayed. | defect fix, not literal bug reproduction |
| **Defect B:** `monte_carlo_sample_alfworld.py:145-162`; `construct_preference_monte_carlo_alfworld.py:232-259` | Retain the five j=0 agent MC samples as requested; record their costs, but global pairs use final reward only. Expert prefix-1 values are likewise computed once and unused in global pairs. | disclosed upstream unused compute |
| **Defect C:** `generate_response.py:15`; `monte_carlo_sample_alfworld.py:15` | No unconditional WebShop import; use only the existing ALFWorld harness. Unused embedded credential assignments are neither imported nor copied. | engineering / credential hygiene |
| **Defect D:** `run_pipeline.sh:25-27,138-140,216-218` | Never execute upstream cleanup or `rm -r`. Fresh attempt directories preserve all prior results, interruptions and rejected samples. | engineering |
| `construct_preference_monte_carlo_alfworld.py:232-259,307-350` | Pure port with both global/local flags and thresholds .01, no step_action. Global pairs only at j=0; entire-continuation pairs at j≥1. Preserve strict thresholds, agent-rule-first ordering, and ALFWorld shorter-continuation flag, including its ability to favor a shorter failed agent continuation when its MC value is higher. | unchanged pair rule |
| `construct_preference_monte_carlo_alfworld.py:158-201` | Mean of exactly five known real outcomes. Incomplete or unknown sampling stops the stage and retains evidence; never fall back to immediate reward for missing samples. Pair loaders reconstruct data from checked source receipts and verify raw call/episode hashes. | evidence integrity |
| `run_pipeline.sh:33-61,249-291` | SFT 2e-5/cosine/.03/3 epochs; DPO 3e-6/constant_with_warmup/.1/3 epochs, beta .1, wd0. Both original global batches 48 shrink by X5: min(48,max(1,floor(N/8))), N=K or this iteration's P. Training seed 0, no accumulation of pairs. | X5 plus fixed training seed |
| `fastchat/train/dpo_trainer.py:807-873,940-946,1002-1007`; `run_pipeline.sh:263-291` | Shared summed-logp scorer + exact `-logsigmoid(.1*delta) - log pi(chosen)`. NLL weight 1, no token normalization. Current iterate initialization; separate frozen SFT reference for all six iterations, also used for every MC stage. Tests execute isolated official `dpo_with_sft_loss`. | unchanged objective/reference |
| `run_pipeline.sh:56,284`; upstream Llama/flash-attn training paths | Shared full text-parameter FSDP2 on four B200s, fp32 masters/AdamW state, bf16 compute, fp32 reductions, SDPA, Gemma4UnifiedTextDecoderLayer wraps, gradient checkpointing/clip1. Frozen multimodal towers exported unchanged. Microbatch one turn/pair, exact partial trajectory/pair batches and zero-weight collective padding. Preference pairs score at most one turn per side per forward and sum scores by side. Zero-weight turn padding aligns forward/backward rounds across sides and ranks. | X6/X7; exact X5 realization |
| `run_pipeline.sh:58,288-290`; `dpo_trainer.py:336-338` | 32768 whole-conversation limit; explicit overlength failure, no truncation. Supervise all assistant continuation spans, including both sides' native turn ends. | fixed no-truncation protocol |
| `run_pipeline.sh:91,239-291` | Six iterations even when P=0: report P and kind counts before training; publish a hash-bound carry-forward receipt with zero exposure and unchanged policy. No empty-data trainer invocation. | fixed user empty-iteration rule |
| `run_pipeline.sh:77` and per-iteration evaluation | Separate dev140-only evaluation through existing full-checkpoint launcher; no test/sealed generation, evaluation writes or checkpoint selection. BATCH_INVARIANT=1 / VLLM_BATCH_INVARIANT=1 for exploration, MC and evaluation. | X3 / fixed serving protocol |
| `run_pipeline.sh:39-291` | Hash-bound stage completions, retained attempts, dependency/config/source/model/data provenance, raw outputs, negative outcomes, prefix transitions and token/exposure accounting. Completed stages resume; partial stages restart in a new attempt. Worker count is outside identity/seed keys. | engineering |

The small replay controller adapts the repository's `analysis/mechanism/branch.py`
checks into a separate TRAIN-only path; it does not import diagnostic policies or
change their masks, seeds or eligibility. New IPR code does not copy the upstream
repository wholesale. ETO's shared sigmoid implementation retains its existing
Apache-2.0/DPO/HuggingFace notice; the additional NLL term is an independently
written direct implementation of the cited objective.

SAG reproducibility is also fixed in Stage 3c: its eight-sample request now carries
`seed = int(sha256([sampling_seed, task_id, turn_index])[:8], 16)`, with default
sampling seed 0. The suite supplies task identity separately from visible prompt
fields. Seeds are retained in the request blob, call ledger and step metadata.
Default greedy requests are byte-identical. AgentEvol uses greedy requests;
mechanism sampled paths already seed requests and are unchanged.

Real distributed training, B200 memory/throughput and seeded live vLLM behavior
still require a separately authorized hardware run. Offline tests are engineering
evidence only.
Turn prefixes repeat history compute. Recorded physical forward tokens include
sequence padding and zero-weight turn/rank padding for both policy and reference;
long-pair memory fit has not been exercised here.
