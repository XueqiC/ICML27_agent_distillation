# ETO port registration — 2026-10-04

Official pin: `a2fc5da38f8d00cfaf3f9b6370d586eebaf72904` ([source](https://github.com/Yifan-Song793/ETO/tree/a2fc5da38f8d00cfaf3f9b6370d586eebaf72904)).
Paths in the first column are relative to `third_party/ETO/`. Fixed user decisions
X1–X7 take precedence over options in the port specification. This is an
algorithm-faithful, protocol-harmonised port with offline engineering validation;
no ALFWorld learning or baseline reproduction result is claimed.

| Official file:line | Our replacement/change | Reason class |
| --- | --- | --- |
| `setup.sh:2`; `requirements.txt:1-3,7,20`; `fastchat/train/train.py:294-302` | Port algorithm into installed torch cu128 / transformers 5.14.1 / peft 0.20 / vLLM 0.27.1; no old Trainer/TRL dependency. | engineering (X1) |
| `run_eto.sh:1,20-21`; `fastchat/train/train.py:294-302` | Configured, pinned `google/gemma-4-12B-it`; same `train.modeling.load_student` class, no model substitution or remote loading. | engineering (X1/X2) |
| `fastchat/train/train.py:123-189`; `fastchat/train/train_dpo.py:117-189,223-234` | Reuse `train.dataset.episode_examples` exactly: Gemma OFF template, all assistant replies + native 106, masked system/user/observation. No synthetic “OK” turn. Boundary errors raise instead of silently zeroing the labels. | engineering (X2) |
| `fastchat/train/train.py:123-189`; `fastchat/train/train_dpo.py:223-234` | SFT defaults to recorded `packing=turn`, with the exact OFF generation prefix and masked empty thought marker before every supervised reply. Full native history stays masked. X5 still samples trajectories; accumulate every selected trajectory's turns with assistant-token weights to retain episode-mean CE, identical update sets/step counts and exposure. DPO likewise scores each continuation turn at its exact generation prefix and sums all assistant token log-probs per side; sigmoid objective/reference unchanged. Record packing and logical/physical forward tokens in config/checkpoint/update ledgers. | engineering: Gemma thought-marker prefix equivalence |
| `eval_agent/envs/alfworld_env.py:15-18,34-42,50-62,71,78-96`; `eval_agent/configs/task/alfworld.json:11` | Shared `sed.react` prompt, visible-field whitelist, admissible commands every turn, last-Action parser, parser-feedback policy turns and unchanged 40-turn cap in seed SFT, exploration and evaluation. No official one-shot example or rollout-only context. | protocol harmonisation X3 |
| `eval_agent/agents/fastchat_agent.py:32-35,61-68`; `eval_agent/configs/model/fastchat.json:6-7` | Greedy unchanged; 1,024 student tokens instead of 512, thinking OFF; shared serving stop/template behavior instead of ETO stop strings. | protocol harmonisation X3 |
| `run_eto.sh:13,113-116,138`; `eval_agent/tasks/alfworld.py:51-53`; `construct_preference.py:118-122` | Exactly K checked teacher seed receipts and exactly their TRAIN task IDs, including replacement tasks and nested K=5/10 subsets. No new teacher, external demonstrations or reward model. K counts trusted trajectories; student rollouts add zero trusted seeds. | protocol harmonisation X3 (fixed expert-source X4) |
| `construct_preference.py:119,135` | Frozen manifest trial-directory task IDs, not basename parsing or output-folder prefix matching. Prompt equality checked before pairing. | engineering (X8) |
| `run_eto.sh:14-16,141-143` | Global `b'=min(b_orig,max(1,floor(N/8)))`; N=K SFT episodes or P current-iteration pairs. Record original batch, N, b', final partial batch, steps and exposures. | few-shot dose X5 |
| `run_eto.sh:15-16,142-143`; `fastchat/train/dpo_trainer.py:842-883` | Microbatch 1 turn / preference pair. A pair scores at most one chosen and one rejected turn per forward, summing scores by side. Zero-weight turn padding aligns rounds across sides and ranks with unequal continuation lengths; every rank executes the same forward/backward graph. Keep every final partial trajectory/pair batch; no DistributedSampler replication or dropping. FSDP collective padding has zero loss and is separately counted. Scale by ranks/actual logical batch, including SFT turn token weights, before rank averaging. | engineering (exact X5 realization) |
| `run_eto.sh:37-38,173-174`; `fastchat/train/train.py:83-92` | FSDP2 `fully_shard` per `Gemma4UnifiedTextDecoderLayer` plus root. Policy fp32 master/sharded optimizer state, bf16 compute, fp32 reductions, AdamW 0.9/0.999/1e-8, gradient clip 1.0 and checkpointing. Full text parameters train; unused multimodal components remain unchanged. | engineering (X6) |
| `fastchat/train/train.py:302`; `fastchat/train/train_dpo.py:286,294` | SDPA instead of flash-attn 2; exact Gemma4Unified wrap class verified in installed transformers 5.14.1 at `modeling_gemma4_unified.py:496`. | engineering (X7) |
| `run_eto.sh:40,176-178`; `fastchat/train/train.py:114-120`; `fastchat/train/train_dpo.py:225,231` | Max length 32768, whole episodes, no right truncation and no fixed 4096 padding. Overlong examples fail explicitly. ALFWorld's existing no-rollout-truncation rule is preserved (`construct_preference.py:155-156`; commit `1a4c84c`). | no truncation |
| `fastchat/train/dpo_trainer.py:737-839,860-883,934-943` | Two loss methods copied verbatim. Materialize logits only at supervised causal predecessors; call unchanged scorer with already-aligned labels, no second shift, sum all assistant log-probs. Small log-softmax chunks reduce temporary memory. Dense-vs-selected tests verify equality. Reward chosen/rejected/accuracy/margin and logps logged; full-vocabulary mean-logit diagnostic omitted. | engineering |
| `run_eto.sh:154-156`; `fastchat/train/dpo_trainer.py:904-923` | Separate frozen, sharded, eval-mode reference reloaded from the exact iteration-start checkpoint; no precomputed reference and no SFT term. | engineering (explicit invariant) |
| `fastchat/train/dpo_trainer.py:904-923` | Load and shard the frozen reference directly in bf16, retaining eval/no-grad, identical decoder/root wraps and `MixedPrecisionPolicy(param_dtype=bfloat16, reduce_dtype=float32)`. This halves reference parameter storage on host/GPU; policy fp32 masters and AdamW state are unchanged. The former fp32 reference already cast parameters to bf16 on every forward, so reference log-probs are numerically identical. CPU tiny-Gemma tests require bitwise equality using the same parameter-cast semantics and native loader buffers. Retain only detached summed reference log-probs before the policy forward; tests check activation/logit release. Record `baseline_training.reference_storage_dtype=bfloat16` in provenance. | engineering (numerically identical) |
| `run_eto.sh:20-42,154-180` | Explicit training/sampler seed 0, epoch permutation seeded by `seed+epoch` instead of relying on HF's implicit seed 42/sampler implementation. Exhaustive, no duplicate logical exposures. All published stage checkpoints are final. | engineering (fixed user seed) |
| `fastchat/train/train.py:83-92,334-342`; `fastchat/train/train_dpo.py:338-345` | Stream tensor gathers into standalone bf16 HF safetensors checkpoint; original base non-text tensor values and tokenizer/processor assets copied unchanged. Save artifact hashes, source binding, timing and peak memory per rank. | engineering |
| `run_eto.sh:62,195`; `eval_agent/tasks/alfworld.py:57-59` | Evaluation removed from training loop; separate frozen dev140 step through `scripts/eval_adapters.sh` full-checkpoint mode. No sealed evaluation or dev-driven generation/selection. | protocol harmonisation X3 |
| `eval_agent/agents/fastchat_agent.py:70-96`; `run_eto.sh:113-116` | Existing student client has no HTTP retry. Retain all failed/raw attempts, token accounting, unknowns, event logs; real reset/replay required before pair eligibility. Replays cost additional episodes/steps. Unknown execution is never mapped to a failed preference. | engineering / protocol harmonisation X3 |
| `run_eto.sh:118-123` | **Exit-status bug not reproduced.** Official `wait` status is overwritten by `rm` and then `echo` before the check. Checked subprocess exits stop the stage; a failing exploration cannot proceed to pairs. | engineering |
| `run_eto.sh:76,138,154-180` | Explicit `P_i=0` stop with “no pairs at iteration i”, instead of invoking DPO on an empty dataset. Successful rollouts remain skipped ties. | engineering (fixed user stop rule) |
| `run_eto.sh:20-210` | Hash-bound stage completions, preserved incomplete attempts and process cleanup on signals. Completed stages resume without rerunning. An interrupted stage is retried from its start in a new attempt directory; all additional interactions/exposures remain disclosed and do not increase K or unique tasks. | engineering |

Unchanged optimization: SFT 2e-5 / cosine / warmup .03 / wd 0 / 3 epochs;
DPO iter 1 beta .1 / lr 1e-6, iter 2–3 beta .5 / lr 5e-7, each
constant-with-warmup / warmup .1 / wd 0 / 3 epochs. Warmup steps use ceil;
optimizer update 1 uses scheduler step 0 (LR zero when warmup > 0). SFT is a
mean over supervised tokens **per episode**, then mean over the global batch.
DPO uses sigmoid on **summed** assistant log-probs, smoothing 0, no SFT term.
Independent turn prefixes repeat history compute. DPO physical forward-token
counts include sequence padding and zero-weight turn/rank padding for policy
and reference. Long-pair memory fit remains a hardware acceptance check.

This storage fix follows the reported hpg K=20/s0 iteration-1 failure: 11
preference pairs on 2 × B200 180 GB, CUDA OOM in the policy Gemma MLP with
170 GB allocated. CPU checks establish the storage/cast equivalence and tensor
lifetime behavior; they do not establish that the failed GPU run now fits.

Full-repo or paper performance equivalence is not asserted. GPU FSDP, B200
memory fit, long-sequence throughput and live full-checkpoint serving remain
acceptance checks for a separately authorized job.
