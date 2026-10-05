# Baseline porting specs: ETO, IPR, AgentGym (BC/AgentEvol), WKM, LEAP, Kang AD (2026-10-04)

Scope: a porting spec for each official agent-distillation repo, written for our setting. Our setting is ALFWorld via our harness:
observation plus admissible commands, 40 steps, ReAct `Thought: ...\nAction: <cmd>`. The student is gemma-4-12B-it with
thinking OFF; the teacher is a black-box OpenAI model; there are K = 5/10/20 trusted teacher trajectories; evaluation is
on valid_seen (140 games).

How this was produced:
- Everything comes from reading code only. Nothing was installed, trained or called.
- Every claim cites `file:line` inside the repo named in its section.
- "not in code" means the repo does not contain it.
- **[ext]** marks a claim that rests on library knowledge rather than this repo; re-check those before relying on them.
- Claims marked (re-checked) were read a second time directly from source while writing this report.

## Repos and commits

| Method | Repo (local clone under `third_party/`) | Commit | Commit date |
|---|---|---|---|
| ETO (ACL'24) | `ETO` ← Yifan-Song793/ETO | a2fc5da38f8d00cfaf3f9b6370d586eebaf72904 | 2024-10-30 |
| IPR (EMNLP'24) | `IPR` ← WeiminXiong/IPR | 93335a3a818b46c1d0e081cd76288cf1cbf7dee4 | 2024-10-18 |
| AgentGym BC / AgentEvol | `AgentGym` ← WooooDyy/AgentGym | 3ef9235d23e68e7c2920c5422ad957dc8ced5c6c | 2026-05-30 |
| WKM (NeurIPS'24) | `WKM` ← zjunlp/WKM | f1fe28fc189c59801a98cabda01188bc9b64eae6 | 2024-12-17 |
| LEAP (ICLR'25) | `leap_llm` ← sanjibanc/leap_llm | a3376d89a2eeee88f03555a4174a0bdf7faf8be6 | 2024-12-19 |
| Kang AD (NeurIPS'25) | `agent-distillation` ← Nardien/agent-distillation | 8884b80ea3d22e53a2e1e4b9fd324600d13e0430 | 2025-10-22 |

### Facts about our own stack
These were checked locally and are used throughout the report.

- **Student class.** The checkpoint at revision 707f0a3 has `model_type = gemma4_unified` and architecture
  `Gemma4UnifiedForConditionalGeneration`, with 48 text layers.
  - Our loader uses that class (`train/modeling.py:25`).
  - The FSDP wrap class is therefore **`Gemma4UnifiedTextDecoderLayer`** (transformers 5.14.1
    `models/gemma4_unified/modeling_gemma4_unified.py:496`), **not** `Gemma4TextDecoderLayer`.
- **Special tokens.** `<|turn>` = 105 (turn start), `<turn|>` = 106 (turn end), `<bos>` = 2, `<eos>` = 1.
- **Generation config.** `do_sample: true, top_k: 64, top_p: 0.95, eos [1,106,50]`. Every port must set its sampling
  parameters explicitly, or it inherits these.
- **PEFT targets.** PEFT 0.20 has a default LoRA target entry for `gemma4` (`peft/utils/constants.py:100`) but **none for
  `gemma4_unified`**. Any repo that relies on PEFT's default `target_modules` (LEAP) must pass the targets explicitly.
  Restrict them to `.language_model.` modules, as our `attach_lora` does (`train/modeling.py:30-40`).
- **K=20 seed statistics (s0).** From `logs/train_seedsft_k20_s0_rai.log`, preparation line:
  - 20 trajectories, 382 assistant turns (ΣT = 382, about 19 steps per episode);
  - 10,407 supervised tokens and about 101.7k forward tokens per pass, i.e. about 5.1k tokens per episode, about 270 per step.
  - s2 has ΣT = 319.
- **Measured inference rate.** Base dev40 on one hpg B200 with 8 concurrent episodes: 40 episodes (1,147 steps) in 3 min
  wall time (`reports/pilot_thinking_0410_ZH.md`). That is about 800 full episodes per B200-hour, used below as a
  conservative rate.

---

## 0. Cross-cutting porting decisions (apply to every method)

| # | Issue | Evidence | Options (★ = closest to the original algorithm) |
|---|---|---|---|
| X1 | **Software stack.** All 2024 repos pin torch 2.1/cu118, transformers 4.37.2, flash-attn 2.4.2, trl 0.7.7 (ETO, IPR, WKM); LEAP is unpinned; Kang pins vllm 0.9.1 / trl 0.19.1. None of these runs Gemma 4 on sm_100. | ETO `setup.sh:2`, `requirements.txt:1-3,7,20`; IPR `requirements.txt:1-3`, `setup.sh:20`; WKM `requirements.txt:1-3`; Kang `pyproject.toml:25-40` | ★ Port the **algorithm** (pair rules, losses, data flow) into our stack (torch cu128, transformers 5.14.1, peft 0.20, vLLM 0.27.1) and our harness. Patching their trainers onto new APIs adds glue but no fidelity. |
| X2 | **Chat template and label mask.** FastChat Llama-2 separators (ETO/IPR/WKM) or raw `[INST]` strings with eos id 2 (AgentGym). Id 2 is `<bos>` in Gemma 4. | ETO `fastchat/train/train.py:123-189` (re-checked); AgentGym `agentenv/agentenv/trainer/bc_trainer.py:118-131` (re-checked) | ★ Render the Gemma 4 template (thinking OFF). Supervise exactly the assistant turns each repo supervises, each ending in id 106; mask user/observation turns. Our `train/dataset.py` already does this. |
| X3 | **Prompt / visible fields / step cap.** Each repo uses its own prompt. ETO/IPR/WKM show no admissible commands. AgentGym shows them only at reset. LEAP and our harness show them every step. Caps: ETO 40, IPR 20, AgentGym 30, LEAP 20, WKM 40. | per-method sections | ★ (algorithm-faithful, protocol-harmonised) Use our harness format in **every** stage (SFT data, exploration, MC rollouts, eval) and disclose it as a protocol change. Alternative: repo-literal prompts for training/exploration with our harness for eval; this creates a train/eval mismatch, since we evaluate everything with our harness. |
| X4 | **Expert data = our K verified teacher trajectories.** Every repo assumes thousands of expert trajectories (ETO/IPR: 3,321-game train split; AgentGym: about 1,000; LEAP: 3,553 handcoded-expert logs). | ETO `eval_agent/tasks/alfworld.py:51-53`; AgentGym `examples/behavioral_cloning/train_behavioral_clone_alfworld.sh:1,11`; LEAP `scripts/dataproc/collect_logs_alfworld.py:31-52` | Use the K receipts (env-verified wins). They satisfy the "golden reward = 1" assumption that ETO/IPR hard-code (ETO `construct_preference.py:121-122`). |
| X5 | **Few-shot dose.** Global batches of 32–64 with 5–20 trajectories give ≤1 optimizer step per epoch. With AgentGym's official config and K=20, the step count is 0 (§3). | per-method sections | (a) Literal batch: faithful hyperparameters, but the update is a near no-op. ★ (b) Keep lr / epochs / scheduler / β, and shrink the global batch by one pre-registered rule applied to every baseline. Example rule: `b' = min(b_orig, max(1, ⌊N_examples/8⌋))`, giving ≥8 steps per epoch. Report it as a dose adaptation. (c) Match supervised-token exposure to Seed-SFT; this changes the epochs, so it is a larger deviation. |
| X6 | **Full FT vs LoRA.** ETO, IPR and AgentGym are full FT. LEAP, WKM and Kang are LoRA, each with its own rank and targets. | ETO `run_eto.sh:37-38`; IPR `run_pipeline.sh:56`; AgentGym `bc_trainer.py:295-316`; LEAP `bash/train-sft-alfworld-iterx.sh:57-62`; WKM `README.md:113-114`; Kang `exps_research/finetune_sft.py:137-146` | ★ Use each repo's own adaptation type. Full FT of 12B needs about 16 B/param ≈ 190 GB of states, so FSDP over ≥2 (comfortably 4) B200s wrapping `Gemma4UnifiedTextDecoderLayer`, with fp32 master weights. Alternative: our bf16 LoRA for all, a disclosed deviation; full-FT learning rates (1e-6 to 2e-5) then need re-tuning, which is not in any repo. |
| X7 | **Attention kernel.** `flash_attention_2` is hard-coded; flash-attn 2.4.2 predates Blackwell. | ETO `fastchat/train/train.py:302`, `train_dpo.py:286,294`; IPR `fastchat/train/train.py:448` | Use `sdpa`; algorithmically neutral. |
| X8 | **Game IDs.** ETO/IPR/WKM key games by the trial-directory path. AgentGym uses an integer index into its own mapping files. LEAP uses `env_idx` (an os.walk + seed-1234 shuffle, machine-specific) plus `gamefile`. | ETO `construct_preference.py:119,135`; AgentGym `agentenv-alfworld/agentenv_alfworld/env_wrapper.py:30-72,117`; LEAP `scripts/eval/eval_alfworld.py:160-161` | Key every record by our manifest's game path (trial directory). |
| X9 | **Chinese-origin models.** Kang's default teacher and students are Qwen2.5, and its released HF trajectories and LoRA are Qwen-generated. Nothing else in any default path. Unused leftovers: FastChat adapters (ChatGLM/Yi/DeepSeek/Qwen); AgentGym `examples/basic` (Qwen2-7B, deepseek-reasoner). | Kang `scripts/inference/run_agent_teacher_train.sh:4`; ETO `fastchat/model/model_adapter.py:769,1954,1976`; AgentGym `agentenv/examples/basic/base_eval.sh:8` | Kang: reuse code only, never its data or weights. Never pick a BGE/GTE embedder for WKM. |

---

## 1. ETO (Exploration-based Trajectory Optimization)

Paths are relative to `third_party/ETO/`.

### 1.1 Pipeline
1. **SFT.** Behaviour cloning on `data/${task}_sft.json` from Llama-2-7b-chat-hf (`run_eto.sh:1,13,20-42`, re-checked).
2. **Evaluate** the SFT model on `--split test`, which is valid_unseen with 134 games (`run_eto.sh:62`; `eval_agent/tasks/alfworld.py:57-59`).
3. **Three ETO iterations** (`run_eto.sh:76`, re-checked). Each iteration:
   - **Explore.** The current policy runs once on every `train` game, 3,321 games in 4 shards (`run_eto.sh:113-116`; `tasks/alfworld.py:51-53`, re-checked). These are the **same tasks that have expert trajectories**: the golden file is the SFT file (`run_eto.sh:138`), and pairs exist only for keys in both (`construct_preference.py:146`).
   - **Build pairs** (ALFWorld branch, `construct_preference.py:114-174`, re-checked).
     - Golden reward is hard-coded to 1.0 (`:121-122`). Rollout reward = 1.0 if `meta.reward` is truthy, else 0 (`:136`).
     - **Failed rollout → (chosen = expert, rejected = full rollout)** (`:150-157`). A successful rollout is a tie and is skipped (`:166-168`). The "win" branch can never fire for ALFWorld.
     - **No truncation for ALFWorld.** Commit 1a4c84c (2024-05-21) removed `pred_trajs[key][:len(golden)+5*2]` from the ALFWorld branch only; the generic WebShop/SciWorld builder keeps it (`construct_preference.py:87-92`; `git show 1a4c84c`, re-checked).
     - Prompt = instruction + "OK" + task (`:154`). The in-context example is stripped through offset 12 (`:39-44,137`).
   - **DPO.** Policy and reference both start from the current model (`run_eto.sh:154-156`). The next iteration starts from the DPO model (`:208`).
4. There is no checkpoint selection: the final model is the iteration-3 model.

### 1.2 Hyperparameters

| Item | SFT | DPO | Source |
|---|---|---|---|
| Adaptation | full FT, FSDP `full_shard auto_wrap`, `LlamaDecoderLayer` | same, no `peft_config` | `run_eto.sh:37-38,173-174`; `fastchat/train/train_dpo.py:315-326` |
| GPUs × micro × accumulation = global | 4 × 4 × 4 = 64 | 4 × 2 × 4 = 32 | `run_eto.sh:6,14-16,141-143` |
| lr / schedule / warmup / wd | 2e-5 / cosine / 0.03 / 0 | iter 1: 1e-6, β 0.1; iters 2–3: 5e-7, β 0.5 / constant_with_warmup / 0.1 / 0 | `run_eto.sh:32-35,144-150,168-171` |
| Epochs | 3 | 3 per iteration | `run_eto.sh:25,160` |
| Max length | 4096 (every example padded to 4096, right-truncated) | 4096. `max_prompt_length` 512 and `max_target_length` 3072 are passed but **have no effect**, because the `tokenize_row` map is commented out | `run_eto.sh:40,176-178`; `train.py:114-120`; `fastchat/train/dpo_trainer.py:336-338` |
| Loss | CE on assistant turns | sigmoid DPO, label smoothing 0, **summed** log-probs, **no SFT term** | `dpo_trainer.py:131-133,759-773,836-839,943` |
| Reference model | — | separate frozen copy of the iteration-start policy, forward pass every step | `train_dpo.py:289-295`; `dpo_trainer.py:904-923` |
| Masking | Llama-2 split on `" </s><s>"` / `" [/INST] "`. All assistant turns are supervised, including "OK". Any length mismatch silently masks the whole example | DPO also masks the prompt tokens | `train.py:123-189` (re-checked); `train_dpo.py:117-189,223-234` |
| Exploration decoding | **greedy**: `fastchat.json` T=0.0, `max_new_tokens` 512; `top_p` defaults to 0, so the worker takes the greedy branch. `fastchat_explore.json` (T=0.7) exists but is **referenced nowhere** | — | `run_eto.sh:114`; `eval_agent/configs/model/fastchat.json:6-7`; `eval_agent/agents/fastchat_agent.py:32-35`; `fastchat/serve/inference.py:185` (all re-checked) |
| Stop strings | `\nObservation:`, `\nTask:`, `\n---`, EOS | — | `eval_agent/agents/base.py:14`; `fastchat_agent.py:61-68` |
| Seed | not passed (HF default) | not passed | `run_eto.sh:20-42` |

Script bug: the exit-status check after `wait` tests the preceding `rm`, so a failed exploration goes unnoticed (`run_eto.sh:118-123`).

### 1.3 Data requirements
- **Expert SFT file:** `{id, game_file, conversations}`, where `conversations[0:2]` = instruction + "OK" (`construct_preference.py:118-120`). It is downloaded and not in the repo (`setup.sh:58-61`).
- **Rollouts:** one per expert task per iteration.
- **No other models:** no teacher logits, no online teacher, no reward or critic model. The reward is `info['won']` (`eval_agent/envs/alfworld_env.py:41`).
- **Chinese-origin model:** none (`run_eto.sh:1`).

### 1.4 ALFWorld specifics
- **Instruction:** `eval_agent/prompt/instructions/alfworld_inst.txt:1-20`, with the action list at `:4-14`. The "task → take" typo was fixed only there in a2fc5da; the training-side copy still reads "task {obj} from {recep}" (`construct_preference.py:30`).
- **In-context example:** one example, toiletpaper / 5 actions (`eval_agent/prompt/icl_examples/alfworld_icl.json`; used 1-shot at `alfworld_env.py:96`).
  - Rollouts are 1-shot, while SFT/DPO contexts are 0-shot. That asymmetry is in the code.
- **Observation:** "Observation: ..." with "You arrive at loc N." stripped. **Admissible commands are never shown** (`alfworld_env.py:15-18,40-42,71`).
- **Action:** regex `Action:\s?(.*)`. A missing "Action:" yields "Error Input" and still counts as a step (`alfworld_env.py:34-35,50-62`).
- **Max steps 40, success = `won`** (`eval_agent/configs/task/alfworld.json:11`; `alfworld_env.py:78-88`).
- **Splits:** train 3,321; dev = valid_seen 140; test = valid_unseen 134 (`tasks/alfworld.py:51-59`, re-checked).

### 1.5 What breaks, and the minimal faithful change

| Problem | Fix (★ closest) |
|---|---|
| FastChat template, Llama-2 mask (§1.2), `LlamaDecoderLayer`, `flash_attention_2`, HF arg renames (`evaluation_strategy` → `eval_strategy`, `tokenizer=` → `processing_class`) | X1/X2/X7. ★ Reimplement about 40 lines in our trainer: summed-logp `get_batch_logps`, sigmoid loss, frozen reference under `no_grad` (`dpo_trainer.py:737-839,904-923`). Current trl DPOTrainer **[ext]** does not natively mask observations inside a multi-turn completion. |
| Full FT reference = iteration-start policy | ★ Full FT with FSDP (X6). LoRA alternative: merge the previous adapter into the weights before each iteration, train a fresh adapter, use adapter-disabled as reference (the trainer already supports this, `dpo_trainer.py:190-191,908-912`). |
| **Pair supply.** P_i = number of the K tasks on which π_i's greedy rollout fails, so P ≤ K. After SFT on exactly those K tasks the student may memorise them, giving P ≈ 0. | ★ Literal: greedy, one rollout per K task; count P before DPO and report it. Deviation options: T=0.7 sampling (the unused config, which would also need `top_p=1`), or exploring new train tasks. The second breaks ETO's "same tasks as expert" rule, because pairs need a golden trajectory. |
| Dose: SFT 64 / DPO 32 global batch. With K=20 that is 1 step per epoch, and warmup makes the first step lr 0, so about 2 effective updates per stage. | X5(b). |
| Golden success assumed | Our receipts are env-verified (X4). |

### 1.6 Compute, K=20 (assumptions in §7)
- **Episodes:** 3 × 20 exploration + 4 × 140 eval (SFT + 3 iterations) = **620**.
- **Training:**
  - SFT: about 0.3M tokens.
  - DPO: ≤ 20 pairs × (5.1k + ≤ 11k) × 1.33 × 3 epochs ≈ 1.3M token-equivalents per iteration.
- **Total:** ≈ **2 B200-h** with LoRA, ≈ 3–4 with full FT on 4 GPUs. Evaluation dominates.

---

## 2. IPR (Iterative step-level Process Refinement)

Paths are relative to `third_party/IPR/`. IPR is built on ETO's FastChat fork.

### 2.1 Pipeline
1. **SFT** on expert data, from Llama-2-7b-hf (base, not chat), full FT (`run_pipeline.sh:3,32,39-61`).
2. **Evaluate** on test, i.e. valid_unseen (`run_pipeline.sh:77`).
3. **Six iterations** (`run_pipeline.sh:91`). Each iteration:
   - **a. Step-wise exploration** on the expert tasks only. It skips games not in the SFT data (`generate_response.py:208,221-242`). For an expert trajectory of T actions there are T explorations, j = 0…T−1 (`generate_response.py:244-247`, re-checked). Each one:
     - resets the game, sets the 1-shot prompt plus the task, and replays the expert's first j turns;
     - takes one greedy agent action, then continues greedily to the end (`:252-276`);
     - is done by the **current iterate** (`run_pipeline.sh:102-108`).
   - **b. MC value of the agent's step:** 5 rollouts per explored item, by the **SFT model**, with `fastchat_explore` (T=1, top_p 1) (`run_pipeline.sh:90,167-227`). The value is the **mean final success**, with no discount (`construct_preference_monte_carlo_alfworld.py:158-185`).
   - **c. Expert step values** `monte_carlo_step_reward` are read from the shipped `data/sft_data_alfworld.json` (`construct_preference_monte_carlo_alfworld.py:212,278,314`). The script that produced them is not in active code; there is only a commented-out block (`monte_carlo_sample_alfworld.py:280-397`).
   - **d. Pairs** (`run_pipeline.sh:245`: `--global_traj --local_traj --traj_threshold 0.01 --step_threshold 0.01`; no `--step_action`, re-checked). Two kinds go into one file under one loss:
     - **Trajectory-level, j = 0 only.** A pair is made when |agent − expert| > 0.01. Prompt = instruction + "OK" + task; chosen/rejected are the two full trajectories (`construct_preference_monte_carlo_alfworld.py:232-259`).
     - **"Step-level", j ≥ 1:** the "entire" branch. Prompt = expert prefix up to step j; chosen/rejected = **the whole continuation from step j** (agent action plus its greedy rollout, vs. the expert remainder), not a single action (`:307-350`, re-checked).
       - Expert is chosen if expert final > agent final + 0.01 **and** expert MC > agent MC + 0.01.
       - Agent is chosen if (agent final > expert final + 0.01 **or** `flag`) and agent MC > expert MC + 0.01. `flag` (ALFWorld only) = the agent's continuation is shorter than the expert's remainder (`:324-345`).
       - The single-action branch (`:272-306`) is never used.
   - **e. DPO + NLL.** Init = current iterate, ref = **SFT model** (`run_pipeline.sh:263-291`). Only this iteration's pairs are used; nothing accumulates (`:239`).

### 2.2 Hyperparameters

| Item | Value | Source |
|---|---|---|
| SFT | full FT, FSDP, 4 × 4 × 3 = global 48, lr 2e-5 cosine, warmup 0.03, 3 epochs, max length 4096 | `run_pipeline.sh:9,33-61` |
| DPO | full FT, 8 × 2 × 3 = global 48, lr 3e-6 constant_with_warmup, warmup 0.1, 3 epochs, β 0.1, max length 4096 (prompt/target limits unused, as in ETO) | `run_pipeline.sh:249-291`; `fastchat/train/dpo_trainer.py:336-338` |
| Loss | `−logσ(β·Δ) − logπ_θ(chosen)`: the **NLL on chosen** has weight 1 and summed log-probs | `dpo_trainer.py:841-844,940-946` (re-checked) |
| Exploration / MC decoding | T=0 (greedy) / T=1, top_p 1; `max_new_tokens` 512; ETO stop strings | `eval_agent/configs/model/fastchat.json:6-7`; `fastchat_explore.json:6-7`; `run_pipeline.sh:220` |
| MC samples | 5 per (task, j) | `run_pipeline.sh:205` |
| **Max steps** | **20**, counted from game start, so replayed expert steps count against it | `eval_agent/configs/task/alfworld.json:11` (re-checked); `alfworld_env.py:77-82` |
| Masking | all assistant turns after the prompt; a tokenisation mismatch drops the sample | `fastchat/train/train_dpo.py:174-234` |

### 2.3 Data requirements
- **SFT file** as in ETO, plus a golden file keyed `"{id}_{j}"` with `gpt_reward`, `gpt_conversations`, `gpt_step_conversations` and `monte_carlo_step_reward` (`construct_preference_monte_carlo_alfworld.py:212,235-236,277-279`).
- **`data/game_file_to_id.json`** maps games to golden ids (`:213`). `alfworld_game_files.json` lists 3,554 train + 134 valid_unseen paths for MC resets (`monte_carlo_sample_alfworld.py:139-143,200`). All data come from a zip (`setup.sh:52-53`).
- **We must generate the expert MC values ourselves:** 5 SFT-student rollouts after each expert action.
- **No other models.** No teacher logits, no second model, no reward model. `OpenAILMAgent` is unused.
- A hard-coded OpenAI key string sits unused at `generate_response.py:3` and `monte_carlo_sample_*.py:3`. Do not copy it.
- **Chinese-origin model:** none in the default path.

### 2.4 ALFWorld specifics
- **Prompt:** as ETO, but the instruction keeps the "task {obj}" typo (`eval_agent/prompt/instructions/alfworld_inst.txt:6`). 1-shot for sampling and eval, 0-shot for training (`construct_preference_monte_carlo_alfworld.py:28-37`). No admissible commands.
- **Vendored textworld 1.6.1:** it exists only to add `reset_states(game_files)` (`textworld/textworld/gym/envs/textworld_batch.py:125-136`, called at `eval_agent/envs/alfworld_env.py:92-94`).
  - **There is no state save/restore.** Mid-trajectory rollouts are reset + prefix replay, the same as our harness. That is deterministic with `domain_randomization: False`.

### 2.5 Code defects that change a faithful port
- **A. The ALFWorld MC sampler is a copy of the WebShop sampler.**
  - The two files have the same md5, b5c550d3… (re-checked).
  - It cuts the history as `[:2] + [10:]` (`monte_carlo_sample_alfworld.py:219`, re-checked). The ALFWorld 1-shot prompt is 12 messages (instruction + OK + a 10-message example, re-checked against `alfworld_icl.json`).
  - So every released ALFWorld MC rollout first executes the example's last action, "put toiletpaper 1 in/on toiletpaperhanger 1", in the real game, and never sees the "Your task is to" observation (`:237-239`).
  - ★ Fix the cut to 12, i.e. `len(prompt)`, and disclose it; this matches the stated method. Literal reproduction would make the MC values largely meaningless.
- **B.** The j = 0 MC rollouts are sampled but never used (`:145-162` vs `construct_preference_monte_carlo_alfworld.py:232-259`). Skipping them is harmless.
- **C.** An unconditional `from webshop...` import in `generate_response.py:15` and `monte_carlo_sample_alfworld.py:15`; drop it.
- **D.** `rm -r` of output paths (`run_pipeline.sh:25-27,138-140,216-218`): never run it from a project root.

### 2.6 What breaks, and the minimal faithful change

| Problem | Fix (★ closest) |
|---|---|
| Stack, template, mask, FSDP class, flash-attn (X1–X7). `run_pipeline_vllm.sh` uses vLLM 0.4 `use_beam_search` (`fastchat/serve/vllm_worker.py:102-106`). | ★ Our harness + `vllm serve`: T=0 exploration, T=1 / top_p 1 MC, max 512 tokens, stop strings plus id 106. Reimplement DPO+NLL in our trainer. |
| Full FT, ref = SFT, init = previous iterate | ★ Full FT with FSDP. LoRA alternative: merge the SFT LoRA to get W_sft; policy = W_sft + one adapter carried across iterations; ref = adapter disabled (`dpo_trainer.py:979-986`). lr 3e-6 is a full-FT value. |
| Step cap 20 vs our 40 | ★ Use our 40 everywhere (X3). The repo also uses a single limit for exploration, MC and eval. |
| **Pair supply.** Upper bound P ≤ K + Σ_k(T_k − 1) = 382 for K=20 (s0). Expert-chosen pairs need the SFT student's greedy continuation to *fail* on tasks it was trained on, so the real P is likely far smaller. IPR cannot use tasks without an expert trajectory (`construct_preference_monte_carlo_alfworld.py:235,277`). | Count P after SFT before spending GPU time; report it per iteration. |
| Dose: global batch 48 for SFT and DPO | X5(b). The SFT stage is the same as ETO's. |

### 2.7 Compute, K=20 (H = 40, after fix A)

E = ΣT = 382.

| Per iteration | Upper bound (every rollout runs to H) |
|---|---|
| Exploration rollouts / policy calls | 382 / ≤ 11.8k |
| MC rollouts / calls | 1,910 / ≤ 57k |
| DPO tokens | P × ~15k × 1.33 × 3 epochs; P = 100 → 6M; P = 382 → 23M |

- **One-time expert MC:** 5 × 362 = 1,810 rollouts, ≤ 55k calls.
- **Evaluation:** 7 checkpoints × 140 = 980 episodes.
- **Totals:** about **16.5k env episodes** (mostly partial) and ≈ **17–45 B200-h** with LoRA. The low end assumes about 50% early termination and P ≈ 100; the high end assumes the upper bounds. This is the most expensive method.

---

## 3. AgentGym: Behavioural Cloning and AgentEvol

Paths are relative to `third_party/AgentGym/`. Aliases:
- `bc` = `agentenv/agentenv/trainer/bc_trainer.py`
- `evolT` = `agentenv/agentenv/trainer/agentevol_trainer.py`
- `alfC` = `agentenv/agentenv/envs/alfworld.py`
- `wrap` = `agentenv-alfworld/agentenv_alfworld/env_wrapper.py`
- `bc.sh` = `agentenv/examples/behavioral_cloning/train_behavioral_clone_alfworld.sh`
- `evol.sh` = `agentenv/examples/agentevol/train_agentevol_multi_task.sh`
- `filter` = `agentenv/utils/agentevol_filter.py`

### 3.1 Pipeline
- **BC.** Full FT of Llama-2-7b-chat on `alfworld_train_1000.json` (`bc.sh:11-12`, re-checked; the file is not in the repo), then greedy eval.
- **AgentEvol.** The real loop is the shell driver; `AgentEvolTrainer.inference_and_filter()` is dead code with hard-coded `webshop_` ids (`evolT:515-617`). The driver runs `ITER = 0..3` (`evol.sh:5,89`, re-checked). Each iteration:
  1. **Train from the base model**, never from the previous iterate, on `train_iter_{ITER}.json` (`evol.sh:17,100,105`). Iteration 0 is BC.
  2. **Greedy eval.**
  3. **Greedy exploration** (`--do_sample False`) with the just-trained model on `./small_exploration_data/<task>.json` (`evol.sh:183-212`, re-checked). That file is **not in the repo**, so whether it contains tasks beyond the BC set is not in code. It is a separate input from the training file.
  4. **Filter:** keep `reward > 0.99`, i.e. ALFWorld success (`filter:20,41-45`), dedup by `item_id` within the iteration (`:47-58`), and **append the whole current training file** (`--add_original_data True`, `evol.sh:241`; `filter:70-72`, re-checked).
     - So D_{m+1} = F_m ∪ D_m: an accumulating set, expert data included, with no cross-iteration dedup.
     - `train_iter_4` is produced but never trained on.
- There is no teacher, no second model and no reward model; the reward is the env's `won` (`wrap:98`).

### 3.2 Hyperparameters

| Item | Value | Source |
|---|---|---|
| Adaptation | full FT, bf16 weights, gradient checkpointing, no PEFT | `bc_trainer.py:295-316`; `examples/behavioral_cloning/train_behavioral_clone.py:88-91` |
| Optimiser / lr / schedule | `transformers.AdamW` eps 1e-8, wd 0, lr 1e-5, linear decay; warmup `-100` → int(0.1 × total) | `bc:23,285-321` (re-checked); `bc.sh:20-22` |
| Epochs / batch | 1 per iteration; 2 per device × 8 processes × GA 2 = **global 32** | `bc.sh:3,6,15,17`; `evol.sh:6,9,20,22` |
| Total-steps formula | `(len(loader)//num_processes * n_epochs)//GA` | `bc:285-289` (re-checked) |
| Max length | 4096 input (right-truncated); generation `max_length` 4096 = prompt + output | `bc.sh:18`; `agentenv/agentenv/controller/task.py:61-64` |
| Iterations / samples | 4 / **1 per instruction**. `sample_num=3` is parsed but never used | `evol.sh:5,68`; `evolT` never reads it |
| Decoding | greedy for eval and exploration (multi-task driver); the WebShop driver explores at T=1.2 | `evol.sh:155,206`; `train_agentevol_webshop.sh:53-54` |
| Masking | human turns → −100; **every** gpt turn is supervised, including the "OK..." acknowledgement; eos supervised | `bc:118-131` (re-checked) |
| Template | Llama-2 strings `<s>[INST] ... [/INST]` with eos id forced to 2 | `bc:76-79,120,127-129` |
| Accelerate config | `config_file=""`: the user's default, not in the repo | `bc.sh:8` |

### 3.3 Data requirements
- **Format:** `{item_id: "alfworld_<int>", conversations: [...]}`. The int indexes the server's game list: 2,420 train mappings (ids 0–2419) then 200 test mappings in **`valid_train`** (`wrap:30-72,117`).
- AgentGym's ALFWorld test split is not valid_seen.
- **Rollouts:** one per exploration item per iteration.
- **API calls:** none. `APIAgent` is unused.
- **Chinese-origin model:** none in the BC/AgentEvol path.

### 3.4 ALFWorld specifics (their env server)
- **Prompt:** an instruction plus an "OK. I'll follow your instructions..." acknowledgement, with no in-context example (`alfC:227-243`).
- **Admissible commands:** shown as `AVAILABLE ACTIONS:` **only at reset**, not after steps (`alfC:565-566,587-591`).
- **Parsing:** `rsplit("Action:",1)` (`agentenv/agentenv/controller/utils.py:130-150`).
- **Rounds and success:** `max_round = 30` (`bc.sh:47`); success = reward 1 (`wrap:98`).
- **Environment pins:** the server pins alfworld 0.3.3 and hard-codes `ALFWORLD_DATA=~/.cache/alfworld` (`wrap:174`; `agentenv-alfworld/setup.sh:1-5`). That location is outside `~/hq`, which our rules forbid.

### 3.5 What breaks, and the minimal faithful change

| Problem | Fix (★ closest) |
|---|---|
| **K=20 under the official config gives 0 optimizer steps:** `(⌈20/2⌉ // 8 × 1) // 2 = 0`. The linear schedule's multiplier is then 0, so nothing is learned (`bc:285-321`). On one GPU with bs 2 / GA 2, K = 5/10/20 gives 1/2/5 steps. | ★ X5(b): keep lr 1e-5, linear, 10% warmup, wd 0, clip 1, 1 epoch. Scale the global batch so each training round has a pre-registered step count. The original BC run had about 31 steps (1,000 / 32). |
| `transformers.AdamW` was removed **[ext]**; Llama-2 strings; eos id 2 is Gemma `<bos>`; `resize_token_embeddings` | X2: use `torch.optim.AdamW`, the Gemma 4 template, and supervise all model turns, including the acknowledgement, each ending in id 106. |
| Full FT original | ★ Full FT with FSDP `Gemma4UnifiedTextDecoderLayer` and fp32 master weights. Plain bf16 AdamW at 1e-5 loses most updates to rounding; that is my inference, and the repo does not show the authors' accelerate config. |
| Exploration set missing from the repo | ★ Explore every non-seed train game (3,553 − K), one greedy rollout each per iteration, matching the "separate exploration file" design. Alternative: a fixed stratified subsample, e.g. 500. Exploring only the K seed tasks makes the method degenerate. All new tasks are accounted for as new interactions. |
| FastAPI env server vs our harness; 30 rounds; admissible list only at reset | ★ Our harness (X3). The algorithm only needs per-item reward == 1 (`filter:44`). Raise the 4,096 limits, since 40 steps with admissible lists is about 11k tokens. |
| HF `generate` re-prefills every turn | vLLM with prefix caching. Set temperature 0 and top_p 1 explicitly, overriding Gemma's sampling defaults. |

### 3.6 Compute, K=20

**BC alone:** about 0.1M training tokens plus 1 eval, ≈ **0.4 B200-h**.

**AgentEvol, option ★** (explore all 3,533 non-seed games; 4 trainings; 3 useful explorations):

| Item | Value |
|---|---|
| Episodes | 3 × 3,533 + 4 × 140 = **11.2k** |
| Training data, accumulating at an assumed 50% success, about 4.3k tokens per success | D1 ≈ 1.8k trajectories (≈ 7.7M tokens), D2 ≈ 3.6k (15M), D3 ≈ 5.3k (23M); about 46M tokens over all rounds |
| Total | ≈ **20 B200-h** with LoRA, about 25 with full FT |

**AgentEvol, option B** (500 games per iteration): ≈ 4 B200-h.

---

## 4. WKM (World Knowledge Model)

Paths are relative to `third_party/WKM/`; WKM's eval code is ETO's `eval_agent`.

### 4.1 Pipeline (what exists in code)

| # | Stage | Generator | Code |
|---|---|---|---|
| 1 | Expert SFT of an "experienced agent" | — | **not in WKM code**; the closest is ETO's SFT |
| 2 | Rejected trajectories, one per chosen trajectory | **not in code** (README only names the pair format; an unused T=0.7 config exists) | input format at `src/world_knowledge_build.py:24-27,37-46,56`; `src/eval/eval_agent/configs/model/fastchat_explore.json:5-7` |
| 3 | Task knowledge, one per pair: prompt = success trajectory + failed trajectory + a per-type example | a **local** OpenAI-compatible server (`localhost:8000`, key "EMPTY", default model name `mistral`) = the agent model itself ("self-synthesize", README.md:44). **No real OpenAI call.** | `src/utils/llms.py:3-11`; `src/world_knowledge_build.py:15-22,61-66,116`; `src/prompts/knowledge_template.py:74-90`. T 0.5, max 512 tokens; observations cut to 1,000 chars |
| 4 | State knowledge, one per human turn of the chosen trajectory | same local LLM, T 0.5, max 128 tokens, `stop='\n'` | `src/world_knowledge_build.py:79-92`. Only `chosen` is written, so task knowledge is dropped from this output |
| 5 | Merge outputs 3 and 4, and rename the misspelled key `task_knowledege` | **not in code** | needed by `src/train_data_process.py:58,83` |
| 6 | Build training data | — | Agent data: task knowledge is appended to the first observation; state knowledge is **not** given to the agent (commented out) (`src/train_data_process.py:80-89`). Knowledge-model data: first answer = task knowledge; later turns map (previous action + observation) → state knowledge (`:51-61`). **Bug:** the prompt lookup needs `mode ∈ {wm, agent}` (`:7-11,97`), but the knowledge-model branch needs `mode == "world_model"` (`:101`), so the shipped code cannot produce knowledge-model data. |
| 7 | State knowledge base: records {pre_action, state knowledge, action} | embeddings from `OpenAIEmbeddings("text-embedding-ada-002")` pointed at `localhost:8000`, i.e. whatever local FastChat worker answers to that name (mean-pooled last hidden state) | `src/state_base_build.py:6-7,19-29,43`; `src/eval/fastchat/serve/model_worker.py:146-167,205-213` |
| 8 | Two **separate LoRA runs on the same base**: agent and knowledge model | — | README.md:102-131 (old LLaMA-Factory `src/train_bash.py`; version not pinned) |

**Inference** (`src/eval/eval_agent/eto_multi_main_probs.py`, re-checked).

Step 1:
- The knowledge model writes task knowledge. The **injection into the agent is commented out** (`:101`), so the agent never sees task knowledge at test time even though it was trained with it.
- The agent then generates freely (`:104`).

Each later step:
1. The knowledge model writes state knowledge (`:45,53`).
2. The agent writes a Thought up to "Action: " (`:56-59`).
3. A one-token forward pass gives logits over the first tokens of 12 verbs → P_agent (`src/eval/eval_agent/utils/retrieve_action.py:11,22-48`).
4. KNN retrieval on the state knowledge, filtered by the previous verb, gives P_kb (`:51-67,88-96`).
5. The two are mixed as γ·P_agent + (1−γ)·P_kb. If P_agent is all-NaN, P_kb alone is used (`:123-134`, re-checked).
6. Argmax verb, then the agent greedily writes the arguments (`eto_multi_main_probs.py:73-85`).

The call site uses `k = 3000` and `gen_weight γ = 1`; the function defaults are k = 10, γ = 0.7 (`eto_multi_main_probs.py:67-69`; `retrieve_action.py:123`, both re-checked). With `top_k = 1` in the worker, P_agent is one-hot, so in the released configuration the knowledge base acts only as a NaN fallback (`src/eval/eval_agent/agents/fastchat_agent.py:37`; `src/eval/fastchat/serve/inference.py:56-57,173-178`).

### 4.2 Hyperparameters

| Item | Value | Source |
|---|---|---|
| Adaptation | LoRA, `lora_target q_proj,v_proj`; rank/alpha/dropout **not in repo** (LLaMA-Factory defaults of an unknown version) | README.md:113-114 (re-checked) |
| Optimiser | lr 1e-4, cosine, no warmup set, 3 epochs, 4 GPUs × 4 × GA 2 = **global 32**, fp16, `cutoff_len` 2048, `train_on_prompt False` | README.md:104,118-130 (re-checked) |
| Template | placeholder `model_template` | README.md:112 |
| Decoding | agent T 0, max 512 tokens, top_p 0, top_k 1; knowledge model same but max 128 | `configs/model/fastchat.json:5-7`; `eto_multi_main_probs.py:152,159` |
| Max steps | 40, no ICL (`is_icl: false`) | `src/eval/eval_agent/configs/task/alfworld.json:15-16` |

### 4.3 Data, models and ALFWorld specifics
- **Expert data:** sharegpt-format trajectories (count not in repo; data on HF only). Exactly one rejected trajectory per chosen one.
- **Calls:** no OpenAI API anywhere; all "OpenAI" clients are localhost. No Chinese-origin defaults (Mistral/Gemma/Llama in README.md:44).
- **Observation only**, no admissible commands. Lowercasing and punctuation stripping of actions. Parse error → "Error Input" (`src/eval/eval_agent/envs/alfworld_env.py:41-84`).
- `dev` = valid_seen, and the README evaluates `test` = valid_unseen (README.md:163).
- **Training vs eval prompts differ:** `src/prompts/alfworld_prompt.py:1-37` vs `alfworld_inst_{ag,wm}.txt`.

### 4.4 What breaks, and the minimal faithful change

| Problem | Fix (★ closest) |
|---|---|
| Missing pieces: rejected sampling, merge script, knowledge-model data bug, missing knowledge-base JSON and training data | ★ Fill from the code WKM builds on: ETO-style greedy SFT-agent exploration on the K tasks for "rejected", plus a 10-line merge. Fix the `mode` string. Log each as an engineering fill. If a K-task rollout succeeds, there is no failure contrast (WKM never checks, `src/world_knowledge_build.py:37-46`); option: resample at T = 0.7 until a failure, which is a disclosed deviation. |
| **Repo-literal vs paper behaviour** (task knowledge not injected, γ = 1, k = 3000) | **User decision.** ★ Repo-literal mode (γ = 1, one-hot verb, KB fallback, no task knowledge to the agent) follows the "official repo" rule. "Method" mode uncomments `:101`, sets γ = 0.7 and uses an untruncated softmax. Run both as flags if the budget allows. |
| k = 3000 vs a knowledge base of only about 382 entries at K=20 | Literal k returns the whole base, so P_kb becomes the verb frequency given the previous verb. Option: k = 10 (the function default). Report the base size. |
| Embedder | ★ The same base (Gemma-4-12B), mean-pooled last hidden state, L2-normalised; this mirrors the FastChat worker. Option: OpenAI `text-embedding-3-small`. **Never BGE/GTE.** |
| Two models at eval | One vLLM server with multi-LoRA (agent + knowledge adapter). The verb-logit step needs **student** first-token logits only: either a one-token HF forward pass, or vLLM `logprobs` (must cover all 12 verbs). Check that the 12 Gemma 4 first tokens are distinct. |
| Assistant-turn continuation is Mistral-specific (`fastchat_agent.py:136`) | Render the open model turn without 106 (`continue_final_message`). |
| LoRA targets / dose | q_proj, v_proj on `.language_model.` modules only, lr 1e-4, 3 epochs, cosine, bf16. Global batch 32 with K=20 gives 3 steps → X5(b). `cutoff_len` 2048 truncates 40-step episodes of about 11k tokens; option: raise it (disclosed). |
| Prompt has no admissible commands | X3 decision. Keep WKM's 12-verb set either way. |

### 4.5 Compute, K=20
- **Training:** about 0.3M tokens (agent) + 0.5M (knowledge model).
- **Data generation:** 20 rejected episodes; 20 task-knowledge + about 402 state-knowledge local calls; about 382 embeddings.
- **Eval:** 4 dependent generation calls + 1 embedding per step, sequential within an episode, so about 4–5× a normal eval.
- **Total:** episodes ≈ 20 + 140 (+140 for the agent-only SFT row); ≈ **2–3 B200-h**.

---

## 5. LEAP (privileged AI feedback)

Paths are relative to `third_party/leap_llm/`.

### 5.1 Pipeline
- **Iteration 0, behaviour cloning on the ALFWorld handcoded expert** (not on GPT-4o demonstrations):
  1. Expert logs for every train game: `info['extra.expert_plan'][0][0]` (`scripts/dataproc/collect_logs_alfworld.py:31-59`).
  2. GPT-4o writes post-hoc REASONs per step, one call per trajectory (`scripts/dataproc/annotate_reason.py:31-63`).
  3. **Privileged state** = a GPT-4o free-text summary of the expert trajectory, one call per game: Task / Essential Objects / Critical Locations / Optimal Action Sequence (`scripts/dataproc/extract_privileged_state_from_logs.py:24-74`; `prompts/alfworld/extract_privileged_state_template.j2:23-56`). It is not PDDL facts.
  4. One **single-turn example per step**: the user prompt contains the flattened (observation, action) history with no past reasons; the target is `REASON:\n...\nACTION:\n...` (`scripts/dataproc/create_training_data.py:52-86`; `prompts/alfworld/alfworld_template.j2:1-40`). The 85/15 split is by trajectory (`create_training_data.py:165-167`).
  5. SFT from Meta-Llama-3-8B-Instruct (`bash/train-sft-alfworld-iterx.sh:27-28`, re-checked).
- **Iteration i ≥ 1:**
  1. The iteration i−1 student plays **every train game once**, capped at 20 actions (`configs/training_alfworld.yaml:57-68`; `scripts/eval/eval_alfworld.py:116-128`, re-checked).
  2. Only failed episodes (`score <= 0`) go to GPT-4o, one call per trajectory, with the privileged state. It returns corrected reason/action for **every** step (`scripts/dataproc/correct_student_trajectory.py:94-106,177-179`, re-checked; `prompts/alfworld/correction_oracle_template.j2:34-53`).
  3. The whole corrected trajectory becomes per-step examples: context = the student's original history, target = the correction (`create_training_data.py:49-58`).
  4. **DAgger-style aggregation:** SFT at iteration i loads iter0…i data, and **each directory is subsampled to the size of the smallest** (`bash/train-sft-alfworld-iterx.sh:15-22`; `scripts/train/sft.py:59-72`, re-checked).
  5. Training starts from the merged iteration i−1 model with a fresh LoRA (`train-sft-alfworld-iterx.sh:29-31`; `scripts/utils/upload_models_to_hf.py:23-26`).
- **DPO variant:** chosen = correction, rejected = the student's original step, current iteration only, with a hard-coded starting model (`bash/train-dpo-alfworld-iterx.sh:4,38-40`; `scripts/train/dpo.py:35-41`).
- **Iterations:** SFT iter0–3 checkpoints are referenced (`configs/eval_alfworld.yaml:36-50`). The notebook's valid_seen results: iter0 68.6, iter1 96.4, iter2 90.0, iter3 94.2 (`notebooks/alfworld_evaluation.ipynb`, cells at lines 185–266).

### 5.2 Hyperparameters

| Item | Value | Source |
|---|---|---|
| SFT LoRA | r 128, α 64, dropout 0.05; targets unset → PEFT default (llama → q,v). **gemma4_unified has no PEFT default**, so pass `.language_model.*(q_proj\|v_proj)` explicitly | `train-sft-alfworld-iterx.sh:57-62`; `src/leap_llm/utils/trainer.py:143-152`; peft 0.20 `utils/constants.py:100` (checked) |
| SFT optimiser | lr 3e-5, cosine, warmup_steps 10, adamw_torch_fused, max_grad_norm 0.3, bf16, seed 42; 4 × GA 8 = global 32 on one process (`device_map="auto"`); 1 epoch; max_seq_length 8000; eval/save every 200 steps, best by eval_loss | `train-sft-alfworld-iterx.sh:40-66` (re-checked); `scripts/train/sft.py:113-119` |
| DPO | lr 5e-7, β 0.01, 1 epoch, 2 × 8, max_length 8000, max_prompt 6000 | `bash/train-dpo-alfworld-iterx.sh:13-19,41-65` |
| Masking | `DataCollatorForCompletionOnlyLM` on a `<\|response\|>` text marker. One reply per example, so only the reply is supervised. The marker is absent at inference | `sft.py:9,35,89-96,122-145`; `src/leap_llm/agents/hf_agent.py:82-87` |
| Teacher | `gpt-4o`, T 0.3, top_p 1, n 1, JSON code-fence parsing, ≤ 3 attempts | `correct_student_trajectory.py:33-35,102-104`; `src/leap_llm/utils/openai.py:38-40,57-58` |
| Student sampling | HF `generate`, max_new_tokens 256, T 0.3, stop on `<\|eot_id\|>`; prompt left-truncated at 6000 | `hf_agent.py:43,88-99`; `eval_alfworld.py:66` |
| Library versions | TRL/transformers **unpinned**; the API usage implies TRL ≤ 0.11 **[ext]** | `requirements.txt:1-24` |

### 5.3 Data and privileged information
- **What the teacher sees.** The privileged state is derived from the handcoded expert's *executed* trajectory per game. The expert wrapper only exists on the train split (installed alfworld 0.4.2, `alfred_tw_env.py:263-266`).
- **Can our harness provide it?** Yes, audit-side: run `AlfredExpert` per rollout task, then the extraction prompt (one teacher call), then insert the result into the correction prompt only.
- **Leak paths.** The student prompt has no privileged slot (`alfworld_template.j2:1-41`). The leak risk is the corrected REASON; LEAP relies on prompt instructions only (`correction_oracle_template.j2:20-31`).
- **Corrected actions are not checked against the admissible list** on the main path (`correct_student_trajectory.py:51-70`).
- **Training context.** LEAP builds it from the *teacher's echoed copies* of the original observation/action (`configs/training_alfworld.yaml:32-36`). Overwrite these from our rollout log.
- **Models:** no second model; no Chinese-origin model anywhere.

### 5.4 ALFWorld specifics
- **Prompt:** a single user message with no ICL. It contains the task, a repr of past (observation, action) pairs, the current observation and `candidate_actions` (the admissible list, every step), plus rules ("MUST choose from candidate_actions") (`alfworld_template.j2:4-35`; `eval_alfworld.py:139`).
- **Parser:** first line after `ACTION:`, lowercased, characters outside `[a-z0-9 /]` stripped (`src/leap_llm/utils/parser.py:15-28`).
- **Success** = last score > 0 (`eval_alfworld.py:153-171`).
- **Splits:** rollouts and BC both use all of train. The reported table is valid_seen, which matches our dev140.

### 5.5 What breaks, and the minimal faithful change

| Problem | Fix (★ closest) |
|---|---|
| **Privileged teacher** | ★ Option A (LEAP): handcoded expert audit-side → extraction → correction; disclose it as an oracle resource used only during training. Option B: the repo's no-privileged template (`prompts/alfworld_realizability_ablation/generation_oracle_0_privileged_template.j2`) = DAgger-style correction; label it "LEAP (0% privileged ablation)", not LEAP. |
| Iteration-0 BC data: theirs is 3,553 expert trajectories with post-hoc GPT-4o reasons; ours is K seeds with the teacher's own Thoughts | Use the K seeds; skip `annotate_reason` (disclose). The 85/15 split leaves int(0.85K) for training (17 of 20). Faithful: keep it. Option: train on all K and select on a separate train-task dev set. |
| Prompt format (single-turn, flattened history, REASON/ACTION) vs our multi-turn ReAct | ★ A: keep LEAP's single-turn prompt with REASON/ACTION mapped to Thought/Action; our eval harness must then render LEAP's prompt for LEAP students. B: our multi-turn ReAct, one example per step with only turn t supervised; past Thoughts become visible, which is a deviation. |
| **Dose:** about 10–15 steps total at global 32, 1 epoch. Warmup is 10 and the first eval/save is at step 200, so no checkpoint selection happens. **`N = min` truncates every correction directory to the seed-set size.** | Faithful: literal (near no-op). Options: X5(b); upsample the seed directory before taking the min (disclosed). |
| Rollout pool: all 3,553 train tasks per iteration; cap 20 vs our 40 | ★ All non-seed train tasks (faithful, expensive in API calls), or a stratified M = 100–200 per iteration (budgeted, disclosed). Use 40 steps (X3); this roughly doubles correction tokens. |
| Code bugs | `correct_student_trajectory.py:8` imports only `List, Tuple` but annotates `Optional`/`Dict` → NameError at definition. `response`/`cost` are undefined if the first call raises (`:118,131`). `eval_alfworld.py:141-143,171` reuses a stale `score`. Fix all three as engineering. |
| TRL API drift (removed collator, `tokenizer=`, `max_seq_length=`) **[ext]**; `<\|eot_id\|>` hard-coded | Our trainer with an assistant-only mask (identical supervised tokens). Stop on id 106. Set T = 0.3 and max 256 tokens explicitly in vLLM. |
| Chaining LoRA across iterations | ★ Merge each iteration and start a fresh LoRA, as the repo does. That costs about 24 GB of disk per merge; our serving is adapter-only. |

### 5.6 Compute and API, K=20 (3 iterations after iteration 0)

| Scenario | Rollout episodes | GPU | Teacher calls |
|---|---|---|---|
| Faithful pool (all 3,553 games, f = 0.3) | 10.7k | ≈ **15 B200-h** | about 1,070 correction calls per iteration (≈ 14M input / 9M output tokens at the 40-step cap), plus 3,553 one-time extraction calls (≈ 12M / 1M) |
| Budgeted (M = 200, f = 0.5) | 600 | ≈ **3 B200-h** | about 100 calls per iteration (≈ 1.35M / 0.9M) plus 200 extraction calls (≈ 0.7M / 0.06M) |

- Both scenarios add 4 × 140 eval episodes.
- Token figures are estimated at about 4 characters per token and exclude the teacher's reasoning tokens.

---

## 6. Kang et al., Agent Distillation (smolagents CodeAgent; QA/math)

Paths are relative to `third_party/agent-distillation/`; `uf/` = `exps_research/unified_framework/`, `sa/` = `src/smolagents/`.

### 6.1 Exact training recipe
- **(a) First-thought prefix (FTP)** is used for teacher data generation only.
  - **Prefix:** the first paragraph of the teacher's CoT answer, `response.split("\n\n")[0] + "\n\n"`, prefixed with "Thought: " (`exps_research/build_prefix_memory.py:31-32`).
  - **CoT source:** the same teacher (Qwen2.5-32B-Instruct) run in `--experiment_type reasoning`, i.e. plain CoT with `enable_thinking: False` (`build_prefix_memory.py:45`; `uf/processors/reasoning.py:102-111`; `sa/prompts/teacher_model.yaml:1-12`; `sa/models.py:1251`).
  - **Storage:** `{question: prefix}` JSON.
  - **Injection:** on the agent's first step only, as a final assistant message with `continue_final_message=True`; the prefix is prepended to the returned text, so it is **supervised in SFT** (`uf/processors/agent.py:112-116`; `sa/agents.py:1455-1492`; `sa/models.py:1264-1283`).
  - The student never receives a prefix (`scripts/inference/run_agent_student.sh:114-128`).
  - `OpenAIServerModel` has **no prefix support** (`sa/models.py:1102-1123`).
- **(b) Self-consistent action generation (SAG)** is test-time only, for the student.
  - **Sampling:** N = 8, T 0.4, top_p 0.8, max 1,024 tokens (`run_agent_student.sh:10-13,122-123`, re-checked).
  - **Filter:** a candidate passes if its code parses and executes without error in the shared executor (`sa/agents.py:1435-1453`).
  - **Vote:** passing candidates are grouped by execution output. The largest group's first member wins; if all groups are singletons, the first passing candidate; if none passes, `choices[0]` (`sa/agents.py:1461-1515`, re-checked).
  - Teacher runs use n = 1, T 0 (`run_agent_teacher_train.sh:112`).
- **(c) Teacher trajectories.**
  - CodeAgent + Wikipedia retriever (e5-base-v2), `max_steps = 5`, n = 1, T 0, top_p 0.8, max 1,024 tokens (`run_agent_teacher_train.sh:4,23,106-115`; `uf/run_experiment.py:73`).
  - Data: 1,000 HotpotQA-hard + 1,000 MATH L4–5 + 1,000 MATH L2–3 (`run_agent_teacher_train.sh:11-13`).
  - Filter: keep correct answers (LLM judge gpt-4o-mini for QA, `math_equal` for math) with no "Error in code parsing"; execution errors are kept (`uf/filter_agent_training_data.py:48-85`; `uf/score_answers.py:29-124`).
  - The final count is not in code; the HF dataset is named "trajectories_2k".
- **(d) SFT.**
  - LoRA r 64, α 128, all-linear, dropout 0.05 (`exps_research/finetune_sft.py:137-146`).
  - lr 2e-4, 2 epochs, 1 per device × GA 2 × 4 GPUs = **effective 8**, max length 10,240, bf16 + FSDP, seed 42 (`exps_research/scripts_train/finetune_sft_agent.sh:11-21`, re-checked; `scripts/training/train_agent_ftp.sh:5`).
  - Scheduler and warmup: library defaults, not in code.
  - **One sequence per trajectory.** Tool responses become user turns, which are masked; **assistant-only loss** via a multi-turn completion collator (`exps_research/train_utils/preprocess.py:155-213`; `exps_research/train_utils/utils.py:213-239`).
  - The short system prompt is used at both train and test time (`preprocess.py:176-198`; `uf/processors/agent.py:127`).

### 6.2 Released vs missing

| Released | Missing |
|---|---|
| teacher trajectory script, prefix-memory builder, filter, SFT script + collator, student eval (SAG), retriever | **teacher CoT run on the train split (the FTP input):** the code path exists but there is no script (README.md:144 TODO); **prefix-memory JSONs / `logs/`**; question-building scripts; the `cot_memory` builder; an OpenAI judge key file. Teacher trajectories and the LoRA are on HF only, and are **Qwen-generated or Qwen-based** |

Bugs: `train_agent.sh` and `train_agent_ftp.sh` share `SAVENAME`, so the FTP run overwrites the baseline (`:9` in both). The "Reached max steps" filter can never fire (`uf/filter_agent_training_data.py:67` vs `sa/agents.py:718-720`).

### 6.3 Models
- **Teacher:** Qwen/Qwen2.5-32B-Instruct, **Chinese-origin, default** (`run_agent_teacher_train.sh:4`; `build_prefix_memory.py:45`).
- **Students:** Qwen2.5-0.5B default; 1.5B in the README. Allowed list: Llama-3.x, Qwen2.5, Phi (`finetune_sft.py:37-48`). An unknown model triggers `pdb.set_trace()`, then a Llama-3.2-1B fallback (`:119-121`).
- **Judge:** gpt-4o-mini. **Retriever:** intfloat/e5-base-v2 (Microsoft).
- Under our rule only the code is usable: replace the teacher with our OpenAI model and the student with Gemma 4.

### 6.4 ALFWorld adaptation that reuses the code most faithfully

| Component | ★ Closest option | Alternative |
|---|---|---|
| Data conversion | One sequence per trajectory: [system (short ReAct prompt, identical at train and test), user (task + observation + admissible), assistant "Thought...\nAction...", ...]. Drop the trailing user turn (`preprocess.py:148-153`); supervise all assistant turns. | Per-step sequences (not original) |
| SFT | Our trainer with **their** hyperparameters: r64 / α128 / all-linear on `.language_model.` modules, lr 2e-4, 2 epochs, effective batch 8. Raise max length above 10,240 if 40-step episodes exceed it (disclosed). | Our r16 / lr 1e-5 (deviation) |
| FTP | Re-collect the K teacher trajectories under FTP. Make one teacher "reasoning" call per attempted task (task + initial observation), take `split("\n\n")[0]`, prepend "Thought: ". The OpenAI chat API cannot prefill an assistant turn **[ext]**, so instruct the model to continue from the given prefix and store prefix + continuation as the first assistant turn (mirrors `sa/models.py:1283`). Key by game path. Disclose the emulated prefill. | No FTP (plain Kang SFT on our K seeds) = their non-FTP baseline (`train_agent.sh`) |
| SAG | N = 8, T 0.4, top_p 0.8 per step. "Integrity" = parses into Thought/Action **and** the action is in the admissible list. Vote on the action string; ties → first group; all distinct → first valid; none valid → first sample. ALFWorld is deterministic given state + action, so equal commands give equal observations; voting on the command is the image of voting on the execution output. | Vote on next observations via replay: closer in form, but N real executions per step are impossible without state forking |
| Teacher filter | env success and no parse failures; keep trajectories that contain inadmissible "Nothing happens" steps (analogue of kept execution errors) | strict: drop any invalid action |

Stop strings become `\nObservation` plus id 106. Raise `max_steps` from 5 to 40.

### 6.5 Compute, K=20
- **Teacher (FTP data):** about 29 attempts (s0 needed 29 for 20 wins) × about 19 steps ≈ 550 trajectory calls, plus about 29 CoT calls.
- **SFT:** about 0.2M tokens, about 5 optimizer steps at effective batch 8 (X5 applies).
- **Eval with SAG:** 140 × ≤ 40 × 8 samples, i.e. 8× the decode of greedy (prompt shared within an n = 8 request).
- **Total:** ≈ **1–1.5 B200-h**; episodes ≈ 29 + 140 (+140 without SAG).

---

## 7. Compute summary (one K=20 run, single seed)

Assumptions, labelled as estimates:
- **Generation:** ≈ 800 full episodes per B200-hour. Measured: base dev40, 8 concurrent, 3 min. Higher concurrency could be 2–4× faster.
- **LoRA training:** ≈ 7M tokens per B200-hour. This scales our rai Seed-SFT rate (≈ 2.0M tokens in about 45 min on an RTX-class Blackwell card) by about 2.5×; uncertainty about ±2×.
- **DPO:** about 1.33× per pair-token for the reference forward.
- **Per checkpoint:** about 0.1 h for merge and server start; one dev140 eval ≈ 0.3 B200-h.
- **Episode size:** about 5.1k tokens per seed episode; about 11k tokens per 40-step failure.

| Method | Env episodes | Teacher calls | B200-h (LoRA) | Notes |
|---|---|---|---|---|
| Seed-SFT / AgentGym BC | 140 | 0 | ≈ 0.4 | reference row |
| ETO | ≈ 620 | 0 | ≈ 2 (full FT ≈ 3–4) | P ≤ 20 pairs per iteration |
| IPR | ≈ 16.5k (mostly partial) | 0 | ≈ 17–45 | MC dominates; full FT is heavier |
| AgentEvol (all 3,533 games explored) | ≈ 11.2k | 0 | ≈ 20 (full FT ≈ 25) | ≈ 4 h with 500 games per iteration |
| WKM | ≈ 160–300 | 0 (self-synthesis) | ≈ 2–3 | eval about 4–5× normal |
| LEAP (all games / M = 200) | ≈ 11.3k / ≈ 1.2k | ≈ 3.2k + 3.6k / ≈ 300 + 200 | ≈ 15 / ≈ 3 | plus API tokens (§5.6) |
| Kang (FTP + SAG) | ≈ 170–310 | ≈ 580 | ≈ 1–1.5 | the FTP re-collection is the only teacher cost |

Multiply by 3 for K ∈ {5, 10, 20}; smaller K is cheaper only for training. Exploration-pool methods (AgentEvol, LEAP) cost about the same at every K.

---

## 8. Blockers and decisions needed from the user

1. **Kang:**
   - The default teacher and students are Qwen, and the released data/LoRA are Qwen-derived, so only the code can be used.
   - The FTP-input generation script is missing; we would write the train-split CoT run with the existing `reasoning` processor.
   - FTP prefill must be emulated on the OpenAI API.
   - Decision: run "Kang SFT (+SAG)" without FTP as the faithful core, and FTP as an emulated variant?
2. **WKM:**
   - Code is incomplete: no rejected sampler, no merge step, and the knowledge-model data mode bug.
   - Released inference disables task-knowledge injection and the KB mixing (γ = 1).
   - Decision: repo-literal vs method mode, k = 3000 vs 10, and whether filling missing glue from ETO is acceptable under the "we do not invent" rule.
3. **LEAP:** faithful LEAP needs the ALFWorld handcoded expert (privileged, audit-side only). Decision: disclosed privileged LEAP, or the 0%-privileged ablation (= DAgger-style). Also the rollout pool size (API cost, §5.6).
4. **IPR:** the official ALFWorld MC sampler is the WebShop file and corrupts every MC rollout. Decision: fix it (recommended, disclosed) vs literal.
5. **All methods:**
   - The few-shot dose rule (X5).
   - Full FT for ETO/IPR/AgentGym vs LoRA everywhere (X6).
   - Harmonised harness prompt (obs + admissible, 40 steps) vs repo-literal prompts (X3).
   - Exploration pools beyond the K tasks for AgentEvol and LEAP (new interactions, accounted).
6. **No other blockers:** none of ETO, IPR, AgentGym BC/AgentEvol, WKM or LEAP uses a Chinese-origin checkpoint in its default pipeline, and none needs teacher logits.
