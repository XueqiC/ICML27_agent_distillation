# Implementation specs for the baselines added on 2026-10-18/19

Written before any implementation, from each paper's method section (read 10-18/19) and, where an official repo exists, its code. User decisions:
- 10-19: run every baseline completely on the same grid, faithful to the repo, or to the paper where no complete repo exists.
- Q1–Q5 resolved by the recommended defaults (`docs/direction/2026-10-19_full_baseline_grid_plan_ZH.md` §4).

## Shared protocol (all methods; identical to the existing ports)

- **Grid.** K ∈ {5, 10, 20, 50, 100} × support seeds s0–s2 (`data/verified/seeds_k{20,100}_s*` and their nested subsets), plus the full-training cell (`data/verified/seeds_full_train`, every eligible TRAIN game).
- **Models.**
  - Student: `google/gemma-4-12B-it@707f0a3b`, thinking OFF, 1,024 output tokens per step.
  - Teacher: gpt-6-luna via `teachers/capped.py` (black box, text only).
- **Harness H0.** `sed.react` prompt and parser, admissible actions every turn, 40 policy turns. Training targets are the visible assistant turns with the Gemma OFF prefix (`train.dataset.encode_turn`).
- **Exploration.** Only on the support set's TRAIN tasks; no outside task pool.
- **Evaluation.** dev140, batch-invariant vLLM, greedy, seed 0, for **every** iterate/checkpoint. The HF/vLLM equivalence gate is required before evaluation. No dev-based selection.
- **Batch.** The original batch shrinks as $`b'=\min(b,\max(1,\lfloor N/8\rfloor))`$. Every deviation is registered in the method's `baselines/<method>/DEVIATIONS.md` *before* the first GPU run.
- **Pipeline infrastructure.** Hash-bound resumable stages (`baselines.common.pipeline`), CPU tests with `CUDA_VISIBLE_DEVICES=''`, a single-GPU smoke at K20 s0, and stage-split hpg chains (teacher-only stages CPU-only).
- **Teacher calls.** Each method has a hard per-cell cap (below). Every call is recorded in the capped ledger.

---

## AD-FTP: Kang et al., 2025 (repo Nardien/agent-distillation; FTP workflow not released, so implemented from the paper and partial helpers)

- **Data generation, per support task.**
  1. CoT call: the teacher answers the task with a chain-of-thought prompt and no tools, $`y_1\sim p_T(\cdot\mid x,I_{\rm CoT})`$. The prompt contains the task text plus the initial observation, and asks the teacher to think step by step about how to solve the task without taking actions.
  2. The first-thought prefix is the first paragraph of that response: `response.split("\n\n")[0]`, as in `exps_research/first_thought_prefix/build_prefix_memory.py`. The agent's first thought starts with `"Thought: " + prefix`.
  3. Agent trajectory: $`\tau\sim p_T(\cdot\mid x,y_1,I_{\rm agent})`$ in H0. The first teacher turn must begin with the prefix. If the API cannot prefill an assistant message, put the prefix in the prompt as the required beginning of the first reply and reject replies that do not start with it; register this.
  4. Execute in the environment and keep the trajectory only on verified success. Up to 3 attempts per task (one CoT call per attempt).
  5. If all attempts fail, keep the existing trusted demonstration for that task, so the task set equals the support set. Count and register these fallbacks.
- **Training (unchanged AD recipe).** LoRA r64/α128/dropout 0.05 on all language-model linear layers, learning rate 2e-4 linear, 2 epochs, batch 8 → $`b'`$, training seed 0, turn packing. The student is trained on the full trajectory, including the prefix tokens.
- **Test time.** SAG with N = 8, temperature 0.4, top-p 0.8, majority vote over admissible parsed actions (existing implementation). Also evaluate greedy.
- **Collections.**
  - One FTP collection per K100 support set (s0–s2). Subsets K5–K50 are the nested views, exactly as for the trusted demonstrations.
  - One FTP collection for the full-training set.
- **Cap.** ≤ 3 × (1 CoT + 40 agent) calls per task.

## SAD: Structured Agent Distillation, Liu et al., 2025 (no repo; implemented from the paper)

- **Loss.** The paper sums span-masked KL to the teacher's token distribution, $`\lambda_r\sum m_r\,\mathrm{KL}(p_T\Vert p_S)+\lambda_a\sum m_a\,\mathrm{KL}(p_T\Vert p_S)`$, with $`\lambda_r=\lambda_a=1`$.
  - Black-box teacher (user Q1): $`p_T`$ is one-hot on the teacher's tokens, so each KL term is the token cross-entropy.
  - Reason span: the `Thought:` text of each reply. Action span: the `Action:` line plus the turn-end token. Without the turn-end token the reply could never stop; this is a harness requirement, registered.
  - Nothing else in the reply is supervised. Observations are masked.
- **Curriculum.** $`C(\tau)=\alpha\,\mathrm{len}(r_{1:k})+\beta\,\mathrm{len}(a_{1:m})+\gamma\,\mathrm{entropy}(\pi_T(\tau))`$; trajectories are sorted once by $`C`$, easy first, and every epoch follows that order.
  - The paper gives no α, β, γ: use α = β = 1 (token counts).
  - γ = 0, because the teacher's entropy is unavailable to a black box.
  - Register both.
- **Recipe (paper).**
  - Full fine-tuning (reuse the ETO/IPR SFT trainer: FSDP2, fp32 masters, bf16 compute, turn packing; shared-history packing once merged).
  - AdamW, learning rate 2e-5 with linear warmup (3% of steps) then cosine decay, clipping 1.0, batch 64 → $`b'`$.
  - **Training amount.** The paper trains 20k steps at batch 64 on sequences truncated to 512 tokens, i.e. 655M token positions. Our protocol forbids truncation, so the run is defined by the same **655M processed token positions**, not 20k steps. Register this.
  - Checkpoints at 1/16, 1/8, 1/4, 1/2 and 1 of that budget, all evaluated.
- **Teacher calls.** 0.

## WKM: Qiao et al., 2024 (repo zjunlp/WKM @ f1fe28f is incomplete: no rejected sampler, no merge; the rest from the paper)

1. **Experienced agent.** LoRA SFT of the student on the K demonstrations: LoRA r8/α16, learning rate 1e-4, 3 epochs, batch 32 → $`b'`$ (paper and repo LLaMA-Factory recipe).
2. **Rejected trajectories.** The experienced agent explores every support task once, greedily; failed trajectories become the rejected set. This is the missing sampler, written from the paper.
3. **Knowledge synthesis by the agent model itself** (paper: "we prompt the agent model itself", so no teacher calls):
   - *Task knowledge* per task, by contrasting the expert (chosen) and rejected trajectories, with the repo's prompt.
   - *State knowledge* at every expert step, by summarizing the current state.
4. **World knowledge model.** A separate LoRA on the same backbone is trained to generate task knowledge (from the task) and state knowledge (from the history); recipe as in step 1.
5. **Agent retraining.** The agent LoRA is retrained on expert trajectories with the task knowledge inserted, as in the repo's data format.
6. **State knowledge base.** Action–state–action triplets from the expert trajectories. Retrieval uses sentence-embedding cosine similarity with a non-Chinese embedding model such as `sentence-transformers/all-MiniLM-L6-v2`; register the choice.
7. **Inference.**
   - The world knowledge model writes the task knowledge once per task.
   - At each step it writes the state knowledge; retrieve the top N triplets (paper N = 3000, capped by the size of the knowledge base).
   - Over the admissible actions, choose the argmax of $`\gamma\,p_{\rm agent}(a)+(1-\gamma)\,p_{\rm know}(a)`$, with γ = 0.4 for ALFWorld.
   - $`p_{\rm know}`$ is the frequency distribution of next actions in the retrieved triplets, as in the paper.
   - This needs two LoRAs at evaluation time. The harness turn format must stay H0, with the knowledge inserted where the paper inserts it; register it.
- **Teacher calls.** 0.

## STeCa: Wang et al., 2025 (repo WangHanLinHenry/STeCa is incomplete: no SFT, no reflection scripts; the rest from the paper)

1. **SFT warm-up** on the K demonstrations: full fine-tuning (IPR framework), learning rate 3e-6, batch 16 → $`b'`$, 3 epochs (IPR SFT epochs; register).
2. **Exploration with Monte Carlo step rewards.** Reuse IPR's expert-prefix exploration and Monte Carlo routine, N = 5 completions per step by the SFT model.
   - The first student step whose reward falls below the preceding expert step's reward by more than δ = 0 is the deviated step $`t`$.
3. **Reflection.**
   - The teacher (gpt-6-luna, in place of GPT-4o) receives the expert sub-trajectory up to $`t-1`$, the deviated action and the ground-truth expert action. It writes a reflective thought, using the paper's prompt.
   - Calibrated trajectory = prefix + reflection-augmented step + the subsequent expert steps: $`D_c`$.
4. **Further data.**
   - $`D_s`$: the expert sub-trajectory from $`t`$, for failed explorations.
   - $`D_e`$: successful explored trajectories paired with their expert trajectories.
5. **Reinforced training, 1 epoch.** Reward-weighted log-likelihood:

```math
\mathcal L=\mathbb E\big[r_c\log\pi_\theta(e^c_{t:m}\mid e_{1:t-1})\big]+\mathbb E\big[r_s\log\pi_\theta(e_{t:m}\mid e_{1:t-1})\big]+\mathbb E\big[r_e\log\pi_\theta(\tilde e_{1:m}\mid u)\big],
```

  - Rewards: $`r_c=1+\eta\,d_{\rm nDTW}(e^c_{t:m},e^d_{t:m})`$, $`r_s=1+\eta\,d_{\rm nDTW}(e_{t:m},\hat e_{t:m})`$, $`r_e=1-\eta\,d_{\rm nDTW}(\tilde e_{1:m},e_{1:m})`$, with η = 0.01 for ALFWorld.
  - The learning rate is the SFT learning rate (the paper does not state the RL learning rate; register).
- **Cap.** ≤ 1 teacher call per deviated trajectory.

## HAD: Harness-Aware Distillation, Choi et al., 2026 (code "to be released"; implemented from the paper)

- **HAD harness, for this method only (user Q2).** It adds text to each observation and never blocks actions:
  - *State tracking* every step: available actions, current location and inventory, visit history, step counter and budget.
  - *Behaviour monitoring* when triggered: an action repeated at the same state; a stall, i.e. recent states with no progress.
  - Implement as a deterministic function of the environment state and the interaction history.
- **On-policy loop.**
  - Each round, the current student rolls out on support tasks with the HAD harness; the paper uses 16 tasks per round, so use min(16, K) tasks, cycling.
  - At each visited state, query the teacher twice: with the harness information ($`\tilde x_t`$) and without it ($`x_t`$).
  - Each collected step is used once, within two policy versions.
- **Losses.**
  - Response distillation: token-mean negative log-likelihood of the teacher's with-harness reply $`\tilde y_t`$ at the student's history.
  - Action preference:
    - $`\Delta_t=\tfrac1{\lvert\tilde a_t\rvert}\log\pi_\theta(\tilde a_t\mid\tilde x_t,\tilde z_t)-\tfrac1{\lvert a_t\rvert}\log\pi_\theta(a_t\mid\tilde x_t,\tilde z_t)`$, scored under the student's own reasoning $`\tilde z_t`$;
    - loss $`-\log\sigma(\beta\Delta_t)`$;
    - masked when the preferred action contradicts the harness records (validity checker).
  - Total $`\ell_{\rm dist}+\lambda\,m_t\,\ell_{\rm pref}`$, with λ balanced per minibatch by gradient-norm matching.
  - β is taken from the paper; if it is not stated, register 0.1.
- **Recipe (paper).** Full fine-tuning, AdamW, cosine schedule, learning rate 1e-5, batch 60 states → $`b'`$ by the shared rule, 250 updates.
- **Evaluation.** dev140 with the HAD harness **and** with H0. The fair comparison with the other methods is the H0 number.
- **Cap.** 2 × 60 × 250 = 30,000 teacher calls per cell.

## GAD: Black-box on-policy distillation, Ye et al., 2025 (repo microsoft/LMOps/gad + YTianZHU/verl)

- **Per-turn adaptation (user Q5).** A "prompt" is one history visited by the student on the support tasks. The teacher response is the teacher's reply at that same history.
- **Data collection rounds.** Roll out the current student on the support tasks (greedy), collect its histories, then query the teacher once per history.
- **Models.** The generator is the student. The discriminator is initialized from the generator, with a scalar head on the final hidden state.
- **Warm-up, 1 epoch.** Cross-entropy of the generator on teacher replies. The discriminator is trained with the Bradley–Terry loss $`-\log\sigma\big(D(y_T)-D(y_S)\big)`$, 10 discriminator steps before joint training.
- **GAD stage, 2 epochs.**
  - GRPO for the generator with reward $`D(y_S)`$: group size 8, advantages normalized within the group, KL weight 0.001.
  - The discriminator is updated jointly with Bradley–Terry.
  - Learning rate 1e-6 for both stages (GPT-5-Chat-teacher setting).
  - Batch 256 → $`b'`$.
  - Max response 1,536 tokens, H0's 1,024-token step cap enforced at generation.
- Use the official repo's GRPO implementation where possible. Port the ALFWorld per-turn data path, and register.
- **Cap.** 1 teacher call per collected history; at most 3 rounds × 40 steps × |support| per cell.

## Chain-of-Agents: Li et al., 2025 (repo OPPO-PersonalAI/Agent_Foundation_Models: SFT + DAPO; multi-agent trajectory generation from the paper)

- **Teacher (user Q3).** A multi-agent system built from gpt-6-luna, following the paper's roles:
  - thinking agent, planning agent, reflection agent and verification agent (`<double_check>`);
  - one tool agent, the ALFWorld actor, which emits H0 actions.
  - Report this system's own dev140 score.
- **Trajectories.** Record the system's execution on the support tasks as single chain-of-agents trajectories: state, activated agent, output. Filter as in the paper:
  - at least 5 agent–tool interactions;
  - correctness, meaning environment success;
  - reflective trajectories kept and those without reflection downsampled;
  - error-correction trajectories upsampled.
- **Agentic SFT.** $`\mathcal L=-\sum_{t\notin\mathcal O}\log\pi_\theta(\tau_t\mid\tau_{<t},q)`$ with observations masked, in the paper's tag format (`<think>`, `<plan>`, `<reflection>`, `<double_check>`, tool call). Use the repo's SFT recipe.
- **Agentic RL.** DAPO with the repo's settings, reward = environment success on the support tasks.
  - Query filter as in the paper: discard tasks the SFT model solves in all 8 samples.
- **Evaluation.** The student generates the chain-of-agents format, and the H0 action is parsed from its tool-call tag. The H0 step cap counts environment actions.
- **Cap.** ≤ 3 system runs per task × (agents per step) calls.

## AgentDistill: Qiu et al., 2025 (training-free; repo jtk16/AgentDistillation lacks the MCP part; implemented from the paper)

- **Teacher MCP generation.** A manager agent and an MCP-creation module (both gpt-6-luna) solve the support tasks. They write self-contained Python tools (MCPs) whose bodies may only issue H0 actions to the environment, and a tool returns the resulting observations. MCPs come only from successful runs.
- **MCP box.** Built with the paper's prompts:
  - abstraction to task-agnostic, parameterized form;
  - code-level clustering by function;
  - consolidation into one tool per cluster.
- **Student.** The student agent uses the box at inference (tool calls) with no weight update.
  - Every environment action a tool executes counts toward the 40-step cap.
  - Results are labelled "different interface" (user Q4).
- **Cap.** ≤ 3 teacher runs per task plus the box-construction calls.
