# Chain-of-Agents: registered implementation choices

Implementation only; no paid calls, environment benchmark, GPU experiment or
scientific result has been produced by this port.

Authority: `docs/baselines/NEW_METHOD_SPECS_1019.md`, Chain-of-Agents and shared
protocol; Li et al., arXiv:2508.13167 §§3.1–3.3; official
OPPO-PersonalAI/Agent_Foundation_Models at
`9a6594d4661fa994f189074ecfdfd259eea87c6d`. `pipeline.official_sources` verifies
HEAD and the exact content of every used upstream source before preparation.
Upstream source is read only. The small DAPO mathematical adaptation in `rl.py`
is attributed to AFM's Apache-2.0 licensed verl code; no upstream runtime or
model checkpoint is imported.

## Models, environment and data

- The required student is the pinned Gemma 4 12B; every teacher role uses only
  sampled gpt-6-luna text. No upstream Qwen models, GRM, reward models, external
  corpora or pretraining are used. Model names and revision are checked.
- Only the `--verified` collection's support TRAIN tasks may generate training
  data. Shared `registered_support` validation handles nested K views and
  full-TRAIN by collection rule. K always counts original trusted trajectories;
  parent IDs, instances, groups, expanded executions and duplicate exposures are
  separate. No trusted H0 seeds are silently substituted for empty CoA SFT data.
- Five roles: think, plan, reflection, double_check, and alfworld. The official
  prompt uses tool-specific XML names (`web_search`, `crawl_page`, `code`), so the
  one allowed tool is `<alfworld>Action: H0 command</alfworld>`. The paper's
  schematic `<tools>` is not an extra executable tag. No web/code tools exist.
- H0 still owns admissibility, parsing, parser feedback, resets, replay, success,
  and termination. The adapter extracts the sole final tool call and bridges it
  to `Action: ...`. Every original tagged response is preserved as `coa_reply`
  and in transport/agent logs. Role invocations do not consume environment
  actions. The shared 40-policy-turn guard also bounds malformed tool responses,
  as for all other policies; malformed tags never trigger a guessed action.
- No generated answer can stop the environment or determine success. A terminal
  `<answer>` is unnecessary for ALFWorld and is not synthesized after success.

## Unreleased teacher generation and filtering

- The exact OAgents synthesis implementation, activation prompts and sampling
  rates are absent. The paper's role descriptions and released AFM prompt guide
  this explicit reimplementation, not a claim of identical teacher generation.
- Thinking selects the next specialist/tool dynamically using a JSON routing
  envelope; its visible text becomes `<think>`. Plan/reflection/double_check
  provide visible tagged task work. Each specialist can activate at most once
  per H0 turn; thinking precedes every activation. This gives at most eight
  invocations per turn (four think + three specialist + one tool) across five
  distinct role agents. Every activation records public state, raw output,
  rendered output, validation outcome and complete call metadata.
- Each task gets exactly three reserved TRAIN system runs, including unsuccessful
  or malformed runs. Cap = `3 * 40 * 8 * unique_support_tasks` transport attempts;
  no SDK/HTTP retries. Each call uses max_output_tokens=1024; all other sampling
  parameters use the configured provider defaults. Responses may collectively
  exceed the student's per-step generation cap; none is silently truncated.
- Runs, run indices, cache identities and the durable counter survive stage
  retries. Completed per-run receipts are content checked. An interrupted run
  without a receipt consumes its reserved slot and is recorded as unknown;
  remaining slots can continue. Pending/failed transport reservations still
  halt the capped client. No interrupted environment state is fabricated.
- Complexity means at least five **actually executed** environment actions,
  excluding role chatter and parser rejection. Require known environment
  success, well-formed tags and a consistent real reset/prefix replay. No
  teacher correctness judgment is substituted for environment success.
- Reflective means an explicit `<reflection>` activation. Keep all reflective
  eligible trajectories; keep nonreflective ones with deterministic probability
  0.5 from SHA256(seed=0, task_id, system_run). Paper/repo give no numerical rate.
- Error correction means a `<double_check>` score 1 or 2, subsequent visible
  thinking/planning/reflection, and eventual environment success. Selected such
  trajectories have two training copies. Factor 2 is unspecified upstream and
  registered here. Scores are teacher assessments, never verification labels;
  this replaces the unavailable/disallowed GRM credibility model. Corrections
  are identified only within the same real execution, not invented branches.
- No extra teacher quality judge is added. Every rejection and failed attempt
  remains on disk. Repeated ALFWorld commands can be necessary; the upstream
  qualitative 'redundant tool inputs' criterion has no hidden-state heuristic.

## SFT recipe and serialization

- Select the released **web-agent 7B SFT launcher** (not its code-agent recipe):
  full tuning, LR 1.4e-5, 4 epochs, warmup 0.1, batch 1 per device times 16
  accumulation times 8 devices = 128. The actual script says NNODES=1; its
  contradictory '2 nodes' comment is not used. Unspecified HF defaults are
  linear scheduler, AdamW betas 0.9/0.999, epsilon 1e-8, weight decay 0, clip 1.
  Shared batch shrink is `min(128,max(1,N//8))`, N including resampled exposures.
- Released dataset registration uses default Alpaca instruction/input/output
  fields. `alpaca.json` exports each execution as one chain with all role and
  observation tags. The released masking code masks observation delimiters,
  contents and trailing blank lines, not just their contents.
- Gemma uses its native OFF chat template, not the upstream Qwen template.
  Physical training rows use shared `encode_turn` with exact evaluation
  prefixes: tagged observations are user turns, tagged agent sequences are
  assistant turns. All observation tags/content and prompts have label -100;
  only the assistant output and native turn end are targets. A trajectory is
  still the batch/exposure unit. Turns are weighted by their target-token count
  to preserve the trajectory token mean. The bridge `Action:` outside tool tags
  is never supervised. This chat serialization is the explicit ALFWorld/Gemma
  adaptation of the released single-assistant concatenated output.
- Shared FSDP2 full text-parameter training replaces DeepSpeed/LLaMA-Factory.
  Training RNG seed 0 replaces upstream HF default 42, matching the existing
  ports. FP32 masters, BF16 compute/export, gradient checkpointing and SDPA match the
  other full-tuning ports. Frozen multimodal weights come from the original
  base. No LoRA substitution: upstream's selected recipe is full fine-tuning.
- Context limit 32768, no silent truncation. A too-long training example fails
  preparation with its retained source evidence. Inference remains OFF/1024.

## Agentic RL recipe

- Select the released **web-agent DAPO launcher**: LR 1e-6, train/mini batch 64,
  generation batch 128, eight samples at temperature 1, epochs 2, at most 60
  updates, constant schedule with 3 warmup steps, AdamW weight decay 0.1,
  clip-low/high 0.2/0.28, dual clip 10, PPO epochs 1, no KL or entropy bonus.
  Default GRPO normalization is sample standard deviation + 1e-6. The loss is
  response-token mean, with observations excluded. Both train/generation batch
  sizes use the shared shrink rule over eligible support queries.
- The required **code-agent query filter** is applied once using the frozen CoA
  SFT model: eight independent resets/samples per support task; discard only
  known 8/8 successes. Unknown, missing or replay-inconsistent samples fail
  closed. The web-agent's separate 32-sample no-tool contamination filter is
  deliberately not used (the user explicitly specifies the eight-sample filter).
- Reward is exactly binary real ALFWorld success. No format bonus, verifier
  model, hindsight label, judge, outside TRAIN pool or dev reward is used.
- Dynamic sampling retains only mixed-reward eight-sample groups, accumulating
  until a full shrunken query batch is available. Both all-zero and all-one
  groups are discarded from gradients but retained for accounting. Two shuffled
  passes over eligible queries bound acquisition, as upstream's finite data
  epochs do; an incomplete final mixed batch is retained but not trained.
  The official torch RandomSampler seed 1 and drop_last=True are preserved;
  omitted generation tails are disclosed in schedule.json. Each generation
  batch remains within an epoch and has unique task IDs.
- Each update is a separate rollout/training stage; optimizer moments and FP32
  masters persist through DCP. Behavior log probabilities are recomputed by the
  student before updates; no teacher logits are requested. FSDP BF16 forwards
  correspond to the exported behavior weights. Padded rank slots have zero
  loss. H0 turn-prefix scoring replaces verl's SGLang multiturn tool runtime.
- Upstream response/prompt limits (30k/2k) and 25 tool turns are replaced by the
  shared 32768 context, 1024 per response, 40 H0 turns. There is no upstream
  overlong shaping in the selected launcher, so none is added. Generated
  length-truncated responses are not assigned a fabricated EOS target.
- Save every update as a standalone checkpoint, rather than upstream save_freq=5;
  every published checkpoint is in models.json and independently evaluated.
  Only the final SFT checkpoint is published (intermediate epoch snapshots are
  not used for selection). No dev-based early stopping or checkpoint selection.

## Evaluation, scheduling and accounting

- Student dev140 uses the existing suite with an opt-in CoA policy. It requires
  a CoA-format, TRAIN-only HF/vLLM token equivalence gate before any evaluation
  environment starts. Numerical thresholds match the recent full-checkpoint
  gate: mean absolute difference <=0.05 and >0.5 outliers <=max(2,ceil(5%)).
  Gate inputs are the first eight verified seed turns converted to tool tags;
  this conversion is measurement-only and never training data.
- Multi-agent teacher dev140 is a separate CPU mode, one system run per dev
  task, cap `1 * 40 * 8 * 140` (within the allowed three runs). Its counter,
  raw calls, receipts and disclosure are separate from TRAIN. The capped
  wrapper's explicit evaluation-only mode permits unsealed DEV and rejects
  TRAIN/test; its default generation behavior and identity remain unchanged.
- Stage selector executes exactly one selected stage. Missing prerequisites
  fail before invoking other stages. Teacher/data jobs have no GPU reservation;
  allocation checks reject teacher jobs even if CUDA visibility was hidden.
- Exploration uses serial environment workers per eight-sample group; evaluation
  uses the existing isolated worker pool. This changes throughput, not the
  sampling distribution. vLLM is batch invariant, evaluation greedy seed 0.
- Original teacher usage remains in the verified seed receipts; new teacher
  usage includes failed/reserved attempts, returned model IDs/timestamps and
  reasoning-token counts as accounting only. Student calls, transitions,
  replay work, unsuccessful stages, resampled examples and update exposures
  are retained. Unknown/interrupted costs are not assigned zero verification.


## 2026-10-19 engineering fix: fail loudly on teacher transport

Transport failures (HTTP errors, timeouts, incomplete/unknown responses, halted
collectors and exhausted teacher budgets) raise through the stage boundary.
Failed stage attempts retain sanitized usage, raw returned outputs, counters and
real execution prefixes; they never publish a completion with missing labels.
Completed malformed/invalid outputs keep this method's existing rejection or
masking behavior. `teacher_report.json` counts terminal transport failures,
failed HTTP attempts, parser rejections and method rejections separately. Existing
reports include the same counters; disclosure ledgers retain historical failures.

This is an **engineering fix, no algorithm change**: prompts, labeling/masking,
selection, trusted K, task scope and configured caps are unchanged. See
[`docs/TEACHER_TRANSPORT_FAILURES.md`](../../docs/TEACHER_TRANSPORT_FAILURES.md)
for recovery, non-refundable reservations and the offline regression matrix.
