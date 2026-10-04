# Research contract

Read `/home/xueqi/hq/projects/tc-alignment/docs/handoff/2026-10-04_paper1_context_ZH.md`
before changing the research protocol. Sections 6, 7, 10, 11, 12, and 14
govern this repository. Later explicit user instructions take precedence.

- K counts trusted, actually executed seed trajectories, **not tasks** or
  all experience available during training. Report unique instances and
  task groups separately. Rephrasing and retrying do not create new tasks.
- New tasks, teacher outputs, and real interactions are allowed. Log and
  disclose every expansion source, parent seed, raw output, rejection,
  failed attempt, token count, transition, and training exposure.
- The teacher is black box: sampled text/actions only. Never request
  logprobs, hidden states, parameters, gradients, or hidden reasoning.
  Usage counts of reasoning tokens are accounting metadata only.
- No Chinese model checkpoints anywhere, including indirect reward,
  teacher, generator, or verifier models. Do not infer origin from authors.
  Do not silently substitute the configured teacher or student.
- Post-training only. Do not pretrain a new foundation model.
- Freeze splits before experiments. Test/sealed instances and states must
  never drive generation, failure attribution, prompt tuning, or checkpoint
  selection. Evaluation never writes training data or performs updates.
- Hidden/audit fields never enter prompts. Training loaders read an explicit
  whitelist. Never serialize arbitrary environment info or full event rows
  into model inputs. Oracle fields include `extra.expert_plan`, PDDL facts,
  `policy_commands`, internal state, and hidden success labels.
- All policies share admissible commands, visible fields, action parsing,
  and termination semantics. State branching requires actual reset and
  prefix replay with consistency checks. Never fake a restored branch.
- Every run writes provenance: effective config, code/content hashes,
  dependency versions, environment seed/version, data manifest/split freeze,
  student revision, teacher alias and returned model ID/timestamp.
- Real execution determines success; teacher claims are not ground truth.
  Preserve unknown verification and negative results. Do not claim Stage 1
  smoke runs establish a scientific finding or a faithful baseline reproduction.
- First discovery baselines: base, teacher, Seed-SFT, grounded Augmented-SFT,
  DAgger-hard. Add SmartAD and DART-SD before related mechanism claims;
  disclose adaptations and keep their distinct labeling/masking protocols.
- Tests run with `CUDA_VISIBLE_DEVICES=''`, no network, and no OpenAI API.
  Use fake clients and fake environments for unit tests. Never print or
  persist secrets, complete process environments, or raw SDK exceptions.
- Do not copy old repositories wholesale. Attribute any small port and
  preserve this repository's independent protocol. No commits or pushes
  unless the user authorizes them.
