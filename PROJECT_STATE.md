# PROJECT_STATE — Paper 1: Sufficient Experience for Black-Box Agent Distillation

Governing document: `~/hq/projects/tc-alignment/docs/handoff/2026-10-04_paper1_context_ZH.md` (sections 6, 7, 10–12, 14).
Discord channel: tc (1536171352573349909). Compute: hpg primary (`/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation`), rai for CPU/smoke.
Teacher: gpt-6-luna via the original OpenAI key, no spend cap (user, 2026-10-04). Student: google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7.

## RUNNING JOBS
- **Teacher dev140 done 22:2x CDT: 114/140 = 81.4%** (0 parser errors, 82 inadmissible / 2,862 steps; 12.33M input tokens of which
  11.14M cached, 166.7k output; 183 min, throttled by the old limiter) → reports/runs/teacher-dev140.
- rai GPU1: K=10 s0 done 21:46 CDT (reports/runs/train/seedsft-k10-s0-rai, ckpts 3/5/10/20 passes); K=5 s0 training since 21:46
  (scripts/rai_train_k10_k5_s0.sh, worktree ICML27_run_e70fa64). Queued behind it: **K-ladder eval** scripts/rai_eval_ladder_s0.sh
  (pid 481960, worktree ICML27_run_57ae004): base + k20/k10/k5 s0 × 3/5/10/20 passes (12 adapters) on one server, schema-2 gate,
  32 workers → reports/runs/eval/ladder-s0-rai-dev140, log logs/rai_eval_ladder_s0.log. ~2.5 h (~01:00 CDT).
- Merged today: c96fc4e equivalence gate schema 2 (tolerance = max(0.05, 1.5 × measured HF-vs-vLLM base noise)), 944263c subset
  provenance fix, dc30492 stage 3a (baselines/common + ETO port; GPU/FSDP paths not yet run; hpg needed for full FT).
- Teacher limiter fixed in 57ae004 (chars/4 + 8/message + output cap; 450 RPM / 180k TPM; provider headers logged).
- **First dev140 numbers (rai GPU1, 20:3x CDT):** base 59/140 = 42.1%; Seed-SFT K=20 s0 3 passes 44/140 = 31.4% (paired 41/18/3/78,
  McNemar p = 0.0015). Parser errors 23 → 0, inadmissible 283 → 499. p05 check failed (one 34-token prompt 0.056 > 0.05; pooled 0.031,
  effect 0.84) → launcher stopped before p05/p10/p20. Output reports/runs/eval/k20s0-rai-dev140 (base + p03 valid).
- hpg: nothing running; tunnel down since 17:31 CDT (told the user once).

## Done today (moved from RUNNING JOBS)
- Seeds: s0 20/29 attempts, s1 20/38 (done 19:38 CDT), s2 20/24 → data/verified/seeds_k20_s{0,1,2} with subsets/k5, subsets/k10.
- Seed-SFT K=20 s0 (rai GPU1, episode packing 96531ed): first run OOM at update 120/233 (fragmentation; eager attention on the
  14,005-token episode) → reports/runs/train/seedsft-k20-s0-rai-oom; rerun with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  19:37–20:21 CDT → reports/runs/train/seedsft-k20-s0-rai, 233 updates, checkpoints 3/5/10/20 passes; 3/5/10 adapters byte-identical
  to the OOM run (determinism confirmed). ~10 s/update; episodes are never split so updates carry 512–~1,200 supervised tokens.
- Codex stage 2e committed (d8b0eea): unmerged LoRA serving, HF/vLLM equivalence gate, rai/hpg adapter eval launchers.

## Stage 2c implementation (2026-10-04)

- Teacher: at most 10 HTTP attempts, full-jitter exponential backoff (2 s base,
  60 s cap), provider timing headers and shared cooldowns; file-locked run-wide
  RPM/TPM admission (300 / 400,000 defaults). All attempts remain in usage ledgers.
- Frozen `dev140.json`: all 140 valid_seen instances, 137 task groups. Original
  manifest, split freeze, dev40 and K=10/seed=0 selection unchanged; sealed untouched.
- Generated all K=5/10/20 selections for seeds 0/1/2; tested per-type nesting.
  `collect_seeds.py --subset-k 5 10` collects K=20 once and materializes smaller
  verified views sharing its receipts and trajectories, with full reuse disclosure.
  Zero new live teacher trajectories collected in this stage.
- Full offline suites: 137 passed / 2 skipped without data override; 138 passed /
  1 skipped with real ALFWorld TRAIN replay. Torch module unavailable in project
  venv. No GPU, network, OpenAI calls, commits or pushes. Exact later rai commands
  and implementation limits: [reports/stage2c_validation.md](reports/stage2c_validation.md).

## Stage 2b implementation (2026-10-04)

- Implemented TRAIN-only stratified seed selection and generated `seeds_k10_s0.json`
  (10 initial groups, 1,455 replacement groups; zero newly verified trajectories).
- Added resumable teacher collection with real-environment win checks, all-attempt
  accounting, same-type replacement and separate per-attempt caches.
- Added exact Gemma OFF evaluation-prefix masks (assistant reply + `<turn|>` id 106),
  complete-turn token-budget LoRA SFT, checkpoint profiling, checked merge/reload,
  and B200 train/merged-dev40 launchers. ReAct, thinking OFF, student cap 1,024 fixed.
- Validation: 106 project tests passed including real ALFWorld replay; torch module
  skipped there and four tokenizer/torch tests passed in the separate RAI reference
  venv with CUDA hidden. Real ReAct dev episode mask report: 16 turns / 473 supervised
  tokens; diagnostic only, excluded from training.
- No GPU, network, OpenAI calls, jobs, commits or pushes for Stage 2b. Collection,
  full-model training/merge, GPU profiling and Seed-SFT dev40 remain future runs.
  Files, semantics and exact commands: [reports/stage2b_validation.md](reports/stage2b_validation.md).

## Log (times CDT)
- 2026-10-04 15:00 gpt-6-luna canary: HTTP 200, returned model id `gpt-6-luna` (no dated snapshot), 58 in / 59 out (45 reasoning).
- 2026-10-04 15:29 Stage 1 skeleton (Codex, reviewed): event log, ALFWorld harness (official textworld.start + AlfredDemangler, no expert,
  visible = observation + admissible commands only, audit separate), reset + prefix-replay branching with consistency checks, frozen manifest
  (train 3,553 playable / valid_seen→dev 140 / valid_unseen→sealed 134), black-box teacher client (Responses API, cache, usage ledger).
  54 tests pass incl. real ALFWorld integration. Commit e7d4895 pushed to github.com/XueqiC/ICML27_agent_distillation.
  Project venv `.venv` (uv, Python 3.11): alfworld 0.4.2, textworld 1.7.0, openai 2.8.1, PyYAML, pytest. Data root (rai):
  `ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1`.
- 2026-10-04 15:45 ALFWorld overview figure sent (analysis/fig_alfworld_overview.py → reports/figs/alfworld_overview.png).
- 2026-10-04 15:32 **First full episode with the teacher**: train/pick_and_place_simple-AlarmClock-None-Desk-307/trial_T20190907_072303_146844,
  success in 4 steps (go to sidetable 1 → take alarmclock 1 → go to desk 1 → move alarmclock 1 to desk 1), 12 s, teacher 2,193 in / 201 out.
  Replay matched (4/4 steps). Artifacts: reports/runs/smoke-teacher-1 (gitignored).
  Stage 1 direct-script import issue (`harness/alfworld` shadowed the installed package) is fixed in Stage 2a by renaming it to `harness/alfworld_env`.

- 2026-10-04 Stage 2a local implementation complete: shared ReAct teacher/student format, vLLM student with per-request thinking switch,
  dev40 frozen (40 instances / 40 groups; seed 0), process suite, and HPG B200 SLURM launcher/config. Final checks: 91 passed / 1 skipped
  without data override; 92 passed with real ALFWorld integration. Installed vLLM 0.27.1 emits `message.reasoning`; legacy
  `reasoning_content` and raw Gemma thought channels are also handled. No GPU, network, OpenAI calls, jobs, commits or pushes in this stage.
  Pilot OFF/ON results remain pending. Details and submission commands: [reports/stage2a_validation.md](reports/stage2a_validation.md).

- 2026-10-04 16:11 Stage 2a reviewed and pushed (ec704e4): shared ReAct template/parser (last Action line; parser_error does not step the env;
  inadmissible commands are executed as written and logged), student over vLLM with per-request enable_thinking (vLLM 0.27.1 returns
  `message.reasoning`; thinking never re-enters history), frozen dev40 (valid_seen, 7/7/7/7/6/6 by type, seed 0), suite runner, hpg launcher.
  92 tests pass on rai and on hpg; hpg client venv built with uv (Python 3.11); dev40's 120 game files on hpg match the manifest hashes.
- 2026-10-04 16:14 User: "可以,先开始这个实验吧" → pilot submitted (see RUNNING JOBS).

- 2026-10-04 16:40 **Thinking pilot done** (reports/pilot_thinking_0410_ZH.md): OFF 19/40, ON 21/40; paired 17 both / 2 only-off / 4 only-on /
  17 neither (McNemar p = 0.69); ON costs ~5x tokens, 4.5x time, 31 vs 9 parser errors. By the pre-agreed rule (not clearly better) →
  **ReAct + enable_thinking OFF for all methods**. Harness issue: with thinking OFF the model sometimes enters the thought channel anyway and
  hits the 4,096 cap with an empty reply (all 9 OFF parser errors, all look_at_obj). Proposed: keep counting as parser error, cap per-step at
  1,024 tokens, report the rate. Thinking text holds explicit multi-step plans (two_obj 2/6 → 5/6) — candidate for mechanism 2 later.

- 2026-10-04 16:43 User approved the harness rule → student cap 1,024 tokens, teacher output cap 2,048 (0ba4d07).
- 2026-10-04 16:47 Teacher dev40 (rai, 8 workers) **invalid**: 151 HTTP 429 with 0.5 s retries → 27/40 episodes aborted (TeacherCallError);
  the 13 completed episodes all succeeded (9.2 steps). Fix pending: Retry-After-aware exponential backoff + global teacher rate limiter, then rerun.
- 2026-10-04 16:48 Codex stage 2b started (seed selection/collection, SFT dataset + masks, LoRA trainer, merge check, hpg scripts).

## Open decisions (asked the user)
0. **Baseline porting (asked 19:40 CDT 10-04, reports/baseline_port_specs_1004.md attached).** Decided by me (engineering): port algorithms
   into our stack; one harness for all stages (X3); each repo's own adaptation type (ETO/IPR/AgentGym full FT, LEAP/Kang LoRA);
   batch shrink b' = min(b_orig, max(1, floor(N/8))) with lr/epochs/scheduler/beta unchanged; dev selection over iterations, official
   final also reported; fix + disclose the IPR MC-sampler bug (WebShop file copy); Kang = SFT + SAG faithful core, FTP emulated variant.
   Asked the user: A) WKM incomplete code → drop from main table? B) LEAP faithful (privileged handcoded-expert summaries at training
   time, flagged) + 0%-privileged ablation? C) AgentEvol/LEAP explore all 3,553 train games as official, with env/teacher counts per row?

1. Interaction format: **ReAct accepted** (user, 2026-10-04 15:39). Gemma `enable_thinking`: the user asked why off. I explained
   (teacher reasoning is not returned, so the only supervisable reasoning is its visible Thought; the Gemma template strips earlier-turn
   thinking via `strip_thinking`, while ReAct Thought stays in history; one fixed mode for all methods). I proposed a pilot: base student,
   ReAct, 40 fixed dev games, thinking off vs on, then fix the mode. User replied by quoting the plan (15:46); taken as approval
   and said so on Discord. Codex stage 2a (ReAct template/parser, student policy over vLLM with enable_thinking switch, dev40,
   suite runner, hpg slurm script, harness/alfworld rename) started 15:48.
2. K=10 seed selection rule (re-asked 16:46 with specifics): 10 distinct train task groups, seeded type order, first 4 types x2 and last 2 x1;
   sampling seeds 0/1/2; up to 3 teacher attempts per task, keep the first real success, replace with the next same-type candidate after 3 failures.
   Awaiting reply.

## Current plan (user 2026-10-04 16:52: multiple K; table + line plot of teacher / base / distilled students incl. SmartAD's three
## and other feasible SOTA; then systematic analysis separating failure mechanism 1 (knowledge in experience but not absorbed/used)
## from mechanism 2 (student needs more explicit decision information: better teaching or legal re-observation/querying))
- Table: K ∈ {5, 10, 20}, 1 support seed first (3 for key cells later); rows teacher, base, Seed-SFT, SmartAD (10 candidates/task, NLL
  selection, segment weights 1/1.5/2), SAD, Kang (first-thought prefix) — interpreted "SmartAD 那三个" as SmartAD/SAD/Kang (told the user);
  then grounded Augmented-SFT, DAgger-hard, DART-SD; logprob-dependent OPD variants listed as not runnable black-box. Eval on full dev
  (valid_seen 140); valid_unseen sealed.
- Mechanism analysis: auto-classify failures; branch-replay from the first critical error with one intervention each: (1) restate facts
  already in history, (2) real legal re-observation result, (3) teacher Thought/plan without action, (4) teacher next action (upper bound);
  offline knowledge probes; rule out format/dose/coverage/teacher-error confounds.
- **User rule (16:53): every agent baseline must follow its official repo; no released code → do not run ("我们不编").**
  Code check 16:57: SmartAD (no code on ACL page/paper), SAD (no official code; only a third-party repo), DART-SD (no code),
  GAD black-box OPD (no code) → not run. Kang AD has an official repo (Nardien/agent-distillation) but only QA/math with retrieval+code
  tools, FTP generation not fully released → not faithful on ALFWorld. Official ALFWorld code: ETO (Yifan-Song793/ETO, ACL'24) and
  IPR (WeiminXiong/IPR, EMNLP'24), both FastChat-based SFT+DPO needing expert trajectories (= teacher seeds here). Asked the user:
  (1) ETO + IPR as ALFWorld SOTA baselines? (2) add a Kang-supported env (HotpotQA) to compare Kang faithfully? (3) Seed-SFT /
  Augmented-SFT / DAgger as protocol controls (not SOTA)?
- User 17:01: two baselines are not enough — find more LLM→SLM agent distillation baselines (also non-agent methods using ReAct-like
  distillation); **adapt Kang to ALFWorld and compare** (reuse repo parts, log every adaptation); controls OK. A background survey of
  candidates with official code is running (ETO, IPR, AgentTuning, Agent-FLAN, FireAct, NAT, AgentGym/AgentEvol, WKM, KnowAgent, MPO,
  STeCa, QLASS, Lumos, AutoAct, Distilling step-by-step, TinyAgent, 2025–2026 methods).
- 17:12 Baseline survey done (reports/baseline_survey_1004_ZH.md): usable with official ALFWorld code — ETO, IPR, LEAP (teacher sees
  privileged state; flag), AgentGym BC/AgentEvol, WKM, optional QLASS; Kang adapted. Excluded: SmartAD/SAD/AgentRefine/ATLaS (no code),
  AgentTuning/Agent-FLAN (no training code), SAD + OPD variants (need teacher logits), KnowSelf (DeepSeek in pipeline). All 2024 repos need
  dependency/template/mask ports for Gemma 4 on B200 (log as engineering). Proposed set to the user: ETO, IPR, LEAP, AgentEvol, WKM,
  Kang(adapted) + controls Seed-SFT/Augmented-SFT/DAgger — awaiting confirmation.
- 17:15 Stage 2b reviewed and pushed (6a2d177): seed selection/collection with real-win receipts, Gemma 4 ReAct SFT dataset (assistant-only,
  token 106), LoRA trainer with supervised-token exposure, merge consistency check, hpg train/eval scripts. 106 tests pass.
- 17:16 Codex stage 2c: teacher 429 backoff + cross-process limiter, frozen dev140, nested K=5⊂10⊂20 seeds for seeds 0/1/2.
- 17:28 Stage 2c pushed (8b53713): teacher backoff + cross-process limiter, dev140 (140 instances / 137 groups), nested seeds K=5/10/20 × s0–2.
- 17:31 **Tunnel down** (ssh hpg: localhost:2222 refused) — told the user once to restart autossh on the Mac; waiting. Nothing running on hpg.
- Supervised tokens per pass (seed 0): K=5 2,146 (79 turns), K=10 5,311 (200), K=20 10,407 (382).
- ETA told the user: table core ~1–1.5 days, + Augmented-SFT/DAgger + plot ~1 day, first mechanism round +2–3 days.

## Next
Run the enabled base dev40 thinking OFF/ON pilot on HPG, inspect paired outcomes and parser/service/length failures, then fix thinking mode.
Remaining Stage 2: student serving/eval (vLLM, merged LoRA), Seed-SFT trainer with assistant-only loss masks, K=10 seed collection, base/teacher/Seed-SFT
smoke on a fixed dev subset, hpg environment + profiling report.
