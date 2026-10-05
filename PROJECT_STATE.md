# PROJECT_STATE — Paper 1: Sufficient Experience for Black-Box Agent Distillation

Governing document: `~/hq/projects/tc-alignment/docs/handoff/2026-10-04_paper1_context_ZH.md` (sections 6, 7, 10–12, 14).
Discord channel: tc (1536171352573349909). Compute: hpg primary (`/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation`), rai for CPU/smoke.
Teacher: gpt-6-luna via the original OpenAI key, no spend cap (user, 2026-10-04). Student: google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7.

## RUNNING JOBS
- **hpg (tree ICML27_runs/74cdb97, two-pass DPO + bf16 reference)**: 44826292 ETO K20 s0 (2 GPUs, 200G) → baselines/eto-k20s0-tp4; 44826293 IPR
  K20 s0 (2 GPUs, 200G, 24 h) → baselines/ipr-k20s0; 44826360 LEAP privileged and 44826361 LEAP 0%-privileged K20 s0 (1 GPU each) →
  baselines/leap-{privileged,zero}-k20s0. ETO/IPR pending on dept QOSGrpMemLimit.
- **OpenAI key on hpg** (user approved 11:28 CDT): only the OPENAI_API_KEY line, /blue/fsu-compsci-dept/xc25.fsu/hq/secrets/openai.env
  (600, dir 700), used via SED_SECRETS_FILE by LEAP jobs only. Compute nodes reach api.openai.com (HTTP 401 without key).
- AgentGym K20 s0 DONE: BC 59, AgentEvol iter-1/2/3 = 64/62/66 (base 62).
- rai GPU1: mechanism run on k5s0-t400k (pid 1513532) — FCE probing. rai GPU3: free after probes run 1.
- Codex (pid 1609910): MECHANISM_PLAN amendment 5 (procedural coverage) in analysis/mechanism.
- **User 11:30 CDT: widen K to the whole few-shot range** → proposed K ∈ {1, 2, 5, 10, 20, 50, 100} (nested; K=1/2 subsets of K=5;
  K=50/100 by extending the K=20 collections, ~350 more teacher attempts, ~8–10 h TPM-bound). Codex stage 2f (pid 1633979): K=100
  selections, --extend-from collector, subset views with exact K=20/10/5 identity checks, dose grid v2, scripts/rai_collect_k100.sh.
- User 11:28 CDT: plots must show each baseline as its own line (distinct colours), separate from Seed-SFT/teacher/base.
- Interim report sent 10:0x CDT: reports/interim_1005_ZH.md (+ seedsft_sweep.png).
- Codex 4f (pid 1474865): plan §4 — needed-step coverage from the shortest teacher takeover at t*, absorption probe on the seed states that
  demonstrated each needed pattern (M1a not absorbed / M1b absorbed-not-invoked), training-set fit; scripts/rai_mechanism_probes.sh.
- Copying best hpg adapters (k10s0-t400k, k5s1-t400k, k5s0-t400k, k20s0-p40) to rai reports/runs/train/hpg-adapters for the next mechanism runs.
- rai GPU1: BI dev140 of hpg adapters (scripts/rai_bi_dev140_hpg_adapters.sh, pid 1482167): k5s0-t400k **74/140** (hpg 72) → gap set G=43, R=4;
  k10s0-t400k 96/140 (hpg 95). **Mechanism run on k5s0-t400k RUNNING since 10:2x CDT** (scripts/rai_mech_k5s0.sh, pid 1513532, tree 70314a3, caps 700/episode, 30k/run)
  → reports/runs/mechanism/k5s0-t400k-bi/run; ~12 h (TPM-bound FCE).
- hpg 44815253: AgentGym evaluate — **bc 59/140, iter-1 64/140**, iter-2/3 pending (tunnel down at 11:3x).
- rai GPU3 (A100): plan §4 coverage + absorption probes on run2's k20s0-p20 (scripts/rai_mechanism_probes.sh, pid 1589071, tree 2d47700)
  → reports/runs/mechanism/k20s0-p20-bi/probes.
- Committed: 6cac4fc two-pass per-turn DPO/IPR backward (fixes the turn-packed OOM); 2d47700 stage 4f + MECHANISM_PLAN amendment 4.
  Next on hpg when back: resubmit ETO K20 s0 (4 GPUs, two-pass) and IPR K20 s0.

- ETO K=20 s0 tp3 (44811450, 4×B200, bf16 ref) FAILED at DPO-1 again: OOM at 169.7 GB (static ~54 GB) — the turn-packed preference graph holds
  all turns of both continuations. P=4 this time (SFT model solved 16/20 seed tasks; the 2-GPU run's solved 9/20). Codex pid 1480242: exact
  two-pass per-turn backward for DPO and IPR. Then resubmit ETO; IPR after ETO's DPO works.
- AgentGym K=20 s0 training DONE (44806889, 1:07; F_m 8/12/7). **hpg 44815253** (dept, 1×B200): AgentGym evaluate mode — dev140 of bc / iter-1..3, one model per server (canonical).
- **Mechanism run2 DONE (10:2x CDT)** on k20s0-p20 (rai-trained, BI, G=24): I0 0/24, I1 3, I2 2, **I3 advice 11**, I4 action 9, **I5 seed
  example 8**; cells M2-only 7, M1-only 4 (coverage pending), both 4, neither 9. Heat → M1 (3/6 via I5 only); look_at_obj/two_obj → M2;
  cool/far-from-goal → unresolved. Readout: reports/runs/mechanism/k20s0-p20-bi/run2/launches/attempt-000/readout. Next: plan §4 coverage +
  knowledge probes; repeat on K5/K10 students (larger gaps) and other support seeds.
- Contamination tests done: no server-state contamination (k10s2 check fails identically alone → OOD check prompts); but **BI determinism
  needs an identical server config**: k20s0-p40 alone (max-loras 1) 112 vs 115 on the 6-adapter sweep server, 70/140 identical
  trajectories, 7 discordant. → Paper/sealed evals: one adapter per server. Dev sweep numbers stand with ±7/140 cross-config reference.
- Kang K=20 s0 turn-packed (official recipe): **SAG 63/140, greedy 66/140** (0 parser errors) ≈ base 62 — official dose ≈ 2 passes.
- **hpg 44806889** (dept, 2×B200, 240G, tree 97eb54b): AgentGym BC + AgentEvol K=20 s0 (4 iterations, explore only the K tasks) →
  reports/runs/baselines/agentgym-k20s0. Pending.
- SAG-on-Seed-SFT (44806873): k20s0-p40 + SAG **115/140 = greedy** (SAG adds nothing). k10s2-t200k / k5s1-t400k failed the check (mean
  0.17 / 0.24 nats) with K20-s0 check prompts (out of their training distribution) → **hpg 44809012** (k20s0-p40 alone, fresh server: compare
  trajectories with the sweep = server-state contamination test) and **hpg 44809013** (k10s2-t200k alone, same s0 prompts: prompt-OOD test).
  vLLM prefix-cache hashes include lora_name (v1/core/kv_cache_utils.py:517-529), so cache sharing across adapters is unlikely.
- ETO K=20 s0 (44806752, group 2×B200): SFT + export OK; explore-1 → 11 pairs (9 ties / 11 lose); **DPO-1 OOM** (fp32 policy + fp32 reference on 2 GPUs). Codex pid 1329456: bf16-stored reference (bitwise-identical); then resubmit ETO (and IPR) on 4 GPUs. AgentGym K=20 s0 (44806889, dept): data-0/train-0 running.
- Schema-5 re-eval 44802354 done: k10s2 p30 86 / p40 98 / t200k 104 / t400k 103; k20s2 p30 92 / p40 98 / t400k 98; k5 p03 still fail (b).
- rai GPU1: mechanism run2 (pid 899949) — FCE probing, 660 probe episodes at 07:18 CDT.
- hpg tunnel dropped 06:43–07:1x CDT (told the user once); back now.
- **Sweep analysis committed** (reports/figs/seedsft-sweep/: table, plot): by supervised tokens all K coincide below ~100k (dip below base
  at 30–60k tokens, min ≈ 33%), then K separates: K20 → ~80% at 400k (= teacher), K5/K10 plateau ~60% with large seed spread. Sent to user
  05:1x CDT (footer first sent stale "0R+3P", corrected to the script's "2R+2P").
- Codex 4e (pid 1120214): analysis/traits.py (frozen dev140 trait table per addendum A) + analysis/stratified.py (per-bucket rows, Wilson,
  McNemar) + sanity readout on teacher/base/4 Seed-SFT cells. Next Codex: allow --gpus 2/4/8 in ETO/IPR/AgentGym pipelines (world-size-
  independent math) so full-FT baselines can run on 2 group GPUs.
- hpg 44798587 robustness: both merges FAILED merge_lora's check (merged-bf16 vs PEFT all-vocab logits mean |Δ| 1.13 / 1.41) → unmerged vLLM
  LoRA (mean 0.04 nats) is the most faithful bf16 serving path; vLLM supports LoRA dtype only fp16/bf16 (no fp32). Plan: gate amendment 4
  (outlier tokens ≤ 5% instead of 2×base+2), evaluate the 8 failed high-dose cells with a flag, fp16-LoRA behavioural sensitivity on 2 ckpts.
- Kang K=20 s0 (first live baseline): SAG 65/140, greedy 65/140, but 140 parser errors = step 0 of every episode → port bug: episode
  packing supervises non-final turns after '<|turn>model\n' while eval generates after the empty thought marker; harmless at lr 1e-5
  (Seed-SFT), breaks step 0 at Kang's lr 2e-4. Codex (pid 1168415, main tree): turn packing for baseline SFT (exact eval prefix per turn),
  default for Kang/AgentGym/ETO/IPR SFT stages; check DPO/IPR continuation scoring and LEAP. Then re-run Kang. Result invalid until then.
- Stage 4e committed (0ee65a1): data/manifest/dev140_traits.json + analysis/stratified.py; first readout reports/figs/stratified-seedsft
  (e.g. look_at_obj: k20s0-p40 8/13 vs teacher 13/13 and k10s0-t400k 13/13 — seed coverage of task types).
- Codex branch gate5-gpus-loradtype (pid 1124640, worktree ICML27_gate5): gate schema 5 (outliers ≤ max(2×base+2, 5% of tokens)),
  --gpus 2/4/8 for ETO/IPR/AgentGym, LORA_DTYPE option (fp16 sensitivity).
- Dept QOS is memory-capped (~980/1000G in use by others at 05:1x CDT); group account had 2 B200 free → 1-GPU jobs go there.
- **hpg 44777572** (dept, 4×B200, request shrunk to 16 CPU / 300G at 00:4x CDT after QOSGrpCpuLimit; same tree): ETO K=20 s0 stage `sft` only (full-FT FSDP2 smoke) → reports/runs/baselines/eto-k20s0;
  VLLM_BATCH_INVARIANT=1 exported (not yet recorded in e1bad3f provenance). Pending.
- rai BI dev140 done: base 60/140 = 42.9%, k20s0-p20 96/140 = 68.6% → reports/runs/eval/bi-dev140-rai. Gap set G = 24, R = 7.
- **rai GPU1: mechanism run2** (pid 899949, worktree ICML27_run_7cbfb58, 01:40 CDT 10-05): k20s0-p20 (BI), G = 24, amendment 3
  (interventions at closest approach t*), WORKERS=8, teacher cache seeded from the stopped serial run (5 FCE tasks done there),
  caps 700/episode, 15,000/run → reports/runs/mechanism/k20s0-p20-bi/run2, log logs/mechanism_k20s0_p20_bi_run2.log.
  Serial run1 (old FCE definition) stopped 01:39; kept as the cache source. Example finding (task 1): D = 1 from t = 24, student loops "look".
- Mechanism run2 rate: TPM-bound (≈27 teacher calls/min; mean limiter wait 20 s; 4.9k input + 2,048 reserved output per call vs the
  provider's 200k TPM). FCE ETA ~08:30 CDT, then I0–I5 (student-side) — pre-registered probe grid kept.
- Baseline ports committed: 3a ETO (dc30492), 3b AgentGym BC/AgentEvol + Kang/SAG (50aebcc), 3c IPR + deterministic SAG/MC seeds
  (5e14289; expert MC values from the repo's commented-out official block — tell the user; WebShop-cut sampler bug fixed). Codex 3d
  (pid 1034548): LEAP faithful (privileged handcoded-expert summaries, K tasks only) + 0%-privileged ablation; trains on all K (85/15
  split only serves eval-loss selection that cannot trigger at this dose — registered). Codex 3c's first attempt hit "model at capacity".
- hpg tunnel was down ~01:00–04:50 CDT 10-05; back now.
- hpg data: data/verified synced (checksums equal). hpg git cannot fetch GitHub (no credentials) → code goes by `git archive <sha> |
  ssh hpg tar -x`, one tree per commit under /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/<sha> with .venv symlinked to the base tree.
- **Protocol: batch-invariant serving (VLLM_BATCH_INVARIANT=1) for every paper evaluation** — base 40/40 and LoRA 40/40 identical
  trajectories across server instances (rai, 23:12–23:51 CDT); ~2.8x slower for base. Earlier non-BI numbers are preliminary.
- Baseline decisions (user 23:43 CDT): WKM/STeCa out (incomplete code); LEAP faithful + 0%-privileged ablation; exploration only on
  the K tasks for every method. Final set: ETO, IPR, AgentGym BC/AgentEvol, LEAP(+ablation), Kang (adapted) + Seed-SFT/teacher/base.
- User goal restated (23:44): honestly measure what SOTA baselines and our student achieve, and which environment traits amplify or
  shrink each gap → MECHANISM_PLAN addendum A (trait stratification, pre-registered).

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
- **Asked 07:5x CDT 10-05: OpenAI key on hpg for LEAP?** LEAP needs live teacher calls; rai disk (~97 GB free) cannot hold its 4 × 24 GB
  merged checkpoints → must run on hpg. Proposed: key in a /blue file, chmod 600, LEAP only. Fallback: split teacher stages (rai) and GPU
  stages (hpg). Not done until the user answers.
- **Decided by the user 2026-10-04 23:43 CDT** (answers to A/B/C): the few-shot black-box principle never changes; an official repo
  with incomplete code is treated as nonexistent (→ WKM and STeCa dropped); B approved (LEAP faithful with privileged expert
  summaries at training time, flagged, plus the 0%-privileged ablation); C: exploration only on the K training tasks for every
  method (AgentEvol, LEAP, …) — if a method fails under that, that is its mechanism's limitation.
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
