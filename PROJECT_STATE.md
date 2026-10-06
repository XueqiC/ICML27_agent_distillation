# PROJECT_STATE — Paper 1: Sufficient Experience for Black-Box Agent Distillation

Governing document: `~/hq/projects/tc-alignment/docs/handoff/2026-10-04_paper1_context_ZH.md` (sections 6, 7, 10–12, 14).
Discord channel: tc (1536171352573349909). Compute: hpg primary (`/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation`), rai for CPU/smoke.
Teacher: gpt-6-luna via the original OpenAI key, no spend cap (user, 2026-10-04). Student: google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7.

## RUNNING JOBS
- **hpg tree d9e151b** (13:3x CDT, scripts/hpg_baseline_cells.slurm = one K×support-seed cell per array task, official recipe + the paper's
  dev140 BI eval; collections on the nested ladder: K5/10 = seeds_k20_s*/subsets, K1/2/50 = seeds_k100_s*/subsets):
  44837378 kang-cells-g (group, %1: k5s0 k10s0 k5s1 k10s1), 44837379 kang-cells-d (dept, %2: k5s2 k10s2 k20s1 k20s2),
  44837380 agentgym-cells (dept, 2×B200 160G, %1: k5s0 k10s0 k5s1 k10s1 k5s2 k10s2 k20s1 k20s2). Outputs
  $B/reports/runs/baselines/{kang-<cell>-tp,agentgym-<cell>}, Kang SAG eval $B/reports/runs/eval/baselines/kang-<cell>-tp-sag.
  Memory right-sized from sacct MaxRSS: Kang train 2.9G / SAG eval 9.7G; AgentGym 2-GPU train 137G.
- **hpg tree 57c9365**: K=1/2 Seed-SFT — 44842264 train array (6 runs k1s0,k2s0,k1s1,k2s1,k1s2,k2s2, %3; grid
  configs/seedsft_dose_grid_v2_k12_ready.json prepared on hpg, committed), 44842265 BI dev140 eval array (aftercorr) →
  $B/reports/runs/train/seedsft-k{1,2}s*-hpg, $B/reports/runs/eval/seedsft-sweep-bi-v2k12 (add as --sweep-root/--grid in main_results).
- **Kang cells DONE** (official + SAG): K5 57/98/68 (mean 74.3), K10 80/71/85 (78.7), K20 63/75/110 (82.7) (base 62). 10–20 updates per cell (lr 2e-4 r64).
  Per-type table reports/figs/coverage-by-type/coverage_by_type.md: gains come from heat/cool (base 1/16, 1/25); K5 s1 omits only the
  easy type (best cell for both methods), K5 s0 omits cool (cool stays at base); coverage necessary not sufficient (s0 heat seed does
  not transfer). Kang K20 s0/s1 collapse on heat/cool/two with s2's coverage → suspected recipe instability.
  **Asked user 14:48 CDT**: re-run Kang K20 s0 with 1–2 more training seeds to separate training instability from seed content?
- **hpg 18:1x CDT**: K=50/100 s0 Seed-SFT 44871876 (train, tree b521a4e, grid configs/seedsft_dose_grid_v2_k50_100_s0_ready.json:
  22.9k / 48.3k tokens per pass, to 916k / 1.93M) + 44871877 eval (aftercorr) → eval/seedsft-sweep-bi-v2k50.
  Baseline cells k1s0 k2s0 k50s0 k100s0 (tree c79f6e8, K1/2 on K100 views): 44871885 Kang (%2), 44871886 AgentGym (%1).
  K=1/2 s1/s2 baseline cells after K100 s1/s2 (and after moving s1/s2 K20 views).
- **K=100 s1 COMPLETE** 19:35 CDT (100 trusted / 115 tasks / 160 attempts); s2 running (~23:00 CDT). hpg s1 K20 views moved to _trash
  (identical to K100 s1 views, checked). seeds_k100_s1 synced. Tree 3472df2: s1 grid being prepared; s1 baseline cells
  k1s1 k2s1 k50s1 k100s1: 44879553 Kang (%2), 44879554 AgentGym (%1).
  K=50/100 s1 Seed-SFT: 44880903 train + 44880904 eval (grid 25.4k / 52.8k tokens per pass) → eval/seedsft-sweep-bi-v2k50.
- **K=100 collection ALL COMPLETE** 22:49 CDT (s2: 100 trusted / 110 tasks / 142 attempts); rai collector done. All hpg K20 k1/k2 views
  now in $B/_trash/k20_views_k1k2_1005/ (identical to K100 views for s0/s1/s2). K=1/2 grid re-pointed:
  configs/seedsft_dose_grid_v2_k12_ready_k100views.json. Tree 3b133d5: k2s2 eval re-run 44893700 (→ eval/seedsft-sweep-bi-v2k12-r2),
  s2 baseline cells k1s2 k2s2 k50s2 k100s2: 44893701 Kang, 44893702 AgentGym; K50/100 s2 Seed-SFT 44894413 train + 44894414 eval
  (24.5k / 49.2k tokens per pass). **Every K × seed cell for Seed-SFT/Kang/AgentGym is now submitted.**
- **hpg tree 74cdb97**: 44826292 ETO K20 s0, 44826293 IPR K20 s0 (dept 2×B200 200G, pending QOSGrpMemLimit — dept mem 980/1000G used);
  44836718/44836719 LEAP privileged/zero K20 s0 FAILED 13:4x CDT at merge-0 (data-0, train-0 completed): our merge check
  (allclose atol .125 / rtol .01 on all vocab logits) rejects the bf16 merge — mean |dlogit| 0.083 / 0.145, max 0.56 / 0.81, argmax
  equal. Upstream merges without any check (fp32 load + merge_and_unload; serving in bf16 gives the same bf16(W+BA)), so the gate is
  stricter than the official recipe → fix: LEAP records the report and accepts finite + argmax-equal + reload-identical merges
  (disclosed). Fixed in 7e26ca7 (LEAP-only policy; strict result kept as diagnostic). Resume refused by design (code hashes bound),
  so **resubmitted fresh: 44843960 privileged / 44843961 zero** (group, 64G, tree 7e26ca7) → $B/reports/runs/baselines/leap-{privileged,zero}-k20s0-m2.
  44843960 passed merge-0 under the new policy but FAILED at explore-1: vLLM cannot load the merged checkpoint (no processor_config.json;
  merge_adapter saves only weights+tokenizer, the full-FT export copies base assets). 44843961 cancelled before start (same failure).
  Fixed in 442d142 (copy_base_assets shared with full-FT export). **Resubmitted 44850280 privileged / 44850281 zero** (group, tree 442d142)
  → $B/reports/runs/baselines/leap-{privileged,zero}-k20s0-m3.
  m3 privileged ran through iteration 1 (40 min) but **all 16 privileged-state extractions got HTTP 400**: gpt-6-luna rejects
  temperature AND top_p (error type invalid_request_error, param set, code null — probed 16:2x CDT); our fallback only fires on
  code unsupported_parameter/value → never switched to default sampling → 0 states → 12 failed rollouts excluded → stopped_empty_min.
  0% variant cancelled (would hit the same 400 at correct-1). Fixed in e36fe93 (code-null shape; live probe from rai sets the flag).
  **Resubmitted 44857832 privileged / 44857833 zero** (group, tree e36fe93) → $B/reports/runs/baselines/leap-{privileged,zero}-k20s0-m4.
  m4 running 17:0x CDT; privileged extraction 16/16 OK (4 tasks have no handcoded expert), sampling fell back to configured default.
  m4 privileged: iteration 1 worked (11 failed rollouts → 7 corrections, 3 no-state, 1 rejected; 280 examples) then FAILED in train-1:
  validate_initialization uses the full-FT validator (needs non_text_source) on the merged checkpoint. 0% m4 (44857833) cancelled
  at correct-1 (same train-1 path). Codex: fix + one real end-to-end CPU chain test (tiny Gemma-4) merge→train-1→merge-1→serve assets.
  Fixed in 7f45f64 (+ eval preflight same gap; tests/test_leap_chain_torch.py runs the real chain on CPU; remaining unexercised paths
  listed in reports/leap_chain_validation.json). **Resubmitted 44872102 privileged / 44872103 zero** (group, tree 7f45f64) → -m5.
  **m5 privileged TRAINING COMPLETE** (23:0x CDT): corrections per iteration 8 / 4 / 4 (rejected 0 / 5 / 5; no-state 4 / 2 / 3; failed
  student rollouts on the 20 K tasks 12 / 11 / 12); official min-subsampling → N = 640 / 480 / 640 examples. Eval 44894390 (group, same
  tree, LEAP_MODE=evaluate) → iter-0..3 on dev140.
  **LEAP privileged K20 s0 dev140: iter-0 58, iter-1 62, iter-2 57, iter-3 60** (base 62) — all within noise of base.
  LEAP 0% K20 s0 training complete (corrections 11 / 7 / 8, rejected 2 / 5 / 7, N 764 / 840 / 1120); eval 44907658 submitted.
- **K=100 s0 collection COMPLETE** (21:17Z = 16:17 CDT: 100 trusted / 117 tasks / 156 attempts, 3 h 37 min; k5/k10/k20 views identical
  to the K=20 ones). s1 running since 16:17 CDT; at this pace s1, s2 ≈ 3.5 h each → ~23:30 CDT. Synced seeds_k100_s0 to hpg.
  Tree b521a4e: K=50/100 s0 grid being prepared on hpg (configs/seedsft_dose_grid_v2_k50_100_s0.json); baseline cells for
  k1s0 k2s0 k1s1 k2s1 k1s2 k2s2 k50s0 k100s0: 44857766 Kang, 44857767 AgentGym — **CANCELLED** (pending) 16:5x CDT: the K=1/2 views I
  wrote into hpg seeds_k20_s*/subsets changed the K=20 tree that seeds_k100_s0 binds (tree_sha256) → every K100 load fails
  ("Extension source identity/evidence changed"); grid prep failed the same way. K100's own k1/k2 views are identical (s0 checked:
  verified, attempts, selection hash). Plan: wait for the K=1/2 Seed-SFT evals (they read the K20-derived views), then move
  hpg seeds_k20_s*/subsets/k{1,2} to $B/_trash/k20_views_k1k2_1005/, then K50/100 s0 grid + Seed-SFT + baseline cells
  (wrapper bfc4cd7: K1/2 → K100 views). K1/K2 s1/s2 baselines after the s1/s2 collections.
  17:5x CDT: K=1/2 Seed-SFT evals done except k2s2 (aborted after p20 on 1 error episode → re-run 44869648 →
  eval/seedsft-sweep-bi-v2k12-r2). s0 views moved to $B/_trash/k20_views_k1k2_1005/seeds_k20_s0 (s1/s2 views stay until the k2s2
  re-run is done and before K100 s1/s2 reach hpg). K50/100 s0 grid preparation restarted on hpg (tree b521a4e).
  **K=1/2 Seed-SFT results** (hpg BI; p03/p05 gate-fail = effect within serving noise): k1s0 peak 72 (t200k), k2s0 71 (t200k),
  k1s1 ≤60, **k2s1 85–87 (t100k–t400k)**, k1s2 ≤63, k2s2 pending. k2s1 = clean Fork + cool Bread: adding the cool seed to k1s1 gives
  +28 → hard-type coverage again. ETO/IPR pending jobs lowered to 160G (scontrol, MB units) 15:2x CDT.
- **K=1/2 views** materialized on hpg (13:5x CDT) under $B/data/verified/seeds_k20_s{0,1,2}/subsets/k{1,2} (materialize_subsets, nested in
  k5 checked). Not yet on rai (the K100 collector reads seeds_k20_s* — add after it finishes). Codex (pid 1881152): partial v2 dose grid
  (configs/seedsft_dose_grid_v2_k12.json) so K=1/2 Seed-SFT can run before K=100 finishes.
- **rai CPU + OpenAI: K=100 seed collection** (scripts/rai_collect_k100.sh, pid 1731893, tree 0834361, started 12:40 CDT), sequential
  s0→s1→s2, each extends seeds_k20_s* by 80 tasks (+ subsets k1/2/5/10/20/50). s0 at 13:05 CDT: 29 tasks / 38 attempts / 24 wins in 24 min
  → ~65 min per seed. 15:47 CDT: s0 103 tasks / 137 attempts (≈1.4 min/attempt) → ~2.2 h per seed, all three ≈ 19:30–20:00 CDT
  (my 14:01 message said ~15 h — correct this in the next report). Shares the provider TPM with the K5 mechanism run: 429s occur but every one retried (0 exhausted).
  After it finishes: rsync seeds_k100_s* to hpg, prepare dose grid v2 (scripts/prepare_seedsft_dose_grid.py, CPU), Seed-SFT sweep
  K∈{1,2,50,100}×3 seeds, then baseline cells for K∈{1,2,50,100}.
- **rai GPU1: mechanism run on k5s0-t400k** (scripts/rai_mech_k5s0.sh, pid 1513532, tree 70314a3; G=43, R=4), FCE phase, ~12 h TPM-bound
  → reports/runs/mechanism/k5s0-t400k-bi/run; then coverage/probes (amendment 5) and readout.
- Monitors: b1iw9ybuc (K100 collection), bidkfdhyu (K5 mechanism), hpg job/stage watcher (re-armed with the ids above).

## Done today (moved from RUNNING JOBS)
- (13:3x CDT snapshot of the previous RUNNING JOBS list, superseded by the list above)
- **hpg (tree 74cdb97)**: 44826292 ETO K20 s0, 44826293 IPR K20 s0 (dept, pending QOSGrpMemLimit); 44836718/44836719 LEAP privileged/zero (resubmitted after shipping third_party/leap_llm templates; first attempts died in 10 s)
  K20 s0 (moved to the group account 12:3x CDT, pending Priority).
- **rai CPU + OpenAI: K=100 seed collection** (scripts/rai_collect_k100.sh, pid 1731893, tree 0834361, started 12:40 CDT) extends
  seeds_k20_s{0,1,2} → data/verified/seeds_k100_s{0,1,2} (+ subsets k1/k2/k5/k10/k20/k50 with exact identity checks); logs
  logs/collect_k100_s*.log. Shares the 200k TPM with the K5 mechanism run.
- rai GPU1: mechanism run on k5s0-t400k (pid 1513532), FCE probing (145 probe episodes at 12:01 CDT).
- **Run 2 coverage-qualified attribution (amendment 5)**: M1b 4/24 (absorbed but not invoked), **M1a 0**, M2 7/24, both 4, execution 1, unresolved 8; training fit 93.7% → experience-side failures are invocation/transfer, not absorption. reports/runs/mechanism/k20s0-p20-bi/probes-a5.
- Committed: c723ae8 amendment 5; 0834361 stage 2f (K ladder 1..100); MECHANISM_PLAN header times corrected to commit times.
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
