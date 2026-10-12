# PROJECT_STATE — Paper 1: Sufficient Experience for Black-Box Agent Distillation

Governing document: `~/hq/projects/tc-alignment/docs/handoff/2026-10-04_paper1_context_ZH.md` (sections 6, 7, 10–12, 14).
Discord channel: tc (1536171352573349909). Compute: hpg primary (`/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation`), rai for CPU/smoke.
Teacher: gpt-6-luna via the original OpenAI key, no spend cap (user, 2026-10-04). Student: google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7.

**Direction (user, 2026-10-07, v2 supersedes v1): Same Tools, Different Policies — interaction-strategy distillation under fixed tools/API**
(docs/direction/2026-10-07_same_tools_different_policies_ZH.md): is the teacher's best interaction style what the student should learn?
Core claim to test: the value of an interaction depends on who executes it (H1 executor × interaction, H2 adjusted targets > more teacher
trajectories, H3 predictable by capability). Experiments E0–E5 (E2 = executor × native interaction cross). v1 text follows:
**v1 (superseded framing): Drop-In Agent Distillation** — from K trusted samples (expansion to new instances allowed and
reported separately), distill a student that reliably replaces the teacher inside the unchanged harness H0. Measure output fit,
local decision quality (fixed history bank) and autonomous execution separately. First hypothesis H1: normal routes learned,
feedback response after deviations not. Plan P0–P4 (docs/direction/2026-10-07_*.md; weekly report reports/weekly/week01_ZH.md).
ETO/IPR/LEAP = old "seed-instance-only" setting, finish and keep. Closest prior: HAD (2610.02858, no code), SOPD (black-box, no
code), PivotOPD (recovery, code soon), FutureBridge (needs logits). P1 preview: K≥20 matches teacher mean but ~10 tasks differ
each way (T-only 9–17, S-only 8–15) → need a teacher re-run for teacher self-disagreement.

## RUNNING JOBS
- **00:35 CDT 10-12: R0 + noise reference SUBMITTED** on tree a943c21 (ICML27_runs/a943c21; .venv and data/verified/r03_surrogate
  symlinked to $B). Jobs (all 1×B200, PENDING Priority): noise-a 45604558, noise-b 45604559 (cold p40 on dev300, two servers),
  r0a 45604560 (warmth_dump at state-update-1/4/8, ckpt 31221/104070/416280), r0b 45604561–64 (CRN unions A,B,A∪B,0 at
  update 8 and p10, 4 probe pairs, r=4, T=10; 52 items/state, 2 shards/state). Out: $B/reports/runs/r03/{noise,r0a,r0b}.
  rai smoke (A100, real 12B, tiny 5-update run): CRN repeat bitwise identical; one real update vs zero message = +0.002–0.006
  nats/token; continue_from_state f=0 reproduces adapter + m,v,step bitwise. Item cost ~280 s on A100 with the 80-trajectory
  surrogate (scoring ~230 s) → switched to r03_surrogate40 (deviation from prereg, recorded). R0a has 6 states, not 9: the
  probe's cold/warm-up states are not saved (deviation). Watcher: hpg_check_once.sh now includes these 7 jobs.
- **22:44 CDT 10-11: R0–R3 pre-registered** (docs/direction/2026-10-12_real_model_R0-R3_prereg_ZH.md, 8cde4c1; sent to the user).
  Start z10 = pilot cold p10 checkpoint (update 115; = main-table k20s0); surrogate = K100 s0 \ K20 s0 (80 teacher successes);
  dev300 = dev140 + 160 group-disjoint train games; pool200 for rollouts/queries; CRN tails T=10 (~1.4 min each).
  Budgets: R0 4 / R3 3 / R1 10 / R2 7 GPU-h (~24, hard cap 40); teacher ≤640 (R1) + ≤1,500 (R2) calls, hard caps.
  Order: noise ref + R0 → R3 → R1 → R2. Engineering by Codex in parallel (no GPU): A = tools/crn_tail.py, warmth_dump,
  grad_stats, amplitude_prediction, train/continue_from_state.py; B = data/manifest/build_r03_sets.py, rollout_pool,
  teachers/capped.py, collect_corrections, collect_rewrites, r2_allocate. Then my review → rai smoke → hpg (SLURM mine).
- **22:32 CDT 10-11: theoretical_framework/ rewritten to v0.16 and pushed** (d8051a2, bf546a4, c366d08, 93e966e). User accepted
  the v0.16 direction ("认可", 21:15 CDT) and approved R0–R3 (~40 B200 GPU-h; "R2 你可以选择几组合适的" = I pick the query
  budgets). New layout: README (Level 0–2 overview, per-section mermaid graphs at 20px, §1 Definitions 1.1–1.5, Level 3 exact
  statements, status, concordance appendix N01–N20 → new numbers) + proofs/0K_*.md and experiments/0K_*.md for §2–§9 (EN+ZH,
  32 pages; old N-pages in _trash/theoretical_framework_v015_nodes_20261011). §5 optimizer state, §6 Theorem A, §7 Theorem B,
  §8 verification + Theorem C (Thm 8.8), §9 CED. 25 figures (fig2_1…fig9_2, ≥12 pt at 900 px; 5 new for A–C). Validator now
  enforces declared K.J labels, anchors sK-J/cK-J, EN/ZH parity, suite-prefixed codes (91 tests); full tree offline PASS,
  GitHub file view PASS. Errors in the v0.16 draft found while re-deriving (errata at the top of the draft): Cor 4.2(b) error
  term and (c) "iff"; Cor 5.4 cap wording; Thm 3.5 condition (E) vacuous for singular designs → regularized design; Remark 3.6
  √dim factor unproved → 4d^{3/2}; Prop 2.4(c) v-block remainder is first order in η; Remark 5.7 medians recomputed (0.0035/
  0.0035/0.0195/0.0045); "7% incl. calibration" → 7.0% aligned (no calibration) vs 11.2% (4.0% in tail updates) calibrated.
  NEXT: write R0–R3 pre-registration (configs, predictions, criteria, GPU and teacher-token budgets with hard caps), show the
  user, then run R0 → R1 certificate → R3 → R2 (teacher labels last).
- **15:25 CDT 10-11: README BFS restructure + file-view rendering fix pushed.** User screenshot: N02(b) display math shown as
  raw code. Cause (measured with pushed probe refs): GitHub's file view leaves a ```math fence inside a list as code whenever
  that list also has inline math; the /markdown API renders it, so --online had passed. 107 fences / 30 pages moved out of
  lists (d005eef); figure specs converted. Validator: new `math-fence-in-list` rule and `--blob` mode (pushes the working
  tree to hidden ref refs/render-check/tmp, fetches contents API HTML, deletes the ref in finally) = authoritative pre-push
  check; slug rule fixed to GitHub's \p{Word}. README (EN+ZH) now Level 0 (one paragraph) → Level 1 (six blocks, LR
  graph) → Level 2 (20 nodes plain-language, LR graph) → §3 notation → Level 3 statements (block I also has derivation
  outlines + experiment results + figures; II–VI pending the user's OK on the sample). Plain-language layers already
  tightened per the user's review (W3 plateau, warm-Adam approximation, N20 norm/b, "tested static proxies").
- **17:11 CDT 10-11: v0.16 controlled checks C1–C6b done** (theory/v016_checks, CPU ~2 min, 35 tests): 15/16 PASS.
  FAIL = C4.amplitude, one boundary setting (mean-field crossing 68.6 vs 77.0±4.0). Negative results recorded in the
  draft: first-order rankers fail where the v-channel dominates (Remark 4.5); first-order certificate certified 0/1,584
  (Remark 5.7) → §5 rebuilt on finite CRN paired replacement (Thm 5.3): point estimate 2.8% wrong-sign, 7% of audit
  cost, but rigorous certificate with a-priori B=2 certifies ~nothing; observed-range diagnostic 45–57% (aligned, 0
  wrong) / 23–28% (short-tail, 22–28 wrong) → open gaps: provable tighter range B, testable calibration transfer
  (Remark 5.8). Still awaiting the user's two decisions (accept direction → rewrite framework; approve R0–R3).
- **15:57 CDT 10-11: v0.16 revision draft written** (docs/direction/2026-10-11_v016_revision_ZH.md; file-view render PASS,
  26/26 display blocks). Answers the user's pasted review. Standard math language, no Node organization (user directive).
  New results: Theorem A = learner-relative information bound (Thm 3.3 Le Cam lower bound for ANY distiller incl.
  replay/synthesis; Lemma 3.4 Elfving/Sion design minimax; Thm 3.5 c-optimal achievability, scalar tight within [1/4,1];
  Cor 3.7 aliasing; Cor 3.8 demo↔query exchange rate; Prop 3.10 coverage model + water-filling; Prop 3.11 SGD weight ∝ d²
  vs warm-Adam ∝ d (frequent) / d² sign-only (isolated); Cor 3.12 learner separation). Theorem B = Adam state-channel
  decomposition (Thm 4.1 exact adjoint) + Cor 4.2 magnitude law Val ∝ (1−β1^{T+1})λ − ½ρω_Tλ², λ*=(1−β1^{T+1})/(ρω_T),
  harmful beyond 2λ* (scratch sim within 2–3%) + Prop 4.3 LESS/param-only inconsistency. Theorem C = CRN group-means
  certificate (Thm 5.3) replacing oracle ∇V. Fixes: W3 (Prop 2.5 exact κ̄(s)=(1−β2)/(β2(1−β2^s)), plateau), warm-Adam
  statistic (Prop 2.4), partial clipping (Prop 2.6), N20 norm (Remark 2.7). Closest prior found: ISO (Liu & Li 2026,
  arXiv 2608.19762, short-horizon m/v channels, no magnitude law); Adam In-Run Shapley (fixed-state). Experiments §7:
  controlled C1–C6 (Codex, CPU), real R0–R3 (~40 B200 GPU-h + teacher labels for R2). Awaiting user review before
  rewriting theoretical_framework/ and before any GPU/teacher spend.

- **12:3x CDT 10-11: doc rendering/reference fix DONE** (Codex⑦–⑩ + my edits). All display math → ```math, inline math →
  $`…`$ (GitHub renders `$…$` only after a space and fails on `_` next to punctuation), macro denylist, node part labels
  (a)(b)…, ~1,150 linked node references, check-ID legend + links, source concordance table at README end, proof-page
  self-links removed, "proof of (x)" headers. Validator theoretical_framework/tools/check_markdown.py: offline + --online
  (GitHub /markdown API) both PASS on 86 files. Progress report rewritten with readable references + pilot eval (§4.6).
  Package docs (v015_revision, hash-pinned) untouched, so their own rendering stays as shipped.
- **11:40 CDT 10-11: pilot-eval 45476444 COMPLETED** (09:10–10:00 CDT, 50:06, exit 0; backfilled; BI dev140, one server, all 6
  adapter checks passed, 0 errors/parser/overflow). Success: p10 cold 66 · mw32 77 · lw32 74; p40 cold 112 · mw32 99 · lw32 104
  (/140). Paired: p10 mw32 vs cold 19/8 (McNemar p=.052), lw32 vs cold 19/11 (p=.20); p40 mw32 vs cold 6/19 (p=.015),
  lw32 vs cold 9/17 (p=.17). Noise reference: cold adapters are byte-identical to the main-table k20s0 p10/p40 adapters;
  the two BI servers give 70→66 (8 flips) and 115→112 (7 flips). Warm-up arms ahead at p10, behind at p40; one training
  run per arm, so arm differences cannot be attributed to warm-up vs trajectory divergence. All pilot jobs done.
- **11:33 CDT 10-11: doc rendering/reference fix** (user screenshots: `\operatorname` rejected, mid-paragraph `$$` broken by
  emphasis, Prop/Thm/(A9) labels not visible). Verified with GitHub /markdown API: only standalone `$$` or ```math render.
  Codex⑦ (logs/codex_20261011_123346.log; rai log names are EDT) converting all display math to ```math, macro denylist, node part labels
  (a)(b)…, linked node references, source concordance, check-ID legend, validator theoretical_framework/tools/check_markdown.py.
  Then: my review + online render check of all pages, push, report.
- **11:24 CDT 10-11: progress document written** (5362f31) (user: "把v0.15以来所有的理论框架和实验证明上的进展都整理成一个文档给我"):
  docs/direction/2026-10-11_v015_progress_ZH.md — framework N01–N20 summary, N12/N18–N20 statements, V1–V9 + E0–E12 table,
  Gemma pilot (warmth, B-only warm-up, clipping-dominated response), gaps and next steps. Final versions only.
- **02:40 CDT 10-11: pilot-probe 45476443 COMPLETED** (01:46–02:35 CDT, 49:17, exit 0; 9 states × 4 pairs, 0 teacher calls;
  $B/reports/runs/pilot-warm-1010/probe/cold.json). Restore of saved update states 1/4/8 bitwise equal (adapter, m, v, step, RNGs).
  float64 reference vs native deltas: max |diff| 9.3e-10 (float32 rounding of parameter subtraction; ν agrees to 4 decimals).
  Median ν_centered (non-additivity of the A/B/AB update response; SGD-unclipped = gradient floor 0.020–0.041):
  AdamW unclipped: cold 0.578 · W8 0.598 · W32 0.426 · u1 0.506 · u4 0.553 · u8 0.518 · 31k 0.398 · 104k 0.273 · 416k 0.151.
  AdamW clipped (the real recipe): 0.45–0.63 at every state, no decline; SGD clipped 0.31–0.62 — clip 1.0 is active on
  every probe batch (pre-clip norms ≥4), so clipping, not Adam coldness, dominates the response after the first ~20 updates:
  the realised learner is C4 "always-active clipping" (first-order statistic = mean unit direction). Moment warm-up states
  stay non-additive (W32 0.426) because LoRA A keeps v=0 (45.33%). Momentum share 0.73–0.97 after step 0 (current batch
  ≈ (1−β1) of a step). cos_centered(A,B) cold Adam 0.28–0.33 vs ≈0 at checkpoints (sign-like coupling only when cold).
  eval 45476444 PENDING (Priority), SLURM estimated start 10-13 13:35 CDT (slipped from 10-12 04:54).
- **23:40 CDT 10-10: pilot-lw32 45476442 COMPLETED** (22:56–23:37 CDT, 41:36, exit 0; 469 updates, 6 checkpoints). All three
  training arms done. Loss u118–469: cold 0.071 / mw32 0.076 / lw32 0.071; last-50 0.018 / 0.025 / 0.013 (training loss, 1 seed).
  probe 45476443 + eval 45476444 PENDING (Priority), SLURM estimated start 10-12 04:54 CDT.
- **23:28 CDT 10-10: pilot-mw32 45476441 COMPLETED** (21:21–22:05 CDT, 44:25, exit 0; 469 updates). pilot-lw32 45476442
  RUNNING since 22:56 CDT (GPU 100%, 88 GB; 373/469 at 23:29). probe 45476443 PENDING (Priority), eval 45476444 (Dependency).
  **Finding (C4(g) on the real model): lr=0 moment warm-up at LoRA B=0 warms only B.** A has zero gradient while B=0, so
  45.33% of coordinates (the A blocks) keep v=0 through all 32 warm-up steps; the first real update moves B, A becomes
  live at update 2 and is exactly as cold as in the cold arm (v=0 fraction 0.4533 at u1–u2, 0.0001 from u3 in both arms).
  So mw32 is "B-warm, A-cold", not fully warm. Training loss u11–50: cold 0.494, mw32 0.476, lw32 0.703 (LR ramp);
  u51–117: 0.271 / 0.258 / 0.289 (single seed, training loss only). κ q99<1: u18 / u18 / u20.
- **21:02 CDT 10-10: pilot-cold 45476440 COMPLETED** (18:53–19:36 CDT, 42:50, exit 0; 469 updates, 6 checkpoints, states 1–9).
  Warmth over the full run: κ q50<1 from u3, q90<1 from u8, q99<1 from u18; fraction κ>1 <10% u8, <1% u18, <0.1% u25.
  0.0073% of coordinates (~4,800) keep v=0 through u469 (never receive gradient). Clip (1.0) active on 439/469 updates (median
  pre-clip norm 4.1). Loss first-10 mean 0.886 → last-50 mean 0.018. Tunnel was down 19:40–21:02 (VPN leg; one notice sent).
  Still PENDING (Priority): mw32 45476441, lw32 45476442, probe 45476443; eval 45476444 (Dependency).
- **18:53 CDT 10-10: pilot-cold 45476440 RUNNING** (backfilled; GPU 97%, 88 GB; 77 updates by 19:02). Early --log-warmth:
  update 1 all v=0; update 2 45% of coordinates still v=0 (LoRA A frozen at step 1 — C4(g)); fraction κ>1: 100%, 77%,
  18–41% (u3–6), 2.8% (u10), 0.8% (u30), 0.01% (u77); median κ 0.022 (u10) → 0.0005 (u77). Cold window ≈ first 10 of 469 updates.
  Theory work done and pushed 8ca77e5 (N18–N20 + 20 figures); completion message sent 18:5x.
- **18:4x CDT 10-10: central theorem added (user: "按照你的想法补上非平凡的中心定理…").** Optimizer-transformed linear
  response (docs/direction/v015_extension/Optimizer_Transformed_Linear_Response_EN.md, a1566ce → corrected 5568846/db10cab):
  C1 expansion in optimizer increments, C2 run-level representation + certificate, C3 sufficiency/necessity/minimality,
  C4 catalog (SGD, momentum, warm/cold Adam(W), clipping, Lion, LoRA), C5 dimension-free estimation, C6 margin criterion.
  Checks (Codex⑤, theory/v015_checks/otlr.py): V8/V9/E11 PASS; E10 INFO (smooth optimizers' exact cells within bound;
  C23 cold-Adam gap 0.07691 predicted 0.07568, naive ≈0; multi-round Adam remainder first order in eta — first-order term
  reliable only under the C6 margin); E12 INFO (0 wrong signs on 56 margin rows; static proxies 36% there; oracle 29%
  outside). Framework nodes N18–N20 written (EN+ZH, db10cab). Codex⑥ (pid 3619094) drawing N18–N20 figures + polish.
  theoretical_framework/ N01–N17 pushed f1200f0. User: "都做好了告诉我" → send one completion message.
- **17:31 CDT 10-10: v0.15 A pilot SUBMITTED** on tree **c0b7b01** (Codex①+③ reviewed; clean-worktree CPU suite 1086 passed,
  5 env skips). scripts/ops_pilot_warm_1010.sh, out $B/reports/runs/pilot-warm-1010: 45476440 cold (save states 1–9), 45476441 mw32
  (moment warm-up 32), 45476442 lw32 (LR warm-up 32) — all K20 s0 main-table recipe 416,280 tokens, --log-warmth, 1×B200 32G 1h15;
  45476443 probe on cold (cold, warmup:8/32, update:1/4/8, checkpoint:31221/104070/416280; 96G 2h; afterok cold);
  45476444 BI dev140 eval of {cold,mw32,lw32}×{p10,p40} on one server (64G 3h; afterok all three). Monitor JOBS += these 5.
- **17:0x CDT 10-10: user asked for theoretical_framework/** (main page + proofs/ + experiments/, EN+ZH, verdict-first,
  PDF figures, final version only — memory feedback-theory-framework-docs-final-only). 68 pages + 17 figure specs written
  (4 forks); Codex④ (pid 3251901, logs/codex_20261010_182538.log) writing figures/make_figures.py. User Q "best paper level?":
  answered honestly — not yet (core characterization near-definitional, affine-only specializations, no real-model evidence).
- **16:49 CDT 10-10: v0.15 A+B status.** B: E8/E9 committed d8b3da5 (E8 PASS: W1 holds on 2,496 law pairs, max gap/bound 0.988; warm-up
  removes the short-horizon C23 gap — moment N=3 eta .1 R=1 0.077→0.0004 at W=10, R=4 0.120→0.0016 — but NOT the R=40 gap 0.005–0.05;
  E9 negative: warm-up does not improve cheap proxies, wrong-sign +1–8 pts). A: Codex① (train/sft.py moment/LR warm-up, per-update
  states; tools/adam_response_probe.py; train/optimizer_state.py) reviewed, uncommitted; Codex③ pid 2830575 (log logs/codex_20261010_174619.log,
  spec scratchpad/codex_probe_fix_task.md): momentum-centred nu (old nu divides by a norm containing the gradient-free momentum step),
  float64 Adam reference, checkpoint:<n> probe states (sft checkpoints already store optimizer m/v), --log-warmth. Then clean-worktree
  suite → commit → new hpg tree → scripts/ops_pilot_warm_1010.sh <sha> (3 arms cold/mw32/lw32 on main-table K20 s0 recipe, probe on
  cold, one 6-adapter BI dev140 eval {arm}×{p10,p40}). hpg-b200 backlog: 852 pending; SLURM estimates IPR r5 sft start 10-14 09:34.
- **15:33 CDT 10-10: IPR r5 SUBMITTED** on tree **5e8443c** (parallel hashing + parallel listing + rank-0 verification; hpg read-only
  check job 45468513 on the r4 tree: all ~770k file hashes and pairs-1 digest/report identical; stage re-hash 235 s vs ~24 min,
  pair re-derivation 300 s vs ~31 min → DPO startup idle ~9 min vs ~55 min). CPU suite on a clean worktree: all pass once third_party
  is linked (main-tree failures were Codex editing concurrently). Chain (scripts/ops_chain_ipr_r5.sh, out reports/runs/baselines/ipr-k20s0-r5):
  45470256 sft (4 GPU 96G) → 45470257 x1 → 45470258 dpo1 (4 GPU 256G) → … x6 45470267 → dpo6 45470268 → 45470269 eval. Monitor JOBS = ETO 3 + r5 14.
- **15:05 CDT 10-10: user "关于AB你可以按照你想的来，做完以后最后结合这次所有的v0.15写成报告"** → doing both: B theory (docs/direction/v015_extension/
  Warm_Adam_Response_EN.md, Props W1–W2, Cor W3; Codex E8/E9 controlled checks) and A Gemma pilot (Codex: train/sft.py moment warm-up /
  lr warmup / per-update state saves + tools/adam_response_probe.py; then hpg 1×B200 runs, ~8 GPU-h). Final: one combined v0.15 report.
- **14:03 CDT 10-10: user sent v0.15 revision** (Thm 19 erratum = v0.14.1 + optimizer-state/few-shot addendum + Followup). DONE 14:35:
  verified (report docs/direction/2026-10-10_v015_verification_ZH.md; checks theory/v015_checks, commit 90ec27f/b8d79c4). C23 Adam gap =
  cold-start optimizer effect (frozen-Jacobian Adam reproduces it; warm m,v removes ~95%). Awaiting user: A micro-LoRA pilot plan (GPU) /
  B theory (warm-Adam lazy-regime response theorem; CPU).
- **GPU-idle fix status**: ec17316 (parallel hashing + rank-0 verification) built on hpg; read-only hashcheck job 45466718 (hpg-milan):
  hashes equal, but listing is the new bottleneck (expert-mc 352k files 328 s ≈ 0.17 s/rollout dir) → Codex round 2 parallelizes the
  rglob listing (3.11/3.12 semantics). r5 chain script scripts/ops_chain_ipr_r5.sh drafted (not submitted; will use the round-2 tree).
- **13:4x CDT 10-10: USER CHOSE (a)** for the GPU-idle fix: ETO tp6 finishes on tree 6bac2da (45058903 x3 → 45058904 dpo3 160G → 45058905 eval);
  IPR r4 withdrawn — cancelled all 12 pending r4 jobs (45439203 dpo1, 45082399–45082409). Reason: identity.json is frozen and binds
  code_sha256, so a code fix cannot resume r4; r4 output tree (sft + x1 + pairs-1 done, dpo-1 attempt-000 OOM) stays in place as an
  abandoned run (disclose). Next: Codex fix (parallel artifact_hashes, byte-identical; rank-0-only validation/pair derivation in
  fsdp_train with gloo handoff) → review/tests → new tree → CPU-job hash-equality + timing check on the r4 tree → IPR **r5** fresh chain
  (sft→x1→dpo1…dpo6→eval; sizing sft 4-GPU 96G, x 1-GPU 32G, dpo 4-GPU 256G, eval 32G). Monitor JOBS = ETO 3 only.
- **13:39 CDT 10-10: hpg back** (user re-opened ControlMaster); no change during outage, all 15 pending (heads 45439203 + 45058903 Priority).
- **10:40 CDT 10-10: hpg ControlMaster expired** (Permission denied keyboard-interactive) — one notice sent (msg 1558504551915454487);
  waiting for user `ssh -fN -M hpg`. Last seen 09:40: all 15 pending (IPR dpo1 45439203 Priority). Monitor now dedupes the tunnel notice
  (scratchpad/hpg_tunnel_notified marker; prints TUNNEL_BACK when reachable again).
- **09:31 CDT 10-10: IPR r4 dpo1 45082398 OUT_OF_MEMORY** (160G; my 10-07 shrink sized from 4-GPU SFT, but DPO loads policy+reference
  per rank, ~52G/rank). GPUs were 0% for all 56 min: 24 min pipeline re-hash of completed stages + 31 min per-rank pair re-derivation (Lustre
  small files) before weight load. Resubmitted **45439203** (256G, same submit line, tree 6bac2da, no code change); x2 45082399 → afterok:45439203;
  dpo2–dpo6 MinMemoryNode → 256G; ETO tp6 dpo3 45058904 96G → 160G (2-GPU DPO peak 104G). Monitor JOBS list updated. Open: ~55 min GPU-idle
  startup in every resumed DPO segment (idle-GPU policy risk) — proposed Codex fix (hash cache + rank-0 pair derivation), needs a new tree.
- **03:22 CDT 10-10: USER DECISION on v0.14** — "V0.14和v2关系不大，他是一个解决这类问题的通用理论框架": v0.14 is a general theory
  framework, NOT a theory layer of v2. Goal: complete the framework, derive a suitable novel agent-distillation method from it, test it
  experimentally. First step (in progress): verify the theory per the whole package (= plan gate G1 + §9 finite experiments).
  Done: manifest hashes 19/19 OK; v0.14 + v0.13 scripts rerun bit-identical (theory/v014_repro/, env theory/.venv py3.12 pinned deps + sympy).
  My proof pass Prop1–Prop21: no math error found; one statement gap (Thm 19/(39) implicitly assumes visited certified states are audited cells).
  DONE 05:1x CDT: independent suite C1–C23 (theory/independent_checks/, Codex-built, reviewed + rerun by me, identical): 22 PASS / 1 FAIL
  (C15 = Thm 19 (39) coverage gap, exact counterexample). Controlled small net: affine holds in lazy SGD; Adam breaks it (sign update →
  threshold learner). Report docs/direction/2026-10-10_v014_verification_ZH.md sent. Awaiting user: which next step (fix Thm 19 wording →
  v0.14.1; theory-completion priorities: optimizer-aware info requirement / role of K / lazy-regime response theorem / state shift;
  G2/G3 restore audit). Still open from 10-09: B (K counting), seeds, baselines.
- **02:14 CDT 10-10: still all 15 pending** (IPR r4 dpo1 45082398 now QOSGrpMemLimit = dept mem cap; ETO tp6 x3 45058903 Priority; rest Dependency).
  Monitoring = session cron c7e05476 hourly at :17 running scratchpad/hpg_check_once.sh (silent on NO_CHANGE; seen-file dedupes events).
  Session-only: after a restart, recreate the script (copy of the loop in hpg_monitor_1009.sh without the loop) + cron.
- **22:4x CDT 10-09: hpg reachable again; v0.14 handoff received** (English theory + proof/novelty audits + CED method + coding-agent plan;
  stored docs/direction/v014_handoff/, reading note docs/direction/2026-10-09_v014_reading_note_ZH.md). Not yet adopted — asked user:
  v0.14 vs v2 relation, K per-family (K=32 sits in ALFWorld saturation), pilot power (8 eval tasks/fork), baseline list vs official-code rule, seeds.
  LEAP done 10-08 14:3x CDT: K50 52/53/75/**81**, K100 49/68/92/**82** (last iterate). Still pending since 10-07 (mlperf reservation to 10-12 08:00 CDT):
  IPR r4 dpo1 45082398 (QOSGrpGRES) → … → 45082409 eval; ETO tp6 x3 45058903 (Priority) → 45058904 → 45058905.
- **00:37 CDT 10-08: hpg ControlMaster expired again** (Permission denied keyboard-interactive) — one notice sent; waiting.
  Last seen 00:15: all chain heads pending — LEAP K50 g4 45082132 + K100 g4 45082137 (QOSGrpGRES, group), IPR r4 dpo1 45082398 +
  ETO tp6 x3 45058903 (Priority, dept). LEAP K100 round 3: 18 corrections, N=2,880. Weekly report v2 sent (week01_ZH.md);
  5 user decisions pending (adaptations, teacher re-run, IPR, second workflow, capability student) + _trash "删".
- **15:4x CDT 10-07: research re-framing sent** (reports/research_plan_1007_ZH.md). Gap vs handoff: grounded Augmented-SFT /
  free-expansion arm (§8.2) not run; valid_seen saturates (K=50 Seed-SFT = teacher). Proposed N2 expansion controls at K=5/10 ×
  s0–2 (resample / covered types / uniform / diagnosis-targeted / matched irrelevant), N3 residual-barrier testbed (a: held-out-scene
  dev-unseen from train; b: second env, lean ScienceWorld). **Awaiting user: N2 go?, N3 a/b/both?, N5 AgentGym-only control?**
  Also awaiting "删" for rai _trash (167G, peer hq-2e asked; user said 可以 in DM per peer — confirm in channel).
- **12:4x CDT 10-07: IPR → r4 with 4-GPU FSDP segments** (user "尽量快"): 45082396 sft → 45082397 x1 → 45082398 dpo1 … 45082409 eval
  (scripts/ops_chain_ipr_r4.sh). r3 chain cancelled. Scheduler --test-only estimates put 1/2/4-GPU B200 starts at 10-13 (mlperf
  reservation 32 nodes 10-08 08:00 → 10-12 08:00 CDT) — watch actual starts; expect slow progress through the reservation.
- **12:4x CDT 10-07: OpenAI topped up (probe 200).** LEAP chains (scripts/ops_chain_leap_1007.sh; teacher stages CPU-only hpg-milan):
  K50 45082130 → 45082131 (cpu) → 45082132 → 45082133 eval; K100 iteration 2 invalidated (moved to $B/_trash/leap_k100_credit_exhausted_1007)
  → 45082134 (cpu correct-2) → 45082135 → 45082136 (cpu) → 45082137 → 45082138 eval. Disclose the K100 correct-2 re-run.
- **12:3x CDT 10-07: hpg back (user re-opened ControlMaster).** Summary 10-04→now sent (reports/summary_1004_1007_ZH.md + main figure).
  Open user decisions: OpenAI top-up; IPR 2 vs 4 GPUs (~3 days on 2); dose controls (only AgentGym on 1 GPU?); mechanism at K=10/50.
- **12:28 CDT 10-07: hpg ssh "Permission denied (keyboard-interactive)"** — ControlMaster expired (Duo); asked user to run
  `ssh -fN -M hpg` on rai. Chains keep running. Last seen: ETO tp6 dpo-2 45058902 running; IPR r3 dpo-1 45078017 queued/running;
  AgentGym K100 done (BC 102 / 103 / 99 / 96). After reconnect: re-arm the monitor (persistent seen-set), check chain health.
- **08:4x CDT 10-07: stage-split chains (idle-GPU policy; no code change, same identities, NGPUS=2 in every segment; FSDP
  segments 2 GPUs, explore/MC/eval segments 1 GPU; afterok; scripts/ops_chain_1007.sh, log ~/chain_1007.log)**. Tree 6bac2da
  (empty-visible-turn fix). ETO K20 s0 fresh → baselines/eto-k20s0-tp6: 45058898 sft, 45058899 x1, 45058900 dpo1, 45058901 x2,
  45058902 dpo2, 45058903 x3, 45058904 dpo3, 45058905 eval. IPR K20 s0 fresh → baselines/ipr-k20s0-r3: 45058906 sft,
  45058907 x1 done (P=95); tail resubmitted 11:5x with 24 h dpo limits: 45078017 dpo-1, 45078018/20/22/24/26 x2–6,
  45078019/21/23/25/27 dpo-2–6, 45078028 eval (IPR ≈ 9–13 h DPO per iteration → ~3 days total on 2×B200). AgentGym k100s0 (def44a9 resume;
  held 45027584_3 cancelled): 45058921 train-2, 45058922 explore/filter-2, 45058923 train-3, 45058924 rest → then direct evals
  (ops_agentgym_direct_eval.sh k100s0). mlperf reservation = 32 of the B200 nodes only (10-08 08:00 → 10-12 08:00 CDT).
  Still held: 45027583 LEAP K50 (credits), 45008706 IPR control (old tree; to be redone as a chain).
- **07:4x CDT 10-07**: ETO K20 s0 tp5 (45008702) FAILED 06:52 CDT at dpo-3 start (dpo-1, dpo-2 complete): encode_turn refuses empty
  replies; 9/20 pairs-3 rejected continuations contain assistant turns with content '' (finish_reason length, 1,024 tokens, nothing visible).
  Codex (pid 2105709, main working tree, log scratchpad/codex_empty_turns.out): mask empty visible turns in student-sourced trajectories
  (kept in history, no labels), teacher seeds still strict, counts in result.json, DEVIATIONS entries. Code change ⇒ fresh ETO run on a new
  tree; bundle with the idle-GPU fix (still awaiting user). LEAP control iter-0 = 108 (unmerged 115). Monitor now keeps a persistent
  seen-set (scratchpad/hpg_seen_jobs.txt) — the tp5 failure was missed between two re-arms.
- **02:55 CDT 10-07 status** (report sent with reports/figs/main-results, sources regenerated from synced evals; quota moves done:
  16 dirs, 0 failures, dept 19.0T). Main table K=1..100 (mean successes/140): Seed-SFT t400k 59.7/71.7/84.7/94.3/107.7/114.7/111.0;
  Kang+SAG 51.3/62.0/74.3/78.7/82.7/94.3/92.3; AgentGym BC 50/48/61.7/61.0/58.3/93/—; AgentEvol iter-3 47/55/32.3/43.0/69.3/102/—;
  LEAP priv s0 66/60/62/55/60. Running: ETO tp5 45008702 (dpo-1). Queued: LEAP control eval 45027585. HELD: 45008706 (IPR control),
  45027584_3 (AgentGym k100s0), 45027583 (LEAP K50). Not resubmitted: IPR r2 (root-cancelled), ETO control (post-export nonzero exit).
  Waiting on user: UFRC email? / idle-GPU fix approval; OpenAI top-up. Rai copies of LEAP K1–K10 come from the yd24f archive.
- **01:2x CDT 10-07 BLOCKERS (asked user)**: (a) **OpenAI credits exhausted** — probe 429 insufficient_quota/credit_balance_exhausted.
  LEAP K100 (44943706, COMPLETED 08:40) iteration 2 is INVALID: all 44 correction requests (132 attempts, 00:51–01:12 CDT) failed 429 →
  0 corrections → "stopped_empty_min_aggregate". Iteration 1 valid (67 failed, 19 excluded, 43 corrections, 3,440 examples). After
  top-up: move stages correct-2+ and audit/teacher/correction-2 (outcomes cached as "exhausted") to _trash, resume. Engineering fix
  needed: insufficient_quota must fail the stage, never count as an empty correction set. LEAP K50 resume 45027583 HELD (iters 1–2
  valid: 19, 21 corrections; needs teacher for iter 3). (b) **Root cancellations ≈ UFRC idle-GPU policy** (4 jobs; policy threatens
  account suspension): explore/eval/MC stages pin CUDA_VISIBLE_DEVICES to the first GPU of 2-GPU jobs. nvidia-smi 01:1x: IPR r2 1/2 idle,
  ETO control 2/2 idle (loading), LEAP K100 0% (teacher wait), ETO tp5 2/2 100%. HELD: IPR control 45008706, AgentGym k100s0 45027584_3.
  01:4x: IPR r2 45008703 CANCELLED by root after 1:42 (the job with 1/2 idle) → policy confirmed; not resubmitted. Both ETO jobs in DPO
  (2/2 at 100%), evals/Kang 1 GPU busy → nothing cancelled. Proposed: Codex adds a second replica on GPU 2 for single-model stages (batch-invariant → same episodes).
- **00:5x CDT 10-07: hpg back; dept /blue QUOTA FULL** (project quota 2273 on /blue/fsu-compsci-dept: 19.63T/20T, shared by the whole
  dept; group yd24f project 2228: 11.84T/18T). Disk quota exceeded killed AgentGym k50s0 (eval-iter-2), k100s0 (train-2), k5s0-r2
  (eval-iter-3) and LEAP K50 (train-2) between 23:46 and 23:58 CDT; running jobs (LEAP K100, ETO tp5, IPR r2, Kang k100s1, AgentGym
  control, k5s1 iter-2 eval) show no quota errors. Our baseline checkpoints ≈ 2.0T (23G per full-FT/merged checkpoint, 45G fp32 init).
  Relief (moves, not deletes; cross-project mv = copy+unlink, slow): ~/quota_move_1007.sh + ~/quota_move_worker.sh on login11, log
  ~/quota_move_1007.log. Superseded → /blue/yd24f.fsu/xc25.fsu/hq/_trash/ICML27_agent_distillation/superseded_1007 (leap-privileged-k20s0-m2/m3/m4,
  leap-zero-k20s0-m3/m4, eto-k20s0-tp/tp2/tp3/tp4, ipr-k20s0); evaluated LEAP cells → /blue/yd24f.fsu/xc25.fsu/hq/ICML27_archive/baselines_evaluated_1007
  (leap-privileged-k1s0/k2s0/k5s0/k10s0/k20s0-m5, leap-zero-k20s0-m5) — rai sync for these now reads from the archive path.
  Results seen: AgentGym k50s0 BC 93, iter-1 89; k5s0 (r2) BC 53, iter-1 44, iter-2 54. **LEAP control (Seed-SFT init, K20 s0)
  stopped after iteration 1: 6/20 training rollouts failed, all 6 on tasks without a privileged state → 0 corrections** (status
  stopped_empty_min_aggregate). Submitted: LEAP K50 resume 45027583 (def44a9, group), AgentGym k100s0 resume 45027584_3 (def44a9),
  LEAP control iter-0 eval 45027585 (8b9230a, group), direct evals AgentGym k50s0 iter-2/3 45027610/1, k5s0 iter-3 45027871.
- **23:31 CDT 10-06: hpg tunnel down again** — notice sent; waiting. Last seen 23:0x: LEAP K10 iter 61/64/60/55, K2 59/54/62/60, K5 62/62
  (all ≈ base); LEAP control (init) passed validation and is training; AgentGym k5s0-r2, Kang/AgentGym old-tree k50s0/k100s0 running;
  ETO tp5, IPR r2, AgentGym/ETO/IPR controls pending. AgentGym k2s0 direct evals 45014404/5. After reconnect: check old-tree
  AgentGym k50s0/k100s0 for overflow (→ scripts/ops_agentgym_direct_eval.sh CELL), submit LEAP K50/K100 + control evals, sync, figure.
- **21:3x CDT resubmission (tree 8b9230a = overflow + init-validator fix; scripts/ops_resubmit_1006.sh)**: cancelled old-tree resumes
  44948828/45003751/45003752. Direct full-checkpoint dev140 evals (hpg_eval_adapters, MERGED_MODEL_PATH, one model per server) of trained
  AgentGym iterates whose eval stage failed → eval/baselines/agentgym-<cell>-<label>: k5s1 iter-2/3 (45007915/6), k10s0 iter-3 (45008246),
  k10s1 iter-2/3 (45008362/3), k5s2 bc/iter-1..3 (45008370-3), k20s1 iter-3 (45008699). Fresh: AgentGym k5s0 45008700, Kang k100s1
  45008701 (old dirs → $B/_trash/overflow_resubmit_1006), ETO K20 s0 45008702 (eto-k20s0-tp5), IPR K20 s0 45008703 (ipr-k20s0-r2),
  dose controls 45008704 AgentGym / 45008705 ETO / 45008706 IPR / 45008707 LEAP (…-k20s0-init).
  AgentGym evals so far: k5s1 bc 88, iter-1 40, iter-2 47(1 err); k10s0 bc 68, iter-1 56.
- **10-06 20:4x CDT hpg back.** Failure diagnosis (~/diag_1006.txt on hpg):
  * Context overflow: degenerate students (1024-token Thought spam, no action) exceed max-model-len 32768 → vLLM HTTP 400 → episode
    'policy_error' (unknown) → stage fails. Hit AgentGym k5s0 explore-1, k5s1/k10s1 eval-iter-2, k5s2 eval-bc, k20s1 eval-iter-3, and ETO
    K20 s0 explore-3 (rounds 1–2 done). Codex (pid 682405): known terminal failure 'context_overflow' (+ init validator fix).
  * External 'CANCELLED … SIGNAL Terminated' (not me): IPR 44826293 (12:11 CDT), Kang k100s1 (13:16), AgentGym k10s0 (14:55) →
    resubmitted in their own trees to resume: 45003751 (AgentGym k10s0, d9e151b), 45003752 (Kang k100s1, def44a9); IPR 44948828.
  * Dose-control init validator rejected the real Seed-SFT adapter (short target_modules names) → LEAP control failed in 45 s;
    AgentGym/ETO/IPR controls cancelled while pending (44944146-8). Resubmit all four after the fix.
  * Done: Seed-SFT K50/K100 all (t400k K50 112/116/116, K100 115/106/112); Kang K20 s0 seeds 0/1/2 = 63/74/86; LEAP K1 iter-0 66,
    K2/K10 complete, K5 stopped after iter-1; LEAP evals K2/K5/K10 45002760-2. LEAP K50/K100 running.
- **12:50 CDT 10-06: hpg tunnel down** (localhost:2222 banner timeout) — sent the one notice; waiting. Before it dropped: IPR K20 s0
  44826293 was terminated externally (12:11 CDT, SIGNAL Terminated on c1000a-s25) and resubmitted as 44948828 (same tree 74cdb97,
  resumes stages). LEAP K1 s0 stopped after iteration 0 (no usable correction) → eval 44944505. Pending after reconnect: LEAP
  ladder evals as each finishes; AgentGym/LEAP control evals after their training; sync + main figure (K50/100, Kang ladder, LEAP).
- **User 10:57 CDT: all four morning proposals approved, in parallel.** (1) LEAP across K with option (b) — all-data/final-epoch beyond 200
  updates, disclosed; privileged at K=1/2/5/10/50/100 s0, 0% stays at K20. (2) AgentGym K1/2/50/100 trimmed to s0 (44921818_[4-11]
  cancelled), throttles raised (44837380 %2, 44921818 %4), nice reset. (3) Kang K20 s0 training seeds 1, 2. (4) Dose controls
  (Seed-SFT k20s0-p40 init) for AgentEvol/ETO/IPR/LEAP at K20 s0. Codex A pid 3672270 (main: LEAP>200 + Kang seed); Codex D pid
  3673961 (worktree ~/hq/projects/ICML27_dosectl, branch dose-controls: --init-checkpoint).
  LEAP privileged K1/2/5/10/50 s0 submitted (tree def44a9, group, no >200 change needed): 44938864–44938868 → $B/reports/runs/baselines/leap-privileged-k{K}s0.
  K100 after Codex A. Evals: submit LEAP_MODE=evaluate per cell when training completes (same tree).
  **12:3x CDT**: 61001ec (LEAP >200 updates option b; Kang --training-seed) + 060a118 (merge of dose-controls 53c59b2).
  LEAP privileged K100 s0 44943706 (group, tree 61001ec, 36 h); Kang K20 s0 training seeds 1/2: 44943707→44943708 (SAG eval),
  44943709→44943710 → baselines/kang-k20s0-tp-seed{1,2}, eval/baselines/kang-k20s0-tp-seed{1,2}-sag.
  **Dose controls K20 s0** (tree 060a118, INIT = seedsft-k20s0-hpg/checkpoint-tokens-416280 = main-table p40):
  44944146 AgentGym, 44944147 ETO, 44944148 IPR (dept, 2×B200), 44944149 LEAP privileged (group) → baselines/<method>-k20s0-init.
  Evals for AgentGym/LEAP need a separate evaluate-mode job after training (ETO/IPR evaluate per their pipelines).
- **10-06 09:1x CDT results**: Seed-SFT K50 s0 peak **114 = teacher** at p30 (t400k 112, p40 109); Kang K1 51/53, K2 51/62, K50s0 90;
  k2s2 t400k 59 (K2 = 70/86/59). ETO K20 s0 running (15 pairs, dpo-1). Morning report sent 06:00 (reports/morning_1006_ZH.md), 4 decisions pending.
- **Baseline cells for K∉{5,10,20} CANCELLED 04:4x CDT** (44871885/6, 44879553/4, 44893701/2): baselines/common/pipeline.py (and eto)
  still guard 'Registered few-shot K is 5/10/20' (Kang k1s0 died in 2 min). Codex (K-guard task): REGISTERED_K = ladder + audit of
  K-dependent preconditions per method (LEAP's <200-updates assumption at K=50/100, X5 at K=1) → reports/baseline_k_ladder_audit.md.
  Resubmit after review; LEAP at K≥50 may need a user decision if the official 200-step save/holdout would apply.
  Fixed in def44a9 (REGISTERED_K; audit reports/baseline_k_ladder_audit.md: Kang/AgentGym/ETO/IPR run at every K; LEAP K=50 OK;
  **LEAP K=100 iteration 3 may need 220–239 updates > the <200 precondition → user decision before LEAP K=100**). Resubmitted
  (tree def44a9, 12 cells k1/k2/k50/k100 × s0–s2): 44921817 Kang (%3), 44921818 AgentGym (%2, Nice 200). Failed k1s0/k2s0 Kang dirs →
  $B/_trash/baseline_kguard_failed_1006.
- **rai GPU1: K5 mechanism re-run run-a6** (pid 2982972, tree 70314a3, 03:0x CDT): first run (run/) finished FCE 41/43; 2 episodes hit the
  700-call cap → launcher exit 2 before interventions. Amendment 6 (0da23fd): one retry at 1,400/episode, run cap 40k, fresh workspace
  seeded read-only from run/'s teacher cache → reports/runs/mechanism/k5s0-t400k-bi/run-a6; log logs/mechanism_k5s0_t400k_bi_a6.log.
  15 min in: 14,414 cache hits, 61 new HTTP calls (57 on the two capped tasks, 4 on two others) → after FCE, diff all 41 non-capped
  FCE results (t*, D curves, fce) between run/ and run-a6 (amendment 6 item 3) and report any change.
  **a6 FCE 43/43 done; diff vs run/: 41 identical, 0 changed** (amendment 6 holds); capped tasks resolved at t*=16 (D 8 / 14).
  **K5 mechanism a6 DONE (06:1x CDT)**: G=43; 13 cool-type gaps excluded from the split (no cool seed in K5 s0 → separate);
  on 30 paired t* states, excess rescues over I0: I1 4, I2 4, I3 8, **I4 13**, I5 5; Section-6 cells both 4, M2 4, M1(coverage
  pending) 3, I4-only/unresolved 19/43. Taxonomy: F-precondition 18, F-search 15, F-loop 5, F-goal 3, F-cap 2. vs K20 run2 (I3 11, I4 9,
  I5 8, unresolved 9/24): the K5 student is short on executable action knowledge, not on advice. Readout run-a6/launches/attempt-000/readout.
  Coverage + probes (amendment 5) launched: pid 3303750, worktree 5093783, rai GPU1 → reports/runs/mechanism/k5s0-t400k-bi/probes-a5.
  **K5 probes DONE**: coverage-qualified attribution (43): no-seed-type 13 (cool), knowledge_not_in_experience 2,
  M2_with_knowledge_not_in_experience 4, M1b 1, **M1a 0**, M2 4, execution_or_capacity 4, unresolved 15; training fit 97.5%.
  → absorption never fails (K20: M1a 0 too); at small K the gap is knowledge absent from the K seeds (19/43).
- **hpg capacity (10-06 ~03:0x CDT)**: B200 partition 466/480 GPUs allocated; nothing of mine running for hours (pending reasons
  QOSGrpGRES/MemLimit are stale — the partition is full). **Reservation `mlperf-benchmark` takes 32 hpg-b200 nodes 10-08 09:00 →
  10-12 09:00 (hpg local, = 08:00 CDT)** → almost no B200 for 4 days from Thursday morning. Lowered AgentGym arrays to Nice=200 so
  Seed-SFT K50/100, Kang, ETO/IPR start first. Remaining need ≈ 150 GPU-h; tell the user in the morning report, propose priorities
  (Seed-SFT K50/100 > Kang > ETO/IPR K20 > AgentGym extra cells) and ask whether to trim AgentGym to s0 outside K=5/10/20.
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
  LEAP 0% K20 s0 (corrections 11 / 7 / 8, rejected 2 / 5 / 7, N 764 / 840 / 1120): **dev140 iter-0 58, iter-1 59, iter-2 52, iter-3 58**.
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
