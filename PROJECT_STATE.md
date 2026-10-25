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
### 01:4x CDT 10-25 — all 8 variant tsft/AD cells done (dev140 solved, converged column, seed 0)
| student | base | tsft K30 | tsft K100 | AD K30 | AD K100 |
|---|---|---|---|---|---|
| 12B (seed 0) | 62 | 88 | 112 | 109 | 107 |
| E4B | 43 | 86 | 107 | 112 | 102 |
| E2B | 9 | 84 | 96 | 95 | 100 |
Teacher 114. e4b-tsft-k100 47600525 107 (vs 12B 9/14 p=.41); e4b-adc-k100 47598299 102 (vs 12B 6/11 p=.33, vs E4B AD30
8/18 p=.08); e2b-adc-k100 47597009 100 (vs 12B 9/16 p=.23). All serving checks passed, 0 error/parser/overflow.
Figure reports/analysis/main_grid_seeds/student_variants.png (3 panels, legend top, y from 0 so the E2B base shows).
SAD trains submitted 01:22-01:32 CDT on c265b5d: e4b k30 47601174, e4b k100 47601175, e2b k30 47601896.

### 01:3x CDT 10-25 — E2B tsft K100 96, E2B AD K30 95: E2B falls behind once there is more signal
- e2b-tsft-k100-e64-lr1e-4-eval 47596876: 96/140 (68.6%); e2b-adc-k30-e64-lr1e-4-eval 47596920: 95/140 (67.9%); both
  serving checks passed, 0 error/parser/overflow. Paired: E2B AD30 vs 12B AD30 10/24 p=.024, vs E4B AD30 9/26 p=.006,
  vs E2B tsft30 24/13 p=.10; E2B tsft100 vs 12B tsft100 (112) 8/24 p=.007, vs E2B tsft30 23/11 p=.06.
- Reading so far: K30 tsft is capacity-insensitive (12B 88 / E4B 86 / E2B 84); AD K30 and tsft K100 separate E2B
  (95-96) from 12B/E4B (107-112). e4b-tsft-k100 LONI training valid (L_last 1.09e-5); eval 47600525 running.

### 01:2x CDT 10-25 — E2B tsft K30 + base; SAD export fix deployed (plan v22)
- e2b-tsft-k30-e64-lr5e-5-eval 47596875: E2B tsft K30 84/140 (60.0%); E2B base 9/140 (6.4%; 25 parser errors, 1 context
  overflow = base behaviour, harness unchanged); serving check passed (adapter max |diff| 2.4e-4). Paired: vs 12B tsft30
  (88) 15/19 p=.61, vs E4B tsft30 (86) 19/21 p=.88, vs E2B base 77/2. K30 tsft: 12B 88 / E4B 86 / E2B 84 despite bases
  62 / 43 / 9.
- SAD export fix 5df9eb4 (branch eseries-kvexport, Codex, reviewed): model_type gemma4 only, unused KV-shared base tensors
  copied byte-for-byte + sha256 at export and validate_checkpoint; tower hash flattens before the uint8 view (E-series
  towers hold 0-dim buffers; bytes unchanged). Checkpoint/serving/eseries/stage3 tests: same 24 failures as unchanged
  2506fa9, 21 new tests pass; Codex full suite 2,404 pass / 57 fail after vs 2,383 / 57 before (same failures), real E4B
  meta check = exactly 54 unused tensors. Trees: c265b5d = 46b8527 + 5df9eb4 (SAD prep/train), 17eec0d = 0c2cd9e + 4f47296 + 5df9eb4
  (SAD eval), both on hpg sha-identical. Plan v22 (26a093d): 4 SAD cells unheld, train limit 14 h; preps stay done.

### 01:0x CDT 10-25 — first E-series result: E4B AD K30 = 112/140 (80.0%)
- e4b-adc-k30-e64-lr1e-4-eval 47596874 (tree 46b8527): serving check passed (adapter max |diff| 4.4e-5); 0 error /
  parser / overflow / failed calls. vs 12B AD K30 conv 109 (77.9%): both 96, E4B-only 16, 12B-only 13, McNemar p=.71.
  Teacher 114 (81.4%).
- e4b-tsft-k30-e64-lr5e-5-eval 47596869: E4B tsft K30 86/140 (61.4%), E4B base 43/140 (30.7%); serving check passed
  (adapter max |diff| 1.3e-4); 0 error/parser/overflow. 12B: base 62, tsft K30 88, AD K30 109. Paired: E4B AD vs E4B tsft
  36 vs 10 (p=.0002); E4B tsft vs 12B tsft 16 vs 18 (p=.86); E4B tsft vs E4B base 47 vs 4. E4B starts 19 tasks lower but
  K30 distillation lands at the 12B level for both tsft and AD.
- e4b-adc-k100 LONI training valid (L_last 2.31e-5, no collapse); eval 47598299 queued. e4b-tsft-k100 still on LONI.

### 00:4x CDT 10-25 — two E-series engineering failures fixed/in progress; SmartAD moved to LONI (user: "咱们尽量并行")
- Failure 1 (eval 47583436, 23:57 CDT): vLLM engine died at CUDA-graph capture, "FlashAttention forward only supports head
  dimension at most 256". vLLM 0.27.1 picks FLASH_ATTN for model_type gemma4 (E-series); VLLM_BATCH_INVARIANT=1 makes
  fa_utils downgrade FA4 -> FA2, which cannot do global_head_dim 512. 12B (gemma4_unified) ran TRITON_ATTN all along
  (FLASH_ATTN not a candidate). NB the exported env VLLM_ATTENTION_BACKEND=FLASH_ATTN in serve provenance is ignored by
  vLLM 0.27.1. Fix 4f47296 (eseries-template, Codex, reviewed): `--attention-backend TRITON_ATTN` iff config model_type
  == gemma4; 12B command byte-identical (tests); 4 failures in the serve-related suite pre-exist on 2506fa9 (LEAP/probes).
  Eval tree 46b8527 = 95a62b2 + 4f47296, deployed on hpg (1,246 files sha-identical). Plan v20 (ff20203): all 12 variant
  adapter evals on 46b8527. Partial eval dirs -> hpg _trash/es_eval_fa2_1025. Cancelled our doomed evals 47586755/56,
  47587029, 47592039. Relaunched 01:38 EDT: 47596869 (e4b tsft K30 + base), 47596874/75/76, 47596920, 47597009.
- Failure 2 (e4b-sadc-k30-b1-train 47578027, 23:39 CDT, update 218/~3400): full-parameter export_checkpoint ->
  checkpoint_key_map "Base checkpoint tensor missing from model: model.language_model.layers.24.self_attn.k_norm.weight".
  E4B base stores k_proj/v_proj/k_norm for its 18 KV-shared layers (24-41; 54 tensors) that HF never instantiates.
  Fix in progress (Codex, worktree ICML27_eseries2 branch eseries-kvexport): copy those tensors byte-for-byte from base,
  12B strict. Variant SAD cells held (plan v20); e4b-sadc-k100 train 47590689 cancelled before start.
- LONI: all 4 K30 + e2b K100 trainings valid (e2b tsft K100 L_last 1.1e-5, adc 1.4e-5); e4b K100 (1128645/48) running.
- SmartAD -> LONI (plan v21 c7ed96f): hpg keeps only the 4 smc evals (gated eval_<cell>). smc_loni.sh e4b (pid 3116967) /
  e2b (3116968): wait <tag>_smartad_selected, stream the selected collection (~6 GB) hpg -> LONI, full sha256 manifest
  compare, submit prep (LONI single, CPU) + train (gpu2, afterok, NGPUS=2: global batch = plan indices, rank count only
  splits microsteps), then sync_and_gate_v2 -> eval gates. Tree f453db8 on LONI (1,275 files sha-identical). Logs
  logs/smc_loni_<tag>.log. Merges: e4b 47587371 (310/500 attempts at 00:39 CDT; 12B merge took 4 h), e2b 47595553.

### 23:2x CDT 10-24 — variant K30 trainings done (LONI), evals queued on hpg
- LONI K30 done + synced (adapter sha verified) + eval gates opened: e2b-adc 22:43, e4b-adc 22:44, e2b-tsft 22:50,
  e4b-tsft 23:03 CDT. All 4 valid (L_last within rule, no collapse flag): e4b tsft 3.1e-5, e4b adc 1.8e-5, e2b tsft
  4.1e-5, e2b adc 1.6e-5. LONI K100 (1128645/48/51/54) still RUNNING (2 h in).
- hpg: e4b selection s0-s2 done 22:18 CDT; e4b s3 + e2b s0 RUNNING, e2b s1 done, e2b s2/s3 PENDING; e4b-sadc-k30 train
  47578027 RUNNING (2 GPU); first E-series eval e4b-adc-k30 47583436 PENDING QOSGrpGRES (scontrol top not permitted).
  Dept QOS still full (24/24 GPU, QOSGrpCpuLimit). Watcher b8j8zlgl5 (watch_first_es_eval.sh) reports the vLLM E-series
  LoRA serving gate verdict of 47583436.

### 21:4x CDT 10-24 — variants training on real weights; hpg limited to 3 GPUs
- LONI (2506fa9) all 8 updating, loss falling (e4b tsft K30 0.83 -> ~0.2). Measured 21:36-21:46 CDT: K30 4-8 s/update
  (done ~22:45 CDT), K100 10-15 s/update (done ~0:15-1:15 CDT). Then sync_and_gate_v2 copies + opens eval gates.
- hpg: group QOS 8 GPUs, yd26 holds 5 (reachguard 4 RTX, 20 h limit; parser 1 L4); dept 24/24 GPU, 127/128 CPU -> we run
  3: e4b selection s0-s2 (47569122/23, 47569925) RUNNING; e4b s3 + e2b s0-s3 PENDING QOSGrpGRES. SAD preps (4) done
  (47570116/25, 47571651/866); SAD trains wait for GPUs. SmartAD pool too large to copy to LONI (stays on hpg).
- Told user: tsft/AD 8 cells ~1-2 AM CDT; SmartAD/SAD tomorrow morning-noon depending on group GPUs.

### 21:1x CDT 10-24 — E-series fix landed; variants relaunched (plan v19 d8cead0)
- Fix 2506fa9 (branch eseries-template, Codex reviewed): template-aware OFF prefix (_off_prefix: 12B empty-thought prefill
  or no prefill), shared history allows generation_start == prompt_tokens + flex hook on Gemma4TextAttention,
  load_student/FSDP dispatch on model_type (Gemma4TextDecoderLayer; towers frozen + FSDP ignored_params), LoRA targets =
  same 7 projection names restricted to model.language_model.layers.N.(self_attn|mlp).*, k_eq_v derivation no-op for
  gemma4, serving conversion branch. Tests: 267 pass on c6b15b2+fix; 297/267/267 on e599953/3318b54/b9f9a80 twins;
  20cb697 twin 2 golden failures only vs the c6b15b2 reference (empty_visible_reply_masked field absent in 20cb697) ->
  52/52 pass with 20cb697 as reference. CPU only; GPU paths (real-weight FSDP/flex, vLLM E-series LoRA) unverified.
- ES_TREE: c6b15b2->2506fa9, e599953->f453db8, 3318b54->4c2bcfc, b9f9a80->0c2cd9e, 20cb697->95a62b2; deployed on hpg
  (every tracked file sha-identical, third_party 1,857 files) and 2506fa9 on LONI. Partial first deploys -> _trash.
- Old variant outputs -> hpg _trash/variants_pre_eseries_1024 (4 SAD preps), LONI _trash/variants_pre_eseries_1024 (8).
  Keeper state of the 26 variant cells reset (state.bak.es_relaunch.json). Gates e4b_ready/e2b_ready reopened.
- LONI resubmitted on 2506fa9: e4b 1128644/45/46/48, e2b 1128649/51/52/54 (all RUNNING 21:1x CDT). Watchers still up:
  sync_and_gate_v2 (1964069 e4b, 1970367 e2b), gate_e{4,2}b_smartad (1964070, 1970368).

### 19:4x CDT 10-24 — E4B/E2B blocked on E-series support (engineering); variant cells held
- All 8 LONI variant trainings FAILED (1128289-92, 1128302-06; ~25 min, data encoding): "Gemma OFF generation prefix
  changed". Cause: the E-series chat template's thinking-OFF generation prompt has no empty-thought prefill (ends
  "<|turn>model\n"; thinking via "<|think|>" in the system turn) while train/dataset.py requires the 12B prefill.
- Deeper: E-series = HF Gemma4ForConditionalGeneration (Gemma4TextAttention, audio+vision towers, 18/42 (E4B) KV-shared
  layers without k/v_proj, per-layer input gates/projections), not Gemma4Unified*; fsdp_train wrap class, shared-history
  flex hook, LoRA target/vLLM key conversion are unified-class specific. vLLM 0.27.1 (hpg + rai envs/vllm-serve) has
  Gemma4ForConditionalGeneration with SupportsLoRA + KV sharing; Gemma4Unified subclasses it.
- Actions: gates e4b_ready/e2b_ready moved to gates/_closed; plan v18 (6810d73) holds all 26 variant cells; hpg
  selection shards (8) scancelled. SAD preps of both variants COMPLETED (validation only). Codex (pid 1994425, worktree
  ../ICML27_eseries branch eseries-template from c6b15b2, task scratchpad/codex_eseries.txt): template-aware encoding,
  shared-history no-prefill + E-series attention hook, FSDP/LoRA for Gemma4ForConditionalGeneration, serving conversion;
  12B byte-identical; tiny-model shared-history == per-turn tests. Then cherry-pick onto c6b15b2/e599953/3318b54/
  20cb697/b9f9a80 (train/dataset.py identical in all but 20cb697), deploy, GPU smoke (rai GPU ok per user), relaunch.
- Told user: results likely 10-25 morning CDT.

### 19:1x CDT 10-24 — STUDENT VARIANTS E4B + E2B (user 18:38-18:55 CDT); multi-seed STOPPED
- User: stop the multi-seed runs; only change the student base (everything else incl. the recipe unchanged), seed 0,
  K30 + K100 first, compare with teacher and the base student; E4B (google/gemma-4-E4B-it @ee0ef602) and, in parallel,
  E2B (google/gemma-4-E2B-it @3e22461f); final figure = 3 panels side by side (12B | E4B | E2B), legend on top.
  "现阶段可以假设不同机器上跑的结果不变，但是要确保数据都是完全一致的".
- Seeds: plan v15 holds all 48 seed cells; adc-k500-e64-lr5e-5-s1/s2 (47516734/47540378) scancelled 18:43 CDT; LONI seed
  offload (v14, 18 cells) never submitted. Kept: tsft K500 s1/s2 (+evals 125/120), tsft K30 s1/s2 trains (no eval).
- Cells per variant (12B converged rung + lr reused): tsft K30 e64 5e-5, tsft K100 e64 1e-4, AD K30/K100 e64 1e-4,
  SmartAD K30 e32 / K100 e64 1e-4 (re-selected with the variant's NLL from the same 5,000 teacher candidates), SAD K30 b1 /
  K100 b05; base student eval rides on the tsft K30 eval (INCLUDE_BASE=1). configs/base_{e4b,e2b}.yaml = base.yaml with
  only models.student changed (sha faf549ff / 2c6a40b5 on rai, hpg, LONI). Plan v16/v17 (ebaf4ac): e4b-*/e2b-* cells.
- Data identity checked: model.safetensors sha256 = HF LFS on hpg + LONI (+ rai for E4B): E4B cfbd3d2f, E2B 2db5482b;
  tokenizer cc8d3a0c (same as 12B family); receipts (manifest/selection/identity) of seeds_full_train_r2 (+k30/k100) and
  adftp-chain collection (+k30/k100) identical on rai/hpg/LONI; LONI c6b15b2 tracked files sha-identical to git (1,258).
- Where: LONI gpu2 (2 A100, tree c6b15b2): tsft/AD K30/K100 for both variants = 8 jobs 1128289-92 (e4b), 1128302-06
  (e2b), all RUNNING 19:54-20:0x EDT. hpg: SmartAD selection 4 shards + merge, SmartAD prep/train, SAD prep/train/eval,
  all evals. rai: no torch env for this project -> not used.
- Watchers (rai): sync_and_gate_v2.sh e4b pid 1964069, e2b pid 1970367 (stream LONI -> hpg, verify count/bytes/adapter
  sha, open gates/eval_<cell>; log logs/sync_adapters_loni_hpg.log); gate_e{4,2}b_smartad.sh pids 1964070 / 1970368.
- Gates opened: e4b_ready 18:54 CDT, e2b_ready 19:1x CDT. Unverified risk: Gemma-4 E-series (KV-shared layers, per-layer
  embeddings) under fsdp_train / vLLM LoRA serving; first evidence = first updates on LONI + the serving gate.

### 18:4x CDT 10-24 — literature re-check from the PDFs (user: "重新回顾这些论文 … 而不是说在你印象里")
- reports/analysis/main_grid_seeds/literature_student_vs_teacher.md (10 papers read; PDFs in session scratchpad papers/).
  Main-table papers (Kang, SAD, SmartAD): students 2-64x smaller, all below teacher; SmartAD reports no teacher score.
  ALFWorld agent-tuning papers: fine-tuned 7-8B > prompted GPT-4-class is common (ETO/IPR/STeCa/WKM/AgentGym/LEAP BC);
  closest to us = IPR (GPT-4 rollouts, correct only) and AgentGym (reward-filtered). LEAP/HAD beat the teacher via extra
  signal (privileged corrections on failures / harness contrast).

### 18:3x CDT 10-24 — user asked why students beat the teacher / normal in papers / success-only? (answered)
- reports/analysis/main_grid_seeds/student_vs_teacher.md. Teacher TRAIN pass@1 80.3% (= dev 81.4%), pass@3 87.2%; all 1,674 failed
  teacher attempts hit the 40-step cap; 35% of first-attempt failures solved on retry. Significant vs teacher only tsft K500 s1
  (p=.003) and SAD K300 (p=.004). All 4 baselines train on successes only (code checked). SmartAD paper Table 2 reports no teacher score.
- Offered again: 2 more teacher runs on dev140 -> dev pass@3 reference line (awaiting user).

### 18:3x CDT 10-24 — final-figure script with teacher line (user 18:10 "在图上画出来teacher的表现，作为依据")
- reports/analysis/main_grid_seeds/: seed_scores.py (hpg; per-seed solved sets, conv + paper, SAD one run) -> seed_scores.json;
  fig_seeds.py -> baseline_perf_seeds.png: converged column, 4 lines = seed means, open circles = single seeds, teacher
  gpt-6-luna 114 (one run, verified 114/140 from teacher140 per_task.tsv) + base 62 as reference lines. Rerun both as seeds land.
- tsft-k30-e64-lr5e-5-s1 47516238 / -s2 47520028 COMPLETED 17:05 CDT: both valid (L_last ~0); collapse flags s1 ep28, s2 ep11,
  recovered -> c. Their evals wait: group GPU cap 8 full (our adc-k500 s1 4 B200 + yd26 4 RTX6000), dept 127/128 CPU, 22/24 GPU.
  adc-k500-e64-lr5e-5-s2 47540378 PENDING (QOSGrpGRES).
- Tunnel: second outage ~16:16 to ~18:0x CDT (notice 16:21); back (user 18:10).

### 16:2x CDT 10-24 — first seed cell done: tsft K500 converged = 122 / 125 / 120 (s0/s1/s2), mean 122.3
- Evals 47516706 (s1) 125/140, 47516235 (s2) 120/140; serving check passed; adapters = own seed's attempt-000 checkpoint;
  0 error/truncated/overflow. Paired vs s0: s1 +6/-3, s2 +4/-6. Per task: 115 solved 3/3, 7 2/3, 8 1/3, 10 0/3
  -> seed noise on this cell ~±3 tasks; 15 tasks are seed-dependent.
- Now: tsft-k30-e64-lr5e-5-s1 47516238 + -s2 47520028 (2 B200 each), adc-k500-e64-lr5e-5-s1 47516734 (4 B200) = 8/8 group
  GPUs. All 16 SmartAD seed preps done (trainings wait for GPUs).

### 15:2x CDT 10-24 — tunnel back; tsft K500 s1/s2 trained (both valid, s1 = c); their evals running
- Tunnel down 10:01 to ~14:54 CDT (keeper SSH_FAIL; one notice 10:15). Submitted jobs unaffected; keeper resumed 14:54 CDT.
- tsft-k500-e64-lr5e-5-s1-train 47473037 COMPLETED 15:11 CDT (6:21 h): L_last 3.7e-5 = L_min -> valid; collapse flags
  ep 2 (0.687 vs 0.268) and ep 11-13 (0.22/0.51/0.61, gmax 458,617 at ep 12), recovered by ep 14 -> kept, marked c.
  s2 47473038 COMPLETED 15:07 CDT (6:10 h): L_last 1.65e-5 valid, no flag (seed 0: 1.86e-5, no flag).
- Running: evals 47516235 (s2), 47516706 (s1), 1 B200 each; tsft-k30-e64-lr5e-5-s1-train 47516238 (2 B200, started 15:11);
  PENDING adc-k500-e64-lr5e-5-s1-train 47516734 (4 B200). SmartAD preps: 10 done; k100/k30 e2 s1/s2 running (CPU).
- hpg $HOME is 100% full (40G/40G; ~/.cache/vllm 29G, last write 10-22). Our evals compile-cache under /blue
  tools/vllm-cache (server.log checked) -> not affecting us; $HOME not authorized -> left to the control plane / user.

### 09:3x CDT 10-24 — seed 1/2 cells all queued (48 trainings + 48 evals); first runs verified
- Trees: tsft c6b15b2; AD e6459b9 (seed-ad, 23 new tests); SmartAD 4f5a138 (seed-smc, 252 tests incl. seal binding). Both hpg
  trees verified by file count + bytes (1,259 / 93,636,659 B; 1,274 / 93,868,494 B). Gates seed_ad_ready, seed_smc_ready open.
- Plan v13 (d5d0702): 48 seed cells, longest first; limits unchanged (10 jobs, 8 group GPUs; dept CPU QOS full).
- RUNNING: tsft-k500-e64-lr5e-5-s1-train 47473037 + -s2 47473038 (4 B200 each, ~6.3 h, ETA ~15:15 CDT); s1 identity
  recipe training_seed 1, lr 5e-5, warmup 0, epochs 64 (checked). SmartAD CPU preps fill the remaining job slots.
- 09:2x CDT: both tsft K500 seed trains past startup (idle GPUs ~20 min = normal pipeline startup, logs checked); first
  update indices differ from seed 0 (s0 [419,459,...], s1 [376,179,...], s2 [240,7,...]) -> seed reaches the sampler. ETA ~15:20 CDT.
- Sent: method pain-point analysis (reports/analysis/main_grid_perf_1024/method_painpoints.md, c389fe7).
- User 08:48: finish 12B first (no E4B pilot meanwhile); E4B full grid estimated 150-200 GPU-h, 1-1.5 days, later.

### 08:5x CDT 10-24 — PRE-REGISTERED: training seeds 1, 2 of the main grid (user 08:32 CDT "可以按照你建议的来")
- Seed = training seed only: LoRA init, dropout stream, epoch sampler order. Same K views, data (AD-FTP / SmartAD pools),
  reported rung (epochs) and lr of each seed-0 cell, GPU count, recipe, eval tree/decoding. No ladder; every seed run is evaluated.
- Cells: 12 converged + 12 paper-recipe LoRA cells (tsft/AD/SmartAD) x seeds 1, 2 = 48 trainings + 48 evals. Paper tsft/AD
  cells use c6b15b2 with explicit lr 2e-4 / warmup 0 (same trainer argv + recipe as the 20cb697 defaults of seed 0).
- SAD is NOT retrained: full-parameter (no random init), fixed ascending-complexity curriculum, no active dropout
  (seed-0 result dropout_equivalence "exact") -> training does not depend on the seed; table notes "SAD: eval noise only".
- Trees: tsft c6b15b2 (KANG_TRAINING_SEED exists); AD c6b15b2 + --training-seed (branch seed-ad); SmartAD e599953 +
  --training-seed + fsdp_train SmartAD seed check relaxed (branch seed-smc). Seed 0 default byte-identical (tests).
- Fallback (as seed 0): training INVALID (L_last > max(1.1 L_min, L_min + 0.02)) or dev140 <= 5/140 -> that seed is rerun once
  with lr halved. Collapse-and-recover runs are kept and marked c. SmartAD K30 serving gate fails on tolerance -> same as seed 0
  (float32 reference, then recorded override per the user's option B), marked ‡.
- Reporting (user 08:38): per cell mean over seeds 0-2 (+ min-max); final figure = converged column only, one line per method
  (4 lines), each the seed mean. Paired comparisons on per-task success averaged over seeds.
- Analysis goal (user 08:34): convergence is only the measurement condition; analyses must yield pain points a NEW
  agent-distillation method should fix. Experiments as parallel as possible (08:36).
- Capacity 08:5x CDT: group yd24f 8 B200 free (nobody running); dept QOS CPUs 123/128 held by others' multi-day CPU jobs ->
  dept effectively closed; ~300 GPU-h on 8 GPUs -> ~1.5 days.

### 08:0x CDT 10-24 — user asked: cost of 2 more seeds, reuse converged rungs? (answered, awaiting seed-type choice)
- Estimate from seed-0 sacct: per seed ~200 GPU-h (converged trains ~165, paper trains ~27, 30 evals ~12); 2 seeds ~400 GPU-h,
  ~2 days on group 8 GPUs, ~1-1.5 days with dept GPUs; longest job ~6.3 h (tsft/AD K500 e64, 4 B200); +10-20% for lr-halved reruns.
- Proposed: training seeds 1,2 on the same K views (all data reused, no teacher calls), reuse each cell's seed-0 rung + lr,
  pre-registered: record L_last + collapse flag, apply the same lr-halving fallback if a cell scores ~0. Data seeds would need
  new AD-FTP + SmartAD teacher collections. Waiting for the user's choice before submitting.

### 03:1x CDT 10-24 — MAIN GRID COMPLETE (32/32): AD K300 converged = 116/140; full performance analysis delivered
- Converged: tsft 88/112c/124c/122; AD 109/107c/116†/118; SAD 49/109/127/116; SmartAD 93‡/120c/111c/114 (base 62).
  Paper: tsft 10/102†/109/108; AD 73/95/84/102†; SAD 49/104†/127/123; SmartAD 77/87/96/112.
- Headline: once converged, no baseline beats trajectory SFT at K>=100 (only AD K30, p=.001); paper recipes undertrain
  tsft (2 epochs) so the paper-column ranking is a training-budget artifact; failures left are search (never found the object).
- Report + figure + scripts: reports/analysis/main_grid_perf_1024/ (main_grid_perf_report.md, baseline_perf_k.png).
- No hpg jobs of ours running. Next: user decision on what comes after the baseline grid.

### 02:4x CDT 10-24 — AD K300 converged rung = e64 (5e-5); eval gate opened
- adc-k300-e64-lr5e-5-train 47396738 COMPLETED 02:39 CDT. L_last e32 .0009 -> e64 .0000 (delta < .01) -> converged, report e64; no collapse flag at e64.
- Gate logs/queue/gates/eval_adc-k300-e64-lr5e-5 opened 02:40 CDT (SAG decoding, eval tree 20cb697, ~25 min). Last cell (32/32); then the full analysis
  (AD_K300=adc-k300-e64-lr5e-5 analyze.py, AD_K300_COLLAPSE=0 fig.py, failure_modes.py incl. the new cell).

### 01:1x CDT 10-24 — SmartAD K30 converged = 93/140 (serving gate overridden, user option B); 31/32 cells in
- smc-k30-e32-lr1e-4-eval 47412779: 93/140; check/OVERRIDDEN + override.json recorded; summary.json carries serving_gate_override.
  Measurement identical to the fp32 diagnostic (0.0672 vs tol 0.061). Table marks it ‡.
- Only AD K300 converged left: adc-k300-e64-lr5e-5-train 47396738 (ETA ~03:00 CDT) -> ladder rule -> eval gate.
### 00:4x CDT 10-24 — SmartAD K30 e32 option B wired: tree 8b47413 (recorded serving-gate override), cell unblocked
- 8b47413 = fea3a8e + SERVING_GATE_OVERRIDE_{ADAPTER,REASON,APPROVED_BY,APPROVED_AT} (Codex, reviewed; 249 tests pass incl. 49 new).
  Only the tolerance verdict is waivable; identity checks kept; HF-reference stage not overridable; override.json bound to the
  report sha256; summary.json carries serving_gate_override. Deployed on hpg (file list = fea3a8e + new test; third_party
  1,857 files / 67,969,659 B identical).
- Plan c315cdf: smc-k30-e32-lr1e-4-eval -> tree 8b47413, REFERENCE_DTYPE=float32 + override env. Old OUT ->
  _trash/smc-k30-e32-lr1e-4-eval-fp32ref-1023. Keeper submits it next (1 GPU).
### 00:2x CDT 10-24 — user chose B for SmartAD K30 e32 ("可以按照你倾向的来", Discord 00:24 CDT): evaluate + annotate
- Annotation: "serving check failed (bf16 numerical sensitivity of a sharp adapter; |vLLM - HF fp32| 0.067 > tol 0.061;
  vLLM bit-identical across runs; HF bf16 is further from fp32 than vLLM)".
- Implementation: Codex adds a recorded per-adapter override on ../ICML27_fp32ref (identity checks kept; only the tolerance
  verdict waivable; override.json + report sha bound; summary.json carries serving_gate_override). Then new tree, old OUT ->
  _trash, plan eval stage -> new tree + override env, unblock the keeper cell.
### 00:2x CDT 10-24 — third tunnel outage 23:2x-00:2x CDT (notice 00:15, user restored); 30/32 cells in
- adc-k300-e64-lr5e-5-train 47396738: training since 23:11 CDT, epoch 20/64 at 00:23 (L .0424, no collapse); ETA train ~03:00,
  eval ~03:30 CDT. Ladder L_last e2 .2529 e4 .2005 e8 .1554 e16 .0456 e32 .0009.
- Missing: AD K300 converged (above), SmartAD K30 converged (user A/B, re-asked 00:2x CDT). Then the full performance analysis.
### 23:0x CDT 10-23 — SAD K100 paper = 104/140 (lr halved); paper column complete (16/16)
- sadc-k100-b1-lr1e-5-eval 47392746: 104/140 (was 0). Paper column: tsft 10/102†/109/108, AD 73/95/84/102†, SAD 49/104†/127/123,
  SmartAD 77/87/96/112 († = lr halved after divergence).
- AD K300 5e-5 ladder: e8 + e4 trained (22:24 / 22:50 CDT); e64 train 47396738 since 22:50 CDT (~4.4 h); e2 47398117 since 23:00.
- Converged column missing: AD K300 (e64 + eval, ETA ~03:30 CDT 10-24), SmartAD K30 (user A/B).
### 21:3x CDT 10-23 — tunnel back by 21:14 CDT; AD K300 5e-5: e16->e32 not converged -> e64 rung; SAD K100 eval next
- adc-k300 5e-5: e16 L_last .0456, e32 .0009 (delta .045) -> e64 added (plan 423139e; 4 GPU, ~4.4 h). e32 shows a one-epoch blip
  at ep6 (.161 -> .494 -> .156; collapse flag raised) unlike 1e-4 (.214 -> 5.02, 25 epochs to recover, 4/140).
- Running: adc-k300-e8-lr5e-5-train 47383356, adc-k300-e4-lr5e-5-train 47383368 (both 4 GPU, submitted 21:22 CDT).
  Next (plan order now sadlr first): sadc-k100-b1-lr1e-5-eval (~40 min) + adc-k300-e64-lr5e-5-train once e4 frees 4 GPUs.
- ETA: SAD K100 paper ~22:30 CDT; AD K300 converged (e64 + eval) ~02:45 CDT 10-24. SmartAD K30 still awaits the user's A/B.
### 18:2x CDT 10-23 — tunnel back by 18:14 CDT (hourly check); SmartAD K500 = 114/140; SAD K100 x1 lr 1e-5 trained (valid), eval gate open
- smc-k500-e32-lr5e-5-eval 47325633: 114/140 (serving check passed). SmartAD converged row: K30 N/A (pending user), 120c, 111c, 114.
- sadc-k100-b1-lr1e-5-train 47303363 COMPLETED ~16:09 CDT; loss monotone to ~0 (lr-2e-5 run sat at ~2300) -> gate
  eval_sadc-k100-b1-lr1e-5 opened (eval tree b9f9a80).
- adc-k300-e32-lr5e-5-train 47325624 running since 15:16 CDT (4 GPU, ~4 h); e16/e8/e4/e2 follow as GPUs free.
- Remaining before the full analysis: SAD K100 paper (eval), AD K300 converged (ladder + eval), SmartAD K30 (user A/B).
### 15:3x CDT 10-23 — hpg down again (SSH hangs from 15:19 CDT, ControlMaster lost ~15:3x); one notice sent 15:33
- On hpg at the time: adc-k300-e32-lr5e-5-train 47325624 (4 GPU, since 15:16 CDT), smc-k500-e32-lr5e-5-eval 47325633
  (since 15:16), sadc-k100-b1-lr1e-5-train 47303363. On recovery: read the smc K500 eval, check the SAD loss, open its gate.
- Note: paper-column tsft K30 = 10/140 is NOT a divergence (loss 0.87 -> 0.35 smooth, 20 updates): undertrained, keep as is.
### 15:2x CDT 10-23 — tunnel down 13:41-15:14 CDT; SmartAD K500 5e-5 converged at e32, eval gate opened
- hpg tunnel lost 13:41 CDT (banner-exchange timeout), back 15:14 CDT; one notice sent 14:16. Jobs unaffected.
- smc K500 5e-5 ladder all trained: L_last e2 .1541, e4 .1237, e8 .0718, e16 .0030 (collapse flag ep8), e32 .0001, e64 .0000.
  e16->e32 delta .0029 < .01 -> converged, report e32 (no collapse flag). Adapter + complete.json present, OUT absent ->
  gate eval_smc-k500-e32-lr5e-5 opened (greedy, tree 20cb697).
- sadc-k100-b1-lr1e-5-train 47303363 still running (2 GPU). AD K300 5e-5 ladder (4 GPU each) waits for the keeper (backoff <= 15 min).
- SmartAD K30 e32: still awaiting the user's A/B.
### 13:2x CDT 10-23 — SmartAD K30 e32: float32-reference gate also failed (0.067 > 0.061) -> N/A per pre-registration
- Mean only (count rule passes 8 <= 11). vLLM bit-identical across runs; HF bf16 is further from fp32 (0.078) than vLLM is
  (0.067): bf16 sensitivity of a sharp adapter, not a serving bug. Asked the user: keep N/A (rule) or evaluate + annotate
  (needs a registered gate override in the eval path). Keeper cell blocked by the FAILED eval.
### 13:1x CDT 10-23 — paper-column reruns: tsft K100 = 102, AD K500 = 102 (lr halved after divergence)
- tsft-k100-e2-lr1e-4-eval 47299031: 102/140 (was 0/140 at lr 2e-4). adc-k500-e2-lr1e-4-eval 47299041: 102/140 (was 0/140).
- sadc-k100-b1-lr1e-5: prep done (identity records lr 1e-05), train 47303363 (2 GPU) since 13:06 CDT (startup, ~3 h).
- smc-k30-e32-lr1e-4-eval 47302386 (float32 reference, tree fea3a8e) since 12:59 CDT.
### 12:5x CDT 10-23 — user: finish ALL cells before the analysis; rerun every ~0 cell; diagnose the instability
- User (Discord 11:55-12:15 CDT): convergence is a fairness device, performance + its analysis is the point; run everything
  first, then analyse; cells at ~0 must be rerun; find whether instability comes from hyperparameters, method or trainer.
- Interim performance report (14/16 converged + 16/16 paper): reports/analysis/main_grid_perf_1023/perf_report.md (+ figure).
- Diagnosis (reports/engineering/training_instability_diagnosis_1023.md; inventory spike_inventory.{py,json}): hyperparameters,
  not data / method / trainer bug. Same batch -> gnorm 3,710 in one rung vs 0.5-6 in the others (smc K300, identical order);
  spikes in all 4 methods; 3-5-update runaway (gnorm x10-1000, then loss explodes); clip 1.0 active on every update;
  spikes per 1k updates: LoRA 2e-4 2.1 (2/10 runs unrecovered), 1e-4 0.9 (2/33), 5e-5 0.1 (0/10); SAD 2e-5 0.8 (1/10).
  Recovered training loss != recovered policy (adc K300 e32: loss 0.004, dev140 4/140).
- Reruns (registered halve-lr fallback; annotated "lr halved after divergence"):
  * paper tsft K100 -> tsft-k100-e2-lr1e-4 (LONI-trained, valid: epochs 0.3375 / 0.2097) copy -> hpg, eval.
  * paper AD K500 -> adc-k500-e2-lr1e-4 (LONI-trained, valid: 0.3096 / 0.2243) copy -> hpg, eval (SAG). (adc-k500-e4-lr1e-4 on
    LONI diverged in epoch 0: 2.59.)
  * paper SAD K100 x1 -> lr 1e-5: sadc-k100-b1-lr1e-5 on tree 7cbc62b (= fb22f09 + --learning-rate / SAD_LR; default
    argv/recipe byte-identical; 16 new + 68 SAD tests pass); eval on b9f9a80 (evaluate mode does not re-derive the recipe).
  * 13:0x CDT: tsft-k100-e2-lr1e-4 and adc-k500-e2-lr1e-4 copied LONI -> hpg (verified: count, bytes, adapter sha) and
    their eval gates opened.
  * converged AD K300 -> registered 5e-5 ladder e32/e16/e8/e4/e2 on hpg (plan 40455ad), 4 GPUs each.
- SmartAD K30 e32 rerun with REFERENCE_DTYPE=float32 on tree fea3a8e (plan 40455ad; old OUT ->
  _trash/smc-k30-e32-lr1e-4-eval-bf16ref-1023). max_jobs back to 10.
### 11:4x CDT 10-23 — user (Discord 11:42 CDT): "1选A,2可以" -> group cap 8 GPUs; SmartAD K30 fp32-reference diagnostic
- Cap: plan limits max_gpus = max_group_gpus = 8 (generator e822529); installed plan has max_jobs 9 until the K30 rerun is
  settled (restore 10 after).
- SmartAD K30 e32 serving gate, option A. Pre-registered BEFORE running (11:4x CDT):
  * Code: branch fp32ref from 20cb697 (worktree ../ICML27_fp32ref) adds --reference-dtype {bfloat16,float32} to
    tools/check_lora_serving.py and REFERENCE_DTYPE to scripts/eval_adapters.sh; default path unchanged. New hpg tree.
  * Run: the keeper's smc-k30-e32-lr1e-4-eval stage on the new tree with REFERENCE_DTYPE=float32 (old OUT -> _trash).
    Gate rule and thresholds unchanged (pooled error <= max(0.05, 1.5 x base noise), >0.5-count rule, effect rules);
    only the HF reference is computed in float32 (base weights upcast; LoRA weights float32).
  * Decision: check passes -> the dev140 number is reported, annotated "serving check vs float32 HF reference; the bf16
    reference failed (0.078 > 0.068)". Check fails -> SmartAD K30 converged = N/A (serving not verified), diagnostics
    reported. Either way, report |HF bf16 - HF fp32| (reference error) and |vLLM - HF fp32| (serving error) per prompt
    from the old and new report.json.
### 10:2x CDT 10-23 — SmartAD K300 = 111/140; K500 5e-5 e64 done (L_last 0.0000), e16 running, e32 next
- smc-k300-e32-lr1e-4-eval 47281234: dev140 111/140 (79.3%), serving check passed (adapter error 0.031 <= 0.076).
- smc-k500-e64-lr5e-5-train 47239796 COMPLETED 09:54 CDT: L_last 0.0000, valid, no collapse flag.
- smc-k500-e16-lr5e-5-train 47273587 (2 GPU) at ep 8/16: transient spike in epoch 8 (25-update means 0.93/1.14 at updates
  900-950, back to 0.074) -> provisional collapse flag ep8; final validity is judged at the end of the run.
- smc-k500-e32-lr5e-5-train 47284892 (4 GPU, 4,000 updates) RUNNING since 10:17 CDT (~14:00 CDT).
- K500 5e-5 e8/e4/e2 moved to the 2-GPU lane too (plan e26c37a, preps -> _trash/smc_prep_4gpu_1023c; fresh NGPUS=2 preps); they
  follow e16 (~11:30 CDT) on that lane, so the K500 decision waits only for e32.
### 09:1x CDT 10-23 — SmartAD K100 = 120/140; K300 converged at e32, eval gate opened
- smc-k100-e64-lr1e-4-eval 47268981: dev140 120/140 (85.7%), serving check passed (adapter error 0.028 <= 0.080).
- SmartAD K300 1e-4 ladder L_last: e2 0.2546, e4 0.1277, e8 0.1016, e16 0.0019, e32 0.0001. First pair with delta < 0.01 =
  e16->e32 (0.0018) -> converged rung e32 (valid; collapse flags e4@ep2, e32@ep6 annotated). e2 had a transient spike inside
  epoch 1 (updates 31-50, recovered; same data order as e4 -- first 12 update losses match) that the collapse flag cannot see
  (no earlier epoch). Gate opened -> smc-k300-e32-lr1e-4-eval (1 GPU) starts when K500 e64 frees GPUs (~10:00 CDT).
- smc-k500-e16-lr5e-5-train 47273587 (2 GPU) RUNNING since 08:58 CDT; smc-k500-e64-lr5e-5 at ep 58/64.
### 09:2x CDT 10-23 — SmartAD K100 converged (e64), eval running; K300 rungs e16/e4 done; K500 5e-5 e16 -> 2-GPU lane
- SmartAD K100 1e-4 ladder L_last: e2 0.1514, e4 0.1133, e8 0.0601, e16 0.0339, e32 0.0002, e64 0.0000. First pair with
  delta < 0.01 = e32->e64 (0.0002) -> converged rung e64 (valid; collapse flags e8@ep4, e16@ep8, e32@ep12, e64@ep2,
  annotated). Eval gate opened -> smc-k100-e64-lr1e-4-eval 47268981 (1 GPU, greedy).
- SmartAD K300 1e-4: e16 0.0019 (done 08:50 EDT keeper = 07:50 CDT), e4 done 08:16 CDT; e2 train next (2-GPU lane). With
  e8 0.1016 / e16 0.0019 / e32 0.0001, the decision waits only for e2/e4 (rule = first pair with delta < 0.01).
- K500 5e-5 e16 (2,000 updates) moved to 2 GPUs (plan e4c0e8d; 4-GPU prep -> _trash/smc_prep_4gpu_1023b), so it runs beside
  the 4-GPU e64 -> e32 -> e8 -> e4 -> e2 sequence. smc-k500-e64-lr5e-5-train 47239796 at ep 48/64 at 08:2x CDT (~10:00 CDT).
### 03:4x CDT 10-23 — AD row complete (K500 118/140); SmartAD K30 e32 blocked by the serving gate (asked the user)
- adc-k500-e64-lr5e-5-eval 47241702: dev140 118/140, check passed. Converged AD row 109/107/4c/118.
- smc-k30-e32-lr1e-4-eval 47241704 FAILED at the serving-equivalence gate (pooled 0.078 > 0.068, >0.5-count 14 > 11; each
  prompt passes alone). Not a k_eq_v-type bug (error ~5% of effect, spread over prompts); likely bf16 serving error
  amplified by a sharp adapter on off-training check prompts (tsft K30 view). Keeper cell blocked; OUT must go to _trash
  before any rerun. Options sent to the user: (a) eval anyway + annotate, (b) fp32 HF reference diagnostic first, (c) N/A.
### 03:1x CDT 10-23 — AD K500 + SmartAD K30 converged, evals running; K100/K300 rungs -> 2 GPUs (plan 9ba549f)
- adc-k500-e64-lr5e-5 train 47156075 COMPLETED 6:12: 64 epochs x 63 updates, L_last 0.00004 = L_min, no collapse; e32 L_last
  0.0003 -> delta < 0.01 -> AD K500 converged rung = 5e-5 e64. Eval gate opened -> adc-k500-e64-lr5e-5-eval 47241702 (SAG).
- SmartAD K30 1e-4 ladder L_last: e2 0.1490, e4 0.1190, e8 0.0620, e16 0.0086, e32 0.0001, e64 0.0000 (collapse flag ep11),
  e128 0.0000. First pair with delta < 0.01 = e16->e32 (0.0085) -> converged rung e32 (valid, no collapse); eval gate
  opened -> smc-k30-e32-lr1e-4-eval 47241704 (greedy). 2-GPU K30 trains took 3-60 min each.
- smc-k500-e64-lr5e-5-train 47239796 RUNNING since 02:51 CDT (yd24f, 4 GPU, 8,000 updates, ~7 h).
- Untrained SmartAD K100 (e32/e16/e8/e4/e2) and K300 (e16/e4/e2) 1e-4 rungs moved to 2 GPUs (global batch b'=4 unchanged,
  same as K30); their 4-GPU preps (identity gpus=4) -> _trash/smc_prep_4gpu_1023 on hpg, preps rerun (state.bak.k100k300_2gpu).
  Done 4-GPU rungs (K100 e64, K300 e8/e32) keep their 4-GPU training.
### 00:2x CDT 10-23 — GPU stages group-only (plan 463450e); 8 stuck dept trains withdrawn
- dept QOS 127/128 CPUs held by other users' CPU-only jobs (rx21a 64, lz23b 50; 1-9 days left; first 8 CPUs free ~06:45 CDT).
  Our 8 dept trains (smc-k300-e16/e4/e2, smc-k100-e32/e16, smc-k30-e64/e32, smc-k500-e64-lr5e-5) pended in QOSGrpCpuLimit
  while holding 8 of the 10 job slots -> cancelled (all PENDING, never started; state.bak.grouponly.json), active=None so the
  keeper resubmits. Limits now max_gpus = max_group_gpus = 6 -> every GPU stage goes to yd24f or waits. Burst QOS for preps
  rejected (sbatch --test-only start estimate ~2 days). Keeper submitted 7 preps (K30 e16-e2, K100 e4/e2, K500 5e-5 e32).
- Running: adc-k500-e64-lr5e-5 47156075 (yd24f, 4 GPU; epoch 34/64 at 00:15 CDT, ~6 s/update, ETA ~03:20 CDT; e32 L_last
  0.0003 so e64 converges, no e128), smc-k30-e128 47217342 (yd24f, 2 GPU, ep 88/128).
- Remaining SmartAD 4-GPU train work ~19 h (K500 5e-5 ladder ~14 h): at 6 group GPUs table ~00:30 CDT 10-24; asked the user
  whether to raise the group cap to 8 (yd26's jobs all held) -> ~15:00 CDT 10-23. Awaiting answer.
### 22:3x CDT 10-22 — SmartAD K500 -> 5e-5 fallback; queue reorder (plan 4329053); LONI not used for SmartAD
- SmartAD ladder (hpg, L_last = last complete epoch token-mean): K300 e8 0.1016, e32 0.0001 (valid; collapse at ep6, annotated);
  K500 e8 0.0686 valid, e32 4.7861 INVALID (L_min 0.1012, collapse at ep7) -> pre-registered 5e-5 fallback for SmartAD K500:
  cells smc-k500-e{64,32,16,8,4,2}-lr5e-5 added; pending 1e-4 K500 e2/e4/e16 (47199126, 47160590, 47158097) cancelled +
  superseded in state. K100 e64 (1e-4) done (L_last 0.0000, collapse flag at ep2).
- Bottleneck = queue, not compute: sealed-prep trains 3.2-3.4 s/update (K500 e32 3:31; K100 e64 1:29) while 4-GPU dept jobs sit
  in QOSGrpCpuLimit; group account (yd24f) had only our adc-k500-e64 (4 GPUs) -> 2 free GPUs under our 6-GPU cap = K30's
  2-GPU rungs. Order now: K500 5e-5 e64, K30 ladder, K100, K300, K500 rest. smc time limits resized (3.5 s/update x 1.6
  + 45 min; 2-GPU 7 s).
- LONI for SmartAD dropped (told the user it was the plan at 22:2x): LONI A100 runs 25-45 s/update vs 3.3 s on hpg, and the
  SmartAD selected collection lives only on hpg (LONI file quota 3.53M/4M). SmartAD idle guard re-armed (pid 2470694).
### 22:1x CDT 10-22 — SAD row complete; SmartAD paper-recipe column complete
- Reruns after the off-home fix all clean (0 error episodes, serving checks passed): SAD K500 x2 (converged rung, full-parameter,
  greedy) 116/140; SmartAD paper recipe (2e-4, 2 epochs, greedy) K300 96/140, K100 87/140.
- Table so far (dev140, base 62). Converged: tsft 88/112c/124c/122; AD 109/107/4c/(K500 5e-5 e64 training, hpg 47156075);
  SAD 49/109/127/116; SmartAD ladder running (K300/K500 e8+e32 trained; K30 not started). Paper recipe: tsft 10/0*/109/108;
  AD 73/95/84/0*; SAD 49/0*/127/123; SmartAD 77/87/96/112. (c = collapsed mid-run, annotated under rule A; * = diverged)
### 20:3x CDT 10-22 — hpg $HOME full -> three evals reset_error; fixed (plan 8a6f26f)
- hpg /home/xc25.fsu 40G/40G (29G = ~/.cache/vllm/torch_compile_cache, written by every vLLM start; full at ~19:34 CDT).
  alfworld.info mkdirs ~/.cache/alfworld at import -> on compute nodes OSError [Errno 28] (reproduced via srun --overlap in
  47201466; the login node does not hit it) -> every dev140 episode reset_error, 0 steps: sadc-k500-b2-eval 47197395,
  smc-k300-e2-lr2e-4-eval 47199950, smc-k100-e2-lr2e-4-eval 47200835 (not model results).
- Fix: every keeper stage exports ALFWORLD_DATA=hq/tools/alfworld-data-empty, VLLM_CACHE_ROOT=hq/tools/vllm-cache,
  TRITON_CACHE_DIR=hq/tools/triton-cache, VLLM_NO_USAGE_STATS=1 (all on /blue); verified 6/6 resets in a job. Failed smc
  eval outputs -> _trash/eval_reset_error_1022; the three cells unblocked (SAD eval resumes as attempt-001).
- hpg home itself still full (outside the authorized paths; shared by every session): ASKED the user to let ctl clear the
  vLLM compile cache.
- SmartAD paper recipe K500 (smc-k500-e2-lr2e-4, 2e-4, 2 epochs, greedy) dev140 112/140 (80.0%), passed the serving check.
- LONI cold move: 35 dirs verified + deleted by 20:35 CDT. tc-hotpotqa-scan5 VERIFY MISMATCH = equal-size tie in the
  largest-file pick (count/bytes equal; nothing deleted). Tie-break by path fixed (atomic replace; running pid 2017009 keeps
  the old copy) -> RERUN scripts/ops/loni_cold_move.sh after this pass to verify the skipped dirs.
### 19:0x CDT 10-22 — back after a ~5 h usage-limit pause (13:5x-19:02 CDT); LONI cold move started
- Pause: no monitoring or replies 13:5x-19:02 CDT. In that window hpg finished smc-k500-e32 train and several smc preps;
  SAD K500 x2 eval never ran (SAD cells behind the SmartAD ladder at the 10-job cap) -> plan 2b75f6d: SAD before SmartAD,
  one pending smc train cancelled to free a slot.
- LONI cold move (user: A + "直接走RAI吧"): scripts/ops/loni_cold_move.sh pid 2017009 (throttle 4 MB/s via
  scripts/ops/throttle_pipe.py; per dir: tar LONI -> rai -> hpg /orange/yd24f.fsu/xc25.fsu/tc-cold/loni_20261022, verify
  count + bytes + sha256 of the largest file per immediate subdir, then rm the LONI source; resumable via
  logs/loni_cold_move.done; log logs/loni_cold_move.log). Tested on tc-rtd-alf-projection-jsonfix-20260920 (verified, deleted).
  ~2.9 TB at 4 MB/s ~ 8-9 days (deadline: LONI purge eligibility ~Nov 11).
### 13:5x CDT 10-22 — LONI cold-move list
- Move list reports/engineering/loni_cold_move_list_1022.txt: 73 hq/ dirs (2,257 GB, 775k files) + hq/tc-alignment/results
  (610 GB) = ~2.9 TB (corrected from 2.2 TB in the channel). Keep: ICML27_data, ICML27_runs, tc-alignment (.venv, .venv-fla,
  envs), hf-cache, scripts. Waiting for the user: Globus (user explained; LONI collections are LSU HPC, authorize with the
  LONI login) vs rai tunnel (~5 days at 6.6 MB/s).
### 13:4x CDT 10-22 — trajectory SFT row complete
- Converged trajectory SFT: K30 88 (5e-5 e64), K100 112 (1e-4 e64, collapse annotated), K300 124 (1e-4 e64, collapse annotated),
  K500 122 (5e-5 e64). SmartAD paper K30 77. hpg back 13:44 CDT; /orange 25 TB free.
### 13:4x CDT 10-22 — user chose A (old-phase LONI data -> hpg /orange); hpg ControlMaster dropped
- User: "A" = move the old-phase LONI data (~2.2 TB) to hpg /orange (copy -> verify -> delete). Measured: LONI->rai 39 MB/s,
  rai->hpg (tunnel via the user's Mac) 6.6 MB/s -> ~4 days; rai cannot stage 2.2 TB (1.2 TB free, keep >= 1 TB). ASKED: Globus
  (user logs in once; LONI QB -> HiPerGator /orange/yd24f.fsu/xc25.fsu/tc-cold/loni_20261022/) vs the rai tunnel (~4 days).
  Assumed: tonight's purge of the 2025 non-project data is left alone.
- hpg ControlMaster dropped right after my 300 MB throughput test (13:4x CDT); notified once (hpg_tunnel_notified).
### 13:3x CDT 10-22 — LONI /work purge scan (user request)
- /work/xueqic: 4.54 TB, 2.73M files (scan /work/xueqic/hq/scan_usage_1022.tsv). Purge-eligible tonight (atime > 60 d):
  1.13 TB / 323k files, all 2025-era non-project data: hf_cache 704 GB (Llama-2-7b, Llama-3.1-70B, Qwen3), .cache 389 GB
  (huggingface 347 GB, pip 14 GB, vllm 20 GB), miniconda3 33 GB, att 7 GB, ability 0.1 GB. Nothing under hq/ is eligible.
- Project data (hq/, Sep-Oct 2026, ~3.4 TB) safe until ~mid-Nov. Keep on LONI: hf-cache 341 GB, tc-alignment/.venv,
  ICML27_runs, ICML27_data (100 GB, 1.25M files; tar after the grid). Old phase (Sep RTD/TVD/HotpotQA/ALF) ~2.2 TB:
  tc-alf-vllm 849, tc-alignment/results 610, tc-hotpotqa-* ~830, archive 296, tc-rtd-* ~150 GB.
- ASKED USER: (1) OK to let tonight's purge take the 2025 data (not this session's); (2) old phase: A move to hpg /orange
  (copy -> verify -> delete), B delete, C keep part (e.g. archive). Nothing deleted or moved yet.
### 13:1x CDT 10-22 — tsft K300 converged 124
- tsft K300 (1e-4 e64; collapse at epoch 35, annotated under rule A) dev140 124/140. LONI /work purge scan running
  (/work/xueqic/hq/scan_usage_1022.sh -> scan_usage_1022.tsv); user asked to clean up LONI (purge tonight 23:00).
### 13:0x CDT 10-22 — USER DECISION: rule A; full table requested
- User (13:03 CDT): rule A = keep the registered convergence rule (last complete epoch + cross-rung + delta < 0.01); mark
  cells whose training collapsed mid-run; priority = finish training, then the full 4-method x 4-K table + analysis.
  "我要在所有k上几个方法（轨迹SFT，AD，SAD，SmartAD）的完整表现表格，你得到以后给我".
- Cancelled hedge 5e-5 ladders (LONI adc K100/K300, tsft K100/K300; hpg pending twins 47091122/23, 47133998, 47134000) and
  held 23 hedge cells in the keeper. Still needed: adc K500 5e-5 e64 (hpg twin + LONI 1124132), SmartAD paper + ladder,
  evals tsft K300 e64 (1e-4), tsft K500 e64 (5e-5), SAD K500 x2, SmartAD K30 e2 (gate opened).
- smc-k500-e32-lr1e-4 collapsing at epochs 6-8 (0.088 -> 0.467 -> 3.98 -> 4.23); rule A decides at its last epoch.
- ETA told to user: full table ~tomorrow (10-23) around noon CDT.
### 12:1x CDT 10-22 — K300/K500 trajectory SFT converge under the registered rule; collapses at 5e-5 too
- tsft K300 converged at e64 (1e-4) and tsft K500 at e64 (5e-5) by the registered rule; collapse test flags tsft K300 e32+e64
  (1e-4) and tsft K500 e32 (5e-5, ep4 0.189 -> 0.854). Eval gates opened for both e64 rungs (reported number follows the user's
  rule decision, not the dev result). adc K500 5e-5 needs an e64 rung (16->32 delta 0.059). Still awaiting the user on the
  collapse criterion; with it, some ladders would need a further fallback (none registered beyond 5e-5).
### 09:1x CDT 10-22 — sealed prep works; AD K300 converged rung collapsed (4/140); collapse criterion = user decision
- SmartAD GPU start 3 min with the seal -> all smc cells unheld. SAD K500 converged at x2 -> eval gate.
- adc K300 e32 (1e-4) = registered converged rung scored 4/140: mid-run collapse (epoch 5: 0.214 -> 5.02) then re-memorised to
  0.0048; the registered within-rung/cross-rung guard does not catch it. Scan: reported converged rungs adc K100 e64 and
  tsft K100 e64 also collapsed at epoch 13 (scored 107 / 112). ASKED USER whether to add a collapse criterion (any epoch mean >
  2x the lowest earlier mean and > +0.1 -> rung invalid -> 5e-5 fallback). 5e-5 hedge ladders running: adc K100/K300, tsft K100.
### 09:2x CDT 10-22 — sealed prep deployed; one SmartAD cell released for measurement
- main e599953 = merge wt/sealedprep (memoized collection validation; CPU prep publishes a sealed prepared dataset under
  training/prepared/<identity digest>/; GPU train verifies seal + top-level collection files, trains without scans; refuses
  without a valid seal). hpg tree e599953 deployed. Plan 1c8aea2: smc = prep (hpg-milan 8 CPU/32G/4 h) -> train (GPU,
  1 h startup allowance) -> eval. Only smc-k30-e2-lr2e-4 released; 26 smc cells still held. Guard pid 1051472.
- tsft-k300-e64-lr1e-4 now trains on hpg (47066822, 89 updates at 08:15 CDT); dedupe cancelled its LONI twin 1122217.
- adc-k300-e32-lr1e-4 converged (e32): copy re-verifying after my stray-attempt error (see exp_log); gate opens on OK.
### 07:1x CDT 10-22 — AD K300 converged (e32)
- adc K300 converged at e32 (LONI) -> copy + eval gate (sync_and_gate). tsft K300 1e-4 e64 recovering; if it completes valid the
  1e-4 K300 ladder decides (32->64) and the 5e-5 hedge is unused. Timing job 47115431 warm pass still running (2h58m).
### 06:2x CDT 10-22 — SmartAD startup measured: collection scans dominate
- Cold CPU prepare 1h40m; snapshot hashing fixed (0.04 s warm); collection_load repeated ~12x per process (190-910 s each),
  warm barely faster -> a GPU stage would still idle ~1.5 h. Codex wt/sealedprep (pid 853599, prompt
  scratchpad/codex_sealedprep.md): memoize collection validation per process + CPU prepare writes a sealed prepared dataset;
  GPU train verifies the seal (top-level manifests/identities, code, snapshot receipts, examples sha256) and trains from it.
- SmartAD stays held. Warm-prepare pass of 47115431 still running (for the record).
### 04:2x CDT 10-22 — faststart deployed; SmartAD prepare timing
- main fbd65bc = + wt/faststart2 (test-only fixes; full suite 2,529 passed, 55 env failures = missing vendored files /
  sandbox Gloo / theory venv). hpg tree fbd65bc deployed (.venv + third_party). Old smc-* trees -> baselines/_superseded/smc-102a35d.
- Timing job 47115431 (hpg-milan, logs/smc_prepare_timing.slurm in the fbd65bc tree): SmartAD K30 paper-cell prepare twice
  (cold, warm) into baselines/smc-k30-e2-lr2e-4. Warm duration ~= GPU-stage startup. If short: smc cells = CPU prep -> GPU train
  (startup allowance from the warm time) -> eval on tree fbd65bc; then unhold.
### 02:2x CDT 10-22 — faststart merged (main 6186263), tests follow-up
- Guard cancelled smc-k30-e2 / smc-k30-e64 at 52 min without updates (01:42/01:46 CDT); no smc running; all smc held.
- main 6186263 = merge wt/faststart (persistent receipts for snapshot/checkpoint hashes; CPU --mode prepare for smartad/kang/adftp;
  startup_timing.jsonl). 19 tests in tests/test_convergence_training.py fail (prepare now runs the real subprocess under
  monkeypatched tests) -> Codex wt/faststart2 (pid 438383) fixes tests + full-suite run. Launchers default the receipt cache to
  $HOME/.cache (small hpg quota): always set SED_HASH_CACHE_DIR=$B/../.sed-hash-receipts.
- Next: deploy, measure cold vs warm prepare on hpg-milan for one smc cell (warm = GPU startup), then smc cells =
  prep (CPU) -> train (GPU, short startup) -> eval, new output dirs (old smc-* trees have 102a35d identities -> _superseded).
### 01:4x CDT 10-22 — user OK on the SmartAD hold
- smc-k30-e2 / smc-k30-e64 at ~50 min with no updates even with the warm receipt cache -> guard cancels them; SmartAD fully
  paused until wt/faststart is merged, deployed and verified on hpg (CPU prepare stage + fast GPU start). Told the user.
### 01:1x CDT 10-22 — ROOT IDLE-GPU CANCEL (account risk): SmartAD held
- smc-k30-e128-lr1e-4 47095421 CANCELLED by root at 72 min (GPUs idle in startup hashing). 25 smc cells held
  (blocked "held 10-22 01:2x CDT ..."); 2 running (smc-k30-e2 47100227, smc-k30-e64 47101414) watched by guard pid 259291
  (cancel at 50 min without updates). smc-k30-e2 read only 4.6 GB at 25 min (persistent cache warm) vs 61 GB before.
- Codex wt/faststart (pid 261081, prompt scratchpad/codex_faststart.md): artifact_hashes via receipts, CPU prepare stages
  for smartad/kang/adftp, startup timing log. Then: add CPU prepare stages to the plan, unhold smc.
### 00:2x CDT 10-22 — SmartAD startup ~1 h
- smc-k30-e2 timed out before its first update (startup: pipeline prepare ~52 min, 61 GB read, then rank-0 prepare_data).
  Plan 1e67101: smc train limits = 3 h + 10 s/update x 1.6. smc-k30-e128-lr1e-4 47095421 running (watch when updates start).
- Follow-up (engineering, Codex once timings are known): reuse snapshot artifact hashes and the persistent receipt cache
  across the pipeline process and torchrun rank 0 so each SmartAD job does not re-read tens of GB.
### 23:1x CDT 10-21 — K30/K100 rows complete (tsft, AD, SAD)
- Converged column (dev140, base 62): tsft K30 88 (5e-5 e64) / K100 112 (1e-4 e64); AD K30 109 / K100 107 (1e-4 e64);
  SAD K30 49 (x1) / K100 109 (x0.5) / K300 127 (x1). Pending: tsft K300/K500 (5e-5), AD K300 (1e-4) / K500 (5e-5),
  SAD K500 x2, SmartAD (all K).
### 22:2x CDT 10-21 — eval attempt-path fix; SmartAD training gate open
- adc-k100-e64-lr1e-4-eval failed (attempt-000 path; checkpoint is in attempt-001). TRAIN_ATTEMPT override map in
  queue_plan_1019.py; RULE: before opening an eval gate for an hpg-trained cell, verify the adapter path exists.
- gates/smartad_selected opened 22:13 CDT -> smc cells (4 paper e2 + 23 lr 1e-4 rungs) submit as slots allow.
- tsft-k30-e64-lr5e-5 copied to hpg, eval gate open.
### 22:1x CDT 10-21 — tunnel back; SmartAD selected; more 1e-4 divergence
- Tunnel back ~22:0x CDT (user). SmartAD merge COMPLETED 21:19 CDT -> gate smartad_selected after keeper CELL_DONE.
- Converged: adc K100 e64 (eval gate), tsft K30 5e-5 e64 (copy + gate). tsft K500 1e-4 e32 INVALID (4.30) -> 5e-5 fallback;
  tsft K300 1e-4 e64 diverging -> 5e-5 hedge. Plan fea2ac6. 1e-4 has now failed in tsft K30, tsft K500, adc K500 (and
  tsft K300 e64 in progress): long rungs at near-zero loss blow up; 5e-5 has been stable so far.
### 19:1x CDT 10-21 — hpg tunnel down (reverse tunnel port 2222 times out)
- Notified once (scratchpad/hpg_tunnel_notified): restart autossh on the Mac, then `ssh -fN -M hpg` on rai. hpg jobs keep
  running (smartadselc-merge retry 47061793, ladder rungs); LONI unaffected. Gate watcher 1904567 (smartad_selected) and the
  dedupe loop resume when hpg is reachable.
### 17:2x CDT 10-21 — long rungs prefer hpg
- dedupe_ladder.py: for K>=300 and E>=32 keep the hpg twin queued even when LONI starts first; cancel LONI once hpg logs
  updates (hpg ~6x faster). Restored hpg twins: tsft-k300-e64, tsft-k500-e32 (LONI ~50% done), adc-k300-e32. Kept LONI-only:
  tsft-k300-e32, adc-k300-e16 (~2 h left). Cancelled superseded adc-k500-e32-lr1e-4 (LONI 1119727; 1e-4 ladder invalid).
  Plan 9f048e9: hpg train for adc-k500-e16/e32-lr5e-5 (dedupe covers them).
- smartadselc-merge retry 47061793 running (8 h limit).
### 16:2x CDT 10-21 — SAD K300 converged at x1; SmartAD merge retry
- SAD convergence complete except K500: K30 x1 (49), K100 x0.5 (109), K300 x1 (127); K500 x2 queued (x0.5->x1 delta 0.162).
- smartadselc-merge hit its 2 h limit in the final verification (full-TRAIN support re-hash). Retry: 8 h, 8 CPUs, persistent
  SED_HASH_CACHE_DIR (also added to SmartAD train stages). Gate watcher 1904567 still waiting for CELL_DONE.
### 15:1x CDT 10-21 — results
- tsft K100 converged (e64) 112/140. SAD K300 x1 (paper) 127/140 (x0.5 training; if x0.5->x1 converges, 127 is also the
  converged entry). SmartAD merge 47046998 at 1:17 of 2:00, publishing the collection (selection json written 14:44 CDT).
### 14:4x CDT 10-21 — SmartAD ladder enabled
- main 102a35d = merge wt/smartadepochs (SmartAD trainer accepts positive epochs; DEVIATIONS registered; 235 tests pass).
  hpg tree 102a35d deployed (.venv + third_party). Plan 1f360cd: 27 smc cells (4 paper 2e-4/e2 + 23 lr 1e-4 rungs, K30 e2..e128,
  K100 e2..e64, K300/K500 e2..e32), all train on 102a35d, gated on gates/smartad_selected (watcher pid 1904567).
  Time limits assume 10 s/update (no measurement yet) -> tighten from the first e2 runs; consider LONI twins.
- Merge 47046998 running 39 min (CPU).
### 14:3x CDT 10-21 — ladder decisions
- tsft K100 converged at e64 (copy + gate). AD K500 1e-4 e4 rung failed (1.856) -> 5e-5 fallback ladder (LONI 1122212-16;
  eval-only gated cells in plan 9afb4fc). tsft K300/K500 need e64: LONI 1122217/18 + hpg train stages (dedupe HPG dict updated).
- Codex wt/smartadepochs (pid 1901568): SmartAD trainer guard epochs==2 -> positive epochs.
### 14:3x CDT 10-21 — SmartAD selection merged next; training cells staged
- Selection shards s0..s3 done (13:43-13:53 CDT); merge 47046998 running (CPU). Gate watcher scratchpad/gate_smartad_selected.sh
  pid 1904567 opens gates/smartad_selected when the merge is CELL_DONE and collection + K30/K100/K300 manifests exist.
- Plan 7e4358e: smc-k{500,300,100,30}-e2-lr2e-4 (paper recipe; tree 3318b54; train 4 GPUs (K30: 2), greedy eval with
  check prompts from the tsft support view). The 1e-4 ladder is staged behind SMC_LADDER_TREE: fsdp_train still pins
  SmartAD to epochs == 2 -> Codex wt/smartadepochs (pid 1901568) relaxes it; then set SMC_LADDER_TREE to the merge commit.
- hourly: SAD K300 x0.5 train + x1 eval in preflight (disk reads 39G/56G), not idle-GPU problems.
### 13:1x CDT 10-21 — hourly
- sadc-k300-b1-train COMPLETED (2:16, 4 GPUs) -> gate eval_sadc-k300-b1 opened (paper column); x0.5 46930070 pending.
- SmartAD selection 4 shards running (~1 h). adc-k300-e32 / adc-k100-e32 lr1e-4 run on LONI (hpg twins cancelled by dedupe).
### 12:1x CDT 10-21 — first converged-column results
- Converged column (dev140, base 62): AD K30 (1e-4, e64, SAG) 109; SAD K30 (x1) 49; SAD K100 (x0.5) 109.
- SmartAD pool verified on hpg 12:12 CDT (889,928 files); selection shards submitted 12:13 CDT: smartadselc-s0..s3of4 47038188/90/93/94 (merge gate via gate_smartad_scored.sh pid 884142).
- sadc-k500-b2 prep done, train 47035772 pending.
### 11:2x CDT 10-21 — tunnel back (11:16); results snapshot
- SAD K100 x1 (paper) 0/140 (collapsed). SAD K100 converged at x0.5 (x0.25 -> x0.5 delta 0.0008): eval gate eval_sadc-k100-b05.
  SAD K500 x0.5 -> x1 delta 0.162: not converged; sadc-k500-b2 added (plan 28c45d7). SAD K300: x1 / x0.5 pending.
- LoRA ladder: adc K30 converged at e64 (32->64 delta 0.0004) -> gate eval_adc-k30-e64-lr1e-4. Others still training
  (ladder script scratchpad/probe/ladder.py; ledgers rsynced into scratchpad/probe/lad).
- SmartAD copy re-verification running (watcher bsyvtc6qs reran sync_smartad_chain_hpg.sh). Keeper was in post-SSH-fail
  backoff (heartbeat 65 min), resumes by itself.
### 10:1x CDT 10-21 — hpg tunnel down (ControlMaster gone; Duo needed)
- Notified the user once (scratchpad/hpg_tunnel_notified): run `ssh -fN -M hpg` on rai. Submitted hpg jobs unaffected.
- SmartAD pool rsync attempt 3 finished, but verification ran with hpg unreachable -> MISMATCH with empty remote
  (local 889,928 files / 9,514,767,857 bytes / 83f5dce1f79961c4). After the tunnel is back: rerun
  scratchpad/sync_smartad_chain_hpg.sh (rsync is incremental) -> gate smartad_chain_on_hpg.
### 09:2x CDT 10-21 — hourly
- hpg: adc-k500-e8-lr1e-4 and adc-k300-e8-lr1e-4 train COMPLETED (--nodes=1 fix confirmed). sadc-k500-b05-train running
  (4 GPUs 97-100%; a 0% sample was a sync point; 30 updates). SmartAD pool rsync attempt 3 in progress (pid 875919).
### 09:1x CDT 10-21 — SmartAD collection COMPLETE; selection set up
- chain_k500_s0_r2 COMPLETE (launcher exit 0 14:02Z = 09:02 CDT): 500 tasks x 10 candidates = 5,000; 3,147 correct; tasks by
  #correct: 0:40, 1:34, 2:30, 3:29, 4:27, 5:31, 6:28, 7:40, 8:49, 9:57, 10:135. api_attempts 102,387 (new teacher calls 97,859),
  amendment bounds 943 + 82, effective cap 206,025. 13G.
- Copy to hpg: scratchpad/sync_smartad_chain_hpg.sh pid 875919 (first 2 attempts failed: parent dir missing on hpg; created),
  gate smartad_chain_on_hpg on verified match.
- hpg tree 3318b54 deployed (with .venv + third_party). Plan 1215afc: smartadselc-s{0..3}of4 (1 GPU, 48G, 3 h; gate
  smartad_chain_on_hpg) + smartadselc-merge (CPU; gate smartad_chain_scored, opened by scratchpad/gate_smartad_scored.sh pid
  884142 after 4 CELL_DONE). Output reports/smartad-selection/chain_k500_s0_r2. Then: SmartAD 1e-4 ladder + evals.
### 08:2x CDT 10-21 — SmartAD collection finishing
- Old launcher 652148 finished; third launcher 1429312 (tree 3318b54) running 32 shards, 0 failures; 4,998/5,000 done; the 2
  operator-interrupt slots are resampling in resampled/attempt-1. Then the deferred candidate-01 pass + candidates.json.
  Watcher b0vr45wn9 waits for candidates.json status=complete.
- Next: copy chain_k500_s0_r2 (13G) to hpg (+ LONI), SmartAD selection (1 GPU per shard, --shard-index/--shard-count, then
  --merge-scores on CPU; SMARTAD_PATH_MAP for the relocated pool), then the 1e-4 training ladder, then gated evals.
### 08:1x CDT 10-21 — paper-recipe column nearly complete
- AD K500 paper recipe 0/140 (training diverged at update ~36, 2e-4). SAD K30 x1 (= converged) 49/140.
- Paper-recipe column (dev140, base 62): tsft K30 10 / K100 0* / K300 109 / K500 108; AD K30 73 / K100 95 / K300 84 /
  K500 0*; SAD K30 49 / K100 (x1 diverged*, eval pending) / K300 (pending) / K500 123. (* = training diverged)
- adc-k500-e16-lr1e-4 COMPLETED on hpg; sadc-k100-b025-prep done.
### 07:2x CDT 10-21 — SmartAD progress
- 4,721/5,000 candidates done; launcher 652148 has 11 live shards; third launcher 1429312 (tree 3318b54) still waiting on
  .smartad.launcher.lock (expected). Watcher b7cecxbq9 hit its 2 h cap; check SmartAD at each hourly check instead.
### 07:1x CDT 10-21 — first SAD result
- SAD K500 x1 (paper budget; full-parameter; greedy) dev140 **123/140** (base 62; tsft K500 paper 108; AD K300 paper 84).
- 3 more AD hpg trainings failed from the pre-fix node split (submitted 06:13-06:16 CDT); unblocked; post-fix submissions
  verified NumNodes=1. dedupe_ladder.py: an hpg twin counts as started only after it logs a training update.
### 06:1x CDT 10-21 — hpg multi-node split fixed; AD K500 eval check prompts
- 3 AD 1e-4 hpg trainings FAILED (invalid device ordinal): --gpus=4 without --nodes=1 let SLURM split jobs over 2 nodes
  (torchrun --standalone). Plan a47c93d: --nodes=1 on every hpg GPU stage (all 617 stages checked). Split running
  adc-k300-e16-lr1e-4 cancelled. Cells cleared; LONI twins resubmitted (adc-k500-e8/e32, adc-k300-e16/e32: 1119726-29) because the
  dedupe loop had cancelled them when the split hpg jobs started.
- adc-k500-e2-eval FAILED in the HF/vLLM check (strict encoding of 8 VERIFIED prompts hit the empty reply); AD K500 evals use
  the K300 view for check prompts; rerun queued.
- Completed on hpg (1e-4): adc-k30-e32/e64/e128, tsft-k300-e8. SAD K500 x1 eval running (passed check). SAD K100 x1 eval submitted.
### 05:2x CDT 10-21 — 1e-4 not stable at K30 (trajectory SFT): 5e-5 fallback triggered
- tsft-k30-e4-lr1e-4 (completed) FAILED the within-rung guard (epoch means 0.399 0.248 0.377 0.445; max gnorm 1968); e32 and
  e128 also spiked. Per the pre-registered rule, the tsft K30 ladder is rerun at 5e-5: LONI 1119668-74 (tsft-k30-e{2..128}-lr5e-5,
  tree c6b15b2). Cancelled superseded tsft-k30-e32/e128-lr1e-4 (LONI 1119423/1119425). b' = 3 at K30 (noisier updates).
- adc K30 1e-4: e64/e128 on hpg recovered big spikes (guard ok so far); e2..e32 pending on LONI. Other K: no failed rung yet.
- AD paper recipe K500 (adc-k500-e2) COMPLETED on LONI; copy + gate watcher pid 1433503.
### 05:2x CDT 10-21 — second SmartAD amendment applied; third launcher queued
- main 3318b54 = merge wt/smartadresume3: amendment smartad_operator_interrupt_20261021_v1 (teachers/smartad_operator_interrupt.py)
  for exactly 501b2815/candidate-10 + 581264b0/candidate-09: archived (8 calls void, 2 in-flight with unknown usage, never
  refunded), fresh 41-cap resample (<= 82 calls). 87 tests pass. Applied offline (recovery/smartad_operator_interrupt_20261021_v1/).
- Third launcher queued: runtime ../ICML27_run_3318b54, scratchpad/run_smartad_resume3.sh pid 1429312, log
  logs/chain_r2_smartad_resume3.log. It waits on .smartad.launcher.lock until launcher 652148 (30 shards) ends by itself, then
  runs shards 10/13's remainder + the deferred candidate-01 pass + finalization. Do not signal 652148/652151.
### 04:4x CDT 10-21 — SmartAD shards 10/13 stopped on the 2 interrupted candidates
- Relaunch f796565 at +30 min: 30 shards fine (2,099/5,000 candidates done, 119 active ledgers); shards 10 and 13 failed with
  TeacherCallError at 501b2815/candidate-10 and 581264b0/candidate-09 (no calls): by design _candidate never re-enters a
  started attempt and require_execution refuses the unresolved episode — recover() alone cannot resume them.
- Codex wt/smartadresume3 (pid 1246148, prompt scratchpad/codex_smartadresume4.md): second amendment
  smartad_operator_interrupt_20261021_v1 for exactly these 2 slots (archive, fresh 41-cap resample, <= 82 calls, 2 in-flight
  requests usage unknown). After it lands: when launcher 652148 finishes (it holds .smartad.launcher.lock), apply + relaunch
  --resume for shards 10/13's remainder, the deferred candidate-01 pass and finalization. Do NOT signal launcher 652151.
### 04:2x CDT 10-21 — SAD eval fix deployed
- main b9f9a80 = merge wt/sadeval: full-checkpoint serving check under TRAIN_PYTHON (CUDA hidden) + SAD_EVAL_FRACTIONS.
  hpg tree ICML27_runs/b9f9a80 deployed WITH .venv symlink + third_party (layout identical to fb22f09). Plan 3de1b54: all SAD
  eval stages on b9f9a80 with SAD_EVAL_FRACTIONS=1 (final checkpoint only). sadc-k100-b1 unblocked; gates open:
  eval_sadc-k30-b1 / k100-b1 / k500-b1 (paper-recipe column; K100 x1 is the diverged model, reported as such).
### 04:1x CDT 10-21 — SAD K100 x1 diverged; SmartAD launcher handoff
- SAD convergence: K100 x0.5 (48 epochs, L_last 3e-5, valid) vs x1 (96 epochs): x1 DIVERGED at update 45 (loss 451,
  grad norm 8.3e4, lr ~2e-5 at peak), stuck at 4.22 for 89 epochs -> invalid (within-rung + cross-rung). K500 x1 valid
  (L_last 0.0043, smooth). Added sadc-k100-b025 (x0.25, tree fb22f09; plan d85e0ef): if x0.25 -> x0.5 delta < 0.01, x0.5
  is the converged rung; x1 stays the paper-recipe column, reported as diverged. reports/runs/convergence/sadc-k100-b05_vs_b1.json.
- hpg: sadc-k500-b1 + sadc-k100-b05 train COMPLETED. adc-k30-e128-lr1e-4-train 46965917 running (preflight hashing at check).
  Dedupe: tsft-k30-e128/e32 -> LONI (hpg twins cancelled by our dedupe), tsft-k30-e64 / tsft-k300-e16 / adc-k30-e128 -> hpg.
  LONI now runs 9 of ours (tsft K30 e4..e128, K100 e2/e4/e8, adc-k500-e2).
- SmartAD: relaunch (652148) was blocked by the old launcher's .smartad.launcher.lock. I SIGTERMed old launcher 218145 —
  its handler terminated its last 2 shards (10, 13) mid-episode (my error; memory feedback-read-signal-handlers-before-kill):
  2 counters halted (501b2815/candidate-10 at 2 attempts, 581264b0/candidate-09 at 6) -> teachers.recover
  --reason interruption_reviewed (no refund). Old launcher hung in futex_wait -> kill -9 after all children reaped. The
  relaunch then started 32 shards (~04:2x CDT). Watcher checks failures at +30 min.
### 03:3x CDT 10-21 — checks
- LONI tsft-k30-e2-lr1e-4 1119419 running: update 1 lr 0.0001 (override plumbing verified), loss 0.87.
- LONI adc-k500-e2 1119418 (paper recipe) running past startup: lr 2e-4; provenance empty_visible_turns_masked 1 at
  look_at_obj_in_light-Pillow-None-DeskLamp-320 step 15 (as scanned).
### 03:2x CDT 10-21 — AD paper-recipe results; two failures fixed
- AD paper recipe (FTP data, 2e-4, 2 epochs, SAG), dev140: K30 73, K100 95, K300 84 (K500 rerun 1119418 queued on LONI).
- tsft-k30-e128-lr1e-4-train 46963841 FAILED (9 s, empty log): hpg tree c6b15b2 lacked .venv symlink + third_party/ (git
  archive excludes untracked); fixed in place, cell unblocked. Deploy rule: hpg trees need `.venv ->
  $B/.venv` and third_party/ copied (LONI trees do not: LONI env sets the pythons).
- sadc-k100-b1-eval 46962140 FAILED: tools.check_full_serving --endpoint runs under ENV_PYTHON (no torch/transformers).
  Latent since affb477; no SAD full-checkpoint eval has passed on hpg. Codex wt/sadeval (pid 664502): run it with
  TRAIN_PYTHON (CUDA hidden) + optional SAD_EVAL_FRACTIONS (default all five checkpoints). eval_sadc-k30-b1 gate held
  (gates/_held_eval_sadc-k30-b1) until the fix; sadc-k100-b1 blocked in keeper state.
### 03:2x CDT 10-21 — SmartAD r2 relaunched on the race fix
- main f796565 = merge wt/smartadresume2: after COMPLETE, amend() verifies the frozen plan only (no classify while shards run);
  45 resume tests pass. Status before relaunch: completed 1,238 + imported 265, restart 21 (2 of 23 already done),
  in_progress 2, never started 3,474, refused 0.
- Relaunched: runtime tree ../ICML27_run_f796565, scratchpad/run_smartad_resume2.sh pid 652148, log
  logs/chain_r2_smartad_resume2.log. The 2 still-alive shards of the first launcher (218145) keep their shard locks; the
  new shards with those indices wait for them. Watcher checks shard failures 15 min after start.
### 02:5x CDT 10-21 — first paper-recipe results
- Trajectory SFT, paper recipe (lr 2e-4, 2 epochs, greedy), dev140 (base 62/140): K30 10, K100 0 (diverged training,
  78% inadmissible), K300 109, K500 108. AD paper-recipe evals K30/K100/K300 running (46958579/52/51); SAD K100 x1 eval 46962140.
- SAD K30 converged at x1 (x0.5 vs x1 delta ~1e-8); eval gate eval_sadc-k30-b1 opened.
- dedupe_ladder.py fixed (ls rc=2 read as 'loni unreachable'). tsft-k500-e32-lr1e-4-train 46962401 submitted on hpg (QOSGrpGRES).
### 02:4x CDT 10-21 — SmartAD r2 resumed (amendment)
- main 9d909b0 = merge wt/smartadresume (bec2b1b): amendment smartad_interrupted_sample_20261021_v1 (teachers/smartad_resume.py,
  tools/amend_smartad_resume.py; disclosure docs/baselines/CHAIN_COLLECTION_RECOVERY_1020.md): the 23 candidates whose r1
  attempt hit the credited quota 429 and whose r2 retry exhausted the inherited shortened cap are archived (both attempts
  void: r1 701 calls, r2 265 calls) and resampled once from step 0 with cap 41 (<= 943 calls). 352 tests pass.
  Applied offline 02:4x CDT (recovery/smartad_interrupted_sample_20261021_v1/{plan,complete}.json); dry run: completed 1,172 +
  imported 265, restart 23, never started 3,540, refused 0, upper bound 146,083 calls.
- Resume launched on runtime tree ../ICML27_run_9d909b0: scratchpad/run_smartad_resume.sh pid 218142, log
  logs/chain_r2_smartad_resume.log, 32 shards.
  BUG: each shard's collect() re-runs amend() -> classify(); shards starting later met candidate dirs just created by other
  shards and raised "Not an unfinished candidate in the recovery classification" -> 23/32 shards exited at startup (no calls),
  8 run (6,10,13,15,16,20,26,29). Codex wt/smartadresume2 (pid 555720, prompt scratchpad/codex_smartadresume3.md): skip
  classification once COMPLETE exists (verify archives only). Then relaunch the 23 failed shard commands on the fixed tree.
- hpg ladder twin: plan 30eb228 adds hpg train stages (tree c6b15b2) for each K's 3 longest 1e-4 rungs, tsft before adc;
  dedupe loop scratchpad/dedupe_ladder_loop.sh pid 224983 (logs/dedupe_ladder.log). LONI start estimate 15:41 CDT (gpu2 full).
  Gate adc_ready opened (AD r2 on hpg verified).
- Credits: settled since the $20 top-up (01:12Z) ~138M input (91% cached) + 2.4M output (r2 cells incl. imported rows) —
  about 1/4 of the top-up by the 20:1x estimate; remaining ~100k calls expected (~3.8k input / 70 output per call). Margin
  is thin; asked the user to consider topping up.

### 02:0x CDT 10-21 — stab merged (main c6b15b2), 1e-4 ladder submitted on LONI
- main c6b15b2 = merge wt/stab (6104f0d): KANG_/ADFTP_/SMARTAD_ LR+WARMUP overrides, within- + cross-rung divergence check,
  AD-FTP `--allow-empty-visible-replies` (verified on real data: K500 masks exactly 1 step, K300 0). 193 CPU tests pass.
  Deployed LONI /work/xueqic/hq/ICML27_runs/c6b15b2 and hpg ICML27_runs/c6b15b2 (git archive + GIT_COMMIT).
- LONI: adc-k500-e2 rerun 1119418 (paper recipe, c6b15b2; failed run moved to baselines/_superseded/adc-k500-e2-b086c72-emptyreply).
  1e-4 ladder 1119419-1119469 (46 jobs; script /work/xueqic/hq/loni_submit_ladder.sh, cells <m>-k<K>-e<E>-lr1e-4):
  tsft + adc, K30 e2..e128, K100 e2..e64, K300/K500 e2..e32. Cancelled superseded 2e-4 tsft-k500-e8 (diverged) / tsft-k300-e16.
- Paper-recipe adapters copied to hpg + verified (logs/sync_adapters_loni_hpg.log): tsft-k30/k100/k300/k500-e2, adc-k30-e2,
  adc-k100-e2; gate watcher scratchpad/sync_and_gate.sh pid 4045348 (also adc-k300-e2 when done). Keeper submitted evals
  tsft-k500/k300/k100/k30-e2-eval 46955026/29/30/31. Plan (35e5ca6): tsft/adc eval-only on hpg, AD evals read r2, gated
  eval cells for every 1e-4 rung (open only the reported rung). dedupe loop stopped (obsolete).

### 01:4x CDT 10-21 — probe done, lr 1e-4 chosen; AD K500 empty reply; SmartAD r2 stalled
- Probe (K100, 4 epochs = 52 updates, epoch means): 2e-4 paper 0.33/0.26/1.10/1.14 (diverged); 2e-4+10% warmup 0.44/0.39/0.22/0.186
  (1 spike); **1e-4 0.35/0.23/0.19/0.171 (no spike, chosen)**; 5e-5 0.37/0.23/0.20/0.188 (no spike, smallest grad norms = fallback).
  Report wt/stab reports/engineering/lora_stability_probe_1021.md. Old 2e-4 ladder: 6/19 tsft rungs fail the guard; K500 e8
  diverged in epoch 1 (1.84, 4.47 … 0.99) -> cross-rung check added: L_last(2E) <= max(1.1 L_last(E), L_last(E)+0.02).
- Codex wt/stab follow-up (pid 3876882, prompt scratchpad/codex_stab2.md): cross-rung check + `--allow-empty-visible-replies` for
  AD-FTP training only. Cause: adc-k500-e2 (LONI 1119272) FAILED at startup "Template transformed or removed the visible teacher
  reply" — 1/500 trajectories (look_at_obj_in_light-Pillow-None-DeskLamp-320, step 15) has action_raw "" (teacher used the
  1024-token cap on reasoning; parser_error; episode succeeded). K30/K100/K300 views do not contain it (scan: CPU, 500 episodes, 1 bad).
- SmartAD chain r2 exited 1 at 05:59Z: 23/32 shards died ~04:02-04:05Z on BudgetExceeded — each shard's first item was a
  candidate interrupted by the 22:44Z quota 429; r2 restarted it with the inherited remaining cap 41-(old_attempts-1), the
  23 retries made 265 calls (264 ok, 1 parse rejection; in per-retry ledgers, not the top-level usage.jsonl — my first
  'no request made' reading was wrong) and hit 41 before finishing; 9 shards finished. 1,437 done (incl. 265 imported),
  3,540 never started, 23 refused (Codex dry run). Recovery tool refuses r2 as a source (no repeated credits). Codex
  wt/smartadresume (pid 3948286, worktree ../ICML27_wt_smartadresume, prompt scratchpad/codex_smartadresume.md): restart the 23
  fresh with cap 41 (<= 943 extra calls) as an amendment inside r2, dry run only.
- hpg: AD-FTP collection copy verified (269,669 files, gate adftp_chain_on_hpg). sadc-k500-b1-train 46930054 started 01:59 EDT
  (preflight hashing ~20 min, GPUs idle by design); sadc-k30-b1 / sadc-k100-b1 running; k300-b1, k500/k300/k100-b05 pending.
- LONI: adc-k30-e2 COMPLETED; adc-k100-e2 / adc-k300-e2 running; tsft-k500-e8, tsft-k300-e16, tsft-k100-e32 running (2e-4, superseded
  as main numbers by the 1e-4 ladder, kept for the record).

### 00:2x CDT 10-21 — user OK (05:08Z): stabilise all LoRA methods uniformly
- Plan approved: probe (LONI 1119261-63 RUNNING) -> one optimiser setting for trajectory SFT / AD / SmartAD (registered
  deviation; paper recipe kept as the 'paper recipe' column); divergence guard: rung valid iff L_last <=
  max(1.1*L_min, L_min+0.02) over complete-epoch means. Codex wt/stab (pid 3678875): lr/warmup overrides + guard.
- AD paper-recipe e2 rungs on LONI (tree b086c72, reads chain_k500_s0_r2): adc-k30/k100/k300/k500-e2 1119268-72.
- AD-FTP collection verified on LONI (269,669 files, sha ba4d8087…); hpg tar copy failed 05:10Z (tunnel), rsync retry
  pid 3686874 (logs/sync_adftp_chain_hpg.log, gate adftp_chain_on_hpg).

### 00:0x CDT 10-21
- AD-FTP chain_k500_s0_r2 COMPLETE (exit 0, 00:00 CDT): 500 trusted (482 new FTP-verified + 18 fallback), K30/K100/K300 views
  published, 1,372 new calls in r2, api_attempts 13,953 total, 3 cap extensions (120 calls). Collection 4.6G; sync to LONI
  (layout symlink /work/xueqic/hq/ICML27_data/data/verified -> ../verified) and hpg running (scratchpad/sync_adftp_chain.sh
  pid 3648301, gates adftp_chain_on_{loni,hpg}). SmartAD r2 still collecting (pid 2822224).
- Stability probe first submission FAILED in 1 s (my script sourced the LONI env before PROJECT_ROOT); fixed and resubmitted
  1119261-63 (lr 1e-4 / 2e-4+10% warmup / 5e-5; K100 e4); watcher b430kr3qn checks they log updates.
- hpg: tsft-k500-e16-train COMPLETED (1:55); sadc-k30-b05-train COMPLETED (2:11); sadc-k30-b1, sadc-k100-b1 running.

### 22:5x CDT 10-20 — collections recovered
- main b086c72 = + wt/chainrecover (rule chain_quota_void_20261020_v1: hash-verified import, exact credits for the 32
  quota-rejected calls per method, bounded cap_extension (AD-FTP: 3 tasks x 40 = 120 calls) for attempts cut by a cap that
  counted a void call). Dry runs: AD-FTP 460 imported / 40 to run; SmartAD 18 tasks (265 candidates) imported / 482
  tasks (4,735 candidates) to run. 290 tests pass.
- Launched 22:49 CDT (scratchpad/run_chain_recover.sh, run tree ../ICML27_run_b086c72, outputs
  reports/runs/{adftp,smartad}-chain/chain_k500_s0_r2, logs logs/chain_r2_*.log, SED_HASH_CACHE_DIR ~/hq/projects/
  .sed-hash-receipts): adftp pid 2822223, smartad pid 2822224. Watcher bzneg9z3o.

### 22:3x CDT 10-20 — AD recipe instability (asked user, Discord 1562306651661541409)
- Trajectory SFT (AD recipe: LoRA r64/a128, lr 2e-4 linear, no warmup, clip 1.0) diverges intermittently on Gemma-4-12B:
  K100 e2 (loss 0.26 -> 2.17 at upd 14-17, ends 0.5), K100 e4 (-> 2.1 at upd 29-40, last epoch 1.14), K500 e8 LONI
  (0.26 -> 4.4, diverged); spikes recovered in K100 e8 (gnorm 2046), K300 e16 (2228), K500 e4 (1857), hpg K500 e16
  (6026; final ~0.000). Stable: K30 all rungs (e16 0.009), K100 e16 (0.035), K300 e4/e8. Same instability as old
  per-turn single-GPU AD-FTP runs (not the new code). Convergence rule falsely 'converges' K100 e2->e4 (1.145 -> 1.143).
  Convergence so far: K30 e8->e16 0.0798 -> 0.0126 (not converged; e32/e64 running).
- Proposed: stability probe (K100 e4, LONI): lr 1e-4 (1118988), lr 2e-4 + 10% warmup (1118989), lr 5e-5 (1118990) —
  submitted already (training loss only); pick one config for ALL LoRA methods (trajectory SFT, AD, SmartAD) as a
  registered deviation; add a divergence guard to the convergence rule. Awaiting user decision.
- AD-FTP/SmartAD recovery: Codex wt/chainrecover done (460 imports, 40 to run, 32 quota voids; 3 tasks need a fresh-
  attempt cap extension) -> follow-up for a bounded cap_extension (pid 2621568) running.

### 21:4x CDT 10-20
- Collections calling again since 21:08 (AD) / 21:10 (SmartAD). AD-FTP chain collection exited 21:43 CDT: BudgetExceeded on 3 tasks
  (work 42167910…, b979abd1…, a524fd09…; chain positions 19/125/130) — each had 1 quota-rejected call (unbilled) counted in
  the per-task cap 123, so attempt 3 hit the cap; their 3 shards aborted; 460/500 final, 40 missing (spread over all K).
  Codex wt/chainrecover (pid 2203392): recovery like full_train_recovery — new dir, hash-verified import, void quota
  calls credited against the cap, run only the 40; SmartAD variant too. SmartAD collection still running (pid 1415431).
- hpg: tsft-k500-e16 193/1008 upd at 21:14 (~12 s/upd, ETA ~00:15 CDT); sadc-k30-b05 95M/327M logical (ETA ~23:50).

### 21:0x CDT 10-20 decision (user OK 21:04)
- Main table: each method's training to convergence (doubling ladder, train-loss rule) = main number; ALSO report the paper's
  own epoch count (2 epochs for AD/SmartAD/trajectory SFT; SAD's x1 budget) as a 'paper recipe' column. So every e2 rung is
  evaluated too. Rationale explained to user: epochs are a data-size-dependent hyperparameter (AD: 2 epochs on ~2k
  trajectories = ~500 updates; our K30 = 20 updates); the method components are unchanged.
- Submitted tsft-k300-e2 / tsft-k500-e2 on LONI (paper-epoch column).

### 20:5x CDT 10-20
- Parser-rejected reply masking merged (main fb22f09; wt/parsemask): K500 5 replies / 115 target tokens masked, other 9,176
  unchanged; Kang/AD unchanged; 149 tests pass. Deployed hpg ICML27_runs/fb22f09; SAD K100-K500 cells reset on fb22f09
  (20cb697 outputs -> $BL/_superseded/sadc-20cb697). SAD K30 stays on 20cb697.
- LONI tsft done: k30-e2/e4/e8, k100-e2 (COMPLETED). Convergence (last complete epoch, token-mean): K30 e2 0.3948 ->
  e4 0.2106 (d 0.184) -> e8 0.0798 (d 0.131): not converged; reports /work/xueqic/hq/ICML27_data/reports/runs/convergence/.
  Submitted ahead: tsft-k30-e32 1118787, tsft-k30-e64 1118788, tsft-k100-e32 1118789 (LONI).
- Running: LONI tsft k30-e16, k100-e4/e8/e16, k300-e4/e8/e16, k500-e4/e8; hpg tsft-k500-e16 46922829 (group), sadc-k30-b05.
- Collections: 64 shard workers each re-hash the 12G source before calling (rai load 164 of 255 cores; others ~13);
  watcher b7brl9vv0 fires at the first post-resume success.

### 20:3x CDT 10-20
- User priority (20:23): 轨迹 SFT > AD > SmartAD > SAD; parallelize more; LONI CPU need not be saved.
  Keeper plan d30c1ea orders tsft, adc before SAD; pending SAD jobs nice=500; rebalancer restarted (pid 1419682,
  CORE '^(tsft|adc|smc|sadc)-', log logs/rebalance_accounts.log).
- LONI: tsft K30/K100 (8 rungs) — k30-e2 COMPLETED (loss 0.349, bit-matches the smoke run), 6 running. tsft K300/K500
  e4/e8/e16 ALSO submitted on LONI (1118751-57, gpu2 whole node 64 cores): dedupe loop (scratchpad/dedupe_loop.sh pid
  1433024, log logs/dedupe_tsft.log) cancels the other cluster's copy once one starts running.
- Plan: AD + SmartAD training and SmartAD selection on LONI (rai->LONI direct copy); all evals on hpg B200.
- Chain collections relaunched 20:22 CDT, both launchers in preflight (pids 1415430 adftp, 1415431 smartad).

### 20:2x CDT 10-20
- User topped up $20 (01:12Z). resume_chain_after_topup.sh: probe 200, recovered 32+32 quota-halted counters (transport_fixed),
  relaunched adftp + smartad chain collections with --resume (logs/chain_collect_*.log). SmartAD reuse NOT applied for K<=500
  (would change the collection identity and void 4,269 calls; saves only ~$2) — revisit for larger K.
- User asked whether ALFWorld is required / smaller common benchmark: recommended keeping ALFWorld (standard for ETO, IPR,
  WKM, LEAP, AgentGym, SAD), K<=500 first, WebShop as a second benchmark later. Awaiting user.

### 20:1x CDT 10-20
- User: last top-up was $20. That bought ~615M input (92% cached) + 9.4M output (rai ledgers since 04:44Z). K<=500 remainder
  ~$12 (AD ~$1, SmartAD ~$11); suggested topping up $20. Offered SmartAD reuse: full-TRAIN trajectory as candidate-01 (provider
  default T=1.0; ~10%) + old K100 pools' 61/500 overlapping tasks (12%) — awaiting user OK (Discord 1562271632678977547).
- sadc-k100-b1-train 46923526 FAILED (data encode): 'SAD reply requires an Action line' — 5/9,181 K500 teacher replies with
  'Action:' not at line start (H0 parser_error step, episode still won; chain positions 73/173/317/380/477; K30 clean).
  SAD K100-K500 cells held; pending/running doomed trains cancelled (46923522, 46923524, 46925184). Codex wt/parsemask
  (pid 1336457): SAD+SmartAD mask parser-rejected replies (kept as history, no labels), Kang/AD unchanged.
  After merge: new tree for SAD K>=100 (move sadc-k{100,300,500}-* outputs aside: identity bound to 20cb697).

### 19:4x CDT 10-20 queue state
- hpg (keeper, tree 20cb697, 10-job cap): pending sadc-k500/k300/k100-b1-train, tsft-k500-e{4,8,16}, tsft-k300-e16;
  sadc-*-b05 preps done/running; all GPU jobs Priority (B200 partition full). tsft-k500-e2/k300-e2 held (only needed
  if e4 were converged vs e2). max_gpus 36. tsft K30/K100 cells on hpg are eval-only.
- LONI (wrapper ICML27_loni_wrap.sh, loni_submit_tsft.sh): tsft-k30-e{2,4,8,16} 1118561-64, tsft-k100-e{2,4,8,16}
  1118565-69, gpu2 x2 A100, outputs /work/xueqic/hq/ICML27_data/reports/runs/baselines/<cell>; copy adapters to hpg
  $BL/<cell> before opening eval_<cell>. All pending (Priority).
- After top-up: scratchpad/resume_chain_after_topup.sh (probe 200 -> verify all 64 halted counters are non-retryable
  429 after 22:45Z (checked: 32+32) -> teachers.recover --reason transport_fixed -> relaunch with --resume on d7ad7fe).
- main bbba030 = + validation cache (wt/viewcache: K30 load 303 s -> 77 s warm); not deployed (running trees fixed).
- Watcher bjglg32mm: notifies when the first new-grid train is RUNNING (then check its early log).

### 19:1x CDT 10-20 update
- B200 acceptance 46909768 (shared-history SFT/SmartAD, dropout 0, seeds_k20_s0 4 rows): pre-set gates fail, but
  kernel-only control (per_turn SDPA vs Flex, same packing) shows the same gradient errors: lora_B rel L2 median
  0.42 (kernel) vs 0.31 (packing); early layers L0-L11 ~1.2-1.7 in both, late layers 0.03-0.1; loss rel diff <0.9%.
  ACCEPTED on the SAD/DPO basis (packing differences at kernel-swap bf16 level). Report $T/reports/engineering/
  sft_shared_history_gpu_r1-{sft,smartad}.json (tree 20cb697). Gates shsft_gpu_ok, tsft_ready opened 19:1x CDT.
- OpenAI credits exhausted: last success 17:45 CDT (22:45:37Z); chain collections (adftp 3,504 / smartad 4,269 calls
  recorded) stopped fail-loud (TeacherQuotaError, 32/32 shards each). User asked to top up (Discord 1562257725650370621).
  Resume after top-up with the transport-failure recovery procedure (no duplicate calls).
- Data verified: seeds_full_train_r2 without subsets = sha 14f44691f84be61c on rai, hpg and LONI (980,253 files);
  subsets (12 files) sha 560b83092b156642 on hpg and LONI. Gate chain_views_on_hpg opened; keeper submitting SAD.
- LONI smoke tsft-k30-e2-loni 1117994 COMPLETED 20.6 min (A100x2): 20 updates, b'=3, shared_history, ~24 s/update,
  loss 0.87 -> 0.35 (not converged at 2 epochs). LONI env works.
- Session usage limit hit ~17:10-19:10 CDT (no progress in that window).

### 17:1x CDT 10-20: chain grid launch state
- main 20cb697 = views (4001161) + convergence (7f3771b) + chain collections (feec885) + shared-history/multi-GPU
  SFT/SmartAD (wt/shsft; dropout 0.05 kept, expected-gradient equivalence; user OK 16:47). CPU tests: 99 + 165 + 112
  (torch, incl. 2-process gloo == 1-process LoRA) pass. Deployed hpg ICML27_runs/20cb697, LONI /work/xueqic/hq/ICML27_runs/20cb697.
- Views written on rai: seeds_full_train_r2/subsets/k{30,100,300,500} (full order sha c0b43778…; all distinct groups).
- Keeper plan b4bf9ff: 40 cells on 20cb697: sadc-k{K}-b{1,05}, tsft-k{K}-e{2,4,8,16}, adc-k{K}-e{2,4,8,16}; gates
  chain_views_on_hpg (SAD), tsft_ready, adc_ready, eval_<cell> (only the reported rung is evaluated). All closed.
- hpg: shsft-gpucheck 46909768 (B200 acceptance, VERIFIED seeds_k20_s0) running; watcher bjkkg9tbo -> open tsft_ready.
- rai: chain collections launched 16:44 CDT (scratchpad/run_chain_collect.sh, run tree ../ICML27_run_d7ad7fe, limiter
  data/teacher_rate_limit_chain): adftp pid 43085 (logs/chain_collect_adftp.log), smartad pid 43086; still in preflight.
- Data sync to hpg via the Mac tunnel still copying (pid 3655297); views need a separate copy after it. LONI copy done
  (original 980,253 files match; subsets copied separately). LONI smoke tsft-k30-e2-loni job 1117994 (gpu2, 2 A100),
  wrapper /work/xueqic/hq/ICML27_loni_wrap.sh, env /work/xueqic/hq/ICML27_loni.env. Codex wt/viewcache (pid 44089).

### 15:4x-16:0x CDT 10-20: NEW MAIN GRID (user) — supersedes the K5-K100 x 3-seed grid
- Grid: K in {30, 100, 300, 500} first (later 1000, 2000, all = 3,098), seed 0 only, nested stratified views of
  data/verified/seeds_full_train_r2 (rule stratified_nested_v2). Target (user): K<=500 x 4 methods by 23:00 CDT 10-20.
- Methods (user OK 15:57): 轨迹 SFT = K trusted trajectories + AD's SFT hyperparameters (LoRA r64/a128, lr 2e-4 linear,
  batch 8), greedy; AD = same + FTP-recollected trajectories, SAG eval; SAD = paper objective + 655M budget; SmartAD =
  10 candidates/task, min-NLL selection, segment-weighted SFT, greedy. Evals on hpg B200 only (fixed serving).
- Convergence (user 15:42): rung r converged if last-epoch token-mean train loss drops < 0.01 nats/token vs rung r-1
  (half the training); report rung r. Rungs: epochs 1,2(paper),4,8,16,32 (SFT/AD/SmartAD); SAD budget x0.5,x1,x2.
- Old grid: ALL cells held in the keeper; running/pending jobs cancelled 15:5x CDT (user: stop the non-converged runs);
  results kept as a separate small-K study. Low priority paused (0 jobs). hpg queue empty.
- Engineering (Codex gpt-6-astra, 4 parallel worktrees): wt/kchain (views, pid 3680502), wt/shsft (shared-history +
  multi-GPU for sft/smartad, pid 3689062), wt/convtrain (epochs/budget overrides + convergence checker, pid 3689064),
  wt/chaincollect (AD-FTP/SmartAD collections on the K500 view + sub-views + sharded selection, pid 3861166).
  Outputs scratchpad/codex_*.out. Watcher b0bwry6dn on kchain.
- Data: seeds_full_train_r2 -> hpg sync running (scratchpad/sync_full_train.sh pid 3655297, log logs/sync_full_train.log).
- LONI usable again (user re-authed 15:50): new allocation loni_depedlab04 ~1.5M SU (exp 2027-10-01); A100 80GB (gpu2
  50x2, gpu4 10x4), queue full (227 pending). /work/xueqic/hq/tc-alignment/.venv: py3.11, torch 2.13+cu130, tf 5.14.1,
  peft 0.20.0; gemma-4-12B-it snapshot 707f0a3b present. Plan: LoRA training rungs there, evals on hpg.
- Estimates told user (15:55): AD likely by 23:00; SAD partly (x1 ~5 h; x2 tomorrow if needed); SmartAD tomorrow
  04-06 CDT (90k teacher calls for K500 + selection + ladder); 轨迹 SFT fast after the trainer speed-up.

### 07:4x CDT 10-20: serving fix verified + rolled out (plan v9, keeper restarted pid 1593872)
- Fix 5554db8 (main; merged from wt/kveqlora, Codex; 501 CPU tests pass) cherry-picked: 2348ce9->8960e0a, 70ed412->3b82db4,
  071c72f->0411901, deployed to hpg ICML27_runs/ (configs identical to base trees). Real-adapter CPU check: k_eq_v layers
  [5,11,...,47], 96 v_proj tensors, validate OK. GPU pilot adftp-k50s0-eval 46842484 (8960e0a): gate PASS, adapter_err
  0.346 -> 0.026 (base noise 0.037), mean logp HF -0.591 / vLLM -0.588.
- Keeper: stage-level "tree" (commit 304f1df); eval stages of adftp/sadsh/smartad on the fix trees, training keeps its tree
  (SAD/SmartAD train re-derive code-hashed identity; evaluate_models does not). Other eval stages gated on
  gates/lora_kveq_fixed (low-prio trees have no fix yet). max_gpus 20 (pending 4-GPU SAD trains counted).
- Pre-fix AD-FTP evals moved to $BL/<cell>/training/_superseded_prefix_kveq/ and re-queued: k5s0 65/70, k10s0 37/43,
  k5s1 57/49, k10s1 56/57, k20s1 54/64 (greedy/SAG, pre-fix).
- Kang re-eval: 15 cells kangfix-k{K}s{S} (SAG, 8960e0a, OUT=reports/runs/eval/baselines/kang-<cell>-tp-sag-kveq; old dirs kept).
- Pending user decision: AD-FTP multi-Action first replies (18/21/18 of 100 episodes).
- 15:3x-15:4x CDT user: (a) OK continuation 'as long as fully trained'; (b) MAIN TABLE = 轨迹 SFT, AD (= AD-FTP + SAG, named
  'AD'), SAD, SmartAD, all 15/15 -> report, then add quick-to-complete baselines; (c) full support set = training set?
  (d) 'ALL training in our tables must converge, per standard LLM distillation'.
  Answers: train split 3,553 games; full-TRAIN r2 collection complete: 3,098 trusted (87.2%), 455 never solved; types
  simple 777 / two 752 / clean 530 / cool 399 / heat 342 / look 298. Full-cell costs: 轨迹 SFT, SAD no teacher calls;
  AD needs FTP re-collection ~84k calls (~0.7x full collection tokens); SmartAD ~640k calls / ~3.3B input (~5x, ~30 h at
  key limit) + ~70 GPU-h selection -> user to decide. Sync of seeds_full_train_r2 (17G, 980k files) to hpg started:
  scratchpad/sync_full_train.sh pid 3655297, log logs/sync_full_train.log, gate gates/full_train_on_hpg.
  Convergence evidence (s1 train loss end): 轨迹 SFT 0.000/0.006/0.014 (K5/20/100) vs AD 0.20-0.24, AD-FTP 0.30-0.35,
  SmartAD 0.18-0.19 (2 epochs = 10-50 updates). PROPOSED (awaiting user, Discord 1562202861033619528): converged when
  doubling training changes last-epoch mean train loss by < 0.01 nats/token; only epochs change (schedule stretched,
  retrain from scratch; ladder 2->4->8->16...); 轨迹 SFT continues (const lr): K5 converged, K20/K100 not; alternative =
  val-loss early stopping on 200 held-out full-TRAIN teacher trajectories. Replaces the dev-based continuation rule.
- 15:3x CDT user: keep ONE reference, named plainly '轨迹 SFT' (no AD-recipe row); AD = trajectory SFT + FTP + SAG +
  its own recipe. Chosen: the existing 15 Seed-SFT runs (standard objective, largest training amount). 'Sufficient
  training' not yet met for K5-K20 (still rising at end) -> proposed: continue those 9 runs from their last checkpoint
  (train/continue_from_state.py, optimizer state, constant lr) to 800k tokens, eval 400k+800k with fixed serving; stop
  when the doubling gain is within eval noise (~+-7), else double again; same rule for all K, endpoint only.
  ASKED USER to confirm the dev140-based stopping rule (Discord 1562201320306516020) before submitting.
- 15:2x CDT user on wording: Seed-SFT = trajectory SFT (standard objective: token-mean NLL of teacher agent outputs y_t
  given h_t, x/o_t conditioning only) + OUR long schedule (LoRA r16, const 1e-5, max(40 passes, 400k tokens)). Renamed in
  reports: '轨迹 SFT（长训练配置）'; paper-recipe row '轨迹 SFT（AD 配方）' = Kang SFT stage, greedy (offered, 15 evals, no
  retrain; awaiting user). It is a lower-bound reference, not 'what the data supports': pre-fix 3-seed means still rising
  at the end for K5-K20 (K5 75->85, K10 88->94, K20 101->108), plateau ~108-115 at K>=50 by 10-20 passes. Comparisons with
  baselines conflate training amount (K20: 469 vs AD 20 updates) -> proposed matched-update comparison via p03/p05/p10.
- 15:1x CDT: user OK'd Seed-SFT re-eval after SAD -> plan cells seedsftfix-k{K}s{S} (last checkpoint = main-table
  'end of schedule', greedy, 8960e0a, OUT reports/runs/eval/seedsft-kveq/<cell>; pre-fix sweep kept). Kang K100 s0
  same-checkpoint SAG re-eval 46892763: 61 (orig 62), 7/8 flips, inadmissible 18.7% -> collapse is the checkpoint, not
  eval noise; retrain 46892764 pending. SAD: sadsh-k5s0-train 46893249 RUNNING on 2 B200 (98-100% util, 112G/GPU,
  host RSS 17G), 14.4M/655M logical in ~6 min -> ~4.5-5 h/cell; K<=10 limit -> 10 h (pending jobs updated).
- 15:0x CDT user Qs answered (Discord 1562195029861081188 / ...144055328789 / ...333516361741):
  (1) variance: Spearman(inadmissible rate, success) = -0.77 over 55 core cells; all 8 cells <=40 are loop collapses
  (16-81% inadmissible vs 2-8%); small-K support content (Kang K5 s0 has no cool demo -> cool 0/25 vs s1 12/25); eval
  +-7 / retrain 5-10 from earlier measurements. Proposed: report per-seed + median + loop-collapse rate; optional extra
  training seeds at K20/K100 (user to decide). Table reports/analysis/core_cells_variance_1020_ZH.md.
  (2) naming: main table AD = AD-FTP(SAG) (FTP + SAG, full paper method); Kang(SAG) = ablation AD w/o FTP (official repo
  releases no end-to-end FTP workflow); AD-FTP(greedy) = ablation AD w/o SAG.
  (3) inventory (main grid 15): Kang 15/15; AD-FTP 14/15; SmartAD 12/15; AgentGym 11/15 (full FT, missing K50/K100 s1,s2);
  Seed-SFT 15/15 but unmerged LoRA incl. k_proj served pre-fix -> needs re-eval (proposed after SAD, awaiting user);
  LEAP priv/0% 0/15 valid (60 s-timeout runs invalidated 10-19; reruns at 1-4/9 stages); ETO/IPR/WKM/CoA/AgentDistill/
  STeCa/GAD/HAD 0/15.
  (4) gpt-6-luna logprobs: probe 15:0x CDT, Responses + Chat Completions both 400 unsupported_parameter ('logprobs are not
  supported with reasoning models'); rejected before generation. SAD keeps one-hot CE.
- 14:29 CDT user: "把SAD先修好，AD-FTP和SmartAD都尽可能要到15/15"; 14:30 "Kang K50->K100 drop is odd, recheck".
  SAD: budget 655M logical positions/cell for every K (K20: 4,600 updates, b'=2); shared-history/per-turn physical
  ratio 13.6 (K20 s0). rank_slots gives one trajectory per rank, idle ranks run zero-weight duplicates -> K5/K10 (b'=1)
  and K20 (b'=2) waste 2-3 of 4 GPUs. All 15 cells -> 2 GPUs (same math; K50/K100 3/6 rounds per update, equal
  GPU-hours). sadsh-k20s0-train 46826441 (4 GPU, started 14:34) OOM-killed 2 min in: every rank loads fp32 model on CPU
  (~50G x ranks); 144G was sized from post-load MaxRSS 95G (wrong). 2-GPU trains 160G, walltime K<=10 20 h, K>=20 12 h
  (tighten after the first measured rate). Old prep-only dirs + the OOM attempt -> $BL/_superseded/sadsh-g4-prep-1020.
  Low-prio limits 2 jobs / 4 GPUs. k5s0/s1/s2 trains 46893249/46893270/46893296 pending (dept, Priority).
  AD-FTP K100 s0: diverged attempt -> training/_superseded_diverged_1020, identical retrain 46892289 (keeper).
  Kang K100 s0 recheck: train curve normal (loss 0.25 end; gnorm spike 572 at update 25, lr 1.5e-5), seed data has no
  loops (inadmissible ~2%, no repeats, s0 K50 vs extra 50 alike), but eval 19.7% inadmissible (others ~2%), 20/140
  episodes with >=10 identical actions in a row; family drop heat 4/16, two 3/24. Replications (plan cells kangchk-*):
  same checkpoint SAG re-eval 46892763; identical retrain on def44a9 (176/176 code hashes match identity) 46892764.
- 14:14 CDT: SmartAD K50s2 = 45/140 greedy (gate 0.012, 0 error/overflow) -> 12/15: K5 53 59 53; K10 63 56 63; K20 82 73 59;
  K50 88 67 45. K50s2 train curve normal (loss 0.59->0.26, gnorm max 174 like s0/s1), but 26.3% inadmissible actions vs
  4.5%/3.7% (s0/s1), 79% of them repeat the previous action (open/close/move loops); worse in every task family.
  K100 retrains (4 h limit) running: s0 16/50, s1 18/50, s2 8/50 updates. SAD 4-GPU trains still pending.
- 13:14 CDT: SmartAD dev140 greedy 11/15 (base 62), s0 s1 s2: K5 53 59 53; K10 63 56 63; K20 82 73 59; K50 88 67 -.
  Gates pass (0.004-0.027), 0 error/overflow/parser, all served with the k_eq_v rule. smartad-k100s0-train 46869920 hit its
  2 h limit (44/50 updates, ~2.5 min/update; SIGTERM via --signal) -> K100 train limit 04:00:00 (plan.json + generator);
  doomed k100s1 46872515 (35/50 at 1:29) and k100s2 46877838 (8/50 at 0:23) cancelled (ours); 3 cells unblocked, keeper resubmits.
- 12:14 CDT: SmartAD dev140 (greedy, eval on 0411901 with k_eq_v fix; base 62) 5/15: K5 s0 53, s1 59; K10 s0 63, s1 56;
  K20 s0 82. Gates pass (0.005-0.027), 0 error/overflow/parser. Running: k100s0/k50s1/k100s1/k10s2 train, k20s1/k50s0 eval;
  k5s2 eval pending. SAD 4-GPU trains still pending (QOSGrpGRES, dept).
- 11:14 CDT: Kang re-eval COMPLETE 15/15 (SAG, post-fix; pre-fix in parens), s0 s1 s2: K5 56(57) 94(98) 67(68);
  K10 75(80) 69(71) 88(85); K20 77(63) 76(75) 97(110); K50 94(90) 94(89) 104(104); K100 62(67) 94(102) 106(108).
  Means K5 72.3, K10 77.3, K20 83.3, K50 97.3, K100 87.3 (base 62). Gates pass (mean adapter_err 0.009-0.018), 0 error/overflow.
  SmartAD K100 selections done on hpg (46855474/46857329/46858097, 53-59 min): trusted fallback 6/9/7 of 100 tasks; gates
  smartad_k100_s{0,1,2}_synced open 10:43-10:48 CDT; keeper running SmartAD train/eval (k5s0, k10s0 trains done, evals running).
  SAD trains sadsh-k20s0/k5s0/k5s1 still pending on dept (Priority).
- 10:14 CDT: AD-FTP grid done (14/15; K100s0 diverged). SAG s0/s1/s2: K5 69/46/58, K10 37/55/47, K20 5/60/38, K50 13/93/93,
  K100 div/99/99. Kang re-eval (SAG, post-fix; pre-fix in parens) 11/15: K5 56(57) 94(98) 67(68); K10 75(80) 69(71) -;
  K20 77(63) 76(75) -; K50 94(90) 94(89) -; K100 62(67) 94(102) - (s0 s1 s2). grid_check regex now includes smartadsel|kangfix.
- 09:14 CDT: AD-FTP post-fix grid (greedy/SAG): K5 62/69 58/46 61/58; K10 31/37 60/55 46/-; K20 4/5 57/60 35/-;
  K50 5/- 90/93 86/-; K100 div 89/99 95/- (s0 s1 s2). Seed 0 collapses with K (inadmissible-loop policy, mid-run loss rises);
  root cause open (FTP multi-Action first turns only weakly associated with gradient spikes). Pending user: FTP fix + retrain?
- SmartAD pools complete 1000/1000 x3 (09:04 CDT). Selection on rai too slow (36 s/candidate, ~10 h/seed): moving to hpg B200 —
  Codex wt/smartadrelocate (path map + hash checks, bg bkz8jn30c), pool sync to hpg (bg bfy7xics9, log logs/sync_smartad_pools.log).
  -> 09:4x CDT: fix e7e4311 deployed (main 1ddc8ca); plan v10 cells smartadsel-k100s{0,1,2} (top priority, gated on
  gates/smartad_pool_k100_sX_on_hpg); s0 running 46855474 (8.7 s/candidate, ~2.4 h). rai fallback stopped (226 scores, unused).
  Gate watcher scratchpad/smartad_sel_gate.sh (pid 2667278) opens gates/smartad_k100_sX_synced when hpg result.json is complete.
- SmartAD collection: first launcher exited 1 at 08:02 CDT (956/950/964 of 1000; one HTTP 400 in k100_s1 shard 7 made the
  launcher skip its completion pass, so deferred candidate-01 slots never ran). Failed slot quarantined (reports/runs/
  smartad-collections/_quarantine/k100_s1/README.md); relaunched --resume 08:23 CDT (launcher pid 1874049, bg borrsbjsb).
  smartad_post.sh (pid 783964) still waits for complete pools -> select on rai -> sync -> gates.
  Follow-up (code, later): SmartAD slots should use teachers.stage.execution_attempt; launcher completion pass should run
  for cells whose shards all succeeded even if another cell failed.
- 08:14 CDT check: post-fix AD-FTP dev140 K5s0 62/69, K10s0 31/37 (greedy/SAG; pre-fix 65/70, 37/43).
  adftp-k100s0 DIVERGED in training (loss 0.41->9.0 at updates 10-14, gnorm<=42k; model -6.2 nats/token on own data) -> gate
  cannot pass; blocked pending user (retrain? coupled with FTP re-collection decision). agd-k20s0-eval FAILED: gate compare runs
  under ENV_PYTHON without transformers (fix: train_python) - low prio, deferred, cell blocked.
- 07:42 CDT check: adftp-k10s2-eval 46832873 FAILED (pre-fix gate 0.0514 > tol 0.050, same k_eq_v cause) -> cell unblocked, eval reruns on 8960e0a.

### 06:5x CDT 10-20: vLLM Gemma-4 LoRA serving bug found; all evals gated
- adftp K20/K50/K100 s0 evals FAILED the LoRA gate (adapter_err 0.082/0.346/... vs tol ~0.056, growing with K). Root cause verified in source:
  Gemma-4 k_eq_v full-attention layers (8/48, no v_proj) -> HF V = k_proj(x) incl. LoRA delta; vLLM 0.27.1 loads base K into the
  qkv V slot and applies k_proj LoRA to the K slice only. Served model != trained model. Kang adapters had the same bias (<=0.026, passed).
- Fix (engineering, serving equivalence): derive serving adapter with v_proj LoRA = bitwise copy of k_proj LoRA on those layers; HF reference unchanged.
  Codex: worktree ../ICML27_wt_kveqlora (branch wt/kveqlora from 018332f), log logs/codex_20261020_074906.log, bg task bakizaydg.
  Then: cherry-pick onto 2348ce9 (adftp), 70ed412 (sadsh), 071c72f (smartad) -> new trees; pilot = re-run adftp-k50s0-eval; open gates/lora_kveq_fixed;
  switch cells' "tree" to fixed trees (AD-FTP evaluate mode does not call prepare(); check SAD/SmartAD evaluate paths before switching).
- Keeper plan v7: every not-yet-submitted eval stage gated on logs/queue/gates/lora_kveq_fixed (86 stages; plan.v6.json = before).
  Reset adftp-k20s0/k50s0/k100s0/k5s2/k50s1 (eval pending at gate). Cancelled our adftp-k5s2-eval 46836116 and adftp-k50s1-eval 46838916.
  Registered orphan adftp-k100s2-train 46838924 (sbatch went through during a keeper SSH_FAIL).
- Pre-fix AD-FTP dev140 (superseded): K5s0 greedy 65 / SAG 70; K10s0 37 / 43 (base 62). k10s1/k20s1 evals also pre-fix -> re-eval
  (move old eval-* stage dirs aside first: stage key does not include serving rule, stage_run would reuse them). Kang evals: re-eval for fairness.
- AD-FTP FTP defect (pending user decision): 18/21/18 of 100 first-turn replies contain >1 Thought/Action (no prefill; parser executes last).
  Proposed: reject such first replies like prefix mismatch, re-collect, retrain 15 AD-FTP cells.
- AD-FTP trains done 11/15 (k100s1, k100s2 46838924, k20s2, k50s2 running). sadsh-k20s0-train 46826441 pending (QOSGrpGRES).

- **05:1x CDT 10-20: AD-FTP collections done; SAD on T5; SmartAD collecting.**
  - **AD-FTP**: all three K100 collections are complete. They were synced to hpg ($B/reports/adftp-collections/k100_s{0,1,2}) and verified by file count, file bytes and a sha256 manifest; all gates are open.
    - adftp-k5s0-train 46811950 COMPLETED in 5 min: K=5, 5 tasks from collection/subsets/k5, 10 examples / 4,494 supervised tokens, LoRA checkpoint written.
    - The slurm log is empty by design; stage logs live in $BL/adftp-k*/training/stages/.
    - Eval and further trains are running (group).
  - **SAD → T5 shared-history (cells sadsh-kKsS, $BL/sadsh-*).**
    - GPU validation r2 (46824704, T5): the first try failed because the job lacked SED_SECRETS_FILE, so the config digest differed.
    - Report $T5/reports/engineering/sad_shared_history_gpu_r2.json:

      | Comparison | side rel diff | grad rel L2 max | token MAE |
      | --- | --- | --- | --- |
      | packing-only | 1.78e-2 | 0.117 | 0.029 |
      | kernel-only (SDPA→Flex, same packing) | 1.33e-2 | 0.206 | 0.037 |

      The pre-set gates (1e-3 / 0.02) are below the bf16 floor. Packing differences sit at kernel-swap level, so this is accepted on the same basis as DPO/IPR.
    - The T4 per-turn sad-k20s0-train (46789747) was cancelled at 04:49 CDT, after its 1/16 and 1/8 checkpoints.
      - Rate: ~31 h per cell, which exceeds the 24 h limit.
      - Its updates.jsonl (667 updates) is the per-turn reference curve for sadsh-k20s0.
      - Train memory 144G, from sstat MaxRSS 95G; walltime 16 h.
  - **SmartAD**:
    - wt/smartadshard was merged as 071c72f (sharded collection). SmartAD/teacher tests: 396 passed.
    - Tree T6 = 071c72f is deployed on hpg. The rai run tree is ../ICML27_run_071c72f.
    - Collection: launcher nohup pid 620007 (scratchpad/run_smartad_collect.sh; log logs/smartad_collect_launcher.log). It runs K100 s0/s1/s2 × 16 shards and writes to reports/runs/smartad-collections.
    - It shares the r2 limiter file (TEACHER_RATE_LIMIT_DIR = seeds_full_train_r2/data/teacher_rate_limit), so both collectors stay inside one 1.8M-tpm window.
    - Temperature 1.0 was accepted by the provider. The seed receipts do not prove identical sampling, so candidate 1 is executed fresh: 10 per task, not 9.
    - Rate: ~11 candidates/min with r2 sharing the window. ETA ~09:30 CDT.
    - scratchpad/smartad_post.sh (nohup pid 783964, log logs/smartad_post.log) runs per pool:
      - selection on ONE free rai GPU, never the last free one, using the tc-alignment-taskeq python;
      - rsync to $B/reports/smartad-selection/k100_sX, then verification;
      - gate gates/smartad_k100_sX_synced.
    - Selection runs on rai because pool receipts hold absolute rai paths; the selected collection is self-contained.
  - **Full-TRAIN r2**: 2,962 / 3,553 tasks attempted, 2,597 wins (05:0x CDT).
  - **Keeper plan v6**: sadsh-k20s0 → AD-FTP ×15 → SmartAD ×15 (gated) → sadsh ×14 → low-priority cells.
    - Limits: max_jobs 10 (the hard limit, so any manual hpg job requires lowering it first), max_group_gpus 6.
    - scratchpad/rebalance_accounts.sh (nohup pid 789861, log logs/rebalance_accounts.log) moves our pending core jobs from dept to group. It acts only when the dept QOS is full and group has room, with our group share kept ≤ 6 GPUs.
- **02:0x CDT 10-20: overnight status.**
  - **AD-FTP collections** (T4): all three halted once each between 00:18 and 01:0x on IncompleteTeacherResponse. Reasoning used up the 1,024-token step cap and output_text was empty; about 1 per 1,900 calls.
    - The fail-loud code treats this as transport; Codex fix wt/truncated (../ICML27_wt_truncated) will treat it as a truncated model output.
    - scratchpad/adftp_autorecover.sh (nohup pid 4038570, log logs/adftp_autorecover.log) resumes a cell ONLY when every halted counter is a pure max_output_tokens truncation, via teachers.recover --reason interruption_reviewed. The attempt re-executes fresh.
    - Progress: s0 86/100, s1 72/100, s2 18/100. sync_adftp.sh is relaunched under nohup (pid 4040651, log logs/adftp_sync.log).
  - **Full-TRAIN r2**: 32 shards (launcher pid 3587096, worktree ../ICML27_run_4aad41d). 06:13Z → 07:0xZ: K 1,406 → 1,479, attempted 1,631 → 1,707, logical calls 69,138 / 160k, reserved 333M / 700M.
  - **SAD**: per-turn sad-k20s0 runs as reference (group, 4 GPU). Codex wt/sadpack (shared-history SAD) is running; the rest of the SAD grid is held.
  - **SmartAD**: Codex wt/smartad running.
  - **Plan v5**: other baselines at low priority, ≤4 jobs / 8 GPUs (WKM, AgentDistill and CoA collections running).
- **23:4x–00:1x CDT 10-19/20: CREDITS TOPPED UP (user 23:44; probe HTTP 200). USER PRIORITY: core agent-distillation methods all-night.**
  - Scope: AD/Kang, AD-FTP, SAD, SmartAD, each over K5–K100 × s0–s2. Other baselines wait until tomorrow. The user also asked to combine the APIs and avoid any duplicate teacher use.
  - **Teacher gate open** (logs/queue/gates/teacher_ok).
  - **AD/Kang**: K grid 15/15 already done (SAG); full cell waits for the full-TRAIN collection.
  - **AD-FTP**:
    - Teacher collections K100 s0/s1/s2 running in parallel on rai (T4 worktree ../ICML27_run_2348ce9; pids 2958214, 2983901 launcher, 2983902 launcher; shared limiter data/adftp_rate_limit). Zero failures so far.
    - Serial within a set, about 2 min per task, so ETA about 3–3.5 h (roughly 03:30–04:00 CDT).
    - scratchpad sync_adftp.sh rsyncs each finished collection to hpg $B/reports/adftp-collections/k100_sX, verifies file count + du -sb, and touches gates/adftp_k100_sX_synced.
    - Keeper plan v4b then runs 15 AD-FTP cells (train 1 GPU 64G 2h → eval greedy+SAG).
  - **SAD**: 15 cells on T4, ahead of everything except AD-FTP. Preps done for 8; trains k20s0 (group), k5s0, k5s1 (dept, QOSGrpMemLimit) pending. sad-k20s0 (T2) retired to _trash.
  - **SmartAD**: not implemented before tonight. Spec taken from the paper PDF (K=10 candidates at T=1.0, correct-only, min per-turn macro-mean NLL under the initial student, Eq.5 weights 1/1.5/2 normalized by Σw, LoRA 2 ep lr 2e-4, greedy).
    - Codex wt/smartad (../ICML27_wt_smartad, launched 23:57 CDT).
    - Reuses the trusted seed as candidate 1 when its sampling matches, so 9 new candidates per task.
  - **Full-TRAIN recovery**: Codex wt/ftrecover (../ICML27_wt_ftrecover, launched 23:48 CDT). The tool will import all valid attempts by hash-verified copy into seeds_full_train_r2 with zero new calls, and exclude transport-failed / unknown attempts.
    - Reason: in 549503d the quota errors consumed attempt slots.
  - Non-core cells on hold (plan "hold": true): pending AG t1 ×3, gad x0 and lpz-k20s0 g1 were cancelled; lpp k20s0/k5s0 extract c0 completed with the gate open.
  - Keeper caps: max_jobs 8, max_gpus 16, group allowance 4 GPUs (group: yd26's pending jobs are user-held or waiting on dependencies; ~430G free). Dept QOS is memory-bound (QOSGrpMemLimit, about 917/1000G used by others).
- **22:1x CDT 10-19: T4 = 2348ce9 deployed; keeper plan v3 (43 cells / 349 stages).**
  - T4 contents:
    - main b1149a7, plus wt/transport bd5873c (baseline teacher transport per upstream / SDK: 600 s, 3 HTTP attempts, separate 429 retries, LEAP 3 tries; seed collection stays 60 s; docs/baselines/TRANSPORT_POLICY.md);
    - plus wt/dpopack (shared-history; conflicts with SAD/STeCa/HAD resolved by keeping both sides; shared_history only for dpo/ipr).
  - CPU suite on 2348ce9: 1,807 passed, 1 failed (theory venv missing, environmental).
  - Pre-submit check passed: all VERIFIED inputs exist, no T4 output dirs exist, all scripts present.
  - Old LEAP dirs (60 s timeout) moved to $B/_trash/leap_60s_timeout_1019/ (14 dirs; leap-privileged-k20s0-init kept).
  - Plan v3 order:
    1. SAD (T2), WKM (T3), AgentGym ×4 (T1)
    2. ETO k20s0, IPR k20s0 (T4; widen the grid after a first DPO stage works)
    3. STeCa, GAD k20s0
    4. LEAP zero/privileged ×15 K cells
    5. AgentDistill, CoA k20s0
  - Teacher stages gated on logs/queue/gates/teacher_ok (create after the top-up).
  - Not yet in the plan: full cells (wait for the collection), HAD (needs a co-scheduled pair stage in the keeper), and the rest of the ETO/IPR/SAD/WKM/STeCa/GAD grids.
  - Budget estimate docs/direction/2026-10-19_teacher_token_budget_ZH.md: about 27× the 14:10–17:55 collection usage, roughly 8B input tokens.
- **21:0x–21:2x CDT 10-19: OPENAI CREDITS EXHAUSTED at ~17:55 CDT** (last success 22:55:53Z).
  - Probe: HTTP 429 `insufficient_quota` / `credit_balance_exhausted` ("You have no credits remaining").
  - Every teacher call since then failed, on rai and hpg. The 19:4x "429 contention" diagnosis was incomplete: before 17:55 there was real contention; after 17:55 it was the credits.
  - Full-TRAIN collection (old tree 549503d, retried quota 429s for 3 h) **stopped by me at 21:1x CDT** (TERM to launcher 228787; all shards exited).
    - State: 1,406 trusted / 1,907 attempted of 3,553; 4,542 attempts recorded (2,189 ok, 2,353 error, mostly quota).
    - Settled usage 14:10–17:55: 57,342 calls, 292.9M input / 3.46M output tokens.
    - Resume needs the fail-loud recovery procedure (docs/TEACHER_TRANSPORT_FAILURES.md) so that quota-failed attempts do not count as task failures.
  - User asked to top up (Discord 1561924281720381494). **All teacher work paused until then**: collection, AD-FTP, LEAP, STeCa, GAD, CoA, AgentDistill, HAD.
  - Keeper teacher gate is now the file logs/queue/gates/teacher_ok (create it by hand).
- **21:0x CDT 10-19: LEAP TRANSPORT BUG → whole LEAP grid reruns (s0 included).**
  - The bug: LEAP teacher calls used Teacher(max_attempts=1, timeout 60 s) per method try, while upstream LEAP uses openai.OpenAI() SDK defaults (600 s, max_retries 2) inside a 3-try loop.
  - Evidence: success latencies are censored at the timeout (correction p50 48 s, p90 58 s, max 60.1 s; extraction p50 3.8 s). Lost requests were recorded as rejections: 119 APITimeoutError + 200 RateLimitError. privileged k100s0 lost 11/114 to timeouts (10-06/07), k50s0 3/63.
  - **Plan v2**:
    - Keeper runs only teacher-free cells: SAD k20s0 (08d3c47), WKM k20s0 (b1149a7, after the CPU suite passes), AgentGym K50/K100 s1,s2 (625f6a9).
    - LEAP cells removed; lpp-k100s1/s2-g1 cancelled.
    - STeCa/GAD/AgentDistill/CoA/HAD (teacher callers) wait for tree T4.
  - Codex wt/transport (../ICML27_wt_transport, launched 21:01 CDT): one baseline transport policy (upstream client settings, else SDK 600 s / 3 HTTP attempts), shared-key 429s retry without consuming tries, fail-loud on exhaustion, seed protocol unchanged (60 s).
  - Then **T4 = main + wt/transport + wt/dpopack** for LEAP full grid (both variants, all s), ETO/IPR grid, and the teacher-calling new methods.
  - Old LEAP K50/K100 dirs (625f6a9) still in place; move them to _trash before the T4 cells start (same output paths).
  - Already moved to $B/_trash/tainted_or_failed_1019/: 14 429-tainted LEAP K5–K20 cells, plus wkm/gad/had k20s0 and new_baselines agentdistill/coa k20s0.
  - main = b1149a7 (merged wt/failloud 0cfbd46 and wt/wkmfix 2d769a0); tree b1149a7 deployed on hpg (third_party copied from 08d3c47).
  - **Shared-history scoring ACCEPTED** (wt/dpopack c92aa09, reports/engineering/shared_history_gpu_acceptance_1019.md): fp32 CPU bitwise equal; bf16 GPU packing diffs are the same size as the SDPA→flex kernel-only control; 6.3 s vs 190.5 s per pair.
- **20:3x–20:5x CDT 10-19: HARD RULE ≤10 hpg jobs at once.** Faculty email (Guang Wang, 20:32) forwarded by the user: another faculty member saw 124 jobs / 148 GPUs from xc25.fsu.
  Cause: batch-1/2 pre-submitted afterok chains (pending dependency links count as queue load).
  - Actions: cancelled every pending/held job (184-line list in hpg:~/queue_before_cancel_1019.txt) and the idle had-teacher; 5 LEAP K100 g1 jobs kept running. User notified, with a draft reply to the professor.
  - **Throttled queue keeper** on rai: `scripts/hpg_queue_keeper.py --dir logs/queue` (pid in logs/queue/keeper.pid, 180 s loop).
    - Plan from `scripts/queue_plan_1019.py`; cells keep their original tree (identity binds code_sha256).
    - One job per cell, next stage only after COMPLETED. Caps (running+pending, all accounts) max_jobs 8, max_gpus 16, group GPUs 0, because groupmate yd26 has ~29 pending GPU jobs on yd24f.
    - Any other terminal state blocks the cell. PAUSE file stops submission. Old-tree LEAP teacher stages are gated on rai collector pid 228787 exiting (no 429 retry in 625f6a9).
    - Hourly cron check prints keeper status. **Any manual sbatch must also keep the total ≤10.**
  - Plan v1 (15 cells): SAD k20s0 train→eval (08d3c47); AgentGym K50/K100 s1,s2 from x0 (625f6a9); LEAP K100 ×5 from g1 (running) and K50 ×5 from c1 (625f6a9).
  - Not in the plan yet:
    - batch-1 LEAP K5/K10/K20 cells: the **429 scan found nearly all of them TAINTED** (e.g. lpz k20s1 14/14 correction requests lost, lpp k20s2 12/12, k10 cells 3–6 of 5–6; scan in scratchpad leap429_scan.txt). Restart fresh on the new tree, old dirs to _trash.
    - New-method K20 s0 cells (STeCa, WKM, GAD, AgentDistill, CoA, HAD) on a new tree after merging wt/failloud and wt/wkmfix.
    - R2 held arms (user decision).
  - Older finding from the same scan: LEAP privileged k100s0 lost 5/48, 2/44 and 4/22 correction requests, and k50s0 lost 0/2/1, all on 10-06/07. Check whether these were quota or transport failures before trusting those s0 numbers.
  - shared-history GPU check 3 (46751335/36) FAILED its gate. packing_only (per_turn_flex vs shared) side-sum 0.84% / grad 12%, vs kernel_only (SDPA vs flex, same packing) 1.5% / 20%. Packing adds no error beyond a kernel swap (token mean |Δ| 0.024 nats, under the 0.05 HF/vLLM tolerance). Next: an fp32 exactness check on rai.
- **19:4x CDT 10-19: 429 CONTENTION INCIDENT.** rai full-TRAIN collection paced at 1.8M/2M tpm + hpg teacher stages → HTTP 429s: LEAP correction stages recorded rejected corrections
  (lpz k5s0 rej 1/0/2, lpz k5s1 1/2/1, lpp k5s1 2/2/3, lpz k10s0 3, lpz k10s1 5 of 5, lpp k10s1 6 of 7 …; 429 files in audit/teacher/correction-*), AgentDistill k20s0 teacher halted (first call 429),
  CoA k20s0 teacher halted → data empty → result stopped_empty_sft (silent). ACTIONS: held 50 pending LEAP teacher (c) jobs; cancelled CoA/GAD/AgentDistill dependents; Codex wt/retry429 (pid 1634290):
  rate-limit 429 retry with backoff, quota 429 still halts, shared limiter dir. TODO after collection ends (~21:30) + fix: halted teacher must fail stages (CoA etc.); scan all ledgers since 14:10;
  invalidate (move to _trash) every LEAP correct-i stage with 429-reduced corrections and all later stages, rerun; rerun AgentDistill/CoA/GAD k20s0 fresh. LEAP 0% K5 s0 result (64/62/61/64) is TAINTED.
  Also: GAD failed for missing third_party/LMOps/gad in tree (synced to 08d3c47 and 1155125). Shared-history GPU check now runs but fails thresholds (side sums 0.37%, grads 6–15%);
  per-turn diffs random incl. turn 1 → likely kernel numerics; Codex adding per_turn_flex control (pid 1625002).
- **18:2x CDT 10-19:** main 08d3c47 (all 8 new methods merged incl. CoA; 226 CPU tests pass). Batch 2 SUBMITTED (K20 s0 of SAD/WKM/STeCa/GAD/AgentDistill/CoA, see exp_log).
  Shared-history fix (tree e20cb73) GPU checks 46741186/87 queued. Codex: wt/hadco co-scheduled HAD runner (pid 1409396). AD-FTP collection waits for full-TRAIN collection.
- **18:1x CDT 10-19:** first batch-1 result: LEAP 0% K5 s0 dev140 64/62/61/64 (iter 0–3; base 62); corrections 2/2/1. Batch 1: 66 links done, 0 failures; group-account K5 chains nearly done.
- **16:5x CDT 10-19 (usage limit reached; resume here):** main = bc119be (+ GAD, AgentDistill merged; all new-method CPU tests pass). Tree bc119be deployed to hpg (+leap_llm);
  all-MiniLM-L6-v2 copied to hpg HF cache (snapshot 1110a243…) for WKM/STeCa EMBEDDING_SNAPSHOT. NEXT: (1) batch-2 smoke chains at K20 s0 for SAD (prepare CPU → train 4 GPU 24h → eval),
  WKM (experienced/explore/synthesize/world/agent GPU1; merge/knowledge-base CPU), STeCa (sft/reinforced 4 GPU; expert-mc/explore/agent-mc GPU1; detect/reflect/datasets CPU), GAD (explore/teacher/train × r0–2),
  AgentDistill (CPU stages, evaluate 1 GPU; AGENTDISTILL_OUT outside reports/runs); (2) HAD needs a co-scheduled GPU+CPU runner (per-round jobs ≈150/cell) before submitting;
  (3) Codex still running: wt/dpopack head_dim-512 kernel_options fix (pid 814843) → rerun tc-shcheck on hpg; wt/coa (pid 820040); (4) AD-FTP collection after the full-TRAIN collection (shared API limits).
- **16:2x CDT 10-19:** main = 18030f5: merged wt/adftp-sad (AD-FTP, SAD), wt/wkm-steca (WKM, STeCa), wt/had (HAD); merged-main CPU tests pass (171). Not yet deployed to hpg.
  Shared-history DPO GPU acceptance (hpg 46723617/18, tree 31e4875) FAILED before comparing: InductorError, FlexAttention default tiles exceed shared memory for Gemma-4 global layers (head_dim 512).
  Reproduced on rai GPU1; kernel_options BLOCK_M=BLOCK_N=32 compiles fwd+bwd and matches SDPA in bf16. Codex fixing (pid 814843). Measured on one ETO pair: per-turn path 368,738 forward tokens, 187 s fwd+bwd; packed 20,697 tokens (18× fewer).
  Moved 4 small LEAP chains (lpz-k5s0/s1/s2, lpp-k5s1) to yd24f (fairshare 0.32 vs dept 0.22; dept estimates were 16–30 h). Codex launched: wt/gad (819635), wt/coa (820040), wt/agentdistill (820526).
  Full-TRAIN collection: 754 attempts by 15:22 CDT (~10.5/min, 81% success), ETA ~21:10 CDT.
- **14:1x CDT 10-19: full-TRAIN teacher collection RUNNING on rai** (pid 228787, 32 shards, worktree ../ICML27_run_549503d, output data/verified/seeds_full_train; caps 160k calls/700M tokens; rate limits raised to 4,500 rpm/1.8M tpm after the 10-19 header probe 5,000/2M). Merged wt/fulltrain (549503d). Codex still running: wt/dpopack follow-up (GPU validation tool, pid 19353),
  wt/adftp-sad (4149528), wt/wkm-steca (4150075), wt/had (4150904).
- **13:4x CDT 10-19: FULL BASELINE GRID started (user 10-19: run every baseline completely, same fair setting, repo or paper; no GPU cap; hpg main, rai/LONI in parallel).**
  Plan: docs/direction/2026-10-19_full_baseline_grid_plan_ZH.md (625f6a9). Grid (user 13:30): K ∈ {5,10,20,50,100} × s0–s2 + 1 full-training cell (support = all 3,553 TRAIN games) = 16 cells/method.
  User approved my recommended defaults Q1–Q5 (SAD one-hot teacher; HAD own harness, eval on HAD harness and H0, ≤30k teacher calls/cell; CoA gpt-6-luna multi-agent teacher + DAPO on K tasks;
  AgentDistill tools compose admissible actions, steps count to 40, flagged; GAD per-turn). Kang = AD; 13 methods total incl. SAD, AgentDistill, CoA, HAD, GAD, WKM, STeCa.
  **Batch 1 SUBMITTED** (tree 625f6a9, scripts/ops_batch1_1019.sh, dept QOS, 252 jobs 46706449–46706702, log ~/batch1_1019.log on hpg): AgentGym K50/K100 s1,s2 (4 chains × 9);
  LEAP privileged K5–K100 s1,s2 (10 × 9); LEAP 0% K5–K100 s0–s2 except k20s0 (14 × 9). Stage-split: AgentGym train 2 GPU 4C/160G, explore 1 GPU 8C/48G; LEAP GPU 4C/48G, teacher stages hpg-milan 2C/8G.
  Monitor: scratchpad grid_check_once.sh, cron :47 hourly (e520283c, session-only, 7-day expiry).
  **Codex (parallel worktrees):** ../ICML27_wt_fulltrain (pid 4109453: all_train_v1 selection, parallel capped collector ≤160k calls/700M tokens, full cell in K guards);
  ../ICML27_wt_dpopack (pid 4111993: exact shared-history packed scoring for turn-packed DPO/IPR; old path ~168 s/pair-slot). Next: specs + Codex for AD-FTP, SAD, WKM, STeCa, HAD, GAD, CoA, AgentDistill.
  Storage: /blue dept 15T/20T used — LEAP merges ~24 GB each; plan moving evaluated checkpoints to /orange (copy→verify→delete).
- **13:5x CDT 10-18: baseline failure analysis delivered (user: list every baseline's principle, find where each fails on the benchmark, explain by first principles; analysis only, no method).**
  docs/analysis/baseline_failure_analysis{,_ZH}.md + 5 figs (faad17b); scripts analysis/baseline_failures/{build_data,plot}.py (255 dev140 cells, 244 with episodes).
  Key: no baseline median beats Seed-SFT end-of-schedule at any K; 26/52 AgentGym, 4/24 Kang, 4/23 LEAP cells below base−7; heat/cool are the weak types;
  mechanisms: epoch-fixed dose (BC stages 10–30 updates vs 469), no new information except LEAP, static success selection (AgentEvol −54..+24),
  DPO saturation on ≤K pairs (ETO 2/20→1/20 train re-solves), LEAP corrections few/partly inadmissible and dose-confounded, SAG = greedy mode.
  Checked and excluded: kept successes carrying inadmissible actions; thought-text drift. Summary sent in 2 Discord msgs (1561453463588704296/…495444574478).
  hpg tunnel back 10-18 ~13:36 CDT (user "hpg好了"); only held R2 arms remain on hpg. Pending user decisions unchanged (R2 held arms, R1 continuation, P1 D1/D2, T1 control arm).
- **16:4x CDT 10-15: T1 eval T11 COMPLETED → dev300 224/300 vs T00 242 (−18; descriptive only per prereg). All T1 jobs terminal.**
  54 flips (18 T11-only / 36 T00-only) vs same-checkpoint server reference 15–20 flips (noise a, noise b, T00 are one bitwise-identical
  adapter; net ≤1). Per type T00→T11: look_at +3, clean −6, cool −7, heat −7, others ≈0 — the R2 A3_n100 pattern, from optimizer state
  alone (θ held at θ0). Sign is OPPOSITE to surrogate M = +0.0143: surrogate40 says the transplant helps, dev300 says it hurts (cf. Remark
  8.11, α/ε_sur uncalibrated). Exploratory, not a prereg readout. Open (needs user): random-perturbation arm (tail + score + dev300
  ~0.75 GPU-h) to separate feedback-specific memory from generic divergence; optional dev300 for N1/T10/T01 (~0.28 each).
  Cumulative GPU-h ≈ 36.4 of 40. Still pending on hpg: only the held R2 arms (4 held + 4 eval deps).
- **14:4x CDT 10-15: T1 PRIMARY READOUT — POSITIVE per prereg (|M| > τ).** T11/N1/summary COMPLETED (exit 0). Surrogate40 F_tilde_geo_prob after the
  same 254-update reference tail (216–469): T00 = T00′ = A0 = 0.487964 → δ_num = 0 (T00/T00′ adapters bitwise identical), τ = 0.0012 (floor).
  M = F(T11)−F(T00) = +0.0143; M_m = −0.0149; M_v = +0.0356; I = −0.0064; P = F(N1)−F(T11) = −0.0040; total N1−T00 = +0.0103.
  Prereg wording: memory channel has a measurable effect at z10 / feedback block B1 / fixed reference tail; channel split descriptive only;
  NOT a test of Thm 7.1 adjoint prediction, NOT deployment benefit (addendum). Exploratory (outside prereg): β2 = 0.999 → 0.999^254 ≈ 0.78 of
  the v difference survives the tail, β1 = 0.9 → m difference gone after ~20 updates, so M_m acts via early θ divergence; δ_num = 0 measures
  repeatability only, not sensitivity of a deterministic 254-update tail to any same-size (m,v) perturbation (no such control was run).
  Prereg next step = propose replication at a second branch point (needs approval; ~4 GPU-h > remaining ~3.7). eval-T11 45924350 PENDING (QOSGrpGRES).
  GPU-h since 12:5x: T1 tails ×5 + eval T00 = 2.46 → cumulative ≈ 36.1 (+ eval T11 ~0.26) of 40.
- **13:4x CDT 10-15: T1 arms T00p/T10/T01 COMPLETED (~26 min each); T11 RUNNING (GPU 100%); N1 → summary → eval T11 pending. eval T00 dev300 242/300 (descriptive; = cold p40 241/242).** Primary T1 readout (surrogate F channels M, M_m, M_v, P, δ_num) comes from the summary job.
- **12:5x CDT 10-15: tunnel back (down ~05:40–12:49 CDT; user "vpn"/"恢复了"). 12 jobs COMPLETED overnight, all exit 0.**
  - **R1 FULL 4-WAY PRE-REGISTERED RANKING (written before any R1 continuation): P2 > P1 > P4 > P3** (Ŝ, F_tilde_geo_prob,
    B=2, δ=0.05; cross-run merge of p1p3-0..3 + p2p4-0..3, 256 rows, provenance/comparability checks passed, tail seeds equal per
    replica; merge on rai CPU with HEAD crn_tail (cross-run mode 2f9e108); out $B/reports/runs/r03/r1/merge_all_F_tilde_{geo_prob,per_token}.json).
    Per-token: P2 > P4 > P1 > P3 (only P1/P4 swap). Paired Ŝ (geo-prob): P2−P1 +0.00035 (40/64 >0, paired t 3.0), P2−P4 +0.00039
    (46/64, t 4.0), P2−P3 +0.00172 (60/64, t 12), P1−P3 +0.00138 (57/64), P4−P3 +0.00134 (56/64), P4−P1 −0.00004 (31/64, t −0.4).
    **No pair is certified under the 4-way merge**: diagnostic radii 0.0012–0.0019 (union over 6 pairs; P1−P3 was diagnostic-certified
    in the 2-way merge at 0.00121, now 0.00169 > 0.00138); strict a-priori radius 0.49 (vacuous). Task certificate not asserted (α/ε_sur
    uncalibrated). Predictions: P2 first ✓; P4−P1 within diagnostic radius ✓ (Cor 6.7); P3 between ✗ (P3 last). Static scores: grad-norm
    mean P2 94 > P1 27 > P4 22 > P3 4.2 → same order as Ŝ(geo-prob); loss P2 2.05 > P4 0.23 > P1 0.21 > P3 0.13 → same order as per-token
    Ŝ. So in R1 the static rankers order the 4 candidates like the certificate; they miss the magnitude (P2's 3.5× grad over P1 buys
    1/4 of the P1−P3 gap). R1 continuation (3 arms ~14 GPU-h + evals) still exceeds the 40 cap → NOT submitted; needs the user.
  - **R2 n=100 (dev300): A0 = cold p40 241/242, A1_n100 221, A3_n100 170. PRIMARY CRITERION FAILS IN THE OPPOSITE DIRECTION:**
    A3−A1 = −51 (discordant 10 vs 61, exact McNemar p = 4.6e-10). A1 vs cold −20 (23 vs 43, p = .019/.013 vs servers a/b); A3 vs cold
    −71 (p < 1e-11). Both evals passed the HF-equivalence check; 0 errors/overflow; all failures hit the 40-step limit. Per type
    (cold_a/A1/A3): look_at 27/35/35, simple 62/60/54, clean 43/38/21, cool 36/23/7, heat 30/20/13, two 43/45/40. A3 inadmissible
    1.39/episode vs 0.63 (A1), 0.61 (cold). Predicted (Prop 6.10 scale-free) A3 0.0464 < A1 0.0582 < A0 → observed order is the
    reverse (Spearman on 3 arms = wrong sign; 7-arm criterion not computable, 4 arms held).
    Mechanism evidence: reference part identical in A1/A3 (297 updates, same replaced indices, seed 1202); only the extra pool differs.
    A3 water-filling (W∝d, k_s) put 56/100 queries in look_at (K20 coverage k=10, three bins k=0), 0 in clean, 5 heat, 12 cool; A1
    spread 10–22 per type. Each of ~99 single-Action labels (~8.5 sup. tokens) is repeated ~60× (token-matched 515-token extra updates,
    98 of 395 updates) → ~5,900 extra examples per arm; A3 extra-pool loss 0.194 vs A1 0.158. A3's drop is in the types it did NOT query
    (clean −17, cool −16 vs A1) while its heavily queried type gains nothing over A1 (35 = 35): cross-cluster interference, i.e. the
    state-owned / additive-coverage assumption of Props 6.10–6.11 fails badly on the real model (prereg already flagged it as a proxy).
    Design confound (mine, same for all arms): replacement removes 25% of replay + heavy label repetition; A1 < cold says the insertion
    scheme itself hurts at n=100. Held arms (A1_n400 45733059, A2_n100 61, A2_n400 63, A3_n400 67 + evals) stay held → user decision.
  - T1: smoke, B0, B1, A0, T00 COMPLETED; T00p RUNNING since 12:45 CDT (GPU 100%); T10/T01/T11/N1 → summary → dev300 T00/T11 pending.
  - GPU-h of jobs finished since 10-13 03:45: 18.35 (R1 shards 6.6, R2 A1/A3 + evals 9.6, T1 so far 2.1) → cumulative ≈ 33.6 + T1 rest ~2.6.
- **04:41 CDT 10-15: R1 p2p4-1 45729775 (P4) RUNNING since 04:31 CDT on c1107a-s15, GPU 78%, 3/32 rows at 04:40 (~3 min/item → ETA ~06:10 CDT, limit 2:30).** Still pending (QOSGrpGRES; both accounts full): p2p4-2 45950751, p2p4-3 45729777, R2 A1_n100/A3_n100, T1 smoke → stages.
- **21:xx CDT 10-14: P1 empirical findings (user: 'empirical findings…以图的形式'), commit 3a627e6, docs/proposals/P1_empirical_findings*.md.**
  Existing data only: A AgentEvol Δ −54..+24 over 13 sets; B dose grid 11/12 runs below base at 36k–77k tokens (post-hoc window);
  C R1 static loss/grad 16–24× for P2 but post-clip update norms equal, P2≈P1>P3 (P2 interim 32/64); D K5s0 (no cool demo) cool
  0–3/25 through 400k tokens vs 7–13 with one demo; E AdamW memory share 0.72–1.00, ν 0.45–0.63 vs SGD 0.02–0.04, mw32 −13 @p40 p=.015.
  Scripts analysis/p1_findings/{build_data,plot}.py (R1 rows + probe copied from hpg to scratchpad; rerun needs --r1-dir/--probe).
- **19:xx CDT 10-14: two method proposals written (user: 'propose 两个ICML级别的方法'), commit 6ac88ba, docs/proposals/.**
  P1 RCD: shared flaws of all agent-distillation methods (static valuation / evidence-vs-processing + learner-blind
  allocation / parameter-snapshot learner) → one general fix over any baseline's generator; E1–E7, E1+E6 first (reuse z10/R1).
  P2 RE-JEPA: six WM conditions (response equivalence not realism, trust region, closure, no free information, legality,
  Prop 8.15 cost threshold ρ>0.77 at 1/10 cost) + JEPA design + X1–X7, X1 first (TextWorld engine injection). No GPU used;
  phases B/C need budget approval. User directives: shared flaw + general fix, not per-method patches; method sections only.
- **17:0x CDT 10-14: tunnel back (down 13:41–~17:00 CDT). R1 p2p4-0 45729774 COMPLETED 2:28:22 (limit 2:30), 32 P2 rows.**
  Shards 0/2 are P2 (~4.3 min/item), 1/3 are P4; crn_tail has no resume, so shard 2 (45729776, 2:30) was cancelled before
  start and resubmitted from its SubmitLine with --time=03:15:00 → **45950751**. Pending: p2p4-1 45729775, p2p4-2 45950751,
  p2p4-3 45729777, R2 A1_n100/A3_n100, T1 smoke → stages (all QOSGrpGRES/Dependency; dept GPU full). README top reorganized
  (d8a3452, user: framework-flow paragraph instead of 'unifying object'; central results as blocks).
- **13:1x CDT 10-14: user "不是必要的都可以撤了" + "作业改小以后再并行提交".** Cancelled the whole IPR r5 chain (14 jobs,
  45470256-69) and ETO tp6 dpo3/eval (45058904/05). ETO tp6 x3 45058903 had already COMPLETED (11 min) after the
  BadConstraints fix; its output stays in place. Kept (necessary): R1 p2p4 x4 (p2p4-0 RUNNING since 13:01 CDT, GPU 100%),
  R2 A1_n100/A3_n100 + evals, 4 held R2 arms (budget decision pending; note A2_n100/A2_n400/A3_n400 still have 1:30 limits),
  T1. **T1 split into parallel stage jobs** (prereg addendum 2, tree a68f109, Python identical to 58e46e0, stub dry-run on rai):
  smoke 45867088 (gate) → B0 45924339 / B1 45924340 / A0 45924341 → 6 arm jobs 45924342-47 (compose+post+score)
  → summary 45924348 (CPU, hpg-milan); dev300 T00 45924349, T11 45924350. ~5.7 GPU-h; ~2 h wall-clock once slots free.
- **12:0x CDT 10-14: pending requests resized to measured usage** (user: "我们以后申请的量应该按照实际估算"). sacct 10-01..10-14:
  1-GPU continuation jobs (r0/r1/r3/gradstats/pilot, 20 jobs) 1.0 busy core, peak RSS ~23G; evals (53) 1.8-2.2 cores, <=23G;
  IPR r4 sft 4.0 cores/24G; IPR/ETO explore 1.2-2.6 cores/<=15G; ETO 2-GPU DPO 2.0 cores, 99.5G; IPR 4-GPU DPO OOM at 160G.
  Applied with scripts/ops_shrink_1014.sh (NumCPUs+MinCPUsNode+CpusPerTask together; every job re-checked, dependencies intact):
  R1 p2p4 x4, R2 arms x6 (incl. 4 held), T1 smoke/transplant 6CPU/96G -> 3/48G; 8 evals 64G -> 48G (CPUs stay 14: pre-registered
  comparisons use the 14-CPU noise-reference settings); IPR r5 sft 96G -> 48G; IPR r5 x1-x6 6 -> 4 CPU; ETO tp6 dpo3 8 -> 4 CPU.
  DPO 4-GPU 256G kept. **ETO tp6 x3 45058903 was BadConstraints since 10-07** (NumCPUs=6 but MinCPUsNode=14, left by
  ops_shrink_1007.sh) -> fixed to 4/32G, now eligible, so the ETO chain x3 -> dpo3 -> eval can run. Pending totals: 302 -> 234 CPU,
  3744G -> 2992G. Templates (hpg_r03_*, hpg_t1_*) changed the same way; OMP_NUM_THREADS stays 6 (numerics unchanged).
- **11:3x CDT 10-14: framework v0.17.1 — the user's review of v0.17 (16:19 UTC msg, "作为后续工作的统一底稿，不再推倒重写").**
  The user re-ran the 22+5 checks (PASS) and asked for local fixes only, freezing the main structure:
  (1) Thm 7.1(e) now holds only for exogenous tail randomness; the new 7.1(f) covers state-dependent sampling:
  grad V = sum p chi + sum V grad p, with baseline and score forms. Check 7.10 (v017:C23.sampling): the costate
  average alone misses 13% of the momentum component. (2) Thm 6.5: new (T) tail control (from (E) plus Chebyshev
  when Psi is bounded); the proof splits on theta_hat in U; the linear term uses Phi_c (pseudo-inverse).
  (3) Remark 9.5 has three layers: general identities / conditional theorems / practical method (G4); new rows
  for (T) and sampling. (4) The T1 prereg gets an addendum (written before any T1 job ran): the negative wording,
  M=M_m+M_v+I cancellation, channel readings beyond the prereg are exploratory, T1 does not test the adjoint/R15,
  positive != deployment gain. Mirrored in 9.4 and the 7.6 experiments page. (5) Remark 9.6 + README: formulation +
  composition + conditional guarantees are theoretical contributions; the open items are efficiency at real scale
  and the method's benefit. User's next priority after R2/T1: one real result showing a decision changed by the
  connections (info-compilation radius guiding a decision, OR the response interface saving compute on held-out
  candidates/checkpoints). hpg tunnel was down 04:40–11:21 CDT; all T1/R2/R1 jobs still PENDING (QOSGrpCpuLimit).
- **00:5x CDT 10-14: framework v0.17 written and merged into theoretical_framework/** (user "你来写", 23:00 CDT 10-13). Basis: the
  user's second revision + my review2 with errata. Title → "…for Continued Training"; response operator as the unifying object.
  New items: Prop 3.4 (joint decomposition), Def 4.8–Prop 4.13 (response operator, spectral sufficiency, Hankel/PSR realization,
  static≠dynamic closure, factorization through the next predictive state, finite-horizon Gramians), Props 6.14–6.16 (convex
  realizability, linear-Gaussian target, information–compilation radius), Prop 7.6 (exact Adam memory construction), Props
  8.12–8.15 + Cor 8.16 (catalog certificate, shared features, anchor reuse + drift certificate, two-fidelity, composed guarantee),
  Remarks 9.5–9.6 (assumption audit, contribution units/prior work). Modified: Lemma 6.4 (singular designs), Thm 6.5 (generability
  scope), Cor 7.2 → restricted mean-field model + R3 negative (Check 7.6), Theorem B = Thm 7.1 + Prop 7.6, Alg 9.2 (gap routing,
  legal compilation, strict/diagnostic apart), Rule 7, Remark 9.4 rewritten with R0/R1/R3 results, Remark 8.12 → 8.17 (6 open
  problems). New check suite theory/v017_checks (22 synthetic + 5 record checks recomputed from hpg outputs in records/; all PASS);
  18 new references verified against primary records. Independent proof audit: 1 error (8.12(e) bias term with frozen estimated
  triples), 1 gap (8.16 vacuity depended on undeclared B), 12 wording — all fixed EN+ZH. Validator: 452 files, 0 errors.
  Note: R3 |prediction/measured| is 28–339× from the records (earlier "15–300×" was wrong).
- **23:26 CDT 10-13: T1 optimizer-state transplant pilot SUBMITTED** (user approved "可以批" 23:00 CDT; prereg
  docs/direction/2026-10-14_T1_optimizer_state_transplant_prereg_ZH.md, 776819b). Tree 58e46e0 (ICML27_runs/58e46e0; Codex code
  train/transplant.py reviewed, CPU tests 35/35). Jobs: smoke 45867088 (0:50 limit; 4-update blocks, compose, 4-update post, scores)
  → main 45867089 afterok (6:30; B0/B1 = first 100 updates of reference / A3_n100 from z10, compose T10/T01/T11, 6 posts of the
  fixed reference plan 216→469, 7 surrogate40 scores, summary.json with the pre-registered verdict) → eval 45867090 afterok (dev300
  for T00, T11). Out: $B/reports/runs/r03/{t1_smoke,t1,t1_eval}. Est. ~4.2 + 0.8 + 0.3 GPU-h; committed total ≈ 34 ≤ 40.
- **22:5x CDT 10-13: user rebutted six points of my review2; I accept all six** (errata appended to review2):
  closure dim d does not imply p_train ≥ d (≥ ⌈d/p⌉ to span); R³ is the per-term |·|-sum certification scheme, not a lower bound
  (signed sum O(σ²R²/ε²); random factorial design also R²); R4 noise floor is that estimator's (paired U-statistic removes it,
  still vacuous at n=10–100); my "0.73–0.76" optimal-α threshold was an arithmetic error (forgot α²) → nested MFMC threshold
  0.575 at c=0.1; R16 inner max over a box is convex maximisation (2^r vertices, r ≤ 6 → enumerable); warm-Adam Δv/v =
  (1−β₂)(g²/v−1), not 0.1%. Also: keep strict bounds strict-but-vacuous; value-model route must compare with DR-OPE (Jiang–Li 2016).
  User's priorities: (1) tighten certificate wording; (2) check whether R16 is feasible from existing logs; (3) warm-checkpoint
  state transplant (2×2 m/v arms) + short-vs-full correlation; (4) cost lower bounds per explicit decision target; R2 as planned;
  title "continued / post-checkpoint training". Findings: R1 jsonl keeps only per-(candidate, replica) aggregates over the 40
  surrogate trajectories (geo_prob/per_token/per_traj/macro), no per-item outputs, and no logged "target feedback" for 𝓜 →
  R16 not credible from current logs. continue_from_state restores the full optimizer.pt, so m/v transplant is mechanically easy.
- **22:0x CDT 10-13: user shared the SECOND revision of that proposal** (docs/direction/2026-10-13_learning_response_framework_revision2_ZH.md,
  doc-dated 10-14; adds §3.4 shared/anchor reuse R1–R4, §3.5 multifidelity R5–R10, §4.4 bridge R=C·T (R11), §5.3 exact Adam memory
  counterexample + full AdamW adjoint R13–R15, §6.4 composed bound R19–R20, §8.5 information–compilation radius R16–R18, new neighbours
  Lessard 2019 / Gunn 2026 / Saunshi 2022). My review: docs/direction/2026-10-13_learning_response_framework_review2_ZH.md.
  All new derivations correct; Adam example re-derived exactly (ratio² = 1960/1249); appendix A and C scripts PASS, C output = appendix D.
  Accepted their correction of my cost unit (m·n·p_train, not pmn → ~400 GPU-h/checkpoint for 20 cand × 4 reps). Main new findings:
  R4 and R19–R20 are vacuous at our scale (dev300 eval noise SD 0.013; Hoeffding term → Δ̂ 0.72 at n=10 pairs, 0.40 at n=100,
  ~5e6 pairs for 0.03; per-step β ≥ ~2.6 summed over steps); multifidelity cuts replicates per cell, not cells (≤ n_H-fold);
  R8 needs short/full correlation ≳ 0.77; bridge needs closed tests (p_train ≥ d) — tension with the p_train=1 cost estimate;
  Adam example is elementary + cold-start (t=2); E1 as designed unrealisable at 12B → proposed optimizer-state transplant.
  Scores: novelty medium (organisational, no non-obvious composed result), completeness medium–high, rigor high, generality high.
  Suggested: R16 routing first (offline on R1/R2 data, no GPU), an estimation-cost lower bound as the composed theorem; still not
  merged into theoretical_framework (no user go-ahead).
- **19:4x CDT 10-13: user shared a revision proposal "学习响应谱统一框架" (docs/direction/2026-10-13_learning_response_framework_revision_ZH.md,
  based on snapshot 8658698) and asked for an assessment (novelty/completeness/rigor/generality). My review
  (docs/direction/2026-10-13_learning_response_framework_review_ZH.md): rigor high (derivations checked; appendix script re-run PASS;
  §11 evidence numbers match the repo), generality high but non-predictive, completeness medium (no estimation-cost theory, no
  cross-checkpoint reuse, no separation vs LESS/TRAK), novelty low–medium (classical instantiations; Thm 4 = simulation lemma,
  response matrix ≈ counterfactual datamodels; machine teaching uncovered in old and new). Flaw: §2.1 existence bound vacuous under
  its own mixture measure (B²=2J → r+1 ≥ J/(2ε²)). Recommended: absorb the rigor fixes (assumption audit, Thm B → conditional + R3
  negative, R0 prereg failure, Thm 4 into §8, singular-design Lemma 6.4, convex realizability); do not recentre the paper on the
  response spectrum unless a learner-specific structural theorem or a cost-matched separation result is added. Awaiting the user.
- **03:45 CDT 10-13: R2 A1_n100 45733057 TIMEOUT at its 1:30 limit (123/395 updates, no checkpoint) — R2 cost was underestimated ~8x.**
  Cause (engineering, not the method): an extra-pool update is token-matched (~515 supervised tokens) but each teacher label is a
  single Action turn (~8.5 supervised tokens) with the full student prefix, so one extra update = ~60 full-context forward/backward
  passes (microbatch 1, eager attention, grad checkpointing — the cold recipe, kept for comparability with A0): median 170 s vs
  4 s for a reference update. Per arm ≈ 98 extra × ~165 s + 297 ref × 4 s ≈ 4.9 h (A2 arms shorter: mean state step 8.7/11.3 vs
  ~17). Projection R0–R3 ≈ 62 GPU-h (used 15.2; R2 arms ~27 + evals 2.4; R1 shards 3.3; R1 continuation 3 arms ~14 + evals
  1.3) vs the approved ~40 cap → asked the user. Actions within the cap: dead eval 45733058 cancelled; A1_n100 partial output
  moved to $B/_trash/r03_r2_A1_n100_timeout_20261013/; resubmitted A1_n100 45754199 (6:30) + eval 45754200 afterok; A3_n100
  45733065 raised to 6:30 and released (A1_n100 + A3_n100 = the primary A3−A1 test at n=100, ~10 GPU-h, total ≈ 29 < 40).
  A1_n400 45733059 (limit 6:30), A2_n100 61, A2_n400 63, A3_n400 67 HELD (scontrol hold; limits 1:30 except 59 — raise to 6:30
  before release; raising a pending job's limit works for users). Their evals 60/62/64/68 wait on them.
- **22:02 CDT 10-12: hpg Qwen + garbage/cache deletion DONE** (user, tc channel 21:46 CDT: "tc上面所有和qwen相关的全部删掉，垃圾缓存全部删掉").
  305 items deleted (290 Qwen3.5-4B items ≈1,705 GiB; _trash dirs incl. this project's 47.5 GiB; vllm/uv/triton/inductor/envs caches).
  dept /blue 17,615 → 15,751 GiB used (of 20,480). Record before deletion: tc_code notes/hpg_qwen_deletion_20261012.{tsv,md}
  (b10ec9d3); log tc-alignment/logs/hpg_qwen_deletion_20261012.log. After: ENV/TRAIN/VLLM pythons import OK; ALFWorld data present.
  rai not touched (asked the user). Supersedes the "FROZEN until the user decides" note below.
- **23:05 CDT 10-12: R2 continuation protocol fixed BEFORE any R2 training** (prereg R2 + addendum B; details the prereg left open):
  start z10 = pilot-warm-1010 cold checkpoint-tokens-104070; target p40 = 416,280 supervised tokens (continue_from_state default);
  fraction 0.25 of the remaining updates replaced by batches from the arm's labels (same rule as R1), --seed 1202 for every
  arm; arm label pools = r2_union filtered to the arm's states (allocations_scalefree/A{1,2,3}_n{100,400}.json; rejected
  labels simply absent, counts recorded); A0 = the existing cold p40 (no queries, all-replay plan; dev300 241/242). 6 new arms,
  dev300 eval each (r03_dev300, same serving setup as the noise reference). Criteria = addendum B. R1 arms (P2/P3/P4,
  fraction 0.25, --seed 1201) only after the full 4-way certificate ranking is written here.
- **22:45 CDT 10-12: hpg dept /blue full (18.24T/20T) — cleanup coordinated by hq (control plane); tc gave a list only.**
  Live dependencies (never delete): tc-alignment/envs/ (vllm-serve venv = VLLM_PYTHON, alfworld data), tc-alignment-vllm/,
  tools/hf-cache, tools/uv-pythons, ICML27_*; on yd24f: /blue/yd24f.fsu/xc25.fsu/hq/tc-alignment/.venv (TRAIN_PYTHON; the
  dept path is a symlink) and /blue/yd24f.fsu/xc25.fsu/hq/tools/uv-pythons. tc-alignment/results/appworld_students (1.2T)
  and similar dirs hold ~1.5T of Qwen3.5-4B student weights: FROZEN until the user decides (/orange, rai, or delete; his
  09-23 "move everything Qwen" missed them). If hub_merged dirs are ever deleted: alfabl_CE_conseq_s0, crcd_hyb_stage1_s0,
  crcd_r2_b3pref_s0 hub_merged are training bases of 7 downstream students — re-export them before any downstream retrain;
  re-export needs Qwen3.5-4B @851bf6e (only on rai now). RTD dirs (9, 70G) re-merge from results/rtd_v1/*/round-N/lora.
  To do myself after sdl's rsync finishes: delete tools/vllm-cache (8.8G torch.compile cache; hq agreed).
- **20:40 CDT 10-12: teacher collection RUNNING with the new key** (user sent it 20:16 CDT, valid 30 days; sha8 c43d7f3d; in
  rai llm_apis.env and hpg secrets/openai.env, old files kept as .bak-20261012-oldopenai). teacher-r1 45727129 (tree 9cf3201,
  run IDs r03-r1-p2-k2 / r03-r1-p4-k2 — the old run ID is halted by its 401 by design): P2 400/600 success, 0 failed.
  teacher-r2 45727130 afterok (r03-r2-union, 558 new calls + 125 cache hits). Next: P2/P4 CRN items (seed 1101, r=64, T=10)
  → certificate → full 4-way ranking here → R1 continuation arms; R2 arms after its labels.
- **14:48 CDT 10-12: R1 certificate P1 vs P3 DONE — PRE-REGISTERED (partial) RANKING: P1 > P3.** 45624783–86 (tree 57131a6),
  r=64, T=10, tail seed 1101. Ŝ(P1−P3): geo-prob +0.00138 (57/64 replicas positive, SE 1.5e-4, diagnostic radius 0.00121 →
  certified by the diagnostic radius only; strict a-priori radius 0.325), per-token +0.00156 (54/64, diagnostic 0.00150 →
  borderline). Prediction ordered P3 above P1 → this pair goes against it on the surrogate. Reading: the student already fits
  its own successes (message loss 0.128 vs 0.211, grad norm 2.42 vs 3.76), and the surrogate is 40 held-out teacher
  trajectories, which favours teacher-distribution data (surrogate alignment uncalibrated, Remark 8.11). This ranking is fixed
  before any continuation; P2/P4 join with the same seed stream once the teacher key works; the full 4-way ranking is
  written here before the R1 continuation arms are submitted.
- **07:18 CDT 10-12: R3 DONE — Cor 7.2 prediction NOT supported on the real model at p10.** Measured δF(c) (per-token, vs zero
  message, r=16): +0.0005/+0.0012/+0.0011/+0.0007/+0.0002 for c=0.5/1/2/4/8 clip off, all inside the diagnostic radius
  (0.004–0.007); predicted −0.019…−0.080 (15–300× larger, opposite sign). Clip on c=1 vs 4: equal means (diff 8e-8) but
  per-replica |Z| up to 0.003 although consumed gradients agree up to rounding → a floating-point chaos noise floor of ~3e-3
  per replica after 10 tail updates. Likely failed assumption: linear response (tail gradients independent of θ, H=0). The
  measured curve is flat-to-concave within noise. To record in the §7 experiments page as a real-model check (negative).
- **06:24 CDT 10-12: R2 allocation fixed before any query** (prereg addendum B, 2b86cd4): σ² spans 1e4–1e8, so A3 = dσ² ≈ A2;
  Prop 6.11(b)'s Adam weight uses a scale-free variance → A3 W ∝ d. Predicted (scale-free) errors n=100: A3 .046 < A2 .055 <
  A1 .058; n=400: A3 .028 < A1 .031 < A2 .039. Teacher total R1 600 + R2 558 = 1,158 calls (cap 2,140). Still blocked on the key.
- **04:58 CDT 10-12: teacher BLOCKED — OpenAI key invalid.** teacher-r1 45624540 FAILED after 1 attempt: HTTP 401
  "Incorrect API key provided" (free /v1/models probe also 401). Same key on rai llm_apis.env and hpg openai.env
  (sha8 c7cee533, written 09-10). Asked the user to put a new key into both files (never via Discord). No spend.
  Meanwhile R1 certificate for P1 (K20 replay) and P3 (39 self successes) SUBMITTED: 45624783–86 (tree 57131a6, items
  inputs/r1/items_p1p3.json, r=64, T=10, tail seed stream 1101 — P2/P4 will reuse it for pairing).
- **04:54 CDT 10-12: rollout-p10 DONE** (45605273, 33 min): 90/200 pool games succeed; P3 = 39 successes in R1_P3 subset;
  R2 statistics: 24 clusters (look_at_obj 5-9/10-19/20+ have k=0 teacher visits; long "20+" buckets have high d).
  SUBMITTED: teacher-r1 45624540 (tree b09b49b, hpg-milan CPU, P2 corrections cap 600 [dry run 600], P4 rewrites cap 40
  [dry run 20]; cache/ledger $B/data/raw/r03_teacher_*), gradstats 45624570 (tree 57131a6, σ² per cluster at z10).
- **04:19 CDT 10-12: R0b DONE** (45604561–64). Response non-additivity ν_resp (median over 4 pairs, pair means over r=4):
  update 8: 0.19 (prereg prediction [0.26,0.76] around update-map ν 0.511 → MISSED; criterion >0.2 → NOT met, borderline);
  p10: 0.89 (prediction [0.38,0.88] around 0.631 → at the edge, per-token 0.86 inside; criterion >0.2 → met). Caveat: at p10
  the mean δF(A∪B) ≈ −4e-4 is below the per-replica scatter (±1e-3), so ν is a ratio of noisy small numbers; r=4 too few.
  Sign: one K20 batch raises held-out loglik at update 8 (+6–10e-3) but lowers it at p10 (−2–6e-4), as R3's prediction said.
  rollout-p10 45605273 RUNNING since 04:15 CDT.
- **03:32 CDT 10-12: R0a DONE** (45604560, 19 min). Pre-registered criterion E_fd ≤ 0.1·|Σ L̂_i Δθ_i| FAILS at all 6 states, but
  that denominator is mis-specified (L̂_i = |∂F̃/∂θ_i| times signed Δθ cancels; 7e-6 at p10). Against the true first-order term
  |Σ ∂F̃/∂θ_i Δθ_i| (post hoc, labelled so): ratio 29.7 (update 1), 2.1 (4), 0.98 (8), 0.100 (p03), 0.097 (p10), 0.084 (p40).
  Quantiles say κ>1 for only 0.01% of coordinates at p10, yet the weighted frozen-denominator error is ~10% of the first-order
  term: quantiles overstate warmth (Remark 5.11). Top terms: LoRA-B, layer-13 MLP gate_proj.
- **03:20 CDT 10-12: noise reference DONE**: cold p40 on dev300 = 241/300 (server a, 45604558) and 242/300 (server b, 45604559);
  15 task flips between identical servers (7/8). Use paired McNemar on flips; differences ≲ ±8 tasks are server noise.
- **02:56 CDT 10-12: framework reorganized per result + references** (11588ae). User: every linked page must be readable alone;
  one statement/proof/experiments set per numbered item; README Level 2+3 merged with per-item links; complete correct
  references; no earlier labels anywhere (appendix = sources only, f69bbf0). Now: statements/KK-JJ_<slug>.md (67 items),
  proofs/ (46), experiments/ (48), EN+ZH; references.md (38 entries verified online). Validator enforces the layout.
  Checks: offline 356 files 0 errors, MathJax 31,886 expressions 0 errors, file view 329 pages pass; max 371 expr/page.
- **01:54 CDT 10-12: framework pages split for GitHub's browser math limit** (e0e9fb0). User screenshot: README "Unable to render
  expression" on valid math. Cause: undocumented per-page limit in GitHub's client-side renderer (README 1,251 expressions;
  known-good pages 386 and 666; local MathJax renders all; HTML wraps all). Now every page ≤ 600 (README 202): Level 3 →
  statements/0K_*.md; proofs 04→04+04_rates_and_composition, 05→05+05_warm_adam+05_learner_statistics,
  08→08+08_certificate; experiments 05→05+05_warm_adam_and_learners. Validator enforces ≤ 600/page. Asked the user to confirm.
- **00:46 CDT 10-12: R1/R2 rollout SUBMITTED** on tree fa6ccd5: rollout-p10 45605273 (cold p10 adapter, greedy, BI serving, all
  200 pool games, 2 temperature-1 label samples per visited state for R2's σ²; no teacher). Out $B/reports/runs/r03/rollouts_p10.
- **00:36 CDT 10-12: R3 SUBMITTED** on tree d6fe9df: r3-amp 45604739 (Corollary 7.2(a) prediction summed over coordinates at
  z10, batch plan[115]=[11,12,2,19], fine grid 0–16; written before any CRN outcome), r3-0..5 45604740–45 (afterok:amp; 8
  messages {zero, c∈{0.5,1,2,4,8} clip off, c∈{1,4} clip on} × r=16, T=10; 128 items, 6 shards). Out $B/reports/runs/r03/r3.
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
