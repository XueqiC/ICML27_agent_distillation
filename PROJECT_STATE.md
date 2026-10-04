# PROJECT_STATE — Paper 1: Sufficient Experience for Black-Box Agent Distillation

Governing document: `~/hq/projects/tc-alignment/docs/handoff/2026-10-04_paper1_context_ZH.md` (sections 6, 7, 10–12, 14).
Discord channel: tc (1536171352573349909). Compute: hpg primary (`/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation`), rai for CPU/smoke.
Teacher: gpt-6-luna via the original OpenAI key, no spend cap (user, 2026-10-04). Student: google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7.

## RUNNING JOBS
- none

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
  ⚠ Running `python harness/run_episode.py` as a file fails: `harness/alfworld` shadows the installed `alfworld` package (ModuleNotFoundError at reset).
  Use `python -m harness.run_episode`; rename the package in stage 2.

## Open decisions (asked the user)
1. Interaction format: **ReAct accepted** (user, 2026-10-04 15:39). Gemma `enable_thinking`: the user asked why off. I explained
   (teacher reasoning is not returned, so the only supervisable reasoning is its visible Thought; the Gemma template strips earlier-turn
   thinking via `strip_thinking`, while ReAct Thought stays in history; one fixed mode for all methods). I proposed a pilot: base student,
   ReAct, 40 fixed dev games, thinking off vs on, then fix the mode. User replied by quoting the plan (15:46); taken as approval
   and said so on Discord. Codex stage 2a (ReAct template/parser, student policy over vLLM with enable_thinking switch, dev40,
   suite runner, hpg slurm script, harness/alfworld rename) started 15:48.
2. K=10 seed selection rule: proposed stratified random over the 6 task types, 3 sampling seeds, up to 3 teacher attempts per task,
   keep only execution-verified successes. Awaiting reply.

## Next
Stage 2: student serving/eval (vLLM, merged LoRA), Seed-SFT trainer with assistant-only loss masks, K=10 seed collection, base/teacher/Seed-SFT
smoke on a fixed dev subset, hpg environment + profiling report.
