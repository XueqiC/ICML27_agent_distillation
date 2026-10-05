# Baseline dose controls (draft 2026-10-05, not yet approved)

## Why
Seed-SFT on K verified seeds dips below base at 30–60k supervised tokens and only clears it after ~100k (seedsft_sweep.md).
Every official baseline recipe trains its SFT stage for 1–3 epochs, which at K ≤ 20 lands inside that dip (Kang K20 s0: 63/140,
AgentGym BC 59, AgentEvol 62–66; base 62). An official-recipe row alone therefore cannot tell "the method's ingredient does not help"
apart from "the recipe was calibrated for thousands of trajectories and is under-dosed at few-shot K". The official rows stay the
baselines (they are what the papers specify); the controls below are labelled as controls and never replace them.

## Control: the method's ingredient on an adequately dosed SFT student
For each cell (K, support seed) the start point is the Seed-SFT checkpoint the main table reports for that cell
(K20: 40 passes; K ≤ 10: the ~400k-token point; K ≥ 50: chosen by the same rule once the v2 grid lands, never by sealed data).
Everything after the SFT stage follows the official recipe unchanged, including exploration only on the K training tasks.

| Method | Ingredient tested | Start | What changes vs official |
| --- | --- | --- | --- |
| Kang | SAG decoding (FTP has no released code) | Seed-SFT LoRA | nothing else; already run for K20 s0 (115 = greedy 115) |
| AgentEvol | explore on K tasks, keep successes, retrain | Seed-SFT merged in fp32 | iteration 0 = Seed-SFT instead of 1-epoch BC; each retrain from base on D_m at the Seed-SFT exposure (passes over D_m matched) |
| ETO | failure/success pairs on K tasks, DPO | Seed-SFT merged in fp32 | SFT stage replaced by Seed-SFT; DPO stages as official |
| IPR | step-level MC rewards, iterative DPO | Seed-SFT merged in fp32 | same as ETO |
| LEAP (privileged and 0%) | teacher corrections on student rollouts | Seed-SFT merged in fp32 | iteration 0 = Seed-SFT |

Full-FT methods start from base + LoRA delta merged in fp32 (the FSDP master precision), so training starts exactly at the
Seed-SFT function; serving is bf16 like every full-FT baseline. Expected mechanism: a dosed student already fits 90%+ of its
K training tasks (run 2 training fit 93.7%), so exploration finds few failures and pairs/corrections become scarce; that is a
finding about these ingredients at few-shot K, reported with the pair/correction counts per iteration.

## Order and cost
1. Official cells first (submitted 10-05: Kang and AgentGym K5/K10 × 3 seeds + K20 s1/s2; ETO/IPR/LEAP K20 s0 validating).
2. Controls at K20 s0 for all five methods (≈ 1 GPU-day on hpg), then the cells where the official row and Seed-SFT differ most.
3. Implementation (Codex): `--init-checkpoint` for the AgentGym/ETO/IPR/LEAP pipelines (fp32 merge, provenance hash of the
   Seed-SFT checkpoint, disclosure.json field `init`), tests on CPU, one dry run on rai before hpg.
