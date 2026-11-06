# P1 setting A — final report (rsi, 2026-11-06)

Source: `reports/p1_runs/A/eval/report.json` on rsi (sha256 ec8d738c…, merge 16:16 CST; plan fb8724cb…; tree d082a6a; E4 frozen from 3138f4a per prereg §11; A shards resumed after the rsi reboot per §12). Point estimates only (user 11-05).

## Generated table (report.md, verbatim)

# P1 A

Point estimates; M2 K20: original K20 tasks. Unknown outcomes are retained.

| Condition | M1 | decision_flip_rate | M2_probe | M2_k20 | M3 | M4 |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 22.41% | 0.00% | 69.01% | 100.00% | 0.00% | 55.71% |
| E1 | 37.93% | 21.74% | 70.42% | 100.00% | 8.00% | 57.14% |
| E2_0 | 18.97% | 0.00% | 66.20% | 100.00% | 2.00% | 52.14% |
| E2_1 | 13.79% | 0.00% | 59.15% | 100.00% | 2.00% | 60.00% |
| E2_2 | 24.14% | 0.00% | 69.01% | 100.00% | 2.50% | 55.00% |
| E2_3 | 18.97% | 0.00% | 63.38% | 100.00% | 2.00% | 55.71% |
| E2_4 | 17.24% | 0.00% | 66.20% | 100.00% | 2.50% | 55.71% |
| E3 | 27.59% | 15.22% | 66.20% | 100.00% | 9.00% | 55.00% |
| E4 | 50.00% | 50.00% | 71.83% | 100.00% | 15.00% | 58.57% |
| E5 | 27.59% | 13.04% | 63.38% | 100.00% | 9.00% | 52.86% |
| E2_mean | 18.62% | 0.00% | 64.79% | 100.00% | 2.20% | 55.71% |

McNemar results are in the JSON appendix only. M4 remains pending until standalone dev140 results are imported.

Counts: M1 E0 13/58, E1 22/58, E2 seeds 11/8/14/11/10, E3 16/58, E4 29/58, E5 16/58. M3 E0 0/200, E1 16/200, E3 18/200, E4 30/200, E5 18/200. M4 E0 78, E1 80, E2 73/84/77/78/78, E3 77, E4 82, E5 74 (of 140).

## Prereg §6 criteria

| Criterion | Required | Result | Met |
|---|---|---|---|
| 1. dev140 E1 vs E0, E2 mean, E3, E5 | ≥ +3 pp each | +1.4 / +1.4 / +2.1 / +4.3 pp | no |
| 2. M1 E1 vs E0, E2 mean, E3, E5 | ≥ +5 pp each | 37.9% vs 22.4 / 18.6 / 27.6 / 27.6 → +15.5 / +19.3 / +10.3 / +10.3 pp | yes |
| 3. dev140 E1 ≥ E4 | ≥ | 57.1% < 58.6% | no |
| 4. M3 ≤ 5%, K20 originals no drop | ≤ 5% | M3 8.0%; K20 20/20 | no (M3) |

Pre-registered branch: criterion 1 not met → "on existing data, teacher-localized weight editing does not beat simple methods" (consistent with P0 and D-param).

## Reading

- The teacher's edit is a real local repair: on held-out failure states of the same categories (M1) it beats no edit, random edits of the same structure, the fixed-rule direction and the automatic localized LoRA by 10–19 pp.
- The local gain does not carry to whole tasks: dev140 moves +2 of 140 (+16 / −14 task flips), inside the spread of random edits (73–84), and M2 synthetic held-out tasks move 49 → 50 of 71.
- SFT on the same repair data (E4) is better on every efficacy metric (M1 50.0%, dev140 58.6%, M2 51/71) but has the largest collateral (15%).
- Path check: the unedited merged export E0 scores 78/140 vs 82/140 for theta0 through the LoRA path on the same rsi GPUs (18 task flips), so merge/export alone moves ~13% of tasks.
- Same-hardware references (LoRA path, rsi): v0 uniform 99/140, D-data 91/140; every P1 condition is ≤ 84/140.

## Appendix: dev140 run-to-run noise on the merged path (added 17:1x CST)

Every A condition was evaluated a second time with the same formal runner (pre-registered 16:2x/16:5x CST; outputs `A/dev140_rep2`, not imported into the report). E0 was additionally run a third time and once with all compile caches cleared.

| Condition | Run 1 (in report) | Run 2 | Outcome flips |
|---|---:|---:|---:|
| E0 | 78 | 76 (rep3 78, cold-cache 78) | 24 |
| E1 | 80 | 80 | 0 |
| E2_0 | 73 | 80 | 21 |
| E2_1 | 84 | 78 | 16 |
| E2_2 | 77 | 77 | 0 |
| E2_3 | 78 | 76 | 12 |
| E2_4 | 78 | 78 | 0 |
| E3 | 77 | 77 | 0 |
| E4 | 82 | 82 | 0 |
| E5 | 74 | 71 | 25 |

- Half of the conditions reproduce exactly; the other half shift by 2–7 tasks with 12–25 flips. The compile cache is not the cause (cold-cache E0 equals run 1), and the servers did not cross-talk (each client hit its own port and model). The likely source is timing-dependent request scheduling in vLLM (16 concurrent workers) that BATCH_INVARIANT=1 does not fully neutralize; not yet verified.
- Consequence: dev140 differences of a few tasks between P1 conditions are within run-to-run noise.
- Two-run means: E0 77.0, E1 80.0, E2 77.9, E3 77.0, E4 82.0, E5 72.5. Criterion 1 is still not met (E1 +2.1 / +1.5 / +2.1 / +5.4 pp over E0 / E2 / E3 / E5), and E1 is still below E4, so the verdict is unchanged.
