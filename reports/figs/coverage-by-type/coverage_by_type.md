# Per-task-type successes vs seed coverage (dev140, 2026-10-05 15:0x CDT)

Seeds row = number of seed trajectories of that type in the K set; bold 0 = type absent. Kang = official recipe + SAG (hpg BI). Seed-SFT = main-table checkpoint (K5/K10 t400k, K20 p40; hpg sweep). — = not yet available here (k10s1 Kang running; k10s2-t400k / k20s2-p40 Seed-SFT live in the schema-5 re-eval root).

| row | simple /35 | look /13 | clean /27 | heat /16 | cool /25 | two /24 | total |
|---|---|---|---|---|---|---|---|
| base | 31 | 7 | 11 | 1 | 1 | 11 | 62 |
| teacher | 34 | 13 | 19 | 12 | 15 | 21 | 114 |
| **K5s0** seeds | 1 | 1 | 1 | 1 | **0** | 1 | 5 |
| K5s0 Kang | 30 | 11 | 6 | 2 | 0 | 8 | 57 |
| K5s0 Seed-SFT t400k | 31 | 11 | 8 | 3 | 3 | 16 | 72 |
| **K5s1** seeds | **0** | 1 | 1 | 1 | 1 | 1 | 5 |
| K5s1 Kang | 32 | 13 | 12 | 8 | 14 | 19 | 98 |
| K5s1 Seed-SFT t400k | 33 | 10 | 16 | 9 | 13 | 20 | 101 |
| **K5s2** seeds | 1 | **0** | 1 | 1 | 1 | 1 | 5 |
| K5s2 Kang | 31 | 10 | 7 | 4 | 3 | 13 | 68 |
| K5s2 Seed-SFT t400k | 32 | 8 | 12 | 4 | 7 | 18 | 81 |
| **K10s0** seeds | 2 | 2 | 2 | 2 | 1 | 1 | 10 |
| K10s0 Kang | 35 | 13 | 13 | 3 | 7 | 9 | 80 |
| K10s0 Seed-SFT t400k | 34 | 13 | 14 | 9 | 11 | 14 | 95 |
| **K10s1** seeds | 1 | 2 | 2 | 1 | 2 | 2 | 10 |
| K10s1 Kang | — | — | — | — | — | — | — |
| K10s1 Seed-SFT t400k | 34 | 9 | 11 | 2 | 15 | 14 | 85 |
| **K10s2** seeds | 2 | 1 | 2 | 2 | 2 | 1 | 10 |
| K10s2 Kang | 35 | 8 | 13 | 10 | 9 | 10 | 85 |
| K10s2 Seed-SFT t400k | — | — | — | — | — | — | — |
| **K20s0** seeds | 3 | 3 | 4 | 4 | 3 | 3 | 20 |
| K20s0 Kang | 29 | 12 | 14 | 4 | 1 | 3 | 63 |
| K20s0 Seed-SFT p40 | 35 | 8 | 22 | 12 | 17 | 21 | 115 |
| **K20s1** seeds | 3 | 3 | 4 | 3 | 4 | 3 | 20 |
| K20s1 Kang | 32 | 10 | 13 | 5 | 5 | 10 | 75 |
| K20s1 Seed-SFT p40 | 35 | 11 | 19 | 8 | 16 | 21 | 110 |
| **K20s2** seeds | 3 | 3 | 4 | 3 | 4 | 3 | 20 |
| K20s2 Kang | 33 | 11 | 21 | 11 | 14 | 20 | 110 |
| K20s2 Seed-SFT p40 | — | — | — | — | — | — | — |

## K=1/2 Seed-SFT (added 2026-10-06 13:0x CDT; K100 views; t200k/t400k; K2s2 pending sync — its total is 59 at t400k)

| row | simple /35 | look /13 | clean /27 | heat /16 | cool /25 | two /24 | total |
|---|---|---|---|---|---|---|---|
| base | 31 | 7 | 11 | 1 | 1 | 11 | 62 |
| **K1s0** seeds | · | · | · | 1 | · | · | 1 |
| K1s0 Seed-SFT t200k | 31 | 4 | 11 | 5 | 4 | 17 | 72 |
| K1s0 Seed-SFT t400k | 31 | 6 | 8 | 4 | 6 | 12 | 67 |
| **K1s1** seeds | · | · | 1 | · | · | · | 1 |
| K1s1 Seed-SFT t200k | 31 | 10 | 10 | 0 | 2 | 4 | 57 |
| K1s1 Seed-SFT t400k | 31 | 10 | 10 | 0 | 1 | 5 | 57 |
| **K1s2** seeds | · | · | 1 | · | · | · | 1 |
| K1s2 Seed-SFT t200k | 31 | 4 | 11 | 0 | 2 | 8 | 56 |
| K1s2 Seed-SFT t400k | 31 | 5 | 11 | 0 | 1 | 7 | 55 |
| **K2s0** seeds | · | · | 1 | 1 | · | · | 2 |
| K2s0 Seed-SFT t200k | 32 | 7 | 15 | 5 | 5 | 7 | 71 |
| K2s0 Seed-SFT t400k | 31 | 7 | 13 | 4 | 2 | 13 | 70 |
| **K2s1** seeds | · | · | 1 | · | 1 | · | 2 |
| K2s1 Seed-SFT t200k | 32 | 8 | 16 | 5 | 14 | 12 | 87 |
| K2s1 Seed-SFT t400k | 34 | 10 | 15 | 3 | 15 | 9 | 86 |
| **K2s2** seeds | · | · | 1 | · | 1 | · | 2 |

Reading: one heat seed (K1s0) lifts heat 1→4–5 and cool 1→4–6; one clean seed (K1s1/K1s2) adds nothing on clean and hurts two-object (11→4–8); adding one cool seed (K2s1) lifts cool 1→14–15 (+28 total). K2s2 has the same coverage (clean + cool) but stays at 59 → which trajectory is in the seed set matters, not only its type.
