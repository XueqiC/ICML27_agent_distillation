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
