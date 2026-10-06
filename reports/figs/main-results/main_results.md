Teacher 114/140; base 62/140. Cells: mean successes over support seeds [per-seed values]; dev140, batch-invariant serving, hpg B200.

| method | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| Seed-SFT (3 passes) | — | — | 61.0 [61] | 61.0 [59, 65, 59] | 52.7 [49, 52, 57] |
| Seed-SFT (40 passes) | 58.0 [54, 59, 61] | 55.0 [54, 56] | 74.7 [63, 86, 75] | 88.0 [86, 80, 98] | 107.7 [115, 110, 98] |
| Seed-SFT (~400k tokens) | 59.7 [67, 57, 55] | 78.0 [70, 86] | 84.7 [72, 101, 81] | 94.3 [95, 85, 103] | 107.7 [115, 110, 98] |
| Kang (official) + SAG | — | — | 74.3 [57, 98, 68] | 78.7 [80, 71, 85] | 82.7 [63, 75, 110] |
| AgentGym BC | — | — | — | — | 59.0 [59] |
| AgentEvol (final iter.) | — | — | — | — | 66.0 [66] |
| LEAP (privileged) | — | — | — | — | 60.0 [60] |
| LEAP (0% privileged) | — | — | — | — | 58.0 [58] |
