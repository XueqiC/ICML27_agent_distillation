Teacher 114/140; base 62/140. Cells: mean successes over support seeds [per-seed values]; dev140, batch-invariant serving, hpg B200.

| method | K=5 | K=10 | K=20 |
|---|---|---|---|
| Seed-SFT (3 passes) | 61.0 [61] | 61.0 [59, 65, 59] | 52.7 [49, 52, 57] |
| Seed-SFT (40 passes) | 74.7 [63, 86, 75] | 88.0 [86, 80, 98] | 107.7 [115, 110, 98] |
| Seed-SFT (~400k tokens) | 84.7 [72, 101, 81] | 94.3 [95, 85, 103] | 107.7 [115, 110, 98] |
| Kang (official) + SAG | — | — | 63.0 [63] |
| AgentGym BC | — | — | 59.0 [59] |
| AgentEvol (final iter.) | — | — | 66.0 [66] |
