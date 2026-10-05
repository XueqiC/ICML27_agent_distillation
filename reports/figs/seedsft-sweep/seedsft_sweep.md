Base 62/140 = 44.3%; teacher 114/140 = 81.4%. Cells: successes/140; F = equivalence gate failed (not evaluated); · = merged into a nearby pass point (within 5%); † = evaluated after gate amendment 4 (schema 5).

| K | support seed | p03 | p05 | p10 | p20 | p30 | p40 | t100k | t200k | t400k |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | s0 | F | 64 | 57 | 49 | 54 | 63 | 71 | 75 | 72 |
| 10 | s0 | 59 | 55 | 46 | 68 | 79 | 86 | 65 | 83 | 95 |
| 20 | s0 | 49 | 45 | 70 | 100 | 106 | 115 | · | · | · |
| 5 | s1 | 61 | 63 | 60 | 66 | 68 | 86 | 86 | 88 | 101 |
| 10 | s1 | 65 | 60 | 58 | 68 | 76 | 80 | · | · | 85 |
| 20 | s1 | 52 | 51 | 73 | 100 | 105 | 110 | · | · | · |
| 5 | s2 | F | 65 | 61 | 45 | 52 | 75 | 77 | 82 | 81 |
| 10 | s2 | 59 | 58 | 49 | 70 | 86† | 98† | 81 | 104† | 103† |
| 20 | s2 | 57 | 45 | 71 | 96 | 92† | 98† | 79 | 99 | 98† |

Per-K means over the support seeds that have an evaluated cell (n seeds in brackets):

| K | 3 passes | 10 passes | 20 passes | 40 passes | ~400k tokens | best cell (seed, dose): success, vs base only-row/only-base, p |
|---|---|---|---|---|---|---|
| 5 | 61.0 [1] | 59.3 [3] | 53.3 [3] | 74.7 [3] | 84.7 [3] | s1 t400k: 101/140, 43/4, p=2.8e-09 |
| 10 | 61.0 [3] | 51.0 [3] | 68.7 [3] | 88.0 [3] | 94.3 [3] | s2 t200k: 104/140, 46/4, p=4.5e-10 |
| 20 | 52.7 [3] | 71.3 [3] | 98.7 [3] | 107.7 [3] | 107.7 [3] | s0 p40: 115/140, 56/3, p=1.2e-13 |

Serving-noise reference: base vs base on two non-invariant servers flipped 12/140 outcomes; under batch-invariant serving every number above is exactly reproducible.

