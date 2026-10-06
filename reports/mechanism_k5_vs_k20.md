# Failure mechanisms vs K: K=5 and K=20 students (2026-10-06)

Frozen gap set = dev140 episodes the teacher solves and the student fails. Attribution follows MECHANISM_PLAN Section 6 with amendments 3–6 (closest-approach interventions, procedural coverage gate, M1a/M1b probes on seed states). Counts are episodes; rates in parentheses are over the gap set.

| attribution | K=20 s0 (20 passes, rai 96/140) | K=5 s0 (~400k tokens, rai 74/140) |
|---|---|---|
| task type absent from the K seeds (separate) | 0 (0%) | 13 (30%) |
| experience-type rescue, needed step not covered | 0 (0%) | 2 (5%) |
| decision-info rescue, needed step not covered | 0 (0%) | 4 (9%) |
| M1a: covered, not absorbed | 0 (0%) | 0 (0%) |
| M1b: covered, absorbed, not invoked | 4 (17%) | 1 (2%) |
| both rescue signals | 4 (17%) | 0 (0%) |
| M2: needs more explicit decision information | 7 (29%) | 4 (9%) |
| only the teacher's action rescues | 1 (4%) | 4 (9%) |
| no single-step intervention rescues | 8 (33%) | 15 (35%) |
| **gap size** | 24 | 43 |
| training-set fit (teacher-forced seed states) | 93.7% | 97.5% |

Reading: absorption failure (M1a) is zero at both K; the student fits its seeds. As K shrinks, the gap is dominated by knowledge that is not in the K seeds at all (task type absent or needed step uncovered: K=5 19/43 = 44% vs K=20 ≈ 0), and single-step rescues shift from advice (I3) to the teacher's action (I4). Intervals are wide (Wilson 95% in the readouts); one student per K, one support seed.
