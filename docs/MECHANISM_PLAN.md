# Mechanism analysis plan: why distilled students fail (pre-registered 2026-10-04, before any distilled-student result)

User request (2026-10-04 16:52): separate the reasons a distilled student fails into at least two testable mechanisms:

- **M1. The knowledge is in the experience, but the student did not absorb it or does not invoke it.**
- **M2. The student needs more explicit decision information**: better teaching, or learning to legally re-observe / query.

This file fixes the populations, measurements, interventions and readout before the first Seed-SFT or baseline
number exists, so the analysis cannot be fitted to the results. Changes after results are appended as dated
amendments, never edited in place.

## 1. Populations (dev = valid_seen 140; valid_unseen stays sealed)

For each analysed student S (Seed-SFT at each K, then each baseline at its dev-selected checkpoint):

- **Gap set G(S)**: dev games where the teacher run succeeds and S fails (greedy). Primary population.
- **Regression set R(S)**: games where the base student succeeds and S fails. Reported separately (a distillation
  side effect, not a missing capability).
- Games where the teacher fails are excluded from the mechanism tests (teacher error is a confound, Section 5).

Greedy decoding is deterministic up to serving noise. Every branch replay first re-runs the unmodified prefix and
checks that S reproduces its own logged action at the branch point (replay check, Section 3, arm I0).

## 2. Failure taxonomy (automatic, auditable)

Computed from the event log; audit fields (PDDL facts, won) are used for labelling only, never in any prompt.
Primary label = first matching rule in this order; all matching rules are also kept.

| Label | Rule |
|---|---|
| F-format | the episode ends with ≥3 consecutive parser errors, or ≥25% of its steps are parser errors or inadmissible |
| F-goal | S places or acts on an object or receptacle whose type does not match the goal (misread goal) |
| F-search | a goal object is never visible in any observation of the episode |
| F-precondition | the goal object was held, but a required transformation or order is wrong (e.g. heat/cool/clean missing, attempted at the wrong receptacle, put before transform) |
| F-loop | the same (action, observation) pair occurs ≥3 times, or a cycle of length ≤4 repeats ≥3 times |
| F-cap | none of the above, 40-step limit reached |

## 3. First critical error and branch interventions

**First critical error (FCE).** For an episode in G(S), T(t) = 1 if the teacher, taking over after S's first t
steps (same real environment, reset + action-prefix replay with consistency checks), succeeds. The FCE is the
first t with T(t) = 1 and T(t+1) = 0, located by a scan at t = 0, 4, 8, … and then refined step by step inside the
first interval where T drops (2 teacher samples per probe point; T = 1 if either succeeds). If T(0) = 0 the game
is moved to "teacher-unsolvable from the start" and excluded.

**Interventions at the FCE state** (before S's erroneous action at step t+1). Each arm changes only the input of that
one step; afterwards S continues on its own under the unchanged harness. Outcome = success of the continuation
(greedy; plus 4 samples at temperature 0.7 for a rescue rate with a confidence interval).

| Arm | Change at the FCE step | What a rescue means |
|---|---|---|
| I0 | none (replay check) | must reproduce the failure; otherwise the episode is flagged as non-deterministic |
| I1 restate | append a deterministic summary of facts already in the visible history (receptacles visited, objects seen where, what is held) | information was present in context but not used |
| I2 re-observe | execute one legal information action in the real environment (`look`, `inventory`, or `examine` of the current receptacle, chosen by a fixed rule) and show its real result; costs one step | the student lacks the habit of legally re-observing |
| I3 teacher advice | the teacher sees the same visible history and writes a short plan/thought without an action; inserted as "Advice:"; S chooses the action | more explicit decision information (better teaching) suffices |
| I4 teacher action | the teacher's action replaces S's action at this step; S continues | upper bound: a correct decision at this step suffices |
| I5 experience in context | insert the most similar seed trajectory (same task type, from the K seeds S was trained on) as a worked example | the knowledge exists in the experience and is usable when in context |

The same arms are run for the base student on the same states where meaningful (I5 on base = pure in-context use of
the seeds; it separates "not absorbed by training" from "not usable at all").

## 4. Knowledge probes and coverage (M1 evidence outside the episode)

- **Coverage.** For each FCE, mark whether some seed trajectory of S's training set demonstrates the needed step
  pattern (e.g. "heat X with microwave 1" after taking X; finding X in a drawer or cabinet). Rule-based, from the
  seed event logs.
- **Probes.** Templated questions built only from the seeds (procedure: "To heat a mug, what must you do after
  taking it?"; location priors: "Where did you find a <obj> in the examples?"), scored against the seed text.
  Run on base and S without environment context. Probe accuracy splits M1 into **M1a not absorbed** (probe fails)
  and **M1b absorbed but not invoked** (probe passes, behaviour fails).
- **Training-set fit.** Teacher-action agreement of S on its own training states (teacher forcing) and on fresh
  rollouts of its training tasks; low fit points to dose/optimisation rather than to either mechanism.

## 5. Confounds checked before any mechanism claim

1. Format: F-format episodes are reported but analysed separately; the parser-error rate is reported per student.
2. Dose: the student is its dev-selected checkpoint; the same analysis is repeated on one earlier and one later
   checkpoint of the same run.
3. Coverage: task types absent from the seeds are reported separately (no M1 claim without coverage).
4. Teacher error: games the teacher fails are excluded; FCE requires T(t) = 1 before the error.
5. Noise: rescues are counted only if greedy and the sampled rate (≥3/4) agree; paired McNemar for student-level
   comparisons; dev140 resolves only about ±8 pp per arm, so arm-level differences under ~10 pp are not claimed.

## 6. Readout (fixed in advance)

Per student: |G|, taxonomy shares, FCE position, and the rescue rate of each arm with a 95% Wilson interval.
Decision table per FCE episode:

| | I5 or I1 rescues | neither rescues |
|---|---|---|
| **I3 or I2 rescues** | both mechanisms contribute | **M2** (needs explicit decision information) |
| **neither rescues** | **M1** (knowledge in experience; M1a/M1b by probe) | only I4 rescues → execution/capacity; none → unresolved |

An M1 label also requires coverage (Section 4). The headline statement is the share of gap episodes in each cell,
per K, with intervals, and how the shares move as K grows.

## 7. Budget

Teacher calls: about 6 probe points × 2 samples × ~10 steps per gap episode for the FCE, plus one call per I3/I4
arm. With ~40–60 gap episodes per student this is a few thousand calls per student, cached, with the usage ledger.
Student branch replays run on our own GPUs.

## Amendment 1 (2026-10-04 22:40 CDT) — taxonomy definitions, before any FCE or intervention result

Made after reviewing the Stage 4a classifier's labelled samples on the real base / K=20 s0 3-pass / teacher dev140 runs
(reports/stage4a_validation.md). Only the descriptive taxonomy (Section 2) changes; populations, FCE, interventions,
confounds and the Section 6 readout are unchanged.

1. **F-format = parser errors only**: ≥3 consecutive parser errors at the end, or ≥25% of steps are parser errors.
   Well-formed but inadmissible commands are not a format failure (reviewed example: "go to drawer 5" repeated while
   already at drawer 5, 80% of steps).
2. **New F-inadmissible**, placed after F-format: ≥25% of steps are inadmissible commands.
3. **F-goal counts executed (legal) actions only**, and excludes the task type's own transformation appliance:
   a legal move of a goal-type object into a receptacle whose type is neither the target receptacle nor, for heat /
   cool / clean tasks, the microwave / fridge / sinkbasin; or a legal heat / cool / clean / use applied to an object of
   the wrong type. Reviewed examples that the old rule mislabelled: an inadmissible "move plate 1 to stoveburner 1"
   (never executed), and "move tomato 3 to microwave 1" on a heat task (a procedure error → F-precondition).

New order: F-format, F-inadmissible, F-goal, F-search, F-precondition, F-loop, F-cap.
