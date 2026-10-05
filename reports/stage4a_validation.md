# Stage 4a engineering validation

This update implements Amendment 1 at commit `862eaed`, validated plan SHA-256
`ca177ad227c36db4ef4f3ec0dc5f05f64ebb7b4dc797cac1b4314b729f4ba9cd`.
The [Amendment 1 validation](#amendment-1-validation) is appended below; the
original validation and its artifact hashes remain historical records.

Validated on 2026-10-04 (America/New_York). This implements the tooling requested
for [MECHANISM_PLAN.md](../docs/MECHANISM_PLAN.md), without changing that plan.
No live FCE or intervention experiment was executed. No GPU, model server,
OpenAI API call, job submission, commit, or push was performed. Existing run
outputs, `data/`, and `third_party/` were read only. The launcher lifecycle tests
used fake executables and never started vLLM.

The original validation plan SHA-256 was
`9c49edb993e0c4d42f640d62643fddebbe2865d240c0822c2a11d7bda10283bf`.
The evaluation prompt/parser, visible-field whitelist, configuration, Gemma
template, default greedy decoding, 1,024-token student cap and 40-turn limit
remain unchanged. Small shared-code changes expose student message/sampling
hooks, extract the existing exact replay comparison, account for teacher usage
inside diagnostic wrappers, and include mechanism code/the plan in provenance.

The implemented entry points are `goals`, `taxonomy`, `branch`, `fce`,
`interventions`, and `readout` under `analysis.mechanism`, plus the CPU preflight
module `launcher` and `scripts/rai_mechanism.sh`. Commands and ownership/resume
rules are documented in [README.md](../README.md#stage-4a-mechanism-analysis-tooling).

The required checks passed:

| Check | Result |
|---|---|
| `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q` | **294 passed, 4 skipped**, 109.52 seconds |
| `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q tests/test_stage3a_torch.py tests/test_lora_serving_torch.py tests/test_stage2b_torch.py` | **37 passed**, 27.59 seconds; one existing PEFT configuration warning |
| Seven module CLIs with `--help` | Passed |
| `bash -n scripts/rai_mechanism.sh` | Passed |
| `git diff --check` | Passed |
| Unchanged plan, AGENTS.md, ReAct prompt/parser, visibility module and base configuration | Verified by Git diff |

The main environment skips the three torch modules because torch is absent;
the second command runs them in the requested torch environment. The fourth
skip is the optional real ALFWorld integration test because `ALFWORLD_DATA_ROOT`
was not set for pytest. During development, a resume test rejected a source
hash change made while the suite was running; subsequent runs with fixed source
files passed. No weakened resume check was introduced.

The new tests cover real task IDs and three complete real PDDL fact lists;
taxonomy order, thresholds and triggering indices; visible-only search and
restatement; transformation/lamp audit predicates; actual reset/prefix execution
against fake environments; mismatched observations, rewards, game content and
audit hashes; parser-error replay; remaining-turn accounting; I0 noise flags;
single-turn intervention text retained in later requests; I2 precedence and
step cost; verified TRAIN-only example selection; exact checkpoint/seed-set
binding; sampled HTTP request seeds/temperature; teacher cache namespaces;
episode/run caps including retries; actionless advice and its template hash;
I3/I4 shared output reuse; unknown probes; immutable attempt receipts; FCE
refinement/resume; rescue thresholds/Wilson intervals; coverage gating; and
launcher gate order, free-port arguments, cleanup and restart directories.

The fake pipeline exercises both teacher samples at each FCE probe, all 30 arm
continuations, persistent shared caps and repeat invocations without new calls
for completed work. A separate invocation of the readout CLI also validated its
saved probe/branch receipts and wrote
[fake JSON](runs/stage4a-validation/fake-readout/readout.json) and
[fake Markdown](runs/stage4a-validation/fake-readout/readout.md). These outputs
are explicitly from fake policies and a fake backend.

The real-artifact classifier command was:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m analysis.mechanism.taxonomy \
  --run reports/runs/eval/k20s0-rai-dev140/k20s0-p03/eval \
  --teacher-run reports/runs/teacher-dev140 \
  --base-run reports/runs/eval/k20s0-rai-dev140/google/gemma-4-12B-it/eval \
  --output reports/runs/stage4a-validation/taxonomy-final
```

The snapshot provenance timestamp is **2026-10-05 03:15:02 UTC**
(2026-10-04 23:15:02 EDT). Episodes were read individually; this is not an atomic
snapshot of the running teacher suite. Unknown source episodes may advance
after capture. Per-episode snapshot hashes are preserved. Each run contains
140 unique instances and 137 task groups in this snapshot.

| Run | Observed episodes | Success | Known failure | Unknown | Missing/reset-incomplete |
|---|---:|---:|---:|---:|---:|
| Base | 140 | 59 | 81 | 0 | 0 |
| Seed-SFT `k20s0-p03` | 140 | 44 | 96 | 0 | 0 |
| Teacher snapshot | 140 | 113 | 22 | 5 | 0 |

**Taxonomy sanity check only; no interpretation.** Primary-label shares use
known failures as the denominator. All-matching labels are also retained in
JSON. There are zero unclassified known failures in these three snapshots.

| Run | F-format | F-goal | F-search | F-precondition | F-loop | F-cap |
|---|---:|---:|---:|---:|---:|---:|
| Base, n=81 | 6 (7.4%) | 16 (19.8%) | 31 (38.3%) | 13 (16.0%) | 13 (16.0%) | 2 (2.5%) |
| Seed-SFT, n=96 | 15 (15.6%) | 16 (16.7%) | 29 (30.2%) | 17 (17.7%) | 15 (15.6%) | 4 (4.2%) |
| Teacher snapshot, n=22 | 0 (0.0%) | 4 (18.2%) | 13 (59.1%) | 2 (9.1%) | 0 (0.0%) | 3 (13.6%) |

The snapshot contains **|G|=69** and **|R|=18**. Unknown teacher outcomes are
excluded from G. Parser-error rates across observed turns are 0.5327% for base,
0% for Seed-SFT, and 0% for teacher. F-format also includes inadmissible actions.

The review file contains ten deterministically sampled episodes per matching
label where available, otherwise every available episode. Samples include
triggered action indices, visible observations, rule evidence and source paths.
Counts were checked against the all-matching populations:

| Run | F-format | F-goal | F-search | F-precondition | F-loop | F-cap |
|---|---:|---:|---:|---:|---:|---:|
| Base | 6 | 10 | 10 | 10 | 10 | 2 |
| Seed-SFT | 10 | 10 | 10 | 10 | 10 | 4 |
| Teacher snapshot | 0 | 4 | 10 | 2 | 2 | 3 |

Artifacts:

- [Taxonomy, per-episode evidence, shares, G and R](runs/stage4a-validation/taxonomy-final/taxonomy.json)
- [Human-review samples](runs/stage4a-validation/taxonomy-final/review_samples.json)
- [Share table](runs/stage4a-validation/taxonomy-final/shares.md)
- [Provenance and snapshot hashes](runs/stage4a-validation/taxonomy-final/provenance.json)
- [Complete real-fact unit-test fixtures](../tests/fixtures/mechanism_real_facts.json)

The taxonomy JSON SHA-256 is
`f469b7c747dd5ee0f5d5d8259c3217190e541bdb7db592109609deba0f6cb738`;
the review-sample JSON SHA-256 is
`0465cb2bce5f3a74f4337f6ee0170d3bb62f6beaa269210c2f0d2311dca981f3`.
The snapshot records ALFWorld 0.4.2, TextWorld 1.7.0, OpenAI SDK 2.8.1,
httpx 0.28.1, and pytest 7.4.0. Generated validation artifacts under
`reports/runs/` follow the repository's existing ignored-output convention.

A CPU-only FCE dry run on this frozen G used explicit caps of 400 HTTP attempts
per parent episode and 12,000 per run. It made **zero teacher calls**, constructed
no environments and created no run directory. Its coarse scan bound is 30,360
policy calls; including the maximum refinement interval and ten HTTP attempts
per policy call gives 460,920 attempts before caps. These are conservative
planning bounds, not observed costs or recommendations for a live budget.

Operational details fixed by the implementation: F-search excludes the task
instruction from sightings; search/navigation is not itself a wrong-goal
commitment. Restatement consumes only visible text and successful feedback.
I3 has a separate advice-only teacher system prompt, avoiding the normal
teacher action-format requirement; both template and instruction hashes are
recorded. Student intervention text persists in the original user turn. I2's
actual information action/feedback adds one normal turn; I4 replaces one action.
One I3/I4 output is reused across that arm's five continuations, with new billed
usage zero on reuse. Teacher actionless advice remains in the existing raw
usage ledger and the shared-output receipt. Completed attempts are immutable;
only owned analysis workspace summaries/counters advance on resume.

Live server/adapter equivalence, real-environment takeover and intervention
outcomes remain untested in this stage. Human review of the sampled labels is
still needed; these automated sanity checks do not establish classifier accuracy.
The readout keeps F-format, missing seed task types, I0 noise, teacher-unsolvable
starts and unknown results separate. M1 attribution requires additional
needed-step coverage evidence, and M1a/M1b requires the section-4 probes.
Training fit and earlier/later checkpoint controls are not supplied by this
engineering validation. No mechanism finding or faithful baseline reproduction
is claimed.

## Amendment 1 validation

Implemented [Amendment 1](../docs/MECHANISM_PLAN.md#amendment-1-2026-10-04-2240-cdt--taxonomy-definitions-before-any-fce-or-intervention-result)
(commit `862eaed`; plan SHA-256 `ca177ad227c36db4ef4f3ec0dc5f05f64ebb7b4dc797cac1b4314b729f4ba9cd`).
After the classifier snapshot, concurrent work appended Amendment 2 (commit
`07af904`), making the current plan file SHA-256
`9505f0410d3bab7736b9be2047b131a1c458279b43c0c3ad47c271aa3e57a82e`.
That later amendment is outside this update; the snapshot retains the exact
Amendment 1 plan hash it used.

F-format uses parser errors only; F-inadmissible follows it at the 25% threshold.
F-goal uses legal executed placements of goal-type objects and wrong-type
heat/cool/clean/use actions, excluding the task’s own transformation appliance
as a placement destination. The readout uses the amended label order and keeps
only F-format in the separate format cohort. The populations, FCE, interventions
and confound rules are unchanged.

This command read the same student/config/manifest/freeze inputs as the original
validation, the specified ladder base run, and the now-complete teacher run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m analysis.mechanism.taxonomy \
  --run reports/runs/eval/k20s0-rai-dev140/k20s0-p03/eval \
  --teacher-run reports/runs/teacher-dev140 \
  --base-run reports/runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval \
  --output reports/runs/stage4a-validation/taxonomy-amend1
```

Snapshot provenance timestamp: **2026-10-05T03:34:01.131394+00:00**.
All three snapshots are complete. Each contains 140 unique instances and 137
task groups. All 420 source episode hashes were rechecked after classification.

| Run | Episodes | Success | Known failure | Unknown | Missing/reset-incomplete |
|---|---:|---:|---:|---:|---:|
| Base | 140 | 59 | 81 | 0 | 0 |
| Seed-SFT `k20s0-p03` | 140 | 44 | 96 | 0 | 0 |
| Teacher | 140 | 114 | 26 | 0 | 0 |

**Descriptive shares only; no interpretation.** Primary-label shares use known
failed episodes as the denominator. There are zero unclassified known failures.

| Run | F-format | F-inadmissible | F-goal | F-search | F-precondition | F-loop | F-cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, n=81 | 0 (0.0%) | 6 (7.4%) | 4 (4.9%) | 34 (42.0%) | 21 (25.9%) | 13 (16.0%) | 3 (3.7%) |
| Seed-SFT `k20s0-p03`, n=96 | 0 (0.0%) | 15 (15.6%) | 3 (3.1%) | 31 (32.3%) | 23 (24.0%) | 19 (19.8%) | 5 (5.2%) |
| Teacher, n=26 | 0 (0.0%) | 0 (0.0%) | 2 (7.7%) | 17 (65.4%) | 2 (7.7%) | 1 (3.8%) | 4 (15.4%) |

All-matching shares (labels may overlap; the same denominators apply):

| Run | F-format | F-inadmissible | F-goal | F-search | F-precondition | F-loop | F-cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base, n=81 | 0 (0.0%) | 6 (7.4%) | 4 (4.9%) | 35 (43.2%) | 25 (30.9%) | 34 (42.0%) | 3 (3.7%) |
| Seed-SFT `k20s0-p03`, n=96 | 0 (0.0%) | 15 (15.6%) | 4 (4.2%) | 34 (35.4%) | 30 (31.2%) | 49 (51.0%) | 5 (5.2%) |
| Teacher, n=26 | 0 (0.0%) | 0 (0.0%) | 2 (7.7%) | 19 (73.1%) | 2 (7.7%) | 2 (7.7%) | 4 (15.4%) |

**|G|=70; |R|=18.** Parser-error rates over all observed turns: Base 0.5408%, Seed-SFT `k20s0-p03` 0.0000%, Teacher 0.0000%.

The two new [offline fixtures](../tests/fixtures/mechanism_amend1_episodes.json)
retain all 40 logged actions, outcomes and visible fields, source episode/audit
hashes, and an explicit projection of classifier audit predicates. Identical
projected audit states are deduplicated. Fixture content and classifications
were checked against the read-only real episodes:

| Source episode | Reviewed action | Outcome | Amended labels in order |
|---|---|---|---|
| [089-03b7ad0af331](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/089-03b7ad0af331/episode.json) | Step 40: `move plate 1 to stoveburner 1` | inadmissible | F-precondition |
| [047-b9f24a290a01](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/047-b9f24a290a01/episode.json) | Step 6: `move tomato 3 to microwave 1` | legal | F-inadmissible, F-precondition, F-loop |

Review samples below are selected from the deterministic seed-0 per-run
`review_samples.json` lists, taking one entry from base, student and teacher
in rotation until five are selected per label, skipping exhausted lists.
Selection uses all-matching labels, so a sample’s primary label can differ.
Each row gives all triggering indices and one evidence action; full observations,
audit evidence and task IDs are in the linked source and review JSON.
**F-format has zero matching episodes across all three runs, so no real review
samples exist for that label.** Each of the other six labels has five samples.

### F-inadmissible: five review samples

| Run / source episode | Goal (task type; object → target) | Primary | Trigger indices | Evidence example |
|---|---|---|---|---|
| Base / [050-fc2e9c60d66f](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/050-fc2e9c60d66f/episode.json) | pick_clean_then_place_in_recep; Cloth → Cabinet | F-inadmissible | 6, 9, 12, 16–17, 19, 23, 26–27, 29–40 | `inadmissible_fraction_at_least_0.25`; step 6: `put towel 1 in cabinet 1` (inadmissible) |
| Seed-SFT `k20s0-p03` / [131-fa06d1ff5588](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/131-fa06d1ff5588/episode.json) | pick_cool_then_place_in_recep; Mug → CoffeeMachine | F-inadmissible | 15–40 | `inadmissible_fraction_at_least_0.25`; step 15: `move cup 2 to coffeemachine 1` (inadmissible) |
| Base / [123-ed536bed1250](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/123-ed536bed1250/episode.json) | pick_two_obj_and_place; CreditCard → Safe | F-inadmissible | 14, 27–30, 32, 34, 36, 38, 40 | `inadmissible_fraction_at_least_0.25`; step 14: `go to drawer 7` (inadmissible) |
| Seed-SFT `k20s0-p03` / [043-b017652e42bb](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/043-b017652e42bb/episode.json) | pick_two_obj_and_place; ToiletPaper → Toilet | F-inadmissible | 5–40 | `inadmissible_fraction_at_least_0.25`; step 5: `take toiletpaper 3 from cabinet 1` (inadmissible) |
| Base / [002-c47e4c28c9ac](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/002-c47e4c28c9ac/episode.json) | pick_two_obj_and_place; CreditCard → Dresser | F-inadmissible | 10–11, 13–14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40 | `inadmissible_fraction_at_least_0.25`; step 10: `take creditcard 2 from sidetable 1` (inadmissible) |

### F-goal: five review samples

| Run / source episode | Goal (task type; object → target) | Primary | Trigger indices | Evidence example |
|---|---|---|---|---|
| Base / [036-922e4568944b](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/036-922e4568944b/episode.json) | pick_cool_then_place_in_recep; Pan → StoveBurner | F-goal | 28 | `manipulated_object_type_mismatch`; step 28: `clean pot 1 with sinkbasin 1` (legal) |
| Seed-SFT `k20s0-p03` / [109-795966ddc51a](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/109-795966ddc51a/episode.json) | pick_clean_then_place_in_recep; ButterKnife → DiningTable | F-goal | 28 | `manipulated_object_type_mismatch`; step 28: `clean knife 1 with sinkbasin 1` (legal) |
| Teacher / [067-50b406ea6951](runs/teacher-dev140/episodes/067-50b406ea6951/episode.json) | pick_cool_then_place_in_recep; Cup → Microwave | F-goal | 39 | `manipulated_object_type_mismatch`; step 39: `cool mug 1 with fridge 1` (legal) |
| Base / [029-b3e40defaec0](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/029-b3e40defaec0/episode.json) | pick_cool_then_place_in_recep; Pan → DiningTable | F-goal | 40 | `manipulated_object_type_mismatch`; step 40: `clean pot 1 with sinkbasin 1` (legal) |
| Seed-SFT `k20s0-p03` / [036-922e4568944b](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/036-922e4568944b/episode.json) | pick_cool_then_place_in_recep; Pan → StoveBurner | F-goal | 35 | `placement_receptacle_type_mismatch`; step 35: `move pan 1 to sinkbasin 1` (legal) |

### F-search: five review samples

| Run / source episode | Goal (task type; object → target) | Primary | Trigger indices | Evidence example |
|---|---|---|---|---|
| Base / [130-0890c4113418](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/130-0890c4113418/episode.json) | pick_cool_then_place_in_recep; Apple → GarbageCan | F-search | 0–40 | `no_goal_type_in_observation_text`; step 40: `go to drawer 10` (legal) |
| Seed-SFT `k20s0-p03` / [081-0d30cac66153](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/081-0d30cac66153/episode.json) | pick_cool_then_place_in_recep; Bread → CounterTop | F-search | 0–40 | `no_goal_type_in_observation_text`; step 40: `go to countertop 1` (legal) |
| Teacher / [009-4c450b0062a7](runs/teacher-dev140/episodes/009-4c450b0062a7/episode.json) | pick_clean_then_place_in_recep; Ladle → DiningTable | F-search | 0–40 | `no_goal_type_in_observation_text`; step 40: `open cabinet 6` (legal) |
| Base / [035-3e2dd9052b1f](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/035-3e2dd9052b1f/episode.json) | pick_cool_then_place_in_recep; Egg → Microwave | F-search | 0–40 | `no_goal_type_in_observation_text`; step 40: `go to cabinet 16` (legal) |
| Seed-SFT `k20s0-p03` / [035-3e2dd9052b1f](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/035-3e2dd9052b1f/episode.json) | pick_cool_then_place_in_recep; Egg → Microwave | F-search | 0–40 | `no_goal_type_in_observation_text`; step 40: `open cabinet 20` (legal) |

### F-precondition: five review samples

| Run / source episode | Goal (task type; object → target) | Primary | Trigger indices | Evidence example |
|---|---|---|---|---|
| Base / [113-8a1f7a290f4a](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/113-8a1f7a290f4a/episode.json) | pick_cool_then_place_in_recep; Pot → DiningTable | F-precondition | 28, 40 | `put_before_required_transformation`; step 28: `move pot 1 to diningtable 1` (legal) |
| Seed-SFT `k20s0-p03` / [135-e2ed1031dbb7](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/135-e2ed1031dbb7/episode.json) | pick_cool_then_place_in_recep; Apple → CounterTop | F-precondition | 21, 27, 40 | `put_before_required_transformation`; step 21: `move apple 1 to fridge 1` (legal) |
| Teacher / [071-0c5585508068](runs/teacher-dev140/episodes/071-0c5585508068/episode.json) | pick_cool_then_place_in_recep; Apple → Microwave | F-precondition | 40 | `held_goal_but_required_transformation_missing_at_end`; step 40: `go to fridge 1` (legal) |
| Base / [100-1829ff0dd02c](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/100-1829ff0dd02c/episode.json) | pick_heat_then_place_in_recep; Potato → Fridge | F-precondition | 40 | `put_before_required_transformation`; step 40: `move potato 1 to microwave 1` (legal) |
| Seed-SFT `k20s0-p03` / [026-97a31b1fcc5d](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/026-97a31b1fcc5d/episode.json) | pick_heat_then_place_in_recep; Mug → CoffeeMachine | F-precondition | 22, 40 | `put_before_required_transformation`; step 22: `move mug 3 to microwave 1` (legal) |

### F-loop: five review samples

| Run / source episode | Goal (task type; object → target) | Primary | Trigger indices | Evidence example |
|---|---|---|---|---|
| Base / [121-40d4bd381ccd](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/121-40d4bd381ccd/episode.json) | pick_clean_then_place_in_recep; SoapBar → Cabinet | F-precondition | 15–16, 18–40 | `same_action_observation_pair_at_least_3`; step 15: `help` (legal) |
| Seed-SFT `k20s0-p03` / [016-0e208600b509](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/016-0e208600b509/episode.json) | pick_two_obj_and_place; SprayBottle → Toilet | F-loop | 20–40 | `same_action_observation_pair_at_least_3`; step 20: `look` (legal) |
| Teacher / [045-93056ae58467](runs/teacher-dev140/episodes/045-93056ae58467/episode.json) | pick_heat_then_place_in_recep; Cup → Cabinet | F-goal | 8, 10–12, 14, 17–40 | `same_action_observation_pair_at_least_3`; step 8: `look` (legal) |
| Base / [094-5cf124a5227d](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/094-5cf124a5227d/episode.json) | pick_two_obj_and_place; SprayBottle → Toilet | F-loop | 5, 10, 15–17, 20–23, 27–32, 35–40 | `same_action_observation_pair_at_least_3`; step 5: `go to shelf 2` (legal) |
| Seed-SFT `k20s0-p03` / [069-f8c705583660](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/069-f8c705583660/episode.json) | pick_two_obj_and_place; RemoteControl → Ottoman | F-loop | 25–40 | `same_action_observation_pair_at_least_3`; step 25: `look` (legal) |

### F-cap: five review samples

| Run / source episode | Goal (task type; object → target) | Primary | Trigger indices | Evidence example |
|---|---|---|---|---|
| Base / [052-d97cb5baca36](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/052-d97cb5baca36/episode.json) | pick_two_obj_and_place; Glassbottle → Shelf | F-cap | 40 | `no_other_rule_and_40_turn_limit`; step 40: `take glassbottle 1 from countertop 1` (legal) |
| Seed-SFT `k20s0-p03` / [070-51b60ec31595](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/070-51b60ec31595/episode.json) | pick_heat_then_place_in_recep; Plate → DiningTable | F-cap | 40 | `no_other_rule_and_40_turn_limit`; step 40: `go to garbagecan 1` (legal) |
| Teacher / [029-b3e40defaec0](runs/teacher-dev140/episodes/029-b3e40defaec0/episode.json) | pick_cool_then_place_in_recep; Pan → DiningTable | F-cap | 40 | `no_other_rule_and_40_turn_limit`; step 40: `go to stoveburner 4` (legal) |
| Base / [040-5a00e00d4a00](runs/eval/ladder-s0-rai-dev140/google/gemma-4-12B-it/eval/episodes/040-5a00e00d4a00/episode.json) | pick_and_place_simple; WineBottle → Shelf | F-cap | 40 | `no_other_rule_and_40_turn_limit`; step 40: `go to diningtable 1` (legal) |
| Seed-SFT `k20s0-p03` / [045-93056ae58467](runs/eval/k20s0-rai-dev140/k20s0-p03/eval/episodes/045-93056ae58467/episode.json) | pick_heat_then_place_in_recep; Cup → Cabinet | F-cap | 40 | `no_other_rule_and_40_turn_limit`; step 40: `move mug 1 to microwave 1` (legal) |

Artifacts:

- [Amended taxonomy, evidence, shares, G and R](runs/stage4a-validation/taxonomy-amend1/taxonomy.json)
- [Complete review samples](runs/stage4a-validation/taxonomy-amend1/review_samples.json)
- [Primary share table](runs/stage4a-validation/taxonomy-amend1/shares.md)
- [Provenance and input snapshot hashes](runs/stage4a-validation/taxonomy-amend1/provenance.json)

Taxonomy JSON SHA-256: `04c7f09b71763d93588d542be15bb53d207683fb3a9f6f9e8a05c4be35b41aa4`.
Review JSON SHA-256: `57d01c01d89e8f82e3d961ab12af0cbc5dbf3faa78b14074ad0c0f5a5ecfb356`.
Earlier validation artifacts are retained unchanged. No FCE/intervention
experiment, GPU, server, network/API request, job, commit or push was used.

Validation checks:

| Check | Result |
|---|---|
| `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q` | **365 passed, 4 skipped**, 156.79 seconds |
| Full source episodes versus both fixture projections | Identical classifications and projection content; source hashes match |
| Real classifier inputs after output | All 420 episode hashes unchanged; all three snapshots complete |
| Review sample counts | Match `min(10, all_matching_count)` per run/label; report has five per nonempty label |
| Snapshot plan SHA-256 versus commit `862eaed` | Identical |
| `git diff --check` | Passed |

The four skips are the three modules requiring unavailable torch and the optional
real ALFWorld integration test without `ALFWORLD_DATA_ROOT`. Tests used the
existing network/API prohibition fixture. The earlier focused run passed 95
checks and failed one resume-identity check when the unrelated
`scripts/rai_eval_ladder_s0_b.sh` appeared during the test; the source-hash
comparison identified that addition. The full rerun passed without weakening
resume validation or editing that script. `tools/check_lora_serving.py` and
`scripts/eval_adapters.sh` were not edited for this amendment.
