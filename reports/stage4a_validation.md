# Stage 4a engineering validation

Validated on 2026-10-04 (America/New_York). This implements the tooling requested
for [MECHANISM_PLAN.md](../docs/MECHANISM_PLAN.md), without changing that plan.
No live FCE or intervention experiment was executed. No GPU, model server,
OpenAI API call, job submission, commit, or push was performed. Existing run
outputs, `data/`, and `third_party/` were read only. The launcher lifecycle tests
used fake executables and never started vLLM.

The plan SHA-256 is
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
