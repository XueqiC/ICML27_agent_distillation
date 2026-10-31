# E0.6: no-hint evaluation of trained adapters

`tools.joint_guidance run --eval-adapter --modes U` evaluates any adapter that
passes the same `validate_report` gate used by coverage, including server
provenance, original/converted adapter hashes, and the configured base model and
revision. The served name need not be `e4b-tsft-k20-e32-lr5e-5`. Without the flag,
the existing theta0 restrictions, prompts, step records and teacher accounting
remain in effect. `G`, `G1`, and the default mixed modes are refused with the flag.

For preregistration §6(a), use the frozen 38-case V1b prefix spec:

```bash
python -m tools.joint_guidance run \
  --eval-adapter --modes U --spec /path/to/v1b-prefixes.json \
  --config configs/base_e4b.yaml --data-root /path/to/alfworld \
  --endpoint http://127.0.0.1:8000/v1 --model-name offline-update-100 \
  --server-provenance /path/to/server/provenance.json \
  --lora-check /path/to/lora-check/report.json \
  --output /path/to/eval-prefixes
```

Greedy plus sampled seeds 0–7 is the default. Add `--from-start` and use a new
output directory for §6(b). For E-short, pass `--seeds` with no values to run
only greedy. These options do not alter the source case selection.

For E-sel, evaluate the frozen **D_unseen_quick 24** tasks from the start:

```bash
PYTHONPATH=src:. python -m tools.v2_task_eval run \
  --set D_unseen_quick --repair-splits data/manifest/repair_v01_splits.json \
  --greedy --sampled-seeds 0 1 2 \
  --config configs/base_e4b.yaml --data-root /path/to/alfworld \
  --endpoint http://127.0.0.1:8000/v1 --model-name offline-update-100 \
  --server-provenance /path/to/server/provenance.json \
  --lora-check /path/to/lora-check/report.json \
  --workers 4 --output /path/to/eval-quick
```

For §6(c), use the same command with `--set D_probe` and a new output directory
(for example `--output /path/to/eval-probe`), keeping `--greedy --sampled-seeds 0 1 2`.
This runs all 60 frozen tasks. **The main comparison is `subsets.non_quick_36`**
(D_probe minus D_unseen_quick). `subsets.quick_24` is labelled as checkpoint
selection; its totals and the full 60-task totals are descriptive.
`tools.repair_coverage` remains a training-pool coverage tool and refuses D_probe
groups, including the quick subset.

The new tool preserves the full frozen listing in `tasks.json`, records the
adapter and leakage check in sealed `plan.json`, and runs the shared student
suite once in `greedy` and each `sampled-sN` directory. The sealed `summary.json`
contains per-task successes (`true`/`false`/`null`), valid sampled seed counts,
per-task sampled means, and aggregate counts. Its sampled success rate averages
valid seeds within each task, then averages tasks with at least one valid seed.
Unknown/error episodes are reported per decoding and excluded from success-rate
denominators; they are never failures. The underlying suite's pilot summary
retains its existing denominator convention; use the top-level V2 summary for
E-sel and §6(c). Exit status is 1 if any episode is unknown/error, with its
evidence and the V2 summary retained.

Outputs are exclusive. `--resume` requires the same plan, adapter, manifests,
config, code, dependencies and decodings, and only fills absent decoding
directories. Completed decodings have sealed receipts binding their artifact
hashes. An existing decoding directory without a completion receipt is refused;
retain it as interrupted evidence and use a fresh output directory. A completed
resume validates evidence and leaves the summary unchanged. Theta0 is accepted
under its original served name `e4b-tsft-k20-e32-lr5e-5` without a V2 training
manifest and recorded as `arm: theta0`, `arm_index: 0`. Missing exposure metadata
retains the `no_exposure_ids_declared` disclosure described below.

Use the same hardware class and serving check for every compared arm, including
a fresh theta0 evaluation.

## Adapter provenance and leakage

`plan.json.identity.evaluated_adapter` and every `records/*.json` contain the
evaluated adapter's name, original path, weights/config SHA256, base and revision.
The path refers to the original adapter even when serving uses a validated
conversion. The server provenance retains the served path and conversion chain.

For an original path ending in `update-NN/checkpoint`, both
`update-NN/manifest.json` and the training run's `manifest.json` are required.
The nested `evaluated_adapter.training` records both file hashes, the arm, and
the **checkpoint's** optimizer update count (not the run's final update count).
Checkpoint manifest weights/config hashes must match the evaluated adapter.

The guard unions task/group IDs from the run and checkpoint manifests'
`training_exposure` declarations and their embedded interchange
`input_manifests[*].training_exposure_tasks` declarations, including replay.
Structured exposure supports `task_ids`/`group_ids` or `tasks`/`groups` lists;
scalar exposure counts are never interpreted as identifiers. Interchange
`training_exposure_tasks` currently contains group IDs, despite its name.
Full task IDs also exclude all other trials in their group. Bare group names
are normalized to `train/<group>`.

This is a conservative exclusion of declared training inventories: the canary
run declares 220 groups across its two interchange manifests, even though only
48 examples were consumed by update 3. The recorded union is not a claim that
all those groups were sampled. The sources and this scope are recorded alongside
the IDs. An overlap refuses the run before creating output or making model/env
calls. Successful checks, evaluated IDs and exclusion IDs are in
`plan.json.identity.leakage_check`. Non-V2 adapters without these declarations
are marked `no_exposure_ids_declared`, not certified training-disjoint.

## Meaning of `prefix_verified`

Replay already uses the source theta0's recorded public replies, including its
public Thought and Action text, through the shared action parser. The evaluated
student is called only at the branch. It need not reproduce any theta0 reply.
No replay implementation change is needed for adapter evaluation.

Before continuation, the executor checks the game hash, environment seed/version,
reset observation, and each replayed transition against the source: visible
observation and admissible commands, parsed action, reward, terminal/truncated
flags, and audit digest. Audit data is used only for verification and never
enters the student prompt. As in V1, the source reference's environment-version
field is rebound to the current harness; this is not proof that the original
and current software versions are identical.

`prefix_verified=true` means the continuation callback was reached after those
checks. It does not assert suffix success or student reply equality. With
`--from-start`, only the reset is compared and the first student decision has
empty history. Divergent replay is void, with no student decision at the branch.
The evaluated adapter receives the original public history without guidance,
training metadata or verifier fields. The definition is recorded in the plan
and each adapter-evaluation record.

## Teacher accounting and summaries

`--budget-ledger` and teacher caps can be omitted. If supplied, the path/caps are
recorded but the ledger is never opened, created, locked, reconciled or changed,
including on resume and summarize. No reviewer is constructed. Run and record
checks reject teacher usage or guidance; a teacher reservation fails immediately.
The `global_budget` summary explicitly reports **run-local** zero calls/tokens,
not the supplied external ledger's cumulative usage. Per-execution environment
accounting remains active. `training_exposure=0` on evaluation records continues
to mean this evaluation performs no training; adapter exposure is nested under
`evaluated_adapter.training`.

The existing summarize command and metrics work without extra flags:

```bash
python -m tools.joint_guidance summarize \
  --rollouts /path/to/eval-prefixes --output /path/to/eval-prefixes-summary
```

Resume requires identical evaluated weights, manifests, cases, config and code.
`--dry-run` with no serving arguments uses an explicitly synthetic adapter
identity with null artifact hashes. Supplying serving arguments in a dry run
requires the complete serving gate, but still uses fake environment/student
execution. Synthetic runs establish no scientific result.
