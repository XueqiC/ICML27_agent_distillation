# E2 interchange reference

Copied from the evaluation-side M0b-3 handoff for tree `46b8527`, supplied in
`_m0b3_draft/EVAL_SIDE_M0b-3.md`. This is a format reference, not imported
evaluation code. The training implementation is based on `2506fa9`.

## E2 interchange contract (training branch needs no repair imports)

The file is `e2_training_collection.json`, with `format` equal to
`repair_e2_training_collection` and integer `schema_version: 1`.
Canonical JSON for every **object hash** uses sorted keys, compact separators
`,:` with no spaces, UTF-8, `ensure_ascii=False`, and no NaN/Infinity. A
`content_sha256` is SHA-256 of that entire document excluding the
`content_sha256` key. **File hashes** hash the exact bytes, including the final
newline. Do not interchange these two hash types.

Top-level fields:

| Field | Meaning |
|---|---|
| `source_collection_sha256` | Exact byte hash of the adjacent finalized `collection.json` |
| `source_collection_content_sha256` | Its canonical content seal |
| `theta0` | The collection's checkpoint object: `identity` from existing `inspect_adapter`, plus `files`, mapping every checkpoint filename to its byte hash |
| `tokenizer` | `id` (hash of metadata file-hash map), `files`, `chat_template_sha256` (canonical string hash) |
| `input_hashes` | Original collection input file path → byte hash, including frozen manifests and repair sources |
| `split` | Seed, rule, coverage `failure_order`, two probes and ordered `training_candidates` |
| `D_repair_train`, `D_repair_probe` | Ordered task ID lists; four approved pairs and two held-out repair-update tasks respectively |
| `K_trusted_seed_trajectories` | Original verified support size, never inferred from repair task count |
| `unique_instances`, `task_groups` | Counts for the selected repair-training tasks |
| `dry_run`, `canary`, `training_exposure` | Execution mode and exposure (zero on this branch) |
| `ready_for_training`, `training_blockers` | Ready only with four approved pairs, seed references for each task, and neither synthetic nor canary data |
| `arms` | Exactly `N` and `F`, each an ordered list in D_repair_train order |

Each arm entry contains `task_id`, `group_id`, `learning_episode_id`,
`repair_id`, `trajectory_id`, `source_bucket: "successful_repair"`,
`episode`, `episode_sha256` (canonical object hash), `human_review`, and
`seed_demonstrations`. `episode` is the public projection of the LearningEpisode
in the **existing train.dataset.episode_examples schema**, not a new trainer
serialization:

```json
{
  "schema_version": 2,
  "task_id": "train/<game>/<trial>",
  "split": "train",
  "sealed": false,
  "initial": {"visible": {
    "observation": "Exact initial harness observation",
    "admissible_commands": ["Exact public command"]
  }},
  "steps": [{
    "event_id": "<learning_episode_id>:0",
    "action_raw": "Thought: Public-prefix-supported reasoning.\nAction: exact executed command",
    "visible": {
      "observation": "Exact post-action harness observation",
      "admissible_commands": ["Exact public command"]
    }
  }]
}
```

`steps` has one entry per actual executed decision, in order. The input visible
context for decision zero is `initial.visible`; subsequent contexts come from
the preceding step. A step's `visible` is its **post-action** observation.
Command lists use the unchanged harness whitelist's sorted unique strings.
`action_raw` is the reconstructed public reply (or the unchanged original
prefix reply); its parsed action must equal the witnessed executed action.
The exporter checks this and equality with LearningEpisode.student_history.
There are no rewards, success labels, PDDL facts, audit dictionaries, diagnosis,
native input IDs, or cached token masks in `episode`. Task IDs and event IDs are
accounting fields, never extra prompt text. The normal encoder builds the
shared public messages and supervises assistant replies/native turn ends.
Tests feed this exact projection through both this tree's encoder and the
read-only encoder at **2506fa9**, for turn and episode packing.

`human_review` contains `status: "approved"`, `reviewed: true`,
`approved: true`, `evidence_sha256`, `episode_sha256`, and `decision`.
`decision` has `trajectory_id`, `episode_id` (the LearningEpisode ID), `reviewer`,
`notes`, and the three explicit boolean decisions from the review JSON. The
evidence hash binds `review_requests.json`'s public execution, labels and prefix
checks; the episode hash additionally binds the exact public training payload.
Pending/rejected targets remain in `collection.json` and do not enter `arms`.
An incomplete export has empty/short arms and `ready_for_training: false`.

Each `seed_demonstrations` item contains:

- `task_id`, `episode_id`: the trusted original executed demonstration.
- `verified_collection_path`, `verified_manifest_sha256`: selected K view and
  exact byte hash of its `manifest.json`.
- `evidence_root`, `episode_dir`: resolved largest-collection root and the
  receipt-relative directory (important for nested K views).
- `episode_sha256`: exact byte hash of `<evidence_root>/<episode_dir>/episode.json`.
- `receipt_sha256`: canonical object hash of `receipt`; `receipt` is the full
  original verified receipt, including `episode_hashes` and execution evidence.

These are evidence references, never model inputs. Paths can be remapped across
hosts, but the byte hashes/receipts must match. Resolve paths beneath the stated
root. Validate through the training tree's existing verified-collection reader
and checked episode reader; do not trust an arbitrary path or a teacher success
claim. Only the four D_repair_train tasks may contribute seed replay. Both arms
carry identical seed references. Probe and other support demonstrations cannot
receive training exposure in this update.

**Required training-reader checks:** reject unknown format/version, failed
content seal, source-collection mismatch, synthetic/canary/not-ready exports,
missing/duplicate/misordered tasks, train/probe overlap (including groups), any
unreviewed/unapproved target or changed episode hash, absent seed evidence,
changed theta0 files/base revision, and changed tokenizer/template. Construct
model inputs from the schema-2 whitelist above and use the existing encoder;
never pass a full arm entry or receipt to a model. Re-tokenize on the training
tree; do not import native token arrays from collection audit records. Record
all training exposure on that branch. `repair.export.require_approved` is an
executable reference guard, not a required cross-branch dependency.

## Fixed-checkpoint observation and training receipt

`tools.repair_probes` serves one observation schedule at a time:
`shared/0`, `N/{1,5,20}`, or `F/{1,5,20}`. The collection must be real and sealed.
The existing LoRA equivalence report must bind the served adapter, configured
base/revision and endpoint. Server provenance must show **BATCH_INVARIANT=1**.
Checkpoint zero must match collection theta0 hashes.

For each fixed repair probe, h_i is immediately before the first changed Near
action. A_i+ is built only from successful execution witnesses with the same
reset and every preceding original action/observation/audit transition
unchanged. Earlier-changing Far routes are excluded. A real reset and exact
original reply-prefix replay precede eight sampled turns, with seeds 0..7,
T=0.7, top_p=0.95, n=1, thinking off and output cap 1024. Request seeds use the
existing sampled student's hash of `[seed, task_id, zero_based_turn]`. Neither
repaired reasoning nor a diagnosis enters the probe prompt.

The first two seeds each get another independent reset+prefix replay and an
autonomous continuation to the original H=40. Prefix turns consume H; parser
errors use the existing feedback/turn semantics. Context overflow is a known
failure, transport/replay errors remain unknown. Failure position means the
first observed parser error after the prefix, otherwise the terminal boundary;
it is not a claimed causal diagnosis. Unknown runs have no failure position.
A_i+ is a witnessed subset, so a miss is not automatically an incorrect action.

Only shared/0 and N/F/20 run the 24 frozen D_unseen_quick tasks, greedily from
initial state through **eval.run_suite.run_suite**, saving existing per-task
records. Evaluation makes no teacher calls and writes no training examples.
Per-checkpoint `checkpoint.json` is sealed and accompanied by provenance and a
manifest. The combined JSON/TSVs report average and per-probe
`delta_m_N_minus_delta_m_F`, continuation outcomes, paired quick-eval outcomes,
and missing checkpoints/unknowns. No missing measurement becomes a zero.

For nonzero checkpoints, the training branch must produce a JSON receipt with
this contract (additional accounting fields are permitted):

```json
{
  "format": "repair_e2_training_run",
  "schema_version": 1,
  "arm": "N",
  "dry_run": false,
  "theta0": {"identity": "<copy the export's identity object>", "files": {}},
  "training_collection_sha256": "<exact e2_training_collection.json byte hash>",
  "source_collection_sha256": "<exact finalized collection.json byte hash>",
  "checkpoints": [{
    "update": 1,
    "status": "saved",
    "path": "/training/host/checkpoint/path",
    "sha256": {"adapter_config.json": "<byte hash>", "adapter_model.safetensors": "<byte hash>"}
  }]
}
```

`theta0` is the complete unchanged export object, not the illustrative abbreviated
value above. `checkpoints` includes one saved record for each fixed update
1/5/20. `sha256` maps **every file** in that checkpoint directory, including any
training-state/metadata files. The probe reader validates the actual served
adapter directory against that map, so the historical `path` can differ across
hosts. It also validates arm, export/source file hashes, all episode approvals,
and theta0. This is a receipt interface only; this branch creates no trainer.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m tools.repair_probes \
  --collection "$NEW_APPROVED_DIR/collection.json" --arm N --checkpoint 20 \
  --training-manifest "$TRAINING_MANIFEST" \
  --endpoint "$ENDPOINT" --model-name "$MODEL_NAME" \
  --server-provenance "$SERVER_PROVENANCE" --lora-check "$LORA_CHECK" \
  --output "$NEW_PROBE_DIR"
# For shared/0 omit --training-manifest. --training-collection defaults to the
# adjacent e2_training_collection.json; it may be supplied explicitly.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m tools.repair_probes \
  --combine "$CP0/checkpoint.json" "$N1/checkpoint.json" "$F1/checkpoint.json" \
  "$N5/checkpoint.json" "$F5/checkpoint.json" "$N20/checkpoint.json" "$F20/checkpoint.json" \
  --output "$NEW_COMBINED_DIR"
```

