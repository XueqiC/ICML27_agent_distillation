# V0 training interchange, version 1

This is the V0 training contract aligned with the eval producer at `e5bf24f`
(`repair/v0_data.py`, `repair/v0_common.py`, `repair/v0_synth.py`, and
`tools/V0_ROUND.md` in `ICML27_wt_repair_eval`). The implementations are
`v0_data.py`, `v0_grad_features.py`,
`v0_groups.py`, and `train_v0_arm.py`. They import the V2 reader, encoder,
adapter loader, scorer, training loop, accounting, and exporter; no V2 module
is patched. There are no teacher, generator, or environment calls here.

The controlling design is the main checkout's
`docs/direction/2026-11-02_v0_prereg_ZH.md` §2 D/E and
`docs/direction/2026-10-26_agent_distillation_pipeline_v0_1_ZH.md` §§9/11,
with the explicit implementation request taking precedence: six groups,
bucket -> weighted episode -> uniform row sampling, B=16, and V2 SFT loss.
This does **not** implement the older optional 0.25/0.75 reason/action loss,
task-first sampling, historical-buffer mixture, or checkpoint selection.

## Files and seals

```
dataset/
  manifest.json
  examples.jsonl
  replay/                 # Unchanged, sealed V2 CONTINUE bundle
    manifest.json
    examples.jsonl
    examples.sha256
```

Root `examples.jsonl` contains R/G/S **decision** rows. Each line has exactly
the V2 fields plus `v0`:

```json
{
  "arm": "V0",
  "evidence": {"episode_ref": "records/repair-1.json", "step": 3,
    "executor": "patch", "local_event": null, "episode_success": true,
    "label_source": "teacher_rationale", "label_check": "passed"},
  "example_id": "unique-stable-decision-id",
  "group_id": "synth-group-1",
  "kind": "decision",
  "messages": ["...exact V2 role/content messages, see below..."],
  "source": "producer-source-label",
  "supervised_assistant_indices": [2],
  "task_id": "train/synth/example-1",
  "task_type": "pick_and_place_simple",
  "weight": 1.0,
  "v0": {"bucket": "R", "episode_id": "repair-1", "generator": "G1",
    "bin": "near", "edit_decision": true,
    "synthetic_manifest_sha256": "<64 lowercase hex>"}
}
```

The abbreviated `messages` above is illustrative, not a valid row. Actual
messages are dictionaries with **only** `role` and string `content`, exactly
matching `sed.react.build_messages`: system then alternating user/assistant,
ending in assistant. User content has only `observation` and
`admissible_commands`. A decision supervises only its final assistant index.
All successful-trajectory decisions should be supplied; the producer omits
rejected/repeated-rejected targets. The consumer rejects supervised actions
outside admissible commands, ungrounded instance references, extra message
fields, and `Guidance:`. V2 action-only labels remain supported. Hidden audit
fields, evidence, task manifests, and `v0` metadata never enter tokenization.
The training encoder reads the same explicit fields as V2.

Root evidence uses executor `student`, `patch`, or `handback-student`, with
`edit_decision=true` exactly for `patch`. Unlabeled patches use
`label_source="action_only_no_label"`, `label_check="not_available"`; these
must still have an action-only supervised target. The V0 validator adapts
these names for V2's existing evidence checks without altering stored rows
or relaxing the evidence field whitelist. D retains its original V2 evidence.

`example_id` is a unique nonempty producer ID; unlike V2, it need not use a
particular digest formula. An `episode_id` identifies exactly one task,
group, generator, bin, bucket, and evidence episode reference. Its decision step
indices are unique. Row `weight` stays 1; weighting happens at episode sampling.
Every row must contain exactly the six `v0` keys shown above. `edit_decision`
is a JSON boolean. Allowed generators are `G1|G2|G3|G5|seed`; allowed bins are
`near|mid|far|very_far|seed`. Non-synthetic task IDs must be θ₀ seed tasks, use
generator and bin `seed`, and use `""` for `synthetic_manifest_sha256`.

Bucket meanings follow the newer preregistration: R is the initial synthesis
repair pool, G the expansion repair/success pool, S initial-synthesis student
success (at most one episode per task), D the original K demonstrations.
Synthetic tasks can therefore occur in R, G, or S. D is mandatory and comes
only from `replay/`; do not duplicate it in root rows. Its V2 bytes and masks
are preserved. In memory, D rows get arm V0 and metadata:
`episode_id="seed/" + example_id`, generator/bin `seed`, `edit_decision=false`,
empty synthetic hash. Original trajectory targets, including V2's diagnostic
exceptions for historical seed labels, remain unchanged.
The consumer's `dataset_sha256` includes these virtual rows and therefore
includes `bin`. Feature/group artifacts made with the older five-key reader
must be regenerated to match this identity.

Root `manifest.json` requires:

```json
{
  "schema": "v0-train-interchange/1",
  "arm": "V0",
  "created": "<producer timestamp>",
  "code_revision": "<producer revision>",
  "examples_sha256": "<raw examples.jsonl file hash>",
  "replay_manifest_sha256": "<raw replay/manifest.json hash>",
  "splits_sha256": "<raw method_v1_splits.json hash>",
  "theta0": {"adapter_weights_sha256": "<adapter_model.safetensors hash>"},
  "inputs": {"<input path>": "<raw SHA256>"}
}
```

`inputs` must seal the configured task manifest, configured split freeze,
method split freeze, and synthetic manifest when used. Byte-identical producer
copies of those task/split files are accepted when the consumer uses another
checkout: their raw SHA256 seals establish identity across paths.
Every listed producer copy is still verified, and provenance retains both
the producer inputs and configured local files. Semantically equivalent JSON
with different bytes does not satisfy this check. The inputs must also include
the producer's collection/verification and expansion accounting artifacts:
parent seeds, raw outputs, rejected candidates, failed attempts, tokens,
transitions, generator/teacher actual IDs and timestamps. Additional manifest
fields are audit-only and are preserved. Listed inputs are byte-verified;
this consumer does not replay environments or independently certify the
producer's success claims. Unknown or failed episode-success labels cannot
be used as successful training trajectories. Rejections and unknowns stay in
the producer's audit artifacts, not the training rows.

The unchanged replay bundle has the complete existing V2 contract, including
hashed method split inputs and exclusions. θ₀ must pass V2's registered
E4B K20 recipe checks. The reader **always** follows θ₀'s original hashed
`baseline_training.verified` view and validates its executed seed receipts,
even when the method split already lists K20. A CONTINUE bundle may contain
the same K rows in both its main and nested replay files. After validating
both sealed bundles, the V0 reader deduplicates by `example_id`, requiring
identical canonical JSON bytes for every duplicate (including evidence and
supervision). Conflicting duplicates fail. The first occurrence and its
origin are retained; the source bundles and V2 arm behavior stay unchanged.
Inspection and feature/training provenance record `replay_deduplication`
with `input_rows`, `unique_rows`, and `duplicates_removed`. Deduplicated replay
task multiplicities must match the verified receipts exactly; missing or
extra seed trajectories still fail. K is the verified trajectory count; unique tasks,
groups, episodes, and repeated training exposure are reported separately.

## Synthetic task manifest

Pass the path with `--synthetic-manifest`. The producer also records this path
in the dataset manifest's audit-only `synthetic_manifest` field; the reader
uses the explicit argument and requires it to be a sealed input. Its SHA256
is the hash of the raw file bytes, not canonicalized JSON or a digest of the
`tasks` list. This is exactly `SyntheticManifest.sha256` on the producer.
Every synthetic row must carry that hash. The minimal identity fields are:

```json
{
  "schema": "v0-synthetic-tasks/1",
  "seed_task_ids": ["train/original-group/trial_0"],
  "tasks": [{
    "task_id": "train/synth/example-1",
    "parent_seed_task_id": "train/original-group/trial_0",
    "group_id": "synth-group-1",
    "task_type": "pick_and_place_simple",
    "generator": "G1",
    "bin": "near",
    "split": "train",
    "sealed": false,
    "heldout_probe": false,
    "status": "valid",
    "item_fingerprint": "<effective state/goal or recovery-state fingerprint>"
  }]
}
```

IDs have exactly three slash-separated components, with a nonempty final
component other than `.` or `..`, and no backslashes. Every entry, including
held-out probes, must have `status="valid"` and a parent in θ₀'s verified K
seed task set. A synthetic task can never be a parent. The manifest's
`seed_task_ids` list cannot authorize additional parents: the reader derives
membership from θ₀'s executed seed receipts.

The generator/bin pairs match the producer's `SyntheticManifest`:

| Bin | Generators |
| --- | --- |
| `near` | `G1`, `G3` |
| `mid` | `G2` |
| `far` | `G2` |
| `very_far` | `G5` |

`seed` is reserved for seed rows, never synthetic manifest entries. Training
entries require `heldout_probe=false`, `split="train"`, `sealed=false`.
Held-out probes require **all three** of `heldout_probe=true`,
`split="holdout"`, `sealed=true`. The flags must be JSON booleans; missing or
contradictory markings are rejected. Probes remain in the hashed freeze but
are always rejected as training rows. All entries have unique task IDs and
`item_fingerprint` values, and training/probe group IDs cannot overlap.
Task type, group, generator **and bin** must match each training row.

The full producer manifest also records seed/split source hashes, seed
provenance, K/seed, grouping/witness policy, and per-task edits, allocation,
fingerprints, asset paths/hashes and validation receipts. These additional
fields remain audit-only; none enters the model input whitelist. Missing
manifest, unknown synthetic ID, hash mismatch, parent outside K, or diagnostic
identity is an error. Freeze the manifest before collection/experiments.
Expansion publishes a new combined manifest and rebinds inherited rows to
its raw-byte hash in a new dataset. Rephrases and retries retain the original
task identity.

## Offline contract tests

`tests/test_v0_producer_contract.py` imports the sibling producer read-only,
with path insertion confined to the test and bytecode writes disabled. It
temporarily isolates the producer's repair/teacher imports from the reader's
cached modules and restores local imports before testing the reader. It
uses the producer's allocator, `SyntheticManifest`, `decision_rows`, and
`publish_dataset` with fake candidates/executions and sealed fake K20 receipts.
It covers every allowed synthetic generator/bin pair, R/G/S and virtual D,
raw-byte manifest hashing, unchanged replay bytes, and the producer's consumer
dataset hash with `seed_bin=True`. Negative cases include unknown bins and
generators, held-out probe rows, and row/manifest bin or generator mismatches.
The cross-worktree tests skip with an explicit reason if the sibling cannot
be imported; the standalone reader tests still run.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest \
  tests/test_v0_training.py tests/test_v0_producer_contract.py \
  tests/test_v0_expansion_weights.py -q
```

## Gradient artifacts

`v0_grad_features.py` loads θ₀ once on one allocated GPU. `--buckets R,G`
(the default) restricts gradient extraction and module moments to repair
buckets. S/D episodes are inventoried but contribute no gradients or moments.
Selected episodes are processed in lexical episode-ID order, using only rows
with `edit_decision=true`. An R/G episode without any such row is refused
before model loading/backward unless `--allow-no-edit` explicitly enables
whole-episode loss. This also applies to student successes in expansion G;
their presence does not silently enable a fallback. Loss is the **mean of V2 per-row
token-mean SFT losses**, using the same encoded masks and shared-history
scorer. No optimizer is created or stepped. Gradients are cleared before and
after every episode; adapter parameters are hashed before/after. Inherited
V2 LoRA dropout is 0.05: extraction temporarily sets dropout to zero and
restores it afterward; training retains V2 dropout. The CRN trainable-parameter and
dropout helpers are reused. Its `surrogate_gradient` and `message_backward`
pool tokens across examples, so cannot replace V2's row-mean objective.

Coordinates concatenate trainable LoRA A/B tensors sorted by full runtime
parameter name, row-major element order. CountSketch uses SplitMix64 on
`coordinate + 17` (standard add/mix constants in code), low bits modulo 2048
for bucket, highest bit for sign (0 -> +1). CPU float64 bucket sums avoid
device-dependent CUDA atomic reduction order; output is normalized float32.
The coordinate map is independent of device. Numerical model gradients can
still differ across hardware/precision. Raw gradient norms are float64 host
reductions. Each LoRA A or B tensor is a separate SNR module; vectors and
squared norms accumulate in float32 **on the model device**, with
`variance=max(E||g||²-||E g||²,0)` and `eps=1e-12` in the SNR denominator.

Files: `sketch.npy`, `norms.json` (episode -> raw norm/flags), `modules.json`,
`index.json`, and `provenance.json`. Index records sorted IDs, row selections,
parameter names/sizes/hash, adapter/dataset hashes, artifact seals, and
episodes/second. `bucket_filter` records the effective bucket list;
`dataset_episodes` inventories every dataset episode by ID/bucket so S/D can
receive explicit unit weights. `available_episodes` counts the entire dataset,
and `eligible_episodes` counts episodes passing the bucket filter.
`allow_no_edit`, `no_edit_count`, and `no_edit_episode_ids` disclose the override
and every selected episode using whole-episode fallback. `modules.json` binds
moments to the same bucket filter and extracted episode IDs.
`--max-episodes` takes the lexical prefix **after** filtering. Omitting any R/G
episode, by limit or bucket filter, marks grouping outputs as an incomplete
smoke subset. Excluding S/D does not. Production training refuses weights/module
masks from incomplete subsets. Explicit S/D extraction is possible for separate
diagnostics, but its artifacts cannot be used by V0 grouping.

## Grouping, weights, and module masks

`v0_groups.py --features FEATURES --output GROUPS` runs on CPU. It requires an
explicit nonempty `bucket_filter` restricted to R/G, matching feature entries
and module moments. Old unfiltered artifacts must be regenerated; mixed-bucket
moments cannot be corrected by dropping sketches afterward. Only R/G episodes
with nonzero raw gradient and sketch participate. ZERO_GRADIENT
means raw norm <1e-12; these remain trainable with neutral weight before R/G
normalization. Nonzero gradients projected to zero have ZERO_SKETCH status
and the same treatment. All-zero inputs produce no directional groups.

Clustering uses cosine distance, max-raw-norm first medoid, farthest-point
initialization, k=min(6,floor(N/2)) or one group for N<4, and at most 20
assignment/update rounds. Every tie is lexical episode ID, including
assignment of medoids themselves; empty groups disappear. Singleton groups
merge into the nearest other medoid until none remain or only one group is
left. The merged group's medoid is recomputed. Farthest representative
excludes the medoid and is null for a sole member.

Consistency is group size times mean cosine over **distinct unordered pairs**;
a sole member uses cosine 1. R/G episodes receive their group's consistency,
normalized jointly to mean 1 over extracted R/G episodes. S/D are never clustered
and always have explicit weight 1, using the index's dataset inventory. Omitted
R/G episodes in a smoke subset have no computed weight.
Signed consistency can be negative: the formula then cannot define sampling
probabilities. The tool preserves groups and module ranking, writes an
invalid-status weights artifact, and fails instead of silently clipping or
shifting it. All-zero R/G weights or a nonempty zero-mass bucket also fail.
These cases require an explicit weighting-protocol decision before training.

Outputs have versioned schemas: `groups.json`, `weights.json` with an episode
ID -> weight map, and `trainable_modules.json` with `selected` full tensor
names and complete descending-SNR ranking from R/G moments only. Ties use tensor name, selecting
ceil(module_count/4). Artifacts bind dataset, adapter, parameter-name hash,
and source feature index. Do not manually edit these identities.

Expanded datasets may use choices bound to their verified parent dataset.
The producer seals the parent `manifest.json` and `examples.jsonl` under the
expanded manifest's `inputs`; the reader requires a unique parent, validates
it with the same V0 reader, and recomputes its composite dataset hash. The
artifact must match that hash and the current adapter and parameter names.
Inherited rows must remain identical except for their synthetic-manifest
reference, and additions must be new G episodes. Missing or changed parents
and unrelated dataset bindings are refused. Relocation applies to these
references without rewriting any source artifact.

For parent-bound D-data weights, `groups.json` beside `weights.json` must be
a sealed expansion input with matching source identity. The original weights
must reproduce its consistency formula. Each new G episode resolves through
its task to an expansion unit, cross-checked against `allocation.json`, sealed
`synth/allocation_execution.json`, sealed `synth/generated.json`, and the
synthetic manifest. The unit's source must be a parent R episode assigned to
that gradient group; the task's `group_id` is a separate identity. Missing or
conflicting units/groups fail. R and G raw consistency weights are normalized
together to episode mean 1 using `v0_groups.episode_weights`; excluded parent
gradients retain neutral raw mass and S/D stay at 1. New G gradients enter only
at the next round (§10.5), so this check does not extract gradients.

The check report, training manifest, and run provenance record the verified
parent, per-bucket row/episode and weight summaries, effective-weight digest,
and each G episode's task, unit, source repair, group, consistency, and final
weight. Provenance hashes the mapping artifacts. These remain audit-only.
Parent-bound D-param module selections need only the verified binding and
existing tensor-name checks; they do not depend on G weights or unit mapping.

## Training and commands

Each of 16 global batch slots draws a bucket (default .50/.25/.15/.10,
renormalized over nonempty buckets), then an episode proportional to its
weight, then a uniform row. Sampling is with replacement and deterministic
from `--seed`, independent of input row order and GPU rank count. Longer
episodes do not receive extra mass merely because they have more rows.
`--bucket-probabilities R G S D` overrides bucket masses explicitly.

`--trainable-modules` freezes every tensor except the selected existing LoRA
tensors before optimizer creation. Existing AdamW moments are subset by
parameter name/order; frozen tensors get no optimizer state. All adapter
tensors, including frozen ones, remain in exported checkpoints. Precision,
LR, AdamW settings, clipping, schedule restart, and distributed gradient
averaging come from V2. The final update always gets a checkpoint; no
checkpoint selection occurs. Exposure logs distinguish planned counts from
counts completed at each optimizer callback, by bucket and episode.

```bash
python tools/train_v0_arm.py inputs --check \
  --dataset DATASET --theta0 THETA0 --snapshot SNAPSHOT \
  --synthetic-manifest SYNTHETIC_JSON

python tools/v0_grad_features.py \
  --dataset DATASET --theta0 THETA0 --snapshot SNAPSHOT \
  --synthetic-manifest SYNTHETIC_JSON --device cuda:0 --output FEATURES

python tools/v0_groups.py --features FEATURES --output GROUPS

python tools/train_v0_arm.py run \
  --dataset DATASET --theta0 THETA0 --snapshot SNAPSHOT \
  --synthetic-manifest SYNTHETIC_JSON --weights GROUPS/weights.json \
  --updates 100 --checkpoints 25 50 75 100 --seed 0 --gpus 2 --output RUN
```

For D-param use `--trainable-modules GROUPS/trainable_modules.json`; weights
are optional and default to 1. Updates and checkpoints are explicit; the
launcher does not guess rounding of `4 * episode_count / 16`. Input validation
supports the same explicit `--relocate OLD=NEW`, `--splits`, `--repair-splits`,
and `--dev140` paths as V2, without rewriting source artifacts.

`--dry-run` uses V2's tiny CPU model and byte tokenizer; optionally use
`--dry-run-tokenizer snapshot` for the real offline tokenizer. With a module
mask, requested real names are validated from θ₀ metadata, then selected A/B
sides map explicitly to fake A/B tensors. This only exercises orchestration,
not production model mask fidelity. Synthetic artifacts/counts are marked
and do not claim real model training. `inputs --check` loads no model.

The HPG launcher uses the V2 job sizing (2 B200, 4 CPUs, 64 GB, 12 hours) and
environment. Required env: `TRAIN_TREE DATASET OUT UPDATES CHECKPOINTS`;
optional `WEIGHTS MODULES SYNTHETIC_MANIFEST THETA0 SNAPSHOT SEED`. Supply
`SYNTHETIC_MANIFEST` for synthetic rows. No GPU is used by unit tests. A local
GPU smoke requires a fresh `bash ~/hq/ops/gpu_free.sh` check and, on rai,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`.
