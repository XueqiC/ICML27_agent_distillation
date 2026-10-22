# Collection and artifact hash receipts

`load_verified_collection` memoizes each successfully validated collection for
the life of the process, returning the same validated tuple on repeated calls.
The key includes the resolved root, fresh byte hashes of `manifest.json`,
`identity.json`, and `selection.json`, the task manifest and split-freeze digests,
and receipt-cache mode. Checked support bindings are memoized too. A hit checks
the top-level hashes of all transitive parents/support collections; it does not
scan episode trees. `allow_subset=False` restrictions still apply to cached views.
Failed validation is never cached. Callers must treat returned objects as immutable.

On the first load it reads the collection, selection, identity, attempt receipts, episodes and
environment verification, and runs every existing comparison and TRAIN/split
check. Both failed and successful attempts remain checked. `artifact_hashes`
and its batch variants also use the same receipts outside collection validation,
including student snapshots, checkpoints, and the episode checks during encoding.
Hash values, ordering and all comparisons remain unchanged.

The separate artifact-hash memo is bound to the resolved collection and directory paths,
fresh byte SHA256 hashes of `manifest.json`, `selection.json`, and `identity.json`,
and each artifact's relative path, size, nanosecond mtime and inode. Every use
enumerates the directory again and stats every file. Additions, deletions and
metadata changes therefore invalidate the affected entries. The memo holds at
most 16,384 directory receipts; evicted entries can still use disk receipts.
Standalone artifact memo entries use the resolved artifact directory with no
collection anchors. They use the same persistent receipt format and metadata
checks; every caller still performs its original identity/hash comparisons.

Persistent receipts are JSON with a version, resolved directory, list of
`[relative_path, size, mtime_ns, inode, sha256]`, and payload checksum. A receipt
hit reuses a SHA256 only when the file metadata matches. The resulting hash map
is compared against the collection's recorded episode/teacher hashes exactly as
before. Missing, malformed, checksum-invalid, or unreadable receipts cause
rehashing. Evidence I/O errors still propagate. Writes use private temporary
files and atomic replacement; cache failures do not prevent validation.

Configure the external directory with `SED_HASH_CACHE_DIR`. Otherwise, an
explicit `TMPDIR` uses `$TMPDIR/sed-hash-receipts-<uid>`; without it the default is
`~/.cache/sed-hash-receipts`. A cache inside the collection (including a resolved
symlink into it) is refused and only the process memo is used. An unwritable
cache also falls back to memory. No receipt, lock or other cache file is written
into a collection or hashed artifact tree. Deleting the external cache is safe.

Set **`SED_HASH_CACHE=0`** to disable artifact hash reuse and use the original full-hash
path, including its existing `SED_HASH_WORKERS` scheduling. Whole-collection
validation is still memoized once per process, with a separate key for this mode.
Use a new CPU preparation process for a fresh full audit. Enabled validation
batches work by directory, using at most two concurrent trees (also bounded by
`SED_HASH_WORKERS`) to avoid per-file futures and metadata/GIL contention. Each
directory is scanned once with `scandir`; file links are hashed and directory
links are not traversed, as before. `provenance.json`
records `hash_cache.mode`, scope, cache directory, metadata fields, version and modes
observed so far in the process. Provenance written before loading records the
configured mode; subsequent writes also record observed fallbacks.

Metadata caching has a deliberate detection boundary: a same-size in-place
content modification with mtime restored and inode unchanged can reuse the old
hash. Inode reuse, coarse timestamp resolution, or filesystem metadata spoofing
can have the same effect.
Use full rehash for adversarial-storage audits. Receipt checksums detect
accidental corruption; they do not authenticate a cache against someone who can
rewrite both its payload and checksum. Keep the cache in trusted local storage.
Collections are immutable by protocol after publication, including all episode,
teacher, receipt, view, parent, and support files. The process memo and prepared
seal rely on this assumption. Neither detects an episode-only edit made after
validation without changing the collection anchors. A fresh CPU preparation
checks episode evidence again; use full rehash for a byte-level audit.
A metadata change during
a new byte hash is rejected, but neither this cache nor the original loader
provides a transactional filesystem snapshot.

## CPU preparation before GPU training

SmartAD, Kang trajectory SFT and AD-FTP support `--mode prepare`. SmartAD and
AD-FTP also require `--through train`, consuming the already published collection.
Preparation performs pipeline identity checks, full selected/support/parent
collection validation, SmartAD selection/student binding, snapshot hashes, and
the exact tokenizer, encoding and data checks formerly run at training startup. It uses
`TRAIN_PYTHON` directly, without torchrun, model loading or updates. Each call
starts a fresh CPU trainer process, validates each distinct collection root once
in that process, and retains an attempt under `training/preparations/` (SmartAD
and AD-FTP) or `preparations/` (Kang). Repeated accesses in the pipeline process
also share validation. No checks, training masks, hash values, or example values
are weakened by memoization.

Preparation publishes `prepared/<pipeline-identity-digest>/examples.json` and
`seal.json` beneath the training output (`training/` for SmartAD/AD-FTP, the
output itself for Kang). The seal is published last and binds:

- Pipeline identity, recipe/config and every effective trainer option except
  the attempt output and pipeline transport path.
- Every reached collection's top-level manifest, identity and selection byte
  hashes, including views, parents and support; SmartAD selection metadata too.
- Config, task manifest, split freeze and pipeline task-list byte hashes.
- Snapshot artifact hashes, tokenizer assets and chat-template digest, code
  hashes, exact encoded-example bytes and semantic digest/count.

The train pipeline and torchrun rank zero independently verify the seal, current
identity/recipe/code, bound top-level files, snapshot artifact receipts, and
encoded examples. Rank zero broadcasts the verified examples to other ranks.
Neither GPU entry point re-scans episode files, loads the tokenizer, or encodes
training data. Missing, invalid or stale seals refuse training with an explicit
instruction to run CPU `--mode prepare`; there is no full-validation fallback.
This applies to resumed/completed pipeline stages as well. Direct trainer calls
require `--pipeline-output` pointing at the prepared pipeline output; pipeline
launchers supply it automatically. Trainer provenance records the seal's path,
file SHA256 and encoded-example SHA256 in `config.baseline_training.prepared_seal`.

The seal and examples reside on trusted storage, like the frozen collection.
Its checksum detects corruption, not an adversary who can rewrite both the
artifact and its hashes. Keep prepared data immutable while training. Re-running
CPU preparation can replace a damaged seal/examples for the same frozen identity;
a code/config/recipe identity change requires a new pipeline output.

For a separate CPU SLURM allocation, use `--partition=hpg-milan --gpus=0` and:

| Launcher | Preparation environment |
| --- | --- |
| `scripts/hpg_smartad.slurm` | `SMARTAD_THROUGH=train SMARTAD_MODE=prepare` |
| `scripts/hpg_kang.slurm` | `KANG_MODE=prepare` |
| `scripts/hpg_adftp.slurm` | `ADFTP_THROUGH=train ADFTP_MODE=prepare` |

After preparation succeeds, run the same launcher with its mode set to `train`
in the intended GPU allocation. Keep `VERIFIED`, `MODEL_PATH`, output directory,
config, recipe overrides, and **NGPUS** identical: NGPUS describes the future
training world size even during CPU preparation. The launchers and Python entry
points refuse preparation when GPU allocation metadata or visible GPU IDs are
present, including when CUDA was hidden inside a GPU allocation.
Existing identities remain frozen to their code hashes. For an output tree
created by an older revision, use a new pipeline output directory with the same
published input collection; do not edit the old identity to accept the patch.

Use the same trusted **shared filesystem** `SED_HASH_CACHE_DIR` in both jobs.
The three launchers default to `$HOME/.cache/sed-hash-receipts`, independently
of node-local `TMPDIR`. Paths and file stat identities must stay the same;
copying data to a different node-local tree will require new hashes.
`SED_HASH_CACHE=0` prevents snapshot hash warming/reuse and makes fresh CPU
validation hash every evidence byte. GPU startup still requires a seal in this
mode: it checks collection anchors and encoded examples, without traversing
episode trees. Snapshot receipt verification still scans the snapshot's files.

## Startup timing provenance

Pipeline `startup_timing.jsonl` records each invocation's mode and inclusive
`identity_prepare`, `selection_validation` (SmartAD), `collection_load`, and
`snapshot_hashing` seconds. Trainer timing lives in each train/preparation
attempt's `startup_timing.jsonl`, alongside its output directory; its
`provenance.json` links that log and the pipeline parent log. The trainer adds
`data_prepare`, `encode`, `model_load`, and `first_update`. Both pipeline and
trainer add `seal_verification` during training; `data_prepare`/`encode` and
collection validation occur during CPU preparation. First update is
recorded after the first optimizer/scheduler step (CUDA synchronized), once per
rank-zero invocation, with both training-loop duration and elapsed trainer
startup seconds. Compare its timestamp to the parent pipeline's `startup` event
for total pipeline-to-first-update wall time. Nested phase durations overlap
and must not be summed.

Events are durable as phases finish, including failed phases. Logs survive a
failure before training provenance is published; a killed job with no first
update has no `first_update` event. Timing sidecars do not modify frozen
identities or existing checkpoint provenance after publication. These are
engineering diagnostics; local CPU tests do not establish hpg startup timings.
