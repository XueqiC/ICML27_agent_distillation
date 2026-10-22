# Collection and artifact hash receipts

`load_verified_collection` caches artifact SHA256 values during validation. It
still reads the collection, selection, identity, attempt receipts, episodes and
environment verification, and runs every existing comparison and TRAIN/split
check. Both failed and successful attempts remain checked. `artifact_hashes`
and its batch variants also use the same receipts outside collection validation,
including student snapshots, checkpoints, and the episode checks during encoding.
Hash values, ordering and all comparisons remain unchanged.

The in-process memo is bound to the resolved collection and directory paths,
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

Set **`SED_HASH_CACHE=0`** to disable both caches and use the original full-hash
path, including its existing `SED_HASH_WORKERS` scheduling. Enabled validation
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
Collections should remain immutable during validation: a metadata change during
a new byte hash is rejected, but neither this cache nor the original loader
provides a transactional filesystem snapshot.

## CPU preparation before GPU training

SmartAD, Kang trajectory SFT and AD-FTP support `--mode prepare`. SmartAD and
AD-FTP also require `--through train`, consuming the already published collection.
Preparation performs pipeline identity checks, full selected/support/parent
collection validation, SmartAD selection/student binding, snapshot hashes, and
the same tokenizer, encoding and data checks as the trainer. It uses
`TRAIN_PYTHON` directly, without torchrun, model loading or updates. Each call
revalidates and retains an attempt under `training/preparations/` (SmartAD and
AD-FTP) or `preparations/` (Kang). Train always repeats validation; preparation
is a cache warmup, never an authorization or validation shortcut.

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
`SED_HASH_CACHE=0` intentionally prevents warming/reuse. Semantic reads and
per-file metadata scans still run on every GPU startup; this optimization does
not eliminate filesystem metadata latency.

## Startup timing provenance

Pipeline `startup_timing.jsonl` records each invocation's mode and inclusive
`identity_prepare`, `selection_validation` (SmartAD), `collection_load`, and
`snapshot_hashing` seconds. Trainer timing lives in each train/preparation
attempt's `startup_timing.jsonl`, alongside its output directory; its
`provenance.json` links that log and the pipeline parent log. The trainer adds
`data_prepare`, `encode`, `model_load`, and `first_update`. First update is
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
