# Verified collection hash receipts

`load_verified_collection` caches artifact SHA256 values during validation. It
still reads the collection, selection, identity, attempt receipts, episodes and
environment verification, and runs every existing comparison and TRAIN/split
check. Both failed and successful attempts remain checked. Generic
`artifact_hashes` calls outside collection validation remain full byte hashes.

The in-process memo is bound to the resolved collection and directory paths,
fresh byte SHA256 hashes of `manifest.json`, `selection.json`, and `identity.json`,
and each artifact's relative path, size, nanosecond mtime and inode. Every use
enumerates the directory again and stats every file. Additions, deletions and
metadata changes therefore invalidate the affected entries. The memo holds at
most 16,384 directory receipts; evicted entries can still use disk receipts.

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
into a collection. Deleting the external cache is safe.

Set **`SED_HASH_CACHE=0`** to disable both caches and use the original full-hash
path, including its existing `SED_HASH_WORKERS` scheduling. Enabled validation
batches work by directory, using at most two concurrent trees (also bounded by
`SED_HASH_WORKERS`) to avoid per-file futures and metadata/GIL contention. Each
directory is scanned once with `scandir`; file links are hashed and directory
links are not traversed, as before. `provenance.json`
records `hash_cache.mode`, cache directory, metadata fields, version and modes
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
