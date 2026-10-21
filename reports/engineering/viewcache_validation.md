# Verified-view hash cache validation

Measured on 2026-10-20, branch `wt/viewcache`, Python 3.11.7, Linux x86-64,
`SED_HASH_WORKERS=16` (cached validation caps tree workers at two), CPU only.
Source (read-only):
`/home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2`.
The worktree has no copy of this bank. No collection files were changed and no
teacher/environment calls or training were performed.

## Before: measured bottleneck

The real K30 view validates the full source: 4,772 attempts, 3,098 verified
trajectories/unique instances and 1,383 task groups. Attempts cover 3,553 unique
instances in 1,465 groups. K30 and K500 have respectively 30 and 500 trusted
trajectories, unique instances and groups. Attempt evidence contains 894,502 file
hashes; the old loader then rehashes 245,292 successful-episode files, followed by
the selected view's files.

The initial K30 cProfile load took **411.531 seconds**. The profile attributes
381.822 seconds (92.8%) to artifact readers and their directory scans/hashing,
including 2,902,756 `ThreadPoolExecutor.submit` calls. The main thread spent
231.996 seconds in lock acquisition/waits; worker I/O overlaps these waits, so
those figures are not additive. Instrumented artifact/receipt hashing made
1,146,951 calls. JSON decoding took 6.696 seconds across 19,001 calls. A separate
read of the 206,900,215-byte source manifest took 0.331 seconds, and parsing it
took 0.874 seconds. Artifact traversal, repeated hashing and fine-grained thread
scheduling dominate; manifest JSON parsing does not explain the delay.

Profiling adds overhead. The final timing table uses unprofiled measurements
where available and labels the profiled first load separately. “Cold” refers to
an empty **receipt cache**, not a flushed operating-system page cache. No system
cache eviction was attempted. Large view loads run sequentially. These are
single engineering measurements, not experiment results or statistical claims.

## Change and invariants

Collection validation uses an external per-directory receipt plus an in-process
memo bound to resolved paths, fresh manifest/selection/identity byte hashes and
all artifact path/size/mtime_ns/inode tuples. It enumerates every directory on
each load, rehashes changed files, and keeps all existing manifest, receipt,
environment-win, frozen TRAIN, selection and count comparisons. Failed attempts
are retained and checked. Concurrent matching loads share memo work.

The cached path batches whole directory trees, replacing millions of tiny
scan/file futures with bounded directory-level work. A single `scandir` per
directory and `DirEntry.stat` avoid duplicate traversal, directory stats and
constructing/reparsing Paths for unchanged files. The metadata cache uses at
most two concurrent trees, also bounded by `SED_HASH_WORKERS`; the raw-hash
baseline retains its original 16 workers. On the 2,323 files in the real K30
episodes, the initial pathlib cache took 0.470–0.471 seconds warm at 16 workers.
The single-scan version took 0.175–0.195 seconds at 16 workers and 0.074 seconds
at two workers. The cold single-scan sample also favored two workers (0.280
seconds versus 0.542 at 16). These small calibration runs occurred during an
intermediate cold-cache pilot; final large view loads remain sequential.

The first cache implementation still used pathlib traversal and 16 directory
workers. It took 345.892 seconds cold and 251.001 seconds warm for K30, despite
reducing warm artifact byte reads to zero (only 33 receipt-reference hashes).
This negative pilot is retained in the raw artifact. It motivated the single
scan and worker-limit changes; those results are not presented as final timings.

`SED_HASH_CACHE=0` executes the original full-hash implementation. Generic artifact hashing outside
collection validation also retains its original behavior.

Receipts are versioned, checksummed JSON with atomic replacement. Bad or
unreadable receipts are discarded. Cache destinations inside a collection are
refused; an unavailable destination falls back to memory. The provenance writer
records configured cache mode and observed modes/fallbacks. See
[`docs/HASH_RECEIPTS.md`](../../docs/HASH_RECEIPTS.md) for configuration and the
explicit same-size/forged-mtime/unchanged-inode detection limit. Checksums detect
corruption, not hostile cache rewriting; full rehash is the audit mode.

## Reproduce

Use the real source path above as `$SOURCE` and a fresh external directory as
`$CACHE`. Run from the worktree with `PYTHONPATH=src:.`:

```sh
CUDA_VISIBLE_DEVICES='' SED_HASH_CACHE=0 .venv/bin/python -m tools.profile_verified_views --source "$SOURCE" --ks 30 30 500 500
CUDA_VISIBLE_DEVICES='' SED_HASH_CACHE=1 SED_HASH_CACHE_DIR="$CACHE" .venv/bin/python -m tools.profile_verified_views --source "$SOURCE" --ks 30 30
CUDA_VISIBLE_DEVICES='' SED_HASH_CACHE=1 SED_HASH_CACHE_DIR="$CACHE" .venv/bin/python -m tools.profile_verified_views --source "$SOURCE" --ks 30
```

Repeat the two enabled commands with K500 and a separate fresh cache directory.
The first enabled load is cold; the second in that process uses the memo. The
separate process proves persistent receipt reuse. `--profile /tmp/view.pstats`
optionally captures the first load with cProfile. Hash-call counters exclude the
three fresh collection-anchor hashes per source/view; they include the selected
view's `receipt.json` reference hashes, which intentionally remain full hashes.

## Tests

All tests ran with `CUDA_VISIBLE_DEVICES=''`, fake clients/environments and the
repository's network/API-blocking fixture. Source files were held fixed for the
final runs:

- **195 passed in 448.97 seconds:** `test_seed_evidence_cache.py`,
  `test_full_train_views.py`, `test_chain_collections.py`, `test_stage2c_seeds.py`,
  `test_stage2f.py`, `test_hash_parallel.py`, `test_full_train.py`,
  `test_full_train_recovery.py`, with the final single-scan implementation.
- **39 passed in 159.54 seconds with `SED_HASH_CACHE=0`:**
  `test_full_train_views.py`, `test_chain_collections.py`.

The 25 new cache cases cover full-hash equivalence, persistent/process hits,
concurrent memo reuse, metadata and file-set changes, all three identity anchors,
unselected-source tampering, malformed/checksum-invalid/unreadable receipts,
unusable or internal cache paths, mid-hash mutation, disabled caches, provenance,
evidence-read errors, link/filename traversal equivalence, and the explicitly
demonstrated forged-mtime limit.

An earlier combined run had two failures because source edits changed code hashes
while tests were comparing collection identities. Both passed in the final
stable-source run above; no identity check was weakened.

## Final timings

Seconds, without cProfile:

| View | Before, full rehash | After, empty receipts | After, warm process memo | After, warm receipts / fresh process |
|---|---:|---:|---:|---:|
| K30 | 302.642 | 131.277 | 59.039 | 76.885 |
| K500 | 360.167 / 474.173 | 159.300 | 70.156 | 86.415 |

K30's first baseline load was profiled (411.531 seconds); its unprofiled repeat
is the before value above. Both K500 before values are unprofiled consecutive
loads. Disabling caching rehashed all evidence on every load. The final K30
fresh-process measurement was repeated independently after its own cold/memo
pair; the earlier final-code persistent read took 83.010 seconds. Runtime
variability is visible, so these numbers do not establish a fixed speedup on
other storage systems. Compared with the faster unprofiled baseline, warm memo
loads are about **5.1× faster** for both views; fresh-process receipt loads are
about **3.9× (K30) / 4.2× (K500)** faster.

| View | Before artifact + reference hash calls | Cold after | Warm after, either cache |
|---|---:|---:|---:|
| K30 | 1,146,951 | 894,535 | 33 |
| K500 | 1,264,590 | 895,059 | 557 |

Cold loads hash each of the 894,502 source artifacts once. The remaining 33/557
calls are the view's attempt-receipt reference hashes. Warm loads rehash **zero
episode/teacher artifacts**, while still enumerating and stat-checking every
file and rerunning semantic validation. Collection-anchor hashing remains fresh
and is excluded from those counters. The source/view anchor files, frozen data
inputs and final code hashes were checked again and matched the recorded values.

[`viewcache_profile.json`](viewcache_profile.json) preserves the full timing
rows, negative intermediate pilot, calibration timings, source/freeze/code
hashes, Python/OS/dependency versions, cache settings and test results. The local
cProfile artifact and external benchmark receipts are under
`/tmp/sed-viewcache-profile`; no cache was placed in the collection. Small offline
tests and calibration overlapped some measurements, but large view loads were
sequential throughout.
