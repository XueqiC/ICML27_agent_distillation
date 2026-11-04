# V0 raw hash receipts and CPU input validation

Worktree: `/home/xueqi/hq/projects/ICML27_wt_repair_train`, base HEAD
`81506cffda0abeec94b8c61262af8cb90c9da9d4`. Measurements were made on rai
(`sn4622122543`) on 2026-11-04. Changes remain unstaged. No commits, GPU
execution, model updates, teacher/API calls, or network access were used.
Inputs were read without modifying their contents. Scratch output and receipts
are under this worktree's `logs/v0_hash_profile/`.

## Original-code profile

The profiler was started before implementation changes. It imported the
original reader and trainer, wrapped `sed.storage.file_hash` before those
imports, then executed the requested D-data check twice in the same Python
process. It recorded resolved paths, sizes, byte-hash calls and nested phase
timings. The second call reused the original collection cache. Profile JSON
serialization happened outside the measured intervals. The profiler itself
adds a resolve/stat per hash; some runs overlapped CPU tests or diagnostic
checks. These instrumented times are not directly comparable to the user's
unprofiled 27-minute rai measurement or the reported Lustre job timings.

| Original D-data check | Wall seconds | Byte-hash calls | Unique byte-hashed paths | Bytes hashed |
| --- | ---: | ---: | ---: | ---: |
| Cold receipts | 2,444.163 | 1,021,739 | 971,193 | 54,904,513,359 |
| Warm, same process | 2,212.802 | 127,237 | 76,691 | 43,346,290,624 |

Both calls contain **50,546 repeated hashes** beyond their unique paths.
The original collection cache does help: its byte-hash calls drop by
894,502. It writes collection receipts, but does not cover the V0 reader's
raw-file hashes. No raw-file receipt namespace existed in the baseline.

Phase timings below are **exclusive**, so they can be summed. Hash counts
and bytes are attributed to the innermost active phase. Current-dataset and
parent inclusive times additionally contain their nested seal and row work.
The assembly phase includes both datasets, so its inclusive total must not be
added to their inclusive totals.

| Phase | Cold wall s | Warm wall s | Cold calls | Warm calls | Cold bytes | Warm bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current dataset, excluding nested work | 559.159 | 346.824 | 894,534 | 32 | 12,889,449,275 | 1,331,226,540 |
| Synthetic manifests | 0.032 | 0.031 | 2 | 2 | 2,704,845 | 2,704,845 |
| Sealed inputs, across current and parent | 732.589 | 719.119 | 127,068 | 127,068 | 36,797,241,690 | 36,797,241,690 |
| Parent reread, excluding nested work | 358.065 | 354.326 | 33 | 33 | 1,343,810,424 | 1,343,810,424 |
| Rows and dataset assembly | 773.779 | 772.436 | 80 | 80 | 2,722,514,859 | 2,722,514,859 |
| Continuation recipe | 0.885 | 0.835 | 21 | 21 | 1,148,777,517 | 1,148,777,517 |
| Batch schedule | 0.015 | 0.013 | 0 | 0 | 0 | 0 |
| Other orchestration | 19.640 | 19.218 | 1 | 1 | 14,749 | 14,749 |

Unique byte-hashed paths per phase (cold / warm): current dataset
894,533 / 31; sealed inputs 76,660 / 76,660; parent reread 32 / 32;
rows/assembly 40 / 40; synthetic manifests 2 / 2; continuation recipe
18 / 18; schedule 0 / 0; other 1 / 1. These per-phase sets overlap.

Current-dataset inclusive time: 1,328.085 s cold / 1,108.602 s warm.
Parent reread inclusive time: 1,095.536 s / 1,084.133 s.
Actual byte-hash function time, summed across threads, was approximately
178.147 s cold / 36.807 s warm. Most wall time therefore lies outside the
byte-reading loop: filesystem discovery/resolution, metadata work, receipt
processing, sealed-input inventories and semantic/row validation.

Representative repeated paths, in each original check:

| Input | Calls | Total bytes hashed |
| --- | ---: | ---: |
| Gemma snapshot's resolved base-model blob | 2 | 31,985,191,768 |
| theta0 `adapter_model.safetensors` | 12 | 6,698,182,272 |
| Full TRAIN seed-bank `manifest.json` | 7 | 1,448,301,505 |
| Parent `dataset/examples.jsonl` | 3 | 231,613,161 |
| Parent `dataset/manifest.json` | 5 | 62,919,420 |
| Expanded `synth/manifest.json` | 3 | 4,604,145 |
| Parent `synth/manifest.json` | 4 | 4,680,520 |

The current and parent synthetic manifests are both verified; their hashes
are not a major byte cost. Their task files and parent references participate
in repeated sealed-input verification. Rephrased/generated instances retain
their original task/seed accounting; this change introduces no new data.

An additional warm collection cProfile recorded 133,856,149 calls in
99.028 profiled seconds, including 845,310 `os.scandir` calls and 1,217,612
`DirEntry.stat` calls. Receipt parsing spent time constructing Paths and
checking SHA characters in Python. This motivated equivalent string/compiled
regex checks and reuse of immutable metadata tuples. The repeated filesystem
operations explain sensitivity to per-file filesystem latency; **Lustre
latency itself was not measured in this worktree**.

Raw profile artifacts:

- `logs/v0_hash_profile/profile_baseline.py`
- `logs/v0_hash_profile/baseline_{cold,warm_in_process}_profile.json`
- `logs/v0_hash_profile/baseline_{cold,warm_in_process}_compact.json`
- `logs/v0_hash_profile/baseline_{cold,warm_in_process}_output.json`
- `logs/v0_hash_profile/collection_warm.pstats`

## Implementation and preserved validation

`sed.raw_hash_cache` scopes V0 hashing through `sed.storage.file_hash` and
shares a process memo keyed by `(resolved path, size, mtime_ns, inode)`.
Striped locks prevent concurrent duplicate byte reads. Metadata is checked
again after a byte read; changed inputs fail without publishing that hash.
Collection cold misses feed the same memo, so the raw and collection readers
do not independently byte-hash the same unchanged file.

Persistent raw receipts reuse `sed.hash_receipts`' versioned, checksummed,
atomically published format in a separate `raw-files/` namespace. Missing,
malformed or inaccessible receipts fall back to hashing. Input/evidence roots
are protected against receipt writes, including shard-directory symlinks.
The CPU check flushes receipts before returning; rank 0 flushes before
broadcast/model startup. `SED_HASH_CACHE_DIR` must be shared between the CPU
check and GPU job on the same filesystem/path layout.

`SED_HASH_CACHE=0` and `--full-rehash` bypass both process memo and receipts,
including provenance. As with the existing collection receipts, a same-size
edit with restored mtime and unchanged inode requires this byte-audit mode.
Receipt trust has not been expanded to semantic validation results.

Raw cold misses use bounded ordered futures (default eight workers,
`SED_HASH_WORKERS` or `--hash-workers` configurable). Warm hits avoid per-file
future overhead. The existing collection tree worker cap remains two.
Check output and training provenance report the cache mode, worker setting,
calls, process hits, receipt hits, misses, byte hashes and bytes. Raw counters
exclude collection receipt hits that never call the raw hash service.

The V0 check avoids building a second million-file collection inventory:
the existing recursive collection loader still enumerates every artifact,
stats it, compares its hash and checks collection semantics. Metadata anchors,
receipt references and binding checks remain explicit. Directory and file
symlink confinement is checked even on receipt hits. Full audit/listing paths
retain the expanded inventory. An already validated replay-inspector seed
collection is reused immediately instead of validated a second time.
The parent dataset is still independently reread and validated.

Path resolution shares directory work only within one read-only verification
pass, checks every leaf and resets between passes. Seals are compared on
every verification. Row validation renders the complete prompt prefix once
instead of rebuilding every growing prefix: with the unchanged empty task
argument, the complete rendering contains all earlier prefixes unchanged.
Every visible-field whitelist, turn parse, supervised-action check and exact
rendered-message comparison remains enforced.

Distributed `run` initializes its process groups before input validation.
Only rank 0 executes dataset, parent, recipe and choice verification. It
broadcasts verified objects and the dataset digest through the existing
long-timeout Gloo helper. Other ranks do not validate or hash the input bank.
A failure flag is broadcast before rank 0 rethrows; every rank fails.
Tests use a fake object process group, including actual pickle serialization
of a complete verified fixture. No multi-GPU or HPG execution was performed.

## Real checks and regression tests

All four real checks passed with exit 0 and empty stderr. Each check used a
fresh interpreter; cold means an empty receipt directory, not a flushed
operating-system cache. Each variant had its own new directory, with its
warm check immediately following its cold check. The CPU test suite was
running in the background; the initial D-data cold run also overlapped the
focused V0 suite. Later quiet warm repeats are recorded separately.

| Variant | Cold seconds | Warm seconds | Cold byte hashes / misses | Cold bytes | Warm receipt hits | Warm process hits | Warm misses / bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-data | 437.878 | 222.352 | 971,193 | 30,179,370,940 | 78,084 | 54,785 | 0 / 0 |
| D-param | 443.098 | 212.349 | 970,526 | 30,123,579,953 | 77,417 | 54,781 | 0 / 0 |

Cold process hits: 56,178 for D-data; 56,174 for D-param. Cold receipt hits
were zero for both. Raw hash calls were 1,027,371 / 132,869 cold/warm for
D-data and 1,026,700 / 132,198 for D-param. D-data's cold byte-hash count is
exactly the original profile's unique-path count. Its raw receipt directory
contains 62,742 per-directory JSON receipt files.

The paired warm checks **do not meet the requested under-two-minute target**.
Raw byte hashing has been eliminated, while filesystem discovery/stat calls,
receipt parsing and full semantic validation remain. No Lustre performance
claim follows from these rai measurements.

After the full CPU suite finished, two additional fresh-process warm checks
ran sequentially with no other check/test job from this task running:
**223.700 s for D-data and 225.593 s for D-param**. Both exited 0 with empty
stderr and reproduced their complete earlier warm report, including cache
counters, exactly. The target miss therefore persists without the test
workload. These are third checks against the same receipt directories, not
replacements for the immediate second-run measurements above. Artifacts:
`logs/v0_hash_profile/quiet_warm/{D-data,D-param}.json` and
`logs/v0_hash_profile/quiet_warm/measurements.json`.

All original output fields in all four reports are byte-identical after
canonical JSON encoding. This includes rows, buckets, dataset and effective
weight digests, trainable parameter counts, parent binding and G inheritance
records; the only added output field is `hash_cache`. Comparisons use the
original D-data cold profile output and the supplied D-param rai report.

| Verified value | D-data | D-param |
| --- | --- | --- |
| Rows | 8,772 | 8,408 |
| Trainable parameters | 139,526,144 | 39,026,688 |
| dataset_sha256 | `2a37f5e77bad4bc4283254523a7304c296cfb0a19709cd000a6b2a53e875fa07` | `546da5aa2bd3ce5bc07b1497bd658a265dc2a2a9b693c20a042f0d1028e7a873` |
| episode_weights_sha256 | `546bd510ddb65b924a6264bfe5b0c9e982a107849a8ed5eee01cfe0771e55231` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |

Exact bucket values and comparisons:
`logs/v0_hash_profile/output_comparison.json`. The final reports are
`logs/v0_hash_profile/release/{D-data,D-param}_{cold,warm}.json`.

The three requested V0 suites plus `tests/test_v0_hash_cache.py` passed:
**145 passed in 114.87 s**, including **33 new tests**. A separate run of the
new tests and existing collection-receipt suite passed **58 tests in 30.10 s**.
Coverage includes alias/nested/thread memo deduplication; persistent hits;
size, mtime and inode invalidation; corrupt/missing receipts; CLI/environment
audit bypass; deterministic bounded workers; changing-file rejection;
protected evidence roots and symlink escapes, including unchanged-inode
escapes with warm receipts; repeat verification; prompt equivalence; check/run
provenance; and rank-0-only validation with serialized fake broadcasts and
fatal failure propagation. Logs: `v0_release.log` and `receipt_fast_tests.log`
under `logs/v0_hash_profile/`.

The full CPU suite finished with **3,282 passed, 57 failed, 6 skipped**,
181 warnings, in **2,719.50 s (45:19)**; exit 1 reflects the existing failures.
Its **exact set of 57 failed test IDs matches the saved pre-change set**:
zero new failures and zero resolved failures. This is 33 more passing tests
than the recorded 3,249-pass baseline. The comparison checks IDs, not just
the failure count. Artifacts:

- `logs/v0_hash_profile/full_cpu_release.log`
- `logs/v0_hash_profile/full_cpu_release.xml`
- `logs/v0_hash_profile/pre_full_lastfailed.json`
- `logs/v0_hash_profile/failure_comparison.json`

The final full-suite command was:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TMPDIR="$PWD/logs/v0_hash_profile/tmp" \
TORCHINDUCTOR_CACHE_DIR="$PWD/logs/v0_hash_profile/torch-cache" \
/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  --basetemp=logs/v0_hash_profile/pytest-release \
  --junitxml=logs/v0_hash_profile/full_cpu_release.xml
```

The exact commands are retained in
`logs/v0_hash_profile/release/real_measurements.json` and generated by
`logs/v0_hash_profile/check_release.py`. They use the user-supplied datasets,
theta0, snapshot, relocation, weights/module mask and configuration. Environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
SED_HASH_CACHE=1 SED_HASH_CACHE_DIR=<worktree>/logs/v0_hash_profile/release/<variant>-receipts \
/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python tools/train_v0_arm.py inputs --check ...
```

The V0 and collection test runs additionally set `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and worktree-local temporary/cache directories. The test
fixture forbids CUDA, sockets/DNS and live OpenAI transport. Dependency
versions and explicit changed-source hashes are in
`logs/v0_hash_profile/validation_metadata.json`.

Earlier optimization trials are retained under `logs/v0_hash_profile/`,
including interrupted checks/suites. They are not counted as completed
validation. An earlier completed D-data pair measured 616.725 s cold and
326.874 s warm; subsequent metadata/prompt optimizations reduced a separate
warm check to 210.692 s. These are diagnostic measurements, not the final
four-run comparison. This is an engineering validation, not a research result
or a baseline reproduction.

## Changed files

| Files | Change |
| --- | --- |
| `src/sed/raw_hash_cache.py`, `src/sed/storage.py` | Scoped raw hashing, process memo, persistent receipts and bounded workers |
| `src/sed/hash_receipts.py`, `src/sed/seed_evidence.py` | Shared memo/context propagation, equivalent receipt parsing optimizations, relocation confinement |
| `src/sed/provenance.py` | Capture cache accounting after input/code hashes |
| `repair/v2_inputs.py`, `tools/v0_data.py` | V0 verification integration, compact validated collection inventory, prompt/path work deduplication |
| `tools/train_v0_arm.py` | Rank-0 validation/broadcast, flags, check and training cache reports |
| `tests/test_v0_hash_cache.py` | 33 focused CPU/offline tests |
| `scripts/hpg_train_v0.slurm`, `tools/V0_DATASET.md` | Preserve exported cache directory and document CPU warming / GPU reuse |
| `reports/engineering/v0_check_profile.md` | Profile, measurements, identity comparisons and regression results |

`git diff --check` and the launcher shell syntax check pass. HEAD remains
`81506cf`; the index is unchanged and every modification/new file is unstaged.
