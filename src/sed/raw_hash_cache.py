"""Scoped raw-file hash acceleration for V0, never semantic validation.

Receipts use hash_receipts' checked, atomically published directory format in
a separate namespace. Trust and full-rehash escape hatch are identical: a
same-size edit with restored mtime and inode requires an explicit byte audit.
"""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from functools import wraps
import os
from pathlib import Path
from threading import RLock

from sed import hash_receipts, storage


_MEMO = {}
_MEMO_LOCK = RLock()
_FILE_LOCKS = [RLock() for _ in range(256)]


class RawFileHashes(hash_receipts.HashReceipts):
    """One operation's counters/receipt buffers; byte hashes share process memo."""
    def __init__(self, workers=None):
        self.workers = int(os.environ.get("SED_HASH_WORKERS", "8") if workers is None else workers)
        if self.workers < 1:
            raise ValueError("Hash workers must be positive")
        self.enabled = hash_receipts.cache_enabled()
        self.cache = hash_receipts.cache_directory() if self.enabled else None
        self.protected = []
        self.boundaries = {}
        self.rows = {}
        self.dirty = set()
        self.byte_hashed_paths = set()
        self.lock = RLock()
        self.counts = Counter(calls=0, memo_hits=0, receipt_hits=0, misses=0,
                              byte_hashes=0, bytes_hashed=0)

    def protect(self, *directories):
        for directory in directories:
            if directory is not None:
                root = Path(directory).resolve()
                if root not in self.protected:
                    self.protected.append(root)
                if self.cache is not None and self.cache.is_relative_to(root):
                    self.cache = None

    def receipt_path(self, directory):
        path = super().receipt_path(directory)
        return self.cache / "raw-files" / path.relative_to(self.cache) if path is not None else None

    def register_relocation(self, relocation):
        self.boundaries.update(relocation.destinations)

    def checked_path(self, value):
        """The inventory's relocation confinement, also applied during scans."""
        path = Path(value).absolute()
        resolved = path.resolve()
        for new, boundary in self.boundaries.items():
            if (path.is_relative_to(new) or path.is_relative_to(boundary)) and not resolved.is_relative_to(boundary):
                raise ValueError("Relocation symlink escape in collection evidence")
        return resolved

    def _rows(self, directory):
        # Caller holds self.lock. Read a directory receipt once per operation.
        if directory not in self.rows:
            self.rows[directory] = self._read(self.receipt_path(directory), directory)
        return self.rows[directory]

    def hash(self, path, hash_file):
        path = Path(path).resolve()
        return self._hash(path, hash_file)

    def hash_entry(self, entry, signature, hash_file):
        # The collection walker starts at a resolved root and never descends
        # directory symlinks. Only a file symlink needs further resolution.
        # Reuse its fresh DirEntry stat instead of resolving/stating a million
        # deep paths again. Collection receipts already persist these hashes.
        path = Path(entry.path)
        if entry.is_symlink():
            path = path.resolve()
        return self.hash_resolved(path, hash_file, signature=signature, persistent=False)

    def hash_resolved(self, path, hash_file, *, signature=None, persistent=True):
        """Reuse the caller's fresh confinement/path resolution, never its seal."""
        def uncached(item):
            token = storage._FILE_HASH_CACHE.set(None)
            try:
                return hash_file(item)
            finally:
                storage._FILE_HASH_CACHE.reset(token)
        return self._hash(path, uncached, signature=signature, persistent=persistent)

    def likely_cached(self, path):
        """Scheduling hint only; hash() still checks fresh file metadata.

        Warm files need no thread/future. A stale hint merely hashes that file
        synchronously; it can never grant a cache hit or skip a validation.
        """
        if not self.enabled or not hash_receipts.cache_enabled():
            return False
        with self.lock:
            return Path(path).name in self._rows(Path(path).parent)

    def _hash(self, path, hash_file, *, signature=None, persistent=True):
        with _FILE_LOCKS[hash(path) % len(_FILE_LOCKS)]:
            if signature is None:
                signature = hash_receipts._signature(path.stat())
            key = (str(path), *signature)
            with self.lock:
                self.counts["calls"] += 1
            # Recheck the environment too: nested audit scopes must bypass both
            # memo and receipts, including already-populated InputFiles objects.
            enabled = self.enabled and hash_receipts.cache_enabled()
            sha, hit = None, "misses"
            if enabled:
                with _MEMO_LOCK:
                    sha = _MEMO.get(key)
                if sha is not None:
                    hit = "memo_hits"
                elif persistent:
                    with self.lock:
                        previous = self._rows(path.parent).get(path.name)
                    if previous is not None and previous[:3] == signature:
                        sha, hit = previous[3], "receipt_hits"
            if sha is None:
                sha = hash_file(path)
                with self.lock:
                    self.counts["byte_hashes"] += 1
                    self.counts["bytes_hashed"] += signature[0]
                    self.byte_hashed_paths.add(path)
                if hash_receipts._signature(path.stat()) != signature:
                    raise ValueError("Input changed while hashing")
            with self.lock:
                self.counts[hit] += 1
                if enabled and persistent:
                    rows = self._rows(path.parent)
                    row = (*signature, sha)
                    if rows.get(path.name) != row:
                        rows[path.name] = row
                        self.dirty.add(path.parent)
            if enabled:
                with _MEMO_LOCK:
                    _MEMO[key] = sha
            return sha

    def map(self, function, values, *, parallel_if=lambda value: True):
        """At most workers futures; ordered results and errors, inherited scope."""
        if self.workers == 1:
            yield from map(function, values)
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = deque()
            for value in values:
                if not parallel_if(value):
                    while pending:
                        yield pending.popleft().result()
                    yield function(value)
                    continue
                pending.append(pool.submit(copy_context().run, function, value))
                if len(pending) == self.workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    def flush(self):
        with self.lock:
            for directory in sorted(self.dirty):
                self._write(self.receipt_path(directory), directory,
                            dict(sorted(self.rows[directory].items())))
            self.dirty.clear()

    def report(self):
        with self.lock:
            enabled = self.enabled and hash_receipts.cache_enabled()
            return {"scope": "v0_raw_input_hashes", "mode": (
                "metadata_receipts_and_process_memo" if enabled and self.cache else
                "process_memo_only" if enabled else "full_rehash"),
                "directory": str(self.cache) if enabled and self.cache else None,
                "stat_fields": ["size", "mtime_ns", "inode"],
                "receipt_version": hash_receipts._VERSION, "workers": self.workers,
                **self.counts}


@contextmanager
def raw_input_cache(*, full_rehash=False, workers=None):
    """Nest across reader, parent, recipe and provenance without global patching."""
    previous = os.environ.get("SED_HASH_CACHE")
    if full_rehash:
        os.environ["SED_HASH_CACHE"] = "0"
    active = storage.active_file_hash_cache()
    token = None
    try:
        cache = active if active is not None else RawFileHashes(workers)
        if active is None:
            token = storage._FILE_HASH_CACHE.set(cache)
        yield cache
    finally:
        if token is not None:
            try:
                cache.flush()
            finally:
                storage._FILE_HASH_CACHE.reset(token)
        if full_rehash:
            if previous is None:
                os.environ.pop("SED_HASH_CACHE", None)
            else:
                os.environ["SED_HASH_CACHE"] = previous


def cached_inputs(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with raw_input_cache():
            return function(*args, **kwargs)
    return wrapped
