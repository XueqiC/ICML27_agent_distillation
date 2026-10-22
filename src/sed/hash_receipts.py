"""Disposable, external receipts for collections and artifacts (never evidence).

Metadata reuse assumes trusted local storage: an in-place, same-size edit with
restored mtime and unchanged inode is not detectable. SED_HASH_CACHE=0 restores
byte hashing. Receipt checksums detect corruption, not a malicious cache writer.
Semantic validation and comparisons with the collection's hashes still run.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
import errno
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from threading import RLock

from sed.storage import canonical_bytes, digest, file_hash


_ACTIVE = ContextVar("collection_hash_receipts", default=None)
_MEMO = OrderedDict()
_LOCK = RLock()
_DIRECTORY_LOCKS = [RLock() for _ in range(256)]
_MAX_DIRECTORIES = 16384  # Includes the full TRAIN bank's episode + teacher trees.
_MODES = set()
_VERSION = 1


def cache_enabled():
    return os.environ.get("SED_HASH_CACHE", "1") != "0"


def cache_directory():
    configured = os.environ.get("SED_HASH_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.environ.get("TMPDIR"):
        return (Path(os.environ["TMPDIR"]) / f"sed-hash-receipts-{os.getuid()}").resolve()
    return (Path.home() / ".cache/sed-hash-receipts").resolve()


def cache_provenance():
    return {"mode": "metadata_receipts_and_process_memo" if cache_enabled() else "full_rehash",
            "observed_modes": sorted(_MODES),
            "scope": "verified_collection_validation_and_artifact_hashes",
            "stat_fields": ["size", "mtime_ns", "inode"],
            "tree_worker_cap": 2,
            "receipt_version": _VERSION,
            "directory": str(cache_directory()) if cache_enabled() else None}


def active_cache():
    return _ACTIVE.get() if cache_enabled() else None


def record_full_rehash():
    _MODES.add("full_rehash")


@contextmanager
def collection_cache(root):
    """Bind memo entries to fresh byte hashes of all three collection anchors."""
    if not cache_enabled():
        record_full_rehash()
        token = _ACTIVE.set(None)
    else:
        root = Path(root).resolve()
        anchors = tuple(file_hash(root / name) for name in
                        ("manifest.json", "selection.json", "identity.json"))
        token = _ACTIVE.set(HashReceipts(root, anchors))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _signature(info):
    return info.st_size, info.st_mtime_ns, info.st_ino


def _missing(error):
    return (error.errno in (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP)
            or getattr(error, "winerror", None) in (21, 123, 1921))


def _file_entries(directory):
    """One scandir per directory; follow file links, never descend directory links.

    Retain pathlib's ignored lock files, missing roots and permission handling.
    DirEntry avoids constructing and reparsing a Path for every warm-cache hit.
    We still stat every file; directory metadata is never a cache-validity key.
    """
    if not directory.is_dir():
        return []
    files, pending = [], [(str(directory), "")]
    while pending:
        path, prefix = pending.pop()
        try:
            entries = os.scandir(path)
        except OSError as error:
            if sys.version_info >= (3, 12) or isinstance(error, PermissionError):
                continue
            raise
        try:
            with entries:
                entries = list(entries)
        except PermissionError:
            continue
        for entry in entries:
            relative = prefix + entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError as error:
                if sys.version_info >= (3, 12) or _missing(error):
                    is_dir = False
                else:
                    raise
            if is_dir:
                pending.append((entry.path, relative + "/"))
            else:
                files.append((relative, entry))
    # Path sorts by components (a/file precedes a.txt), not the joined string.
    return sorted(files, key=lambda row: tuple(row[0].split("/")))


class HashReceipts:
    def __init__(self, root, anchors):
        self.root, self.anchors = root, anchors
        self.cache = cache_directory()
        # Reject a cache anywhere inside the collection, including siblings of
        # an episode and symlinks into the collection. Never create it there.
        protected = [root]
        protected.extend(p for p in root.parents
                         if (p / "identity.json").exists() or (p / "selection.json").exists())
        self.protected = protected
        if any(self.cache.is_relative_to(p) for p in protected):
            self.cache = None
        _MODES.add("metadata_receipts_and_process_memo" if self.cache else "process_memo_only")

    def receipt_path(self, directory):
        if self.cache is None or self.cache.is_relative_to(directory):
            return None
        key = digest(str(directory))
        return self.cache / key[:2] / (key + ".json")

    def _read(self, path, directory):
        if path is None:
            return {}
        try:
            envelope = json.loads(path.read_bytes())
            payload = envelope["payload"]
            if (payload["version"] != _VERSION or payload["directory"] != str(directory)
                    or envelope["sha256"] != digest(payload)):
                return {}
            rows = payload["files"]
            if not isinstance(rows, list):
                return {}
            result = {}
            for row in rows:
                if (not isinstance(row, list) or len(row) != 5
                        or not isinstance(row[0], str) or row[0] in result
                        or Path(row[0]).is_absolute() or ".." in Path(row[0]).parts
                        or any(type(v) is not int for v in row[1:4])
                        or row[1] < 0 or row[3] < 0
                        or not isinstance(row[4], str) or len(row[4]) != 64
                        or any(c not in "0123456789abcdef" for c in row[4])):
                    return {}
                result[row[0]] = tuple(row[1:])
            return result
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            return {}  # A receipt is an optional optimization, never authority.

    def _write(self, path, directory, rows):
        if path is None:
            return
        temporary = None
        try:
            # Also reject a shard-directory symlink into the evidence tree.
            parent = path.parent.resolve()
            if any(parent.is_relative_to(p) for p in [directory, *self.protected]):
                _MODES.add("process_memo_only")
                return
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = {"version": _VERSION, "directory": str(directory),
                       "files": [[name, *row] for name, row in rows.items()]}
            data = canonical_bytes({"payload": payload, "sha256": digest(payload)}) + b"\n"
            fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            # Atomic publication is sufficient for a disposable cache. No fsync
            # per episode; a torn/lost receipt is simply recomputed next time.
            os.replace(temporary, path)
        except OSError:
            _MODES.add("process_memo_only")
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def hashes(self, directory, hash_file):
        directory = Path(directory).resolve()
        key = (str(self.root), self.anchors, str(directory))
        # Concurrent loads of the same collection share one hashing operation.
        # Stripes bound lock storage without serializing unrelated directories.
        with _DIRECTORY_LOCKS[hash(key) % len(_DIRECTORY_LOCKS)]:
            return self._hashes(directory, hash_file, key)

    def _hashes(self, directory, hash_file, key):
        path = self.receipt_path(directory)
        with _LOCK:
            old = _MEMO.get(key)
            if old is not None:
                _MEMO.move_to_end(key)
        if old is None:
            old = self._read(path, directory)
        rows, result = {}, {}
        # Always enumerate the tree: stat'ing only old files would miss additions.
        for relative, entry in _file_entries(directory):
            try:
                info = entry.stat(follow_symlinks=True)
            except OSError as error:
                # Match Path.is_file(), including dangling links.
                if _missing(error):
                    continue
                raise
            if not stat.S_ISREG(info.st_mode) or entry.name.endswith(".lock"):
                continue
            signature = _signature(info)
            previous = old.get(relative)
            if previous is not None and previous[:3] == signature:
                sha = previous[3]
            else:
                item = Path(entry.path)
                sha = hash_file(item)
                if _signature(item.stat()) != signature:
                    raise ValueError("Evidence changed while hashing")
            rows[relative] = (*signature, sha)
            result[relative] = sha
        if rows != old:
            self._write(path, directory, rows)
        with _LOCK:
            _MEMO[key] = rows
            _MEMO.move_to_end(key)
            while len(_MEMO) > _MAX_DIRECTORIES:
                _MEMO.popitem(last=False)
        return result

    @contextmanager
    def batch(self, directories, workers, hash_file):
        with hash_batch(directories, workers, lambda path: self.hashes(path, hash_file)) as readers:
            yield readers


@contextmanager
def hash_batch(directories, workers, hash_directory):
    # Profiling on the real bank: many tiny metadata syscalls contend with
    # Python work at 16 threads. Two concurrent trees are faster cold/warm.
    workers = min(workers, 2)
    def read(value):
        return hash_directory(value() if callable(value) else value)
    if workers == 1:
        yield (lambda value=value: read(value) for value in directories)
        return
    # One task per tree avoids millions of tiny file/scan futures on warm
    # loads. Bounded prefetch and ordered result() preserve error ordering.
    from collections import deque
    from itertools import islice
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def readers():
            values = iter(directories)
            pending = deque(pool.submit(read, value) for value in islice(values, workers))
            while pending:
                yield pending.popleft().result
                for value in values:
                    pending.append(pool.submit(read, value))
                    break
        yield readers()
