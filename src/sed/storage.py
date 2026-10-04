"""Durable POSIX storage. JSONL is logically append-only, atomically published.

Stage 1 logs are small: copy-on-append avoids ever publishing a torn JSON line.
An advisory lock serializes writers; fsync + same-directory rename publishes
the unchanged old prefix plus one line. Readers see the old or new whole log.
This deliberately costs O(log size) per append. Use a transactional segmented
log before scaling, not unsafe tail truncation or silent crash recovery.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path):
    """Persist newly created directory entries, including blob hash shards."""
    path = Path(path)
    if path.is_dir():
        return
    durable_mkdir(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    fsync_dir(path.parent)


@contextmanager
def locked(path):
    path = Path(path)
    durable_mkdir(path.parent)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def atomic_write(path, content, *, exclusive=False):
    path = Path(path)
    durable_mkdir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(tmp, path)  # Atomic create-if-absent, never overwrite a freeze.
            os.unlink(tmp)
        else:
            os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value, *, exclusive=False):
    atomic_write(path, canonical_bytes(value) + b"\n", exclusive=exclusive)


class JsonlLedger:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, value, *, unique_key=None):
        line = canonical_bytes(value) + b"\n"
        with locked(str(self.path) + ".lock"):
            prefix = self.path.read_bytes() if self.path.exists() else b""
            if prefix and not prefix.endswith(b"\n"):
                raise ValueError("Damaged JSONL tail; refusing to modify existing evidence")
            records = [json.loads(row) for row in prefix.splitlines()]
            if unique_key and any(r[unique_key] == value[unique_key] for r in records):
                raise ValueError(f"Duplicate {unique_key}")
            atomic_write(self.path, prefix + line)


class BlobStore:
    def __init__(self, root):
        self.root = Path(root)

    def put(self, value):
        content = canonical_bytes(value)
        sha = hashlib.sha256(content).hexdigest()
        path = self.root / sha[:2] / (sha + ".json")
        try:
            atomic_write(path, content, exclusive=True)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError("Corrupt content-addressed blob") from None
        return "sha256:" + sha

    def get(self, ref):
        import re
        if not isinstance(ref, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", ref):
            raise ValueError("Invalid blob reference")
        sha = ref.removeprefix("sha256:")
        content = (self.root / sha[:2] / (sha + ".json")).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError("Blob hash mismatch")
        return json.loads(content)
