"""Process-local validation of immutable collections, with transitive anchors.

Only successful validations are memoized. Callers must treat returned objects
and collection contents as immutable. No episode metadata is scanned on a hit.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

from sed.hash_receipts import cache_enabled
from sed.storage import digest, file_hash

_MEMO = {}
_LOCK = RLock()
_CAPTURE = ContextVar("validated_collection_bindings", default=None)


def anchors(root):
    return {name: file_hash(Path(root) / name)
            for name in ("manifest.json", "identity.json", "selection.json")}


@contextmanager
def capture_collections():
    bindings = {}
    parent = _CAPTURE.get()
    token = _CAPTURE.set(bindings)
    try:
        yield bindings
    finally:
        _CAPTURE.reset(token)
        if parent is not None:
            parent.update(bindings)


def memoized(kind, root, manifest, freeze, compute, *, binding=None):
    root = str(Path(root).resolve())
    current = anchors(root)
    key = (kind, root, digest(current), digest(manifest), digest(freeze),
           cache_enabled(), digest(binding))
    # Reentrant for view -> parent -> support; concurrent callers share work.
    with _LOCK:
        entry = _MEMO.get(key)
        if entry is not None:
            result, dependencies = entry
            if all((current if path == root else anchors(path)) == hashes
                   for path, hashes in dependencies.items()):
                if _CAPTURE.get() is not None:
                    _CAPTURE.get().update(dependencies)
                return result
        with capture_collections() as dependencies:
            dependencies[root] = current
            result = compute()
            if any(anchors(path) != hashes for path, hashes in dependencies.items()):
                raise ValueError("Collection anchors changed during validation")
        _MEMO[key] = result, dependencies
        return result


def clear_validation_cache():
    """Start a fresh validation session (also used by mutation/audit tests)."""
    with _LOCK:
        _MEMO.clear()
