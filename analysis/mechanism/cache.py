"""Copy a frozen teacher-cache snapshot; never lock or mutate the source run."""
import hashlib
import json
from pathlib import Path

from analysis.mechanism.common import read_json
from sed.storage import atomic_write, digest, file_hash, write_json


def seed_summary(seed):
    return {key: value for key, value in seed.items() if key != "entries"} if seed else None


def _validate_entry(relative, content):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[0] not in {"teacher_cache", "teacher_blobs"}:
        raise ValueError("Unsafe cache manifest path")
    row = json.loads(content)
    if path.parts[0] == "teacher_cache":
        if row["cache_key"] != path.stem or digest(row["result"]) != row["result_sha256"]:
            raise ValueError("Seed teacher cache integrity check failed")
    elif hashlib.sha256(content).hexdigest() != path.stem:
        raise ValueError("Seed teacher blob integrity check failed")


def prepare_cache_seed(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Cache source and destination workspaces must be separate")
    saved = output/"teacher_cache_seed.json"
    if saved.exists():
        seed = read_json(saved)
        if seed["source_path"] != str(source) or digest(seed["entries"]) != seed["manifest_sha256"]:
            raise ValueError("Cache seed manifest changed")
        return seed  # A growing live source must not change an already frozen import.
    root = source if (source/"teacher_cache").is_dir() else source/"fce"
    if not (root/"teacher_cache").is_dir() or not (root/"teacher_blobs").is_dir():
        raise ValueError("Source needs teacher_cache/ and teacher_blobs/ (directly or under fce/)")
    entries = {}
    for name in ("teacher_cache", "teacher_blobs"):
        directory = root/name
        if directory.is_symlink() or root.is_symlink():
            raise ValueError("Cache source directories must not be symlinks")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("Cache source entries must not be symlinks")
            if not path.is_file() or path.suffix != ".json" or path.name.startswith("."):
                continue  # Exclude locks and unpublished atomic-write temporaries.
            relative = str(path.relative_to(root))
            content = path.read_bytes()
            _validate_entry(relative, content)
            entries[relative] = hashlib.sha256(content).hexdigest()
    return {"source_path": str(source), "cache_workspace_path": str(root),
            "entry_count": sum(p.startswith("teacher_cache/") for p in entries),
            "blob_entry_count": sum(p.startswith("teacher_blobs/") for p in entries),
            "file_count": len(entries), "manifest_sha256": digest(entries), "entries": entries}


def copy_cache_seed(seed, output):
    output = Path(output)
    manifest = output/"teacher_cache_seed.json"
    if manifest.exists():
        if read_json(manifest) != seed:
            raise ValueError("Cache seed manifest changed")
    else:
        write_json(manifest, seed, exclusive=True)
    # Blobs first, so no published cache entry lacks its original prompt/output.
    for relative, expected in sorted(seed["entries"].items()):
        destination = output/relative
        if any(p.is_symlink() for p in [destination, *destination.parents] if p != output.parent):
            raise ValueError("Cache destination must not contain symlinks")
        if destination.exists():
            if file_hash(destination) != expected:
                raise ValueError("Cache seed conflicts with destination entry")
            continue
        source = Path(seed["cache_workspace_path"])/relative
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Cache source changed after manifest freeze")
        _validate_entry(relative, content)
        atomic_write(destination, content, exclusive=True)
