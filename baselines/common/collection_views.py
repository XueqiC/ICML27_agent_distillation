"""Teacher-side task-exact views; support sampling belongs to sed, never here."""
import json
import os
from pathlib import Path
import shutil
import tempfile

from sed.manifest import frozen_write
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.storage import digest, file_hash, fsync_dir, locked


def read(path):
    return json.loads(Path(path).read_text())


def parse_views(entries):
    result = {}
    for entry in entries:
        key, sep, path = entry.partition("=")
        if not sep or not key.isdecimal() or int(key) < 1 or not path or int(key) in result:
            raise ValueError("--subset-view requires distinct positive K=PATH entries")
        result[int(key)] = str(Path(path).resolve())
    return result


def bind_support(path, collection, selection):
    return {"path": str(Path(path).resolve()), "K": collection["K"],
            "manifest_sha256": digest(collection), "selection_sha256": digest(selection),
            "manifest_file_sha256": file_hash(Path(path) / "manifest.json"),
            "selection_file_sha256": file_hash(Path(path) / "selection.json"),
            "task_ids": sorted(r["task_id"] for r in collection["verified"])}


def support_reference(binding, output, *, resolver=None):
    path = resolver.resolve(binding["path"]) if resolver else Path(binding["path"])
    return {**binding, "path": os.path.relpath(path, Path(output).resolve())}


def bind_views(verified, source, selection, manifest, freeze, entries=(), *, subset_ks=()):
    if subset_ks and selection.get("selection_rule") == "stratified_nested_v2":
        raise ValueError("Nested v2 requires explicit --subset-view; no second sampling rule")
    if entries and subset_ks:
        raise ValueError("Use explicit --subset-view OR legacy --subset-k")
    bindings = []
    large = {r["task_id"] for r in source["verified"]}
    previous = set()
    for k, path in sorted(parse_views(entries).items()):
        support, small, _ = load_verified_collection(path, manifest, freeze)
        ids = {r["task_id"] for r in support["verified"]}
        if (support["K"] != k or len(ids) != k or k >= source["K"] or not ids <= large
                or not previous <= ids or small["seed"] != selection["seed"]):
            raise ValueError("Support view task_id mismatch, non-nested tasks, K or seed mismatch")
        bindings.append(bind_support(path, support, small))
        previous = ids
    return bindings


def check_support(binding, manifest, freeze, *, resolver=None):
    path = resolver.resolve(binding["path"]) if resolver else Path(binding["path"])
    support, selection, _ = load_verified_collection(path, manifest, freeze)
    actual = bind_support(path, support, selection)
    if {k: v for k, v in actual.items() if k != "path"} != {k: v for k, v in binding.items() if k != "path"}:
        raise ValueError("Support view manifest/selection/task_id binding changed")
    return support, selection, path


def collection_identity(output, proposed):
    """Keep durable receipt keys across code-only upgrades; freeze every protocol field.

    Called under the collector's identity lock. Runtime changes are disclosed in
    separate immutable records, never used to invalidate or reissue old calls.
    """
    output = Path(output)
    path = output / "identity.json"
    if path.exists():
        old = read(path)
        protocol = lambda value: {k: v for k, v in value.items() if k not in {"code_sha256", "dependencies"}}
        if protocol(old) != protocol(proposed):
            raise ValueError("Refusing different frozen artifact: collection identity changed; source/config/limiter/views must match")
        if old != proposed:
            runtime = {"identity_sha256": digest(old), "code_sha256": proposed["code_sha256"],
                       "dependencies": proposed["dependencies"]}
            frozen_write(output / "resume_versions" / (digest(runtime) + ".json"), runtime)
        return old
    frozen_write(path, proposed)
    return proposed


def copy_bundle(source, target):
    """Private atomic copy: edits to a view cannot mutate the parent evidence."""
    expected = artifact_hashes(source)
    if target.exists():
        if artifact_hashes(target) != expected:
            raise ValueError("Collection view evidence changed")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=".view-", dir=target.parent))
    shutil.copytree(source, partial, dirs_exist_ok=True)
    if artifact_hashes(partial) != expected:
        raise ValueError("Collection view evidence changed during copy")
    os.rename(partial, target)
    fsync_dir(target.parent)


def materialize_views(root, manifest, freeze, bindings, *, resolver=None):
    """Normal verified publications bound to the supplied support, without sampling.

    Private evidence copies keep the generic verified loader (and its path escape
    checks) unchanged. Every receipt is identical to its largest-cell receipt.
    """
    root = Path(root).resolve()
    source, _, _ = load_verified_collection(root, manifest, freeze)
    parent_identity = read(root / "identity.json")
    for binding in bindings:
        support, small, path = check_support(binding, manifest, freeze, resolver=resolver)
        by_task = {r["task_id"]: r for r in source["verified"]}
        ids = [r["task_id"] for r in support["verified"]]
        if not set(ids) <= by_task.keys():
            raise ValueError("Collection view task_id mismatch")
        receipts = [by_task[tid] for tid in ids]
        attempts = [r for r in source["attempts"] if r["task_id"] in set(ids)]
        target = root / "subsets" / f"k{binding['K']}"
        identity = {"method": parent_identity["method"], "selection": small,
                    "mode": "support_task_ids_v1", "attempts_per_candidate": 3,
                    "support_view": support_reference(binding, target, resolver=resolver),
                    "parent_collection": {"path": "../..", "manifest_sha256": digest(source),
                                          "identity_sha256": digest(parent_identity)}}
        value = {"schema_version": 1, "status": "complete", "K": len(receipts),
                 "K_counts": "trusted_executed_trajectories", "identity_sha256": digest(identity),
                 "selection_sha256": digest(small), "trusted_trajectories": len(receipts),
                 "unique_instances": len(set(ids)), "task_groups": len({r['group_id'] for r in receipts}),
                 "attempted_unique_instances": len({r['task_id'] for r in attempts}),
                 "attempted_task_groups": len({r['group_id'] for r in attempts}),
                 "attempts": attempts, "verified": receipts,
                 "parent_seed_ids": [p for r in receipts for p in r.get("parent_seed_ids", [])],
                 "source": "support_task_ids_view", "support_view": identity["support_view"],
                 "parent_collection": identity["parent_collection"], "new_teacher_calls": 0,
                 "new_environment_steps": 0, "training_exposure": 0,
                 "accounting": "View of largest-cell receipts and costs; no new calls/executions. Do not sum views."}
        with locked(target / ".view.lock"):
            frozen_write(target / "identity.json", identity)
            for row in attempts:
                relative = Path(row["episode_dir"]).parent
                copy_bundle(root / relative, target / relative)
            frozen_write(target / "selection.json", small)
            frozen_write(target / "provenance.json", {"parent": read(root / "provenance.json"), "details": identity})
            frozen_write(target / "manifest.json", value)
            loaded, _, _ = load_verified_collection(target, manifest, freeze)
            if {r["task_id"] for r in loaded["verified"]} != set(ids):
                raise ValueError("Collection view task_id mismatch")


def validate_shard(shard_index, shard_count):
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("Positive shard count required")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("Shard index must be in [0, shard_count)")

