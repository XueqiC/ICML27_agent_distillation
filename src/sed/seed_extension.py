"""Read-only inheritance of a completed seed collection and its receipt identity."""

import json
import os
from pathlib import Path

from sed.seed_selection import validate_extension_selection
from sed.storage import atomic_write, digest, file_hash, write_json


def extension_source(source, output, selection, manifest, freeze, *, attempts, config):
    from sed.seed_evidence import artifact_hashes, load_verified_collection

    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Extension output and source must be separate trees")
    collection, small, _ = load_verified_collection(source, manifest, freeze, allow_subset=False)
    identity = json.loads((source / "identity.json").read_text())
    if "extension_source" in identity:
        raise ValueError("Extend from the original completed collection, not an extension")
    validate_extension_selection(small, selection)
    if attempts != identity["attempts_per_candidate"]:
        raise ValueError("Extension must preserve attempts per candidate")
    # Host paths, rate limits and code versions can change; sampling and the
    # environment protocol cannot. Both effective configs remain in provenance.
    old = identity["config"]
    for key in ("requested_name", "provider_model_id", "sampling", "access"):
        if old["models"]["teacher"][key] != config["models"]["teacher"][key]:
            raise ValueError("Extension teacher identity/sampling changed")
    if old["models"]["student"] != config["models"]["student"]:
        raise ValueError("Extension student identity changed")
    portable = lambda c: {k: v for k, v in c["environment"].items() if k not in {"data_root", "python"}}
    if portable(old) != portable(config):
        raise ValueError("Extension environment protocol changed")
    expected = {str(Path(r["episode_dir"]).parent / "receipt.json") for r in collection["attempts"]}
    actual = {str(p.relative_to(source)) for p in (source / "attempts").glob("*/receipt.json")}
    started = {str(p.parent.relative_to(source) / "receipt.json")
               for p in (source / "attempts").glob("*/started.json")}
    directories = {str(p.relative_to(source) / "receipt.json")
                   for p in (source / "attempts").iterdir() if p.is_dir()}
    if actual != expected or directories != expected or not started <= expected:
        raise ValueError("Source has unaccounted attempts; finish it before extending")
    # These are mandatory identity anchors for the Stage 2f K20 extension.
    if small["K"] == 20:
        for k in (5, 10):
            load_verified_collection(source / "subsets" / f"k{k}", manifest, freeze)
    reference = {"path": os.path.relpath(source, output), "K": small["K"],
                 "manifest_sha256": digest(collection), "identity_sha256": digest(identity),
                 "selection_sha256": digest(small), "tree_sha256": digest(artifact_hashes(source))}
    return reference, collection


def copy_inherited_attempts(source, output, reference):
    """Atomic copies, no hardlinks to the read-only original; resumable bootstrap."""
    from sed.seed_evidence import artifact_hashes

    source, output = Path(source), Path(output)
    marker = output / "inherited_ready.json"
    if marker.exists() and json.loads(marker.read_text()) != reference:
        raise ValueError("Inherited source identity changed")
    for relative, sha in artifact_hashes(source / "attempts").items():
        target = output / "attempts" / relative
        if target.exists():
            if file_hash(target) != sha:
                raise ValueError("Inherited attempt evidence changed")
        elif marker.exists():
            raise ValueError("Inherited attempt evidence missing")
        else:
            atomic_write(target, (source / "attempts" / relative).read_bytes(), exclusive=True)
            if file_hash(target) != sha:
                raise ValueError("Source evidence changed during copy")
    if not marker.exists():
        write_json(marker, reference, exclusive=True)


def check_inherited_views(root, source, large, manifest, freeze, subset_builder):
    """Reject changes to old receipts before publishing or loading any new view."""
    from sed.seed_evidence import load_verified_collection
    from sed.seed_selection import select_seeds

    identity = json.loads((Path(root) / "identity.json").read_text())
    reference = identity.get("extension_source")
    if reference is None:
        return
    origin = (Path(root) / reference["path"]).resolve()
    current, inherited = extension_source(origin, root, large, manifest, freeze,
        attempts=identity["attempts_per_candidate"], config=identity["config"])
    if current != reference or source.get("extension_source") != reference:
        raise ValueError("Extension source identity/evidence changed")
    # Full receipts include task IDs, attempt numbers, episode paths and every
    # trajectory hash. Checking them also preserves rejection and cost records.
    if source["attempts"][:len(inherited["attempts"])] != inherited["attempts"]:
        raise ValueError("Extension changed inherited attempts")
    ks = (5, 10, 20) if reference["K"] == 20 else (reference["K"],)
    for k in ks:
        small = select_seeds(manifest, freeze, k=k, seed=large["seed"])
        _, view = subset_builder(root, source, large, small)
        baseline = inherited if k == reference["K"] else load_verified_collection(
            origin / "subsets" / f"k{k}", manifest, freeze)[0]
        if view["verified"] != baseline["verified"] or view["attempts"] != baseline["attempts"]:
            raise ValueError(f"Inherited K={k} receipt identity mismatch")
    if large["K"] > 5:
        _, five = subset_builder(root, source, large, select_seeds(manifest, freeze, k=5, seed=large["seed"]))
        for k in (1, 2):
            _, view = subset_builder(root, source, large, select_seeds(manifest, freeze, k=k, seed=large["seed"]))
            expected = [next(r for r in five["verified"] if r["task_type"] == kind)
                        for kind in large["type_order"][:k]]
            if view["verified"] != expected:
                raise ValueError(f"K={k} differs from the first K=5 type seeds")
