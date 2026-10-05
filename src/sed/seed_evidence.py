"""Validate environment evidence separately from model-visible training fields."""
import json
from collections import Counter
from pathlib import Path

from sed.manifest import authorize_task, resolve_task, frozen_write
from sed.seed_selection import select_seeds, validate_selection
from sed.storage import BlobStore, file_hash, digest, locked


def environment_won(directory, episode):
    if not (episode.get("complete") is True and episode.get("status") == "ok"
            and episode.get("success") is True and episode.get("steps")):
        return False
    final = episode["steps"][-1]
    verification = BlobStore(Path(directory) / "blobs").get(final["verification_ref"])
    return (verification.get("fact_source") == "real_environment"
            and verification.get("executable") is True
            and verification.get("audit", {}).get("won") is True
            and final.get("environment_steps") == 1)


def artifact_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and not p.name.endswith(".lock")}


def checked_episode(root, receipt):
    directory = (Path(root) / receipt["episode_dir"]).resolve()
    if not directory.is_relative_to(Path(root).resolve()):
        raise ValueError("Episode path escapes collection")
    if artifact_hashes(directory) != receipt["episode_hashes"]:
        raise ValueError("Episode evidence changed")
    episode = json.loads((directory / "episode.json").read_text())
    if not environment_won(directory, episode):
        raise ValueError("Seed lacks a real environment win")
    return episode


def receipt_path(root, receipt):
    path = (Path(root) / receipt["episode_dir"]).resolve().parent / "receipt.json"
    if not path.is_relative_to(Path(root).resolve() / "attempts"):
        raise ValueError("Receipt path escapes collection attempts")
    return path


def _subset_values(root, source, large, small):
    if (small["seed"] != large["seed"] or small["K"] >= large["K"]
            or any(small["selected"][kind] != large["selected"][kind][:small["quotas"][kind]]
                   for kind in small["type_order"])):
        raise ValueError("Selections are not nested per-type prefixes")
    parents = {r["task_id"] for rows in small["selected"].values() for r in rows}
    verified = [r for r in source["verified"] if r["selection_parent_task_id"] in parents]
    attempts = [r for r in source["attempts"] if r["selection_parent_task_id"] in parents]
    if len(verified) != small["K"] or {r["selection_parent_task_id"] for r in verified} != parents:
        raise ValueError("Source lacks verified evidence for every subset slot")
    reference = {"path": "../..", "manifest_sha256": digest(source),
                 "selection_sha256": digest(large), "K": large["K"]}
    identity = {"selection": small, "mode": "reuse_verified_parent_slots_v1",
                "source_collection": reference}
    value = {"schema_version": 1, "status": "complete", "K": small["K"],
             "K_counts": "trusted_executed_trajectories", "identity_sha256": digest(identity),
             "selection_sha256": digest(small), "trusted_trajectories": len(verified),
             "unique_instances": len({r["task_id"] for r in verified}),
             "task_groups": len({r["group_id"] for r in verified}),
             "attempted_unique_instances": len({r["task_id"] for r in attempts}),
             "attempted_task_groups": len({r["group_id"] for r in attempts}),
             "attempts": attempts, "verified": verified, "parent_seed_ids": [],
             "source": "reused_verified_nested_collection", "source_collection": reference,
             "receipt_refs": [{"path": "../../" + str(receipt_path(root, r).relative_to(Path(root).resolve())),
                               "sha256": file_hash(receipt_path(root, r))} for r in attempts],
             "reused_trajectories": len(verified), "new_teacher_calls": 0,
             "new_environment_steps": 0, "training_exposure": 0,
             "accounting": "Slot attempts are a view of source costs, not additional calls. "
                           "The source manifest retains ALL largest-K attempts and costs; "
                           "do not sum these views as independent collections."}
    return identity, value


def load_verified_collection(root, manifest, freeze, *, allow_subset=True):
    """Validate evidence first; callers separately whitelist model-visible fields."""
    root = Path(root).resolve()
    collection = json.loads((root / "manifest.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    validate_selection(selection, manifest, freeze)
    if (collection["status"] != "complete" or collection["selection_sha256"] != digest(selection)
            or len(collection["verified"]) != selection["K"]
            or collection["identity_sha256"] != digest(identity) or identity["selection"] != selection):
        raise ValueError("Collection must contain exactly K verified trajectories")
    if "source_collection" in collection:
        if not allow_subset or root.parent.name != "subsets" or root.name != f"k{selection['K']}":
            raise ValueError("Invalid nested collection location or chained source")
        evidence_root = root.parent.parent
        source, large, _ = load_verified_collection(evidence_root, manifest, freeze, allow_subset=False)
        expected_identity, expected = _subset_values(evidence_root, source, large, selection)
        if identity != expected_identity or collection != expected:
            raise ValueError("Nested collection differs from shared source receipts")
    else:
        evidence_root = root
    candidates = {r["task_id"] for kind in selection["type_order"]
                  for r in selection["selected"][kind] + selection["replacements"][kind]}
    parents = {r["task_id"]: kind for kind in selection["type_order"] for r in selection["selected"][kind]}
    # Every failed/interrupted attempt remains referenced and checked too.
    for receipt in collection["attempts"]:
        if json.loads(receipt_path(evidence_root, receipt).read_text()) != receipt:
            raise ValueError("Stored receipt differs from collection")
        directory = (evidence_root / receipt["episode_dir"]).resolve()
        if (artifact_hashes(directory) != receipt["episode_hashes"]
                or artifact_hashes(directory.parent / "teacher") != receipt["teacher_hashes"]):
            raise ValueError("Attempt evidence changed")
        task = resolve_task(manifest, freeze, receipt["task_id"])
        authorize_task(task, "generation")
        if (task["source_split"] != "train" or receipt["task_id"] not in candidates
                or receipt["group_id"] != task["group_id"] or receipt["task_type"] != task["task_type"]
                or parents.get(receipt["selection_parent_task_id"]) != task["task_type"]):
            raise ValueError("Receipt differs from frozen TRAIN selection")
    groups, slots, counts = set(), set(), Counter()
    for receipt in collection["verified"]:
        task = resolve_task(manifest, freeze, receipt["task_id"])
        if (receipt not in collection["attempts"] or receipt["won"] is not True
                or task["group_id"] in groups or receipt["selection_parent_task_id"] in slots):
            raise ValueError("Seeds must be distinct TRAIN groups and selection slots")
        episode = checked_episode(evidence_root, receipt)
        if (episode["task_id"] != task["task_id"] or episode["game_sha256"] != task["game_sha256"]
                or episode["split"] != "train"):
            raise ValueError("Episode differs from frozen TRAIN task")
        groups.add(task["group_id"])
        slots.add(receipt["selection_parent_task_id"])
        counts[task["task_type"]] += 1
    if (any(counts[kind] != quota for kind, quota in selection["quotas"].items())
            or collection["K"] != selection["K"] or collection["trusted_trajectories"] != len(groups)
            or collection["unique_instances"] != len(groups) or collection["task_groups"] != len(groups)):
        raise ValueError("Verified trajectories differ from stratified quotas/counts")
    return collection, selection, evidence_root


def verified_collection_inputs(root, manifest, freeze):
    """Return provenance inputs for a validated collection or nested subset."""
    root = Path(root)
    collection, _, evidence_root = load_verified_collection(root, manifest, freeze)
    if "source_collection" in collection:
        return [root / "manifest.json", root / "identity.json", root / "selection.json",
                evidence_root / "manifest.json", evidence_root / "provenance.json"]
    return [root / "manifest.json", root / "provenance.json"]


def materialize_subsets(root, manifest, freeze, ks):
    """Write immutable views under largest-K/subsets/kK; never copy trajectories."""
    root = Path(root).resolve()
    with locked(root / ".collection.lock"):
        source, large, _ = load_verified_collection(root, manifest, freeze, allow_subset=False)
        values = []
        for k in sorted(set(ks)):
            small = select_seeds(manifest, freeze, k=k, seed=large["seed"], rule=large["selection_rule"])
            identity, value = _subset_values(root, source, large, small)
            values.append((root / "subsets" / f"k{k}", small, identity, value))
        for output, small, identity, value in values:
            frozen_write(output / "selection.json", small)
            frozen_write(output / "identity.json", identity)
            frozen_write(output / "manifest.json", value)
        return [value for _, _, _, value in values]
