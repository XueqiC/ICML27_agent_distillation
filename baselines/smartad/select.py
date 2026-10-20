"""Base-student scoring and canonical verified publication, without API calls."""
import math
import os
from pathlib import Path
import shutil
import tempfile

from baselines.smartad.collect import load_pool
from baselines.common.collection_views import (bind_support, support_reference, materialize_views,
    check_support, validate_shard, collection_identity as resume_identity)
from baselines.smartad.data import score_episode
from baselines.smartad.paths import PathResolver
from baselines.smartad.teacher import CALLS_PER_TASK, read
from sed.manifest import frozen_write
from sed.provenance import code_hashes, versions, write_provenance
from sed.seed_evidence import artifact_hashes, load_verified_collection, materialize_subsets
from sed.seed_selection import is_full_train
from sed.storage import digest, file_hash, fsync_dir, locked


def copy_evidence(source, target, *, resolver, hashes):
    """Atomic publish; existing evidence must be byte-identical."""
    source, target = resolver.resolve(source, hashes), Path(target)
    if target.exists():
        if hashes != artifact_hashes(target):
            raise ValueError("Published SmartAD evidence changed")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=".smartad-partial-", dir=target.parent))
    shutil.copytree(source, partial, dirs_exist_ok=True)
    os.rename(partial, target)
    fsync_dir(target.parent)


def select(pool_root, snapshot, config, manifest, freeze, output, *, scorer=None, path_map=(),
           shard_index=0, shard_count=1, merge=False):
    validate_shard(shard_index, shard_count)
    resolver = PathResolver(path_map)
    pool_root, output, snapshot = Path(pool_root).resolve(), Path(output).resolve(), Path(snapshot).resolve()
    pool_root = resolver.resolve(pool_root)
    pool, pool_identity, source, selection, source_root = load_pool(pool_root, manifest, freeze, resolver=resolver)
    if "parent_pool" in pool:
        raise ValueError("Select the largest candidate pool once; its task views inherit the selections")
    bindings = pool_identity.get("subset_views", [])
    for binding in bindings:
        check_support(binding, manifest, freeze, resolver=resolver)
    if not snapshot.is_dir() or any((snapshot / name).exists() for name in ("adapter_config.json", "training_state.json")):
        raise ValueError("SmartAD selection requires the initial base student snapshot")
    if (output == pool_root or pool_root.is_relative_to(output) or output.is_relative_to(pool_root / "work")
            or any(output.is_relative_to(Path(config["paths"][name]).resolve()) for name in ("raw", "verified"))
            or output.is_relative_to(source_root)):
        raise ValueError("Selection output must be separate from candidate/source evidence")
    identity = {"method": "smartad", "selection": selection, "pool_root": str(resolver.published(pool_root)),
        "pool_sha256": digest(pool), "pool_identity_sha256": digest(pool_identity),
        "snapshot": str(snapshot), "snapshot_artifact_sha256": artifact_hashes(snapshot),
        "config_sha256": digest(config),
        "student": config["models"]["student"], "dtype": "bfloat16", "packing": "turn",
        "nll": "mean_turn(mean_token(NLL)); exact H0 OFF prefix; no truncation",
        "code_sha256": code_hashes(Path(__file__).resolve().parents[2]), "dependencies": versions(),
        # This field belongs to the shared seed publication schema. Ten SmartAD
        # candidate slots are separate from its <=3 inherited seed attempts.
        "attempts_per_candidate": 3, "smartad_candidates_per_task": 10}
    if selection.get("selection_rule") == "stratified_nested_v2":
        support_path = resolver.resolve(pool_identity["source_verified"])
        identity["support_view"] = support_reference(bind_support(support_path, source, selection), output / "collection")
    with locked(output / ".pipeline.lock"):
        identity = resume_identity(output, identity)
        if not (output / "provenance.json").exists():
            write_provenance(output, command="smartad_select", config=config,
                inputs=[pool_root / "candidates.json", pool_root / "provenance.json"],
                details={**identity, "path_map": resolver.disclosure})
        elif read(output / "provenance.json")["details"].get("path_map", []) != resolver.disclosure:
            raise ValueError("SmartAD selection relocation changed")
    candidates = [(task, candidate) for task in pool["tasks"] for candidate in task["candidates"]]
    candidate_list_sha = digest([[task["task_id"], candidate] for task, candidate in candidates])
    def score_path(task, candidate):
        return output / "scores" / digest(task["task_id"])[:20] / (candidate["candidate_id"] + ".json")

    if not merge:
        for index in range(shard_index, len(candidates), shard_count):
            task, candidate = candidates[index]
            path = score_path(task, candidate)
            with locked(output / "score_locks" / (digest([task["task_id"], candidate["candidate_id"]]) + ".lock")):
                if path.exists():
                    checked_score(path, identity, candidate)
                    continue
                episode_dir = resolver.resolve(candidate["episode_dir"], candidate["episode_hashes"])
                if candidate["correct"] and scorer is None:
                    scorer = gpu_scorer(snapshot)
                score = scorer(read(episode_dir / "episode.json")) if candidate["correct"] else None
                record = {"candidate_id": candidate["candidate_id"], "binding_sha256": digest([identity, candidate]),
                    "episode_hashes": candidate["episode_hashes"], "correct": candidate["correct"],
                    "score": score, "rejection": candidate["rejection"]}
                validate_score(record, identity, candidate)
                frozen_write(path, record)
        if shard_count > 1:
            files = {str(score_path(*candidates[i]).relative_to(output)): file_hash(score_path(*candidates[i]))
                     for i in range(shard_index, len(candidates), shard_count)}
            receipt = {"identity_sha256": digest(identity), "candidate_list_sha256": candidate_list_sha,
                       "shard_index": shard_index, "shard_count": shard_count, "scores_sha256": files}
            with locked(output / ".pipeline.lock"):
                frozen_write(output / "score_shards" / f"{shard_count}-{shard_index}.json", receipt)
            return output
    with locked(output / ".pipeline.lock"):
        if merge and shard_count > 1:
            for index in range(shard_count):
                path = output / "score_shards" / f"{shard_count}-{index}.json"
                if not path.exists():
                    raise ValueError("Missing SmartAD score shard; merge requires every shard")
                files = {str(score_path(*candidates[i]).relative_to(output)): file_hash(score_path(*candidates[i]))
                         for i in range(index, len(candidates), shard_count)}
                expected = {"identity_sha256": digest(identity), "candidate_list_sha256": candidate_list_sha,
                            "shard_index": index, "shard_count": shard_count, "scores_sha256": files}
                if read(path) != expected:
                    raise ValueError("SmartAD score shard hash/binding changed")
        selected = []
        for task in pool["tasks"]:
            records = []
            for candidate in task["candidates"]:
                path = score_path(task, candidate)
                if not path.exists():
                    raise ValueError("Missing SmartAD candidate score; cannot publish partial selection")
                records.append(checked_score(path, identity, candidate))
            eligible = [r for r in records if r["correct"]]
            chosen = min(eligible, key=lambda r: (r["score"]["nll"], r["candidate_id"])) if eligible else None
            selected.append({"task_id": task["task_id"], "scores": records,
                "chosen_id": chosen["candidate_id"] if chosen else "trusted-fallback",
                "fallback": chosen is None, "reuse": task["reuse"], "reuse_reason": task["reuse_reason"]})
        result = {"identity_sha256": digest(identity), "pool_sha256": digest(pool),
            "snapshot_artifact_sha256": identity["snapshot_artifact_sha256"],
            "tasks": selected, "fallback_count": sum(r["fallback"] for r in selected),
            "scores_sha256": artifact_hashes(output / "scores")}
        frozen_write(output / "smartad_selection.json", result)
        root = output / "collection"
        work_hashes = {}
        for path, sha in pool["work_hashes"].items():
            key, _, relative = path.partition("/")
            work_hashes.setdefault(key, {})[relative] = sha
        receipts = []
        for task, decision in zip(pool["tasks"], selected):
            original = task["source_receipt"]
            key = digest(task["task_id"])[:20]
            destination = root / "attempts" / (key + "-1")
            chosen = next((c for c in task["candidates"] if c["candidate_id"] == decision["chosen_id"]), None)
            episode_dir = Path(chosen["episode_dir"]) if chosen else source_root / original["episode_dir"]
            copy_evidence(episode_dir, destination / "episode", resolver=resolver,
                          hashes=(chosen or original)["episode_hashes"])
            copy_evidence(pool_root / "work" / key, destination / "teacher/candidates",
                          resolver=resolver, hashes=work_hashes[key])
            copy_evidence((source_root / original["episode_dir"]).parent / "teacher",
                          destination / "teacher/trusted_source_teacher", resolver=resolver,
                          hashes=original["teacher_hashes"])
            copy_evidence(source_root / original["episode_dir"], destination / "teacher/trusted_source_episode",
                          resolver=resolver, hashes=original["episode_hashes"])
            frozen_write(destination / "teacher/selection.json", decision)
            episode = read(destination / "episode/episode.json")
            usages = [c["usage"] for c in task["candidates"] if not c["reused"]]
            usage = {key: sum(u[key] for u in usages) for key in
                     ("unknown_usage_calls", "transport_failures", "rejections", "interrupted_requests", "ledger_rows")}
            usage["known_billed_tokens"] = {key: sum(u["known_billed_tokens"][key] for u in usages)
                for key in ("input_tokens", "output_tokens", "reasoning_tokens")}
            receipt = {**original, "attempt": 1, "source": "smartad_trusted_fallback" if decision["fallback"] else "smartad_selected",
                "parent_seed_ids": task["candidates"][0]["parent_seed_ids"],
                "episode_dir": f"attempts/{key}-1/episode", "episode_hashes": artifact_hashes(destination / "episode"),
                "teacher_hashes": artifact_hashes(destination / "teacher"), "metrics": episode["metrics"],
                "smartad_selection": decision, "fallback": decision["fallback"], "usage": usage,
                "source_usage_reused": original["usage"],
                "api_attempts": sum(c["api_attempts"] for c in task["candidates"])}
            frozen_write(destination / "receipt.json", receipt)
            receipts.append(receipt)
        all_receipts = receipts
        if is_full_train(selection):
            by_task = {r["task_id"]: r for r in receipts}
            all_receipts = []
            for original in source["attempts"]:
                if original["task_id"] in by_task:
                    if original["won"]:
                        all_receipts.append(by_task[original["task_id"]])
                    continue
                origin = source_root / original["episode_dir"]
                destination = root / original["episode_dir"]
                for name in ("episode", "teacher"):
                    copy_evidence(origin.parent / name, destination.parent / name,
                                  resolver=resolver, hashes=original[name + "_hashes"])
                frozen_write(destination.parent / "receipt.json", original)
                all_receipts.append(original)
        collection_identity = {**identity, "smartad_selection_sha256": digest(result)}
        frozen_write(root / "identity.json", collection_identity)
        frozen_write(root / "selection.json", selection)
        frozen_write(root / "smartad_selection.json", result)
        frozen_write(root / "provenance.json", read(output / "provenance.json"))
        new = [c for task in pool["tasks"] for c in task["candidates"] if not c["reused"]]
        collection = {"schema_version": 1, "status": "complete", "K": source["K"],
            "K_counts": "trusted_executed_trajectories", "identity_sha256": digest(collection_identity),
            "selection_sha256": digest(selection), "trusted_trajectories": len(receipts),
            "unique_instances": source["unique_instances"], "task_groups": source["task_groups"],
            "attempted_unique_instances": len({r["task_id"] for r in all_receipts}),
            "attempted_task_groups": len({r["group_id"] for r in all_receipts}),
            "attempts": all_receipts, "verified": receipts,
            "parent_seed_ids": [p for r in receipts for p in r["parent_seed_ids"]],
            "source": "smartad_support_tasks_only", "training_exposure": 0, "new_task_instances": 0,
            "fallback_count": result["fallback_count"], "reused_trusted": sum(t["reuse"] for t in pool["tasks"]),
            "candidate_slots": 10 * len(receipts), "new_verified_trajectories": sum(c["correct"] for c in new),
            "api_attempts": pool["api_attempts"], "per_cell_call_cap": CALLS_PER_TASK * len(receipts),
            "baseline_transport": pool_identity["baseline_transport"], "sampling_fallback": pool["sampling_fallback"],
            "new_environment_steps": sum((c["metrics"] or {}).get("environment_steps", 0) for c in new),
            "known_billed_tokens": {key: sum(c["usage"]["known_billed_tokens"][key] for c in new)
                                   for key in ("input_tokens", "output_tokens", "reasoning_tokens")},
            "unknown_usage_calls": sum(c["usage"]["unknown_usage_calls"] for c in new),
            "transport_failures": sum(c["usage"]["transport_failures"] for c in new),
            "accounting": "Canonical receipt is a publication slot, not an execution. Ten candidates and all failures "
                "remain in hashed evidence. Seed reuse/fallback costs are inherited, never additional calls/tasks."}
        if is_full_train(selection):
            collection.update(selection_rule=selection["selection_rule"], candidates=selection["candidates"],
                              failed_candidates=source["failed_candidates"])
        frozen_write(root / "manifest.json", collection)
        load_verified_collection(root, manifest, freeze)
        if pool_identity["subset_ks"]:
            materialize_subsets(root, manifest, freeze, pool_identity["subset_ks"])
        if bindings:
            materialize_views(root, manifest, freeze, bindings, resolver=resolver)
        frozen_write(output / "result.json", {"status": "complete", "verified": str(root),
                                             "fallback_count": result["fallback_count"],
                                             "path_map": resolver.disclosure})
        return root


def gpu_scorer(snapshot):
    import torch
    from train.dataset import load_tokenizer
    from train.modeling import load_student
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("SmartAD selection requires one allocated GPU per shard")
    tokenizer = load_tokenizer(snapshot)
    model = load_student(snapshot, dtype="bfloat16", device="cuda")
    model.set_attn_implementation("sdpa")
    model.requires_grad_(False)
    model.eval()
    return lambda episode: score_episode(model, tokenizer, episode)


def validate_score(record, identity, candidate):
    expected = {"candidate_id": candidate["candidate_id"], "binding_sha256": digest([identity, candidate]),
                "episode_hashes": candidate["episode_hashes"], "correct": candidate["correct"],
                "rejection": candidate["rejection"]}
    if {k: v for k, v in record.items() if k != "score"} != expected:
        raise ValueError("SmartAD score binding changed")
    score = record.get("score")
    if candidate["correct"]:
        if not isinstance(score, dict) or not isinstance(score.get("nll"), (int, float)) or not math.isfinite(score["nll"]) or score["nll"] < 0:
            raise ValueError("Invalid candidate NLL")
    elif score is not None:
        raise ValueError("Incorrect candidate has a score")


def checked_score(path, identity, candidate):
    record = read(path)
    validate_score(record, identity, candidate)
    return record


def validate_selection(verified, manifest, freeze, snapshot):
    """Bind training, including nested views, to the same initial student."""
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    identity = read(root / "identity.json")
    if identity.get("mode") == "support_task_ids_v1":
        reference = identity["parent_collection"]
        parent_root = (root / reference["path"]).resolve()
        parent, _, _ = load_verified_collection(parent_root, manifest, freeze)
        if (digest(parent) != reference["manifest_sha256"]
                or digest(read(parent_root / "identity.json")) != reference["identity_sha256"]
                or any(r not in parent["verified"] for r in collection["verified"])):
            raise ValueError("SmartAD view parent/receipt binding changed")
        binding = {**identity["support_view"], "path": str((root / identity["support_view"]["path"]).resolve())}
        support, _, _ = check_support(binding, manifest, freeze)
        if {r["task_id"] for r in collection["verified"]} != {r["task_id"] for r in support["verified"]}:
            raise ValueError("SmartAD view task_id mismatch")
        return validate_selection(parent_root, manifest, freeze, snapshot)
    identity, selection = read(root / "identity.json"), read(root / "smartad_selection.json")
    if (identity.get("method") != "smartad" or identity["smartad_selection_sha256"] != digest(selection)
            or selection["identity_sha256"] != digest({k: v for k, v in identity.items() if k != "smartad_selection_sha256"})
            or identity["snapshot_artifact_sha256"] != artifact_hashes(snapshot)):
        raise ValueError("SmartAD selection/student binding changed")
    return selection
