"""Base-student scoring and canonical verified publication, without API calls."""
import math
import os
from pathlib import Path
import shutil
import tempfile

from baselines.smartad.collect import load_pool
from baselines.smartad.data import score_episode
from baselines.smartad.teacher import CALLS_PER_TASK, read
from sed.manifest import frozen_write
from sed.provenance import code_hashes, versions, write_provenance
from sed.seed_evidence import artifact_hashes, load_verified_collection, materialize_subsets
from sed.seed_selection import is_full_train
from sed.storage import digest, fsync_dir, locked


def copy_evidence(source, target):
    """Atomic publish; existing evidence must be byte-identical."""
    source, target = Path(source), Path(target)
    if target.exists():
        if artifact_hashes(source) != artifact_hashes(target):
            raise ValueError("Published SmartAD evidence changed")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=".smartad-partial-", dir=target.parent))
    shutil.copytree(source, partial, dirs_exist_ok=True)
    os.rename(partial, target)
    fsync_dir(target.parent)


def select(pool_root, snapshot, config, manifest, freeze, output, *, scorer=None):
    pool_root, output, snapshot = Path(pool_root).resolve(), Path(output).resolve(), Path(snapshot).resolve()
    pool, pool_identity, source, selection, source_root = load_pool(pool_root, manifest, freeze)
    if not snapshot.is_dir() or any((snapshot / name).exists() for name in ("adapter_config.json", "training_state.json")):
        raise ValueError("SmartAD selection requires the initial base student snapshot")
    if (output == pool_root or pool_root.is_relative_to(output) or output.is_relative_to(pool_root / "work")
            or any(output.is_relative_to(Path(config["paths"][name]).resolve()) for name in ("raw", "verified"))
            or output.is_relative_to(source_root)):
        raise ValueError("Selection output must be separate from candidate/source evidence")
    identity = {"method": "smartad", "selection": selection, "pool_root": str(pool_root),
        "pool_sha256": digest(pool), "pool_identity_sha256": digest(pool_identity),
        "snapshot": str(snapshot), "snapshot_artifact_sha256": artifact_hashes(snapshot),
        "config_sha256": digest(config),
        "student": config["models"]["student"], "dtype": "bfloat16", "packing": "turn",
        "nll": "mean_turn(mean_token(NLL)); exact H0 OFF prefix; no truncation",
        "code_sha256": code_hashes(Path(__file__).resolve().parents[2]), "dependencies": versions(),
        # This field belongs to the shared seed publication schema. Ten SmartAD
        # candidate slots are separate from its <=3 inherited seed attempts.
        "attempts_per_candidate": 3, "smartad_candidates_per_task": 10}
    with locked(output / ".pipeline.lock"):
        frozen_write(output / "identity.json", identity)
        if not (output / "provenance.json").exists():
            write_provenance(output, command="smartad_select", config=config,
                inputs=[pool_root / "candidates.json", pool_root / "provenance.json"],
                details=identity)
        if scorer is None:
            import torch
            from train.dataset import load_tokenizer
            from train.modeling import load_student
            if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
                raise RuntimeError("SmartAD selection requires one allocated GPU")
            tokenizer = load_tokenizer(snapshot)
            model = load_student(snapshot, dtype="bfloat16", device="cuda")
            model.set_attn_implementation("sdpa")
            model.requires_grad_(False)
            model.eval()
            scorer = lambda episode: score_episode(model, tokenizer, episode)
        selected = []
        for task in pool["tasks"]:
            key = digest(task["task_id"])[:20]
            records = []
            for candidate in task["candidates"]:
                binding = digest([identity, candidate])
                score_path = output / "scores" / key / (candidate["candidate_id"] + ".json")
                if score_path.exists():
                    record = read(score_path)
                    if record["binding_sha256"] != binding:
                        raise ValueError("SmartAD score binding changed")
                else:
                    score = scorer(read(Path(candidate["episode_dir"]) / "episode.json")) if candidate["correct"] else None
                    if score is not None and (not math.isfinite(score["nll"]) or score["nll"] < 0):
                        raise ValueError("Invalid candidate NLL")
                    record = {"candidate_id": candidate["candidate_id"], "binding_sha256": binding,
                        "episode_hashes": candidate["episode_hashes"], "correct": candidate["correct"],
                        "score": score, "rejection": candidate["rejection"]}
                    frozen_write(score_path, record)
                records.append(record)
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
        receipts = []
        for task, decision in zip(pool["tasks"], selected):
            original = task["source_receipt"]
            key = digest(task["task_id"])[:20]
            destination = root / "attempts" / (key + "-1")
            chosen = next((c for c in task["candidates"] if c["candidate_id"] == decision["chosen_id"]), None)
            episode_dir = Path(chosen["episode_dir"]) if chosen else source_root / original["episode_dir"]
            copy_evidence(episode_dir, destination / "episode")
            copy_evidence(pool_root / "work" / key, destination / "teacher/candidates")
            copy_evidence((source_root / original["episode_dir"]).parent / "teacher",
                          destination / "teacher/trusted_source_teacher")
            copy_evidence(source_root / original["episode_dir"], destination / "teacher/trusted_source_episode")
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
                    copy_evidence(origin.parent / name, destination.parent / name)
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
        frozen_write(output / "result.json", {"status": "complete", "verified": str(root),
                                             "fallback_count": result["fallback_count"]})
        return root


def validate_selection(verified, manifest, freeze, snapshot):
    """Bind training, including nested views, to the same initial student."""
    _, _, root = load_verified_collection(verified, manifest, freeze)
    identity, selection = read(root / "identity.json"), read(root / "smartad_selection.json")
    if (identity.get("method") != "smartad" or identity["smartad_selection_sha256"] != digest(selection)
            or selection["identity_sha256"] != digest({k: v for k, v in identity.items() if k != "smartad_selection_sha256"})
            or identity["snapshot_artifact_sha256"] != artifact_hashes(snapshot)):
        raise ValueError("SmartAD selection/student binding changed")
    return selection
