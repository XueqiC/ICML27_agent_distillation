"""Stage 3b input identity and accounting, using Stage 3a stage transactions."""
import json
import os
from pathlib import Path

from baselines.common import REGISTERED_K
from baselines.common.explore import training_tasks
from baselines.eto.pipeline import update_disclosure as exploration_disclosure
from sed.manifest import frozen_write, load_frozen
from sed.provenance import code_hashes, write_provenance
from sed.seed_evidence import artifact_hashes, checked_episode, load_verified_collection
from sed.storage import digest, write_json


def batch_invariant():
    if os.environ.get("BATCH_INVARIANT", "1") != "1":
        raise ValueError("Stage 3b requires BATCH_INVARIANT=1")
    os.environ["BATCH_INVARIANT"] = "1"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"


def prepare(args, config, *, method, recipe):
    output = Path(args.output).resolve()
    for name in ("raw", "verified"):
        if output.is_relative_to(Path(config["paths"][name]).resolve()):
            raise ValueError("Baseline output must be outside seed storage")
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, selection, root = load_verified_collection(args.verified, manifest, freeze)
    if selection["K"] not in REGISTERED_K:
        raise ValueError("Registered few-shot K is " + "/".join(map(str, REGISTERED_K)))
    parents = {r["task_id"]: [checked_episode(root, r)["episode_id"]] for r in collection["verified"]}
    ids = list(parents)
    listing = {"name": method + "_seed_train", "task_ids": ids, "parent_seed_ids": parents,
        "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze), "list_sha256": digest(ids)}
    training_tasks(listing, manifest, freeze)
    identity = {"method": method, "recipe": recipe, "K": collection["K"], "support_seed": selection["seed"],
        "unique_instances": collection["unique_instances"], "task_groups": collection["task_groups"],
        "verified": str(Path(args.verified).resolve()), "collection_sha256": digest(collection),
        "snapshot": str(Path(args.snapshot).resolve()), "snapshot_artifact_sha256": artifact_hashes(args.snapshot),
        "config_sha256": digest(config), "gpus": args.gpus, "tasks_sha256": digest(listing), "batch_invariant": True,
        "code_sha256": code_hashes(Path(__file__).resolve().parents[2])}
    from baselines.common.dose_control import bind_init
    bind_init(args, config, identity)
    output.mkdir(parents=True, exist_ok=True)
    frozen_write(output / "identity.json", identity)
    frozen_write(output / "tasks.json", listing)
    if not (output / "provenance.json").exists():
        write_provenance(output, command=method + "_pipeline", config={**config, method: identity},
            inputs=[args.config, Path(args.verified) / "manifest.json", config["paths"]["manifest"], config["paths"]["split_freeze"]])
    return identity


def update_disclosure(output, identity):
    """Include logical exposure in failed/retried training, never just final models."""
    output = Path(output)
    exploration_disclosure(output, identity)
    path = output / "disclosure.json"
    row = json.loads(path.read_text())
    trainings = []
    for attempt in sorted((output / "stages").glob("train-*/attempt-*")):
        updates = attempt / "output/updates.jsonl"
        last, partial = {}, False
        if updates.exists():
            for line in updates.read_text().splitlines():
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    partial = True
        result = attempt / "output/result.json"
        record = json.loads(result.read_text()) if result.exists() else {}
        trainings.append({"path": str(attempt.relative_to(output)), "complete": result.exists(),
            "examples_seen": last.get("examples_seen", 0),
            "supervised_tokens_seen": last.get("supervised_tokens_seen", last.get("cumulative_supervised_tokens", 0)),
            "logical_forward_tokens": record.get("logical_forward_tokens"),
            "scoring": record.get("scoring", last.get("scoring")),
            "flex_attention": record.get("flex_attention", last.get("flex_attention")),
            "physical_policy_forward_tokens": record.get("physical_policy_forward_tokens", last.get("physical_policy_forward_tokens")),
            "physical_reference_forward_tokens": record.get("physical_reference_forward_tokens", last.get("physical_reference_forward_tokens")),
            "unknown_exposure_possible": partial or not result.exists(),
            "accounting": "Durable completed updates only; an interrupted update can have unknown additional compute."})
    row.update(training_attempts=trainings, batch_invariant=True,
        iterations=[json.loads(p.read_text()) for p in sorted(output.glob("iteration-*.json"))])
    if identity.get("method") == "kang":
        row.update(source="trusted_teacher_seed_sft", expansion=False,
            training_seed=identity["recipe"]["training_seed"])
    row["totals"].update(examples_seen=sum(t["examples_seen"] for t in trainings),
        supervised_tokens_seen=sum(t["supervised_tokens_seen"] for t in trainings), teacher_calls=0)
    row["token_accounting"] = "Every raw student candidate/call and usage lives in the exploration attempt; summary totals include all returned completions. Seed teacher usage/model/timestamps remain bound in original receipts."
    write_json(path, row)
