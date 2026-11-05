"""Build V0 D replay from theta0's original executed support, offline on CPU.

The small trajectory renderer/row construction is adapted from
ICML27_wt_repair_eval tools/build_v2_dataset.py and repair/supervision.py
(V2 producer 7748da5). No producer runtime, teacher, or environment is imported.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

from repair import v2_train as v2
from repair.artifacts import provenance, read_json
from repair.v2_inputs import InputFiles, Relocation, ValidationError, require, training_view_inputs
from sed.config import load_config
from sed.manifest import load_frozen, resolve_task
from sed.raw_hash_cache import raw_input_cache
from sed.react import build_messages
from sed.storage import active_file_hash_cache, canonical_bytes, digest, file_hash, write_json
from sed.thinking import split_thinking
from sed.visibility import visible_observation
from tools.v0_data import round_seed_count


def trajectory_row(episode, task, episode_ref, seed_k):
    """Preserve every original assistant turn, including diagnostic exceptions."""
    require(type(seed_k) is int and seed_k in (10, 20, 30), "Unsupported V0 seed K")
    require(episode.get("schema_version") == 2 and episode.get("steps")
            and episode.get("success") is True and episode.get("complete") is True
            and episode.get("status") == "ok" and episode.get("split") == "train"
            and episode.get("sealed") is False and episode["task_id"] == task["task_id"],
            "Replay requires a verified successful TRAIN trajectory")
    history, visible = [], visible_observation(episode["initial"]["visible"])
    for step in episode["steps"]:
        reply, hidden = split_thinking(step["action_raw"])
        require(not hidden, "Replay contains hidden thinking")
        messages = build_messages(history, {}, visible)
        history.append({"visible": visible, "reply": reply})
        visible = visible_observation(step["visible"])
    messages.append({"role": "assistant", "content": reply})
    source = f"k{seed_k}_replay"
    row = dict(example_id=digest(["CONTINUE", source, task["task_id"], episode_ref, None]),
        arm="CONTINUE", source=source, kind="trajectory", task_id=task["task_id"],
        group_id=task["group_id"], task_type=task["task_type"], messages=messages,
        supervised_assistant_indices=list(range(2, len(messages), 2)), weight=1.0,
        evidence=dict(episode_ref=episode_ref, step=None, executor="teacher", local_event=None,
                      episode_success=True, label_source="teacher_reply", label_check="passed"))
    v2.validate_example(row, "CONTINUE", task, ({task["task_id"]}, {task["group_id"]}),
                        (set(), set()), seed_k=seed_k)
    return row


def build_replay(output, *, round_manifest, theta0, config, splits, snapshot=None, relocate=()):
    relocation = Relocation(output, relocate)
    output = relocation.dataset
    require(not output.exists() and not Path(output).is_symlink(), "Output already exists")
    files = InputFiles(relocation)
    round_path, splits = files.reference(round_manifest), files.reference(splits)
    seed_k = round_seed_count(read_json(round_path))
    theta0 = relocation.path(theta0)
    config = deepcopy(config)
    for key in ("manifest", "split_freeze"):
        config["paths"][key] = str(files.reference(config["paths"][key]))
    source_path = files.add(theta0.parent / "provenance.json")
    source = read_json(source_path)
    snapshot = relocation.path(snapshot or source["config"]["sft"]["base_snapshot"]["path"])
    view, collection = training_view_inputs(files, source, validate=True, config=config, seed_k=seed_k)
    protected = [theta0, snapshot, view, view.parent.parent, round_path, splits]
    require(not any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected),
            "Output must be separate from source inputs")
    if active_file_hash_cache() is not None:
        active_file_hash_cache().protect(*protected)
    inherited = v2.continuation_recipe(theta0, config, snapshot, 1, seed_k=seed_k)
    for name, sha in inherited["files"].items():
        files.add(theta0 / name, sha)
    frozen, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    rows = []
    for receipt in collection["verified"]:
        # The recursive collection check above already verified all attempts,
        # raw outputs, receipts and episode hashes; retain their provenance.
        episode_dir = view.parent.parent / receipt["episode_dir"]
        episode_path = files.add(episode_dir / "episode.json", receipt["episode_hashes"]["episode.json"])
        task = resolve_task(frozen, freeze, receipt["task_id"])
        rows.append(trajectory_row(read_json(episode_path), task, str(episode_path), seed_k))
    rows.sort(key=lambda row: row["example_id"])
    require(len(rows) == seed_k, "Replay must contain exactly K executed seed trajectories")
    files.verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v0-replay-", dir=output.parent) as temporary:
        staging = Path(temporary)
        prov = provenance(staging, config, command="v0_replay", inputs=files.provenance_inputs([Path(__file__)]),
            details={"K": seed_k, "training_view": str(view), "inherited": inherited,
                     "new_teacher_calls": 0, "new_environment_steps": 0, "training_exposure": 0})
        (staging / "examples.jsonl").write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
        (staging / "examples.sha256").write_text(file_hash(staging / "examples.jsonl") + "\n")
        write_json(staging / "manifest.json", {
            "schema": "v2-train-interchange/1", "arm": "CONTINUE", "K": seed_k,
            "created": prov["timestamp"], "code_revision": prov["git_head"],
            "code_sha256": prov["code_sha256"], "dependency_versions": prov["dependencies"],
            "effective_config": prov["config"], "inputs": prov["inputs"],
            "counts": {**{"by_" + k: dict(Counter(row[k] for row in rows))
                          for k in ("source", "kind", "task_type")},
                       "examples": len(rows), "unique_instances": len({r["task_id"] for r in rows}),
                       "task_groups": len({r["group_id"] for r in rows})},
            "supervised_turns": sum(len(row["supervised_assistant_indices"]) for row in rows),
            "example_ids_sha256": digest([row["example_id"] for row in rows]),
            "splits_sha256": file_hash(splits), "theta0": {"adapter_path": str(theta0),
                "adapter_weights_sha256": inherited["identity"]["adapter_weights_sha256"]},
            "training_exposure_tasks": sorted({row["group_id"] for row in rows}), "excluded": [],
            "new_teacher_calls": 0, "new_environment_steps": 0, "training_exposure": 0})
        inspection = v2.inspect_inputs(staging, theta0=theta0, snapshot=snapshot, config=config,
            splits=splits, relocate=relocate, seed_k=seed_k)
        v2.read_dataset(staging, theta0=theta0, config=config, _inspection=inspection, seed_k=seed_k)
        staging.rename(output)
    return {"K": seed_k, "rows": len(rows), "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("round-manifest", "theta0", "splits", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--config", default="configs/base_e4b.yaml")
    parser.add_argument("--relocate", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        with raw_input_cache():
            relocation = Relocation(args.output, args.relocate)
            result = build_replay(args.output, round_manifest=args.round_manifest, theta0=args.theta0,
                config=load_config(relocation.path(args.config)), splits=args.splits,
                snapshot=args.snapshot, relocate=args.relocate)
        print(json.dumps(result))
        return 0
    except ValidationError as exc:
        print(f"V0 replay refused: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"V0 replay failed ({type(exc).__name__}); no bundle published.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
