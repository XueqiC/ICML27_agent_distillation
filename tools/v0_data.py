"""V0 input contract and CPU-only validation; see V0_DATASET.md."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from repair import v2_train as v2
from repair.artifacts import read_json
from repair.v2_inputs import Relocation, require, training_view_inputs
from repair.v2_validation import EXAMPLE_FIELDS, _json, _sha, _validate_evidence
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.react import build_messages, parse_action
from sed.storage import digest, file_hash
from sed.visibility import visible_observation
from repair.labels import instance_ids

BUCKETS = ("R", "G", "S", "D")
V0_FIELDS = {"bucket", "episode_id", "generator", "edit_decision", "synthetic_manifest_sha256"}


def episodes(rows):
    result = defaultdict(list)
    for i, row in enumerate(rows):
        result[row["v0"]["episode_id"]].append(i)
    return {key: sorted(result[key], key=lambda i: rows[i]["example_id"]) for key in sorted(result)}


def validate_row(row):
    require(isinstance(row, dict) and set(row) == EXAMPLE_FIELDS | {"v0"}, "Invalid V0 row fields")
    require(row["arm"] == "V0", "Expected arm V0")
    require(all(isinstance(row[k], str) and row[k] for k in
                ("example_id", "source", "task_id", "group_id", "task_type")), "Missing row identity")
    meta = row["v0"]
    require(isinstance(meta, dict) and set(meta) == V0_FIELDS, "Invalid v0 metadata fields")
    require(meta["bucket"] in BUCKETS and meta["generator"] in ("G1", "G2", "G3", "seed")
            and isinstance(meta["episode_id"], str) and meta["episode_id"]
            and type(meta["edit_decision"]) is bool, "Invalid v0 metadata")
    require(row["kind"] in ("decision", "trajectory"), "Invalid row kind")
    require(type(row["weight"]) in (int, float) and row["weight"] == 1,
            "Row weight must be 1; use episode weights.json")
    messages, indices = row["messages"], row["supervised_assistant_indices"]
    require(isinstance(messages, list) and len(messages) >= 3 and len(messages) % 2 == 1,
            "Messages must end with an assistant reply")
    require(all(isinstance(m, dict) and set(m) == {"role", "content"}
                and isinstance(m["content"], str) and "Guidance:" not in m["content"] for m in messages),
            "Invalid message fields or Guidance: text")
    expected = list(range(2, len(messages), 2)) if row["kind"] == "trajectory" else [len(messages) - 1]
    require(isinstance(indices, list) and all(type(i) is int for i in indices) and indices == expected,
            "Invalid supervised assistant indices")
    _validate_evidence(row)
    require(row["evidence"]["episode_success"] is True, "V0 requires executed successful episodes")
    history = []
    for i in range(2, len(messages), 2):
        require(messages[i - 1]["role"] == "user" and messages[i]["role"] == "assistant",
                "Expected alternating public user/assistant messages")
        visible = _json(messages[i - 1]["content"])
        require(isinstance(visible, dict) and set(visible) == {"observation", "admissible_commands"},
                "User fields exceed the public observation whitelist")
        require(build_messages(history, {}, visible_observation(visible)) == messages[:i],
                "Messages differ from the exact student prompt contract")
        action = parse_action(messages[i]["content"])
        if i in indices and meta["bucket"] != "D":
            require(action in visible["admissible_commands"], "Rejected action cannot be supervised")
            require(instance_ids(messages[i]["content"]) <= instance_ids("\n".join(m["content"] for m in messages[:i])),
                    "Label cites an unseen instance ID")
        history.append({"visible": visible, "reply": messages[i]["content"]})


def read_synthetic_manifest(path, seed_tasks):
    """Training membership is explicit; held-out entries remain in the freeze."""
    document = read_json(path)
    require(document.get("schema") == "v0-synthetic-tasks/1" and isinstance(document.get("tasks"), list),
            "Invalid synthetic-task manifest schema")
    tasks = {}
    for task in document["tasks"]:
        require(isinstance(task, dict) and {"task_id", "parent_seed_task_id", "group_id", "task_type",
                "generator", "split", "sealed"} <= task.keys(), "Incomplete synthetic task")
        tid = task["task_id"]
        require(isinstance(tid, str) and tid.startswith("train/synth/") and len(tid.split("/")) == 3
                and tid.split("/")[-1] not in ("", ".", "..") and tid not in tasks,
                "Invalid or duplicate synthetic task id")
        require(task["parent_seed_task_id"] in seed_tasks, "Synthetic parent is not in theta0 K seeds")
        require(task["generator"] in ("G1", "G2", "G3") and task["split"] in ("train", "holdout")
                and type(task["sealed"]) is bool and all(isinstance(task[k], str) and task[k]
                for k in ("group_id", "task_type")), "Invalid synthetic task fields")
        tasks[tid] = task
    training_groups = {t["group_id"] for t in tasks.values() if t["split"] == "train" and not t["sealed"]}
    held_groups = {t["group_id"] for t in tasks.values() if t["split"] != "train" or t["sealed"]}
    require(not training_groups & held_groups, "Synthetic training/holdout group overlap")
    return tasks, file_hash(path)


def read_dataset(directory, *, theta0, snapshot=None, config, synthetic_manifest=None,
                 splits=None, relocate=(), repair_splits=None, dev140=None):
    """Validate V2 replay, original K evidence, split freezes and all V0 seals."""
    relocation = Relocation(directory, relocate)
    root = relocation.dataset
    manifest = read_json(root / "manifest.json")
    require(manifest.get("schema") == "v0-train-interchange/1" and manifest.get("arm") == "V0",
            "Invalid V0 dataset manifest")
    require(all(isinstance(manifest.get(k), str) and manifest[k] for k in ("created", "code_revision")),
            "Missing V0 dataset provenance")
    # The replay bundle is unchanged V2: retain its mature evidence/split checks.
    inspection = v2.inspect_inputs(root / "replay", theta0=theta0, snapshot=snapshot, config=config,
        splits=splits, relocate=Relocation(root / "replay", relocate),
        repair_splits=repair_splits, dev140=dev140)
    return _finish_read(root, manifest, inspection, synthetic_manifest)


def _finish_read(root, manifest, inspection, synthetic_manifest):
    files = inspection["files"]
    theta0, config = inspection["theta0"], inspection["config"]
    replay = v2.read_dataset(root / "replay", theta0=theta0, config=config, _inspection=inspection)
    require(all(r["source"] == "k20_replay" for r in replay["rows"]), "replay/ must contain only K seed rows")
    view, collection = training_view_inputs(files, inspection["theta0_provenance"], validate=True, config=config)
    seeds = {r["task_id"] for r in collection["verified"]}
    require(len(collection["verified"]) == collection["K"], "K must count verified seed trajectories")
    require(Counter(r["task_id"] for r in replay["rows"]) == Counter(r["task_id"] for r in collection["verified"]),
            "D replay must cover exactly theta0's K seed trajectories")
    require(isinstance(manifest.get("inputs"), dict) and manifest["inputs"], "Missing hashed V0 inputs")
    main_inputs = {files.reference(name, sha, root=root) for name, sha in manifest["inputs"].items()}
    for path in (inspection["splits"], Path(config["paths"]["manifest"]), Path(config["paths"]["split_freeze"])):
        require(path in main_inputs, "V0 inputs must seal the task manifest and both split freezes")
    files.add(root / "manifest.json")
    files.add(root / "examples.jsonl", manifest.get("examples_sha256"), reason="V0 examples hash mismatch")
    require(_sha(manifest.get("examples_sha256")), "Missing V0 examples_sha256")
    require(manifest.get("replay_manifest_sha256") == file_hash(root / "replay/manifest.json"),
            "V0 replay manifest hash mismatch")
    require(manifest.get("splits_sha256") == file_hash(inspection["splits"]), "V0 split freeze hash mismatch")
    require(manifest.get("theta0", {}).get("adapter_weights_sha256") == replay["theta0_sha256"],
            "V0 theta0 adapter hash mismatch")
    synthetic, synth_sha = {}, ""
    if synthetic_manifest is not None:
        path = files.relocation.path(synthetic_manifest)
        require(path in main_inputs, "Synthetic manifest must be a sealed V0 input")
        files.verify(paths=[path])
        synthetic, synth_sha = read_synthetic_manifest(path, seeds)
    files.verify()
    rows = [_json(line) for line in (root / "examples.jsonl").read_text().splitlines() if line.strip()]
    frozen, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    sets = read_json(inspection["splits"])["sets"]
    # The V2 reader already checks all external diagnostic sources and overlap
    # against K. Synthetic identities must not masquerade as those instances.
    forbidden_tasks, forbidden_groups = set(), set()
    for name, members in sets.items():
        if name not in ("K20", "V2_pool"):
            tasks, groups = v2._members(members)
            forbidden_tasks.update(tasks); forbidden_groups.update(groups)
    sources = inspection["split_sources"]
    if sources:
        repair_sets = read_json(sources["repair_splits"])["sets"]
        for name in ("U_mech", "D_probe", "D_unseen_quick"):
            tasks, groups = v2._members(repair_sets[name])
            forbidden_tasks.update(tasks); forbidden_groups.update(groups)
        for tid in read_json(sources["dev140"])["task_ids"]:
            forbidden_tasks.add(tid)
            forbidden_groups.add(resolve_task(frozen, freeze, tid)["group_id"])
    for row in rows:
        validate_row(row)
        meta, tid = row["v0"], row["task_id"]
        require(meta["bucket"] != "D" and row["source"] != "k20_replay", "D rows must come from unchanged replay/")
        require(row["kind"] == "decision", "Non-D V0 rows must be decisions")
        if tid.startswith("train/synth/"):
            require(synthetic_manifest is not None, "Synthetic task requires --synthetic-manifest")
            require(tid in synthetic, "Synthetic id is absent from the manifest")
            require(meta["synthetic_manifest_sha256"] == synth_sha, "Synthetic manifest hash mismatch")
            task = synthetic[tid]
            require(task["split"] == "train" and not task["sealed"], "Held-out synthetic task cannot enter training")
            require(meta["generator"] == task["generator"], "Synthetic generator mismatch")
        else:
            require(tid in seeds and meta["generator"] == "seed", "Non-synthetic task must be a theta0 K seed")
            require(meta["synthetic_manifest_sha256"] == "", "Seed rows use an empty synthetic manifest hash")
            task = resolve_task(frozen, freeze, tid)
            authorize_task(task, "generation")
        require(row["group_id"] == task["group_id"] and row["task_type"] == task["task_type"], "Task identity mismatch")
        require(tid not in forbidden_tasks and row["group_id"] not in forbidden_groups, "Diagnostic task/group in training")
    origins = [str(root / "examples.jsonl")] * len(rows)
    for row in replay["rows"]:
        row = deepcopy(row)
        row["arm"] = "V0"
        row["v0"] = {"bucket": "D", "episode_id": "seed/" + row["example_id"], "generator": "seed",
                     "edit_decision": False, "synthetic_manifest_sha256": ""}
        validate_row(row)
        rows.append(row)
        origins.append(str(root / "replay/examples.jsonl"))
    require(len({r["example_id"] for r in rows}) == len(rows), "Duplicate example id")
    grouped = episodes(rows)
    for indices in grouped.values():
        require(len({(rows[i]["task_id"], rows[i]["group_id"], rows[i]["v0"]["bucket"],
                      rows[i]["v0"]["generator"], rows[i]["evidence"]["episode_ref"]) for i in indices}) == 1,
                "Episode crosses tasks, buckets, generators or evidence references")
        require(len({rows[i]["evidence"]["step"] for i in indices}) == len(indices), "Duplicate episode decision")
    # One successful S trajectory per instance, as preregistered.
    s_tasks = [rows[indices[0]]["task_id"] for indices in grouped.values()
               if rows[indices[0]]["v0"]["bucket"] == "S"]
    require(len(set(s_tasks)) == len(s_tasks), "At most one S episode per task")
    dataset_hash = digest({"manifest": file_hash(root / "manifest.json"), "rows": rows,
                           "replay_manifest": file_hash(root / "replay/manifest.json")})
    return {"rows": rows, "episodes": grouped, "origins": origins, "manifest": manifest,
            "dataset_sha256": dataset_hash, "adapter_sha256": replay["theta0_sha256"],
            "K": collection["K"], "seed_task_ids": sorted(seeds), "training_view": str(view),
            "inspection": inspection, "config": config, "theta0": theta0, "snapshot": inspection["snapshot"]}


def safe_output(output, data):
    destination = Path(output).resolve()
    inspection = data["inspection"]
    protected = [inspection["files"].relocation.dataset.parent, data["theta0"], data["snapshot"], data["training_view"]]
    require(not any(destination.is_relative_to(Path(p).resolve()) for p in protected),
            "Output must be outside dataset, theta0, snapshot and seed view")
    require(not destination.exists() and not Path(output).is_symlink(), "Output already exists")
    return destination


def add_inputs(parser):
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--theta0", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--config", default="configs/base_e4b.yaml")
    parser.add_argument("--synthetic-manifest")
    parser.add_argument("--splits")
    parser.add_argument("--repair-splits")
    parser.add_argument("--dev140")
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")


def load_args(args):
    from sed.config import load_config
    relocation = Relocation(args.dataset, args.relocate)
    return read_dataset(args.dataset, theta0=args.theta0, snapshot=args.snapshot,
        config=load_config(relocation.path(args.config)), synthetic_manifest=args.synthetic_manifest,
        splits=args.splits, relocate=args.relocate, repair_splits=args.repair_splits, dev140=args.dev140)


def provenance_inputs(data, extra=()):
    sources = [*Path(__file__).parent.glob("v0_*.py"), Path(__file__).with_name("train_v0_arm.py"),
               Path(__file__).with_name("train_v2_arm.py"), Path(__file__).with_name("crn_tail.py"),
               Path(__file__).with_name("grad_stats.py")]
    return data["inspection"]["files"].provenance_inputs([*sources, *extra])
