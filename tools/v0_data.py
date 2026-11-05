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
from sed.raw_hash_cache import cached_inputs, raw_input_cache
from sed.storage import active_file_hash_cache, canonical_bytes, digest, file_hash
from sed.visibility import visible_observation
from repair.labels import instance_ids

BUCKETS = ("R", "G", "S", "D")
V0_FIELDS = {"bucket", "episode_id", "generator", "bin", "edit_decision", "synthetic_manifest_sha256"}
# Contract mirrored from repair/v0_common.py in the eval producer (e5bf24f).
BIN_GENERATORS = {"near": ("G1", "G3"), "mid": ("G2",), "far": ("G2",), "very_far": ("G5",)}
GENERATORS = ("G1", "G2", "G3", "G5", "seed")
BINS = (*BIN_GENERATORS, "seed")
SEED_COUNTS = (10, 20, 30)


def round_seed_count(manifest, synthetic=None):
    """K is declared by the round/synthetic manifest, never inferred from tasks."""
    values = [doc["K"] for doc in (manifest, synthetic) if doc is not None and "K" in doc]
    require(values and all(type(k) is int and k in SEED_COUNTS for k in values),
            "V0 requires explicit K in {10, 20, 30} in the round/synthetic manifest")
    require(len(set(values)) == 1, "Round/synthetic manifest K mismatch")
    return values[0]


def episodes(rows):
    result = defaultdict(list)
    for i, row in enumerate(rows):
        result[row["v0"]["episode_id"]].append(i)
    return {key: sorted(result[key], key=lambda i: rows[i]["example_id"]) for key in sorted(result)}


def _validate_v0_evidence(row):
    """Check producer executor/label names through V2 without rewriting rows."""
    evidence = row["evidence"]
    require(isinstance(evidence, dict), "Invalid evidence schema")
    if row["v0"]["bucket"] == "D":
        _validate_evidence(row)
        return
    executor = evidence.get("executor")
    require(executor in ("student", "patch", "handback-student"), "Invalid V0 executor")
    require(row["v0"]["edit_decision"] == (executor == "patch"), "V0 edit decision/executor mismatch")
    evidence = dict(evidence, executor="teacher" if executor == "patch" else "student")
    if evidence.get("label_source") == "action_only_no_label":
        require(evidence.get("label_check") == "not_available", "Invalid V0 action-only label evidence")
        # V2's invalid-label path checks the same action-only target invariant.
        evidence["label_source"] = "action_only_invalid_label"
    _validate_evidence(dict(row, evidence=evidence))


def validate_row(row):
    require(isinstance(row, dict) and set(row) == EXAMPLE_FIELDS | {"v0"}, "Invalid V0 row fields")
    require(row["arm"] == "V0", "Expected arm V0")
    require(all(isinstance(row[k], str) and row[k] for k in
                ("example_id", "source", "task_id", "group_id", "task_type")), "Missing row identity")
    meta = row["v0"]
    require(isinstance(meta, dict) and set(meta) == V0_FIELDS, "Invalid v0 metadata fields")
    require(meta["bucket"] in BUCKETS and meta["generator"] in GENERATORS
            and isinstance(meta["episode_id"], str) and meta["episode_id"]
            and type(meta["edit_decision"]) is bool, "Invalid v0 metadata")
    require(meta["bin"] in BINS, "Invalid v0 bin")
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
    _validate_v0_evidence(row)
    require(row["evidence"]["episode_success"] is True, "V0 requires executed successful episodes")
    history = []
    for i in range(2, len(messages), 2):
        require(messages[i - 1]["role"] == "user" and messages[i]["role"] == "assistant",
                "Expected alternating public user/assistant messages")
        visible = _json(messages[i - 1]["content"])
        require(isinstance(visible, dict) and set(visible) == {"observation", "admissible_commands"},
                "User fields exceed the public observation whitelist")
        if i == len(messages) - 1:
            # With task={}, the complete rendered prefix contains every earlier
            # prefix unchanged. Compare it once instead of rendering all growing
            # prefixes quadratically; every turn is still parsed and checked.
            require(build_messages(history, {}, visible_observation(visible)) == messages[:i],
                    "Messages differ from the exact student prompt contract")
        action = parse_action(messages[i]["content"])
        if i in indices and meta["bucket"] != "D":
            require(action in visible["admissible_commands"], "Rejected action cannot be supervised")
            require(instance_ids(messages[i]["content"]) <= instance_ids("\n".join(m["content"] for m in messages[:i])),
                    "Label cites an unseen instance ID")
        history.append({"visible": visible, "reply": messages[i]["content"]})


@cached_inputs
def read_synthetic_manifest(path, seed_tasks):
    """Training membership is explicit; held-out entries remain in the freeze."""
    document = read_json(path)
    require(document.get("schema") == "v0-synthetic-tasks/1" and isinstance(document.get("tasks"), list),
            "Invalid synthetic-task manifest schema")
    tasks, fingerprints = {}, set()
    for task in document["tasks"]:
        require(isinstance(task, dict) and {"task_id", "parent_seed_task_id", "group_id", "task_type",
                "generator", "bin", "split", "sealed", "heldout_probe", "status", "item_fingerprint"}
                <= task.keys(), "Incomplete synthetic task")
        tid = task["task_id"]
        require(isinstance(tid, str) and tid.startswith("train/synth/") and len(tid.split("/")) == 3
                and tid.split("/")[-1] not in ("", ".", "..") and "\\" not in tid and tid not in tasks,
                "Invalid or duplicate synthetic task id")
        parent = task["parent_seed_task_id"]
        require(isinstance(parent, str) and parent in seed_tasks and not parent.startswith("train/synth/"),
                "Synthetic parent is not in theta0 K seeds")
        require(task["generator"] in GENERATORS[:-1] and task["bin"] in tuple(BIN_GENERATORS)
                and task["status"] == "valid" and type(task["sealed"]) is bool
                and all(isinstance(task[k], str) and task[k]
                for k in ("group_id", "task_type")), "Invalid synthetic task fields")
        require(task["generator"] in BIN_GENERATORS[task["bin"]], "Synthetic generator does not match distance bin")
        probe = task["heldout_probe"]
        require(type(probe) is bool and task["split"] == ("holdout" if probe else "train")
                and task["sealed"] == probe, "Synthetic probe split mismatch")
        fingerprint = task["item_fingerprint"]
        require(isinstance(fingerprint, str) and fingerprint and fingerprint not in fingerprints,
                "Invalid or duplicate synthetic training/probe fingerprint")
        fingerprints.add(fingerprint)
        tasks[tid] = task
    training_groups = {t["group_id"] for t in tasks.values() if not t["heldout_probe"]}
    held_groups = {t["group_id"] for t in tasks.values() if t["heldout_probe"]}
    require(not training_groups & held_groups, "Synthetic training/holdout group overlap")
    return tasks, file_hash(path)


@cached_inputs
def read_dataset(directory, *, theta0, snapshot=None, config, synthetic_manifest=None,
                 splits=None, relocate=(), repair_splits=None, dev140=None):
    """Validate V2 replay, original K evidence, split freezes and all V0 seals."""
    relocation = Relocation(directory, relocate)
    root = relocation.dataset
    active_file_hash_cache().protect(root, relocation.path(theta0),
        relocation.path(snapshot) if snapshot else None,
        relocation.path(synthetic_manifest).parent if synthetic_manifest else None)
    manifest = read_json(root / "manifest.json")
    require(manifest.get("schema") == "v0-train-interchange/1" and manifest.get("arm") == "V0",
            "Invalid V0 dataset manifest")
    require(all(isinstance(manifest.get(k), str) and manifest[k] for k in ("created", "code_revision")),
            "Missing V0 dataset provenance")
    seed_k = round_seed_count(manifest, read_json(relocation.path(synthetic_manifest))
                              if synthetic_manifest is not None else None)
    # The replay bundle is unchanged V2: retain its mature evidence/split checks.
    inspection = v2.inspect_inputs(root / "replay", theta0=theta0, snapshot=snapshot, config=config,
        splits=splits, relocate=Relocation(root / "replay", relocate),
        repair_splits=repair_splits, dev140=dev140, seed_k=seed_k)
    return _finish_read(root, manifest, inspection, synthetic_manifest, seed_k)


def _finish_read(root, manifest, inspection, synthetic_manifest, seed_k):
    files = inspection["files"]
    theta0, config = inspection["theta0"], inspection["config"]
    replay = v2.read_dataset(root / "replay", theta0=theta0, config=config, _inspection=inspection, seed_k=seed_k)
    require(all(r["source"] == f"k{seed_k}_replay" for r in replay["rows"]), "replay/ must contain only K seed rows")
    # CONTINUE may store the same seeds in both its main and replay bundles.
    # V2 uses both shares; V0's D bucket contains each example exactly once.
    seen, replay_rows, replay_origins = {}, [], []
    for row, origin in zip(replay["rows"], replay["origins"]):
        ident, content = row["example_id"], canonical_bytes(row)
        if ident in seen:
            require(seen[ident] == content, f"Conflicting V0 replay duplicate example_id: {ident}")
        else:
            seen[ident] = content
            replay_rows.append(row)
            replay_origins.append(origin)
    inspection["replay_deduplication"] = {"input_rows": len(replay["rows"]),
        "unique_rows": len(replay_rows), "duplicates_removed": len(replay["rows"]) - len(replay_rows)}
    if "training_collection" in inspection:
        # inspect_inputs already ran this exact validation for the K fallback.
        view, collection = inspection["training_view"], inspection["training_collection"]
    else:
        view, collection = training_view_inputs(files, inspection["theta0_provenance"], validate=True,
                                               config=config, seed_k=seed_k)
    require(collection["K"] == seed_k, "Round K differs from theta0's original support manifest")
    seeds = {r["task_id"] for r in collection["verified"]}
    require(len(collection["verified"]) == collection["K"], "K must count verified seed trajectories")
    require(Counter(r["task_id"] for r in replay_rows) == Counter(r["task_id"] for r in collection["verified"]),
            "D replay must cover exactly theta0's K seed trajectories")
    require(isinstance(manifest.get("inputs"), dict) and manifest["inputs"], "Missing hashed V0 inputs")
    with files.relocation.resolution_batch():
        main_inputs = {files.reference(name, sha, root=root): sha for name, sha in manifest["inputs"].items()}
    main_hashes = set(manifest["inputs"].values())
    for path in (inspection["splits"], Path(config["paths"]["manifest"]), Path(config["paths"]["split_freeze"])):
        # Producer/consumer checkouts can contain byte-identical copies. Match
        # raw seals, not JSON equivalence; files.verify() still checks every
        # producer input, and provenance retains both local and sealed paths.
        require(path in main_inputs or file_hash(path) in main_hashes,
                "V0 inputs must seal the task manifest and both split freezes")
    files.add(root / "manifest.json")
    files.add(root / "examples.jsonl", manifest.get("examples_sha256"), reason="V0 examples hash mismatch")
    require(_sha(manifest.get("examples_sha256")), "Missing V0 examples_sha256")
    require(manifest.get("replay_manifest_sha256") == file_hash(root / "replay/manifest.json"),
            "V0 replay manifest hash mismatch")
    require(manifest.get("splits_sha256") == file_hash(inspection["splits"]), "V0 split freeze hash mismatch")
    require(manifest.get("theta0", {}).get("adapter_weights_sha256") == replay["theta0_sha256"],
            "V0 theta0 adapter hash mismatch")
    synthetic, synth_sha, synth_path = {}, "", None
    if synthetic_manifest is not None:
        synth_path = files.relocation.path(synthetic_manifest)
        require(synth_path in main_inputs, "Synthetic manifest must be a sealed V0 input")
        files.verify(paths=[synth_path])
        synthetic, synth_sha = read_synthetic_manifest(synth_path, seeds)
    files.verify()
    rows = [_json(line) for line in (root / "examples.jsonl").read_text().splitlines() if line.strip()]
    frozen, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    sets = read_json(inspection["splits"])["sets"]
    if f"K{seed_k}" in sets:
        require(v2._members(sets[f"K{seed_k}"]) == (seeds, {r["group_id"] for r in collection["verified"]}),
                "Frozen K membership differs from theta0's original support manifest")
    # The V2 reader already checks all external diagnostic sources and overlap
    # against K. Synthetic identities must not masquerade as those instances.
    forbidden_tasks, forbidden_groups = set(), set()
    for name, members in sets.items():
        if name not in (f"K{seed_k}", "V2_pool"):
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
        require(meta["bucket"] != "D" and row["source"] not in {f"k{k}_replay" for k in SEED_COUNTS},
                "D rows must come from unchanged replay/")
        require(row["kind"] == "decision", "Non-D V0 rows must be decisions")
        if tid.startswith("train/synth/"):
            require(synthetic_manifest is not None, "Synthetic task requires --synthetic-manifest")
            require(tid in synthetic, "Synthetic id is absent from the manifest")
            require(meta["synthetic_manifest_sha256"] == synth_sha, "Synthetic manifest hash mismatch")
            task = synthetic[tid]
            require(not task["heldout_probe"] and task["split"] == "train" and not task["sealed"],
                    "Held-out synthetic task cannot enter training")
            require(meta["generator"] == task["generator"], "Synthetic generator mismatch")
            require(meta["bin"] == task["bin"], "Synthetic bin mismatch")
        else:
            require(tid in seeds and meta["generator"] == "seed", "Non-synthetic task must be a theta0 K seed")
            require(meta["bin"] == "seed", "Seed rows use bin seed")
            require(meta["synthetic_manifest_sha256"] == "", "Seed rows use an empty synthetic manifest hash")
            task = resolve_task(frozen, freeze, tid)
            authorize_task(task, "generation")
        require(row["group_id"] == task["group_id"] and row["task_type"] == task["task_type"], "Task identity mismatch")
        require(tid not in forbidden_tasks and row["group_id"] not in forbidden_groups, "Diagnostic task/group in training")
    origins = [str(root / "examples.jsonl")] * len(rows)
    for row, origin in zip(replay_rows, replay_origins):
        row = deepcopy(row)
        row["arm"] = "V0"
        row["v0"] = {"bucket": "D", "episode_id": "seed/" + row["example_id"], "generator": "seed", "bin": "seed",
                     "edit_decision": False, "synthetic_manifest_sha256": ""}
        validate_row(row)
        rows.append(row)
        origins.append(origin)
    require(len({r["example_id"] for r in rows}) == len(rows), "Duplicate example id")
    grouped = episodes(rows)
    for indices in grouped.values():
        require(len({(rows[i]["task_id"], rows[i]["group_id"], rows[i]["v0"]["bucket"],
                      rows[i]["v0"]["generator"], rows[i]["v0"]["bin"],
                      rows[i]["evidence"]["episode_ref"]) for i in indices}) == 1,
                "Episode crosses tasks, buckets, generators, bins or evidence references")
        require(len({rows[i]["evidence"]["step"] for i in indices}) == len(indices), "Duplicate episode decision")
    # One successful S trajectory per instance, as preregistered.
    s_tasks = [rows[indices[0]]["task_id"] for indices in grouped.values()
               if rows[indices[0]]["v0"]["bucket"] == "S"]
    require(len(set(s_tasks)) == len(s_tasks), "At most one S episode per task")
    dataset_hash = digest({"manifest": file_hash(root / "manifest.json"), "rows": rows,
                           "replay_manifest": file_hash(root / "replay/manifest.json")})
    return {"rows": rows, "episodes": grouped, "origins": origins, "manifest": manifest,
            "synthetic_manifest": synth_path, "synthetic_tasks": synthetic, "input_seals": main_inputs,
            "dataset_sha256": dataset_hash, "adapter_sha256": replay["theta0_sha256"],
            "K": collection["K"], "seed_task_ids": sorted(seeds), "training_view": str(view),
            "inspection": inspection, "config": config, "theta0": theta0, "snapshot": inspection["snapshot"]}


@cached_inputs
def sealed_input(data, path):
    """Resolve an expansion input and recheck its seal before interpreting it."""
    relocation = data["inspection"]["files"].relocation
    path = relocation.local(path)
    seal = data["input_seals"].get(path)
    require(seal is not None and path.is_file(), f"Missing sealed expansion input: {path}")
    require(file_hash(path) == seal, f"Expansion input hash mismatch: {path}")
    return path


@cached_inputs
def verified_parent(data):
    """Reconstruct the parent identity from the producer's sealed input bundle.

    v0_expand seals the parent manifest/examples in inputs, rather than storing
    the consumer's composite dataset digest as a top-level manifest field.
    """
    inspection = data["inspection"]
    relocation = inspection["files"].relocation
    root = relocation.dataset.parent  # The inspection is for dataset/replay.
    candidates = {}
    for path in data["input_seals"]:
        if path.name != "manifest.json":
            continue
        path = sealed_input(data, path)
        document = read_json(path)
        if path != root / "manifest.json" and document.get("schema") == "v0-train-interchange/1":
            candidates[path] = document
    require(len(candidates) == 1, "Weights/modules dataset or adapter hash mismatch: no unique sealed parent dataset")
    path, manifest = next(iter(candidates.items()))
    sealed_input(data, path.parent / "examples.jsonl")
    require(isinstance(manifest.get("synthetic_manifest"), str), "Parent dataset lacks synthetic manifest reference")
    synthetic = relocation.path(manifest["synthetic_manifest"], root=path.parent)
    sources = inspection["split_sources"] or {}
    parent = read_dataset(path.parent, theta0=data["theta0"], snapshot=data["snapshot"], config=data["config"],
        synthetic_manifest=synthetic, splits=inspection["splits"],
        relocate=[f"{old}={new}" for old, new in relocation.rules],
        repair_splits=sources.get("repair_splits"), dev140=sources.get("dev140"))
    require(parent["adapter_sha256"] == data["adapter_sha256"], "Parent dataset adapter hash mismatch")
    # Expansion only appends G decisions and refreshes the synthetic-manifest
    # reference on inherited rows. Do not accept an unrelated sealed input.
    inherited = {r["example_id"]: r for r in parent["rows"]}
    current = {r["example_id"]: r for r in data["rows"]}
    require(inherited.keys() <= current.keys(), "Expanded dataset dropped parent rows")
    for eid, row in inherited.items():
        updated = dict(current[eid], v0=dict(current[eid]["v0"],
            synthetic_manifest_sha256=row["v0"]["synthetic_manifest_sha256"]))
        require(canonical_bytes(updated) == canonical_bytes(row), "Expanded dataset changed parent rows")
    require(all(row["v0"]["bucket"] == "G" and row["v0"]["episode_id"] not in parent["episodes"]
                for eid, row in current.items() if eid not in inherited), "Expansion must append only new G episodes")
    data["parent_binding"] = {"path": str(path.parent), "manifest_sha256": file_hash(path),
        "dataset_sha256": parent["dataset_sha256"], "adapter_sha256": parent["adapter_sha256"]}
    return parent


def expansion_units(data, episode_ids):
    """Audit-only G episode -> task -> allocated unit; never added to prompts."""
    require(data["synthetic_manifest"] is not None, "G weights require a synthetic manifest")
    synth = data["synthetic_manifest"].parent
    generated_path = sealed_input(data, synth / "generated.json")
    execution_path = sealed_input(data, synth / "allocation_execution.json")
    # allocation.json itself is not sealed by older producers. Its units must
    # agree with both sealed generated.json and allocation_execution.json.
    allocation_path = synth.parent / "allocation.json"
    require(allocation_path.is_file(), "G weights require expansion allocation.json")
    allocation = read_json(allocation_path)
    require(allocation.get("variant") == "D-data" and isinstance(allocation.get("units"), list),
            "G weights require a D-data expansion allocation")
    units = {}
    for unit in allocation["units"]:
        require(isinstance(unit, dict) and type(unit.get("unit")) is int and unit["unit"] >= 0
                and unit["unit"] not in units, "Invalid or duplicate expansion unit")
        units[unit["unit"]] = unit
    generated, execution = read_json(generated_path), read_json(execution_path)
    require(isinstance(generated, list) and isinstance(execution, list), "Invalid expansion task/unit mapping")
    tasks, executed, seen_units = {}, {}, set()
    for item in generated:
        require(isinstance(item, dict) and isinstance(item.get("task_id"), str)
                and item["task_id"] not in tasks, "Invalid or duplicate generated task")
        tasks[item["task_id"]] = item
    for item in execution:
        unit = item.get("unit", {})
        uid = unit.get("unit")
        require(type(uid) is int and uid in units and unit == units[uid] and uid not in seen_units,
                "Expansion execution unit differs from allocation")
        seen_units.add(uid)
        if item.get("status") == "filled":
            tid = item.get("generated_task_id")
            require(isinstance(tid, str) and tid not in executed, "Duplicate or missing executed expansion task")
            executed[tid] = unit
    result = {}
    for eid in sorted(episode_ids):
        row = data["rows"][data["episodes"][eid][0]]
        tid = row["task_id"]
        unit = executed.get(tid)
        require(unit is not None and tid in tasks and tid in data["synthetic_tasks"],
                f"Unresolved G episode expansion unit: {eid}")
        require(tasks[tid].get("expansion_unit") == unit
                and data["synthetic_tasks"][tid].get("expansion_unit") == unit,
                f"Conflicting G episode expansion unit: {eid}")
        require(isinstance(unit.get("group"), str) and isinstance(unit.get("source_episode_id"), str),
                f"Unresolved G episode expansion group: {eid}")
        result[eid] = {"task_id": tid, "unit": unit["unit"], "group": unit["group"],
                       "source_episode_id": unit["source_episode_id"]}
    return result, [allocation_path, generated_path, execution_path]


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
    parser.add_argument("--full-rehash", action="store_true", help="Bypass hash receipts and process memo")
    parser.add_argument("--hash-workers", type=int, default=None,
                        help="Bound concurrent input hashes (default SED_HASH_WORKERS or 8)")


def load_args(args):
    from sed.config import load_config
    relocation = Relocation(args.dataset, args.relocate)
    with raw_input_cache(full_rehash=getattr(args, "full_rehash", False), workers=getattr(args, "hash_workers", None)):
        return read_dataset(args.dataset, theta0=args.theta0, snapshot=args.snapshot,
            config=load_config(relocation.path(args.config)), synthetic_manifest=args.synthetic_manifest,
            splits=args.splits, relocate=args.relocate, repair_splits=args.repair_splits, dev140=args.dev140)


def provenance_inputs(data, extra=()):
    sources = [*Path(__file__).parent.glob("v0_*.py"), Path(__file__).with_name("train_v0_arm.py"),
               Path(__file__).with_name("train_v2_arm.py"), Path(__file__).with_name("crn_tail.py"),
               Path(__file__).with_name("grad_stats.py")]
    return data["inspection"]["files"].provenance_inputs([*sources, *extra])
