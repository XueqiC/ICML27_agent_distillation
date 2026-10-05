"""Freeze Addendum A dev140 traits from resets, verified seeds and teacher lengths.

Audit-only output: never a prompt or training input. No policy is constructed.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import re

from analysis.mechanism.goals import Facts, normal_type, parse_goal, refuse_sealed
from harness.alfworld_env.environment import Environment
from sed.config import load_config
from sed.dev_subset import build_dev140
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import code_hashes, utc_now, versions
from sed.seed_evidence import load_verified_collection
from sed.storage import BlobStore, digest, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"pick_and_place_simple": "simple", "look_at_obj_in_light": "look",
            "pick_two_obj_and_place": "multi-object", "pick_heat_then_place_in_recep": "transform",
            "pick_cool_then_place_in_recep": "transform", "pick_clean_then_place_in_recep": "transform"}
BUCKETS = {"task_type": sorted(FAMILIES), "family": ["simple", "transform", "multi-object", "look"],
           "hidden_target": ["no", "yes"], "room_size": ["low", "middle", "high"],
           "distractors": ["0", "1", ">=2"], "state_change": ["no", "yes"],
           "minimum_length": ["low", "middle", "high", "unknown"], "seed_coverage": ["0", "1", ">=2"]}


class Inputs:
    """Snapshot exact consumed files and detect edits during a read-only analysis."""
    def __init__(self):
        self.hashes = {}

    def add(self, path):
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        sha = file_hash(path)
        if key in self.hashes and self.hashes[key] != sha:
            raise ValueError("Input changed during analysis")
        self.hashes[key] = sha
        return path

    def read(self, path):
        path = self.add(path)
        result = json.loads(path.read_text())
        self.add(path)
        return result

    def verify(self):
        for path, sha in self.hashes.items():
            if file_hash(ROOT / path) != sha:
                raise ValueError("Input changed during analysis")


def provenance(settings, inputs):
    code = code_hashes(ROOT)
    for name in ("analysis/traits.py", "analysis/stratified.py"):
        if (ROOT / name).exists():
            code[name] = file_hash(ROOT / name)
    return {"created_at": utc_now(), "settings": settings, "python_version": platform.python_version(), "dependencies": versions(),
            "code_sha256": digest(code), "code_files_sha256": code,
            "inputs_sha256": dict(sorted(inputs.hashes.items()))}


def require_dev(task_id):
    refuse_sealed(task_id)
    if not task_id.startswith("valid_seen/"):
        raise ValueError("Traits/readout require valid_seen dev tasks")


def count_bucket(count):
    if type(count) is not int or count < 0:
        raise ValueError("Expected a nonnegative integer count")
    return str(count) if count < 2 else ">=2"


def room_receptacles(observation):
    # Count only the initial room inventory, excluding the goal and any object text.
    match = re.search(r"Looking quickly around you, you see\s+([^\n]+)", observation, re.I)
    if not match:
        raise ValueError("Initial observation lacks the room receptacle list")
    names = re.findall(r"\b(?:a|an)\s+([a-z][a-z0-9]*\s+\d+)\b", match[1], re.I)
    if not names:
        raise ValueError("Initial room receptacle list is empty")
    return sorted({name.lower() for name in names})


def initial_traits(task_id, observation, initial_facts):
    require_dev(task_id)
    goal, facts = parse_goal(task_id), Facts(initial_facts)
    if goal.task_type not in FAMILIES or goal.movable_receptacle or "issliced" in goal.required:
        raise ValueError("Unsupported dev task type")
    objects = sorted(o for o, kind in facts.get("objecttype") if normal_type(kind) == goal.base_object)
    required = 2 if goal.task_type == "pick_two_obj_and_place" else 1
    if len(objects) < required:
        raise ValueError("Initial facts have fewer goal-type objects than required")
    closed = {args[0] for args in facts.get("openable") - facts.get("opened")}
    receptacles = {args[0] for args in facts.get("receptacletype")}
    parents = {o: sorted(r for obj, r in facts.get("inreceptacle") if obj == o) for o in objects}
    # Direct initial containment, as specified; no visibility inferred from a rollout.
    hidden = all(any(r in closed and r in receptacles for r in parents[o]) for o in objects)
    names = room_receptacles(observation)
    transform = next((verb for verb in ("heat", "cool", "clean") if f"_{verb}_" in goal.task_type), None)
    distractors = len(objects) - required
    return {"task_type": goal.task_type, "family": FAMILIES[goal.task_type],
            "hidden_target": "yes" if hidden else "no", "room_size": len(names),
            "distractors": distractors, "distractors_bucket": count_bucket(distractors),
            "state_change": "yes" if transform else "no", "transformation": transform,
            "goal_object_count": len(objects), "required_object_count": required,
            "initial_evidence": {"observation": observation, "room_receptacles": names,
                                 "goal_objects": objects, "goal_object_receptacles": parents,
                                 "closed_openable_receptacles": sorted(closed & receptacles),
                                 "facts_sha256": digest(initial_facts)}}


def tertiles(values):
    """Empirical inverse-CDF cuts (ceil(n/3), ceil(2n/3)); never split ties."""
    known = sorted(v for v in values if v is not None)
    return {"method": "inverse_ecdf_ceiling_ranks; low <= q1, middle <= q2, high > q2; ties stay together",
            "known_n": len(known), "unknown_n": len(values) - len(known),
            "cutpoints": [known[math.ceil(len(known) * q / 3) - 1] for q in (1, 2)] if known else []}


def tertile_bucket(value, cuts):
    if value is None:
        return "unknown"
    if len(cuts["cutpoints"]) != 2:
        raise ValueError("No known tertile cutpoints")
    lo, hi = cuts["cutpoints"]
    return "low" if value <= lo else "middle" if value <= hi else "high"


def dev_episode(episode, games):
    require_dev(episode["task_id"])
    authorize_task(episode, "diagnosis")
    task_id = episode["task_id"]
    if episode["split"] != "dev" or task_id not in games:
        raise ValueError("Episode is outside the frozen dev140 population")
    game = games[task_id]
    if episode.get("game_sha256") != game["game_sha256"] or episode.get("seed") != game["environment_seed"]:
        raise ValueError("Episode game hash or environment seed differs from frozen traits")


def known_outcome(episode):
    if episode.get("complete") is not True or episode.get("status") != "ok":
        return None
    value = episode.get("success")
    return value if type(value) is bool else None


def teacher_lengths(directory, games, inputs):
    """Only this trait may use policy outcomes. Verify wins against final audit blobs."""
    directory = Path(directory)
    result = {task: {"steps": None, "successful_trajectories": 0, "observed_trajectories": 0,
                     "shortest_episode_paths": []} for task in games}
    paths = sorted(directory.glob("episodes/*/episode.json"))
    if not paths:
        raise ValueError("Teacher directory has no dev trajectories")
    for name in ("provenance.json", "tasks.json", "summary.json"):
        if (directory / name).exists():
            value = inputs.read(directory / name)
            if name == "tasks.json":
                for task_id in value["task_ids"]:
                    require_dev(task_id)
                if set(value["task_ids"]) != set(games):
                    raise ValueError("Teacher task list differs from dev140")
    for path in paths:
        episode = inputs.read(path)
        dev_episode(episode, games)
        row = result[episode["task_id"]]
        row["observed_trajectories"] += 1
        if known_outcome(episode) is not True:
            continue
        final = episode["steps"][-1] if episode["steps"] else episode["initial"]
        ref = final["verification_ref"]
        verification = BlobStore(path.parent / "blobs").get(ref)
        inputs.add(path.parent / "blobs" / ref[7:9] / (ref[7:] + ".json"))
        audit = verification["audit"]
        if (digest(audit) != final["audit_sha256"] or audit.get("won") is not True
                or verification.get("fact_source") != "real_environment"):
            raise ValueError("Teacher success lacks matching real environment evidence")
        length = len(episode["steps"])
        row["successful_trajectories"] += 1
        if row["steps"] is None or length < row["steps"]:
            row.update(steps=length, shortest_episode_paths=[str(path)])
        elif length == row["steps"]:
            row["shortest_episode_paths"].append(str(path))
    return result


def coverage_counts(collection):
    receipts = collection["verified"]
    if collection["status"] != "complete" or len(receipts) != collection["K"]:
        raise ValueError("Coverage needs exactly K verified trajectories")
    counts = Counter()
    for receipt in receipts:
        refuse_sealed(receipt["task_id"])
        if not receipt["task_id"].startswith("train/") or receipt.get("won") is not True:
            raise ValueError("Coverage requires trusted executed train seeds")
        kind = parse_goal(receipt["task_id"]).task_type
        if kind != receipt["task_type"]:
            raise ValueError("Seed task type differs from its task ID")
        counts[kind] += 1
    return {kind: counts[kind] for kind in sorted(FAMILIES)}


def seed_coverage(root, manifest, freeze, inputs):
    result = {}
    for seed in (0, 1, 2):
        parent = Path(root) / f"seeds_k20_s{seed}"
        for k in (5, 10, 20):
            directory = parent if k == 20 else parent / "subsets" / f"k{k}"
            for folder in {directory, parent}:
                for name in ("manifest.json", "identity.json", "selection.json"):
                    inputs.add(folder / name)
            # Existing read-only verifier checks all receipts, successes and hashes,
            # including the shared source evidence of nested subsets.
            collection, selection, _ = load_verified_collection(directory, manifest, freeze)
            if selection["seed"] != seed or collection["K"] != k:
                raise ValueError("Seed collection does not match its K/support seed")
            counts = coverage_counts(collection)
            result[f"k{k}s{seed}"] = {"K": k, "support_seed": seed, "collection": str(directory),
                                     "counts_by_task_type": counts,
                                     "trusted_trajectories": len(collection["verified"]),
                                     "unique_instances": collection["unique_instances"],
                                     "task_groups": collection["task_groups"]}
    return result


def freeze_traits(path, body):
    value = dict(body, traits_sha256=digest(body))
    write_json(path, value, exclusive=True)
    return value


def load_traits(path):
    value = json.loads(Path(path).read_text())
    if digest({k: v for k, v in value.items() if k != "traits_sha256"}) != value.get("traits_sha256"):
        raise ValueError("Frozen traits hash mismatch")
    ids = value["task_ids"]
    if (value.get("schema_version") != 1 or value.get("split") != "dev" or len(ids) != 140
            or len(set(ids)) != 140 or set(ids) != set(value["games"])):
        raise ValueError("Expected frozen dev140 traits")
    for task_id in ids:
        require_dev(task_id)
    return value


def build_traits(*, config, tasks_path, teacher, verified_root, output, env_factory=Environment):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite frozen traits")
    for protected in (ROOT / "reports/runs", ROOT / "data/verified", Path(verified_root), Path(teacher),
                      Path(config["environment"]["data_root"])):
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Traits output cannot modify source evidence")
    inputs = Inputs()
    for name in ("manifest", "split_freeze"):
        inputs.add(config["paths"][name])
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    listing = inputs.read(tasks_path)
    for task_id in listing["task_ids"]:
        require_dev(task_id)
    if listing != build_dev140(manifest, freeze):
        raise ValueError("Task list differs from frozen dev140")
    inputs.add(ROOT / "docs/MECHANISM_PLAN.md")
    env_config = config["environment"]
    seed = env_config["seed"]
    games = {}
    env = env_factory(env_config["data_root"], seed=seed, split="dev",
                      max_episode_steps=env_config["max_episode_steps"])
    try:
        for task_id in listing["task_ids"]:
            task = resolve_task(manifest, freeze, task_id)
            authorize_task(task, "diagnosis")
            verify_task_files(env_config["data_root"], task)
            directory = Path(env_config["data_root"]) / task_id
            for name in task["files"]:
                inputs.add(directory / name)
            visible, _, _, _, audit = env.reset(task_id)  # Exactly one reset, no actions.
            if env.game_hash != task["game_sha256"]:
                raise ValueError("Reset game differs from manifest")
            row = initial_traits(task_id, visible["observation"], audit["internal_facts"])
            if row["task_type"] != task["task_type"]:
                raise ValueError("Goal ID and frozen game metadata disagree")
            row.update(game_sha256=env.game_hash, environment_seed=seed, group_id=task["group_id"],
                       initial_audit_sha256=digest(audit))
            games[task_id] = row
    finally:
        env.close()
    lengths = teacher_lengths(teacher, games, inputs)
    coverage = seed_coverage(verified_root, manifest, freeze, inputs)
    for task_id, row in games.items():
        row["minimum_length"] = lengths[task_id]["steps"]
        row["minimum_length_evidence"] = lengths[task_id]
        row["seed_coverage"] = {key: {"count": cov["counts_by_task_type"][row["task_type"]],
                                      "bucket": count_bucket(cov["counts_by_task_type"][row["task_type"]])}
                                for key, cov in coverage.items()}
    cuts = {trait: tertiles([r[trait] for r in games.values()]) for trait in ("room_size", "minimum_length")}
    for row in games.values():
        for trait, cut in cuts.items():
            row[trait + "_bucket"] = tertile_bucket(row[trait], cut)
    inputs.verify()
    settings = {"tasks": str(tasks_path), "teacher": str(teacher), "verified_root": str(verified_root),
                "environment": env_config, "paths": {k: config["paths"][k] for k in ("manifest", "split_freeze")},
                "environment_resets": len(games), "environment_steps": 0, "policy_calls": 0,
                "minimum_length_unit": "logged policy steps, len(episode.steps); shortest verified success",
                "audit_only": True}
    return freeze_traits(output, {"schema_version": 1, "split": "dev", "task_ids": listing["task_ids"],
                                  "unique_instances": len(games), "task_groups": listing["task_groups"],
                                  "dev140": listing, "buckets": BUCKETS, "tertiles": cuts,
                                  "coverage": coverage, "games": games, "provenance": provenance(settings, inputs)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--tasks", default="data/manifest/dev140.json")
    parser.add_argument("--teacher", default="reports/runs/teacher-dev140")
    parser.add_argument("--verified-root", default="data/verified")
    parser.add_argument("--output", default="data/manifest/dev140_traits.json")
    args = parser.parse_args(argv)
    result = build_traits(config=load_config(args.config), tasks_path=args.tasks, teacher=args.teacher,
                          verified_root=args.verified_root, output=args.output)
    print(f"{len(result['games'])} games; traits_sha256={result['traits_sha256']}")


if __name__ == "__main__":
    main()
