"""Seed-only ALFWorld task edits; validation evidence is never training data.

No environment or expert is imported here. All real execution is delegated to
``repair.synth_validation`` in a separate, offline CPU process. PDDL templates
and language templates are copied from the selected verified seeds only.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys


TASK_TYPES = (
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)
GROUP_RE = re.compile(r"(?:" + "|".join(TASK_TYPES) + r")-[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+-\d+")
TASK_RULE = re.compile(r'("task"\s*:\s*\[\s*\{\s*"rhs"\s*:\s*)("(?:\\.|[^"\\])*")')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def group_name(value):
    matches = GROUP_RE.findall(str(value))
    if len(set(matches)) != 1:
        raise ValueError("Expected exactly one supported ALFWorld task group")
    return matches[0]


def group_parts(value):
    return tuple(group_name(value).split("-"))


def section_span(problem, name):
    """Locate a balanced PDDL section, preserving its exact source text."""
    start = problem.index("(:" + name)
    depth = 0
    comment = False
    for index in range(start, len(problem)):
        char = problem[index]
        if char == "\n":
            comment = False
        if char == ";":
            comment = True
        if comment:
            continue
        depth += (char == "(") - (char == ")")
        if depth == 0:
            return start, index + 1
    raise ValueError("Unbalanced PDDL section")


def section(problem, name):
    start, end = section_span(problem, name)
    return problem[start:end]


def replace_section(problem, name, replacement):
    start, end = section_span(problem, name)
    return problem[:start] + replacement + problem[end:]


def init_facts(game):
    body = re.sub(r";[^\n]*", "", section(game["pddl_problem"], "init"))
    atoms = [tuple(x.split()) for x in re.findall(r"\(([^()]*)\)", body)]
    if any(not x or x[0].startswith(":") for x in atoms):
        raise ValueError("Unsupported init form")
    # Fail closed on numeric/nested initial expressions, rather than dropping them.
    rest = re.sub(r"\([^()]*\)", "", body[len("(:init"): -1])
    if rest.strip():
        raise ValueError("Non-atomic init is unsupported")
    return atoms


def fact_map(facts, predicate):
    return {f[1]: f[2] for f in facts if f[0] == predicate}


def problem_hash(game):
    return digest({"init": sorted(set(init_facts(game))),
                   "goal": re.findall(r"[^\s()]+|[()]", section(game["pddl_problem"], "goal"))})


def fingerprint(group, game):
    return {"group": group_name(group), "tuple": list(group_parts(group)),
            "problem_sha256": problem_hash(game)}


def task_text(game):
    match = TASK_RULE.search(game["grammar"])
    if not match:
        raise ValueError("Unsupported task grammar rule")
    text = json.loads(match[2])
    if not text.startswith("Your task is to: ") or "#" in text:
        raise ValueError("Task rule must be a literal instruction")
    return text


@dataclass
class Seed:
    task_id: str
    game: dict
    actions: list
    provenance: dict

    @property
    def group(self):
        return group_name(self.task_id)


def load_seeds(manifest_path, data_root):
    manifest_path, data_root = Path(manifest_path).resolve(), Path(data_root).resolve()
    manifest = json.loads(manifest_path.read_text())
    verified = manifest.get("verified")
    if not isinstance(verified, list) or len(verified) != manifest["K"]:
        raise ValueError("K must equal the verified trajectory count")
    collection = (manifest_path.parent / manifest.get("source_collection", {}).get("path", ".")).resolve()
    seeds = []
    for row in verified:
        if not row.get("success") or not row.get("won") or row.get("rejection"):
            raise ValueError("Unverified seed trajectory")
        task_id = row["task_id"]
        if not task_id.startswith("train/") or ".." in Path(task_id).parts:
            raise ValueError("Only native train seeds are allowed")
        game_path = (data_root / task_id / "game.tw-pddl").resolve()
        episode_dir = (collection / row["episode_dir"]).resolve()
        if not game_path.is_relative_to(data_root) or not episode_dir.is_relative_to(collection):
            raise ValueError("Seed path escapes its source")
        episode_path, events_path = episode_dir / "episode.json", episode_dir / "events.jsonl"
        for path in (episode_path, events_path):
            if row["episode_hashes"].get(path.name) != file_hash(path):
                raise ValueError("Seed receipt hash mismatch")
        episode = json.loads(episode_path.read_text())
        if not episode.get("complete") or not episode.get("success") or episode["task_id"] != task_id:
            raise ValueError("Seed episode is incomplete or unsuccessful")
        if episode["game_sha256"] != file_hash(game_path):
            raise ValueError("Seed game differs from the verified game")
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        actions = [event["action_parsed"] for event in events if event.get("environment_steps") == 1]
        if not actions or any(not isinstance(a, str) for a in actions):
            raise ValueError("Missing executed seed actions")
        seeds.append(Seed(task_id, json.loads(game_path.read_text()), actions, {
            "game_path": str(game_path), "game_sha256": file_hash(game_path),
            "episode_path": str(episode_path), "episode_sha256": file_hash(episode_path),
            "events_path": str(events_path), "events_sha256": file_hash(events_path),
            "environment_seed": episode["seed"], "action_count": len(actions),
        }))
    if len({s.task_id for s in seeds}) != len(seeds):
        raise ValueError("This factory requires distinct seed instances; retries are not new tasks")
    return sorted(seeds, key=lambda s: s.task_id), manifest


def identity_groups(value):
    """Read identity fields only; observations and hidden states are not inputs."""
    groups = set()
    identity = {"task_id", "task_ids", "group", "group_id", "group_ids", "task_groups",
                "task_name", "task_names", "task_name_exclusions"}
    if isinstance(value, list):
        for item in value:
            groups.update(identity_groups(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in identity:
                if isinstance(item, (str, list, dict)):
                    groups.update(GROUP_RE.findall(json.dumps(item)))
            elif isinstance(item, (dict, list)) and key not in {
                "observations", "events", "history", "facts", "internal_facts", "games", "source",
            }:
                groups.update(identity_groups(item))
    return groups


def freeze_heldout(data_root, main_checkout, repo_root, extra_paths=()):
    """Freeze identities before generating. Never open held-out games or states.

    Native valid_seen is excluded conservatively in full, as is valid_unseen.
    General training catalogs/seed selections are not held-out sets. Explicit
    r03 pools and method-v1 lists are conservatively excluded even when unlabeled.
    """
    groups, sources = set(), []
    for split in ("valid_seen", "valid_unseen"):
        directory = Path(data_root) / split
        if not directory.is_dir():
            raise ValueError("Missing native held-out split directory")
        names = sorted(p.name for p in directory.iterdir() if p.is_dir() and GROUP_RE.fullmatch(p.name))
        if not names:
            raise ValueError("Empty native held-out split")
        groups.update(names)
        sources.append({"path": str(directory.resolve()), "identity_only": True,
                        "sha256": digest(names), "groups": len(names)})
    main_checkout, repo_root = Path(main_checkout), Path(repo_root)
    paths = set(map(Path, extra_paths))
    for base in {main_checkout / "data/manifest", repo_root / "data/manifest"}:
        for path in base.glob("*.json"):
            if path.name.startswith("seeds_") or path.name in {"alfworld_manifest.json", "alfworld_split_freeze.json", "dev140_traits.json"}:
                continue
            paths.add(path)
    method = main_checkout / "reports/runs/method_v1"
    # Pruning execution stores avoids inspecting any held-out trajectories.
    prune = {"records", "episodes", "blobs", "teacher_blobs", "cache", "data", "hf", "calls", "locks",
             "checkpoints", "adapters", "server", "teacher_rate_limit", "coverage", "__pycache__"}
    for directory, dirs, files in os.walk(method):
        dirs[:] = sorted(d for d in dirs if d not in prune and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".json") and (Path(directory) == method or re.search(r"listing|manifest|splits", name, re.I)):
                paths.add(Path(directory) / name)
    for path in sorted(paths):
        value = json.loads(path.read_text())
        if path.name == "run_manifest.json":
            # A run's checkpoint embeds its *training* provenance, including K
            # seeds. Only the evaluated task selection defines held-out IDs.
            value = {key: value[key] for key in ("tasks", "task_ids", "group_ids", "cases") if key in value}
        if path.name in {"repair_v01_splits.json", "method_v1_splits.json"}:
            value = {"sets": value["sets"], "task_name_exclusions": value.get("task_name_exclusions", [])}
        found = identity_groups(value)
        groups.update(found)
        sources.append({"path": str(path.resolve()), "sha256": file_hash(path),
                        "groups": len(found), "identity_only": True})
    if not any(Path(s["path"]).name == "dev140.json" for s in sources):
        raise ValueError("dev140 split freeze is required")
    return {"groups": sorted(groups), "sources": sources,
            "sha256": digest(sorted(groups)), "policy": "identity exclusion only; all valid_seen and valid_unseen"}


def legal_pairs(seeds):
    evidence = defaultdict(set)
    for seed in seeds:
        facts = init_facts(seed.game)
        objects, receptacles = fact_map(facts, "objectType"), fact_map(facts, "receptacleType")
        for fact in facts:
            if fact[0] == "canContain":
                evidence[(fact[2], fact[1])].add(seed.task_id + ":canContain")
            if fact[0] == "inReceptacle" and fact[1] in objects and fact[2] in receptacles:
                evidence[(objects[fact[1]], receptacles[fact[2]])].add(seed.task_id + ":observed")
    return {pair: sorted(sources) for pair, sources in evidence.items()}


def relocate(game, obj, destination):
    """Atomic relocation; reject nesting/unknown dependencies instead of guessing."""
    facts = init_facts(game)
    membership = [f for f in facts if f[0] == "inReceptacle" and f[1] == obj]
    locations = [f for f in facts if f[0] == "objectAtLocation" and f[1] == obj]
    target_locations = [f[2] for f in facts if f[0] == "receptacleAtLocation" and f[1] == destination]
    if len(membership) != 1 or len(locations) != 1 or len(target_locations) != 1:
        raise ValueError("ambiguous_location")
    if membership[0][2] == destination:
        raise ValueError("unchanged_location")
    if ("full", destination) in facts:
        raise ValueError("full_receptacle")
    static_object = {"objectType", "pickupable", "moveable", "cleanable", "heatable", "coolable",
                     "sliceable", "toggleable", "isReceptacleObject", "isClean", "isHot", "isCool", "isSliced", "isOn", "isToggled"}
    for fact in facts:
        if obj in fact[1:] and fact[0] not in static_object | {"inReceptacle", "objectAtLocation"}:
            raise ValueError("unsupported_object_dependency:" + fact[0])
    receptacle_locations = fact_map(facts, "receptacleAtLocation")
    if locations[0][2] != receptacle_locations.get(membership[0][2]):
        raise ValueError("inconsistent_parent_location")
    changed = deepcopy(game)
    initial = section(game["pddl_problem"], "init")
    replacements = [(membership[0], ("inReceptacle", obj, destination)),
                    (locations[0], ("objectAtLocation", obj, target_locations[0]))]
    for old, new in replacements:
        pattern = r"\(\s*" + r"\s+".join(map(re.escape, old)) + r"\s*\)"
        initial, count = re.subn(pattern, "(" + " ".join(new) + ")", initial)
        if count != 1:
            raise ValueError("ambiguous_duplicate_fact")
    changed["pddl_problem"] = replace_section(game["pddl_problem"], "init", initial)
    changed["walkthrough"], changed["solvable"] = [], False
    return changed


def goal_edit(parent, template, object_type, receptacle_type):
    """Substitute only type constants in an exact seed goal and its task rule."""
    if parent.game["pddl_domain"] != template.game["pddl_domain"]:
        raise ValueError("template_domain_mismatch")
    task_type, old_obj, mrecep, old_recep, _ = group_parts(template.group)
    if mrecep != "None":
        raise ValueError("movable_receptacles_unsupported")
    goal = section(template.game["pddl_problem"], "goal")
    substitutions = {old_obj + "Type": object_type + "Type", old_recep + "Type": receptacle_type + "Type"}
    goal = re.sub(r"\b(?:" + "|".join(map(re.escape, substitutions)) + r")\b", lambda m: substitutions[m[0]], goal)
    text = task_text(template.game)
    words = {old_obj.lower(): object_type.lower(), old_recep.lower(): receptacle_type.lower()}
    text = re.sub(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", lambda m: words[m[0]], text)
    game = deepcopy(parent.game)
    game["pddl_problem"] = replace_section(game["pddl_problem"], "goal", goal)
    game["grammar"] = TASK_RULE.sub(lambda m: m[1] + json.dumps(text), game["grammar"], count=1)
    game["walkthrough"], game["solvable"] = [], False
    group = "-".join((task_type, object_type, "None", receptacle_type, group_parts(parent.group)[4]))
    return game, group


def candidate_record(parent, game, group, generator, edit):
    fp = fingerprint(group, game)
    ident = digest({"generator": generator, "parent": parent.task_id, "fingerprint": fp})[:24]
    return {"id": ident, "task_id": "train/synth/" + ident, "generator": generator,
            "parent_seed_task_id": parent.task_id, "fingerprint": fp,
            "task_text": task_text(game), "edit": edit,
            "pddl_diff": "".join(difflib.unified_diff(parent.game["pddl_problem"].splitlines(True),
                game["pddl_problem"].splitlines(True), fromfile="parent.pddl", tofile="synthetic.pddl")),
            "grammar_diff": "".join(difflib.unified_diff(parent.game["grammar"].splitlines(True),
                game["grammar"].splitlines(True), fromfile="parent.grammar", tofile="synthetic.grammar"))}


def generate_candidates(parent, seeds, generator, rng):
    facts = init_facts(parent.game)
    objects, receptacles = fact_map(facts, "objectType"), fact_map(facts, "receptacleType")
    pairs = legal_pairs(seeds)
    options = []
    if generator == "G1":
        goal_obj = group_parts(parent.group)[1] + "Type"
        for obj, typ in sorted(objects.items()):
            if typ != goal_obj or ("pickupable", obj) not in facts:
                continue
            old = fact_map(facts, "inReceptacle").get(obj)
            for recep, recep_type in sorted(receptacles.items()):
                if recep != old and (typ, recep_type) in pairs:
                    options.append((obj, recep, typ, recep_type))
        rng.shuffle(options)
        for obj, recep, typ, recep_type in options:
            edit = {"object": obj, "destination": recep, "pair_evidence": pairs[(typ, recep_type)]}
            try:
                game = relocate(parent.game, obj, recep)
            except ValueError as error:
                yield None, {"generator": generator, "parent_seed_task_id": parent.task_id,
                             "edit": edit, "rejection": str(error)}
                continue
            yield game, candidate_record(parent, game, parent.group, generator, edit)
    elif generator == "G2":
        templates = {}
        for seed in seeds:
            templates.setdefault(group_parts(seed.group)[0], seed)
        types = sorted({typ[:-4] for obj, typ in objects.items() if ("pickupable", obj) in facts})
        for task_type, template in sorted(templates.items()):
            for obj_type in types:
                instances = [o for o, typ in objects.items() if typ == obj_type + "Type" and ("pickupable", o) in facts]
                if task_type == "pick_two_obj_and_place" and len(instances) < 2:
                    continue
                requirement = {"pick_clean_then_place_in_recep": ("cleanable", "SinkBasinType"),
                               "pick_heat_then_place_in_recep": ("heatable", "MicrowaveType"),
                               "pick_cool_then_place_in_recep": ("coolable", "FridgeType")}.get(task_type)
                if requirement and (not any((requirement[0], obj) in facts for obj in instances)
                                    or requirement[1] not in receptacles.values()):
                    continue
                if task_type == "look_at_obj_in_light":
                    targets = sorted({typ[:-4] for obj, typ in objects.items()
                                      if typ in {"DeskLampType", "FloorLampType"} and ("toggleable", obj) in facts
                                      and obj in fact_map(facts, "inReceptacle")})
                else:
                    targets = sorted({typ[:-4] for typ in receptacles.values() if (obj_type + "Type", typ) in pairs
                                      and ("canContain", typ, obj_type + "Type") in facts})
                for target in targets:
                    options.append((template, obj_type, target))
        rng.shuffle(options)
        for template, obj_type, target in options:
            game, group = goal_edit(parent, template, obj_type, target)
            yield game, candidate_record(parent, game, group, generator, {
                "template_parent_seed_task_id": template.task_id,
                "object_type": obj_type, "receptacle_type": target,
                "template_goal_sha256": digest(section(template.game["pddl_problem"], "goal")),
            })
    else:
        raise ValueError("Unknown game generator")


def rejection_reason(record, heldout, seed_hashes, seen):
    if record["fingerprint"]["group"] in heldout:
        return "heldout_group"
    key = record["fingerprint"]["problem_sha256"]
    if key in seed_hashes:
        return "seed_duplicate"
    if key in seen:
        return "synthetic_duplicate"
    return None


def training_task_view(record):
    """Runtime task selection only; provenance, labels and PDDL stay audit-only."""
    if record.get("status") != "valid":
        raise ValueError("Only validated tasks can be exposed")
    return {k: record[k] for k in ("task_id", "task_text")}


def safe_output(out, data_root, main_checkout):
    out = Path(out).resolve()
    for protected in (Path(data_root).resolve(), Path(main_checkout).resolve()):
        if out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError("Output overlaps a read-only input")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be new or empty; no overwrite of previous evidence")
    return out


def summarize_records(records, seed_task_ids, generators):
    """Separate new instances, existing seed groups, and recovery situations."""
    stats = {}
    for generator in generators:
        rows = [r for r in records if r["generator"] == generator]
        stats[generator] = {"generated": len(rows), "valid": sum(r.get("status") == "valid" for r in rows),
                            "rejected_by_reason": dict(Counter(r["rejection"] for r in rows if r.get("rejection")))}
    seed_groups = {group_name(task_id) for task_id in seed_task_ids}
    games = [r for r in records if r.get("status") == "valid" and r["generator"] != "G3"]
    output_groups = {r["fingerprint"]["group"] for r in games}
    stats["accounting"] = {"K": len(seed_task_ids), "unique_seed_instances": len(set(seed_task_ids)),
        "seed_task_groups": len(seed_groups), "new_executable_tasks": len(games),
        "synthetic_task_groups": len(output_groups), "new_task_groups": len(output_groups - seed_groups),
        "recovery_prefixes": stats.get("G3", {}).get("valid", 0), "teacher_calls": 0,
        "teacher_tokens": 0, "training_exposure": 0,
        "environment_transitions": sum(r.get("validation", {}).get("environment_transitions", 0) for r in records)}
    return stats


def build_factory(*, seeds_path, data_root, out, main_checkout, repo_root,
                  per_seed=5, g3_per_seed=None, generators=("G1", "G2", "G3"), seed=0,
                  extra_heldout=(), validation_timeout=7200, validation_workers=4):
    if per_seed < 0 or (g3_per_seed is not None and g3_per_seed < 0):
        raise ValueError("Counts must be nonnegative")
    if not generators or not set(generators) <= {"G1", "G2", "G3"}:
        raise ValueError("Unsupported generators")
    if validation_workers < 1:
        raise ValueError("Validation workers must be positive")
    out = safe_output(out, data_root, main_checkout)
    seeds, source_manifest = load_seeds(seeds_path, data_root)
    freeze = freeze_heldout(data_root, main_checkout, repo_root, extra_heldout)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "split_freeze.json", freeze)
    rng = random.Random(seed)
    records, jobs, seen = [], [], set()
    seed_hashes = {problem_hash(s.game) for s in seeds}
    heldout = set(freeze["groups"])
    seed_by_id = {s.task_id: s for s in seeds}
    for parent in seeds:
        for generator in generators:
            count = g3_per_seed if generator == "G3" and g3_per_seed is not None else per_seed
            if generator == "G3":
                for index in range(count):
                    error_type = ("refused_action", "wrong_object", "useless_search", "wrong_receptacle")[index % 4]
                    ident = digest([generator, parent.task_id, index, seed])[:24]
                    record = {"id": ident, "generator": generator, "task_id": parent.task_id,
                              "parent_seed_task_id": parent.task_id, "fingerprint": fingerprint(parent.group, parent.game),
                              "pddl_diff": "", "task_text": task_text(parent.game), "error_type": error_type,
                              "prefix_index": index, "new_task": False}
                    records.append(record)
                    if parent.group in heldout:
                        record.update(status="rejected", rejection="heldout_group")
                    else:
                        jobs.append(dict(record))
                continue
            iterator = iter(generate_candidates(parent, seeds, generator, rng))
            for index in range(count):
                try:
                    game, record = next(iterator)
                except StopIteration:
                    records.append({"id": digest([parent.task_id, generator, index])[:24],
                                    "generator": generator, "parent_seed_task_id": parent.task_id,
                                    "status": "rejected", "rejection": "candidate_space_exhausted"})
                    continue
                record.setdefault("id", digest([parent.task_id, generator, index])[:24])
                record["new_task"] = True
                records.append(record)
                if game is None:
                    record["status"] = "rejected"
                    continue
                reason = rejection_reason(record, heldout, seed_hashes, seen)
                record["game_path"] = f"data_root/{record['task_id']}/game.tw-pddl"
                # Rejections retain the exact raw candidate, outside the accepted data root.
                if reason:
                    record["game_path"] = f"rejected/{record['id']}/game.tw-pddl"
                write_json(out / record["game_path"], game)
                record["game_sha256"] = file_hash(out / record["game_path"])
                if reason:
                    record.update(status="rejected", rejection=reason)
                    continue
                seen.add(record["fingerprint"]["problem_sha256"])
                jobs.append(dict(record))
    for job in jobs:
        parent = seed_by_id[job["parent_seed_task_id"]]
        job["parent_actions"] = parent.actions
        job["parent_game_sha256"] = parent.provenance["game_sha256"]
    provenance = {
        "effective_config": {"seeds": str(Path(seeds_path).resolve()), "data_root": str(Path(data_root).resolve()),
                             "out": str(out), "per_seed": per_seed, "g3_per_seed": g3_per_seed,
                             "generators": list(generators), "seed": seed, "max_steps": 40,
                             "validation_workers": validation_workers,
                             "per_seed_semantics": "candidate attempts, not guaranteed acceptances"},
        "K": len(seeds), "K_counts": "verified executed seed trajectories",
        "seed_unique_instances": len(seed_by_id), "seed_task_groups": len({s.group for s in seeds}),
        "source_manifest_sha256": file_hash(seeds_path), "split_freeze_sha256": freeze["sha256"],
        "seed_sources": {s.task_id: s.provenance for s in seeds},
        "failed_seed_attempts_excluded": len(source_manifest.get("attempts", [])) - len(seeds),
        "teacher": {"alias": None, "returned_model_id": None, "timestamp": None, "calls": 0,
                    "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
        "student_revision": None, "training_exposure": 0, "gpu_used": False,
        "code_sha256": {str(p.relative_to(repo_root)): file_hash(p) for p in [
            Path(__file__).resolve(), Path(__file__).with_name("synth_validation.py").resolve(),
            Path(repo_root) / "tools/synth_tasks.py", Path(repo_root) / "harness/alfworld_env/environment.py"]},
        "dependencies": {name: importlib.metadata.version(name) for name in ("alfworld", "textworld", "numpy")},
        "python": sys.version, "environment_version": "ALFWorld tw-pddl via unchanged Environment",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
    }
    write_json(out / "provenance.json", provenance)
    write_json(out / "attempts.before_validation.json", records)
    write_json(out / "validation_requests.json", {"jobs": jobs, "seed": seed,
               "workers": validation_workers, "data_root": str(Path(data_root).resolve())})
    # No live SDKs, credentials, or network access are passed to the validator.
    worker_env = {"PATH": os.defpath, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                  "PYTHONHASHSEED": "0", "PYTHONPATH": str(Path(repo_root) / "src") + os.pathsep + str(repo_root),
                  "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-m", "repair.synth_validation", "--out", str(out)],
                   cwd=out, env=worker_env, check=True, timeout=validation_timeout)
    for record in records:
        if record.get("status") == "rejected":
            continue
        result = json.loads((out / "validation" / (record["id"] + ".json")).read_text())
        record.update(result)
        if record["generator"] != "G3":
            path = out / record["game_path"]
            if record["status"] == "valid":
                game = json.loads(path.read_text())
                game["solvable"] = True
                write_json(path, game)
                record["validated_game_sha256"] = record["game_sha256"]
                record["game_sha256"] = file_hash(path)
                record["post_validation_metadata_change"] = "solvable=false -> true; PDDL/grammar unchanged; walkthrough empty"
            else:
                rejected_path = out / "rejected" / record["id"] / "game.tw-pddl"
                rejected_path.parent.mkdir(parents=True, exist_ok=True)
                path.rename(rejected_path)
                path.parent.rmdir()
                record["game_path"] = str(rejected_path.relative_to(out))
    stats = summarize_records(records, list(seed_by_id), generators)
    write_json(out / "manifest.json", {"schema_version": 1, "tasks": records,
        "provenance_path": "provenance.json", "split_freeze_path": "split_freeze.json",
        "witness_policy": "validation-only; never-for-training; use training_task_view whitelist"})
    write_json(out / "stats.json", stats)
    return stats
