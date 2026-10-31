"""E0.5: sealed V2 interchange -> inherited trajectory-SFT continuation.

Only this worktree's renderer, scorer, optimizer and adapter loader are used.
Evidence stays in audit artifacts; only validated public messages are encoded.
"""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import random
import re

from baselines.common.conversation import encode_messages, group_turns, sequence_rows
from baselines.common.schedule import rank_slots
from baselines.common.shared_history import pack_shared_history
from repair.artifacts import checkpoint_files, provenance, read_json, tokenizer_identity
from repair.labels import instance_ids
from repair.train_arms import inherited_recipe, load_existing_adapter
from repair.v2_inputs import ValidationError, require, inspect_inputs, input_hash_policy, FORBIDDEN_SETS
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.react import build_messages, parse_action
from sed.storage import digest, file_hash, write_json
from sed.visibility import visible_observation

SCHEMA = "v2-train-interchange/1"
ARMS = {
    "JOINT": {"joint_local", "joint_complete"},
    "JOINT_COMPLETE_ONLY": {"joint_complete"},
    "PREFIX_EXPERT": {"prefix_expert"},
    "OFFLINE": {"offline_teacher"},
    "CONTINUE": set(),
}
EXAMPLE_FIELDS = {"example_id", "arm", "source", "kind", "task_id", "group_id",
                  "task_type", "messages", "supervised_assistant_indices", "weight", "evidence"}
EVIDENCE_FIELDS = {"episode_ref", "step", "executor", "local_event", "episode_success",
                   "label_source", "label_check"}
LOCAL_EVENTS = {"target_seen", "target_taken", "hands_freed", "transform_done",
                "placed", "lamp_used_with_target", "task_success"}
EVIDENCE_HASH_FIELDS = {"public_prefix_sha256", "fact_state_sha256",
                        "facts_before_sha256", "facts_after_sha256"}
DECISION_EVIDENCE_FIELDS = EVIDENCE_HASH_FIELDS | {"local_events", "complete_success_eligible",
                                                "history_rewrites", "next_legal_target_take"}


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite JSON value")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_evidence(row):
    evidence = row["evidence"]
    optional = DECISION_EVIDENCE_FIELDS if row["kind"] == "decision" else set()
    require(isinstance(evidence, dict) and EVIDENCE_FIELDS <= evidence.keys()
            and evidence.keys() <= EVIDENCE_FIELDS | optional, "Invalid evidence schema")
    event, step = evidence["local_event"], evidence["step"]
    require(isinstance(evidence["episode_ref"], str) and evidence["episode_ref"]
        and (step is None if row["kind"] == "trajectory" else type(step) is int and step >= 0)
        and evidence["executor"] in ("student", "student_after_guidance", "teacher")
        and (event is None or isinstance(event, str) and event in LOCAL_EVENTS)
        and (row["kind"] != "trajectory" or event is None)
        and type(evidence["episode_success"]) in (bool, type(None))
        and evidence["label_source"] in ("student_thought", "teacher_rationale", "teacher_reply")
        and evidence["label_check"] == "passed", "Invalid or unchecked label evidence")
    for field in evidence.keys() & optional:
        value = evidence[field]
        if field in EVIDENCE_HASH_FIELDS:
            valid = isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
        elif field == "local_events":
            valid = (isinstance(value, list) and all(isinstance(v, str) and v in LOCAL_EVENTS for v in value)
                     and event == (value[0] if value else None))
        elif field == "complete_success_eligible":
            valid = type(value) is bool
        elif field == "history_rewrites":
            valid = isinstance(value, list) and all(isinstance(v, dict) for v in value)
        else:  # next_legal_target_take: int (not bool), or null; no range restriction.
            valid = value is None or type(value) is int
        require(valid, f"Invalid decision evidence: {field}")


def _read_bundle(bundle, theta0, files):
    """Validate byte seals and summaries before any tokenizer/model allocation."""
    root, manifest = bundle["root"], bundle["manifest"]
    fields = {"schema", "arm", "created", "code_revision", "inputs", "counts", "supervised_turns",
              "example_ids_sha256", "splits_sha256", "theta0", "training_exposure_tasks", "excluded"}
    require(isinstance(manifest, dict) and fields <= manifest.keys(), "Incomplete interchange manifest")
    require(manifest["schema"] == SCHEMA and manifest["arm"] in ARMS, "Unknown interchange schema/arm")
    require(all(isinstance(manifest[k], str) and manifest[k] for k in ("created", "code_revision")),
            "Missing interchange provenance")
    identity = manifest["theta0"]
    require(isinstance(identity, dict) and set(identity) == {"adapter_path", "adapter_weights_sha256"}
            and isinstance(identity["adapter_path"], str) and _sha(identity["adapter_weights_sha256"]),
            "Invalid theta0 identity")
    require(file_hash(Path(theta0) / "adapter_model.safetensors") == identity["adapter_weights_sha256"],
            "theta0 adapter hash differs from dataset manifest")
    require(file_hash(root / "examples.jsonl") == bundle["examples_sha256"], "examples.jsonl sha256 mismatch")
    rows = [_json(line) for line in (root / "examples.jsonl").read_text().splitlines()]
    require(all(isinstance(row, dict) and set(row) == EXAMPLE_FIELDS for row in rows),
            "Example schema must exactly match the interchange whitelist")
    ids = [row["example_id"] for row in rows]
    require(all(_sha(i) for i in ids) and ids == sorted(set(ids)), "Example IDs must be sorted and unique")
    require(manifest["example_ids_sha256"] == digest(ids), "Example ID seal mismatch")
    required_counts = {"by_" + k: dict(Counter(row[k] for row in rows))
                       for k in ("source", "kind", "task_type")}
    for row in rows:
        _validate_evidence(row)
    expected_counts = {**required_counts,
        "by_local_event": dict(Counter(event for row in rows if row["kind"] == "decision"
            for event in row["evidence"].get("local_events", [row["evidence"]["local_event"]])
            if event is not None)),
        "examples": len(rows), "task_groups": len({row["group_id"] for row in rows}),
        "unique_instances": len({row["task_id"] for row in rows})}
    counts = manifest["counts"]
    require(isinstance(counts, dict) and required_counts.keys() <= counts.keys()
            and counts.keys() <= expected_counts.keys(), "Invalid manifest counts keys")
    for name, value in counts.items():
        expected = expected_counts[name]
        valid_type = (isinstance(value, dict) and all(type(n) is int for n in value.values())
                      if isinstance(expected, dict) else type(value) is int)
        require(valid_type and value == expected, f"Manifest counts mismatch: {name}")
    require(all(isinstance(row["supervised_assistant_indices"], list) for row in rows),
            "Supervised indices must be a list")
    require(type(manifest["supervised_turns"]) is int and manifest["supervised_turns"] == sum(
        len(row["supervised_assistant_indices"]) for row in rows), "Supervised turn count mismatch")
    require(manifest["training_exposure_tasks"] == sorted({row["group_id"] for row in rows}),
            "Training exposure groups mismatch")
    require(isinstance(manifest["excluded"], list) and all(isinstance(item, dict)
        and set(item) == {"reason", "count"} and isinstance(item["reason"], str)
        and type(item["count"]) is int and item["count"] >= 0 for item in manifest["excluded"]),
        "Invalid excluded-attempt accounting")
    inputs = manifest["inputs"]
    require(isinstance(inputs, dict) and inputs, "Missing hashed interchange inputs")
    paths = []
    for name in inputs:
        # The shared preflight already verified every input in both bundles.
        path = files.relocation.path(name, root=root)
        paths.append(path.resolve())
    return {"root": root, "manifest": manifest, "rows": rows,
            "inputs": [root / "manifest.json", root / "examples.sha256", root / "examples.jsonl", *paths]}


def _members(value):
    require(isinstance(value, dict) and {"task_ids", "group_ids"} <= value.keys(),
            "Split set requires task_ids and group_ids")
    tasks, groups = value["task_ids"], value["group_ids"]
    require(all(isinstance(v, list) and all(isinstance(x, str) for x in v)
                and len(v) == len(set(v)) for v in (tasks, groups)), "Invalid split members")
    return set(tasks), set(groups)


def validate_example(row, arm, task, allowed, forbidden):
    """Validate decisions strictly; return informational trajectory turn counts."""
    require(row["arm"] == arm and row["source"] in ARMS[arm] | {"k20_replay"}, "Arm/source mismatch")
    require(row["kind"] in {"trajectory", "decision"}, "Unknown example kind")
    require(type(row["weight"]) in (int, float) and row["weight"] == 1.0,
            "Inherited SFT has no sample-weight hook: weights must all be 1.0")
    require(row["task_id"] == task["task_id"] and row["group_id"] == task["group_id"]
            and row["task_type"] == task["task_type"], "Task identity differs from frozen manifest")
    authorize_task(task, "generation")
    require(row["task_id"] in allowed[0] and row["group_id"] in allowed[1], "Task outside V2_pool/K20")
    require(row["task_id"] not in forbidden[0] and row["group_id"] not in forbidden[1],
            "Diagnostic/dev/test task or group cannot enter training")
    evidence = row["evidence"]
    _validate_evidence(row)
    require(row["example_id"] == digest([arm, row["source"], row["task_id"],
                                         evidence["episode_ref"], evidence["step"]]), "Canonical example ID mismatch")
    if row["source"] in {"k20_replay", "offline_teacher"}:
        require(row["kind"] == "trajectory", "Replay/offline supervision must be a trajectory")
    messages = row["messages"]
    require(isinstance(messages, list) and len(messages) >= 3 and len(messages) % 2 == 1,
            "Messages must end with an assistant reply")
    require(all(isinstance(m, dict) and set(m) == {"role", "content"}
                and isinstance(m["content"], str) and "Guidance:" not in m["content"] for m in messages),
            "Invalid message fields or Guidance: text")
    indices = row["supervised_assistant_indices"]
    expected = list(range(2, len(messages), 2)) if row["kind"] == "trajectory" else [len(messages) - 1]
    require(all(type(i) is int for i in indices) and indices == expected,
            "Supervised indices must be all assistant turns, or the last assistant for a decision")
    history = []
    diagnostics = {"inadmissible_supervised_turns": 0, "not_yet_seen_instance_supervised_turns": 0}
    for i in range(2, len(messages), 2):
        require(messages[i - 1]["role"] == "user" and messages[i]["role"] == "assistant",
                "Expected alternating user/assistant messages")
        visible = _json(messages[i - 1]["content"])
        require(isinstance(visible, dict) and set(visible) == {"observation", "admissible_commands"},
                "User fields exceed the public observation whitelist")
        require(build_messages(history, {}, visible_observation(visible)) == messages[:i],
                "Messages differ from the exact student prompt contract")
        action = parse_action(messages[i]["content"])
        if i in indices:
            admissible = action in visible["admissible_commands"]
            # Reuse this tree's M0c syntax check (step/turn numbers are not objects).
            prefix_refs = instance_ids("\n".join(m["content"] for m in messages[:i]))
            grounded = instance_ids(messages[i]["content"]) <= prefix_refs
            if row["kind"] == "decision":
                require(admissible, "Supervised action is not admissible")
                require(grounded, "Label cites an unseen instance ID")
            else:
                # Rule 5 preserves baseline trajectories, including refused actions
                # and ungrounded Thoughts. Count turns without changing their bytes.
                diagnostics["inadmissible_supervised_turns"] += int(not admissible)
                diagnostics["not_yet_seen_instance_supervised_turns"] += int(not grounded)
        history.append({"visible": visible, "reply": messages[i]["content"]})
    return diagnostics


def read_dataset(directory, *, theta0, config, splits=None, repair_splits=None, dev140=None,
                 relocate=(), _inspection=None):
    """No eval-side imports. Both inline replay and a sealed replay/ bundle work."""
    inspection = _inspection or inspect_inputs(directory, theta0=theta0, config=config,
        splits=splits, repair_splits=repair_splits, dev140=dev140, relocate=relocate, runtime=False)
    files = inspection["files"]
    require(files.all_hashes_verified, "Dataset requires a completed input hash verification")
    theta0, config, splits = (inspection[k] for k in ("theta0", "config", "splits"))
    main = _read_bundle(inspection["bundles"][0], theta0, files)
    bundles = [main]
    if len(inspection["bundles"]) > 1:
        bundles.append(_read_bundle(inspection["bundles"][1], theta0, files))
        require(all(row["source"] == "k20_replay" for row in bundles[1]["rows"]),
                "replay/ must contain only k20_replay examples")
    if splits is None:
        candidates = [p for p in main["inputs"] if p.name == "method_v1_splits.json"]
        require(len(candidates) == 1, "Provide --splits or a hashed method_v1_splits.json input")
        splits = candidates[0]
    splits = Path(splits).resolve()
    split_doc = read_json(splits)
    for bundle in bundles:
        require(bundle["manifest"]["splits_sha256"] == file_hash(splits), "Split freeze sha256 mismatch")
        require(splits in bundle["inputs"], "Split freeze must be an interchange input")
    frozen, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    for key, value in (("manifest_sha256", frozen), ("split_freeze_sha256", freeze)):
        if key in split_doc:
            require(split_doc[key] == digest(value), "Method split differs from configured manifest/freeze")
    sets = dict(split_doc.get("sets", {}))
    sources = inspection["split_sources"]
    if sources:
        repair_sets = read_json(sources["repair_splits"]).get("sets", {})
        for name in ("U_mech", "D_probe", "D_unseen_quick"):
            require(name in repair_sets, f"Repair split source is missing {name}")
            if name in sets:
                require(_members(sets[name]) == _members(repair_sets[name]), "Conflicting split members")
            sets[name] = repair_sets[name]
        dev = read_json(sources["dev140"])
        require(isinstance(dev.get("task_ids"), list) and all(isinstance(t, str) for t in dev["task_ids"])
                and len(dev["task_ids"]) == len(set(dev["task_ids"])), "Invalid dev140 task_ids")
        # dev140 records task IDs only. Resolve groups through the frozen task
        # manifest, never guess them or treat a missing group list as empty.
        groups = sorted({resolve_task(frozen, freeze, t)["group_id"] for t in dev["task_ids"]})
        if "group_ids" in dev:
            require(_members(dev)[1] == set(groups), "dev140 groups differ from frozen manifest")
        dev_members = {"task_ids": dev["task_ids"], "group_ids": groups}
        if "dev140" in sets:
            require(_members(sets["dev140"]) == _members(dev_members), "Conflicting split members")
        sets["dev140"] = dev_members
    require("V2_pool" in sets and all(name in sets for name in FORBIDDEN_SETS),
            "Method split freeze must include V2_pool and all diagnostic exclusions")
    pool = _members(sets["V2_pool"])
    forbidden = (set(), set())
    for name in FORBIDDEN_SETS:
        members = _members(sets[name])
        forbidden[0].update(members[0]); forbidden[1].update(members[1])
    inputs = [*[p for bundle in bundles for p in bundle["inputs"]], *sources.values()]
    if "K20" in sets:
        replay = _members(sets["K20"])
    else:
        # The contract permits the original K20 view instead of a split entry.
        view, collection = (inspection[k] for k in ("training_view", "training_collection"))
        require(collection["K"] == 20, "Expected original K20 view")
        replay = ({r["task_id"] for r in collection["verified"]}, {r["group_id"] for r in collection["verified"]})
        inputs.append(view / "manifest.json")
    require(not (pool[0] & replay[0] or pool[1] & replay[1]), "V2_pool overlaps K20")
    require(not ((pool[0] | replay[0]) & forbidden[0] or (pool[1] | replay[1]) & forbidden[1]),
            "Training split overlaps a diagnostic group")
    rows, origins, trajectory_diagnostics = [], [], []
    for bundle in bundles:
        for row in bundle["rows"]:
            task = resolve_task(frozen, freeze, row["task_id"])
            diagnostics = validate_example(row, bundle["manifest"]["arm"], task,
                                           replay if row["source"] == "k20_replay" else pool, forbidden)
            if row["kind"] == "trajectory":
                trajectory_diagnostics.append({"example_id": row["example_id"], **diagnostics})
            rows.append(row)
            origins.append(str(bundle["root"] / "examples.jsonl"))
    require(rows, "Empty training dataset")
    pools = training_pools(rows, arm=main["manifest"]["arm"],
                           root_count=len(main["rows"]) if len(bundles) > 1 else None)
    return {"arm": main["manifest"]["arm"], "rows": rows, "origins": origins,
            "pools": pools,
            "trajectory_diagnostics": trajectory_diagnostics,
            "manifests": [b["manifest"] for b in bundles],
            "relocation": files.report(),
            "inputs": sorted(set([*inputs, splits, Path(config["paths"]["manifest"]),
                                  Path(config["paths"]["split_freeze"])])),
            "theta0_sha256": main["manifest"]["theta0"]["adapter_weights_sha256"]}


def encode_example(row, tokenizer, *, max_length=32768):
    # validate_example has checked the assistant indices. Retain every public
    # history turn, but tokenize only the requested supervision targets.
    indices = [(i - 2) // 2 for i in row["supervised_assistant_indices"]]
    encoded = encode_messages(row["messages"], tokenizer, task_id=row["task_id"],
                              max_length=max_length, packing="turn", turns=set(indices))
    by_index = {turn["label_provenance"]["turn_index"]: turn for turn in encoded["turns"]}
    turns = [by_index[index] for index in indices]
    result = group_turns(turns)
    result["label_provenance"] = {"example_id": row["example_id"], "task_id": row["task_id"],
        "group_id": row["group_id"], "source": row["source"], "kind": row["kind"], "training_eligible": True}
    return result


def _replay_key(row):
    # IDs contain arm: exclude arm/ID so identical replay content orders identically.
    return digest({k: v for k, v in row.items() if k not in {"arm", "example_id"}})


def training_pools(rows, *, arm, root_count=None):
    if arm == "CONTINUE":
        require(all(r["source"] == "k20_replay" for r in rows), "CONTINUE can only train original K20 replay")
        # Keep both copies and their origins: root supplies the arm share,
        # replay/ supplies the common replay share. A root-only bundle keeps
        # the existing independent epoch streams over the same K20 pool.
        pools = {"new": list(range(len(rows) if root_count is None else root_count)),
                 "replay": list(range(0 if root_count is None else root_count, len(rows)))}
    else:
        pools = {"new": [i for i, r in enumerate(rows) if r["source"] != "k20_replay"],
                 "replay": [i for i, r in enumerate(rows) if r["source"] == "k20_replay"]}
    for name, indices in pools.items():
        require(len({_replay_key(rows[i]) for i in indices}) == len(indices),
                f"Duplicate {name} trajectories; a row in both pools is only allowed for CONTINUE")
        indices.sort(key=lambda i: _replay_key(rows[i]) if name == "replay" or arm == "CONTINUE"
                     else rows[i]["example_id"])
    return pools


def batch_schedule(rows, *, updates, batch_examples, replay_ratio, seed, arm, pools=None):
    require(type(updates) is int and updates > 0 and type(batch_examples) is int and batch_examples > 0,
            "Positive integer updates/batch-examples required")
    require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer")
    require(math.isfinite(replay_ratio) and 0 <= replay_ratio <= 1, "Replay ratio must be in [0,1]")
    quota = Decimal(str(replay_ratio)) * batch_examples
    require(quota == int(quota), "B * replay-ratio must be integral; no silent ratio rounding")
    replay_n = int(quota)
    pools = training_pools(rows, arm=arm) if pools is None else pools
    require((replay_n == 0 or pools["replay"]) and (replay_n == batch_examples or pools["new"]),
            "Requested mixture has an empty pool")
    rngs = {name: random.Random(digest(["v2-epochs", seed, name])) for name in pools}
    slots_rng = random.Random(digest(["v2-slots", seed]))
    remaining = {name: [] for name in pools}
    epochs = {name: -1 for name in pools}
    batches = []
    for update in range(1, updates + 1):
        slots = ["replay"] * replay_n + ["new"] * (batch_examples - replay_n)
        slots_rng.shuffle(slots)
        selections = []
        for pool in slots:
            if not remaining[pool]:
                remaining[pool] = list(pools[pool])
                rngs[pool].shuffle(remaining[pool])
                epochs[pool] += 1
            index = remaining[pool].pop()
            row = rows[index]
            selections.append({"index": index, "pool": pool, "pool_epoch": epochs[pool],
                "example_id": row["example_id"], "source": row["source"], "kind": row["kind"],
                "replay_key": _replay_key(row) if row["source"] == "k20_replay" else None})
        batches.append({"update": update, "epoch": epochs["new"], "indices": [s["index"] for s in selections],
                        "selections": selections})
    return batches


def accounting(examples, batches, *, world_size=1):
    """Physical context positions include zero-loss rank padding; targets do not."""
    lengths = [len(pack_shared_history(e)["input_ids"]) for e in examples]
    targets = [sum(r["supervised_tokens"] for r in sequence_rows(e)) for e in examples]
    by_source, by_kind = Counter(), Counter()
    tasks, groups = set(), set()
    counts = {"optimizer_updates": 0, "processed_context_tokens": 0, "supervised_target_tokens": 0,
              "examples_seen": 0, "logical_context_tokens": 0, "zero_weight_slots": 0,
              "examples_seen_by_source": {}, "examples_seen_by_kind": {},
              "unique_instances_seen": 0, "task_groups_seen": 0}
    result = {0: deepcopy(counts)}
    for batch in batches:
        indices = batch["indices"]
        for index in indices:
            info = examples[index]["label_provenance"]
            by_source[info["source"]] += 1; by_kind[info["kind"]] += 1
            tasks.add(info["task_id"]); groups.add(info["group_id"])
            counts["supervised_target_tokens"] += targets[index]
            counts["logical_context_tokens"] += lengths[index]
        slots = rank_slots(list(range(len(indices))), world_size)
        counts["processed_context_tokens"] += sum(lengths[indices[i if i is not None else 0]]
                                                   for slot in slots for i in slot)
        counts["zero_weight_slots"] += sum(i is None for slot in slots for i in slot)
        counts.update(optimizer_updates=batch["update"], examples_seen=counts["examples_seen"] + len(indices),
                      examples_seen_by_source=dict(by_source), examples_seen_by_kind=dict(by_kind),
                      unique_instances_seen=len(tasks), task_groups_seen=len(groups))
        result[batch["update"]] = deepcopy(counts)
    return result


def continuation_recipe(theta0, config, snapshot, updates):
    inherited = inherited_recipe(theta0, config, snapshot)
    recipe = inherited["recipe"]
    require(recipe["optimizer"] == {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8}
        and recipe["compute_dtype"] == recipe["frozen_base_dtype"] == "bfloat16"
        and recipe["adapter_master_dtype"] == "float32" and recipe["attention"] == "flex_attention",
        "Unsupported inherited optimizer/precision/attention; cannot silently change recipe")
    inherited["scheduler_policy"].update(total_updates=updates,
        decision=f"theta0 schedule ended; restart same shape over {updates} updates in every arm")
    return inherited


def publish_checkpoint(model, output, prov, state, counts, inherited, *, dry_run=False):
    from baselines.common.fsdp_train import export_adapter
    update = state["updates"]
    destination = Path(output) / f"update-{update:02d}"
    destination.mkdir()
    write_json(destination / "provenance.json", prov, exclusive=True)
    saved = export_adapter(model, destination / "checkpoint", prov, {
        **state, **counts, "epochs": 1, "adaptation": inherited["recipe"]["adaptation"],
        "packing": "turn", "gradient_clip": 1.0,
        "loss_reduction": "per_example_mean_then_global_batch_mean",
        "compute_dtype": "float32" if dry_run else "bfloat16", "master_dtype": "float32",
        "trainable_parameter_count": sum(inherited["parameters"].values()),
        "actual_trainable_parameter_names": sorted(inherited["parameters"]),
        "actual_trainable_module_names": inherited["trainable_modules"],
        "optimizer_state": inherited["optimizer_state"], "scheduler_policy": inherited["scheduler_policy"],
        "dry_run": dry_run, "synthetic": dry_run})
    item = {"update": update, "status": "synthetic_saved" if dry_run else "saved", **counts,
            "path": str((destination / "checkpoint").resolve()), "sha256": checkpoint_files(destination / "checkpoint"),
            "provenance_sha256": file_hash(destination / "provenance.json"), "trainer_state": saved}
    write_json(destination / "manifest.json", item, exclusive=True)
    return item


class FakeTokenizer:
    """Explicit CPU byte tokenizer; never represented as the E4B tokenizer."""
    all_special_ids = [106]
    chat_template = "v2-fake-byte-OFF/1"

    def encode(self, text, **_):
        parts = text.split("<turn|>")
        ids = []
        for i, part in enumerate(parts):
            if i:
                ids.append(106)
            ids.extend(b + 200 for b in part.encode())
        return ids

    def decode(self, ids, **_):
        parts, pending = [], []
        for token in ids:
            if token == 106:
                parts.append(bytes(pending).decode()); pending = []
                parts.append("<turn|>")
            else:
                pending.append(token - 200)
        return "".join(parts) + bytes(pending).decode()

    def apply_chat_template(self, messages, *, add_generation_prompt, **_):
        from train.dataset import EMPTY_THOUGHT
        text = "".join(m["role"] + "\n" + m["content"].strip()
                       + ("<turn|>\n" if m["role"] == "assistant" else "\n") for m in messages)
        return text + ("assistant\n" + EMPTY_THOUGHT if add_generation_prompt else "")


def fake_model(vocab_size=512):
    """Tiny synthetic projection, actual shared scorer/AdamW/backward/export on CPU."""
    import torch
    from types import SimpleNamespace
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.ones(1, 1))
            self.lora_B = torch.nn.Parameter(torch.zeros(vocab_size, 1))
            text = SimpleNamespace(model_type="gemma4_text", attention_dropout=0,
                                   layer_types=["full_attention"], _attn_implementation="flex_attention")
            self.config = SimpleNamespace(get_text_config=lambda: text)

        def forward(self, input_ids, logits_to_keep, **_):
            logits = (self.lora_B @ self.lora_A).flatten()
            return SimpleNamespace(logits=logits.expand(len(input_ids), len(logits_to_keep), -1))

        def save_pretrained(self, destination, *, safe_serialization):
            from safetensors.torch import save_file
            require(safe_serialization, "Fake export must use safetensors")
            write_json(Path(destination) / "adapter_config.json", {
                "peft_type": "LORA", "r": 1, "lora_alpha": 1, "lora_dropout": 0,
                "target_modules": ["fake_projection"], "base_model_name_or_path": "synthetic/v2-tiny-fake",
                "revision": "fake-only", "bias": "none"})
            save_file({"base_model.model.fake_projection.lora_A.weight": self.lora_A.detach().contiguous(),
                       "base_model.model.fake_projection.lora_B.weight": self.lora_B.detach().contiguous()},
                      str(Path(destination) / "adapter_model.safetensors"))
    return Tiny()


def fake_scoring_examples(examples):
    """Bijectively compact real IDs for the tiny model, after production encoding.

    Lengths, masks, boundaries and all token distinctions are preserved. The
    original encoding is hashed separately; this synthetic vocabulary is never
    used for production training or token-length validation.
    """
    vocabulary = sorted({token for example in examples for turn in sequence_rows(example)
                         for token in turn["input_ids"]})
    mapping = {token: index for index, token in enumerate(vocabulary)}
    result = deepcopy(examples)
    for example in result:
        for turn in sequence_rows(example):
            turn["input_ids"] = [mapping[token] for token in turn["input_ids"]]
            turn["labels"] = [-100 if token == -100 else mapping[token] for token in turn["labels"]]
    return result, vocabulary


@input_hash_policy
def run_arm(dataset, config, output, *, theta0, snapshot, updates, batch_examples,
            replay_ratio, checkpoints, seed=1, gpus=1, dry_run=False, splits=None,
            repair_splits=None, dev140=None, inputs=(), relocate=(), dry_run_tokenizer="bytes",
            full_rehash=False):
    """Continuation on 1 GPU or the existing torchrun replicated-LoRA route."""
    require(gpus in (1, 2), "gpus must be 1 or 2")
    require(dry_run_tokenizer in ("bytes", "snapshot"), "Unknown dry-run tokenizer")
    require(type(updates) is int and updates > 0, "Positive integer updates required")
    require(all(type(n) is int for n in checkpoints), "Checkpoints must be integer update counts")
    checkpoints = sorted(set([0, *checkpoints]))
    require(all(type(n) is int and 0 <= n <= updates for n in checkpoints), "Checkpoint outside update range")
    inspection = inspect_inputs(dataset, theta0=theta0, snapshot=snapshot, config=config,
        splits=splits, repair_splits=repair_splits, dev140=dev140, inputs=inputs, relocate=relocate,
        full_rehash=full_rehash)
    theta0, snapshot, config = (inspection[k] for k in ("theta0", "snapshot", "config"))
    dataset = inspection["files"].relocation.dataset
    data = read_dataset(dataset, theta0=theta0, config=config, _inspection=inspection)
    inherited = continuation_recipe(theta0, config, snapshot, updates)
    require(inherited["identity"]["adapter_weights_sha256"] == data["theta0_sha256"], "theta0 changed during validation")
    from train.dataset import load_tokenizer
    byte_tokenizer = dry_run and dry_run_tokenizer == "bytes"
    tokenizer = FakeTokenizer() if byte_tokenizer else load_tokenizer(snapshot)
    examples = [encode_example(row, tokenizer, max_length=inherited["recipe"]["max_length"]) for row in data["rows"]]
    batches = batch_schedule(data["rows"], updates=updates, batch_examples=batch_examples,
                             replay_ratio=replay_ratio, seed=seed, arm=data["arm"], pools=data["pools"])
    scoring_examples, fake_vocabulary = (fake_scoring_examples(examples)
        if dry_run and not byte_tokenizer else (examples, None))
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                         (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    require(world == (1 if dry_run else gpus), "Launch with the requested GPU rank count")
    counts = accounting(examples, batches, world_size=world)
    output = Path(output)
    require(not any(output.resolve().is_relative_to(Path(p).resolve()) for p in (dataset, theta0, snapshot)),
            "Output must be outside the sealed dataset, theta0 and base snapshot")
    require(not output.exists() and not output.is_symlink(), "Output already exists")
    import torch
    import torch.distributed as dist
    from baselines.common.fsdp_train import train_loop, rank_zero_data
    from transformers import set_seed
    from datetime import timedelta
    if not dry_run:
        require(torch.cuda.is_available(), "Production requires allocated GPUs; use --dry-run on CPU")
        torch.cuda.set_device(local)
    device = torch.device("cpu" if dry_run else f"cuda:{local}")
    group = None
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        if world > 1:
            group = dist.new_group(backend="gloo", timeout=timedelta(hours=4))
        set_seed(seed + rank)
        model = (fake_model(len(fake_vocabulary) if fake_vocabulary is not None else 512) if dry_run
                 else load_existing_adapter(snapshot, theta0, device, inherited["recipe"]))
        actual = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
        real_inherited = deepcopy(inherited)
        if dry_run:
            inherited.update(parameters=actual, trainable_modules=["fake_projection"],
                             optimizer_state={"mode": "synthetic_fresh_AdamW", "path": None, "sha256": None})
        else:
            require(actual == inherited["parameters"], "Trainable adapter tensors differ from theta0")
            if world > 1:
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        dist.broadcast(parameter.data, src=0)
        opt = inherited["optimizer_state"]
        optimizer = torch.load(opt["path"], map_location="cpu", weights_only=True) if opt["path"] else None
        if optimizer is not None and "optimizer" in optimizer:
            if "trainable" in optimizer:
                from train.optimizer_state import assert_trainable_equal
                require(list(optimizer["trainable"]) == list(actual), "Optimizer parameter order differs")
                assert_trainable_equal(model, optimizer["trainable"])
            optimizer = optimizer["optimizer"]

        def record():
            output.mkdir(parents=True, exist_ok=False)
            effective = deepcopy(config)
            effective["sft"] = deepcopy(inherited["sft"])
            effective["v2_training"] = {"arm": data["arm"], "updates": updates, "batch_examples": batch_examples,
                "replay_ratio": replay_ratio, "seed": seed, "requested_gpus": gpus, "world_size": world,
                "dry_run": dry_run, "dry_run_tokenizer": dry_run_tokenizer if dry_run else None,
                "full_rehash": inspection["files"].full_rehash,
                "inherited": real_inherited, "schedule_sha256": digest(batches),
                "loss_normalization": "per_example_mean_then_global_batch_mean",
                "data_parallel": "replicated_lora_gradient_mean_before_clip"}
            if dry_run:
                # Never label a fake artifact as a continuation of the real base.
                fake_base = output / "fake-base"; fake_base.mkdir()
                write_json(fake_base / "config.json", {"synthetic": True})
                effective["models"]["student"] = {"requested_repo": "synthetic/v2-tiny-fake", "revision": "fake-only"}
                effective["sft"] = {"rank": 1, "alpha": 1, "dropout": 0, "targets": ["fake_projection"],
                    "packing": "turn", "base_snapshot": {"path": str(fake_base.resolve()),
                    "metadata_sha256": {"config.json": file_hash(fake_base / "config.json")}}}
            prov = provenance(output, effective, command="v2_train", inputs=inspection["files"].provenance_inputs([
                *data["inputs"], Path(__file__), Path("tools/train_v2_arm.py")]),
                details={"relocation": data["relocation"]})
            zero = {"update": 0, "status": "theta0_reference", "path": str(Path(theta0).resolve()),
                    "sha256": inherited["files"], **counts[0]}
            manifest = {"schema": "v2-training-run/1", "status": "running", "dry_run": dry_run,
                "dry_run_tokenizer": dry_run_tokenizer if dry_run else None,
                "arm": data["arm"], "theta0": real_inherited["identity"], "provenance_sha256": digest(prov),
                "input_manifests": data["manifests"], "schedule_sha256": digest(batches),
                "relocation": data["relocation"],
                "trajectory_diagnostics": data["trajectory_diagnostics"],
                "encoded_examples_sha256": digest(examples), "tokenizer":
                    {"synthetic": True, "chat_template_sha256": digest(tokenizer.chat_template)} if byte_tokenizer
                    else {"synthetic": False, **tokenizer_identity(snapshot, tokenizer)},
                "fake_scoring_vocabulary": fake_vocabulary,
                "scoring_examples_sha256": digest(scoring_examples),
                "optimizer_state": opt, "scheduler_policy": inherited["scheduler_policy"],
                "trainable_parameter_count": sum(actual.values()), "actual_trainable_parameter_names": sorted(actual),
                "actual_trainable_module_names": inherited["trainable_modules"],
                "unique_instances": len({r["task_id"] for r in data["rows"]}),
                "task_groups": len({r["group_id"] for r in data["rows"]}),
                "pool_inventory": {name: {
                    "examples": len(pool), "unique_instances": len({r["task_id"] for r in pool}),
                    "task_groups": len({r["group_id"] for r in pool}),
                    "episode_references": len({r["evidence"]["episode_ref"] for r in pool}),
                    "by_source": dict(Counter(r["source"] for r in pool)),
                    "by_kind": dict(Counter(r["kind"] for r in pool))}
                    for name, indices in data["pools"].items()
                    for pool in [[data["rows"][i] for i in indices]]},
                "K_trusted_seed_trajectories": 20, "training_exposure": 0,
                "teacher_calls_this_run": 0, "environment_transitions_this_run": 0,
                "checkpoints": [zero], "planned_accounting": counts,
                "planned_checkpoints": checkpoints, "loss_curve": "updates.jsonl"}
            write_json(output / "batches.json", batches, exclusive=True)
            write_json(output / "example_sources.json", [{"file": origin, **{k: row[k] for k in
                ("example_id", "arm", "source", "kind", "task_id", "group_id", "evidence")}}
                for origin, row in zip(data["origins"], data["rows"])], exclusive=True)
            write_json(output / "manifest.json", manifest, exclusive=True)
            return prov, manifest
        prov, manifest = rank_zero_data(rank, world, record, group)

        def save(current_model, state):
            n = state["updates"]
            require(state["examples_seen"] == counts[n]["examples_seen"]
                and state["supervised_tokens_seen"] == counts[n]["supervised_target_tokens"]
                and state["physical_policy_forward_tokens"] == counts[n]["processed_context_tokens"],
                "Actual training accounting differs from the fixed schedule")
            if n not in checkpoints:
                return
            def publish():
                with torch.random.fork_rng(devices=[] if dry_run else [local]):
                    item = publish_checkpoint(current_model, output, prov, state, counts[n], inherited, dry_run=dry_run)
                manifest["checkpoints"].append(item)
                manifest["synthetic_training_exposure" if dry_run else "training_exposure"] = state["examples_seen"]
                write_json(output / "manifest.json", manifest)
            rank_zero_data(rank, world, publish, group)

        recipe = inherited["recipe"]
        with nullcontext() if dry_run else torch.autocast("cuda", dtype=torch.bfloat16):
            state = train_loop(model, scoring_examples, output, objective="sft", learning_rate=recipe["learning_rate"],
                scheduler=recipe["scheduler"], warmup_ratio=recipe["warmup_ratio"], warmup_steps=recipe["warmup_steps"],
                original_batch=batch_examples, seed=seed, weight_decay=recipe["weight_decay"],
                packing=recipe["packing"], scoring=recipe["scoring"], flex_kernel_options=recipe.get("flex_kernel_options"),
                rank=rank, world_size=world, device=device, replicated_lora=True,
                fixed_batches=batches, optimizer_state=optimizer, checkpoint_callback=save)
        if rank == 0:
            manifest.update(status="synthetic_complete" if dry_run else "complete", trainer_state=state,
                            accounting=counts[updates])
            manifest["synthetic_training_exposure" if dry_run else "training_exposure"] = state["examples_seen"]
            manifest["artifacts_sha256"] = {name: file_hash(output / name) for name in
                ("provenance.json", "batches.json", "example_sources.json", "updates.jsonl")}
            write_json(output / "manifest.json", manifest)
            return manifest
    finally:
        if world > 1:
            dist.destroy_process_group()
