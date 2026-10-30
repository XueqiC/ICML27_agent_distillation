"""E0.5 contract tests: sealed synthetic inputs, tiny CPU model, no network/API."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import pytest

from repair import v2_train as v2
from sed.manifest import build_manifest, freeze_splits
from sed.react import build_messages
from sed.storage import canonical_bytes, digest, file_hash, write_json
from test_repair_e2_reader import adapter_fixture


def example(tid, source, kind, arm="JOINT", step=...):
    if step is ...:
        step = None if kind == "trajectory" else 1
    visible = {"observation": "You see apple 1.", "admissible_commands": ["look", "take apple 1"]}
    first = "Thought: Check apple 1.\nAction: look"
    messages = build_messages([{"visible": visible, "reply": first}], {}, visible)
    messages.append({"role": "assistant", "content": "Thought: Inspect once more.\nAction: look"})
    evidence = {"episode_ref": f"records/{tid.replace('/', '-')}.json", "step": step,
                "executor": "teacher" if source in ("offline_teacher", "k20_replay") else "student_after_guidance",
                "local_event": None if kind == "trajectory" else "target_seen", "episode_success": True,
                "label_source": "teacher_reply" if source in ("offline_teacher", "k20_replay") else "teacher_rationale",
                "label_check": "passed"}
    return {"example_id": digest([arm, source, tid, evidence["episode_ref"], step]),
            "arm": arm, "source": source, "kind": kind, "task_id": tid,
            "group_id": tid.rsplit("/", 1)[0], "task_type": "pick_and_place_simple",
            "messages": messages, "supervised_assistant_indices": [2, 4] if kind == "trajectory" else [4],
            "weight": 1.0, "evidence": evidence}


def seal_bundle(root, rows, theta, splits, arm="JOINT"):
    root.mkdir(exist_ok=True, parents=True)
    rows = sorted(rows, key=lambda row: row["example_id"])
    (root / "examples.jsonl").write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
    (root / "examples.sha256").write_text(file_hash(root / "examples.jsonl") + "\n")
    write_json(root / "manifest.json", {"schema": v2.SCHEMA, "arm": arm, "created": "2026-10-27",
        "code_revision": "test-only", "inputs": {str(splits): file_hash(splits)},
        "counts": {"by_" + k: dict(Counter(row[k] for row in rows)) for k in ("source", "kind", "task_type")},
        "supervised_turns": sum(len(row["supervised_assistant_indices"]) for row in rows),
        "example_ids_sha256": digest([row["example_id"] for row in rows]), "splits_sha256": file_hash(splits),
        "theta0": {"adapter_path": str(theta), "adapter_weights_sha256": file_hash(theta / "adapter_model.safetensors")},
        "training_exposure_tasks": sorted({row["group_id"] for row in rows}), "excluded": [{"reason": "test", "count": 1}]})


@pytest.fixture
def corpus(tmp_path, data_root):
    theta, snapshot, config = adapter_fixture(tmp_path)
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    for key, value in (("manifest", manifest), ("split_freeze", freeze)):
        path = tmp_path / (key + ".json")
        write_json(path, value)
        config["paths"][key] = str(path)
    sets = {name: {"task_ids": [], "group_ids": []} for name in v2.FORBIDDEN_SETS}
    for name, group in (("V2_pool", "a"), ("K20", "b")):
        sets[name] = {"task_ids": [f"train/room_{group}/trial_{i}" for i in range(2)],
                      "group_ids": [f"train/room_{group}"]}
    splits = tmp_path / "method_v1_splits.json"
    write_json(splits, {"sets": sets})
    root = tmp_path / "dataset"
    rows = [example(f"train/room_a/trial_{i}", "joint_local", "decision") for i in range(2)]
    replay = [example(f"train/room_b/trial_{i}", "k20_replay", "trajectory", "CONTINUE", step=None) for i in range(2)]
    seal_bundle(root, rows, theta, splits)
    seal_bundle(root / "replay", replay, theta, splits, "CONTINUE")
    return {"root": root, "rows": rows, "replay": replay, "theta": theta, "snapshot": snapshot,
            "config": config, "splits": splits}


def read(c):
    return v2.read_dataset(c["root"], theta0=c["theta"], config=c["config"])


def reseal(c):
    seal_bundle(c["root"], c["rows"], c["theta"], c["splits"])


def test_token_masks_match_main_grid_and_decision_masks_native_prefix():
    from baselines.common.conversation import encode_messages
    from baselines.common.shared_history import pack_shared_history
    tokenizer = v2.FakeTokenizer()
    trajectory = example("train/a/0", "joint_complete", "trajectory")
    decision = deepcopy(trajectory)
    decision.update(kind="decision", supervised_assistant_indices=[4])
    full = v2.encode_example(trajectory, tokenizer)
    local = v2.encode_example(decision, tokenizer)
    baseline = encode_messages(trajectory["messages"], tokenizer, task_id=trajectory["task_id"], packing="turn")
    assert full["turns"] == baseline["turns"]
    assert local["turns"] == baseline["turns"][-1:]
    for row in (full, local):
        for turn in row["turns"]:
            assert turn["labels"][-1] == turn["input_ids"][-1] == 106
            assert set(turn["labels"][:turn["prompt_tokens"]]) == {-100}
        packed = pack_shared_history(row)
        assert [v for v in packed["labels"] if v != -100] == [
            v for turn in row["turns"] for v in turn["labels"] if v != -100]
    assert local["supervised_tokens"] < full["supervised_tokens"]
    turn = local["turns"][0]
    assert "Check apple 1" in tokenizer.decode(turn["input_ids"][:turn["prompt_tokens"]])
    assert tokenizer.decode([v for v in turn["labels"] if v != -100]) == decision["messages"][-1]["content"] + "<turn|>"
    with pytest.raises(ValueError, match="no truncation"):
        v2.encode_example(decision, tokenizer, max_length=8)


def test_separate_and_inline_replay(corpus):
    data = read(corpus)
    assert len(data["rows"]) == 4 and data["arm"] == "JOINT"
    assert all("evidence" not in v2.encode_example(row, v2.FakeTokenizer()) for row in data["rows"])
    inline = deepcopy(corpus["replay"])
    for row in inline:
        row["arm"] = "JOINT"
        e = row["evidence"]
        row["example_id"] = digest(["JOINT", row["source"], row["task_id"], e["episode_ref"], e["step"]])
    corpus["rows"] += inline
    reseal(corpus)
    with pytest.raises(ValueError, match="Duplicate replay"):
        read(corpus)
    shutil.rmtree(corpus["root"] / "replay")
    assert len(read(corpus)["rows"]) == 4


TRAJECTORY_ISSUES = [
    ("Thought: Try apple 1 again.\nAction: eat apple 1", 1, 0, "not admissible"),
    ("Thought: apple 99 and mug 42 are here.\nAction: look", 0, 1, "unseen instance"),
]


@pytest.mark.parametrize("reply,inadmissible,unseen,match", TRAJECTORY_ISSUES)
@pytest.mark.parametrize("arm,source,layout", [
    ("OFFLINE", "offline_teacher", "root"),
    ("CONTINUE", "k20_replay", "root"),
    ("JOINT", "k20_replay", "inline"),
    ("CONTINUE", "k20_replay", "separate"),
])
def test_trajectory_issues_are_diagnostics_and_preserve_baseline_bytes(
        corpus, arm, source, layout, reply, inadmissible, unseen, match):
    from train.dataset import episode_examples
    group = "a" if source == "offline_teacher" else "b"
    row = example(f"train/room_{group}/trial_0", source, "trajectory", arm)
    row["messages"][2]["content"] = reply
    # Future observations cannot retroactively ground the first supervised turn.
    later = {"observation": "Now apple 99 and mug 42 are visible.", "admissible_commands": ["look"]}
    row["messages"][3]["content"] = canonical_bytes(later).decode()
    if layout == "separate":
        seal_bundle(corpus["root"] / "replay", [row], corpus["theta"], corpus["splits"], arm)
    else:
        shutil.rmtree(corpus["root"] / "replay")
        rows = [*corpus["rows"], row] if layout == "inline" else [row]
        seal_bundle(corpus["root"], rows, corpus["theta"], corpus["splits"], arm)
    original = canonical_bytes(row)
    data = read(corpus)
    assert data["trajectory_diagnostics"] == [{"example_id": row["example_id"],
        "inadmissible_supervised_turns": inadmissible,
        "not_yet_seen_instance_supervised_turns": unseen}]
    loaded = next(r for r in data["rows"] if r["example_id"] == row["example_id"])
    tokenizer = v2.FakeTokenizer()
    encoded = v2.encode_example(loaded, tokenizer)
    assert canonical_bytes(loaded) == original
    # Compare directly to the main-grid episode renderer, including both targets.
    episode = {"schema_version": 2, "task_id": row["task_id"], "split": "train", "sealed": False,
        "initial": {"visible": json.loads(row["messages"][1]["content"])},
        "steps": [{"event_id": f"teacher:{i}", "action_raw": row["messages"][i]["content"],
                   "visible": later} for i in (2, 4)]}
    baseline = episode_examples(episode, tokenizer, packing="turn")
    assert len(encoded["turns"]) == len(baseline) == 2
    for actual, expected in zip(encoded["turns"], baseline):
        assert actual["input_ids"] == expected["input_ids"]
        assert actual["labels"] == expected["labels"]
        assert tokenizer.decode(actual["input_ids"]).encode() == tokenizer.decode(expected["input_ids"]).encode()
    assert tokenizer.decode([v for v in encoded["turns"][0]["labels"] if v != -100]) == reply + "<turn|>"


@pytest.mark.parametrize("reply,inadmissible,unseen,match", TRAJECTORY_ISSUES)
def test_decision_rejects_trajectory_issues_in_supervised_turn(corpus, reply, inadmissible, unseen, match):
    corpus["rows"][0]["messages"][-1]["content"] = reply
    reseal(corpus)
    with pytest.raises(ValueError, match=match):
        read(corpus)


@pytest.mark.parametrize("reply,inadmissible,unseen,match", TRAJECTORY_ISSUES)
def test_decision_context_is_not_subject_to_supervised_checks(corpus, reply, inadmissible, unseen, match):
    corpus["rows"][0]["messages"][2]["content"] = reply
    reseal(corpus)
    data = read(corpus)
    assert {d["example_id"] for d in data["trajectory_diagnostics"]} == {
        row["example_id"] for row in corpus["replay"]}
    assert all(d["inadmissible_supervised_turns"] == d["not_yet_seen_instance_supervised_turns"] == 0
               for d in data["trajectory_diagnostics"])


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r["messages"][2].update(content="Guidance: look\nAction: look"), "Guidance:"),
    (lambda r: r["messages"][-1].update(content="Guidance: look\nAction: look"), "Guidance:"),
    (lambda r: r["messages"][2].update(content="Unparseable"), "Action line"),
    (lambda r: r["messages"][-1].update(content="Unparseable"), "Action line"),
    (lambda r: r["messages"][0].update(content="Different system prompt"), "prompt contract"),
    (lambda r: r.update(supervised_assistant_indices=[4]), "all assistant turns"),
])
def test_trajectory_keeps_other_validity_checks(corpus, mutate, match):
    mutate(corpus["replay"][0])
    seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    with pytest.raises(ValueError, match=match):
        read(corpus)


def test_loss_uses_token_mean_per_example_then_batch_mean():
    import torch
    from baselines.common.fsdp_train import sft_objective
    torch.set_num_threads(1)
    short = example("train/a/0", "joint_local", "decision")
    long = example("train/a/1", "joint_complete", "trajectory")
    long["messages"][-1]["content"] = "Thought: " + "z" * 200 + "\nAction: look"
    rows = [v2.encode_example(r, v2.FakeTokenizer()) for r in (short, long)]
    actual, expected = v2.fake_model(), v2.fake_model()
    with torch.no_grad():
        actual.lora_B.copy_(torch.linspace(-2, 2, 512).reshape(512, 1))
        expected.load_state_dict(actual.state_dict())
    loss = sum(sft_objective(actual, r, "cpu", scoring="shared_history")[0] for r in rows) / 2
    logits = (expected.lora_B @ expected.lora_A).flatten().log_softmax(0)
    targets = [torch.tensor([x for t in r["turns"] for x in t["labels"] if x != -100]) for r in rows]
    reference = sum(-logits[t].mean() for t in targets) / 2
    pooled = -logits[torch.cat(targets)].mean()
    assert loss.item() == pytest.approx(reference.item(), abs=1e-6)
    assert abs(loss.item() - pooled.item()) > 1e-3
    loss.backward(); reference.backward()
    for a, e in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, e.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(weight=.5), "sample-weight"),
    (lambda r: r.update(weight=True), "sample-weight"),
    (lambda r: r.update(kind="suffix"), "kind"),
    (lambda r: r.update(supervised_assistant_indices=[2]), "last assistant"),
    (lambda r: r.update(supervised_assistant_indices=[2, 4]), "last assistant"),
    (lambda r: r.update(supervised_assistant_indices=[True]), "last assistant"),
    (lambda r: r.update(example_id="f" * 64), "Canonical"),
    (lambda r: r.update(arm="OFFLINE"), "Arm/source"),
    (lambda r: r.update(source="offline_teacher"), "Arm/source"),
    (lambda r: r.update(group_id="train/spoof"), "Task identity"),
    (lambda r: r.update(task_type="spoof"), "Task identity"),
    (lambda r: r["messages"][-1].update(content="Guidance: look\nAction: look"), "Guidance:"),
    (lambda r: r["messages"][2].update(content="Guidance: look\nAction: look"), "Guidance:"),
    (lambda r: r["messages"][-1].update(content="Action: impossible"), "admissible"),
    (lambda r: r["messages"][2].update(content="Unparseable"), "Action line"),
    (lambda r: r["messages"][-1].update(content="Thought: apple 99 exists.\nAction: look"), "unseen instance"),
    (lambda r: r["messages"][0].update(content="Different system prompt"), "prompt contract"),
    (lambda r: r["messages"][1].update(content='{"observation":"room","admissible_commands":["look"],"policy_commands":["secret"]}'), "whitelist"),
    (lambda r: r["messages"][1].update(secret="oracle"), "message fields"),
    (lambda r: r["evidence"].update(label_check="failed"), "unchecked"),
    (lambda r: r["evidence"].update(episode_success="true"), "unchecked"),
    (lambda r: r.update(hidden_state={"secret": 3}), "whitelist"),
])
def test_validation_failures(corpus, mutate, match):
    mutate(corpus["rows"][0]); reseal(corpus)
    with pytest.raises(ValueError, match=match):
        read(corpus)


@pytest.mark.parametrize("field,value", [("schema", "v2/2"), ("counts", {}),
    ("example_ids_sha256", "0" * 64), ("supervised_turns", 999),
    ("training_exposure_tasks", []), ("splits_sha256", "f" * 64), ("inputs", {})])
def test_manifest_corruption(corpus, field, value):
    path = corpus["root"] / "manifest.json"
    doc = json.loads(path.read_text()); doc[field] = value; write_json(path, doc)
    with pytest.raises(ValueError):
        read(corpus)


@pytest.mark.parametrize("bundle", ["", "replay"])
@pytest.mark.parametrize("fault", [None, "by_source", "by_kind", "by_task_type", "by_local_event",
                                   "examples", "task_groups", "unique_instances", "unknown", "bool"])
def test_manifest_rule6_counts(corpus, bundle, fault):
    # Two trials share a group; null decisions and trajectories do not
    # contribute to by_local_event. Rule 8 requires null trajectory events.
    corpus["rows"][0]["evidence"]["local_event"] = None
    reseal(corpus)
    seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    path = corpus["root"] / bundle / "manifest.json"
    doc = json.loads(path.read_text())
    info = {"by_local_event": {} if bundle else {"target_seen": 1}, "examples": 2,
            "task_groups": 1, "unique_instances": 2}
    # Each informational key is independently optional.
    for key, value in info.items():
        single = deepcopy(doc); single["counts"][key] = value
        write_json(path, single)
        assert len(read(corpus)["rows"]) == 4
    doc["counts"].update(info)
    if fault == "unknown":
        doc["counts"]["tokens"] = 5
    elif fault == "bool":
        doc["counts"]["task_groups"] = True
    elif fault:
        doc["counts"][fault] = {"wrong": 1} if fault.startswith("by_") else 99
    write_json(path, doc)
    if fault:
        with pytest.raises(ValueError, match="counts"):
            read(corpus)
    else:
        assert len(read(corpus)["rows"]) == 4


@pytest.mark.parametrize("key", ["by_source", "by_kind", "by_task_type"])
def test_manifest_rule6_required_counts_cannot_be_omitted(corpus, key):
    path = corpus["root"] / "replay/manifest.json"
    doc = json.loads(path.read_text()); del doc["counts"][key]; write_json(path, doc)
    with pytest.raises(ValueError, match="counts keys"):
        read(corpus)


@pytest.mark.parametrize("kind,step,accepted", [("trajectory", None, True), ("trajectory", 0, False),
    ("decision", 0, True), ("decision", None, False), ("decision", True, False),
    ("decision", 1.0, False), ("trajectory", -1, False), ("trajectory", "0", False)])
def test_evidence_step_by_kind(corpus, kind, step, accepted):
    row = example("train/room_a/trial_0", "joint_local", kind, step=step)
    row["evidence"].update(local_event=None, executor="teacher", label_source="teacher_reply")
    corpus["rows"] = [row]; reseal(corpus)
    if accepted:
        assert row in read(corpus)["rows"]
    else:
        with pytest.raises(ValueError, match="unchecked label evidence"):
            read(corpus)


@pytest.mark.parametrize("kind", ["decision", "trajectory"])
@pytest.mark.parametrize("field,value", [("executor", "generator"), ("local_event", "unknown"),
    ("label_source", "teacher_thought"), ("label_check", "unchecked"), ("unknown_field", None)])
def test_evidence_unknown_fields_and_enums_refused(corpus, kind, field, value):
    corpus["rows"] = [example("train/room_a/trial_0", "joint_local", kind)]
    corpus["rows"][0]["evidence"][field] = value
    reseal(corpus)
    with pytest.raises(ValueError, match="evidence"):
        read(corpus)


@pytest.mark.parametrize("events,expected", [
    (["target_seen", "placed", "target_seen"], {"target_seen": 3, "placed": 1}),
    ([], {"target_seen": 1}),
])
def test_rule6_plural_events_count_every_element_and_singular_fallback(corpus, events, expected):
    corpus["rows"][0]["evidence"].update(local_events=events, local_event=events[0] if events else None)
    reseal(corpus)
    path = corpus["root"] / "manifest.json"
    doc = json.loads(path.read_text())
    doc["counts"]["by_local_event"] = expected
    write_json(path, doc)
    assert len(read(corpus)["rows"]) == 4
    doc["counts"]["by_local_event"] = {"target_seen": 2}
    write_json(path, doc)
    with pytest.raises(ValueError, match="by_local_event"):
        read(corpus)


OPTIONAL_EVIDENCE = [
    ("local_events", ["target_seen", "placed"]), ("local_events", []),
    ("complete_success_eligible", True), ("complete_success_eligible", False),
    *[(field, "aB" * 32) for field in sorted(v2.EVIDENCE_HASH_FIELDS)],
    ("history_rewrites", [{"original_thought": "AUDIT_ONLY_PRIVATE_TEXT", "replacement": "look"}]),
    ("history_rewrites", []), ("next_legal_target_take", None),
    ("next_legal_target_take", 0), ("next_legal_target_take", -1),
]


@pytest.mark.parametrize("field,value", OPTIONAL_EVIDENCE)
def test_rule8_optional_evidence_is_typed_audit_only(corpus, field, value):
    row = corpus["rows"][0]
    row["evidence"][field] = value
    if field == "local_events":
        row["evidence"]["local_event"] = value[0] if value else None
    expected = v2.encode_example(row, v2.FakeTokenizer())
    reseal(corpus)
    actual = next(r for r in read(corpus)["rows"] if r["example_id"] == row["example_id"])
    assert v2.encode_example(actual, v2.FakeTokenizer()) == expected
    assert "AUDIT_ONLY_PRIVATE_TEXT" not in json.dumps(expected)
    del row["evidence"][field]
    assert v2.encode_example(row, v2.FakeTokenizer()) == expected


@pytest.mark.parametrize("field,value", [
    ("local_events", None), ("local_events", "target_seen"), ("local_events", [None]),
    ("local_events", ["unknown"]), ("local_events", [{}]),
    ("complete_success_eligible", 1), ("complete_success_eligible", None),
    ("complete_success_eligible", "true"),
    *[(field, bad) for field in sorted(v2.EVIDENCE_HASH_FIELDS)
      for bad in (None, 64, "a" * 63, "g" * 64)],
    ("history_rewrites", {}), ("history_rewrites", ["text"]), ("history_rewrites", [None]),
    ("next_legal_target_take", True), ("next_legal_target_take", 1.0),
    ("next_legal_target_take", "1"),
])
def test_rule8_bad_optional_types_refused(corpus, field, value):
    corpus["rows"][0]["evidence"][field] = value
    reseal(corpus)
    with pytest.raises(ValueError, match="decision evidence"):
        read(corpus)


@pytest.mark.parametrize("event,events", [("target_seen", ["placed", "target_seen"]),
    (None, ["target_seen"]), ("target_seen", [])])
def test_rule8_bad_local_event_pairing_refused(corpus, event, events):
    corpus["rows"][0]["evidence"].update(local_event=event, local_events=events)
    reseal(corpus)
    with pytest.raises(ValueError, match="local_events"):
        read(corpus)


@pytest.mark.parametrize("field,value", OPTIONAL_EVIDENCE)
def test_rule8_optional_fields_only_allowed_on_decisions(corpus, field, value):
    corpus["replay"][0]["evidence"][field] = value
    seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    with pytest.raises(ValueError, match="evidence schema"):
        read(corpus)


def test_rule8_trajectory_local_event_must_be_null(corpus):
    corpus["replay"][0]["evidence"]["local_event"] = "target_seen"
    seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    with pytest.raises(ValueError, match="unchecked label evidence"):
        read(corpus)


def test_rule7_continue_root_and_replay_keep_origins_and_common_draw(corpus):
    other = read(corpus)
    seal_bundle(corpus["root"], corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    assert (corpus["root"] / "examples.jsonl").read_bytes() == (corpus["root"] / "replay/examples.jsonl").read_bytes()
    data = read(corpus)
    assert len(data["rows"]) == 4
    assert data["pools"] == v2.training_pools(data["rows"], arm="CONTINUE", root_count=2)
    draws = []
    for d in (other, data):
        batches = v2.batch_schedule(d["rows"], updates=2, batch_examples=16, replay_ratio=.25,
                                   seed=0, arm=d["arm"], pools=d["pools"])
        draws.append([[(i, s["replay_key"], s["pool_epoch"]) for i, s in enumerate(b["selections"])
                       if s["pool"] == "replay"] for b in batches])
        for batch in batches:
            assert Counter(s["pool"] for s in batch["selections"]) == {"new": 12, "replay": 4}
            if d is data:
                for s in batch["selections"]:
                    path = corpus["root"] / ("replay/examples.jsonl" if s["pool"] == "replay" else "examples.jsonl")
                    assert data["origins"][s["index"]] == str(path)
    assert draws[0] == draws[1]


@pytest.mark.parametrize("arm", sorted(set(v2.ARMS) - {"CONTINUE"}))
def test_rule7_other_arms_refuse_row_in_both_pools(corpus, arm):
    rows = deepcopy(corpus["replay"])
    for row in rows:
        row["arm"] = arm
        e = row["evidence"]
        row["example_id"] = digest([arm, row["source"], row["task_id"], e["episode_ref"], e["step"]])
    seal_bundle(corpus["root"], rows, corpus["theta"], corpus["splits"], arm)
    with pytest.raises(ValueError, match="Duplicate replay"):
        read(corpus)


@pytest.fixture
def external_corpus(corpus):
    """Method freeze and source shapes emitted by E0.4; fabricated members only."""
    c = corpus
    doc = json.loads(c["splits"].read_text())
    root = c["root"].parent / "producer"
    repair = root / "data/manifest/repair_v01_splits.json"
    dev = root / "data/manifest/dev140.json"
    write_json(repair, {"sets": {name: doc["sets"].pop(name) for name in
                                ("U_mech", "D_probe", "D_unseen_quick")}})
    del doc["sets"]["dev140"]
    frozen = json.loads(Path(c["config"]["paths"]["manifest"]).read_text())
    dev_tasks = [r["task_id"] for r in frozen["tasks"] if r["source_split"] == "valid_seen"]
    assert dev_tasks
    write_json(dev, {"task_ids": dev_tasks})
    doc["rebuild_arguments"] = {"repair_splits": str(repair.relative_to(root)),
        "dev140": str(dev.relative_to(root)), "output": "data/manifest/method_v1_splits.json"}
    doc["exclusion_sources"] = [{"category": category, "path": str(path.relative_to(root)),
        "sha256": file_hash(path)} for category, path in
        [(name, repair) for name in ("U_mech", "D_probe", "D_unseen_quick")] + [("dev140", dev)]]
    doc["input_sha256"] = {str(p.relative_to(root)): file_hash(p) for p in (repair, dev)}
    c["splits"].unlink()
    c.update(splits=root / doc["rebuild_arguments"]["output"], repair_splits=repair, dev140=dev)
    write_json(c["splits"], doc)
    reseal(c)
    seal_bundle(c["root"] / "replay", c["replay"], c["theta"], c["splits"], "CONTINUE")
    return c


def refreeze_external(c):
    doc = json.loads(c["splits"].read_text())
    for record in doc["exclusion_sources"]:
        key = "dev140" if record["category"] == "dev140" else "repair_splits"
        record["sha256"] = file_hash(c[key])
        doc["input_sha256"][record["path"]] = record["sha256"]
    write_json(c["splits"], doc)
    reseal(c)
    seal_bundle(c["root"] / "replay", c["replay"], c["theta"], c["splits"], "CONTINUE")


@pytest.mark.parametrize("source", ["repair_splits", "dev140"])
@pytest.mark.parametrize("fault", ["missing", "altered", "unrecorded", "conflicting_hash"])
def test_external_split_sources_fail_closed(external_corpus, source, fault, monkeypatch):
    c = external_corpus
    assert {c["repair_splits"], c["dev140"]} <= set(read(c)["inputs"])
    def forbidden(*_, **__):
        raise AssertionError("Must reject sources before allocating a tokenizer/model")
    monkeypatch.setattr(v2, "FakeTokenizer", forbidden)
    if fault == "missing":
        c[source].unlink()
    elif fault == "altered":
        c[source].write_bytes(c[source].read_bytes() + b" ")
    else:
        doc = json.loads(c["splits"].read_text())
        record = next(r for r in doc["exclusion_sources"] if r["path"] == doc["rebuild_arguments"][source])
        if fault == "unrecorded":
            del record["sha256"]
        else:
            record["sha256"] = "f" * 64
        write_json(c["splits"], doc)
        reseal(c)
        seal_bundle(c["root"] / "replay", c["replay"], c["theta"], c["splits"], "CONTINUE")
    with pytest.raises(ValueError, match="missing file|sha256"):
        run(c, c["root"].parent / "refused")


@pytest.mark.parametrize("name", ["V2_pool", *v2.FORBIDDEN_SETS])
def test_external_missing_set_is_not_empty(external_corpus, name):
    c = external_corpus
    path = c["splits"] if name in ("V2_pool", "U_dev") else c["dev140"] if name == "dev140" else c["repair_splits"]
    doc = json.loads(path.read_text())
    if name == "dev140":
        del doc["task_ids"]
    else:
        del doc["sets"][name]
    write_json(path, doc); refreeze_external(c)
    with pytest.raises(ValueError, match="missing|must include|task_ids"):
        read(c)


@pytest.mark.parametrize("name", ["U_mech", "D_probe", "D_unseen_quick", "dev140"])
def test_external_diagnostic_groups_are_excluded(external_corpus, name):
    c = external_corpus
    path = c["dev140"] if name == "dev140" else c["repair_splits"]
    doc = json.loads(path.read_text())
    if name == "dev140":
        # Deriving dev140 groups must exclude sibling trials too.
        doc["task_ids"] = ["train/room_a/trial_1"]
    else:
        doc["sets"][name]["group_ids"] = ["train/room_a"]
    write_json(path, doc); refreeze_external(c)
    with pytest.raises(ValueError, match="diagnostic group"):
        read(c)


@pytest.mark.parametrize("explicit", [False, True])
def test_external_sources_relocate_and_cli_hash_checks(external_corpus, tmp_path_factory, capsys, explicit):
    from tools import train_v2_arm as cli
    from repair.v2_inputs import inspect_inputs
    c = external_corpus
    before = read(c)
    cfg = c["root"].parent / "config.json"; write_json(cfg, c["config"])
    overrides = {}
    if explicit:
        for key in ("repair_splits", "dev140"):
            replacement = c["root"].parent / (key + "-copy.json")
            c[key].rename(replacement)
            overrides[key] = replacement
    old, new, rules = move_corpus(c, tmp_path_factory)
    after = v2.read_dataset(c["root"], theta0=c["theta"], config=c["config"], relocate=rules, **overrides)
    assert after["rows"] == before["rows"] and after["manifests"] == before["manifests"]
    inspected = inspect_inputs(c["root"], relocate=rules, **overrides)
    assert set(inspected["split_sources"].values()) <= inspected["files"].files.keys()
    assert all(inspected["files"].files[p]["seals"] for p in inspected["split_sources"].values())
    args = ["inputs", "--dataset", str(c["root"]), "--config", str(cfg), "--relocate", rules[0], "--check"]
    for key, path in overrides.items():
        args += ["--" + key.replace("_", "-"), str(path)]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    inspected["split_sources"]["dev140"].write_text("{}")
    assert cli.main(args) == 1
    assert "dev140 source sha256 mismatch" in capsys.readouterr().err


def test_cli_run_with_explicit_external_sources(external_corpus, capsys):
    from tools import train_v2_arm as cli
    import torch
    torch.set_num_threads(1)
    c = external_corpus
    cfg = c["root"].parent / "config.json"; write_json(cfg, c["config"])
    out = c["root"].parent / "external-cli-run"
    args = ["run", "--dataset", str(c["root"]), "--theta0", str(c["theta"]),
        "--snapshot", str(c["snapshot"]), "--config", str(cfg), "--output", str(out),
        "--updates", "1", "--batch-examples", "4", "--replay-ratio", ".25",
        "--checkpoints", "1", "--gpus", "1", "--dry-run"]
    for key in ("repair_splits", "dev140"):
        replacement = c["root"].parent / (key + "-copy.json")
        c[key].rename(replacement)
        args += ["--" + key.replace("_", "-"), str(replacement)]
    assert cli.main(args) == 0, capsys.readouterr().err
    result = json.loads((out / "manifest.json").read_text())
    assert result["status"] == "synthetic_complete" and result["training_exposure"] == 0
    inputs = json.loads((out / "provenance.json").read_text())["inputs"]
    assert inputs[str(replacement)] == file_hash(replacement)


def test_hashes_theta0_and_json_refusals(corpus):
    weights = corpus["theta"] / "adapter_model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="theta0 adapter hash"):
        read(corpus)
    weights.write_bytes(original)
    path = corpus["root"] / "examples.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256"):
        read(corpus)
    reseal(corpus)
    (corpus["splits"]).write_text("{}")
    with pytest.raises(ValueError, match="Split freeze sha256 mismatch"):
        read(corpus)
    with pytest.raises(ValueError, match="Duplicate"):
        v2._json('{"key":1,"key":2}')
    with pytest.raises(ValueError, match="Nonfinite"):
        v2._json('{"key":NaN}')


def test_nonpool_training_task_and_wrong_replay_membership(corpus):
    corpus["rows"][0] = example("train/room_b/trial_0", "joint_local", "decision")
    reseal(corpus)
    with pytest.raises(ValueError, match="outside V2_pool/K20"):
        read(corpus)
    corpus["rows"][0] = example("train/room_a/trial_0", "joint_local", "decision")
    reseal(corpus)
    corpus["replay"][0] = example("train/room_a/trial_0", "k20_replay", "trajectory", "CONTINUE")
    seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    with pytest.raises(ValueError, match="outside V2_pool/K20"):
        read(corpus)


def test_step_number_is_not_an_unseen_object(corpus):
    corpus["rows"][0]["messages"][-1]["content"] = "Thought: At step 1, inspect apple 1.\nAction: look"
    reseal(corpus)
    assert read(corpus)


@pytest.mark.parametrize("name", v2.FORBIDDEN_SETS)
def test_diagnostic_group_exclusion(corpus, name):
    doc = json.loads(corpus["splits"].read_text())
    doc["sets"][name]["group_ids"] = ["train/room_a"]
    write_json(corpus["splits"], doc)
    reseal(corpus)
    seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    with pytest.raises(ValueError, match="diagnostic group"):
        read(corpus)


def schedule(rows, **kwargs):
    return v2.batch_schedule(rows, **{"updates": 9, "batch_examples": 4, "replay_ratio": .25,
                                      "seed": 1, "arm": "JOINT", **kwargs})


def test_schedule_determinism_epochs_ratio_and_cross_arm_replay(corpus):
    rows = read(corpus)["rows"]
    a = schedule(rows)
    assert a == schedule(rows) and digest(a) == digest(schedule(rows))
    assert a != schedule(rows, seed=2)
    alternative = [example(f"train/other/{i}", "offline_teacher", "trajectory", "OFFLINE") for i in range(7)]
    alternative += deepcopy(corpus["replay"])
    for r in alternative[7:]:
        r["arm"] = "OFFLINE"
        r["example_id"] = digest([r["arm"], r["source"], r["task_id"], r["evidence"]["episode_ref"], r["evidence"]["step"]])
    b = schedule(alternative, arm="OFFLINE")
    replay_draws = lambda batches: [[(i, s["replay_key"], s["pool_epoch"]) for i, s in enumerate(batch["selections"])
                                    if s["pool"] == "replay"] for batch in batches]
    assert replay_draws(a) == replay_draws(b)
    for batch in a:
        assert len(batch["indices"]) == 4
        assert sum(s["pool"] == "replay" for s in batch["selections"]) == 1
    for pool in ("replay", "new"):
        epochs = {}
        for batch in a:
            for s in batch["selections"]:
                if s["pool"] == pool:
                    epochs.setdefault(s["pool_epoch"], []).append(s["index"])
        assert all(len(values) == len(set(values)) for values in epochs.values())
        assert all(len(values) == 2 for epoch, values in epochs.items() if epoch < max(epochs))
    # CONTINUE fills non-replay slots with an independent K20 epoch stream.
    cont = schedule(deepcopy(corpus["replay"]), arm="CONTINUE")
    assert replay_draws(cont) == replay_draws(a)
    assert {s["source"] for batch in cont for s in batch["selections"]} == {"k20_replay"}


@pytest.mark.parametrize("kwargs", [{"replay_ratio": .3}, {"replay_ratio": -1},
    {"replay_ratio": float("nan")}, {"batch_examples": 0}, {"updates": 0}, {"seed": -1}])
def test_schedule_invalid_options(corpus, kwargs):
    with pytest.raises(ValueError):
        schedule(read(corpus)["rows"], **kwargs)


def test_empty_pools_and_endpoint_ratios(corpus):
    with pytest.raises(ValueError, match="empty pool"):
        schedule(corpus["rows"])
    assert schedule(corpus["rows"], replay_ratio=0)
    assert schedule(corpus["replay"], replay_ratio=1)
    with pytest.raises(ValueError, match="Duplicate replay"):
        schedule(corpus["replay"] * 2, replay_ratio=1)


def test_accounting_includes_rank_padding_but_not_padding_targets(corpus):
    rows = read(corpus)["rows"]
    examples = [v2.encode_example(r, v2.FakeTokenizer()) for r in rows]
    batches = schedule(rows, batch_examples=3, replay_ratio=0, updates=2)
    one = v2.accounting(examples, batches)
    two = v2.accounting(examples, batches, world_size=2)
    assert one[0]["examples_seen"] == one[0]["processed_context_tokens"] == 0
    assert one[2]["examples_seen"] == 6
    assert one[2]["examples_seen_by_source"] == {"joint_local": 6}
    assert one[2]["examples_seen_by_kind"] == {"decision": 6}
    assert one[2]["supervised_target_tokens"] == sum(examples[i]["supervised_tokens"] for b in batches for i in b["indices"])
    assert two[2]["supervised_target_tokens"] == one[2]["supervised_target_tokens"]
    padding = sum(len(v2.pack_shared_history(examples[b["indices"][0]])["input_ids"]) for b in batches)
    assert two[2]["processed_context_tokens"] == one[2]["processed_context_tokens"] + padding
    assert two[2]["zero_weight_slots"] == 2


def test_inherited_recipe_and_schedule_semantics(corpus):
    spec = v2.continuation_recipe(corpus["theta"], corpus["config"], corpus["snapshot"], 7)
    assert spec["sft"]["rank"] == 64 and spec["sft"]["alpha"] == 128 and spec["sft"]["dropout"] == .05
    assert spec["scheduler_policy"]["total_updates"] == 7
    assert spec["scheduler_policy"]["restore_scheduler_state"] is False
    assert spec["recipe"]["learning_rate"] == 5e-5
    assert spec["optimizer_state"]["mode"] == "identical_reinitialization_for_both"
    (corpus["theta"] / "optimizer.pt").write_bytes(b"must not deserialize during inspection")
    spec = v2.continuation_recipe(corpus["theta"], corpus["config"], corpus["snapshot"], 7)
    assert spec["optimizer_state"]["mode"] == "restore_for_both"


def run(c, out, **kwargs):
    return v2.run_arm(c["root"], c["config"], out, theta0=c["theta"], snapshot=c["snapshot"],
        **{"updates": 5, "batch_examples": 4, "replay_ratio": .25, "checkpoints": [0, 1, 5],
           "seed": 1, "gpus": 1, "dry_run": True, **kwargs})


def test_run_manifest_records_trajectory_diagnostics_without_mutating_inputs(corpus, tmp_path):
    import torch
    torch.set_num_threads(1)
    expected = []
    for row, (reply, inadmissible, unseen, _) in zip(corpus["replay"], TRAJECTORY_ISSUES):
        row["messages"][2]["content"] = reply
        expected.append({"example_id": row["example_id"], "inadmissible_supervised_turns": inadmissible,
                         "not_yet_seen_instance_supervised_turns": unseen})
    seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"], corpus["splits"], "CONTINUE")
    before = {p: p.read_bytes() for p in corpus["root"].rglob("*") if p.is_file()}
    output = tmp_path / "diagnostics-run"
    result = run(corpus, output, updates=1, checkpoints=[1])
    expected.sort(key=lambda d: d["example_id"])
    assert result["trajectory_diagnostics"] == expected
    assert json.loads((output / "manifest.json").read_text())["trajectory_diagnostics"] == expected
    assert {p: p.read_bytes() for p in corpus["root"].rglob("*") if p.is_file()} == before


def test_dry_run_real_updates_manifests_and_determinism(corpus, tmp_path, monkeypatch):
    import torch
    from harness.lora_adapters import inspect_adapter
    torch.set_num_threads(1)
    def forbidden(*_, **__):
        raise AssertionError("A dry run must never load real weights/tokenizer/optimizer")
    monkeypatch.setattr(v2, "load_existing_adapter", forbidden)
    monkeypatch.setattr("train.dataset.load_tokenizer", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    # This file is hashed, but never interpreted by the fake run.
    (corpus["theta"] / "optimizer.pt").write_bytes(b"real optimizer is not a fake optimizer")
    first = run(corpus, tmp_path / "first")
    second = run(corpus, tmp_path / "second", gpus=2)
    assert first["schedule_sha256"] == second["schedule_sha256"]
    assert first["status"] == "synthetic_complete" and first["training_exposure"] == 0
    assert first["synthetic_training_exposure"] == 20
    assert first["trainable_parameter_count"] == 513
    assert first["accounting"]["examples_seen_by_source"] == {"joint_local": 15, "k20_replay": 5}
    assert first["pool_inventory"]["replay"]["unique_instances"] == 2
    assert first["pool_inventory"]["replay"]["task_groups"] == 1
    assert first["checkpoints"][0]["status"] == "theta0_reference"
    assert first["checkpoints"][0]["sha256"]["adapter_model.safetensors"] == file_hash(corpus["theta"] / "adapter_model.safetensors")
    for ckpt, other in zip(first["checkpoints"][1:], second["checkpoints"][1:]):
        directory = Path(ckpt["path"])
        assert ckpt["sha256"] == v2.checkpoint_files(directory)
        assert {"adapter_config.json", "adapter_model.safetensors", "training_state.json"} <= ckpt["sha256"].keys()
        assert ckpt["sha256"]["adapter_model.safetensors"] == other["sha256"]["adapter_model.safetensors"]
        assert ckpt["trainer_state"]["actual_trainable_module_names"] == ["fake_projection"]
        assert ckpt["trainer_state"]["synthetic"] is True
        prov = json.loads((directory.parent / "provenance.json").read_text())
        assert ckpt["provenance_sha256"] == file_hash(directory.parent / "provenance.json")
        # Same artifact schema used by check_lora_serving, with honest fake identity.
        identity = inspect_adapter("fake", directory, prov["config"]["models"]["student"], tmp_path / "first/fake-base")
        assert identity["base_model"] == "synthetic/v2-tiny-fake"
        assert ckpt["supervised_target_tokens"] == first["planned_accounting"][ckpt["update"]]["supervised_target_tokens"]
    assert first["checkpoints"][1]["sha256"]["adapter_model.safetensors"] != first["checkpoints"][2]["sha256"]["adapter_model.safetensors"]
    logs = [json.loads(s) for s in (tmp_path / "first/updates.jsonl").read_text().splitlines()]
    assert [r["update"] for r in logs] == [1, 2, 3, 4, 5]
    assert all(r["loss"] > 0 for r in logs)
    assert [r["lr"] for r in logs] == pytest.approx([5e-5, 4e-5, 3e-5, 2e-5, 1e-5])
    assert first["accounting"]["supervised_target_tokens"] == first["trainer_state"]["supervised_tokens_seen"]
    with pytest.raises(ValueError, match="Output already exists"):
        run(corpus, tmp_path / "first")


class SnapshotTokenizerStub(v2.FakeTokenizer):
    """Offline stand-in with IDs outside the byte model's 512-token vocabulary."""
    chat_template = "snapshot-tokenizer-stub"

    def encode(self, text, **kwargs):
        return [token if token == 106 else token + 1000 for token in super().encode(text, **kwargs)]

    def decode(self, ids, **kwargs):
        return super().decode([token if token == 106 else token - 1000 for token in ids], **kwargs)

    def convert_tokens_to_ids(self, token):
        assert token == "<turn|>"
        return 106


def test_snapshot_dry_run_cli_offline_tokenizer_no_weights(corpus, tmp_path, monkeypatch):
    from tools import train_v2_arm as cli
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    calls = []
    tokenizer = SnapshotTokenizerStub()
    def load(path, **kwargs):
        assert Path(path) == corpus["snapshot"]
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        calls.append(path)
        return tokenizer
    def forbidden(*_, **__):
        raise AssertionError("Snapshot dry run must not load model or optimizer weights")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(v2, "load_existing_adapter", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    cfg = tmp_path / "config.json"
    write_json(cfg, corpus["config"])
    out = tmp_path / "snapshot-run"
    assert cli.main(["run", "--dataset", str(corpus["root"]), "--theta0", str(corpus["theta"]),
        "--snapshot", str(corpus["snapshot"]), "--config", str(cfg), "--output", str(out),
        "--updates", "2", "--batch-examples", "16", "--replay-ratio", ".25", "--seed", "0",
        "--checkpoints", "1", "2", "--gpus", "1", "--dry-run", "--dry-run-tokenizer", "snapshot"]) == 0
    result = json.loads((out / "manifest.json").read_text())
    assert len(calls) == 1
    assert result["status"] == "synthetic_complete" and result["training_exposure"] == 0
    assert result["dry_run_tokenizer"] == "snapshot" and result["tokenizer"]["synthetic"] is False
    assert result["tokenizer"]["chat_template_sha256"] == digest(tokenizer.chat_template)
    assert [c["update"] for c in result["checkpoints"]] == [0, 1, 2]
    examples = [v2.encode_example(r, tokenizer) for r in read(corpus)["rows"]]
    assert result["encoded_examples_sha256"] == digest(examples)
    scoring, vocabulary = v2.fake_scoring_examples(examples)
    assert result["fake_scoring_vocabulary"] == vocabulary and max(vocabulary) > 512
    assert result["scoring_examples_sha256"] == digest(scoring)
    assert result["trainable_parameter_count"] == len(vocabulary) + 1
    for before, after in zip(examples, scoring):
        for a, b in zip(before["turns"], after["turns"]):
            assert a["input_ids"] == [vocabulary[i] for i in b["input_ids"]]
            assert a["labels"] == [-100 if i == -100 else vocabulary[i] for i in b["labels"]]
            assert {k: v for k, v in a.items() if k not in ("input_ids", "labels")} == {
                k: v for k, v in b.items() if k not in ("input_ids", "labels")}
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["config"]["v2_training"]["dry_run_tokenizer"] == "snapshot"
    assert prov["config"]["models"]["student"]["requested_repo"] == "synthetic/v2-tiny-fake"


def test_snapshot_dry_run_enforces_inherited_context_limit(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr("train.dataset.load_tokenizer", lambda _: SnapshotTokenizerStub())
    original = v2.continuation_recipe
    def short_recipe(*args):
        inherited = original(*args)
        inherited["recipe"]["max_length"] = 8
        return inherited
    def forbidden(*_, **__):
        raise AssertionError("Context overflow must fail before model allocation")
    monkeypatch.setattr(v2, "continuation_recipe", short_recipe)
    monkeypatch.setattr(v2, "fake_model", forbidden)
    out = tmp_path / "overflow"
    with pytest.raises(ValueError, match="max_length=8; no truncation"):
        run(corpus, out, dry_run_tokenizer="snapshot")
    assert not out.exists()


def test_serving_checkpoint_export_keeps_original_lora(corpus, tmp_path):
    from harness.lora_adapters import inspect_adapter
    spec = v2.continuation_recipe(corpus["theta"], corpus["config"], corpus["snapshot"], 1)
    prov = json.loads((corpus["theta"].parent / "provenance.json").read_text())
    class Adapter:
        def save_pretrained(self, path, safe_serialization):
            assert safe_serialization
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copyfile(corpus["theta"] / name, path / name)
    out = tmp_path / "publish"; out.mkdir()
    item = v2.publish_checkpoint(Adapter(), out, prov, {"updates": 1, "supervised_tokens_seen": 9},
                                 {"optimizer_updates": 1, "processed_context_tokens": 50, "supervised_target_tokens": 9}, spec)
    identity = inspect_adapter("v2", item["path"], corpus["config"]["models"]["student"], corpus["snapshot"])
    assert identity["rank"] == 64 and identity["adapter_weights_sha256"] == file_hash(corpus["theta"] / "adapter_model.safetensors")


@pytest.mark.parametrize("key", ["root", "theta", "snapshot"])
def test_refuses_output_inside_immutable_input(corpus, key):
    with pytest.raises(ValueError, match="outside the sealed"):
        run(corpus, corpus[key] / "new-run")
    assert not (corpus[key] / "new-run").exists()


def test_cli_dry_run_and_two_gpu_launcher(corpus, tmp_path, monkeypatch):
    from tools import train_v2_arm as cli
    import torch
    torch.set_num_threads(1)
    cfg = tmp_path / "config.json"
    write_json(cfg, corpus["config"])
    args = ["run", "--dataset", str(corpus["root"]), "--theta0", str(corpus["theta"]),
            "--snapshot", str(corpus["snapshot"]), "--config", str(cfg), "--updates", "1",
            "--batch-examples", "4", "--replay-ratio", ".25", "--checkpoints", "1", "--gpus", "2",
            "--output", str(tmp_path / "cli")]
    assert cli.main([*args, "--dry-run"]) == 0
    calls = []
    from types import SimpleNamespace
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: calls.append(command) or SimpleNamespace(returncode=0))
    assert cli.main(args) == 0
    assert calls[0][1:6] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2"]


def move_corpus(c, tmp_path_factory):
    """Copy no seals: move their filesystem intact to a second host prefix."""
    old = c['root'].parent
    new = tmp_path_factory.mktemp('hpg')
    for child in old.iterdir():
        shutil.move(str(child), new / child.name)
    c['root'] = new / c['root'].name
    return old, new, [f'{old}={new}']


def test_relocation_component_prefixes_order_and_single_substitution(tmp_path):
    from repair.v2_inputs import Relocation
    mapping = Relocation(tmp_path / 'sealed', ['/rai/a=/hpg/a', '/rai/b/cache=/hpg/cache'])
    assert [str(old) for old, _ in mapping.rules] == ['/rai/b/cache', '/rai/a']
    assert mapping.path('/rai/a/dir/file') == Path('/hpg/a/dir/file')
    assert mapping.path('/rai/abc/file') == Path('/rai/abc/file')
    assert mapping.path('relative/file', root=tmp_path) == tmp_path / 'relative/file'
    assert mapping.path('/rai/a') == Path('/hpg/a')
    assert mapping.path('/rai/a/dir/file') == Path('/hpg/a/dir/file')
    assert mapping.report()['relocated_paths_per_prefix']['/rai/a'] == 2
    one = Relocation(tmp_path / 'sealed', ['/rai/a=/rai/a/moved'])
    assert one.path('/rai/a/file') == Path('/rai/a/moved/file')
    with pytest.raises(v2.ValidationError, match='relative tail'):
        mapping.path('/rai/a/../b/file')


@pytest.mark.parametrize('rules,match', [
    (['/rai/a'], 'OLD=NEW'), (['rai/a=/hpg/a'], 'absolute'),
    (['/rai/a=hpg/a'], 'absolute'), (['/rai/a=/hpg/a', '/rai/a/=/hpg/b'], 'overlapping'),
    (['/rai/a=/hpg/a', '/rai/a/b=/hpg/b'], 'overlapping'),
    (['/rai/a/b=/hpg/b', '/rai/a=/hpg/a'], 'overlapping'),
    (['/rai/a/../b=/hpg/a'], 'relative tail'), (['/rai/a=/hpg/a/../b'], 'relative tail'),
])
def test_relocation_rule_refusals(tmp_path, rules, match):
    from repair.v2_inputs import Relocation
    with pytest.raises(v2.ValidationError, match=match):
        Relocation(tmp_path / 'sealed', rules)


def test_relocation_rejects_files_and_destinations_inside_seal(tmp_path):
    from repair.v2_inputs import Relocation
    sealed = tmp_path / 'sealed'; sealed.mkdir()
    file = tmp_path / 'file'; file.write_text('file')
    for rule in (f'{file}=/new', f'/old={file}'):
        with pytest.raises(v2.ValidationError, match='directory'):
            Relocation(sealed, [rule])
    alias = tmp_path / 'alias'; alias.symlink_to(sealed, target_is_directory=True)
    for new in (sealed, sealed / 'child', alias / 'child'):
        with pytest.raises(v2.ValidationError, match='outside the sealed'):
            Relocation(sealed, [f'/old={new}'])


def test_relocation_replay_theta0_and_default_refusal(corpus, tmp_path_factory):
    before = read(corpus)
    original = {p.relative_to(corpus['root']): p.read_bytes()
                for p in corpus['root'].rglob('*') if p.is_file()}
    old, new, rules = move_corpus(corpus, tmp_path_factory)
    after = v2.read_dataset(corpus['root'], theta0=corpus['theta'], config=corpus['config'], relocate=rules,
                            splits=corpus['splits'])
    assert after['rows'] == before['rows']
    assert after['manifests'] == before['manifests']
    assert all(p.is_file() for p in after['inputs'])
    assert {p.relative_to(corpus['root']): p.read_bytes()
            for p in corpus['root'].rglob('*') if p.is_file()} == original
    assert after['relocation']['all_hashes_verified']
    assert after['relocation']['map'] == [{'old': str(old), 'new': str(new)}]
    with pytest.raises(v2.ValidationError, match='missing file'):
        read(corpus)


@pytest.mark.parametrize('fault', ['hash', 'missing', 'symlink'])
def test_relocation_failures_precede_any_model_load(corpus, tmp_path_factory, monkeypatch, fault):
    import os
    def forbidden(*args, **kwargs):
        raise AssertionError('No model/tokenizer allocation before input verification')
    monkeypatch.setattr(v2, 'load_existing_adapter', forbidden)
    monkeypatch.setattr('train.dataset.load_tokenizer', forbidden)
    # Put the failing input only in replay, so root-only verification cannot pass.
    bad = corpus['root'].parent / 'replay-only.txt'; bad.write_text('sealed')
    manifest = corpus['root'] / 'replay/manifest.json'
    doc = json.loads(manifest.read_text()); doc['inputs'][str(bad)] = file_hash(bad); write_json(manifest, doc)
    old, new, rules = move_corpus(corpus, tmp_path_factory)
    target = new / bad.name
    if fault == 'hash':
        stat = target.stat(); target.write_text('edited')
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert target.stat().st_size == stat.st_size
        match = '1 offending paths.*sha256 mismatch'
    elif fault == 'missing':
        target.unlink(); match = '1 offending paths.*missing file'
    else:
        external = tmp_path_factory.mktemp('outside') / 'same.txt'; external.write_text('sealed')
        target.unlink(); target.symlink_to(external); match = 'symlink escape'
    with pytest.raises(v2.ValidationError, match=match):
        run(corpus, new / 'run', dry_run=False, relocate=rules)
    assert not (new / 'run').exists()


def test_relocation_reports_first_missing_paths_and_total(corpus, tmp_path_factory):
    from repair.v2_inputs import inspect_inputs
    manifest = corpus['root'] / 'manifest.json'
    doc = json.loads(manifest.read_text())
    for i in range(7):
        doc['inputs'][str(corpus['root'].parent / f'missing-{i}.txt')] = '0' * 64
    write_json(manifest, doc)
    _, new, rules = move_corpus(corpus, tmp_path_factory)
    with pytest.raises(v2.ValidationError, match='7 offending paths') as error:
        inspect_inputs(corpus['root'], relocate=rules)
    assert str(new / 'missing-0.txt') in str(error.value)
    assert str(new / 'missing-4.txt') in str(error.value)
    assert 'missing-5.txt' not in str(error.value)


def test_inputs_cli_list_check_exclusive_output_and_no_model_import(corpus, tmp_path_factory, monkeypatch, capsys):
    from tools import train_v2_arm as cli
    import builtins
    cfg = corpus['root'].parent / 'run-config.json'; write_json(cfg, corpus['config'])
    old, new, rules = move_corpus(corpus, tmp_path_factory)
    original_import = builtins.__import__
    def cpu_import(name, *args, **kwargs):
        assert name.split('.')[0] not in ('torch', 'transformers', 'peft')
        assert name not in ('repair.v2_train', 'train.dataset')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', cpu_import)
    args = ['inputs', '--dataset', str(corpus['root']), '--relocate', rules[0],
            '--config', str(cfg), '--snapshot', str(corpus['snapshot']), '--splits', str(corpus['splits'])]
    assert cli.main([*args, '--list']) == 0
    listing = json.loads(capsys.readouterr().out)
    entries = [entry for group in listing['groups'] for entry in group['files']]
    assert len(entries) == len({e['path'] for e in entries}) == listing['file_count']
    assert all(e['size_bytes'] == Path(e['path']).stat().st_size for e in entries)
    needed = [new / 'method_v1_splits.json', new / 'baseline/checkpoint/adapter_model.safetensors',
              new / 'baseline/provenance.json', corpus['root'] / 'replay/manifest.json', new / cfg.name]
    assert all(str(p) in {e['path'] for e in entries} for p in needed)
    assert not listing['relocation']['all_hashes_verified']
    assert cli.main([*args, '--check']) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked['status'] == 'verified' and checked['relocation']['all_hashes_verified']
    assert checked['file_count'] == listing['file_count']
    out = new / 'inventory.json'
    assert cli.main([*args, '--list', '--output', str(out)]) == 0
    assert json.loads(out.read_text()) == listing
    assert cli.main([*args, '--list', '--output', str(out)]) == 1
    assert json.loads(out.read_text()) == listing
    capsys.readouterr()
    assert cli.main([*args, '--list', '--output', str(corpus['root'] / 'new.json')]) == 1
    assert not (corpus['root'] / 'new.json').exists()


def test_relocated_run_provenance_and_cli_config(corpus, tmp_path_factory):
    from tools import train_v2_arm as cli
    import torch
    torch.set_num_threads(1)
    expected = digest(schedule(read(corpus)['rows'], updates=1))
    cfg = corpus['root'].parent / 'run-config.json'; write_json(cfg, corpus['config'])
    sealed = json.loads((corpus['root'] / 'manifest.json').read_text())
    old, new, rules = move_corpus(corpus, tmp_path_factory)
    out = new / 'run'
    args = ['run', '--dataset', str(corpus['root']), '--theta0', str(corpus['theta']),
            '--snapshot', str(corpus['snapshot']), '--config', str(cfg), '--updates', '1',
            '--batch-examples', '4', '--replay-ratio', '.25', '--checkpoints', '1', '--gpus', '1',
            '--output', str(out), '--dry-run', '--relocate', rules[0]]
    assert cli.main(args) == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    provenance = json.loads((out / 'provenance.json').read_text())
    assert manifest['schedule_sha256'] == expected
    assert manifest['input_manifests'][0] == sealed
    report = manifest['relocation']
    assert report == provenance['details']['relocation']
    assert report['map'] == [{'old': str(old), 'new': str(new)}]
    assert report['all_hashes_verified'] and report['verified_hashed_files'] > 0
    assert report['relocated_paths_per_prefix'][str(old)] == 6  # theta, snapshot, config, manifest/freeze, splits
    assert manifest['checkpoints'][0]['path'] == str(new / 'baseline/checkpoint')
    assert str(new / cfg.name) in provenance['inputs']
    assert json.loads((corpus['root'] / 'manifest.json').read_text()) == sealed


def test_theta0_training_view_reference_relocated_without_rewriting_provenance(corpus, tmp_path_factory, monkeypatch):
    view = corpus['root'].parent / 'k20'; view.mkdir()
    support = {'K': 20, 'verified': [{'task_id': r['task_id'], 'group_id': r['group_id']} for r in corpus['replay']]}
    write_json(view / 'manifest.json', support)
    write_json(view / 'selection.json', {})
    write_json(view / 'identity.json', {})
    provenance = corpus['theta'].parent / 'provenance.json'
    doc = json.loads(provenance.read_text())
    doc['config']['baseline_training']['verified'] = str(view)
    doc['inputs'] = {str(view / 'manifest.json'): file_hash(view / 'manifest.json')}
    write_json(provenance, doc)
    split_doc = json.loads(corpus['splits'].read_text()); del split_doc['sets']['K20']
    write_json(corpus['splits'], split_doc)
    reseal(corpus)
    seal_bundle(corpus['root'] / 'replay', corpus['replay'], corpus['theta'], corpus['splits'], 'CONTINUE')
    old, new, rules = move_corpus(corpus, tmp_path_factory)
    calls = []
    def fake_collection(root, *_):
        calls.append(root)
        return support, {}, root
    monkeypatch.setattr('sed.seed_evidence.load_verified_collection', fake_collection)
    result = v2.read_dataset(corpus['root'], theta0=corpus['theta'], config=corpus['config'], relocate=rules)
    assert calls == [new / 'k20']
    assert result['relocation']['all_hashes_verified']
    assert json.loads((new / 'baseline/provenance.json').read_text()) == doc
    from repair.v2_inputs import inspect_inputs
    inventory = inspect_inputs(corpus['root'], relocate=rules)
    assert {new / 'k20' / name for name in ('manifest.json', 'selection.json', 'identity.json')} <= inventory['files'].files.keys()
    (new / 'k20/manifest.json').write_text('{}')
    with pytest.raises(v2.ValidationError, match="K20 view differs"):
        v2.read_dataset(corpus['root'], theta0=corpus['theta'], config=corpus['config'], relocate=rules)


def test_inputs_preserve_hf_aliases_within_new_prefix(corpus, tmp_path_factory):
    from repair.v2_inputs import inspect_inputs
    old = corpus['root'].parent
    blob = old / 'blobs/tokenizer'; blob.parent.mkdir(); blob.write_text('same bytes')
    alias = corpus['snapshot'] / 'extra_tokenizer.json'
    alias.symlink_to('../blobs/tokenizer')
    old, new, rules = move_corpus(corpus, tmp_path_factory)
    result = inspect_inputs(corpus['root'], relocate=rules)
    entries = [e for group in result['files'].listing()['groups'] for e in group['files']]
    entry = next(e for e in entries if e['path'] == str(new / 'snapshot/extra_tokenizer.json'))
    assert str(new / 'blobs/tokenizer') in {e['path'] for e in entries}
    assert entry['resolved_path'] == str(new / 'blobs/tokenizer')
    assert entry['size_bytes'] == len('same bytes')
    assert result['files'].report()['all_hashes_verified']


@pytest.fixture
def collection_corpus(tmp_path, monkeypatch):
    """Real collection validator, 24 fake executions, hash-bound K20 view."""
    from test_repair_e2_reader import build_corpus
    from sed import hash_receipts
    c = build_corpus(tmp_path)
    support = json.loads((c.support / 'manifest.json').read_text())
    view = json.loads((c.view / 'manifest.json').read_text())
    replay_ids = {r['task_id'] for r in view['verified']}
    new = [r for r in support['verified'] if r['task_id'] not in replay_ids]
    splits = tmp_path / 'method_v1_splits.json'
    sets = {name: {'task_ids': [], 'group_ids': []} for name in v2.FORBIDDEN_SETS}
    sets['V2_pool'] = {key: sorted({r[key[:-1]] for r in new}) for key in ('task_ids', 'group_ids')}
    write_json(splits, {'sets': sets})
    rows = [example(r['task_id'], 'offline_teacher', 'trajectory', 'OFFLINE') for r in new]
    replay = [example(r['task_id'], 'k20_replay', 'trajectory', 'CONTINUE') for r in view['verified']]
    root = tmp_path / 'interchange'
    seal_bundle(root, rows, c.theta, splits, 'OFFLINE')
    seal_bundle(root / 'replay', replay, c.theta, splits, 'CONTINUE')
    # The real bundles also reference selected evidence directly. These seals
    # must compare with collection hashes without an early second byte pass.
    evidence = {}
    for receipt in view['attempts']:
        episode = c.support / receipt['episode_dir']
        for directory, key in ((episode, 'episode_hashes'), (episode.parent / 'teacher', 'teacher_hashes')):
            evidence.update({str(directory / name): sha for name, sha in receipt[key].items()})
        evidence[str(episode.parent / 'receipt.json')] = file_hash(episode.parent / 'receipt.json')
    for directory in (root, root / 'replay'):
        doc = json.loads((directory / 'manifest.json').read_text())
        doc['inputs'].update(evidence)
        write_json(directory / 'manifest.json', doc)
    # Start with no receipts from fixture construction; never touch a real bank.
    monkeypatch.setenv('SED_HASH_CACHE', '1')
    monkeypatch.setenv('SED_HASH_CACHE_DIR', str(tmp_path / 'verification-cache'))
    hash_receipts._MEMO.clear()
    hash_receipts._MODES.clear()
    return dict(root=root, theta=c.theta, snapshot=c.snapshot, config=c.config, splits=splits,
                support=c.support, view=c.view, collection=support, replay_ids=replay_ids)


def inspect_collection(c, **kwargs):
    from repair.v2_inputs import inspect_inputs
    return inspect_inputs(c['root'], theta0=c['theta'], config=c['config'], **kwargs)['files']


def test_collection_coverage_cold_warm_and_noncollection_bytes(collection_corpus, monkeypatch):
    from repair import v2_inputs
    from sed import hash_receipts, seed_evidence
    c = collection_corpus
    extra = c['support'] / 'unrelated-config.json'
    extra.write_text('{}')
    optimizer = c['theta'] / 'optimizer.pt'
    optimizer.write_bytes(b'unsealed optimizer input')
    snapshot = c['snapshot'] / 'extra-model-file'
    snapshot.write_bytes(b'unsealed snapshot input')
    byte_calls, evidence_calls = [], []
    def byte_hash(path):
        byte_calls.append(Path(path))
        return file_hash(path)
    def evidence_hash(path):
        evidence_calls.append(Path(path))
        return file_hash(path)
    monkeypatch.setattr(v2_inputs, 'file_hash', byte_hash)
    monkeypatch.setattr(seed_evidence, 'file_hash', evidence_hash)
    for warm in (False, True):
        byte_calls.clear(); evidence_calls.clear()
        hash_receipts._MEMO.clear()  # Warm pass must reuse disk receipts.
        files = inspect_collection(c, inputs=[extra])
        report = files.report()
        assert files.collection_covered and not files.collection_covered.intersection(byte_calls)
        assert set(byte_calls) == set(files.hashes)
        assert {extra, optimizer, snapshot, c['splits']} <= set(byte_calls)
        assert all(p in byte_calls for root in (c['root'], c['root'] / 'replay')
                   for p in root.iterdir() if p.is_file())
        assert report['collection_validation_roots'] == sorted(map(str, (c['support'], c['view'])))
        assert report['collection_covered_files'] + report['byte_hashed_files'] == len(files.files)
        assert report['hash_cache'] == hash_receipts.cache_provenance()
        assert report['collection_validation_anchors'][str(c['support'])]['semantic_sha256']['manifest.json'] == digest(c['collection'])
        assert not report['full_rehash']
        # Subset receipt references always hash; warm episode/teacher trees don't.
        assert bool([p for p in evidence_calls if p.name != 'receipt.json']) is (not warm)
        evidence = next(p for p in files.collection_covered if p.name == 'episode.json')
        alias = c['root'].parent / 'evidence-alias'
        if not alias.exists():
            alias.symlink_to(evidence)
        assert alias not in files.provenance_inputs([alias])
        assert not set(files.provenance_inputs()).intersection(files.collection_covered)


@pytest.mark.parametrize('fault', ['episode', 'teacher', 'receipt', 'missing', 'selection'])
def test_collection_tampering_refused_by_real_loader(collection_corpus, monkeypatch, fault):
    from repair import v2_inputs
    c = collection_corpus
    inspect_collection(c)  # Prime receipts before tampering with an unselected task.
    receipt = next(r for r in c['collection']['attempts'] if r['task_id'] not in c['replay_ids'])
    episode = c['support'] / receipt['episode_dir']
    target = {'episode': episode / 'episode.json', 'missing': episode / 'episode.json',
              'teacher': episode.parent / 'teacher' / next(iter(receipt['teacher_hashes'])),
              'receipt': episode.parent / 'receipt.json', 'selection': c['support'] / 'selection.json'}[fault]
    if fault == 'missing':
        target.unlink()
    else:
        if fault in ('selection', 'receipt'):
            doc = json.loads(target.read_text()); doc['tampered'] = True
            write_json(target, doc)
        else:
            target.write_bytes(target.read_bytes() + b' ')
    def no_second_hash(path):
        assert not Path(path).is_relative_to(c['support'] / 'attempts')
        return file_hash(path)
    monkeypatch.setattr(v2_inputs, 'file_hash', no_second_hash)
    with pytest.raises(v2.ValidationError, match='collection validation failed'):
        inspect_collection(c)


@pytest.mark.parametrize('mode', ['flag', 'environment'])
def test_full_rehash_covers_all_and_disables_receipts(collection_corpus, monkeypatch, mode):
    from repair import v2_inputs
    from sed import hash_receipts
    c = collection_corpus
    inspect_collection(c)
    if mode == 'environment':
        monkeypatch.setenv('SED_HASH_CACHE', '0')
    def forbidden(*args, **kwargs):
        raise AssertionError('Full rehash must bypass memory and disk receipts')
    monkeypatch.setattr(hash_receipts.HashReceipts, 'hashes', forbidden)
    calls = []
    def counted(path):
        calls.append(Path(path))
        return file_hash(path)
    monkeypatch.setattr(v2_inputs, 'file_hash', counted)
    files = inspect_collection(c, full_rehash=(mode == 'flag'))
    assert set(calls) == set(files.files) == set(files.provenance_inputs())
    report = files.report()
    assert report['collection_covered_files'] == 0
    assert report['byte_hashed_files'] == len(files.files)
    assert report['full_rehash'] and report['hash_cache']['mode'] == 'full_rehash'
    assert os.environ['SED_HASH_CACHE'] == ('1' if mode == 'flag' else '0')


def test_full_rehash_refuses_same_metadata_tamper_and_restores_policy(collection_corpus):
    c = collection_corpus
    inspect_collection(c)
    receipt = c['collection']['attempts'][0]
    path = c['support'] / receipt['episode_dir']
    target = path.parent / 'teacher' / next(iter(receipt['teacher_hashes']))
    before = target.stat()
    target.write_bytes(b'x' * before.st_size)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(v2.ValidationError, match='collection validation failed'):
        inspect_collection(c, full_rehash=True)
    assert os.environ['SED_HASH_CACHE'] == '1'


def test_collection_coverage_still_checks_interchange_seals(collection_corpus):
    c = collection_corpus
    manifest = c['root'] / 'manifest.json'
    doc = json.loads(manifest.read_text())
    evidence = next(p for p in doc['inputs'] if '/episode/' in p)
    doc['inputs'][evidence] = '0' * 64
    write_json(manifest, doc)
    with pytest.raises(v2.ValidationError, match='Interchange input sha256 mismatch'):
        inspect_collection(c)


@pytest.mark.parametrize('full', [False, True])
def test_collection_cli_check_run_and_provenance(collection_corpus, monkeypatch, capsys, full):
    import torch
    from tools import train_v2_arm as cli
    from sed import provenance as sed_provenance
    c = collection_corpus
    torch.set_num_threads(1)
    cfg = c['root'].parent / 'config.json'
    write_json(cfg, c['config'])
    args = ['--dataset', str(c['root']), '--theta0', str(c['theta']), '--config', str(cfg)]
    if full:
        args.append('--full-rehash')
    assert cli.main(['inputs', '--check', *args]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked['prep_seconds'] > 0 and checked['relocation']['full_rehash'] is full
    provenance_calls = []
    def counted(path):
        provenance_calls.append(Path(path).resolve())
        return file_hash(path)
    monkeypatch.setattr(sed_provenance, 'file_hash', counted)
    output = c['root'].parent / 'run'
    assert cli.main(['run', *args, '--snapshot', str(c['snapshot']), '--output', str(output),
                     '--updates', '1', '--batch-examples', '4', '--replay-ratio', '.25',
                     '--checkpoints', '1', '--gpus', '1', '--dry-run']) == 0
    prov = json.loads((output / 'provenance.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    report = prov['details']['relocation']
    assert report == manifest['relocation']
    assert report['hash_cache'] == prov['hash_cache']
    assert report['full_rehash'] is prov['config']['v2_training']['full_rehash'] is full
    assert bool(report['collection_covered_files']) is (not full)
    assert bool([p for p in provenance_calls if p.is_relative_to(c['support'] / 'attempts')]) is full
    assert prov['inputs'][str(cfg)] == file_hash(cfg)
    assert os.environ['SED_HASH_CACHE'] == '1'


def test_inputs_list_refuses_full_rehash(capsys):
    from tools import train_v2_arm as cli
    with pytest.raises(SystemExit) as exc:
        cli.main(['inputs', '--dataset', 'unused', '--list', '--full-rehash'])
    assert exc.value.code == 2
    assert '--full-rehash requires inputs --check' in capsys.readouterr().err


def test_relocated_theta0_mismatch_in_check(corpus, tmp_path_factory, capsys):
    from tools import train_v2_arm as cli
    _, new, rules = move_corpus(corpus, tmp_path_factory)
    (new / 'baseline/checkpoint/adapter_model.safetensors').write_bytes(b'wrong checkpoint')
    assert cli.main(['inputs', '--dataset', str(corpus['root']), '--relocate', rules[0], '--check']) == 1
    error = capsys.readouterr().err
    assert 'theta0 adapter hash' in error and '1 offending paths' in error
