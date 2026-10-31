"""Selected-turn encoding must preserve all training bytes and history guards."""

import json
from pathlib import Path
import random

import pytest

from baselines.common.conversation import conversation, encode_messages, episode_messages
from repair.v2_train import FakeTokenizer, encode_example
from sed.storage import canonical_bytes, digest
from tools import v2_encode_check as check
from train import dataset


def episode(count=6):
    def visible(index):
        return {"observation": f"Room {index}: café, apple 1 — inspect.",
                "admissible_commands": ["look", "take apple 1"]}

    return {"schema_version": 2, "task_id": "train/encode/trial_0", "split": "train", "sealed": False,
            "initial": {"visible": visible(0)},
            "steps": [{"action_raw": f"Thought: Check apple 1 at step {index}.\nAction: look",
                       "event_id": f"step-{index}", "visible": visible(index + 1),
                       "action_outcome": "legal", "audit": "HIDDEN_AUDIT"}
                      for index in range(count)]}


def example(indices, *, count=6, kind="decision", arm="PREFIX_EXPERT", source="prefix_expert"):
    return {"example_id": digest([indices, count, kind, arm, source]),
            "arm": arm, "source": source, "kind": kind, "task_id": "train/encode/trial_0",
            "group_id": "train/encode", "task_type": "pick_and_place_simple",
            "messages": episode_messages(episode(count)),
            "supervised_assistant_indices": [2 + 2 * index for index in indices], "weight": 1.0,
            "evidence": {"episode_ref": "synthetic.json", "step": indices[-1] if kind == "decision" else None,
                         "executor": "teacher", "local_event": None, "episode_success": True,
                         "label_source": "teacher_reply", "label_check": "passed"}}


def rows():
    return [
        *(example([count - 1], count=count) for count in (1, 4, 16)),
        *(example([index], count=16, arm="JOINT", source="joint_local") for index in (0, 7, 15)),
        example([0, 3, 5]),
        # Exercise the encoder's order preservation independently of the loader,
        # which currently restricts decision rows to their final assistant turn.
        example([5, 1, 3, 1]),
        example(list(range(6)), kind="trajectory", arm="OFFLINE", source="offline_teacher"),
        example(list(range(16)), count=16, kind="trajectory", arm="CONTINUE", source="k20_replay"),
    ]


def assert_identical(row, tokenizer):
    before = canonical_bytes(row)
    old = check.encode_example_old(row, tokenizer)
    new = encode_example(row, tokenizer)
    assert canonical_bytes(new) == canonical_bytes(old)
    assert digest(new) == digest(old)
    assert canonical_bytes(row) == before
    assert [turn["label_provenance"]["turn_index"] for turn in new["turns"]] == [
        (index - 2) // 2 for index in row["supervised_assistant_indices"]]
    return old, new


def test_fake_tokenizer_bytes_and_encoded_examples_sha256_unchanged():
    pairs = [assert_identical(row, FakeTokenizer()) for row in rows()]
    # This is exactly the digest used by v2_training.encoded_examples_sha256.
    assert digest([old for old, _ in pairs]) == digest([new for _, new in pairs])


@pytest.fixture(scope="module")
def real_tokenizer():
    snapshot = Path.home() / (
        ".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/"
        "ee0ef6023621cff504d758262d4e04895a5af4a2")
    if not snapshot.is_dir():
        pytest.skip("Pinned E4B tokenizer snapshot is not installed locally")
    return dataset.load_tokenizer(snapshot)


def test_real_tokenizer_bytes_and_encoded_examples_sha256_unchanged(real_tokenizer):
    selected = [rows()[index] for index in (2, 4, 7, 8, 9)]
    pairs = [assert_identical(row, real_tokenizer) for row in selected]
    assert digest([old for old, _ in pairs]) == digest([new for _, new in pairs])


@pytest.mark.parametrize("position", [0, 5])
@pytest.mark.parametrize("violation", ["", "hidden"])
def test_unsupervised_empty_or_hidden_reply_rejected_exactly_as_before(position, violation):
    row = example([3])
    row["messages"][2 + 2 * position]["content"] = (
        "<|channel>thought\nSECRET<channel|>Action: look" if violation else "")
    with pytest.raises(ValueError) as old:
        check.encode_example_old(row, FakeTokenizer())
    with pytest.raises(type(old.value)) as new:
        encode_example(row, FakeTokenizer())
    assert str(new.value) == str(old.value)
    assert ("Hidden thought channel" if violation else "visible teacher reply") in str(new.value)


@pytest.mark.parametrize("tokenizer_fixture", ["fake", "real_tokenizer"])
def test_supervised_empty_reply_rejected_exactly_as_before(tokenizer_fixture, request):
    tokenizer = FakeTokenizer() if tokenizer_fixture == "fake" else request.getfixturevalue(tokenizer_fixture)
    row = example([5])
    row["messages"][-1]["content"] = ""
    with pytest.raises(ValueError) as old:
        check.encode_example_old(row, tokenizer)
    with pytest.raises(type(old.value)) as new:
        encode_example(row, tokenizer)
    assert str(new.value) == str(old.value) == "Template transformed or removed the visible teacher reply"


def test_only_supervised_steps_are_tokenized(monkeypatch):
    original = dataset.encode_turn
    calls = []

    def counted(tokenizer, messages, reply, **kwargs):
        calls.append((len(messages) - 2) // 2)
        return original(tokenizer, messages, reply, **kwargs)

    monkeypatch.setattr(dataset, "encode_turn", counted)
    for row in rows():
        calls.clear()
        encode_example(row, FakeTokenizer())
        assert calls == sorted({(i - 2) // 2 for i in row["supervised_assistant_indices"]})


def test_episode_selection_is_sorted_deduplicated_and_preserves_every_field():
    value, tokenizer = episode(), FakeTokenizer()
    full = dataset.episode_examples(value, tokenizer, packing="turn", turns=None)
    selected = dataset.episode_examples(value, tokenizer, packing="turn", turns=[4, 0, 4, 2])
    assert canonical_bytes(selected) == canonical_bytes([full[index] for index in (0, 2, 4)])


@pytest.mark.parametrize("selected", [{3}, {1, 3}, {0, 1, 3}])
def test_allowed_empty_history_preserved_and_only_selected_steps_encoded(monkeypatch, selected):
    value, tokenizer = episode(4), FakeTokenizer()
    value["steps"][1]["action_raw"] = ""
    full = dataset.episode_examples(value, tokenizer, packing="turn", allow_empty_visible=True)
    calls = []
    for name in ("encode_turn", "_encode_empty_history"):
        original = getattr(dataset, name)

        def counted(tokenizer, messages, *args, _original=original, **kwargs):
            calls.append((len(messages) - 2) // 2)
            return _original(tokenizer, messages, *args, **kwargs)

        monkeypatch.setattr(dataset, name, counted)
    actual = dataset.episode_examples(value, tokenizer, packing="turn", allow_empty_visible=True,
                                      turns=selected)
    assert calls == sorted(selected)
    assert canonical_bytes(actual) == canonical_bytes([full[index] for index in sorted(selected)])
    messages = episode_messages(value)
    full_group = encode_messages(messages, tokenizer, task_id=value["task_id"],
                                 source="student_exploration", packing="turn")
    group = encode_messages(messages, tokenizer, task_id=value["task_id"],
                            source="student_exploration", packing="turn", turns=selected)
    assert group["messages"] == full_group["messages"] == messages
    assert canonical_bytes(group["turns"]) == canonical_bytes([
        full_group["turns"][index] for index in sorted(selected)])


@pytest.mark.parametrize("selected,match", [(set(), "Trajectory has no assistant turns"),
                                           ({1}, "Nothing to score")])
def test_empty_checks_apply_to_returned_rows(selected, match):
    value = episode(3)
    value["steps"][1]["action_raw"] = ""
    with pytest.raises(ValueError, match=match):
        dataset.episode_examples(value, FakeTokenizer(), packing="turn", allow_empty_visible=True,
                                 turns=selected)


@pytest.mark.parametrize("selected", [{0}, set()])
def test_episode_packing_rejects_turn_selection(selected):
    with pytest.raises(ValueError, match="requires packing='turn'"):
        dataset.episode_examples(episode(), FakeTokenizer(), packing="episode", turns=selected)


@pytest.mark.parametrize("selected", [{-1}, {6}, {0, 99}, {True}, {1.5}])
def test_turn_selection_requires_in_range_step_indices(selected):
    with pytest.raises(ValueError, match="Turn index out of range"):
        dataset.episode_examples(episode(), FakeTokenizer(), packing="turn", turns=selected)


@pytest.mark.parametrize("violation", ["hidden", "empty", "visible", "reply_type", "schema", "split"])
def test_selection_preserves_validation_after_last_selected_step(violation):
    value = episode(4)
    if violation == "hidden":
        value["steps"][-1]["action_raw"] = "<|channel>thought\nSECRET<channel|>Action: look"
    elif violation == "empty":
        value["steps"][-1]["action_raw"] = ""
    elif violation == "visible":
        value["steps"][-1]["visible"]["observation"] = 7
    elif violation == "reply_type":
        value["steps"][-1]["action_raw"] = 7
    elif violation == "schema":
        value["schema_version"] = 1
    else:
        value["split"] = "test"
    with pytest.raises((ValueError, TypeError)) as old:
        dataset.episode_examples(value, FakeTokenizer(), packing="turn", turns=None)
    with pytest.raises(type(old.value)) as new:
        dataset.episode_examples(value, FakeTokenizer(), packing="turn", turns={0})
    assert str(new.value) == str(old.value)


@pytest.mark.parametrize("packing", ["turn", "episode"])
def test_default_none_is_identical_for_existing_entry_points(packing):
    value, tokenizer = episode(), FakeTokenizer()
    messages = episode_messages(value)
    for function, args, kwargs in (
        (dataset.episode_examples, (value, tokenizer), {}),
        (conversation, (value, tokenizer), {"source": "trusted_teacher_seed"}),
        (encode_messages, (messages, tokenizer), {"task_id": value["task_id"]}),
    ):
        assert canonical_bytes(function(*args, packing=packing, **kwargs)) == canonical_bytes(
            function(*args, packing=packing, turns=None, **kwargs))


def test_selected_max_length_retains_no_truncation_check():
    row, tokenizer = example([5]), FakeTokenizer()
    length = len(encode_example(row, tokenizer)["turns"][0]["input_ids"])
    assert encode_example(row, tokenizer, max_length=length) == check.encode_example_old(
        row, tokenizer, max_length=length)
    for encoder in (check.encode_example_old, encode_example):
        with pytest.raises(ValueError, match="no truncation"):
            encoder(row, tokenizer, max_length=length - 1)


@pytest.mark.parametrize("limit", [None, 3, 99])
@pytest.mark.parametrize("mismatch", [False, True])
def test_check_cli_sample_summary_and_exit_status(tmp_path, monkeypatch, capsys, limit, mismatch):
    values = rows()
    (tmp_path / "examples.jsonl").write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in values))
    monkeypatch.setattr(check, "load_tokenizer", lambda _: FakeTokenizer())
    seen = []

    def new(row, tokenizer):
        seen.append(row["example_id"])
        encoded = encode_example(row, tokenizer)
        if mismatch:
            encoded["supervised_tokens"] += 1
        return encoded

    monkeypatch.setattr(check, "encode_example", new)
    args = ["--dataset", str(tmp_path), "--snapshot", str(tmp_path), "--seed", "19"]
    if limit is not None:
        args += ["--limit", str(limit)]
    assert check.main(args) == int(mismatch)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    expected = random.Random(19).sample(values, limit) if limit is not None and limit < len(values) else values
    assert seen == [row["example_id"] for row in expected]
    assert report["rows"] == len(expected)
    assert report["identical"] is not mismatch
    assert report["mismatches"] == (seen[:5] if mismatch else [])
    assert report["seconds_old"] >= 0 and report["seconds_new"] >= 0
    assert set(report) == {"rows", "identical", "mismatches", "seconds_old", "seconds_new"}
