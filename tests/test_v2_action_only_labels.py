"""V2 fallback labels and shared CPU preflight validation; synthetic inputs only."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from repair import v2_train as v2
from repair import v2_validation
from repair.v2_inputs import inspect_inputs
from sed.storage import canonical_bytes, digest, write_json
from test_v2_train import corpus, example, read, reseal, seal_bundle
from tools import train_v2_arm as cli
from tools.v2_encode_check import encode_example_old


ALLOWED_PAIRS = [
    ("student_thought", "passed"),
    ("teacher_rationale", "passed"),
    ("teacher_reply", "passed"),
    ("action_only_no_thought", "passed"),
    ("action_only_ungrounded_student", "label_instances_not_public"),
    ("action_only_invalid_label", "label_grounding_failed"),
    ("action_only_invalid_label", "reasoning_token_cap"),
    ("action_only_invalid_label", "invalid_json"),
    ("action_only_invalid_label", "reasoning_control_text"),
    ("action_only_invalid_label", "builder_error_2"),
]
ACTION_ONLY_PAIRS = [pair for pair in ALLOWED_PAIRS if pair[0].startswith("action_only_")]
REJECTED_PAIRS = [
    ("action_only_no_label", "not_available"),
    ("unknown_source", "passed"),
    ("action_only_unknown", "passed"),
    ("action_only_invalid_label", "passed"),
    ("action_only_invalid_label", "INVALID_JSON"),
    ("action_only_invalid_label", "invalid json"),
    ("action_only_invalid_label", ""),
    ("action_only_invalid_label", "invalid_json\n"),
    ("action_only_invalid_label", "1invalid_json"),
    ("action_only_invalid_label", "invalid-json"),
    ("action_only_ungrounded_student", "passed"),
    ("action_only_no_thought", "label_grounding_failed"),
    ("teacher_rationale", "label_grounding_failed"),
]


def set_label(row, source, check):
    row["evidence"].update(label_source=source, label_check=check)
    if source.startswith("action_only_"):
        for i in row["supervised_assistant_indices"]:
            row["messages"][i]["content"] = "Action: look"


def check_args(c):
    config = c["root"].parent / "config.json"
    write_json(config, c["config"])
    return ["inputs", "--check", "--dataset", str(c["root"]), "--theta0", str(c["theta"]),
            "--snapshot", str(c["snapshot"]), "--config", str(config)]


@pytest.mark.parametrize("source,check", ALLOWED_PAIRS)
def test_allowed_label_pair_preserves_row_and_encoding(corpus, source, check):
    row = corpus["rows"][0]
    set_label(row, source, check)
    reseal(corpus)
    before = canonical_bytes(row)
    v2._validate_evidence(row)
    loaded = next(r for r in read(corpus)["rows"] if r["example_id"] == row["example_id"])
    assert canonical_bytes(loaded) == before == canonical_bytes(row)
    tokenizer = v2.FakeTokenizer()
    assert digest(v2.encode_example(loaded, tokenizer)) == digest(encode_example_old(row, tokenizer))
    # The earlier Thought is context only and must remain intact.
    assert row["messages"][2]["content"].startswith("Thought:")


@pytest.mark.parametrize("source,check", REJECTED_PAIRS)
def test_rejected_label_pair(source, check):
    row = example("train/room_a/trial_0", "joint_local", "decision")
    set_label(row, source, check)
    with pytest.raises(ValueError, match="^Invalid or unchecked label evidence$"):
        v2._validate_evidence(row)


@pytest.mark.parametrize("source,check", ACTION_ONLY_PAIRS)
def test_action_only_trajectory_is_rejected(source, check):
    row = example("train/room_b/trial_0", "k20_replay", "trajectory", "CONTINUE")
    set_label(row, source, check)
    with pytest.raises(ValueError, match="^Invalid or unchecked label evidence$"):
        v2._validate_evidence(row)


@pytest.mark.parametrize("target", [
    "Thought: Inspect once more.\nAction: look",
    "Action: look\nAction: look",
    "Action: look\n",
    "Action: ",
    "look",
])
@pytest.mark.parametrize("source,check", ACTION_ONLY_PAIRS)
def test_action_only_rejects_non_action_target(source, check, target):
    row = example("train/room_a/trial_0", "joint_local", "decision")
    set_label(row, source, check)
    row["messages"][-1]["content"] = target
    with pytest.raises(ValueError, match="^Action-only label with non-action target$"):
        v2._validate_evidence(row)


@pytest.mark.parametrize("index", [2, 4])
def test_action_only_checks_every_supervised_target(index):
    row = example("train/room_a/trial_0", "joint_local", "decision")
    row["supervised_assistant_indices"] = [2, 4]
    set_label(row, "action_only_no_thought", "passed")
    row["messages"][index]["content"] = "Thought: Inspect.\nAction: look"
    with pytest.raises(ValueError, match="^Action-only label with non-action target$"):
        v2._validate_evidence(row)


@pytest.mark.parametrize("bundle", ["main", "replay"])
def test_inputs_check_refuses_bad_evidence_like_run(corpus, capsys, bundle):
    row = corpus["rows" if bundle == "main" else "replay"][0]
    set_label(row, "action_only_no_label", "not_available")
    if bundle == "main":
        reseal(corpus)
    else:
        seal_bundle(corpus["root"] / "replay", corpus["replay"], corpus["theta"],
                    corpus["splits"], "CONTINUE")
    with pytest.raises(ValueError, match="^Invalid or unchecked label evidence$"):
        read(corpus)
    assert cli.main(check_args(corpus)) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "V2 training refused: Invalid or unchecked label evidence"
    assert not captured.out


def test_inputs_check_refuses_non_action_target(corpus, capsys):
    row = corpus["rows"][0]
    set_label(row, "action_only_invalid_label", "invalid_json")
    row["messages"][-1]["content"] = "Thought: Inspect.\nAction: look"
    reseal(corpus)
    assert cli.main(check_args(corpus)) == 1
    assert capsys.readouterr().err.strip() == "V2 training refused: Action-only label with non-action target"


def test_inputs_check_counts_labels_with_shared_reader_and_no_allocation(corpus, monkeypatch, capsys):
    from train import dataset

    rows = []
    for step, (source, check) in enumerate(ALLOWED_PAIRS):
        row = example("train/room_a/trial_0", "joint_local", "decision", step=step)
        set_label(row, source, check)
        rows.append(row)
    corpus["rows"] = rows
    reseal(corpus)
    before = deepcopy(rows)
    expected_labels = dict(Counter(
        f'{row["evidence"]["label_source"]}|{row["evidence"]["label_check"]}'
        for row in rows + corpus["replay"]))
    expected_count = len(rows) + len(corpus["replay"])
    called = []
    read_bundle = v2._read_bundle
    assert read_bundle is v2_validation._read_bundle

    def tracked(bundle, *args, **kwargs):
        called.append(bundle["root"])
        return read_bundle(bundle, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must not allocate a tokenizer or model, or encode examples")

    monkeypatch.setattr(v2_validation, "_read_bundle", tracked)
    for name in ("FakeTokenizer", "fake_model", "encode_example", "inherited_recipe", "load_existing_adapter"):
        monkeypatch.setattr(v2, name, forbidden)
    monkeypatch.setattr(dataset, "load_tokenizer", forbidden)
    assert cli.main(check_args(corpus)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert result["examples_validated"] == expected_count
    assert result["label_provenance"] == expected_labels
    assert called == [corpus["root"], corpus["root"] / "replay"]
    assert rows == before

    # The metadata-only reader branch uses the same checks and summary too.
    checked = inspect_inputs(corpus["root"], theta0=corpus["theta"], config=corpus["config"], runtime=False)
    assert checked["examples_validated"] == expected_count
    assert checked["label_provenance"] == expected_labels


def test_inputs_list_does_not_validate_examples(corpus, capsys):
    set_label(corpus["rows"][0], "action_only_no_label", "not_available")
    reseal(corpus)
    args = check_args(corpus)
    args[args.index("--check")] = "--list"
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert "examples_validated" not in result
    assert "label_provenance" not in result
