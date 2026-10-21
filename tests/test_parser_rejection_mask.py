"""SAD/SmartAD's opt-in H0 rejection mask; SFT keeps its assistant-all recipe."""
from copy import deepcopy

import pytest

from baselines.common.conversation import conversation, sequence_rows
from baselines.common.parser_mask import parser_rejection_counts
from baselines.common.shared_history import pack_shared_history
from baselines.sad.data import encode_episode as sad_encode, spans
from baselines.smartad.data import encode_episode as smartad_encode, segment_spans, WEIGHTS
from test_sad import CharTokenizer, episode
from train.dataset import TURN_ID, TURN_TOKEN


REJECTED = "Thought: I'll look for lettuce in the fridge first.Action: go to fridge 1"


def rejected_trajectory(raw=REJECTED, outcome="parser_error", index=1):
    source = episode("Thought: check\nAction: look")
    source["steps"] = [dict(source["steps"][0], event_id=f"e{i}", action_outcome="legal")
                       for i in range(3)]
    source["steps"][index].update(action_raw=raw, action_outcome=outcome)
    return source


def assert_original_labels(row, raw, tokenizer, *, method, final=False):
    """Original span rules, evaluated only on H0-accepted replies."""
    reply = raw.strip()
    target = tokenizer(reply + TURN_TOKEN, add_special_tokens=False, return_offsets_mapping=True)
    ranges = spans(reply) if method == "sad" else segment_spans(reply, final=final)
    kinds = []
    for token, (left, right) in zip(target["input_ids"], target["offset_mapping"]):
        overlaps = [kind for lo, hi, kind in ranges if left < hi and right > lo]
        if method == "sad":
            assert len(set(overlaps)) <= 1
            kinds.append("action" if token == TURN_ID else next(iter(overlaps), None))
        else:
            kinds.append("reason" if token == TURN_ID else next(iter(overlaps), "reason"))
    start = row["prompt_tokens"]
    expected = [-100] * start + [token if kind else -100
                                for token, kind in zip(target["input_ids"], kinds)]
    assert row["labels"] == expected
    assert row["supervised_tokens"] == sum(v != -100 for v in expected)
    if method == "smartad":
        assert row["segment_labels"] == [None] * start + kinds
        assert row["token_weights"] == [0.0] * start + [WEIGHTS[kind] for kind in kinds]


@pytest.mark.parametrize("encoder,method", [(sad_encode, "sad"), (smartad_encode, "smartad")])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_rejected_reply_labels_history_and_counts(encoder, method, index):
    source = rejected_trajectory(index=index)
    original = deepcopy(source)
    tokenizer = CharTokenizer()
    # This is also the unchanged Kang/AD trajectory-SFT encoding.
    baseline = conversation(source, tokenizer, source="trusted_teacher_seed", packing="turn")
    encoded = encoder(source, tokenizer)
    assert source == original
    assert encoded["messages"] == baseline["messages"]
    assert encoded["messages"][2 + 2 * index]["content"] == REJECTED
    removed = baseline["turns"][index]["supervised_tokens"]
    counts = {"episodes": 1, "replies": 1, "supervised_tokens_removed": removed}
    assert encoded["label_provenance"]["parser_rejection_mask"] == counts
    assert parser_rejection_counts([encoded]) == counts
    assert len(sequence_rows(encoded)) == 2
    for i, (row, old) in enumerate(zip(encoded["turns"], baseline["turns"])):
        for key in ("input_ids", "prompt_tokens", "generation_start", "attention_mask"):
            assert row[key] == old[key]
        if i == index:
            assert set(row["labels"]) == {-100}
            assert row["supervised_tokens"] == 0
            assert row["assistant_spans"] == []
            assert row["label_provenance"]["parser_rejected_supervised_tokens_removed"] == removed
            if method == "sad":
                assert row["reason_tokens"] == row["action_tokens"] == 0
                assert row["sad_spans"] == []
            else:
                assert set(row["token_weights"]) == {0.0}
                assert set(row["segment_labels"]) == {None}
        else:
            assert_original_labels(row, source["steps"][i]["action_raw"], tokenizer,
                                   method=method, final=i == 2)
    packed = pack_shared_history(encoded)
    assert [v for v in packed["labels"] if v != -100] == [
        v for row in encoded["turns"] for v in row["labels"] if v != -100]
    assert encoded["supervised_tokens"] == sum(v != -100 for v in packed["labels"])
    if index < 2:
        assert REJECTED in tokenizer.decode(encoded["turns"][-1]["input_ids"])
        history = [v for v, branch in zip(packed["input_ids"], packed["branches"]) if branch == 0]
        assert REJECTED in tokenizer.decode(history)


@pytest.mark.parametrize("encoder", [sad_encode, smartad_encode])
@pytest.mark.parametrize("raw", [REJECTED, "Thought: no action", "Action: look\nAction:   ", ""])
@pytest.mark.parametrize("recorded", [True, False])
def test_rejection_uses_h0_including_last_empty_action(encoder, raw, recorded):
    source = rejected_trajectory(raw)
    if not recorded:
        del source["steps"][1]["action_outcome"]
    result = encoder(source, CharTokenizer())
    assert result["label_provenance"]["parser_rejection_mask"]["replies"] == 1
    assert set(result["turns"][1]["labels"]) == {-100}


@pytest.mark.parametrize("encoder", [sad_encode, smartad_encode])
@pytest.mark.parametrize("raw,outcome", [(REJECTED, "legal"), (REJECTED, "inadmissible"),
                                        ("Action: look", "parser_error")])
def test_recorded_outcome_mismatch_fails_closed(encoder, raw, outcome):
    with pytest.raises(ValueError, match="action_outcome disagrees with H0"):
        encoder(rejected_trajectory(raw, outcome), CharTokenizer())


@pytest.mark.parametrize("encoder,method", [(sad_encode, "sad"), (smartad_encode, "smartad")])
def test_inadmissible_but_parsed_reply_keeps_original_labels(encoder, method):
    source = rejected_trajectory("Thought: try\nAction: nonexistent command", "inadmissible")
    tokenizer = CharTokenizer()
    result = encoder(source, tokenizer)
    assert result["label_provenance"]["parser_rejection_mask"] == {
        "episodes": 0, "replies": 0, "supervised_tokens_removed": 0}
    for i, row in enumerate(result["turns"]):
        assert_original_labels(row, source["steps"][i]["action_raw"], tokenizer,
                               method=method, final=i == 2)


@pytest.mark.parametrize("encoder", [sad_encode, smartad_encode])
def test_counts_distinguish_episodes_from_replies(encoder):
    source = rejected_trajectory()
    source["steps"][0].update(action_raw=REJECTED, action_outcome="parser_error")
    tokenizer = CharTokenizer()
    encoded = encoder(source, tokenizer)
    clean = encoder(episode("Action: look"), tokenizer)
    assert parser_rejection_counts([encoded, clean]) == {
        "episodes": 1, "replies": 2,
        "supervised_tokens_removed": 2 * len(tokenizer.encode(REJECTED + TURN_TOKEN))}
