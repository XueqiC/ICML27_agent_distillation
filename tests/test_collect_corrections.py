import json

import pytest

from data.manifest.build_r03_sets import load_r03_dataset
from sed.react import build_messages
from tools.collect_corrections import collect, resolve_selection, select_r1_states, state_ref
from test_capped_teacher import RawClient, capped
from test_rollout_pool import rollout
from test_r03_sets import TinyTokenizer


def test_only_teacher_turn_supervised_and_shared_r2_cache(rollout, tmp_path):
    m, f, task, output, pool, records = rollout
    state = records[0][0]["states"][1]
    client = RawClient(texts=["Thought: teacher thought\nAction: go north"])
    teacher = capped(tmp_path / "teacher", client, run="r1")
    result = collect(records, [state_ref(state)], teacher, tmp_path / "p2", m, f)
    assert result["episode_count"] == 1 and result["verification"] == "unknown"
    assert result["trusted_executed_trajectories"] == 0
    assert client.requests[0]["input"] == build_messages(state["history"], task, state["visible_context"])
    tokenizer = TinyTokenizer()
    rows = load_r03_dataset(tmp_path / "p2", tokenizer, m, f)
    assert len(rows) == 1 and rows[0]["packing"] == "turn"
    decoded = tokenizer.decode([n for n in rows[0]["labels"] if n != -100], skip_special_tokens=False)
    assert decoded == "Action: go north<turn|>"
    text = tokenizer.decode(rows[0]["input_ids"], skip_special_tokens=False)
    assert "Action: look" in text and "SECRET" not in text and "teacher thought" not in text
    collect(records, [state_ref(state)], capped(tmp_path / "teacher", client, run="r2", cap=0),
            tmp_path / "r2", m, f, source="R2")
    assert len(client.requests) == 1


def test_rejected_label_logged_not_supervised(rollout, tmp_path):
    m, f, _, _, _, records = rollout
    state = records[0][0]["states"][0]
    result = collect(records, [state_ref(state)], capped(tmp_path / "teacher", RawClient(texts=["Action: fly"])),
                     tmp_path / "p2", m, f)
    assert result["episode_count"] == 0 and result["rejected_labels"] == 1
    assert "Action: fly" in (tmp_path / "p2/collection.jsonl").read_text()


def test_selection_dedupe_bad_ref_and_dry_run(rollout, tmp_path):
    m, f, task, _, _, records = rollout
    state = records[0][0]["states"][0]
    selection = select_r1_states([records[0][0]], [task["task_id"]], cap=1)
    assert len(selection["states"]) == 1 and selection["omitted_by_cap"] == 1
    assert len(resolve_selection(records, [state_ref(state)] * 2)) == 1
    with pytest.raises(ValueError, match="hash"):
        resolve_selection(records, [{**state_ref(state), "state_hash": "bad"}])
    client = RawClient()
    result = collect(records, [state_ref(state)] * 2, capped(tmp_path / "teacher", client, dry_run=True),
                     tmp_path / "p2", m, f)
    assert result["would_call"] == 1 and not client.requests and not (tmp_path / "p2").exists()
