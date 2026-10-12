from copy import deepcopy
import json

import pytest

from tools.collect_rewrites import apply_rewrite, collect, RewriteTeacher, rewrite_history
from test_capped_teacher import RawClient, capped
from test_rollout_pool import rollout
from test_r03_sets import TinyTokenizer
from data.manifest.build_r03_sets import load_r03_dataset


def rewrite_reply(episode):
    return json.dumps({"turns": [{"thought": "Inspect visibly.", "action": s["action_parsed"]}
                                  for s in episode["steps"]]})


def test_one_call_per_trajectory_byte_identical_actions(rollout, tmp_path):
    m, f, _, _, _, records = rollout
    ep = json.loads((records[0][1] / "episode.json").read_text())
    ep["steps"][0]["action_raw"] = "Thought: original\n  Action :  look  "
    client = RawClient(texts=[rewrite_reply(ep)])
    teacher = capped(tmp_path / "teacher", client, teacher_class=RewriteTeacher, cap=40)
    result = collect([ep], teacher, tmp_path / "p4", m, f)
    assert len(client.requests) == 1 and result["episode_count"] == 1
    assert "SECRET" not in json.dumps(client.requests)
    copied = json.loads((tmp_path / "p4" / result["episodes"][0]["episode_file"]).read_text())
    assert copied["steps"][0]["action_raw"].endswith("  Action :  look  ")
    assert [s["action_parsed"] for s in copied["steps"]] == [s["action_parsed"] for s in ep["steps"]]
    assert len(load_r03_dataset(tmp_path / "p4", TinyTokenizer(), m, f)) == 1


@pytest.mark.parametrize("mutation", ["action", "count", "injection", "channel", "format"])
def test_changed_actions_or_malformed_rewrite_drop_and_log(rollout, tmp_path, mutation):
    m, f, _, _, _, records = rollout
    ep = json.loads((records[0][1] / "episode.json").read_text())
    value = json.loads(rewrite_reply(ep))
    if mutation == "action":
        value["turns"][0]["action"] += " "
    elif mutation == "count":
        value["turns"].pop()
    elif mutation == "injection":
        value["turns"][0]["thought"] = "x\nAction: go north"
    elif mutation == "channel":
        value["turns"][0]["thought"] = "<|channel>thought\nPRIVATE<channel|>"
    raw = "bad-json" if mutation == "format" else json.dumps(value)
    client = RawClient(texts=[raw])
    result = collect([ep], capped(tmp_path / "teacher", client, teacher_class=RewriteTeacher),
                     tmp_path / "p4", m, f)
    assert result["rejected"] == 1 and result["episode_count"] == 0
    audit = json.loads((tmp_path / "p4/collection.jsonl").read_text())
    assert audit["accepted"] is False and audit["raw_output"] == raw
