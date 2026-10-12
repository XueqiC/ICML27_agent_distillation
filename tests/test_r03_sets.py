import json
from pathlib import Path

import pytest

from data.manifest.build_r03_sets import (build_sets, dataset_entry, guarded_task, load_r03_dataset,
                                        publish_dataset, load_surrogate_examples)
from sed.dev_subset import build_dev140
from sed.manifest import TASK_TYPES, freeze_splits
from sed.storage import digest, file_hash, write_json


@pytest.fixture
def bank():
    tasks = []
    kinds = sorted(TASK_TYPES)
    for source, count in (("train", 110), ("valid_seen", 24), ("valid_unseen", 1)):
        for k, kind in enumerate(kinds):
            n = count - (source == "valid_seen" and k >= 2)
            for i in range(n):
                tasks.append({"task_id": f"{source}/{kind}-{i}/trial", "group_id": f"{source}/{kind}-{i}",
                    "source_split": source, "eligible": True, "task_type": kind, "files": {},
                    "game_sha256": digest([source, kind, i])})
    manifest = {"tasks": tasks}
    freeze = freeze_splits(manifest)
    large, small = [], []
    for k, kind in enumerate(kinds):
        for i in range(17 if k < 4 else 16):
            r = {"task_id": f"train/{kind}-{i}/trial", "episode_hashes": {"episode.json": digest([kind, i])}}
            large.append(r)
            if i < (4 if k < 2 else 3):
                small.append(r)
    return manifest, freeze, {"verified": large}, {"verified": small}, build_dev140(manifest, freeze)


class TinyTokenizer:
    """Deterministic byte-like fake of the already tested Gemma OFF boundaries."""
    all_special_ids = [106]
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        from train.dataset import EMPTY_THOUGHT
        assert not tokenize and not enable_thinking
        text = ""
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            text += f"<|turn>{role}\n" + m["content"].strip() + "<turn|>\n"
        if add_generation_prompt:
            text += "<|turn>model\n" + EMPTY_THOUGHT
        return text
    def encode(self, text, *, add_special_tokens):
        parts = text.split("<turn|>")
        result = []
        for i, part in enumerate(parts):
            if i:
                result.append(106)
            result.extend(ord(c) + 1000 for c in part)
        return result
    def decode(self, ids, *, skip_special_tokens):
        return "".join("<turn|>" if n == 106 else chr(n - 1000) for n in ids)


def test_counts_disjoint_stratification_determinism_and_eval_format(bank, tmp_path):
    m, f, large, small, dev = bank
    a = build_sets(*bank)
    assert a == build_sets(*bank)
    assert [a[n]["unique_instances"] for n in ("surrogate", "pool200", "dev300")] == [80, 200, 300]
    assert a["dev300"]["task_ids"][:140] == dev["task_ids"]
    assert set(a["surrogate"]["counts_by_task_type"].values()) == {13, 14}
    assert set(a["pool200"]["counts_by_task_type"].values()) == {33, 34}
    groups = [{f["assignments"][r["task_id"]]["group_id"] for r in large["verified"]},
              set(a["pool200"]["group_ids"]), set(a["dev300"]["group_ids"])]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    from eval.run_suite import frozen_tasks
    path = tmp_path / "dev300.json"
    write_json(path, a["dev300"])
    listing, tasks = frozen_tasks(path, m, f)
    assert len(tasks) == 300 and listing == a["dev300"]


def test_sealed_guard_and_forged_dev(bank):
    m, f, *rest = bank
    task = next(t for t in m["tasks"] if t["source_split"] == "valid_unseen")
    for generation in (True, False):
        with pytest.raises(ValueError, match="Sealed"):
            guarded_task(m, f, task["task_id"], generation=generation)
    m, f, large, small, dev = bank
    dev = dict(dev, task_ids=[task["task_id"], *dev["task_ids"][1:]])
    with pytest.raises(ValueError, match="dev140"):
        build_sets(m, f, large, small, dev)


def test_surrogate_loader_masks_audit_and_refuses_training(bank, tmp_path):
    m, f, large, *_ = bank
    tid = large["verified"][0]["task_id"]
    visible = {"observation": "VISIBLE", "admissible_commands": ["look"], "facts": "SECRET"}
    ep = {"schema_version": 2, "episode_id": "e", "task_id": tid, "split": "train",
          "sealed": False, "initial": {"visible": visible}, "audit": "SECRET",
          "steps": [{"event_id": "e0", "action_raw": "Thought: inspect\nAction: look", "visible": visible}]}
    entry = dataset_entry(tmp_path, ep, source="surrogate")
    publish_dataset(tmp_path, [entry], m, f, source="surrogate", purpose="surrogate_only")
    tokenizer = TinyTokenizer()
    with pytest.raises(ValueError, match="purpose"):
        load_r03_dataset(tmp_path, tokenizer, m, f)
    rows = load_r03_dataset(tmp_path, tokenizer, m, f, purpose="surrogate_only")
    assert len(rows) == 1 and rows[0]["labels"][-1] == 106
    assert "SECRET" not in tokenizer.decode(rows[0]["input_ids"], skip_special_tokens=False)
    assert rows[0]["label_provenance"]["training_eligible"] is False
    listing = tmp_path / "surrogate.json"
    write_json(listing, {"dataset_root": ".", "dataset_manifest_sha256": file_hash(tmp_path / "manifest.json")})
    assert load_surrogate_examples(listing, tokenizer, m, f) == rows
    (tmp_path / entry["episode_file"]).write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        load_r03_dataset(tmp_path, tokenizer, m, f, purpose="surrogate_only")
