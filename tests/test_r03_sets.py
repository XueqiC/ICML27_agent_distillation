import json
from pathlib import Path

import pytest

from data.manifest.build_r03_sets import (build_sets, dataset_entry, guarded_task, load_r03_dataset,
                                        publish_dataset, load_surrogate_examples, surrogate_subset)
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
    assert [a[n]["unique_instances"] for n in ("surrogate", "surrogate40", "pool200", "dev300")] == [80, 40, 200, 300]
    subset = a["surrogate40"]
    assert len(set(subset["task_ids"])) == subset["trusted_trajectories"] == subset["task_groups"] == 40
    assert set(subset["task_ids"]) < set(a["surrogate"]["task_ids"])
    assert subset["parent_list_sha256"] == a["surrogate"]["list_sha256"]
    assert subset["purpose"] == "surrogate_only" and subset["training_eligible"] is False
    assert subset["seed"] == 0
    for kind, n in a["surrogate"]["counts_by_task_type"].items():
        assert n // 2 <= subset["quotas"][kind] <= (n + 1) // 2
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


def test_surrogate_subset_uses_proportional_largest_remainder():
    from collections import Counter
    counts = dict(zip(sorted(TASK_TYPES), [5, 11, 12, 14, 17, 21]))
    tasks = [{"task_id": f"{kind}/{i}", "task_type": kind}
             for kind, n in counts.items() for i in range(n)]
    selected = surrogate_subset(tasks)
    assert selected == surrogate_subset(list(reversed(tasks)))
    assert len(selected) == len({t["task_id"] for t in selected}) == 40
    quotas = Counter(t["task_type"] for t in selected)
    # Both even strata get their exact quota; exactly two odd strata round up.
    assert all(n // 2 <= quotas[kind] <= (n + 1) // 2 for kind, n in counts.items())
    assert sum(quotas[kind] > n // 2 for kind, n in counts.items()) == 2
    with pytest.raises(ValueError, match="size"):
        surrogate_subset(tasks, 81)


def test_builder_emits_loadable_subset_without_changing_frozen_artifacts(bank, tmp_path, monkeypatch):
    import data.manifest.build_r03_sets as builder
    m, f, large, small, dev = bank
    output, root = tmp_path / "lists", tmp_path / "episodes"
    dev_path = tmp_path / "dev140.json"
    write_json(dev_path, dev)
    monkeypatch.setattr(builder, "load_frozen", lambda *_: (m, f))
    monkeypatch.setattr(builder, "load_verified_collection", lambda path, *_:
                        (large if path == "large" else small, {"seed": 0}, "unused"))
    def episode(_, receipt):
        tid = receipt["task_id"]
        visible = {"observation": "VISIBLE", "admissible_commands": ["look"], "facts": "SECRET"}
        return {"schema_version": 2, "episode_id": tid, "task_id": tid, "split": "train",
                "sealed": False, "initial": {"visible": visible}, "audit": "SECRET",
                "steps": [{"event_id": f"e{i}", "action_raw": "Action: look", "visible": visible}
                          for i in range(2)]}
    monkeypatch.setattr(builder, "checked_episode", episode)
    monkeypatch.setattr("sed.provenance.write_provenance", lambda *_, **__: None)
    args = ["--k100", "large", "--k20", "small", "--dev140", str(dev_path),
            "--output-dir", str(output), "--surrogate-root", str(root)]
    builder.main(args)
    # Simulate artifacts frozen by the previous builder, before surrogate40 existed.
    (output / "r03_surrogate40.json").unlink()
    collection = json.loads((root / "manifest.json").read_text())
    collection["builder_sha256"] = "original-builder"
    write_json(root / "manifest.json", collection)
    for path in output.glob("*.json"):
        value = json.loads(path.read_text())
        value["builder_sha256"] = "original-builder"
        if "dataset_manifest_sha256" in value:
            value["dataset_manifest_sha256"] = file_hash(root / "manifest.json")
        write_json(path, value)
    frozen_paths = [*output.glob("*.json"), *root.rglob("*.json")]
    before = {path: path.read_bytes() for path in frozen_paths}
    builder.main(args)
    assert all(path.read_bytes() == value for path, value in before.items())
    subset = output / "r03_surrogate40.json"
    value = json.loads(subset.read_text())
    assert value["builder_sha256"] == file_hash(builder.__file__)
    assert value["dataset_manifest_sha256"] == file_hash(root / "manifest.json")
    before[subset] = subset.read_bytes()
    builder.main(args)
    assert all(path.read_bytes() == value for path, value in before.items())
    tokenizer = TinyTokenizer()
    from tools.crn_tail import Experiment
    for packing, multiplier in (("episode", 1), ("turn", 2)):
        context = Experiment.__new__(Experiment)
        context.datasets, context.dataset_purposes, context.inputs = {}, {}, []
        context.manifest, context.freeze, context.tokenizer = m, f, tokenizer
        context.recipe = {"max_length": 32768, "packing": packing}
        # All four diagnostics load --surrogate through this shared entrypoint.
        rows = context.load_dataset(str(subset), purpose="surrogate_only")
        full = load_surrogate_examples(output / "r03_surrogate.json", tokenizer, m, f, packing=packing)
        ids = set(value["task_ids"])
        assert len(rows) == 40 * multiplier and len(full) == 80 * multiplier
        assert rows == [row for row in full if row["label_provenance"]["task_id"] in ids]
        assert all(row["label_provenance"]["training_eligible"] is False for row in rows)
        inputs = {path.resolve() for path in context.inputs}
        assert subset in inputs and root / "manifest.json" in inputs
        with pytest.raises(ValueError, match="both"):
            context.load_dataset(str(subset))
    for ids, match in (([value["task_ids"][0]] * 2, "unique"), (["missing"], "missing")):
        write_json(subset, {**value, "task_ids": ids, "list_sha256": digest(ids)})
        with pytest.raises(ValueError, match=match):
            load_surrogate_examples(subset, tokenizer, m, f)
    write_json(subset, {**value, "list_sha256": "changed"})
    with pytest.raises(ValueError, match="frozen inputs"):
        load_surrogate_examples(subset, tokenizer, m, f)
    # Content drift still fails closed despite retaining historical builder hashes.
    changed = json.loads((output / "r03_surrogate.json").read_text())
    write_json(output / "r03_surrogate.json", {**changed, "trusted_trajectories": 79})
    with pytest.raises(ValueError, match="frozen artifact"):
        builder.main(args)


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
