"""V0 contract/numerics and inherited-loop tests, all offline CPU-only."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from repair import v2_train as v2
from repair.artifacts import read_json
from sed.storage import digest, file_hash, write_json
from tools import v0_data, v0_grad_features as features, v0_groups as groups, train_v0_arm as trainer
from test_repair_e2_reader import corpus as seed_corpus
from test_v2_train import example, seal_bundle


@pytest.fixture
def corpus(seed_corpus, tmp_path):
    from sed.seed_evidence import load_verified_collection
    from sed.manifest import load_frozen
    c = seed_corpus
    frozen, freeze = load_frozen(c.config["paths"]["manifest"], c.config["paths"]["split_freeze"])
    collection, _, _ = load_verified_collection(c.view, frozen, freeze)
    sets = {name: {"task_ids": [], "group_ids": []} for name in v2.FORBIDDEN_SETS}
    sets["V2_pool"] = {"task_ids": [], "group_ids": []}
    sets["K20"] = {"task_ids": sorted({r["task_id"] for r in collection["verified"]}),
                   "group_ids": sorted({r["group_id"] for r in collection["verified"]})}
    splits = tmp_path / "method_v1_splits.json"
    write_json(splits, {"sets": sets})
    root = tmp_path / "v0"
    replay = [example(tid, "k20_replay", "trajectory", "CONTINUE") for tid in sets["K20"]["task_ids"]]
    seal_bundle(root / "replay", replay, c.theta, splits, "CONTINUE")
    synthetic = tmp_path / "synthetic.json"
    write_json(synthetic, {"schema": "v0-synthetic-tasks/1", "tasks": [{"task_id": "train/synth/a",
        "parent_seed_task_id": sets["K20"]["task_ids"][0], "group_id": "synthetic/a",
        "task_type": "pick_and_place_simple", "generator": "G1", "split": "train", "sealed": False}]})
    row = example("train/synth/a", "joint_local", "decision", "V0")
    row.update(group_id="synthetic/a", v0={"bucket": "R", "episode_id": "repair-a", "generator": "G1",
        "edit_decision": True, "synthetic_manifest_sha256": file_hash(synthetic)})
    config_path = tmp_path / "v0-config.json"
    write_json(config_path, c.config)
    result = {"root": root, "rows": [row], "theta": c.theta, "snapshot": c.snapshot,
              "config": c.config, "config_path": config_path, "splits": splits, "synthetic": synthetic}
    seal(result)
    return result


def seal(c):
    path = c["root"] / "examples.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in c["rows"]))
    inputs = [c["splits"], c["synthetic"], *[Path(c["config"]["paths"][k]) for k in ("manifest", "split_freeze")]]
    write_json(c["root"] / "manifest.json", {"schema": "v0-train-interchange/1", "arm": "V0",
        "created": "test-only", "code_revision": "test-only", "examples_sha256": file_hash(path),
        "inputs": {str(p): file_hash(p) for p in inputs}, "splits_sha256": file_hash(c["splits"]),
        "replay_manifest_sha256": file_hash(c["root"] / "replay/manifest.json"),
        "theta0": {"adapter_weights_sha256": file_hash(c["theta"] / "adapter_model.safetensors")}})


def read(c, **kwargs):
    return v0_data.read_dataset(c["root"], theta0=c["theta"], snapshot=c["snapshot"], config=c["config"],
                               synthetic_manifest=kwargs.get("synthetic_manifest", c["synthetic"]))


def test_count_sketch_determinism_linearity_and_coordinate_offsets():
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((2, 8193)).astype(np.float32)
    first = features.count_sketch(a)
    np.testing.assert_array_equal(first, features.count_sketch(a))
    np.testing.assert_allclose(features.count_sketch(2 * a.astype(np.float64) - b),
                               2 * first - features.count_sketch(b), atol=1e-12)
    np.testing.assert_allclose(first, features.count_sketch(a[:4000]) + features.count_sketch(a[4000:], offset=4000))
    import torch
    np.testing.assert_array_equal(first, features.count_sketch(torch.from_numpy(a)))
    assert features.coordinate_hash(np.arange(4, dtype=np.uint64)).tolist() == [
        9260656408219841379, 1234184003990712370, 13564971763896621636, 3900778703475868044]


def test_kmedoids_permutation_ties_merging_and_six_group_cap():
    x = np.array([[1., 0], [1., 0], [1., 0], [0., 1]])
    ids = ["a", "b", "c", "d"]
    actual = groups.k_medoids(x, ids, [1, 1, 1, 1])
    assert actual["initial_medoids"] == ["a", "d"]
    assert len(actual["groups"]) == 1
    assert actual["groups"][0]["medoid"] == "a"
    assert actual["groups"][0]["farthest_member"] == "d"
    assert actual["groups"][0]["consistency"] == pytest.approx(2)
    order = [3, 1, 0, 2]
    assert actual == groups.k_medoids(x[order], [ids[i] for i in order], [1] * 4)
    assert len(groups.k_medoids(np.eye(3), ["c", "a", "b"], [1, 3, 2])["groups"]) == 1
    equal = groups.k_medoids(np.ones((8, 2)), list("abcdefgh"), [1] * 8)
    assert len(equal["groups"]) == 1  # Equal distances assign by ID, never invent balanced ties.
    six = groups.k_medoids(np.repeat(np.eye(6), 2, axis=0), [f"e{i:02d}" for i in range(12)], [1] * 12)
    assert len(six["groups"]) == 6 and six["iterations"] <= 20


def test_snr_exact_device_accumulation_on_toy_model():
    import torch
    model = v2.fake_model(2)
    params = features.lora_parameters(model)
    moments = features.ModuleMoments(params)
    samples = []
    for i in (1., 3., 5.):
        gradients = {"lora_A": torch.tensor([[i]]), "lora_B": torch.tensor([[i], [-2 * i]])}
        samples.append(gradients)
        moments.add(gradients)
    result = moments.result()
    for name in params:
        values = torch.stack([s[name] for s in samples])
        mean = values.mean(0)
        variance = (values - mean).square().flatten(1).sum(1).mean()
        assert result[name]["snr"] == pytest.approx(float(mean.norm() / variance.sqrt()), rel=1e-6)
        assert result[name]["mean_squared_norm"] == pytest.approx(float(values.square().flatten(1).sum(1).mean()))
        assert moments.sums[name].dtype == torch.float32 and result[name]["episodes"] == 3
    zero = features.ModuleMoments(params)
    zero.add({n: None for n in params})
    assert all(v["snr"] == 0 for v in zero.result().values())


def sampler_rows():
    rows = []
    for bucket in v0_data.BUCKETS:
        for episode in ("a", "b"):
            for decision in range(1 if episode == "a" else 3):
                rows.append({"example_id": f"{bucket}{episode}{decision}",
                             "v0": {"bucket": bucket, "episode_id": bucket + episode}})
    return rows


def test_bucket_sampler_distribution_determinism_and_episode_normalization():
    rows = sampler_rows()
    weights = {r["v0"]["episode_id"]: 3 if r["v0"]["episode_id"].endswith("b") else 1 for r in rows}
    batches, probabilities = trainer.batch_schedule(rows, updates=2000, seed=17, weights=weights)
    assert batches == trainer.batch_schedule(rows, updates=2000, seed=17, weights=weights)[0]
    counts = trainer.exposure_counts(batches)[2000]
    for bucket, probability in probabilities.items():
        assert counts["bucket_exposures"][bucket] / 32000 == pytest.approx(probability, abs=.01)
        assert counts["episode_exposures"][bucket + "b"] / counts["bucket_exposures"][bucket] == pytest.approx(.75, abs=.025)
    assert trainer.batch_schedule(rows[4:8], updates=1, seed=1)[1] == {"G": 1.0}
    assert len(batches[0]["indices"]) == 16
    with pytest.raises(ValueError, match="zero episode weight"):
        trainer.batch_schedule(rows, updates=1, seed=1, weights={"Ra": 0, "Rb": 0})


def test_module_mask_freezes_tensors_and_subsets_optimizer_moments():
    import torch
    model = v2.fake_model(4)
    with torch.no_grad():
        model.lora_B.fill_(1)
    original = list(dict(model.named_parameters()))
    all_optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    (model.lora_A.sum() + model.lora_B.sum()).backward()
    all_optimizer.step()
    saved = all_optimizer.state_dict()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert trainer.mask_modules(model, ["lora_B"]) == {"lora_B": 4}
    subset = trainer.subset_optimizer_state(saved, original, ["lora_B"])
    assert len(subset["state"]) == 1 and len(subset["param_groups"][0]["params"]) == 1
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.1)
    optimizer.load_state_dict(subset)
    (model.lora_B @ model.lora_A).sum().backward()
    optimizer.step()
    assert torch.equal(model.lora_A, before["lora_A"])
    assert not torch.equal(model.lora_B, before["lora_B"])
    assert model.lora_A.grad is None
    with pytest.raises(ValueError, match="does not exist"):
        trainer.mask_modules(model, ["missing"])
    with pytest.raises(ValueError, match="nonempty"):
        trainer.mask_modules(model, [])


def test_episode_gradient_is_v2_row_mean_with_edit_selection_and_no_update(tmp_path):
    import torch
    from baselines.common.fsdp_train import sft_objective
    torch.set_num_threads(1)
    rows = [example("train/a/0", "joint_local", "decision", "V0") for _ in range(3)]
    for i, row in enumerate(rows):
        row["example_id"] = str(i)
        row["messages"][-1]["content"] = "Thought: " + "look " * (10 * i + 1) + ".\nAction: look"
        row["v0"] = {"episode_id": "a" if i < 2 else "b", "bucket": "R", "edit_decision": i == 0}
    examples = [v2.encode_example(r, v2.FakeTokenizer()) for r in rows]
    model = v2.fake_model()
    with torch.no_grad():
        model.lora_B.copy_(torch.linspace(-1, 1, 512).reshape(512, 1))
    reference = deepcopy(model)
    features.episode_backward(model, examples[:2])
    (sum(sft_objective(reference, e, "cpu", scoring="shared_history")[0] for e in examples[:2]) / 2).backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    from types import SimpleNamespace
    model.dropout = torch.nn.Dropout(.05)
    model.peft_config = {"default": SimpleNamespace(lora_dropout=.05)}
    before = features.parameter_hash(features.lora_parameters(model))
    result = features.collect_features(model, examples, rows, v0_data.episodes(rows), tmp_path,
        metadata={"dataset_sha256": "d", "adapter_sha256": "a"})
    assert result["episodes"][0]["gradient_row_ids"] == ["0"]
    assert result["episodes"][1]["gradient_row_ids"] == ["2"]  # No edit -> all rows.
    assert before == features.parameter_hash(features.lora_parameters(model))
    assert all(p.grad is None for p in model.parameters())
    assert model.dropout.p == .05 and model.peft_config["default"].lora_dropout == .05
    x = np.load(tmp_path / "sketch.npy")
    assert x.shape == (2, 2048) and x.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
    out = tmp_path.parent / (tmp_path.name + "-groups")
    groups.run(tmp_path, out)
    assert len(read_json(out / "trainable_modules.json")["selected"]) == 1
    weights = read_json(out / "weights.json")["weights"]
    assert sum(weights.values()) / len(weights) == pytest.approx(1)


def test_zero_gradient_is_retained_but_not_directionally_clustered(tmp_path):
    import torch
    model = v2.fake_model()
    with torch.no_grad():
        model.lora_A.zero_()
    row = example("train/a/0", "joint_local", "decision", "V0")
    row["v0"] = {"episode_id": "a", "bucket": "R", "edit_decision": False}
    features.collect_features(model, [v2.encode_example(row, v2.FakeTokenizer())], [row], {"a": [0]}, tmp_path,
        metadata={"dataset_sha256": "d", "adapter_sha256": "a"})
    assert read_json(tmp_path / "norms.json")["a"]["ZERO_GRADIENT"] is True
    out = tmp_path.parent / (tmp_path.name + "-groups")
    result = groups.run(tmp_path, out)
    assert not result["groups"]
    assert read_json(out / "weights.json")["weights"] == {"a": 1.0}


def test_peft_subset_checkpoint_preserves_frozen_adapter_tensors(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 2, bias=False)

        def forward(self, x):
            return self.projection(x)
    model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=2, target_modules=["projection"], lora_dropout=.05))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    selected = [n for n in names if ".lora_B." in n]
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if n not in selected}
    trainer.mask_modules(model, selected)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.1)
    model(torch.ones(4, 3)).square().mean().backward()
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
    model.save_pretrained(tmp_path, safe_serialization=True)
    saved = load_file(tmp_path / "adapter_model.safetensors")
    assert len(saved) == 2
    for name in names:
        torch.testing.assert_close(saved[name.replace(".default", "")], dict(model.named_parameters())[name])


def test_contract_accepts_valid_synthetic_and_preserves_replay(corpus):
    data = read(corpus)
    assert data["K"] == 20 and len(data["rows"]) == 21 and len(data["episodes"]) == 21
    assert Counter(r["v0"]["bucket"] for r in data["rows"]) == {"R": 1, "D": 20}
    original = [json.loads(s) for s in (corpus["root"] / "replay/examples.jsonl").read_text().splitlines()]
    for a, b in zip(data["rows"][1:], original):
        assert {k: v for k, v in a.items() if k not in ("arm", "v0")} == {k: v for k, v in b.items() if k != "arm"}
    assert "evidence" not in v2.encode_example(data["rows"][0], v2.FakeTokenizer())


@pytest.mark.parametrize("mutation,match", [
    ("bad_id", "absent"), ("missing_manifest", "requires --synthetic-manifest"),
    ("hash", "Synthetic manifest hash mismatch"), ("parent", "not in theta0 K seeds"),
    ("holdout", "Held-out"), ("bucket", "Episode crosses"), ("hidden", "whitelist"),
    ("unknown_success", "executed successful"), ("sealed_bytes", "hash mismatch")])
def test_contract_rejections(corpus, mutation, match):
    row = corpus["rows"][0]
    if mutation == "missing_manifest":
        with pytest.raises(ValueError, match=match):
            read(corpus, synthetic_manifest=None)
        return
    if mutation == "bad_id":
        row["task_id"] = "train/synth/missing"
    elif mutation == "hash":
        row["v0"]["synthetic_manifest_sha256"] = "0" * 64
    elif mutation in ("parent", "holdout"):
        doc = read_json(corpus["synthetic"])
        doc["tasks"][0].update({"parent_seed_task_id": "train/not-a-seed/0"} if mutation == "parent" else {"split": "holdout"})
        write_json(corpus["synthetic"], doc)
        row["v0"]["synthetic_manifest_sha256"] = file_hash(corpus["synthetic"])
    elif mutation == "bucket":
        other = deepcopy(row)
        other["example_id"] = "other"
        other["v0"]["bucket"] = "G"
        corpus["rows"].append(other)
    elif mutation == "hidden":
        visible = json.loads(row["messages"][1]["content"])
        visible["extra.expert_plan"] = "private"
        row["messages"][1]["content"] = json.dumps(visible)
    elif mutation == "unknown_success":
        row["evidence"]["episode_success"] = None
    seal(corpus)
    if mutation == "sealed_bytes":
        with (corpus["root"] / "examples.jsonl").open("a") as handle:
            handle.write("\n")
    with pytest.raises(ValueError, match=match):
        read(corpus)


def test_inputs_check_never_loads_model_and_cpu_dry_run(corpus, tmp_path, monkeypatch):
    args = ["--dataset", str(corpus["root"]), "--theta0", str(corpus["theta"]),
            "--snapshot", str(corpus["snapshot"]), "--config", str(corpus["config_path"]),
            "--synthetic-manifest", str(corpus["synthetic"])]
    def forbidden(*_, **__):
        raise AssertionError("Model loading forbidden")
    monkeypatch.setattr(v2, "load_existing_adapter", forbidden)
    assert trainer.main(["inputs", "--check", *args]) == 0
    data = read(corpus)
    inherited = v2.continuation_recipe(corpus["theta"], corpus["config"], corpus["snapshot"], 2)
    selected = [n for n in inherited["parameters"] if ".lora_B." in n][:1]
    metadata = {"dataset_sha256": data["dataset_sha256"], "adapter_sha256": data["adapter_sha256"],
                "parameter_names_sha256": digest(sorted(inherited["parameters"]))}
    module_path = tmp_path / "modules.json"
    write_json(module_path, {"schema": "v0-trainable-modules/1", **metadata, "selected": selected})
    weight_path = tmp_path / "weights.json"
    weights = {eid: 1 for eid in data["episodes"]}
    write_json(weight_path, {"schema": "v0-episode-weights/1", **metadata, "weights": weights})
    args += ["--trainable-modules", str(module_path), "--weights", str(weight_path)]
    assert trainer.main(["inputs", "--check", *args]) == 0
    output = tmp_path / "trained"
    assert trainer.main(["run", *args, "--output", str(output), "--updates", "2",
                         "--checkpoints", "1", "2", "--dry-run"]) == 0
    manifest = read_json(output / "manifest.json")
    assert manifest["status"] == "synthetic_complete" and manifest["training_exposure"] == 0
    assert manifest["synthetic_training_exposure"] == 32
    assert manifest["actual_trainable_parameter_names"] == ["lora_B"]
    assert manifest["selected_real_tensors"] == selected
    assert sum(manifest["realised_exposure"]["bucket_exposures"].values()) == 32
    assert [c["update"] for c in manifest["checkpoints"]] == [0, 1, 2]
    write_json(module_path, {"schema": "v0-trainable-modules/1", **metadata, "selected": ["missing"]})
    assert trainer.main(["inputs", "--check", *args]) == 1


def test_weights_keep_signed_consistency_exact_and_module_ranking_ties():
    entries = [{"episode_id": eid, "bucket": b} for eid, b in (("a", "R"), ("b", "G"), ("c", "D"))]
    result = {"assignment": {"a": "x", "b": "y"}, "groups": [
        {"group_id": "x", "consistency": 2}, {"group_id": "y", "consistency": 4}]}
    assert groups.episode_weights(result, entries) == {"a": 2 / 3, "b": 4 / 3, "c": 1}
    result["groups"][0]["consistency"] = -1
    with pytest.raises(ValueError, match="Negative group consistency"):
        groups.episode_weights(result, entries)
    modules = {f"m{i}.lora_A.weight": {"snr": 2} for i in range(5)}
    assert groups.rank_modules(modules)["selected"] == sorted(modules)[:2]
