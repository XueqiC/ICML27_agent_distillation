"""K10/K30 use real readers and row construction with fake executed evidence."""
from collections import Counter
import json
import shutil

import pytest

from repair import v2_train as v2
from repair.artifacts import read_json
from sed.manifest import load_frozen
from sed.seed_evidence import materialize_full_train_views
from sed.storage import digest, file_hash, write_json
from tools import train_v0_arm as trainer, v0_grad_features as features, v0_groups as groups, v0_replay
from tools.v0_data import round_seed_count
from test_repair_e2_reader import build_corpus
from test_v0_training import make_corpus, read, seal
from test_v2_train import seal_bundle


@pytest.fixture(params=[10, 30])
def grid_corpus(request, tmp_path):
    k = request.param
    seed = build_corpus(tmp_path, task_count=36)
    frozen, freeze = load_frozen(seed.config["paths"]["manifest"], seed.config["paths"]["split_freeze"])
    bank = seed.view.parent.parent
    materialize_full_train_views(bank, frozen, freeze, [k])
    seed.view = bank / "subsets" / f"k{k}"
    provenance = read_json(seed.theta.parent / "provenance.json")
    epochs = 64 if k == 30 else 32
    provenance["config"]["baseline_training"].update(K=k, epochs=epochs, verified=str(seed.view))
    provenance["inputs"] = {str(seed.view / "manifest.json"): file_hash(seed.view / "manifest.json")}
    write_json(seed.theta.parent / "provenance.json", provenance)
    state = read_json(seed.theta / "training_state.json")
    state.update(identity_sha256=digest(provenance), passes=epochs)
    write_json(seed.theta / "training_state.json", state)
    return make_corpus(seed, tmp_path, seed_k=k)


def arguments(c):
    return ["--dataset", str(c["root"]), "--theta0", str(c["theta"]), "--snapshot", str(c["snapshot"]),
            "--config", str(c["config_path"]), "--synthetic-manifest", str(c["synthetic"])]


@pytest.mark.parametrize("split_membership", [True, False])
def test_replay_inputs_features_groups_and_schedule(grid_corpus, tmp_path, monkeypatch, capsys, split_membership):
    import torch
    c, k = grid_corpus, grid_corpus["K"]
    if not split_membership:
        splits = read_json(c["splits"])
        del splits["sets"][f"K{k}"]
        write_json(c["splits"], splits)
    output = tmp_path / "built-replay"
    def forbidden(*args, **kwargs):
        raise AssertionError("No real model loading")
    monkeypatch.setattr(v2, "load_existing_adapter", forbidden)
    assert v0_replay.main(["--round-manifest", str(c["synthetic"]), "--theta0", str(c["theta"]),
        "--snapshot", str(c["snapshot"]), "--config", str(c["config_path"]), "--splits", str(c["splits"]),
        "--output", str(output)]) == 0
    manifest = read_json(output / "manifest.json")
    assert manifest["K"] == manifest["counts"]["examples"] == k
    assert manifest["counts"]["by_source"] == {f"k{k}_replay": k}
    assert manifest["new_teacher_calls"] == manifest["new_environment_steps"] == manifest["training_exposure"] == 0
    shutil.rmtree(c["root"] / "replay")  # Replace only this test's placeholder rows.
    output.rename(c["root"] / "replay")
    seal(c)
    capsys.readouterr()
    assert trainer.main(["inputs", "--check", *arguments(c)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["K_trusted_seed_trajectories"] == report["seed_unique_instances"] == k
    data = read(c)
    assert Counter(r["v0"]["bucket"] for r in data["rows"]) == {"R": 1, "D": k}
    inherited = v2.continuation_recipe(c["theta"], c["config"], c["snapshot"], 2, seed_k=k)
    assert inherited["recipe"]["epochs"] == (64 if k == 30 else 32)
    assert inherited["recipe"]["seed"] == 0
    # CPU fake tensors deliberately use the declared adapter names so the
    # feature -> group -> training choice binding is exercised without a GPU.
    model = v2.fake_model()
    names = sorted(inherited["parameters"])
    parameters = {name: (model.lora_A if i == 0 else model.lora_B if i == 1
                         else torch.nn.Parameter(torch.zeros(1, 1))) for i, name in enumerate(names)}
    monkeypatch.setattr(features, "lora_parameters", lambda _: parameters)
    encoded = [v2.encode_example(r, v2.FakeTokenizer()) for r in data["rows"]]
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    index = features.collect_features(model, encoded, data["rows"], data["episodes"], feature_dir,
        metadata={"dataset_sha256": data["dataset_sha256"], "adapter_sha256": data["adapter_sha256"]})
    assert index["episode_ids"] == ["repair-a"] and index["optimizer_steps"] == 0
    group_dir = tmp_path / "groups"
    groups.run(feature_dir, group_dir)
    for option, filename in (("--weights", "weights.json"), ("--trainable-modules", "trainable_modules.json")):
        assert trainer.main(["inputs", "--check", *arguments(c), option, str(group_dir / filename)]) == 0
    weights = read_json(group_dir / "weights.json")["weights"]
    batches, mixture = trainer.batch_schedule(data["rows"], updates=8, seed=0, weights=weights)
    assert batches == trainer.batch_schedule(data["rows"], updates=8, seed=0, weights=weights)[0]
    assert mixture == pytest.approx({"R": 5 / 6, "D": 1 / 6})
    assert sum(trainer.exposure_counts(batches)[8]["bucket_exposures"].values()) == 128
    assert all(weights[eid] == 1 for eid in data["episodes"] if eid.startswith("seed/"))
    assert trainer.main(["run", *arguments(c), "--output", str(tmp_path / "trained"), "--updates", "2",
                         "--checkpoints", "1", "2", "--dry-run"]) == 0


@pytest.mark.parametrize("mutation", ["K", "support_path", "support_seal", "recipe_epochs", "source", "split"])
def test_grid_rejects_wrong_adapter_or_replay(grid_corpus, mutation):
    c = grid_corpus
    if mutation in ("K", "support_path", "support_seal", "recipe_epochs"):
        path = c["theta"].parent / "provenance.json"
        prov = read_json(path)
        recipe = prov["config"]["baseline_training"]
        if mutation == "K":
            recipe["K"] = 20
        elif mutation == "support_path":
            recipe["verified"] = recipe["verified"].rsplit("/", 1)[0] + "/k20"
        elif mutation == "support_seal":
            prov["inputs"][recipe["verified"] + "/manifest.json"] = "0" * 64
        else:
            recipe["epochs"] = 64 if c["K"] == 10 else 32
        write_json(path, prov)
        state = read_json(c["theta"] / "training_state.json")
        state["identity_sha256"] = digest(prov)
        write_json(c["theta"] / "training_state.json", state)
    elif mutation == "source":
        root = c["root"] / "replay"
        rows = [json.loads(s) for s in (root / "examples.jsonl").read_text().splitlines()]
        for row in rows:
            row["source"] = "k20_replay"
            row["example_id"] = digest([row["arm"], row["source"], row["task_id"], row["evidence"]["episode_ref"], None])
        seal_bundle(root, rows, c["theta"], c["splits"], "CONTINUE")
        seal(c)
    else:
        splits = read_json(c["splits"])
        splits["sets"][f'K{c["K"]}']["task_ids"].append("train/non-seed/0")
        write_json(c["splits"], splits)
        root = c["root"] / "replay"
        rows = [json.loads(s) for s in (root / "examples.jsonl").read_text().splitlines()]
        seal_bundle(root, rows, c["theta"], c["splits"], "CONTINUE")
        seal(c)
    assert trainer.main(["inputs", "--check", *arguments(c)]) == 1


@pytest.mark.parametrize("value", [None, 0, 5, 40, 20.0, True, "20"])
def test_unknown_or_missing_round_k_fails(value):
    with pytest.raises(ValueError, match="explicit K"):
        round_seed_count({} if value is None else {"K": value})


def test_manifest_k_sources_must_agree():
    for k in (10, 20, 30):
        assert round_seed_count({"K": k}) == round_seed_count({}, {"K": k}) == k
    with pytest.raises(ValueError, match="K mismatch"):
        round_seed_count({"K": 10}, {"K": 30})
