"""Partial v2 preparation with fake collections/tokenizer and real provenance hashes."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sed.storage import digest, file_hash, write_json
import train.dose_grid as dose


FULL_TEMPLATE = Path("configs/seedsft_dose_grid_v2.json")
PARTIAL_TEMPLATE = Path("configs/seedsft_dose_grid_v2_k12.json")


@pytest.fixture
def partial_template():
    return json.loads(PARTIAL_TEMPLATE.read_text())


@pytest.fixture
def fake_collections(tmp_path, monkeypatch):
    def install(template, data_root="."):
        runs = {Path(data_root) / r["verified"]: r for r in template["runs"]}
        measured = []
        evidence = tmp_path / "receipt.json"
        write_json(evidence, {"source": "fake executed collection"})

        def collection(path, *_):
            run = runs[path]
            selection = {"K": run["K"], "seed": run["support_seed"]}
            if "selection_rule" in run:
                selection["selection_rule"] = run["selection_rule"]
            return selection, selection, path

        def examples(path, tokenizer, *_args, **kwargs):
            assert kwargs == {"max_length": 32768, "packing": "episode"}
            measured.append(path)
            count = 17 * runs[path]["K"]
            return [{"labels": [-100] + [106] * count, "supervised_tokens": count}]

        monkeypatch.setattr(dose, "load_verified_collection", collection)
        monkeypatch.setattr(dose, "build_verified_dataset", examples)
        monkeypatch.setattr(dose, "verified_collection_inputs", lambda *_: [evidence])
        return measured, evidence

    return install


def test_partial_template_matches_registered_protocol_and_k20_views(partial_template):
    expected = json.loads(FULL_TEMPLATE.read_text())
    expected["partial"] = True
    expected["runs"] = [r for r in expected["runs"] if r["K"] in (1, 2)]
    for run in expected["runs"]:
        run["verified"] = f"data/verified/seeds_k20_s{run['support_seed']}/subsets/k{run['K']}"
    assert partial_template == expected
    assert [r["name"] for r in partial_template["runs"]] == [
        "k1s0", "k2s0", "k1s1", "k2s1", "k1s2", "k2s2"]


def test_partial_preparation_hashes_and_round_trip(partial_template, fake_collections, tmp_path):
    before = deepcopy(partial_template)
    measured, evidence = fake_collections(partial_template, tmp_path)
    manifest, freeze = {"manifest": "fake"}, {"freeze": "fake"}
    grid = dose.prepare_grid(partial_template, SimpleNamespace(chat_template="fake"),
                             manifest, freeze, data_root=tmp_path)
    assert partial_template == before
    assert grid["status"] == "ready" and grid["partial"] is True
    assert grid["partial_of"] == {"path": str(FULL_TEMPLATE),
                                  "sha256": digest(json.loads(FULL_TEMPLATE.read_text()))}
    assert grid["template_sha256"] == digest(before)
    assert grid["manifest_sha256"] == digest(manifest)
    assert grid["split_freeze_sha256"] == digest(freeze)
    assert grid["template_chat_sha256"] == digest("fake")
    assert measured == [tmp_path / r["verified"] for r in before["runs"]]
    output = tmp_path / "ready.json"
    write_json(output, grid)
    for index, run in enumerate(grid["runs"]):
        count = 17 * run["K"]
        assert run["supervised_tokens_per_pass"] == count
        assert run["total_tokens"] == 400_000
        assert run["collection_sha256"] == digest({"K": run["K"], "seed": run["support_seed"]})
        assert run["examples_sha256"] == digest([
            {"labels": [-100] + [106] * count, "supervised_tokens": count}])
        assert run["inputs"] == {str(evidence): file_hash(evidence)}
        assert dose.load_ready_run(output, index) == run


@pytest.mark.parametrize("names", [["k50s1"], ["k100s2", "k1s0"], None])
def test_partial_accepts_any_nonempty_registered_subset(names, fake_collections):
    template = json.loads(FULL_TEMPLATE.read_text())
    template["partial"] = True
    if names is not None:
        template["runs"] = [r for r in template["runs"] if r["name"] in names]
    fake_collections(template)
    grid = dose.prepare_grid(template, SimpleNamespace(chat_template="fake"), {}, {})
    assert [r["name"] for r in grid["runs"]] == [r["name"] for r in template["runs"]]


@pytest.mark.parametrize("field,value", [
    ("rule", "changed"), ("pass_checkpoints", [3, 5]),
    ("token_checkpoints", [100_000]), ("merge_relative_to", "pass_point"),
    ("merge_fraction", 0.1), ("schema_version", 1),
])
def test_partial_rejects_changed_protocol(partial_template, field, value):
    partial_template[field] = value
    with pytest.raises(ValueError, match="dose protocol"):
        dose.prepare_grid(partial_template, None, {}, {})


@pytest.mark.parametrize("invalid", ["duplicate", "unregistered_k", "unregistered_seed", "empty", "name"])
def test_partial_rejects_invalid_runs(partial_template, invalid):
    runs = partial_template["runs"]
    if invalid == "duplicate":
        runs.append(deepcopy(runs[0]))
    elif invalid == "unregistered_k":
        runs[0].update(K=20, name="k20s0")
    elif invalid == "unregistered_seed":
        runs[0].update(support_seed=3, name="k1s3")
    elif invalid == "empty":
        runs.clear()
    else:
        runs[0]["name"] = "wrong"
    with pytest.raises(ValueError, match="registered runs|run names"):
        dose.prepare_grid(partial_template, None, {}, {})


@pytest.mark.parametrize("partial", [None, False, "true", 1])
def test_non_partial_still_requires_all_twelve_runs(partial):
    template = json.loads(FULL_TEMPLATE.read_text())
    template["runs"].pop()
    if partial is not None:
        template["partial"] = partial
    with pytest.raises(ValueError, match="requires exactly"):
        dose.prepare_grid(template, None, {}, {})


@pytest.mark.parametrize("invalid", ["selection", "labels"])
def test_partial_preserves_measurement_checks(partial_template, fake_collections, monkeypatch, invalid):
    fake_collections(partial_template)
    if invalid == "selection":
        monkeypatch.setattr(dose, "load_verified_collection",
                            lambda *_: ({}, {"K": 2, "seed": 0}, None))
        message = "K/support seed"
    else:
        monkeypatch.setattr(dose, "build_verified_dataset",
                            lambda *_a, **_k: [{"labels": [-100, 106], "supervised_tokens": 2}])
        message = "accounting differs"
    with pytest.raises(ValueError, match=message):
        dose.prepare_grid(partial_template, SimpleNamespace(chat_template="fake"), {}, {})


@pytest.mark.parametrize("invalid", ["unprepared", "checkpoint", "merged"])
def test_partial_loader_preserves_schedule_checks(partial_template, fake_collections, tmp_path, invalid):
    fake_collections(partial_template)
    grid = dose.prepare_grid(partial_template, SimpleNamespace(chat_template="fake"), {}, {})
    if invalid == "unprepared":
        grid = partial_template
        message = "Prepare"
    elif invalid == "checkpoint":
        grid["runs"][0]["checkpoints"][0]["tokens"] += 1
        message = "measured pass/token schedule"
    else:
        grid["runs"][0]["merged_token_points"] = {"t100k": "p03"}
        message = "measured pass/token schedule"
    output = tmp_path / "invalid.json"
    write_json(output, grid)
    with pytest.raises(ValueError, match=message):
        dose.load_ready_run(output, 0)


def test_prepare_cli_accepts_partial_template(partial_template, fake_collections, tmp_path, monkeypatch, capsys):
    from scripts import prepare_seedsft_dose_grid as cli

    fake_collections(partial_template, tmp_path)
    manifest, freeze = tmp_path / "manifest.json", tmp_path / "freeze.json"
    write_json(manifest, {})
    write_json(freeze, {})
    config = {"paths": {"manifest": str(manifest), "split_freeze": str(freeze)},
              "models": {"student": "fake"}}
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "load_frozen", lambda *_: ({}, {}))
    monkeypatch.setattr(cli, "snapshot_metadata", lambda _: {"snapshot": "fake"})
    monkeypatch.setattr(cli, "load_tokenizer", lambda _: SimpleNamespace(chat_template="fake"))
    monkeypatch.setattr(cli, "versions", lambda: {})
    monkeypatch.setattr(cli, "code_hashes", lambda _: {})
    output = tmp_path / "ready.json"
    cli.main(["--template", str(PARTIAL_TEMPLATE), "--config", str(config_path),
              "--snapshot", str(tmp_path / "snapshot"), "--data-root", str(tmp_path),
              "--output", str(output)])
    assert json.loads(capsys.readouterr().out) == {"output": str(output), "runs": 6, "status": "ready"}
    grid = json.loads(output.read_text())
    assert grid["preparation"]["inputs"][str(PARTIAL_TEMPLATE)] == file_hash(PARTIAL_TEMPLATE)
    assert dose.load_ready_run(output, 5) == grid["runs"][5]


def test_nested_grid_and_future_prefixes(fake_collections, tmp_path):
    from sed.seed_selection import NESTED_RULE, NESTED_GRID_K
    template = json.loads(Path("configs/seedsft_dose_grid_full_train_views.json").read_text())
    assert [r["K"] for r in template["runs"]] == [30, 100, 300, 500]
    for k in (1000, 2000):
        template["runs"].append(dict(template["runs"][0], K=k, name=f"k{k}s0"))
        template["runs"][-1]["verified"] = f"source/subsets/k{k}"
    measured, _ = fake_collections(template)
    grid = dose.prepare_grid(template, SimpleNamespace(chat_template="fake"), {}, {})
    assert len(measured) == 6
    assert grid["support_registration"]["selection_rule"] == NESTED_RULE
    assert grid["support_registration"]["Ks"] == list(NESTED_GRID_K)
    path = tmp_path / "ready.json"
    write_json(path, grid)
    assert dose.load_ready_run(path, 5)["K"] == 2000


@pytest.mark.parametrize("mutation", ["seed", "duplicate", "legacy_collection", "legacy_grid"])
def test_nested_grid_rejects_wrong_identity(fake_collections, mutation):
    template = json.loads(Path("configs/seedsft_dose_grid_full_train_views.json").read_text())
    template["runs"] = [template["runs"][1]]  # K100 belongs to both grids; rule must distinguish them.
    if mutation == "seed":
        template["runs"][0].update(support_seed=1, name="k100s1")
    elif mutation == "duplicate":
        template["runs"].append(deepcopy(template["runs"][0]))
    elif mutation == "legacy_collection":
        template["runs"][0]["selection_rule"] = "stratified_groups_v1"
    else:
        del template["support_selection_rule"]
    fake_collections(template)
    with pytest.raises(ValueError, match="registered runs|selection rule"):
        dose.prepare_grid(template, SimpleNamespace(chat_template="fake"), {}, {})
