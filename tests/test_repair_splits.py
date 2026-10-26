import json
from pathlib import Path

import pytest

from data.manifest.build_repair_splits import KINDS, SEED_GRID, build_splits, main, read_source
from sed.manifest import freeze_splits
from sed.storage import digest, file_hash, write_json


@pytest.fixture
def universe():
    rows = []
    for source, count in (("train", 30), ("valid_seen", 2), ("valid_unseen", 2)):
        for kind in KINDS:
            for i in range(count):
                for trial in range(2):
                    group = f"{source}/{kind}-{i}"
                    rows.append({"task_id": f"{group}/trial_{trial}", "group_id": group,
                        "source_split": source, "eligible": True, "task_type": kind})
    manifest = {"tasks": rows}
    freeze = freeze_splits(manifest)
    dev = [t["task_id"] for t in rows if t["source_split"] == "valid_seen"]
    listing = {"task_ids": dev, "list_sha256": digest(dev),
        "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze)}
    return manifest, freeze, listing


def test_stratification_hash_order_and_group_exclusions(universe):
    m, f, dev = universe
    exclusions = [f"train/{k}-0/trial_0" for k in KINDS]
    result = build_splits(m, f, [{"task_ids": exclusions}], dev)
    assert result == build_splits(m, f, [{"task_ids": exclusions}], dev)
    assert result["assertions"]["all_passed"]
    assert all(n == 0 for pair in result["assertions"]["overlap_counts"].values() for n in pair.values())
    probe, quick, mech = [result["sets"][n] for n in ("D_probe", "D_unseen_quick", "U_mech")]
    assert [s["unique_instances"] for s in (probe, quick, mech)] == [60, 24, 20]
    assert list(mech["quotas"].values()) == [4, 4, 3, 3, 3, 3]
    for listing in (probe, quick, mech):
        assert listing["task_groups"] == listing["unique_instances"]
        assert listing["counts_by_task_type"] == listing["quotas"]
        assert listing["task_ids"] == sorted(listing["task_ids"], key=lambda t: digest([listing["seed"], t]))
        assert listing["list_sha256"] == digest(listing["task_ids"])
        assert not any(g.endswith("-0") for g in listing["group_ids"])
        assert all(t.startswith("train/") for t in listing["task_ids"])
    for k in KINDS:
        expected = [t for t in probe["task_ids"] if t.startswith(f"train/{k}-")][:4]
        assert expected == [t for t in quick["task_ids"] if t.startswith(f"train/{k}-")]
    assert not set(probe["group_ids"]) & set(mech["group_ids"])
    # Exact first distinct groups of each type under the specified hash rule.
    for k in KINDS:
        rows = sorted((t for t in m["tasks"] if t["source_split"] == "train" and t["task_type"] == k
                       and not t["group_id"].endswith("-0")), key=lambda t: digest([20261026, t["task_id"]]))
        expected, groups = [], set()
        for t in rows:
            if t["group_id"] not in groups and len(expected) < 10:
                groups.add(t["group_id"])
                expected.append(t["task_id"])
        assert expected == [t for t in probe["task_ids"] if t.startswith(f"train/{k}-")]


def test_shortfalls_and_actual_exposure_exhaustion_are_explicit(universe):
    m, f, dev = universe
    ids = [t["task_id"] for t in m["tasks"] if t["source_split"] == "train"
           and t["task_type"] == KINDS[0] and int(t["group_id"].rsplit("-", 1)[1]) >= 2]
    result = build_splits(m, f, [{"task_ids": ids}], dev)
    assert result["sets"]["D_probe"]["shortfalls"][KINDS[0]] == 8
    assert result["sets"]["D_unseen_quick"]["shortfalls"][KINDS[0]] == 2
    assert result["sets"]["U_mech"]["shortfalls"][KINDS[0]] == 4
    all_ids = [t["task_id"] for t in m["tasks"] if t["source_split"] == "train"]
    result = build_splits(m, f, [{"task_ids": all_ids}], dev)
    for listing in result["sets"].values():
        assert listing["task_ids"] == []
        assert listing["shortfalls"] == listing["quotas"]


def test_eval_groups_and_train_derived_dev_are_never_selected(universe):
    m, _, _ = universe
    f = freeze_splits(m, dev_source="train", dev_count=6)
    dev = [t for t, a in f["assignments"].items() if a["split"] == "dev"]
    listing = {"task_ids": dev, "list_sha256": digest(dev), "manifest_sha256": digest(m),
               "split_freeze_sha256": digest(f)}
    result = build_splits(m, f, [], listing)
    for value in result["sets"].values():
        assert not set(value["task_ids"]) & set(dev)
        assert all(f["assignments"][t]["split"] == "train" for t in value["task_ids"])


@pytest.mark.parametrize("fields", [("selected",), ("task_order",), ("selected", "task_order")])
def test_only_selected_groups_excluded_despite_full_train_candidates(universe, tmp_path, fields):
    m, f, dev = universe
    train = [t for t in m["tasks"] if t["source_split"] == "train"]
    chosen = [t for t in train if t["task_id"].endswith("-0/trial_0")]
    ids = [t["task_id"] for t in chosen]
    all_ids = [t["task_id"] for t in train]
    selection = {"selected": {k: [t for t in chosen if t["task_type"] == k] for k in KINDS},
                 "task_order": ids}
    value = {key: selection[key] for key in fields}
    value.update(replacements={"all": train}, candidates=train, candidate_order=all_ids,
                 full_order=all_ids, task_ids=all_ids,
                 unrelated={"selected": {"all": train}, "task_order": all_ids})
    # These unselected fields are never interpreted, even if their IDs are stale.
    value["candidates"].append({"task_id": "train/unknown/trial_0"})
    path = tmp_path / "selection.json"
    write_json(path, value)
    source = read_source(path, "k500", m, f)
    assert source["task_ids"] == sorted(ids)
    assert source["group_ids"] == sorted(t["group_id"] for t in chosen)
    assert source["task_groups"] == 6
    assert source["sha256"] == file_hash(path)
    assert source["list_sha256"] == digest(source["task_ids"])
    assert source["group_list_sha256"] == digest(source["group_ids"])
    result = build_splits(m, f, [source], dev)
    assert result["E"]["task_groups"] == 6
    assert result["eligible_after_exclusions"]["task_groups"] == 6 * 29
    assert result["eligible_after_exclusions"]["unique_instances"] == 6 * 29 * 2
    assert result["eligible_after_exclusions"]["groups_by_task_type"] == dict.fromkeys(KINDS, 29)
    assert [s["unique_instances"] for s in result["sets"].values()] == [60, 24, 20]
    for listing in result["sets"].values():
        assert not set(listing["group_ids"]) & set(source["group_ids"])
        assert set(listing["task_ids"]) <= set(all_ids) - set(ids)


@pytest.mark.parametrize("value, message", [
    ({"candidates": []}, "selected or task_order"),
    ({"selected": []}, "mapping"),
    ({"selected": {"kind": ["train/unknown/trial_0"]}}, "absent"),
    ({"task_order": ["train/unknown/trial_0"]}, "absent"),
])
def test_bad_selection_fails_closed(universe, tmp_path, value, message):
    m, f, _ = universe
    path = tmp_path / "selection.json"
    write_json(path, value)
    with pytest.raises(ValueError, match=message):
        read_source(path, "k500", m, f)


def test_selection_consistency_and_frozen_input_hashes(universe, tmp_path):
    m, f, _ = universe
    path = tmp_path / "selection.json"
    write_json(path, {"selected": {"kind": [m["tasks"][0]]}, "task_order": [m["tasks"][2]["task_id"]]})
    with pytest.raises(ValueError, match="differs"):
        read_source(path, "k500", m, f)
    write_json(path, {"selected": {"kind": [{**m["tasks"][0], "group_id": m["tasks"][2]["group_id"]}]}})
    with pytest.raises(ValueError, match="group differs"):
        read_source(path, "k500", m, f)
    write_json(path, {"manifest_sha256": "changed"})
    with pytest.raises(ValueError, match="frozen"):
        read_source(path, "k500", m, f)


def test_r03_excludes_all_named_tasks_and_group_only_entries(universe, tmp_path):
    m, f, dev = universe
    path = tmp_path / "r03.json"
    rows = m["tasks"]
    write_json(path, {"task_ids": [rows[0]["task_id"]], "group_ids": [rows[2]["group_id"]],
                      "nested": {rows[4]["group_id"]: [rows[6]["task_id"]]}})
    source = read_source(path, "r03", m, f, selected_only=False)
    assert source["unique_instances"] == 2
    assert source["task_groups"] == 4
    result = build_splits(m, f, [source], dev)
    assert result["eligible_after_exclusions"]["task_groups"] == 6 * 30 - 4
    for listing in result["sets"].values():
        assert not set(listing["group_ids"]) & set(source["group_ids"])
    write_json(path, {"group_ids": ["train/unknown_group"]})
    with pytest.raises(ValueError, match="absent"):
        read_source(path, "r03", m, f, selected_only=False)


@pytest.fixture
def cli_sources(universe, tmp_path):
    m, f, dev = universe
    seed_dir, subsets = tmp_path / "seeds", tmp_path / "subsets"
    for name in SEED_GRID:
        row = m["tasks"][2 if name == "seeds_k20_s1.json" else 0]
        write_json(seed_dir / name, {"selected": {"kind": [row]}, "replacements": m["tasks"]})
    write_json(subsets / "k30" / "selection.json", {"task_order": [m["tasks"][4]["task_id"]]})
    write_json(subsets / "k30" / "subsets" / "k10" / "selection.json", {"task_order": [m["tasks"][6]["task_id"]]})
    # Non-selection collection metadata must not contribute exclusions.
    write_json(subsets / "k30" / "manifest.json", {"task_ids": [t["task_id"] for t in m["tasks"]]})
    fixtures = {"manifest": m, "freeze": f, "dev": dev,
                "full": {"selected": {"kind": m["tasks"]}},
                "chain": {"task_order": [m["tasks"][10]["task_id"]]}}
    for name, value in fixtures.items():
        write_json(tmp_path / f"{name}.json", value)
    r03_paths = [tmp_path / f"r03_{name}.json" for name in ("pool200", "surrogate40", "surrogate", "dev300")]
    for path in r03_paths:
        write_json(path, {"task_ids": [m["tasks"][12]["task_id"]], "group_ids": [m["tasks"][14]["group_id"]]})
    args = ["--manifest", str(tmp_path / "manifest.json"), "--freeze", str(tmp_path / "freeze.json"),
        "--dev140", str(tmp_path / "dev.json"),
        "--seed-dir", str(seed_dir), "--full-train", str(tmp_path / "full.json"),
        "--chain-selection", str(tmp_path / "chain.json"), "--chain-subsets", str(subsets),
        "--legacy-chain-collection", str(tmp_path / "legacy"),
        "--r03", *map(str, r03_paths), "--output", str(tmp_path / "result.json")]
    return args


@pytest.mark.parametrize("legacy_present", [False, True])
def test_cli_sources_hashes_nested_views_and_rebuild(universe, cli_sources, tmp_path, capsys, legacy_present):
    m, _, _ = universe
    if legacy_present:
        write_json(tmp_path / "legacy" / "selection.json", {"task_order": [m["tasks"][16]["task_id"]]})
        write_json(tmp_path / "legacy" / "subsets" / "k10" / "selection.json", {"selected": {"kind": [m["tasks"][18]]}})
    else:
        # A directory with no selection metadata is also an absent optional source.
        (tmp_path / "legacy").mkdir()
    args = cli_sources
    first = main(args)
    assert first["E"]["task_groups"] == (9 if legacy_present else 7)
    assert len(first["E"]["task_ids"]) == (8 if legacy_present else 6)
    assert len(first["optional_legacy_chain"]["selection_sources_found"]) == (2 if legacy_present else 0)
    assert len(first["exclusion_sources"]) == (21 if legacy_present else 19)
    assert not any(s["path"] == str(tmp_path / "full.json") for s in first["exclusion_sources"])
    considered, = first["considered_not_excluded"]
    assert considered["path"] == str(tmp_path / "full.json")
    assert considered["sha256"] == file_hash(considered["path"])
    assert considered["excluded_groups"] == 0
    assert "not actual baseline training exposure" in considered["reason"]
    assert all(s["sha256"] == file_hash(s["path"]) for s in first["exclusion_sources"])
    assert all(s["task_groups"] == len(s["group_ids"]) == s["excluded_train_groups"] for s in first["exclusion_sources"])
    assert first["content_sha256"] == digest({k: v for k, v in first.items() if k != "content_sha256"})
    content = (tmp_path / "result.json").read_bytes()
    assert main(args) == first
    assert (tmp_path / "result.json").read_bytes() == content
    write_json(tmp_path / "chain.json", {"task_order": [m["tasks"][20]["task_id"]]})
    with pytest.raises(ValueError, match="overwrite"):
        main(args)
    (tmp_path / "chain.json").unlink()
    with pytest.raises(FileNotFoundError):
        main(args)


@pytest.mark.parametrize("missing", ["chain.json", "subsets/k30/selection.json",
    "subsets/k30/subsets/k10/selection.json", "seeds/seeds_k5_s2.json", "r03_dev300.json", "full.json"])
def test_missing_required_source_fails_closed(cli_sources, tmp_path, missing):
    (tmp_path / missing).unlink()
    with pytest.raises(FileNotFoundError):
        main(cli_sources)
    assert not (tmp_path / "result.json").exists()


def test_declared_optional_chain_view_cannot_silently_lose_selection(cli_sources, tmp_path):
    (tmp_path / "legacy" / "subsets" / "k100").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        main(cli_sources)


def test_dev140_inside_train_universe_fails_closed(universe):
    m, f, _ = universe
    ids = [m["tasks"][0]["task_id"]]
    dev = {"task_ids": ids, "list_sha256": digest(ids), "manifest_sha256": digest(m),
           "split_freeze_sha256": digest(f)}
    with pytest.raises(ValueError, match="universe overlaps"):
        build_splits(m, f, [], dev)
