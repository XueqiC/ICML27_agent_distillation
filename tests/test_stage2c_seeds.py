from copy import deepcopy
import json
from pathlib import Path

import pytest

from eval.run_suite import frozen_tasks
from sed.dev_subset import build_dev140
from sed.manifest import frozen_write, load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import load_verified_collection, materialize_subsets, verified_collection_inputs
from sed.seed_selection import select_seeds
from sed.storage import digest, file_hash, write_json
from teachers.collect_seeds import collect
from test_stage2b_seeds import seed_bank, factories


def test_dev140_full_frozen_deterministic_and_rejects_tamper(tmp_path):
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json", "data/manifest/alfworld_split_freeze.json")
    value = build_dev140(manifest, freeze)
    assert value == build_dev140(manifest, freeze)
    assert value == json.loads(Path("data/manifest/dev140.json").read_text())
    assert value["unique_instances"] == len(set(value["task_ids"])) == 140
    assert set(value["task_ids"]) == {t["task_id"] for t in manifest["tasks"]
                                     if t["source_split"] == "valid_seen" and t["eligible"]}
    assert all(freeze["assignments"][t]["split"] == "dev" and not freeze["assignments"][t]["sealed"]
               for t in value["task_ids"])
    assert value["quotas"] == value["counts_by_task_type"]
    assert sum(value["quotas"].values()) == 140
    assert value["list_sha256"] == digest(value["task_ids"])
    reverse = dict(manifest, tasks=list(reversed(manifest["tasks"])))
    assert build_dev140(reverse, dict(freeze, manifest_sha256=digest(reverse)))["task_ids"] == value["task_ids"]
    path = tmp_path / "dev140.json"
    frozen_write(path, value)
    frozen_write(path, value)
    assert len(frozen_tasks(path, manifest, freeze, policy="teacher")[1]) == 140
    with pytest.raises(ValueError, match="overwrite"):
        frozen_write(path, dict(value, seed=1))
    # Even a recomputed list hash cannot turn an incomplete list into dev140.
    bad = dict(value, task_ids=value["task_ids"][:-1], list_sha256=digest(value["task_ids"][:-1]))
    write_json(path, bad)
    with pytest.raises(ValueError, match="full-dev"):
        frozen_tasks(path, manifest, freeze)
    with pytest.raises(ValueError, match="140"):
        build_dev140(dict(manifest, tasks=[]), dict(freeze, manifest_sha256=digest(dict(manifest, tasks=[]))))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nested_frozen_selections_and_k_independent_shuffles(seed):
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json", "data/manifest/alfworld_split_freeze.json")
    selections = [select_seeds(manifest, freeze, k=k, seed=seed) for k in (5, 10, 20)]
    for value in selections:
        assert value == json.loads(Path(f"data/manifest/seeds_k{value['K']}_s{seed}.json").read_text())
        assert value["trusted_trajectories"] == 0
    for small, large in zip(selections, selections[1:]):
        assert small["type_order"] == large["type_order"]
        small_groups, large_groups = set(), set()
        for kind in small["type_order"]:
            s, l = small["selected"][kind], large["selected"][kind]
            assert s == l[:small["quotas"][kind]]
            assert s + small["replacements"][kind] == l + large["replacements"][kind]
            small_groups.update(r["group_id"] for r in s)
            large_groups.update(r["group_id"] for r in l)
        assert small_groups < large_groups


def test_collect_once_subsets_reuse_receipts_replacements_and_loader(seed_bank, tmp_path, monkeypatch):
    import train.dataset as dataset
    root, manifest, freeze, config = seed_bank
    large = select_seeds(manifest, freeze, k=20)
    failed = large["selected"][large["type_order"][0]][0]["task_id"]
    calls = []
    environment, teacher = factories(root, config, {failed}, calls)
    output = tmp_path / "k20"
    source = collect(large, manifest, freeze, config, output, attempts=2,
                     env_factory=environment, teacher_factory=teacher)
    assert len(calls) == 22 and source["trusted_trajectories"] == 20
    subsets = materialize_subsets(output, manifest, freeze, [5, 10])
    assert materialize_subsets(output, manifest, freeze, [5, 10]) == subsets
    assert collect(large, manifest, freeze, config, output, attempts=2,
                   env_factory=environment, teacher_factory=teacher) == source
    assert len(calls) == 22
    sets = []
    for k, value in zip((5, 10), subsets):
        subset_dir = output / "subsets" / f"k{k}"
        checked, small, evidence_root = load_verified_collection(subset_dir, manifest, freeze)
        assert checked == value and evidence_root == output
        assert value["K"] == value["trusted_trajectories"] == value["reused_trajectories"] == k
        assert value["new_teacher_calls"] == value["new_environment_steps"] == 0
        assert len(value["attempts"]) == k + 2  # failed attempts stay visible
        assert not (subset_dir / "attempts").exists()
        for receipt, ref in zip(value["attempts"], value["receipt_refs"]):
            path = subset_dir / ref["path"]
            assert receipt in source["attempts"]
            assert json.loads(path.read_text()) == receipt
            assert file_hash(path) == ref["sha256"]
            assert path.resolve() == output / receipt["episode_dir"].replace("/episode", "/receipt.json")
        parents = {r["task_id"] for rows in small["selected"].values() for r in rows}
        assert {r["selection_parent_task_id"] for r in value["verified"]} == parents
        assert all(r in source["verified"] for r in value["verified"])
        sets.append({r["task_id"] for r in value["verified"]})
        # Exercise the training entry point without torch/tokenizer dependencies.
        monkeypatch.setattr(dataset, "episode_examples", lambda e, *_args, **_kwargs: [digest(e)])
        examples = dataset.build_verified_dataset(subset_dir, None, manifest, freeze)
        assert examples == [digest(json.loads((output / r["episode_dir"] / "episode.json").read_text()))
                            for r in value["verified"]]
    assert sets[0] < sets[1] < {r["task_id"] for r in source["verified"]}
    # Forged refs, altered source, or altered raw evidence cannot be loaded.
    subpath = output / "subsets/k5/manifest.json"
    bad = deepcopy(subsets[0])
    bad["receipt_refs"][0]["path"] = "../../outside/receipt.json"
    write_json(subpath, bad)
    with pytest.raises(ValueError, match="shared source receipts"):
        load_verified_collection(subpath.parent, manifest, freeze)
    write_json(subpath, subsets[0])
    episode = output / source["verified"][0]["episode_dir"] / "episode.json"
    episode.write_text(episode.read_text() + " ")
    with pytest.raises(ValueError, match="evidence changed"):
        load_verified_collection(subpath.parent, manifest, freeze)


def test_verified_collection_provenance_inputs(seed_bank, tmp_path, monkeypatch):
    root, manifest, freeze, config = seed_bank
    environment, teacher = factories(root, config, set(), [])
    source = tmp_path / "seeds_k20_s0"
    collect(select_seeds(manifest, freeze, k=20), manifest, freeze, config, source,
            env_factory=environment, teacher_factory=teacher)
    materialize_subsets(source, manifest, freeze, [5, 10])
    monkeypatch.chdir(tmp_path)
    for k in (20, 5, 10):
        directory = Path(source.name) if k == 20 else Path(source.name) / "subsets" / f"k{k}"
        expected = [directory / "manifest.json"]
        if k == 20:
            expected.append(directory / "provenance.json")
        else:
            assert {p.name for p in directory.iterdir()} == {
                "manifest.json", "identity.json", "selection.json"}
            expected.extend([directory / "identity.json", directory / "selection.json",
                             source / "manifest.json", source / "provenance.json"])
        inputs = verified_collection_inputs(str(directory), manifest, freeze)
        assert inputs == expected
        output = tmp_path / f"provenance_k{k}"
        record = write_provenance(output, command="sft", config=config, inputs=inputs)
        expected_hashes = {str(path): file_hash(path) for path in expected}
        assert record["inputs"] == expected_hashes
        assert json.loads((output / "provenance.json").read_text())["inputs"] == expected_hashes


def test_partial_collection_cannot_materialize_verified_subsets(seed_bank, tmp_path):
    _, manifest, freeze, _ = seed_bank
    large = select_seeds(manifest, freeze, k=20)
    identity = {"selection": large}
    write_json(tmp_path / "identity.json", identity)
    write_json(tmp_path / "selection.json", large)
    write_json(tmp_path / "manifest.json", {"status": "exhausted", "identity_sha256": digest(identity)})
    with pytest.raises(ValueError, match="exactly K"):
        materialize_subsets(tmp_path, manifest, freeze, [5, 10])
    assert not (tmp_path / "subsets").exists()
