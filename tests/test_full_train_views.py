"""Offline nested full-TRAIN views, real evidence loaders and fake collectors."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sed.manifest import TASK_TYPES
from sed.seed_evidence import (artifact_hashes, load_verified_collection, materialize_full_train_views,
                               materialize_subsets, verified_collection_inputs)
from sed.seed_selection import (NESTED_RULE, select_nested_verified, stratified_nested_order)
from sed.storage import digest, write_json
from teachers.collect_seeds import collect
from test_full_train import full_bank, factories


@pytest.fixture
def source_bank(full_bank, tmp_path):
    _, manifest, freeze, selection, config = full_bank
    output = tmp_path / "source"
    env, teacher = factories(full_bank, tmp_path / "calls.jsonl", {selection["candidate_order"][0]})
    source = collect(selection, manifest, freeze, config, output, env_factory=env, teacher_factory=teacher)
    return output, source, manifest, freeze, config, env


def synthetic_universe():
    rows = []
    for i, kind in enumerate(sorted(TASK_TYPES)):
        for g in range(13 + i * 3):
            for trial in range(1 + g % (i + 2)):
                rows.append(dict(task_id=f"{kind}/{g}/{trial}", group_id=f"{kind}/{g}", task_type=kind))
    return {"verified": rows}


def test_determinism_nesting_stratification_and_group_rounds():
    source = synthetic_universe()
    order = stratified_nested_order(source)
    assert order == stratified_nested_order(source)
    assert order == stratified_nested_order({"verified": list(reversed(source["verified"]))})
    assert order != stratified_nested_order(source, seed=1)
    assert {r["task_id"] for r in order} == {r["task_id"] for r in source["verified"]}
    assert len(order) == len({r["task_id"] for r in order})
    totals = Counter(r["task_type"] for r in order)
    counts, N = Counter(), len(order)
    for n, row in enumerate(order, 1):
        counts[row["task_type"]] += 1
        assert all(abs(counts[t] * N - n * total) <= N for t, total in totals.items())
    for kind in totals:
        group_sizes = Counter(r["group_id"] for r in order if r["task_type"] == kind)
        seen, rounds = Counter(), defaultdict(list)
        for row in (r for r in order if r["task_type"] == kind):
            group = row["group_id"]
            remaining = [g for g in group_sizes if seen[g] < group_sizes[g]]
            assert seen[group] == min(seen[g] for g in remaining)
            seen[group] += 1
            rounds[seen[group]].append(group)
        for n, groups in rounds.items():
            assert groups == [g for g in rounds[1] if group_sizes[g] >= n]
    # Exact ties use lexicographic type order; no randomized type order.
    tied = {"verified": [dict(task_id=t, group_id=t, task_type=t) for t in sorted(TASK_TYPES)]}
    assert [r["task_type"] for r in stratified_nested_order(tied)] == sorted(TASK_TYPES)


def test_future_grid_prefixes_use_one_verified_universe():
    # Exact real-source type totals, synthetic identities only. Audit metadata,
    # failed candidates and unverified rows must not affect ordering.
    rows = []
    for kind, count in zip(sorted(TASK_TYPES), (298, 777, 530, 399, 342, 752)):
        rows.extend(dict(task_id=f"{kind}/{i}", group_id=f"{kind}/{i // 3}", task_type=kind)
                    for i in range(count))
    source = {"status": "complete", "verified": rows, "attempts": [{"task_id": "failed"}]}
    large = {"selection_rule": "all_train_v1", "manifest_sha256": "fake", "split_freeze_sha256": "fake"}
    selections = [select_nested_verified(source, large, k=k) for k in (30, 100, 300, 500, 1000, 2000)]
    assert len({s["full_order_sha256"] for s in selections}) == 1
    for small, big in zip(selections, selections[1:]):
        assert small["task_order"] == big["task_order"][:small["K"]]
    assert all(s["universe_size"] == 3098 for s in selections)
    with_audit = {**source, "verified": [{**r, "audit": "ignored"} for r in reversed(rows)],
                  "attempts": [{"task_id": "different_failed_candidate"}]}
    assert stratified_nested_order(source) == stratified_nested_order(with_audit)
    order = stratified_nested_order(source)
    totals, counts = Counter(r["task_type"] for r in order), Counter()
    for n, row in enumerate(order, 1):
        counts[row["task_type"]] += 1
        assert all(abs(counts[t] * 3098 - n * total) <= 3098 for t, total in totals.items())


def test_selected_retries_are_retained_without_new_costs(full_bank, tmp_path):
    from conftest import FakeBackend
    from harness.alfworld_env import Environment
    root, manifest, freeze, selection, config = full_bank
    _, teacher = factories(full_bank, tmp_path / "calls.jsonl")
    seen = Counter()

    def environment(task):
        seen[task["task_id"]] += 1
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 1 if seen[task["task_id"]] == 2 else 99
            return result
        return Environment(root, max_episode_steps=1, backend_factory=backend)

    output = tmp_path / "retried"
    collect(selection, manifest, freeze, config, output, env_factory=environment, teacher_factory=teacher)
    materialize_full_train_views(output, manifest, freeze, [1])
    view, _, _ = load_verified_collection(output / "subsets/k1", manifest, freeze)
    assert [r["won"] for r in view["attempts"]] == [False, True]
    assert view["verified"] == view["attempts"][1:]
    assert view["attempted_unique_instances"] == view["unique_instances"] == view["K"] == 1
    assert view["new_teacher_calls"] == view["new_environment_steps"] == 0


def test_write_dry_run_later_prefixes_and_provenance(source_bank):
    root, source, manifest, freeze, _, _ = source_bank
    before = artifact_hashes(root)
    planned = materialize_full_train_views(root, manifest, freeze, [1, 2], dry_run=True)
    assert artifact_hashes(root) == before
    assert not (root / "subsets").exists()
    assert materialize_full_train_views(root, manifest, freeze, [1, 2]) == planned
    later = materialize_full_train_views(root, manifest, freeze, [3])
    order = [r["task_id"] for r in stratified_nested_order(source)]
    for selection in [*planned, *later]:
        k = selection["K"]
        view = root / "subsets" / f"k{k}"
        collection, checked, evidence = load_verified_collection(view, manifest, freeze)
        assert selection == checked and evidence == root
        assert selection["task_order"] == order[:k]
        assert selection["full_order_sha256"] == digest(order)
        assert selection["source_manifest_sha256"] == digest(source)
        assert collection["source"] == "reused_verified_nested_collection"
        assert collection["new_teacher_calls"] == collection["new_environment_steps"] == 0
        assert collection["trusted_trajectories"] == collection["unique_instances"] == k
        assert [r["task_id"] for r in collection["verified"]] == order[:k]
        assert collection["attempts"] == [r for r in source["attempts"] if r["task_id"] in order[:k]]
        inputs = verified_collection_inputs(view, manifest, freeze)
        assert all((view / name) in inputs for name in ("manifest.json", "selection.json", "identity.json"))
        assert root / "manifest.json" in inputs and root / "provenance.json" in inputs
        assert {p.name for p in view.iterdir()} == {"manifest.json", "selection.json", "identity.json"}
    assert later[0]["selected_task_groups"] == 2 and later[0]["multi_trial_groups"] == 1
    assert {p: sha for p, sha in artifact_hashes(root).items() if not p.startswith("subsets/")} == before
    # Retain the legacy API's full-TRAIN refusal and its old semantics.
    with pytest.raises(ValueError, match="nested"):
        materialize_subsets(root, manifest, freeze, [1])


@pytest.mark.parametrize("ks", [[], [0], [-1], [True], [1.5], [1, 1], [1, 4]])
def test_refuses_invalid_ks_without_writing(source_bank, ks):
    root, _, manifest, freeze, _, _ = source_bank
    before = artifact_hashes(root)
    with pytest.raises(ValueError):
        materialize_full_train_views(root, manifest, freeze, ks)
    assert artifact_hashes(root) == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_refuses_any_existing_target_before_writes(source_bank, dry_run):
    root, _, manifest, freeze, _, _ = source_bank
    (root / "subsets/k2").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_full_train_views(root, manifest, freeze, [1, 2], dry_run=dry_run)
    assert not (root / "subsets/k1").exists()


def test_refuses_incomplete_or_non_full_source(source_bank):
    root, source, manifest, freeze, _, _ = source_bank
    write_json(root / "manifest.json", {**source, "status": "incomplete"})
    with pytest.raises(ValueError, match="exactly K"):
        materialize_full_train_views(root, manifest, freeze, [1])
    assert not (root / "subsets").exists()
    with pytest.raises(ValueError, match="all_train_v1"):
        select_nested_verified(source, {"selection_rule": "stratified_groups_v1"}, k=1)


def rebind_local_files(view, selection, identity, collection):
    """An attacker can recompute local hashes but cannot change the source prefix."""
    selection.pop("selection_sha256", None)
    selection["selection_sha256"] = digest(selection)
    identity["selection"] = selection
    collection["selection_sha256"] = digest(selection)
    collection["identity_sha256"] = digest(identity)
    for name, value in (("selection", selection), ("identity", identity), ("manifest", collection)):
        write_json(view / (name + ".json"), value)


@pytest.mark.parametrize("mutation", ["order", "type_order", "counts", "groups", "multi_trial",
                                     "full_hash", "source_hash", "receipt_ref", "episode", "source_selection"])
def test_loader_rejects_tampering_even_with_rebound_local_hashes(source_bank, mutation):
    root, _, manifest, freeze, _, _ = source_bank
    materialize_full_train_views(root, manifest, freeze, [3])
    view = root / "subsets/k3"
    selection, identity, collection = [json.loads((view / (n + ".json")).read_text())
                                       for n in ("selection", "identity", "manifest")]
    if mutation == "order":
        selection["task_order"].reverse()
        collection["verified"].reverse()
    elif mutation == "type_order":
        selection["type_order"].reverse()
    elif mutation == "counts":
        selection["counts_by_task_type"][selection["type_order"][0]] += 1
    elif mutation == "groups":
        selection["selected_task_groups"] += 1
    elif mutation == "multi_trial":
        selection["multi_trial_groups"] = 0
    elif mutation == "full_hash":
        selection["full_order_sha256"] = "0" * 64
    elif mutation == "source_hash":
        selection["source_manifest_sha256"] = "0" * 64
    elif mutation == "receipt_ref":
        collection["receipt_refs"][0]["sha256"] = "0" * 64
    elif mutation == "episode":
        episode = root / collection["verified"][0]["episode_dir"] / "episode.json"
        episode.write_text(episode.read_text() + " ")
    else:
        original = json.loads((root / "selection.json").read_text())
        original["seed"] += 1
        write_json(root / "selection.json", original)
    rebind_local_files(view, selection, identity, collection)
    from sed.validation_cache import clear_validation_cache
    clear_validation_cache()  # Evidence mutation is checked in a fresh session.
    with pytest.raises(ValueError):
        load_verified_collection(view, manifest, freeze)
    with pytest.raises(ValueError):
        verified_collection_inputs(view, manifest, freeze)


def test_cli_read_only_and_creation(source_bank, capsys):
    from tools.make_full_train_views import main
    root, _, manifest, freeze, config, _ = source_bank
    args = ["--source", str(root), "--ks", "1", "3", "--manifest", config["paths"]["manifest"],
            "--freeze", config["paths"]["split_freeze"]]
    before = artifact_hashes(root)
    assert main([*args, "--dry-run"]) == 0
    assert "| 3 |" in capsys.readouterr().out
    assert artifact_hashes(root) == before
    assert main(args) == 0
    assert load_verified_collection(root / "subsets/k3", manifest, freeze)[0]["K"] == 3


def test_baseline_preparation_sft_dataset_and_dose_loader(source_bank, tmp_path):
    from baselines.common import registered_support
    from baselines.common.pipeline import prepare
    from train.dataset import build_verified_dataset
    from test_r03_sets import TinyTokenizer
    from train.dose_grid import prepare_grid

    root, _, manifest, freeze, config, _ = source_bank
    materialize_full_train_views(root, manifest, freeze, [3])
    view = root / "subsets/k3"
    tokenizer = TinyTokenizer()
    tokenizer.chat_template = "fake Gemma OFF template"
    assert len(build_verified_dataset(view, tokenizer, manifest, freeze)) == 3
    for k in (30, 100, 300, 500, 1000, 2000):
        assert registered_support({"selection_rule": NESTED_RULE, "K": k})
    assert not registered_support({"selection_rule": "stratified_groups_v1", "K": 300})
    for method in ("kang", "adftp", "smartad", "sad"):
        args = SimpleNamespace(verified=view, output=tmp_path / method, snapshot=tmp_path / "snapshot",
                               config="configs/base.yaml", gpus=1)
        identity = prepare(args, config, method=method, recipe={})
        assert identity["K"] == identity["unique_instances"] == 3 and identity["task_groups"] == 2
        tasks = json.loads((args.output / "tasks.json").read_text())
        assert tasks["task_ids"] == json.loads((view / "selection.json").read_text())["task_order"]
    # The real registered grid uses larger fixtures; small K remains rejected by
    # the registration rather than silently changing its scientific cells.
    template = json.loads(Path("configs/seedsft_dose_grid_full_train_views.json").read_text())
    template["runs"] = [dict(template["runs"][0], K=3, name="k3s0", verified=str(view))]
    with pytest.raises(ValueError, match="registered runs"):
        prepare_grid(template, tokenizer, manifest, freeze)


@pytest.mark.parametrize("method", ["adftp", "smartad"])
def test_teacher_task_lists_and_published_collections_keep_view_prefix(source_bank, tmp_path, method):
    from baselines.adftp import collect as ftp
    from baselines.smartad import collect as sm
    from baselines.smartad.select import select, validate_selection
    from test_adftp import stub_factory
    from test_smartad import factory
    from train.dataset import build_verified_dataset
    from test_r03_sets import TinyTokenizer

    root, _, manifest, freeze, config, env = source_bank
    materialize_full_train_views(root, manifest, freeze, [3])
    view = root / "subsets/k3"
    selection = json.loads((view / "selection.json").read_text())
    before = artifact_hashes(root)
    config["environment"]["max_episode_steps"] = 40
    calls = []
    kwargs = dict(rate_limit_dir=tmp_path / "limiter", env_factory=env)
    if method == "adftp":
        result = ftp.collect(view, manifest, freeze, config, tmp_path / "ftp",
                             teacher_factory=stub_factory(calls), **kwargs)
        assert {tid for tid, _ in calls} == set(selection["task_order"])
    else:
        pool = sm.collect(view, manifest, freeze, config, tmp_path / "pool",
                          teacher_factory=factory(calls), **kwargs)
        tasks = sm.load_pool(pool, manifest, freeze)[0]["tasks"]
        assert [r["task_id"] for r in tasks] == selection["task_order"]
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        result = select(pool, snapshot, config, manifest, freeze, tmp_path / "selected",
                        scorer=lambda _: {"nll": 1., "turns": [{"nll_mean": 1.}]})
        validate_selection(result, manifest, freeze, snapshot)
    collection, selected, _ = load_verified_collection(result, manifest, freeze)
    assert selected == selection
    assert [r["task_id"] for r in collection["verified"]] == selection["task_order"]
    assert collection["task_groups"] == 2 and collection["K"] == 3
    assert len(build_verified_dataset(result, TinyTokenizer(), manifest, freeze)) == 3
    assert view / "selection.json" in verified_collection_inputs(result, manifest, freeze)
    assert artifact_hashes(root) == before
    selection["full_order_sha256"] = "bad"
    write_json(view / "selection.json", selection)
    with pytest.raises(ValueError):
        load_verified_collection(result, manifest, freeze)
