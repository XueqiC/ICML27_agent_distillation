"""Stage 2f: fake environments/teachers, immutable inheritance and CPU planning."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sed.manifest import TASK_TYPES, build_manifest, freeze_splits, load_frozen
from sed.seed_evidence import artifact_hashes, load_verified_collection, materialize_subsets, verified_collection_inputs
from sed.seed_selection import select_seeds, validate_extension_selection
from sed.storage import write_json
from teachers.collect_seeds import collect
from train.dose_grid import dose_checkpoints, load_ready_run, prepare_grid
from test_stage2b_seeds import seed_bank, factories


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full_few_shot_quotas_and_frozen_candidate_identity(seed):
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json", "data/manifest/alfworld_split_freeze.json")
    values = [select_seeds(manifest, freeze, k=k, seed=seed) for k in (1, 2, 5, 10, 20, 50, 100)]
    for s in values:
        assert list(s["quotas"].values()) == [s["K"] // 6 + int(i < s["K"] % 6) for i in range(6)]
        assert sum(s["quotas"].values()) == s["K"]
        if s["K"] in (5, 10, 20, 100):
            assert s == json.loads(Path(f"data/manifest/seeds_k{s['K']}_s{seed}.json").read_text())
    for small, large in zip(values, values[1:]):
        validate_extension_selection(small, large)
    bad = deepcopy(values[-1])
    kind = bad["type_order"][0]
    bad["replacements"][kind].reverse()
    with pytest.raises(ValueError, match="candidate order"):
        validate_extension_selection(values[-2], bad)


@pytest.fixture
def large_bank(seed_bank):
    root, _, _, config = seed_bank
    for kind in TASK_TYPES:
        for group in range(5, 22):
            folder = root / "train" / f"{kind}-{group}" / "trial_0"
            folder.mkdir(parents=True)
            write_json(folder / "traj_data.json", {"task_type": kind})
            write_json(folder / "game.tw-pddl", {"solvable": True})
    manifest = build_manifest(root)
    return root, manifest, freeze_splits(manifest), config


def test_extend_reuses_all_attempts_resume_subsets_and_training(large_bank, tmp_path, monkeypatch):
    import teachers.collect_seeds as collector
    import train.dataset as dataset

    root, manifest, freeze, config = large_bank
    small, large = [select_seeds(manifest, freeze, k=k) for k in (20, 100)]
    kind = small["type_order"][0]
    losing = {r["task_id"] for r in small["selected"][kind][:2]}
    calls = []
    env, teacher = factories(root, config, losing, calls)
    source, target = tmp_path / "k20", tmp_path / "k100"
    old = collect(small, manifest, freeze, config, source, env_factory=env, teacher_factory=teacher)
    old_views = materialize_subsets(source, manifest, freeze, [5, 10])
    before = artifact_hashes(source)
    attempted = {r["task_id"] for r in old["attempts"]}
    old_calls = len(calls)
    # A further failed candidate must consume its three attempts, then the next
    # unattempted candidate of this type. Never revisit any inherited failure.
    new_failure = next(r["task_id"] for r in large["selected"][kind] if r["task_id"] not in attempted)
    losing.add(new_failure)
    def guarded(task, *args):
        assert task["task_id"] not in attempted
        return teacher(task, *args)
    original_write = collector.write_json
    def interrupt(path, *args, **kwargs):
        if Path(path).name == "receipt.json" and len(calls) == old_calls + 4:
            raise KeyboardInterrupt
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(collector, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect(large, manifest, freeze, config, target, extend_from=source,
                env_factory=env, teacher_factory=guarded)
    monkeypatch.setattr(collector, "write_json", original_write)
    result = collect(large, manifest, freeze, config, target, extend_from=source,
                     env_factory=env, teacher_factory=guarded)
    assert len(calls) == old_calls + 83  # 80 new wins + 3 failed attempts; no replayed call
    assert result["status"] == "complete" and result["trusted_trajectories"] == 100
    assert result["reused_attempts"] == len(old["attempts"])
    assert result["reused_trajectories"] == 20 and result["new_verified_trajectories"] == 80
    assert result["attempts"][:len(old["attempts"])] == old["attempts"]
    copied_hashes = artifact_hashes(target / "attempts")
    for relative, sha in artifact_hashes(source / "attempts").items():
        assert copied_hashes[relative] == sha
    assert collect(large, manifest, freeze, config, target, extend_from=source,
                   env_factory=env, teacher_factory=guarded) == result
    assert len(calls) == old_calls + 83
    views = materialize_subsets(target, manifest, freeze, [1, 2, 5, 10, 20, 50])
    assert views == materialize_subsets(target, manifest, freeze, [1, 2, 5, 10, 20, 50])
    by_k = {v["K"]: v for v in views}
    for k, baseline in [(5, old_views[0]), (10, old_views[1]), (20, old)]:
        assert by_k[k]["verified"] == baseline["verified"]
        assert by_k[k]["attempts"] == baseline["attempts"]
    for k in (1, 2):
        assert by_k[k]["verified"] == by_k[5]["verified"][:k]
    # Exercise the actual training collection path without model dependencies.
    monkeypatch.setattr(dataset, "episode_examples", lambda e, *_a, **_k: [e["task_id"]])
    for k in (1, 2, 5, 10, 20, 50, 100):
        path = target if k == 100 else target / "subsets" / f"k{k}"
        collection, selection, evidence = load_verified_collection(path, manifest, freeze)
        assert evidence == target and selection["K"] == k
        assert dataset.build_verified_dataset(path, None, manifest, freeze) == [r["task_id"] for r in collection["verified"]]
        inputs = verified_collection_inputs(path, manifest, freeze)
        assert source / "provenance.json" in inputs and all(p.is_file() for p in inputs)
    assert artifact_hashes(source) == before
    bad = deepcopy(result)
    bad["verified"][0], bad["verified"][1] = bad["verified"][1], bad["verified"][0]
    write_json(target / "manifest.json", bad)
    with pytest.raises(ValueError, match="receipt identity mismatch"):
        materialize_subsets(target, manifest, freeze, [1])
    write_json(target / "manifest.json", result)
    # Reuse is integrity checked; neither missing evidence nor an altered raw
    # teacher cache can be repaired silently by issuing new teacher calls.
    receipt = old["attempts"][0]
    evidence = target / receipt["episode_dir"] / "episode.json"
    evidence.write_text(evidence.read_text() + " ")
    with pytest.raises(ValueError, match="Inherited attempt evidence changed"):
        collect(large, manifest, freeze, config, target, extend_from=source,
                env_factory=env, teacher_factory=guarded)
    assert len(calls) == old_calls + 83 and artifact_hashes(source) == before


def test_extension_refuses_incompatible_sources_before_calls(seed_bank, tmp_path):
    root, manifest, freeze, config = seed_bank
    env, teacher = factories(root, config, set(), [])
    source = tmp_path / "k5"
    collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, source,
            env_factory=env, teacher_factory=teacher)
    large = select_seeds(manifest, freeze, k=20)
    with pytest.raises(ValueError, match="separate trees"):
        collect(large, manifest, freeze, config, source / "child", extend_from=source)
    with pytest.raises(ValueError, match="attempts per candidate"):
        collect(large, manifest, freeze, config, tmp_path / "a", extend_from=source, attempts=2)
    with pytest.raises(ValueError, match="prefixes"):
        collect(select_seeds(manifest, freeze, k=20, seed=1), manifest, freeze, config,
                tmp_path / "b", extend_from=source)
    changed = deepcopy(config)
    changed["models"]["teacher"]["sampling"]["max_output_tokens"] += 1
    with pytest.raises(ValueError, match="teacher identity"):
        collect(large, manifest, freeze, changed, tmp_path / "c", extend_from=source)
    assert not any((tmp_path / p).exists() for p in ("a", "b", "c"))


def test_interrupted_inheritance_copy_resumes_without_source_calls(seed_bank, tmp_path, monkeypatch):
    import sed.seed_extension as extension
    root, manifest, freeze, config = seed_bank
    calls = []
    env, teacher = factories(root, config, set(), calls)
    source, target = tmp_path / "k5", tmp_path / "k10"
    old = collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, source,
                  env_factory=env, teacher_factory=teacher)
    before = artifact_hashes(source)
    original = extension.atomic_write
    copies = []
    def interrupt(path, *args, **kwargs):
        original(path, *args, **kwargs)
        copies.append(path)
        if len(copies) == 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(extension, "atomic_write", interrupt)
    large = select_seeds(manifest, freeze, k=10)
    with pytest.raises(KeyboardInterrupt):
        collect(large, manifest, freeze, config, target, extend_from=source,
                env_factory=env, teacher_factory=teacher)
    assert len(calls) == 5 and not (target / "inherited_ready.json").exists()
    monkeypatch.setattr(extension, "atomic_write", original)
    result = collect(large, manifest, freeze, config, target, extend_from=source,
                     env_factory=env, teacher_factory=teacher)
    assert len(calls) == 10 and result["reused_trajectories"] == 5
    assert result["verified"][:5] == old["verified"] and artifact_hashes(source) == before


def test_extension_exhaustion_keeps_inherited_evidence(seed_bank, tmp_path):
    root, manifest, freeze, config = seed_bank
    losing, calls = set(), []
    env, teacher = factories(root, config, losing, calls)
    source, target = tmp_path / "k5", tmp_path / "k20"
    old = collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, source,
                  env_factory=env, teacher_factory=teacher)
    large = select_seeds(manifest, freeze, k=20)
    kind = large["type_order"][0]
    losing.update(r["task_id"] for r in large["selected"][kind] + large["replacements"][kind])
    result = collect(large, manifest, freeze, config, target, extend_from=source,
                     env_factory=env, teacher_factory=teacher)
    assert result["status"] == "exhausted" and result["verified"] == old["verified"]
    assert len(calls) == 5 + 4 * 3
    assert collect(large, manifest, freeze, config, target, extend_from=source,
                   env_factory=env, teacher_factory=teacher) == result
    assert len(calls) == 17
    with pytest.raises(ValueError, match="exactly K"):
        materialize_subsets(target, manifest, freeze, [5])


def test_dose_schedule_reproduces_v1_and_merge_boundary():
    old = json.loads(Path("configs/seedsft_dose_grid.json").read_text())
    for run in old["runs"]:
        checkpoints, _ = dose_checkpoints(run["supervised_tokens_per_pass"])
        assert checkpoints == run["checkpoints"]
        assert max(c["tokens"] for c in checkpoints) == run["total_tokens"]
    assert dose_checkpoints(9500)[1]["t100k"] == "p10"  # exactly 5%
    assert "t100k" not in dose_checkpoints(9499)[1]
    with pytest.raises(ValueError, match="Prepare"):
        load_ready_run("configs/seedsft_dose_grid_v2.json", 0)


def test_prepare_grid_measures_labels_and_all_twelve_runs(tmp_path, monkeypatch):
    import train.dose_grid as module
    template = json.loads(Path("configs/seedsft_dose_grid_v2.json").read_text())
    measured = []
    def collection(path, *_):
        run = next(r for r in template["runs"] if Path(r["verified"]) == path)
        return {"K": run["K"]}, {"K": run["K"], "seed": run["support_seed"]}, path
    def examples(path, tokenizer, *_args, **kwargs):
        assert kwargs == {"max_length": 32768, "packing": "episode"}
        k = collection(path)[0]["K"]
        measured.append(path)
        return [{"labels": [-100] + [106] * (17 * k), "supervised_tokens": 17 * k}]
    monkeypatch.setattr(module, "load_verified_collection", collection)
    monkeypatch.setattr(module, "build_verified_dataset", examples)
    monkeypatch.setattr(module, "verified_collection_inputs", lambda *_: [])
    grid = prepare_grid(template, SimpleNamespace(chat_template="fake"), {}, {})
    assert grid["status"] == "ready" and len(measured) == 12
    assert template["status"] == "requires_collection_tokenization"
    for run in grid["runs"]:
        assert run["supervised_tokens_per_pass"] == 17 * run["K"]
        assert run["total_tokens"] == max(c["tokens"] for c in run["checkpoints"])
    output = tmp_path / "grid.json"
    write_json(output, grid)
    assert load_ready_run(output, 11) == grid["runs"][11]


def test_rai_launcher_preflight_resume_and_sequential_fakes(tmp_path):
    collection_root, logs = tmp_path / "collections", tmp_path / "logs"
    for seed in range(3):
        write_json(collection_root / f"seeds_k20_s{seed}" / "manifest.json", {})
    existing = collection_root / "seeds_k100_s2"
    write_json(existing / "identity.json", {})
    ledger = tmp_path / "calls.jsonl"
    fake = tmp_path / "python"
    fake.write_text("#!" + sys.executable + "\n" + '''
import json, os, sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
args = sys.argv[1:]
with Path(os.environ['FAKE_LEDGER']).open('a') as f:
    f.write(json.dumps(args) + '\\n')
out = Path(args[args.index('--output') + 1])
out.mkdir(exist_ok=True)
(out / 'identity.json').write_text('{}')
''')
    fake.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), ENV_PYTHON=str(fake),
               COLLECTION_ROOT=str(collection_root), LOG_ROOT=str(logs), FAKE_LEDGER=str(ledger))
    command = ["bash", "scripts/rai_collect_k100.sh"]
    refusal = subprocess.run(command, env=env, capture_output=True, text=True)
    assert refusal.returncode == 2 and "--resume" in refusal.stderr
    assert not ledger.exists()  # Even s0 must not start when s2 already exists.
    resumed = subprocess.run([*command, "--resume"], env=env, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stderr
    calls = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(calls) == len(list(logs.glob("*.log"))) == 3
    for seed, args in enumerate(calls):
        assert args[args.index("--extend-from") + 1] == str(collection_root / f"seeds_k20_s{seed}")
        assert args[args.index("--subset-k") + 1:] == ["1", "2", "5", "10", "20", "50"]
