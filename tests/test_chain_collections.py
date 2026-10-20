"""Teacher-side exact support views, receipt reuse and score sharding (offline)."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.adftp import collect as ftp
from baselines.common.collection_views import bind_views, materialize_views, read
from baselines.smartad import collect as sm
from baselines.smartad.select import select, validate_selection
from sed.seed_evidence import artifact_hashes, load_verified_collection, materialize_subsets
from sed.storage import digest, write_json
from test_adftp import stub_factory as original_ftp_factory
from test_smartad import factory, setup_source
from test_stage2b_seeds import seed_bank


def stub_factory(calls):
    make = original_ftp_factory(calls)
    def shared(*args):
        teacher = make(*args)
        teacher.teacher.limiter.limits = {"requests_per_minute": 100000, "tokens_per_minute": 100000000}
        return teacher
    return shared


def task_ids(collection):
    return {r["task_id"] for r in collection["verified"]}


def test_exact_views_for_both_methods_and_refusal(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=3)
    materialize_subsets(source, manifest, freeze, [1, 2])
    entries = [f"{k}={source}/subsets/k{k}" for k in (1, 2)]
    support = {k: load_verified_collection(source / f"subsets/k{k}", manifest, freeze)[0] for k in (1, 2)}
    before = artifact_hashes(source)
    calls = []
    kwargs = dict(rate_limit_dir=tmp_path / "limit", env_factory=env, subset_views=entries)
    output = tmp_path / "ftp"
    for index in (1, 0, 1):
        ftp.collect(source, manifest, freeze, config, output, shard_count=2, shard_index=index,
                    teacher_factory=stub_factory(calls), **kwargs)
    assert len(calls) == 6
    collection = load_verified_collection(output / "collection", manifest, freeze)[0]
    for k in (1, 2):
        view, _, _ = load_verified_collection(output / f"collection/subsets/k{k}", manifest, freeze)
        assert task_ids(view) == task_ids(support[k])
        assert view["support_view"]["manifest_sha256"] == digest(support[k])
        assert view["new_teacher_calls"] == view["new_environment_steps"] == 0
        assert all(r in collection["verified"] for r in view["verified"])
    pool = sm.collect(source, manifest, freeze, config, tmp_path / "pool", teacher_factory=factory([]), **kwargs)
    for k in (1, 2):
        view = sm.load_pool(pool / f"subsets/k{k}", manifest, freeze)[0]
        assert {t["task_id"] for t in view["tasks"]} == task_ids(support[k])
        assert all(len(t["candidates"]) == 10 for t in view["tasks"])
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    selected = select(pool, snapshot, config, manifest, freeze, tmp_path / "selection",
                      scorer=lambda _: {"nll": 1.0})
    large = load_verified_collection(selected, manifest, freeze)[0]
    for k in (1, 2):
        view, _, _ = load_verified_collection(selected / f"subsets/k{k}", manifest, freeze)
        assert task_ids(view) == task_ids(support[k])
        assert all(r in large["verified"] for r in view["verified"])
        validate_selection(selected / f"subsets/k{k}", manifest, freeze, snapshot)
    assert artifact_hashes(source) == before
    # A supplied label must match the actual support size, before any teacher call.
    with pytest.raises(ValueError, match="mismatch"):
        ftp.collect(source, manifest, freeze, config, tmp_path / "bad", rate_limit_dir=tmp_path / "limit",
                    subset_views=[f"2={source}/subsets/k1"], teacher_factory=lambda *a: pytest.fail("teacher called"))
    # Tamper an explicit pool view: never silently intersect an unexpected task set.
    path = pool / "subsets/k1/candidates.json"
    bad = read(path)
    bad["tasks"] = []
    write_json(path, bad)
    with pytest.raises(ValueError, match="task_id"):
        sm.load_pool(path.parent, manifest, freeze)
    # Frozen manifest/selection hashes remain binding even if task IDs still match.
    bindings = read(pool / "identity.json")["subset_views"]
    write_json(source / "subsets/k1/manifest.json", {**support[1], "extra": "changed"})
    with pytest.raises(ValueError):
        materialize_views(selected, manifest, freeze, bindings)


def test_actual_task_mismatch_and_mixed_rules_refused(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path / "large", k=2)
    other, _, _, _, _ = setup_source(seed_bank, tmp_path / "other", k=3)
    collection, selection, _ = load_verified_collection(source, manifest, freeze)
    with pytest.raises(ValueError, match="mismatch"):
        bind_views(source, collection, selection, manifest, freeze, [f"3={other}"])
    materialize_subsets(other, manifest, freeze, [1])
    wrong = {**collection, "verified": collection["verified"][1:]}
    with pytest.raises(ValueError, match="task_id mismatch"):
        bind_views(source, wrong, selection, manifest, freeze, [f"1={other}/subsets/k1"])
    with pytest.raises(ValueError, match="OR"):
        bind_views(source, collection, selection, manifest, freeze, [f"1={other}"], subset_ks=[1])


def test_sharded_scores_equal_serial_and_merge_hash_checks(seed_bank, tmp_path):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=2)
    pool = sm.collect(source, manifest, freeze, config, tmp_path / "pool", rate_limit_dir=tmp_path / "limit",
                      env_factory=env, teacher_factory=factory([]))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scores = []
    def scorer(episode):
        scores.append(episode["episode_id"])
        return {"nll": int(digest(episode["episode_id"])[:4], 16) / 10000}
    serial = select(pool, snapshot, config, manifest, freeze, tmp_path / "serial", scorer=scorer)
    expected = read(serial / "smartad_selection.json")
    scores.clear()
    output = tmp_path / "parallel"
    def run(**kw):
        return select(pool, snapshot, config, manifest, freeze, output, scorer=scorer, shard_count=4, **kw)
    run(shard_index=1)
    assert len(scores) == 5 and not (output / "collection/manifest.json").exists()
    with pytest.raises(ValueError, match="Missing"):
        run(merge=True)
    for index in (3, 0, 2, 1):
        run(shard_index=index)
    assert len(scores) == 20
    actual = run(merge=True)
    assert read(actual / "smartad_selection.json") == expected
    assert read(actual / "manifest.json") == read(serial / "manifest.json")
    score_path = next((output / "scores").rglob("*.json"))
    record = read(score_path)
    record["score"]["nll"] += 0.1
    write_json(score_path, record)
    with pytest.raises(ValueError, match="hash/binding"):
        run(merge=True)


def test_legacy_identity_and_receipts_survive_code_upgrade(seed_bank, tmp_path, monkeypatch):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    for module, teacher_factory in ((ftp, stub_factory), (sm, factory)):
        calls = []
        output = tmp_path / module.__name__.split('.')[1]
        kwargs = dict(rate_limit_dir=tmp_path / "limit", env_factory=env, teacher_factory=teacher_factory(calls))
        monkeypatch.setattr(module, "code_hashes", lambda _: {"old.py": "old"})
        module.collect(source, manifest, freeze, config, output, **kwargs)
        before = (output / "identity.json").read_bytes(), len(calls)
        monkeypatch.setattr(module, "code_hashes", lambda _: {"new.py": "new"})
        module.collect(source, manifest, freeze, config, output, shard_count=7, shard_index=0, **kwargs)
        assert (output / "identity.json").read_bytes() == before[0] and len(calls) == before[1]
        assert len(list((output / "resume_versions").glob("*.json"))) == 1
        assert "subset_views" not in read(output / "identity.json")
        bad = dict(config, training={**config["training"], "new_setting": True})
        with pytest.raises(ValueError, match="identity changed"):
            module.collect(source, manifest, freeze, bad, output, **kwargs)


@pytest.mark.parametrize("method", ["adftp", "smartad"])
def test_launcher_arguments_and_dry_run(seed_bank, tmp_path, method):
    import yaml
    source, manifest, freeze, config, _ = setup_source(seed_bank, tmp_path, k=2)
    materialize_subsets(source, manifest, freeze, [1])
    for key, data in (("manifest", manifest), ("split_freeze", freeze)):
        config["paths"][key] = str(tmp_path / (key + ".json"))
        write_json(config["paths"][key], data)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "ENV_PYTHON": sys.executable, "PROJECT_ROOT": str(Path.cwd()),
           "TEACHER_RATE_LIMIT_DIR": str(tmp_path / "shared-limit")}
    output = tmp_path / "new-output"
    command = ["bash", f"scripts/rai_collect_{method}.sh", "--cell", "chain_k500_s0",
               "--source-view", str(source), "--subset-view", f"1={source}/subsets/k1",
               "--config", str(config_path), "--output-root", str(output), "--shards", "32", "--dry-run"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    cell = plan["cells"][0]
    assert cell["cell"] == "chain_k500_s0" and cell["source_view"] == str(source)
    assert len(cell["commands"]) == 32 and len(cell["subsets"]) == 1
    assert plan["rate_limit_dir"] == str(tmp_path / "shared-limit")
    assert all("--subset-view" in c and "--subset-k" not in c for c in cell["commands"])
    assert not output.exists() and not (tmp_path / "shared-limit").exists()
    for extra in (["--shards", "0"], ["--cell", "../escape"], ["--subset-view", "1=/bad"]):
        bad = subprocess.run(command + extra, env=env, capture_output=True, text=True)
        assert bad.returncode == 2 and not output.exists()


@pytest.mark.parametrize("merge", [False, True])
def test_hpg_array_and_cpu_merge_arguments(tmp_path, merge):
    fake = tmp_path / "python"
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                    'open(os.environ["ARGS"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = {**os.environ, "PROJECT_ROOT": str(Path.cwd()), "HPG_CONFIG": str(Path('configs/hpg.env').resolve()),
           "ENV_PYTHON": str(fake), "TRAIN_PYTHON": str(fake), "VLLM_PYTHON": str(fake),
           "MODEL_PATH": "base", "VERIFIED": "pool", "SMARTAD_OUT": str(tmp_path / "selected"),
           "SMARTAD_THROUGH": "select", "SMARTAD_SELECTION_SHARDS": "8", "SLURM_ARRAY_TASK_ID": "3",
           "SLURM_GPUS_ON_NODE": "0" if merge else "1", "SMARTAD_MERGE_SCORES": str(int(merge)),
           "SMARTAD_PATH_MAP": "/rai=/hpg", "ARGS": str(tmp_path / "args.json")}
    env.pop("SLURM_JOB_GPUS", None)
    result = subprocess.run(["bash", "scripts/hpg_smartad.slurm"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = read(tmp_path / "args.json")
    assert args[args.index("--shard-count") + 1] == "8"
    assert args[args.index("--shard-index") + 1] == "3"
    assert args[args.index("--path-map") + 1] == "/rai=/hpg"
    assert ("--merge-scores" in args) == merge


def test_v2_support_interface_when_kchain_is_integrated(seed_bank, tmp_path):
    import sed.seed_evidence as evidence
    if not hasattr(evidence, "materialize_full_train_views"):
        pytest.skip("stratified_nested_v2 loader belongs to wt/kchain; run integration with its src on pythonpath")
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, full=True)
    evidence.materialize_full_train_views(source, manifest, freeze, [2, 4], seed=0)
    verified = source / "subsets/k4"
    entries = [f"2={source}/subsets/k2"]
    kwargs = dict(rate_limit_dir=tmp_path / "limit", env_factory=env, subset_views=entries)
    root = ftp.collect(verified, manifest, freeze, config, tmp_path / "ftp", teacher_factory=stub_factory([]), **kwargs)
    view = evidence.load_verified_collection(root / "subsets/k2", manifest, freeze)[0]
    support = evidence.load_verified_collection(source / "subsets/k2", manifest, freeze)[0]
    assert task_ids(view) == task_ids(support)
    pool = sm.collect(verified, manifest, freeze, config, tmp_path / "pool", teacher_factory=factory([]), **kwargs)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    moved_source, moved_pool = tmp_path / "hpg-source", tmp_path / "hpg-pool"
    source.rename(moved_source)
    pool.rename(moved_pool)
    mapping = [f"{source}={moved_source}", f"{pool}={moved_pool}"]
    calls = []
    def score(_):
        calls.append(1)
        return {"nll": 1.0}
    for index in range(4):
        select(moved_pool, snapshot, config, manifest, freeze, tmp_path / "selected", scorer=score,
               path_map=mapping, shard_count=4, shard_index=index)
    selected = select(moved_pool, snapshot, config, manifest, freeze, tmp_path / "selected",
                      path_map=mapping, shard_count=4, merge=True)
    assert len(calls) == 40
    validate_selection(selected / "subsets/k2", manifest, freeze, snapshot)
    assert task_ids(evidence.load_verified_collection(selected / "subsets/k2", manifest, freeze)[0]) == task_ids(support)
    assert {t["task_id"] for t in sm.load_pool(moved_pool / "subsets/k2", manifest, freeze,
        resolver=sm.PathResolver(mapping))[0]["tasks"]} == task_ids(support)


def test_adftp_overlapping_shards_execute_each_task_once(seed_bank, tmp_path, monkeypatch):
    from sed.storage import JsonlLedger
    monkeypatch.setenv("SED_HASH_WORKERS", "1")
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path, k=2)
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(2)
    ledger = tmp_path / "calls.jsonl"
    def teacher_factory(*args):
        teacher = stub_factory([])(*args)
        create = teacher.teacher.client.responses.with_raw_response.create
        def logged(**kwargs):
            JsonlLedger(ledger).append({"task_id": args[2]["task_id"], "pid": os.getpid()})
            # Two tasks must be in flight before either can finish.
            from baselines.adftp.teacher import COT_PROMPT
            if kwargs["input"][0]["content"] == COT_PROMPT:
                barrier.wait(timeout=30)
            return create(**kwargs)
        teacher.teacher.client.responses.with_raw_response.create = logged
        return teacher
    processes = []
    output = tmp_path / "ftp"
    for index, count in ((0, 2), (1, 2), (0, 2), (0, 1)):
        process = ctx.Process(target=ftp.collect, args=(source, manifest, freeze, config, output),
            kwargs=dict(rate_limit_dir=tmp_path / "limit", env_factory=env, teacher_factory=teacher_factory,
                        shard_index=index, shard_count=count))
        process.start()
        processes.append(process)
    for process in processes:
        process.join(timeout=40)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 4 and len({r["pid"] for r in rows}) == 2
    assert load_verified_collection(output / "collection", manifest, freeze)[0]["api_attempts"] == 4


def test_old_adftp_launcher_keeps_cells_sources_and_subsets(tmp_path):
    fake = tmp_path / "python"
    ledger = tmp_path / "args.jsonl"
    fake.write_text('#!' + sys.executable + '\n' + r'''
import json, os, sys
from pathlib import Path
if sys.argv[1] != "-":
    with open(os.environ["ARGS"], "a") as f:
        f.write(json.dumps(sys.argv[1:]) + "\n")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / "identity.json").write_text("{}")
''')
    fake.chmod(0o755)
    sources = tmp_path / "sources"
    for cell in ("k100_s0", "k100_s1", "k100_s2", "full_train_r2"):
        write_json(sources / ("seeds_" + cell) / "manifest.json", {})
    env = {**os.environ, "ENV_PYTHON": str(fake), "PROJECT_ROOT": str(Path.cwd()),
           "COLLECTION_ROOT": str(sources), "ADFTP_ROOT": str(tmp_path / "outputs"),
           "LOG_ROOT": str(tmp_path / "logs"), "ARGS": str(ledger)}
    for flags in ([], ["--resume"]):
        result = subprocess.run(["bash", "scripts/rai_collect_adftp.sh", *flags], env=env, capture_output=True)
        assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(calls) == 8 and calls[:4] == calls[4:]
    assert [Path(r[r.index("--verified") + 1]).name for r in calls[:4]] == [
        "seeds_k100_s0", "seeds_k100_s1", "seeds_k100_s2", "seeds_full_train_r2"]
    for i, args in enumerate(calls[:4]):
        assert "--shard-count" not in args and "--subset-view" not in args
        assert args[args.index("--rate-limit-dir") + 1] == "data/adftp_rate_limit"
        assert ("--subset-k" in args) == (i < 3)
        if i < 3:
            assert args[args.index("--subset-k") + 1:] == ["5", "10", "20", "50"]
