"""Parallel verification preserves evidence bytes, error order and rank handoff."""
import builtins
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import pickle
import threading

import pytest

from baselines.common import explore
from baselines.common.fsdp_train import rank_zero_data
from baselines.ipr import rollouts
from sed import seed_evidence
from sed.manifest import build_manifest, freeze_splits
from sed.seed_selection import select_seeds
from sed.storage import canonical_bytes, digest, file_hash, write_json
from teachers.collect_seeds import collect
from test_stage2b_seeds import seed_bank, factories
from test_stage3a import listing_for
from test_stage3c import fake_env, FakeStudent, fake_worker


def old_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and not p.name.endswith(".lock")}


@pytest.fixture
def hash_tree(tmp_path):
    root = tmp_path / "tree"
    for name in ("nested/deeper", ".hidden", "empty", "directory.lock"):
        (root / name).mkdir(parents=True)
    for name, content in (("z", b"last"), ("nested/deeper/a", b"nested"), (".dot", b"dot"),
                          (".hidden/file", b"hidden"), ("empty_file", b""),
                          ("large", bytes(range(256)) * 8193), ("ignore.lock", b"lock"),
                          (".hidden/ignore.lock", b"lock"), ("directory.lock/keep", b"keep")):
        (root / name).write_bytes(content)
    (root / "link_file").symlink_to("nested/deeper/a")
    (root / "link_dir").symlink_to("nested", target_is_directory=True)
    (root / "link.lock").symlink_to("large")
    (root / "dangling").symlink_to("missing")
    return root


@pytest.mark.parametrize("workers", [1, 16])
def test_artifact_hashes_reference(hash_tree, monkeypatch, workers):
    monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
    expected = old_hashes(hash_tree)
    actual = seed_evidence.artifact_hashes(hash_tree)
    assert list(actual.items()) == list(expected.items())
    assert canonical_bytes(actual) == canonical_bytes(expected)
    assert digest(actual) == digest(expected)
    assert "link_file" in actual and "directory.lock/keep" in actual
    assert not any(name.startswith("link_dir/") for name in actual)


@pytest.mark.parametrize("workers", [1, 16])
def test_artifact_hashes_many_order(hash_tree, monkeypatch, workers):
    monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
    directories = [hash_tree / "nested", hash_tree, hash_tree / "empty", hash_tree / ".hidden",
                   hash_tree / "absent", hash_tree / "nested"]
    expected = [old_hashes(d) for d in directories]
    actual = seed_evidence.artifact_hashes_many(iter(directories))
    assert [list(row.items()) for row in actual] == [list(row.items()) for row in expected]
    assert canonical_bytes(actual) == canonical_bytes(expected)
    assert seed_evidence.artifact_hashes_many([]) == []


@pytest.mark.parametrize("workers", [1, 16])
@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_hash_io_failure_propagates(tmp_path, monkeypatch, workers, failure):
    monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
    victim = tmp_path / "victim"
    victim.write_bytes(b"evidence")
    original_open = builtins.open
    def open_file(path, *args, **kwargs):
        if Path(path) == victim:
            # Occurs after is_file(), exactly where the old implementation raises.
            if failure == "missing":
                victim.unlink()
            else:
                raise PermissionError("unreadable evidence")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", open_file)
    with pytest.raises(FileNotFoundError if failure == "missing" else PermissionError):
        seed_evidence.artifact_hashes(tmp_path)


@pytest.mark.parametrize("workers", [1, 16])
def test_file_checks_run_in_workers(tmp_path, monkeypatch, workers):
    monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
    for i in range(4):
        (tmp_path / str(i)).write_bytes(b"evidence")
    parent = threading.get_ident()
    seen, checks = set(), set()
    barrier = threading.Barrier(4)
    original_stat, original_hash = Path.is_file, seed_evidence.file_hash
    def is_file(path):
        checks.add(threading.get_ident())
        return original_stat(path)
    def hash_file(path):
        seen.add(threading.get_ident())
        if workers > 1:
            barrier.wait(timeout=10)
        return original_hash(path)
    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(seed_evidence, "file_hash", hash_file)
    assert len(seed_evidence.artifact_hashes(tmp_path)) == 4
    assert checks == seen
    if workers == 1:
        assert seen == {parent}
    else:
        assert parent not in seen and len(seen) == 4


def test_default_pool_is_shared_and_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv("SED_HASH_WORKERS", raising=False)
    pools = []
    class Pool(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            pools.append(kwargs["max_workers"])
            super().__init__(**kwargs)
    monkeypatch.setattr(seed_evidence, "ThreadPoolExecutor", Pool)
    directories = []
    for i in range(20):
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / "data").write_text(str(i))
        directories.append(directory)
    assert seed_evidence.artifact_hashes_many(directories) == [old_hashes(d) for d in directories]
    assert pools == [16]


def test_batch_defers_path_and_io_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("SED_HASH_WORKERS", "16")
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    (bad / "data").write_text("evidence")
    def fail_io(path):
        raise PermissionError("later receipt")
    def fail_path():
        raise ValueError("invalid later path")
    monkeypatch.setattr(seed_evidence, "file_hash", fail_io)
    with seed_evidence.artifact_hashes_batch([good, bad, fail_path]) as readers:
        assert next(readers)() == {}
        with pytest.raises(PermissionError, match="later receipt"):
            next(readers)()
        with pytest.raises(ValueError, match="invalid later path"):
            next(readers)()


@pytest.fixture
def common_evidence(seed_bank, tmp_path):
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 40
    config["paths"].update(manifest=str(tmp_path / "manifest.json"), split_freeze=str(tmp_path / "freeze.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    tasks = tmp_path / "tasks.json"
    write_json(tasks, listing_for(manifest, freeze, freeze["seed_pool_candidates"][:2]))
    output = tmp_path / "common"
    def worker(job):
        return explore.explore_one(job, env_factory=fake_env, policy_factory=FakeStudent)
    explore.run_exploration(config=config, tasks_path=tasks, output=output,
        spec={"enable_thinking": False, "max_tokens": 1024}, decoding="greedy", workers=1, worker_fn=worker)
    return output, lambda: explore.checked_rollouts(output, manifest, freeze), "episode_dir", "episode_hashes", "Exploration evidence changed"


@pytest.fixture
def ipr_evidence(seed_bank, tmp_path):
    root, _, _, config = seed_bank
    for path in root.rglob("game.tw-pddl"):
        path.write_text(json.dumps({"solvable": True, "win_at": 3}))
    manifest, freeze = build_manifest(root), freeze_splits(build_manifest(root))
    config["environment"]["max_episode_steps"] = 40
    config["paths"].update(manifest=str(tmp_path / "manifest.json"), split_freeze=str(tmp_path / "freeze.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    _, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collect(select_seeds(manifest, freeze, k=1), manifest, freeze, config, verified,
            env_factory=lambda task: fake_env(config, task), teacher_factory=teacher)
    server = tmp_path / "server.json"
    write_json(server, {"config": {"server": {"batch_invariant": True, "model_path": "sft",
                                            "training_provenance_sha256": "sft-hash"}}})
    output = tmp_path / "ipr"
    rollouts.run_rollouts(config=config, verified=verified, kind="explore", iteration=1, output=output,
        spec={"enable_thinking": False, "max_tokens": 1024,
              "model": config["models"]["student"]["requested_repo"],
              "revision": config["models"]["student"]["revision"]},
        server_provenance=server, workers=1, worker_fn=fake_worker)
    return output, lambda: rollouts.checked_rollouts(output, config), "directory", "artifact_sha256", "IPR rollout evidence changed"


@pytest.mark.parametrize("fixture", ["common_evidence", "ipr_evidence"])
@pytest.mark.parametrize("workers", [1, 16])
def test_checked_rollouts_intact_tamper_and_first_error(request, monkeypatch, fixture, workers):
    monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
    output, checked, path_field, hash_field, error = request.getfixturevalue(fixture)
    intact = checked()
    assert len(intact) >= 2
    monkeypatch.setenv("SED_HASH_WORKERS", "1")
    assert canonical_bytes(checked()) == canonical_bytes(intact)
    monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
    receipts = json.loads((output / "receipts.json").read_text())
    first, second = [output / r[path_field] for r in receipts[:2]]
    target = first / ("episode/episode.json" if path_field == "directory" else "episode.json")
    target.write_text("{}")
    with pytest.raises(ValueError, match=error):
        checked()
    # A later worker's I/O failure must never hide the first receipt's mismatch.
    original_hash = seed_evidence.file_hash
    def hash_file(path):
        if path.is_relative_to(second):
            raise PermissionError("later receipt unreadable")
        return original_hash(path)
    monkeypatch.setattr(seed_evidence, "file_hash", hash_file)
    with pytest.raises(ValueError, match=error):
        checked()
    # Nor may eagerly resolving a malformed later receipt hide that mismatch.
    del receipts[1][path_field]
    write_json(output / "receipts.json", receipts)
    with pytest.raises(ValueError, match=error):
        checked()


class Broadcast:
    """CPU object collective, using pickle as broadcast_object_list does."""
    def __init__(self):
        self.condition = threading.Condition()
        self.payload = None
        self.waiters = 0
        self.peers_waiting = threading.Event()

    def __call__(self, message, *, src, group):
        assert src == 0 and group is self
        with self.condition:
            if message[0] is not None:
                self.payload = pickle.dumps(message)
                self.condition.notify_all()
            else:
                self.waiters += 1
                if self.waiters == 3:
                    self.peers_waiting.set()
                assert self.condition.wait_for(lambda: self.payload is not None, timeout=10)
                message[:] = pickle.loads(self.payload)


@pytest.mark.parametrize("failure", [False, True])
def test_rank_zero_handoff_waits_and_propagates(failure):
    group = Broadcast()
    release = threading.Event()
    original = ValueError("original validation error")
    examples = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "messages": ["é"],
                 "label_provenance": {"source": "trusted_teacher_seed"}}]
    computed = []
    def compute():
        computed.append(0)
        assert release.wait(timeout=10)
        if failure:
            raise original
        return examples
    def forbidden():
        pytest.fail("A nonzero rank attempted evidence validation")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(rank_zero_data, rank, 4, compute if rank == 0 else forbidden,
                               group, broadcast=group) for rank in range(4)]
        try:
            assert group.peers_waiting.wait(timeout=10)
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        if failure:
            assert futures[0].exception(timeout=10) is original
            assert pickle.loads(group.payload) == [(False, None)]
            for future in futures[1:]:
                with pytest.raises(RuntimeError, match="Rank 0 startup validation failed"):
                    future.result(timeout=10)
        else:
            for future in futures:
                result = future.result(timeout=10)
                assert canonical_bytes(result) == canonical_bytes(examples)
                assert digest(result) == digest(examples)
    assert computed == [0]


def test_rank_zero_single_process():
    expected = [{"input_ids": [1, 2], "labels": [-100, 2]}]
    def forbidden(*args, **kwargs):
        pytest.fail("Single-process startup needs no collective")
    assert rank_zero_data(0, 1, lambda: expected, None, broadcast=forbidden) is expected
    error = ValueError("original")
    def fail():
        raise error
    with pytest.raises(ValueError) as caught:
        rank_zero_data(0, 1, fail, None, broadcast=forbidden)
    assert caught.value is error


@pytest.mark.parametrize("objective", ["sft", "accumulated_sft", "dpo", "ipr"])
def test_main_validates_only_rank_zero(tmp_path, monkeypatch, objective):
    """Exercise main through both handoffs, stopping before model allocation."""
    import sys
    from types import SimpleNamespace
    from baselines.common import fsdp_train as train
    from baselines.eto import pairs as eto_pairs
    from baselines.ipr import pairs as ipr_pairs
    from baselines.agentgym import data as agentgym
    from sed.config import load_config

    calls, groups, wire = [], [], []
    current = {"rank": 0, "broadcast": 0}
    group = object()
    def new_group(**kwargs):
        assert calls[-1] == (current["rank"], "init")
        assert kwargs["backend"] == "gloo" and kwargs["timeout"].total_seconds() >= 4 * 3600
        groups.append(current["rank"])
        return group
    def broadcast(message, *, src, group):
        assert src == 0 and group is verification_group
        index = current["broadcast"]
        if current["rank"] == 0:
            wire.append(pickle.dumps(message))
        else:
            message[:] = pickle.loads(wire[index])
        current["broadcast"] += 1
    verification_group = group
    dist = SimpleNamespace(
        init_process_group=lambda *a, **k: calls.append((current["rank"], "init")),
        new_group=new_group, broadcast_object_list=broadcast,
        barrier=lambda: calls.append((current["rank"], "barrier")), destroy_process_group=lambda: None)
    torch = SimpleNamespace(distributed=dist, device=lambda *a: "cuda",
        cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda _: None),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "torch.distributed.device_mesh", SimpleNamespace(init_device_mesh=lambda *a: None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=lambda _: None))
    config = load_config()
    config["environment"]["max_episode_steps"] = 40
    config["training"].update(thinking_mode="off", student_step_token_cap=1024)
    expected = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "label_provenance": {"task_id": "seed"}}]
    def counted(name, result):
        def run(*args, **kwargs):
            calls.append((current["rank"], name))
            return deepcopy(result)
        return run
    monkeypatch.setattr(train, "load_config", counted("config", config))
    monkeypatch.setattr(train, "validate_checkpoint", counted("checkpoint", {}))
    monkeypatch.setattr(train, "load_frozen", counted("frozen", ({}, {})))
    monkeypatch.setattr(train, "load_tokenizer", counted("tokenizer", SimpleNamespace(chat_template="template")))
    monkeypatch.setattr(train, "load_verified_collection", counted("collection", ({"verified": [{}]}, {}, tmp_path)))
    monkeypatch.setattr(train, "checked_episodes", counted("episodes", [{}]))
    monkeypatch.setattr(train, "conversation", counted("conversation", expected[0]))
    monkeypatch.setattr(agentgym, "training_examples", counted("accumulated", expected))
    for module in (eto_pairs, ipr_pairs):
        monkeypatch.setattr(module, "load_pair_data", counted("pairs", [{}]))
    monkeypatch.setattr(train, "preference_conversation", counted("encode", expected[0]))
    monkeypatch.setattr(ipr_pairs, "preference_conversation", counted("encode", expected[0]))
    monkeypatch.setattr(train, "snapshot_metadata", counted("metadata", {}))
    monkeypatch.setattr(train, "artifact_hashes", counted("hashes", {"shard": "unchanged"}))
    provenance_configs = []
    def provenance(*args, **kwargs):
        provenance_configs.append(deepcopy(kwargs["config"]))
        calls.append((current["rank"], "provenance"))
        return {}
    monkeypatch.setattr(train, "write_provenance", provenance)
    class TrainingReached(Exception):
        pass
    def prepare(*args, **kwargs):
        assert current["broadcast"] == 2
        raise TrainingReached()
    monkeypatch.setattr(train, "prepare_model", prepare)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write_json(snapshot / "training_state.json", {})
    write_json(tmp_path / "provenance.json", {})
    pair_root = tmp_path / "pairs"
    pair_root.mkdir()
    stages = {name: str(tmp_path / name) for name in ("expert_mc", "exploration")}
    write_json(pair_root / "provenance.json", {"details": stages})
    for path in stages.values():
        Path(path).mkdir()
        write_json(Path(path) / "provenance.json", {"config": {"server_provenance": {"config": {"server": {
            "model_path": str(snapshot), "training_provenance_sha256": file_hash(tmp_path / "provenance.json")}}}}})
    argv = ["--objective", "sft" if objective == "accumulated_sft" else objective,
            "--snapshot", str(snapshot), "--base-snapshot", str(snapshot), "--output", str(tmp_path / "out"),
            "--learning-rate", ".001", "--scheduler", "cosine", "--warmup-ratio", "0", "--original-batch", "8"]
    if objective in {"dpo", "ipr"}:
        argv += ["--reference", str(snapshot), "--pairs", str(pair_root / "pairs.json")]
    elif objective == "accumulated_sft":
        argv += ["--sft-data", "dataset.json"]
    else:
        argv += ["--verified", "verified"]
    monkeypatch.setenv("WORLD_SIZE", "4")
    for rank in range(4):
        current.update(rank=rank, broadcast=0)
        monkeypatch.setenv("RANK", str(rank))
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        with pytest.raises(TrainingReached):
            train.main(argv)
    assert groups == [0, 1, 2, 3]
    assert all(rank == 0 for rank, name in calls if name not in {"init", "barrier"})
    assert calls.count((0, "checkpoint")) == (2 if objective in {"dpo", "ipr"} else 0)
    assert calls.count((0, "pairs")) == int(objective in {"dpo", "ipr"})
    assert calls.count((0, "collection")) == int(objective == "sft")
    assert calls.count((0, "episodes")) == int(objective == "sft")
    assert calls.count((0, "accumulated")) == int(objective == "accumulated_sft")
    assert provenance_configs[0]["baseline_training"]["examples_sha256"] == digest(expected)
    handed_off = pickle.loads(wire[0])[0][1][0]
    assert canonical_bytes(handed_off) == canonical_bytes(expected)
