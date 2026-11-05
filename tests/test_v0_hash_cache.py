"""V0 raw hash receipts and distributed validation, offline with CPU fakes."""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from repair.v2_inputs import InputFiles, Relocation
from sed import hash_receipts, raw_hash_cache, seed_evidence, storage
from tools import train_v0_arm as trainer, v0_data
from test_v0_training import corpus, seed_corpus


@pytest.fixture(autouse=True)
def fresh_raw_cache(monkeypatch):
    monkeypatch.setenv("SED_HASH_CACHE", "1")
    raw_hash_cache._MEMO.clear()
    yield
    raw_hash_cache._MEMO.clear()


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "evidence/data"
    path.parent.mkdir()
    path.write_bytes(b"original")
    return path


def count_bytes(monkeypatch):
    calls = []
    original = storage._byte_file_hash
    def counted(path):
        calls.append(Path(path))
        return original(path)
    monkeypatch.setattr(storage, "_byte_file_hash", counted)
    return calls


def test_memo_deduplicates_aliases_nested_scopes_and_threads(target, tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    calls = count_bytes(monkeypatch)
    with raw_hash_cache.raw_input_cache(workers=8) as cache:
        values = list(cache.map(storage.file_hash, [target, alias] * 16))
        with raw_hash_cache.raw_input_cache() as nested:
            assert nested is cache
            assert storage.file_hash(target) == values[0]
        assert len(set(values)) == 1 and calls == [target]
        assert cache.report()["memo_hits"] == 32
    # Separate invocations in one interpreter also share the metadata memo.
    with raw_hash_cache.raw_input_cache() as cache:
        assert storage.file_hash(alias) == values[0]
        assert cache.report()["memo_hits"] == 1
    assert calls == [target] and storage.active_file_hash_cache() is None


def test_receipts_survive_process_memo_reset(target, monkeypatch):
    calls = count_bytes(monkeypatch)
    with raw_hash_cache.raw_input_cache() as cold:
        expected = storage.file_hash(target)
    assert cold.report()["misses"] == 1
    raw_hash_cache._MEMO.clear()
    with raw_hash_cache.raw_input_cache() as warm:
        assert storage.file_hash(target) == expected
        assert warm.report()["receipt_hits"] == 1
        assert warm.report()["byte_hashes"] == 0
    assert calls == [target]


@pytest.mark.parametrize("collection_first", [False, True])
def test_collection_and_raw_reader_share_one_byte_hash(target, monkeypatch, collection_first):
    root = target.parent.parent / "collection"
    root.mkdir()
    for name in ("manifest.json", "selection.json", "identity.json"):
        (root / name).write_text("{}")
    calls = count_bytes(monkeypatch)
    with raw_hash_cache.raw_input_cache():
        def collection():
            with hash_receipts.collection_cache(root):
                return hash_receipts.active_cache().hashes(target.parent, storage.file_hash)[target.name]
        first = collection() if collection_first else storage.file_hash(target)
        second = storage.file_hash(target) if collection_first else collection()
        assert first == second
    assert calls.count(target) == 1


@pytest.mark.parametrize("change", ["size", "mtime", "inode"])
@pytest.mark.parametrize("persistent", [False, True])
def test_metadata_invalidates_both_caches(target, monkeypatch, change, persistent):
    with raw_hash_cache.raw_input_cache():
        storage.file_hash(target)
    info = target.stat()
    if change == "size":
        target.write_bytes(b"longer content")
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    elif change == "mtime":
        target.write_bytes(b"modified")
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    else:
        replacement = target.with_name("replacement")
        replacement.write_bytes(b"replaced")
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        replacement.replace(target)
        assert target.stat().st_ino != info.st_ino
    if persistent:
        raw_hash_cache._MEMO.clear()
    calls = count_bytes(monkeypatch)
    with raw_hash_cache.raw_input_cache() as cache:
        assert storage.file_hash(target) == storage._byte_file_hash(target)
        assert cache.report()["misses"] == 1
    assert len(calls) == 2  # One cache miss and the independent comparison.


@pytest.mark.parametrize("audit", ["environment", "flag"])
def test_full_rehash_bypasses_memory_and_disk(target, monkeypatch, audit):
    with raw_hash_cache.raw_input_cache():
        before = storage.file_hash(target)
    info = target.stat()
    target.write_bytes(b"tampered")
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    calls = count_bytes(monkeypatch)
    if audit == "environment":
        monkeypatch.setenv("SED_HASH_CACHE", "0")
    def forbidden(*args):
        raise AssertionError("Audit touched receipts")
    monkeypatch.setattr(raw_hash_cache.RawFileHashes, "_read", forbidden)
    with raw_hash_cache.raw_input_cache(full_rehash=audit == "flag") as cache:
        assert storage.file_hash(target) == storage.file_hash(target) != before
        assert cache.report()["mode"] == "full_rehash"
        assert cache.report()["byte_hashes"] == 2
    assert len(calls) == 2
    assert os.environ["SED_HASH_CACHE"] == ("0" if audit == "environment" else "1")


@pytest.mark.parametrize("corruption", ["truncated", "checksum", "schema", "missing"])
def test_disposable_receipt_corruption_rehashes(target, corruption, monkeypatch):
    with raw_hash_cache.raw_input_cache() as cache:
        expected = storage.file_hash(target)
    receipt = cache.receipt_path(target.parent)
    if corruption == "truncated":
        receipt.write_text('{"payload":')
    elif corruption == "checksum":
        value = json.loads(receipt.read_text())
        value["payload"]["files"][0][-1] = "0" * 64
        receipt.write_text(json.dumps(value))
    elif corruption == "schema":
        receipt.write_text("[]")
    else:
        receipt.unlink()
    raw_hash_cache._MEMO.clear()
    calls = count_bytes(monkeypatch)
    with raw_hash_cache.raw_input_cache() as cache:
        assert storage.file_hash(target) == expected
        assert cache.report()["misses"] == 1
    assert calls == [target]


def test_parallel_order_and_worker_bound(target, monkeypatch):
    paths = []
    slots = set()
    i = 0
    while len(paths) < 16:
        p = target.parent / str(i)
        i += 1
        slot = hash(p.resolve()) % len(raw_hash_cache._FILE_LOCKS)
        if slot in slots:
            continue
        slots.add(slot)
        p.write_bytes(str(i).encode())
        paths.append(p)
    original = storage._byte_file_hash
    expected = [original(p) for p in paths]
    barrier = Barrier(4)
    lock, running, peak = Lock(), 0, 0
    def blocked(path):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        barrier.wait(timeout=10)
        value = original(path)
        with lock:
            running -= 1
        return value
    monkeypatch.setattr(storage, "_byte_file_hash", blocked)
    with raw_hash_cache.raw_input_cache(workers=4) as cache:
        assert list(cache.map(storage.file_hash, paths)) == expected
        assert cache.report()["byte_hashes"] == len(paths)
    assert peak == 4
    raw_hash_cache._MEMO.clear()
    def forbidden(*args, **kwargs):
        pytest.fail("Warm receipt hit submitted a hashing future")
    monkeypatch.setattr(raw_hash_cache.ThreadPoolExecutor, "submit", forbidden)
    with raw_hash_cache.raw_input_cache(workers=4) as cache:
        assert list(cache.map(storage.file_hash, paths, parallel_if=lambda p: not cache.likely_cached(p))) == expected
        assert cache.report()["byte_hashes"] == 0


def test_tree_worker_cap_concurrency_and_order(tmp_path, monkeypatch):
    cache = hash_receipts.HashReceipts(tmp_path, ())
    monkeypatch.setattr(seed_evidence, "active_cache", lambda: cache)
    paths = [tmp_path / str(i) for i in range(16)]
    expected = [{"file": str(path)} for path in paths]
    baseline = None
    for configured, workers, effective in [(None, 16, 2), ("8", 16, 8), ("8", 4, 4), ("1", 16, 1)]:
        monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
        if configured is None:
            monkeypatch.delenv("SED_TREE_WORKERS", raising=False)
        else:
            monkeypatch.setenv("SED_TREE_WORKERS", configured)
        barrier = Barrier(effective)
        lock, running, peak = Lock(), 0, 0
        def read_tree(path, hash_file):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            try:
                barrier.wait(timeout=10)
                return {"file": str(path)}
            finally:
                with lock:
                    running -= 1
        monkeypatch.setattr(cache, "hashes", read_tree)
        with seed_evidence.artifact_hashes_batch(paths) as readers:
            actual = [read() for read in readers]
        assert peak == effective and running == 0
        assert actual == expected
        if baseline is None:
            baseline = actual
        assert actual == baseline
        assert hash_receipts.cache_provenance()["tree_worker_cap"] == (2 if configured is None else int(configured))


@pytest.mark.parametrize("value", ["", "invalid", "1.5", "0", "-1"])
def test_invalid_tree_worker_cap(tmp_path, monkeypatch, value):
    monkeypatch.setenv("SED_TREE_WORKERS", value)
    cache = hash_receipts.HashReceipts(tmp_path, ())
    with pytest.raises(ValueError):
        with cache.batch([], 16, storage.file_hash):
            pytest.fail("Invalid tree workers accepted")
    with pytest.raises(ValueError):
        hash_receipts.cache_provenance()


def test_worker_configuration_and_audit_scope_restore(monkeypatch):
    monkeypatch.setenv("SED_HASH_WORKERS", "3")
    with raw_hash_cache.raw_input_cache() as cache:
        assert cache.workers == 3
    with pytest.raises(ValueError, match="positive"):
        with raw_hash_cache.raw_input_cache(full_rehash=True, workers=0):
            pytest.fail("Invalid workers accepted")
    assert os.environ["SED_HASH_CACHE"] == "1"
    assert storage.active_file_hash_cache() is None


def test_hashing_mutation_never_publishes_receipt(target, monkeypatch):
    original = storage._byte_file_hash
    def changed(path):
        sha = original(path)
        path.write_bytes(b"changed during hashing")
        return sha
    monkeypatch.setattr(storage, "_byte_file_hash", changed)
    with pytest.raises(ValueError, match="changed while hashing"):
        with raw_hash_cache.raw_input_cache():
            storage.file_hash(target)
    assert not raw_hash_cache._MEMO
    assert not list(hash_receipts.cache_directory().rglob("*.json"))


def test_cache_cannot_write_inside_protected_inputs(target, monkeypatch):
    directory = target.parent / "cache"
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(directory))
    with raw_hash_cache.raw_input_cache() as cache:
        cache.protect(target.parent)
        storage.file_hash(target)
        assert cache.report()["mode"] == "process_memo_only"
    assert not directory.exists()


def test_receipt_shard_symlink_cannot_write_into_input(target, tmp_path):
    with raw_hash_cache.raw_input_cache() as cache:
        cache.protect(target.parent)
        receipt = cache.receipt_path(target.parent)
        receipt.parent.parent.mkdir(parents=True)
        receipt.parent.symlink_to(target.parent, target_is_directory=True)
        storage.file_hash(target)
    assert sorted(p.name for p in target.parent.iterdir()) == ["data"]


def test_inputfiles_rechecks_metadata_and_all_seals(target):
    files = InputFiles(Relocation(target.parent, ()))
    sha = storage.file_hash(target)
    files.add(target, sha)
    with raw_hash_cache.raw_input_cache():
        files.verify()
        target.write_bytes(b"changed input")
        with pytest.raises(ValueError, match="sha256 mismatch"):
            files.verify()


def test_directory_resolution_is_fresh_for_each_verification(tmp_path):
    evidence, dataset = tmp_path / "evidence", tmp_path / "dataset"
    folder = evidence / "nested"
    folder.mkdir(parents=True)
    target = folder / "data"
    target.write_bytes(b"sealed")
    files = InputFiles(Relocation(dataset, [f"/old/evidence={evidence}"]))
    files.add(target, storage.file_hash(target))
    with raw_hash_cache.raw_input_cache():
        files.verify()
        outside = tmp_path / "outside"
        folder.rename(outside)
        folder.symlink_to(outside, target_is_directory=True)
        # Bytes/inode/mtime are unchanged; directory symlink confinement still fails.
        with pytest.raises(ValueError, match="symlink escape"):
            files.verify()


def test_batch_resolution_matches_pathlib_and_audit_bypasses_it(target, tmp_path):
    link = tmp_path / "linked"
    link.symlink_to(target.parent, target_is_directory=True)
    paths = [target, link / target.name, link / "missing", link / "../linked/data"]
    relocation = Relocation(tmp_path / "dataset")
    with raw_hash_cache.raw_input_cache(), relocation.resolution_batch():
        assert [relocation.local(p) for p in paths] == [p.resolve() for p in paths]
    assert relocation._directory_resolution is None
    with raw_hash_cache.raw_input_cache(full_rehash=True), relocation.resolution_batch():
        assert relocation._directory_resolution is None


@pytest.mark.parametrize("mutation", ["whitespace", "oracle"])
def test_complete_prefix_render_check_rejects_earlier_bad_turns(corpus, mutation):
    from sed.react import build_messages
    row = deepcopy(corpus["rows"][0])
    visible = json.loads(row["messages"][1]["content"])
    reply = row["messages"][-1]["content"]
    row["messages"] = build_messages([{"visible": visible, "reply": reply}] * 3, {}, visible)
    row["messages"].append({"role": "assistant", "content": reply})
    row["supervised_assistant_indices"] = [len(row["messages"]) - 1]
    v0_data.validate_row(row)
    bad = dict(visible)
    if mutation == "oracle":
        bad["extra.expert_plan"] = ["FORBIDDEN"]
    row["messages"][1]["content"] = json.dumps(bad, indent=2)
    with pytest.raises(ValueError, match="prompt contract|whitelist"):
        v0_data.validate_row(row)


def arguments(c):
    return ["--dataset", str(c["root"]), "--theta0", str(c["theta"]),
            "--snapshot", str(c["snapshot"]), "--config", str(c["config_path"]),
            "--synthetic-manifest", str(c["synthetic"])]


@pytest.mark.parametrize("warm", [False, True])
def test_compact_inventory_rejects_symlink_escape_even_with_same_inode(corpus, tmp_path, monkeypatch, warm):
    c = corpus
    source = json.loads((c["theta"].parent / "provenance.json").read_text())
    view = Path(source["config"]["baseline_training"]["verified"])
    support = view.parent.parent
    manifest = json.loads((support / "manifest.json").read_text())
    episode = support / manifest["attempts"][0]["episode_dir"]
    target = episode / "episode.json"
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(tmp_path / "new-receipts"))
    def read():
        return v0_data.read_dataset(c["root"], theta0=c["theta"], snapshot=c["snapshot"], config=c["config"],
            synthetic_manifest=c["synthetic"], relocate=[f"/old/collection={support}"])
    if warm:
        read()
    outside = tmp_path / "outside-evidence"
    os.link(target, outside)
    target.unlink()
    target.symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="collection validation failed|symlink escape"):
            read()
    finally:
        # The seed fixture shares its immutable support bank between copies.
        target.unlink()
        os.link(outside, target)
        outside.unlink()


def test_replay_inspector_collection_is_not_validated_again(corpus, monkeypatch):
    from repair import v2_inputs
    c = corpus
    splits = json.loads(c["splits"].read_text())
    del splits["sets"]["K20"]
    storage.write_json(c["splits"], splits)
    for root in (c["root"] / "replay", c["root"]):
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["splits_sha256"] = storage.file_hash(c["splits"])
        manifest["inputs"][str(c["splits"])] = manifest["splits_sha256"]
        if root == c["root"]:
            manifest["replay_manifest_sha256"] = storage.file_hash(c["root"] / "replay/manifest.json")
        storage.write_json(path, manifest)
    original, calls = v2_inputs.training_view_inputs, []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(v2_inputs, "training_view_inputs", counted)
    monkeypatch.setattr(v0_data, "training_view_inputs", counted)
    v0_data.read_dataset(c["root"], theta0=c["theta"], snapshot=c["snapshot"], config=c["config"],
                        synthetic_manifest=c["synthetic"])
    assert calls == [1]


def test_cli_reports_hits_full_rehash_and_run_provenance(corpus, tmp_path, monkeypatch):
    # The fixture keeps some sealed files at tmp_path itself; put receipts
    # outside that input directory, just as on the cluster.
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(tmp_path.parent / (tmp_path.name + "-receipts")))
    args = arguments(corpus)
    reports = []
    for label, extra in (("cold", []), ("warm", []), ("full", ["--full-rehash"])):
        raw_hash_cache._MEMO.clear()
        output = tmp_path / f"{label}.json"
        assert trainer.main(["inputs", "--check", *args, *extra, "--output", str(output)]) == 0
        reports.append(json.loads(output.read_text()))
    cold, warm, full = [r["hash_cache"]["raw_files"] for r in reports]
    assert cold["misses"] > 0 and warm["receipt_hits"] > 0
    assert warm["byte_hashes"] == 0
    assert full["mode"] == "full_rehash" and full["memo_hits"] == full["receipt_hits"] == 0
    for key in ("rows", "buckets", "dataset_sha256", "episode_weights_sha256", "trainable_parameter_count"):
        assert reports[0][key] == reports[1][key] == reports[2][key]
    raw_hash_cache._MEMO.clear()
    output = tmp_path / "run"
    assert trainer.main(["run", *args, "--updates", "1", "--checkpoints", "1", "--dry-run",
                         "--output", str(output)]) == 0
    prov = json.loads((output / "provenance.json").read_text())
    assert prov["hash_cache"]["raw_files"]["receipt_hits"] > 0
    assert prov["details"]["input_validation"]["rank"] == 0


class FakeProcessGroup:
    def __init__(self):
        self.message = None
        self.calls = []

    def broadcast(self, message, *, src, group):
        assert src == 0 and group is self
        self.calls.append(group)
        if message[0] is not None:
            # Match object-collective serialization, including InputFiles/Paths.
            self.message = pickle.dumps(message[0])
        else:
            message[0] = pickle.loads(self.message)


def test_complete_verified_dataset_broadcast_never_reopens_inputs(corpus, monkeypatch):
    c = corpus
    args = SimpleNamespace(command="run", updates=1, weights=None, trainable_modules=None,
        dataset=str(c["root"]), theta0=str(c["theta"]), snapshot=str(c["snapshot"]),
        config=str(c["config_path"]), synthetic_manifest=str(c["synthetic"]),
        splits=None, repair_splits=None, dev140=None, relocate=[], full_rehash=False, hash_workers=2)
    group = FakeProcessGroup()
    with raw_hash_cache.raw_input_cache():
        first = trainer.validate_inputs(args, rank=0, world=2, group=group, broadcast=group.broadcast)
    def forbidden(*args, **kwargs):
        pytest.fail("Nonzero rank tried to validate input files")
    monkeypatch.setattr(trainer, "load_args", forbidden)
    monkeypatch.setattr(trainer.v2, "continuation_recipe", forbidden)
    monkeypatch.setattr(trainer, "load_choices", forbidden)
    monkeypatch.setattr(storage, "_byte_file_hash", forbidden)
    second = trainer.validate_inputs(args, rank=1, world=2, group=group, broadcast=group.broadcast)
    assert first[0]["rows"] == second[0]["rows"]
    assert first[0]["dataset_sha256"] == second[0]["dataset_sha256"]
    assert first[1:] == second[1:]
    assert second[0]["inspection"]["files"].all_hashes_verified


@pytest.mark.parametrize("failure", [False, True])
def test_only_rank_zero_validates_and_failure_is_fatal(monkeypatch, failure):
    group, calls = FakeProcessGroup(), []
    args = SimpleNamespace(command="run", updates=1, weights=None, trainable_modules=None, config="config")
    def load(_):
        calls.append("validation")
        if failure:
            raise ValueError("bad input")
        files = InputFiles(Relocation(Path.cwd(), ()))
        return {"K": 20, "theta0": "theta", "config": {}, "snapshot": "snapshot", "dataset_sha256": "verified",
                "inspection": {"files": files}}
    monkeypatch.setattr(trainer, "load_args", load)
    monkeypatch.setattr(trainer.v2, "continuation_recipe", lambda *args, **kwargs: {})
    monkeypatch.setattr(trainer, "load_choices", lambda *args, **kwargs: ({}, None, []))
    if failure:
        with pytest.raises(ValueError, match="bad input"):
            trainer.validate_inputs(args, rank=0, world=2, group=group, broadcast=group.broadcast)
        with pytest.raises(RuntimeError, match="Rank 0 startup validation failed"):
            trainer.validate_inputs(args, rank=1, world=2, group=group, broadcast=group.broadcast)
    else:
        first = trainer.validate_inputs(args, rank=0, world=2, group=group, broadcast=group.broadcast)
        second = trainer.validate_inputs(args, rank=1, world=2, group=group, broadcast=group.broadcast)
        assert first[0]["dataset_sha256"] == second[0]["dataset_sha256"] == "verified"
        assert second[0]["input_validation"]["rank"] == 0
    assert calls == ["validation"] and len(group.calls) == 2
