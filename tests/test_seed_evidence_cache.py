"""CPU-only receipt equivalence, invalidation, isolation and threat boundary."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sed import hash_receipts, seed_evidence
from sed.provenance import write_provenance
from sed.storage import file_hash, write_json
from test_full_train import full_bank
from test_full_train_views import source_bank


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setenv("SED_HASH_CACHE", "1")
    hash_receipts._MEMO.clear()
    hash_receipts._MODES.clear()
    yield
    hash_receipts._MEMO.clear()
    hash_receipts._MODES.clear()


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "collection"
    episode = root / "attempts/one/episode"
    (episode / "nested").mkdir(parents=True)
    (episode / "nested/data").write_bytes(b"evidence")
    (episode / "other").write_bytes(b"other")
    (episode / "ignored.lock").write_bytes(b"lock")
    for name in ("manifest", "selection", "identity"):
        write_json(root / (name + ".json"), {})
    return root, episode


def cached(tree):
    root, episode = tree
    with hash_receipts.collection_cache(root):
        return seed_evidence.artifact_hashes(episode)


def count_hashes(monkeypatch):
    calls = []
    def counted(path):
        calls.append(Path(path))
        return file_hash(path)
    monkeypatch.setattr(seed_evidence, "file_hash", counted)
    return calls


@pytest.mark.parametrize("workers", [1, 4])
def test_cache_and_disk_hits_equal_full_hash(tree, monkeypatch, workers):
    monkeypatch.setenv("SED_HASH_WORKERS", str(workers))
    monkeypatch.setenv("SED_HASH_CACHE", "0")
    expected = seed_evidence.artifact_hashes(tree[1])
    monkeypatch.setenv("SED_HASH_CACHE", "1")
    calls = count_hashes(monkeypatch)
    assert cached(tree) == expected and len(calls) == 2
    calls.clear()
    assert cached(tree) == expected and calls == []
    hash_receipts._MEMO.clear()  # Simulate the next process.
    assert cached(tree) == expected and calls == []
    receipts = list(hash_receipts.cache_directory().rglob("*.json"))
    assert len(receipts) == 1
    row = json.loads(receipts[0].read_text())["payload"]["files"][0]
    info = (tree[1] / row[0]).stat()
    assert row[1:4] == [info.st_size, info.st_mtime_ns, info.st_ino]
    assert row[4] == expected[row[0]]


def test_concurrent_loads_hash_once(tree, monkeypatch):
    calls = count_hashes(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: cached(tree), range(4)))
    assert all(value == values[0] for value in values)
    assert len(calls) == 2


def test_cached_walk_matches_full_hash_for_links_and_names(tree, monkeypatch):
    root, episode = tree
    (episode / "link-file").symlink_to("nested/data")
    (episode / "link-dir").symlink_to("nested", target_is_directory=True)
    (episode / "dangling").symlink_to("absent")
    (episode / ".hidden").write_bytes(b"hidden")
    (episode / "nested.txt").write_bytes(b"sort after nested/data")
    (episode / "directory.lock").mkdir()
    (episode / "directory.lock/keep").write_bytes(b"keep")
    (episode / "line\nbreak").write_bytes(b"newline")
    monkeypatch.setenv("SED_HASH_CACHE", "0")
    expected = seed_evidence.artifact_hashes(episode)
    monkeypatch.setenv("SED_HASH_CACHE", "1")
    assert list(cached(tree).items()) == list(expected.items())
    assert list(cached(tree).items()) == list(expected.items())


@pytest.mark.parametrize("change", ["size", "mtime", "inode", "add", "remove"])
def test_metadata_and_file_set_changes(tree, monkeypatch, change):
    before = cached(tree)
    episode = tree[1]
    target = episode / "nested/data"
    info = target.stat()
    if change == "size":
        target.write_bytes(b"longer evidence")
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    elif change == "mtime":
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    elif change == "inode":
        replacement = episode / "replacement"
        replacement.write_bytes(target.read_bytes())
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        replacement.replace(target)
        assert target.stat().st_ino != info.st_ino
    elif change == "add":
        (episode / "new").write_bytes(b"new")
    else:
        target.unlink()
    calls = count_hashes(monkeypatch)
    actual = cached(tree)
    assert len(calls) == (0 if change == "remove" else 1)
    assert actual == {str(p.relative_to(episode)): file_hash(p)
                      for p in sorted(episode.rglob("*")) if p.is_file() and not p.name.endswith(".lock")}
    if change in ("size", "add", "remove"):
        assert actual != before


def test_forged_mtime_limit_and_disabled_escape_hatch(tree, monkeypatch):
    before = cached(tree)
    target = tree[1] / "nested/data"
    info = target.stat()
    target.write_bytes(b"tampered")  # Same length, same inode, forged timestamp.
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert hash_receipts._signature(target.stat()) == hash_receipts._signature(info)
    assert cached(tree) == before  # Documented metadata-cache limitation.
    hash_receipts._MEMO.clear()
    assert cached(tree) == before  # Persistent receipts have the same limitation.
    monkeypatch.setenv("SED_HASH_CACHE", "0")
    assert cached(tree) != before


def test_disabled_bypasses_both_caches(tree, monkeypatch):
    expected = cached(tree)
    monkeypatch.setenv("SED_HASH_CACHE", "0")
    calls = count_hashes(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled path touched cache")
    monkeypatch.setattr(hash_receipts.HashReceipts, "hashes", forbidden)
    assert cached(tree) == cached(tree) == expected
    assert len(calls) == 4
    assert hash_receipts.cache_provenance()["mode"] == "full_rehash"


@pytest.mark.parametrize("anchor", ["manifest", "selection", "identity"])
def test_memo_is_bound_to_all_collection_anchors(tree, monkeypatch, anchor):
    # An unusable receipt destination proves hits come from memory alone.
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(tree[0] / "forbidden-cache"))
    expected = cached(tree)
    calls = count_hashes(monkeypatch)
    assert cached(tree) == expected and calls == []
    write_json(tree[0] / (anchor + ".json"), {"changed": True})
    assert cached(tree) == expected and len(calls) == 2
    assert not (tree[0] / "forbidden-cache").exists()


@pytest.mark.parametrize("corruption", ["truncated", "checksum", "schema", "unreadable"])
def test_bad_receipt_rehashes(tree, monkeypatch, corruption):
    expected = cached(tree)
    receipt = next(hash_receipts.cache_directory().rglob("*.json"))
    if corruption == "truncated":
        receipt.write_text('{"payload":')
    elif corruption == "checksum":
        value = json.loads(receipt.read_text())
        value["payload"]["files"][0][-1] = "0" * 64
        write_json(receipt, value)
    elif corruption == "schema":
        write_json(receipt, ["not a receipt"])
    else:
        original = Path.read_bytes
        def read_bytes(path):
            if path == receipt:
                raise PermissionError("unreadable receipt")
            return original(path)
        monkeypatch.setattr(Path, "read_bytes", read_bytes)
    hash_receipts._MEMO.clear()
    calls = count_hashes(monkeypatch)
    assert cached(tree) == expected and len(calls) == 2


def test_collection_cache_symlink_is_never_written(tree, tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    alias.symlink_to(tree[0], target_is_directory=True)
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(alias / "receipts"))
    cached(tree)
    assert not (tree[0] / "receipts").exists()


def test_unwritable_cache_falls_back_to_memory(tree, tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(blocked / "receipts"))
    expected = cached(tree)
    calls = count_hashes(monkeypatch)
    assert cached(tree) == expected and calls == []
    assert "process_memo_only" in hash_receipts.cache_provenance()["observed_modes"]


def test_changing_during_hash_does_not_publish_receipt(tree, monkeypatch):
    def racing_hash(path):
        sha = file_hash(path)
        path.write_bytes(path.read_bytes() + b"changed")
        return sha
    monkeypatch.setattr(seed_evidence, "file_hash", racing_hash)
    with pytest.raises(ValueError, match="changed while hashing"):
        cached(tree)
    assert not list(hash_receipts.cache_directory().rglob("*.json"))
    assert not hash_receipts._MEMO


def test_evidence_read_errors_propagate(tree, monkeypatch):
    def unreadable(path):
        raise PermissionError("unreadable evidence")
    monkeypatch.setattr(seed_evidence, "file_hash", unreadable)
    with pytest.raises(PermissionError, match="unreadable evidence"):
        cached(tree)
    assert not hash_receipts._MEMO


def test_two_views_reuse_source_hashes_and_reject_unselected_tampering(source_bank, monkeypatch):
    root, _, manifest, freeze, _, _ = source_bank
    seed_evidence.materialize_full_train_views(root, manifest, freeze, [1, 2])
    hash_receipts._MEMO.clear()
    first = seed_evidence.load_verified_collection(root / "subsets/k1", manifest, freeze)
    calls = count_hashes(monkeypatch)
    seed_evidence.load_verified_collection(root / "subsets/k2", manifest, freeze)
    # _subset_values deliberately rehashes receipt.json identity references;
    # episode/teacher artifacts must all be metadata hits.
    assert all(path.name == "receipt.json" for path in calls)
    collection = json.loads((root / "manifest.json").read_text())
    selected = {r["task_id"] for r in first[0]["verified"]}
    receipt = next(r for r in collection["verified"] if r["task_id"] not in selected)
    target = root / receipt["episode_dir"] / "episode.json"
    target.write_text(target.read_text() + " ")
    from sed.validation_cache import clear_validation_cache
    clear_validation_cache()  # A new process/audit revalidates immutable evidence.
    with pytest.raises(ValueError, match="Attempt evidence changed"):
        seed_evidence.load_verified_collection(root / "subsets/k1", manifest, freeze)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_provenance_records_mode(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("SED_HASH_CACHE", enabled)
    value = write_provenance(tmp_path / "run", command="test", config={})
    assert value["hash_cache"]["mode"] == (
        "full_rehash" if enabled == "0" else "metadata_receipts_and_process_memo")
