"""Standalone snapshot/checkpoint receipts share collection cache semantics."""
import os
from pathlib import Path

import pytest

from sed import hash_receipts, seed_evidence
from sed.storage import file_hash


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("SED_HASH_CACHE", "1")
    hash_receipts._MEMO.clear()
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "weights.safetensors").write_bytes(b"fake weights")
    (root / "config.json").write_bytes(b"{}")
    return root


def count_bytes(monkeypatch):
    calls = []
    def counted(path):
        calls.append(Path(path))
        return file_hash(path)
    monkeypatch.setattr(seed_evidence, "file_hash", counted)
    return calls


def test_standalone_memory_and_persistent_receipts_read_no_artifact_bytes(snapshot, monkeypatch):
    calls = count_bytes(monkeypatch)
    expected = seed_evidence.artifact_hashes(snapshot)
    assert len(calls) == 2
    calls.clear()
    assert seed_evidence.artifact_hashes(snapshot) == expected and not calls
    hash_receipts._MEMO.clear()  # New process must reuse disk, not process memo.
    assert seed_evidence.artifact_hashes(snapshot) == expected and not calls
    assert expected == {p.name: file_hash(p) for p in snapshot.iterdir()}


@pytest.mark.parametrize("change", ["mtime", "size", "inode", "add", "remove"])
def test_standalone_invalidates_changed_artifact(snapshot, monkeypatch, change):
    seed_evidence.artifact_hashes(snapshot)
    path = snapshot / "weights.safetensors"
    info = path.stat()
    if change == "mtime":
        path.write_bytes(b"next weights")
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    elif change == "size":
        path.write_bytes(b"larger fake weights")
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    elif change == "inode":
        other = snapshot / "replacement"
        other.write_bytes(path.read_bytes())
        os.utime(other, ns=(info.st_atime_ns, info.st_mtime_ns))
        other.replace(path)
    elif change == "add":
        (snapshot / "new").write_bytes(b"new")
    else:
        path.unlink()
    hash_receipts._MEMO.clear()
    calls = count_bytes(monkeypatch)
    actual = seed_evidence.artifact_hashes(snapshot)
    assert len(calls) == (0 if change == "remove" else 1)
    assert actual == {p.name: file_hash(p) for p in snapshot.iterdir()}


def test_disabled_standalone_always_rehashes(snapshot, monkeypatch):
    expected = seed_evidence.artifact_hashes(snapshot)
    monkeypatch.setenv("SED_HASH_CACHE", "0")
    hash_receipts._MODES.clear()
    calls = count_bytes(monkeypatch)
    assert seed_evidence.artifact_hashes(snapshot) == seed_evidence.artifact_hashes(snapshot) == expected
    assert len(calls) == 4
    assert hash_receipts.cache_provenance()["observed_modes"] == ["full_rehash"]


def test_standalone_never_writes_cache_inside_snapshot(snapshot, monkeypatch):
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(snapshot / "receipts"))
    seed_evidence.artifact_hashes(snapshot)
    assert not (snapshot / "receipts").exists()


def test_checkpoint_training_state_comparison_unchanged(snapshot):
    from sed.storage import write_json
    expected = seed_evidence.artifact_hashes(snapshot)
    write_json(snapshot / "training_state.json", {"artifact_sha256": expected})
    actual = seed_evidence.artifact_hashes(snapshot)
    actual.pop("training_state.json")
    assert actual == expected
