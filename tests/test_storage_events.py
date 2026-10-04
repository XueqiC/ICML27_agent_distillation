from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields, replace
import json

import pytest

from sed.events import Event, EventWriter
from sed.storage import BlobStore, JsonlLedger


def make_event(writer):
    ref = writer.blobs.put({"long": "observation" * 1000})
    return Event("run", "episode", "event", "train", "family", "task", [], "audit_only",
                 "revision", None, "environment", "scripted", ref, ref, "look", "look",
                 ref, ref, "real_environment", "unknown_until_audit", 0, 0, 1,
                 0.01, False, False, "ok", None)


def test_event_schema_and_durable_blobs(tmp_path):
    writer = EventWriter(tmp_path)
    event = make_event(writer)
    writer.append(event)
    row = json.loads((tmp_path / "events.jsonl").read_text())
    expected = "run_id episode_id event_id split task_family task_id parent_seed_ids mechanism_id student_revision teacher_provider_id environment_revision policy_condition history_ref visible_context_ref action_raw action_parsed observation_ref verification_ref feedback_source new_environment_information teacher_tokens_in teacher_tokens_out environment_steps latency_seconds terminated truncated status error_type".split()
    expected += ["action_outcome", "inadmissible", "policy_metadata_ref"]
    assert set(row) == set(expected) == {f.name for f in fields(Event)}
    assert writer.blobs.get(row["observation_ref"])["long"].startswith("observation")
    assert writer.blobs.put({"long": "observation" * 1000}) == event.observation_ref
    with pytest.raises(ValueError, match="Duplicate"):
        writer.append(event)
    with pytest.raises(TypeError):
        Event(**{k: v for k, v in asdict(event).items() if k != "task_id"})


@pytest.mark.parametrize("change", [
    {"teacher_tokens_in": -1}, {"environment_steps": True}, {"terminated": 1},
    {"latency_seconds": float("nan")}, {"latency_seconds": -1}, {"parent_seed_ids": "seed"},
    {"split": "valid_unseen"}, {"history_ref": "a/path"}, {"status": "error"},
    {"feedback_source": "teacher_guess_is_ground_truth"},
])
def test_invalid_schema_rejected(tmp_path, change):
    writer = EventWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.append(replace(make_event(writer), **change))
    assert not (tmp_path / "events.jsonl").exists()


def test_dangling_reference_rejected(tmp_path):
    writer = EventWriter(tmp_path)
    with pytest.raises(FileNotFoundError):
        writer.append(replace(make_event(writer), history_ref="sha256:" + "0" * 64))


def test_atomic_append_crash_preserves_prefix(tmp_path, monkeypatch):
    from sed import storage
    log = JsonlLedger(tmp_path / "events.jsonl")
    log.append({"i": 1})
    original = log.path.read_bytes()
    def crash(*args):
        raise OSError("simulated interruption before publish")
    with monkeypatch.context() as scoped:
        scoped.setattr(storage.os, "replace", crash)
        with pytest.raises(OSError):
            log.append({"i": 2})
    assert log.path.read_bytes() == original
    log.append({"i": 2})
    assert [json.loads(s)["i"] for s in log.path.read_text().splitlines()] == [1, 2]


def test_concurrent_appends_preserve_every_record(tmp_path):
    path = tmp_path / "events.jsonl"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: JsonlLedger(path).append({"i": i}), range(20)))
    assert sorted(json.loads(s)["i"] for s in path.read_text().splitlines()) == list(range(20))


def test_corruption_is_not_silently_repaired(tmp_path):
    log = JsonlLedger(tmp_path / "bad.jsonl")
    log.path.write_bytes(b'{"old":1}\n{"torn":')
    before = log.path.read_bytes()
    with pytest.raises(ValueError, match="Damaged"):
        log.append({"new": 2})
    assert log.path.read_bytes() == before
    store = BlobStore(tmp_path / "blobs")
    ref = store.put({"a": 1})
    next((tmp_path / "blobs").rglob("*.json")).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.get(ref)
