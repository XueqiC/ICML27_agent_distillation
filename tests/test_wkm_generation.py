"""Offline WKM generation, merge and accounting regressions with real transport."""
import json
from types import SimpleNamespace

import httpx
import pytest

from baselines.common.method_data import read
from baselines.wkm.data import PromptStudent, merge, synthesize
from baselines.wkm.pipeline import update_disclosure
from baselines.wkm.policy import WKMStudent
from sed.storage import BlobStore, write_json
from students.vllm_student import StudentCallError
from test_wkm_steca import seed_bank, wkm_sources


def ledger(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def student(path, handler):
    class SynthesisStudent(PromptStudent):
        def sampling(self):
            return {"temperature": .5, "top_p": 1, "top_k": -1, "seed": 0}
    return SynthesisStudent(endpoint="http://fake/v1", model="experienced", revision="pin",
        enable_thinking=False, log_dir=path, transport=httpx.MockTransport(handler))


def response(body, text, finish):
    return httpx.Response(200, json={"model": body["model"], "choices": [{
        "message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": body["max_tokens"]}})


def disclose(path):
    update_disclosure(path, {"method": "wkm", "K": 5, "unique_instances": 5, "task_groups": 5})
    return read(path / "disclosure.json")


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_synthesis_request_merge_and_truncation_accounting(wkm_sources, tmp_path, finish):
    verified, explored, config = wkm_sources
    run = tmp_path / "run"
    attempt = run / "stages/synthesize/attempt-000"
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return response(body, "  task knowledge  " if body["max_tokens"] == 512 else "  state knowledge  ", finish)
    policy = student(attempt / "synthesis_calls", handler)
    try:
        rows = synthesize(verified, explored, attempt / "output", config, policy)
    finally:
        policy.close()
    assert len(rows) == 5
    expected = [(512, None), (128, ["\n"])] * 5
    assert [(r["max_tokens"], r.get("stop")) for r in requests] == expected
    assert all("stop" not in r for r in requests if r["max_tokens"] == 512)
    assert all((r["temperature"], r["top_p"], r["top_k"], r["seed"]) == (.5, 1, -1, 0) for r in requests)
    calls = [c for row in rows for c in row["calls"]]
    assert len(calls) == len(requests)
    blobs = BlobStore(attempt / "synthesis_calls/blobs")
    for call, request in zip(calls, requests):
        metadata = call["metadata"]
        assert metadata["finish_reason"] == finish
        assert metadata["prompt_tokens"] == 11
        assert metadata["completion_tokens"] == request["max_tokens"]
        assert blobs.get(metadata["request_ref"]) == request
        assert call["raw_output"] == ("task knowledge" if request["max_tokens"] == 512 else "state knowledge")
        assert blobs.get(metadata["raw_response_ref"])["content"] == "  " + call["raw_output"] + "  "
    merge(verified, explored, attempt / "output", tmp_path / "merged", config)
    assert all(r["task_knowledge"] == "task knowledge" and r["states"] == ["state knowledge"]
               for r in read(tmp_path / "merged/data.json"))
    report = disclose(run)
    truncated = len(calls) if finish == "length" else 0
    assert report["synthesis_attempts"][0]["truncated_outputs"] == truncated
    assert report["totals"]["synthesis_truncated_outputs"] == truncated
    assert report["totals"]["student_calls"] == len(calls)
    assert report["totals"]["prompt_tokens"] == 11 * len(calls)
    assert report["totals"]["completion_tokens"] == sum(r["max_tokens"] for r in requests)
    # Matching raw text and label must still be non-empty and have an allowed finish.
    for text, reason in [("", finish), (" \n ", finish), ("bad", "content_filter"), ("bad", None)]:
        rows[0]["states"][0] = text.strip()
        rows[0]["calls"][1].update(raw_output=text)
        rows[0]["calls"][1]["metadata"]["finish_reason"] = reason
        write_json(attempt / "output/knowledge.json", rows)
        with pytest.raises(ValueError, match="accepted non-empty"):
            merge(verified, explored, attempt / "output", tmp_path / "invalid-merge", config)


@pytest.mark.parametrize("cap", [512, 128], ids=["task", "state"])
@pytest.mark.parametrize("text,finish", [("", "stop"), (" \n ", "length"),
    ("rejected", "content_filter"), ("unknown", None), (None, "transport_error")])
def test_synthesis_and_world_failures_retain_evidence(wkm_sources, tmp_path, cap, text, finish):
    verified, explored, config = wkm_sources
    run = tmp_path / "run"
    attempt = run / "stages/synthesize/attempt-000"
    def handler(request):
        body = json.loads(request.content)
        if body["max_tokens"] == cap:
            if finish == "transport_error":
                raise httpx.ConnectError("SECRET_TRANSPORT_ERROR", request=request)
            return response(body, text, finish)
        return response(body, "task knowledge", "length")
    policy = student(attempt / "synthesis_calls", handler)
    try:
        with pytest.raises(StudentCallError if finish == "transport_error" else ValueError):
            synthesize(verified, explored, attempt / "output", config, policy)
    finally:
        policy.close()
    assert not (attempt / "output/knowledge.json").exists()
    calls = ledger(attempt / "synthesis_calls/calls.jsonl")
    if finish == "transport_error":
        assert calls[-1]["status"] == "failed" and calls[-1]["error_type"] == "ConnectError"
    else:
        assert calls[-1]["finish_reason"] == finish
        assert calls[-1]["prompt_tokens"] == 11 and calls[-1]["completion_tokens"] == cap
        raw = [p for p in (attempt / "output").glob("*.json") if p.name != "provenance.json"]
        assert len(raw) == 1 and read(raw[0])["calls"][-1]["raw_output"] == text.strip()
        assert BlobStore(attempt / "synthesis_calls/blobs").get(calls[-1]["raw_response_ref"])["content"] == text
    report = disclose(run)
    truncated = int(cap == 128) + int(finish == "length")
    assert report["totals"]["synthesis_truncated_outputs"] == truncated
    assert report["synthesis_attempts"][0]["truncated_outputs"] == truncated
    assert report["synthesis_attempts"][0]["unknown_usage"] == int(finish == "transport_error")
    # The same failures must stop inference before agent generation or scoring.
    world_dir = tmp_path / "inference"
    policy = WKMStudent(endpoint="http://fake/v1", model="agent", revision="pin", enable_thinking=False,
        knowledge_model="world", knowledge_base="unused", log_dir=world_dir,
        kb=SimpleNamespace(identity={"sha256": "stub"}), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(StudentCallError):
            policy.generate([], {"task_id": "train/a"}, {"observation": "room", "admissible_commands": ["look"]})
    finally:
        policy.close()
    assert not (world_dir / "agent/calls.jsonl").exists()
    assert not (world_dir / "decisions.jsonl").exists()
    world_calls = ledger(world_dir / "world/calls.jsonl")
    assert world_calls[-1]["status"] == ("failed" if finish == "transport_error" else "completed")
    assert all("SECRET_TRANSPORT_ERROR" not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


def test_disclosure_counts_truncation_across_retries(tmp_path):
    from sed.storage import JsonlLedger
    for attempt, reason in enumerate(["length", "stop", "length"]):
        log = JsonlLedger(tmp_path / f"stages/synthesize/attempt-{attempt:03}/synthesis_calls/calls.jsonl")
        log.append({"status": "started"})
        log.append({"status": "completed", "finish_reason": reason, "prompt_tokens": 7, "completion_tokens": 128})
    report = disclose(tmp_path)
    assert [a["truncated_outputs"] for a in report["synthesis_attempts"]] == [1, 0, 1]
    assert report["totals"]["synthesis_truncated_outputs"] == 2
    assert report["totals"]["completion_tokens"] == 384
