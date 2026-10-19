"""Shared SAD/STeCa gate: exact pooled rule and offline report round trip."""
from copy import deepcopy
import json

import pytest

from sed.storage import digest, file_hash, write_json
from tools import check_full_serving as gate


@pytest.mark.parametrize("errors,passed", [
    ([.05], True), ([.0501], False),
    ([.51] * 2 + [0.] * 19, True),  # ceil(.05 * 21) permits two
    ([.51] * 6 + [0.] * 94, False),  # mean passes; outlier count fails
])
def test_pooled_thresholds(errors, passed):
    # Unequal prompt sizes must not turn this into a mean of prompt means.
    cases = [{"hf_logprobs": [0.], "vllm_logprobs": errors[:1]}]
    if len(errors) > 1:
        cases.append({"hf_logprobs": [0.] * (len(errors) - 1), "vllm_logprobs": errors[1:]})
    assert gate.compare(cases)["passed"] is passed


@pytest.mark.parametrize("cases", [[], [{"hf_logprobs": [], "vllm_logprobs": []}],
    [{"hf_logprobs": [float("inf")], "vllm_logprobs": [0.]}],
    [{"hf_logprobs": [0.], "vllm_logprobs": [0., 0.]}]])
def test_invalid_scores_fail(cases):
    with pytest.raises(ValueError):
        gate.compare(cases)


def test_reference_to_evaluation_report_round_trip(tmp_path, monkeypatch):
    from train import dataset
    config = {"models": {"student": {}}, "paths": {}}
    for name in ("manifest", "split_freeze"):
        path = tmp_path / (name + ".json")
        write_json(path, {})
        config["paths"][name] = str(path)
    checkpoint = tmp_path / "run/checkpoint"
    write_json(checkpoint / "training_state.json", {"final": True})
    write_json(checkpoint.parent / "provenance.json", {"training": "fixture"})
    verified = tmp_path / "verified"
    write_json(verified / "manifest.json", {})
    monkeypatch.setattr(gate, "load_config", lambda _: config)
    monkeypatch.setattr(gate, "validate_checkpoint", lambda *a, **k: {"final": True})
    monkeypatch.setattr(gate, "load_frozen", lambda *a: ({}, {}))
    from types import SimpleNamespace
    monkeypatch.setattr(dataset, "load_tokenizer", lambda _: SimpleNamespace(chat_template="fixture"))
    examples = [{"input_ids": [2, 7, 106], "prompt_tokens": 1, "packing": "turn",
        "label_provenance": {"task_id": "train/a", "turn_index": i, "training_eligible": True}}
        for i in range(8)]
    monkeypatch.setattr(dataset, "build_verified_dataset", lambda *a, **k: deepcopy(examples))
    captured = []
    def provenance(output, **kwargs):
        row = {"code_sha256": "fixture", **kwargs}
        captured.append(row)
        write_json(output / "provenance.json", row)
        return row
    # Inputs are Paths in production; keep the test provenance JSON serializable.
    def record(output, **kwargs):
        kwargs["inputs"] = list(map(str, kwargs["inputs"]))
        return provenance(output, **kwargs)
    monkeypatch.setattr(gate, "write_provenance", record)
    endpoint = "http://127.0.0.1:1234/v1"
    server = {"config": {"server": {"model_path": str(checkpoint.resolve()),
        "training_provenance_sha256": file_hash(checkpoint.parent / "provenance.json"),
        "endpoint": endpoint, "batch_invariant": True, "merged_model": True,
        "checkpoint_kind": "full_parameter", "served_model_name": "fixture-student"}}}
    server_path = tmp_path / "server.json"
    write_json(server_path, server)
    reference_path = tmp_path / "hf/reference.json"
    args = ["--config", "fixture", "--snapshot", "base", "--checkpoint", str(checkpoint),
        "--verified", str(verified), "--hf-reference", str(reference_path),
        "--endpoint", endpoint, "--server-provenance", str(server_path)]
    # Capture the expected identity without any model/GPU/network work.
    assert gate.main([*args, "--output", str(tmp_path / "identity")]) == 1
    prior = captured[-1]
    prompts = gate.fixed_prompts(examples)
    write_json(reference_path.parent / "provenance.json", prior)
    write_json(reference_path, {"identity": prior["details"], "provenance_sha256": digest(prior),
        "prompts": [{**row, "hf_logprobs": [-1., -.1]} for row in prompts]})
    calls = []
    def scores(client, endpoint, model, row):
        calls.append((endpoint, model))
        return [-1.01, -.1]
    monkeypatch.setattr(gate, "vllm_logprobs", scores)
    output = tmp_path / "comparison"
    assert gate.main([*args, "--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    gate.validate_report(report, server, endpoint)
    assert calls == [(endpoint, "fixture-student")] * 8
    assert report["passed"] and report["tokens"] == 16
    changed = deepcopy(report)
    changed["prompts"][0]["source"]["training_eligible"] = False
    with pytest.raises(ValueError):
        gate.validate_report(changed, server, endpoint)
    changed = deepcopy(report)
    changed["mean_absolute_error"] = 0.
    with pytest.raises(ValueError):
        gate.validate_report(changed, server, endpoint)
