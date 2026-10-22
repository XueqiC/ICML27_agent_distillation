"""Seal integrity checks use no models, environment execution, or network."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.common import prepared
from sed.config import load_config
from sed.storage import digest, write_json
from sed.validation_cache import anchors


@pytest.fixture
def sealed(tmp_path):
    config = load_config()
    for name in ("manifest", "split_freeze"):
        config["paths"][name] = str(tmp_path / (name + ".json"))
        write_json(config["paths"][name], {})
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    snapshot = tmp_path / "snapshot"
    write_json(snapshot / "tokenizer_config.json", {"chat_template": "fake template"})
    (snapshot / "weights.safetensors").write_bytes(b"fake weights")
    bindings = {}
    for name in ("view", "parent", "support"):
        root = tmp_path / name
        for filename in ("manifest.json", "identity.json", "selection.json"):
            write_json(root / filename, {"synthetic": name})
        bindings[str(root)] = anchors(root)
    output = tmp_path / "training"
    collection = {"K": 1, "unique_instances": 1, "task_groups": 1}
    identity = {"method": "kang", "config_sha256": digest(config),
                "recipe": {"epochs": 2}, "code_sha256": prepared.code_hashes(prepared.ROOT),
                "verified": str(tmp_path / "view"), "snapshot": str(snapshot),
                "snapshot_artifact_sha256": prepared.artifact_hashes(snapshot), "gpus": 1,
                "collection_sha256": digest(collection)}
    write_json(output / "identity.json", identity)
    write_json(output / "tasks.json", {})
    args = SimpleNamespace(config=str(config_path), pipeline_output=str(output),
        verified=identity["verified"], snapshot=str(snapshot), base_snapshot=str(snapshot),
        output=str(tmp_path / "prepare-attempt"), epochs=2)
    examples = [{"input_ids": [1, 2], "labels": [-100, 2], "weights": [0., 1.5],
                 "label_provenance": {"source": "synthetic TRAIN"}}]
    data = (config, collection, examples, "fake template")
    ref = prepared.write_prepared(args, data, bindings)
    return args, identity, data, ref


def test_seal_round_trip_and_attempt_output_independent(sealed):
    args, identity, data, ref = sealed
    args.output += "-train"
    loaded, used = prepared.load_prepared(args)
    assert loaded == data and used == ref
    seal = prepared.read(ref["path"])["seal"]
    assert len(seal["input_collections"]) == 3
    assert seal["tokenizer_sha256"] == digest(seal["tokenizer"])


@pytest.mark.parametrize("tamper", ["examples", "seal", "missing", "manifest", "identity", "selection",
                                   "parent_manifest", "support_identity", "code", "config", "recipe",
                                   "pipeline_identity", "snapshot", "tokenizer", "tasks", "split"])
def test_seal_refuses_every_changed_binding(sealed, monkeypatch, tamper):
    args, identity, data, ref = sealed
    path = Path(ref["path"])
    if tamper == "missing":
        path.unlink()
    elif tamper == "recipe":
        args.epochs += 1
    elif tamper == "code":
        monkeypatch.setattr(prepared, "code_hashes", lambda _: {"changed.py": "changed"})
    else:
        target = {
            "examples": path.with_name("examples.json"), "seal": path,
            "manifest": Path(args.verified) / "manifest.json",
            "identity": Path(args.verified) / "identity.json",
            "selection": Path(args.verified) / "selection.json",
            "parent_manifest": Path(args.verified).parent / "parent/manifest.json",
            "support_identity": Path(args.verified).parent / "support/identity.json",
            "config": Path(args.config), "pipeline_identity": Path(args.pipeline_output) / "identity.json",
            "snapshot": Path(args.snapshot) / "weights.safetensors",
            "tokenizer": Path(args.snapshot) / "tokenizer_config.json",
            "tasks": Path(args.pipeline_output) / "tasks.json",
            "split": Path(data[0]["paths"]["split_freeze"]),
        }[tamper]
        # Whitespace still changes a byte binding; pipeline identity is semantic.
        if tamper in {"pipeline_identity", "config", "seal"}:
            value = prepared.read(target)
            value["changed"] = True
            if tamper == "seal":
                value["seal"]["examples_count"] += 1
            write_json(target, value)
        else:
            target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(prepared.PreparedDatasetError, match="Run the CPU --mode prepare"):
        prepared.load_prepared(args)


def test_missing_seal_does_not_call_full_validation(sealed, monkeypatch):
    args, identity, data, ref = sealed
    Path(ref["path"]).unlink()
    monkeypatch.setattr(prepared, "load_verified_collection", lambda *a, **kw: pytest.fail("fallback"))
    with pytest.raises(prepared.PreparedDatasetError, match="Run the CPU --mode prepare"):
        prepared.verify_prepared(args.pipeline_output, identity, data[0])
    args.pipeline_output = None
    with pytest.raises(prepared.PreparedDatasetError, match="missing --pipeline-output"):
        prepared.load_prepared(args)
