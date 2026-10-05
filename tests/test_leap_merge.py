"""CPU merge-policy regressions with fake model loading and no network."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from baselines.leap import merge as leap_merge
from sed.config import load_config
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, write_json
from train import merge_lora


@pytest.fixture
def merge_case(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    def make(reference, in_memory, standalone):
        base, adapter = tmp_path / "base", tmp_path / "training/adapter"
        base.mkdir()
        adapter.mkdir(parents=True)
        write_json(base / "config.json", {"model_type": "gemma4_unified"})
        (adapter / "adapter_model.safetensors").write_bytes(b"fake adapter")
        tensors = [torch.tensor(values, dtype=torch.float32)
                   for values in (reference, in_memory, standalone)]

        def save(output, *, safe_serialization):
            assert safe_serialization
            write_json(output / "config.json", {"model_type": "gemma4_unified", "source": "merged"})
            write_json(output / "generation_config.json", {"source": "merged"})
            (output / "model.safetensors").write_bytes(b"fake merged weights")

        merged = NS(logits=tensors[1], save_pretrained=save)

        def merge(*, safe_merge):
            assert safe_merge
            return merged

        adapted = NS(logits=tensors[0], merge_and_unload=merge)

        def from_pretrained(model, path, *, local_files_only, is_trainable):
            assert Path(path) == adapter and local_files_only and not is_trainable
            return adapted

        def load(path, *, device, dtype):
            assert device == "cpu"
            return NS(logits=tensors[0] if Path(path) == base else tensors[2])

        monkeypatch.setitem(sys.modules, "peft", NS(PeftModel=NS(from_pretrained=from_pretrained)))
        monkeypatch.setattr(merge_lora, "load_student", load)
        monkeypatch.setattr(merge_lora, "next_token_logits", lambda model, ids: model.logits.clone())
        return NS(base=base, adapter=adapter, output=tmp_path / "merged")

    return make


def run_merge(case, **kwargs):
    return merge_lora.merge_adapter(case.base, case.adapter, case.output, [1, 2],
                                   device="cpu", **kwargs)


def read_report(case):
    return json.loads((case.output / "consistency.json").read_text())


@pytest.mark.parametrize("policy", [merge_lora.LEAP_MERGE_POLICY, "strict_allclose"])
@pytest.mark.parametrize("with_tokenizer", [False, True])
def test_merge_preserves_base_assets_before_reload_and_hashing(merge_case, monkeypatch, policy, with_tokenizer):
    case = merge_case([0, 1, 3], [0, 1, 3], [0, 1, 3])
    assets = {
        "processor_config.json": b'{"processor_class": "FakeProcessor"}\n',
        "tokenizer_config.json": b'{"source": "base tokenizer"}\n',
        "tokenizer.json": b'{"source": "base vocabulary"}\n',
        "chat_template.jinja": b"base template\n",
        "alternate_template.jinja2": b"alternate template\n",
        "tokenizer.model": b"\x00base tokenizer model\xff",
        "vocab.txt": b"base vocabulary\n",
    }
    excluded = ["weights.index.json", "weights.index.txt", "model.safetensors", "weights.bin", "README.md"]
    for name, value in assets.items():
        (case.base / name).write_bytes(value)
    for name in excluded:
        (case.base / name).write_bytes(b"base bytes must not be copied")
    write_json(case.base / "generation_config.json", {"source": "base"})
    (case.base / "directory.json").mkdir()
    (case.base / "directory.json/nested.json").write_text("{}")

    tokenizer_assets = {name: b"tokenizer saved " + name.encode() for name in (
        "tokenizer_config.json", "tokenizer.json", "chat_template.jinja")} if with_tokenizer else {}

    def save_tokenizer(output):
        for name, value in tokenizer_assets.items():
            (output / name).write_bytes(value)

    expected = {**assets, **tokenizer_assets}
    copied = sorted(assets.keys() - tokenizer_assets.keys())
    reloads = []
    load = merge_lora.load_student

    def checked_load(path, **kwargs):
        if Path(path) == case.output:
            reloads.append(path)
            for name, value in expected.items():
                assert (case.output / name).read_bytes() == value
            assert json.loads((case.output / "config.json").read_text())["source"] == "merged"
            assert json.loads((case.output / "generation_config.json").read_text())["source"] == "merged"
            assert (case.output / "model.safetensors").read_bytes() == b"fake merged weights"
            assert not (case.output / "directory.json").exists()
            assert all(not (case.output / name).exists() for name in excluded if name != "model.safetensors")
        return load(path, **kwargs)

    monkeypatch.setattr(merge_lora, "load_student", checked_load)
    report = run_merge(case, policy=policy, tokenizer=NS(save_pretrained=save_tokenizer) if with_tokenizer else None)
    assert len(reloads) == 1 and report["passed"]
    assert read_report(case) == report
    assert report["copied_base_assets"] == copied
    provenance = json.loads((case.output / "provenance.json").read_text())
    assert provenance["details"]["copied_base_assets"] == copied
    hashes = artifact_hashes(case.output)
    hashes.pop("consistency.json")
    assert report["merged_sha256"] == hashes
    assert hashes["processor_config.json"] == file_hash(case.base / "processor_config.json")


def test_leap_accepts_bf16_rounding_and_retains_strict_failure(merge_case):
    case = merge_case([0, 1, 3], [0.5, 1.25, 3.75], [0.5, 1.25, 3.75])
    report = run_merge(case, policy=merge_lora.LEAP_MERGE_POLICY)
    assert read_report(case) == report
    assert report["passed"] and report["policy"] == "leap_official_bf16_merge"
    assert report["all_logits_finite"] and report["argmax_equal"]
    assert report["strict_allclose_passed"] is False
    assert report["reload_consistency"]["exact"]
    assert report["reload_consistency"]["max_absolute_error"] == 0
    for check in report["checks"].values():
        assert check["passed"] is False
        assert check["max_absolute_error"] == 0.75
        assert check["mean_absolute_error"] == 0.5
    provenance = json.loads((case.output / "provenance.json").read_text())
    assert provenance["config"]["merge"]["policy"] == report["policy"]


def test_other_callers_default_to_strict_allclose(merge_case):
    case = merge_case([0, 1, 3], [0.5, 1.25, 3.75], [0.5, 1.25, 3.75])
    with pytest.raises(ValueError, match="exceed tolerance; checkpoint must not be served"):
        run_merge(case)
    report = read_report(case)
    assert report["policy"] == "strict_allclose"
    assert report["passed"] is report["strict_allclose_passed"] is False


@pytest.mark.parametrize("policy,accepted", [(merge_lora.LEAP_MERGE_POLICY, False), ("strict_allclose", True)])
def test_argmax_gate_is_leap_only_even_when_strict_allclose_passes(merge_case, policy, accepted):
    case = merge_case([0, 0.01], [0.02, 0.01], [0.02, 0.01])
    if accepted:
        run_merge(case, policy=policy)
    else:
        with pytest.raises(ValueError, match="checkpoint must not be served"):
            run_merge(case, policy=policy)
    report = read_report(case)
    assert report["passed"] is accepted
    assert report["strict_allclose_passed"] and report["reload_consistency"]["passed"]
    assert report["argmax_equal"] is False


@pytest.mark.parametrize("policy", [merge_lora.LEAP_MERGE_POLICY, "strict_allclose"])
@pytest.mark.parametrize("location", [0, 1, 2], ids=["reference", "in_memory", "reloaded"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_logits_rejected_with_durable_report(merge_case, policy, location, value):
    values = [[0, 2], [0, 2], [0, 2]]
    values[location][0] = value
    case = merge_case(*values)
    with pytest.raises(ValueError, match="checkpoint must not be served"):
        run_merge(case, policy=policy)
    report = read_report(case)
    assert report["passed"] is report["all_logits_finite"] is report["strict_allclose_passed"] is False
    failed = [check for check in report["checks"].values() if not check["all_logits_finite"]]
    assert failed and all(check["max_absolute_error"] is None for check in failed)
    # A rejection remains valid JSON instead of failing while serializing NaN/Inf.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("delta,accepted", [(0, True), (5e-7, True), (1e-6, True), (2e-6, False)])
def test_leap_reload_absolute_tolerance(merge_case, delta, accepted):
    case = merge_case([0, 2], [0, 2], [delta, 2])
    if accepted:
        run_merge(case, policy=merge_lora.LEAP_MERGE_POLICY)
    else:
        with pytest.raises(ValueError, match="checkpoint must not be served"):
            run_merge(case, policy=merge_lora.LEAP_MERGE_POLICY)
    report = read_report(case)
    assert report["passed"] is accepted
    assert report["strict_allclose_passed"] and report["argmax_equal"]
    reload = report["reload_consistency"]
    assert reload["passed"] is accepted and reload["exact"] is (delta == 0)
    assert reload["atol"] == 1e-6 and reload["rtol"] == 0
    assert reload["max_absolute_error"] == pytest.approx(delta)


def test_unknown_merge_policy_fails_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="Unknown merge consistency policy"):
        merge_lora.merge_adapter("base", "adapter", tmp_path / "output", [1], policy="typo")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("variant", ["privileged", "zero"])
@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("failure", [None, "argmax", "nonfinite", "reload"])
def test_leap_publication_records_policy_and_never_publishes_rejections(merge_case, monkeypatch, variant, chained, failure):
    values = [[0, 1, 3], [0.5, 1.25, 3.75], [0.5, 1.25, 3.75]]
    if failure == "argmax":
        values[1] = values[2] = [5, 1.25, 3.75]
    elif failure == "nonfinite":
        values[2][0] = float("inf")
    elif failure == "reload":
        values[2][0] += 1e-4
    case = merge_case(*values)
    write_json(case.base / "processor_config.json", {"processor_class": "FakeProcessor"})
    if chained:
        for name in ("consistency.json", "provenance.json", "training_state.json"):
            write_json(case.base / name, {"previous_run": True})
    config = load_config()
    training = {"config": {"models": config["models"],
        "sft": {"base_snapshot": merge_lora.snapshot_metadata(case.base)},
        "leap_training": {"snapshot_artifact_sha256": artifact_hashes(case.base)}}}
    write_json(case.adapter.parent / "provenance.json", training)
    write_json(case.adapter / "training_state.json", {
        "identity_sha256": digest(training), "adaptation": "leap_lora",
        "artifact_sha256": artifact_hashes(case.adapter), "leap_iteration": int(chained),
        "leap_variant": variant, "privileged_teacher": variant == "privileged"})
    monkeypatch.setattr(leap_merge, "load_tokenizer", lambda _: NS(
        apply_chat_template=lambda *a, **kw: [1, 2], save_pretrained=lambda path: None))
    monkeypatch.setattr(leap_merge, "merge_adapter", lambda *a, **kw:
        merge_lora.merge_adapter(*a, device="cpu", **kw))
    argv = ["--snapshot", str(case.base), "--adapter", str(case.adapter), "--output", str(case.output)]
    if failure:
        with pytest.raises(ValueError, match="checkpoint must not be served"):
            leap_merge.main(argv)
        assert not (case.output / "checkpoint").exists()
        assert not (case.output / "result.json").exists()
        assert (case.output / ".checkpoint.partial/consistency.json").is_file()
        assert not (case.output / ".checkpoint.partial/training_state.json").exists()
        return
    leap_merge.main(argv)
    checkpoint = case.output / "checkpoint"
    state = json.loads((checkpoint / "training_state.json").read_text())
    assert state == json.loads((case.output / "result.json").read_text())
    assert state["merge_passed"] and state["strict_allclose_passed"] is False
    assert state["policy"] == "leap_official_bf16_merge"
    assert state["leap_variant"] == variant
    assert state["checkpoint_kind"] == "merged_lora"
    hashes = artifact_hashes(checkpoint)
    hashes.pop("training_state.json")
    assert state["artifact_sha256"] == hashes
    assert hashes["processor_config.json"] == file_hash(case.base / "processor_config.json")
    consistency = json.loads((checkpoint / "consistency.json").read_text())
    assert consistency["copied_base_assets"] == ["processor_config.json"]
    assert consistency["merged_sha256"]["processor_config.json"] == hashes["processor_config.json"]
    assert consistency["merged_sha256"] == {name: value for name, value in hashes.items()
                                            if name != "consistency.json"}
    provenance = json.loads((checkpoint / "provenance.json").read_text())
    assert provenance["details"]["copied_base_assets"] == ["processor_config.json"]
    assert not (case.output / ".checkpoint.partial").exists()
