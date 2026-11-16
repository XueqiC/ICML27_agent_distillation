"""Tiny CPU PEFT factors compared with independent dense algebra."""
import json

import pytest
torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file

from harness.lora_adapters import inspect_adapter
from sed.seed_evidence import artifact_hashes
from sed.storage import digest, file_hash, write_json
from tools import wsp_adapter_ops as ops


NAMES = [f"base_model.model.model.language_model.layers.{i}.{block}.{target}"
         for i, block, target in [(0, "self_attn", "q_proj"), (1, "self_attn", "q_proj"),
                                  (1, "mlp", "down_proj"), (2, "mlp", "up_proj")]]


def adapter(path, seed=3, *, rslora=False, zero=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = {}
    for index, name in enumerate(NAMES):
        tensors[name + ".lora_A.weight"] = torch.randn(2, 8 + index, generator=generator)
        tensors[name + ".lora_B.weight"] = torch.randn(10 + index, 2, generator=generator) * (index + 1)
        if zero:
            tensors[name + ".lora_B.weight"].zero_()
    return write_adapter(path, tensors, rslora=rslora)


def write_adapter(path, tensors, **overrides):
    snapshot = path.parent / "base-snapshot"
    snapshot.mkdir(exist_ok=True)
    write_json(snapshot / "config.json", {"fixture": "tiny-local-base"})
    path = path / "checkpoint"
    path.parent.mkdir()
    path.mkdir()
    config = {"peft_type": "LORA", "base_model_name_or_path": "fake-local-base", "r": 2,
              "lora_alpha": 5, "use_rslora": overrides.pop("rslora", False), "bias": "none",
              "rank_pattern": {}, "alpha_pattern": {}, "use_dora": False,
              "target_modules": [name.removeprefix("base_model.model.") for name in NAMES],
              "lora_dropout": .05, "task_type": "CAUSAL_LM", "revision": "fake-revision",
              "inference_mode": True, "lora_bias": False, "modules_to_save": None, **overrides}
    (path / "adapter_config.json").write_text(json.dumps(config))
    save_file(tensors, path / "adapter_model.safetensors")
    provenance = {"command": "synthetic-test-fixture", "config": {
        "models": {"student": {"requested_repo": "fake-local-base", "revision": "fake-revision"}},
        "sft": {"rank": 2, "lora_alpha": 5, "base_snapshot": {"path": str(snapshot.resolve()),
                "metadata_sha256": {"config.json": file_hash(snapshot / "config.json")}}}}}
    bind_source(path, provenance)
    return path


def bind_source(path, provenance=None, **metadata):
    """Seal synthetic fixture bytes, including deliberate malformed-math cases."""
    provenance_path = path.parent / "provenance.json"
    if provenance is None:
        provenance = json.loads(provenance_path.read_text())
    write_json(provenance_path, provenance)
    hashes = artifact_hashes(path)
    hashes.pop("training_state.json", None)
    write_json(path / "training_state.json", {"final": False, "identity_sha256": digest(provenance),
        "artifact_sha256": hashes, "supervised_tokens": 12, "passes": 1.25, "packing": "turn", **metadata})


def dense(path):
    config = json.loads((path / "adapter_config.json").read_text())
    tensors = load_file(path / "adapter_model.safetensors", device="cpu")
    scale = config["lora_alpha"] / (config["r"] ** .5 if config.get("use_rslora") else config["r"])
    return {name: scale * tensors[name + ".lora_B.weight"].double() @ tensors[name + ".lora_A.weight"].double()
            for name in NAMES}


def best(matrix, rank):
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def check_output(source, output, rank):
    output = output / "checkpoint"
    original = json.loads((source / "adapter_config.json").read_text())
    actual = json.loads((output / "adapter_config.json").read_text())
    assert actual == {**original, "r": rank, "lora_alpha": rank, "use_rslora": False}
    tensors = load_file(output / "adapter_model.safetensors")
    assert set(tensors) == set(load_file(source / "adapter_model.safetensors"))
    assert all(t.dtype == torch.float32 and t.device.type == "cpu" for t in tensors.values())
    report = json.loads((output / "wsp_ops.json").read_text())
    assert report["inputs"][0]["adapter_model_sha256"] == ops.sha256(source / "adapter_model.safetensors")
    return report


def check_gated_output(output, sources, operation, rank, *, snapshot=None):
    checkpoint = output / "checkpoint"
    provenance = json.loads((output / "provenance.json").read_text())
    original = json.loads((sources[0].parent / "provenance.json").read_text())
    state = json.loads((checkpoint / "training_state.json").read_text())
    student = original["config"]["models"]["student"]
    snapshot = snapshot or original["config"]["sft"]["base_snapshot"]["path"]
    inspected = inspect_adapter(operation, checkpoint, student, snapshot)
    assert inspected["rank"] == rank
    assert {p.name for p in output.iterdir()} == {"checkpoint", "provenance.json"}
    assert {p.name for p in checkpoint.iterdir()} == {
        "adapter_config.json", "adapter_model.safetensors", "wsp_ops.json", "training_state.json"}
    inputs = []
    for source in sources:
        source_state = json.loads((source / "training_state.json").read_text())
        inputs.append({"adapter": str(source.resolve()),
            "adapter_model_sha256": file_hash(source / "adapter_model.safetensors"),
            "adapter_config_sha256": file_hash(source / "adapter_config.json"),
            "base_model_name_or_path": json.loads((source / "adapter_config.json").read_text())["base_model_name_or_path"],
            "training_state_sha256": file_hash(source / "training_state.json"),
            "provenance_sha256": digest(json.loads((source.parent / "provenance.json").read_text())),
            "checkpoint_metadata": {key: source_state[key] for key in ("supervised_tokens", "passes", "packing")}})
    report = json.loads((checkpoint / "wsp_ops.json").read_text())
    derived = {"operation": operation,
               "parameters": {**report["parameters"], "rank": rank, "lora_alpha": rank},
               "tool_sha256": file_hash(ops.__file__), "inputs": inputs}
    if "wsp_derived" in original:
        derived["parent"] = original["wsp_derived"]
    assert provenance == {**original, "wsp_derived": derived}
    hashes = artifact_hashes(checkpoint)
    hashes.pop("training_state.json")
    assert state == {"final": True, "wsp_derived": True, "identity_sha256": digest(provenance),
                     "artifact_sha256": hashes, **inputs[0]["checkpoint_metadata"], "wsp_sources": inputs}
    assert inspected["checkpoint_metadata"] == inputs[0]["checkpoint_metadata"]
    assert report["inputs"] == [{key: row[key] for key in (
        "adapter", "adapter_model_sha256", "adapter_config_sha256", "base_model_name_or_path")} for row in inputs]


@pytest.mark.parametrize("operation", ["truncate", "average", "scale", "chained"])
def test_derived_adapters_pass_existing_gate(tmp_path, operation):
    source = adapter(tmp_path / "first")
    original_hashes = artifact_hashes(source.parent)
    output = tmp_path / "out"
    if operation in {"average", "chained"}:
        second = adapter(tmp_path / "second", seed=9)
        bind_source(second, supervised_tokens=24, passes=2.5, packing="conversation")
        second_hashes = artifact_hashes(second.parent)
        ops.average([source, second], output, rank=4)
        check_gated_output(output, [source, second], "average", 4)
        assert artifact_hashes(second.parent) == second_hashes
        if operation == "chained":
            chained = tmp_path / "chained"
            ops.truncate(output / "checkpoint", chained, rank=1)
            check_gated_output(chained, [output / "checkpoint"], "truncate", 1)
            for name, matrix in dense(chained / "checkpoint").items():
                torch.testing.assert_close(matrix, best(dense(output / "checkpoint")[name], 1),
                                           rtol=1e-6, atol=2e-5)
    elif operation == "truncate":
        ops.truncate(source, output, rank=1)
        check_gated_output(output, [source], operation, 1)
    else:
        ops.scale(source, output, factor=.5)
        check_gated_output(output, [source], operation, 2)
    assert artifact_hashes(source.parent) == original_hashes


@pytest.mark.parametrize("operation", ["average", "overlap"])
def test_source_comparison_accepts_cluster_specific_snapshot_paths(tmp_path, operation):
    sources, snapshots = [], []
    for cluster, seed in (("cluster-a", 3), ("cluster-b", 9)):
        directory = tmp_path / cluster
        directory.mkdir()
        source = adapter(directory / "source", seed=seed)
        provenance = json.loads((source.parent / "provenance.json").read_text())
        snapshot = provenance["config"]["sft"]["base_snapshot"]["path"]
        config = json.loads((source / "adapter_config.json").read_text())
        write_json(source / "adapter_config.json", {**config, "base_model_name_or_path": snapshot})
        bind_source(source)
        sources.append(source)
        snapshots.append(snapshot)
    assert snapshots[0] != snapshots[1]
    output = tmp_path / "out"
    if operation == "average":
        report = ops.average(sources, output, rank=4)
        check_output(sources[0], output, 4)
        check_gated_output(output, sources, operation, 4, snapshot=snapshots[1])
        for name, matrix in dense(output / "checkpoint").items():
            expected = (dense(sources[0])[name] + dense(sources[1])[name]) / 2
            torch.testing.assert_close(matrix, expected, rtol=1e-6, atol=2e-5)
    else:
        report = ops.overlap(*sources, output)
        assert json.loads(output.read_text()) == report
    assert [row["base_model_name_or_path"] for row in report["inputs"]] == snapshots


@pytest.mark.parametrize("operation", ["truncate", "average", "scale"])
@pytest.mark.parametrize("change,message", [
    ("adapter_config.json", "artifacts changed"),
    ("adapter_model.safetensors", "artifacts changed"),
    ("missing_provenance", "missing provenance.json"),
    ("missing_state", "missing training_state.json"),
    ("identity", "provenance mismatch"),
    ("missing_hash", "artifacts changed"),
    ("missing_metadata", "schema"),
    ("invalid_json", "schema"),
])
def test_refuse_unverifiable_sources(tmp_path, operation, change, message):
    valid = adapter(tmp_path / "valid")
    source = adapter(tmp_path / "source", seed=9)
    state_path = source / "training_state.json"
    if change.startswith("adapter_"):
        with (source / change).open("ab") as handle:
            handle.write(b"\n")
    elif change == "missing_provenance":
        (source.parent / "provenance.json").unlink()
    elif change == "missing_state":
        state_path.unlink()
    elif change == "invalid_json":
        state_path.write_text("{")
    else:
        state = json.loads(state_path.read_text())
        if change == "identity":
            state["identity_sha256"] = "0" * 64
        elif change == "missing_hash":
            state["artifact_sha256"].pop("adapter_config.json")
        else:
            state.pop("packing")
        write_json(state_path, state)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        if operation == "average":
            ops.average([valid, source], output, rank=2)
        elif operation == "truncate":
            ops.truncate(source, output, rank=1)
        else:
            ops.scale(source, output, factor=.5)
    assert not output.exists()


@pytest.mark.parametrize("field", ["requested_repo", "revision", "metadata_sha256"])
def test_average_refuses_different_base_identity(tmp_path, field):
    first, second = adapter(tmp_path / "first"), adapter(tmp_path / "second", seed=9)
    provenance = json.loads((second.parent / "provenance.json").read_text())
    config = json.loads((second / "adapter_config.json").read_text())
    if field == "metadata_sha256":
        provenance["config"]["sft"]["base_snapshot"][field] = {"config.json": "0" * 64}
    else:
        provenance["config"]["models"]["student"][field] = "different"
        config["revision" if field == "revision" else "base_model_name_or_path"] = "different"
        write_json(second / "adapter_config.json", config)
    bind_source(second, provenance)
    with pytest.raises(ValueError, match=field):
        ops.average([first, second], tmp_path / "out", rank=4)
    assert not (tmp_path / "out").exists()


def test_chained_derivation_refuses_modified_report(tmp_path):
    source = adapter(tmp_path / "source")
    output = tmp_path / "scaled"
    ops.scale(source, output, factor=.5)
    write_json(output / "checkpoint/wsp_ops.json", {"operation": "tampered"})
    with pytest.raises(ValueError, match="artifacts changed"):
        ops.truncate(output / "checkpoint", tmp_path / "out", rank=1)
    assert not (tmp_path / "out").exists()


def test_analysis_accepts_adapters_without_training_provenance(tmp_path):
    source = adapter(tmp_path / "source")
    (source / "training_state.json").unlink()
    (source.parent / "provenance.json").unlink()
    ops.spectrum(source, tmp_path / "spectrum.json")
    ops.overlap(source, source, tmp_path / "overlap.json")


@pytest.mark.parametrize("rslora", [False, True])
def test_spectrum_dense(tmp_path, rslora):
    source = adapter(tmp_path / "in", rslora=rslora)
    report = ops.spectrum(source, tmp_path / "spectrum.json")
    norms = {}
    for name, matrix in dense(source).items():
        row = report["modules"][name]
        s = torch.linalg.svdvals(matrix)[:2]
        norms[name] = float(matrix.square().sum())
        assert row["singular_values"] == pytest.approx(s.tolist())
        assert row["squared_frobenius_norm"] == pytest.approx(norms[name])
        assert row["stable_rank"] == pytest.approx(norms[name] / float(s[0] ** 2))
        for k, value in row["top_k_share"].items():
            assert value == pytest.approx(float(s[:int(k)].square().sum()) / norms[name])
    summary = report["summary"]
    total = sum(norms.values())
    assert summary["squared_frobenius_norm"] == pytest.approx(total)
    assert summary["share_by_layer"]["1"] == pytest.approx((norms[NAMES[1]] + norms[NAMES[2]]) / total)
    assert summary["share_by_module_type"]["q_proj"] == pytest.approx((norms[NAMES[0]] + norms[NAMES[1]]) / total)
    for p, count in summary["modules_to_reach_percent"].items():
        ordered = sorted(norms.values(), reverse=True)
        assert sum(ordered[:count]) >= total * int(p) / 100
        assert sum(ordered[:count - 1]) < total * int(p) / 100


@pytest.mark.parametrize("rank", [None, 1, 2, 6])
@pytest.mark.parametrize("rslora", [False, True])
def test_truncate_dense(tmp_path, rank, rslora):
    source = adapter(tmp_path / "in", rslora=rslora)
    output = tmp_path / "out"
    report = ops.truncate(source, output, rank=rank)
    check_output(source, output, rank or 2)
    original = dense(source)
    for name, value in dense(output / "checkpoint").items():
        expected = original[name] if rank is None else best(original[name], rank)
        torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-5)
        expected_share = float(expected.square().sum() / original[name].square().sum())
        assert report["modules"][name]["kept_share"] == pytest.approx(expected_share)


def test_top_modules_and_filters_zero_both_factors(tmp_path):
    source = adapter(tmp_path / "in")
    output = tmp_path / "top"
    report = ops.truncate(source, output, top_modules=2)
    expected = sorted(NAMES, key=lambda n: -float(dense(source)[n].square().sum()))[:2]
    assert {n for n, v in report["modules"].items() if v["selected"]} == set(expected)
    filtered = tmp_path / "filtered"
    report = ops.truncate(source, filtered, targets="q_proj,down_proj", layers="1-1,9", top_modules=1)
    expected = max(NAMES[1:3], key=lambda n: float(dense(source)[n].square().sum()))
    tensors = load_file(filtered / "checkpoint/adapter_model.safetensors")
    for name, matrix in dense(filtered / "checkpoint").items():
        if name == expected:
            torch.testing.assert_close(matrix, dense(source)[name], rtol=1e-6, atol=1e-5)
        else:
            assert torch.count_nonzero(matrix) == 0
            assert torch.count_nonzero(tensors[name + ".lora_A.weight"]) == 0
            assert torch.count_nonzero(tensors[name + ".lora_B.weight"]) == 0
    assert report["total_kept_share"] == pytest.approx(float(dense(source)[expected].square().sum()) /
                                                       sum(float(m.square().sum()) for m in dense(source).values()))


@pytest.mark.parametrize("same,rank,weights", [(True, 2, None), (False, 4, None),
                                               (False, 1, None), (False, 4, [2., -.5]), (True, 4, [1., -1.])])
def test_average_dense(tmp_path, same, rank, weights):
    first = adapter(tmp_path / "a")
    second = first if same else adapter(tmp_path / "b", seed=9, rslora=True)
    output = tmp_path / "out"
    report = ops.average([first, second], output, rank=rank, weights=weights)
    check_output(first, output, rank)
    w1, w2 = weights or [.5, .5]
    for name, matrix in dense(output / "checkpoint").items():
        expected = w1 * dense(first)[name] + w2 * dense(second)[name]
        torch.testing.assert_close(matrix, best(expected, rank), rtol=1e-6, atol=2e-5)
    if rank >= 4 and weights != [1., -1.]:
        assert report["total_kept_share"] == pytest.approx(1.)


@pytest.mark.parametrize("factor", [-2.5, 0., 1., .125])
def test_scale_dense(tmp_path, factor):
    source = adapter(tmp_path / "in", rslora=True)
    output = tmp_path / "out"
    ops.scale(source, output, factor=factor)
    check_output(source, output, 2)
    for name, value in dense(output / "checkpoint").items():
        torch.testing.assert_close(value, dense(source)[name] * factor, rtol=1e-6, atol=2e-5)


def test_overlap_identical_and_dense(tmp_path):
    a, b = adapter(tmp_path / "a"), adapter(tmp_path / "b", seed=9)
    identical = ops.overlap(a, a, tmp_path / "same.json", k=[1, 2, 16])
    for row in identical["modules"].values():
        assert row["frobenius_cosine"] == pytest.approx(1.)
        for item in row["subspaces"].values():
            assert item["left"] == pytest.approx(1.) and item["right"] == pytest.approx(1.)
    different = ops.overlap(a, b, tmp_path / "different.json", k=[1, 2])
    cosines, weights = [], []
    for name, row in different["modules"].items():
        x, y = dense(a)[name], dense(b)[name]
        cosine = float((x * y).sum() / (torch.linalg.norm(x) * torch.linalg.norm(y)))
        assert row["frobenius_cosine"] == pytest.approx(cosine)
        cosines.append(cosine)
        weights.append(float((x.square().sum() + y.square().sum()) / 2))
        ux, _, vx = torch.linalg.svd(x, full_matrices=False)
        uy, _, vy = torch.linalg.svd(y, full_matrices=False)
        for k, values in row["subspaces"].items():
            k = int(k)
            assert values["left"] == pytest.approx(float((ux[:, :k].T @ uy[:, :k]).square().sum()) / k)
            assert values["right"] == pytest.approx(float((vx[:k] @ vy[:k].T).square().sum()) / k)
            assert values["chance_left"] == k / x.shape[0] and values["chance_right"] == k / x.shape[1]
    assert different["summary"]["mean"]["frobenius_cosine"] == pytest.approx(sum(cosines) / 4)
    assert different["summary"]["squared_norm_weighted_mean"]["frobenius_cosine"] == pytest.approx(
        sum(c * w for c, w in zip(cosines, weights)) / sum(weights))
    assert different["summary"]["by_module_type"]["q_proj"]["mean"]["frobenius_cosine"] == pytest.approx(
        sum(different["modules"][name]["frobenius_cosine"] for name in NAMES[:2]) / 2)


def test_orthogonal_and_zero_overlap(tmp_path):
    paths = []
    for i in range(2):
        tensors = {}
        for name in NAMES:
            a = torch.eye(8)[2*i:2*i+2]
            tensors[name + ".lora_A.weight"] = a
            tensors[name + ".lora_B.weight"] = a.T @ torch.diag(torch.tensor([2., 1.]))
        paths.append(write_adapter(tmp_path / str(i), tensors))
    result = ops.overlap(*paths, tmp_path / "orthogonal.json", k=[1, 2])
    for row in result["modules"].values():
        assert row["frobenius_cosine"] == pytest.approx(0.)
        assert all(v["left"] == 0. and v["right"] == 0. for v in row["subspaces"].values())
    zero = adapter(tmp_path / "zero", zero=True)
    report = ops.spectrum(zero, tmp_path / "zero.json")
    assert report["summary"]["squared_frobenius_norm"] == 0
    assert all(row["stable_rank"] == 0 for row in report["modules"].values())
    overlap = ops.overlap(zero, zero, tmp_path / "zero-overlap.json")
    assert overlap["summary"]["mean"]["frobenius_cosine"] is None


@pytest.mark.parametrize("field,value", [("rank_pattern", {"q_proj": 1}), ("alpha_pattern", {"q_proj": 2}),
                                          ("use_dora", True), ("bias", "all"), ("lora_bias", True)])
def test_refuse_unsupported_config(tmp_path, field, value):
    source = adapter(tmp_path / "in")
    config = json.loads((source / "adapter_config.json").read_text())
    config[field] = value
    (source / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        ops.spectrum(source, tmp_path / "out.json")


@pytest.mark.parametrize("operation", ["spectrum", "truncate", "average", "scale", "overlap"])
def test_cli_and_no_overwrite(tmp_path, operation):
    source = adapter(tmp_path / "in")
    out = tmp_path / "out"
    args = [operation, str(source)]
    if operation in {"average", "overlap"}:
        args += [str(source)]
    args += ["--out", str(out)]
    if operation == "average":
        args += ["--rank", "4"]
    if operation == "scale":
        args += ["--factor", "2"]
    ops.main(args)
    before = out.read_bytes() if out.is_file() else (out / "checkpoint/adapter_model.safetensors").read_bytes()
    with pytest.raises(SystemExit) as error:
        ops.main(args)
    assert error.value.code == 2
    assert before == (out.read_bytes() if out.is_file() else (out / "checkpoint/adapter_model.safetensors").read_bytes())


@pytest.mark.parametrize("change", ["base", "modules", "dimensions", "missing_b", "extra", "nan"])
def test_refuse_incompatible_or_malformed(tmp_path, change):
    a, b = adapter(tmp_path / "a"), adapter(tmp_path / "b")
    tensors = load_file(b / "adapter_model.safetensors")
    if change == "base":
        config = json.loads((b / "adapter_config.json").read_text())
        config["base_model_name_or_path"] = "different"
        (b / "adapter_config.json").write_text(json.dumps(config))
    elif change == "modules":
        for part in "AB":
            tensors.pop(NAMES[0] + f".lora_{part}.weight")
    elif change == "dimensions":
        tensors[NAMES[0] + ".lora_B.weight"] = torch.zeros(11, 2)
    elif change == "missing_b":
        tensors.pop(NAMES[0] + ".lora_B.weight")
    elif change == "extra":
        tensors["bias"] = torch.zeros(2)
    else:
        tensors[NAMES[0] + ".lora_A.weight"][0, 0] = float("nan")
    save_file(tensors, b / "adapter_model.safetensors")
    bind_source(b)
    with pytest.raises(ValueError):
        ops.average([a, b], tmp_path / "out", rank=4)
    assert not (tmp_path / "out").exists()


def test_small_core_only_and_invalid_arguments(tmp_path, monkeypatch):
    source = adapter(tmp_path / "in")
    svd = torch.linalg.svd
    def small_core(matrix, **kwargs):
        assert max(matrix.shape) <= 4
        assert matrix.dtype == torch.float64 and matrix.device.type == "cpu"
        return svd(matrix, **kwargs)
    monkeypatch.setattr(torch.linalg, "svd", small_core)
    ops.average([source, source], tmp_path / "avg", rank=4)
    for kwargs in ({"rank": 0}, {"top_modules": -1}, {"layers": "2-0"}):
        with pytest.raises(ValueError):
            ops.truncate(source, tmp_path / "bad", **kwargs)
    with pytest.raises(ValueError, match="weights"):
        ops.average([source, source], tmp_path / "bad", rank=4, weights=[1.])
