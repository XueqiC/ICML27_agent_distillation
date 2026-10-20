"""Offline CPU coverage; synthetic checkpoints, fake HTTP, no vLLM imports."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import struct
import sys
from types import SimpleNamespace

import httpx
import pytest

from harness.lora_adapters import (adapter_identity, convert_adapter, inspect_adapter,
                                   parse_adapters, safetensors_header)
from harness.serve_student import free_port, server_command
from sed.config import load_config
from sed.storage import digest, file_hash, write_json
from tools.check_lora_serving import (compare_prompts, comparison, fixed_prompts,
                                     summarize_prompts, validate_cases, validate_report, vllm_logprobs)


@pytest.fixture
def adapter_bank(tmp_path):
    student = load_config()["models"]["student"]
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified", "text_config": {
        "num_hidden_layers": 1, "attention_k_eq_v": False, "layer_types": ["full_attention"]}})
    write_json(snapshot / "model.safetensors.index.json", {"weight_map": {
        f"model.language_model.layers.0.self_attn.{p}_proj.weight": "model.safetensors"
        for p in ("k", "v")}})
    adapter = tmp_path / "train" / "checkpoint"
    adapter.mkdir(parents=True)
    module = "model.language_model.layers.0.self_attn.q_proj"
    config = {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none",
              "base_model_name_or_path": str(snapshot), "revision": None,
              "target_modules": [module]}
    write_json(adapter / "adapter_config.json", config)
    prefix = "base_model.model." + module
    header = {prefix + ".lora_A.weight": {"dtype": "F32", "shape": [1, 2], "data_offsets": [0, 8]},
              prefix + ".lora_B.weight": {"dtype": "F32", "shape": [2, 1], "data_offsets": [8, 16]},
              "__metadata__": {"format": "pt"}}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (adapter / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + struct.pack("<ffff", 1.0, -0.0, 2.0, 3.0))
    provenance = {"config": {"models": {"student": student}, "sft": {"base_snapshot": {
        "path": str(snapshot), "metadata_sha256": {"config.json": file_hash(snapshot / "config.json")}}}}}
    write_json(adapter.parent / "provenance.json", provenance)
    state = {"supervised_tokens": 10501, "passes": 1.05, "packing": "episode",
             "identity_sha256": digest(provenance), "artifact_sha256": {
                 p.name: file_hash(p) for p in adapter.iterdir()}}
    write_json(adapter / "training_state.json", state)
    return adapter, snapshot, student


def test_adapter_identity_and_command(adapter_bank, monkeypatch):
    path, snapshot, student = adapter_bank
    row = inspect_adapter("sft", path, student, snapshot)
    assert row["checkpoint_metadata"] == {"supervised_tokens": 10501, "passes": 1.05, "packing": "episode"}
    assert row["adapter_weights_sha256"] == file_hash(path / "adapter_model.safetensors")
    assert row["adapter_config_sha256"] == file_hash(path / "adapter_config.json")
    args = SimpleNamespace(model_path=snapshot, merged_model_path=None, python="fake", port=23456)
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.42")
    cmd = server_command(args, student, {"gemma4_parser": True}, [row, dict(row, name="later", rank=32)])
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.42"
    assert cmd[cmd.index("--max-lora-rank") + 1] == "32"
    assert cmd[cmd.index("--max-loras") + 1] == "2"
    assert cmd[cmd.index("--lora-modules") + 1:] == [f"sft={path}", f"later={path}"]
    assert "--lora-dtype" not in cmd
    args.lora_dtype = "bfloat16"
    assert server_command(args, student, {"gemma4_parser": True}, [row, dict(row, name="later", rank=32)]) == cmd
    args.lora_dtype = "float16"
    fp16 = server_command(args, student, {"gemma4_parser": True}, [row, dict(row, name="later", rank=32)])
    position = fp16.index("--lora-dtype")
    assert fp16[position + 1] == "float16"
    assert fp16[:position] + fp16[position + 2:] == cmd
    args.merged_model_path = str(snapshot)
    with pytest.raises(ValueError, match="unmerged"):
        server_command(args, student, {"gemma4_parser": False}, [row])
    assert "--enable-lora" not in server_command(args, student, {"gemma4_parser": False})
    assert "--lora-dtype" not in server_command(args, student, {"gemma4_parser": False})
    for value in ("nan", "inf", "0", "1.1", "bad"):
        monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", value)
        with pytest.raises(ValueError):
            server_command(args, student, {"gemma4_parser": False})


@pytest.mark.parametrize("values", [["base=/tmp"], ["a=/tmp", "a=/tmp/b"],
                                    ["=x"], ["a="], ["a"], ["../x=y"], ["-x=y"]])
def test_names_refused(values):
    with pytest.raises(ValueError):
        parse_adapters(values, "base")


def test_names_paths_with_spaces_and_equals(tmp_path):
    path = tmp_path / "space = path"
    assert parse_adapters(["sft=" + str(path)], "base") == [("sft", path)]


@pytest.mark.parametrize("field,value", [("base_model_name_or_path", "wrong/model"),
                                         ("revision", "wrong"), ("rank_pattern", {"q_proj": 8})])
def test_adapter_mismatch_refused(adapter_bank, field, value):
    path, snapshot, student = adapter_bank
    config = json.loads((path / "adapter_config.json").read_text())
    config[field] = value
    write_json(path / "adapter_config.json", config)
    with pytest.raises(ValueError):
        inspect_adapter("sft", path, student, snapshot)


def test_provenance_and_weights_tamper_refused(adapter_bank):
    path, snapshot, student = adapter_bank
    with (path / "adapter_model.safetensors").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        inspect_adapter("sft", path, student, snapshot)


def test_relocated_snapshot_and_training_revision(adapter_bank, tmp_path):
    path, snapshot, student = adapter_bank
    relocated = tmp_path / "hpg-snapshot"
    shutil.copytree(snapshot, relocated)
    assert inspect_adapter("sft", path, student, relocated)["revision"] == student["revision"]
    provenance_path = path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["config"]["models"]["student"]["revision"] = "different"
    write_json(provenance_path, provenance)
    state = json.loads((path / "training_state.json").read_text())
    state["identity_sha256"] = digest(provenance)
    write_json(path / "training_state.json", state)
    with pytest.raises(ValueError, match="revision"):
        inspect_adapter("sft", path, student, relocated)


@pytest.mark.parametrize("mode", ["targets", "gemma4"])
def test_conversion_pure_rename(adapter_bank, tmp_path, mode):
    path, snapshot, student = adapter_bank
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    original = inspect_adapter("sft", path, student, snapshot)
    dest = tmp_path / "out" / mode
    row = convert_adapter(original, dest, mode)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    assert row["conversion"]["tensor_bytes_identical"] is True
    old_header, old_offset = safetensors_header(path / "adapter_model.safetensors")
    new_header, new_offset = safetensors_header(dest / "adapter_model.safetensors")
    assert before["adapter_model.safetensors"][old_offset:] == (dest / "adapter_model.safetensors").read_bytes()[new_offset:]
    assert list(old_header.values()) == list(new_header.values())
    cfg = json.loads((dest / "adapter_config.json").read_text())
    old_cfg = json.loads(before["adapter_config.json"])
    assert cfg.pop("target_modules") == ["q_proj"]
    old_cfg.pop("target_modules")
    assert cfg == old_cfg  # rank, alpha, scaling, and every other field unchanged
    assert adapter_identity(row) == adapter_identity(original)
    if mode == "targets":
        assert row["conversion"]["adapter_weights_sha256"] == original["adapter_weights_sha256"]
    else:
        assert all("base_model.model.language_model.model." in k
                   for k in new_header if k != "__metadata__")
    with pytest.raises(ValueError, match="outside"):
        convert_adapter(original, path / "conversion", mode)


def test_select_port_by_binding(monkeypatch):
    calls = []
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): calls.append("close")
        def bind(self, address): calls.append(address)
        def getsockname(self): return "127.0.0.1", 34567
    monkeypatch.setattr("harness.serve_student.socket.socket", lambda *a: Socket())
    assert free_port() == 34567
    assert calls == [("127.0.0.1", 0), "close"]


def test_listener_must_belong_to_child_group(tmp_path, monkeypatch):
    from harness.serve_student import owns_listener
    (tmp_path / "net").mkdir()
    (tmp_path / "net/tcp").write_text(
        "header\n0: 0100007F:8707 00000000:0000 0A 0 0 0 0 0 987\n")
    directory = tmp_path / "123/fd"
    directory.mkdir(parents=True)
    (directory / "4").symlink_to("socket:[987]")
    monkeypatch.setattr("harness.serve_student.os.getpgid", lambda pid: 456)
    assert owns_listener(456, 34567, proc=tmp_path)
    assert not owns_listener(999, 34567, proc=tmp_path)
    assert not owns_listener(456, 34568, proc=tmp_path)


@pytest.mark.parametrize("lora_dtype", [None, "bfloat16", "float16"])
def test_server_provenance_and_all_models_readiness(adapter_bank, tmp_path, monkeypatch, lora_dtype):
    import harness.serve_student as serving
    path, snapshot, student = adapter_bank
    monkeypatch.setattr(serving, "inspect_runtime", lambda _: {
        "gemma4_parser": True, "vllm": "fake", "parser_registry_path": str(snapshot / "config.json")})
    monkeypatch.setattr(serving, "stop_server", lambda _: None)
    monkeypatch.setattr(serving.signal, "signal", lambda *a: None)
    monkeypatch.setattr(serving.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, wait=lambda: 0))
    seen = []
    monkeypatch.setattr(serving, "wait_ready", lambda process, port, timeout, models, **kw: seen.extend(models))
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
        def getsockname(self): return "127.0.0.1", 34567
    monkeypatch.setattr(serving.socket, "socket", lambda *a: Socket())
    # Provenance uses subprocess.run; avoid globally replacing Popen's internals.
    monkeypatch.setattr(serving, "write_provenance", lambda out, **kw: write_json(out / "provenance.json", kw))
    out = tmp_path / "server"
    dtype_args = ["--lora-dtype", lora_dtype] if lora_dtype else []
    assert serving.main(["--python", "fake", "--model-path", str(snapshot), "--port", "0",
                         "--lora-adapter", "sft=" + str(path), "--output", str(out), *dtype_args]) == 0
    record = json.loads((out / "provenance.json").read_text())["config"]["server"]
    assert seen == [student["requested_repo"], "sft"]
    assert record["port"] > 0 and str(record["port"]) in record["endpoint"]
    assert record["lora_adapters"][0] == inspect_adapter("sft", path, student, snapshot)
    assert record["lora_dtype"] == (lora_dtype or "bfloat16")
    if lora_dtype == "float16":
        assert record["command"][record["command"].index("--lora-dtype") + 1] == lora_dtype
    else:
        assert "--lora-dtype" not in record["command"]
    assert not (out / "ready.json").exists()


def test_invalid_lora_dtype_refused_before_serving(tmp_path):
    from harness.serve_student import main
    with pytest.raises(SystemExit) as exc:
        main(["--python", "unused", "--model-path", str(tmp_path), "--output", str(tmp_path / "out"),
              "--lora-dtype", "float32"])
    assert exc.value.code == 2 and not (tmp_path / "out").exists()


def cases():
    return [{"input_ids": [2, 3, 10, 106], "prompt_tokens": 2,
             "input_sha256": digest([2, 3, 10, 106]), "hf_logprobs": [-1.0, -2.0],
             "hf_base_logprobs": [-1.29, -2.29],
             "source": {"task_id": "train/task/1", "turn_index": i,
                        "training_eligible": True}} for i in range(8)]


def fake_completion(request):
    body = json.loads(request.content)
    assert body["prompt"] == [2, 3, 10, 106]
    assert body["prompt_logprobs"] == 1 and body["return_token_ids"]
    assert body["add_special_tokens"] is False and body["max_tokens"] == 0
    values = [-1.01, -2.01] if body["model"] == "sft" else [-1.3, -2.3]
    return httpx.Response(200, json={"model": body["model"], "choices": [{
        "prompt_token_ids": body["prompt"], "prompt_logprobs": [None, {"3": {"logprob": -9}},
            {"10": {"logprob": values[0]}}, {"106": {"logprob": values[1]}}]}]})


@pytest.fixture
def bound_report(adapter_bank):
    path, snapshot, student = adapter_bank
    row = inspect_adapter("sft", path, student, snapshot)
    with httpx.Client(transport=httpx.MockTransport(fake_completion)) as client:
        report = compare_prompts(cases(), client, "http://127.0.0.1:34567/v1", "sft", "base")
    server = {"config": {"server": {"lora_adapters": [row], "served_model_name": "base",
                                    "endpoint": "http://127.0.0.1:34567/v1", "merged_model": False}}}
    report.update(model_name="sft", server_provenance_sha256=digest(server),
                  identity={"adapter": adapter_identity(row)})
    return report, server


def test_logprob_equivalence_and_report_gate(bound_report):
    report, server = bound_report
    assert report["schema_version"] == 5 and report["passed"] and len(report["prompts"]) == 8
    for stats in [*report["prompts"], report["aggregate"]]:
        for metric, expected in (("adapter_error", 0.01), ("base_noise", 0.01), ("effect", 0.29)):
            assert stats[metric]["mean_abs_diff"] == pytest.approx(expected)
            assert stats[metric]["max_abs_diff"] == pytest.approx(expected)
        for metric in ("adapter_error", "base_noise"):
            assert stats[metric]["tokens_abs_diff_gt_0_25"] == 0
            assert stats[metric]["tokens_abs_diff_gt_0_5"] == 0
        if stats is report["aggregate"]:
            assert stats["mean_abs_diff_tolerance_nats"] == 0.05
            assert stats["tokens_abs_diff_gt_0_5_tolerance"] == 2
            assert stats["min_effect_nats"] == pytest.approx(0.02)
        else:
            assert stats["mean_abs_diff_tolerance_nats"] is None
            assert stats["tokens_abs_diff_gt_0_5_tolerance"] is None
            assert stats["min_effect_nats"] == pytest.approx(0.01)
    validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")
    with pytest.raises(ValueError, match="Endpoint"):
        validate_report(report, server, "sft", "http://127.0.0.1:34568/v1")
    report["server_provenance_sha256"] = "stale"
    with pytest.raises(ValueError, match="passing"):
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


def test_thresholds_fail_closed():
    assert comparison([-1], [-1.02], [-2], [-2], pooled=True)["passed"]
    assert not comparison([-1], [-1.06], [-2], [-2], pooled=True)["passed"]
    assert not comparison([-1], [-1.02], [-1.05], [-1.03], pooled=True)["passed"]
    for pooled in (False, True):
        assert not comparison([-1], [-1.02], [-1.02], [-1.03], pooled=pooled)["passed"]
    # Prompt equivalence error is diagnostic; only its effect versus noise gates it.
    assert comparison([-1], [-1.06], [-1.16], [-1.15])["passed"]
    assert not comparison([-1], [-1.06], [-1.56], [-1])["passed"]


@pytest.mark.parametrize("noise,passed", [(0.038, True), (0.05, True), (0.01, False)])
def test_observed_error_relative_to_base_noise(noise, passed):
    stats = comparison([-1], [-1.0563], [-1.8963], [-1.8963 + noise], pooled=True)
    assert stats["adapter_error"]["mean_abs_diff"] == pytest.approx(0.0563)
    assert stats["base_noise"]["mean_abs_diff"] == pytest.approx(noise)
    assert stats["effect"]["mean_abs_diff"] == pytest.approx(0.84)
    assert stats["mean_abs_diff_tolerance_nats"] == pytest.approx(max(0.05, 1.5 * noise))
    assert stats["passed"] is passed


def test_threshold_equality_is_inclusive():
    # Binary-exact values put pooled error/noise and effect/noise at equality.
    assert comparison([-1], [-1.09375], [-1.21875], [-1.15625], pooled=True)["passed"]
    assert not comparison([-0.999], [-1.09375], [-1.21875], [-1.15625], pooled=True)["passed"]
    assert not comparison([-1], [-1.09375], [-1.21875], [-1.156], pooled=True)["passed"]
    assert comparison([0], [-0.05], [-1], [-1], pooled=True)["passed"]
    assert comparison([-1], [-1.125], [-1.25], [-1.125])["passed"]
    assert not comparison([-1], [-1.125], [-1.25], [-1.124])["passed"]
    # The specified inclusive inequalities also hold at a zero noise/effect floor.
    assert comparison([-1], [-1], [-1], [-1], pooled=True)["passed"]
    # Outlier diagnostics use strict >, including at both cutoffs.
    stats = comparison([0] * 4, [-0.25, -0.5, -0.75, 0], [-10] * 4,
                       [-9.75, -9.5, -9.25, -10])
    for metric in ("adapter_error", "base_noise"):
        assert stats[metric]["tokens_abs_diff_gt_0_25"] == 2
        assert stats[metric]["tokens_abs_diff_gt_0_5"] == 1


@pytest.mark.parametrize("tokens", [32, 41, 200, 236])
@pytest.mark.parametrize("base_outliers", [0, 1, 6])
@pytest.mark.parametrize("excess", [0, 1])
def test_pooled_outlier_limit_is_inclusive(tokens, base_outliers, excess):
    # Integer ceiling is independent of the checker's floating-point formula.
    limit = max(2 * base_outliers + 2, (tokens + 19) // 20)
    count = limit + excess
    error = [0.51] * count + [0] * (tokens - count)
    noise = [1.0] * base_outliers + [0.0625 if base_outliers else 0] * (tokens - base_outliers)
    stats = comparison([0] * tokens, [-v for v in error], [-10] * tokens,
                       [-10 + v for v in noise], pooled=True)
    assert stats["adapter_error"]["mean_abs_diff"] <= stats["mean_abs_diff_tolerance_nats"]
    assert stats["effect"]["mean_abs_diff"] >= stats["min_effect_nats"]
    assert stats["adapter_error"]["tokens_abs_diff_gt_0_5"] == count
    assert stats["tokens_abs_diff_gt_0_5_tolerance"] == limit
    assert stats["passed"] is (excess == 0)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")], [float("-inf")], [None], [True]])
def test_invalid_logprobs_refused(index, bad):
    values = [[-1], [-1.01], [-2], [-2.01]]
    values[index] = bad
    with pytest.raises(ValueError):
        comparison(*values)
    with pytest.raises(ValueError):
        comparison([], [], [], [])


@pytest.mark.parametrize("bad", [None, [], [-1.29], [-1.29, float("nan")]])
def test_reference_requires_aligned_finite_hf_base(bad):
    rows = cases()
    if bad is None:
        rows[0].pop("hf_base_logprobs")
    else:
        rows[0]["hf_base_logprobs"] = bad
    with pytest.raises(ValueError, match="hf_base_logprobs"):
        validate_cases(rows)


def measured_cases(errors, noises, effects):
    """Construct aligned token measurements, with distinct TRAIN turn identities."""
    rows = cases()
    for row, error, noise, effect in zip(rows, errors, noises, effects):
        n = len(error)
        row["input_ids"] = [2, 3] + [10] * (n - 1) + [106]
        row["input_sha256"] = digest(row["input_ids"])
        row["hf_logprobs"] = [-1.0] * n
        row["adapter_vllm_logprobs"] = [-1 - value for value in error]
        row["base_vllm_logprobs"] = [a - e for a, e in zip(row["adapter_vllm_logprobs"], effect)]
        row["hf_base_logprobs"] = [b + v for b, v in zip(row["base_vllm_logprobs"], noise)]
    return rows


@pytest.mark.parametrize("name,outliers,pooled_error,effect", [
    ("k20s2-p40", 7, 0.045, 0.93), ("k10s2-t400k", 6, 0.042, 0.98)])
def test_hpg_large_adapter_rounding_passes(bound_report, name, outliers, pooled_error, effect):
    # Synthetic measurements match supplied pooled figures, not unavailable HPG arrays.
    lengths = [27, 34, 28, 30, 24, 34, 35, 24]
    remainder = (236 * pooled_error - outliers * 0.75) / (236 - outliers)
    errors = [[remainder] * n for n in lengths]
    errors[1][:outliers] = [0.75] * outliers
    # The pooled base-noise mean was not supplied; choose a small synthetic value.
    rows = measured_cases(errors, [[0.01] * n for n in lengths], [[effect] * n for n in lengths])
    report, server = bound_report
    report.update(summarize_prompts(rows))
    pooled = report["aggregate"]
    assert pooled["tokens"] == 236
    assert pooled["adapter_error"]["mean_abs_diff"] == pytest.approx(pooled_error)
    assert pooled["effect"]["mean_abs_diff"] == pytest.approx(effect)
    assert pooled["adapter_error"]["tokens_abs_diff_gt_0_5"] == outliers > 2
    assert pooled["base_noise"]["tokens_abs_diff_gt_0_5"] == 0
    assert pooled["tokens_abs_diff_gt_0_5_tolerance"] == 12
    assert report["thresholds"]["outlier_token_fraction"] == 0.05
    assert "ceil(0.05 * pooled_tokens)" in report["rule"]
    assert report["passed"], name
    validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


def test_p20_outliers_and_token_weighted_pool(bound_report):
    # Synthetic arrays matching rounded p20 prompt means, maxima and outlier counts;
    # real artifacts stay read-only and are not required by the test suite.
    lengths = [27, 34, 28, 30, 24, 34, 35, 24]
    error_means = [0.008796, 0.017867, 0.105254, 0.000448, 0.011256, 0.070332, 0.002872, 0.033135]
    noise_means = [0.033476, 0.062088, 0.057430, 0.051241, 0.034818, 0.043492, 0.028150, 0.058981]
    effect_means = [1.193503, 1.722613, 1.502924, 0.904344, 1.435949, 1.327441, 1.202816, 1.119152]
    def with_outliers(means, outliers):
        return [out + [(n * mean - sum(out)) / (n - len(out))] * (n - len(out))
                for n, mean, out in zip(lengths, means, outliers)]
    errors = with_outliers(error_means, [[], [0.3466], [2.7192], [], [], [1.0218, 0.6, 0.3], [], [0.4373]])
    noises = with_outliers(noise_means, [[], [0.8124], [0.3634, 0.26], [0.6555], [],
                                       [0.3621, 0.26], [0.3238], [0.4991, 0.26, 0.26]])
    effects = [[effect] * n for effect, n in zip(effect_means, lengths)]
    report, server = bound_report
    report.update(summarize_prompts(measured_cases(errors, noises, effects)))
    assert report["passed"] and all(p["passed"] for p in report["prompts"])
    old_failures = [i for i, p in enumerate(report["prompts"])
                    if p["adapter_error"]["mean_abs_diff"] > max(0.05, 1.5 * p["base_noise"]["mean_abs_diff"])]
    assert old_failures == [2, 5]
    pooled = report["aggregate"]
    assert pooled["tokens"] == 236 and pooled["passed"]
    for metric, means in (("adapter_error", error_means), ("base_noise", noise_means), ("effect", effect_means)):
        assert pooled[metric]["mean_abs_diff"] == pytest.approx(sum(n * m for n, m in zip(lengths, means)) / 236)
    assert pooled["adapter_error"]["max_abs_diff"] == pytest.approx(2.7192)
    assert pooled["base_noise"]["max_abs_diff"] == pytest.approx(0.8124)
    for metric, counts in (("adapter_error", (6, 3)), ("base_noise", (10, 2))):
        for field, count in zip(("tokens_abs_diff_gt_0_25", "tokens_abs_diff_gt_0_5"), counts):
            assert pooled[metric][field] == count == sum(p[metric][field] for p in report["prompts"])
    validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


@pytest.mark.parametrize("pooled_error,effect", [(0.0499, 0.144), (0.0467, 0.337), (0.0396, 0.47)])
def test_light_adapter_and_isolated_token_error_pass(bound_report, pooled_error, effect):
    # Synthetic 236-token cases shaped like the light adapters; no real run is required.
    lengths = [27, 34, 28, 30, 24, 34, 35, 24]
    remainder = (236 * pooled_error - 1.82) / 235
    errors = [[remainder] * n for n in lengths]
    errors[1][0] = 1.82
    noise_remainder = (236 * 0.0461 - 0.81) / 235
    noises = [[noise_remainder] * n for n in lengths]
    noises[1][0] = 0.81
    effects = [[effect] * n for n in lengths]
    report, server = bound_report
    report.update(summarize_prompts(measured_cases(errors, noises, effects)))
    assert report["passed"] and all(p["passed"] for p in report["prompts"])
    assert report["aggregate"]["adapter_error"]["mean_abs_diff"] == pytest.approx(pooled_error)
    assert report["aggregate"]["base_noise"]["mean_abs_diff"] == pytest.approx(0.0461)
    assert report["aggregate"]["effect"]["mean_abs_diff"] == pytest.approx(effect)
    prompt = report["prompts"][1]
    assert prompt["adapter_error"]["mean_abs_diff"] > 0.1 * prompt["effect"]["mean_abs_diff"]
    assert prompt["adapter_error"]["tokens_abs_diff_gt_0_5"] == 1
    assert prompt["base_noise"]["tokens_abs_diff_gt_0_5"] == 1
    validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


def test_outlier_limit_is_pooled_not_per_prompt():
    errors = [[0.75] * 4 + [0] * 46] + [[0] * 50] * 7
    noises = [[0] * 50] * 7 + [[0.75] + [0] * 49]
    report = summarize_prompts(measured_cases(errors, noises, [[1.0] * 50] * 8))
    assert report["passed"]
    assert report["prompts"][0]["adapter_error"]["tokens_abs_diff_gt_0_5"] == 4
    assert report["prompts"][0]["base_noise"]["tokens_abs_diff_gt_0_5"] == 0
    assert report["aggregate"]["tokens_abs_diff_gt_0_5_tolerance"] == 20


@pytest.mark.parametrize("failure", ["prompt", "pooled_error", "pooled_effect", "pooled_outliers"])
def test_validation_recomputes_each_gate(bound_report, failure):
    report, server = bound_report
    if failure == "prompt":
        errors, noises, effects = [[0.01]] * 8, [[0.1]] + [[0.01]] * 7, [[0.05]] + [[0.3]] * 7
    elif failure == "pooled_error":
        errors, noises, effects = [[0.049]] * 4 + [[0.149]] * 4, [[0]] * 4 + [[0.1]] * 4, [[2.0]] * 8
    elif failure == "pooled_effect":
        errors, noises, effects = [[0.01]] * 8, [[0.04]] * 8, [[0.06]] * 8
    else:
        errors = [[0.75] * 2 + [0] * 18] + [[0.75] + [0] * 19] * 7
        noises, effects = [[0.01] * 20] * 8, [[1.0] * 20] * 8
    rows = measured_cases(errors, noises, effects)
    actual = summarize_prompts(rows)
    assert not actual["passed"]
    # Each gate can fail independently while every other requirement holds.
    assert all(row["passed"] for row in actual["prompts"]) is (failure != "prompt")
    assert actual["aggregate"]["passed"] is (failure == "prompt")
    pooled = actual["aggregate"]
    assert (pooled["adapter_error"]["mean_abs_diff"] <= pooled["mean_abs_diff_tolerance_nats"]) is (
        failure != "pooled_error")
    assert (pooled["effect"]["mean_abs_diff"] >= pooled["min_effect_nats"]) is (failure != "pooled_effect")
    assert (pooled["adapter_error"]["tokens_abs_diff_gt_0_5"]
            <= pooled["tokens_abs_diff_gt_0_5_tolerance"]) is (failure != "pooled_outliers")
    # Leave the old passing summaries/flags in place: raw measurements govern.
    for row, measured in zip(report["prompts"], rows):
        row.update(measured)
    report["thresholds"] = {"mean_abs_diff_nats": 999, "min_effect_noise_multiplier": 0,
                            "outlier_count_multiplier": 999, "outlier_count_allowance": 999,
                            "outlier_token_fraction": 1}
    with pytest.raises(ValueError, match="per-prompt effect"):
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


@pytest.mark.parametrize("missing", ["one", "all"])
def test_missing_adapter_is_refused(bound_report, missing):
    report, server = bound_report
    errors, noises = [[0.02]] * 8, [[0.01]] * 8
    effects = [[0.0]] + [[0.3 if missing == "one" else 0.0]] * 7
    report.update(summarize_prompts(measured_cases(errors, noises, effects)))
    assert not report["passed"] and not report["prompts"][0]["passed"]
    assert report["aggregate"]["passed"] is (missing == "one")
    with pytest.raises(ValueError, match="passing"):
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


@pytest.mark.parametrize("schema", [None, 1, 2, 3, 4, 6, "5", 5.0, True])
def test_report_schema_refused(bound_report, schema):
    report, server = bound_report
    if schema is None:
        report.pop("schema_version")
    else:
        report["schema_version"] = schema
    with pytest.raises(ValueError, match="Schema version 5"):
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


@pytest.mark.parametrize("field", ["hf_logprobs", "hf_base_logprobs", "adapter_vllm_logprobs", "base_vllm_logprobs"])
@pytest.mark.parametrize("schema", [2, 3, 4, 5])
def test_recomputation_requires_all_measurements(bound_report, field, schema):
    report, server = bound_report
    report["schema_version"] = schema
    for row in report["prompts"]:
        row.pop(field)
    with pytest.raises((ValueError, KeyError), match=field):
        report.update(summarize_prompts(report["prompts"]))
    if schema == 5:
        with pytest.raises((ValueError, KeyError), match=field):
            validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


@pytest.mark.parametrize("schema", [2, 3, 4])
def test_complete_legacy_report_can_be_recomputed(bound_report, schema):
    report, server = bound_report
    report.update(schema_version=schema, passed=False, rule=f"old schema {schema} rule", thresholds={})
    for row in report["prompts"]:
        row["passed"] = False
        row["mean_abs_diff_tolerance_nats"] = 0
        for metric in ("adapter_error", "base_noise"):
            row[metric].pop("tokens_abs_diff_gt_0_25")
            row[metric].pop("tokens_abs_diff_gt_0_5")
    with pytest.raises(ValueError, match="Schema version 5"):
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")
    report.update(summarize_prompts(report["prompts"]))
    assert report["schema_version"] == 5 and report["passed"]
    assert report["thresholds"]["min_effect_noise_multiplier"] == 2
    assert "every prompt: effect >= base_noise" in report["rule"]
    validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


@pytest.mark.parametrize("field,value", [("passed", False), ("model_name", "other"),
                                         ("identity", {"adapter": {}})])
def test_failed_or_mismatched_report_refused(bound_report, field, value):
    report, server = bound_report
    report[field] = value
    with pytest.raises(ValueError, match="passing"):
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")


def test_prompt_ids_must_match():
    def fake(request):
        response = fake_completion(request)
        data = response.json()
        data["choices"][0]["prompt_token_ids"] = [99]
        return httpx.Response(200, json=data)
    with httpx.Client(transport=httpx.MockTransport(fake)) as client:
        with pytest.raises(ValueError, match="alignment"):
            vllm_logprobs(client, "http://127.0.0.1/v1", "sft", cases()[0])


def test_fixed_selection_training_only():
    examples = [{"input_ids": [2, 3, 106], "prompt_tokens": 2, "packing": "turn",
                 "label_provenance": {"task_id": "train/task/1", "turn_index": i,
                                      "training_eligible": True}} for i in reversed(range(10))]
    assert [r["source"]["turn_index"] for r in fixed_prompts(examples)] == list(range(8))
    examples[-1]["label_provenance"]["training_eligible"] = False
    with pytest.raises(ValueError, match="training-eligible"):
        fixed_prompts(examples)
    with pytest.raises(ValueError, match="eight"):
        fixed_prompts(examples[:2])


def test_hf_only_writes_both_references_with_one_model_load(adapter_bank, tmp_path, monkeypatch):
    import tools.check_lora_serving as check
    path, snapshot, student = adapter_bank
    verified = tmp_path / "verified"
    verified.mkdir()
    write_json(verified / "manifest.json", {})
    examples = [{"input_ids": row["input_ids"], "prompt_tokens": row["prompt_tokens"],
                 "label_provenance": row["source"], "packing": "turn"} for row in cases()]
    loads, scores = [], []
    base = object()
    class FakePeft:
        disabled = False

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        @classmethod
        def from_pretrained(cls, loaded, adapter, **kwargs):
            assert loaded is base and adapter == str(path)
            assert kwargs == {"local_files_only": True, "is_trainable": False,
                              "autocast_adapter_dtype": False}
            return model

    model = FakePeft()
    def score(loaded, row):
        assert loaded is model
        scores.append((row["source"]["turn_index"], model.disabled))
        return [-1.29, -2.29] if model.disabled else [-1.0, -2.0]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: None)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeft))
    monkeypatch.setattr("train.modeling.load_student", lambda value: loads.append(value) or base)
    monkeypatch.setattr("train.dataset.load_tokenizer", lambda _: SimpleNamespace(chat_template="fake"))
    monkeypatch.setattr("train.dataset.build_verified_dataset", lambda *a, **kw: examples)
    monkeypatch.setattr(check, "hf_logprobs", score)
    output = tmp_path / "hf"
    assert check.main(["--snapshot", str(snapshot), "--adapter", str(path), "--verified", str(verified),
                       "--model-name", "sft", "--hf-only", "--output", str(output)]) == 0
    reference = json.loads((output / "reference.json").read_text())
    assert reference["schema_version"] == 5 and reference["prompts"] == cases()
    assert reference["provenance_sha256"] == digest(json.loads((output / "provenance.json").read_text()))
    assert loads == [str(snapshot)] and not model.disabled
    assert scores == [(i, disabled) for disabled in (False, True) for i in range(8)]
    assert not (output / "report.json").exists()


@pytest.mark.parametrize("lora_dtype", [None, "bfloat16", "float16"])
@pytest.mark.parametrize("scenario", ["pass", "prompt_failure", "pooled_error_failure",
                                      "pooled_effect_failure", "legacy_reference"])
def test_check_cli_json_report_and_exit_status(adapter_bank, tmp_path, monkeypatch, scenario, lora_dtype):
    import tools.check_lora_serving as check
    from sed.provenance import write_provenance
    path, snapshot, student = adapter_bank
    adapter = inspect_adapter("sft", path, student, snapshot)
    verified = tmp_path / "verified"
    verified.mkdir()
    write_json(verified / "manifest.json", {})
    identity = check.source_identity(SimpleNamespace(verified=str(verified), snapshot=str(snapshot)),
                                     load_config(), adapter)
    hf_dir = tmp_path / "hf"
    old = write_provenance(hf_dir, command="check_lora_serving", config={})
    rows = cases()
    if scenario == "legacy_reference":
        for row in rows:
            row.pop("hf_base_logprobs")
    if scenario == "pooled_effect_failure":
        for row in rows:
            row["hf_base_logprobs"] = [-1.09, -2.09]
    write_json(hf_dir / "reference.json", {"schema_version": 2, "identity": identity, "prompts": rows,
               "provenance_sha256": digest(old), "template_sha256": "fake-template"})
    server = {"config": {"server": {"lora_adapters": [adapter], "served_model_name": "base",
                                    "endpoint": "http://127.0.0.1:34567/v1", "merged_model": False}}}
    server_file = tmp_path / "server.json"
    if lora_dtype is not None:
        server["config"]["server"]["lora_dtype"] = lora_dtype
    write_json(server_file, server)
    prompt_index = 0
    def completion(request):
        nonlocal prompt_index
        assert scenario != "legacy_reference"  # Refuse incomplete references before serving calls.
        data = fake_completion(request).json()
        if scenario == "pooled_error_failure" and data["model"] == "sft":
            data["choices"][0]["prompt_logprobs"][2]["10"]["logprob"] = -1.5
        if scenario == "prompt_failure" and data["model"] == "base" and prompt_index == 0:
            data["choices"][0]["prompt_logprobs"][2]["10"]["logprob"] = -1.01
            data["choices"][0]["prompt_logprobs"][3]["106"]["logprob"] = -2.01
        if scenario == "pooled_effect_failure" and data["model"] == "base":
            data["choices"][0]["prompt_logprobs"][2]["10"]["logprob"] = -1.06
            data["choices"][0]["prompt_logprobs"][3]["106"]["logprob"] = -2.06
        if data["model"] == "base":
            prompt_index += 1
        return httpx.Response(200, json=data)
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        transport=httpx.MockTransport(completion), **kw))
    output = tmp_path / "check"
    status = check.main(["--snapshot", str(snapshot), "--adapter", str(path),
                        "--verified", str(verified), "--model-name", "sft",
                        "--endpoint", "http://127.0.0.1:34567/v1", "--server-provenance", str(server_file),
                        "--hf-reference", str(hf_dir / "reference.json"), "--output", str(output)])
    passes = scenario == "pass"
    assert status == (0 if passes else 1)
    assert (output / "provenance.json").exists()
    if scenario == "legacy_reference":
        assert not (output / "report.json").exists()
        assert json.loads((output / "failure.json").read_text()) == {"passed": False, "error_type": "ValueError"}
        return
    report = json.loads((output / "report.json").read_text())
    assert report["lora_dtype"] == (lora_dtype or "bfloat16")
    assert report["server_provenance_sha256"] == digest(server)
    assert report["passed"] is passes and len(report["prompts"]) == 8
    assert report["schema_version"] == 5 and report["thresholds"]["base_noise_multiplier"] == 1.5
    assert report["thresholds"]["min_effect_noise_multiplier"] == 2
    assert report["thresholds"]["outlier_count_multiplier"] == 2
    assert report["thresholds"]["outlier_count_allowance"] == 2
    assert report["thresholds"]["outlier_token_fraction"] == 0.05
    assert "every prompt: effect >= base_noise" in report["rule"]
    assert report["aggregate"]["adapter_error"]["tokens_abs_diff_gt_0_25"] == (
        8 if scenario == "pooled_error_failure" else 0)
    assert report["aggregate"]["passed"] is (scenario in {"pass", "prompt_failure"})
    assert all(p["passed"] for p in report["prompts"]) is (scenario != "prompt_failure")
    assert all("hf_base_logprobs" in row for row in report["prompts"])
    if passes:
        validate_report(report, server, "sft", "http://127.0.0.1:34567/v1")
