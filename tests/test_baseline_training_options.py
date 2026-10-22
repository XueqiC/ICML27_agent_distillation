"""User-approved LEAP dose extension and Kang training-seed identity semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baselines.common import pipeline as common
from baselines.kang import pipeline as kang
from baselines.leap import pipeline as leap_pipeline, train as leap
from sed.config import load_config
from sed.storage import digest, write_json


@pytest.mark.parametrize("updates", [199, 200, 201, 220, 239, 224])
def test_leap_final_epoch_past_official_step_with_provenance(tmp_path, monkeypatch, updates):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    torch.set_num_threads(1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    config = load_config()
    for key in ("manifest", "split_freeze"):
        config["paths"][key] = str(tmp_path / (key + ".json"))
        write_json(config["paths"][key], {})
    config_path, snapshot, verified = tmp_path / "config.json", tmp_path / "base", tmp_path / "verified"
    write_json(config_path, config)
    write_json(snapshot / "config.json", {})
    write_json(verified / "manifest.json", {})
    dataset = tmp_path / "examples.json"
    write_json(dataset, [])
    # Use the real dose, sampler, optimizer, scheduler, ledgers and publication.
    # Fake evidence/model scoring keeps this CPU test independent of any weights.
    n = 32 * updates - 3  # Also exercise the final partial optimizer batch.
    example = {"input_ids": [0, 1], "labels": [-100, 1], "label_provenance": {"source": "fake"}}
    monkeypatch.setattr(leap, "load_set", lambda *a, **kw: [example] * n)
    monkeypatch.setattr(leap, "encode_rows", lambda rows, tokenizer: rows)
    monkeypatch.setattr(leap, "load_tokenizer", lambda *a: SimpleNamespace(chat_template="fake"))
    monkeypatch.setattr(leap, "snapshot_metadata", lambda *a: {"source": "fake"})

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.peft_config = {"default": SimpleNamespace(revision=None)}
            self.saves = []

        def save_pretrained(self, path, **kwargs):
            self.saves.append(path)
            write_json(path / "fake_weights.json", {"weight": float(self.weight.detach())})

    model = FakeModel()
    monkeypatch.setattr(leap, "prepare_model", lambda *a, **kw: model)
    monkeypatch.setattr(leap, "batch_logps", lambda model, rows, device: -model.weight.square().expand(len(rows)))
    root = tmp_path / "run"
    output = root / "stages/train-0/attempt-000/output"
    leap.main(["--config", str(config_path), "--snapshot", str(snapshot), "--base-snapshot", str(snapshot),
        "--verified", str(verified), "--output", str(output), "--iteration", "0", "--variant", "zero",
        "--sets", str(dataset)], device="cpu")
    state = json.loads((output / "adapter/training_state.json").read_text())
    training = json.loads((output / "provenance.json").read_text())["config"]["leap_training"]
    assert state == json.loads((output / "result.json").read_text())
    for record in (state, training):
        assert record["total_steps"] == updates
        assert record["official_eval_save_step"] == 200
        assert record["updates_exceed_official_eval_save_step"] is (updates >= 200)
        assert "2026-10-06 user decision (b)" in record["eval_save_deviation"]
    assert state["updates"] == updates and state["examples_seen"] == n and state["final"]
    assert state["supervised_tokens_seen"] == n
    assert training["train_split"] == 1. and training["checkpoint_selection"] is False
    assert len(model.saves) == 1  # Final epoch only, including after update 200.
    assert model.weight.item() < 1
    ledger = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert len(ledger) == updates and ledger[-1]["global_batch"] == 29
    assert sorted(i for row in ledger for i in row["indices"]) == list(range(n))
    leap_pipeline.update_disclosure(root, {"K": 100, "unique_instances": 100, "task_groups": 100,
        "recipe": {"variant": "zero", "label": "LEAP (0% privileged ablation)"}})
    disclosure = json.loads((root / "disclosure.json").read_text())
    assert disclosure["eval_save_deviation"] == state["eval_save_deviation"]
    attempt = disclosure["training_attempts"][0]
    assert attempt["dose"]["total_steps"] == updates
    assert attempt["dose"]["updates_exceed_official_eval_save_step"] is (updates >= 200)
    # Preserve dose disclosure even if publication was interrupted after training.
    (output / "result.json").unlink()
    leap_pipeline.update_disclosure(root, {"K": 100, "unique_instances": 100, "task_groups": 100,
        "recipe": {"variant": "zero", "label": "LEAP (0% privileged ablation)"}})
    attempt = json.loads((root / "disclosure.json").read_text())["training_attempts"][0]
    assert not attempt["complete"] and attempt["dose"]["total_steps"] == updates


def test_kang_cli_seed_identity_provenance_and_strict_resume(tmp_path, monkeypatch):
    from baselines.common import fsdp_train
    config = load_config()
    for key in ("manifest", "split_freeze", "raw", "verified"):
        config["paths"][key] = str(tmp_path / key)
    for key in ("manifest", "split_freeze"):
        write_json(config["paths"][key], {})
    config_path, snapshot, verified = tmp_path / "config.json", tmp_path / "base", tmp_path / "verified"
    write_json(config_path, config)
    write_json(snapshot / "config.json", {})
    write_json(verified / "manifest.json", {})
    collection = {"K": 1, "unique_instances": 1, "task_groups": 1, "verified": [{"task_id": "train/task"}]}
    monkeypatch.setattr(common, "load_frozen", lambda *a: ({}, {}))
    monkeypatch.setattr(common, "load_verified_collection", lambda *a: (collection, {"K": 1, "seed": 0}, verified))
    monkeypatch.setattr(common, "checked_episode", lambda *a: {"episode_id": "seed"})
    monkeypatch.setattr(common, "training_tasks", lambda listing, *a: listing["task_ids"])
    # Legacy preparation has no training_seed argument and uses the unchanged recipe.
    output = tmp_path / "seed0"
    args = SimpleNamespace(config=str(config_path), snapshot=str(snapshot), verified=str(verified), output=output, gpus=1)
    common.prepare(args, config, method="kang", recipe=kang.RECIPE)
    original = (output / "identity.json").read_bytes()
    original_provenance = (output / "provenance.json").read_bytes()
    calls = []

    def fake_train(args):
        calls.append(args)
        write_json(Path(args.output) / "result.json", {"seed": args.seed})

    monkeypatch.setattr(fsdp_train, "train_lora", fake_train)
    monkeypatch.setattr(kang, "run_command", lambda command, log: fsdp_train.main(command[command.index("--config"):]))
    cli = ["--config", str(config_path), "--snapshot", str(snapshot), "--verified", str(verified)]
    kang.main([*cli, "--output", str(output)])
    kang.main([*cli, "--output", str(output), "--training-seed", "0"])
    assert len(calls) == 1 and calls[0].seed == 0  # The second call resumes the completed stage.
    assert (output / "identity.json").read_bytes() == original
    assert (output / "provenance.json").read_bytes() == original_provenance
    assert json.loads((output / "disclosure.json").read_text())["training_seed"] == 0
    with pytest.raises(ValueError, match="different frozen artifact"):
        kang.main([*cli, "--output", str(output), "--training-seed", "42"])
    second = tmp_path / "seed42"
    kang.main([*cli, "--output", str(second), "--training-seed", "42"])
    kang.main([*cli, "--output", str(second), "--training-seed", "42"])
    assert len(calls) == 2 and calls[1].seed == 42
    identity = json.loads((second / "identity.json").read_text())
    legacy = json.loads(original)
    assert identity["recipe"]["training_seed"] == 42
    expected = {**legacy, "recipe": {**legacy["recipe"], "training_seed": 42}}
    assert identity == expected and digest(identity) != digest(legacy)
    assert json.loads((second / "provenance.json").read_text())["config"]["kang"] == identity
    assert json.loads((second / "disclosure.json").read_text())["training_seed"] == 42
    assert (output / "stages/train-sft/complete.json").read_bytes() != (second / "stages/train-sft/complete.json").read_bytes()
    with pytest.raises(ValueError, match="different frozen artifact"):
        kang.main([*cli, "--output", str(second)])
    # A changed code tree still cannot reuse the old identity, even for seed 0.
    monkeypatch.setattr(common, "code_hashes", lambda *a: {**legacy["code_sha256"], "changed.py": "changed"})
    with pytest.raises(ValueError, match="different frozen artifact"):
        kang.main([*cli, "--output", str(output)])
    assert len(calls) == 2 and kang.RECIPE["training_seed"] == 0


@pytest.mark.parametrize("seed", [None, "0", "0042"])
def test_kang_launchers_forward_seed_unchanged(tmp_path, seed):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    for name in ("hpg_baseline_cells.slurm", "hpg_kang.slurm", "hpg_training_startup.sh"):
        (scripts / name).write_bytes((Path("scripts") / name).read_bytes())
    (scripts / "hpg_eval_adapters.slurm").write_text('#!/bin/bash\n"$ENV_PYTHON" --fake-eval\n')
    fake = tmp_path / "fake-python"
    fake.write_text("#!" + sys.executable + "\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-c":
    print(json.loads(Path(args[-1]).read_text())[0]["adapter"])
    sys.exit(0)
with open(os.environ["FAKE_LEDGER"], "a") as stream:
    stream.write(json.dumps({"args": args, "seed_env": os.environ.get("KANG_TRAINING_SEED")}) + "\\n")
if args[0] == "-m":
    output = Path(args[args.index("--output") + 1])
    output.mkdir(parents=True)
    (output / "models.json").write_text(json.dumps([{"adapter": str(output / "checkpoint")}]))
''')
    fake.chmod(0o755)
    gpu = tmp_path / "nvidia-smi"
    gpu.write_text("#!/bin/sh\necho fake-cpu-test\n")
    gpu.chmod(0o755)
    data = tmp_path / "collections"
    write_json(data / "data/verified/seeds_k20_s0/manifest.json", {})
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    env = os.environ.copy()
    env.pop("KANG_TRAINING_SEED", None)
    if seed is not None:
        env["KANG_TRAINING_SEED"] = seed
    ledger = tmp_path / "ledger.jsonl"
    env.update(PROJECT_ROOT=str(project), HPG_CONFIG=str(Path("configs/hpg.env").resolve()),
        ENV_PYTHON=str(fake), TRAIN_PYTHON=str(fake), MODEL_PATH=str(snapshot), ENABLE_THINKING="off",
        MAX_TOKENS="1024", MAX_LENGTH="32768", BATCH_INVARIANT="1", METHOD="kang", CELLS="k20s0",
        DATA_ROOT=str(data), SLURM_ARRAY_TASK_ID="0", FAKE_LEDGER=str(ledger), PATH=str(tmp_path) + ":" + env["PATH"])
    result = subprocess.run(["bash", str(scripts / "hpg_baseline_cells.slurm")], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    training, evaluation = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert training["seed_env"] == evaluation["seed_env"] == seed
    args = training["args"]
    assert args[:2] == ["-m", "baselines.kang.pipeline"]
    assert args[args.index("--training-seed") + 1] == (seed or "0")
    assert args[args.index("--output") + 1] == str(data / "reports/runs/baselines/kang-k20s0-tp")
    assert evaluation["args"] == ["--fake-eval"]
