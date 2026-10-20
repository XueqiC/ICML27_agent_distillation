"""CPU numerical accounting, ladder stopping, frozen overrides, and launchers."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baselines.common import pipeline as common
from baselines.common.schedule import dose, lr_multiplier, warmup_count
from baselines.sad.data import token_plan, TOKEN_BUDGET
from baselines.sad.pipeline import processed_budget
from sed.config import load_config
from sed.storage import digest, write_json
from train.convergence import doubling_passes
from train.sft import exposure_plan


@pytest.mark.parametrize("method", ["kang", "adftp", "smartad", "sad"])
def test_amount_binds_pipeline_identity_and_command(tmp_path, monkeypatch, method):
    pipeline = importlib.import_module(f"baselines.{method}.pipeline")
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
    if method == "smartad":
        for name in ("validate_selection", "validate_h0"):
            monkeypatch.setattr(pipeline, name, lambda *a: None)
        monkeypatch.setattr(pipeline, "load_frozen", lambda *a: ({}, {}))
    args = ["--config", str(config_path), "--snapshot", str(snapshot), "--verified", str(verified), "--mode", "prepare"]
    if method in {"adftp", "smartad"}:
        args += ["--through", "train"]
    option, value = ("--budget-multiplier", ".5") if method == "sad" else ("--epochs", "4")
    first, second = tmp_path / "first", tmp_path / "second"
    pipeline.main([*args, "--output", str(first)])
    pipeline.main([*args, "--output", str(first)])
    with pytest.raises(ValueError, match="different frozen artifact"):
        pipeline.main([*args, "--output", str(first), option, value])
    pipeline.main([*args, "--output", str(second), option, value])
    suffix = Path("training/identity.json") if method in {"adftp", "smartad"} else Path("identity.json")
    old, new = (json.loads((root / suffix).read_text()) for root in (first, second))
    key, expected = ("processed_token_budget", TOKEN_BUDGET // 2) if method == "sad" else ("epochs", 4)
    assert new["recipe"] == {**old["recipe"], key: expected}
    assert digest(new) != digest(old)
    command_args = SimpleNamespace(train_python="python", config="config", snapshot="base", verified="verified",
                                  gpus=1, scoring="shared_history", epochs=4, budget_multiplier=.5)
    command = pipeline.training_command(command_args, "out")
    flag = "--processed-token-budget" if method == "sad" else "--epochs"
    assert command[command.index(flag) + 1] == str(expected)
    assert old["recipe"][key] == (TOKEN_BUDGET if method == "sad" else 2)


def test_amount_stretches_schedules_and_sad_checkpoint_fractions():
    assert processed_budget(1) == TOKEN_BUDGET
    assert processed_budget(1.1) == TOKEN_BUDGET * 11 // 10
    old, new = (dose(16, 8, epochs=e) for e in (2, 4))
    assert new["total_steps"] == 2 * old["total_steps"]
    for step in range(old["total_steps"] + 1):
        assert lr_multiplier(step, scheduler="linear", total_steps=old["total_steps"], warmup_ratio=0) == \
               lr_multiplier(2 * step, scheduler="linear", total_steps=new["total_steps"], warmup_ratio=0)
    examples = [{"input_ids": [0, 1], "labels": [-100, 1]}] * 16
    plans = [token_plan(examples, token_budget=processed_budget(m)) for m in (.5, 1., 2.)]
    assert [p["processed_token_budget"] for p in plans] == [327500000, 655000000, 1310000000]
    for p in plans:
        assert p["checkpoint_tokens"][-1] == p["processed_token_budget"]
        assert p["checkpoint_tokens"][0] == (p["processed_token_budget"] + 15) // 16
    assert warmup_count(plans[-1]["total_steps"], .03) > warmup_count(plans[1]["total_steps"], .03)
    for steps in (100, 200):
        assert warmup_count(steps, .03) == steps * .03
        assert lr_multiplier(steps, scheduler="cosine", total_steps=steps, warmup_ratio=.03) == 0
    assert lr_multiplier(50, scheduler="cosine", total_steps=100, warmup_ratio=.03) == \
           lr_multiplier(100, scheduler="cosine", total_steps=200, warmup_ratio=.03)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 1e-12])
def test_invalid_budget_multiplier(value):
    with pytest.raises(ValueError):
        processed_budget(value)


def test_trajectory_cli_connects_frozen_plan_identity_and_outcome(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import train.sft as sft
    from test_smartad_torch import Tiny

    class Model(Tiny):
        def save_pretrained(self, path, **kwargs):
            write_json(path / "weights.json", {"fake": True})

    config = {"training": {"thinking_mode": "off", "student_step_token_cap": 1024},
              "paths": {"manifest": "unused", "split_freeze": "unused"}}
    row = {"input_ids": [0, 1, 2], "labels": [-100, 1, 2], "supervised_tokens": 2,
           "turn_count": 1, "label_provenance": {"source": "fake_train_only"}}
    tokenizer = SimpleNamespace(chat_template="fake", save_pretrained=lambda path: None)
    monkeypatch.setattr(sft, "load_config", lambda path: config)
    monkeypatch.setattr(sft, "snapshot_metadata", lambda path: {"path": "fake"})
    monkeypatch.setattr(sft, "deterministic_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft, "load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(sft, "load_frozen", lambda *a: ({}, {}))
    monkeypatch.setattr(sft, "build_verified_dataset", lambda *a, **kw: [row])
    monkeypatch.setattr(sft, "mask_report", lambda *a: {"supervised_tokens_per_pass": 2,
                                                      "estimated_forward_tokens_per_pass": 3})
    monkeypatch.setattr(sft, "verified_collection_inputs", lambda *a: [])
    def provenance(output, **kwargs):
        record = {"config": kwargs["config"]}
        write_json(output / "provenance.json", record)
        return record
    monkeypatch.setattr(sft, "write_provenance", provenance)
    def load_student(path):
        # The production gate above is stubbed; all actual training stays CPU.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        return Model()
    monkeypatch.setattr(sft, "load_student", load_student)
    monkeypatch.setattr(sft, "attach_lora", lambda model, **kw: model)
    output = tmp_path / "run"
    sft.main(["--snapshot", "fake", "--verified", "fake", "--output", str(output),
              "--passes-ladder", "--max-passes", "4"])
    frozen = json.loads((output / "provenance.json").read_text())
    recipe = frozen["config"]["sft"]
    assert recipe["rank"] == 16 and recipe["learning_rate"] == 1e-5 and recipe["scheduler"] == "constant"
    assert recipe["convergence"]["rungs"] == [1, 2, 4]
    assert recipe["plan_sha256"] == digest(json.loads((output / "exposure_plan.json").read_text()))
    result = json.loads((output / "result.json").read_text())
    assert result["identity_sha256"] == digest(frozen)
    assert result["convergence"]["reported_passes"] == 2
    assert result["supervised_tokens"] == 4


@pytest.mark.parametrize("stop_at", [2, 4, None])
def test_trajectory_ladder_stops_at_first_endpoint_and_records_last_pass(tmp_path, monkeypatch, stop_at):
    torch = pytest.importorskip("torch")
    import train.sft as sft

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def save_pretrained(self, path, **kwargs):
            write_json(path / "weights.json", {"weight": self.weight.item()})

    rows = [{"input_ids": [0, 1], "labels": [-100, 1], "supervised_tokens": 1},
            {"input_ids": [0, 1, 2, 3], "labels": [-100, 1, 2, 3], "supervised_tokens": 3}]
    cap = 8 if stop_at else 4
    plan = exposure_plan([1, 3], token_budget=1, passes=cap, packing="episode", flush_passes=True,
                         checkpoint_tokens=[4 * p for p in doubling_passes(cap)])
    # Pass 3 is deliberately unlike pass 4: averaging the whole interval is wrong.
    means = [2.5, 2.499, 100, 2.498] if stop_at == 2 else [2.5, 2.48, 100, 2.475 if stop_at == 4 else 2.0]
    calls = []

    def loss(model, row, device):
        epoch = len(calls) // 2
        calls.append(row)
        # Distinct token losses within a pass expose unweighted update averaging.
        value = means[epoch] + (-1.5 if row["supervised_tokens"] == 1 else .5)
        return model.weight * 0 + value * row["supervised_tokens"]

    monkeypatch.setattr(sft, "turn_loss_sum", loss)
    saved = sft.train_loop(Model(), rows, plan, tmp_path, identity_sha256="frozen",
                           passes_ladder=True, converge_delta=.01)
    endpoint = stop_at or cap
    assert len(calls) == 2 * endpoint
    assert [c["passes"] for c in saved] == [1, 2] + ([4] if endpoint == 4 else [])
    assert saved[-1]["final"]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result == json.loads((tmp_path / "convergence_provenance.json").read_text())
    convergence = result["convergence"]
    assert convergence["status"] == ("converged" if stop_at else "cap_reached")
    assert convergence["reported_passes"] == stop_at
    decision = convergence["decisions"][-1]
    assert decision["L_last_prev"] == pytest.approx(means[endpoint // 2 - 1])
    assert decision["L_last"] == pytest.approx(means[endpoint - 1])
    assert decision["converged"] is bool(stop_at)
    assert result["identity_sha256"] == "frozen"
    assert convergence["reported_checkpoint"] == (saved[-1]["path"] if stop_at else None)
    from tools.convergence_check import last_epoch
    # train_loop is called directly; the CLI normally writes this dataset metadata.
    write_json(tmp_path / "example_sources.json", [{"source": "fake_train_only"}] * len(rows))
    assert last_epoch(tmp_path)["L_last"] == pytest.approx(means[endpoint - 1])


@pytest.mark.parametrize("objective", ["sft", "smartad", "sad"])
def test_baseline_records_true_token_numerator_without_changing_objective(tmp_path, objective):
    torch = pytest.importorskip("torch")
    from baselines.common.fsdp_train import train_loop
    from baselines.common.schedule import epoch_batches
    from test_smartad_torch import Tiny
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = Tiny()
    short = {"input_ids": [1, 2], "labels": [-100, 2], "token_weights": [0, 2.]}
    long = {"input_ids": [3, 2, 4, 5], "labels": [-100, 2, 4, 5], "token_weights": [0, 1., 1.5, 2.]}
    examples = [{"packing": "turn", "turns": [r]} for r in [short, long] * 8]
    batch = [0, 1] if objective == "sad" else next(epoch_batches(16, 2, epochs=1, seed=0))["indices"]
    expected = 0.
    with torch.no_grad():
        for i in batch:
            row = examples[i]["turns"][0]
            ids = torch.tensor([row["input_ids"]])
            logits = model(ids, torch.ones_like(ids)).logits[0, :-1]
            ce = torch.nn.functional.cross_entropy(logits, torch.tensor(row["labels"][1:]), reduction="none")
            weights = torch.tensor(row["token_weights"][1:]) if objective == "smartad" else 1
            expected += (ce * weights).sum().item()
    state = train_loop(model, examples, tmp_path, objective=objective, learning_rate=1e-5,
        scheduler="linear", warmup_ratio=0, original_batch=8, epochs=1, packing="turn", scoring="per_turn",
        **({"processed_token_budget": 6} if objective == "sad" else {}))
    first = json.loads((tmp_path / "updates.jsonl").read_text().splitlines()[0])
    assert first["supervised_loss_sum"] == pytest.approx(expected, rel=1e-6)
    assert first["objective"] == objective and first["global_batch"] == 2
    assert first["supervised_tokens_seen"] == sum(len(examples[i]["turns"][0]["labels"]) - 1 for i in batch)
    assert state["updates"] >= 1


@pytest.mark.parametrize("method,flag,env_name,value", [
    ("kang", "--epochs", "KANG_EPOCHS", "4"), ("adftp", "--epochs", "ADFTP_EPOCHS", "8"),
    ("smartad", "--epochs", "SMARTAD_EPOCHS", "16"), ("sad", "--budget-multiplier", "SAD_BUDGET_MULT", "0.5"),
    ("train_sft", "--max-passes", "TRAIN_MAX_PASSES", "32")])
def test_slurm_amount_wiring(tmp_path, method, flag, env_name, value):
    fake = tmp_path / "python"
    fake.write_text("#!" + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    verified, base = tmp_path / "verified", tmp_path / "base"
    write_json(verified / "manifest.json", {})
    base.mkdir()
    env = {"PATH": os.defpath, "PROJECT_ROOT": str(Path.cwd()), "HPG_CONFIG": str(Path("configs/hpg.env").resolve()),
           "MODEL_PATH": str(base), "VERIFIED": str(verified), "ENV_PYTHON": str(fake), "TRAIN_PYTHON": str(fake),
           "SLURM_GPUS_ON_NODE": "1", "TRAIN_PASSES_LADDER": "1", "TRAIN_OUT": str(tmp_path / "run"),
           method.upper() + "_OUT": str(tmp_path / "run"), method.upper() + "_THROUGH": "train", env_name: value}
    script = f"scripts/hpg_{method}.slurm"
    subprocess.run(["bash", "-n", script], check=True, env=env)
    result = subprocess.run(["bash", script], check=True, env=env, capture_output=True, text=True)
    argv = json.loads(result.stdout)
    assert argv[argv.index(flag) + 1] == value
    if method == "train_sft":
        assert "--passes-ladder" in argv and "--passes" not in argv
        assert argv[argv.index("--converge-delta") + 1] == "0.01"
        assert argv[argv.index("--verified") + 1] == str(verified)
        assert argv[argv.index("--output") + 1] == str(tmp_path / "run")
