"""Offline CPU preparation, identical train validation, and allocation guards."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baselines.common import fsdp_train, pipeline as common
from sed import hash_receipts, seed_evidence
from sed.storage import write_json
from test_full_train import full_bank
from test_full_train_views import source_bank
from test_sad import CharTokenizer

METHODS = ["kang", "adftp", "smartad"]
GPU_ENV = {"SLURM_JOB_GPUS": "0", "SLURM_STEP_GPUS": "0", "SLURM_GPUS_ON_NODE": "1",
           "SLURM_GPUS": "1", "SLURM_GPUS_PER_NODE": "b200:1", "SLURM_GPUS_PER_TASK": "1",
           "CUDA_VISIBLE_DEVICES": "0"}


@pytest.fixture(autouse=True)
def cpu_allocation(monkeypatch):
    for key in GPU_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("gpu_key", GPU_ENV)
def test_pipeline_prepare_refuses_gpu_before_io(method, gpu_key, monkeypatch, tmp_path):
    monkeypatch.setenv(gpu_key, GPU_ENV[gpu_key])
    module = importlib.import_module(f"baselines.{method}.pipeline")
    argv = ["--mode", "prepare", "--snapshot", "absent", "--verified", "absent", "--output", str(tmp_path / "out")]
    if method != "kang":
        argv += ["--through", "train"]
    with pytest.raises(ValueError, match="must not hold a GPU"):
        module.main(argv)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("gpu_key", [None, *GPU_ENV])
def test_slurm_prepare_cpu_and_gpu_guard(tmp_path, method, gpu_key):
    fake = tmp_path / "python"
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                    'open(os.environ["ARGS"],"w").write(json.dumps({"argv":sys.argv[1:],'
                    '"cuda":os.environ["CUDA_VISIBLE_DEVICES"],"cache":os.environ["SED_HASH_CACHE_DIR"]}))\n')
    fake.chmod(0o755)
    env = {**os.environ, "PROJECT_ROOT": str(Path.cwd()), "HPG_CONFIG": str(Path("configs/hpg.env").resolve()),
           "ENV_PYTHON": str(fake), "TRAIN_PYTHON": str(fake), "VLLM_PYTHON": str(fake),
           "MODEL_PATH": str(tmp_path), "VERIFIED": "input", "ARGS": str(tmp_path / "args.json"),
           "NGPUS": "4", method.upper() + "_OUT": str(tmp_path / "out"),
           method.upper() + "_MODE": "prepare", method.upper() + "_THROUGH": "train"}
    if gpu_key:
        env[gpu_key] = GPU_ENV[gpu_key]
    result = subprocess.run(["bash", f"scripts/hpg_{method}.slurm"], env=env, capture_output=True, text=True)
    if gpu_key:
        assert result.returncode != 0 and "must not hold a GPU" in result.stderr
        assert not (tmp_path / "args.json").exists()
    else:
        assert result.returncode == 0, result.stderr
        row = json.loads((tmp_path / "args.json").read_text())
        argv = row["argv"]
        assert argv[argv.index("--mode") + 1] == "prepare"
        assert argv[argv.index("--gpus") + 1] == "4"  # Planned world size, not allocated GPUs.
        assert row["cuda"] == "" and row["cache"] == env["SED_HASH_CACHE_DIR"]


@pytest.mark.parametrize("method", METHODS)
def test_prepare_warms_full_bindings_and_train_revalidates(source_bank, tmp_path, monkeypatch, method):
    from baselines.adftp import collect as ftp
    from baselines.smartad import collect as sm
    from baselines.smartad.select import select
    from test_adftp import stub_factory
    from test_smartad import factory

    root, _, manifest, freeze, config, env = source_bank
    seed_evidence.materialize_full_train_views(root, manifest, freeze, [1, 2])
    config["environment"]["max_episode_steps"] = 40
    snapshot = tmp_path / "snapshot"
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified"})
    (snapshot / "weights.safetensors").write_bytes(b"fake snapshot bytes")
    kwargs = dict(rate_limit_dir=tmp_path / "limit", env_factory=env, subset_views=[f"1={root / 'subsets/k1'}"])
    verified = root / "subsets/k1"
    if method == "adftp":
        selected = ftp.collect(root / "subsets/k2", manifest, freeze, config, tmp_path / "ftp",
                               teacher_factory=stub_factory([]), **kwargs)
        verified = selected / "subsets/k1"
    elif method == "smartad":
        pool = sm.collect(root / "subsets/k2", manifest, freeze, config, tmp_path / "pool",
                          teacher_factory=factory([]), **kwargs)
        selected = select(pool, snapshot, config, manifest, freeze, tmp_path / "selected",
                          scorer=lambda _: {"nll": 1., "turns": [{"nll_mean": 1.}]})
        verified = selected / "subsets/k1"
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    monkeypatch.setattr(fsdp_train, "load_tokenizer", lambda _: CharTokenizer())
    module = importlib.import_module(f"baselines.{method}.pipeline")
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    commands = []
    def run_cpu(command, log):
        commands.append(command)
        assert "torch.distributed.run" not in command
        fsdp_train.main(command[command.index("baselines.common.fsdp_train") + 1:])
    monkeypatch.setattr(common, "run_command", run_cpu)
    output = tmp_path / "run"
    argv = ["--config", str(config_path), "--snapshot", str(snapshot), "--verified", str(verified),
            "--output", str(output)]
    if method != "kang":
        argv += ["--through", "train"]
    # Remove both kinds of warm state from fixture construction.
    import shutil
    shutil.rmtree(hash_receipts.cache_directory(), ignore_errors=True)
    hash_receipts._MEMO.clear()
    module.main([*argv, "--mode", "prepare"])
    training = output if method == "kang" else output / "training"
    identity = (training / "identity.json").read_bytes()
    assert len(commands) == 1
    assert not (training / "stages").exists() and not (output / "models.json").exists()
    prepared = next(training.glob("preparations/*/output/result.json"))
    assert json.loads(prepared.read_text())["training_updates"] == 0
    provenance = json.loads(prepared.with_name("provenance.json").read_text())
    events = [json.loads(line) for line in Path(provenance["startup_timing"]["path"]).read_text().splitlines()]
    assert {"collection_load", "encode", "snapshot_hashing", "data_prepare"} <= {r["phase"] for r in events}
    assert all(r["seconds"] >= 0 for r in events)

    # Simulate a new allocation. Preparation is not a validation bypass token.
    hash_receipts._MEMO.clear()
    calls = []
    original = seed_evidence.file_hash
    def counted(path):
        calls.append(Path(path))
        return original(path)
    monkeypatch.setattr(seed_evidence, "file_hash", counted)
    class ModelReached(Exception):
        pass
    def train_to_model(command, log):
        assert "--prepare-only" not in command and "torch.distributed.run" in command
        fsdp_train.main(command[command.index("baselines.common.fsdp_train") + 1:])
    # Fake device setup only; stop before any model or CUDA tensor allocation.
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, set_device=lambda _: None),
                            device=lambda *a: "cuda", distributed=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch.distributed)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=lambda _: None))
    monkeypatch.setattr(fsdp_train, "prepare_lora_model", lambda *a, **kw: (_ for _ in ()).throw(ModelReached()))
    monkeypatch.setattr(module, "run_command", train_to_model)
    with pytest.raises(ModelReached):
        module.main([*argv, "--mode", "train"])
    assert (training / "identity.json").read_bytes() == identity
    assert all(p.name == "receipt.json" for p in calls), calls
    trainer_argv = commands[0][commands[0].index("baselines.common.fsdp_train") + 1:]
    trainer_argv = [arg for arg in trainer_argv if arg != "--prepare-only"]
    if method == "smartad":
        weights = snapshot / "weights.safetensors"
        original_weights = weights.read_bytes()
        weights.write_bytes(original_weights + b"changed")
        for extra in ([], ["--prepare-only"]):
            with pytest.raises(ValueError, match="selection/student binding changed"):
                fsdp_train.main([*trainer_argv, *extra])
        weights.write_bytes(original_weights)

    # Even an unselected trajectory in the full support bank remains checked.
    collection = json.loads((root / "manifest.json").read_text())
    selected_ids = {r["task_id"] for r in seed_evidence.load_verified_collection(verified, manifest, freeze)[0]["verified"]}
    unselected = next(r for r in collection["verified"] if r["task_id"] not in selected_ids)
    victim = root / unselected["episode_dir"] / "episode.json"
    victim.write_text(victim.read_text() + " ")
    for mode in ("prepare", "train"):
        with pytest.raises(ValueError, match="evidence changed"):
            module.main([*argv, "--mode", mode])
    for extra in ([], ["--prepare-only"]):
        with pytest.raises(ValueError, match="evidence changed"):
            fsdp_train.main([*trainer_argv, *extra])


def test_timing_first_update_is_logged_after_cpu_optimizer_step(tmp_path):
    torch = pytest.importorskip("torch")
    from test_smartad_torch import Tiny
    from sed.startup_timing import startup_timing
    model = Tiny()
    row = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    with startup_timing(tmp_path / "startup.jsonl", process="trainer", mode="train", snapshot=tmp_path):
        result = fsdp_train.train_loop(model, [row], tmp_path / "output", objective="sft", learning_rate=2e-4,
            scheduler="linear", warmup_ratio=0, warmup_steps=0, original_batch=8, epochs=2)
    events = [json.loads(line) for line in (tmp_path / "startup.jsonl").read_text().splitlines()]
    first, = [r for r in events if r["phase"] == "first_update"]
    assert first["elapsed_seconds"] >= first["seconds"] >= 0
    assert result["updates"] == 2
