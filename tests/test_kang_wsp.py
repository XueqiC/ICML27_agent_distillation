"""Optional Kang controls: offline defaults, seal binding and tiny CPU updates."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.common import fsdp_train as trainer
from baselines.kang import pipeline
from sed.storage import canonical_bytes, digest


def default_observation(tmp_path, monkeypatch, options=()):
    """Golden captured on the untouched tree; variable provenance is fixed."""
    torch = pytest.importorskip("torch")
    from test_stage2b_torch import tiny_model
    torch.set_num_threads(1)
    captured = {}
    args = SimpleNamespace(train_python="TRAIN", config="CONFIG", snapshot="BASE", verified="SEEDS")
    command = pipeline.training_command(args, "OUT") + list(options)
    with monkeypatch.context() as m:
        m.setattr(trainer, "train_lora", lambda args: captured.update(args=vars(args).copy()))
        trainer.main(command[command.index("baselines.common.fsdp_train") + 1:])
    with monkeypatch.context() as m:
        m.setattr(pipeline, "load_config", lambda _: {})
        def prepare(args, config, *, method, recipe):
            captured["recipe"] = recipe
            return {"method": method, "recipe": recipe, "code_sha256": {"fixed.py": "fixed"}}
        m.setattr(pipeline, "prepare", prepare)
        m.setattr(pipeline, "train_pipeline", lambda args, identity: captured.update(cell=digest([identity, "sft"])))
        m.setattr(pipeline, "update_disclosure", lambda *a: None)
        m.setattr(pipeline.signal, "signal", lambda *a: None)
        pipeline.main(["--snapshot", "BASE", "--verified", "SEEDS", "--output", str(tmp_path / "pipeline"), *options])
    torch.manual_seed(17)
    with monkeypatch.context() as m:
        m.setattr(trainer, "load_student", lambda *a, **kw: tiny_model())
        from train.lora_options import shape
        model = trainer.prepare_lora_model("BASE", device="cpu",
                    **(shape(SimpleNamespace(**captured["args"])) if options else {}))
    target_names = sorted(model.peft_config["default"].target_modules)
    # Exercise the production provenance builder without a GPU/model load.
    with monkeypatch.context() as m:
        from baselines.common import prepared
        m.setattr(torch.cuda, "is_available", lambda: True)
        m.setattr(torch.cuda, "set_device", lambda _: None)
        m.setattr(torch, "device", lambda *a: "cpu")
        m.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
        config = {"models": {"student": {"revision": "fixed"}},
                  "paths": {"manifest": "MANIFEST", "split_freeze": "SPLIT"}}
        collection = {"K": 1, "unique_instances": 1, "task_groups": 1}
        m.setattr(prepared, "load_prepared", lambda args: ((config, collection, [{"label_provenance": {}}], "template"), {"seal": "fixed"}))
        m.setattr(trainer, "prepare_lora_model", lambda *a, **kw: model)
        m.setattr(trainer, "snapshot_metadata", lambda _: {"snapshot": "fixed"})
        m.setattr(trainer, "artifact_hashes", lambda _: {"weights": "fixed"})
        def provenance(output, **kw):
            captured["config"] = deepcopy(kw["config"])
            return {"config": kw["config"]}
        m.setattr(trainer, "write_provenance", provenance)
        m.setattr(trainer, "train_loop", lambda *a, **kw: {"epochs": 3, "supervised_tokens_seen": 2})
        def export(model, path, provenance, state):
            path.mkdir()
            return state
        m.setattr(trainer, "export_adapter", export)
        train_args = SimpleNamespace(**{**captured["args"], "output": str(tmp_path / "production")})
        trainer.train_lora(train_args)
        captured["config"]["baseline_training"]["output"] = "OUT"
    rows = [{"input_ids": [2, 7, 8, 106], "labels": [-100, -100, 8, 106]}]
    state = trainer.train_loop(model, rows, tmp_path / "loop", objective="sft", learning_rate=2e-4,
        scheduler="linear", warmup_ratio=0, warmup_steps=0, original_batch=8, epochs=2, seed=3)
    trainer.export_adapter(model, tmp_path / "adapter", {"fixture": "fixed"}, state)
    tensors = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            tensors.update(name.encode())
            tensors.update(parameter.detach().numpy().tobytes())
    return {"RECIPE": pipeline.RECIPE, "command": command, **captured,
            "targets": target_names, "trained_tensors_sha256": tensors.hexdigest(),
            "adapter_file_sha256": hashlib.sha256((tmp_path / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()}


def test_default_bytes_pinned(tmp_path, monkeypatch):
    golden = Path(__file__).with_name("fixtures") / "kang_wsp_default.json"
    assert canonical_bytes(default_observation(tmp_path, monkeypatch)) == canonical_bytes(json.loads(golden.read_text()))


@pytest.mark.parametrize("options,expected", [
    (["--lora-rank", "4"], {"rank": 4, "alpha": 8}),
    (["--lora-alpha", "32"], {"rank": 64, "alpha": 32}),
    (["--lora-targets", "down_proj,q_proj", "--lora-layers", "0-1,4"],
     {"targets": ["q_proj", "down_proj"], "lora_layers": [0, 1, 4]}),
    (["--lora-modules", "22.down_proj,1.q_proj"],
     {"targets": ["q_proj", "down_proj"], "lora_modules": ["1.q_proj", "22.down_proj"]}),
    (["--checkpoint-updates", "1", "4"], {"checkpoint_updates": [1, 4]})])
def test_options_bind_pipeline_recipe_command_and_trainer(tmp_path, monkeypatch, options, expected):
    captured = {}
    def prepare(args, config, *, method, recipe):
        captured.update(args=args, recipe=recipe)
        return {"recipe": recipe}
    monkeypatch.setattr(pipeline, "prepare", prepare)
    monkeypatch.setattr(pipeline, "load_config", lambda _: {})
    monkeypatch.setattr(pipeline, "train_pipeline", lambda *a: None)
    monkeypatch.setattr(pipeline, "update_disclosure", lambda *a: None)
    monkeypatch.setattr(pipeline.signal, "signal", lambda *a: None)
    pipeline.main(["--snapshot", "BASE", "--verified", "SEEDS", "--output", str(tmp_path), *options])
    assert all(captured["recipe"][k] == v for k, v in expected.items())
    assert digest(captured["recipe"]) != digest(pipeline.RECIPE)
    for key in ("lora_layers", "lora_modules", "checkpoint_updates"):
        assert (key in captured["recipe"]) == (key in expected)
    command = pipeline.training_command(captured["args"], "OUT")
    received = []
    monkeypatch.setattr(trainer, "train_lora", received.append)
    trainer.main(command[command.index("baselines.common.fsdp_train") + 1:])
    from baselines.common.prepared import check_trainer_identity
    check_trainer_identity({"method": "kang", "recipe": captured["recipe"]}, received[0])
    from train.lora_options import shape
    actual = shape(received[0])
    assert actual["rank"] == captured["recipe"]["rank"] and actual["alpha"] == captured["recipe"]["alpha"]


@pytest.mark.parametrize("options", [["--lora-rank", "0"], ["--lora-rank", "257"], ["--lora-alpha", "0"],
    ["--lora-targets", ""], ["--lora-targets", "bogus"], ["--lora-layers", "2-1"],
    ["--lora-layers", "-1"], ["--lora-modules", "1.bogus"], ["--lora-modules", "no-layer.q_proj"],
    ["--lora-modules", "1.q_proj", "--lora-targets", "q_proj"],
    ["--lora-modules", "1.q_proj", "--lora-layers", "1"],
    ["--checkpoint-updates", "0"], ["--checkpoint-updates", "2", "1"], ["--checkpoint-updates", "1", "1"]])
def test_bad_options_are_parser_errors(tmp_path, options):
    command = pipeline.training_command(SimpleNamespace(train_python="TRAIN", config="CONFIG",
                                        snapshot="BASE", verified="SEEDS"), "OUT")
    for main, argv in ((pipeline.main, ["--snapshot", "BASE", "--verified", "SEEDS", "--output", str(tmp_path)]),
                       (trainer.main, command[command.index("baselines.common.fsdp_train") + 1:])):
        with pytest.raises(SystemExit) as error:
            main([*argv, *options])
        assert error.value.code == 2


@pytest.mark.parametrize("objective,adaptation", [("sft", "full"), ("sft", "wkm_lora"), ("smartad", "kang_lora"), ("dpo", "kang_lora")])
def test_controls_illegal_for_other_recipes(objective, adaptation, capsys):
    command = pipeline.training_command(SimpleNamespace(train_python="TRAIN", config="CONFIG", snapshot="BASE", verified="SEEDS"), "OUT")
    argv = command[command.index("baselines.common.fsdp_train") + 1:]
    argv[argv.index("--objective") + 1] = objective
    argv[argv.index("--adaptation") + 1] = adaptation
    with pytest.raises(SystemExit):
        trainer.main([*argv, "--checkpoint-updates", "1"])
    assert "require --objective sft --adaptation kang_lora" in capsys.readouterr().err


def test_attach_filters_and_missing_modules():
    torch = pytest.importorskip("torch")
    from train.modeling import attach_lora, LORA_TARGETS
    from test_stage2b_torch import tiny_model
    model = tiny_model()
    model.model.vision_tower = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in LORA_TARGETS})
    adapted = attach_lora(model, targets=["q_proj", "down_proj"], layers=iter([1]), gradient_checkpointing=False)
    names = adapted.peft_config["default"].target_modules
    assert len(names) == 2 and all(".language_model.layers.1." in name for name in names)
    adapted = attach_lora(tiny_model(), modules=iter(["0.q_proj", "1.down_proj"]), gradient_checkpointing=False)
    assert len(adapted.peft_config["default"].target_modules) == 2
    for kwargs, missing in [({"layers": [7]}, "q_proj"), ({"modules": ["7.down_proj"]}, "7.down_proj"),
                           ({"modules": ["0.q_proj"], "targets": ["q_proj"]}, "mutually exclusive")]:
        with pytest.raises(ValueError, match=missing):
            attach_lora(tiny_model(), **kwargs)
    model = tiny_model()
    del model.model.language_model.layers[1].self_attn.k_proj
    del model.model.language_model.layers[1].self_attn.v_proj
    with pytest.raises(ValueError, match="1.k_proj"):
        attach_lora(model, modules=["1.k_proj"])
    with pytest.raises(ValueError, match="k_proj"):
        attach_lora(model, layers=[1], targets=["k_proj"])


def test_intermediate_exports_preserve_rng_updates_and_final_bytes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import random
    import numpy as np
    from test_stage2b_torch import tiny_model
    from train.modeling import attach_lora
    torch.set_num_threads(1)
    rows = [{"packing": "turn", "turns": [{"input_ids": [2, 7, 8, 106], "labels": [-100, -100, 8, 106]}]},
            {"packing": "turn", "turns": [{"input_ids": [2, 6, 106], "labels": [-100, 6, 106]}]}]
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    student = {"requested_repo": "fixture", "revision": "fixed"}
    provenance = {"config": {"models": {"student": student}, "sft": {"base_snapshot": {
        "path": "", "metadata_sha256": {"config.json": hashlib.sha256(b"{}").hexdigest()}}}}}
    original_export = trainer.export_adapter
    checkpoints = []
    def export(model, path, provenance, state, **kw):
        before = (torch.get_rng_state().clone(), random.getstate(), np.random.get_state(), model.training)
        result = original_export(model, path, provenance, state, **kw)
        assert torch.equal(before[0], torch.get_rng_state())
        assert before[1] == random.getstate()
        after = np.random.get_state()
        assert before[2][0] == after[0] and np.array_equal(before[2][1], after[1]) and before[2][2:] == after[2:]
        assert before[3] == model.training
        checkpoints.append(Path(path))
        return result
    monkeypatch.setattr(trainer, "export_adapter", export)
    models = []
    for enabled in (False, True):
        torch.manual_seed(17)
        model = attach_lora(tiny_model(), rank=2, alpha=4, dropout=.05)
        model.peft_config["default"].revision = "fixed"
        model.set_attn_implementation("sdpa")
        output = tmp_path / str(enabled)
        state = trainer.train_loop(model, rows, output, objective="sft", learning_rate=5e-5, scheduler="linear",
            warmup_ratio=0, warmup_steps=0, original_batch=8, epochs=2, seed=3, packing="turn",
            **({"checkpoint_updates": [1, 3, 4], "adapter_provenance": provenance} if enabled else {}))
        original_export(model, output / "final", provenance, state)
        models.append(model)
    assert len(checkpoints) == 3
    updates = [json.loads(line) for line in (tmp_path / "True/updates.jsonl").read_text().splitlines()]
    assert (tmp_path / "True/updates.jsonl").read_bytes() == (tmp_path / "False/updates.jsonl").read_bytes()
    for path in checkpoints:
        saved = json.loads((path / "training_state.json").read_text())
        assert not saved["final"]
        update = updates[saved["update"] - 1]
        for key in ("examples_seen", "supervised_tokens_seen"):
            assert saved[key] == update[key]
        from harness.lora_adapters import inspect_adapter
        inspected = inspect_adapter("intermediate", path, student, base)
        assert inspected["checkpoint_metadata"]["passes"] == saved["examples_seen"] / len(rows)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        assert (tmp_path / "True/final" / name).read_bytes() == (tmp_path / "False/final" / name).read_bytes()
        assert (checkpoints[-1] / name).read_bytes() == (tmp_path / "True/final" / name).read_bytes()
    assert all(torch.equal(a, b) for a, b in zip(models[0].parameters(), models[1].parameters()))
    with pytest.raises(ValueError, match="beyond last update 4"):
        trainer.train_loop(models[0], rows, tmp_path / "invalid", objective="sft", learning_rate=5e-5,
            scheduler="linear", warmup_ratio=0, original_batch=8, epochs=2,
            checkpoint_updates=[5], adapter_provenance=provenance)
    assert not (tmp_path / "invalid").exists()


def test_checkpoint_publication_rank_zero_only(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from test_smartad_torch import Tiny
    monkeypatch.setattr(trainer, "export_adapter", lambda *a, **kw: pytest.fail("nonzero rank export"))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda *a: None)
    trainer.train_loop(Tiny(), [{"input_ids": [1, 2], "labels": [-100, 2]}], tmp_path,
        objective="sft", learning_rate=1e-4, scheduler="linear", warmup_ratio=0, original_batch=8,
        epochs=1, rank=1, world_size=2, checkpoint_updates=[1], adapter_provenance={})


def test_sealed_shape_mismatch_and_reuse(tmp_path, monkeypatch):
    from baselines.common import prepared
    from sed.config import load_config
    from sed.storage import write_json
    from sed.validation_cache import anchors
    config = load_config()
    for name in ("manifest", "split_freeze"):
        config["paths"][name] = str(tmp_path / (name + ".json"))
        write_json(config["paths"][name], {})
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    snapshot, verified = tmp_path / "base", tmp_path / "seeds"
    write_json(snapshot / "config.json", {})
    for filename in ("manifest.json", "identity.json", "selection.json"):
        write_json(verified / filename, {})
    data = (config, {"K": 1, "unique_instances": 1, "task_groups": 1},
            [{"input_ids": [1, 2], "labels": [-100, 2]}], "template")
    identity = {"method": "kang", "recipe": pipeline.RECIPE.copy(), "config_sha256": digest(config),
                "code_sha256": prepared.code_hashes(prepared.ROOT), "verified": str(verified), "snapshot": str(snapshot),
                "snapshot_artifact_sha256": prepared.artifact_hashes(snapshot), "gpus": 1,
                "collection_sha256": digest(data[1])}
    source, target = tmp_path / "source", tmp_path / "target"
    for output in (source, target):
        write_json(output / "identity.json", identity)
        write_json(output / "tasks.json", {})
    args = SimpleNamespace(config=str(config_path), pipeline_output=str(source), verified=str(verified),
                           snapshot=str(snapshot), base_snapshot=str(snapshot), output="unused", epochs=2, original_batch=8)
    ref = prepared.write_prepared(args, data, {str(verified): anchors(verified)})
    with pytest.raises(ValueError, match="rank differs"):
        prepared.check_trainer_identity(identity, SimpleNamespace(lora_rank=4))
    args.lora_rank, args.lora_alpha = 4, 8
    from train.lora_options import recipe_overrides
    new_identity = {**identity, "recipe": {**identity["recipe"], **recipe_overrides(args)}}
    write_json(target / "identity.json", new_identity)
    args.pipeline_output, args.reuse_prepared = str(target), str(source)
    monkeypatch.setattr(trainer, "prepare_lora_data", lambda *a: pytest.fail("re-encoding"))
    loaded, reused = prepared.reuse_prepared(args)
    assert loaded == data
    old_bytes = Path(ref["path"]).with_name("examples.json")
    new_bytes = Path(reused["path"]).with_name("examples.json")
    assert old_bytes.stat().st_ino == new_bytes.stat().st_ino
    assert prepared.load_prepared(args)[0] == data
    del args.reuse_prepared
    assert prepared.load_prepared(args)[0] == data
    args.lora_rank = 8
    with pytest.raises(ValueError, match="trainer recipe mismatch"):
        prepared.load_prepared(args)
    args.lora_rank = 4
    new_identity["recipe"]["epochs"] = 3
    write_json(target / "identity.json", new_identity)
    args.reuse_prepared = str(source)
    with pytest.raises(ValueError, match="differs beyond"):
        prepared.reuse_prepared(args)


@pytest.mark.parametrize("options,rank,alpha,extra", [
    (["--lora-rank", "3", "--lora-alpha", "9"], 3, 9, {}),
    (["--lora-layers", "1", "--lora-targets", "q_proj,down_proj"], 64, 128, {"layers": [1]}),
    (["--lora-modules", "1.down_proj"], 64, 128, {"modules": ["1.down_proj"]})])
def test_effective_sft_provenance(tmp_path, monkeypatch, options, rank, alpha, extra):
    observation = default_observation(tmp_path, monkeypatch, options)
    config = observation["config"]["sft"]
    assert (config["rank"], config["alpha"], config["dropout"]) == (rank, alpha, .05)
    assert config["targets"] == observation["targets"]
    for key in ("layers", "modules"):
        assert (key in config) == (key in extra)
        if key in extra:
            assert config[key] == extra[key]


@pytest.mark.parametrize("populated", [False, True])
def test_new_slurm_only_appends_nonempty_options(tmp_path, populated):
    import os
    import subprocess
    import sys
    original = Path("scripts/hpg_kang.slurm").read_bytes()
    # Pin the protected script, including all SBATCH lines and guards.
    assert hashlib.sha256(original).hexdigest() == "5c8a0fd2f3ecd4e17ce33ae76bfe1f6ca6a4b3f4bee3563e38c2cdcc40fa56f2"
    new = Path("scripts/hpg_kang_wsp.slurm").read_text()
    assert new.split('command=("$ENV_PYTHON"')[0] == original.decode().split('"$ENV_PYTHON" -m')[0]
    fake = tmp_path / "python"
    fake.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                    'open(os.environ["ARGS"],"w").write(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), HPG_CONFIG=str(Path("configs/hpg.env").resolve()),
               ENV_PYTHON=str(fake), TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake), MODEL_PATH=str(tmp_path),
               VERIFIED="SEEDS", KANG_OUT=str(tmp_path / "out"), ARGS=str(tmp_path / "args.json"),
               NGPUS="1", ENABLE_THINKING="off", MAX_TOKENS="1024", MAX_LENGTH="32768")
    variables = {"KANG_LORA_RANK": "4", "KANG_LORA_ALPHA": "8", "KANG_LORA_TARGETS": "q_proj",
                 "KANG_LORA_LAYERS": "0-1,4", "KANG_LORA_MODULES": "1.q_proj",
                 "KANG_CHECKPOINT_UPDATES": "1 3 5"}
    env.update({k: v if populated else "" for k, v in variables.items()})
    result = subprocess.run(["bash", "scripts/hpg_kang_wsp.slurm"], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    argv = json.loads((tmp_path / "args.json").read_text())
    if populated:
        for key, value in variables.items():
            flag = "--" + key.removeprefix("KANG_").lower().replace("_", "-")
            values = value.split() if key == "KANG_CHECKPOINT_UPDATES" else [value]
            assert argv[argv.index(flag) + 1:argv.index(flag) + 1 + len(values)] == values
    else:
        assert not any(arg.startswith(("--lora-", "--checkpoint-")) for arg in argv)
