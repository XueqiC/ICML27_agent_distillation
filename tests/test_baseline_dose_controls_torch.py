"""Offline dose-control validation, real tiny Gemma merge/retrain, CLI routing."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from peft import PeftModel

from baselines.common import dose_control as control
from baselines.common.fsdp_train import prepare_model, text_parameter, validate_checkpoint
from baselines.agentgym import dose_train
from baselines.agentgym.data import prepare_dataset
from baselines.leap.train import validate_initialization
from harness.lora_adapters import inspect_adapter
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes, verified_collection_inputs
from sed.seed_selection import select_seeds
from sed.storage import digest, file_hash, write_json
from teachers.collect_seeds import collect
from train.dataset import build_verified_dataset
from train.merge_lora import LEAP_MERGE_POLICY
from train.modeling import attach_lora, load_student, snapshot_metadata
from train.sft import exposure_plan, train_loop
from test_checkpoint_export_torch import read_tensors
from test_leap_chain_torch import tiny_snapshot
from test_stage2b_seeds import seed_bank, factories


@pytest.fixture
def cell(seed_bank, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    root, manifest, freeze, config = seed_bank
    config["environment"]["max_episode_steps"] = 40
    config["paths"].update(manifest=str(tmp_path / "manifest.json"), split_freeze=str(tmp_path / "freeze.json"))
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    cfg = tmp_path / "config.json"
    write_json(cfg, config)
    base = tmp_path / "base"
    tokenizer = tiny_snapshot(base)
    env, teacher = factories(root, config, set(), [])
    verified = tmp_path / "verified"
    collect(select_seeds(manifest, freeze, k=5), manifest, freeze, config, verified,
        env_factory=env, teacher_factory=teacher)
    examples = build_verified_dataset(verified, tokenizer, manifest, freeze, packing="episode")
    plan = exposure_plan([e["supervised_tokens"] for e in examples], passes=2, packing="episode",
                         turn_counts=[e["turn_count"] for e in examples])
    trained = tmp_path / "seed-sft"
    trained.mkdir()
    recipe = {"snapshot": str(base), "verified": str(verified), "output": str(trained),
        "base_snapshot": snapshot_metadata(base), "rank": 16, "alpha": 32, "dropout": 0.0,
        "learning_rate": 1e-5, "tokens_per_update": 512, "packing": "episode", "optimizer": "AdamW",
        "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8, "gradient_clip": 1.0,
        "scheduler": "constant", "dtype": "bfloat16", "loss": "supervised_token_mean_per_update",
        "attention": "eager", "microbatch": 1, "gradient_checkpointing": True,
        "enable_thinking": False, "max_length": 32768, "seed": 0, "passes": 2,
        "supervised_tokens": None, "checkpoint_tokens": [], "plan_sha256": digest(plan)}
    provenance = write_provenance(trained, command="sft", config={**config, "sft": recipe},
        inputs=verified_collection_inputs(verified, manifest, freeze))
    model = attach_lora(load_student(base, device="cpu", dtype="bfloat16"), gradient_checkpointing=False)
    checkpoints = train_loop(model, examples, plan, trained, tokenizer=tokenizer, identity_sha256=digest(provenance))
    adapter = trained / checkpoints[-1]["path"]
    args = NS(init_checkpoint=str(adapter), snapshot=str(base), verified=str(verified))
    identity = {"method": "agentgym", "collection_sha256": digest(control.read(verified / "manifest.json")),
                "snapshot_artifact_sha256": artifact_hashes(base)}
    control.bind_init(args, config, identity)
    return NS(base=base, verified=verified, adapter=adapter, init=identity["init"], identity=identity,
              config=config, config_path=cfg, args=args, manifest=manifest, freeze=freeze)


def test_fp32_merge_exact_base_plus_delta_and_full_checkpoint_reload(cell, tmp_path):
    model = PeftModel.from_pretrained(load_student(cell.base, device="cpu", dtype="float32"),
        cell.adapter, local_files_only=True)
    expected = {}
    differences = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        delta = (module.lora_B["default"].weight @ module.lora_A["default"].weight) * module.scaling["default"]
        value = module.base_layer.weight.detach() + delta.detach()
        assert value.dtype == torch.float32
        expected[name.removeprefix("base_model.model.") + ".weight"] = value
        differences += int(not torch.equal(value, value.to(torch.bfloat16).float()))
    assert expected and differences  # Actually detect the rounding regression.
    identity = tmp_path / "identity.json"
    write_json(identity, cell.identity)
    control.main(["--config", str(cell.config_path), "--snapshot", str(cell.base),
        "--verified", str(cell.verified), "--identity", str(identity), "--output", str(tmp_path / "merged")])
    checkpoint = tmp_path / "merged/checkpoint"
    state = validate_checkpoint(checkpoint, cell.config["models"]["student"], base_snapshot=cell.base)
    assert state["text_export_dtype"] == "float32"
    saved, base = read_tensors(checkpoint), read_tensors(cell.base)
    assert saved.keys() == base.keys()
    for key, value in expected.items():
        assert torch.equal(saved[key], value), key
    for key, value in base.items():
        if not text_parameter(key):
            assert saved[key].dtype == value.dtype and torch.equal(saved[key], value)
    for name in ("processor_config.json", "preprocessor_config.json", "alternate_template.jinja2", "tokenizer.model"):
        assert (checkpoint / name).read_bytes() == (cell.base / name).read_bytes()
    masters = prepare_model(checkpoint)
    assert all(p.dtype == torch.float32 for p in masters.parameters())
    for key, value in expected.items():
        assert torch.equal(masters.state_dict()[key], value)
    reference = prepare_model(checkpoint, reference=True)
    for key, value in expected.items():
        assert torch.equal(reference.state_dict()[key], value.to(torch.bfloat16))
    assert not any(p.requires_grad for p in reference.parameters())
    from harness.serve_student import server_command
    args = NS(merged_model_path=str(checkpoint), model_path=str(cell.base), python="VLLM", port=1234)
    command = server_command(args, cell.config["models"]["student"], {"gemma4_parser": False})
    assert command[command.index("--dtype") + 1] == "bfloat16"
    args.merged_model_path = None
    assert "--dtype" not in server_command(args, cell.config["models"]["student"], {"gemma4_parser": False})


@pytest.mark.parametrize("variant", ["privileged", "zero"])
def test_leap_seed_init_official_bf16_merge_then_fresh_lora(cell, tmp_path, variant):
    init = {**cell.init, "merge_policy": LEAP_MERGE_POLICY}
    checkpoint = control.export_init(cell.base, cell.adapter, tmp_path / "merged", cell.config, init, variant=variant)
    state = validate_checkpoint(checkpoint, cell.config["models"]["student"], base_snapshot=cell.base)
    assert state["text_export_dtype"] == "bfloat16" and state["policy"] == LEAP_MERGE_POLICY
    assert state["merge_passed"] and state["leap_iteration"] == 0
    assert state["fresh_lora"] is False
    validate_initialization(checkpoint, cell.base, iteration=1, variant=variant, student=cell.config["models"]["student"])
    with pytest.raises(ValueError, match="previous merged iterate"):
        validate_initialization(checkpoint, cell.base, iteration=2, variant=variant, student=cell.config["models"]["student"])
    with pytest.raises(ValueError, match="previous merged iterate"):
        validate_initialization(checkpoint, cell.base, iteration=1, variant="zero" if variant == "privileged" else "privileged",
                                student=cell.config["models"]["student"])
    from baselines.leap.train import prepare_model as leap_model
    fresh = leap_model(checkpoint, device="cpu")
    assert all(torch.count_nonzero(p) == 0 for n, p in fresh.named_parameters() if "lora_B" in n)
    assert fresh.peft_config["default"].r == 128


def test_init_validation_and_identity(cell, tmp_path):
    original = control.read(cell.adapter / "training_state.json")
    for field, value, error in [("final", False, "final Seed-SFT"), ("passes", 0, "positive realized"),
                                 ("passes", 3, "episode exposure")]:
        write_json(cell.adapter / "training_state.json", {**original, field: value})
        with pytest.raises(ValueError, match=error):
            control.bind_init(cell.args, cell.config, deepcopy(cell.identity))
    write_json(cell.adapter / "training_state.json", original)
    config = deepcopy(cell.config)
    config["models"]["student"]["revision"] = "other"
    with pytest.raises(ValueError, match="revision"):
        control.bind_init(cell.args, config, deepcopy(cell.identity))
    # Same K is insufficient: a different collection manifest must be refused.
    other = tmp_path / "other"
    write_json(other / "manifest.json", {**control.read(cell.verified / "manifest.json"), "changed": True})
    with pytest.raises(ValueError, match="different verified collection"):
        control.validate_init(cell.adapter, cell.base, other, cell.config,
                              collection_sha256=digest(control.read(other / "manifest.json")))
    assert cell.init["training_state"] == original
    assert cell.init["artifact_sha256"]["training_state.json"] == file_hash(cell.adapter / "training_state.json")
    assert cell.init["recipe"]["rank"] == 16
    # A changed artifact invalidates the control even if training_state stayed fixed.
    with (cell.adapter / "adapter_model.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        control.bind_init(cell.args, cell.config, deepcopy(cell.identity))


def test_real_agentevol_retrain_uses_seed_sft_and_is_unmerged(cell, tmp_path):
    dataset = tmp_path / "data-0"
    prepare_dataset(cell.verified, dataset, cell.config)
    identity = tmp_path / "identity.json"
    write_json(identity, cell.identity)
    output = tmp_path / "retrain"
    dose_train.main(["--config", str(cell.config_path), "--snapshot", str(cell.base),
        "--verified", str(cell.verified), "--sft-data", str(dataset / "train_iter_0.json"),
        "--identity", str(identity), "--output", str(output)], device="cpu")
    result = control.read(output / "result.json")
    assert result["passes"] == result["requested_passes"] == 2
    assert result["examples_seen"] == 10 and result["fresh_from_base"] and result["unmerged_serving"]
    adapter = output / result["checkpoint"]
    assert not (adapter / "config.json").exists()
    inspected = inspect_adapter("dose-policy", adapter, cell.config["models"]["student"], cell.base)
    assert inspected["rank"] == 16
    recipe = control.read(output / "provenance.json")["config"]["sft"]
    for key in ("learning_rate", "optimizer", "tokens_per_update", "packing", "rank", "alpha", "scheduler"):
        assert recipe[key] == cell.init["recipe"][key]
    plan = control.read(output / "exposure_plan.json")
    assert plan[-1]["examples_seen"] == 10
    assert all(p["packing"] == "episode" for p in plan)
    sources = control.read(output / "example_sources.json")
    assert all(s["source"] == "trusted_teacher_seed" and s["trusted_seed"] for s in sources)


def test_fractional_pass_endpoint_uses_complete_episodes():
    init = {"recipe": {"tokens_per_update": 512, "seed": 0}, "passes": 7 / 5,
            "training_state": {"examples_seen": 7}}
    plan = dose_train.matched_plan([{"supervised_tokens": 33, "turn_count": 2}] * 8, init)
    assert plan[-1]["examples_seen"] == 12  # ceil(1.4 * 8)
    assert plan[-1]["passes"] == 1.5 and plan[-1]["turns_seen"] == 24


@pytest.mark.parametrize("method,variant", [("agentgym", None), ("eto", None), ("ipr", None),
                                           ("leap", "privileged"), ("leap", "zero")])
def test_control_cli_replaces_initial_stage_and_preserves_later_recipe(cell, tmp_path, monkeypatch, method, variant):
    pipeline = importlib.import_module(f"baselines.{method}.pipeline")
    eto = importlib.import_module("baselines.eto.pipeline")
    output = tmp_path / "pipeline"
    commands, serving, oracle = [], [], []
    def command(cmd, log, **kwargs):
        commands.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True)
        if "baselines.eto.pairs" in cmd:
            write_json(out / "report.json", {"P": 1})
        else:
            write_json(out / "result.json", {"checkpoint": "adapter", "passes": 2,
                "examples_seen": 10, "supervised_tokens_seen": 100})
        if "baselines.common.explore" in cmd:
            write_json(out / "summary.json", {})
    @contextmanager
    def served(args, checkpoint, attempt):
        serving.append(str(checkpoint))
        yield "http://fake/v1"
    monkeypatch.setattr(eto, "run_command", command)
    monkeypatch.setattr(pipeline, "run_command", command)
    monkeypatch.setattr(pipeline, "served", served)
    if method == "agentgym":
        def dataset(verified, out, config, *, iteration=0, **kw):
            out.mkdir(parents=True)
            write_json(out / f"train_iter_{iteration}.json", [{}] * (5 + iteration))
            write_json(out / "report.json", {"F_m": 1, "D_next": 6 + iteration, "empty_F": False})
        monkeypatch.setattr(pipeline, "prepare_dataset", dataset)
    if method == "ipr":
        def rollout(args, snapshot, attempt, **kw):
            serving.append((kw["kind"], str(snapshot)))
            (attempt / "output").mkdir()
        def pairs(verified, exploration, expert_mc, agent_mc, out, config):
            out.mkdir(parents=True)
            write_json(out / "report.json", {"P": 1, "P_by_kind": {"global": 1}})
        monkeypatch.setattr(pipeline, "rollout_stage", rollout)
        monkeypatch.setattr(pipeline, "prepare_pairs", pairs)
    if method == "leap":
        monkeypatch.setattr(pipeline, "expert_source_hashes", lambda: {"fake_backend": "offline"})
        def dataset(verified, out, config, *, iteration=0, **kw):
            out.mkdir(parents=True)
            write_json(out / "examples.json", [{"i": i} for i in range(5)])
            write_json(out / "report.json", {"inadmissible_corrections": 0})
        def audit(*args):
            oracle.append(1)
            args[-1].mkdir(parents=True)
        def correct(*args):
            args[-1].mkdir(parents=True)
            write_json(args[-1] / "report.json", {"corrections": 5})
        monkeypatch.setattr(pipeline, "prepare_set", dataset)
        monkeypatch.setattr(pipeline, "collect_experts", audit)
        monkeypatch.setattr(pipeline, "extract_states", audit)
        monkeypatch.setattr(pipeline, "correct_rollouts", correct)
    argv = ["--config", str(cell.config_path), "--snapshot", str(cell.base), "--verified", str(cell.verified),
            "--output", str(output), "--init-checkpoint", str(cell.adapter)]
    if variant:
        argv += ["--variant", variant]
    pipeline.main(argv)
    identity = control.read(output / "identity.json")
    assert identity["control"] == "dose_matched_init"
    assert identity["init"]["artifact_sha256"] == artifact_hashes(cell.adapter)
    models = control.read(output / "models.json")
    init = models[0]["checkpoint"]
    assert "/stages/init/" in init
    assert all(m["label"].endswith(control.SUFFIX) for m in models)
    disclosure = control.read(output / "disclosure.json")
    assert disclosure["control"] == identity["control"] and disclosure["init"] == identity["init"]
    assert len([c for c in commands if "baselines.common.dose_control" in c]) == 1
    training = [c for c in commands if "--objective" in c]
    assert all(c[c.index("--objective") + 1] != "sft" for c in training)
    if method in {"eto", "ipr"}:
        assert len(training) == (3 if method == "eto" else 6)
        assert all(c[c.index("--reference") + 1] == init for c in training)
        assert training[0][training[0].index("--snapshot") + 1] == init
        assert all(c[c.index("--epochs") + 1] == "3" for c in training)
        if method == "ipr":
            assert all(s == init for kind, s in serving if kind in {"expert-mc", "agent-mc"})
        else:
            assert serving[0] == init
    if method == "agentgym":
        retrains = [c for c in commands if "baselines.agentgym.dose_train" in c]
        assert len(retrains) == 3 and len(serving) == 4 and serving[0] == init
        assert all(c[c.index("--snapshot") + 1] == str(cell.base) for c in retrains)
        assert all("adapter" in m for m in models[1:])
        assert not control.read(output / "result.json")["train_iter_4_trained"]
        assert not (output / "stages/train-0").exists()
    if method == "leap":
        trains = [c for c in commands if "baselines.leap.train" in c]
        assert len(trains) == 3 and serving[0] == init
        assert len(oracle) == (2 if variant == "privileged" else 0)
        assert [c[c.index("--iteration") + 1] for c in trains] == ["1", "2", "3"]
        assert trains[0][trains[0].index("--snapshot") + 1] == init
        assert all(len(c[c.index("--sets") + 1:]) == i + 2 for i, c in enumerate(trains))
        assert not (output / "stages/train-0").exists()
    # Resume reuses all completed stages, preserving checkpoint provenance.
    before = len(commands)
    pipeline.main(argv)
    assert len(commands) == before
    # A different verified support set is rejected before any merge/train work.
    other = tmp_path / "other-verified"
    env, teacher = factories(Path(cell.config["environment"]["data_root"]), cell.config, set(), [])
    collect(select_seeds(cell.manifest, cell.freeze, k=5, seed=1), cell.manifest, cell.freeze, cell.config,
        other, env_factory=env, teacher_factory=teacher)
    argv[argv.index("--verified") + 1] = str(other)
    argv[argv.index("--output") + 1] = str(tmp_path / "rejected")
    with pytest.raises(ValueError, match="different verified collection"):
        pipeline.main(argv)
    assert len(commands) == before
    assert not (tmp_path / "rejected/identity.json").exists()


@pytest.mark.parametrize("method", ["agentgym", "eto", "ipr", "leap"])
def test_default_identity_has_no_control_fields_or_empty_cli_drift(cell, tmp_path, monkeypatch, method):
    from baselines.common import pipeline as common
    from baselines.eto import pipeline as eto
    module = importlib.import_module(f"baselines.{method}.pipeline")
    prepare = eto.prepare if method == "eto" else common.prepare
    monkeypatch.setattr(eto if method == "eto" else common, "code_hashes", lambda _: {"source.py": "fixed"})
    args = NS(snapshot=str(cell.base), verified=str(cell.verified), config=str(cell.config_path),
              output=str(tmp_path / "default"), gpus=4)
    kwargs = {} if method == "eto" else {"method": method, "recipe": module.RECIPE}
    original = prepare(args, cell.config, **kwargs)
    first = (Path(args.output) / "identity.json").read_bytes()
    args.init_checkpoint = None
    assert prepare(args, cell.config, **kwargs) == original
    assert (Path(args.output) / "identity.json").read_bytes() == first
    legacy_keys = {"recipe", "K", "support_seed", "unique_instances", "task_groups", "verified", "collection_sha256",
                   "snapshot", "config_sha256", "snapshot_artifact_sha256", "gpus", "tasks_sha256", "code_sha256"}
    if method != "eto":
        legacy_keys |= {"method", "batch_invariant"}
    assert set(original) == legacy_keys
    assert original["recipe"] == module.RECIPE
    assert control.model_record(original, "sft", "checkpoint") == {"name": "sft", "checkpoint": "checkpoint"}
    args.init_checkpoint = ""
    assert prepare(args, cell.config, **kwargs) == original


def test_adapter_exploration_and_evaluation_use_unmerged_serving(cell, tmp_path, monkeypatch):
    from baselines.eto import pipeline
    args = NS(snapshot=str(cell.base), verified=str(cell.verified), config=str(cell.config_path),
        output=str(tmp_path / "run"), env_python="ENV", train_python="TRAIN", vllm_python="VLLM",
        ready_timeout=10, workers=1)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    calls, launches, stopped = [], [], []
    def run(cmd, log, **kw):
        calls.append((cmd, kw))
    def popen(cmd, **kw):
        launches.append(cmd)
        write_json(attempt / "server/ready.json", {"endpoint": "http://127.0.0.1:1234/v1"})
        return NS(poll=lambda: None)
    monkeypatch.setattr(pipeline, "run_command", run)
    monkeypatch.setattr(pipeline.subprocess, "Popen", popen)
    monkeypatch.setattr(pipeline, "stop_server", stopped.append)
    with pipeline.served(args, cell.adapter, attempt) as endpoint:
        assert endpoint == "http://127.0.0.1:1234/v1"
    cmd = launches[0]
    assert "--merged-model-path" not in cmd
    assert cmd[cmd.index("--lora-adapter") + 1] == "dose-policy=" + str(cell.adapter)
    assert "--hf-only" in calls[0][0] and "--hf-reference" in calls[1][0]
    assert len(stopped) == 1
    write_json(Path(args.output) / "models.json", [{"name": "iter-1", "checkpoint": str(cell.adapter),
        "adapter": str(cell.adapter), "label": "iter-1" + control.SUFFIX}])
    pipeline.evaluate_pipeline(args)
    cmd, kwargs = calls[-1]
    env = kwargs["env"]
    assert env["INCLUDE_BASE"] == "0" and env["MERGED_MODEL_PATH"] == ""
    assert env["DOSE_ADAPTER"] == "iter-1=" + str(cell.adapter)
    assert env["VERIFIED"] == str(cell.verified)
    assert 'adapters=("$DOSE_ADAPTER")' in cmd[-1]
