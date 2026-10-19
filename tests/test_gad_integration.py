"""CPU-only checkpoint publication, evaluation gating and interrupted resume."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from baselines.common.method_data import read
from sed.storage import digest, file_hash, write_json
from test_full_train import full_bank, factories


def test_registered_full_train_cell_uses_evidence_not_magic_k(full_bank, tmp_path):
    from baselines.common.pipeline import prepare
    from teachers.collect_seeds import collect
    root, manifest, freeze, selection, config = full_bank
    env, teacher = factories(full_bank, tmp_path / "calls.jsonl")
    verified = tmp_path / "verified"
    collection = collect(selection, manifest, freeze, config, verified, env_factory=env, teacher_factory=teacher)
    snapshot = tmp_path / "base"
    snapshot.mkdir()
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    args = NS(output=str(tmp_path / "cell"), verified=str(verified), snapshot=str(snapshot), gpus=1,
              config=str(config_path), init_checkpoint=None)
    identity = prepare(args, config, method="gad", recipe={})
    assert identity["K"] == collection["K"] == 4  # Not a registered few-shot K.
    assert identity["unique_instances"] == 4 and identity["task_groups"] == 2
    listing = read(Path(args.output) / "tasks.json")
    assert listing["task_ids"] == [r["task_id"] for r in collection["verified"]]
    assert all(listing["parent_seed_ids"][t] for t in listing["task_ids"])


@pytest.mark.parametrize("fail", [False, True])
def test_every_checkpoint_equivalence_gate_before_dev140(tmp_path, monkeypatch, fail):
    import baselines.common.evaluation as common
    import baselines.gad.pipeline as p
    config = {"models": {"student": {"requested_repo": "student"}}}
    args = NS(output=str(tmp_path / "cell"), snapshot="base", verified=str(tmp_path / "verified"),
              config="config", train_python="train", env_python="env", workers=1)
    write_json(Path(args.verified) / "manifest.json", {})
    write_json(Path(args.output) / "identity.json", {"method": "gad", "config_sha256": digest(config),
        "snapshot": args.snapshot, "verified": args.verified})
    models = []
    for i in range(2):
        checkpoint = tmp_path / f"checkpoint-{i}"
        write_json(checkpoint / "training_state.json", {"artifact_sha256": {}})
        models.append({"name": f"update-{i}", "checkpoint": str(checkpoint),
            "checkpoint_state_sha256": file_hash(checkpoint / "training_state.json")})
    write_json(Path(args.output) / "models.json", models)
    events = []
    @contextmanager
    def served(*_):
        events.append("serve")
        yield "endpoint"
        events.append("stop")
    def command(cmd, log, **kw):
        if "--hf-only" in cmd:
            events.append("hf")
        elif "--hf-reference" in cmd:
            events.append("gate")
            if fail:
                raise RuntimeError("gate failed")
        else:
            events.append("eval")
            assert "--full-check" in cmd
            assert cmd[cmd.index("--tasks") + 1] == "data/manifest/dev140.json"
            assert cmd[cmd.index("--decoding") + 1] == "greedy"
            assert kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
    def forbidden(*a, **kw):
        raise AssertionError("Evaluation must never collect/train")
    monkeypatch.setattr(common, "load_config", lambda _: config)
    monkeypatch.setattr(common, "served", served)
    monkeypatch.setattr(common, "run_command", command)
    for name in ("prepare", "collect", "train_round"):
        monkeypatch.setattr(p, name, forbidden)
    if fail:
        with pytest.raises(RuntimeError, match="gate failed"):
            p.evaluate(args, config)
        assert "eval" not in events
    else:
        p.evaluate(args, config)
        assert events == ["hf", "serve", "gate", "eval", "stop"] * 2
        p.evaluate(args, config)
        assert len(events) == 10


def test_interrupted_actor_reuses_completed_samples_and_discriminator(tmp_path, monkeypatch):
    import baselines.gad.training as t
    args = NS(output=str(tmp_path / "cell"), verified="verified", config="config", gpus=1,
              train_python="python", snapshot="base")
    data = tmp_path / "data.json"
    write_json(data, {})
    monkeypatch.setattr(t, "load_rows", lambda *a, **kw: [None])
    monkeypatch.setattr(t, "check_history_policy", lambda *a: None)
    counts = {"samples": 0, "critic": 0, "actor": 0}
    @contextmanager
    def served(*_):
        counts["samples"] += 1
        yield "endpoint"
    fail = [True]
    def command(cmd, log, **kw):
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True)
        if "baselines.gad.sampling" in cmd:
            write_json(out / "groups.json", {})
        else:
            role = cmd[cmd.index("--role") + 1]
            counts[role] += 1
            if role == "actor":
                if fail[0]:
                    fail[0] = False
                    raise RuntimeError("interrupted")
                write_json(out / "checkpoint/training_state.json", {})
            else:
                write_json(out / "rewards.json", {})
    monkeypatch.setattr(t, "served", served)
    monkeypatch.setattr(t, "run_command", command)
    kw = dict(data=data, snapshot="base", previous=None, round_index=0, output=tmp_path / "round")
    with pytest.raises(RuntimeError, match="interrupted"):
        t.train_round(args, {}, {}, **kw)
    assert counts == {"samples": 11, "critic": 11, "actor": 1}
    t.train_round(args, {}, {}, **kw)
    assert counts == {"samples": 11, "critic": 11, "actor": 2}
    root = Path(args.output) / "updates/round-0/stages/actor-00010"
    assert (root / "attempt-000/failure.json").exists()
    assert read(root / "complete.json")["attempt"] == "attempt-001"


def test_full_checkpoint_export_matches_existing_serving_contract(tmp_path):
    torch = pytest.importorskip("torch")
    from test_stage2b_torch import tiny_model
    from baselines.common.fsdp_train import export_checkpoint, validate_checkpoint
    from baselines.gad.objective import optimizer_for
    from baselines.gad.worker import update
    from test_gad_torch import groups
    from sed.provenance import write_provenance
    torch.set_num_threads(1)
    base, output = tmp_path / "base", tmp_path / "trained"
    model = tiny_model()
    model.save_pretrained(base)
    output.mkdir()
    provenance = write_provenance(output, command="gad_actor", config={"models": {"student": {}}})
    result = update(model, optimizer_for(model), groups(), role="actor", kind="ce", device="cpu")
    checkpoint = output / "checkpoint"
    state = export_checkpoint(model, checkpoint, base)
    write_json(checkpoint / "training_state.json", {**result, **state, "method": "gad", "final": True,
        "provenance_sha256": digest(provenance)})
    assert validate_checkpoint(checkpoint, {}, base_snapshot=base)["method"] == "gad"


def test_disclosure_retains_failed_attempts_unknown_usage_and_exposure(tmp_path):
    from baselines.gad.pipeline import update_disclosure
    from sed.storage import JsonlLedger
    identity = {"method": "gad", "K": 5, "unique_instances": 5, "task_groups": 3}
    write_json(tmp_path / "teacher/counter.json", {"attempts": 2, "halted": True})
    ledger = JsonlLedger(tmp_path / "teacher/usage.jsonl")
    ledger.append({"status": "success", "usage": {"input_tokens": 9, "output_tokens": 3}})
    ledger.append({"status": "failed", "usage": None})
    base = tmp_path / "updates/round-0/stages"
    JsonlLedger(base / "sample-00000/attempt-000/output/calls.jsonl").append({"status": "started"})
    JsonlLedger(base / "sample-00000/attempt-001/output/calls.jsonl").append({"status": "started"})
    JsonlLedger(base / "sample-00000/attempt-001/output/calls.jsonl").append({"status": "returned",
        "response": {"choices": [{}] * 8, "usage": {"prompt_tokens": 11, "completion_tokens": 24}}})
    actor = base / "actor-00010/attempt-000/output"
    JsonlLedger(actor / "updates.jsonl").append({"examples_seen": 2, "supervised_tokens_seen": 5,
        "logical_forward_tokens": 17, "physical_forward_tokens": 17})
    # No result.json: completed update exposure survives an interrupted export.
    update_disclosure(tmp_path, identity)
    report = read(tmp_path / "disclosure.json")
    assert report["teacher_cap"] == 600 and report["unknown_teacher_usage_possible"]
    assert report["totals"]["teacher_calls"] == 2 and report["totals"]["teacher_unknown_usage_calls"] == 1
    assert report["totals"]["group_responses"] == 8 and report["totals"]["group_completion_tokens"] == 24
    assert report["totals"]["actor_examples_seen"] == 2
    assert report["training_attempts"][0]["unknown_exposure_possible"]
    assert report["sampling_attempts"][0]["unknown_usage_possible"]
