"""Resumable orchestration and launcher lifecycle; every policy/backend is fake."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from analysis.mechanism.fce import main as fce_main, run_fce
from analysis.mechanism.interventions import run_interventions
from analysis.mechanism.policies import BranchTeacher
from analysis.mechanism.readout import build_readout, validate_intervention_artifacts
from analysis.mechanism.taxonomy import classify_run, comparison
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits
from sed.runner import ScriptedPolicy, run_episode
from sed.storage import digest, file_hash, write_json
from test_mechanism import BranchBackend, FakeResponses, FakeStudent, TASK_ID


@pytest.fixture
def pipeline(tmp_path):
    data = tmp_path/"data"
    for name in ("train", "valid_seen", "valid_unseen"):
        (data/name).mkdir(parents=True)
    directory = data/TASK_ID
    directory.mkdir(parents=True)
    write_json(directory/"game.tw-pddl", {"solvable": True})
    write_json(directory/"traj_data.json", {"task_type": "pick_heat_then_place_in_recep"})
    manifest = build_manifest(data)
    freeze = freeze_splits(manifest)
    config = load_config()
    config["paths"].update(manifest=str(tmp_path/"manifest.json"), split_freeze=str(tmp_path/"freeze.json"))
    config["environment"]["data_root"] = str(data)
    write_json(config["paths"]["manifest"], manifest)
    write_json(config["paths"]["split_freeze"], freeze)
    task = {"task_id": TASK_ID, "split": "dev", "sealed": False, "task_type": "pick_heat_then_place_in_recep"}
    env_factory = lambda *_: Environment(data, split="dev", backend_factory=BranchBackend)
    parent = run_episode(env_factory(), ScriptedPolicy(["look"]), task, config,
                         tmp_path/"source/episodes/000", policy_condition="fake_student", purpose="diagnosis")
    listing = {"task_ids": [TASK_ID], "list_sha256": digest([TASK_ID]),
               "manifest_sha256": digest(manifest), "split_freeze_sha256": digest(freeze)}
    write_json(tmp_path/"source/tasks.json", listing)
    classified = classify_run(tmp_path/"source", config)
    teacher = {"episodes": [dict(classified["episodes"][0], success=True)]}
    taxonomy = {"G": comparison(classified, teacher, "teacher"),
                "runs": {"student": classified, "teacher": teacher}}
    gap = tmp_path/"taxonomy.json"
    write_json(gap, taxonomy)
    calls = []
    class API(FakeResponses):
        def create(self, **kwargs):
            text = kwargs["input"][-1]["content"]
            t = (len(kwargs["input"])-2)//2
            self.raw = "Take the apple." if "Do not issue an action" in text else "Action: win" if t < 1 else "Action: look"
            result = super().create(**kwargs)
            calls.append(kwargs)
            return result
    api = API()
    def teacher_factory(config, workspace, budget, episode_key, *, namespace, sample_index, advice=False):
        return BranchTeacher(namespace=namespace, sample_index=sample_index, budget=budget, episode_key=episode_key,
            advice=advice, cache_dir=Path(workspace)/"cache", usage_ledger=Path(workspace)/"usage.jsonl", run_id="fake",
            client=SimpleNamespace(responses=api), sampling={"max_output_tokens": 8}, sleep=lambda _: None,
            requests_per_minute=10000, tokens_per_minute=100000000)
    seed = dict(parent, task_id=TASK_ID.replace("valid_seen", "train"), split="train", success=True)
    return config, gap, env_factory, teacher_factory, calls, [{"order": 0, "episode": seed}]


def test_fce_caps_unknown_and_resume_completed_probes(pipeline, tmp_path):
    config, gap, env, teacher, calls, _ = pipeline
    from teachers.openai_teacher import TeacherCallError
    for _ in range(2):
        with pytest.raises(TeacherCallError):
            run_fce(gap_path=gap, output=tmp_path/"capped", config=config, max_episode=1, max_run=1,
                    env_factory=env, teacher_factory=teacher)
    assert len(calls) == 1
    assert len(list((tmp_path/"capped").rglob("receipt.json"))) == 1
    assert not (tmp_path/"capped/fce.json").exists()
    assert len(list((tmp_path/"capped").rglob("incomplete.json"))) == 2


@pytest.mark.parametrize("control_rescues", [False, True])
def test_full_fake_fce_interventions_readout_resume(pipeline, tmp_path, control_rescues):
    config, gap, env, teacher, calls, seeds = pipeline
    fce = run_fce(gap_path=gap, output=tmp_path/"fce", config=config, max_episode=1000, max_run=1000,
                  env_factory=env, teacher_factory=teacher)
    assert fce["episodes"][0]["fce"] == 0 and fce["complete"]
    before = len(calls)
    assert run_fce(gap_path=gap, output=tmp_path/"fce", config=config, max_episode=1000, max_run=1000,
                  env_factory=env, teacher_factory=teacher) == fce
    assert len(calls) == before
    fce_file = tmp_path/"fce/fce.json"
    fce_hash = file_hash(fce_file)
    student_calls = []
    def student_factory(**kwargs):
        student_calls.append(kwargs)
        return FakeStudent("win" if control_rescues else "look")
    args = dict(gap_path=gap, fce_path=fce_file, verified=None, output=tmp_path/"arms", config=config,
                spec={"model": "fake", "endpoint": "http://fake/v1"}, max_episode=1000, max_run=1000,
                env_factory=env, teacher_factory=teacher, student_factory=student_factory, seeds=seeds,
                budget_dir=tmp_path/"fce")
    result = run_interventions(**args)
    assert len(student_calls) == 30
    for arm_index in range(6):
        group = student_calls[arm_index*5:arm_index*5+5]
        assert [r["sample_index"] for r in group] == list(range(5))
        assert [r["sampling_seed"] for r in group] == [r["sampling_seed"] for r in student_calls[:5]]
    assert len({r["sampling_seed"] for r in student_calls}) == 5
    assert len(calls) == before+2  # One I3 advice and one I4 action, shared across samples.
    assert result["episodes"][0]["arms"]["I4"]["rescue"]
    validate_intervention_artifacts(result)
    readout = build_readout(json.loads(gap.read_text()), fce, result)
    assert readout["gap_size"] == 1
    assert readout["episodes"][0]["label"] == ("unresolved" if control_rescues else "execution_or_capacity")
    assert readout["non_deterministic"]["count"] == int(control_rescues)
    assert readout["paired_arm_rescue"]["primary"]["I4"]["paired_excess"]["count"] == int(not control_rescues)
    run_fce(gap_path=gap, output=tmp_path/"fce", config=config, max_episode=1000, max_run=1000,
            env_factory=env, teacher_factory=teacher)
    assert file_hash(fce_file) == fce_hash  # Later shared-budget use must not invalidate FCE inputs.
    run_interventions(**args)
    assert len(student_calls) == 30 and len(calls) == before+2
    # A new server can resume only if its serving protocol is unchanged.
    config["server_provenance"] = {"config": {"server": {"batch_invariant": True}}}
    with pytest.raises(ValueError, match="Resume identity changed"):
        run_interventions(**args)
    assert len(student_calls) == 30


def test_dry_run_never_constructs_teacher_or_environment(pipeline, tmp_path, monkeypatch, capsys):
    config, gap, _, _, calls, _ = pipeline
    import analysis.mechanism.fce as module
    monkeypatch.setattr(module, "load_config", lambda _: config)
    def forbidden(*a, **k):
        pytest.fail("Dry-run cannot instantiate a teacher/environment")
    monkeypatch.setattr(module, "make_teacher", forbidden)
    monkeypatch.setattr(module, "_environment", forbidden)
    assert fce_main(["--gap-set", str(gap), "--output", str(tmp_path/"dry"), "--dry-run",
                     "--seed-teacher-cache-from", str(tmp_path/"missing-cache-source"),
                     "--max-teacher-calls", "25", "--max-teacher-calls-run", "100"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["teacher_calls"] == 0 and result["episodes"][0]["coarse_probe_points"][-1] == 40
    assert result["episodes"][0]["densification_policy_call_upper_bound"] == 228
    assert not (tmp_path/"dry").exists() and not calls


@pytest.mark.parametrize("fail_check", [False, True])
@pytest.mark.parametrize("batch_invariant", [None, "0"])
def test_launcher_gate_order_free_port_cleanup_and_resume(tmp_path, fail_check, batch_invariant):
    fake = tmp_path/"fake-python"
    fake.write_text("#!"+sys.executable+"\n"+r'''
import json, os, signal, sys, time
from pathlib import Path
args=sys.argv[1:]
def value(flag): return args[args.index(flag)+1]
def record(kind):
    with open(os.environ["FAKE_LEDGER"], "a") as f:
        f.write(json.dumps({"kind":kind,"args":args,"cuda":os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "batch_invariant":os.environ.get("VLLM_BATCH_INVARIANT")})+"\n")
if args[0]=="-":
    p=Path(args[1]); print(json.loads(p.read_text())["endpoint"] if p.name=="ready.json" else "google/gemma-4-12B-it")
    sys.exit(0)
out=Path(value("--output"))
if args[:2]==["-m","analysis.mechanism.launcher"]:
    record("preflight")
    out.mkdir(parents=True,exist_ok=True)
    launch=out/"launches"/str(len(list((out/"launches").glob("*"))))
    launch.mkdir(parents=True);print(launch);sys.exit(0)
out.mkdir(parents=True,exist_ok=args[0]=="-m")
if args[0]=="harness/serve_student.py":
    record("server");assert value("--port")=="0"
    def stop(*_): record("stopped");sys.exit(0)
    signal.signal(signal.SIGTERM,stop)
    (out/"ready.json").write_text(json.dumps({"endpoint":"http://127.0.0.1:34567/v1"}))
    (out/"provenance.json").write_text("{}")
    while True: time.sleep(.05)
elif args[0]=="tools/check_lora_serving.py":
    kind="hf" if "--hf-only" in args else "check";record(kind)
    if kind=="check" and os.environ.get("FAKE_FAIL")=="1":sys.exit(1)
    (out/("reference.json" if kind=="hf" else "report.json")).write_text("{}")
elif args[0]=="-m":
    record(args[1].rsplit(".",1)[1]);(out/(args[1].rsplit(".",1)[1]+".json")).write_text("{}")
else:raise AssertionError("Unexpected command")
''')
    fake.chmod(0o755)
    (tmp_path/"snapshot").mkdir()
    env = os.environ.copy()  # Never logged; only selected test flags are recorded.
    env.pop("WORKERS", None)
    env.pop("BATCH_INVARIANT", None)
    seed_source = str(tmp_path/"old workspace")
    if batch_invariant is not None:
        env["BATCH_INVARIANT"] = batch_invariant
    env["VLLM_BATCH_INVARIANT"] = "0" if batch_invariant is None else "1"
    env.update(ENV_PYTHON=str(fake), TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake),
        MODEL_PATH=str(tmp_path/"snapshot"), FAKE_LEDGER=str(tmp_path/"ledger.jsonl"),
        FAKE_FAIL="1" if fail_check else "0", MAX_TEACHER_CALLS="300", MAX_TEACHER_CALLS_RUN="3000",
        INCLUDE_BASE="1", READY_TIMEOUT="5", SEED_TEACHER_CACHE_FROM=seed_source)
    command = ["bash", "scripts/rai_mechanism.sh", "3", str(tmp_path/"gap.json"),
               str(tmp_path/"verified"), str(tmp_path/"out"), "student=/fake/adapter"]
    process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=20)
    assert process.returncode == int(fail_check), process.stderr
    rows = [json.loads(line) for line in (tmp_path/"ledger.jsonl").read_text().splitlines()]
    expected = ["preflight", "hf", "server", "check"]
    if not fail_check:
        expected += ["fce", "interventions", "interventions", "readout"]
    assert [r["kind"] for r in rows] == expected+["stopped"]
    for row in rows:
        assert row["batch_invariant"] == (batch_invariant or "1")
        assert row["cuda"] == ("3" if row["kind"] in {"server", "hf", "stopped"} else "")
        if row["kind"] in {"fce", "interventions"}:
            assert row["args"][row["args"].index("--workers")+1] == "8"
        if row["kind"] == "fce":
            assert row["args"][row["args"].index("--seed-teacher-cache-from")+1] == seed_source
        else:
            assert "--seed-teacher-cache-from" not in row["args"]
    if not fail_check:
        env["WORKERS"] = "4"
        again = subprocess.run(command, capture_output=True, text=True, env=env, timeout=20)
        assert again.returncode == 0, again.stderr
        assert len(list((tmp_path/"out/launches").glob("*"))) == 2
        resumed = [json.loads(line) for line in (tmp_path/"ledger.jsonl").read_text().splitlines()][len(rows):]
        assert all(r["args"][r["args"].index("--workers")+1] == "4"
                   for r in resumed if r["kind"] in {"fce", "interventions"})
