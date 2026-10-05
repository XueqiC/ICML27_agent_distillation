"""Stage 4f: fake HTTP clients, immutable evidence, and read-only real CPU smoke."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from analysis.mechanism import coverage, probes
from analysis.mechanism.common import read_json
from analysis.mechanism.readout import build_readout, markdown
from analysis.mechanism.taxonomy import classify, shares
from sed.config import load_config
from sed.react import build_messages
from sed.storage import digest, file_hash, write_json
from students.vllm_student import Student
from test_mechanism import episode, TASK_ID


def seed(actions=None, *, kind="pick_heat_then_place_in_recep", seed_id="seed"):
    actions = actions or ["go to drawer 1", "open drawer 1", "take mug 1 from drawer 1", "heat mug 1 with microwave 1"]
    def visible(i):
        return {"observation": f"Visible state {i}", "admissible_commands": actions + ["look"],
                "expert_plan": "SECRET", "internal_facts": ["SECRET"]}
    return {"schema_version": 2, "episode_id": seed_id, "task_id": f"train/{kind}-Mug-None-CounterTop-1/trial_0",
            "split": "train", "sealed": False, "initial": {"visible": visible(0)},
            "steps": [{"action_raw": "Thought: visible thought\nAction: " + a, "action_parsed": a,
                       "action_outcome": "legal", "environment_steps": 1, "event_id": str(i),
                       "visible": visible(i), "policy_metadata": {"hidden": "SECRET"}}
                      for i, a in enumerate(actions, 1)]}


@pytest.mark.parametrize("action,pattern,nav", [
    ("heat mug 23 with microwave 2", "heat {obj:mug} with {microwave}", False),
    ("take mug 2 from drawer 1", "take {obj:mug} from {drawer}", False),
    ("move mug 1 to countertop 2", "move {obj:mug} to {countertop}", False),
    ("put mug 1 in/on countertop 2", "move {obj:mug} to {countertop}", False),
    ("use desklamp 8", "use {desklamp}", False),
    ("open cabinet 7", "open {cabinet}", False),
    ("go to cabinet 7", "go to {cabinet}", True),
    ("examine mug 1", "examine {mug}", True),
    ("look", "look", True), ("inventory", "inventory", True),
    (None, None, False), ("invent mug", None, False),
])
def test_typed_patterns(action, pattern, nav):
    assert coverage.action_pattern(action) == {"pattern": pattern, "navigation_only": nav}


def test_coverage_all_substantive_patterns_and_scopes():
    one = seed()
    states = coverage.seed_states(one)
    other = coverage.seed_states(seed(kind="pick_and_place_simple", seed_id="other"))
    # Prefer same-task evidence even when it is later in collection order.
    result = coverage.cover_steps(one["steps"], 7, "pick_heat_then_place_in_recep", other + states)
    assert result["covered"] and result["covered_same_task_type"]
    take = result["needed_steps"][2]
    assert take["step"] == 10 and take["takeover_step"] == 3
    assert take["preferred_evidence"]["seed_episode_id"] == "seed"
    assert len(take["any_task_type"]) == 2 and len(take["same_task_type"]) == 1
    result = coverage.cover_steps(one["steps"], 0, "pick_heat_then_place_in_recep", other)
    assert result["covered"] and not result["covered_same_task_type"]
    missing = deepcopy(one["steps"])
    missing[0]["action_parsed"] = "go to sidetable 9"  # Unseen navigation cannot veto coverage.
    assert coverage.cover_steps(missing, 0, "x", states)["covered"] is True
    missing[-1]["action_parsed"] = "heat bowl 1 with microwave 1"
    assert coverage.cover_steps(missing, 0, "x", states)["covered"] is False
    invalid = deepcopy(states)
    invalid[-1]["demonstrated"] = False
    assert coverage.cover_steps(one["steps"], 0, "x", invalid)["covered"] is False
    assert coverage.cover_steps(one["steps"][:1], 0, "x", states)["status"] == "navigation_only"
    missing[-1]["action_parsed"] = "unknown command"
    assert coverage.cover_steps(missing, 0, "x", states)["covered"] is None


def test_seed_prefixes_match_training_and_never_include_oracles(monkeypatch):
    import train.dataset as dataset
    ep = seed()
    captured = []
    def encode(tokenizer, messages, reply, **kwargs):
        captured.append(messages)
        return {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1],
                "supervised_tokens": 1, "prompt_tokens": 1, "turn_forward_tokens": 2}
    monkeypatch.setattr(dataset, "encode_turn", encode)
    examples = dataset.episode_examples(ep, None, packing="turn")
    states = coverage.seed_states(ep)
    assert len(states) == len(ep["steps"])
    for i, state in enumerate(states):
        messages = build_messages(state["history"], {}, state["visible"])
        assert captured[i] == messages
        assert examples[i]["label_provenance"]["message_sha256"] == state["messages_sha256"]
        assert len(messages) == 2+2*i
        assert not any(x in json.dumps(messages) for x in ("SECRET", "internal_facts", ep["task_id"]))
    for split, prefix in (("dev", "valid_seen"), ("test", "valid_unseen")):
        invalid = deepcopy(ep)
        invalid.update(split=split, task_id=ep["task_id"].replace("train", prefix))
        with pytest.raises(ValueError):
            coverage.seed_states(invalid)


def test_shortest_successful_takeover_uses_suffix_and_records_sample(tmp_path, monkeypatch):
    parent = dict(seed(), split="dev", task_id=TASK_ID)
    t = 2
    receipts = []
    for i, distance in enumerate((3, 1)):
        ep = deepcopy(parent)
        ep.update(success=True, complete=True, status="ok")
        ep["steps"] = parent["steps"][:t] + seed(["use desklamp 1"]*distance)["steps"]
        ep["branch"] = {"parent_episode": parent["episode_id"], "parent_episode_sha256": digest(parent),
                        "t": t, "sample_index": i, "arm": "takeover", "replay_matched": True}
        path = tmp_path/f"sample{i}.json"
        write_json(path, ep)
        receipts.append({"episode_path": str(path), "success": True, "branch": ep["branch"]})
    monkeypatch.setattr(coverage, "completed_branch", lambda root: receipts[int(root.name[-1])])
    monkeypatch.setattr(coverage, "audit_states", lambda *a: [{"won": True}])
    row = {"task_id": TASK_ID, "t_star": t, "D_t_star": 1}
    (chosen, steps), status, samples = coverage.shortest_takeover(tmp_path/"fce.json", row, parent)
    assert status == "ok" and chosen["sample_index"] == 1
    assert len(steps) == 1 and steps[0]["action_parsed"] == "use desklamp 1"
    assert samples[0]["teacher_steps_to_go"] == 3
    # A lying teacher's claimed success is insufficient without the environment win.
    monkeypatch.setattr(coverage, "audit_states", lambda *a: [{"won": False}])
    with pytest.raises(ValueError, match="success disagrees"):
        coverage.shortest_takeover(tmp_path/"fce.json", row, parent)


def fake_factory(handler):
    return lambda **kwargs: Student(**kwargs, transport=httpx.MockTransport(handler))


SPEC = {"endpoint": "http://fake/v1", "model": "student", "revision": "fake-rev",
        "enable_thinking": False, "max_tokens": 1024}


def test_fake_student_queries_resume_failed_calls_and_validate_receipts(tmp_path):
    states = coverage.seed_states(seed())
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        i = (len(body["messages"])-2)//2
        if i == 1 and len(requests) <= len(states):
            return httpx.Response(500, text="SECRET HTTP ERROR")
        action = states[i]["seed_action"] if i != 2 else "look"
        return httpx.Response(200, json={"id": "fake", "created": 123, "model": "student",
            "choices": [{"message": {"content": "Action: " + action}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    output, config = tmp_path/"probes", load_config()
    args = dict(output=output, config=config, spec=SPEC, student_factory=fake_factory(handler))
    rows = probes.query_states(states, **args)
    assert [r["agrees"] for r in rows] == [True, None, False, True]
    for i, body in enumerate(requests):
        assert body["messages"] == build_messages(states[i]["history"], {}, states[i]["visible"])
        assert body["temperature"] == 0 and body["max_tokens"] == 1024
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "SECRET" not in json.dumps(body) and "logprobs" not in body
    assert "SECRET" not in "".join(p.read_text() for p in output.rglob("*.json*"))
    args["spec"] = dict(SPEC, endpoint="http://new-server/v1")
    rows = probes.query_states(states, **args)
    assert len(requests) == len(states)+1
    assert all(r["status"] == "ok" for r in rows)
    assert len(list(output.rglob("failure.json"))) == 1
    probes.query_states(states, **args)
    assert len(requests) == len(states)+1
    receipt = Path(rows[0]["receipt_path"])
    (receipt.parent/"payload/result.json").write_text("{}")
    with pytest.raises(ValueError, match="artifacts changed"):
        probes.query_states(states, **args)


def test_absorption_half_threshold_all_patterns_and_unknowns():
    one = seed(["take mug 1 from drawer 1", "take mug 1 from drawer 1", "heat mug 1 with microwave 1"])
    states = coverage.seed_states(one)
    cov = {"episodes": [{"task_id": TASK_ID, "t_star": 0,
                          **coverage.cover_steps(one["steps"], 0, "pick_heat_then_place_in_recep", states)}]}
    rows = [dict(coverage.state_evidence(s), agrees=a) for s, a in zip(states, [True, False, True])]
    report = probes.summarize(cov, rows)
    assert report["episodes"][0]["absorbed"] is True  # Exactly 1/2 for take.
    assert report["training_set_fit"]["overall"]["agreement"] == 2/3
    rows[-1]["agrees"] = False
    assert probes.summarize(cov, rows)["episodes"][0]["absorbed"] is False
    rows[-1]["agrees"] = None
    result = probes.summarize(cov, rows)
    assert result["episodes"][0]["absorbed"] is None
    assert result["training_set_fit"]["overall"]["unknown"] == 1
    assert result["training_set_fit"]["overall"]["agreement"] is None


def test_probe_workspace_end_to_end_resume_identity_and_readout_validation(tmp_path, monkeypatch):
    ep = seed()
    states = coverage.seed_states(ep)
    config = load_config()
    spec = dict(SPEC, revision=config["models"]["student"]["revision"])
    collection = {"path": str(tmp_path/"seeds"), "K": 1}
    cov = {"episodes": [{"task_id": TASK_ID, "t_star": 2,
                          **coverage.cover_steps(ep["steps"], 2, "pick_heat_then_place_in_recep", states)}],
           "gap_path": "unused", "student_binding": {"model": "student", "adapter": "fake"},
           "training_collection": collection}
    coverage_path = tmp_path/"coverage.json"
    write_json(coverage_path, cov)
    monkeypatch.setattr(probes, "validate_coverage", lambda *a: cov)
    monkeypatch.setattr(probes, "gap_episodes", lambda *a: ({}, [({"episode_path": "unused"}, {}, {})]))
    bindings = []
    monkeypatch.setattr(probes, "validate_continuation", lambda *a, **k: bindings.append(k))
    monkeypatch.setattr(probes, "load_training_states", lambda *a: (states, collection, []))
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        index = (len(body["messages"])-2)//2
        return httpx.Response(200, json={"model": "student", "choices": [{"message": {
            "content": "Action: "+states[index]["seed_action"]}, "finish_reason": "stop"}]})
    args = dict(coverage_path=coverage_path, output=tmp_path/"workspace", config=config,
                spec=spec, student_factory=fake_factory(handler))
    result = probes.run_probes(**args)
    assert result["complete"] and result["episodes"][0]["absorbed"]
    assert len(calls) == len(states) and bindings == [{"verified": collection["path"]}]
    path = tmp_path/"workspace/probes.json"
    assert probes.validate_probes(path, cov, config) == result
    args["spec"] = dict(spec, endpoint="http://new-port/v1")
    assert probes.run_probes(**args) == result and len(calls) == len(states)
    config["server_provenance"] = {"config": {"server": {"batch_invariant": True}}}
    with pytest.raises(ValueError, match="Resume identity changed"):
        probes.run_probes(**args)
    changed = deepcopy(result)
    changed["episodes"][0]["absorbed"] = False
    write_json(path, changed)
    with pytest.raises(ValueError, match="scoring changed"):
        probes.validate_probes(path, cov, config)


@pytest.mark.parametrize("covered,absorbed,label", [(True, False, "M1a"), (True, True, "M1b"),
    (False, True, "knowledge_not_in_experience"), (None, True, "M1_requires_coverage"), (True, None, "M1")])
def test_coverage_qualified_decision_table(covered, absorbed, label):
    ep, audits = episode(["look"])
    row = classify(ep, audits)
    tax = {"G": {"task_ids": [TASK_ID]}, "runs": {"student": {"episodes": [row], "shares": shares([row])}}}
    fce = {"episodes": [{"task_id": TASK_ID, "status": "closest_approach_found", "t_star": 0, "episode_length": 1}]}
    arms = {f"I{a}": {"samples": [{"sample_index": i, "success": a == 1} for i in range(5)]} for a in range(6)}
    interventions = {"episodes": [{"task_id": TASK_ID, "t": 0, "arms": arms, "task_type_covered": True}]}
    cov = {"episodes": [{"task_id": TASK_ID, "t_star": 0, "covered": covered}]}
    probe = {"episodes": [{"task_id": TASK_ID, "t_star": 0, "absorbed": absorbed}]}
    result = build_readout(tax, fce, interventions, cov, probe)
    assert result["episodes"][0]["label"] == label
    assert result["coverage_decision_cells"]["M1_cell"][label]["count"] == 1
    assert result["mechanism_attribution"][label]["n"] == 1
    assert "Coverage and absorption" in markdown(result)
    # Both-rescue cells retain the information signal, but never infer uncovered M1.
    for sample in arms["I3"]["samples"]:
        sample["success"] = True
    result = build_readout(tax, fce, interventions, cov, probe)
    if covered is False:
        assert result["episodes"][0]["label"] == "M2_with_knowledge_not_in_experience"
    if covered is not True:
        assert result["episodes"][0]["M1_subtype"] is None


def test_real_run2_coverage_cpu_every_gap_has_status(tmp_path):
    source = Path("reports/runs/mechanism/k20s0-p20-bi/run2")
    verified = Path("data/verified/seeds_k20_s0")
    if not (source/"fce/fce.json").exists() or not (verified/"manifest.json").exists():
        pytest.skip("Local read-only run2 artifacts not present")
    inputs = [source/"fce/fce.json", source/"interventions/interventions.json",
              source/"launches/attempt-000/readout/readout.json", verified/"manifest.json"]
    before = {str(p): file_hash(p) for p in inputs}
    result = coverage.main(["--fce", str(inputs[0]), "--verified", str(verified), "--output", str(tmp_path/"coverage")])
    gap = read_json(result["gap_path"])
    assert {r["task_id"] for r in result["episodes"]} == set(gap["G"]["task_ids"])
    assert len(result["episodes"]) == len(gap["G"]["task_ids"])
    assert all(r["status"] for r in result["episodes"])
    assert result["student_calls"] == result["teacher_calls"] == result["environment_steps"] == 0
    assert {str(p): file_hash(p) for p in inputs} == before
    # Coverage alone can recompute the existing table; absent probes must remain unknown.
    readout = build_readout(gap, read_json(inputs[0]), read_json(inputs[1]), result)
    assert all(r["absorbed"] is None for r in readout["episodes"])
    write_json(tmp_path/"coverage-readout.json", readout)


@pytest.mark.parametrize("failure", ["none", "check", "probes"])
def test_probe_launcher_gate_cleanup_cpu_and_new_readout_attempts(tmp_path, failure):
    fake = tmp_path/"fake-python"
    fake.write_text("#!"+sys.executable+"\n"+r'''
import json, os, signal, sys, time
from pathlib import Path
args=sys.argv[1:]
def value(flag): return args[args.index(flag)+1]
def record(kind):
    with open(os.environ["FAKE_LEDGER"], "a") as f:
        f.write(json.dumps({"kind":kind,"args":args,"cuda":os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "bi":os.environ.get("VLLM_BATCH_INVARIANT")})+"\n")
if args[0]=="-":
    data=json.loads(Path(args[1]).read_text());print(data.get("endpoint",data.get("gap_path")));sys.exit(0)
out=Path(value("--output"))
if args[:2]==["-m","analysis.mechanism.probe_launcher"]:
    record("preflight");out.mkdir(parents=True,exist_ok=True)
    launch=out/"launches"/str(len(list((out/"launches").glob("*"))))
    launch.mkdir(parents=True);print(launch);sys.exit(0)
out.mkdir(parents=True,exist_ok=args[0]=="-m")
if args[0]=="harness/serve_student.py":
    record("server")
    def stop(*_): record("stopped");sys.exit(0)
    signal.signal(signal.SIGTERM,stop)
    (out/"ready.json").write_text(json.dumps({"endpoint":"http://fake/v1"}))
    (out/"provenance.json").write_text("{}")
    while True: time.sleep(.01)
elif args[0]=="tools/check_lora_serving.py":
    kind="hf" if "--hf-only" in args else "check";record(kind)
    if kind==os.environ["FAKE_FAIL"]: sys.exit(1)
    (out/("reference.json" if kind=="hf" else "report.json")).write_text("{}")
elif args[0]=="-m":
    kind=args[1].rsplit(".",1)[1];record(kind)
    (out/(kind+".json")).write_text("{}")
    if kind==os.environ["FAKE_FAIL"]:sys.exit(2)
else:raise AssertionError("Unexpected fake command")
''')
    fake.chmod(0o755)
    (tmp_path/"snapshot").mkdir()
    source = tmp_path/"source"
    write_json(source/"fce/fce.json", {"gap_path": str(tmp_path/"gap.json")})
    before = file_hash(source/"fce/fce.json")
    env = os.environ.copy()  # Only the selected fields above are recorded.
    env.update(ENV_PYTHON=str(fake), TRAIN_PYTHON=str(fake), VLLM_PYTHON=str(fake),
        MODEL_PATH=str(tmp_path/"snapshot"), FAKE_LEDGER=str(tmp_path/"ledger.jsonl"),
        FAKE_FAIL=failure, READY_TIMEOUT="5", BATCH_INVARIANT="1")
    command = ["bash", "scripts/rai_mechanism_probes.sh", "3", str(source), str(tmp_path/"verified"),
               str(tmp_path/"new-out"), "student=/fake/adapter"]
    status = {"none": 0, "check": 1, "probes": 2}[failure]
    for attempt in range(2):
        process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=20)
        assert process.returncode == status, process.stderr
    rows = [json.loads(line) for line in (tmp_path/"ledger.jsonl").read_text().splitlines()]
    expected = ["preflight", "coverage", "hf", "server", "check"]
    if failure != "check":
        expected += ["probes", "readout"]
    expected += ["stopped"]
    assert [r["kind"] for r in rows] == expected*2
    for row in rows:
        assert row["bi"] == "1"
        assert row["cuda"] == ("3" if row["kind"] in {"hf", "server", "stopped"} else "")
        if row["kind"] == "server":
            assert row["args"].count("--lora-adapter") == 1
            assert row["args"][row["args"].index("--port")+1] == "0"
    outputs = [r["args"][r["args"].index("--output")+1] for r in rows if r["kind"] == "readout"]
    assert len(set(outputs)) == len(outputs)
    assert file_hash(source/"fce/fce.json") == before
