"""Offline IPR evidence, official pair rules, orchestration and sampling seeds."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from baselines.ipr import pipeline
from baselines.ipr.branch import run_branch
from baselines.ipr.pairs import build_pairs, mc_mean
from baselines.ipr.rollouts import plan_rollouts, dispatch, rollout_one, run_rollouts, checked_rollouts
from baselines.ipr.sampling import MCStudent
from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits, resolve_task
from sed.runner import ScriptedPolicy, run_episode
from sed.storage import digest, write_json
from students.sag_student import SAGStudent
from test_stage3b import response
from test_stage2b_seeds import seed_bank, factories


def messages(n, label="expert"):
    result = [{"role": "system", "content": "instruction"}, {"role": "user", "content": "task"}]
    for i in range(n):
        result.append({"role": "assistant", "content": f"{label}-{i}"})
        if i + 1 < n:
            result.append({"role": "user", "content": f"obs-{i}"})
    return result


def pair_inputs(j=1, ef=1., af=0., em=.8, am=.2, shorter=False):
    gold = messages(4)
    agent = deepcopy(gold[:2 + j * 2]) + messages(1 if shorter else 4 - j, "agent")[2:]
    return ([{"task_id": "train/fake", "messages": gold, "parent_seed_ids": ["seed"],
              "final_reward": ef, "mc": {j: em}}],
            [{"task_id": "train/fake", "j": j, "messages": agent, "final_reward": af, "mc": am}])


def original_pairs(experts, agents, tmp_path, traj_threshold=.01, step_threshold=.01):
    """Execute ONLY pinned build_preference, with in-memory inputs; no imports/keys."""
    source = Path("third_party/IPR/construct_preference_monte_carlo_alfworld.py").read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_preference")
    expert, agent = experts[0], agents[0]
    j = agent["j"]
    def convert(msgs):
        return [{"from": "gpt" if m["role"] == "assistant" else "human", "value": m["content"]} for m in msgs]
    golden = convert(expert["messages"])
    golden.insert(1, {"from": "gpt", "value": "OK"})
    pred = [{"from": "human", "value": "ICL"}] * 12 + convert(agent["messages"][1:])
    pred.append({"from": "human", "value": "terminal observation"})
    data = {f"1_{j}": {"gpt_reward": expert["final_reward"], "gpt_conversations": golden,
        "gpt_step_conversations": golden[:4 + 2*j], "monte_carlo_step_reward": expert["mc"][j]}}
    paths = {"data/sft_data_alfworld.json": data, "data/game_file_to_id.json": {"game": 1}}
    def fake_open(path, mode="r"):
        return io.StringIO(json.dumps(paths[path])) if path in paths else open(path, mode)
    scope = {"json": json, "open": fake_open, "instruction_part": {"alfworld": [0]*2},
        "icl_prompt_part": {"alfworld": [0]*12}, "template_change": lambda x: x,
        "is_empty_conversations": lambda c: any(not m["value"].strip() for m in c),
        "cal_monte_carlo_reward": lambda _: [{"game_file": "game", "iteration": j,
            "agent_final_reward": agent["final_reward"], "agent_conversations": pred,
            "agent_step_conversations": pred[:14 + 2*j], "monte_carlo_step_reward": agent["mc"]}]}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "isolated_IPR_pair_rule", "exec"), scope)
    output = tmp_path / "official.json"
    with redirect_stdout(io.StringIO()):
        scope["build_preference"](SimpleNamespace(task="alfworld", global_traj=True, local_traj=True,
            step_action=False, traj_threshold=traj_threshold, step_threshold=step_threshold, output_path=str(output)))
    return json.loads(output.read_text())


@pytest.mark.parametrize("j,ef,af,em,am,shorter,winner", [
    (0,1,0,0,1,False,"expert"), (0,0,1,1,0,False,"agent"), (0,1,1,1,0,False,None),
    (0,.51,.5,1,0,False,None), (1,1,0,.8,.2,False,"expert"),
    (1,1,0,.51,.5,False,None), (1,1,1,.8,.2,False,None),
    (1,1,1,.2,.8,True,"agent"), (1,1,1,.2,.8,False,None),
    (1,1,0,.2,.8,True,"agent"), (1,0,1,.2,.8,False,"agent"),
    (1,1,1,.5,.51,True,None), (1,.51,.5,.8,.2,False,None),
])
def test_pair_rule_matches_official_both_kinds_and_flag(tmp_path,j,ef,af,em,am,shorter,winner):
    experts, agents = pair_inputs(j,ef,af,em,am,shorter)
    actual, report = build_pairs(experts, agents)
    expected = original_pairs(experts, agents, tmp_path)
    assert len(actual) == len(expected) == int(winner is not None)
    if winner:
        assert report[winner + "_chosen"] == 1
        for side in ("chosen", "rejected"):
            assert [m["content"] for m in actual[0][side]] == [m["value"] for m in expected[0][side]]
        assert [m["content"] for m in actual[0]["prompt"]] == [m["value"] for i,m in enumerate(expected[0]["prompt"]) if i != 1]
        assert report["P_by_kind"]["trajectory" if j == 0 else "entire"] == 1


def test_mc_unknown_empty_and_prompt_integrity():
    assert mc_mean([True, False, True, True, False]) == .6
    assert mc_mean([True]*4) is None and mc_mean([True]*4 + ["unknown"]) is None
    experts, agents = pair_inputs()
    agents[0]["mc"] = None
    assert build_pairs(experts, agents)[1]["unknown"] == 1
    agents[0]["mc"] = .2
    agents[0]["messages"][-1]["content"] = ""
    assert build_pairs(experts, agents)[1]["empty"] == 1
    agents[0]["messages"][0]["content"] = "injected oracle"
    with pytest.raises(ValueError, match="prefix"):
        build_pairs(experts, agents)


def fake_env(config, task):
    return Environment(config["environment"]["data_root"], seed=0, max_episode_steps=40, backend_factory=FakeBackend)


class FakeStudent(ScriptedPolicy):
    def __init__(self, **kwargs):
        super().__init__(["look"])
        self.spec = {k:v for k,v in kwargs.items() if k != "log_dir"}
    def close(self):
        pass


class FakeMCStudent(MCStudent):
    def __init__(self, **kwargs):
        def handler(request):
            body = json.loads(request.content)
            value = response(["Action: " + ("look" if body["seed"] % 2 else "go north")])
            value["model"] = body["model"]
            return httpx.Response(200,json=value)
        settings = dict(endpoint="http://fake/v1",model="gemma",revision="pin")
        settings.update(kwargs)
        super().__init__(transport=httpx.MockTransport(handler),**settings)


def fake_worker(job):
    return rollout_one(job, env_factory=fake_env, student_factory=FakeStudent, mc_factory=FakeMCStudent)


def branch_fixture(data_root, tmp_path):
    game = data_root / "train/room_a/trial_0/game.tw-pddl"
    game.write_text(json.dumps({"solvable": True, "win_at": 3}))
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    task = resolve_task(manifest, freeze, "train/room_a/trial_0")
    config = load_config()
    config["environment"].update(data_root=str(data_root), seed=0, max_episode_steps=40)
    parent = run_episode(fake_env(config, task), FakeStudent(), task, config, tmp_path / "parent",
                         policy_condition="fake_expert")
    return task, config, parent


def test_expert_mc_replays_only_expert_actions_and_terminal_prefix(data_root, tmp_path):
    task, config, parent = branch_fixture(data_root, tmp_path)
    plan = plan_rollouts([(task,parent)], kind="expert-mc")
    assert len(plan) == 15
    assert [p[2]["prefix"] for p in plan] == [1]*5 + [2]*5 + [3]*5
    assert len({p[2]["rollout_seed"] for p in plan}) == 15
    for prefix in (1,2,3):
        policy = FakeStudent()
        env = fake_env(config, task)
        ep = run_branch(env, policy, task, config, parent, prefix, tmp_path / f"branch-{prefix}",
                        lineage=plan[5*(prefix-1)][2])
        assert ep["success"] is True and ep["ipr_branch"]["replay_matched"]
        assert policy.index == 3-prefix  # Terminal expert action produces no extra request.
        assert [s["action_parsed"] for s in ep["steps"]] == ["look"]*3
        assert [s["visible"] for s in ep["steps"][:prefix]] == [s["visible"] for s in parent["steps"][:prefix]]
    changed = deepcopy(parent)
    changed["steps"][0]["audit_sha256"] = "changed"
    with pytest.raises(RuntimeError):
        run_branch(fake_env(config,task), FakeStudent(),task,config,changed,1,tmp_path / "mismatch",lineage=plan[0][2])
    bad = json.loads((tmp_path / "mismatch/episode.json").read_text())
    assert bad["success"] == "unknown" and not bad["ipr_branch"]["replay_matched"]


def test_parallel_rollout_results_independent_of_workers(data_root, tmp_path):
    task, config, parent = branch_fixture(data_root, tmp_path)
    plan = plan_rollouts([(task,parent)], kind="expert-mc")
    results = []
    for workers in (1,4):
        jobs = [(t,e,lineage,config,{"enable_thinking":False,"max_tokens":512},
                 str(tmp_path / f"w{workers}" / str(i)),[]) for i,(t,e,lineage) in enumerate(plan)]
        receipts = dispatch(jobs,workers,fake_worker)
        projection = []
        for r in receipts:
            ep = json.loads((Path(r["directory"]) / "episode/episode.json").read_text())
            projection.append({"seed":r["rollout_seed"],"j":r["j"],"sample":r["sample"],"success":r["success"],
                "actions":[s["action_raw"] for s in ep["steps"]],"observations":[s["visible"] for s in ep["steps"]],
                "request_seeds":[s["policy_metadata"].get("seed") for s in ep["steps"]]})
        results.append(projection)
    assert results[0] == results[1]


@pytest.mark.parametrize("cls,cap", [(SAGStudent,1024),(MCStudent,512)])
def test_seeded_fake_server_requests_reproducible(tmp_path,cls,cap):
    runs = []
    visible = {"observation":"room","admissible_commands":["look","go north"]}
    for run in range(2):
        requests, metadata = [], []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            # Fake server output actually depends on supplied seed.
            action = "look" if body["seed"] % 2 else "go north"
            return httpx.Response(200,json=response(["Action: "+action]*body.get("n",1)))
        student = cls(endpoint="http://fake/v1",model="gemma",revision="pin",enable_thinking=False,
            max_tokens=cap,sampling_seed=17,log_dir=tmp_path / str(run),transport=httpx.MockTransport(handler),
            **({"task_id":"train/seed-1"} if cls is SAGStudent else {}))
        history = []
        for turn in range(3):
            value = student.generate(history,{},visible)
            assert value.metadata["seed"] == requests[-1]["seed"]
            metadata.append((value.raw_text,value.metadata["seed"]))
            history.append({"visible":visible,"reply":value.raw_text})
        student.close()
        assert len({r["seed"] for r in requests}) == 3
        assert "train/seed-1" not in json.dumps(requests)
        if cls is SAGStudent:
            assert requests[0]["seed"] == int(digest([17,"train/seed-1",0])[:8],16)
        runs.append((requests,metadata))
    assert runs[0] == runs[1]


@pytest.mark.parametrize("pair_count", [0,2])
def test_pipeline_six_iterations_fixed_reference_resume_and_carry(tmp_path,monkeypatch,pair_count):
    commands, rollouts = [], []
    def run(command,log,**kwargs):
        commands.append(command)
        out = Path(command[command.index("--output")+1])
        (out / "checkpoint").mkdir(parents=True)
        write_json(out / "result.json",{"examples_seen":6,"logical_forward_tokens":42})
    def rollout(args,snapshot,attempt,**kwargs):
        rollouts.append((str(snapshot),kwargs))
        (attempt / "output").mkdir()
        write_json(attempt / "output/receipts.json",[])
    def pairs(v,e,g,a,out,config):
        out.mkdir()
        write_json(out / "pairs.json",[{}]*pair_count)
        write_json(out / "report.json",{"P":pair_count,"P_by_kind":{"trajectory":pair_count,"entire":0}})
    monkeypatch.setattr(pipeline,"run_command",run)
    monkeypatch.setattr(pipeline,"rollout_stage",rollout)
    monkeypatch.setattr(pipeline,"prepare_pairs",pairs)
    args = SimpleNamespace(output=str(tmp_path),snapshot="base",verified="seeds",config="cfg",train_python="train",gpus=4,
                           through="all",workers=1)
    identity = {"K":5,"unique_instances":5,"task_groups":5}
    pipeline.train_pipeline(args,identity,{})
    assert len(commands) == (7 if pair_count else 1)
    sft = str(tmp_path / "stages/train-sft/attempt-000/output/checkpoint")
    assert len(rollouts) == 13
    assert all(path == sft for path,kw in rollouts if kw["kind"] != "explore")
    for command in commands[1:]:
        assert command[command.index("--reference")+1] == sft
        assert command[command.index("--objective")+1] == "ipr"
    records = [json.loads((tmp_path / f"iteration-{i}.json").read_text()) for i in range(1,7)]
    assert all(r["carry_forward"] == (pair_count == 0) for r in records)
    assert len(json.loads((tmp_path / "models.json").read_text())) == 7
    n = len(commands)
    args.workers = 4
    pipeline.train_pipeline(args,identity,{})
    assert len(commands) == n and len(rollouts) == 13
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "complete"


def test_ipr_training_recipe():
    args = SimpleNamespace(train_python="TRAIN",config="cfg",snapshot="base",verified="seeds",gpus=4)
    for objective in ("sft","ipr"):
        cmd = pipeline.training_command(args,objective=objective,snapshot="current",output="out",reference="sft",pairs="p")
        opts = {cmd[i]:cmd[i+1] for i in range(len(cmd)-1) if cmd[i].startswith("--")}
        assert opts["--original-batch"] == "48" and opts["--seed"] == "0"
        assert opts["--epochs"] == "3" and opts["--max-length"] == "32768"
        assert float(opts["--learning-rate"]) == (2e-5 if objective == "sft" else 3e-6)
        if objective == "ipr":
            assert opts["--reference"] == "sft" and opts["--snapshot"] == "current"


def test_agent_mc_replays_expert_prefix_plus_agent_action(data_root,tmp_path):
    task,config,parent = branch_fixture(data_root,tmp_path)
    explore = plan_rollouts([(task,parent)],kind="explore",iteration=1)[1]
    policy = FakeStudent()
    policy.actions = ["go north"]
    agent = run_branch(fake_env(config,task),policy,task,config,parent,1,tmp_path / "agent",lineage=explore[2])
    # All agent branches must be supplied to the exhaustive planner.
    explored = [(dict(explore[2],j=j),agent) for j in range(3)]
    plan = plan_rollouts([(task,parent)],kind="agent-mc",iteration=1,exploration=explored)
    _,mc_parent,lineage = plan[5]
    assert lineage["j"] == 1 and lineage["prefix"] == 2
    mc = run_branch(fake_env(config,task),FakeStudent(),task,config,mc_parent,2,tmp_path / "mc",lineage=lineage)
    assert [s["action_parsed"] for s in mc["steps"]] == ["look","go north","look"]


def test_prefix_parser_feedback_counts_toward_40_turn_cap(data_root,tmp_path):
    from teachers.openai_teacher import TeacherResult
    task,config,_ = branch_fixture(data_root,tmp_path)
    class Invalid(FakeStudent):
        def generate(self,*args):
            return TeacherResult("unparseable",None,{})
    parent = run_episode(fake_env(config,task),Invalid(),task,config,tmp_path / "invalid-parent",
                         policy_condition="fake")
    assert len(parent["steps"]) == 40 and not parent["success"]
    lineage = {"kind":"agent-mc","parent_seed_id":"seed"}
    policy = FakeStudent()
    episode = run_branch(fake_env(config,task),policy,task,config,parent,39,tmp_path / "last-turn",lineage=lineage)
    assert policy.index == 1 and episode["truncated"] and episode["ipr_branch"]["replay_matched"]
    assert len(episode["steps"]) == 40 and episode["metrics"]["environment_steps"] == 1
    assert sum(s["action_outcome"] == "parser_error" for s in episode["steps"]) == 39


def test_evidence_chain_pair_reconstruction_and_tamper(seed_bank,tmp_path):
    from teachers.collect_seeds import collect
    from sed.seed_selection import select_seeds
    from baselines.ipr.pairs import prepare_pairs, load_pair_data
    root,_,_,config = seed_bank
    for p in root.rglob("game.tw-pddl"):
        p.write_text(json.dumps({"solvable":True,"win_at":3}))
    manifest = build_manifest(root)
    freeze = freeze_splits(manifest)
    config["environment"]["max_episode_steps"] = 40
    config["paths"].update(manifest=str(tmp_path / "manifest.json"),split_freeze=str(tmp_path / "freeze.json"))
    write_json(config["paths"]["manifest"],manifest)
    write_json(config["paths"]["split_freeze"],freeze)
    _,teacher = factories(root,config,set(),[])
    verified = tmp_path / "verified"
    collect(select_seeds(manifest,freeze,k=1),manifest,freeze,config,verified,
            env_factory=lambda task:fake_env(config,task),teacher_factory=teacher)
    server = tmp_path / "server.json"
    write_json(server,{"config":{"server":{"batch_invariant":True,"model_path":"sft", "training_provenance_sha256":"sft-hash"}}})
    outputs = {}
    for kind in ("expert-mc","explore","agent-mc"):
        output = tmp_path / kind
        result = run_rollouts(config=config,verified=verified,kind=kind,iteration=0 if kind=="expert-mc" else 1,
            output=output,spec={"enable_thinking":False,"max_tokens":1024 if kind=="explore" else 512,
                "model":config["models"]["student"]["requested_repo"],"revision":config["models"]["student"]["revision"]},
            server_provenance=server,exploration=outputs.get("explore") if kind=="agent-mc" else None,
            workers=1,worker_fn=fake_worker)
        assert result["eligible"] == result["rollouts"]
        assert result["K"] == result["unique_instances"] == result["task_groups"] == 1
        assert len(checked_rollouts(output,config)) == result["rollouts"]
        outputs[kind] = output
    pairs = tmp_path / "pairs"
    report = prepare_pairs(verified,outputs["explore"],outputs["expert-mc"],outputs["agent-mc"],pairs,config)
    assert report["P"] == 0 and load_pair_data(pairs / "pairs.json",config) == []
    # Edits to raw model-call evidence are rejected, not just episode edits.
    receipt = json.loads((outputs["expert-mc"] / "receipts.json").read_text())[0]
    calls = outputs["expert-mc"] / receipt["directory"] / "calls/calls.jsonl"
    calls.write_text(calls.read_text()+"{}\n")
    with pytest.raises(ValueError,match="evidence changed"):
        load_pair_data(pairs / "pairs.json",config)
