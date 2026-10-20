"""Full-TRAIN selection, process concurrency, durable caps/resume, real loaders; no API."""

from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.config import load_config
from sed.manifest import build_manifest, freeze_splits, load_frozen
from sed.seed_evidence import load_verified_collection, materialize_subsets
from sed.seed_selection import ALL_TRAIN_RULE, main as select_main, ordered_candidates, select_seeds
from sed.storage import JsonlLedger, digest, write_json
from teachers.collect_seeds import collect
from teachers.full_train import preflight
from teachers.openai_teacher import Teacher, TeacherCallError


def test_frozen_full_selection_counts_determinism_and_cli(tmp_path):
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json",
                                   "data/manifest/alfworld_split_freeze.json")
    selection = select_seeds(manifest, freeze, rule=ALL_TRAIN_RULE)
    assert selection == json.loads(Path("data/manifest/seeds_full_train.json").read_text())
    assert selection == select_seeds(manifest, freeze, rule=ALL_TRAIN_RULE, k=1)
    assert selection["K"] == selection["candidates"] == selection["selected_unique_instances"] == 3553
    assert selection["selected_task_groups"] == 1465 and selection["trusted_trajectories"] == 0
    assert selection["K_counts"] == "candidates"
    assert selection["quotas"] == {"look_at_obj_in_light": 308, "pick_and_place_simple": 790,
        "pick_clean_then_place_in_recep": 650, "pick_cool_then_place_in_recep": 533,
        "pick_heat_then_place_in_recep": 459, "pick_two_obj_and_place": 813}
    assert not any(selection["replacements"].values())
    rows = ordered_candidates(selection)
    assert [r["task_id"] for r in rows] == selection["candidate_order"]
    assert len({r["task_id"] for r in rows}) == 3553
    assert set(selection["candidate_order"]) == set(freeze["seed_pool_candidates"])
    assert [digest([0, r["task_id"]]) for r in rows] == sorted(digest([0, r["task_id"]]) for r in rows)
    assert select_seeds(manifest, freeze, rule=ALL_TRAIN_RULE, seed=1)["candidate_order"] != selection["candidate_order"]
    reversed_manifest = dict(manifest, tasks=list(reversed(manifest["tasks"])))
    assert select_seeds(reversed_manifest, dict(freeze, manifest_sha256=digest(reversed_manifest)),
                        rule=ALL_TRAIN_RULE)["candidate_order"] == selection["candidate_order"]
    output = tmp_path / "selection.json"
    select_main(["--rule", ALL_TRAIN_RULE, "--output", str(output)])
    assert json.loads(output.read_text()) == selection
    # Existing default selections must remain byte-for-byte equivalent as objects.
    assert select_seeds(manifest, freeze, k=100) == json.loads(Path("data/manifest/seeds_k100_s0.json").read_text())


@pytest.fixture
def full_bank(data_root, tmp_path):
    manifest = build_manifest(data_root)
    freeze = freeze_splits(manifest)
    selection = select_seeds(manifest, freeze, rule=ALL_TRAIN_RULE)
    config = load_config()
    config["environment"].update(data_root=str(data_root), max_episode_steps=1)
    for name, value in (("manifest", manifest), ("split_freeze", freeze)):
        config["paths"][name] = str(tmp_path / (name + ".json"))
        write_json(config["paths"][name], value)
    return data_root, manifest, freeze, selection, config


def factories(bank, calls, losing=()):
    root, _, _, _, config = bank
    def environment(task):
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 99 if task["task_id"] in losing else 1
            return result
        return Environment(root, max_episode_steps=1, backend_factory=backend)

    def teacher(task, cache, ledger):
        class Responses:
            def create(self, **kwargs):
                assert kwargs["model"] == config["models"]["teacher"]["provider_model_id"]
                assert kwargs["max_output_tokens"] == config["models"]["teacher"]["sampling"]["max_output_tokens"]
                JsonlLedger(calls).append({"task_id": task["task_id"], "cache": str(cache)})
                return {"output_text": "Thought: inspect.\nAction: look", "model": "gpt-6-luna", "status": "completed",
                        "usage": {"input_tokens": 20, "output_tokens": 10}, "id": "fake"}
        spec = config["models"]["teacher"]
        return Teacher(cache_dir=cache, usage_ledger=ledger, run_id="fake",
                       client=SimpleNamespace(responses=Responses()), sampling=spec["sampling"],
                       max_attempts=spec["max_attempts"])
    return environment, teacher


def run_shards(bank, output, calls, *, cap=160_000, losing=()):
    _, manifest, freeze, selection, config = bank
    environment, teacher = factories(bank, calls, losing)
    def worker(index):
        collect(selection, manifest, freeze, config, output, env_factory=environment,
                teacher_factory=teacher, shard_index=index, shard_count=2, resume=True, max_calls=cap)
    processes = [multiprocessing.get_context("fork").Process(target=worker, args=(i,)) for i in range(2)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(30)
            assert process.exitcode == (1 if cap == 1 else 0)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    return json.loads((output / "manifest.json").read_text())


def test_process_shards_one_collection_no_replacements_and_resume(full_bank, tmp_path):
    _, manifest, freeze, selection, _ = full_bank
    output, calls = tmp_path / "collection", tmp_path / "calls.jsonl"
    failed = selection["candidate_order"][0]
    result = run_shards(full_bank, output, calls, losing={failed})
    assert result["status"] == "complete" and result["K"] == 3 and result["candidates"] == 4
    assert result["trusted_trajectories"] == result["unique_instances"] == 3
    assert result["task_groups"] == 2  # Sibling trials are valid trusted trajectories.
    assert result["failed_candidates"] == [failed]
    assert len(result["attempts"]) == len(calls.read_text().splitlines()) == 6
    assert result["budget"]["logical_calls"] == result["budget"]["transport_reservations"] == 6
    assert all(r["selection_parent_task_id"] == r["task_id"] for r in result["attempts"])
    assert [r["task_id"] for r in result["verified"]] == selection["candidate_order"][1:]
    assert len(list((output / "data/teacher_rate_limit").glob("*.json"))) == 1
    assert load_verified_collection(output, manifest, freeze)[0] == result
    assert run_shards(full_bank, output, calls, losing={failed}) == result
    assert len(calls.read_text().splitlines()) == 6
    with pytest.raises(ValueError, match="nested"):
        materialize_subsets(output, manifest, freeze, [1])
    bad = deepcopy(result)
    bad["K"] = selection["K"]
    write_json(output / "manifest.json", bad)
    with pytest.raises(ValueError, match="Full TRAIN"):
        load_verified_collection(output, manifest, freeze)


@pytest.mark.parametrize("caps,expected", [({"max_calls": 1}, 1), ({"max_input_tokens": 1}, 0)])
def test_cap_stops_cleanly_and_resume_never_repays(full_bank, tmp_path, caps, expected):
    _, manifest, freeze, selection, config = full_bank
    calls, output = tmp_path / "calls.jsonl", tmp_path / "collection"
    environment, teacher = factories(full_bank, calls)
    kwargs = dict(env_factory=environment, teacher_factory=teacher, **caps)
    for resume in (False, True):
        with pytest.raises(TeacherCallError, match="NOT complete"):
            collect(selection, manifest, freeze, config, output, resume=resume, **kwargs)
    result = json.loads((output / "manifest.json").read_text())
    assert result["status"] == "budget_exhausted"
    assert result["budget"]["logical_calls"] == expected
    assert all(r["status"] == "ok" for r in result["attempts"])
    assert (len(calls.read_text().splitlines()) if calls.exists() else 0) == expected
    with pytest.raises(ValueError, match="exactly K"):
        load_verified_collection(output, manifest, freeze)


def test_process_shards_share_hard_cap(full_bank, tmp_path):
    calls, output = tmp_path / "calls.jsonl", tmp_path / "collection"
    result = run_shards(full_bank, output, calls, cap=1)
    assert result["status"] == "budget_exhausted"
    assert result["budget"]["logical_calls"] == result["budget"]["transport_reservations"] == 1
    assert len(calls.read_text().splitlines()) == 1
    again = run_shards(full_bank, output, calls, cap=1)
    assert again == result and len(calls.read_text().splitlines()) == 1


def test_resume_after_episode_before_receipt_and_failed_attempts(full_bank, tmp_path, monkeypatch):
    import teachers.collect_seeds as module
    _, manifest, freeze, selection, config = full_bank
    calls, output = tmp_path / "calls.jsonl", tmp_path / "collection"
    environment, teacher = factories(full_bank, calls, {selection["candidate_order"][0]})
    original = module.write_json
    def interrupt(path, value, **kwargs):
        if Path(path).name == "receipt.json" and value["won"]:
            raise KeyboardInterrupt
        return original(path, value, **kwargs)
    monkeypatch.setattr(module, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect(selection, manifest, freeze, config, output, env_factory=environment, teacher_factory=teacher)
    assert len(calls.read_text().splitlines()) == 4
    monkeypatch.setattr(module, "write_json", original)
    result = collect(selection, manifest, freeze, config, output, resume=True,
                     env_factory=environment, teacher_factory=teacher)
    assert result["status"] == "complete" and result["K"] == 3
    assert len(calls.read_text().splitlines()) == 6  # Neither failures nor the success were repaid.
    assert load_verified_collection(output, manifest, freeze)[0] == result


def test_read_only_preflight_and_train_derived_dev_exclusion(full_bank, tmp_path):
    _, manifest, freeze, selection, config = full_bank
    output = tmp_path / "never-created"
    preflight(selection, manifest, freeze, config, output, shard_count=2)
    assert not output.exists()
    derived = freeze_splits(manifest, dev_source="train", dev_count=1)
    subset = select_seeds(manifest, derived, rule=ALL_TRAIN_RULE)
    dev_groups = {r["group_id"] for r in derived["assignments"].values() if r["split"] == "dev"}
    assert all(r["group_id"] not in dev_groups for r in ordered_candidates(subset))
    for kwargs in ({"attempts": 2}, {"max_calls": 160001}, {"max_input_tokens": 700000001}):
        with pytest.raises(ValueError):
            preflight(selection, manifest, freeze, config, output, **kwargs)


def test_seed_sft_and_baseline_loaders_accept_full(full_bank, tmp_path, monkeypatch):
    from baselines.common import pipeline as common
    from baselines.eto import pipeline as eto
    from train.dataset import build_verified_dataset
    from test_r03_sets import TinyTokenizer
    import train.dose_grid as dose

    _, manifest, freeze, selection, config = full_bank
    output = tmp_path / "collection"
    environment, teacher = factories(full_bank, tmp_path / "calls.jsonl", {selection["candidate_order"][0]})
    result = collect(selection, manifest, freeze, config, output, env_factory=environment, teacher_factory=teacher)
    tokenizer = TinyTokenizer()
    tokenizer.chat_template = "fake Gemma OFF template"
    examples = build_verified_dataset(output, tokenizer, manifest, freeze)
    assert len(examples) == result["K"] == 3
    from baselines.agentgym.data import seed_rows
    from baselines.ipr.rollouts import seeds as ipr_seeds
    from baselines.leap.data import seeds as leap_seeds
    for loader in (seed_rows, ipr_seeds, leap_seeds):
        assert len(loader(output, config)) == 3
    for method in ("eto", "ipr", "agentgym", "leap", "kang"):
        pipeline = eto if method == "eto" else common
        args = SimpleNamespace(output=tmp_path / method, verified=output, snapshot=tmp_path / "snapshot",
                               config="configs/base.yaml", gpus=1)
        kwargs = {} if method == "eto" else {"method": method, "recipe": {}}
        identity = pipeline.prepare(args, config, **kwargs)
        assert identity["K"] == 3 and identity["task_groups"] == 2
        assert len(json.loads((args.output / "tasks.json").read_text())["task_ids"]) == 3
    template = json.loads(Path("configs/seedsft_dose_grid_full_train.json").read_text())
    template["runs"][0]["verified"] = str(output)
    grid = dose.prepare_grid(template, tokenizer, manifest, freeze)
    run = grid["runs"][0]
    assert run["name"] == "full" and run["K"] == 3 and run["candidates"] == 4
    write_json(tmp_path / "ready.json", grid)
    assert dose.load_ready_run(tmp_path / "ready.json", 0) == run


def test_missing_budget_cannot_reset_caps(full_bank, tmp_path):
    _, manifest, freeze, selection, config = full_bank
    calls, output = tmp_path / "calls.jsonl", tmp_path / "collection"
    environment, teacher = factories(full_bank, calls)
    with pytest.raises(TeacherCallError):
        collect(selection, manifest, freeze, config, output, max_calls=1,
                env_factory=environment, teacher_factory=teacher)
    (output / "teacher_budget/reservations.sqlite3").unlink()
    with pytest.raises(ValueError, match="refusing to reset"):
        collect(selection, manifest, freeze, config, output, max_calls=1, resume=True,
                env_factory=environment, teacher_factory=teacher)
    assert len(calls.read_text().splitlines()) == 1


def test_keeps_first_success_after_retry(full_bank, tmp_path):
    root, manifest, freeze, selection, config = full_bank
    calls, output = tmp_path / "calls.jsonl", tmp_path / "collection"
    _, teacher = factories(full_bank, calls)
    attempts = {}
    def environment(task):
        tid = task["task_id"]
        attempts[tid] = attempts.get(tid, 0) + 1
        def backend(path, seed):
            result = FakeBackend(path, seed)
            result.win_at = 1 if attempts[tid] == 2 else 99
            return result
        return Environment(root, max_episode_steps=1, backend_factory=backend)
    result = collect(selection, manifest, freeze, config, output, env_factory=environment, teacher_factory=teacher)
    assert result["K"] == 4 and not result["failed_candidates"]
    assert len(result["attempts"]) == len(calls.read_text().splitlines()) == 8
    assert all(r["attempt"] == 2 for r in result["verified"])
    assert load_verified_collection(output, manifest, freeze)[0] == result


def test_launcher_preflight_resume_and_shared_worker_arguments(tmp_path):
    fake, ledger = tmp_path / "fake-python", tmp_path / "invocations.jsonl"
    fake.write_text("#!" + sys.executable + "\n" + '''
import fcntl, json, os, sys
from pathlib import Path
assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert os.environ["ALFWORLD_DATA_ROOT"]
args = sys.argv[1:]
if args[0] == "-":
    sys.argv = args
    exec(sys.stdin.read())
    sys.exit(0)
with open(os.environ["FAKE_LEDGER"], "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(args) + "\\n")
output = Path(args[args.index("--output") + 1])
if "--preflight" in args:
    sys.exit(2 if output.exists() and "--resume" not in args else 0)
output.mkdir(parents=True, exist_ok=True)
(output / "identity.json").write_text("{}")
value = dict(status="complete", candidates=4, K=4, unique_instances=4, task_groups=2, budget={})
# Atomic per-process temporary file; both fake workers publish the same value.
temporary = output / (str(os.getpid()) + ".json")
temporary.write_text(json.dumps(value))
temporary.replace(output / "manifest.json")
''')
    fake.chmod(0o755)
    logs, output = tmp_path / "logs", tmp_path / "collections"
    env = dict(os.environ, PROJECT_ROOT=str(Path.cwd()), ENV_PYTHON=str(fake),
               COLLECTION_ROOT=str(output), LOG_ROOT=str(logs), FAKE_LEDGER=str(ledger))
    command = ["bash", "scripts/rai_collect_full_train.sh", "--workers", "2"]
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 3 and "--preflight" in rows[0]
    assert len(list(logs.glob("*shard*.log"))) == 2
    shards = rows[1:]
    assert {r[r.index("--shard-index") + 1] for r in shards} == {"0", "1"}
    for args in shards:
        assert "--resume" in args
        assert args[args.index("--shard-count") + 1] == "2"
        assert args[args.index("--max-calls") + 1] == "160000"
        assert args[args.index("--max-input-tokens") + 1] == "700000000"
        assert args[args.index("--output") + 1] == str(output / "seeds_full_train")
    refused = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert refused.returncode == 2
    assert len(ledger.read_text().splitlines()) == 4  # Only the failed preflight.
    resumed = subprocess.run([*command, "--resume"], env=env, capture_output=True, text=True, timeout=20)
    assert resumed.returncode == 0, resumed.stderr
    assert len(ledger.read_text().splitlines()) == 7
