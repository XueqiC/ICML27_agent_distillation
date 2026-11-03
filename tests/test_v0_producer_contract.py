"""Read the eval worktree's actual V0 builder output, using only CPU fakes."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import repair
from repair.artifacts import read_json
from repair.execute import execute_actions
from sed.storage import digest, file_hash, write_json
from tools import v0_data
from test_v0_training import corpus, seed_corpus, read


@pytest.fixture
def producer(monkeypatch):
    sibling = Path(__file__).resolve().parents[2] / "ICML27_wt_repair_eval"
    if not (sibling / "repair/v0_data.py").is_file():
        pytest.skip(f"V0 eval producer is unavailable at {sibling}; cross-worktree contract needs that checkout")
    # Test-only lookup: production training has no dependency on the sibling.
    # repair is already imported by the reader, so extend its package path too.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(sibling))
    monkeypatch.setattr(repair, "__path__", [str(sibling / "repair"), *repair.__path__])
    before = set(sys.modules)
    try:
        try:
            modules = {name: importlib.import_module("repair.v0_" + name) for name in ("common", "synth", "data")}
            fakes = importlib.import_module("repair.dry_fakes")
        except (ImportError, OSError) as exc:
            pytest.skip(f"Cannot import read-only V0 eval producer at {sibling} ({type(exc).__name__})")
        for name, module in modules.items():
            assert Path(module.__file__).resolve() == sibling / "repair" / f"v0_{name}.py"
        yield SimpleNamespace(**modules, fakes=fakes)
    finally:
        # Do not leak sibling modules into subsequent local tests.
        for name in set(sys.modules) - before:
            module = sys.modules[name]
            source = getattr(module, "__file__", None)
            if source and Path(source).resolve().is_relative_to(sibling):
                del sys.modules[name]
                parent, _, child = name.rpartition(".")
                package = sys.modules.get(parent)
                if package is not None and getattr(package, child, None) is module:
                    delattr(package, child)


@pytest.fixture
def produced(corpus, producer, tmp_path):
    seeds = read_json(corpus["synthetic"])["seed_task_ids"]

    class FakeCandidates:
        count = 0

        def next_valid(self, generator, bin_name):
            self.count += 1
            ident = str(self.count)
            return dict(id=ident, task_id="train/synth/" + ident, parent_seed_task_id=seeds[0],
                        task_type="pick_and_place_simple", generator=generator, bin=bin_name,
                        status="valid", item_fingerprint=digest(["fake-item", ident]))

    # The producer itself marks probes and allocates both near generators.
    tasks, _ = producer.synth.allocate_seed(FakeCandidates(), quotas=dict(near=2, mid=1, far=1, very_far=1))
    path = tmp_path / "producer-synthetic.json"
    document = dict(schema="v0-synthetic-tasks/1", seed_task_ids=seeds, tasks=tasks)
    path.write_text(json.dumps(document, indent=2) + "\n\n")
    manifest = producer.common.SyntheticManifest(path, seeds)
    accepted, sha = v0_data.read_synthetic_manifest(path, seeds)
    assert accepted == manifest.tasks and sha == manifest.sha256 == file_hash(path)
    assert sha != digest(document)
    rows = []
    for i, task in enumerate(t for t in tasks if not t["heldout_probe"]):
        execution = execute_actions(producer.fakes.FakeEnvironment, task["task_id"], producer.fakes.N, 40)
        bucket = ("R", "S", "G")[i % 3]
        origins = ["student"] * len(execution.steps)
        if bucket != "S":
            origins[3] = "patch"
            origins[4:] = ["handback-student"] * (len(origins) - 4)
        episode = dict(task_id=task["task_id"], episode_id="fake-" + task["id"],
                       success=execution.success, execution=execution.to_json(), executors=origins,
                       prefix_length=1 if task["generator"] == "G3" else 0)
        built, _ = producer.data.decision_rows(episode, manifest, bucket=bucket,
                                              episode_ref="records/" + episode["episode_id"] + ".json")
        assert built
        rows.extend(built)
    directory = tmp_path / "producer-dataset"
    replay = corpus["root"] / "replay"
    producer.data.publish_dataset(directory, rows, manifest=manifest, replay=replay,
        inputs=producer.common.bind(corpus["config"]["paths"][k] for k in ("manifest", "split_freeze")),
        metadata=dict(code_revision="e5bf24f-contract-fakes", theta0=read_json(replay / "manifest.json")["theta0"]))
    return dict(corpus, root=directory, synthetic=path), manifest


def test_reader_accepts_eval_producer_dataset(produced, producer):
    corpus, manifest = produced
    data = read(corpus)
    _, original = producer.data.load_dataset(corpus["root"])
    assert data["rows"][:len(original)] == original
    assert data["K"] == 20 and len(data["rows"]) == len(original) + 20
    assert {r["v0"]["bucket"] for r in data["rows"]} == {"R", "S", "G", "D"}
    assert {(r["v0"]["generator"], r["v0"]["bin"]) for r in data["rows"]} == {
        ("G1", "near"), ("G3", "near"), ("G2", "mid"), ("G2", "far"), ("G5", "very_far"), ("seed", "seed")}
    assert all(not manifest.tasks[r["task_id"]]["heldout_probe"] for r in original)
    assert data["dataset_sha256"] == producer.data.consumer_dataset_hash(corpus["root"], original, seed_bin=True)
    for filename in ("manifest.json", "examples.jsonl", "examples.sha256"):
        assert (corpus["root"] / "replay" / filename).read_bytes() == (
            corpus["root"].parent / "v0/replay" / filename).read_bytes()


@pytest.mark.parametrize("mutation,match", [
    ("unknown_bin", "Invalid v0 bin"), ("unknown_generator", "Invalid v0 metadata"),
    ("heldout_probe", "Held-out synthetic task"), ("bin_mismatch", "Synthetic bin mismatch"),
    ("generator_mismatch", "Synthetic generator mismatch"),
])
def test_reader_rejects_mutated_eval_producer_rows(produced, mutation, match):
    corpus, manifest = produced
    path = corpus["root"] / "examples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if r["v0"]["generator"] == "G1")
    if mutation == "unknown_bin":
        row["v0"]["bin"] = "unknown"
    elif mutation == "unknown_generator":
        row["v0"]["generator"] = "G4"
    elif mutation == "heldout_probe":
        probe = next(t for t in manifest.tasks.values() if t["heldout_probe"] and t["generator"] == "G1")
        row.update(task_id=probe["task_id"], group_id=probe["group_id"], task_type=probe["task_type"])
        row["v0"].update(generator=probe["generator"], bin=probe["bin"])
    elif mutation == "bin_mismatch":
        row["v0"]["bin"] = "mid"
    elif mutation == "generator_mismatch":
        row["v0"]["generator"] = "G3"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    seal = deepcopy(read_json(corpus["root"] / "manifest.json"))
    seal["examples_sha256"] = file_hash(path)
    write_json(corpus["root"] / "manifest.json", seal)
    with pytest.raises(ValueError, match=match):
        read(corpus)
