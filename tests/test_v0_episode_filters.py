"""Training-only episode ablations on sealed data; all tests use offline CPU fakes."""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from types import ModuleType, SimpleNamespace

import pytest

from repair import v2_train as v2
from repair.artifacts import read_json
from sed.storage import digest, file_hash, write_json
from tools import train_v0_arm as trainer, v0_data
from test_v0_training import corpus, seed_corpus, read, seal


def arguments(c):
    return ["--dataset", str(c["root"]), "--theta0", str(c["theta"]),
            "--snapshot", str(c["snapshot"]), "--config", str(c["config_path"]),
            "--synthetic-manifest", str(c["synthetic"])]


@pytest.fixture
def filter_corpus(corpus):
    c = corpus
    base = c["rows"][0]
    synthetic = read_json(c["synthetic"])
    task = synthetic["tasks"][0]
    synthetic["tasks"], c["rows"] = [], []
    for bucket, count in zip("RGS", (5, 7, 9)):
        for i in range(count):
            generator = v0_data.GENERATORS[i % 4]
            distance = {"G1": "near", "G2": "mid", "G3": "near", "G5": "very_far"}[generator]
            eid = f"{bucket}-{i}"
            item = dict(task, task_id=f"train/synth/{eid}", group_id=f"synthetic/{eid}",
                        item_fingerprint=digest(eid), generator=generator, bin=distance)
            synthetic["tasks"].append(item)
            for step in range(1 + i % 3):
                row = deepcopy(base)
                row.update(example_id=f"{eid}-{step}", task_id=item["task_id"], group_id=item["group_id"])
                row["v0"].update(bucket=bucket, episode_id=eid, generator=generator, bin=distance)
                row["evidence"].update(step=step, episode_ref=f"records/{eid}.json")
                if bucket == "S":
                    row["v0"]["edit_decision"] = False
                    row["evidence"]["executor"] = "student"
                c["rows"].append(row)
    write_json(c["synthetic"], synthetic)
    for row in c["rows"]:
        row["v0"]["synthetic_manifest_sha256"] = file_hash(c["synthetic"])
    seal(c)
    return c


def assert_record(record, rows):
    grouped = v0_data.episodes(rows)
    kept = {eid for ids in record["kept_episode_ids"].values() for eid in ids}
    kept_rows = [r for r in rows if r["v0"]["episode_id"] in kept]
    for field, values in (("bucket", v0_data.BUCKETS), ("generator", v0_data.GENERATORS)):
        for value in values:
            subset = [r for r in kept_rows if r["v0"][field] == value]
            assert record[field + "s"][value] == {
                "rows": len(subset), "episodes": len({r["v0"]["episode_id"] for r in subset})}
    for bucket in v0_data.BUCKETS:
        ids = {eid for eid, indices in grouped.items() if rows[indices[0]]["v0"]["bucket"] == bucket}
        assert set(record["kept_episode_ids"][bucket]) | set(record["dropped_episode_ids"][bucket]) == ids
        assert not set(record["kept_episode_ids"][bucket]) & set(record["dropped_episode_ids"][bucket])
    assert record["dropped_episode_ids"]["D"] == []
    assert record["kept_rows"] == len(kept_rows)
    assert record["updates_for_4_passes"] == math.ceil(4 * len(kept_rows) / 16)


@pytest.mark.parametrize("excluded", [["G1"], ["G2"], ["G3"], ["G5"], ["G1", "G3"]])
def test_exclusion_whole_episodes_and_original_indices(filter_corpus, excluded):
    rows = read(filter_corpus)["rows"]
    original = deepcopy(rows)
    record = trainer.episode_filter(rows, SimpleNamespace(exclude_generators=excluded))
    assert_record(record, rows)
    expected = {i for i, r in enumerate(rows) if r["v0"]["bucket"] == "D"
                or r["v0"]["generator"] not in excluded}
    kept = {eid for ids in record["kept_episode_ids"].values() for eid in ids}
    assert {i for i, r in enumerate(rows) if r["v0"]["episode_id"] in kept} == expected
    batches, _ = trainer.batch_schedule(rows, updates=1000, seed=8, filter_record=record)
    assert {i for batch in batches for i in batch["indices"]} == expected
    for batch in batches:
        for item in batch["selections"]:
            assert rows[item["index"]]["v0"]["episode_id"] == item["episode_id"]
    assert rows == original


@pytest.mark.parametrize("fraction", [.01, .28, .5, .99])
def test_fraction_seed_counts_rounding_and_d_preserved(filter_corpus, fraction):
    rows = read(filter_corpus)["rows"]
    args = SimpleNamespace(episode_fraction=fraction, subsample_seed=0)
    record = trainer.episode_filter(rows, args)
    assert_record(record, rows)
    assert record == trainer.episode_filter(list(reversed(rows)), args)
    assert record == trainer.episode_filter(rows, args)
    if fraction < .99:
        other = trainer.episode_filter(rows, SimpleNamespace(episode_fraction=fraction, subsample_seed=1))
        assert record["kept_episode_ids"] != other["kept_episode_ids"]
    rng = random.Random(0)
    for bucket, count in zip("RGS", (5, 7, 9)):
        ids = sorted({r["v0"]["episode_id"] for r in rows if r["v0"]["bucket"] == bucket})
        rng.shuffle(ids)
        assert record["kept_episode_ids"][bucket] == sorted(ids[:max(1, round(fraction * count))])


def test_episode_generator_consistency_and_d_never_excluded():
    rows = [{"example_id": str(i), "v0": {"bucket": "D", "episode_id": "seed", "generator": "G1"}}
            for i in range(2)]
    args = SimpleNamespace(exclude_generators=["G1"])
    assert trainer.episode_filter(rows, args)["kept_episode_ids"]["D"] == ["seed"]
    rows[1]["v0"]["generator"] = "G2"
    with pytest.raises(ValueError, match="crosses buckets or generators"):
        trainer.episode_filter(rows, args)


@pytest.mark.parametrize("command", ["inputs", "run"])
@pytest.mark.parametrize("flags,match,parser_error", [
    (["--exclude-generators", "G1", "--episode-fraction", ".28", "--subsample-seed", "0"], "not allowed", True),
    (["--exclude-generators", "G4"], "invalid choice", True),
    *[(["--episode-fraction", f, "--subsample-seed", "0"], "strictly between", False)
      for f in ("0", "1", "-0.1", "1.1", "nan", "inf")],
    (["--episode-fraction", ".28"], "requires --subsample-seed", False),
    (["--subsample-seed", "0"], "requires --episode-fraction", False),
    *[(filter_flags + [choice, "unused.json"], "uniform weights and all tensors", False)
      for filter_flags in (["--exclude-generators", "G1"], ["--episode-fraction", ".28", "--subsample-seed", "0"])
      for choice in ("--weights", "--trainable-modules")],
])
def test_flag_refusals(command, flags, match, parser_error, tmp_path, capsys):
    args = [command, "--dataset", "unused", "--theta0", "unused", *flags]
    args += ["--check"] if command == "inputs" else [
        "--output", str(tmp_path / "unused"), "--updates", "1", "--checkpoints", "1", "--dry-run"]
    if parser_error:
        with pytest.raises(SystemExit) as exc:
            trainer.main(args)
        assert exc.value.code == 2
    else:
        assert trainer.main(args) == 1
    assert match in capsys.readouterr().err


@pytest.mark.parametrize("command", ["inputs", "run"])
def test_empty_positive_bucket_refused_but_zero_mass_allowed(corpus, tmp_path, capsys, command):
    args = arguments(corpus) + ["--exclude-generators", "G1"]
    extra = ["--check"] if command == "inputs" else [
        "--output", str(tmp_path / "trained"), "--updates", "1", "--checkpoints", "1", "--dry-run"]
    assert trainer.main([command, *args, *extra]) == 1
    assert "bucket R empty with positive mixture probability" in capsys.readouterr().err
    rows = read(corpus)["rows"]
    record = trainer.episode_filter(rows, SimpleNamespace(exclude_generators=["G1"]))
    batches, probabilities = trainer.batch_schedule(rows, updates=2, seed=0, filter_record=record,
        mixture={"R": 0, "G": 0, "S": 0, "D": 1})
    assert probabilities == {"D": 1}
    assert all(rows[i]["v0"]["bucket"] == "D" for b in batches for i in b["indices"])


@pytest.mark.parametrize("flags", [["--exclude-generators", "G1"],
                                  ["--episode-fraction", ".28", "--subsample-seed", "0"]])
def test_check_and_run_filter_evidence_and_unchanged_dataset(filter_corpus, tmp_path, capsys, monkeypatch, flags):
    import torch
    monkeypatch.setattr(v2, "load_existing_adapter", lambda *a, **kw: pytest.fail("Real model loading"))
    torch.set_num_threads(1)
    c = filter_corpus
    originals = {p: p.read_bytes() for p in c["root"].rglob("*") if p.is_file()}
    rows = read(c)["rows"]
    capsys.readouterr()
    assert trainer.main(["inputs", "--check", *arguments(c), *flags]) == 0
    check = json.loads(capsys.readouterr().out)
    output = tmp_path / "trained"
    assert trainer.main(["run", *arguments(c), *flags, "--output", str(output), "--updates", "40",
                         "--checkpoints", "40", "--dry-run", "--dry-run-tokenizer", "bytes"]) == 0
    manifest = read_json(output / "manifest.json")
    record = manifest["filter"]
    assert_record(record, rows)
    assert check["filter"] == {k: v for k, v in record.items()
                               if k not in ("kept_episode_ids", "dropped_episode_ids")}
    prov = read_json(output / "provenance.json")
    assert prov["details"]["filter"] == prov["config"]["v0_training"]["filter"] == record
    assert manifest["dataset_sha256"] == check["dataset_sha256"] == read(c)["dataset_sha256"]
    sources = read_json(output / "example_sources.json")
    assert [s["example_id"] for s in sources] == [r["example_id"] for r in rows]
    kept = {eid for ids in record["kept_episode_ids"].values() for eid in ids}
    for batch in read_json(output / "batches.json"):
        for item in batch["selections"]:
            assert sources[item["index"]]["v0"]["episode_id"] == item["episode_id"]
            assert item["episode_id"] in kept
    assert {p: p.read_bytes() for p in c["root"].rglob("*") if p.is_file()} == originals


def test_excluded_rows_still_verified(filter_corpus, capsys):
    c = filter_corpus
    c["rows"][0]["evidence"]["episode_success"] = False  # G1, excluded from sampling only.
    seal(c)
    assert trainer.main(["inputs", "--check", *arguments(c), "--exclude-generators", "G1"]) == 1
    assert "successful episodes" in capsys.readouterr().err


@pytest.mark.parametrize("keep,drop", [(True, False), (False, True), (True, True)])
def test_episode_selection_evidence_and_sampling(filter_corpus, tmp_path, capsys, monkeypatch, keep, drop):
    import torch
    torch.set_num_threads(1)
    monkeypatch.setattr(v2, "load_existing_adapter", lambda *a, **kw: pytest.fail("Real model loading"))
    c = filter_corpus
    rows = read(c)["rows"]
    originals = {p: p.read_bytes() for p in c["root"].rglob("*") if p.is_file()}
    selection = tmp_path / "keep.json"
    ids = ["R-1", "R-3"]
    write_json(selection, {"bucket": "R", "episode_ids": ids})
    flags = (["--keep-episodes", str(selection)] if keep else []) + (["--drop-buckets", "G", "S"] if drop else [])
    capsys.readouterr()
    assert trainer.main(["inputs", "--check", *arguments(c), *flags]) == 0
    check = json.loads(capsys.readouterr().out)
    output = tmp_path / "trained"
    assert trainer.main(["run", *arguments(c), *flags, "--output", str(output), "--updates", "8",
                         "--checkpoints", "8", "--dry-run", "--dry-run-tokenizer", "bytes"]) == 0
    manifest = read_json(output / "manifest.json")
    record = manifest["filter"]
    assert_record(record, rows)
    prov = read_json(output / "provenance.json")
    assert record == check["filter"] == prov["details"]["filter"] == prov["config"]["v0_training"]["filter"]
    assert record["keep_episodes"] == (str(selection) if keep else None)
    assert record["keep_episodes_sha256"] == (file_hash(selection) if keep else None)
    assert record["keep_bucket"] == ("R" if keep else None)
    assert record["drop_buckets"] == (["G", "S"] if drop else None)
    if keep:
        assert record["kept_episode_ids"]["R"] == ids
        assert prov["inputs"][str(selection)] == file_hash(selection)
    probabilities = {"R": 5 / 6, "D": 1 / 6} if drop else trainer.DEFAULT_MIXTURE
    assert record["effective_bucket_probabilities"] == pytest.approx(probabilities)
    assert manifest["effective_bucket_probabilities"] == record["effective_bucket_probabilities"]
    assert manifest["selected_real_tensors"] is None
    assert manifest["actual_trainable_parameter_names"] == ["lora_A", "lora_B"]
    assert manifest["episode_weights_sha256"] == digest({})
    expected = {i for i, r in enumerate(rows) if (not drop or r["v0"]["bucket"] not in ("G", "S"))
                and (not keep or r["v0"]["bucket"] != "R" or r["v0"]["episode_id"] in ids)}
    # A larger schedule checks every surviving original index and uniform episode weights.
    batches, mixture = trainer.batch_schedule(rows, updates=1000, seed=8, filter_record=record)
    assert mixture == pytest.approx(probabilities)
    assert {i for batch in batches for i in batch["indices"]} == expected
    for batch in [*batches, *read_json(output / "batches.json")]:
        for item in batch["selections"]:
            assert item["index"] in expected
            assert rows[item["index"]]["v0"]["episode_id"] == item["episode_id"]
    if keep:
        exposures = trainer.exposure_counts(batches)[1000]["episode_exposures"]
        assert exposures[ids[0]] / sum(exposures[eid] for eid in ids) == pytest.approx(.5, abs=.02)
    sources = read_json(output / "example_sources.json")
    assert [s["example_id"] for s in sources] == [r["example_id"] for r in rows]
    assert {p: p.read_bytes() for p in c["root"].rglob("*") if p.is_file()} == originals


@pytest.mark.parametrize("command", ["inputs", "run"])
@pytest.mark.parametrize("selection,dropped,match", [
    ({"bucket": "R", "episode_ids": ["missing"]}, [], "unknown id"),
    ({"bucket": "R", "episode_ids": ["R-0", "R-0"]}, [], "sorted unique string"),
    ({"bucket": "R", "episode_ids": ["R-1", "R-0"]}, [], "sorted unique string"),
    ({"bucket": "R", "episode_ids": [1]}, [], "sorted unique string"),
    ({"bucket": "R", "episode_ids": ["G-0"]}, [], "bucket mismatch"),
    ({"bucket": "X", "episode_ids": ["R-0"]}, [], "bucket/schema"),
    (None, ["R", "G", "S", "D"], "positive mass"),
    ({"bucket": "R", "episode_ids": ["R-0"]}, ["R"], "Cannot drop the bucket"),
])
def test_selection_refusals(filter_corpus, tmp_path, capsys, command, selection, dropped, match):
    flags = []
    if selection is not None:
        path = tmp_path / "keep.json"
        write_json(path, selection)
        flags += ["--keep-episodes", str(path)]
    if dropped:
        flags += ["--drop-buckets", *dropped]
    extra = ["--check"] if command == "inputs" else [
        "--output", str(tmp_path / "trained"), "--updates", "1", "--checkpoints", "1", "--dry-run"]
    assert trainer.main([command, *arguments(filter_corpus), *flags, *extra]) == 1
    assert match in capsys.readouterr().err
    assert not (tmp_path / "trained").exists()


@pytest.mark.parametrize("command", ["inputs", "run"])
@pytest.mark.parametrize("filter_flags", [["--keep-episodes", "unused.json"], ["--drop-buckets", "G", "S"]])
@pytest.mark.parametrize("other,match", [
    (["--weights", "unused.json"], "uniform weights and all tensors"),
    (["--trainable-modules", "unused.json"], "uniform weights and all tensors"),
    (["--exclude-generators", "G1"], "mutually exclusive"),
    (["--episode-fraction", ".5", "--subsample-seed", "0"], "mutually exclusive"),
])
def test_selection_incompatible_flags(command, filter_flags, other, match, tmp_path, capsys):
    test_flag_refusals(command, filter_flags + other, match, False, tmp_path, capsys)


def test_drop_all_positive_custom_mixture_and_absent_buckets(corpus, tmp_path, capsys):
    assert trainer.main(["run", *arguments(corpus), "--drop-buckets", "R",
                         "--bucket-probabilities", "1", "0", "0", "0", "--output", str(tmp_path / "trained"),
                         "--updates", "1", "--checkpoints", "1", "--dry-run"]) == 1
    assert "positive mass" in capsys.readouterr().err
    # Naturally absent G/S buckets are allowed for the new selection filters.
    path = tmp_path / "keep.json"
    write_json(path, {"bucket": "R", "episode_ids": ["repair-a"]})
    assert trainer.main(["inputs", "--check", *arguments(corpus), "--keep-episodes", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["filter"]["effective_bucket_probabilities"] == pytest.approx(
        {"R": 5 / 6, "D": 1 / 6})


@pytest.mark.parametrize("revision", ["085b46b", "41b9138"])
@pytest.mark.parametrize("choices", [False, True])
def test_no_flags_byte_identical_to_revision(corpus, tmp_path, monkeypatch, choices, revision):
    """Compare every output byte at one path with clocks/UUID/cache state fixed.

    Both implementations hash the same on-disk code, holding code identity fixed
    while executing the original trainer directly from the named revision.
    """
    import torch
    from baselines.common import fsdp_train
    from sed import provenance, hash_receipts
    torch.set_num_threads(1)
    monkeypatch.setenv("SED_HASH_CACHE", "0")
    monkeypatch.setattr(provenance, "utc_now", lambda: "2026-11-05T00:00:00+00:00")
    monkeypatch.setattr(provenance.uuid, "uuid4", lambda: "fixed-test-run-id")
    monkeypatch.setattr(fsdp_train, "time", SimpleNamespace(monotonic=lambda: 100.0))
    source = subprocess.check_output(["git", "show", f"{revision}:tools/train_v0_arm.py"],
                                     cwd=Path(trainer.__file__).resolve().parents[1], text=True)
    baseline = ModuleType(f"tools.train_v0_arm_{revision}")
    baseline.__file__ = trainer.__file__
    baseline.__package__ = "tools"
    exec(compile(source, trainer.__file__, "exec"), baseline.__dict__)
    args = arguments(corpus)
    if choices:
        data = read(corpus)
        inherited = v2.continuation_recipe(corpus["theta"], corpus["config"], corpus["snapshot"], 2)
        metadata = {"dataset_sha256": data["dataset_sha256"], "adapter_sha256": data["adapter_sha256"],
                    "parameter_names_sha256": digest(sorted(inherited["parameters"]))}
        weights, modules = tmp_path / "weights.json", tmp_path / "modules.json"
        write_json(weights, {"schema": "v0-episode-weights/1", **metadata,
                             "weights": {eid: 1 for eid in data["episodes"]}})
        write_json(modules, {"schema": "v0-trainable-modules/1", **metadata,
                             "selected": [n for n in inherited["parameters"] if ".lora_B." in n][:1]})
        args += ["--weights", str(weights), "--trainable-modules", str(modules)]
    output = tmp_path / "trained"
    command = ["run", *args, "--output", str(output), "--updates", "2", "--checkpoints", "1", "2",
               "--dry-run", "--dry-run-tokenizer", "bytes"]
    snapshots = []
    for implementation in (baseline, trainer):
        hash_receipts._MODES.clear()
        assert implementation.main(command) == 0
        snapshots.append({str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()})
        shutil.rmtree(output)
    assert snapshots[0].keys() == snapshots[1].keys()
    for name, content in snapshots[0].items():
        assert content == snapshots[1][name], name
