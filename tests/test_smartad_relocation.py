"""Offline relocation keeps published bindings and self-contained evidence."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from baselines.smartad import collect as sm, pipeline
from baselines.smartad.paths import PathResolver
from baselines.smartad.select import select, validate_selection
from conftest import FakeBackend
from harness.alfworld_env import Environment
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.storage import write_json
from test_smartad import factory, setup_source
from test_stage2b_seeds import seed_bank


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("mode", ["reuse", "new", "fallback", "full_failed"])
def test_relocated_cli_selection_is_identical(seed_bank, data_root, tmp_path, monkeypatch, mode):
    a, b = tmp_path / "A", tmp_path / "B"
    if mode == "full_failed":
        from sed.config import load_config
        from sed.manifest import build_manifest, freeze_splits
        manifest = build_manifest(data_root)
        seed_bank = data_root, manifest, freeze_splits(manifest), load_config()
    source, manifest, freeze, config, env = setup_source(
        seed_bank, a, k=2, default=mode in {"new", "fallback"},
        full=mode == "full_failed", failed_source=mode == "full_failed")
    if mode == "fallback":
        env = lambda _: Environment(config["environment"]["data_root"],
                                    max_episode_steps=1, backend_factory=FakeBackend)
    calls = []
    pool_root = sm.collect(source, manifest, freeze, config, a / "pool",
        rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=factory(calls),
        subset_ks=[] if mode == "full_failed" else [1])
    original_pool = read(pool_root / "candidates.json")
    pool_hashes = artifact_hashes(pool_root)
    call_count = len(calls)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write_json(snapshot / "config.json", {"model_type": "gemma4_unified"})
    monkeypatch.setattr(pipeline, "load_config", lambda _: config)
    monkeypatch.setattr(pipeline, "load_frozen", lambda *_: (manifest, freeze))
    scored = []

    def stub_select(*args, **kwargs):
        def scorer(episode):
            scored.append(episode["episode_id"])
            # Exercise stable tie breaking, including a winning reused seed.
            rank = (len(scored) - 1) % 10 + 1
            nll = .1 if rank in ((1, 2) if mode == "reuse" else (2, 3)) else 2.0
            return {"nll": nll, "turns": [{"nll_mean": nll}]}
        return select(*args, **kwargs, scorer=scorer)

    monkeypatch.setattr(pipeline, "select", stub_select)

    def run(pool, output, entries=()):
        return pipeline.main(["--through", "select", "--snapshot", str(snapshot),
            "--verified", str(pool), "--output", str(output),
            *[arg for entry in entries for arg in ("--path-map", entry)]])

    first = run(pool_root, tmp_path / "first")
    original_scored = list(scored)
    scored.clear()
    if mode == "new":
        # Pool-only relocation also covers unmapped trusted evidence.
        b.mkdir()
        pool_root.rename(b / "pool")
        entries = [f"{pool_root}={b / 'pool'}"]
    else:
        a.rename(b)
        # Exercise both a project-prefix map and repeatable disjoint maps.
        entries = ([f"{a / name}={b / name}" for name in ("pool", "source")]
                   if mode == "reuse" else [f"{a}={b}"])
    with pytest.raises(ValueError, match="does not exist"):
        run(b / "pool", tmp_path / "missing-map")
    assert not scored
    second = run(b / "pool", tmp_path / "second", entries)
    assert scored == original_scored
    assert len(calls) == call_count
    assert read(b / "pool/candidates.json") == original_pool
    assert artifact_hashes(b / "pool") == pool_hashes
    assert artifact_hashes(first.parent / "scores") == artifact_hashes(second.parent / "scores")
    assert (first / "smartad_selection.json").read_bytes() == (second / "smartad_selection.json").read_bytes()
    # All selected content (including receipts, bindings and nested views) is
    # byte-identical except run provenance, which discloses the relocation.
    first_hashes, second_hashes = artifact_hashes(first), artifact_hashes(second)
    first_hashes.pop("provenance.json")
    second_hashes.pop("provenance.json")
    assert first_hashes == second_hashes
    disclosure = PathResolver(entries).disclosure
    assert read(second / "provenance.json")["details"]["path_map"] == disclosure
    assert read(second.parent / "result.json")["path_map"] == disclosure
    assert read(first.parent / "result.json")["path_map"] == []
    decisions = read(second / "smartad_selection.json")["tasks"]
    chosen = "trusted-fallback" if mode == "fallback" else "candidate-01" if mode == "reuse" else "candidate-02"
    assert all(task["chosen_id"] == chosen for task in decisions)
    run(b / "pool", tmp_path / "second", entries)
    assert scored == original_scored  # resume uses identical cached bindings
    # Downstream verification needs only the selected collection and snapshot.
    shutil.rmtree(b)
    if a.exists():
        shutil.rmtree(a)
    relocated = tmp_path / "selected-on-third-host"
    second.rename(relocated)
    assert validate_selection(relocated, manifest, freeze, snapshot)["tasks"] == decisions
    collection, _, _ = load_verified_collection(relocated, manifest, freeze)
    assert all(not Path(row["episode_dir"]).is_absolute() for row in collection["attempts"])
    if mode != "full_failed":
        validate_selection(relocated / "subsets/k1", manifest, freeze, snapshot)


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("target", ["episode", "teacher", "reused_episode", "reused_teacher"])
def test_selection_rejects_tampered_evidence_before_scoring(seed_bank, tmp_path, mapped, target):
    a, b = tmp_path / "A", tmp_path / "B"
    source, manifest, freeze, config, env = setup_source(seed_bank, a)
    pool_root = sm.collect(source, manifest, freeze, config, a / "pool",
        rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=factory([]))
    pool = read(pool_root / "candidates.json")
    candidate = pool["tasks"][0]["candidates"][0 if target.startswith("reused") else 1]
    directory = Path(candidate["episode_dir"])
    if target.endswith("teacher"):
        directory = directory.parent / "teacher"
    if mapped:
        a.rename(b)
        pool_root = b / "pool"
        directory = b / directory.relative_to(a)
    # Alter one actual episode file, or one teacher artifact.
    path = directory / "episode.json" if target.endswith("episode") else next(directory.rglob("usage.jsonl"))
    path.write_text(path.read_text() + "\n")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="evidence changed"):
        select(pool_root, snapshot, config, manifest, freeze, tmp_path / "selected",
               path_map=[f"{a}={b}"] if mapped else [],
               scorer=lambda _: pytest.fail("Tampered evidence must fail before scoring"))
    assert not (tmp_path / "selected").exists()


@pytest.mark.parametrize("kind", ["episode", "teacher"])
def test_copy_rechecks_evidence_after_scoring(seed_bank, tmp_path, kind):
    source, manifest, freeze, config, env = setup_source(seed_bank, tmp_path)
    pool_root = sm.collect(source, manifest, freeze, config, tmp_path / "pool",
        rate_limit_dir=tmp_path / "limiter", env_factory=env, teacher_factory=factory([]))
    row = read(pool_root / "candidates.json")["tasks"][0]["candidates"][1]
    directory = Path(row["episode_dir"])
    if kind == "teacher":
        directory = directory.parent / "teacher"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    scored = []

    def scorer(episode):
        scored.append(episode)
        if len(scored) == 10:
            write_json(directory / "tampered.json", {})
        return {"nll": .1 if len(scored) == 2 else 2.0}

    with pytest.raises(ValueError, match="evidence changed"):
        select(pool_root, snapshot, config, manifest, freeze, tmp_path / "selected", scorer=scorer)
    assert len(scored) == 10
    assert not (tmp_path / "selected/result.json").exists()


@pytest.mark.parametrize("entries", [
    ["relative=/new"], ["/old=relative"], ["/old"], ["=/new"], ["/old="],
    ["/old=/new", "/old=/elsewhere"], ["/old=/new", "/old=/new"],
    ["/old=/new", "/old/child=/elsewhere"], ["/old/child=/elsewhere", "/old=/new"],
    ["/old=/new", "/other=/new"], ["/old=/new", "/other=/new/child"],
    ["/old=/new", "/new=/other"], ["/old=/old/child"],
    ["/old=/old"], ["/old/../old=/new", "/old=/other"],
])
def test_cli_rejects_invalid_maps_before_loading(entries, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "load_config", lambda _: pytest.fail("Invalid map reached config loading"))
    with pytest.raises(SystemExit) as error:
        pipeline.main(["--through", "select", "--verified", str(tmp_path / "pool"),
            "--output", str(tmp_path / "out"), "--snapshot", str(tmp_path / "snapshot"),
            *[arg for entry in entries for arg in ("--path-map", entry)]])
    assert error.value.code == 2
    assert "--path-map" in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["collect", "train", "evaluate"])
def test_path_map_only_applies_to_selection(stage, capsys):
    with pytest.raises(SystemExit) as error:
        pipeline.main(["--through", stage, "--verified", "/pool", "--output", "/out",
                       "--snapshot", "/snapshot", "--path-map", "/old=/new"])
    assert error.value.code == 2
    assert "--path-map is for selection only" in capsys.readouterr().err


def test_component_prefix_and_missing_destination(tmp_path):
    old, new, lookalike = (tmp_path / name for name in ("pool", "moved", "pool-other"))
    old.mkdir()
    lookalike.mkdir()
    resolver = PathResolver([f"{old}={new}"])
    assert resolver.resolve(lookalike) == lookalike
    with pytest.raises(ValueError, match="does not exist"):
        resolver.resolve(old)  # never fall back to the old path
    (new / "child").mkdir(parents=True)
    assert resolver.resolve(old / "child") == new / "child"
    assert resolver.published(new / "child") == old / "child"


def test_launcher_map_arguments_are_literal(tmp_path):
    launcher = Path("scripts/hpg_smartad.slurm").read_text()
    # Execute only local argument construction, never the SLURM/HPG launcher.
    snippet = "path_args=()\n" + launcher.split("path_args=()\n", 1)[1].split(
        'if [[ "$SMARTAD_THROUGH" == collect ]]; then', 1)[0]
    command = ["bash", "-c", "set -euo pipefail\n" + snippet + '\nprintf "%s\\0" "${path_args[@]}"']
    entries = ["/old=/new$(>executed)", "/source=/moved/*", "/third=/fourth`>executed`"]
    env = {"SMARTAD_THROUGH": "select", "SMARTAD_PATH_MAP": "\t" + " \n".join(entries)}
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, check=True)
    assert result.stdout.decode().split("\0")[:-1] == [arg for entry in entries for arg in ("--path-map", entry)]
    assert not (tmp_path / "executed").exists()
    env["SMARTAD_THROUGH"] = "train"
    refused = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True)
    assert refused.returncode == 2
