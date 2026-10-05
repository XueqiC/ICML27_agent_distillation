"""Serving protocol guards operate on saved artifacts only."""
import json

import pytest

from analysis.k_ladder import batch_invariant_setting, guard_serving, load_run, main, row


@pytest.mark.parametrize("setting", [False, True, None])
def test_load_serving_setting_from_summary_or_provenance(tmp_path, setting):
    (tmp_path / "summary.json").write_text(json.dumps({"batch_invariant": setting}))
    run = load_run(tmp_path)
    assert run["batch_invariant"] is setting
    assert row("base", run, None)["batch_invariant"] is setting
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "provenance.json").write_text(json.dumps({"config": {
        "server_provenance": {"config": {"server": {"batch_invariant": setting}}}}}))
    assert load_run(tmp_path)["batch_invariant"] is setting
    assert batch_invariant_setting({}, {"config": {"suite": {"batch_invariant": setting}}}) is setting


def test_missing_and_conflicting_serving_metadata(tmp_path):
    assert load_run(tmp_path)["batch_invariant"] is None
    with pytest.raises(ValueError, match="Conflicting"):
        batch_invariant_setting({"batch_invariant": True}, {"config": {"suite": {"batch_invariant": False}}})
    for invalid in ("0", "1", 0, 1):
        with pytest.raises(ValueError, match="boolean"):
            batch_invariant_setting({"batch_invariant": invalid}, {})


@pytest.mark.parametrize("a,b", [(True, False), (True, None), (False, None)])
def test_guard_refuses_mixed_or_unknown_settings(a, b):
    rows = [{"name": "base", "batch_invariant": a}, {"name": "Seed-SFT", "batch_invariant": b}]
    with pytest.raises(ValueError, match="--allow-mixed-serving"):
        guard_serving(rows)
    assert guard_serving(rows, allow_mixed=True)
    assert not guard_serving([rows[0], {"name": "teacher", "batch_invariant": None}])
    assert not guard_serving([rows[0], dict(rows[0], name="Seed-SFT")])


def save_run(path, setting, *, teacher=False):
    episode_dir = path / "episodes" / "000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode.json").write_text(json.dumps({
        "task_id": "valid_seen/pick_and_place_simple-Apple-None-CounterTop-1/trial_0",
        "split": "dev", "success": True, "steps": []}))
    (path / "summary.json").write_text(json.dumps({
        "policy": "teacher" if teacher else "student", "batch_invariant": setting}))


def test_cli_guard_and_labelled_override(tmp_path, monkeypatch):
    import analysis.k_ladder as ladder
    plotted = []
    # Plotting is optional in the CPU test venv; still exercise the actual table CLI.
    monkeypatch.setattr(ladder, "plot_ladder", lambda rows, *args: plotted.extend(rows))
    root, teacher, out = [tmp_path / name for name in ("ladder", "teacher", "out")]
    save_run(root / "google" / "gemma-4-12B-it" / "eval", False)
    save_run(root / "k10s0-p3" / "eval", True)
    save_run(teacher, None, teacher=True)
    args = ["--ladder", str(root), "--teacher", str(teacher), "--out", str(out)]
    with pytest.raises(ValueError, match="--allow-mixed-serving"):
        main(args)
    assert not out.exists()
    main([*args, "--allow-mixed-serving"])
    rows = json.loads((out / "k_ladder.json").read_text())
    assert [r["batch_invariant"] for r in rows] == [None, False, True]
    assert rows[2]["vs_base"]["serving_noise_reference"]["discordant"] == 12
    text = (out / "k_ladder.md").read_text()
    assert "--allow-mixed-serving" in text and "batch_invariant" in text and "12/140" in text
    assert "| base | 0 |" in text and "| Seed-SFT K=10 s0 3 passes | 1 |" in text
    assert json.loads(json.dumps(plotted)) == rows
