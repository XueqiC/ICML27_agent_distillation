"""Trajectory batching and recipe wiring without torch, GPU, or network."""
from types import SimpleNamespace

import pytest

from baselines.common.fsdp_train import main, sft_microsteps
from baselines.common.schedule import dose, epoch_batches, rank_slots
from baselines.agentgym import pipeline as agentgym
from baselines.eto import pipeline as eto
from baselines.ipr import pipeline as ipr
from baselines.kang import pipeline as kang


@pytest.mark.parametrize("n", [1, 5, 20, 25])
@pytest.mark.parametrize("world", [1, 4])
def test_turn_expansion_keeps_x5_trajectory_sets_and_partial_batches(n, world):
    episodes, turns = [], []
    for i in range(n):
        labels = [[i + 1] * (j + 1) for j in range(i % 4 + 1)]
        episodes.append({"labels": [-100] + sum(labels, [])})
        turns.append({"packing": "turn", "turns": [
            {"labels": [-100] + target, "trajectory_index": i} for target in labels]})
    plan = dose(n, 8, epochs=2)
    batches = list(epoch_batches(n, plan["b_prime"], epochs=2, seed=7))
    assert len(batches) == plan["total_steps"]
    for batch in batches:
        episode_steps = sft_microsteps(episodes, batch["indices"])
        turn_steps = sft_microsteps(turns, batch["indices"])
        assert len(episode_steps) == len(batch["indices"])
        assert list(dict.fromkeys(row["trajectory_index"] for row, _ in turn_steps)) == batch["indices"]
        for i in batch["indices"]:
            assert sum(w for r, w in turn_steps if r["trajectory_index"] == i) == pytest.approx(1.)
        slots = rank_slots(list(range(len(turn_steps))), world)
        assert [i for slot in slots for i in slot if i is not None] == list(range(len(turn_steps)))
    assert len(batches[-1]["indices"]) == plan["last_batch"]


def test_all_trajectory_sft_recipes_record_and_pass_turn_packing():
    args = SimpleNamespace(train_python="TRAIN", config="CONFIG", snapshot="BASE", verified="SEEDS", gpus=4)
    commands = [kang.training_command(args, "OUT"),
        agentgym.training_command(args, data="DATA", output="OUT", n_examples=20),
        eto.training_command(args, objective="sft", snapshot="BASE", output="OUT"),
        ipr.training_command(args, objective="sft", snapshot="BASE", output="OUT")]
    assert kang.RECIPE["packing"] == agentgym.RECIPE["packing"] == "turn"
    assert eto.RECIPE["sft"]["packing"] == ipr.RECIPE["sft"]["packing"] == "turn"
    for command in commands:
        assert command[command.index("--packing") + 1] == "turn"


@pytest.mark.parametrize("packing", [None, "episode", "turn"])
def test_cli_passes_packing_to_kang_without_loading_a_model(monkeypatch, packing):
    received = []
    monkeypatch.setattr("baselines.common.fsdp_train.train_lora", lambda args: received.append(args.packing))
    argv = ["--objective", "sft", "--adaptation", "kang_lora", "--snapshot", "BASE",
        "--base-snapshot", "BASE", "--verified", "SEEDS", "--output", "OUT",
        "--learning-rate", "2e-4", "--scheduler", "linear", "--warmup-ratio", "0", "--original-batch", "8"]
    main(argv + (["--packing", packing] if packing else []))
    assert received == [packing or "turn"]
