"""CPU training and optional read-only K500 regression with the pinned tokenizer."""
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from baselines.common.conversation import conversation, sequence_rows
from baselines.common.fsdp_train import train_loop
from baselines.common.parser_mask import parser_rejection_counts
from baselines.common.shared_history import pack_shared_history
from baselines.sad.data import encode_episode as sad_encode
from baselines.smartad.data import encode_episode as smartad_encode
from sed.manifest import load_frozen
from sed.react import ActionParseError, parse_action
from sed.seed_evidence import _subset_values, checked_episodes
from sed.storage import digest, file_hash
from test_parser_rejection_mask import assert_original_labels, rejected_trajectory
from test_sad import CharTokenizer
from test_stage2b_torch import tokenizer
from test_validate_shared_history_gpu import tiny_model


@pytest.mark.parametrize("method,encoder", [("sad", sad_encode), ("smartad", smartad_encode)])
def test_masked_training_both_scoring_paths_and_run_counts(tmp_path, method, encoder):
    example = encoder(rejected_trajectory(), CharTokenizer())
    expected = parser_rejection_counts([example])
    assert expected["episodes"] == expected["replies"] == 1
    logical = sum(len(row["input_ids"]) for row in sequence_rows(example))
    states, parameters = [], []
    for scoring in ("per_turn", "shared_history"):
        model = tiny_model()
        callbacks = []
        options = ({"processed_token_budget": logical + 1,
                    "checkpoint_callback": lambda model, state: callbacks.append(state)}
                   if method == "sad" else {"epochs": 2})
        state = train_loop(model, [example], tmp_path / scoring, objective=method,
            learning_rate=2e-5, scheduler="linear", warmup_ratio=0, warmup_steps=0,
            original_batch=1, packing="turn", scoring=scoring, **options)
        assert state["parser_rejection_mask"] == expected  # input counts, not 2x exposures
        assert state["examples_seen"] == 2
        assert state["supervised_tokens_seen"] == 2 * example["supervised_tokens"]
        assert state["logical_forward_tokens"] == 2 * logical
        assert all(c["parser_rejection_mask"] == expected for c in callbacks)
        states.append(state)
        parameters.append([p.detach().clone() for p in model.parameters()])
    for before, after in zip(*parameters):
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    assert states[0]["updates"] == states[1]["updates"] == 2


def test_real_k500_sad_masks_exactly_five_and_preserves_other_labels(tokenizer):
    view = Path(os.environ.get("SED_K500_VIEW", "data/verified/seeds_full_train_r2/subsets/k500"))
    if not view.is_dir():
        pytest.skip("Set SED_K500_VIEW to the local read-only full-TRAIN K500 view")
    manifest, freeze = load_frozen("data/manifest/alfworld_manifest.json",
                                   "data/manifest/alfworld_split_freeze.json")
    root = view.resolve().parent.parent
    collection, selection, identity = [json.loads((view / (name + ".json")).read_text())
                                       for name in ("manifest", "selection", "identity")]
    parent, parent_selection = [json.loads((root / (name + ".json")).read_text())
                                for name in ("manifest", "selection")]
    # Reconstruct the bound view, then verify only its selected episode receipts.
    # The full production loader also scans every unrelated full-TRAIN attempt.
    expected_identity, expected = _subset_values(root, parent, parent_selection, selection)
    assert collection == expected and identity == expected_identity
    assert selection["manifest_sha256"] == digest(manifest)
    assert selection["split_freeze_sha256"] == digest(freeze)
    assert collection["K"] == 500
    # Hash only the evidence read by this check; no collection writes/materialization.
    paths = [view / name for name in ("manifest.json", "selection.json", "identity.json")]
    paths += [root / "manifest.json", root / "selection.json", root / "identity.json"]
    paths += [root / r["episode_dir"] / "episode.json" for r in collection["verified"]]
    before = {str(path): file_hash(path) for path in paths}
    totals = {"episodes": 0, "replies": 0, "supervised_tokens_removed": 0}
    positions, replies, unchanged = [], 0, 0
    for position, source in enumerate(checked_episodes(root, collection["verified"])):
        example = sad_encode(source, tokenizer)
        baseline = conversation(source, tokenizer, source="trusted_teacher_seed", packing="turn")
        assert example["messages"] == baseline["messages"]
        counts = parser_rejection_counts([example])
        for key in totals:
            totals[key] += counts[key]
        if counts["replies"]:
            positions.append(position)
        for row, old, step in zip(example["turns"], baseline["turns"], source["steps"]):
            replies += 1
            assert row["input_ids"] == old["input_ids"]
            try:
                parse_action(step["action_raw"])
            except ActionParseError:
                assert step["action_outcome"] == "parser_error"
                assert row["labels"] == [-100] * len(row["input_ids"])
                assert row["label_provenance"]["parser_rejected_supervised_tokens_removed"] == old["supervised_tokens"]
            else:
                assert step["action_outcome"] != "parser_error"
                assert_original_labels(row, step["action_raw"], tokenizer, method="sad")
                unchanged += 1
        packed = pack_shared_history(example)
        assert [v for v in packed["labels"] if v != -100] == [
            v for row in sequence_rows(example) for v in row["labels"] if v != -100]
        if (position + 1) % 100 == 0:
            print(f"K500: encoded {position + 1}/500 episodes", flush=True)
    assert replies == 9181 and unchanged == 9176
    assert positions == [73, 173, 317, 380, 477]  # zero-based K500 order
    assert totals["episodes"] == totals["replies"] == 5
    assert totals["supervised_tokens_removed"] > 0
    assert {str(path): file_hash(path) for path in paths} == before
    print(json.dumps({"K": 500, "replies": replies, "unchanged_replies": unchanged,
                      "positions": positions, "parser_rejection_mask": totals}, sort_keys=True))
