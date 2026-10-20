"""Compare last-complete-epoch training losses from adjacent doubling rungs.

Usage: python -m tools.convergence_check PREVIOUS_OUTPUT CURRENT_OUTPUT [--output JSON]
Only training ledgers and their metadata are read; no evaluation data is opened.
"""
import argparse
import json
import math
from pathlib import Path

from sed.storage import file_hash, write_json
from train.convergence import compare_losses


def complete_epochs(rows, epochs, example_count):
    """Prove exhaustive, once-only coverage from the actual update indices."""
    coverage, examples_seen = {}, 0
    for row, epoch in zip(rows, epochs):
        indices = row.get("indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("Per-update indices required to verify complete epochs")
        if any(type(i) is not int or not 0 <= i < example_count for i in indices):
            raise ValueError("Training indices must be integers within the training example count")
        seen = coverage.setdefault(epoch, set())
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError(f"Duplicate training indices in epoch {epoch}; examples must be seen exactly once")
        seen.update(indices)
        examples_seen += len(indices)
        for key, expected in (("global_batch", len(indices)), ("examples_seen", examples_seen)):
            if key in row and (type(row[key]) is not int or row[key] != expected):
                raise ValueError(f"Ledger {key} disagrees with per-update indices")
    return {epoch for epoch, seen in coverage.items() if len(seen) == example_count}


def last_epoch(directory):
    directory = Path(directory).resolve()
    hashes = {}

    def read(name, jsonl=False):
        path = directory / name
        hashes[str(path)] = file_hash(path)
        text = path.read_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)

    rows = read("updates.jsonl", jsonl=True)
    if not rows:
        raise ValueError("Empty training ledger")
    provenance = read("provenance.json") if (directory / "provenance.json").is_file() else {}
    result = read("result.json") if (directory / "result.json").is_file() else {}
    config = provenance.get("config", {})
    baseline = "supervised_tokens_seen" in rows[0]
    settings = config.get("baseline_training" if baseline else "sft", {})
    objective = settings.get("objective", rows[0].get("objective", "sft" if not baseline else None))
    if objective not in {"sft", "smartad", "sad"}:
        raise ValueError("Need training objective sft/smartad/sad in provenance or ledger")
    losses, tokens, epochs = [], [], []
    previous_tokens = previous_passes = 0
    for update, row in enumerate(rows, 1):
        if row.get("update") != update:
            raise ValueError("Training ledger must contain contiguous unique updates starting at 1")
        if row.get("objective", objective) != objective:
            raise ValueError("Mixed objectives in training ledger")
        if baseline:
            count = row["supervised_tokens_seen"] - previous_tokens
            previous_tokens = row["supervised_tokens_seen"]
            epoch = row["epoch"]
            if "supervised_loss_sum" in row:
                numerator = row["supervised_loss_sum"]
            elif objective == "sad":
                numerator = row["loss"] * row["global_batch"]
            elif objective == "sft" and row["global_batch"] == 1:
                numerator = row["loss"] * count
            else:
                raise ValueError("Legacy SFT/SmartAD ledger lacks exact supervised_loss_sum; cannot reconstruct token loss")
        else:
            count = row["supervised_tokens"]
            numerator = row["loss"] * count
            # Legacy plans can cross pass boundaries. Do not invent a loss
            # allocation for such an update; the ladder flushes each pass.
            epoch = row.get("epoch", math.floor(previous_passes + 1e-9))
            if row["passes"] > epoch + 1 + 1e-9:
                raise ValueError("Update crosses a pass boundary; exact last-pass loss is unavailable")
            previous_passes = row["passes"]
        if type(count) is not int or count <= 0 or not math.isfinite(numerator) or numerator < 0:
            raise ValueError("Positive supervised-token counts and finite nonnegative loss sums required")
        if (type(epoch) is not int or epoch < 0 or (not epochs and epoch != 0)
                or (epochs and epoch not in {epochs[-1], epochs[-1] + 1})):
            raise ValueError("Invalid or noncontiguous epoch indices")
        losses.append(numerator)
        tokens.append(count)
        epochs.append(epoch)
    final_epoch = epochs[-1]
    if baseline:
        expected_steps = result.get("total_steps", settings.get("total_steps"))
        if expected_steps is not None and rows[-1]["update"] != expected_steps:
            raise ValueError("Training ledger has not reached its planned endpoint")
        if objective == "sad":
            amount = settings.get("processed_token_budget", result.get("processed_token_budget"))
            if amount is None:
                raise ValueError("SAD requires processed_token_budget metadata to identify its rung")
        else:
            amount = final_epoch + 1
            if settings.get("epochs", amount) != amount:
                raise ValueError("Training ledger has not reached its planned epoch")
    else:
        amount = rows[-1]["passes"]
        if not float(amount).is_integer() or amount != final_epoch + 1:
            raise ValueError("Trajectory endpoint must finish a complete pass")
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Invalid training amount")

    # Never infer the dataset size from a possibly truncated training ledger.
    example_counts = [settings["N_examples"]] if "N_examples" in settings else []
    if (directory / "example_sources.json").is_file():
        sources = read("example_sources.json")
        if not isinstance(sources, list):
            raise ValueError("example_sources.json must contain a list of training example sources")
        example_counts.append(len(sources))
    if not example_counts or any(type(n) is not int or n <= 0 for n in example_counts):
        raise ValueError("Positive training example count required in provenance or example_sources.json")
    if len(set(example_counts)) != 1:
        raise ValueError("Training example counts in provenance and example_sources.json disagree")
    example_count = example_counts[0]
    complete = complete_epochs(rows, epochs, example_count)
    if objective != "sad" and final_epoch not in complete:
        raise ValueError(f"{objective} final epoch is incomplete; every training example must be seen exactly once")
    if not complete:
        raise ValueError("No complete epoch exists in the training ledger")
    if any(epoch not in complete for epoch in range(final_epoch)):
        raise ValueError("Incomplete nonfinal epoch in training ledger")
    # SAD stops on a token budget. Its ascending-complexity curriculum makes
    # the partial final epoch unrepresentative; use the preceding full epoch.
    selected_epoch = max(complete)
    selected = [i for i, epoch in enumerate(epochs) if epoch == selected_epoch]
    mean = math.fsum(losses[i] for i in selected) / sum(tokens[i] for i in selected)
    return {"L_last": mean, "objective": objective, "format": "fsdp_train" if baseline else "trajectory_sft",
            "amount": amount, "final_epoch": final_epoch, "final_epoch_complete": final_epoch in complete,
            "selected_epoch": selected_epoch, "training_examples": example_count, "updates": len(selected),
            "supervised_tokens": sum(tokens[i] for i in selected), "input_hashes": hashes,
            "settings": settings}


def check(previous, current):
    prev, curr = last_epoch(previous), last_epoch(current)
    if (prev["objective"], prev["format"]) != (curr["objective"], curr["format"]):
        raise ValueError("Rungs must use the same training format and objective")
    if curr["amount"] != 2 * prev["amount"]:
        raise ValueError("Current training amount must be double the previous rung")
    for key in ("examples_sha256", "seed", "snapshot", "learning_rate", "scheduler", "packing", "scoring"):
        if key in prev["settings"] and key in curr["settings"] and prev["settings"][key] != curr["settings"][key]:
            raise ValueError(f"Rung training settings differ: {key}")
    return {**compare_losses(prev["L_last"], curr["L_last"]),
            "input_hashes": {"previous": prev["input_hashes"], "current": curr["input_hashes"]},
            "previous": {k: v for k, v in prev.items() if k not in {"settings", "input_hashes"}},
            "current": {k: v for k, v in curr.items() if k not in {"settings", "input_hashes"}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--output", type=Path, help="Default: CURRENT_OUTPUT/convergence.json")
    args = parser.parse_args(argv)
    try:
        result = check(args.previous, args.current)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    destination = args.output or args.current / "convergence.json"
    inputs = {path for group in result["input_hashes"].values() for path in group}
    if str(destination.resolve()) in inputs:
        parser.error("Output must not overwrite a training input")
    write_json(destination, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
