"""Read-only, offline comparison of V2 full-history and selected-turn encoding."""

import argparse
import json
from pathlib import Path
import random
import time

from baselines.common.conversation import encode_messages, group_turns
from repair.v2_train import encode_example
from sed.storage import canonical_bytes
from train.dataset import load_tokenizer


def encode_example_old(row, tokenizer, *, max_length=32768):
    """Frozen cc909f7 selection logic, using the encoder's turns=None path."""
    encoded = encode_messages(row["messages"], tokenizer, task_id=row["task_id"],
                              max_length=max_length, packing="turn", turns=None)
    turns = [encoded["turns"][(i - 2) // 2] for i in row["supervised_assistant_indices"]]
    result = group_turns(turns)
    result["label_provenance"] = {"example_id": row["example_id"], "task_id": row["task_id"],
        "group_id": row["group_id"], "source": row["source"], "kind": row["kind"], "training_eligible": True}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Compare a seeded random sample of at most N rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    with (args.dataset / "examples.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if args.limit is not None and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    tokenizer = load_tokenizer(args.snapshot)
    seconds_old = seconds_new = 0.0
    mismatches = []
    for row in rows:
        start = time.perf_counter()
        old = encode_example_old(row, tokenizer)
        seconds_old += time.perf_counter() - start
        start = time.perf_counter()
        new = encode_example(row, tokenizer)
        seconds_new += time.perf_counter() - start
        if canonical_bytes(old) != canonical_bytes(new) and len(mismatches) < 5:
            mismatches.append(row["example_id"])
    print(json.dumps({"rows": len(rows), "identical": not mismatches, "mismatches": mismatches,
                      "seconds_old": seconds_old, "seconds_new": seconds_new}))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
