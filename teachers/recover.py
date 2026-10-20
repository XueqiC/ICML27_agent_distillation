"""Acknowledge a fixed transport incident without erasing evidence or spent caps.

Usage: python -m teachers.recover --counter PATH --reason transport_fixed
Run only after investigating the retained evidence. Unknown usage stays unknown.
If no budget remains, recovery refuses: use a separately disclosed new run.
"""
import argparse
import json
from pathlib import Path

from sed.provenance import utc_now
from sed.storage import digest, locked, write_json
from teachers.capped import BudgetExceeded


def recover(counter_path, *, reason):
    if reason not in {"transport_fixed", "interruption_reviewed"}:
        raise ValueError("Use a registered recovery reason; never persist provider error text")
    counter_path = Path(counter_path)
    with locked(str(counter_path) + ".lock"):
        counter = json.loads(counter_path.read_text())
        identity = counter["identity"]
        ledger = Path(identity["usage_ledger"])
        rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        rows = [r for r in rows if r.get("run_id") == identity["run_id"]]
        slots = {r.get("logical_call_id", r["call_id"]) for r in rows if r["status"] == "started"}
        counter["attempts"] = max(counter["attempts"], len(slots))
        if counter["attempts"] >= identity["cap"]:
            raise BudgetExceeded("No teacher budget remains; retain evidence and use a separately disclosed new run")
        recovery = {"timestamp": utc_now(), "reason": reason, "through_rows": len(rows),
            "ledger_prefix_sha256": digest(rows), "previous_counter": counter,
            "unknown_usage": "unchanged; no reservations refunded"}
        directory = counter_path.parent / "recoveries"
        number = len(list(directory.glob("*.json")))
        write_json(directory / f"{number:04d}.json", recovery, exclusive=True)
        write_json(counter_path, {**counter, "pending": False, "halted": False,
            "recovery": {key: recovery[key] for key in ("through_rows", "ledger_prefix_sha256")}})
    return recovery


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counter", required=True)
    parser.add_argument("--reason", required=True, choices=["transport_fixed", "interruption_reviewed"])
    args = parser.parse_args(argv)
    recover(args.counter, reason=args.reason)
    print("Recovery recorded; spent reservations and unknown usage preserved")


if __name__ == "__main__":
    main()
