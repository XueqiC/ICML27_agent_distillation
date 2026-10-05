"""ALFWorld preference rule, ETO construct_preference.py:114-174 (a2fc5da).

Golden reward = 1. Failed rollout is rejected in full; a success is a tie.
Unknown execution is preserved as unknown, never relabeled as a loss.
"""
import argparse
import json
from pathlib import Path

from baselines.common.conversation import episode_messages
from baselines.common.explore import checked_rollouts
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, write_json


def build_pairs(experts, rollouts):
    """Pure rule on checked {task_id, won, messages} receipts; no token caps."""
    golden = {r["task_id"]: r for r in experts}
    if len(golden) != len(experts) or any(r["won"] is not True for r in experts):
        raise ValueError("One verified golden win per task required")
    if len({r["task_id"] for r in rollouts}) != len(rollouts):
        raise ValueError("One rollout per task per iteration required")
    pairs, counts = [], dict(win=0, tie=0, lose=0, unknown=0, unmatched=0, invalid_end=0)
    decisions = []
    for rollout in rollouts:
        task_id = rollout["task_id"]
        if task_id not in golden:
            counts["unmatched"] += 1
            decisions.append({"task_id": task_id, "decision": "unmatched"})
            continue
        if type(rollout["won"]) is not bool:
            counts["unknown"] += 1
            decisions.append({"task_id": task_id, "decision": "unknown"})
            continue
        if rollout["won"]:
            counts["tie"] += 1
            decisions.append({"task_id": task_id, "decision": "tie"})
            continue
        counts["lose"] += 1
        expert = golden[task_id]
        predicted = list(rollout["messages"])
        if predicted and predicted[-1]["role"] == "user":
            predicted.pop()
        prompt = expert["messages"][:2]
        chosen, rejected = expert["messages"][2:], predicted[2:]
        if (not chosen or not rejected or chosen[-1]["role"] != "assistant"
                or rejected[-1]["role"] != "assistant"):
            counts["invalid_end"] += 1
            decisions.append({"task_id": task_id, "decision": "invalid_end"})
            continue
        if predicted[:2] != prompt:
            raise ValueError("Expert/rollout initial prompts differ")
        pairs.append({"task_id": task_id, "prompt": prompt, "chosen": chosen, "rejected": rejected,
                      "parent_seed_ids": expert.get("parent_seed_ids", []),
                      "golden_reward": 1.0, "rollout_reward": 0.0})
        decisions.append({"task_id": task_id, "decision": "pair"})
    return pairs, {**counts, "P": len(pairs), "decisions": decisions, "no_truncation": True}


def pair_data(verified, exploration, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, evidence_root = load_verified_collection(verified, manifest, freeze)
    experts = []
    for receipt in collection["verified"]:
        episode = checked_episode(evidence_root, receipt)
        experts.append({"task_id": receipt["task_id"], "won": True,
                        "parent_seed_ids": [episode["episode_id"]], "messages": episode_messages(episode)})
    rollouts = [{"task_id": r["task_id"], "won": r["won"], "messages": episode_messages(e)}
                for r, e in checked_rollouts(exploration, manifest, freeze)]
    if {r["task_id"] for r in rollouts} != {r["task_id"] for r in experts}:
        raise ValueError("ETO explores exactly the SAME K seed tasks")
    pairs, report = build_pairs(experts, rollouts)
    return pairs, report


def load_pair_data(path, config):
    path = Path(path)
    provenance = json.loads((path.parent / "provenance.json").read_text())
    details = provenance["details"]
    pairs, report = pair_data(details["verified"], details["exploration"], config)
    if (json.loads(path.read_text()) != pairs or digest(pairs) != details["pairs_sha256"]
            or json.loads((path.parent / "report.json").read_text()) != report):
        raise ValueError("Pair data differs from checked source evidence")
    return pairs


def prepare_pairs(verified, exploration, output, config, *, inputs=()):
    pairs, report = pair_data(verified, exploration, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="eto_pairs", config=config, inputs=[*inputs,
        Path(verified) / "manifest.json", Path(exploration) / "receipts.json",
        config["paths"]["manifest"], config["paths"]["split_freeze"]],
        details={"pairs_sha256": digest(pairs), "rule": "ETO-ALFWorld-a2fc5da",
                 "verified": str(Path(verified).resolve()), "exploration": str(Path(exploration).resolve())})
    write_json(output / "pairs.json", pairs)
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--verified", required=True)
    parser.add_argument("--exploration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_pairs(args.verified, args.exploration, args.output,
                                  load_config(args.config), inputs=[args.config])))


if __name__ == "__main__":
    main()
