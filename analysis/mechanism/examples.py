"""Deterministic I5 retrieval from the explicitly supplied, verified TRAIN seeds."""
from pathlib import Path

from analysis.mechanism.common import authorize_episode
from analysis.mechanism.goals import normal_type, parse_goal
from sed.manifest import load_frozen
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest
from sed.visibility import visible_observation


def load_seeds(root, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, evidence = load_verified_collection(root, manifest, freeze)
    result = []
    for i, receipt in enumerate(collection["verified"]):
        episode = checked_episode(evidence, receipt)
        result.append({"order": i, "episode": episode, "parent_seed_id": episode["episode_id"],
                       "receipt_sha256": digest(receipt), "collection": str(Path(root).resolve())})
    return result


def select_example(task_id, seeds):
    goal = parse_goal(task_id)
    eligible = []
    for seed in seeds:
        ep = seed["episode"]
        authorize_episode(ep)
        if ep["split"] != "train" or not ep["task_id"].startswith("train/") or ep.get("success") is not True:
            raise ValueError("I5 examples must be successful verified TRAIN trajectories")
        other = parse_goal(ep["task_id"])
        if other.task_type != goal.task_type:
            continue
        score = sum(normal_type(a or "") == normal_type(b or "") and a is not None
                    for a, b in ((goal.object_type, other.object_type),
                                 (goal.movable_receptacle, other.movable_receptacle),
                                 (goal.target_receptacle, other.target_receptacle)))
        eligible.append((-score, seed["order"], seed))
    return min(eligible, key=lambda row: row[:2])[2] if eligible else None


def render_example(seed):
    ep = seed["episode"]
    authorize_episode(ep)
    if ep["split"] != "train" or not ep["task_id"].startswith("train/"):
        raise ValueError("No dev/sealed examples")
    text = ["Example of a solved task:", "Observation: " + visible_observation(ep["initial"]["visible"])["observation"]]
    for step in ep["steps"]:
        text += ["Action: " + (step["action_parsed"] or "[parser error]"),
                 "Observation: " + visible_observation(step["visible"])["observation"]]
    return "\n".join(text)
