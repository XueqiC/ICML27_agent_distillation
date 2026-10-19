"""Cosine top-3000 action-state-action retrieval and full-command frequency mix."""
import math
from pathlib import Path

from baselines.common.embeddings import LocalEncoder
from baselines.common.method_data import checked, read
from baselines.ipr.rollouts import seeds
from sed.react import parse_action
from sed.provenance import write_provenance
from sed.storage import digest, write_json


def cosine(x, y):
    norm = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
    if not norm or not math.isfinite(norm) or len(x) != len(y):
        raise ValueError("Invalid state embedding")
    return sum(a*b for a, b in zip(x, y)) / norm


def build(data, output, config, verified, encode):
    rows, _ = checked(data, config, method="wkm", verified=verified)
    experts = {t["task_id"]: e for t, e in seeds(verified, config)}
    triples = []
    for row in rows:
        expert = experts[row["task_id"]]
        if len(row["states"]) != len(expert["steps"]) or row["parent_seed_id"] != expert["episode_id"]:
            raise ValueError("Knowledge base outside support evidence")
        previous = "look"
        for state, step in zip(row["states"], expert["steps"]):
            action = parse_action(step["action_raw"])
            triples.append({"pre_action": previous, "state": state, "action": action,
                "parent_seed_id": row["parent_seed_id"], "task_id": row["task_id"]})
            previous = action
    vectors = encode([r["state"] for r in triples])
    payload = {"triples": triples, "vectors": vectors, "embedding": encode.identity,
               "source_data_sha256": digest(rows), "top_n": 3000, "gamma": .4}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="wkm_state_knowledge_base", config=config,
        inputs=[data, Path(data).parent / "provenance.json", Path(verified) / "manifest.json"],
        details={"embedding": encode.identity, "source_data_sha256": digest(rows)})
    write_json(output / "knowledge_base.json", {**payload, "sha256": digest(payload)})


class KnowledgeBase:
    def __init__(self, path, encode=None):
        payload = read(path)
        checksum = payload.pop("sha256")
        if digest(payload) != checksum or payload["top_n"] != 3000 or payload["gamma"] != .4:
            raise ValueError("Knowledge base changed")
        self.rows, self.vectors = payload["triples"], payload["vectors"]
        if len(self.rows) != len(self.vectors) or not self.rows:
            raise ValueError("Empty/incomplete state knowledge base")
        self.encode = encode or LocalEncoder(payload["embedding"]["snapshot"], "minilm")
        if self.encode.identity != payload["embedding"]:
            raise ValueError("Retrieval embedding checkpoint changed")
        self.identity = {"sha256": checksum, "embedding": payload["embedding"]}

    def probabilities(self, state, commands, previous="look"):
        vector, = self.encode([state])
        indices = sorted(range(len(self.rows)), key=lambda i: (-cosine(vector, self.vectors[i]), i))[:3000]
        # Released get_action_know_prob_pre_action filters previous action VERB
        # after retrieval. Counts below use complete admissible commands (H0).
        selected = [self.rows[i] for i in indices if self.rows[i]["pre_action"].split()[0] == previous.split()[0]]
        counts = {a: sum(r["action"] == a for r in selected) for a in commands}
        total = sum(counts.values())
        return ({a: v / total for a, v in counts.items()} if total else {}), indices


def mix(commands, logps, knowledge, gamma=.4):
    if not commands or set(logps) != set(commands) or any(not math.isfinite(x) for x in logps.values()):
        raise ValueError("Every admissible action needs a finite agent score")
    highest = max(logps.values())
    scores = {a: math.exp(logps[a] - highest) for a in commands}
    total = sum(scores.values())
    agent = {a: x / total for a, x in scores.items()}
    if knowledge and (set(knowledge) != set(commands) or any(v < 0 for v in knowledge.values())
                      or not math.isclose(sum(knowledge.values()), 1)):
        raise ValueError("Knowledge distribution must cover admissible commands")
    # No matching legal triplet: explicit zero-evidence fallback to the agent.
    knowledge = knowledge or agent
    mixed = {a: gamma*agent[a] + (1-gamma)*knowledge[a] for a in commands}
    return max(commands, key=lambda a: mixed[a]), agent, mixed
