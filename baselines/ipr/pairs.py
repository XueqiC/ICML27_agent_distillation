"""Whole-continuation rules from IPR 93335a3 construct_preference...:232-350.

Flags: --global_traj --local_traj --traj_threshold .01 --step_threshold .01;
no --step_action. j is the zero-based branch action; its expert MC value was
measured after j+1 expert actions. All j=0 MC samples are retained but unused.
"""
import math
from pathlib import Path

from baselines.common.conversation import episode_messages, encode_continuation
from baselines.ipr.rollouts import checked_rollouts, seeds, read
from sed.provenance import write_provenance
from sed.storage import digest, file_hash, write_json


def mc_mean(outcomes):
    if len(outcomes) != 5 or any(type(value) is not bool for value in outcomes):
        return None
    return sum(outcomes) / 5


def known(value):
    return type(value) in (int, float, bool) and math.isfinite(value) and 0 <= value <= 1


def build_pairs(experts, rollouts, *, traj_threshold=.01, step_threshold=.01):
    golden = {r["task_id"]: r for r in experts}
    if len(golden) != len(experts) or min(traj_threshold, step_threshold) < 0:
        raise ValueError("Unique experts and nonnegative thresholds required")
    pairs, decisions = [], []
    counts = {"trajectory": 0, "entire": 0, "agent_chosen": 0, "expert_chosen": 0,
              "tie": 0, "unknown": 0, "empty": 0}
    seen = set()
    for agent in rollouts:
        task_id, j = agent["task_id"], agent["j"]
        if (task_id, j) in seen or task_id not in golden:
            raise ValueError("Duplicate branch or non-seed task")
        seen.add((task_id, j))
        expert = golden[task_id]
        if type(j) is not int or not 0 <= j < (len(expert["messages"]) - 1) // 2:
            raise ValueError("Branch outside expert trajectory")
        cut = 2 + 2 * j
        prompt, gold = expert["messages"][:cut], expert["messages"][cut:]
        predicted = list(agent["messages"])
        if predicted and predicted[-1]["role"] == "user":
            predicted.pop()  # Official agent histories include final feedback.
        if predicted[:cut] != prompt:
            raise ValueError("Agent prefix differs from checked expert prefix")
        continuation = predicted[cut:]
        flag = len(continuation) < len(gold)
        winner, decision = None, "tie"
        ef, af = expert["final_reward"], agent["final_reward"]
        em, am = expert.get("mc", {}).get(j), agent.get("mc")
        if not known(ef) or not known(af) or (j >= 1 and (not known(em) or not known(am))):
            decision = "unknown"
        elif (not continuation or not gold or continuation[-1]["role"] != "assistant"
              or gold[-1]["role"] != "assistant"
              or any(not m["content"].strip() for m in [*expert["messages"], *predicted])):
            decision = "empty"
        elif j == 0:
            if af > ef + traj_threshold:
                winner = "agent"
            elif ef > af + traj_threshold:
                winner = "expert"
        else:
            # Preserve upstream order and its fixed .01 FINAL-reward threshold.
            if (af > ef + .01 or flag) and am > em + step_threshold:
                winner = "agent"
            elif ef > af + .01 and em > am + step_threshold:
                winner = "expert"
        kind = "trajectory" if j == 0 else "entire"
        if winner:
            decision = winner + "_chosen"
            counts[kind] += 1
            pairs.append({"task_id": task_id, "j": j, "kind": kind, "prompt": prompt,
                "chosen": continuation if winner == "agent" else gold,
                "rejected": gold if winner == "agent" else continuation,
                "chosen_source": "student_exploration" if winner == "agent" else "trusted_teacher_seed",
                "rejected_source": "trusted_teacher_seed" if winner == "agent" else "student_exploration",
                "parent_seed_ids": expert["parent_seed_ids"], "agent_episode_id": agent.get("episode_id"),
                "expert_final": ef, "agent_final": af, "expert_mc": em, "agent_mc": am, "flag": flag})
        counts[decision] += 1
        decisions.append({"task_id": task_id, "j": j, "kind": kind, "flag": flag, "decision": decision,
                          "expert_final": ef, "agent_final": af, "expert_mc": em, "agent_mc": am})
    return pairs, {"P": len(pairs), "P_by_kind": {k: counts[k] for k in ("trajectory", "entire")},
        **counts, "decisions": decisions, "no_truncation": True, "training_exposure": 0}


def values(rollouts):
    grouped = {}
    for receipt, _ in rollouts:
        grouped.setdefault((receipt["task_id"], receipt["j"]), []).append(receipt["success"])
    return {key: mc_mean(outcomes) for key, outcomes in grouped.items()}


def pair_data(verified, exploration, expert_mc, agent_mc, config):
    experts = seeds(verified, config)
    explored = checked_rollouts(exploration, config, expected_kind="explore")
    iteration = read(Path(exploration) / "provenance.json")["details"]["iteration"]
    gold_rollouts = checked_rollouts(expert_mc, config, expected_kind="expert-mc", expected_iteration=0)
    agent_rollouts = checked_rollouts(agent_mc, config, expected_kind="agent-mc", expected_iteration=iteration)
    agent_prov = read(Path(agent_mc) / "provenance.json")
    expert_prov = read(Path(expert_mc) / "provenance.json")
    if Path(agent_prov["details"]["exploration"]).resolve() != Path(exploration).resolve():
        raise ValueError("Agent MC must continue this iteration's exploration")
    for provenance in (agent_prov, expert_prov, read(Path(exploration) / "provenance.json")):
        if Path(provenance["details"]["verified"]).resolve() != Path(verified).resolve():
            raise ValueError("IPR stages must share the same verified seeds")
    for key in ("model_path", "training_provenance_sha256"):
        if (agent_prov["config"]["server_provenance"]["config"]["server"][key]
                != expert_prov["config"]["server_provenance"]["config"]["server"][key]):
            raise ValueError("Expert and agent MC require the same fixed SFT model")
    gm, am = values(gold_rollouts), values(agent_rollouts)
    golden = [{"task_id": t["task_id"], "messages": episode_messages(e), "final_reward": 1.,
        "parent_seed_ids": [e["episode_id"]],
        "mc": {j: gm[(t["task_id"], j)] for j in range(len(e["steps"]))}} for t, e in experts]
    agents = [{"task_id": r["task_id"], "j": r["j"], "messages": episode_messages(e),
        "final_reward": float(r["success"]), "mc": am[(r["task_id"], r["j"])],
        "episode_id": e["episode_id"]} for r, e in explored]
    return build_pairs(golden, agents)


def prepare_pairs(verified, exploration, expert_mc, agent_mc, output, config):
    pairs, report = pair_data(verified, exploration, expert_mc, agent_mc, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    details = {k: str(Path(v).resolve()) for k, v in dict(verified=verified, exploration=exploration,
               expert_mc=expert_mc, agent_mc=agent_mc).items()}
    details.update(pairs_sha256=digest(pairs), rule="IPR-93335a3-global-local-entire-.01-.01")
    write_provenance(output, command="ipr_pairs", config=config, details=details,
        inputs=[Path(verified) / "manifest.json", config["paths"]["manifest"], config["paths"]["split_freeze"],
                *[Path(root) / p for root in (exploration, expert_mc, agent_mc)
                  for p in ("provenance.json", "receipts.json", "plan.json")]])
    write_json(output / "pairs.json", pairs)
    write_json(output / "report.json", report)
    return report


def load_pair_data(path, config):
    path = Path(path)
    provenance = read(path.parent / "provenance.json")
    for source, expected in provenance["inputs"].items():
        if file_hash(source) != expected:
            raise ValueError("IPR pair source evidence changed")
    details = provenance["details"]
    pairs, report = pair_data(**{k: details[k] for k in ("verified", "exploration", "expert_mc", "agent_mc")}, config=config)
    if read(path) != pairs or digest(pairs) != details["pairs_sha256"] or read(path.parent / "report.json") != report:
        raise ValueError("IPR pairs differ from checked execution evidence")
    return pairs


def preference_conversation(pair, tokenizer, *, max_length=32768):
    prompt = pair["prompt"]
    if len(prompt) != 2 + 2 * pair["j"] or prompt[-1]["role"] != "user":
        raise ValueError("IPR prompt must end at the branch observation")
    rows = {}
    for side in ("chosen", "rejected"):
        row, ids = encode_continuation(prompt, pair[side], tokenizer, task_id=pair["task_id"],
                                      source=pair[side + "_source"], max_length=max_length)
        provenance = row["label_provenance"]
        provenance.update(source=pair[side + "_source"], parent_seed_ids=pair["parent_seed_ids"],
            trusted_seed=pair[side + "_source"] == "trusted_teacher_seed", kind=pair["kind"], j=pair["j"],
            agent_episode_id=pair["agent_episode_id"], masked_prefix_turns=pair["j"])
        for turn in provenance["turns"]:
            turn["source"] = provenance["source"]
        rows[side] = row
    return {"task_id": pair["task_id"], "prompt_input_ids": ids,
        "empty_visible_turns_masked": sum(r["empty_visible_turns_masked"] for r in rows.values()),
        "label_provenance": {
        "kind": pair["kind"], "j": pair["j"], "parent_seed_ids": pair["parent_seed_ids"],
        **{side: rows[side]["label_provenance"] for side in rows}}, **rows}
