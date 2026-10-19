"""Executed rejected sampler, self-synthesis and missing WKM data merge."""
import argparse
from pathlib import Path

from baselines.common.conversation import episode_messages
from baselines.common.explore import checked_rollouts
from baselines.common.method_data import checked, publish, encode_targets, read
from baselines.ipr.rollouts import seeds
from baselines.wkm.format import task_synthesis, state_synthesis, agent_messages, world_prompt
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.storage import digest, write_json
from students.vllm_student import Student, context_options


class PromptStudent(Student):
    """Student transport/logging reused for explicit, already allowlisted prompts."""
    def messages(self, history, task, visible_context):
        return self.prompt_messages


def source_data(verified, exploration, config):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    experts = seeds(verified, config)
    rollouts = checked_rollouts(exploration, manifest, freeze)
    if {t["task_id"] for t, _ in experts} != {r["task_id"] for r, _ in rollouts}:
        raise ValueError("WKM exploration must equal verified support")
    by_task = {r["task_id"]: (r, e) for r, e in rollouts}
    if len(by_task) != len(experts):
        raise ValueError("WKM explores each support task exactly once")
    for task, expert in experts:
        if by_task[task["task_id"]][0]["parent_seed_ids"] != [expert["episode_id"]]:
            raise ValueError("WKM rollout parent differs from verified support seed")
    return experts, by_task


def synthesize(verified, exploration, output, config, student):
    experts, explored = source_data(verified, exploration, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="wkm_student_self_synthesis", config=config,
        inputs=[Path(verified) / "manifest.json", Path(exploration) / "receipts.json",
                Path(exploration) / "provenance.json", config["paths"]["manifest"], config["paths"]["split_freeze"]],
        details={"verified": str(Path(verified).resolve()), "student": student.spec, "teacher_calls": 0,
                 "sampling": {"temperature": .5, "top_p": 1, "top_k": -1, "seed": 0},
                 "task_max_tokens": 512, "state_max_tokens": 128})
    rows = []
    for task, expert in experts:
        receipt, sampled = explored[task["task_id"]]
        chosen = episode_messages(expert)
        rejected = episode_messages(sampled) if receipt["won"] is False else None
        row = {"task_id": task["task_id"], "parent_seed_id": expert["episode_id"],
            "chosen_sha256": digest(expert), "explored_sha256": digest(sampled),
            "rejected": rejected is not None, "states": [], "calls": []}
        def generate(prompt, max_tokens):
            student.prompt_messages = prompt
            student.spec["max_tokens"] = max_tokens
            result = student.generate([], task, expert["initial"]["visible"])
            row["calls"].append({"prompt": prompt, "raw_output": result.raw_text, "metadata": result.metadata})
            write_json(output / (digest(task["task_id"]) + ".json"), row)
            if not result.raw_text.strip() or result.metadata.get("finish_reason") != "stop":
                raise ValueError("Incomplete self-synthesis; raw output retained")
            return result.raw_text.strip()
        row["task_knowledge"] = generate(task_synthesis(chosen, rejected), 512)
        trajectory = ""
        for j in range(len(expert["steps"])):
            trajectory += chosen[1 + 2*j]["content"] + "\n"
            state = generate(state_synthesis(trajectory), 128)
            row["states"].append(state)
            trajectory += state + "\n" + chosen[2 + 2*j]["content"] + "\n"
        rows.append(row)
        write_json(output / (digest(task["task_id"]) + ".json"), row)
    # Durable raw files are part of stage artifacts; final dataset binds each call.
    write_json(output / "knowledge.json", rows)
    return rows


def merge(verified, exploration, synthesis, output, config):
    experts, explored = source_data(verified, exploration, config)
    knowledge = read(Path(synthesis) / "knowledge.json")
    if len(knowledge) != len(experts):
        raise ValueError("Missing synthesized support knowledge")
    rows = []
    for (task, expert), row in zip(experts, knowledge):
        if (row["task_id"] != task["task_id"] or row["parent_seed_id"] != expert["episode_id"]
                or row["chosen_sha256"] != digest(expert)
                or row["explored_sha256"] != digest(explored[task["task_id"]][1])
                or len(row["states"]) != len(expert["steps"])):
            raise ValueError("Knowledge/execution mismatch")
        calls = row["calls"]
        labels = [row["task_knowledge"], *row["states"]]
        if (len(calls) != len(labels) or any(c["raw_output"].strip() != text or
                c["metadata"].get("finish_reason") != "stop" for c, text in zip(calls, labels))):
            raise ValueError("Knowledge labels differ from completed raw student outputs")
        # Only these synthesized strings become training labels/context.
        rows.append({k: row[k] for k in ("task_id", "parent_seed_id", "task_knowledge", "states")})
    publish(output, rows, method="wkm", config=config, verified=verified,
        inputs=[Path(verified) / "manifest.json", Path(synthesis) / "knowledge.json", Path(synthesis) / "provenance.json",
                Path(exploration) / "receipts.json", Path(exploration) / "provenance.json",
                config["paths"]["manifest"], config["paths"]["split_freeze"]],
        details={"exploration": str(Path(exploration).resolve()), "synthesis": str(Path(synthesis).resolve())})
    for role in ("agent", "world"):
        write_json(Path(output) / (role + ".json"), {"role": role, "data": "data.json"})


def training_examples(path, config, tokenizer, *, verified):
    path = Path(path)
    role = read(path)
    if role not in ({"role": "agent", "data": "data.json"}, {"role": "world", "data": "data.json"}):
        raise ValueError("Unknown WKM training role")
    rows, details = checked(path.parent / "data.json", config, method="wkm", verified=verified)
    experts, _ = source_data(verified, details["exploration"], config)
    # Reconstruct merge from original generated strings, never trust arbitrary messages.
    original = read(Path(details["synthesis"]) / "knowledge.json")
    expected = [{k: r[k] for k in ("task_id", "parent_seed_id", "task_knowledge", "states")} for r in original]
    if rows != expected:
        raise ValueError("WKM merge differs from synthesis")
    result = []
    for (task, expert), row in zip(experts, rows):
        if row["task_id"] != task["task_id"] or row["parent_seed_id"] != expert["episode_id"]:
            raise ValueError("WKM data outside verified support")
        messages = episode_messages(expert)
        if role["role"] == "agent":
            rendered = agent_messages(messages, row["task_knowledge"])
            pairs = [(rendered[:i], rendered[i]["content"]) for i in range(2, len(rendered), 2)]
        else:
            pairs = [(world_prompt(messages[:2]), row["task_knowledge"])]
            pairs += [(world_prompt(messages[:2 + 2*j], row["task_knowledge"], row["states"]), state)
                      for j, state in enumerate(row["states"])]
        example = encode_targets(pairs, tokenizer, task_id=task["task_id"],
            parent_seed_id=expert["episode_id"], source="wkm_" + role["role"])
        for provenance in [example["label_provenance"], *[t["label_provenance"] for t in example["turns"]]]:
            provenance.update(target_source="trusted_teacher_seed" if role["role"] == "agent" else "student_self_synthesis",
                              context_source="visible_expert_history_and_student_self_knowledge")
        result.append(example)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "verified", "exploration", "output", "endpoint", "model-name", "server-provenance"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    config = load_config(a.config)
    server = read(a.server_provenance)
    config["server_provenance"] = server
    class SynthesisStudent(PromptStudent):
        def sampling(self):
            return {"temperature": .5, "top_p": 1, "top_k": -1, "seed": 0}
    student = SynthesisStudent(endpoint=a.endpoint, model=a.model_name,
        revision=config["models"]["student"]["revision"], enable_thinking=False,
        log_dir=Path(a.output).parent / "synthesis_calls", **context_options(config))
    try:
        synthesize(a.verified, a.exploration, a.output, config, student)
    finally:
        student.close()


if __name__ == "__main__":
    main()
