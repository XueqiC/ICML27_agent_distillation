"""Paper Eq. 2/3/5/7-10 and the released ALFWorld reflection prompt.

Small prompt port from WangHanLinHenry/STeCa 76f6ea7,
data_collection/data_cali_alfworld.py:336-356. No upstream API code is imported.
"""
import math
import os
from pathlib import Path
from teachers.stage import teacher_stage, count, require_completed

from baselines.common.conversation import episode_messages, encode_continuation
from baselines.common.method_data import checked, publish, read
from baselines.ipr.rollouts import seeds, checked_rollouts
from baselines.ipr.pairs import values
from sed.react import parse_action
from sed.provenance import write_provenance
from sed.storage import digest, write_json
from teachers.openai_teacher import Teacher
from teachers.capped import from_config
from teachers.baseline_transport import baseline_transport

REFLECTION = ("Above is the interaction history. However, the last step is not optimal and may lead to a wrong direction. "
    "The next step ground truth action is: {action}. Please provide the thought which would lead the agent to generate "
    "the ground truth action and be aware of the last non-optimal action. The thought should follow the format of the interaction history.")


def ndtw(left, right, encode):
    """Eq. 2-3: cosine distance and sqrt(nx^2+ny^2) normalization, NOT exp(-DTW)."""
    if not left or not right:
        raise ValueError("nDTW requires two nonempty action trajectories")
    vectors = encode(left + right)
    if len(vectors) != len(left) + len(right):
        raise ValueError("Embedding count differs")
    def cosine(x, y):
        if len(x) != len(y) or not x:
            raise ValueError("Invalid embeddings")
        norm = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Nonfinite/zero embedding")
        return 1 - max(-1., min(1., sum(a*b for a, b in zip(x, y)) / norm))
    previous = [0.] + [math.inf] * len(right)
    for x in vectors[:len(left)]:
        current = [math.inf]
        for j, y in enumerate(vectors[len(left):], 1):
            current.append(cosine(x, y) + min(previous[j], current[-1], previous[j-1]))
        previous = current
    # Do not silently clip the paper's formula (cosine can make it exceed one).
    return previous[-1] / math.hypot(len(left), len(right))


def deviation(agent_reward, preceding_expert_reward, delta=0):
    for value in (agent_reward, preceding_expert_reward):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Five known MC outcomes required; unknown is not zero")
    return preceding_expert_reward - agent_reward > delta


def plan(verified, exploration, expert_mc, agent_mc, config):
    experts = seeds(verified, config)
    explored = checked_rollouts(exploration, config, expected_kind="explore", expected_iteration=1)
    gold = checked_rollouts(expert_mc, config, expected_kind="expert-mc", expected_iteration=0)
    agent = checked_rollouts(agent_mc, config, expected_kind="agent-mc", expected_iteration=1)
    provs = [read(Path(root) / "provenance.json") for root in (exploration, expert_mc, agent_mc)]
    if any(Path(p["details"]["verified"]).resolve() != Path(verified).resolve() for p in provs):
        raise ValueError("STeCa stages must use the same support collection")
    if Path(provs[2]["details"]["exploration"]).resolve() != Path(exploration).resolve():
        raise ValueError("Agent MC must continue this exploration")
    for key in ("model_path", "training_provenance_sha256"):
        if len({p["config"]["server_provenance"]["config"]["server"][key] for p in provs}) != 1:
            raise ValueError("STeCa exploration and MC must use the fixed warm-up model")
    gm, am = values(gold), values(agent)
    expert_map = {t["task_id"]: (t, e) for t, e in experts}
    result, selected = [], set()
    for receipt, episode in explored:
        task, expert = expert_map[receipt["task_id"]]
        j = receipt["j"]
        expert_messages, student_messages = episode_messages(expert), episode_messages(episode)
        cut = 2 + 2*j
        if student_messages[:cut] != expert_messages[:cut]:
            raise ValueError("Explored prefix differs from verified expert")
        if len(student_messages) <= cut:
            raise ValueError("No explored action after prefix")
        is_deviated = j > 0 and deviation(am[(task["task_id"], j)], gm[(task["task_id"], j-1)])
        first = is_deviated and task["task_id"] not in selected
        if first:
            selected.add(task["task_id"])
        result.append({"task_id": task["task_id"], "parent_seed_id": expert["episode_id"],
            "episode_id": episode["episode_id"], "j": j, "prompt": expert_messages[:cut],
            "expert": expert_messages[cut:], "explored": student_messages[cut:],
            "full_explored": student_messages, "full_expert": expert_messages,
            "success": receipt["success"], "deviated": bool(first),
            "agent_mc": am[(task["task_id"], j)], "preceding_expert_mc": gm.get((task["task_id"], j-1)),
            "decision": "first_deviation" if first else "later_deviation" if is_deviated else
                        "no_preceding_expert" if j == 0 else "no_drop"})
    return result


def reflection_prompt(row):
    # Only canonical visible messages and the executed deviation; no audit fields.
    messages = row["prompt"] + row["explored"][:1]
    history = "\n\n".join(("Gpt" if m["role"] == "assistant" else "Human") + ": " + m["content"] for m in messages)
    return history + "\n\n" + REFLECTION.format(action=parse_action(row["expert"][0]["content"]))


class ReflectionTeacher(Teacher):
    def __init__(self, **kwargs):
        super().__init__(prompt_template=REFLECTION, **kwargs)

    def prompt(self, history, task, visible_context):
        # Collector supplies only the whitelisted rendered prompt in observation.
        return [{"role": "user", "content": visible_context["observation"]}]

    def parse_response(self, raw):
        # The response is a visible reflective explanation, not a ground-truth action.
        return None, None


def calibrated_reply(raw, expert_action):
    from sed.thinking import split_thinking
    visible, hidden = split_thinking(raw)
    if hidden or not visible.strip():
        raise ValueError("Reflection has no eligible visible text")
    # Upstream replace_action forces the demonstrated action. Never execute a
    # teacher-suggested replacement or relabel synthetic text as verified success.
    thought = visible.split("Action:", 1)[0].strip()
    if thought.startswith("Thought:"):
        thought = thought[len("Thought:"):].strip()
    if not thought:
        raise ValueError("Reflection missing thought")
    return "Thought: " + thought + "\nAction: " + expert_action


@teacher_stage("output", name="steca.reflect")
def reflect(planned, config, verified, output, ledger_root, teacher_factory=from_config):
    allocated = [os.environ.get(k, "") for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")]
    if any(value not in ("", "-1", "NoDevFiles") for value in allocated) or os.environ.get("SLURM_GPUS_ON_NODE", "0") != "0":
        raise ValueError("Reflection is a CPU-only stage; resume with --through reflect in a job without GPUs")
    output, ledger_root = Path(output), Path(ledger_root)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="steca_reflection", config=config,
        inputs=[Path(verified) / "manifest.json", config["paths"]["manifest"], config["paths"]["split_freeze"]],
        details={"plan_sha256": digest(planned), "ledger_root": str(ledger_root.resolve()),
                 "cap": sum(r["deviated"] for r in planned), "teacher": "gpt-6-luna",
                 "baseline_transport": baseline_transport("steca").disclosure()})
    candidates = [r for r in planned if r["deviated"]]
    if config["models"]["teacher"]["provider_model_id"] != "gpt-6-luna":
        raise ValueError("STeCa requires the configured gpt-6-luna teacher")
    teacher = teacher_factory(config, output=ledger_root, cache_dir=ledger_root / "cache",
        usage_ledger=ledger_root / "usage.jsonl", run_id="steca-" + digest(planned),
        cap=len(candidates), teacher_class=ReflectionTeacher, baseline_method="steca")
    tasks = {t["task_id"]: t for t, _ in seeds(verified, config)}
    reflections = []
    for row in candidates:
        result = teacher.generate([], tasks[row["task_id"]],
            {"observation": reflection_prompt(row), "admissible_commands": []})
        require_completed(result)
        record = {"task_id": row["task_id"], "episode_id": row["episode_id"], "j": row["j"],
            "raw_output": result.raw_text, "metadata": result.metadata}
        # Persist before validation; one rejected reflection consumes its only call.
        write_json(output / (digest([row["task_id"], row["j"]]) + ".json"), record)
        try:
            record["reply"] = calibrated_reply(result.raw_text, parse_action(row["expert"][0]["content"]))
        except ValueError:
            count("rejections")
            raise
        reflections.append(record)
    write_json(output / "reflections.json", reflections)
    return reflections


def build_datasets(planned, reflections, encode):
    lookup = {(r["task_id"], r["j"]): r for r in reflections}
    expected = {(r["task_id"], r["j"]) for r in planned if r["deviated"]}
    if set(lookup) != expected or len(lookup) != len(reflections):
        raise ValueError("Exactly one reflection per deviated trajectory required")
    datasets = {"Dc": [], "Ds": [], "De": []}
    for row in planned:
        def add(kind, prompt, target, paired):
            # Paper a_t includes visible rationale plus action, not observations.
            left = [m["content"] for m in target if m["role"] == "assistant"]
            right = [m["content"] for m in paired if m["role"] == "assistant"]
            if any(not v.strip() for v in left + right):
                raise ValueError("Empty visible reply cannot define trajectory distance")
            distance = ndtw(left, right, encode)
            reward = 1 + (-1 if kind == "De" else 1) * .01 * distance
            datasets[kind].append({"kind": kind, "task_id": row["task_id"], "j": row["j"],
                "parent_seed_id": row["parent_seed_id"], "episode_id": row["episode_id"],
                "prompt": prompt, "target": target, "paired": paired, "ndtw": distance, "reward": reward})
        if row["deviated"]:
            reflection = lookup[(row["task_id"], row["j"])]
            reply = calibrated_reply(reflection["raw_output"], parse_action(row["expert"][0]["content"]))
            if reflection["reply"] != reply or reflection["episode_id"] != row["episode_id"]:
                raise ValueError("Reflection identity/content changed")
            target = [{"role": "assistant", "content": reply}] + row["expert"][1:]
            add("Dc", row["prompt"], target, row["explored"])
        if row["success"] is False:
            add("Ds", row["prompt"], row["expert"], row["explored"])
        if row["j"] == 0 and row["success"] is True:
            add("De", row["full_explored"][:2], row["full_explored"][2:], row["full_expert"][2:])
    return datasets


def merge(planned, reflections_path, output, config, verified, source_paths, encode):
    cache = {}
    def cached_encode(texts):
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            cache.update(zip(missing, encode(missing)))
        return [cache[t] for t in texts]
    datasets = build_datasets(planned, read(reflections_path), cached_encode)
    rows = [row for values_ in datasets.values() for row in values_]
    # Save source paths, enabling the loader to reconstruct labels from checked
    # execution and reflection evidence without loading an embedding model.
    roots = [str(Path(p).parent.resolve()) for p in source_paths if Path(p).name == "provenance.json"]
    if len(roots) != 3:
        raise ValueError("STeCa dataset requires exploration/expert-MC/agent-MC provenance")
    sources = dict(zip(("exploration", "expert_mc", "agent_mc"), roots))
    serving = read(Path(sources["exploration"]) / "provenance.json")["config"]["server_provenance"]["config"]["server"]
    publish(output, rows, method="steca", config=config, verified=verified,
        inputs=[*source_paths, reflections_path, Path(verified) / "manifest.json",
                config["paths"]["manifest"], config["paths"]["split_freeze"]],
        details={"plan_sha256": digest(planned), "embedding": encode.identity,
                 "vectors_sha256": digest(cache), "sources": sources,
                 "reflections": str(Path(reflections_path).resolve()),
                 "warmup_checkpoint": serving["model_path"],
                 "warmup_provenance_sha256": serving["training_provenance_sha256"],
                 "counts": {k: len(v) for k, v in datasets.items()}})
    write_json(Path(output) / "vectors.json", cache)
    for kind, data in datasets.items():
        write_json(Path(output) / (kind + ".json"), data)
    return len(rows)


def training_examples(path, config, tokenizer, *, verified):
    rows, details = checked(path, config, method="steca", verified=verified)
    experts = {t["task_id"]: e for t, e in seeds(verified, config)}
    planned = plan(verified, config=config, **details["sources"])
    vectors = read(Path(path).parent / "vectors.json")
    if digest(planned) != details["plan_sha256"] or digest(vectors) != details["vectors_sha256"]:
        raise ValueError("STeCa plan/embedding evidence changed")
    expected = build_datasets(planned, read(details["reflections"]), lambda texts: [vectors[t] for t in texts])
    if rows != [row for values_ in expected.values() for row in values_]:
        raise ValueError("STeCa data differs from checked execution/reflections")
    counts = {k: sum(row["kind"] == k for row in rows) for k in ("Dc", "Ds", "De")}
    if counts != details["counts"]:
        raise ValueError("STeCa dataset counts changed")
    result = []
    for row in rows:
        expert = experts[row["task_id"]]
        if row["parent_seed_id"] != expert["episode_id"] or row["kind"] not in counts:
            raise ValueError("STeCa example outside verified support")
        expected_reward = 1 + (-1 if row["kind"] == "De" else 1) * .01 * row["ndtw"]
        if row["reward"] != expected_reward or not math.isfinite(expected_reward) or expected_reward <= 0:
            raise ValueError("Invalid nDTW reward")
        example, _ = encode_continuation(row["prompt"], row["target"], tokenizer,
            task_id=row["task_id"], source="student_exploration" if row["kind"] == "De" else "trusted_teacher_seed")
        source = "steca_" + row["kind"]
        provenance = example["label_provenance"]
        provenance.update(source=source, trusted_seed=False,
            parent_seed_ids=[expert["episode_id"]], masked_prefix_turns=row["j"], reward=row["reward"])
        for i, turn in enumerate(example["turns"]):
            label_source = ("teacher_reflection_and_expert_action" if row["kind"] == "Dc" and i == 0 else
                            "student_exploration" if row["kind"] == "De" else "trusted_teacher_seed")
            turn["label_provenance"].update(source=source, target_source=label_source,
                trusted_seed=False, parent_seed_ids=[expert["episode_id"]])
        provenance["turns"] = [t["label_provenance"] for t in example["turns"]]
        example.update(reward=row["reward"], dataset_scale=len(rows) / counts[row["kind"]])
        result.append(example)
    return result
