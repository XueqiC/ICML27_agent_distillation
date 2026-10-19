"""Paired black-box labels and explicit, source-checked HAD training whitelist."""
import json
import os
from pathlib import Path
import re

from baselines.common.explore import checked_rollouts
from baselines.had import RECIPE
from baselines.had.harness import augment, messages, records, validity
from sed.manifest import load_frozen, resolve_task
from sed.provenance import write_provenance
from sed.react import action_outcome
from sed.seed_evidence import load_verified_collection
from sed.storage import digest, file_hash, JsonlLedger, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from teachers.capped import from_config
from teachers.openai_teacher import Teacher
from train.dataset import encode_turn


class DrawTeacher(Teacher):
    """Independent draws per visited state/view; retries reuse that exact draw."""
    draw_id = None

    def cache_key(self, history, task, visible_context):
        if self.draw_id is None:
            raise ValueError("HAD teacher draw identity missing")
        return digest([super().cache_key(history, task, visible_context), self.draw_id])

    def generate(self, history, task, visible_context):
        result = super().generate(history, task, visible_context)
        # Shared Teacher caches parseable actions. HAD also distills complete
        # unparsable replies (preference mask=0), preserving their error metadata.
        path = self.cache_dir / (result.metadata["cache_key"] + ".json")
        if result.raw_text.strip() and result.metadata.get("response_status") == "completed" and not path.exists():
            value = {"raw_text": result.raw_text, "parsed_action": result.parsed_action, "metadata": result.metadata}
            write_json(path, {"cache_key": result.metadata["cache_key"], "result": value,
                             "result_sha256": digest(value)}, exclusive=True)
        return result


def require_cpu_teacher():
    # Check allocation as well as visibility: hiding CUDA cannot free a GPU job.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Teacher stages require CUDA_VISIBLE_DEVICES=''")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        allowed = {""} if key in {"SLURM_JOB_GPUS", "SLURM_STEP_GPUS"} else {"", "0"}
        if os.environ.get(key, "").strip() not in allowed:
            raise ValueError("Teacher stage cannot run inside a GPU allocation")


def round_tasks(listing, round_index):
    ids = listing["task_ids"]
    count = min(RECIPE["tasks_per_round"], len(ids))
    selected = [ids[(round_index * count + i) % len(ids)] for i in range(count)]
    return {**listing, "task_ids": selected, "list_sha256": digest(selected),
            "parent_seed_ids": {key: listing["parent_seed_ids"][key] for key in selected}}


def visited_states(exploration, verified, config, round_index):
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, root = load_verified_collection(verified, manifest, freeze)
    from sed.seed_evidence import checked_episode
    parents = {r["task_id"]: [checked_episode(root, r)["episode_id"]] for r in collection["verified"]}
    expected = list(parents)
    n = min(16, len(expected))
    expected = [expected[(round_index * n + i) % len(expected)] for i in range(n)]
    checked = checked_rollouts(exploration, manifest, freeze)
    if [r["task_id"] for r, _ in checked] != expected:
        raise ValueError("HAD rollout differs from cycling VERIFIED support tasks")
    result = []
    for receipt, episode in checked:
        if receipt["parent_seed_ids"] != parents[receipt["task_id"]]:
            raise ValueError("HAD rollout parent seed differs")
        task = resolve_task(manifest, freeze, receipt["task_id"])
        history, page = [], visible_observation(episode["initial"]["visible"])
        for index, step in enumerate(episode["steps"]):
            record = records(history, page)
            if step["policy_metadata"].get("had_records") != record:
                raise ValueError("Rollout did not use the deterministic HAD harness")
            result.append({"state_id": digest([round_index, episode["episode_id"], index]),
                "policy_version": round_index, "task_id": task["task_id"],
                "parent_seed_ids": parents[task["task_id"]], "turn_index": index,
                "task": visible_task(task), "history": visible_history(history),
                "visible": page, "student_reply": step["action_raw"], "harness_records": record})
            history.append({"visible": page, "reply": step["action_raw"]})
            page = visible_observation(step["visible"])
    if not result:
        raise ValueError("No student-visited states; cannot train HAD")
    return result


def collect(exploration, verified, output, config, *, cell, identity, round_index, teacher=None):
    require_cpu_teacher()
    states = visited_states(exploration, verified, config, round_index)
    cell, output = Path(cell), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    if teacher is None:
        teacher = from_config(config, output=cell / "teacher", cache_dir=cell / "teacher/cache",
            usage_ledger=cell / "teacher/usage.jsonl", run_id="had-" + digest(identity),
            cap=RECIPE["teacher_cap"], teacher_class=DrawTeacher,
            sampling={**config["models"]["teacher"]["sampling"], "temperature": 1.0})
    effective = {**config, "had_teacher": {"sampling": teacher.teacher.sampling,
        "max_attempts": 1, "cap": RECIPE["teacher_cap"], "counter_path": str(teacher.counter_path)}}
    provenance = write_provenance(output, command="had_teacher", config=effective,
        inputs=[Path(exploration) / "receipts.json", Path(verified) / "manifest.json"],
        details={"round": round_index, "identity_sha256": digest(identity), "exploration": str(exploration),
                 "verified": str(verified), "states_sha256": digest(states), "teacher_cap": 30000})
    ledger, rows = JsonlLedger(output / "queries.jsonl"), []
    for state in states:
        task = dict(resolve_task(manifest, freeze, state["task_id"]), parent_seed_ids=state["parent_seed_ids"])
        labels = {}
        for view in ("with_harness", "without_harness"):
            draw_id = digest([digest(identity), state["state_id"], view])
            teacher.teacher.draw_id = draw_id
            page = augment(state["history"], state["visible"]) if view == "with_harness" else state["visible"]
            ledger.append({"state_id": state["state_id"], "view": view, "draw_id": draw_id, "status": "started"})
            reply = teacher.generate(state["history"], task, page)
            cache = teacher.teacher.cache_dir / (reply.metadata["cache_key"] + ".json")
            labels[view] = {"raw_text": reply.raw_text, "parsed_action": reply.parsed_action,
                "metadata": reply.metadata, "cache_path": str(cache), "cache_sha256": file_hash(cache)}
            ledger.append({"state_id": state["state_id"], "view": view, "draw_id": draw_id,
                           "status": "returned", **labels[view]})
        row = {**state, **labels}
        rows.append(row)
        JsonlLedger(output / "states.jsonl").append(row, unique_key="state_id")
    write_json(output / "data.json", {"rows": rows, "provenance_sha256": digest(provenance)})
    write_json(output / "report.json", {"states": len(rows), "teacher_queries": 2 * len(rows),
        "policy_version": round_index, "training_exposure": 0})


def load_rows(path, config, *, verified, identity, round_index):
    path = Path(path)
    dataset = json.loads(path.read_text())
    provenance = json.loads((path.parent / "provenance.json").read_text())
    detail = provenance["details"]
    if (dataset["provenance_sha256"] != digest(provenance)
            or detail["identity_sha256"] != digest(identity) or detail["round"] != round_index
            or Path(detail["verified"]).resolve() != Path(verified).resolve()):
        raise ValueError("HAD teacher dataset identity changed")
    for source, expected in provenance["inputs"].items():
        if file_hash(source) != expected:
            raise ValueError("HAD teacher input changed")
    states = visited_states(detail["exploration"], verified, config, round_index)
    if digest(states) != detail["states_sha256"] or len(states) != len(dataset["rows"]):
        raise ValueError("HAD visited states changed")
    cache_teacher = DrawTeacher(cache_dir=path.parent, usage_ledger=path.parent / "unused",
        run_id="validation_only", provider_model_id=config["models"]["teacher"]["provider_model_id"],
        sampling={**config["models"]["teacher"]["sampling"], "temperature": 1.0})
    for state, row in zip(states, dataset["rows"]):
        if any(row[key] != value for key, value in state.items()):
            raise ValueError("HAD training history differs from executed student history")
        for view in ("with_harness", "without_harness"):
            label = row[view]
            cache_teacher.draw_id = digest([digest(identity), state["state_id"], view])
            page = augment(state["history"], state["visible"]) if view == "with_harness" else state["visible"]
            expected_key = cache_teacher.cache_key(state["history"],
                {"task_id": state["task_id"], **state["task"]}, page)
            cache = json.loads(Path(label["cache_path"]).read_text())
            if (file_hash(label["cache_path"]) != label["cache_sha256"]
                    or cache["cache_key"] != expected_key
                    or digest(cache["result"]) != cache["result_sha256"]
                    or any(label[k] != cache["result"][k] for k in ("raw_text", "parsed_action"))):
                raise ValueError("HAD label differs from durable capped teacher output")
    return dataset["rows"]


def reasoning_prefix(reply):
    matches = list(re.finditer(r"^[ \t]*Action[ \t]*:[ \t]*([^\r\n]*)", reply, re.MULTILINE))
    return reply[:matches[-1].start(1)].lstrip() if matches and matches[-1][1].strip() else None


def action_row(tokenizer, prompt, prefix, action, *, max_length):
    text = prefix + action
    row = encode_turn(tokenizer, prompt, text, max_length=max_length)
    tokens = tokenizer(text.strip(), add_special_tokens=False, return_offsets_mapping=True)
    if tokens["input_ids"] != row["input_ids"][row["prompt_tokens"]:-1]:
        raise ValueError("Action span tokenizer disagrees with native turn encoder")
    labels = [-100] * len(row["input_ids"])
    for i, (_, end) in enumerate(tokens["offset_mapping"]):
        if end > len(prefix):
            labels[row["prompt_tokens"] + i] = tokens["input_ids"][i]
    if all(v == -100 for v in labels):
        raise ValueError("Empty action token span")
    return {**row, "labels": labels, "supervised_tokens": sum(v != -100 for v in labels)}


def encode_state(row, tokenizer, *, max_length=32768):
    # Explicit whitelist: no serialized event rows, teacher metadata or audit.
    history, page = visible_history(row["history"]), visible_observation(row["visible"])
    prompt = messages(history, visible_task(row["task"]), page)
    record = records(history, page)
    positive = row["with_harness"]["raw_text"]
    negative = row["without_harness"]["raw_text"]
    chosen = action_outcome(positive, page["admissible_commands"])[0]
    rejected = action_outcome(negative, page["admissible_commands"])[0]
    mask, reason = validity(chosen, record)
    prefix = reasoning_prefix(row["student_reply"])
    if prefix is None or rejected is None:
        mask, reason = False, "unparseable_student_or_negative_action"
    response = encode_turn(tokenizer, prompt, positive, max_length=max_length)
    result = {"state_id": row["state_id"], "policy_version": row["policy_version"],
        "response": response, "mask": mask, "mask_reason": reason,
        "label_provenance": {"task_id": row["task_id"], "parent_seed_ids": row["parent_seed_ids"],
                             "source": "had_paired_teacher", "training_eligible": True}}
    # Dummy preference forwards retain collective alignment on masked ranks.
    for key, action in (("chosen", chosen), ("rejected", rejected)):
        result[key] = action_row(tokenizer, prompt, prefix, action, max_length=max_length) if mask else response
    return result
