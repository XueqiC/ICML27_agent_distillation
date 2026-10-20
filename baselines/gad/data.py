"""Executed H0 histories and one durable, capped black-box draw per history."""
from pathlib import Path
from teachers.stage import teacher_stage, count, require_completed

from baselines.common.explore import checked_rollouts
from baselines.common.method_data import read
from baselines.had.data import DrawTeacher, require_cpu_teacher
from sed.manifest import load_frozen, resolve_task
from sed.provenance import write_provenance
from sed.react import build_messages
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, file_hash, JsonlLedger, write_json
from sed.visibility import visible_history, visible_observation, visible_task
from teachers.capped import from_config
from teachers.baseline_transport import baseline_transport
from train.dataset import encode_turn


def visited_states(exploration, verified, config, round_index):
    if round_index not in range(3):
        raise ValueError("GAD has exactly three history collection rounds")
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, selection, root = load_verified_collection(verified, manifest, freeze)
    from baselines.common import registered_support
    if not registered_support(selection):
        raise ValueError("Unregistered VERIFIED support collection")
    parents = {r["task_id"]: [checked_episode(root, r)["episode_id"]] for r in collection["verified"]}
    rollouts = checked_rollouts(exploration, manifest, freeze)
    if [r["task_id"] for r, _ in rollouts] != list(parents):
        raise ValueError("GAD requires exactly the VERIFIED support TRAIN tasks")
    result = []
    for receipt, episode in rollouts:
        if receipt["parent_seed_ids"] != parents[receipt["task_id"]] or len(episode["steps"]) > 40:
            raise ValueError("GAD parent seed or turn budget changed")
        task = resolve_task(manifest, freeze, receipt["task_id"])
        history, page = [], visible_observation(episode["initial"]["visible"])
        for index, step in enumerate(episode["steps"]):
            result.append({"state_id": digest([round_index, episode["episode_id"], index]),
                "round": round_index, "task_id": task["task_id"], "turn_index": index,
                "parent_seed_ids": parents[task["task_id"]], "task": visible_task(task),
                "history": visible_history(history), "visible": page})
            history.append({"visible": page, "reply": step["action_raw"]})
            page = visible_observation(step["visible"])
    if not result:
        raise ValueError("No student-visited histories")
    return result


@teacher_stage("output", name="gad.teacher-r")
def collect(exploration, verified, output, config, *, cell, identity, round_index, teacher=None):
    require_cpu_teacher()
    states = visited_states(exploration, verified, config, round_index)
    cell, output = Path(cell), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cap = 3 * 40 * identity["unique_instances"]
    if teacher is None:
        teacher = from_config(config, output=cell / "teacher", cache_dir=cell / "teacher/cache",
            usage_ledger=cell / "teacher/usage.jsonl", run_id="gad-" + digest(identity), cap=cap,
            teacher_class=DrawTeacher, baseline_method="gad",
            sampling={**config["models"]["teacher"]["sampling"], "max_output_tokens": 1024})
    provenance = write_provenance(output, command="gad_teacher", config=config,
        inputs=[Path(exploration) / "receipts.json", Path(verified) / "manifest.json"],
        details={"round": round_index, "identity_sha256": digest(identity), "exploration": str(exploration),
            "verified": str(Path(verified).resolve()), "states_sha256": digest(states),
            "sampling": teacher.teacher.sampling, "teacher_cap": cap,
            "baseline_transport": baseline_transport("gad").disclosure(),
            "max_attempts": teacher.teacher.max_attempts,
            "retry_rate_limits_only": teacher.teacher.retry_rate_limits_only,
            "verification": "unknown: teacher replies are not executed success labels"})
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    rows, ledger = [], JsonlLedger(output / "queries.jsonl")
    for state in states:
        teacher.teacher.draw_id = digest([digest(identity), state["state_id"]])
        task = dict(resolve_task(manifest, freeze, state["task_id"]), parent_seed_ids=state["parent_seed_ids"])
        ledger.append({"state_id": state["state_id"], "status": "started"})
        reply = teacher.generate(state["history"], task, state["visible"])
        require_completed(reply)
        if reply.metadata.get("parser_error"):
            count("rejections")
        cache = teacher.teacher.cache_dir / (reply.metadata["cache_key"] + ".json")
        label = {"raw_text": reply.raw_text, "parsed_action": reply.parsed_action,
                 "metadata": reply.metadata, "cache_path": str(cache), "cache_sha256": file_hash(cache)}
        row = {**state, "teacher": label}
        ledger.append({"status": "returned", **row})
        rows.append(row)
    write_json(output / "data.json", {"rows": rows, "provenance_sha256": digest(provenance)})
    write_json(output / "report.json", {"states": len(rows), "round": round_index,
        "teacher_cap": cap, "teacher_queries": len(rows), "verification": "unknown"})


def load_rows(path, config, *, verified, identity, round_index):
    path = Path(path)
    dataset, provenance = read(path), read(path.parent / "provenance.json")
    detail = provenance["details"]
    if (dataset["provenance_sha256"] != digest(provenance) or detail["identity_sha256"] != digest(identity)
            or detail["round"] != round_index or provenance["config"] != config
            or Path(detail["verified"]).resolve() != Path(verified).resolve()):
        raise ValueError("GAD teacher data identity changed")
    for source, expected in provenance["inputs"].items():
        if file_hash(source) != expected:
            raise ValueError("GAD teacher input changed")
    states = visited_states(detail["exploration"], verified, config, round_index)
    if digest(states) != detail["states_sha256"] or len(states) != len(dataset["rows"]):
        raise ValueError("GAD visited histories changed")
    teacher = DrawTeacher(cache_dir=path.parent, usage_ledger=path.parent / "unused", run_id="validation",
        provider_model_id=config["models"]["teacher"]["provider_model_id"], sampling=detail["sampling"])
    for state, row in zip(states, dataset["rows"]):
        if any(row[key] != value for key, value in state.items()):
            raise ValueError("GAD training history differs from real execution")
        teacher.draw_id = digest([digest(identity), state["state_id"]])
        key = teacher.cache_key(state["history"], {"task_id": state["task_id"], **state["task"]}, state["visible"])
        label = row["teacher"]
        cache = read(label["cache_path"])
        if (file_hash(label["cache_path"]) != label["cache_sha256"] or cache["cache_key"] != key
                or digest(cache["result"]) != cache["result_sha256"]
                or any(label[k] != cache["result"][k] for k in ("raw_text", "parsed_action"))):
            raise ValueError("GAD teacher label differs from its capped raw output")
    return dataset["rows"]


def encode_state(row, tokenizer):
    messages = build_messages(visible_history(row["history"]), visible_task(row["task"]),
                              visible_observation(row["visible"]))
    encoded = encode_turn(tokenizer, messages, row["teacher"]["raw_text"], max_length=32768)
    if encoded["supervised_tokens"] > 1536:
        raise ValueError("Teacher reply exceeds GAD response capacity; preserve raw output, never truncate")
    return encoded
