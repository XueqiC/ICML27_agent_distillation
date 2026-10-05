"""Greedy student action agreement on logged TRAIN seed prefixes; no environment."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from analysis.mechanism.common import attempt_path, gap_episodes, new_output, read_json, workspace
from analysis.mechanism.coverage import load_training_states, state_evidence, validate_coverage
from analysis.mechanism.policies import add_student_args, student_spec, validate_continuation
from sed.config import load_config
from sed.react import action_outcome, build_messages
from sed.seed_evidence import artifact_hashes
from sed.storage import BlobStore, digest, file_hash, locked, write_json
from students.vllm_student import Student


def agreement(rows, *, absorption=False):
    n = len(rows)
    known = [r["agrees"] for r in rows if type(r.get("agrees")) is bool]
    matches = sum(known)
    value = {"n": n, "known": len(known), "unknown": n-len(known), "matches": matches,
             "agreement": matches/n if n and len(known) == n else None}
    if absorption:
        # A failed HTTP call is unknown, never evidence of failed absorption.
        value["absorbed"] = matches*2 >= n if n and len(known) == n else None
    return value


def summarize(coverage, states):
    by_id = {r["state_id"]: r for r in states}
    if len(by_id) != len(states):
        raise ValueError("Duplicate seed probe state")
    types = defaultdict(list)
    for row in states:
        types[row["task_type"]].append(row)
    episodes = []
    for ep in coverage["episodes"]:
        patterns = []
        seen = set()
        for needed in ep["needed_steps"]:
            pattern = needed["pattern"]
            if pattern in seen or pattern is None:
                continue
            seen.add(pattern)
            scopes = {}
            for scope in ("same_task_type", "any_task_type"):
                ids = [r["state_id"] for r in needed[scope]]
                if any(key not in by_id for key in ids):
                    raise ValueError("Coverage references a missing training state")
                scopes[scope] = {**agreement([by_id[key] for key in ids], absorption=True), "state_ids": ids}
            patterns.append({"pattern": pattern, "navigation_only": needed["navigation_only"], **scopes})
        def absorbed(scope):
            if ep["covered_"+scope] is not True:
                return None
            values = [p[scope]["absorbed"] for p in patterns if not p["navigation_only"]]
            if not values or any(v is None for v in values):
                return None
            return all(values)
        value = absorbed("any_task_type")
        episodes.append({"task_id": ep["task_id"], "t_star": ep["t_star"], "patterns": patterns,
                         "absorbed": value, "absorbed_any_task_type": value,
                         "absorbed_same_task_type": absorbed("same_task_type"),
                         "status": ("not_covered" if ep["covered"] is False else
                                    "absorbed" if value is True else "not_absorbed" if value is False else "unknown")})
    return {"episodes": episodes, "training_set_fit": {
        "overall": agreement(states), "per_task_type": {k: agreement(v) for k, v in sorted(types.items())},
        "scope": "All logged teacher-forced seed training states, including navigation and invalid teacher targets; no fresh rollouts"}}


def completed_state(root, state):
    for path in sorted(Path(root).glob("attempt-*/receipt.json")):
        receipt = read_json(path)
        payload = path.parent/"payload"
        if artifact_hashes(payload) != receipt["hashes"]:
            raise ValueError("Probe response artifacts changed")
        result = read_json(payload/"result.json")
        if result["state"] != state_evidence(state):
            raise ValueError("Probe state differs from training evidence")
        request = BlobStore(payload/"student_calls/blobs").get(result["metadata"]["request_ref"])
        if (request["messages"] != build_messages(state["history"], {}, state["visible"])
                or request["temperature"] != 0 or request["max_tokens"] != 1024
                or request["chat_template_kwargs"] != {"enable_thinking": False}
                or request["model"] != result["metadata"]["returned_model_id"]):
            raise ValueError("Probe request differs from canonical greedy evaluation")
        parsed, status = action_outcome(result["raw_text"], state["visible"]["admissible_commands"])
        if (result["parsed_action"] != parsed or result["action_outcome"] != status
                or result["agrees"] != (parsed is not None and parsed == state["seed_action"])):
            raise ValueError("Probe scoring differs from the shared action parser")
        return {**state_evidence(state), **{k: v for k, v in result.items() if k != "state"},
                "receipt_path": str(path.resolve()), "receipt_sha256": file_hash(path)}
    return None


def query_states(states, *, output, config, spec, inputs=(), student_factory=Student):
    """One durable attempt per state. Completed calls survive new server ports."""
    rows = []
    for state in states:
        root = Path(output)/"states"/state["state_id"]
        completed = completed_state(root, state)
        if completed is not None:
            rows.append(completed)
            continue
        attempt = attempt_path(root, ".")
        new_output(attempt, config=config, command="probe_state", inputs=inputs,
                   details={"state": state_evidence(state), "student": spec, "environment_steps": 0,
                            "teacher_calls": 0, "training_exposure": 0})
        payload = attempt/"payload"
        student = student_factory(**spec, log_dir=payload/"student_calls")
        try:
            messages = build_messages(state["history"], {}, state["visible"])
            if digest(messages) != state["messages_sha256"] or student.messages(state["history"], {}, state["visible"]) != messages:
                raise ValueError("Probe prompt differs from exact evaluation/training messages")
            reply = student.generate(state["history"], {}, state["visible"])
            parsed, status = action_outcome(reply.raw_text, state["visible"]["admissible_commands"])
            result = {"state": state_evidence(state), "status": "ok", "raw_text": reply.raw_text,
                      "parsed_action": parsed, "action_outcome": status,
                      "agrees": parsed is not None and parsed == state["seed_action"], "metadata": reply.metadata}
            write_json(payload/"result.json", result, exclusive=True)
            write_json(attempt/"receipt.json", {"hashes": artifact_hashes(payload)}, exclusive=True)
            rows.append(completed_state(root, state))
        except Exception as exc:
            failure = {"status": "error", "error_type": type(exc).__name__, "agrees": None,
                       "state_id": state["state_id"]}
            write_json(attempt/"failure.json", failure, exclusive=True)
            rows.append({**state_evidence(state), **failure, "attempt_path": str(attempt.resolve())})
        finally:
            student.close()
    return rows


def run_probes(*, coverage_path, output, config, spec, student_factory=Student):
    if spec.get("enable_thinking") is not False or spec.get("max_tokens") != 1024:
        raise ValueError("Probes require canonical OFF/1024-token greedy student")
    coverage = validate_coverage(coverage_path, config)
    verified = coverage["training_collection"]["path"]
    _, population = gap_episodes(coverage["gap_path"], config)
    for row, _, _ in population:
        validate_continuation(config, spec, row["episode_path"], verified=verified)
    states, collection, inputs = load_training_states(verified, config)
    stable_spec = {k: v for k, v in spec.items() if k != "endpoint"}
    identity = {"student": stable_spec, "binding": coverage["student_binding"],
                "batch_invariant": config.get("server_provenance", {}).get("config", {}).get("server", {}).get("batch_invariant"),
                "states": [state_evidence(s) for s in states]}
    inputs = [coverage_path, *inputs]
    output = workspace(output, config=config, command="probes", identity=identity, inputs=inputs)
    with locked(output/".orchestrator.lock"):
        rows = query_states(states, output=output, config=config, spec=spec, inputs=inputs, student_factory=student_factory)
        result = {"schema_version": 1, "purpose": "diagnosis", "coverage_path": str(Path(coverage_path).resolve()),
                  "coverage_sha256": digest(coverage), "coverage_file_sha256": file_hash(coverage_path), "student": stable_spec,
                  "student_binding": coverage["student_binding"], "training_collection": collection,
                  "complete": all(r["status"] == "ok" for r in rows), "states": rows, **summarize(coverage, rows),
                  "absorption_rule": "Each non-navigation needed pattern: exact parsed action agreement >= 1/2 of ALL matching seed states. Episode absorbed iff all needed patterns pass; unknown calls stay unknown.",
                  "environment_steps": 0, "teacher_calls": 0, "training_exposure": 0}
        write_json(output/"probes.json", result)
    return result


def validate_probes(path, coverage, config):
    value = read_json(path)
    if (value["coverage_sha256"] != digest(coverage)
            or value["student_binding"] != coverage["student_binding"]
            or value["training_collection"] != coverage["training_collection"]
            or value["student"]["model"] != coverage["student_binding"]["model"]
            or value["student"]["revision"] != config["models"]["student"]["revision"]):
        raise ValueError("Probe/coverage evidence or student binding differs")
    states, _, _ = load_training_states(coverage["training_collection"]["path"], config)
    expected = {r["state_id"]: r for r in states}
    if len(value["states"]) != len(expected) or {r["state_id"] for r in value["states"]} != set(expected):
        raise ValueError("Probes must report every seed training state exactly once")
    for row in value["states"]:
        state = expected[row["state_id"]]
        if row["status"] == "ok":
            actual = completed_state(Path(row["receipt_path"]).parent.parent, state)
            if actual != row:
                raise ValueError("Probe summary differs from response evidence")
            if (row["metadata"]["returned_model_id"] != value["student"]["model"]
                    or row["metadata"]["student_revision"] != value["student"]["revision"]):
                raise ValueError("Probe response differs from configured student")
        elif row.get("agrees") is not None or any(row.get(k) != v for k, v in state_evidence(state).items()):
            raise ValueError("Unknown probe result cannot be scored")
    for key, expected_value in summarize(coverage, value["states"]).items():
        if value[key] != expected_value:
            raise ValueError("Probe summary scoring changed")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--output", required=True, help="Owned resumable analysis workspace")
    add_student_args(parser)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    spec = student_spec(config, args)
    result = run_probes(coverage_path=args.coverage, output=args.output, config=config, spec=spec)
    print(json.dumps({"output": args.output, "complete": result["complete"], "training_set_fit": result["training_set_fit"]}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
