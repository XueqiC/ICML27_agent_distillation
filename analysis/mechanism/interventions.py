"""Run I0-I5 at frozen FCE states: greedy plus four seeded T=0.7 continuations."""
import argparse
import json
from pathlib import Path

from analysis.mechanism.branch import run_branch
from analysis.mechanism.common import (attempt_path, budget_directory, completed_branch, gap_episodes, read_json,
                                      seal_branch, workspace)
from analysis.mechanism.examples import load_seeds, select_example
from analysis.mechanism.policies import (BranchStudent, TeacherBudget, add_budget_args,
                                      add_student_args, make_teacher, student_spec, validate_continuation)
from sed.config import load_config
from sed.react import build_messages, action_outcome
from sed.runner import _environment
from sed.seed_evidence import verified_collection_inputs
from sed.manifest import load_frozen, verify_task_files
from sed.storage import digest, locked, write_json
from teachers.openai_teacher import Teacher, TeacherResult

ARMS = tuple(f"I{i}" for i in range(6))


class SharedInterventionTeacher(Teacher):
    """Freeze one I3/I4 output across the arm's five student continuations.

    The original output and billed usage stay in the existing teacher ledger.
    Reuse has zero new billed usage and does not pretend to be a fresh sample.
    """
    def __init__(self, teacher, path, advice):
        self.teacher, self.path, self.advice = teacher, Path(path), advice
        self.spec = teacher.spec

    def generate(self, history, task, current):
        key = digest({"messages": build_messages(history, task, current), "advice": self.advice})
        if self.path.exists():
            row = read_json(self.path)
            if row["input_sha256"] != key or digest(row["result"]) != row["result_sha256"]:
                raise ValueError("Shared intervention input/output changed")
            saved = row["result"]
            meta = dict(saved["metadata"], shared_intervention_reuse=True,
                        billed_usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            return TeacherResult(saved["raw_text"], saved["parsed_action"], meta)
        result = self.teacher.generate(history, task, current)
        if self.advice and (not result.raw_text.strip() or action_outcome(result.raw_text, current["admissible_commands"])[0] is not None):
            raise ValueError("Rejected I3 advice; raw response retained in teacher ledger")
        saved = result._asdict()
        write_json(self.path, {"input_sha256": key, "result": saved, "result_sha256": digest(saved)}, exclusive=True)
        return result


def arm_outcome(samples):
    by_index = {s["sample_index"]: s for s in samples}
    if (len(samples) != 5 or set(by_index) != set(range(5))
            or any(type(s.get("sample_index")) is not int or type(s.get("success")) is not bool for s in samples)):
        return {"complete": False, "rescue": None, "sample_successes": None, "greedy_success": None}
    count = sum(by_index[i]["success"] for i in range(1, 5))
    return {"complete": True, "rescue": by_index[0]["success"] and count >= 3,
            "sample_successes": count, "greedy_success": by_index[0]["success"]}


def run_interventions(*, gap_path, fce_path, verified, output, config, spec, max_episode, max_run,
                      budget_dir=None, sampling_seed=0, subject="student", env_factory=_environment,
                      teacher_factory=make_teacher, student_factory=BranchStudent, branch_fn=run_branch,
                      seeds=None):
    _, population = gap_episodes(gap_path, config)
    fce = read_json(fce_path)
    if Path(fce["gap_path"]).resolve() != Path(gap_path).resolve():
        raise ValueError("FCE and intervention gap populations differ")
    fce_rows = {r["task_id"]: r for r in fce["episodes"]}
    if len(fce_rows) != len(fce["episodes"]):
        raise ValueError("Duplicate FCE task")
    seed_inputs = []
    if seeds is None:
        for row, _, _ in population:
            validate_continuation(config, spec, row["episode_path"], subject=subject, verified=verified)
        seeds = load_seeds(verified, config)
        manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        seed_inputs = verified_collection_inputs(verified, manifest, freeze)
    stable_spec = {k: v for k, v in spec.items() if k != "endpoint"}
    # Checkpoint hashes, not a new server UUID/port, bind resumed continuations.
    adapter = config.get("lora_equivalence", {}).get("identity", {}).get("adapter")
    identity = {"student": stable_spec, "adapter": adapter, "subject": subject,
                "batch_invariant": config.get("server_provenance", {}).get("config", {}).get("server", {}).get("batch_invariant"),
                "max_episode": max_episode, "max_run": max_run, "sampling_seed": sampling_seed,
                "budget_dir": str(Path(budget_dir).resolve()) if budget_dir else None,
                "seeds": [{k: v for k, v in s.items() if k != "episode"} for s in seeds]}
    output = workspace(output, config=config, command="interventions", identity=identity,
                       inputs=[gap_path, fce_path, *seed_inputs])
    budget = TeacherBudget(budget_directory(budget_dir, output)/"budget.json", max_episode=max_episode, max_run=max_run)
    results = []
    with locked(output / ".orchestrator.lock"):
        for row, parent, task in population:
            verify_task_files(config["environment"]["data_root"], task)
            located = fce_rows.get(task["task_id"], {})
            if located.get("status") != "fce_found":
                results.append({"task_id": task["task_id"], "status": "no_fce", "fce_status": located.get("status")})
                continue
            if located.get("parent_episode") != parent["episode_id"]:
                raise ValueError("FCE parent episode differs")
            t = located["fce"]
            root = output / "tasks" / digest(task["task_id"])[:16]
            example = select_example(task["task_id"], seeds)
            entry = {"task_id": task["task_id"], "t": t, "parent_episode": parent["episode_id"],
                     "subject": subject, "status": "ok", "non_deterministic": False,
                     "taxonomy_primary": row["primary"], "taxonomy_labels": row["labels"],
                     "task_type_covered": example is not None,
                     "needed_step_coverage": "unknown", "arms": {}}
            for arm in ARMS:
                if arm == "I5" and example is None:
                    entry["arms"][arm] = {"status": "no_same_type_seed", "rescue": None, "samples": []}
                    continue
                samples = []
                for index in range(5):
                    name = f"{arm}/sample{index}"
                    receipt = completed_branch(root/name)
                    if receipt is None:
                        attempt = attempt_path(root, name)
                        attempt.mkdir(parents=True, exist_ok=False)
                        seed = int(digest([sampling_seed, task["task_id"], index])[:8], 16)
                        student = student_factory(**spec, log_dir=attempt/"episode/student_calls",
                                                  sample_index=index, sampling_seed=seed)
                        teacher = None
                        if arm in {"I3", "I4"}:
                            teacher = SharedInterventionTeacher(teacher_factory(config, output, budget, parent["episode_id"],
                                namespace=digest([parent["episode_id"], t, arm, subject]), sample_index=0, advice=arm == "I3"),
                                root/arm/"shared_teacher_output.json", arm == "I3")
                        try:
                            ep = branch_fn(env_factory(config, task), student, task, config, parent, t,
                                attempt/"episode", arm=arm, sample_index=index, sampling_seed=seed,
                                teacher=teacher, example=example, parent_path=row["episode_path"],
                                inputs=[gap_path, fce_path, "docs/MECHANISM_PLAN.md", *seed_inputs],
                                replay_check=subject == "student")
                            receipt = seal_branch(attempt, ep)
                        except Exception as exc:
                            write_json(attempt/"failure.json", {"error_type": type(exc).__name__,
                                "arm": arm, "sample_index": index, "success": "unknown"}, exclusive=True)
                            samples.append({"sample_index": index, "success": "unknown", "status": "error"})
                            break
                    samples.append({"sample_index": index, **receipt})
                    if arm == "I0" and index == 0 and subject == "student":
                        entry["non_deterministic"] = receipt["branch"].get("non_deterministic", False)
                summary = arm_outcome(samples)
                entry["arms"][arm] = {**summary, "samples": samples, "status": "ok" if summary["complete"] else "incomplete"}
                # Amendment 2: I0 divergence is audit metadata, not an exclusion.
                # Incomplete controls remain unknown in the paired readout.
            results.append(entry)
            root.mkdir(parents=True, exist_ok=True)
            write_json(root/"result.json", entry)
            write_json(output/"interventions.json", {"gap_path": str(Path(gap_path).resolve()),
                "fce_path": str(Path(fce_path).resolve()), "subject": subject, "student": stable_spec,
                "episodes": results, "budget": budget.read(),
                "shared_teacher_output": "one I3/I4 output per arm, reused across the five student continuations"})
        write_json(output/"interventions.json", {"gap_path": str(Path(gap_path).resolve()),
            "fce_path": str(Path(fce_path).resolve()), "subject": subject, "student": stable_spec,
            "episodes": results, "budget": budget.read(),
            "shared_teacher_output": "one I3/I4 output per arm, reused across the five student continuations"})
    return read_json(output/"interventions.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--gap-set", required=True)
    parser.add_argument("--fce", required=True)
    parser.add_argument("--verified", required=True, help="Exactly the seed collection used to train S")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", choices=["student", "base"], default="student")
    parser.add_argument("--sampling-seed", type=int, default=0)
    add_student_args(parser)
    add_budget_args(parser)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    spec = student_spec(config, args)
    result = run_interventions(gap_path=args.gap_set, fce_path=args.fce, verified=args.verified,
        output=args.output, config=config, spec=spec, max_episode=args.max_teacher_calls,
        max_run=args.max_teacher_calls_run, budget_dir=args.budget_dir,
        subject=args.subject, sampling_seed=args.sampling_seed)
    incomplete = sum(e.get("status") == "incomplete_replay_check" or any(
        a["status"] == "incomplete" for a in e.get("arms", {}).values()) for e in result["episodes"])
    print(json.dumps({"output": args.output, "incomplete_episodes": incomplete, "budget_used": result["budget"]["total"]}))
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
