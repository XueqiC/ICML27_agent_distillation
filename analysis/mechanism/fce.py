"""Two-sample teacher takeover scan and first-drop refinement (plan section 3)."""
import argparse
import json
from pathlib import Path

from analysis.mechanism.branch import run_branch
from analysis.mechanism.common import (attempt_path, budget_directory, completed_branch, gap_episodes, read_json,
                                      seal_branch, workspace)
from analysis.mechanism.policies import TeacherBudget, add_budget_args, make_teacher
from analysis.mechanism.parallel import process_pool, run_units, validate_workers
from sed.config import load_config
from sed.manifest import verify_task_files
from sed.runner import _environment
from sed.storage import digest, locked, write_json


class IncompleteProbe(RuntimeError):
    pass


def probe_points(length):
    if not 0 <= length <= 40:
        raise ValueError("Episode length must be within the fixed 40-turn cap")
    return sorted(set([*range(0, length+1, 4), length]))


def dry_plan(length, max_attempts=10):
    points = probe_points(length)
    scan = 2*sum(40-t for t in points)
    refinement = max((2*sum(40-t for t in range(a+1, b)) for a, b in zip(points, points[1:])), default=0)
    return {"coarse_probe_points": points, "teacher_samples_per_probe": 2,
            "refinement": "Every integer strictly inside the first coarse interval with T(left)=1,T(right)=0",
            "scan_policy_call_upper_bound": scan, "refinement_policy_call_upper_bound": refinement,
            "http_attempt_upper_bound_before_caps": (scan+refinement)*max_attempts,
            "estimate_uses": "Worst-case remaining turns, including parent steps; early success uses fewer calls"}


def locate_fce(length, probe):
    """Pure scan; probe returns bool or raises IncompleteProbe (never coerced to 0)."""
    values = {}
    def at(t):
        if t not in values:
            value = probe(t)
            if type(value) is not bool:
                raise IncompleteProbe("Unknown takeover result")
            values[t] = value
        return values[t]
    result = {"fce": None, "probes": values}
    try:
        if not at(0):
            return dict(result, status="teacher_unsolvable_from_start")
        previous = 0
        for right in probe_points(length)[1:]:
            if at(previous) and not at(right):
                left = previous
                for t in range(previous+1, right+1):
                    if at(left) and not at(t):
                        return dict(result, status="fce_found", fce=left, erroneous_step=left+1,
                                    coarse_interval=[previous, right])
                    left = t
            previous = right
        return dict(result, status="no_drop_detected")
    except IncompleteProbe:
        return dict(result, status="incomplete")


def _probe_sample(unit):
    row, parent, task, t, index, root, output, config, budget, gap_path, env_factory, teacher_factory, branch_fn = unit
    name = f"probes/t{t:02d}/sample{index}"
    receipt = completed_branch(root / name)
    if receipt is not None:
        return receipt
    attempt = attempt_path(root, name)
    attempt.mkdir(parents=True, exist_ok=False)
    try:
        teacher = teacher_factory(config, output, budget, parent["episode_id"],
            namespace=digest([parent["episode_id"], t, "takeover"]), sample_index=index)
        ep = branch_fn(env_factory(config, task), teacher, task, config, parent, t,
            attempt/"episode", arm="takeover", sample_index=index,
            parent_path=row["episode_path"], inputs=[gap_path, "docs/MECHANISM_PLAN.md"])
        return seal_branch(attempt, ep)
    except Exception as exc:
        incomplete = getattr(exc, "incomplete", False)
        write_json(attempt/("incomplete.json" if incomplete else "failure.json"),
            {"error_type": type(exc).__name__, "t": t, "sample_index": index,
             "status": "incomplete" if incomplete else "error", "success": "unknown"}, exclusive=True)
        return {"success": "unknown"}


def _gap_episode(unit):
    row, parent, task, output, config, budget, gap_path, workers, env_factory, teacher_factory, branch_fn = unit
    verify_task_files(config["environment"]["data_root"], task)
    root = output / "tasks" / digest(task["task_id"])[:16]
    root.mkdir(parents=True, exist_ok=True)
    sample_workers = 1 if workers == 1 else 2
    def probe(t):
        units = [(row, parent, task, t, index, root, output, config, budget, gap_path,
                  env_factory, teacher_factory, branch_fn) for index in range(2)]
        bound = (40-t)*config["models"]["teacher"]["max_attempts"]
        costs = [(parent["episode_id"], 0 if completed_branch(root/f"probes/t{t:02d}/sample{i}")
                  else bound) for i in range(2)]
        samples = run_units(_probe_sample, units, workers=sample_workers,
                            budget=budget, costs=costs, executor=pool)
        if any(type(r["success"]) is not bool for r in samples):
            raise IncompleteProbe("Probe has unknown outcome")
        return any(r["success"] for r in samples)
    with process_pool(sample_workers) as pool:
        result = locate_fce(len(parent["steps"]), probe)
    result.update(task_id=task["task_id"], parent_episode=parent["episode_id"],
                  parent_path=row["episode_path"], episode_length=len(parent["steps"]),
                  taxonomy_primary=row["primary"], taxonomy_labels=row["labels"])
    return result


def run_fce(*, gap_path, output, config, max_episode, max_run, budget_dir=None, workers=1,
            env_factory=_environment, teacher_factory=make_teacher, branch_fn=run_branch):
    validate_workers(workers)
    _, population = gap_episodes(gap_path, config)
    identity = {"gap_path": str(Path(gap_path).resolve()), "max_episode": max_episode, "max_run": max_run,
                "budget_dir": str(Path(budget_dir).resolve()) if budget_dir else None}
    output = workspace(output, config=config, command="fce", identity=identity, inputs=[gap_path])
    budget = TeacherBudget(budget_directory(budget_dir, output)/"budget.json", max_episode=max_episode, max_run=max_run)
    with locked(output / ".orchestrator.lock"):
        if (output/"fce.json").exists():
            saved = read_json(output/"fce.json")
            if saved.get("complete"):
                for result in saved["episodes"]:
                    root = output/"tasks"/digest(result["task_id"])[:16]
                    for t in result["probes"]:
                        for index in range(2):
                            if completed_branch(root/f"probes/t{int(t):02d}/sample{index}") is None:
                                raise ValueError("Completed FCE summary lacks its probe receipts")
                return saved
        state = budget.read()
        units = [(row, parent, task, output, config, budget, gap_path, workers,
                  env_factory, teacher_factory, branch_fn) for row, parent, task in population]
        costs = [(parent["episode_id"], max(0, max_episode-state["episodes"].get(parent["episode_id"], 0)))
                 for _, parent, _ in population]
        results = run_units(_gap_episode, units, workers=workers, budget=budget, costs=costs)
        # Only publish summaries after all workers finish, in frozen population order.
        for result in results:
            write_json(output/"tasks"/digest(result["task_id"])[:16]/"result.json", result)
        write_json(output/"fce.json", {"gap_path": str(Path(gap_path).resolve()),
            "gap_size": len(population), "episodes": results, "budget": budget.snapshot(),
            "complete": all(r["status"] != "incomplete" for r in results)})
    return read_json(output/"fce.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--gap-set", required=True, help="taxonomy.json with teacher comparison")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent gap episodes (two samples per probe)")
    add_budget_args(parser)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if min(args.max_teacher_calls, args.max_teacher_calls_run) <= 0:
        parser.error("Teacher caps must be positive")
    config = load_config(args.config)
    if args.dry_run:
        _, population = gap_episodes(args.gap_set, config)
        plan = {"dry_run": True, "teacher_calls": 0, "max_teacher_calls_per_episode": args.max_teacher_calls,
                "max_teacher_calls_run": args.max_teacher_calls_run, "episodes": [
                    {"task_id": ep["task_id"], **dry_plan(len(ep["steps"]), config["models"]["teacher"]["max_attempts"])}
                    for _, ep, _ in population]}
        print(json.dumps(plan, indent=2))
        return 0
    result = run_fce(gap_path=args.gap_set, output=args.output, config=config,
                     max_episode=args.max_teacher_calls, max_run=args.max_teacher_calls_run,
                     budget_dir=args.budget_dir, workers=args.workers)
    print(json.dumps({"output": args.output, "complete": result["complete"], "budget_used": result["budget"]["total"]}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
