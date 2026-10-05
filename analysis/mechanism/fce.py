"""Teacher distance / closest approach, retaining the section-3 budget FCE."""
import argparse
import json
from pathlib import Path

from analysis.mechanism.branch import run_branch
from analysis.mechanism.cache import prepare_cache_seed, copy_cache_seed, seed_summary
from analysis.mechanism.common import (attempt_path, budget_directory, completed_branch, gap_episodes, read_json,
                                      load_episode, seal_branch, workspace)
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
    densification = max((2*sum(40-t for t in range(max(0, c-3), c+1) if t not in points)
                         for c in points), default=0)
    return {"coarse_probe_points": points, "teacher_samples_per_probe": 2,
            "refinement": "Every integer strictly inside the first coarse interval with T(left)=1,T(right)=0",
            "scan_policy_call_upper_bound": scan, "refinement_policy_call_upper_bound": refinement,
            "densification": "Every prefix in [max(0,t_c-3),t_c]; reuse already probed points, two samples each",
            "densification_policy_call_upper_bound": densification,
            "http_attempt_upper_bound_before_caps": (scan+refinement+densification)*max_attempts,
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


def locate_closest_approach(length, probe):
    """Amendment 3. Probe returns two successful turn counts (None = failure).

    Finish the unchanged coarse grid, retain the original budget-drop refinement,
    then densify around the earliest *coarse* minimum. Never resample a prefix.
    Unknown samples raise IncompleteProbe, never become negative observations.
    """
    distances, values = {}, {}
    result = {"status": "incomplete", "fce": None, "budget_fce": None,
              "budget_fce_status": "incomplete", "probes": values, "D_curve": distances,
              "t_star": None, "D_t_star": None, "D_0": None,
              "turns_remaining_at_t_star": None, "coarse_minimum_prefix": None,
              "dense_probe_points": []}
    def at(t):
        if t not in distances:
            samples = probe(t)
            if len(samples) != 2 or any(d is not None and (type(d) is not int or not 0 <= d <= 40-t)
                                        for d in samples):
                raise IncompleteProbe("Invalid teacher steps-to-go")
            successful = [d for d in samples if d is not None]
            distances[t] = min(successful) if successful else None
            values[t] = bool(successful)
        return distances[t]
    try:
        coarse = probe_points(length)
        for t in coarse:
            at(t)
        budget = locate_fce(length, lambda t: at(t) is not None)
        result.update(fce=budget["fce"], budget_fce=budget["fce"], budget_fce_status=budget["status"])
        for key in ("erroneous_step", "coarse_interval"):
            if key in budget:
                result[key] = budget[key]
        if budget["status"] == "incomplete":
            return result
        successful_coarse = [t for t in coarse if distances[t] is not None]
        if not successful_coarse:
            result["status"] = "teacher_unsolvable_from_student_states"
            return result
        tc = min(successful_coarse, key=lambda t: (distances[t], t))
        dense = list(range(max(0, tc-3), tc+1))
        result.update(coarse_minimum_prefix=tc, dense_probe_points=dense)
        for t in dense:
            at(t)
        t_star = min((t for t, d in distances.items() if d is not None), key=lambda t: (distances[t], t))
        result.update(status="closest_approach_found", t_star=t_star, D_t_star=distances[t_star],
                      turns_remaining_at_t_star=40-t_star)
    except IncompleteProbe:
        pass
    finally:
        result["D_0"] = distances.get(0)
    return result


def probe_episode(episode):
    """Stable executed trajectory; accounting/IDs remain in the full episode log."""
    fields = ("action_raw", "action_parsed", "action_outcome", "visible", "reward", "terminated",
              "truncated", "environment_steps", "audit_sha256")
    branch = episode["branch"]
    return {"schema_version": 1, "task_id": episode["task_id"], "parent_episode": branch["parent_episode"],
            "t": branch["t"], "sample_index": branch["sample_index"], "success": episode["success"],
            "initial": {k: episode["initial"][k] for k in fields},
            "steps": [{k: step[k] for k in fields} for step in episode["steps"]]}


def sample_distance(receipt, t, index):
    """Count policy turns after the replay, including parser/inadmissible turns."""
    distance = None
    if "episode_path" in receipt:
        episode = load_episode(receipt["episode_path"])
        branch = episode["branch"]
        if (branch["t"] != t or branch["sample_index"] != index or branch["arm"] != "takeover"
                or not branch["replay_matched"] or len(episode["steps"]) < t
                or episode["success"] is not receipt["success"]):
            raise ValueError("Teacher sample differs from its executed takeover")
        if receipt["success"] is True:
            distance = len(episode["steps"])-t
    return {"sample_index": index, "success": receipt["success"], "teacher_steps_to_go": distance}


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
        write_json(attempt/"episode/probe_episode.json", probe_episode(ep), exclusive=True)
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
    probe_samples = {}
    def probe(t):
        units = [(row, parent, task, t, index, root, output, config, budget, gap_path,
                  env_factory, teacher_factory, branch_fn) for index in range(2)]
        bound = (40-t)*config["models"]["teacher"]["max_attempts"]
        costs = [(parent["episode_id"], 0 if completed_branch(root/f"probes/t{t:02d}/sample{i}")
                  else bound) for i in range(2)]
        samples = run_units(_probe_sample, units, workers=sample_workers,
                            budget=budget, costs=costs, executor=pool)
        probe_samples[t] = [sample_distance(r, t, i) for i, r in enumerate(samples)]
        if any(type(r["success"]) is not bool for r in samples):
            raise IncompleteProbe("Probe has unknown outcome")
        return [s["teacher_steps_to_go"] for s in probe_samples[t]]
    with process_pool(sample_workers) as pool:
        result = locate_closest_approach(len(parent["steps"]), probe)
    # Retain observed distances even when the other sample was capped/unknown.
    for t, samples in probe_samples.items():
        successful = [s["teacher_steps_to_go"] for s in samples if s["success"] is True]
        result["D_curve"][t] = min(successful) if successful else None
    result["D_0"] = result["D_curve"].get(0)
    result.update(task_id=task["task_id"], parent_episode=parent["episode_id"],
                  parent_path=row["episode_path"], episode_length=len(parent["steps"]),
                  taxonomy_primary=row["primary"], taxonomy_labels=row["labels"], probe_samples=probe_samples)
    return result


def run_fce(*, gap_path, output, config, max_episode, max_run, budget_dir=None, workers=1,
            seed_teacher_cache_from=None, env_factory=_environment, teacher_factory=make_teacher, branch_fn=run_branch):
    validate_workers(workers)
    _, population = gap_episodes(gap_path, config)
    seed = prepare_cache_seed(seed_teacher_cache_from, output) if seed_teacher_cache_from else None
    identity = {"gap_path": str(Path(gap_path).resolve()), "max_episode": max_episode, "max_run": max_run,
                "budget_dir": str(Path(budget_dir).resolve()) if budget_dir else None,
                "teacher_cache_seed": seed_summary(seed)}
    output = workspace(output, config=config, command="fce", identity=identity, inputs=[gap_path],
                       details={"teacher_cache_seed": seed_summary(seed)})
    budget = TeacherBudget(budget_directory(budget_dir, output)/"budget.json", max_episode=max_episode, max_run=max_run)
    with locked(output / ".orchestrator.lock"):
        if seed:
            copy_cache_seed(seed, output)
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
    parser.add_argument("--seed-teacher-cache-from", help="Read-only old FCE workspace or launcher root")
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
                "seed_teacher_cache_from": args.seed_teacher_cache_from,
                "max_teacher_calls_run": args.max_teacher_calls_run, "episodes": [
                    {"task_id": ep["task_id"], **dry_plan(len(ep["steps"]), config["models"]["teacher"]["max_attempts"])}
                    for _, ep, _ in population]}
        print(json.dumps(plan, indent=2))
        return 0
    result = run_fce(gap_path=args.gap_set, output=args.output, config=config,
                     max_episode=args.max_teacher_calls, max_run=args.max_teacher_calls_run,
                     budget_dir=args.budget_dir, workers=args.workers,
                     seed_teacher_cache_from=args.seed_teacher_cache_from)
    print(json.dumps({"output": args.output, "complete": result["complete"], "budget_used": result["budget"]["total"]}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
