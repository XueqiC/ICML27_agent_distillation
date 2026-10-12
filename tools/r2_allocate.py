"""R2 visitation clusters, finite-pool water filling, and shared query selections.

sigma_s^2 is mean_states mean_student_labels ||grad[-log p(label | state)]||^2
at the restored student checkpoint. Use full reply token sums, unclipped and
uncentered, over trainable parameters. With temperature-1 student labels this
is an unbiased estimate of the student's Fisher trace at each sampled state;
teacher/greedy labels, token means and clipped gradients do not estimate it.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from data.manifest.build_r03_sets import guarded_task
from sed.config import load_config
from sed.manifest import TASK_TYPES, load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import checked_episode, load_verified_collection
from sed.storage import digest, write_json
from tools.collect_corrections import select_r1_states, state_ref
from tools.rollout_pool import read_rollouts

BUCKETS = ("0-4", "5-9", "10-19", "20+")
ADAM_WEIGHTS = ("raw_sigma", "scale_free")


def cluster_id(task_type, step):
    if task_type not in TASK_TYPES or type(step) is not int or step < 0:
        raise ValueError("Require a known task type and nonnegative policy-turn index")
    return task_type + "::" + BUCKETS[0 if step < 5 else 1 if step < 10 else 2 if step < 20 else 3]


def largest_remainder(values, budget, capacities=None):
    keys = sorted(values)
    capacities = capacities or {s: budget for s in keys}
    result = {s: min(capacities[s], int(math.floor(max(0, values[s])))) for s in keys}
    left = budget - sum(result.values())
    if left < 0:
        raise ValueError("Continuous allocation exceeds budget")
    rank = sorted(keys, key=lambda s: (-(values[s] - math.floor(max(0, values[s]))), s))
    for s in rank:
        if left and result[s] < capacities[s]:
            result[s] += 1
            left -= 1
    if left:
        raise ValueError("Largest remainder cannot meet budget/capacity")
    return result


def water_fill(weights, coverage, budget, capacities=None):
    """min sum W/(k+n), subject to 0<=n<=available and sum n=budget.

    No capacities means the preregistered unbounded formula. Finite capacities
    prevent querying nonexistent states; clipping/saturation is disclosed.
    """
    keys = sorted(weights)
    if set(keys) != set(coverage) or not keys or type(budget) is not int or budget < 0:
        raise ValueError("Invalid water-filling inputs")
    if any(not math.isfinite(v) or v < 0 for v in [*weights.values(), *coverage.values()]):
        raise ValueError("Weights/coverage must be finite and nonnegative")
    caps = dict(capacities) if capacities is not None else {s: budget for s in keys}
    if (set(caps) != set(keys) or any(type(c) is not int or c < 0 for c in caps.values())
            or budget > sum(caps.values())):
        raise ValueError("Budget exceeds available unique states or capacities are invalid")
    positive = [s for s in keys if weights[s] > 0]
    available_positive = (sum(caps[s] for s in positive) if capacities is not None
                          else math.inf if positive else 0)
    if budget == 0:
        values, lam = {s: 0.0 for s in keys}, None
    elif available_positive <= budget:
        values = {s: float(caps[s]) if weights[s] > 0 else 0.0 for s in keys}
        left = budget - available_positive
        zero_capacity = sum(caps[s] for s in keys if weights[s] == 0)
        if left:
            for s in keys:
                if weights[s] == 0:
                    values[s] = left * caps[s] / zero_capacity
        lam = 0.0
    else:
        def at(lam):
            return {s: min(caps[s] if capacities is not None else math.inf,
                           max(0.0, math.sqrt(weights[s] / lam) - coverage[s]))
                    if weights[s] else 0.0 for s in keys}
        lo, hi = 0.0, max(weights.values())
        while sum(at(hi).values()) > budget:
            hi *= 2
        for _ in range(180):
            mid = (lo + hi) / 2
            if sum(at(mid).values()) > budget:
                lo = mid
            else:
                hi = mid
        lam, values = hi, at(hi)
    integers = largest_remainder(values, budget, caps)
    return {"continuous": values, "integer": integers, "lambda": lam,
            "saturated_clusters": [s for s in keys if capacities is not None and caps[s]
                                   and abs(values[s] - caps[s]) < 1e-7],
            "zero_weight_tie_rule": "capacity-proportional when all positive-weight capacity is exhausted"}


def predicted_error(weights, coverage, allocation):
    total = 0.0
    for s, weight in weights.items():
        denom = coverage[s] + allocation.get(s, 0)
        if weight and denom == 0:
            return None  # JSON null plus explicit infinite flag; never nonstandard Infinity.
        if weight:
            total += weight / denom
    return total


def cluster_statistics(records, demonstrations):
    states, visits, coverage = defaultdict(list), Counter(), Counter()
    seen = set()
    for record in records:
        for state in record["states"]:
            cid = cluster_id(record["task_type"], state["step"])
            visits[cid] += 1
            if state["state_hash"] not in seen:
                states[cid].append(state)
                seen.add(state["state_hash"])
    if not visits:
        raise ValueError("No student-visited states")
    for task_type, episode in demonstrations:
        for i in range(len(episode["steps"])):
            coverage[cluster_id(task_type, i)] += 1
    total = sum(visits.values())
    return {s: {"visits": visits[s], "d": visits[s] / total, "k": coverage[s],
                "unique_states": len(states[s])} for s in sorted(visits)}, dict(states)


def read_sigma(value, clusters):
    """Read grad_stats' state/label-mean, unclipped action-NLL second moment."""
    checkpoint = value.get("checkpoint")
    if value.get("schema_version") != 1 or not isinstance(checkpoint, str):
        raise ValueError("Sigma file requires schema_version=1 and a p10 checkpoint")
    if checkpoint != "p10":
        # grad_stats writes its actual checkpoint path. Validate its saved state
        # rather than renaming or trusting an arbitrary path's spelling.
        path = Path(checkpoint) / "training_state.json"
        state = json.loads(path.read_text()) if path.is_file() else {}
        if state.get("updates") != 115 or state.get("supervised_tokens") != 104070:
            raise ValueError("Sigma checkpoint is not the preregistered p10 state")
    result = {}
    for cid in clusters:
        row = value.get("clusters", {}).get(cid)
        if (not isinstance(row, dict) or type(row.get("sigma2")) not in (int, float)
                or not math.isfinite(row["sigma2"]) or row["sigma2"] < 0
                or type(row.get("sample_count")) is not int or row["sample_count"] < 1):
            raise ValueError(f"Missing/invalid sigma2 or sample_count for {cid}")
        result[cid] = float(row["sigma2"])
    return result


def allocate(records, demonstrations, sigma, *, budgets=(100, 400), seed=0, r1_states=(),
             adam_weight="raw_sigma"):
    if adam_weight not in ADAM_WEIGHTS:
        raise ValueError("adam_weight must be raw_sigma or scale_free")
    stats, clusters = cluster_statistics(records, demonstrations)
    sigma2 = read_sigma(sigma, stats)
    coverage = {s: row["k"] for s, row in stats.items()}
    caps = {s: row["unique_states"] for s, row in stats.items()}
    # Proposition 6.11(b) uses variance after coordinate-wise RMS normalization.
    # Cluster-level raw moments cannot estimate it; scale_free holds it constant.
    adam_weights = {
        "raw_sigma": {s: row["d"] * sigma2[s] for s, row in stats.items()},
        "scale_free": {s: row["d"] for s, row in stats.items()},
    }
    adam = adam_weights[adam_weight]
    sgd = {s: row["d"] ** 2 * sigma2[s] for s, row in stats.items()}
    for s in stats:
        stats[s].update(sigma2=sigma2[s], W_Adam=adam[s], W_SGD=sgd[s])
    # A1 is simple uniform sampling without replacement. Every arm uses these
    # SAME priorities; within-cluster prefixes attain maximal pairwise overlap.
    priority = lambda s: (digest([seed, "r03-query-crn", s["state_hash"]]), s["state_hash"])
    ranked = {s: sorted(rows, key=priority) for s, rows in clusters.items()}
    all_states = sorted([s for rows in ranked.values() for s in rows], key=priority)
    state_cluster = {s["state_hash"]: cid for cid, rows in ranked.items() for s in rows}
    arms, union = {}, {}
    def arm(name, budget, allocation, selected, **extra):
        errors = {}
        for definition, weights in adam_weights.items():
            error = predicted_error(weights, coverage, allocation)
            errors[definition] = {"predicted_error": error,
                                  "predicted_error_infinite": error is None}
        # Preserve the original scalar and ordering under raw_sigma, regardless
        # of A3's allocation choice; report both objectives with explicit labels.
        arms[name] = {"budget": budget, "allocation": allocation,
            "states": [state_ref(s) for s in selected], **errors["raw_sigma"],
            "predicted_error_weight": "raw_sigma", "predicted_errors": errors, **extra}
        for s in selected:
            union[s["state_hash"]] = state_ref(s)
    arm("A0", 0, {s: 0 for s in stats}, [])
    for n in budgets:
        if type(n) is not int or n < 0 or n > len(all_states):
            raise ValueError("Query budget exceeds unique visited states; no replacement/fabrication")
        selected = all_states[:n]
        counts = Counter(state_cluster[s["state_hash"]] for s in selected)
        arm(f"A1_n{n}", n, {s: counts[s] for s in stats}, selected)
        for name, weights in (("A2", sgd), ("A3", adam)):
            solution = water_fill(weights, coverage, n, caps)
            selected = [state for s in sorted(stats) for state in ranked[s][:solution["integer"][s]]]
            arm(f"{name}_n{n}", n, solution["integer"], selected, water_filling=solution)
    r1 = {s["state_hash"]: dict(s) for s in r1_states}
    combined = {**r1, **union}
    return {"schema_version": 1, "checkpoint": "p10", "seed": seed,
        "adam_weight": adam_weight,
        "step_unit": "zero-based policy turn, including parser feedback turns",
        "clusters": stats, "arms": arms, "union_states": [union[s] for s in sorted(union)],
        "union_query_count": len(union), "r1_query_count": len(r1),
        "r1_r2_overlap": len(r1.keys() & union.keys()), "r1_r2_union_query_count": len(combined),
        "r2_additional_after_r1": len(union.keys() - r1.keys()),
        "r1_r2_union_states": [combined[s] for s in sorted(combined)],
        "predicted_error_weight": "raw_sigma",
        "predicted_order_low_error_first": sorted(arms, key=lambda k: (
            arms[k]["predicted_error"] if arms[k]["predicted_error"] is not None else math.inf, k)),
        "sampling": "uniform hash priorities shared by every arm/budget; per-cluster prefixes",
        "finite_pool_adaptation": "clip continuous water-filling at unique available states, then largest remainder",
        "weight_normalization": "d sums to one; W_Adam="
            + ("d*sigma2" if adam_weight == "raw_sigma" else "d")
            + "; W_SGD=d*d*sigma2 (no additional scaling)",
        "adam_weight_definitions": {"raw_sigma": "W_Adam=d*sigma2",
            "scale_free": "W_Adam=d; coordinate-RMS-normalized gradient variance treated as "
                          "constant across clusters (set to 1)"}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--pool", default="data/manifest/r03_pool200.json")
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--k20", default="data/verified/seeds_k20_s0")
    p.add_argument("--sigma", help="Omit only with --statistics-only")
    p.add_argument("--statistics-only", action="store_true")
    p.add_argument("--budgets", nargs="+", type=int, choices=(100, 400), default=[100, 400])
    p.add_argument("--adam-weight", choices=ADAM_WEIGHTS, default="raw_sigma",
                   help="A3 only: raw_sigma uses d*sigma2 (default); scale_free uses d")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, help="Directory containing allocation.json and R1 selection")
    args = p.parse_args(argv)
    if not args.statistics_only and not args.sigma:
        p.error("Allocation requires --sigma")
    config = load_config(args.config)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    pool = json.loads(Path(args.pool).read_text())
    records = read_rollouts(args.rollouts, manifest, freeze, args.pool, subset="R2")
    rows = [r for r, _ in records]
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Use one p10 rollout per pool game")
    collection, selection, evidence = load_verified_collection(args.k20, manifest, freeze)
    if collection["K"] != 20 or selection["seed"] != 0:
        raise ValueError("Coverage requires K20 seed 0")
    demos = [(guarded_task(manifest, freeze, r["task_id"])["task_type"], checked_episode(evidence, r))
             for r in collection["verified"]]
    r1 = select_r1_states(rows, pool["subsets"]["R1_P2"])
    config["r2_allocation"] = {"seed": args.seed, "budgets": args.budgets,
                              "statistics_only": args.statistics_only,
                              "adam_weight": args.adam_weight}
    write_provenance(args.output, command="r03_r2_allocate", config=config,
        inputs=[args.config, args.pool, *args.rollouts, str(Path(args.k20) / "manifest.json"),
                config["paths"]["manifest"], config["paths"]["split_freeze"],
                *([args.sigma] if args.sigma else [])])
    write_json(Path(args.output) / "r1_selection.json", r1, exclusive=True)
    stats, clusters = cluster_statistics(rows, demos)
    statistics = {"schema_version": 1, "checkpoint": "p10", "clusters": stats,
        "adam_weight": args.adam_weight,
        "gradient_sample_states": {s: [state_ref(t) for t in sorted(ts,
            key=lambda t: digest([args.seed, "gradient-probe", t["state_hash"]]))[:10]]
            for s, ts in clusters.items()}}
    write_json(Path(args.output) / "clusters.json", statistics, exclusive=True)
    if args.statistics_only:
        print(json.dumps({"clusters": len(stats), "gradient_samples": sum(
            map(len, statistics["gradient_sample_states"].values()))}))
        return
    result = allocate(rows, demos, json.loads(Path(args.sigma).read_text()),
                      budgets=tuple(args.budgets), seed=args.seed, r1_states=r1["states"],
                      adam_weight=args.adam_weight)
    result["rollout_game_count"] = len(rows)
    result["missing_r2_game_ids"] = [t for t in pool["subsets"]["R2"] if t not in {r["task_id"] for r in rows}]
    write_json(Path(args.output) / "allocation.json", result, exclusive=True)
    for arm, value in result["arms"].items():
        write_json(Path(args.output) / (arm + ".json"), value, exclusive=True)
    print(json.dumps({k: result[k] for k in ("union_query_count", "r1_query_count", "r1_r2_overlap",
                                           "r1_r2_union_query_count", "r2_additional_after_r1")}, indent=2))


if __name__ == "__main__":
    main()
