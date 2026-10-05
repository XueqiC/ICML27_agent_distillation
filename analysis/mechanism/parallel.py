"""Ordered, budget-aware process dispatch. Worker count is execution-only."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
import multiprocessing


def validate_workers(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")


def process_pool(workers):
    return (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            if workers > 1 else nullcontext(None))


def run_units(fn, units, *, workers, budget, costs, dependencies=None, executor=None):
    """Return input order, allowing overlap only with sufficient budget headroom.

    costs are (parent episode, conservative maximum HTTP attempts). These are
    scheduling claims, not charged reservations: only the existing limiter
    charges actual attempts. Near a cap the next costly unit runs alone, so
    admission has the same outcome as serial execution, including retries.
    Dependencies keep shared-output producers ahead of their consumers.
    """
    validate_workers(workers)
    dependencies = dependencies or {}
    if workers == 1:
        return [fn(unit) for unit in units]
    results = [None] * len(units)
    pending, finished, active = list(range(len(units))), set(), {}
    with (nullcontext(executor) if executor is not None else process_pool(workers)) as pool:
        while pending or active:
            state = budget.read()
            claims = [costs[i] for i in active.values()]
            blocked_cost = False
            for i in list(pending):
                if len(active) == workers:
                    break
                episode, bound = costs[i]
                if not set(dependencies.get(i, ())) <= finished:
                    blocked_cost |= bound > 0
                    continue
                if bound:
                    if blocked_cost:
                        continue
                    run_claim = sum(n for _, n in claims)
                    episode_claim = sum(n for ep, n in claims if ep == episode)
                    fits = (state["total"] + run_claim + bound <= budget.limits["max_run"]
                            and state["episodes"].get(episode, 0) + episode_claim + bound
                            <= budget.limits["max_episode"])
                    if run_claim and not fits:
                        blocked_cost = True
                        continue
                active[pool.submit(fn, units[i])] = i
                claims.append((episode, bound))
                pending.remove(i)
            if not active:
                raise ValueError("Unresolvable continuation dependencies")
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                i = active.pop(future)
                results[i] = future.result()
                finished.add(i)
    return results
