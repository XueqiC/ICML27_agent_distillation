#!/usr/bin/env python3
"""Build plan.json for scripts/hpg_queue_keeper.py from the batch-1/batch-2 stage definitions (2026-10-19).

Stage resources, walltimes and exports are the ones scripts/ops_batch1_1019.sh and scripts/ops_batch2_1019.sh used
(sized from sacct of the same stages); only the submission changes from pre-submitted afterok chains to one job per
cell at a time. Cells already started keep the tree they started on, because every run binds code_sha256 in its
identity. Usage: queue_plan_1019.py OUT_DIR  (writes OUT_DIR/plan.json and, the first time, the seed state.json).
"""
import json
import sys
from pathlib import Path

B = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation"
BL = f"{B}/reports/runs/baselines"
SECRETS = "/blue/fsu-compsci-dept/xc25.fsu/hq/secrets/openai.env"
COLLECTOR_PID = 228787  # rai full-TRAIN teacher collection; old trees have no 429 retry, so their teacher stages wait
T1 = "625f6a9"   # batch 1
T2 = "08d3c47"   # batch 2


def verified(k, s):
    if k in (20, 100):
        return f"{B}/data/verified/seeds_k{k}_s{s}"
    if k in (5, 10):
        return f"{B}/data/verified/seeds_k20_s{s}/subsets/k{k}"
    return f"{B}/data/verified/seeds_k100_s{s}/subsets/k50"


def stage(name, script, res, time, gpus, exports, gate=None):
    d = {"name": name, "script": script, "res": res, "time": time, "gpus": gpus, "exports": exports}
    if gate:
        d["gate"] = gate
    return d


def agentgym(k, s, start):
    cell = f"ag-k{k}s{s}"
    e = f"VERIFIED={verified(k, s)},AGENTGYM_OUT={BL}/agentgym-k{k}s{s},NGPUS=2,WORKERS=16,BATCH_INVARIANT=1"
    th = {50: "02:30:00", 100: "03:30:00"}.get(k, "02:00:00")
    out = []
    for m in range(4):
        out.append(stage(f"{cell}-t{m}", "scripts/hpg_agentgym.slurm", "--gpus=2 --cpus-per-task=4 --mem=160G", th, 2,
                         f"{e},AGENTGYM_THROUGH=train-{m}"))
        out.append(stage(f"{cell}-x{m}", "scripts/hpg_agentgym.slurm", "--gpus=1 --cpus-per-task=4 --mem=24G",
                         "01:30:00", 1, f"{e},AGENTGYM_THROUGH=filter-{m}"))
    out.append(stage(f"{cell}-eval", "scripts/hpg_agentgym.slurm", "--gpus=1 --cpus-per-task=4 --mem=24G",
                     "04:00:00", 1, f"{e},AGENTGYM_MODE=evaluate"))
    names = [x["name"] for x in out]
    return {"cell": cell, "tree": T1, "stages": out[names.index(f"{cell}-{start}"):]}


def leap(var, k, s, start, tree=T1, gate=True):
    cell = f"lp{var[0]}-k{k}s{s}"
    e = (f"LEAP_OUT={BL}/leap-{var}-k{k}s{s},LEAP_VARIANT={var},VERIFIED={verified(k, s)},"
         f"SED_SECRETS_FILE={SECRETS},WORKERS=16,BATCH_INVARIANT=1")
    cpu = "--partition=hpg-milan --gpus=0 --cpus-per-task=2 --mem=8G"
    gpu = "--gpus=1 --cpus-per-task=4 --mem=32G"
    gh = {5: "01:30:00", 10: "01:30:00", 20: "02:30:00", 50: "07:00:00", 100: "12:00:00"}[k]
    g = {"pid_gone": COLLECTOR_PID} if gate else None
    first = "extract" if var == "privileged" else "data-0"
    out = [stage(f"{cell}-c0", "scripts/hpg_leap.slurm", cpu, "04:00:00", 0, f"{e},LEAP_THROUGH={first}", g)]
    for i in (1, 2, 3):
        out.append(stage(f"{cell}-g{i}", "scripts/hpg_leap.slurm", gpu, gh, 1, f"{e},LEAP_THROUGH=explore-{i}"))
        out.append(stage(f"{cell}-c{i}", "scripts/hpg_leap.slurm", cpu, "02:00:00", 0, f"{e},LEAP_THROUGH=data-{i}", g))
    out.append(stage(f"{cell}-g4", "scripts/hpg_leap.slurm", gpu, gh, 1, f"{e},LEAP_THROUGH=all"))
    out.append(stage(f"{cell}-eval", "scripts/hpg_leap.slurm", "--gpus=1 --cpus-per-task=4 --mem=24G", "03:00:00", 1,
                     f"{e},LEAP_MODE=evaluate"))
    names = [x["name"] for x in out]
    return {"cell": cell, "tree": tree, "stages": out[names.index(f"{cell}-{start}"):]}


def multi(name, res, time, gpus, wrapper, var, stages, extra, k=20, s=0, mode=None, account=None):
    e = (f"WRAPPER={wrapper},STAGE_VAR={var},STAGES={stages},VERIFIED={verified(k, s)},SED_SECRETS_FILE={SECRETS},"
         f"WORKERS=16,BATCH_INVARIANT=1,{extra}")
    if mode:
        e += f",MODE_VAR={mode[0]},MODE={mode[1]}"
    d = stage(name, "scripts/hpg_multi_stage.slurm", res, time, gpus, e)
    if account:
        d["account"] = account
    return d


G4 = "--partition=hpg-b200 --gpus=4 --cpus-per-task=8 --mem=256G"
G1 = "--partition=hpg-b200 --gpus=1 --cpus-per-task=4 --mem=48G"


def sad_k20s0():
    e = f"SAD_OUT={BL}/sad-k20s0,NGPUS=4"
    return {"cell": "sad-k20s0", "tree": T2, "stages": [
        multi("b2-sad-k20s0-train", G4, "24:00:00", 4, "scripts/hpg_sad.slurm", "SAD_THROUGH", "train", e, account="dept"),
        multi("b2-sad-k20s0-eval", G1, "10:00:00", 1, "scripts/hpg_sad.slurm", "SAD_THROUGH", "all", e,
              mode=("SAD_MODE", "evaluate"))]}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cells = [sad_k20s0()]
    cells += [agentgym(k, s, "x0") for k in (50, 100) for s in (1, 2)]
    cells += [leap("privileged", 100, s, "g1") for s in (1, 2)] + [leap("zero", 100, s, "g1") for s in (0, 1, 2)]
    cells += [leap("privileged", 50, s, "c1") for s in (1, 2)] + [leap("zero", 50, s, "c1") for s in (0, 1, 2)]
    plan = {"limits": {"max_jobs": 8, "max_gpus": 16, "max_group_gpus": 0}, "cells": cells}
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    state_path = out / "state.json"
    if not state_path.exists():  # jobs that were RUNNING when the queue was cleared (10-19 20:3x CDT)
        running = {"lpp-k100s1": "46706560", "lpp-k100s2": "46706569",
                   "lpz-k100s0": "46706677", "lpz-k100s1": "46706686", "lpz-k100s2": "46706695"}
        state = {c: {"done": [], "active": {"name": f"{c}-g1", "id": j, "submitted": "batch1"}, "blocked": None}
                 for c, j in running.items()}
        state_path.write_text(json.dumps(state, indent=1))
    print(f"{len(cells)} cells, {sum(len(c['stages']) for c in cells)} stages -> {out / 'plan.json'}")


if __name__ == "__main__":
    main()
