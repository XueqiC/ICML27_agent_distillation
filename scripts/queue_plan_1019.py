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
T3 = "b1149a7"   # 10-19 evening: + 429 retry, fail-loud teacher transport, WKM upstream stops (fresh cells only)
EMB = ("/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
       "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
NB = f"{B}/reports/new_baselines"  # AgentDistill/CoA outputs must live outside reports/runs
TEACHER_GATE = {"pid_gone": COLLECTOR_PID}  # every teacher stage waits for the rai collection (shared API key)


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
    # extract (privileged) asks the teacher; data-0 (0% privileged) does not
    out = [stage(f"{cell}-c0", "scripts/hpg_leap.slurm", cpu, "04:00:00", 0, f"{e},LEAP_THROUGH={first}",
                 g if var == "privileged" else None)]
    for i in (1, 2, 3):
        out.append(stage(f"{cell}-g{i}", "scripts/hpg_leap.slurm", gpu, gh, 1, f"{e},LEAP_THROUGH=explore-{i}"))
        out.append(stage(f"{cell}-c{i}", "scripts/hpg_leap.slurm", cpu, "02:00:00", 0, f"{e},LEAP_THROUGH=data-{i}", g))
    out.append(stage(f"{cell}-g4", "scripts/hpg_leap.slurm", gpu, gh, 1, f"{e},LEAP_THROUGH=all"))
    out.append(stage(f"{cell}-eval", "scripts/hpg_leap.slurm", "--gpus=1 --cpus-per-task=4 --mem=24G", "03:00:00", 1,
                     f"{e},LEAP_MODE=evaluate"))
    names = [x["name"] for x in out]
    return {"cell": cell, "tree": tree, "stages": out[names.index(f"{cell}-{start}"):]}


def multi(name, res, time, gpus, wrapper, var, stages, extra, k=20, s=0, mode=None, account=None, gate=None):
    e = (f"WRAPPER={wrapper},STAGE_VAR={var},STAGES={stages},VERIFIED={verified(k, s)},SED_SECRETS_FILE={SECRETS},"
         f"WORKERS=16,BATCH_INVARIANT=1,{extra}")
    if mode:
        e += f",MODE_VAR={mode[0]},MODE={mode[1]}"
    d = stage(name, "scripts/hpg_multi_stage.slurm", res, time, gpus, e, gate)
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


C1 = "--partition=hpg-milan --gpus=0 --cpus-per-task=2 --mem=8G"
C4 = "--partition=hpg-milan --gpus=0 --cpus-per-task=4 --mem=16G"


def new_methods_k20s0():
    """Batch-2 first cells, fresh on T3 (stage sets and sizes as in scripts/ops_batch2_1019.sh)."""
    tg = TEACHER_GATE
    e = f"STECA_OUT={BL}/steca-k20s0,EMBEDDING_SNAPSHOT={EMB},NGPUS=4"
    steca = [multi("steca-k20s0-sft", G4, "04:00:00", 4, "scripts/hpg_steca.slurm", "STECA_THROUGH", "sft", e),
             multi("steca-k20s0-mc", G1, "12:00:00", 1, "scripts/hpg_steca.slurm", "STECA_THROUGH",
                   "expert-mc explore agent-mc", e),
             multi("steca-k20s0-c1", C1, "04:00:00", 0, "scripts/hpg_steca.slurm", "STECA_THROUGH",
                   "detect reflect datasets", e, gate=tg),
             multi("steca-k20s0-rl", G4, "06:00:00", 4, "scripts/hpg_steca.slurm", "STECA_THROUGH", "reinforced", e),
             multi("steca-k20s0-eval", G1, "06:00:00", 1, "scripts/hpg_steca.slurm", "STECA_THROUGH", "all", e,
                   mode=("STECA_MODE", "evaluate"))]
    e = f"WKM_OUT={BL}/wkm-k20s0,EMBEDDING_SNAPSHOT={EMB},NGPUS=1"
    wkm = [multi("wkm-k20s0-g1", G1, "08:00:00", 1, "scripts/hpg_wkm.slurm", "WKM_THROUGH",
                 "experienced explore synthesize", e),
           multi("wkm-k20s0-c1", C4, "02:00:00", 0, "scripts/hpg_wkm.slurm", "WKM_THROUGH", "merge knowledge-base", e),
           multi("wkm-k20s0-g2", G1, "08:00:00", 1, "scripts/hpg_wkm.slurm", "WKM_THROUGH", "world agent", e),
           multi("wkm-k20s0-eval", G1, "08:00:00", 1, "scripts/hpg_wkm.slurm", "WKM_THROUGH", "all", e,
                 mode=("WKM_MODE", "evaluate"))]
    e = f"GAD_OUT={BL}/gad-k20s0,NGPUS=4"
    gad = [multi("gad-k20s0-x0", G1, "03:00:00", 1, "scripts/hpg_gad.slurm", "GAD_THROUGH", "explore-0", e)]
    for r in (0, 1, 2):
        gad.append(multi(f"gad-k20s0-t{r}", C1, "04:00:00", 0, "scripts/hpg_gad.slurm", "GAD_THROUGH", f"teacher-{r}", e,
                         gate=tg))
        gad.append(multi(f"gad-k20s0-g{r}", G4, "08:00:00", 4, "scripts/hpg_gad.slurm", "GAD_THROUGH",
                         f"train-{r} explore-{r + 1}" if r < 2 else "train-2", e))
    gad.append(multi("gad-k20s0-eval", G1, "10:00:00", 1, "scripts/hpg_gad.slurm", "GAD_THROUGH", "train-2", e,
                     mode=("GAD_MODE", "evaluate")))
    e = f"AGENTDISTILL_OUT={NB}/agentdistill-k20s0"
    ad = [multi("agd-k20s0-c", C1, "08:00:00", 0, "scripts/hpg_agentdistill.slurm", "AGENTDISTILL_THROUGH",
                "collect abstract cluster consolidate publish", e, gate=tg),
          multi("agd-k20s0-eval", G1, "06:00:00", 1, "scripts/hpg_agentdistill.slurm", "AGENTDISTILL_THROUGH",
                "evaluate", e)]
    o = f"{NB}/coa-k20s0"
    e = f"COA_OUT={o},NGPUS=4"
    rl = " ".join(f"explore-{r} train-rl-{r}" for r in range(60))
    coa = [multi("coa-k20s0-c", C1, "12:00:00", 0, "scripts/hpg_coa.slurm", "COA_THROUGH", "teacher data", e, gate=tg),
           multi("coa-k20s0-g", G4, "48:00:00", 4, "scripts/hpg_coa.slurm", "COA_THROUGH", f"train-sft query-filter {rl}",
                 f"{e},STOP_JSON={o}/result.json"),
           multi("coa-k20s0-eval", G1, "12:00:00", 1, "scripts/hpg_coa.slurm", "COA_THROUGH", "data", e,
                 mode=("COA_MODE", "evaluate"))]
    coa_teval = [multi("coa-k20s0-teval", C1, "12:00:00", 0, "scripts/hpg_coa.slurm", "COA_THROUGH", "teacher", e,
                       mode=("COA_MODE", "teacher-evaluate"), gate=tg)]
    return [{"cell": n, "tree": T3, "stages": st} for n, st in (
        ("steca-k20s0", steca), ("wkm-k20s0", wkm), ("gad-k20s0", gad), ("agentdistill-k20s0", ad),
        ("coa-k20s0", coa), ("coa-k20s0-teval", coa_teval))]


# batch-1 LEAP cells whose correction stages lost teacher requests to the 10-19 429 contention (scan 20:5x CDT):
# restarted fresh on T3; the tainted run directories are moved to _trash first.
LEAP_RESTART = [("zero", k, s) for k, s in ((5, 0), (5, 1), (5, 2), (10, 0), (10, 1), (10, 2), (20, 1), (20, 2))] + \
               [("privileged", k, s) for k, s in ((5, 1), (5, 2), (10, 1), (10, 2), (20, 1), (20, 2))]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    # Plan v2 (10-19 21:0x CDT). LEAP is out: its teacher calls used a 60 s timeout with one HTTP attempt per method
    # slot, which censored correction latencies at 60 s (p50 48 s, 119 timeouts) -> every LEAP cell, s0 included,
    # reruns on a tree with upstream-faithful transport (SDK defaults: 600 s, retries). Methods that call the
    # teacher (STeCa, GAD, AgentDistill, CoA, HAD) wait for that tree too; teacher-free cells run now.
    cells = [sad_k20s0()] + [c for c in new_methods_k20s0() if c["cell"] == "wkm-k20s0"]
    cells += [agentgym(k, s, "x0") for k in (50, 100) for s in (1, 2)]
    plan = {"limits": {"max_jobs": 8, "max_gpus": 12, "max_group_gpus": 0}, "cells": cells}
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
