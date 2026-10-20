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
T4 = "2348ce9"   # + upstream-faithful baseline teacher transport (600 s, 3 HTTP attempts) + shared-history DPO/IPR
T5 = "70ed412"   # + shared-history SAD (default), truncated-output semantics, SmartAD
T6 = "071c72f"   # + sharded SmartAD candidate collection (SmartAD pools are collected on rai with this tree)
# 10-20: vLLM served Gemma-4 k_eq_v layers without the k_proj LoRA delta on V (served model != trained model).
# Serving fix 5554db8 cherry-picked onto each tree; evaluation stages run on these, training keeps its tree
# (SAD/SmartAD training re-derive code-hashed identities, evaluation does not).
EVAL_FIX = {"2348ce9": "8960e0a", "70ed412": "3b82db4", "071c72f": "0411901"}
LORA_GATE = {"file": "/home/xueqi/hq/projects/ICML27_agent_distillation/logs/queue/gates/lora_kveq_fixed"}
EMB = ("/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
       "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
NB = f"{B}/reports/new_baselines"  # AgentDistill/CoA outputs must live outside reports/runs
# Every teacher stage waits for this file. Create it by hand only once the OpenAI key has credits (out since 10-19
# 17:55 CDT) and the API budget across rai and hpg is settled.
TEACHER_GATE = {"file": "/home/xueqi/hq/projects/ICML27_agent_distillation/logs/queue/gates/teacher_ok"}


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
    g = TEACHER_GATE if gate else None
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


def sad(k, s):
    """SAD cell on T5: one-hot KL = CE on the trusted seeds, shared-history packing (same 655M logical positions).
    GPU acceptance 10-20 (sad_shared_history_gpu_r2.json): packing-only differences are at the kernel-swap bf16 level.
    Train memory from sstat of the T4 per-turn run (MaxRSS 95G on 4 GPUs) -> 144G; per-turn needed ~31 h, shared
    history processes each history once, so 16 h is a generous limit."""
    cell = f"sadsh-k{k}s{s}"
    e = f"SAD_OUT={BL}/{cell},NGPUS=4"
    C4 = "--partition=hpg-milan --gpus=0 --cpus-per-task=4 --mem=16G"
    G4S = "--partition=hpg-b200 --gpus=4 --cpus-per-task=8 --mem=144G"
    return {"cell": cell, "tree": T5, "stages": [
        multi(f"{cell}-prep", C4, "02:00:00", 0, "scripts/hpg_sad.slurm", "SAD_THROUGH", "prepare", e, k=k, s=s),
        multi(f"{cell}-train", G4S, "16:00:00", 4, "scripts/hpg_sad.slurm", "SAD_THROUGH", "train", e, k=k, s=s),
        multi(f"{cell}-eval", G1, "10:00:00", 1, "scripts/hpg_sad.slurm", "SAD_THROUGH", "all", e, k=k, s=s,
              mode=("SAD_MODE", "evaluate"))]}


ADFTP_SRC = f"{B}/reports/adftp-collections"   # FTP collections, collected on rai (T4 code), rsynced here


def adftp(k, s):
    """AD-FTP training + evaluation on T4 from the K100 FTP collection of support seed s (K5-K50 nested views).
    Waits for gates/adftp_k100_s{s}_synced (created after the rai collection is complete and rsynced).
    Sizes from sacct of Kang cells (same LoRA recipe): train 5-28 min at 32-96G, SAG/greedy eval ~15 min at 48G."""
    cell = f"adftp-k{k}s{s}"
    src = f"{ADFTP_SRC}/k100_s{s}/collection" + ("" if k == 100 else f"/subsets/k{k}")
    e = f"ADFTP_OUT={BL}/{cell},VERIFIED={src},WORKERS=16,BATCH_INVARIANT=1"
    gate = {"file": f"/home/xueqi/hq/projects/ICML27_agent_distillation/logs/queue/gates/adftp_k100_s{s}_synced"}
    one = "--partition=hpg-b200 --gpus=1 --cpus-per-task=8 --mem=64G"
    return {"cell": cell, "tree": T4, "stages": [
        stage(f"{cell}-train", "scripts/hpg_adftp.slurm", one, "02:00:00", 1, f"{e},ADFTP_THROUGH=train", gate),
        stage(f"{cell}-eval", "scripts/hpg_adftp.slurm", one, "03:00:00", 1,
              f"{e},ADFTP_THROUGH=train,ADFTP_MODE=evaluate")]}


SMARTAD_SEL = f"{B}/reports/smartad-selection"   # SmartAD selected collections (selected on rai, T6), rsynced here


def smartad(k, s):
    """SmartAD training + greedy evaluation on T6 from the selected K100 pool of support seed s (K5-K50 nested views).
    The candidate pool (10 x T=1.0 per task) is collected on rai; selection (min macro per-turn NLL under the initial
    student) also runs on rai because pool receipts hold rai paths; the selected collection is self-contained.
    Waits for gates/smartad_k100_s{s}_synced. Sizes as AD-FTP / Kang (same LoRA trainer)."""
    cell = f"smartad-k{k}s{s}"
    src = f"{SMARTAD_SEL}/k100_s{s}/collection" + ("" if k == 100 else f"/subsets/k{k}")
    e = f"SMARTAD_OUT={BL}/{cell},VERIFIED={src},WORKERS=16,BATCH_INVARIANT=1"
    gate = {"file": f"/home/xueqi/hq/projects/ICML27_agent_distillation/logs/queue/gates/smartad_k100_s{s}_synced"}
    one = "--partition=hpg-b200 --gpus=1 --cpus-per-task=8 --mem=64G"
    return {"cell": cell, "tree": T6, "stages": [
        stage(f"{cell}-train", "scripts/hpg_smartad.slurm", one, "02:00:00", 1, f"{e},SMARTAD_THROUGH=train", gate),
        stage(f"{cell}-eval", "scripts/hpg_smartad.slurm", one, "03:00:00", 1, f"{e},SMARTAD_THROUGH=evaluate")]}


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
    return [{"cell": n, "tree": T3 if n == "wkm-k20s0" else T4, "stages": st} for n, st in (
        ("steca-k20s0", steca), ("wkm-k20s0", wkm), ("gad-k20s0", gad), ("agentdistill-k20s0", ad),
        ("coa-k20s0", coa), ("coa-k20s0-teval", coa_teval))]


def preference(method, k, s):
    """ETO (3 iterations) / IPR (6) on T4 with shared-history scoring; sizes from sacct of IPR r5 (10-10):
    4-GPU SFT 25G peak, 1-GPU explore/MC 15G, 4-GPU DPO ~210G. Explore+MC+pairs share one 1-GPU job per iteration."""
    up = method.upper()
    cell = f"{method}-k{k}s{s}"
    e = f"{up}_OUT={BL}/{cell},VERIFIED={verified(k, s)},NGPUS=4,WORKERS=16,BATCH_INVARIANT=1"
    sft = "--partition=hpg-b200 --gpus=4 --cpus-per-task=8 --mem=96G"
    dpo = "--partition=hpg-b200 --gpus=4 --cpus-per-task=8 --mem=256G"
    one = "--partition=hpg-b200 --gpus=1 --cpus-per-task=6 --mem=32G"
    script = f"scripts/hpg_{method}.slurm"
    out = [stage(f"{cell}-sft", script, sft, "03:00:00", 4, f"{e},{up}_THROUGH=sft")]
    for i in range(1, (3 if method == "eto" else 6) + 1):
        out.append(stage(f"{cell}-x{i}", script, one, "10:00:00", 1, f"{e},{up}_THROUGH=pairs-{i}"))
        out.append(stage(f"{cell}-dpo{i}", script, dpo, "06:00:00", 4, f"{e},{up}_THROUGH=dpo-{i}"))
    out.append(stage(f"{cell}-eval", script, one, "10:00:00", 1, f"{e},{up}_MODE=evaluate"))
    return {"cell": cell, "tree": T4, "stages": out}


K_CELLS = [(k, s) for k in (5, 10, 20, 50, 100) for s in (0, 1, 2)]


# batch-1 LEAP cells whose correction stages lost teacher requests to the 10-19 429 contention (scan 20:5x CDT):
# restarted fresh on T3; the tainted run directories are moved to _trash first.
LEAP_RESTART = [("zero", k, s) for k, s in ((5, 0), (5, 1), (5, 2), (10, 0), (10, 1), (10, 2), (20, 1), (20, 2))] + \
               [("privileged", k, s) for k, s in ((5, 1), (5, 2), (10, 1), (10, 2), (20, 1), (20, 2))]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    # Plan v4 (10-19 23:5x CDT, user priority): core agent-distillation methods first, every K x s0-s2:
    # AD/Kang (K grid already complete), AD-FTP (collection running on rai; training cells added when the FTP
    # collections land), SAD (15 cells here), SmartAD (to implement). Everything else is on hold until tomorrow;
    # running jobs of held cells are still tracked.
    # sadsh-k20s0 first: SAD trains are the long pole (hours on 4 GPUs), AD-FTP stages take 5-30 min.
    cells = [sad(20, 0)] + [adftp(k, s) for s in (0, 1, 2) for k in (5, 10, 20, 50, 100)]
    # SAD: T4 per-turn scoring needed ~31 h per cell (sad-k20s0 reached the 1/8 checkpoint in 3.9 h; cancelled
    # 10-20 04:49 CDT after 1/8, its updates.jsonl is the per-turn reference curve). All 15 cells run on T5 with
    # shared-history packing (sadsh-*), k20s0 first (top of the plan) so its curve can be compared with the per-turn reference.
    # SmartAD cells wait for their selected pools (gates); ahead of the remaining SAD cells because they are short.
    cells += [smartad(k, s) for s in (0, 1, 2) for k in (5, 10, 20, 50, 100)]
    cells += [sad(k, s) for k, s in K_CELLS if (k, s) != (20, 0)]
    # Low-priority order: stages that fit the dept QOS leftovers first (dept is memory-bound, ~80G free):
    # CPU teacher stages and 1-GPU/32-48G stages (WKM, LEAP, AgentDistill/CoA collection), then 4-GPU/160-256G work.
    held = [c for c in new_methods_k20s0() if c["cell"] in ("wkm-k20s0", "agentdistill-k20s0", "coa-k20s0",
                                                             "coa-k20s0-teval")]
    held += [leap(v, 20, 0, "c0", tree=T4) for v in ("zero", "privileged")]
    held += [leap(v, k, s, "c0", tree=T4) for v in ("zero", "privileged") for k, s in K_CELLS if (k, s) != (20, 0)]
    held += [c for c in new_methods_k20s0() if c["cell"] in ("steca-k20s0", "gad-k20s0")]
    held += [preference("eto", 20, 0), preference("ipr", 20, 0)]
    held += [agentgym(k, s, "x0") for k in (50, 100) for s in (1, 2)]
    # User 10-20 00:40 CDT: parallelize as far as the rules allow -> the other baselines run as low priority in at
    # most 4 jobs / 8 GPUs, leaving room for the core cells as they become ready.
    for c in held:
        c["lowprio"] = True
    cells += held
    # Eval stages of trees with a serving fix use it; all other eval stages wait until their tree has one.
    for c in cells:
        for st in c["stages"]:
            if "eval" in st["name"] and not st.get("gate"):
                if c["tree"] in EVAL_FIX and not c.get("lowprio"):
                    st["tree"] = EVAL_FIX[c["tree"]]
                else:
                    st["gate"] = LORA_GATE
    # max_gpus counts pending jobs too: 20 lets 1-GPU evals pass while 4-GPU SAD trains wait on the dept QOS.
    plan = {"limits": {"max_jobs": 10, "max_gpus": 20, "max_group_gpus": 6, "max_lowprio_jobs": 3,
                       "max_lowprio_gpus": 8}, "cells": cells}
    names = [c["cell"] for c in cells]
    assert len(names) == len(set(names)), "duplicate cell"
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(cells)} cells ({len(held)} low priority), {sum(len(c['stages']) for c in cells)} stages -> {out / 'plan.json'}")


if __name__ == "__main__":
    main()
