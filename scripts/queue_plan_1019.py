#!/usr/bin/env python3
"""Build plan.json for scripts/hpg_queue_keeper.py from the batch-1/batch-2 stage definitions (2026-10-19).

Stage resources, walltimes and exports are the ones scripts/ops_batch1_1019.sh and scripts/ops_batch2_1019.sh used
(sized from sacct of the same stages); only the submission changes from pre-submitted afterok chains to one job per
cell at a time. Cells already started keep the tree they started on, because every run binds code_sha256 in its
identity. Usage: queue_plan_1019.py OUT_DIR  (writes OUT_DIR/plan.json and, the first time, the seed state.json).
"""
import json
import re
import sys
from pathlib import Path

B = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation"
BL = f"{B}/reports/runs/baselines"
# 10-22 20:3x CDT the 40G hpg $HOME filled up (29G vLLM compile cache); alfworld.info then failed with ENOSPC creating
# ~/.cache/alfworld at import, and every episode of three evals ended in reset_error. All stages keep caches on /blue.
OFF_HOME = (f"ALFWORLD_DATA={B}/../tools/alfworld-data-empty,VLLM_CACHE_ROOT={B}/../tools/vllm-cache,"
            f"TRITON_CACHE_DIR={B}/../tools/triton-cache,VLLM_NO_USAGE_STATS=1")
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


# 10-21 06:1x CDT: --nodes=1 everywhere: multi-GPU stages were split across 2 nodes (torchrun --standalone -> invalid device ordinal).
G4 = "--partition=hpg-b200 --nodes=1 --gpus=4 --cpus-per-task=8 --mem=256G"
G1 = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=4 --mem=48G"


def sad(k, s):
    """SAD cell on T5: one-hot KL = CE on the trusted seeds, shared-history packing (same 655M logical positions).
    GPU acceptance 10-20 (sad_shared_history_gpu_r2.json): packing-only differences are at the kernel-swap bf16 level.
    10-20 15:xx: 2 GPUs instead of 4. Shared-history SAD runs one trajectory per rank slot and pads idle ranks with
    zero-weight duplicates (schedule.rank_slots), so with b' = min(64, max(1, N // 8)) = 1 (K5, K10) or 2 (K20) a
    4-GPU job had 2-3 padded ranks: 2 GPUs give the same global batch, the same per-update wall time and the same
    gradients (padded slots contribute exact zeros). K50/K100 (b' = 6/12) take 3/6 instead of 2/3 rounds per update at
    equal GPU-hours. 2 is the minimum: full-parameter fp32 masters + AdamW state (~192G) need 2 B200 shards. The
    4-GPU requests sat in Priority/QOSGrpGRES for 10 h. Walltime from the per-turn reference (95.3M physical tokens in
    4.5 h on 4 GPUs) and the K20 s0 shared/per-turn physical ratio 13.6: ~9 h for one active rank (K5/K10), ~4.5 h
    for two (K20-K100); limits 20 h / 12 h until the first cell's measured rate. gpus is part of identity.json, so the
    4-GPU prep-only outputs moved to _superseded/sadsh-g4-prep-1020 and prep reruns (30-45 s CPU)."""
    cell = f"sadsh-k{k}s{s}"
    # Host memory: every rank loads the full float32 model on CPU before sharding (~50G each), so the load peak is
    # ~50G x ranks. 144G (from a post-load sstat MaxRSS of 95G) OOM-killed sadsh-k20s0-train 46826441 on 4 GPUs at 87%
    # of weight loading; the T4 per-turn run survived with 256G. 2 ranks -> 160G (same as the 2-GPU DPO jobs).
    n = 2
    e = f"SAD_OUT={BL}/{cell},NGPUS={n}"
    C4 = "--partition=hpg-milan --gpus=0 --cpus-per-task=4 --mem=16G"
    G2S = f"--partition=hpg-b200 --nodes=1 --gpus={n} --cpus-per-task=8 --mem=160G"
    return {"cell": cell, "tree": T5, "stages": [
        multi(f"{cell}-prep", C4, "02:00:00", 0, "scripts/hpg_sad.slurm", "SAD_THROUGH", "prepare", e, k=k, s=s),
        # Measured 10-20 (sadsh-k5s0-train 46893249, 2 B200): 14.4M of 655M logical positions in ~6 min of training
        # -> ~4.5-5 h per K5 cell; K<=10 limit 20 h -> 10 h (2x margin) so the pending jobs can backfill.
        multi(f"{cell}-train", G2S, "10:00:00" if k <= 10 else "12:00:00", n,
              "scripts/hpg_sad.slurm", "SAD_THROUGH",
              "train", e, k=k, s=s),
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
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=8 --mem=64G"
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
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=8 --mem=64G"
    return {"cell": cell, "tree": T6, "stages": [
        # K100 = 50 updates at ~2.5 min each (46869920 timed out at 2 h after 44); K<=50 finish in <= 1.3 h.
        stage(f"{cell}-train", "scripts/hpg_smartad.slurm", one, "04:00:00" if k == 100 else "02:00:00", 1,
              f"{e},SMARTAD_THROUGH=train", gate),
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
    sft = "--partition=hpg-b200 --nodes=1 --gpus=4 --cpus-per-task=8 --mem=96G"
    dpo = "--partition=hpg-b200 --nodes=1 --gpus=4 --cpus-per-task=8 --mem=256G"
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=6 --mem=32G"
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


def smartadsel(s):
    """SmartAD K100 selection on hpg (rai: ~36 s/candidate on one 48 GB card). The pool was collected on rai and copied
    to the same relative layout; --path-map relocates its absolute evidence paths (hash-checked, disclosed)."""
    cell = f"smartadsel-k100s{s}"
    e = (f"SMARTAD_OUT={SMARTAD_SEL}/k100_s{s},VERIFIED={B}/reports/runs/smartad-collections/k100_s{s},"
         f"SMARTAD_THROUGH=select,SMARTAD_PATH_MAP=/home/xueqi/hq/projects/ICML27_agent_distillation={B},"
         f"WORKERS=16,BATCH_INVARIANT=1")
    gate = {"file": f"/home/xueqi/hq/projects/ICML27_agent_distillation/logs/queue/gates/smartad_pool_k100_s{s}_on_hpg"}
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=8 --mem=64G"
    return {"cell": cell, "tree": "e7e4311", "stages": [stage(f"{cell}-select", "scripts/hpg_smartad.slurm", one,
                                                              "08:00:00", 1, e, gate)]}


def kangfix(k, s):
    """Re-evaluate the finished Kang/AD cells (SAG, as before) with the k_eq_v serving fix, on the AD-FTP eval tree so
    every core method shares one eval harness. New OUT; the pre-fix kang-<cell>-tp-sag results stay in place."""
    cell = f"kangfix-k{k}s{s}"
    tree = EVAL_FIX[T4]
    verified = f"{B}/data/verified/" + (f"seeds_k{k}_s{s}" if k in (20, 100) else
                                        f"seeds_k20_s{s}/subsets/k{k}" if k in (5, 10) else f"seeds_k100_s{s}/subsets/k{k}")
    adapter = f"{BL}/kang-k{k}s{s}-tp/stages/train-sft/attempt-000/output/checkpoint"
    e = (f"ADAPTERS=kang-k{k}s{s}-tp={adapter},TASKS=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/{tree}/data/manifest/dev140.json,DECODING=sag,"
         f"INCLUDE_BASE=0,OUT={B}/reports/runs/eval/baselines/kang-k{k}s{s}-tp-sag-kveq,VERIFIED={verified},"
         f"WORKERS=16,BATCH_INVARIANT=1")
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=8 --mem=48G"   # sacct: kang-tp-sag 15 min, MaxRSS 9.7G
    return {"cell": cell, "tree": tree, "stages": [stage(f"{cell}-eval", "scripts/hpg_eval_adapters.slurm", one,
                                                         "03:00:00", 1, e)]}



def kangchk(k, s, tree_train):
    """User 10-20 14:30 CDT: Kang K50 -> K100 drop must be rechecked. The drop is K100 s0 alone (94 -> 62; s1 94 -> 94,
    s2 104 -> 106): normal training curve, teacher data without loops, but 19.7% inadmissible actions at evaluation.
    Two replications separate evaluation noise from training-run variance: (1) the same checkpoint re-evaluated
    (SAG, new OUT); (2) an identical retrain on the tree whose code hashes match the cell's identity (def44a9, 176/176
    files), into a new output tree, then the same SAG evaluation."""
    v = f"{B}/data/verified/seeds_k{k}_s{s}"
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=8 --mem=48G"
    tasks = f"/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/{EVAL_FIX[T4]}/data/manifest/dev140.json"
    def ev(name, model, adapter, out):
        return stage(name, "scripts/hpg_eval_adapters.slurm", one, "03:00:00", 1,
                     f"ADAPTERS={model}={adapter},TASKS={tasks},DECODING=sag,INCLUDE_BASE=0,OUT={out},VERIFIED={v},"
                     f"WORKERS=16,BATCH_INVARIANT=1")
    old = f"kang-k{k}s{s}-tp"
    re_eval = {"cell": f"kangchk-k{k}s{s}-reeval", "tree": EVAL_FIX[T4], "stages": [
        ev(f"kangchk-k{k}s{s}-reeval-eval", old, f"{BL}/{old}/stages/train-sft/attempt-000/output/checkpoint",
           f"{B}/reports/runs/eval/baselines/{old}-sag-kveq-r2")]}
    new = f"kang-k{k}s{s}-tp-r2"
    train = stage(f"kangchk-k{k}s{s}-retrain-train", "scripts/hpg_kang.slurm", one, "02:00:00", 1,
                  f"KANG_OUT={BL}/{new},VERIFIED={v},KANG_MODE=train,WORKERS=16,BATCH_INVARIANT=1")
    evs = ev(f"kangchk-k{k}s{s}-retrain-eval", new, f"{BL}/{new}/stages/train-sft/attempt-000/output/checkpoint",
             f"{B}/reports/runs/eval/baselines/{new}-sag-kveq")
    evs["tree"] = EVAL_FIX[T4]
    retrain = {"cell": f"kangchk-k{k}s{s}-retrain", "tree": tree_train, "stages": [train, evs]}
    return [re_eval, retrain]


# Last checkpoint of each Seed-SFT run (max of 40 passes and 400k tokens = the "end of schedule" entry of the main
# table), read from reports/runs/train/seedsft-<cell>-hpg on 10-20; pre-fix greedy scores 72..115.
SEEDSFT_LAST = {"k5s0": 400172, "k5s1": 400125, "k5s2": 400446, "k10s0": 400091, "k10s1": 400127, "k10s2": 400059,
                "k20s0": 416280, "k20s1": 407400, "k20s2": 400351, "k50s0": 915840, "k50s1": 1014840,
                "k50s2": 981480, "k100s0": 1933720, "k100s1": 2110160, "k100s2": 1969160}


def seedsftfix(k, s):
    """User 10-20 15:08 CDT: re-evaluate Seed-SFT (unmerged LoRA r16 incl. k_proj, served before the k_eq_v fix) after
    SAD. Same decoding as the original sweep (greedy, batch-invariant dev140), on the AD-FTP/Kang eval tree, new OUT;
    the pre-fix sweep results stay in place."""
    cell = f"seedsftfix-k{k}s{s}"
    tree = EVAL_FIX[T4]
    verified = f"{B}/data/verified/" + (f"seeds_k{k}_s{s}" if k in (20, 100) else
                                        f"seeds_k20_s{s}/subsets/k{k}" if k in (5, 10) else f"seeds_k100_s{s}/subsets/k{k}")
    ckpt = f"{B}/reports/runs/train/seedsft-k{k}s{s}-hpg/checkpoint-tokens-{SEEDSFT_LAST[f'k{k}s{s}']}"
    e = (f"ADAPTERS=seedsft-k{k}s{s}-last={ckpt},TASKS=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/{tree}/data/manifest/"
         f"dev140.json,DECODING=greedy,INCLUDE_BASE=0,OUT={B}/reports/runs/eval/seedsft-kveq/k{k}s{s},VERIFIED={verified},"
         f"WORKERS=16,BATCH_INVARIANT=1")
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=8 --mem=48G"   # as kangfix (15 min, MaxRSS 9.7G)
    return {"cell": cell, "tree": tree, "stages": [stage(f"{cell}-eval", "scripts/hpg_eval_adapters.slurm", one,
                                                         "03:00:00", 1, e)]}


# ---- 10-20 new main grid: nested stratified views of the full-TRAIN collection, seed 0, converged training ----
CHAIN_TREE = "20cb697"   # main: views + convergence overrides + chain collections + shared-history/multi-GPU SFT + k_eq_v fix
CHAIN_GATE = {"file": "/home/xueqi/hq/projects/ICML27_agent_distillation/logs/queue/gates/chain_views_on_hpg"}
GATES = "/home/xueqi/hq/projects/ICML27_agent_distillation/logs/queue/gates"


def chain_view(k):
    return f"{B}/data/verified/seeds_full_train_r2/subsets/k{k}"


def sadc(k, mult, lr=None):
    """SAD on the chain view K with budget x mult (rungs x0.5 and x1 = paper first; x2 only if not converged).
    b' = min(64, N // 8) = 3 / 12 / 37 / 62 for K30/100/300/500: K>=300 fill 4 GPUs without padding (half the wall
    time at equal GPU-hours); K30/K100 stay on 2. Host memory: each rank loads the fp32 model on CPU (~50G):
    160G for 2 ranks, 256G for 4 (the per-turn 4-GPU run's working value). Walltime from sadsh-k5s0 (14.4M logical
    positions in ~6 min on 2 B200, i.e. ~4.5-5 h per 655M on 2 GPUs). Evaluation of a rung is gated: only the reported
    (first converged) rung is evaluated."""
    tag = {0.25: "025", 0.5: "05", 1: "1", 2: "2", 4: "4"}[mult]
    cell = f"sadc-k{k}-b{tag}" + (f"-lr{lr}" if lr else "")   # 10-23: lr suffix = halved-lr rerun (tree 7cbc62b, SAD_LR)
    n = 4 if k >= 300 else 2
    hours = 5.0 * mult * (2 / n) * 1.6 + 0.5
    hh = f"{int(hours):02d}:{int((hours % 1) * 60):02d}:00"
    base = (f"WRAPPER=scripts/hpg_sad.slurm,STAGE_VAR=SAD_THROUGH,VERIFIED={chain_view(k)},SED_SECRETS_FILE={SECRETS},"
            f"WORKERS=16,BATCH_INVARIANT=1,SAD_OUT={BL}/{cell},NGPUS={n},SAD_BUDGET_MULT={mult}" + (f",SAD_LR={lr}" if lr else ""))
    C4 = "--partition=hpg-milan --gpus=0 --cpus-per-task=4 --mem=16G"
    GT = f"--partition=hpg-b200 --nodes=1 --gpus={n} --cpus-per-task=8 --mem={256 if n == 4 else 160}G"
    return {"cell": cell, "tree": CHAIN_TREE, "stages": [
        stage(f"{cell}-prep", "scripts/hpg_multi_stage.slurm", C4, "02:00:00", 0, f"{base},STAGES=prepare", CHAIN_GATE),
        stage(f"{cell}-train", "scripts/hpg_multi_stage.slurm", GT, hh, n, f"{base},STAGES=train"),
        # 10-21: eval on b9f9a80 (serving check under TRAIN_PYTHON; ENV_PYTHON lacks torch) and only the final
        # checkpoint (SAD_EVAL_FRACTIONS=1; the 1/16..1/2 exports are not part of the main table).
        dict(stage(f"{cell}-eval", "scripts/hpg_multi_stage.slurm", G1, "10:00:00", 1,
                   f"{base},STAGES=all,MODE_VAR=SAD_MODE,MODE=evaluate,SAD_EVAL_FRACTIONS=1",
                   {"file": f"{GATES}/eval_{cell}"}), tree=SAD_EVAL_TREE)]}


SAD_EVAL_TREE = "b9f9a80"
FP32REF_TREE = "fea3a8e"
OVERRIDE_TREE = "8b47413"   # fea3a8e + recorded per-adapter serving-gate tolerance override (SERVING_GATE_OVERRIDE_*)




LADDER_TREE = "c6b15b2"
# 10-21 22:2x CDT: training stages retried after the multi-node split finished in attempt-001 (stage_run attempt dirs);
# evals must read that attempt's checkpoint (adc-k100-e64-lr1e-4-eval 47090929 failed on attempt-000). Verify the adapter
# path before opening any eval gate; extend this map for further retries.
TRAIN_ATTEMPT = {"adc-k100-e64-lr1e-4": 1, "adc-k300-e8-lr1e-4": 1, "adc-k500-e8-lr1e-4": 1}
# Long ladder rungs also get an hpg training stage (LONI gpu2 is saturated; first start estimate 10-21 15:41 CDT):
# whichever cluster starts a rung first keeps it (scratchpad dedupe_ladder.py cancels the pending twin).
HPG_LADDER = {500: (64, 32, 16, 8), 300: (64, 32, 16, 8), 100: (64, 32, 16), 30: (128, 64, 32)}


def lora_rung(method, k, epochs, lr=None):
    """Trajectory SFT ("tsft": the K trusted trajectories) or AD ("adc": FTP-recollected trajectories of the same K
    tasks) with AD's SFT recipe (LoRA r64/a128/dropout .05, lr 2e-4 linear, batch 8) for `epochs` epochs:
    shared-history scoring, data parallel over NGPUS (b' = min(8, N // 8) trajectories per update; 3 at K30).
    Rungs 2 (paper), 4, 8, 16 run concurrently (32 only if 16 is not converged); only the reported rung is evaluated
    (trajectory SFT greedy, AD with SAG)."""
    cell = f"{method}-k{k}-e{epochs}" + (f"-lr{lr}" if lr else "")   # 10-21: lr suffix = stabilised ladder (LONI-trained)
    n = 2 if k == 30 else 4
    per_epoch = {30: 10, 100: 13, 300: 38, 500: 63}[k]          # updates per epoch
    hours = 0.5 + epochs * per_epoch * (8 if n == 4 else 6) / 3600 * 2.5   # 8 s/update is an estimate: margin 2.5x
    hh = f"{int(hours):02d}:{int((hours % 1) * 60):02d}:00"
    out = f"{BL}/{cell}"
    if method == "tsft":
        verified, script = chain_view(k), "scripts/hpg_kang.slurm"
        e = (f"KANG_OUT={out},VERIFIED={verified},KANG_MODE=train,KANG_EPOCHS={epochs},KANG_SCORING=shared_history,"
             f"NGPUS={n},WORKERS=16,BATCH_INVARIANT=1" + (f",KANG_LR={lr},KANG_WARMUP=0" if lr else ""))
        # one gate file per method, opened by hand once the chain views are verified on hpg AND the B200
        # shared-history acceptance passed (the keeper takes a single gate per stage)
        adapter, decoding, gate = (f"{out}/stages/train-sft/attempt-{TRAIN_ATTEMPT.get(cell, 0):03d}/output/checkpoint", "greedy",
                                   {"file": f"{GATES}/tsft_ready"})
    else:
        verified = (f"{B}/reports/runs/adftp-chain/chain_k500_s0_r2/collection" +   # 10-21: recovered r2 collection
                    ("" if k == 500 else f"/subsets/k{k}"))
        script = "scripts/hpg_adftp.slurm"
        e = (f"ADFTP_OUT={out},VERIFIED={verified},ADFTP_THROUGH=train,ADFTP_EPOCHS={epochs},"
             f"ADFTP_SCORING=shared_history,NGPUS={n},WORKERS=16,BATCH_INVARIANT=1" +
             (f",ADFTP_LR={lr},ADFTP_WARMUP=0" if lr else ""))
        adapter, decoding = f"{out}/training/stages/train-sft/attempt-{TRAIN_ATTEMPT.get(cell, 0):03d}/output/checkpoint", "sag"
        gate = {"file": f"{GATES}/adc_ready"}   # + the AD-FTP chain collection verified on hpg
    res = f"--partition=hpg-b200 --nodes=1 --gpus={n} --cpus-per-task=8 --mem={160 if n == 4 else 96}G"
    if lr:
        # 10-21: stabilised ladder on tree c6b15b2 (lr/warmup overrides, AD empty-reply masking). sacct 46922829
        # (tsft K500, 4 B200): 6.8 s/update, MaxRSS 50.7G -> 96G (4 GPUs) / 64G (2 GPUs), time margin 1.6x.
        res = f"--partition=hpg-b200 --nodes=1 --gpus={n} --cpus-per-task=8 --mem={96 if n == 4 else 64}G"
        hours = 0.5 + epochs * per_epoch * (8 if n == 4 else 6) / 3600 * 1.6
        hh = f"{int(hours):02d}:{int((hours % 1) * 60):02d}:00"
    train = stage(f"{cell}-train", script, res, hh, n, e, gate)
    if lr:
        train["tree"] = LADDER_TREE
    # 10-21: the serving-equivalence check builds 8 fixed prompts from VERIFIED with strict encoding; the AD K500
    # collection holds one empty teacher reply (masked in training) -> ValueError (adc-k500-e2-eval 46980745). AD K500
    # evals take their prompts from the K300 view of the same collection (harness input only; training unchanged).
    eval_verified = verified + "/subsets/k300" if (method == "adc" and k == 500) else verified
    ev = stage(f"{cell}-eval", "scripts/hpg_eval_adapters.slurm", G1, "03:00:00", 1,
               f"ADAPTERS={cell}={adapter},TASKS=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/{CHAIN_TREE}/data/manifest/"
               f"dev140.json,DECODING={decoding},INCLUDE_BASE=0,OUT={B}/reports/runs/eval/chain/{cell},VERIFIED={eval_verified},"
               f"WORKERS=16,BATCH_INVARIANT=1", {"file": f"{GATES}/eval_{cell}"})
    return {"cell": cell, "tree": CHAIN_TREE, "stages": [train, ev]}


SMARTAD_CHAIN = f"{B}/reports/runs/smartad-chain/chain_k500_s0_r2"
SMARTAD_CHAIN_SEL = f"{B}/reports/smartad-selection/chain_k500_s0_r2"
SMARTAD_SEL_SHARDS = 4
HASH_CACHE = f"{B}/../.sed-hash-receipts"   # persistent metadata receipts (sed.hash_receipts), shared by hpg stages


def smartadselc(i):
    """10-21: SmartAD selection of the K500 chain pool (5,000 candidates, 3,147 correct) on tree 3318b54 (amendment-aware
    evidence paths). sacct 46855474 (1,000 candidates): 59 min, MaxRSS 2.3G on 1 B200 -> 4 shards x ~1.25 h.
    Shard i scores its slice; the CPU merge (smartadselc-merge) runs once gates/smartad_chain_scored exists."""
    cell = f"smartadselc-s{i}of{SMARTAD_SEL_SHARDS}"
    e = (f"SMARTAD_OUT={SMARTAD_CHAIN_SEL},VERIFIED={SMARTAD_CHAIN},SMARTAD_THROUGH=select,"
         f"SMARTAD_PATH_MAP=/home/xueqi/hq/projects/ICML27_agent_distillation={B},"
         f"SMARTAD_SELECTION_SHARDS={SMARTAD_SEL_SHARDS},SMARTAD_SELECTION_SHARD_INDEX={i},WORKERS=16,BATCH_INVARIANT=1")
    one = "--partition=hpg-b200 --nodes=1 --gpus=1 --cpus-per-task=8 --mem=48G"
    return {"cell": cell, "tree": "3318b54", "stages": [stage(f"{cell}-select", "scripts/hpg_smartad.slurm", one,
            "03:00:00", 1, e, {"file": f"{GATES}/smartad_chain_on_hpg"})]}


def smartadselc_merge():
    e = (f"SMARTAD_OUT={SMARTAD_CHAIN_SEL},VERIFIED={SMARTAD_CHAIN},SMARTAD_THROUGH=select,SMARTAD_MERGE_SCORES=1,"
         f"SMARTAD_PATH_MAP=/home/xueqi/hq/projects/ICML27_agent_distillation={B},"
         f"SMARTAD_SELECTION_SHARDS={SMARTAD_SEL_SHARDS},WORKERS=16,BATCH_INVARIANT=1,SED_HASH_CACHE_DIR={HASH_CACHE}")
    # 10-21 16:1x CDT: 47046998 hit its 2 h limit in the final load_verified_collection (re-hashing the 980k-file
    # full-TRAIN support bank on Lustre; the default receipt cache lives in the job TMPDIR). 8 h + persistent cache.
    cpu = "--partition=hpg-milan --gpus=0 --cpus-per-task=8 --mem=32G"
    return {"cell": "smartadselc-merge", "tree": "3318b54", "stages": [stage("smartadselc-merge", "scripts/hpg_smartad.slurm",
            cpu, "08:00:00", 0, e, {"file": f"{GATES}/smartad_chain_scored"})]}


SMARTAD_CHAIN_COLL = f"{SMARTAD_CHAIN_SEL}/collection"
SMC_LADDER_TREE = "e599953"   # sealed CPU preparation + seal-verified GPU start (102a35d trees superseded, never trained)


def smc_rung(k, epochs, lr):
    """SmartAD on the selected K500 chain pool (nested views K30/K100/K300). b' = 4 trajectories per update (paper):
    8 / 25 / 75 / 125 updates per epoch. lr 2e-4 x 2 epochs = paper-recipe column; lr 1e-4 ladder = converged column.
    Greedy evaluation; the serving-check prompts come from the trajectory-SFT support view of the same K (strict-clean;
    teacher candidates at T=1.0 may hold empty replies)."""
    cell = f"smc-k{k}-e{epochs}-lr{lr}"
    n = 2 if k == 30 else 4
    # 10-23 03:1x CDT: with GPU stages group-only (6 GPUs) the K500 5e-5 e64 train holds 4 and the 4-GPU K100/K300 rungs
    # would wait ~7 h beside 2 idle GPUs. The untrained K100/K300 rungs run on 2 GPUs like K30 (global batch stays b' = 4
    # trajectories, recipe.original_batch; only the data-parallel split changes). identity.json binds gpus -> fresh prep.
    if lr == "1e-4" and (k, epochs) in {(100, 32), (100, 16), (100, 8), (100, 4), (100, 2), (300, 16), (300, 4), (300, 2)}:
        n = 2
    # 10-23 08:3x CDT: same reason for the K500 5e-5 fallback ladder: e16 (2,000 updates) runs on the 2-GPU lane beside the
    # 4-GPU e64 -> e32 -> e8 -> e4 -> e2 sequence (~2 h earlier ladder decision). Its 4-GPU prep goes to _trash.
    # 10-23 10:2x CDT: e8/e4/e2 follow e16 on the 2-GPU lane (2-GPU K500 e16 runs 4.4 s/update), so the lane finishes them
    # while e32 trains on 4 GPUs instead of queueing after it (4-GPU preps -> _trash).
    if lr == "5e-5" and k == 500 and epochs in (16, 8, 4, 2):
        n = 2
    per_epoch = {30: 8, 100: 25, 300: 75, 500: 125}[k]
    # 10-22 00:2x CDT: smc-k30-e2 (47091127) timed out at 1:04 before its first update: the pipeline prepare phase alone
    # took ~52 min (61 GB read: collection + snapshot hashing), then rank-0 prepare_data. 3 h startup allowance.
    hours = 3.0 + epochs * per_epoch * 10 / 3600 * 1.6
    hh = f"{int(hours):02d}:{int((hours % 1) * 60):02d}:00"
    out = f"{BL}/{cell}"
    verified = SMARTAD_CHAIN_COLL + ("" if k == 500 else f"/subsets/k{k}")
    e = (f"SMARTAD_OUT={out},VERIFIED={verified},SMARTAD_THROUGH=train,SMARTAD_EPOCHS={epochs},SMARTAD_LR={lr},"
         f"SMARTAD_WARMUP=0,SMARTAD_SCORING=shared_history,NGPUS={n},WORKERS=16,BATCH_INVARIANT=1,"
         f"SED_HASH_CACHE_DIR={HASH_CACHE}")
    res = f"--partition=hpg-b200 --nodes=1 --gpus={n} --cpus-per-task=8 --mem={96 if n == 4 else 64}G"
    # 10-22 09:2x CDT: sealed preparation (main e599953). The CPU prep stage runs the full validation once and publishes the
    # encoded examples + seal; the GPU train stage verifies the seal and trains without collection scans (root cancelled
    # 47095421 for idle GPUs during ~1 h of startup hashing). Startup allowance back to 1 h.
    hours = 1.0 + epochs * per_epoch * 10 / 3600 * 1.6
    # 10-22 22:3x CDT, sized from sacct: sealed-prep trains run 3.2-3.4 s/update incl. startup (smc-k500-e32 4,000 updates
    # 3:31, smc-k100-e64 1,600 updates 1:29); 3.5 s x 1.6 + 45 min (2-GPU K30 rungs: 7 s, twice the per-GPU work).
    hours = 0.75 + epochs * per_epoch * (7.0 if n == 2 else 3.5) * 1.6 / 3600
    hh = f"{int(hours):02d}:{int((hours % 1) * 60):02d}:00"
    prep = stage(f"{cell}-prep", "scripts/hpg_smartad.slurm", "--partition=hpg-milan --gpus=0 --cpus-per-task=8 --mem=32G",
                 "04:00:00", 0, e + ",SMARTAD_MODE=prepare", {"file": f"{GATES}/smartad_selected"})
    train = stage(f"{cell}-train", "scripts/hpg_smartad.slurm", res, hh, n, e)
    prep["tree"] = train["tree"] = SMC_LADDER_TREE
    adapter = f"{out}/training/stages/train-sft/attempt-{TRAIN_ATTEMPT.get(cell, 0):03d}/output/checkpoint"
    ev = stage(f"{cell}-eval", "scripts/hpg_eval_adapters.slurm", G1, "03:00:00", 1,
               f"ADAPTERS={cell}={adapter},TASKS=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs/{CHAIN_TREE}/data/manifest/"
               f"dev140.json,DECODING=greedy,INCLUDE_BASE=0,OUT={B}/reports/runs/eval/chain/{cell},VERIFIED={chain_view(k)},"
               f"WORKERS=16,BATCH_INVARIANT=1", {"file": f"{GATES}/eval_{cell}"})
    # 10-23 12:5x CDT (user chose option A): the K30 e32 serving gate failed against the bf16 HF reference (0.078 > 0.068);
    # rerun the same gate against a float32 HF reference (tree fea3a8e = 20cb697 + --reference-dtype; gate rule unchanged).
    # 10-24 00:2x CDT: the float32-reference gate failed too (0.067 > 0.061); user chose option B (Discord 00:24 CDT,
    # "可以按照你倾向的来"): evaluate with a recorded override of the tolerance verdict only (identity checks kept),
    # annotated in the table.
    if cell == "smc-k30-e32-lr1e-4":
        ev["tree"] = OVERRIDE_TREE
        ev["exports"] += (",REFERENCE_DTYPE=float32,SERVING_GATE_OVERRIDE_ADAPTER=smc-k30-e32-lr1e-4"
                          ",SERVING_GATE_OVERRIDE_REASON=user_option_B_discord_1563422934167461943_bf16_sensitivity_"
                          "of_sharp_adapter_vLLM_closer_to_fp32_than_HF_bf16"
                          ",SERVING_GATE_OVERRIDE_APPROVED_BY=xueqi118_discord"
                          ",SERVING_GATE_OVERRIDE_APPROVED_AT=2026-10-24T00:24-05:00")
    return {"cell": cell, "tree": CHAIN_TREE, "stages": [prep, train, ev]}


def smc_cells():
    paper = [smc_rung(k, 2, "2e-4") for k in (500, 300, 100, 30)]
    if SMC_LADDER_TREE is None:
        return paper
    ladder = {30: (2, 4, 8, 16, 32, 64, 128), 100: (2, 4, 8, 16, 32, 64), 300: (2, 4, 8, 16, 32), 500: (2, 4, 8, 16, 32)}
    lad = {k: [smc_rung(k, e, "1e-4") for e in sorted(ladder[k], reverse=True)] for k in ladder}
    # 10-22 22:3x CDT: smc-k500-e32 (1e-4) fails the registered divergence check (L_last 4.79 vs L_min 0.10, collapse at
    # epoch 7) -> pre-registered 5e-5 fallback for SmartAD K500 (as trajectory SFT / AD K500); its pending 1e-4 rungs are
    # superseded in the keeper state. Order: the long K500 5e-5 e64 rung first, then K30 (2-GPU rungs fit the group
    # account's free GPUs while 4-GPU rungs wait on the dept CPU QOS), K100, K300, the rest of K500.
    fb500 = [smc_rung(500, e, "5e-5") for e in (64, 32, 16, 8, 4, 2)]
    return paper + fb500[:1] + lad[30] + lad[100] + lad[300] + lad[500] + fb500[1:]


def chain_cells():
    sad1 = [sadc(k, 1) for k in (500, 300, 100, 30)]
    tsft = [lora_rung("tsft", k, e) for k in (500, 300, 100, 30) for e in (16, 8, 4, 2)]
    # 19:3x CDT: on hpg the 2-epoch rung of K300/K500 is held (it is needed only if 4 epochs were already converged
    # vs 2, implausible at 2-epoch losses of 0.2-0.35); it frees a job slot for SAD under the 10-job rule.
    # 10-21 01:5x CDT: all trajectory-SFT and AD training runs on LONI (paper recipe 2e-4 rungs and the stabilised
    # 1e-4 ladder); hpg runs only their evaluation stages, gated per cell once the adapter copy is verified on hpg.
    # 19:2x CDT: trajectory-SFT training for K30/K100 runs on LONI (A100, smoke tsft-k30-e2-loni passed; hpg's B200
    # partition is congested); only their evaluation stages run here, after the adapters are copied back.
    for c in tsft:
        c["stages"] = [st for st in c["stages"] if st["name"].endswith("-eval")]
    sad05 = [sadc(k, 0.5) for k in (500, 300, 100, 30)]
    adc = [lora_rung("adc", k, e) for k in (500, 300, 100, 30) for e in (16, 8, 4, 2)]
    # 10-21 02:3x CDT: stabilised ladder (lr 1e-4, no warmup; registered deviation, probe report
    # reports/engineering/lora_stability_probe_1021.md) trained on LONI tree c6b15b2; hpg evaluates only the reported
    # (converged) rung of each method x K, opened per cell by its gate.
    ladder = {30: (2, 4, 8, 16, 32, 64, 128), 100: (2, 4, 8, 16, 32, 64), 300: (2, 4, 8, 16, 32), 500: (2, 4, 8, 16, 32)}
    for c in tsft + adc:
        c["stages"] = [st for st in c["stages"] if st["name"].endswith("-eval")]
    lad = [lora_rung(m, k, e, lr="1e-4") for m in ("tsft", "adc") for k in (500, 300, 100, 30)
           for e in sorted(ladder[k], reverse=True)]
    for c in lad:
        k, e = int(c["cell"].split("-k")[1].split("-")[0]), int(c["cell"].split("-e")[1].split("-")[0])
        if e not in HPG_LADDER[k]:
            c["stages"] = [st for st in c["stages"] if st["name"].endswith("-eval")]
    adc += lad
    # 10-21 14:3x CDT: tsft K300 16->32 still drops ~0.05 -> e64 rungs for tsft K300/K500 (hpg train + LONI twin).
    # Pre-registered 5e-5 fallback ladders (tsft K30: e4 rung failed at 1e-4; adc K500: e4 rung failed) train on LONI;
    # eval-only gated cells here so the reported rung can be evaluated.
    adc += [lora_rung("tsft", k, 64, lr="1e-4") for k in (500, 300)]
    fb = [lora_rung("tsft", 30, e, lr="5e-5") for e in (128, 64, 32, 16, 8, 4, 2)]
    fb += [lora_rung("adc", 500, e, lr="5e-5") for e in (64, 32, 16, 8, 4, 2)]   # e64 added 10-22 12:1x CDT (16->32 delta 0.059)
    # 10-21 22:1x CDT: tsft K500 e32 (1e-4, hpg) ended at 4.30 (invalid) -> tsft K500 5e-5 ladder; tsft K300 e64 (1e-4) diverging
    # mid-run (1.89 at epoch 10) -> tsft K300 5e-5 ladder as a hedge (the rule decides once the 1e-4 rung completes).
    fb += [lora_rung("tsft", k, e, lr="5e-5") for k in (500, 300) for e in (64, 32, 16, 8, 4, 2)]
    # 10-22 09:1x CDT: mid-run collapse then re-memorisation found in 1e-4 rungs the registered guard passes (adc K300 e32:
    # epoch 5 0.214 -> 5.02, dev140 4/140; adc K100 e64 and tsft K100 e64 collapsed at epoch 13). 5e-5 hedge ladders for
    # adc K100/K300 and tsft K100 while the user decides on a collapse criterion.
    fb += [lora_rung("adc", 100, e, lr="5e-5") for e in (64, 32, 16, 8, 4, 2)]
    # 10-23 20:2x CDT: AD K300 5e-5 e16 -> e32 L_last 0.0456 -> 0.0009 (delta 0.045 >= 0.01): not converged -> e64 rung.
    fb += [lora_rung("adc", 300, e, lr="5e-5") for e in (64, 32, 16, 8, 4, 2)]
    fb += [lora_rung("tsft", 100, e, lr="5e-5") for e in (64, 32, 16, 8, 4, 2)]
    for c in fb:
        # 10-21 17:2x CDT: the long AD K500 5e-5 rungs also get an hpg train stage (~6x faster than LONI A100s).
        if c["cell"] not in ("adc-k500-e64-lr5e-5", "adc-k500-e32-lr5e-5", "adc-k500-e16-lr5e-5", "tsft-k500-e64-lr5e-5", "tsft-k500-e32-lr5e-5",
                             "tsft-k300-e64-lr5e-5", "tsft-k300-e32-lr5e-5", "adc-k100-e64-lr5e-5", "tsft-k100-e64-lr5e-5",
                             "adc-k300-e64-lr5e-5", "adc-k300-e32-lr5e-5", "adc-k300-e16-lr5e-5", "adc-k300-e8-lr5e-5", "adc-k300-e4-lr5e-5",
                             "adc-k300-e2-lr5e-5"):
            # 10-23 12:5x CDT: user: cells that scored ~0 are rerun -> AD K300 (1e-4 e32 collapsed, 4/140) runs the
            # registered 5e-5 ladder on hpg (LONI no longer trains): e16/e8/e4/e2 get hpg train stages too.
            c["stages"] = [st for st in c["stages"] if st["name"].endswith("-eval")]
    adc += fb
    # 20:1x CDT: SAD K100-K500 hit 5 teacher replies (of 9,181 in K500) that the H0 parser rejected ("...Action:"
    # without a line break; harness parser_error step, episode still won). The span grammar raises on them; held until
    # the masking fix lands (K30 contains none of them and runs).
    # 21:0x CDT: the fix is main fb22f09 (SAD/SmartAD mask parser-rejected replies; Kang/AD unchanged): SAD K100-K500
    # run on that tree (their 20cb697 outputs moved to _superseded/sadc-20cb697); K30 (no such replies) stays on 20cb697.
    for c in sad1 + sad05:
        if not c["cell"].startswith("sadc-k30-"):
            c["tree"] = "fb22f09"
    # User 20:23 CDT: trajectory SFT, AD and SmartAD are the top priority; SAD after them.
    # 10-21 04:2x CDT: SAD K100 x1 diverged at update 45 (loss 451, grad norm 8e4 at lr ~2e-5; last epoch 4.22, invalid)
    # while x0.5 ended at 3e-5 (valid). The doubling rule one rung lower: x0.25 vs x0.5 decides whether x0.5 is the
    # converged rung; x1 stays the paper-recipe column (reported as diverged).
    sad025 = [sadc(100, 0.25)]
    # 10-21 11:2x CDT: SAD K500 x0.5 -> x1 delta 0.162 (0.167 -> 0.0043): not converged -> x2 rung.
    sad025 += [sadc(500, 2)]
    for c in sad025:
        c["tree"] = "fb22f09"
    sel = [smartadselc(i) for i in range(SMARTAD_SEL_SHARDS)] + [smartadselc_merge()]
    # 10-22 19:0x CDT: SAD (its K500 x2 eval was starving) ahead of the SmartAD ladder, whose pending trains filled the
    # 10-job cap; evals and short stages first.
    # 10-23 12:5x CDT: user: cells that scored ~0 are rerun. SAD K100 x1 (paper budget) diverged -> registered halve-lr
    # fallback: lr 2e-5 -> 1e-5 (SAD_LR, tree 7cbc62b = fb22f09 + --learning-rate); evaluation on b9f9a80 as for every SAD
    # cell (evaluate mode does not re-derive the recipe).
    sadlr = [sadc(100, 1, lr="1e-5")]
    for c in sadlr:
        c["tree"] = "7cbc62b"
    # 10-23 20:2x CDT: the SAD K100 rerun eval (1 GPU) goes ahead of the 4-GPU AD K300 rungs.
    return tsft + sadlr + adc + sel + sad1 + sad05 + sad025 + smc_cells()

# ---- 10-24: training seeds 1, 2 of the main grid (user 08:32 CDT "可以按照你建议的来") ----
# Each reported seed-0 cell is retrained at its reported rung and lr (no ladder) with only the training seed changed
# (LoRA init, dropout stream, epoch sampler order); same data views, GPUs, recipe and evaluation. Every seed run is
# evaluated. SAD is not retrained: full-parameter (no random init), fixed ascending-complexity curriculum, no active
# dropout (dropout_equivalence "exact") -> its training does not depend on the seed.
SEED_AD_TREE = "e6459b9"   # c6b15b2 + AD --training-seed (branch seed-ad); hpg tree verified 10-24 08:5x CDT (1,259 files, 93,636,659 B)
SEED_SMC_TREE = "4f5a138"  # e599953 + SmartAD --training-seed (branch seed-smc); hpg tree verified 10-24 09:3x CDT (1,274 files, 93,868,494 B)
SEED_RUNGS = [  # (method, K, epochs, lr) of the reported seed-0 cell; lr None = paper recipe 2e-4
    ("tsft", 500, 64, "5e-5"), ("adc", 500, 64, "5e-5"), ("tsft", 300, 64, "1e-4"), ("adc", 300, 64, "5e-5"),
    ("smc", 500, 32, "5e-5"), ("smc", 300, 32, "1e-4"), ("tsft", 100, 64, "1e-4"), ("adc", 100, 64, "1e-4"),
    ("smc", 100, 64, "1e-4"), ("tsft", 30, 64, "5e-5"), ("adc", 30, 64, "1e-4"), ("smc", 30, 32, "1e-4"),
    ("tsft", 500, 2, None), ("adc", 500, 2, "1e-4"), ("tsft", 300, 2, None), ("adc", 300, 2, None),
    ("smc", 500, 2, "2e-4"), ("smc", 300, 2, "2e-4"), ("tsft", 100, 2, "1e-4"), ("adc", 100, 2, None),
    ("smc", 100, 2, "2e-4"), ("tsft", 30, 2, None), ("adc", 30, 2, None), ("smc", 30, 2, "2e-4")]
SEED_VAR = {"tsft": "KANG_TRAINING_SEED", "adc": "ADFTP_TRAINING_SEED", "smc": "SMARTAD_TRAINING_SEED"}
# 10-24 18:4x CDT: rungs whose seed-0 training ran on LONI A100 (2 GPUs) train their seeds there too (same hardware as
# seed 0, frees hpg for the K300/K500 rungs); hpg keeps only the eval, gated on the verified copy (gates/eval_<cell>).
LONI_SEED = {("tsft", 100, 64, "1e-4"), ("tsft", 500, 2, None), ("adc", 500, 2, "1e-4"), ("tsft", 300, 2, None),
             ("adc", 300, 2, None), ("tsft", 100, 2, "1e-4"), ("adc", 100, 2, None), ("tsft", 30, 2, None),
             ("adc", 30, 2, None)}


def seed_cell(method, k, epochs, lr, seed):
    """Training seed `seed` of a reported seed-0 cell. Paper-recipe trajectory SFT / AD cells (seed 0 on 20cb697, partly
    LONI) are built with the explicit 2e-4 / warmup-0 overrides of c6b15b2, which give the same trainer argv and recipe as
    the 20cb697 defaults, and the sacct-based LoRA sizes."""
    if method == "smc":
        c = smc_rung(k, epochs, lr)
        name0 = c["cell"]
    else:
        c = lora_rung(method, k, epochs, lr=lr or "2e-4")
        name0 = f"{method}-k{k}-e{epochs}" + (f"-lr{lr}" if lr else "")
    old, new = c["cell"], f"{name0}-s{seed}"
    pat = re.compile(rf"(?<=[/=]){re.escape(old)}(?=[/=,]|$)")
    for st in c["stages"]:
        st["name"] = new + st["name"][len(old):]
        e = pat.sub(new, st["exports"])
        e = re.sub(r"attempt-\d{3}/output/checkpoint", "attempt-000/output/checkpoint", e)   # fresh run: first attempt
        e = ",".join(x for x in e.split(",") if not x.startswith(("REFERENCE_DTYPE=", "SERVING_GATE_OVERRIDE_")))
        assert not pat.search(e), (new, e)
        if st["name"].endswith("-eval"):
            st.pop("gate", None)   # every seed run is evaluated (no rung decision)
            st["tree"] = CHAIN_TREE
        else:
            e += f",{SEED_VAR[method]}={seed}"
            if method == "tsft":
                st["tree"] = LADDER_TREE
            elif method == "adc":
                st["tree"] = SEED_AD_TREE
                st["gate"] = {"file": f"{GATES}/seed_ad_ready"}
            else:
                st["tree"] = SEED_SMC_TREE
                st["gate"] = {"file": f"{GATES}/seed_smc_ready"}
        st["exports"] = e
    c["cell"] = new
    if (method, k, epochs, lr) in LONI_SEED:
        c["stages"] = [st for st in c["stages"] if st["name"].endswith("-eval")]
        c["stages"][0]["gate"] = {"file": f"{GATES}/eval_{new}"}
    return c


def seed_cells():
    ready = {"tsft": True, "adc": SEED_AD_TREE is not None, "smc": SEED_SMC_TREE is not None}
    cells = [seed_cell(m, k, e, lr, s) for m, k, e, lr in SEED_RUNGS if ready[m] for s in (1, 2)]
    # User 10-24 18:38 CDT: "把多seed停下来" -> student-base variant (gemma-4-E2B-it) instead; seed cells held.
    for c in cells:
        c["hold"] = True
    return cells


# 10-24 18:4x CDT, user: student-base variant google/gemma-4-E4B-it (only the student changes; same K views, teacher data,
# per-cell rung and lr of the 12B converged column, eval protocol). Seed 0, converged column, K30 + K100 first.
# configs/base_e4b.yaml = base.yaml with models.student -> E4B (sha256 faf549ff... on rai, hpg, LONI). SmartAD re-selects
# its candidates with the E4B NLL (same 5,000 teacher candidates); no new teacher calls.
VARIANTS = {   # tag -> (config, snapshot); model.safetensors sha256 verified on rai/hpg/LONI against the HF LFS hash
    "e4b": (f"{B}/configs/base_e4b.yaml", "/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--google--gemma-4-E4B-it/"
            "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"),   # cfbd3d2f...; config faf549ff...
    # 10-24 18:51 CDT, user: run the same 8 cells for gemma-4-E2B-it in parallel ("我们还有没有资源同时跑E2B的相同的8格结果")
    "e2b": (f"{B}/configs/base_e2b.yaml", "/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache/hub/models--google--gemma-4-E2B-it/"
            "snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7"),   # 2db5482b...
}
E4B_GATE = {"file": f"{GATES}/e4b_ready"}
ES_TREE = {"c6b15b2": "2506fa9", "e599953": "f453db8", "3318b54": "4c2bcfc", "b9f9a80": "0c2cd9e",
           "20cb697": "95a62b2",   # 12B tree -> same tree + E-series support (branch eseries-template, cherry-picks)
           "fb22f09": "96209eb"}   # fb22f09 + 2506fa9 + 4f47296 + 5df9eb4 (SAD K100: parser mask; serving + export fixes)
ES_EVAL_TREE = "46b8527"   # 95a62b2 + 4f47296 (TRITON_ATTN for model_type gemma4); 12B serving command unchanged
ES_SAD_TREE = "c265b5d"   # 46b8527 + 5df9eb4 (E-series full-parameter export: KV-shared layers' unused base tensors)
ES_SAD_EVAL_TREE = "17eec0d"   # 0c2cd9e + 4f47296 + 5df9eb4


def v_env(tag):
    cfg, model = VARIANTS[tag]
    return f"CONFIG={cfg},MODEL_PATH={model}"


def v_sel(tag):
    return f"{B}/reports/smartad-selection/chain_k500_s0_r2_{tag}"


E4B_RUNGS = [("tsft", 30, 64, "5e-5"), ("tsft", 100, 64, "1e-4"), ("adc", 30, 64, "1e-4"), ("adc", 100, 64, "1e-4"),
             ("smc", 30, 32, "1e-4"), ("smc", 100, 64, "1e-4"), ("sad", 30, 1, None), ("sad", 100, 0.5, None)]
E4B_LONI = {("tsft", 30), ("tsft", 100), ("adc", 30), ("adc", 100),   # trained on LONI A100; hpg evaluates the verified copy
            ("smc", 30), ("smc", 100)}   # 10-25 01:0x CDT (user "咱们尽量并行"): SmartAD prep+train on LONI too; the selected
            # collection (6 GB, self-contained) is copied hpg -> LONI with a full sha256 manifest check before prep


def variant_cell(tag, method, k, epochs, lr):
    """The 12B cell of the same rung with another student: cell/output names prefixed <tag>-, fresh attempt-000 adapters,
    CONFIG/MODEL_PATH exported, SmartAD on the pool re-selected with that student, no per-rung eval gates."""
    c = (sadc(k, epochs) if method == "sad" else smc_rung(k, epochs, lr) if method == "smc"
         else lora_rung(method, k, epochs, lr=lr))
    if method == "sad" and k != 30:
        # 10-25 02:5x CDT: as the 12B SAD K100-K500 (tree override in the 10-20 plan, applied outside sadc()): mask the
        # teacher replies the H0 parser rejects (fb22f09); without it e4b-sadc-k100-b05 failed "SAD reply requires an
        # Action line" (47607394). K30 contains no such reply and stays on 20cb697, as for the 12B.
        c["tree"] = "fb22f09"
    old, new = c["cell"], f"{tag}-" + c["cell"]
    pat = re.compile(rf"(?<=[/=]){re.escape(old)}(?=[/=,]|$)")
    for i, st in enumerate(c["stages"]):
        st["name"] = new + st["name"][len(old):]
        e = pat.sub(new, st["exports"]).replace(SMARTAD_CHAIN_COLL, f"{v_sel(tag)}/collection")
        e = re.sub(r"attempt-\d{3}/output/checkpoint", "attempt-000/output/checkpoint", e)
        e = ",".join(x for x in e.split(",") if not x.startswith(("REFERENCE_DTYPE=", "SERVING_GATE_OVERRIDE_")))
        assert not pat.search(e), (new, e)
        st["exports"] = f"{e},{v_env(tag)}"
        if st["name"].endswith("-eval"):
            st.pop("gate", None)
            if method == "smc":
                st["tree"] = CHAIN_TREE
        elif i == 0:
            st["gate"] = {"file": f"{GATES}/{tag}_smartad_selected"} if method == "smc" else {"file": f"{GATES}/{tag}_ready"}
    if (method, k) == ("tsft", 30):   # base student on dev140 (reference line) in the same eval job
        c["stages"][-1]["exports"] = c["stages"][-1]["exports"].replace("INCLUDE_BASE=0", "INCLUDE_BASE=1")
    c["cell"] = new
    return c


def variant_selection(tag):
    """SmartAD selection of the K500 chain pool with the variant's NLL: 4 shards + CPU merge (as smartadselc)."""
    cells = []
    for c in [smartadselc(i) for i in range(SMARTAD_SEL_SHARDS)] + [smartadselc_merge()]:
        c["cell"] = f"{tag}-" + c["cell"]
        for st in c["stages"]:
            st["name"] = f"{tag}-" + st["name"]
            st["exports"] = (st["exports"].replace(f"SMARTAD_OUT={SMARTAD_CHAIN_SEL},", f"SMARTAD_OUT={v_sel(tag)},")
                             + f",{v_env(tag)}")
            assert v_sel(tag) in st["exports"]
            st["gate"] = ({"file": f"{GATES}/{tag}_ready"} if "merge" not in st["name"]
                          else {"file": f"{GATES}/{tag}_smartad_scored"})
        cells.append(c)
    return cells


def variant_cells(tag, loni_trains=None):
    cells = variant_selection(tag)
    for m, k, e, lr in E4B_RUNGS:
        c = variant_cell(tag, m, k, e, lr)
        if (m, k) in E4B_LONI:
            train = [st for st in c["stages"] if not st["name"].endswith("-eval")]
            if loni_trains is not None:
                loni_trains.extend(train)
            c["stages"] = [st for st in c["stages"] if st["name"].endswith("-eval")]
            c["stages"][0]["gate"] = {"file": f"{GATES}/eval_{c['cell']}"}
        cells.append(c)
    # 10-24 19:3x CDT: E-series template/architecture differs from the 12B (no empty-thought OFF prefill, Gemma4TextAttention,
    # KV-shared layers, per-layer inputs, audio tower); every variant stage failed on trees without that support.
    # 21:xx CDT: fix 2506fa9 (Codex, reviewed; 12B byte-identical by golden tests against each base tree) cherry-picked onto
    # every tree the variant cells use -> variant cells run on the fixed twin of their 12B tree.
    for st in [st for c in cells for st in c["stages"]] + list(loni_trains or []):
        if st.get("tree") in ES_TREE:
            st["tree"] = ES_TREE[st["tree"]]
    for c in cells:
        c["tree"] = ES_TREE[c["tree"]]
    # 10-25 00:0x CDT: the first E-series eval (47583436) failed at vLLM CUDA-graph capture: vLLM picks FLASH_ATTN for
    # model_type gemma4 and batch invariance downgrades FA4 -> FA2, which cannot serve the 512-wide global heads (the 12B
    # runs TRITON_ATTN). Fix 4f47296 pins TRITON_ATTN for gemma4 -> adapter evals run on 95a62b2 + 4f47296 = ES_EVAL_TREE.
    # SAD stays on hold: its full-parameter export rejects the KV-shared layers' unused base tensors (fix pending).
    for c in cells:
        for st in c["stages"]:
            if st["name"].endswith("-eval") and st.get("tree", c["tree"]) == ES_TREE[CHAIN_TREE]:
                st["tree"] = ES_EVAL_TREE
        # 10-25 01:2x CDT: KV-shared export fix 5df9eb4 (unused k_proj/v_proj/k_norm of the shared layers copied from the base,
        # sha256-verified; 12B byte-identical) -> SAD prep/train on ES_SAD_TREE, SAD eval on ES_SAD_EVAL_TREE. E4B SAD ran
        # ~5-6.6 s/update (12B 4.3) -> 14 h train limit (12B K30 took 4.1 h of 8.5 h).
        if c["cell"].startswith(f"{tag}-sadc-"):
            for st in c["stages"]:
                st["tree"] = {ES_TREE[CHAIN_TREE]: ES_SAD_TREE, ES_TREE["b9f9a80"]: ES_SAD_EVAL_TREE,
                              ES_TREE["fb22f09"]: ES_TREE["fb22f09"]}[st.get("tree", c["tree"])]
                if st["name"].endswith("-train"):
                    st["time"] = "14:00:00"
    return cells


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    # Plan v4 (10-19 23:5x CDT, user priority): core agent-distillation methods first, every K x s0-s2:
    # AD/Kang (K grid already complete), AD-FTP (collection running on rai; training cells added when the FTP
    # collections land), SAD (15 cells here), SmartAD (to implement). Everything else is on hold until tomorrow;
    # running jobs of held cells are still tracked.
    # sadsh-k20s0 first: SAD trains are the long pole (hours on 4 GPUs), AD-FTP stages take 5-30 min.
    # 10-20 09:4x CDT: SmartAD selection moved from rai to hpg (top priority: it gates all 15 SmartAD cells).
    cells = [smartadsel(s) for s in (0, 1, 2)]
    cells += [sad(20, 0)] + [adftp(k, s) for s in (0, 1, 2) for k in (5, 10, 20, 50, 100)]
    # 10-20 08:4x CDT: Kang evals re-run with the k_eq_v serving fix (pre-fix served V without the k_proj LoRA delta).
    cells += [kangfix(k, s) for s in (0, 1, 2) for k in (5, 10, 20, 50, 100)]
    cells += kangchk(100, 0, "def44a9")
    # SAD: T4 per-turn scoring needed ~31 h per cell (sad-k20s0 reached the 1/8 checkpoint in 3.9 h; cancelled
    # 10-20 04:49 CDT after 1/8, its updates.jsonl is the per-turn reference curve). All 15 cells run on T5 with
    # shared-history packing (sadsh-*), k20s0 first (top of the plan) so its curve can be compared with the per-turn reference.
    # SmartAD cells wait for their selected pools (gates); ahead of the remaining SAD cells because they are short.
    cells += [smartad(k, s) for s in (0, 1, 2) for k in (5, 10, 20, 50, 100)]
    cells += [sad(k, s) for k, s in K_CELLS if (k, s) != (20, 0)]
    # User 10-20 15:08 CDT: Seed-SFT re-evaluation with the serving fix right after SAD (ahead of low priority).
    cells += [seedsftfix(k, s) for s in (0, 1, 2) for k in (5, 10, 20, 50, 100)]
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
            if ("eval" in st["name"] and not st.get("gate") and c["tree"] not in EVAL_FIX.values()
                    and st.get("tree") not in EVAL_FIX.values()):
                if c["tree"] in EVAL_FIX and not c.get("lowprio"):
                    st["tree"] = EVAL_FIX[c["tree"]]
                else:
                    st["gate"] = LORA_GATE
    # User 10-20 15:42 CDT: new main grid K in {30,100,300,500,1000,2000,all} x seed 0 from the full-TRAIN collection,
    # trained to convergence. The K5-K100 x 3-seed grid is kept only where nearly finished: hold the remaining SAD
    # cells (sadsh-k5s0 runs on as the first end-to-end SAD check), the Seed-SFT re-evaluation and low priority.
    # User 10-20 15:46 CDT: "之前跑的不对的任务停下来" -> every cell of the old K5-K100 grid is held (its trainings use
    # the papers' fixed epochs and are not converged); running jobs were cancelled.
    for c in cells:
        c["hold"] = True
    loni_trains = []
    cells = variant_cells("e4b", loni_trains) + variant_cells("e2b", loni_trains) + seed_cells() + chain_cells() + cells
    for c in cells:
        for st in c["stages"]:
            st["exports"] = ",".join(x for x in (st.get("exports"), OFF_HOME) if x)
    # max_gpus counts pending jobs too: 20 lets 1-GPU evals pass while SAD trains wait on the dept QOS.
    # User 10-20 14:29 CDT: SAD first, then AD-FTP/SmartAD to 15/15 -> low priority shrinks to 2 jobs / 4 GPUs.
    # 19:3x CDT: pending GPUs do not consume the QOS; 36 lets SAD trains queue behind the trajectory-SFT rungs.
    # 10-23 00:3x CDT: the dept QOS's 128 CPUs are held by other users' CPU-only jobs with 1-9 days left, and our dept
    # trains pended in QOSGrpCpuLimit while holding 8 of the 10 job slots. max_gpus = max_group_gpus sends every GPU
    # stage to the group account (a stage that does not fit the group allowance now waits instead of going to dept).
    # 10-23 11:4x CDT: user approved the full group allowance (8 GPUs; yd24f QOS gpu=8) for the remaining SmartAD rungs.
    plan = {"limits": {"max_jobs": 10, "max_gpus": 8, "max_group_gpus": 8, "max_lowprio_jobs": 0,
                       "max_lowprio_gpus": 0}, "cells": cells}
    names = [c["cell"] for c in cells]
    assert len(names) == len(set(names)), "duplicate cell"
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    (out / "e4b_loni_trains.json").write_text(json.dumps(loni_trains, indent=1))
    print(f"{len(cells)} cells ({len(held)} low priority), {sum(len(c['stages']) for c in cells)} stages -> {out / 'plan.json'}")


if __name__ == "__main__":
    main()
