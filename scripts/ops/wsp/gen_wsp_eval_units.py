#!/usr/bin/env python3
"""gen_wsp_eval_units.py SITE OUT_DIR   (SITE = hpg | loni) — short evaluation jobs for the WSP queue.

Why: both clusters only start our jobs by backfill, and backfill only takes short jobs. One evaluation unit = one
adapter (dev140, greedy), or two when one of them has a LoRA rank vLLM cannot take as --max-lora-rank on its own
(vLLM 0.27.1 accepts 1, 8, 16, 32, 64, 128, 256, 320, 512; the harness passes the largest rank in the job, so a
rank-4 adapter is paired with a higher-rank one). Every adapter is evaluated in two runs (two different jobs).
Writes jobs/ev.<unit>.run<r>.env/.res and lists/ev1.list (first runs, most informative first), lists/ev2.list.
Eval tree: wsp_4ba5d0e (E-series serving fix).
"""
import sys
from pathlib import Path

site, out = sys.argv[1], Path(sys.argv[2])
if site == "hpg":
    H = "/blue/fsu-compsci-dept/xc25.fsu/hq"
    A = H + "/ICML27_agent_distillation"
    R, TE = H + "/ICML27_wsp", H + "/ICML27_runs/wsp_4ba5d0e"
    OLD = A + "/reports/runs/baselines"
    VER = {k: f"{A}/data/verified/seeds_full_train_r2/subsets/{k}" for k in ("k20", "k100")}
    ENV = {"PROJECT_ROOT": TE, "CONFIG": TE + "/configs/base_e4b.yaml",
           "MODEL_PATH": H + "/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2",
           "ALFWORLD_DATA": H + "/tools/alfworld-data-empty", "VLLM_CACHE_ROOT": H + "/tools/vllm-cache",
           "TRITON_CACHE_DIR": H + "/tools/triton-cache", "VLLM_NO_USAGE_STATS": "1", "WORKERS": "16", "BATCH_INVARIANT": "1",
           "SED_HASH_CACHE_DIR": H + "/.sed-hash-receipts"}
    RES = dict(ACCOUNT="", PARTITION="hpg-b200", GPUS="1", CPUS="4", MEM="24G", SCRIPT="scripts/hpg_eval_adapters.slurm")
    LIMIT = {1: "00:30:00", 2: "00:45:00", 3: "01:00:00"}       # history: 16 min for one adapter on a B200
    OLDSEEDS = {"k20": (0, 1, 2), "k100": (0, 1, 2)}
    HAS_B3 = True
else:
    W = "/work/xueqic/hq"
    R, TE = W + "/ICML27_wsp", W + "/ICML27_runs/wsp_4ba5d0e"
    OLD = W + "/ICML27_data/reports/runs/baselines"
    VER = {k: f"{W}/ICML27_data/verified/seeds_full_train_r2/subsets/{k}" for k in ("k20", "k100")}
    ENV = {"PROJECT_ROOT": TE, "EXPECT_COMMIT": "4ba5d0e", "HPG_CONFIG": W + "/ICML27_runs/eval_46b8527/configs_ec704e4/loni_eval.env",
           "CONFIG": TE + "/configs/base_e4b.yaml",
           "MODEL_PATH": "/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2",
           "WORKERS": "16", "BATCH_INVARIANT": "1", "VLLM_CACHE_ROOT": W + "/tools/vllm-cache", "TRITON_CACHE_DIR": W + "/tools/triton-cache",
           "VLLM_NO_USAGE_STATS": "1", "SED_HASH_CACHE_DIR": W + "/ICML27_v0_runs/sed-hash-receipts"}
    RES = dict(KIND="gpu", GPUS="1", CPUS="8", STAGE_SCRIPT="scripts/hpg_eval_adapters.slurm")
    LIMIT = {1: "00:50:00", 2: "01:15:00", 3: "01:40:00"}       # history: 27 min for one adapter on an A100
    OLDSEEDS = {"k20": (0,), "k100": (0, 2)}
    HAS_B3 = False
CELL = {"k20": "e4b-tsft-k20-e32-lr5e-5", "k100": "e4b-tsft-k100-e64-lr1e-4"}
(out / "jobs").mkdir(parents=True, exist_ok=True)
(out / "lists").mkdir(exist_ok=True)


def old(k, s):
    return f"{OLD}/{CELL[k]}{'' if s == 0 else f'-s{s}'}/stages/train-sft/attempt-000/output/checkpoint"


def new(cell, update=None):
    base = f"{R}/cells/{cell}/stages/train-sft/attempt-000/output"
    return (f"{base}/checkpoints/update-{update}" if update else f"{base}/checkpoint", "adapter_model.safetensors")


def der(name):
    return (f"{R}/derived/{name}/checkpoint", "training_state.json")


units = []      # (unit name, k, [(served name, (path, required file))], include_base)


def unit(name, k, members, base=False):
    units.append((name, k, members, base))


for k in ("k20", "k100"):
    ck = (10, 40, 80, 160) if k == "k20" else (26, 104, 208, 416)
    lrs = ("-lr5e-5", "-lr2e-4") if k == "k20" else ("",)
    s0 = OLDSEEDS[k][0]
    unit(f"old-{k}-s{s0}", k, [(f"old-{k}-s{s0}", (old(k, s0), "adapter_model.safetensors"))], base=True)   # + raw model
    unit(f"a0-{k}-r64", k, [(f"a0-{k}-r64", new(f"a0-{k}-r64"))])
    for s in OLDSEEDS[k][1:]:
        unit(f"old-{k}-s{s}", k, [(f"old-{k}-s{s}", (old(k, s), "adapter_model.safetensors"))])
    if HAS_B3:
        unit(f"b3-{k}-shared1", k, [(f"b3-{k}-shared1", der(f"b3-{k}-shared1"))])
        unit(f"b3-{k}-noshared1", k, [(f"b3-{k}-noshared1", der(f"b3-{k}-noshared1"))])
    unit(f"c1-{k}-r64", k, [(f"c1-{k}-r64", der(f"c1-{k}-r64"))])
    unit(f"c1-{k}-r1", k, [(f"c1-{k}-r1", der(f"c1-{k}-r1"))])
    if k == "k20":
        unit(f"c1-{k}-r4+r16", k, [(f"c1-{k}-r4", der(f"c1-{k}-r4")), (f"c1-{k}-r16", der(f"c1-{k}-r16"))])
    else:
        unit(f"c1-{k}-r16", k, [(f"c1-{k}-r16", der(f"c1-{k}-r16"))])
    unit(f"b2-{k}-r1", k, [(f"b2-{k}-r1", der(f"b2-{k}-r1"))])
    unit(f"b2-{k}-r4+r16", k, [(f"b2-{k}-r4", der(f"b2-{k}-r4")), (f"b2-{k}-r16", der(f"b2-{k}-r16"))])
    if HAS_B3:
        unit(f"b3-{k}-mean-r64", k, [(f"b3-{k}-mean-r64", der(f"b3-{k}-mean-r64"))])
        unit(f"b3-{k}-shared4", k, [(f"b3-{k}-shared4", der(f"b3-{k}-shared4"))])
        unit(f"b3-{k}-shared1-L24-41", k, [(f"b3-{k}-shared1-L24-41", der(f"b3-{k}-shared1-L24-41"))])
    for u in ck:
        unit(f"a0-{k}-r64-u{u}", k, [(f"a0-{k}-r64-u{u}", new(f"a0-{k}-r64", u))])
    for lr in lrs:
        unit(f"a1-{k}-r16+r4{lr}", k, [(f"a1-{k}-r16{lr}", new(f"a1-{k}-r16{lr}")), (f"a1-{k}-r4{lr}", new(f"a1-{k}-r4{lr}"))])
        unit(f"a1-{k}-r1{lr}", k, [(f"a1-{k}-r1{lr}", new(f"a1-{k}-r1{lr}"))])
    for t in (("mlp", "attn", "L0-13", "L14-27", "L28-41", "b1mods") if k == "k20" else ("mlp", "attn")):
        for lr in lrs:
            unit(f"a2-{k}-r16-{t}{lr}", k, [(f"a2-{k}-r16-{t}{lr}", new(f"a2-{k}-r16-{t}{lr}"))])
    for n in ("top23", "top50", "mlpin", "scale0.5", "scale2"):
        unit(f"b2-{k}-{n}", k, [(f"b2-{k}-{n}", der(f"b2-{k}-{n}"))])
    unit(f"b2-{k}-top90+r4-top90", k, [(f"b2-{k}-top90", der(f"b2-{k}-top90")), (f"b2-{k}-r4-top90", der(f"b2-{k}-r4-top90"))])
    if HAS_B3:
        unit(f"b3-{k}-mean-r4+r16", k, [(f"b3-{k}-mean-r4", der(f"b3-{k}-mean-r4")), (f"b3-{k}-mean-r16", der(f"b3-{k}-mean-r16"))])

names = [u[0] for u in units]
assert len(names) == len(set(names))
lists = {1: [], 2: []}
# interleave K20 and K100 so that both data conditions progress
k20 = [u for u in units if u[1] == "k20"]
k100 = [u for u in units if u[1] == "k100"]
ordered = []
for i in range(max(len(k20), len(k100))):
    for group in (k20, k100):
        if i < len(group):
            ordered.append(group[i])
for name, k, members, base in ordered:
    for run in (1, 2):
        job = f"ev.{name}.run{run}"
        env = {**ENV, "ADAPTERS": " ".join(f"{n}={p}" for n, (p, _) in members), "TASKS": TE + "/data/manifest/dev140.json",
               "DECODING": "greedy", "INCLUDE_BASE": "1" if base else "0", "OUT": f"{R}/eval/{name}.run{run}", "VERIFIED": VER[k]}
        assert all("'" not in v for v in env.values())
        res = {**RES, "TIME": LIMIT[len(members) + (1 if base else 0)]}
        (out / "jobs" / f"{job}.env").write_text("".join(f"{a}='{b}'\n" for a, b in env.items()))
        (out / "jobs" / f"{job}.res").write_text("".join(f"{a}={b}\n" for a, b in res.items())
            + "".join(f"REQUIRE={p}/{req}\n" for _, (p, req) in members) + f"FORBID={R}/eval/{name}.run{run}\n")
        lists[run].append(job)
for run in (1, 2):
    (out / "lists" / f"ev{run}.list").write_text("".join(j + "\n" for j in lists[run]))
print(site, "units", len(units), "jobs", 2 * len(units), "adapter evals per run", sum(len(u[2]) + (1 if u[3] else 0) for u in units))
