#!/usr/bin/env python3
"""gen_wsp_queue.py OUT_DIR — writes the hpg job queue for the WSP-vs-SFT experiments (groups A, B).

For every job NAME it writes  jobs/NAME.env  (KEY=VALUE lines exported before sbatch)
                      and    jobs/NAME.res  (sbatch resources: KEY=VALUE; REQUIRE=<path> lines must exist,
                                             FORBID=<path> lines must not exist at submit time)
plus ordered lists  lists/<wave>.list  (one job name per line) and  lists/prep_<K>.cells (cells of one CPU prep wave).
Nothing is submitted here. Paths are hpg paths; all literals.
"""
import sys
from pathlib import Path

H = "/blue/fsu-compsci-dept/xc25.fsu/hq"
A = H + "/ICML27_agent_distillation"
T = H + "/ICML27_runs/wsp_946b2a9"
TE = H + "/ICML27_runs/wsp_4ba5d0e"   # eval tree: + E-series serving fix (TRITON_ATTN)
R = H + "/ICML27_wsp"
Q = R + "/queue"
MODEL = H + "/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
OLD = A + "/reports/runs/baselines"
COMMON = {
    "PROJECT_ROOT": T, "CONFIG": T + "/configs/base_e4b.yaml", "MODEL_PATH": MODEL,
    "ALFWORLD_DATA": H + "/tools/alfworld-data-empty", "VLLM_CACHE_ROOT": H + "/tools/vllm-cache",
    "TRITON_CACHE_DIR": H + "/tools/triton-cache", "VLLM_NO_USAGE_STATS": "1", "WORKERS": "16",
    "BATCH_INVARIANT": "1", "SED_HASH_CACHE_DIR": H + "/.sed-hash-receipts",
}
KCONF = {  # per data condition: verified view, epochs, reference lr, world size, train resources
    "k20": dict(verified=A + "/data/verified/seeds_full_train_r2/subsets/k20", epochs="32", lr="5e-5", gpus=2,
                cpus=4, mem="30G", time="01:30:00", total_updates=320, ckpts="10 40 80 160"),
    "k100": dict(verified=A + "/data/verified/seeds_full_train_r2/subsets/k100", epochs="64", lr="1e-4", gpus=2,
                 cpus=4, mem="36G", time="03:30:00", total_updates=832, ckpts="26 104 208 416"),
}
out = Path(sys.argv[1])
(out / "jobs").mkdir(parents=True, exist_ok=True)
(out / "lists").mkdir(exist_ok=True)
cells = {}      # cell name -> dict(k=..., env=...)
jobs = {}       # job name -> (env dict, res dict, requires, forbids)


def cell(name, k, *, lr=None, seed=0, rank=None, targets=None, layers=None, modules=None, ckpts=None):
    c = KCONF[k]
    env = {**COMMON, "KANG_OUT": f"{R}/cells/{name}", "VERIFIED": c["verified"], "KANG_EPOCHS": c["epochs"],
           "KANG_SCORING": "shared_history", "NGPUS": str(c["gpus"]), "KANG_LR": lr or c["lr"], "KANG_WARMUP": "0",
           "KANG_TRAINING_SEED": str(seed)}
    if rank is not None:
        env["KANG_LORA_RANK"] = str(rank)
    if targets:
        env["KANG_LORA_TARGETS"] = targets
    if layers:
        env["KANG_LORA_LAYERS"] = layers
    if modules:
        env["KANG_LORA_MODULES"] = modules
    if ckpts:
        env["KANG_CHECKPOINT_UPDATES"] = ckpts
    cells[name] = dict(k=k, env=env)
    res = dict(ACCOUNT="", PARTITION="hpg-b200", GPUS=str(c["gpus"]), CPUS=str(c["cpus"]), MEM=c["mem"], TIME=c["time"],
               SCRIPT="scripts/hpg_kang_wsp.slurm")
    jobs[f"train.{name}"] = ({**env, "KANG_MODE": "train"}, res, [f"{Q}/done/prep.{name}"],
                             [f"{R}/cells/{name}/result.json"])


MLP, ATT = "gate_proj,up_proj,down_proj", "q_proj,k_proj,v_proj,o_proj"
# ---- group A, K20 (reference lr 5e-5; second lr 2e-4 for every restricted arm) ----
cell("a0-k20-r64", "k20", ckpts=KCONF["k20"]["ckpts"])
for r in (16, 4, 1):
    for lr in ("5e-5", "2e-4"):
        cell(f"a1-k20-r{r}-lr{lr}", "k20", rank=r, lr=lr)
for tag, kw in (("mlp", dict(targets=MLP)), ("attn", dict(targets=ATT)), ("L0-13", dict(layers="0-13")),
                ("L14-27", dict(layers="14-27")),
                # E4B layers 24-41 share K/V (no separate k_proj / v_proj), so this band lists the five types that exist
                ("L28-41", dict(layers="28-41", targets="q_proj,o_proj,gate_proj,up_proj,down_proj")),
                ("b1mods", dict(modules="22.down_proj,23.v_proj,41.q_proj"))):
    for lr in ("5e-5", "2e-4"):
        cell(f"a2-k20-r16-{tag}-lr{lr}", "k20", rank=16, lr=lr, **kw)
# ---- group A, K100 (reference lr 1e-4) ----
cell("a0-k100-r64", "k100", ckpts=KCONF["k100"]["ckpts"])
for r in (16, 4, 1):
    cell(f"a1-k100-r{r}", "k100", rank=r)
cell("a2-k100-r16-mlp", "k100", rank=16, targets=MLP)
cell("a2-k100-r16-attn", "k100", rank=16, targets=ATT)


def evaljob(name, adapters, k, *, include_base=False, time=None):
    """adapters: list of (served name, adapter dir)."""
    env = {**COMMON, "ADAPTERS": " ".join(f"{n}={p}" for n, p in adapters), "PROJECT_ROOT": TE, "CONFIG": TE + "/configs/base_e4b.yaml", "TASKS": TE + "/data/manifest/dev140.json",
           "DECODING": "greedy", "INCLUDE_BASE": "1" if include_base else "0", "OUT": f"{R}/eval/{name}",
           "VERIFIED": KCONF[k]["verified"]}
    n = len(adapters) + (1 if include_base else 0)
    res = dict(ACCOUNT="", PARTITION="hpg-b200", GPUS="1", CPUS="4", MEM="24G",
               TIME=time or f"{(18 * n + 20) // 60:02d}:{(18 * n + 20) % 60:02d}:00", SCRIPT="scripts/hpg_eval_adapters.slurm")
    jobs[f"eval.{name}"] = (env, res, [p + "/adapter_model.safetensors" for _, p in adapters], [f"{R}/eval/{name}"])


def old(cellname):
    return f"{OLD}/{cellname}/stages/train-sft/attempt-000/output/checkpoint"


def new(cellname, update=None):
    base = f"{R}/cells/{cellname}/stages/train-sft/attempt-000/output"
    return f"{base}/checkpoints/update-{update}" if update else f"{base}/checkpoint"


# ---- evaluation of the six existing reference adapters (two runs each; run 1 also evaluates the raw model) ----
OLD20 = [("old-k20-s0", old("e4b-tsft-k20-e32-lr5e-5")), ("old-k20-s1", old("e4b-tsft-k20-e32-lr5e-5-s1")),
         ("old-k20-s2", old("e4b-tsft-k20-e32-lr5e-5-s2"))]
OLD100 = [("old-k100-s0", old("e4b-tsft-k100-e64-lr1e-4")), ("old-k100-s1", old("e4b-tsft-k100-e64-lr1e-4-s1")),
          ("old-k100-s2", old("e4b-tsft-k100-e64-lr1e-4-s2"))]
for run in (1, 2):
    evaljob(f"old-k20.run{run}", OLD20, "k20", include_base=(run == 1))
    evaljob(f"old-k100.run{run}", OLD100, "k100", include_base=(run == 2))
# ---- evaluation of the new cells (one adapter group per job, two runs) ----
groups = {
    "a0-k20": [("a0-k20-r64", new("a0-k20-r64"))] + [(f"a0-k20-r64-u{u}", new("a0-k20-r64", u)) for u in (10, 40, 80, 160)],
    "a1-k20-lr5e-5": [(f"a1-k20-r{r}-lr5e-5", new(f"a1-k20-r{r}-lr5e-5")) for r in (16, 4, 1)],
    "a1-k20-lr2e-4": [(f"a1-k20-r{r}-lr2e-4", new(f"a1-k20-r{r}-lr2e-4")) for r in (16, 4, 1)],
    "a2-k20-types": [(f"a2-k20-r16-{t}-lr{lr}", new(f"a2-k20-r16-{t}-lr{lr}")) for t in ("mlp", "attn") for lr in ("5e-5", "2e-4")],
    "a2-k20-bands-lr5e-5": [(f"a2-k20-r16-{t}-lr5e-5", new(f"a2-k20-r16-{t}-lr5e-5")) for t in ("L0-13", "L14-27", "L28-41", "b1mods")],
    "a2-k20-bands-lr2e-4": [(f"a2-k20-r16-{t}-lr2e-4", new(f"a2-k20-r16-{t}-lr2e-4")) for t in ("L0-13", "L14-27", "L28-41", "b1mods")],
    "a0a1-k100": [(n, new(n)) for n in ("a0-k100-r64", "a1-k100-r16", "a1-k100-r4", "a1-k100-r1")],
    "a2-k100": [(n, new(n)) for n in ("a2-k100-r16-mlp", "a2-k100-r16-attn")],
    "a0-k100-ckpt": [(f"a0-k100-r64-u{u}", new("a0-k100-r64", u)) for u in (26, 104, 208, 416)],
}
for g, ads in groups.items():
    for run in (1, 2):
        evaljob(f"{g}.run{run}", ads, "k100" if "k100" in g else "k20")

for name, (env, res, req, forb) in jobs.items():
    (out / "jobs" / f"{name}.env").write_text("".join(f"{k}='{v}'\n" for k, v in env.items()))
    (out / "jobs" / f"{name}.res").write_text("".join(f"{k}={v}\n" for k, v in res.items())
                                              + "".join(f"REQUIRE={p}\n" for p in req) + "".join(f"FORBID={p}\n" for p in forb))
for name, c in cells.items():  # env used by the CPU preparation wave (same variables as the train job, mode added there)
    (out / "jobs" / f"prep.{name}.env").write_text("".join(f"{k}='{v}'\n" for k, v in c["env"].items()))
assert all("'" not in v for e, *_ in jobs.values() for v in e.values())
k20 = [n for n, c in cells.items() if c["k"] == "k20"]
(out / "lists" / "prep_k20a.cells").write_text("".join(n + "\n" for n in k20[0::2]))   # a0 first
(out / "lists" / "prep_k20b.cells").write_text("".join(n + "\n" for n in k20[1::2]))
(out / "lists" / "prep_k100.cells").write_text("".join(n + "\n" for n, c in cells.items() if c["k"] == "k100"))
order = [n for n in jobs if n.startswith("eval.old-") and n.endswith("run1")]
order += [f"train.{n}" for n in cells if n.startswith("a0-")]
order += [n for n in jobs if n.startswith("eval.old-") and n.endswith("run2")]
order += [f"train.{n}" for n in cells if n.startswith("a1-k20")] + [f"train.{n}" for n in cells if n.startswith("a1-k100")]
order += [f"train.{n}" for n in cells if n.startswith("a2-")]
order += [n for n in jobs if n.startswith("eval.") and not n.startswith("eval.old-") and n.endswith("run1")]
order += [n for n in jobs if n.startswith("eval.") and not n.startswith("eval.old-") and n.endswith("run2")]
assert sorted(order) == sorted(jobs), "every job is listed exactly once"
(out / "lists" / "main.list").write_text("".join(n + "\n" for n in order))
print(f"cells {len(cells)}  jobs {len(jobs)}  (train {sum(n.startswith('train.') for n in jobs)}, eval {sum(n.startswith('eval.') for n in jobs)})")
