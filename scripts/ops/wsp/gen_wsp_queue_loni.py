#!/usr/bin/env python3
"""gen_wsp_queue_loni.py OUT_DIR — LONI (QB-D, 2 x A100 per gpu2 node) lane of the WSP-vs-SFT experiments.

Same cells and recipes as the hpg queue (gen_wsp_queue.py): group A for K20 and K100, plus group B2 derived from
the LONI copies of the seed-0 reference adapters. Everything trained on LONI is evaluated on LONI, next to LONI
re-evaluations of the reference adapters and the raw model, so every comparison stays inside one cluster.
Per job NAME: jobs/NAME.env (exported variables, single-quoted) and jobs/NAME.res (KIND, GPUS, CPUS, TIME, STAGE_SCRIPT,
REQUIRE= / FORBID= paths). Lists: lists/main.list, lists/b.list, lists/prep_*.cells; derive/derive_b.sh.
"""
import sys
from pathlib import Path

W = "/work/xueqic/hq"
T = W + "/ICML27_runs/wsp_946b2a9"           # train / eval tree
TT = W + "/ICML27_runs/wsp_a788e7e"          # tool tree
R = W + "/ICML27_wsp"
Q = R + "/queue"
MODEL = "/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
OLD = W + "/ICML27_data/reports/runs/baselines"
VER = {k: f"{W}/ICML27_data/verified/seeds_full_train_r2/subsets/{k}" for k in ("k20", "k100")}
TRAIN_COMMON = {
    "PROJECT_ROOT": T, "EXPECT_COMMIT": "946b2a9", "HPG_CONFIG": W + "/ICML27_loni.env", "CONFIG": T + "/configs/base_e4b.yaml",
    "MODEL_PATH": MODEL, "WORKERS": "16", "BATCH_INVARIANT": "1", "SED_HASH_CACHE_DIR": W + "/ICML27_v0_runs/sed-hash-receipts",
}
EVAL_COMMON = {
    "PROJECT_ROOT": T, "EXPECT_COMMIT": "946b2a9", "HPG_CONFIG": W + "/ICML27_runs/eval_46b8527/configs_ec704e4/loni_eval.env",
    "CONFIG": T + "/configs/base_e4b.yaml", "MODEL_PATH": MODEL, "WORKERS": "16", "BATCH_INVARIANT": "1",
    "VLLM_CACHE_ROOT": W + "/tools/vllm-cache", "TRITON_CACHE_DIR": W + "/tools/triton-cache", "VLLM_NO_USAGE_STATS": "1",
    "SED_HASH_CACHE_DIR": W + "/ICML27_v0_runs/sed-hash-receipts", "TASKS": T + "/data/manifest/dev140.json",
    "DECODING": "greedy",
}
KCONF = {"k20": dict(epochs="32", lr="5e-5", time="01:15:00", ckpts="10 40 80 160"),
         "k100": dict(epochs="64", lr="1e-4", time="06:30:00", ckpts="26 104 208 416")}
out = Path(sys.argv[1])
for d in ("jobs", "lists", "derive"):
    (out / d).mkdir(parents=True, exist_ok=True)
cells, jobs = {}, {}


def cell(name, k, *, lr=None, rank=None, targets=None, layers=None, modules=None, ckpts=None):
    c = KCONF[k]
    env = {**TRAIN_COMMON, "KANG_OUT": f"{R}/cells/{name}", "VERIFIED": VER[k], "KANG_EPOCHS": c["epochs"],
           "KANG_SCORING": "shared_history", "NGPUS": "2", "KANG_LR": lr or c["lr"], "KANG_WARMUP": "0", "KANG_TRAINING_SEED": "0"}
    for key, val in (("KANG_LORA_RANK", rank), ("KANG_LORA_TARGETS", targets), ("KANG_LORA_LAYERS", layers),
                     ("KANG_LORA_MODULES", modules), ("KANG_CHECKPOINT_UPDATES", ckpts)):
        if val is not None:
            env[key] = str(val)
    cells[name] = dict(k=k, env=env)
    jobs[f"train.{name}"] = ({**env, "KANG_MODE": "train"},
                             dict(KIND="gpu", GPUS="2", CPUS="16", TIME=c["time"], STAGE_SCRIPT="scripts/hpg_kang_wsp.slurm"),
                             [f"{Q}/done/prep.{name}"], [f"{R}/cells/{name}/result.json"])


MLP, ATT = "gate_proj,up_proj,down_proj", "q_proj,k_proj,v_proj,o_proj"
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
cell("a0-k100-r64", "k100", ckpts=KCONF["k100"]["ckpts"])
for r in (16, 4, 1):
    cell(f"a1-k100-r{r}", "k100", rank=r)
cell("a2-k100-r16-mlp", "k100", rank=16, targets=MLP)
cell("a2-k100-r16-attn", "k100", rank=16, targets=ATT)


def evaljob(name, adapters, k, *, include_base=False, require=None):
    env = {**EVAL_COMMON, "ADAPTERS": " ".join(f"{n}={p}" for n, p in adapters), "INCLUDE_BASE": "1" if include_base else "0",
           "OUT": f"{R}/eval/{name}", "VERIFIED": VER[k]}
    mins = 28 * (len(adapters) + (1 if include_base else 0)) + 20      # A100: 27 min for one adapter (job 1163511)
    res = dict(KIND="gpu", GPUS="1", CPUS="8", TIME=f"{mins // 60:02d}:{mins % 60:02d}:00", STAGE_SCRIPT="scripts/hpg_eval_adapters.slurm")
    jobs[f"eval.{name}"] = (env, res, require or [p + "/adapter_model.safetensors" for _, p in adapters], [f"{R}/eval/{name}"])


def old(cellname):
    return f"{OLD}/{cellname}/stages/train-sft/attempt-000/output/checkpoint"


def new(cellname, update=None):
    base = f"{R}/cells/{cellname}/stages/train-sft/attempt-000/output"
    return f"{base}/checkpoints/update-{update}" if update else f"{base}/checkpoint"


for run in (1, 2):   # the serving check draws its prompts from VERIFIED, so K20 and K100 adapters go in separate jobs
    evaljob(f"old-k20.run{run}", [("old-k20-s0", old("e4b-tsft-k20-e32-lr5e-5"))], "k20", include_base=True)
    evaljob(f"old-k100.run{run}", [("old-k100-s0", old("e4b-tsft-k100-e64-lr1e-4")),
                                   ("old-k100-s2", old("e4b-tsft-k100-e64-lr1e-4-s2"))], "k100")
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

# ---- group B2 on LONI: derived from the LONI copies of the seed-0 adapters ----
SRC = {"k20": old("e4b-tsft-k20-e32-lr5e-5"), "k100": old("e4b-tsft-k100-e64-lr1e-4")}
derive, bgroups = [], {}
for k in ("k20", "k100"):
    s0 = SRC[k]
    for r in (1, 4, 16):
        derive.append((f"b2-{k}-r{r}", f"truncate {s0} --rank {r}"))
    for m in (23, 50, 90):
        derive.append((f"b2-{k}-top{m}", f"truncate {s0} --top-modules {m}"))
    derive.append((f"b2-{k}-mlpin", f"truncate {s0} --targets gate_proj,up_proj"))
    derive.append((f"b2-{k}-r4-top90", f"truncate {s0} --rank 4 --top-modules 90"))
    derive.append((f"b2-{k}-scale0.5", f"scale {s0} --factor 0.5"))
    derive.append((f"b2-{k}-scale2", f"scale {s0} --factor 2"))
    bgroups[f"b2-{k}-rank"] = [f"b2-{k}-r1", f"b2-{k}-r4", f"b2-{k}-r16", f"b2-{k}-r4-top90"]
    bgroups[f"b2-{k}-sites"] = [f"b2-{k}-top23", f"b2-{k}-top50", f"b2-{k}-top90", f"b2-{k}-mlpin"]
    bgroups[f"b2-{k}-scale"] = [f"b2-{k}-scale0.5", f"b2-{k}-scale2"]
border = []
for g, names in bgroups.items():
    for run in (1, 2):
        evaljob(f"{g}.run{run}", [(n, f"{R}/derived/{n}/checkpoint") for n in names], "k100" if "k100" in g else "k20",
                require=[f"{R}/derived/{n}/checkpoint/training_state.json" for n in names])
        border.append((run, f"eval.{g}.run{run}"))
sh = ["#!/bin/bash",
      "# derive_b.sh (LONI) — group B2 derivations on CPU. Idempotent; half-written outputs are moved to derived/_failed.",
      "set -u",
      f"TT={TT}; R={R}; Q={Q}; PY={W}/tc-alignment/.venv/bin/python",
      'cd "$TT" || exit 2',
      'export PYTHONPATH="$TT/src:$TT" CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
      'mkdir -p "$R/derived/_failed" "$R/analysis" "$Q/done" "$Q/logs"; fail=0',
      'dv() { local name="$1"; shift',
      '  if [ -f "$R/derived/$name/checkpoint/training_state.json" ]; then echo "SKIP $name"; return; fi',
      '  if [ -e "$R/derived/$name" ]; then mv -- "$R/derived/$name" "$R/derived/_failed/$name.$(date +%s)"; fi',
      '  if "$PY" -m tools.wsp_adapter_ops "$@" --out "$R/derived/$name" > "$Q/logs/derive.$name.log" 2>&1; then echo "$(date +%T) DERIVED $name"',
      '  else echo "$(date +%T) DERIVE_FAIL $name"; fail=1; fi; }']
sh += [f"dv {n} {c}" for n, c in derive]
sh += ['if [ "$fail" = 0 ]; then : > "$Q/done/derive.b"; echo "DERIVE_B_DONE"; else echo "DERIVE_B_FAILED"; exit 1; fi']
(out / "derive" / "derive_b.sh").write_text("\n".join(sh) + "\n")

assert all("'" not in v for e, *_ in jobs.values() for v in e.values())
for name, (env, res, req, forb) in jobs.items():
    (out / "jobs" / f"{name}.env").write_text("".join(f"{k}='{v}'\n" for k, v in env.items()))
    (out / "jobs" / f"{name}.res").write_text("".join(f"{k}={v}\n" for k, v in res.items())
                                              + "".join(f"REQUIRE={p}\n" for p in req) + "".join(f"FORBID={p}\n" for p in forb))
for name, c in cells.items():
    (out / "jobs" / f"prep.{name}.env").write_text("".join(f"{k}='{v}'\n" for k, v in c["env"].items()))
k20 = [n for n, c in cells.items() if c["k"] == "k20"]
(out / "lists" / "prep_k20a.cells").write_text("".join(n + "\n" for n in k20[0::2]))
(out / "lists" / "prep_k20b.cells").write_text("".join(n + "\n" for n in k20[1::2]))
(out / "lists" / "prep_k100.cells").write_text("".join(n + "\n" for n, c in cells.items() if c["k"] == "k100"))
a_jobs = [n for n in jobs if not n.startswith("eval.b2-")]
order = [n for n in a_jobs if n.startswith("eval.old-") and n.endswith("run1")]
order += [f"train.{n}" for n in cells if n.startswith("a0-")]
order += [f"train.{n}" for n in cells if n.startswith("a1-k100") or n.startswith("a2-k100")]     # long ones early
order += [f"train.{n}" for n in cells if n.startswith("a1-k20")] + [f"train.{n}" for n in cells if n.startswith("a2-k20")]
order += [n for n in a_jobs if n.startswith("eval.") and not n.startswith("eval.old-") and n.endswith("run1")]
order += [n for n in a_jobs if n.startswith("eval.") and n.endswith("run2")]
assert sorted(order) == sorted(a_jobs)
(out / "lists" / "main.list").write_text("".join(n + "\n" for n in order))
(out / "lists" / "b.list").write_text("".join(j + "\n" for _, j in sorted(border, key=lambda t: t[0])))
print(f"cells {len(cells)}  jobs {len(jobs)}  derived {len(derive)}")
