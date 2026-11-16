# NOTE 11-16: on the clusters every jobs/eval.*.env was repointed from wsp_946b2a9 to the eval tree wsp_4ba5d0e
#            (E-series serving fix); regenerate eval jobs with that tree.
#!/usr/bin/env python3
"""gen_wsp_queue_c.py SITE OUT_DIR   (SITE = hpg | loni) — group C1 of the WSP-vs-SFT experiments.

C1 = one-step gradient edit: G_m = dL/dW_m of the SFT objective on D_K at the raw model (dropout off), for the 258
modules of the standard recipe; Delta(r, eta) = -eta * (best rank-r approximation of G_m) for every module.
eta is chosen per rank on the TRAINING objective only (grid of target norms ||Delta||_F), never on dev140.
Writes: jobs/prep.sc-<K>.env + lists/prep_sc.cells (score cells prepared by the tool tree, whose seal the tool needs),
        derive/c1_<K>.sh (one GPU job each), jobs/eval.c1-<K>.run{1,2}.env/.res, lists/c.list.
"""
import sys
from pathlib import Path

site, out = sys.argv[1], Path(sys.argv[2])
if site == "hpg":
    H = "/blue/fsu-compsci-dept/xc25.fsu/hq"
    A = H + "/ICML27_agent_distillation"
    R, T, TT = H + "/ICML27_wsp", H + "/ICML27_runs/wsp_946b2a9", H + "/ICML27_runs/wsp_2855df1"
    PY = H + "/tc-alignment/.venv/bin/python"
    MODEL = H + "/tools/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
    OLD = A + "/reports/runs/baselines"
    VER = {k: f"{A}/data/verified/seeds_full_train_r2/subsets/{k}" for k in ("k20", "k100")}
    CACHE = H + "/.sed-hash-receipts"
    BASE = {"MODEL_PATH": MODEL, "ALFWORLD_DATA": H + "/tools/alfworld-data-empty", "VLLM_CACHE_ROOT": H + "/tools/vllm-cache",
            "TRITON_CACHE_DIR": H + "/tools/triton-cache", "VLLM_NO_USAGE_STATS": "1", "WORKERS": "16", "BATCH_INVARIANT": "1",
            "SED_HASH_CACHE_DIR": CACHE}
    PREP = {**BASE, "PROJECT_ROOT": TT, "CONFIG": TT + "/configs/base_e4b.yaml"}
    EVAL = {**BASE, "PROJECT_ROOT": T, "CONFIG": T + "/configs/base_e4b.yaml"}
    ENVSRC = 'source "$PROJECT_ROOT/configs/hpg.env"'
    EVALRES = dict(ACCOUNT="", PARTITION="hpg-b200", GPUS="1", CPUS="4", MEM="24G", SCRIPT="scripts/hpg_eval_adapters.slurm")
    PER = 18
else:
    W = "/work/xueqic/hq"
    R, T, TT = W + "/ICML27_wsp", W + "/ICML27_runs/wsp_946b2a9", W + "/ICML27_runs/wsp_2855df1"
    PY = W + "/tc-alignment/.venv/bin/python"
    MODEL = "/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
    OLD = W + "/ICML27_data/reports/runs/baselines"
    VER = {k: f"{W}/ICML27_data/verified/seeds_full_train_r2/subsets/{k}" for k in ("k20", "k100")}
    CACHE = W + "/ICML27_v0_runs/sed-hash-receipts"
    PREP = {"PROJECT_ROOT": TT, "EXPECT_COMMIT": "2855df1", "HPG_CONFIG": W + "/ICML27_loni.env", "CONFIG": TT + "/configs/base_e4b.yaml",
            "MODEL_PATH": MODEL, "WORKERS": "16", "BATCH_INVARIANT": "1", "SED_HASH_CACHE_DIR": CACHE}
    EVAL = {"PROJECT_ROOT": T, "EXPECT_COMMIT": "946b2a9", "HPG_CONFIG": W + "/ICML27_runs/eval_46b8527/configs_ec704e4/loni_eval.env",
            "CONFIG": T + "/configs/base_e4b.yaml", "MODEL_PATH": MODEL, "WORKERS": "16", "BATCH_INVARIANT": "1",
            "VLLM_CACHE_ROOT": W + "/tools/vllm-cache", "TRITON_CACHE_DIR": W + "/tools/triton-cache", "VLLM_NO_USAGE_STATS": "1",
            "SED_HASH_CACHE_DIR": CACHE}
    ENVSRC = 'source "$HPG_CONFIG"'
    EVALRES = dict(KIND="gpu", GPUS="1", CPUS="8", STAGE_SCRIPT="scripts/hpg_eval_adapters.slurm")
    PER = 28
Q = R + "/queue"
KC = {"k20": dict(epochs="32", lr="5e-5", like=f"{OLD}/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint",
                  ranks=(1, 4, 16, 64), norms=(0.25, 0.5, 1, 2, 4, 8)),
      "k100": dict(epochs="64", lr="1e-4", like=f"{OLD}/e4b-tsft-k100-e64-lr1e-4/stages/train-sft/attempt-000/output/checkpoint",
                   ranks=(1, 16, 64), norms=(0.5, 1, 2, 4, 8))}
for d in ("jobs", "lists", "derive"):
    (out / d).mkdir(parents=True, exist_ok=True)


def q(env):
    assert all("'" not in v for v in env.values())
    return "".join(f"{k}='{v}'\n" for k, v in env.items())


order = []
for k, c in KC.items():
    cell = f"sc-{k}"
    (out / "jobs" / f"prep.{cell}.env").write_text(q({**PREP, "KANG_OUT": f"{R}/cells/{cell}", "VERIFIED": VER[k],
        "KANG_EPOCHS": c["epochs"], "KANG_SCORING": "shared_history", "NGPUS": "2", "KANG_LR": c["lr"], "KANG_WARMUP": "0",
        "KANG_TRAINING_SEED": "0"}))
    ranks = " ".join(str(r) for r in c["ranks"])
    small = " ".join(str(r) for r in c["ranks"] if r != 64)
    norms = " ".join(str(n) for n in c["norms"])
    sh = f"""#!/bin/bash
# c1_{k}.sh ({site}, ONE GPU) — group C1 for {k}: full-batch gradient of the SFT objective at the raw model -> rank-64
# adapter -> truncations -> step size per rank by the TRAINING objective (target norms {norms}) -> final adapters
# derived/c1-{k}-r<rank>. Idempotent (finished pieces are skipped). Ends C1_DONE + marker done/c1.{k}.
set -u
R={R}; Q={Q}; PY={PY}; CELL={R}/cells/sc-{k}; LIKE={c["like"]}
set -a; . "$Q/jobs/prep.sc-{k}.env"; set +a
{ENVSRC}
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p "$R/analysis" "$R/derived/_failed" "$Q/done" "$Q/logs"
[ -f "$Q/done/prep.sc-{k}" ] || {{ echo "C1_FAIL score cell not prepared"; exit 1; }}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
step() {{ echo "$(date '+%F %T') $*"; }}
fresh() {{ if [ -e "$1" ]; then mv -- "$1" "$R/derived/_failed/$(basename "$1").$(date +%s)"; fi; }}
G="$R/derived/c1-{k}-g64"
if [ ! -f "$G/checkpoint/training_state.json" ]; then fresh "$G"; step gradient
  "$PY" -m tools.wsp_onestep gradient --cell "$CELL" --like "$LIKE" --out "$G" --rank 64 || {{ echo "C1_FAIL gradient"; exit 1; }}; fi
for r in {small}; do D="$R/derived/c1-{k}-g$r"
  if [ ! -f "$D/checkpoint/training_state.json" ]; then fresh "$D"; step "truncate $r"
    CUDA_VISIBLE_DEVICES= "$PY" -m tools.wsp_adapter_ops truncate "$G/checkpoint" --out "$D" --rank "$r" || {{ echo "C1_FAIL truncate $r"; exit 1; }}; fi
done
if [ ! -f "$R/analysis/c1-{k}.score.ref.json" ]; then step "score reference adapter"
  "$PY" -m tools.wsp_onestep score --cell "$CELL" --out "$R/analysis/c1-{k}.score.ref.json" --adapter "$LIKE" --scales 1 || {{ echo "C1_FAIL score ref"; exit 1; }}; fi
for r in {ranks}; do D="$R/derived/c1-{k}-g$r"; S="$R/analysis/c1-{k}.norm.g$r.json"; L="$R/analysis/c1-{k}.linesearch.r$r.json"; F="$R/derived/c1-{k}-r$r"
  [ -f "$S" ] || CUDA_VISIBLE_DEVICES= "$PY" -m tools.wsp_adapter_ops spectrum "$D/checkpoint" --out "$S" || {{ echo "C1_FAIL spectrum $r"; exit 1; }}
  scales=$("$PY" -c 'import json,sys; n=json.load(open(sys.argv[1]))["summary"]["squared_frobenius_norm"]**0.5; print(" ".join(repr(float(t)/n) for t in sys.argv[2:]))' "$S" {norms}) || exit 1
  if [ ! -f "$L" ]; then step "line search rank $r: scales $scales"
    "$PY" -m tools.wsp_onestep score --cell "$CELL" --out "$L" --adapter "$D/checkpoint" --scales $scales || {{ echo "C1_FAIL linesearch $r"; exit 1; }}; fi
  eta=$("$PY" -c 'import json,sys; rows=[x for x in json.load(open(sys.argv[1]))["rows"] if x.get("scale", 0) > 0]; b=min(rows,key=lambda x:x["loss"]); print(repr(b["scale"]))' "$L") || exit 1
  if [ ! -f "$F/checkpoint/training_state.json" ]; then fresh "$F"; step "final rank $r eta $eta"
    CUDA_VISIBLE_DEVICES= "$PY" -m tools.wsp_adapter_ops scale "$D/checkpoint" --out "$F" --factor "$eta" || {{ echo "C1_FAIL scale $r"; exit 1; }}; fi
done
: > "$Q/done/c1.{k}"; step C1_DONE
"""
    (out / "derive" / f"c1_{k}.sh").write_text(sh)
    names = [f"c1-{k}-r{r}" for r in c["ranks"]]
    for run in (1, 2):
        job = f"eval.c1-{k}.run{run}"
        env = {**EVAL, "ADAPTERS": " ".join(f"{n}={R}/derived/{n}/checkpoint" for n in names), "TASKS": T + "/data/manifest/dev140.json",
               "DECODING": "greedy", "INCLUDE_BASE": "0", "OUT": f"{R}/eval/c1-{k}.run{run}", "VERIFIED": VER[k]}
        mins = PER * len(names) + 20
        res = {**EVALRES, "TIME": f"{mins // 60:02d}:{mins % 60:02d}:00"}
        (out / "jobs" / f"{job}.env").write_text(q(env))
        (out / "jobs" / f"{job}.res").write_text("".join(f"{a}={b}\n" for a, b in res.items())
            + "".join(f"REQUIRE={R}/derived/{n}/checkpoint/training_state.json\n" for n in names) + f"FORBID={R}/eval/c1-{k}.run{run}\n")
        order.append((run, job))
(out / "lists" / "prep_sc.cells").write_text("sc-k20\nsc-k100\n")
(out / "lists" / "c.list").write_text("".join(j + "\n" for _, j in sorted(order)))
if site == "hpg":
    (out / "derive" / "tc_wsp_gpu.slurm").write_text("""#!/bin/bash
#SBATCH --job-name=wsp-gpu
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
# tc_wsp_gpu.slurm — runs one staged GPU script of the WSP queue (CMD_FILE under queue/derive).
set -u
: "${CMD_FILE:?}"
case "$CMD_FILE" in /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp/queue/derive/*.sh) ;; *) echo "REFUSE $CMD_FILE"; exit 2;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec bash "$CMD_FILE"
""")
print("site", site, "eval jobs", len(order))
