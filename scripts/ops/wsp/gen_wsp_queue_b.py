# NOTE 11-16: on the clusters every jobs/eval.*.env was repointed from wsp_946b2a9 to the eval tree wsp_4ba5d0e
#            (E-series serving fix); regenerate eval jobs with that tree.
#!/usr/bin/env python3
"""gen_wsp_queue_b.py OUT_DIR — group B of the WSP-vs-SFT experiments (no training).

Writes  derive/derive_b.sh   : CPU script (run inside tc_wsp_cpu.slurm) that builds every derived adapter with
                               tools/wsp_adapter_ops.py (tree wsp_ee07eab) and the spectrum / overlap reports;
        jobs/eval.b-*.env/.res: evaluation jobs for the derived adapters (tree wsp_83c99b4, same path as group A);
        lists/b.list         : their order.
Sources are the six existing trajectory-SFT adapters (K20 / K100, seeds 0-2). All paths are hpg literals.
"""
import sys
from pathlib import Path

H = "/blue/fsu-compsci-dept/xc25.fsu/hq"
A = H + "/ICML27_agent_distillation"
T = H + "/ICML27_runs/wsp_946b2a9"          # train / eval tree
TT = H + "/ICML27_runs/wsp_a788e7e"         # tool tree (derivations)
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
VER = {"k20": A + "/data/verified/seeds_full_train_r2/subsets/k20", "k100": A + "/data/verified/seeds_full_train_r2/subsets/k100"}
CELL = {"k20": "e4b-tsft-k20-e32-lr5e-5", "k100": "e4b-tsft-k100-e64-lr1e-4"}


def old(k, s):
    return f"{OLD}/{CELL[k]}{'' if s == 0 else f'-s{s}'}/stages/train-sft/attempt-000/output/checkpoint"


out = Path(sys.argv[1])
for d in ("jobs", "lists", "derive"):
    (out / d).mkdir(parents=True, exist_ok=True)
derive, analysis, groups = [], [], {}
for k in ("k20", "k100"):
    s0, seeds = old(k, 0), " ".join(old(k, s) for s in (0, 1, 2))
    for r in (1, 4, 16):
        derive.append((f"b2-{k}-r{r}", f"truncate {s0} --rank {r}"))
    for m in (23, 50, 90):
        derive.append((f"b2-{k}-top{m}", f"truncate {s0} --top-modules {m}"))
    derive.append((f"b2-{k}-mlpin", f"truncate {s0} --targets gate_proj,up_proj"))
    derive.append((f"b2-{k}-r4-top90", f"truncate {s0} --rank 4 --top-modules 90"))
    derive.append((f"b2-{k}-scale0.5", f"scale {s0} --factor 0.5"))
    derive.append((f"b2-{k}-scale2", f"scale {s0} --factor 2"))
    for r in (64, 16, 4):
        derive.append((f"b3-{k}-mean-r{r}", f"average {seeds} --rank {r}"))
    for s in (0, 1, 2):
        analysis.append((f"spectrum.{k}-s{s}", f"spectrum {old(k, s)}"))
    for a, b in ((0, 1), (0, 2), (1, 2)):
        analysis.append((f"overlap.{k}-s{a}.{k}-s{b}", f"overlap {old(k, a)} {old(k, b)} --k 1 4 16"))
    groups[f"b2-{k}-rank"] = [f"b2-{k}-r1", f"b2-{k}-r4", f"b2-{k}-r16", f"b2-{k}-r4-top90"]
    groups[f"b2-{k}-sites"] = [f"b2-{k}-top23", f"b2-{k}-top50", f"b2-{k}-top90", f"b2-{k}-mlpin"]
    groups[f"b3-{k}-mean"] = [f"b3-{k}-mean-r64", f"b3-{k}-mean-r16", f"b3-{k}-mean-r4"]
    groups[f"b2-{k}-scale"] = [f"b2-{k}-scale0.5", f"b2-{k}-scale2"]
for s in (0, 1, 2):
    analysis.append((f"overlap.k20-s{s}.k100-s{s}", f"overlap {old('k20', s)} {old('k100', s)} --k 1 4 16"))

sh = ["#!/bin/bash",
      "# derive_b.sh — group B derivations (CPU). Idempotent: finished outputs are skipped, half-written ones are moved",
      "# to derived/_failed (never removed). Ends DERIVE_B_DONE and writes queue/done/derive.b when everything exists.",
      "set -u",
      f"TT={TT}; R={R}; Q={Q}; PY={H}/tc-alignment/.venv/bin/python",
      'cd "$TT" || exit 2',
      'export PYTHONPATH="$TT/src:$TT" CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',
      'mkdir -p "$R/derived/_failed" "$R/analysis" "$Q/done"; fail=0',
      'dv() { local name="$1"; shift',
      '  if [ -f "$R/derived/$name/checkpoint/training_state.json" ]; then echo "SKIP $name"; return; fi',
      '  if [ -e "$R/derived/$name" ]; then mv -- "$R/derived/$name" "$R/derived/_failed/$name.$(date +%s)"; fi',
      '  if "$PY" -m tools.wsp_adapter_ops "$@" --out "$R/derived/$name" > "$Q/logs/derive.$name.log" 2>&1; then echo "$(date +%T) DERIVED $name"',
      '  else echo "$(date +%T) DERIVE_FAIL $name"; fail=1; fi; }',
      'an() { local name="$1"; shift',
      '  if [ -f "$R/analysis/$name.json" ]; then echo "SKIP $name"; return; fi',
      '  if "$PY" -m tools.wsp_adapter_ops "$@" --out "$R/analysis/$name.json" > "$Q/logs/analysis.$name.log" 2>&1; then echo "$(date +%T) ANALYSED $name"',
      '  else echo "$(date +%T) ANALYSIS_FAIL $name"; fail=1; fi; }']
sh += [f"dv {n} {c}" for n, c in derive] + [f"an {n} {c}" for n, c in analysis]
sh += ['if [ "$fail" = 0 ]; then : > "$Q/done/derive.b"; echo "DERIVE_B_DONE"; else echo "DERIVE_B_FAILED"; exit 1; fi']
(out / "derive" / "derive_b.sh").write_text("\n".join(sh) + "\n")
(out / "derive" / "tc_wsp_cpu.slurm").write_text("""#!/bin/bash
#SBATCH --job-name=wsp-cpu
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
# tc_wsp_cpu.slurm — runs one staged CPU script of the WSP queue (CMD_FILE, absolute path under the queue dir).
set -u
: "${CMD_FILE:?}"
case "$CMD_FILE" in /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp/queue/derive/*.sh) ;; *) echo "REFUSE $CMD_FILE"; exit 2;; esac
exec bash "$CMD_FILE"
""")
order = []
for g, names in groups.items():
    k = "k100" if "k100" in g else "k20"
    for run in (1, 2):
        job = f"eval.{g}.run{run}"
        env = {**COMMON, "ADAPTERS": " ".join(f"{n}={R}/derived/{n}/checkpoint" for n in names),
               "TASKS": T + "/data/manifest/dev140.json", "DECODING": "greedy", "INCLUDE_BASE": "0",
               "OUT": f"{R}/eval/{g}.run{run}", "VERIFIED": VER[k]}
        mins = 18 * len(names) + 20
        res = dict(ACCOUNT="", PARTITION="hpg-b200", GPUS="1", CPUS="4", MEM="24G", TIME=f"{mins // 60:02d}:{mins % 60:02d}:00",
                   SCRIPT="scripts/hpg_eval_adapters.slurm")
        (out / "jobs" / f"{job}.env").write_text("".join(f"{a}='{b}'\n" for a, b in env.items()))
        (out / "jobs" / f"{job}.res").write_text("".join(f"{a}={b}\n" for a, b in res.items())
            + "".join(f"REQUIRE={R}/derived/{n}/checkpoint/training_state.json\n" for n in names) + f"FORBID={R}/eval/{g}.run{run}\n")
        order.append((run, job))
(out / "lists" / "b.list").write_text("".join(j + "\n" for _, j in sorted(order, key=lambda t: t[0])))
print(f"derived {len(derive)}  analyses {len(analysis)}  eval jobs {len(order)}")
