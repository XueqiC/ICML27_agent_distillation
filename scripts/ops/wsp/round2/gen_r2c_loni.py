#!/usr/bin/env python3
"""gen_r2c_loni.py OUT_DIR — LONI jobs of round 2 that use tree wsp_27a83f4 (prereg sections 18-19):
 C2b rule sheet v1 (t2 + dev140; raw t2 reference), a full second ICL set on A100, C5 (two GPU derive scripts),
 W1 functional + W2 (CPU derive script, then dev140 once per derived adapter). Also writes tc_wsp_pass6_loni.sh."""
import os, sys
out = sys.argv[1]
W = "/work/xueqic/hq"; Q = f"{W}/ICML27_wsp/queue"; R = f"{W}/ICML27_wsp"
D = f"{W}/ICML27_data/verified/seeds_full_train_r2/subsets"
MODEL = "/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
T = f"{W}/ICML27_runs/wsp_27a83f4"; TE = f"{W}/ICML27_runs/wsp_4ba5d0e"
EVALENV = f"{W}/ICML27_runs/eval_46b8527/configs_ec704e4/loni_eval.env"
for d in ("jobs", "lists", "derive"): os.makedirs(f"{out}/{d}", exist_ok=True)
env = lambda pairs: "".join(f"{k}='{v}'\n" for k, v in pairs)
common = [("HPG_CONFIG", EVALENV), ("MODEL_PATH", MODEL), ("BATCH_INVARIANT", "1"), ("VLLM_CACHE_ROOT", f"{W}/tools/vllm-cache"),
          ("TRITON_CACHE_DIR", f"{W}/tools/triton-cache"), ("VLLM_NO_USAGE_STATS", "1"),
          ("SED_HASH_CACHE_DIR", f"{W}/ICML27_v0_runs/sed-hash-receipts"), ("DECODING", "greedy")]
def base_job(name, tasks, lim, *, view=None, mode=None, mml=None, extra=None, workers=16, require=()):
    pairs = [("PROJECT_ROOT", T), ("EXPECT_COMMIT", "27a83f4"), ("CONFIG", f"{T}/configs/base_e4b.yaml"), ("WORKERS", str(workers)),
             ("BASE_ONLY", "1"), ("TASKS", tasks), ("OUT", f"{R}/eval/{name}")] + common
    if view: pairs += [("ICL_VIEW", f"{D}/{view}"), ("ICL_MODE", mode), ("MAX_MODEL_LEN", str(mml))]
    if extra: pairs += [("SYSTEM_EXTRA_FILE", extra)]
    open(f"{out}/jobs/ev.{name}.env", "w").write(env(pairs))
    open(f"{out}/jobs/ev.{name}.res", "w").write(f"KIND=gpu\nGPUS=1\nCPUS=8\nSTAGE_SCRIPT=scripts/hpg_eval_adapters.slurm\nTIME={lim}\n"
        + "".join(f"REQUIRE={r}\n" for r in require) + f"FORBID={R}/eval/{name}\n")
    return f"ev.{name}"
T2 = f"{Q}/lists/t2_k20_train.json"; DEV = f"{T}/data/manifest/dev140.json"; RULES = f"{Q}/lists/rules_v1.txt"
c2b = [base_job("t2-k20-raw27.run1", T2, "00:30:00"),
       base_job("t2-k20-rules-v1.run1", T2, "00:30:00", extra=RULES, require=[RULES]),
       base_job("rules-v1.run1", DEV, "00:45:00", extra=RULES, require=[RULES])]
icl = [base_job("icl-none.loni.run1", DEV, "00:45:00"),
       base_job("icl-k1.loni.run1", DEV, "01:00:00", view="k1", mode="fixed", mml=32768),
       base_job("icl-k2.loni.run1", DEV, "01:00:00", view="k2", mode="fixed", mml=32768),
       base_job("icl-type2.loni.run1", DEV, "01:30:00", view="k20", mode="type2", mml=65536),
       base_job("icl-k5.loni.run1", DEV, "02:00:00", view="k5", mode="fixed", mml=65536),
       base_job("icl-k10.loni.run1", DEV, "03:00:00", view="k10", mode="fixed", mml=65536, workers=8),
       base_job("icl-k20.loni.run1", DEV, "05:00:00", view="k20", mode="fixed", mml=131072, workers=8)]
wnames = ["w1-proj-out-k16", "w1-proj-out-k64", "w1-comp-out-k16", "w1-comp-out-k64", "w1-rand-proj-out-k64", "w1-rand-comp-out-k64",
          "w2-r1-equal", "w2-r1-gradprofile", "w2-r1-equal-x0.5", "w2-r1-equal-x2"]
wev = []
for n in wnames:
    ck = f"{R}/derived/{n}/checkpoint"; name = f"{n}.run1"; wev.append(f"ev.{name}")
    open(f"{out}/jobs/ev.{name}.env", "w").write(env([("PROJECT_ROOT", TE), ("EXPECT_COMMIT", "4ba5d0e"), ("CONFIG", f"{TE}/configs/base_e4b.yaml"),
        ("WORKERS", "16"), ("ADAPTERS", f"{n}={ck}"), ("TASKS", f"{TE}/data/manifest/dev140.json"), ("INCLUDE_BASE", "0"),
        ("OUT", f"{R}/eval/{name}"), ("VERIFIED", f"{D}/k20")] + common))
    open(f"{out}/jobs/ev.{name}.res", "w").write("KIND=gpu\nGPUS=1\nCPUS=8\nSTAGE_SCRIPT=scripts/hpg_eval_adapters.slurm\nTIME=00:50:00\n"
        f"REQUIRE={ck}/adapter_model.safetensors\nREQUIRE={Q}/done/w12.derive.k20\nFORBID={R}/eval/{name}\n")
open(f"{out}/lists/c2b.list", "w").write("".join(n + "\n" for n in c2b))
open(f"{out}/lists/icl_loni.list", "w").write("".join(n + "\n" for n in icl))
open(f"{out}/lists/w_ev.list", "w").write("".join(n + "\n" for n in wev))
HEAD = f"""set -u
R={R}; Q={Q}; PY={W}/tc-alignment/.venv/bin/python; T={T}
LIKE={W}/ICML27_data/reports/runs/baselines/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint
step() {{ echo "$(date '+%F %T') $*"; }}
fresh() {{ if [ -e "$1" ]; then mkdir -p "$R/derived/_failed"; mv -- "$1" "$R/derived/_failed/$(basename "$1").$(date +%s)"; fi; }}
"""
open(f"{out}/derive/w12_derive_k20.sh", "w").write("#!/bin/bash\n# w12_derive_k20.sh (LONI, CPU, tree 27a83f4) — prereg section 19: W1 functional arms (projection of the working rank-16\n# K20 delta onto / off the top-K output directions of the full gradient; random-subspace controls), W2 arms (rank-1 working\n# delta with non-learned magnitudes), and the W1 overlap table with the registered quantities. Idempotent. Ends W12_DONE.\n" + HEAD + r"""cd "$T" || exit 2
export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=
ck() { echo "$R/cells/$1/stages/train-sft/attempt-000/output/checkpoint"; }
Y16=$(ck a1-k20-r16-lr2e-4); Y1=$(ck a1-k20-r1-lr2e-4); Y64=$(ck a0-k20-r64); G=$R/derived/c1-k20-g64/checkpoint
for p in "$Y16" "$Y1" "$Y64" "$G" "$R/derived/c1a-k20-g64/checkpoint" "$R/derived/c1t-k20-g64/checkpoint"; do [ -f "$p/adapter_model.safetensors" ] || { echo "W12_FAIL missing $p"; exit 1; }; done
op() { local name=$1; shift; local d="$R/derived/$name"
  if [ ! -f "$d/checkpoint/training_state.json" ]; then fresh "$d"; step "derive $name: $*"
    "$PY" -m tools.wsp_adapter_ops "$@" --out "$d" > "$Q/logs/w12.$name.log" 2>&1 || { echo "W12_FAIL $name"; tail -n 5 "$Q/logs/w12.$name.log"; exit 1; }; fi
  "$PY" -c 'import json,sys; j=json.load(open(sys.argv[1])); print("   %s: captured=%s in=%.4f out=%.4f" % (sys.argv[2], j.get("captured"), j["input_squared_frobenius_norm"]**.5, j["output_squared_frobenius_norm"]**.5))' "$d/checkpoint/wsp_ops.json" "$name" || exit 1; }
for k in 16 64; do
  op w1-proj-out-k$k project "$Y16" --onto "$G" --k $k --side out --pad-rank 16
  op w1-comp-out-k$k project "$Y16" --onto "$G" --k $k --side out --complement --pad-rank 16
done
op w1-rand-proj-out-k64 project "$Y16" --onto "$G" --k 64 --side out --random-seed 0 --pad-rank 16
op w1-rand-comp-out-k64 project "$Y16" --onto "$G" --k 64 --side out --random-seed 0 --complement --pad-rank 16
op w2-r1-equal renorm "$Y1" --mode equal
op w2-r1-gradprofile renorm "$Y1" --mode profile --profile-from "$G"
tot=$("$PY" -c 'import json,sys; print(repr(json.load(open(sys.argv[1]))["old_total_norm"]))' "$R/derived/w2-r1-equal/checkpoint/wsp_ops.json") || exit 1
half=$("$PY" -c "print(repr($tot*0.5))"); twice=$("$PY" -c "print(repr($tot*2))"); step "rank-1 working delta total norm $tot"
op w2-r1-equal-x0.5 renorm "$Y1" --mode equal --total-norm "$half"
op w2-r1-equal-x2 renorm "$Y1" --mode equal --total-norm "$twice"
for y in r1:"$Y1" r16:"$Y16" r64:"$Y64"; do tag=${y%%:*}; path=${y#*:}; o="$R/analysis/w1.signal_overlap.$tag.json"
  if [ ! -s "$o" ]; then step "signal overlap $tag"
    "$PY" -m tools.wsp_signal_overlap --working "$path" --signal-adapter grad-all="$G" --signal-adapter grad-action="$R/derived/c1a-k20-g64/checkpoint" \
      --signal-adapter grad-thought="$R/derived/c1t-k20-g64/checkpoint" --k 1 4 16 64 --out "$o" > "$Q/logs/w12.signal_overlap.$tag.log" 2>&1 || { echo "W12_FAIL overlap $tag"; tail -n 5 "$Q/logs/w12.signal_overlap.$tag.log"; exit 1; }; fi
  echo "== signal overlap, working = $tag"; cat "$Q/logs/w12.signal_overlap.$tag.log" | tail -n 40
done
: > "$Q/done/w12.derive.k20"; step W12_DONE
""")
open(f"{out}/derive/c5a_k20.sh", "w").write("#!/bin/bash\n# c5a_k20.sh (LONI, ONE GPU) — prereg section 18, C5: rank-1 gradients of the SFT loss on Action tokens and on Thought\n# tokens (tree a5e1a76, as c1s_k20.sh), the 6-direction basis and its random control (tree 27a83f4), then the closed-loop\n# coordinate search on the 20 K20 training tasks with the real basis. Idempotent / resumable. Ends C5A_DONE.\n" + HEAD + r"""CELL=$R/cells/sd-k20
( set -a; . "$Q/jobs/prep.sd-k20.env"; set +a; source "$HPG_CONFIG"; cd "$PROJECT_ROOT" || exit 2
  [ "$(cat GIT_COMMIT)" = a5e1a76 ] || { echo "C5A_FAIL wrong gradient tree"; exit 1; }
  export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
  for span in action thought; do G="$R/derived/c1${span:0:1}-k20-g1"
    if [ ! -f "$G/checkpoint/training_state.json" ]; then fresh "$G"; step "gradient rank 1 span $span"
      "$PY" -m tools.wsp_onestep gradient --cell "$CELL" --like "$LIKE" --out "$G" --rank 1 --span "$span" || exit 1; fi
  done ) || { echo "C5A_FAIL gradient"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( cd "$T" || exit 2; export PYTHONPATH="$T/src:$T" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
  for b in c5-basis-k20: c5-basis-k20-rand0:0; do name=${b%%:*}; seed=${b#*:}; B="$R/derived/$name"
    if [ ! -f "$B/basis.json" ]; then fresh "$B"; step "basis $name"
      CUDA_VISIBLE_DEVICES= "$PY" -m tools.wsp_adapter_ops basis --gradient-action "$R/derived/c1a-k20-g1/checkpoint" --gradient-thought "$R/derived/c1t-k20-g1/checkpoint" \
        --like "$LIKE" --out "$B" ${seed:+--random-seed $seed} || exit 1; fi
  done ) || { echo "C5A_FAIL basis"; exit 1; }
: > "$Q/done/c5.basis.k20"
step "search (real basis)"
export PROJECT_ROOT=$T HPG_CONFIG=""" + EVALENV + r""" CONFIG=$T/configs/base_e4b.yaml MODEL_PATH=""" + MODEL + r""" BASIS=$R/derived/c5-basis-k20 LIKE \
  TASKS=$Q/lists/t2_k20_train.json VERIFIED=""" + D + r"""/k20 OUT=$R/search/c5-k20 WORKERS=16 BATCH_INVARIANT=1 \
  VLLM_CACHE_ROOT=""" + W + r"""/tools/vllm-cache TRITON_CACHE_DIR=""" + W + r"""/tools/triton-cache VLLM_NO_USAGE_STATS=1 SED_HASH_CACHE_DIR=""" + W + r"""/ICML27_v0_runs/sed-hash-receipts
mkdir -p "$R/search"
bash "$T/scripts/wsp_search_job.sh"; rc=$?
[ $rc -eq 0 ] || { echo "C5A_FAIL search rc=$rc"; exit 1; }
: > "$Q/done/c5.search.k20"; step C5A_DONE
""")
open(f"{out}/derive/c5b_k20.sh", "w").write("#!/bin/bash\n# c5b_k20.sh (LONI, ONE GPU) — C5 control: the same closed-loop coordinate search with the random 6-direction basis\n# (same module sets and per-module norms as the real basis). Needs done/c5.basis.k20. Resumable. Ends C5B_DONE.\n" + HEAD + r"""[ -f "$Q/done/c5.basis.k20" ] && [ -f "$R/derived/c5-basis-k20-rand0/basis.json" ] || { echo "C5B_FAIL basis not ready"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export PROJECT_ROOT=$T HPG_CONFIG=""" + EVALENV + r""" CONFIG=$T/configs/base_e4b.yaml MODEL_PATH=""" + MODEL + r""" BASIS=$R/derived/c5-basis-k20-rand0 LIKE \
  TASKS=$Q/lists/t2_k20_train.json VERIFIED=""" + D + r"""/k20 OUT=$R/search/c5-k20-rand0 WORKERS=16 BATCH_INVARIANT=1 \
  VLLM_CACHE_ROOT=""" + W + r"""/tools/vllm-cache TRITON_CACHE_DIR=""" + W + r"""/tools/triton-cache VLLM_NO_USAGE_STATS=1 SED_HASH_CACHE_DIR=""" + W + r"""/ICML27_v0_runs/sed-hash-receipts
mkdir -p "$R/search"
bash "$T/scripts/wsp_search_job.sh"; rc=$?
[ $rc -eq 0 ] || { echo "C5B_FAIL search rc=$rc"; exit 1; }
: > "$Q/done/c5.search.k20-rand0"; step C5B_DONE
""")
open(f"{out}/tc_wsp_pass6_loni.sh", "w").write(f"""#!/bin/bash
# tc_wsp_pass6_loni.sh — runs ON LONI. Round 2 (prereg sections 18-19) submission pass: N1, B4 (12B), A3, C2b, second ICL set,
# C5 (GPU derive scripts), W1 / W2 (CPU derive + evals). Submits only what is ready now; safe to run at any time.
Q={Q}; R={R}; S=$Q/tc_wsp_submit_loni.sh
echo "== $(TZ=America/Chicago date '+%F %T %Z') LONI pass6"
inq() {{ squeue -u $USER -h -o %j | grep -q -x -F "$1"; }}
[ ! -f "$Q/done/w12.derive.k20" ] && ! inq wsp-cpu-w12_derive_k20 && [ "$(cut -f1 $Q/submitted.tsv | grep -c -x -F cpu:w12_derive_k20.sh)" -lt 3 ] && bash $S cpu w12_derive_k20.sh 12
[ ! -f "$Q/done/c5.search.k20" ] && [ "$(cut -f1 $Q/submitted.tsv | grep -c -x -F gpucmd:c5a_k20.sh)" -lt 4 ] && bash $S gpucmd c5a_k20.sh 16 05:00:00
[ -f "$Q/done/c5.basis.k20" ] && [ ! -f "$Q/done/c5.search.k20-rand0" ] && [ "$(cut -f1 $Q/submitted.tsv | grep -c -x -F gpucmd:c5b_k20.sh)" -lt 4 ] && bash $S gpucmd c5b_k20.sh 16 05:00:00
for l in n1_main.list n1_ev2.list b4_main.list b4_ev.list a3_main.list a3_ev.list c2b.list w_ev.list icl_loni.list r2d.list; do [ -f "$Q/lists/$l" ] && bash $S list $l 16; done
echo "== queue"; squeue -u $USER -h -o '%i %j %P %T %M %S' | grep ' wsp-' | sort -k2 | cut -c1-105
echo "== markers: $(ls $Q/done | grep -E 'prep\\.(n1|b4|a3)|a3check|w1|w12|c5' | tr '\\n' ' ')"
echo "== prep waves"; for f in $Q/logs/slurm-wsp-prep-prep_a3-*.out; do [ -f "$f" ] && echo "$(basename $f): $(grep -c PREP_OK $f) ok, $(grep -c PREP_FAIL $f) fail, last: $(tail -n 1 $f | cut -c1-90)"; done
echo "== n1 cells"; python3 $Q/tc_wsp_collect.py $R 2>/dev/null | grep -E '^== |n1-|b4-|12b|a3-|rules|icl-|raw27|w1-|w2-|c5' | cut -c1-170
echo "== search"; for s in $R/search/*/state.json; do [ -f "$s" ] && python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); c=j["current"]; b=j["best_so_far"]; print(sys.argv[1].split("/")[-2], "iter", j["iteration"], "step", j["step"], "current", c and (c["solved"], round(c["mean_progress"],4), c["coefficients"]), "best", b and (b["solved"], round(b["mean_progress"],4)))' "$s"; done
echo "== files: $(lfs quota -u $USER /work 2>/dev/null | grep -E '^ +/work' | awk '{{print $6}}')"
echo "== last states"; sacct -u $USER -S now-3hours -X -n -o JobID,JobName%34,State%12,Elapsed 2>/dev/null | grep wsp- | grep -v -E 'PENDING|RUNNING' | tail -n 16
""")
print("ok", len(c2b), len(icl), len(wev))
