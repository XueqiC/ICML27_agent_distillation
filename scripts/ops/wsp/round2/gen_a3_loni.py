#!/usr/bin/env python3
"""gen_a3_loni.py OUT_DIR — A3 (prereg section 18): supervision 2x2 on K20, LONI. Recipe = rank 16, alpha 32, lr 2e-4,
all modules, seed 0, 32 epochs (10 updates per epoch with the trajectory sampler -> 320 updates, as the reference).
Cells: a3-all-action (all 315 steps, Action line only), a3-p90-full (the 90 mismatch steps, whole reply),
a3-p90-action (90 steps, Action line only). Tree wsp_9b3d850 (its own wrapper passes the two options).
Train jobs additionally REQUIRE done/a3check.<cell>, created by hand after the prepare report shows the expected
supervision (guards against a wrapper that silently drops the options)."""
import os, sys
out = sys.argv[1]
W = "/work/xueqic/hq"; Q = f"{W}/ICML27_wsp/queue"; R = f"{W}/ICML27_wsp"
D = f"{W}/ICML27_data/verified/seeds_full_train_r2/subsets"
MODEL = "/work/xueqic/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
TT = f"{W}/ICML27_runs/wsp_9b3d850"; TE = f"{W}/ICML27_runs/wsp_4ba5d0e"; T2 = f"{W}/ICML27_runs/wsp_1f649a2"
FILT = f"{Q}/lists/turn_filter_k20_pairs90.json"
os.makedirs(f"{out}/jobs", exist_ok=True); os.makedirs(f"{out}/lists", exist_ok=True)
env = lambda pairs: "".join(f"{k}='{v}'\n" for k, v in pairs)
cells = [("a3-all-action", [("KANG_SUPERVISE_SPAN", "action")]),
         ("a3-p90-full", [("KANG_TURN_FILTER", FILT)]),
         ("a3-p90-action", [("KANG_SUPERVISE_SPAN", "action"), ("KANG_TURN_FILTER", FILT)])]
evs = []
for cell, extra in cells:
    base = [("PROJECT_ROOT", TT), ("EXPECT_COMMIT", "9b3d850"), ("HPG_CONFIG", f"{W}/ICML27_loni.env"),
            ("CONFIG", f"{TT}/configs/base_e4b.yaml"), ("MODEL_PATH", MODEL), ("WORKERS", "16"), ("BATCH_INVARIANT", "1"),
            ("SED_HASH_CACHE_DIR", f"{W}/ICML27_v0_runs/sed-hash-receipts"), ("KANG_OUT", f"{R}/cells/{cell}"),
            ("VERIFIED", f"{D}/k20"), ("KANG_EPOCHS", "32"), ("KANG_SCORING", "shared_history"), ("NGPUS", "2"),
            ("KANG_LR", "2e-4"), ("KANG_WARMUP", "0"), ("KANG_TRAINING_SEED", "0"), ("KANG_LORA_RANK", "16")] + extra
    open(f"{out}/jobs/prep.{cell}.env", "w").write(env(base))
    open(f"{out}/jobs/train.{cell}.env", "w").write(env(base + [("KANG_MODE", "train")]))
    open(f"{out}/jobs/train.{cell}.res", "w").write(
        "KIND=gpu\nGPUS=2\nCPUS=16\nTIME=00:50:00\nSTAGE_SCRIPT=scripts/hpg_kang_wsp.slurm\n"
        f"REQUIRE={Q}/done/prep.{cell}\nREQUIRE={Q}/done/a3check.{cell}\nFORBID={R}/cells/{cell}/result.json\n")
    ckpt = f"{R}/cells/{cell}/stages/train-sft/attempt-000/output/checkpoint"
    common = [("WORKERS", "16"), ("BATCH_INVARIANT", "1"), ("VLLM_CACHE_ROOT", f"{W}/tools/vllm-cache"),
              ("TRITON_CACHE_DIR", f"{W}/tools/triton-cache"), ("VLLM_NO_USAGE_STATS", "1"),
              ("SED_HASH_CACHE_DIR", f"{W}/ICML27_v0_runs/sed-hash-receipts"), ("MODEL_PATH", MODEL),
              ("HPG_CONFIG", f"{W}/ICML27_runs/eval_46b8527/configs_ec704e4/loni_eval.env"),
              ("ADAPTERS", f"{cell}={ckpt}"), ("DECODING", "greedy"), ("INCLUDE_BASE", "0"), ("VERIFIED", f"{D}/k20")]
    for run in (1, 2):
        name = f"ev.{cell}.run{run}"; evs.append(name)
        open(f"{out}/jobs/{name}.env", "w").write(env([("PROJECT_ROOT", TE), ("EXPECT_COMMIT", "4ba5d0e"),
            ("CONFIG", f"{TE}/configs/base_e4b.yaml"), ("TASKS", f"{TE}/data/manifest/dev140.json"),
            ("OUT", f"{R}/eval/{cell}.run{run}")] + common))
        open(f"{out}/jobs/{name}.res", "w").write(
            "KIND=gpu\nGPUS=1\nCPUS=8\nSTAGE_SCRIPT=scripts/hpg_eval_adapters.slurm\nTIME=00:50:00\n"
            f"REQUIRE={ckpt}/adapter_model.safetensors\nFORBID={R}/eval/{cell}.run{run}\n")
    name = f"ev.t2-k20-{cell}.run1"; evs.append(name)
    open(f"{out}/jobs/{name}.env", "w").write(env([("PROJECT_ROOT", T2), ("EXPECT_COMMIT", "1f649a2"),
        ("CONFIG", f"{T2}/configs/base_e4b.yaml"), ("TASKS", f"{Q}/lists/t2_k20_train.json"),
        ("OUT", f"{R}/eval/t2-k20-{cell}.run1")] + common))
    open(f"{out}/jobs/{name}.res", "w").write(
        "KIND=gpu\nGPUS=1\nCPUS=8\nSTAGE_SCRIPT=scripts/hpg_eval_adapters.slurm\nTIME=00:30:00\n"
        f"REQUIRE={ckpt}/adapter_model.safetensors\nFORBID={R}/eval/t2-k20-{cell}.run1\n")
open(f"{out}/lists/prep_a3.cells", "w").write("".join(c + "\n" for c, _ in cells))
open(f"{out}/lists/a3_main.list", "w").write("".join(f"train.{c}\n" for c, _ in cells))
open(f"{out}/lists/a3_ev.list", "w").write("".join(n + "\n" for n in evs))
open(f"{out}/tc_wsp_prep_wave2.slurm", "w").write(r"""#!/bin/bash
# tc_wsp_prep_wave2.slurm — CPU preparation for a list of Kang-SFT cells; unlike v1 every cell is prepared with the
# job script of ITS OWN tree (PROJECT_ROOT of its env file), so tree-specific options are never dropped by an old wrapper.
# Needs: WSP_QUEUE, PREP_LIST. Marker per cell: done/prep.<cell>.
set -u
: "${WSP_QUEUE:?}" "${PREP_LIST:?}"
mkdir -p "$WSP_QUEUE/done" "$WSP_QUEUE/logs"
fail=0
while IFS= read -r name <&7; do
  [ -n "$name" ] || continue
  envf="$WSP_QUEUE/jobs/prep.$name.env"
  [ -f "$envf" ] || { echo "PREP_NOENV $name"; fail=1; continue; }
  if [ -e "$WSP_QUEUE/done/prep.$name" ]; then echo "PREP_SKIP $name"; continue; fi
  echo "$(date '+%F %T') PREP_START $name"
  ( set -a; . "$envf"; set +a; export KANG_MODE=prepare; cd "$PROJECT_ROOT" || exit 2
    [ "$(cat GIT_COMMIT)" = "${EXPECT_COMMIT:?}" ] || { echo "tree mismatch" >&2; exit 2; }
    exec bash scripts/hpg_kang_wsp.slurm ) > "$WSP_QUEUE/logs/prep.$name.${SLURM_JOB_ID:-manual}.log" 2>&1 < /dev/null
  rc=$?
  if [ "$rc" -eq 0 ]; then : > "$WSP_QUEUE/done/prep.$name"; echo "$(date '+%F %T') PREP_OK $name"
  else echo "$(date '+%F %T') PREP_FAIL $name rc=$rc"; fail=1; fi
done 7< "$PREP_LIST"
if [ "$fail" -eq 0 ]; then echo "PREP_WAVE_DONE"; else echo "PREP_WAVE_FAILED"; exit 1; fi
""")
open(f"{out}/tc_wsp_pass4_loni.sh", "w").write(f"""#!/bin/bash
# tc_wsp_pass4_loni.sh — runs ON LONI. Round 2 (prereg section 18) submission pass: N1 small-K ladder, B4 (12B part), A3.
# Submits only what is ready now (cap 12 GPU-queue jobs); safe to run at any time. Prints queue, markers, results.
Q={Q}; R={R}; S=$Q/tc_wsp_submit_loni.sh
echo "== $(TZ=America/Chicago date '+%F %T %Z') LONI pass4"
for l in n1_main.list n1_ev.list b4_main.list b4_ev.list a3_main.list a3_ev.list; do [ -f "$Q/lists/$l" ] && bash $S list $l 16; done
echo "== queue"; squeue -u $USER -h -o '%i %j %P %T %M %S' | grep ' wsp-' | sort -k2 | cut -c1-105
echo "== markers: $(ls $Q/done | grep -E 'prep\\.(n1|b4|a3)|a3check' | tr '\\n' ' ')"
echo "== prep waves"; for f in $Q/logs/slurm-wsp-prep-prep_a3-*.out; do [ -f "$f" ] && echo "$(basename $f): $(grep -c PREP_OK $f) ok, $(grep -c PREP_FAIL $f) fail, last: $(tail -n 1 $f | cut -c1-90)"; done
echo "== n1 cells"; python3 $Q/tc_wsp_collect.py $R 2>/dev/null | grep -E '^== |n1-|b4-|12b|a3-' | cut -c1-170
echo "== files: $(lfs quota -u $USER /work 2>/dev/null | grep -E '^ +/work' | awk '{{print $6}}')"
echo "== last states"; sacct -u $USER -S now-3hours -X -n -o JobID,JobName%34,State%12,Elapsed 2>/dev/null | grep wsp- | grep -v -E 'PENDING|RUNNING' | tail -n 14
""")
print("ok", [c for c, _ in cells], len(evs))
