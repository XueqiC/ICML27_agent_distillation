#!/bin/bash
# Stage 2f: later live CPU/OpenAI collection on rai. No GPU or scheduler.
set -euo pipefail
resume=0
if [[ ${1:-} == --resume ]]; then resume=1; shift; fi
[[ $# == 0 ]] || { echo "Usage: $0 [--resume]" >&2; exit 2; }
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:.
: "${ENV_PYTHON:=.venv/bin/python}"
: "${COLLECT_CONFIG:=configs/base.yaml}"
: "${COLLECTION_ROOT:=data/verified}"
: "${LOG_ROOT:=logs}"
# Preflight all outputs before starting any paid call. flock also excludes a
# concurrent launcher; the Python collector independently locks its output.
for seed in 0 1 2; do
    source_dir="$COLLECTION_ROOT/seeds_k20_s$seed"
    output="$COLLECTION_ROOT/seeds_k100_s$seed"
    [[ -f "$source_dir/manifest.json" ]] || { echo "Missing source: $source_dir" >&2; exit 2; }
    if [[ -e "$output" && $resume == 0 ]]; then
        echo "Output exists: $output; use --resume for the same frozen code/config" >&2; exit 2
    fi
    if [[ -e "$output" && ! -f "$output/identity.json" ]]; then
        echo "Existing output has no collection identity: $output" >&2; exit 2
    fi
done
mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/collect_k100.lock"
flock -n 9 || { echo "K100 launcher already running" >&2; exit 2; }
for seed in 0 1 2; do
    output="$COLLECTION_ROOT/seeds_k100_s$seed"
    # Repeat the guard under the launcher lock to close the preflight race.
    [[ ! -e "$output" || $resume == 1 ]] || { echo "Output exists: $output" >&2; exit 2; }
    log_file=$(mktemp "$LOG_ROOT/collect_k100_s${seed}.$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")
    echo "[$(date -u +%FT%TZ)] extend s$seed; log=$log_file"
    "$ENV_PYTHON" teachers/collect_seeds.py --config "$COLLECT_CONFIG" \
        --selection "data/manifest/seeds_k100_s${seed}.json" --attempts 3 \
        --extend-from "$COLLECTION_ROOT/seeds_k20_s${seed}" --output "$output" \
        --subset-k 1 2 5 10 20 50 >"$log_file" 2>&1
    echo "[$(date -u +%FT%TZ)] completed s$seed"
done
