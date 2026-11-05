#!/bin/bash
# Example launcher; starts a server except for the CPU-only expand-synth stage.
# Source/cache/model paths come from the same rai.env as run_v2_eval_rai.sh.
set -euo pipefail
: "${V0_ROUND_OUT:?Set the round root, already initialized by synth}"
: "${V0_STAGE:?Set attempt, probe, review, expand or expand-synth}"
: "${ADAPTER:?Set the theta0 checkpoint path}"
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export V0_SOURCE_ROOT=${V0_SOURCE_ROOT:-/home/xueqi/hq/projects/ICML27_agent_distillation}
export HPG_CONFIG=${HPG_CONFIG:-$V0_SOURCE_ROOT/reports/runs/repair_pilot/rai.env}
export MODEL_NAME=${MODEL_NAME:-e4b-tsft-k20-e32-lr5e-5}
case "$MODEL_NAME" in
    e4b-tsft-k10-e32-lr5e-5) seed_k=10 ;;
    e4b-tsft-k20-e32-lr5e-5) seed_k=20 ;;
    e4b-tsft-k30-e64-lr5e-5) seed_k=30 ;;
    *) echo 'Unknown V0 theta0 MODEL_NAME' >&2; exit 2 ;;
esac
export VERIFIED=${VERIFIED:-$V0_SOURCE_ROOT/data/verified/seeds_full_train_r2/subsets/k$seed_k}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
case "$V0_STAGE" in
    expand-synth) export CUDA_VISIBLE_DEVICES='' ;;
    attempt|probe|review|expand) ;;
    *) echo 'Unknown V0_STAGE' >&2; exit 2 ;;
esac
if [[ -n "${V0_ATTEMPT_DIR:-}" ]]; then
    [[ "$V0_STAGE" == probe && "${V0_RESUME:-0}" != 1 ]] || {
        echo 'V0_ATTEMPT_DIR requires a fresh probe' >&2; exit 2;
    }
    # Reserve the entire run exclusively. The phase and serving evidence have
    # separate children so the probe CLI can also reserve its phase exclusively.
    mkdir -p "$(dirname "$V0_ATTEMPT_DIR")"
    mkdir "$V0_ATTEMPT_DIR"
    export V0_ATTEMPT_DIR="$(cd "$V0_ATTEMPT_DIR" && pwd)"
    export REPAIR_OUT="$V0_ATTEMPT_DIR/serve"
else
    export REPAIR_OUT=${REPAIR_OUT:-$V0_ROUND_OUT/serve-$V0_STAGE${V0_VARIANT:+-$V0_VARIANT}}
fi
cd "$PROJECT_ROOT"
source scripts/repair_stage_env.sh
# Fail before starting the GPU server if the round, seed view or adapter differ.
CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$V0_ROUND_OUT" "$VERIFIED" "$MODEL_NAME" "$ADAPTER" <<'PY'
import sys
from pathlib import Path
from repair.v0_common import SyntheticManifest, check_theta0, read, round_identity, seed_manifest_k
from sed.storage import file_hash
root, verified, name, adapter = sys.argv[1:]
k = seed_manifest_k(Path(verified) / 'manifest.json')
manifest = SyntheticManifest(Path(root) / 'synth/manifest.json')
if read(Path(root) / 'round.json') != round_identity(k) or manifest.k != k:
    raise ValueError('V0 round and VERIFIED K differ')
if manifest.value['source_seed_manifest_sha256'] != file_hash(Path(verified) / 'manifest.json'):
    raise ValueError('V0 VERIFIED differs from the frozen seed manifest')
check_theta0(k, name, adapter)
PY
repair_work() {
    local extra=(--workers "${WORKERS:-8}")
    [[ -n "${V0_ATTEMPT_DIR:-}" ]] && extra+=(--attempt-dir "$V0_ATTEMPT_DIR/probe")
    [[ "${V0_RESUME:-0}" == 1 ]] && extra+=(--resume)
    if [[ "$V0_STAGE" == expand || "$V0_STAGE" == expand-synth ]]; then
        extra+=(--label-workers "${LABEL_WORKERS:-16}")
        [[ -n "${V0_SNAPSHOT:-}" ]] && extra+=(--snapshot "$V0_SNAPSHOT")
        : "${V0_VARIANT:?Set D-data or D-param}" "${V0_DATASET:?Set the immutable input dataset}"
        extra+=(--variant "$V0_VARIANT" --dataset "$V0_DATASET"
            --seeds "$VERIFIED/manifest.json" --data-root "$ALFWORLD_DATA_ROOT"
            --main-checkout "$V0_SOURCE_ROOT")
        if [[ "$V0_VARIANT" == D-data ]]; then
            : "${V0_GROUPS:?Set the grouping output directory}"
            extra+=(--groups "$V0_GROUPS/groups.json" --weights "$V0_GROUPS/weights.json")
        fi
    fi
    if [[ "$V0_STAGE" == expand-synth ]]; then
        CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -m tools.v0_round expand --synth-only \
            --config "$CONFIG" --output "$V0_ROUND_OUT" --seed 0 \
            --model-name "$MODEL_NAME" "${extra[@]}"
        return
    fi
    # repair_serve_stage.sh supplies the actual ready endpoint and both gates.
    CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m tools.v0_round "$V0_STAGE" \
        --config "$CONFIG" --output "$V0_ROUND_OUT" --seed 0 \
        --endpoint "$1" --server-provenance "$2" --lora-check "$3" \
        --model-name "$MODEL_NAME" "${extra[@]}"
}
if [[ "$V0_STAGE" == expand-synth ]]; then
    repair_work
else
    source scripts/repair_serve_stage.sh
fi
