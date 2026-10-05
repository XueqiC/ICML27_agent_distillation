#!/bin/bash
# Later authorized live use only. GPU SOURCE_RUN VERIFIED OUT NAME=ADAPTER
# OUT may be an existing probe workspace: responses are validated and reused;
# each launch writes new coverage/probe summaries and queries only missing states.
set -euo pipefail
BATCH_INVARIANT=${BATCH_INVARIANT:-1}
[[ "$BATCH_INVARIANT" == 1 ]] || { echo 'Stage 4f requires BATCH_INVARIANT=1' >&2; exit 2; }
export VLLM_BATCH_INVARIANT=1
[[ $# == 5 && "$1" =~ ^[0-9]+$ && "$5" == *=* ]] || {
    echo 'Usage: rai_mechanism_probes.sh GPU SOURCE_RUN VERIFIED OUT NAME=ADAPTER' >&2; exit 2;
}
gpu=$1
SOURCE_RUN=$2
VERIFIED=$3
OUT=$4
adapter=$5
name=${adapter%%=*}
adapter_path=${adapter#*=}
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PROJECT_ROOT"
ENV_PYTHON=${ENV_PYTHON:-.venv/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/xueqi/hq/projects/tc-alignment-vllm/envs/vllm-serve/.venv/bin/python}
TRAIN_PYTHON=${TRAIN_PYTHON:-/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python}
MODEL_PATH=${MODEL_PATH:-/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
CONFIG=${CONFIG:-configs/base.yaml}
READY_TIMEOUT=${READY_TIMEOUT:-900}
LORA_CONVERSION=${LORA_CONVERSION:-none}
[[ -x "$ENV_PYTHON" && -x "$VLLM_PYTHON" && -x "$TRAIN_PYTHON" && -d "$MODEL_PATH" ]]
launch=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" -m analysis.mechanism.probe_launcher \
    --config "$CONFIG" --source-run "$SOURCE_RUN" --verified "$VERIFIED" --adapter "$adapter" \
    --snapshot "$MODEL_PATH" --output "$OUT")
exec 9>"$OUT/.launcher.lock"
flock -n 9 || { echo 'Probe workspace already has an active launcher' >&2; exit 2; }
launcher_pid=''
work_pid=''
cleanup() {
    trap - EXIT TERM INT
    if [[ -n "$work_pid" ]]; then kill -TERM -- "-$work_pid" 2>/dev/null || true; wait "$work_pid" 2>/dev/null || true; fi
    if [[ -n "$launcher_pid" ]]; then kill -TERM "$launcher_pid" 2>/dev/null || true; wait "$launcher_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
run_work() {
    setsid "$@" &
    work_pid=$!
    local status=0
    wait "$work_pid" || status=$?
    work_pid=''
    return "$status"
}
GAP_SET=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$SOURCE_RUN/fce/fce.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["gap_path"])
PY
)
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m analysis.mechanism.coverage \
    --config "$CONFIG" --gap-set "$GAP_SET" --fce "$SOURCE_RUN/fce/fce.json" \
    --verified "$VERIFIED" --output "$OUT/coverage" >"$launch/coverage-command.json"
coverage_output=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$launch/coverage-command.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["output"])
PY
)
run_work "$TRAIN_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
    --snapshot "$MODEL_PATH" --adapter "$adapter_path" --verified "$VERIFIED" --model-name "$name" \
    --hf-only --output "$launch/hf"
"$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" \
    --model-path "$MODEL_PATH" --lora-adapter "$adapter" --lora-conversion "$LORA_CONVERSION" \
    --port 0 --ready-timeout "$READY_TIMEOUT" --output "$launch/server" &
launcher_pid=$!
deadline=$((SECONDS + READY_TIMEOUT + 30))
until [[ -f "$launch/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
done
endpoint=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$launch/server/ready.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["endpoint"])
PY
)
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
    --snapshot "$MODEL_PATH" --adapter "$adapter_path" --verified "$VERIFIED" --model-name "$name" \
    --hf-reference "$launch/hf/reference.json" --endpoint "$endpoint" \
    --server-provenance "$launch/server/provenance.json" --output "$launch/check"
probe_status=0
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m analysis.mechanism.probes \
    --config "$CONFIG" --coverage "$coverage_output/coverage.json" --output "$OUT/probes" \
    --endpoint "$endpoint" --model-name "$name" --server-provenance "$launch/server/provenance.json" \
    --lora-check "$launch/check/report.json" >"$launch/probes-command.json" || probe_status=$?
# Partial probe outcomes remain unknown in a reviewable readout; restart retries only failed states.
[[ "$probe_status" == 0 || "$probe_status" == 2 ]] || exit "$probe_status"
probe_output=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$launch/probes-command.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["output"])
PY
)
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" -m analysis.mechanism.readout \
    --config "$CONFIG" --gap-set "$GAP_SET" --fce "$SOURCE_RUN/fce/fce.json" \
    --interventions "$SOURCE_RUN/interventions/interventions.json" \
    --coverage "$coverage_output/coverage.json" --probes "$probe_output/probes.json" --output "$launch/readout"
exit "$probe_status"
