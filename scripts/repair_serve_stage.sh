# Shared one-adapter HF/vLLM equivalence gate, following eval_adapters.sh.
# The caller defines repair_work(endpoint, provenance, check) before sourcing.
: "${ADAPTER:?Set adapter path}" "${MODEL_NAME:?Set exact served adapter name}" "${VERIFIED:?Set original verified seeds}"
mkdir -p "$(dirname "$REPAIR_OUT")"
mkdir "$REPAIR_OUT"
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
    local status=0
    setsid "$@" &
    work_pid=$!
    wait "$work_pid" || status=$?
    work_pid=''
    return "$status"
}
run_work "$TRAIN_PYTHON" tools/check_lora_serving.py --config "$CONFIG" --snapshot "$MODEL_PATH" \
    --adapter "$ADAPTER" --verified "$VERIFIED" --model-name "$MODEL_NAME" --hf-only --output "$REPAIR_OUT/hf"
"$ENV_PYTHON" harness/serve_student.py --config "$CONFIG" --python "$VLLM_PYTHON" --model-path "$MODEL_PATH" \
    --lora-adapter "$MODEL_NAME=$ADAPTER" --lora-conversion none --lora-dtype bfloat16 \
    --port 0 --ready-timeout "${READY_TIMEOUT:-900}" --output "$REPAIR_OUT/server" &
launcher_pid=$!
deadline=$((SECONDS + ${READY_TIMEOUT:-900} + 30))
until [[ -f "$REPAIR_OUT/server/ready.json" ]]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then wait "$launcher_pid"; exit 1; fi
    if (( SECONDS >= deadline )); then echo 'Server startup timed out' >&2; exit 1; fi
    sleep 1
done
endpoint=$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$REPAIR_OUT/server/ready.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['endpoint'])
PY
)
CUDA_VISIBLE_DEVICES='' run_work "$ENV_PYTHON" tools/check_lora_serving.py --config "$CONFIG" \
    --snapshot "$MODEL_PATH" --adapter "$ADAPTER" --verified "$VERIFIED" --model-name "$MODEL_NAME" \
    --hf-reference "$REPAIR_OUT/hf/reference.json" --endpoint "$endpoint" \
    --server-provenance "$REPAIR_OUT/server/provenance.json" --output "$REPAIR_OUT/check"
repair_work "$endpoint" "$REPAIR_OUT/server/provenance.json" "$REPAIR_OUT/check/report.json"
