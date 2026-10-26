# Common repair stage setup. Paths/caches may be supplied by a keeper export.
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
source "${HPG_CONFIG:-$PROJECT_ROOT/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${REPAIR_OUT:?Set a fresh stage output directory}"
[[ ! -e "$REPAIR_OUT" && ! -L "$REPAIR_OUT" ]] || { echo 'Output already exists' >&2; exit 2; }
# Match the grid's caller-managed OFF_HOME cache exports. Never choose another
# cache tree or override an exported XDG_CACHE_HOME inside a stage job.
: "${ALFWORLD_DATA:?Export OFF_HOME ALFWORLD_DATA}" "${VLLM_CACHE_ROOT:?Export OFF_HOME VLLM_CACHE_ROOT}"
: "${TRITON_CACHE_DIR:?Export OFF_HOME TRITON_CACHE_DIR}" "${HF_HOME:?Export OFF_HOME HF_HOME}"
: "${XDG_CACHE_HOME:?Export OFF_HOME XDG_CACHE_HOME}" "${SED_HASH_CACHE_DIR:?Export OFF_HOME SED_HASH_CACHE_DIR}"
export ALFWORLD_DATA VLLM_CACHE_ROOT TRITON_CACHE_DIR HF_HOME XDG_CACHE_HOME SED_HASH_CACHE_DIR
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
CONFIG=${CONFIG:-configs/base_e4b.yaml}
