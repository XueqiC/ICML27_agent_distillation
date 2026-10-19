#!/bin/bash
# Submit/resume one HAD cell; arguments are shared account/QOS sbatch options.
# HAD_OUT=/shared/cell VERIFIED=/shared/support bash scripts/ops_had_cell.sh \
#     --account=... --qos=...
# Resume just a killed member: ... ops_had_cell.sh --resume-role gpu [account args]
set -euo pipefail
role=''
worker=0
if [[ "${1:-}" == --worker || "${1:-}" == --resume-role ]]; then
    [[ "$1" != --worker ]] || worker=1
    role=${2:?Expected gpu or teacher}
    shift 2
    [[ "$role" == gpu || "$role" == teacher ]] || exit 2
fi
account_args=("$@")
# Preserve exact argv (including separate option values) on both submissions.
while (( $# )); do
    case "$1" in
        --account=*|--qos=*) shift ;;
        --account|--qos|-A|-q) [[ $# -ge 2 ]] || exit 2; shift 2 ;;
        *) echo 'Only account/QOS options are accepted; resources are fixed per role.' >&2; exit 2 ;;
    esac
done
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
source "${HPG_CONFIG:-$PROJECT_ROOT/configs/hpg.env}"
cd "$PROJECT_ROOT"
: "${HAD_OUT:?Set a persistent HAD cell output}"
: "${VERIFIED:?Set the verified support collection}"
export HAD_OUT VERIFIED NGPUS=4 HAD_MODE=train HAD_THROUGH=explore-0
mkdir -p "$HAD_OUT/scheduling"

submit() {
    local target=$1 dependency=${2:-} job
    local resources=() dependencies=()
    if [[ "$target" == gpu ]]; then
        resources=(--partition=hpg-b200 --gpus=4 --cpus-per-task=8 --mem=256G)
    else
        resources=(--partition=hpg-milan --gpus=0 --cpus-per-task=2 --mem=8G)
    fi
    [[ -z "$dependency" ]] || dependencies=("--dependency=afterany:$dependency")
    job=$(sbatch --parsable "${account_args[@]}" "${resources[@]}" "${dependencies[@]}" \
        --nodes=1 --ntasks=1 --time=48:00:00 --signal=B:USR1@120 \
        --job-name="had-$target" --output="$HAD_OUT/scheduling/$target-%j.out" \
        --export=ALL "$PROJECT_ROOT/scripts/ops_had_cell.sh" --worker "$target" "${account_args[@]}")
    # Log only job IDs/roles, never the process environment or credentials.
    printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$target" "$job" >> "$HAD_OUT/scheduling/submissions.log"
    printf '%s %s\n' "$target" "$job"
}

if (( ! worker )); then
    if [[ -n "$role" ]]; then
        submit "$role"
    else
        submit teacher
        submit gpu
    fi
    exit 0
fi

export HAD_WORKER=$role
child=''
cleanup() {
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
        child=''
    fi
}
rollover() {
    trap '' USR1
    cleanup
    submit "$role" "${SLURM_JOB_ID:?Missing SLURM job ID}"
    exit 0
}
trap cleanup EXIT
# Time-limit warning resubmits either member. Explicit cancellation does not.
trap rollover USR1
trap 'exit 143' TERM
trap 'exit 130' INT
bash "$PROJECT_ROOT/scripts/hpg_had.slurm" &
child=$!
status=0
wait "$child" || status=$?
child=''
if [[ "$role" == gpu && "$status" == 75 ]]; then
    submit gpu "${SLURM_JOB_ID:?Missing SLURM job ID}"
    exit 0
fi
exit "$status"
