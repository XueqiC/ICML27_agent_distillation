# Shared by the SFT launchers after hpg.env. Keep receipts across allocations,
# including when SLURM gives each job a different node-local TMPDIR.
export SED_HASH_CACHE_DIR=${SED_HASH_CACHE_DIR:-$HOME/.cache/sed-hash-receipts}

require_cpu_preparation() {
    local key
    for key in SLURM_JOB_GPUS SLURM_STEP_GPUS; do
        if [[ -n "${!key:-}" ]]; then
            echo 'CPU preparation must not hold a GPU allocation' >&2
            return 2
        fi
    done
    for key in SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_GPUS_PER_NODE SLURM_GPUS_PER_TASK; do
        if [[ -n "${!key:-}" && "${!key}" != 0 ]]; then
            echo 'CPU preparation must not hold a GPU allocation' >&2
            return 2
        fi
    done
    case "${CUDA_VISIBLE_DEVICES:-}" in
        ''|-1|NoDevFiles) ;;
        *) echo 'CPU preparation must not hold a GPU allocation' >&2; return 2 ;;
    esac
    export CUDA_VISIBLE_DEVICES=''
}
