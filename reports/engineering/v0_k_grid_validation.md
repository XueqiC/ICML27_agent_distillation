# V0 training K10 / K20 / K30

Worktree: `ICML27_wt_repair_train`, base HEAD
`dc28c526e7f5fc8d2c9e413830b51f600bf0830d`. Changes are unstaged. No commits,
GPU execution, teacher/API calls, network calls, or source-artifact writes.

## Scope

V0 reads integer K from the round dataset or supplied synthetic manifest;
missing, unsupported or conflicting K fails. It checks the original support
manifest's byte seal, `subsets/k{K}` location, verified trajectory count and
theta0 recipe. Explicit frozen K memberships must agree with that support.
The registered seed-0 recipes remain K10/e32, K20/e32 and K30/e64, lr5e-5.

The replay builder uses the existing V2 trajectory row construction with
`k{K}_replay`, preserving all assistant turns and the audit-only evidence
boundary. Its output is a sealed CONTINUE bundle to place at `dataset/replay`.
Every failed attempt and raw teacher record remains bound through the original
support collection. New interactions and training exposure are zero.

Shared `repair/v2_train.py` and `repair/train_arms.py` needed explicit K
arguments because they contained the replay-source and inherited-recipe gates.
Standalone V2 retains its defaults. `tools/v0_groups.py` already operates on
dataset/adapter/parameter identities without assuming K20. The K20 branches
in `sed.seed_evidence` / `sed.seed_extension` enforce historical extension
ancestry, not V0's support size; these remain unchanged. Buckets, mixture,
weights, module selection, schedules, hash receipts and `SED_TREE_WORKERS`
are unchanged.

## Real K20 regression

Both arms passed CPU `inputs --check` before and after the change. The entire
stdout files are **byte-identical**, including all provenance/binding fields,
bucket summaries, G inheritance records and hash-cache counters. Stderr is
empty and all four commands exit 0. Post-change times: D-data 209.56 s,
D-param 216.35 s. No round-1 inputs were changed.

| Identity | D-data | D-param |
| --- | --- | --- |
| Rows | 8,772 | 8,408 |
| Trainable parameters | 139,526,144 | 39,026,688 |
| Dataset SHA256 | `2a37f5e77bad4bc4283254523a7304c296cfb0a19709cd000a6b2a53e875fa07` | `546da5aa2bd3ce5bc07b1497bd658a265dc2a2a9b693c20a042f0d1028e7a873` |
| Effective weights SHA256 | `546bd510ddb65b924a6264bfe5b0c9e982a107849a8ed5eee01cfe0771e55231` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |

The new renderer also reconstructed all 20 real K20 replay rows identically
under canonical JSON encoding, using the original episode references. This
checks row construction independently of the unchanged replay input files.

Artifacts: `logs/v0_k_grid/{before,after}-{D-data,D-param}.json`, corresponding
`.stderr` / `.measurement.json`, and `replay_row_parity.json`.

Exact commands executed by `logs/v0_k_grid/check_real.py` (the phase only
selects the output filenames):

```bash
cd /home/xueqi/hq/projects/ICML27_wt_repair_train
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 SED_HASH_CACHE=1
PYTHON=/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python
MAIN=/home/xueqi/hq/projects/ICML27_agent_distillation
ROUND=/home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1
SNAPSHOT=/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2
THETA0="$MAIN/reports/runs/baselines_loni/e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint"

SED_HASH_CACHE_DIR="$PWD/logs/v0_hash_profile/release/D-data-receipts" \
"$PYTHON" tools/train_v0_arm.py inputs --check \
  --theta0 "$THETA0" --snapshot "$SNAPSHOT" --config configs/base_e4b.yaml \
  --relocate "/ddnB/work/xueqic/hq/ICML27_data=$MAIN/data" \
  --dataset "$ROUND/expand-D-data/dataset" \
  --synthetic-manifest "$ROUND/expand-D-data/synth/manifest.json" \
  --weights "$ROUND/grad_groups/weights.json"

SED_HASH_CACHE_DIR="$PWD/logs/v0_hash_profile/release/D-param-receipts" \
"$PYTHON" tools/train_v0_arm.py inputs --check \
  --theta0 "$THETA0" --snapshot "$SNAPSHOT" --config configs/base_e4b.yaml \
  --relocate "/ddnB/work/xueqic/hq/ICML27_data=$MAIN/data" \
  --dataset "$ROUND/expand-D-param/dataset" \
  --synthetic-manifest "$ROUND/expand-D-param/synth/manifest.json" \
  --trainable-modules "$ROUND/grad_groups/trainable_modules.json"

cmp logs/v0_k_grid/before-D-data.json logs/v0_k_grid/after-D-data.json
cmp logs/v0_k_grid/before-D-param.json logs/v0_k_grid/after-D-param.json
```

## K10 / K30 replay commands

Run after the per-K synthetic manifests and the complete original theta0
artifacts are available. The local `baselines_hpg` copies currently have
adapter weights/config/training state but lack the original
`output/provenance.json`; the builder correctly refuses incomplete provenance.
No replacement or reconstructed provenance was used. K10/K30 round roots do
not yet exist locally. Their replay construction was exercised with fakes.

The following uses the same frozen exclusions as round 1, without rewriting
them. It derives the relocation prefix solely from the recorded original
support path; K is still read from the round/synthetic manifest and validated.
Outputs are new directories in this worktree, ready to use as each dataset's
replay bundle. The downstream producer must seal their manifest hashes.

```bash
# In the worktree, with PYTHON, MAIN, SNAPSHOT and offline exports above.
export SED_HASH_CACHE_DIR="$PWD/logs/v0_k_grid/build-receipts"
SPLITS=/home/xueqi/hq/projects/ICML27_run_repair_eval_7748da5/data/manifest/method_v1_splits.json
R10=/home/xueqi/hq/projects/ICML27_v0_runs/v0_k10_r1
R30=/home/xueqi/hq/projects/ICML27_v0_runs/v0_k30_r1
T10="$MAIN/reports/runs/baselines_hpg/e4b-tsft-k10-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint"
T30="$MAIN/reports/runs/baselines_hpg/e4b-tsft-k30-e64-lr5e-5/stages/train-sft/attempt-000/output/checkpoint"
original_data_root() {
  "$PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]).parent/"provenance.json"; print(Path(json.loads(p.read_text())["config"]["baseline_training"]["verified"]).parents[3])' "$1"
}

"$PYTHON" tools/v0_replay.py \
  --round-manifest "$R10/synth/manifest.json" \
  --theta0 "$T10" --snapshot "$SNAPSHOT" --config configs/base_e4b.yaml \
  --splits "$SPLITS" --relocate "$(original_data_root "$T10")=$MAIN/data" \
  --output "$PWD/data/v0_k10_r1/replay"

"$PYTHON" tools/v0_replay.py \
  --round-manifest "$R30/synth/manifest.json" \
  --theta0 "$T30" --snapshot "$SNAPSHOT" --config configs/base_e4b.yaml \
  --splits "$SPLITS" --relocate "$(original_data_root "$T30")=$MAIN/data" \
  --output "$PWD/data/v0_k30_r1/replay"
```

## Tests

New fixtures cover K10/K30 with explicit frozen seed sets and with the
original-support fallback: replay CLI, `inputs --check`, CPU fake gradients,
groups, both choice files, deterministic batch scheduling and fake training.
Negative cases cover unsupported/missing/conflicting K, adapter K, support
location/seal, epoch recipe, replay source and frozen membership mismatches.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_v0_*.py --junitxml=logs/v0_k_grid/v0_tests.xml

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TMPDIR="$PWD/logs/v0_k_grid/tmp" \
TORCHINDUCTOR_CACHE_DIR="$PWD/logs/v0_k_grid/torch-cache" \
/home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  --basetemp=logs/v0_k_grid/pytest-full --junitxml=logs/v0_k_grid/full_cpu.xml
```

V0 suite: **175 passed in 392.98 s**, including 24 new K-grid tests.
Full CPU suite: **3,312 passed, 57 failed, 6 skipped**, 181 warnings,
3,291.04 s (54:51), exit 1. All 57 failing test IDs exactly match the saved
baseline: **no new failures and no resolved failures**. This adds 24 passing
tests to the HEAD baseline (3,288 passed, including the six tree-worker tests
added after the earlier 3,282-pass report).

The saved baseline is `logs/v0_k_grid/pre_full_lastfailed.json`; the exact
ID comparison is `logs/v0_k_grid/failure_comparison.json`. Source hashes and
dependency versions are recorded in `logs/v0_k_grid/validation_metadata.json`.
The final source hashes match those recorded during validation. Logs are
`logs/v0_k_grid/v0_tests.log` and `logs/v0_k_grid/full_cpu.log`.

Final diff: six production files modified, one offline replay builder added;
two fixture/test files updated, one K-grid test module added; the interchange
documentation updated and this report added. `git diff --check` passes, the
index is unchanged, and HEAD remains `dc28c526e7f5fc8d2c9e413830b51f600bf0830d`.
