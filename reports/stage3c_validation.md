# Stage 3c: IPR port and SAG reproducibility — 2026-10-05

Engineering validation only. No GPU use, live model/environment server, network
or OpenAI request, job submission, commit or push occurred. `third_party/`,
`reports/runs/`, `data/` and mechanism code are unchanged. Existing local seed
lengths were read solely for the compute estimate below.

The implementation is in `baselines/ipr/{pipeline,rollouts,branch,sampling,pairs}.py`,
with the IPR objective in `baselines/common/fsdp_train.py` and the launcher in
`scripts/hpg_ipr.slurm`. [DEVIATIONS](../baselines/ipr/DEVIATIONS.md) registers
the disabled official expert-MC activation, defects A/B/C/D and every protocol
adaptation. The official pin is `93335a3a818b46c1d0e081cd76288cf1cbf7dee4`.

## Validation

Required offline suite, with test fixtures disabling CUDA, DNS, network sockets
and the live OpenAI transport:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_stage2b_torch.py tests/test_lora_serving_torch.py \
  tests/test_stage3a_torch.py tests/test_stage3b_torch.py tests/test_stage3c_torch.py
bash -n scripts/hpg_ipr.slurm
git diff --check
```

The final full suite passed **482 tests, 6 skipped**, in 407.86 seconds. Five
skips are torch modules unavailable in the project venv; the sixth is the
optional real ALFWorld integration test without `ALFWORLD_DATA_ROOT`. All five
torch modules were run in the requested separate CPU runtime: **51 passed** in
31.81 seconds, with one existing PEFT warning about a missing config in a
synthetic adapter fixture. Shell syntax and whitespace checks passed.

An earlier full-suite attempt hit a mechanism resume-identity guard while source
files were still being edited. The complete passing rerun above held the source
tree fixed; no mechanism code or compatibility exception was changed to pass it.

IPR tests execute the isolated pinned official pair builder on synthetic
records and compare both continuations and the prompt. Cases cover global and
entire pairs, strict .01 equality/ties, both chosen directions, the ALFWorld
shorter-continuation flag (including shorter failures), empty outputs and unknown
MC values. CPU numerical tests execute official `dpo_with_sft_loss`, compare loss
and gradients, verify summed chosen NLL and frozen-reference updates, and run
tiny-model training. The installed real Gemma tokenizer verifies prefix masking,
all continuation assistant turn-end labels 106, reversed expert/student source
labels, whitelist boundaries and explicit overlength rejection.

Fake environment tests exercise real reset/replay after expert actions, terminal
expert prefixes, audit mismatch failure, agent MC after the first agent action,
parser-feedback replay and the shared 40-turn cap. Fake HTTP MC tests compare
actions, observations and request seeds at workers=1/4. SAG tests repeat the same
fake-server run and compare every request and selected response, with distinct
per-turn seeds. Existing tests verify default greedy request bytes are unchanged.
AgentEvol is greedy; mechanism samples already pass seeds and remain unchanged.

An end-to-end fake collection → expert MC → exploration → agent MC → pairs chain
checks the evidence loaders and rejects mutations to raw model-call records.
Fake pipeline tests cover all six iterations, fixed SFT reference, P=0
carry-forward without trainer invocation, current-iteration-only pairs,
hash-checked resume and changed worker count. Existing ETO and AgentGym tests
continue to check the shared machinery.

## Exact HPG commands (not executed)

Use a dedicated experiment checkout, with the installed runtimes configured by
`configs/hpg.env`; do not reuse the worktree running the mechanism experiment.
The launcher uses account/qos `fsu-compsci-dept`, partition `hpg-b200`, 32 CPUs,
480G and a 12-hour wall request. Training is four B200s. The stage chain below
overrides serving-only jobs to one B200; environment workers run with CUDA hidden.
Each rollout stage starts one checked server: current iterate for exploration,
fixed SFT for expert/agent MC. No teacher server is used.

```bash
source configs/hpg.env
cd "$PROJECT_ROOT"
export CONFIG=configs/base.yaml
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0"
export IPR_OUT="$PROJECT_ROOT/reports/runs/ipr/k20-s0"
export BATCH_INVARIANT=1 VLLM_BATCH_INVARIANT=1 WORKERS=8

# Optional CPU-only input freeze; no server/training launched.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. "$ENV_PYTHON" -m baselines.ipr.pipeline \
  --config "$CONFIG" --snapshot "$MODEL_PATH" --verified "$VERIFIED" \
  --output "$IPR_OUT" --gpus 4 --workers "$WORKERS" --mode prepare \
  --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON"

# Stage boundaries avoid requiring the whole MC-heavy run to fit in 12 hours.
ipr_job=$(sbatch --parsable --export=ALL,IPR_MODE=train,IPR_THROUGH=sft scripts/hpg_ipr.slurm)
ipr_job=$(sbatch --parsable --dependency="afterok:$ipr_job" --gpus=1 \
  --export=ALL,IPR_MODE=train,IPR_THROUGH=expert-mc scripts/hpg_ipr.slurm)
for ipr_iteration in 1 2 3 4 5 6; do
  # Current-policy exploration, fixed-SFT MC, then CPU pair construction.
  ipr_job=$(sbatch --parsable --dependency="afterok:$ipr_job" --gpus=1 \
    --export="ALL,IPR_MODE=train,IPR_THROUGH=pairs-$ipr_iteration" scripts/hpg_ipr.slurm)
  ipr_boundary="dpo-$ipr_iteration"
  if [[ "$ipr_iteration" == 6 ]]; then ipr_boundary=all; fi
  ipr_job=$(sbatch --parsable --dependency="afterok:$ipr_job" --gpus=4 \
    --export="ALL,IPR_MODE=train,IPR_THROUGH=$ipr_boundary" scripts/hpg_ipr.slurm)
done

# Separate evaluation of SFT and six iterates, frozen dev140, unchanged harness.
sbatch --dependency="afterok:$ipr_job" --gpus=1 --cpus-per-task=8 --mem=48G \
  --export=ALL,IPR_MODE=evaluate,IPR_THROUGH=all scripts/hpg_ipr.slurm
```

For K=5/10, set `VERIFIED` to
`$PROJECT_ROOT/data/verified/seeds_k20_s0/subsets/k5` or `subsets/k10`, and use
`IPR_OUT=.../k5-s0` or `.../k10-s0`. For another support seed, change the collection
and output directory together. **Training seed is always 0**, independently of
support seed. `--workers` can change on resume; sampling seeds and root identity
do not depend on it. Re-submit the failed boundary with the same inputs to resume;
downstream Slurm jobs whose dependency failed need replacement submissions.

`--gpus 4` in the pipeline registers the training recipe, even in a one-GPU
serving job. Such a job is safe only after its prerequisite training boundaries
have completed, as enforced by the chain above. A full four-GPU monolithic run
is also supported by `IPR_THROUGH=all`, but reserves three idle GPUs during
serving and is unlikely to fit the 12-hour request at K=20.

## Artifacts and reproducibility

`identity.json`, `tasks.json` and root provenance bind config, code, dependencies,
frozen splits, verified collection and base artifacts. Stages are `train-sft`,
`expert-mc`, then `explore-i`, `agent-mc-i`, `pairs-i`, `train-dpo-i` for i=1..6.
Each retains `attempt-NNN`; only successful actions publish a hash-bound
`complete.json`. Changed completed artifacts fail resume. Incomplete stages
restart from their beginning in a fresh attempt; their costs remain disclosed.

Rollout outputs contain `plan.json`, ordered `receipts.json`, `summary.json`,
MC `values.json` (all five outcomes and their mean or unknown), plus individual
`rollouts/*/{receipt.json,episode/,calls/}`. Episodes contain raw/parsed actions,
actual prefix transitions, visible observations, audit hashes and provenance;
calls retain raw returned text, token usage, model identity/timestamps, request
blobs and seeds. Hidden/audit fields are never used to render prompts. Missing
or unknown samples cannot be averaged away or turned into negative labels.

For zero-based branch j, exploration replays j expert actions; agent MC replays
j expert actions plus the first agent action. The matching expert value uses
expert prefix length j+1. Expert MC samples prefix lengths 1..T once, including
terminal prefixes (real replay, zero further requests). This avoids an off-by-one
comparison of different actions. The upstream unused j=0 agent MC is retained
and disclosed, while global pair decisions use final success only.

Pair stages retain both kinds, all tie/rejection decisions, and source provenance.
`iteration-i.json` reports P and kind counts **before** DPO; `models.json` records
all iterates, including unchanged checkpoints for P=0. Training results record
exact X5 batch/steps, logical and physical exposure, reference-forward tokens,
timing and memory. `disclosure.json` includes every rollout/training attempt,
known token totals and missing-usage counts. Environment-step totals include
replay; `prefix_environment_steps` is a subset, not an additional charge.

SAG request seeds use the first eight hex digits of canonical-JSON SHA-256 of
`[sampling_seed, task_id, zero_based_turn]`. Task identity is supplied separately
from visible fields. `--sag-sampling-seed` defaults to 0. IPR rollout seeds hash
`[0, seed_episode_id, task_id, kind, iteration, j, sample]`; each MC request hashes
`[rollout_seed, absolute_turn]`. Worker count, PID, output path, wall time and the
new rollout's random episode UUID are excluded. Timestamps, latencies and UUIDs
remain truthful, so entire run directories are not expected to be byte-identical.

## Compute estimate per K

These are planning calculations, not measured IPR throughput. Existing **s0**
seed lengths give E=ΣT of 79, 200 and 382 for K=5,10,20 respectively. Let H=40.
Per iteration there are E greedy branches and 5E agent-MC samples; one-time
expert MC has 5E reset/replays. Terminal expert samples make zero model calls.
All six iterations plus seven dev140 evaluations therefore require
**41E + 980 resets**, before retries. No new teacher calls or trusted seeds.

| K | E / maximum P per iteration | Expert MC resets | Six-iteration training-related resets | With 980 evaluation resets | Model calls, turn-cap bound including eval |
| --- | --- | --- | --- | --- | --- |
| 5 | 79 | 395 | 3,239 | 4,219 | 125,948 |
| 10 | 200 | 1,000 | 8,200 | 9,180 | 249,853 |
| 20 | 382 | 1,910 | 15,662 | 16,642 | 450,427 |

The fourth column includes the one-time expert stage. Calls are calculated as
`5Σ_kΣ_(t=1..T_k−1)(40−t) + 6Σ_kΣ_(j=0..T_k−1)[(40−j)+5(39−j)] + 980×40`.
Prefix replay uses no model requests. All-action prefix-replay upper counts are
42,207 / 116,347 / 213,163 actual environment steps, respectively; parser-error
turns can reduce them. Continuation steps are additional. Each rollout separately
resets its environment, so CPU replay throughput can also constrain serving.

For a rough serving scenario, the prior 1,147-step/180-second B200 pilot rate
and reported 2.8× batch-invariance slowdown imply about **2.28 requests/second**.
Applying that rate to the turn-cap call counts gives 15.4 / 30.5 / 55.0 active
B200-hours. Half those calls would give 7.7 / 15.2 / 27.5 hours. Neither is an
IPR measurement or a guaranteed bound: sampled replies and long-prefix prefills
can be slower, and server starts, CPU resets, retries and exports add cost.

Training must be estimated separately for full-parameter FSDP. SFT has 3K
episode exposures; each DPO stage has 3P_i pairs, with X5 batches from 48.
Using the specification's illustrative 15k forward tokens per pair and 4/3
reference-forward factor gives `0.060 × ΣP_i` million token-equivalents plus
roughly `.0153K` million for SFT. At P_i=E every round this is 28.52 / 72.15 /
137.83 million. These are **pair-count ceiling scenarios at an assumed average
length**, not length bounds; actual P and lengths are recorded before training.

For example, an explicitly hypothetical effective full-FT throughput of 2 million
logical token-equivalents per **aggregate B200-hour**, including FSDP padding
overhead, gives 14.3 / 36.1 / 68.9 training B200-hours. Combined with the above
serving scenarios, budget roughly **22–30 / 51–67 / 96–124 active B200-hours** for
K=5/10/20 before startup/export/retries. This throughput is a planning assumption,
not the old LoRA rate and not a measured GPU claim. Replace it after a separately
authorized SFT/first-pair-stage profile. Four GPUs divide training wall time;
serving stages use one. Monolithic four-GPU serving would multiply its allocated
GPU-hour portion by four. Retain the staged chain to avoid that allocation cost.

Actual GPU FSDP collectives, 12B memory fit, live batch-invariant sampled serving
and scientific baseline performance remain outside this offline validation.
