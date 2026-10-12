# R0–R3 GPU tools

Implements the frozen design in `direction/2026-10-12_real_model_R0-R3_prereg_ZH.md`.
The later user-requested clarifications below add the bounded certificate
surrogate and fix R2 labels to temperature-1 student samples; the preregistration
file itself is unchanged.
These tools do not call a teacher, submit jobs, or run environment evaluation.
Results are optimizer/surrogate diagnostics, not evidence of task success.

## State, data, and provenance

Run commands from the repository root with `.venv/bin/python -m ...` in an
installed training environment. Models/tokenizers load only local files. The
checkout is not necessarily installed as a package; set these first:

```bash
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The production CLI validates the recorded snapshot, template, reference examples,
exposure plan, training input hashes, and constant AdamW recipe (lr 1e-5,
betas .9/.999, epsilon 1e-8, zero decay, clip 1, zero dropout). Reference data is
the original K20 s0 verified root. The lower-level Python functions also accept
tiny CPU models for tests; they do not impose K20-sized fixtures.

`--checkpoint` accepts either `checkpoint-tokens-N` or `state-update-N` under
a run containing `provenance.json` and `exposure_plan.json`. Token checkpoints
use `train.optimizer_state.load_checkpoint_state`; named update states use
`load_update_state`. The base loads once. A CPU snapshot is restored into the
same model and optimizer for every branch using `restore_state`, with
`assert_trainable_equal`. Native optimizer loading also resets moments, step,
and saved RNG. Legacy token checkpoints only saved torch CPU RNG; dropout
must be zero. These tools reject module and config dropout too.

Every run writes a sibling `OUTPUT.provenance/provenance.json` containing the
original effective config, tool arguments, code hashes, dependency versions,
checkpoint identity, and data/input hashes. Source collection provenance retains
teacher identity, generation timestamps, raw outputs, failures and environment
metadata; these tools neither relabel unknown verification nor call teachers.
Continuation additionally uses the normal in-run provenance layout. Outputs
must be new paths outside the immutable source run.

Data interfaces from the concurrent data job are supported without modification:

- Original verified roots use `train.dataset.build_verified_dataset`.
- P2/P3/P4/R2 `r03_episode_dataset_v1` roots use
  `data.manifest.build_r03_sets.load_r03_dataset(purpose="training")`.
- `--surrogate` accepts the frozen `r03_surrogate.json` listing, its dataset
  root, or `r03_surrogate40.json`, using the surrogate-only loader. The latter
  selects exactly 40 of the same 80 trajectories with seed-0 proportional
  task-type quotas (largest remainder). All four diagnostics accept it; rows
  remain training-ineligible. The original listing and dataset stay unchanged.
- Surrogate groups must be disjoint from the reference and every candidate
  root. Candidate inputs never serialize arbitrary audit rows into prompts;
  the existing allowlisted encoders construct all model inputs.

Reference IDs are zero-based positions in the hashed example list, or unique
`example_id`/`label_provenance.task_id` strings. Repeated-task/correction IDs
are ambiguous: use frozen row positions. External batches use
`{"episodes":{"root":"path/to/pool","ids":[0,1]}}`.

## CRN tails and R0b

Messages are `"zero"` (also `{"zero":true}`),
`{"episodes":[0,1],"loss_multiplier":1,"clip":true}`, or
`{"union":[{"episodes":[0]},{"episodes":[1]}]}`. A union requires disjoint
batches and consistent clipping flags. It accumulates a single token-mean
loss, then clips once; component weights are their supervised-token counts
normalized by total tokens. Child multipliers multiply their token loss sums.
Unions do not average independently clipped gradients. `zero` supplies an
explicit zero gradient to every trainable, so Adam's momentum, variance and
step advance. It is not a skipped update.

An items file is a JSON list, or `{"items":[...]}`. Example:

```json
{"items":[
  {"item_id":"zero","replica":0,"message":"zero","tail_updates":10,"tail_seed":701},
  {"item_id":"batch","replica":0,"message":{"episodes":[0],"loss_multiplier":4,"clip":false},"tail_updates":10,"tail_seed":701},
  {"item_id":"zero","replica":1,"message":"zero","tail_updates":10,"tail_seed":902},
  {"item_id":"batch","replica":1,"message":{"episodes":[0],"loss_multiplier":4,"clip":false},"tail_updates":10,"tail_seed":902}
]}
```

`item_id` names a candidate across replicas. Alternatively use unique item IDs
and a shared `candidate_id`. `(item_id, replica)` must be unique. Replica
counts, messages, tail seeds and horizons must be fixed before scoring.

The tail rule is precisely: call `train.sft.exposure_plan` on reference
supervised-token costs, with `seed=tail_seed`, the recorded `tokens_per_update`
(512), packing and turn counts, **no checkpoint endpoints**, and token horizon
`(T+1)*(tokens_per_update+max(costs))`; take its first T updates. This horizon
exceeds the first T complete batches and cannot induce their final flush.
Thus each pass is exhaustively shuffled using `random.Random(omega)`, and
batches can cross pass boundaries. It starts at the beginning of this newly
shuffled plan, not at the old checkpoint cursor. All tail steps use token-mean
CE, clipping 1 and lr 1e-5. Message and tail updates use the same optimizer.

The builder takes `pools.json`, a mapping of candidate names to pools:

```json
{"P1":{"ids":[0,1],"costs":[600,300]},
 "P2":{"root":"data/r03/P2","ids":[0,1,2],"costs":[20,30,25]}}
```

It seeds a fresh `random.Random(replica)` for each candidate, maps each shared
uniform draw to `floor(u*len(pool))`, and samples **with replacement**. With
`--tokens-per-update 512`, stop at the first complete example reaching 512;
with `--batch-size N`, draw N examples (default 1). The former requires a
positive `costs` entry for each pool ID. Different lengths consume prefixes of
the same stream. A separate `random.Random(--seed)` produces 63-bit tail
seeds, independent of the message stream. Changing `--seed` changes tails;
replica indices define message draws. Zero/control messages for R3 can be
added to the same frozen items file before execution.

```bash
.venv/bin/python -m tools.crn_tail build-items \
  --pools pools.json --replicas 64 --tail-updates 10 --seed 17 \
  --tokens-per-update 512 --output items.json

.venv/bin/python -m tools.crn_tail run \
  --snapshot /local/gemma-snapshot --checkpoint /runs/cold/checkpoint-tokens-104070 \
  --verified /data/k20-s0 --surrogate data/manifest/r03_surrogate.json \
  --items items.json --shard 0 --num-shards 4 --device cuda \
  --output results/crn-0.jsonl

.venv/bin/python -m tools.crn_tail merge \
  --items items.json --results results/crn-0.jsonl results/crn-1.jsonl \
  results/crn-2.jsonl results/crn-3.jsonl \
  --metric F_tilde_geo_prob --B 2 --delta 0.05 --output results/crn-summary.json
```

Launch one process per assigned GPU, setting its `CUDA_VISIBLE_DEVICES` and
its `--shard` separately. Items are selected by file position modulo
`--num-shards`. This tool does not launch SLURM jobs. `--metric F_tilde_geo_prob`
and `--B 2` are the merge defaults and can both be omitted.

Every `run` result row includes `timing_seconds` for `restore` (initial and
final restores combined), `message_update`, `tail_updates` (including planning),
and `surrogate_scoring`. CUDA devices used by the model are synchronized before
each clock read. `wall_seconds` also includes result construction overhead.

The strict certificate's default score is **`F_tilde_geo_prob`**:
`mean_trajectories exp(trajectory_log_likelihood_sum / trajectory_supervised_tokens)`.
Each trajectory contributes its geometric-mean token probability, regardless
of its length. Exponentiation happens **before** averaging trajectories. This
score is in `(0,1]`, so paired differences are in `[-1,1]` and **a-priori B=2**
is valid without estimating a range from results. Numerical probability
underflow is an error rather than silently emitting zero.

Scores use `turn_loss_sum` with teacher-forced labels: `F_tilde_per_token` remains
total log likelihood divided by all supervised tokens; `F_tilde_per_traj` is
total log likelihood divided by trajectory count (mean trajectory **sum**).
`F_tilde_macro_token_mean` also reports the average trajectory token mean.
All three log-likelihood metrics remain outputs and explicit `merge --metric`
choices; they require an explicit, independently justified `--B`. The R0a/R3
sensitivity gradients and R1 rankers still use micro token-mean log likelihood;
they do not become derivatives of the bounded certificate metric.
Higher is better. Episode packing treats each row as a trajectory; turn
packing groups by explicit `trajectory_id` or task ID. Surrogate datasets with
multiple trajectories per task should provide distinct trajectory IDs.

Merge requires exactly all frozen items, matching experiment identities and
CRN tails, and at least two complete replicas. It emits every ordered pair's
Z values, S_hat, unbiased sample variance and
`sqrt(2*V*log(4*M*M/delta)/r) + 7*B*log(4*M*M/delta)/(3*(r-1))`.
Here **M is the number of candidates minus one**, as in Definition 8.6.
B is an a-priori **interval length for Z**: if F is in [a,b], B=2(b-a)
suffices. The geometric probability metric defaults to B=2 and rejects B<2.
Raw log likelihood has no automatic finite lower bound. Merely
passing B does not prove it valid. An observed range exceeding B is rejected;
a smaller observed range does not validate the supplied bound. The empirical
observed-Z-range radius is separately labeled `DIAGNOSTIC`, never certified.
No task-side certificate is asserted without calibrated alpha/epsilon_sur;
the latter must also absorb T=10 truncation.

For R0b, add `--unions unions.json`, where the file is a list such as
`[{"zero":"zero","A":"A","B":"B","union":"AB"}]`. Merge computes the
preregistered centered response per replica using the union's token weights.
Each pair's `nu_resp` is the absolute mean residual divided by the absolute
mean union response. `median_pair_nu_resp` is the median of these ratios across
the supplied batch pairs. Per-replica ratios and their median are separate
diagnostics; zero denominators remain null and are counted. Run/merge distinct
checkpoint states separately. Comparison against 0.631 is downstream analysis.

## R0a warmth

```bash
.venv/bin/python -m tools.warmth_dump \
  --snapshot /local/gemma-snapshot --checkpoint /runs/cold/state-update-8 \
  --verified /data/k20-s0 --surrogate data/manifest/r03_surrogate.json \
  --message batch.json --output results/warmth-8.json
```

`batch.json` is a message object. For the actual consumed gradient G and raw
saved moments at step t-1, the calculation is:

- `a=sqrt(beta2*v/(1-beta2**t))`,
  `n=(beta1*m+(1-beta1)*G)/(1-beta1**t)`;
- `kappa=(1-beta2)*G**2/(beta2*v)`, with v=0 mapped to infinity even at G=0;
- `u=abs(n)/(a+eps)`, `L=abs(grad F_tilde)`;
- `term=eta*L*u*min(kappa/2,1)`, `E_fd=sum(term)`.

Outputs contain E_fd, maximum term, top 20 named coordinates, literal
`sum L_i*Delta_theta_i` signed/absolute, and the distinct true-gradient
`sum dF_i*Delta_theta_i`. Delta is measured from a native AdamW step with
float64 parameter subtraction. Reported breakdowns cover LoRA A/B, layer,
and layer × A/B. Old inverse-empirical-CDF kappa quantiles and gradient share
are retained. Quantile reductions and coordinate summaries use CPU float64;
model forward/backward and Adam updates run on the requested device.

The preregistered criterion is `E_fd <= .1*abs(sum L*Delta_theta)`. Sensitivity
is a **proxy**, not a proven coordinate Lipschitz constant. Cancellation and
the use of absolute derivatives make this first-order comparison different
from the signed-gradient directional derivative; both are exposed.

No coordinate arrays are written by default. Explicit
`--arrays-float16-npz results/coordinates.npz` writes kappa, u, L and Delta,
with parameter names in NPZ keys. This is large and lossy; float16 can overflow
large finite values. JSON metrics are computed before this optional conversion.

## R1 rankers and R2 moments

`stats-spec.json` contains candidate messages and cluster example sets:

```json
{"candidates":{"P1":{"episodes":[0,1]},
               "P2":{"episodes":{"root":"data/r03/P2","ids":[0,1]}}},
 "example_sets":{"pick_and_place_simple::0-4":
                 {"episodes":{"root":"data/r03/R2-samples","ids":[0,1,2]}}}}
```

```bash
.venv/bin/python -m tools.grad_stats \
  --snapshot /local/gemma-snapshot --checkpoint /runs/cold/checkpoint-tokens-104070 \
  --verified /data/k20-s0 --surrogate data/manifest/r03_surrogate.json \
  --spec stats-spec.json --output results/gradient-statistics.json
```

For R2, use rollout records directly (collect them with
`tools.rollout_pool run --label-samples 2`, for example):

```bash
.venv/bin/python -m tools.grad_stats \
  --snapshot /local/gemma-snapshot --checkpoint /runs/cold/checkpoint-tokens-104070 \
  --verified /data/k20-s0 --rollouts data/raw/r03_p10/rollouts.jsonl \
  --pool data/manifest/r03_pool200.json --states-per-cluster 10 --seed 0 \
  --output results/r03_sigma.json
```

`--rollouts FILE...` is an alternative to `--spec`; `--surrogate` is optional
for moments and required for candidate rankers. State selection uses the same
seed-0 hash priorities as `r2_allocate --statistics-only`, taking up to
`--states-per-cluster` states (default 10), before inspecting labels or gradients.
Frozen R2-pool membership and original episode evidence are checked. Serving
checkpoint metadata and adapter weight hashes must match the restored checkpoint.
The loader rebuilds prompts from visible history/observations, masks the whole
prefix, and creates one example per **(state, sampled label)**. Its supervised
reply is the sampled raw response, including any visible Thought/Action text,
not a rewritten parsed command. It preserves whitespace, includes the native
stop token on a stop finish, and adds no stop token on a length-cap finish.
If the server omits its stop token, the loader assumes Gemma's native `<turn|>`;
an exact Fisher interpretation requires this to match the served stop semantics.
Malformed/inadmissible actions are retained; failed/missing labels or unexpected
thinking cause an error rather than silently selecting successful samples.

For cluster s, **sigma_s² = mean_states mean_labels
||grad[-log p(sampled action | visible state)]||²**, evaluated at the restored
student checkpoint. The loss is the **full reply token sum**, and the gradient
is **unclipped**. For independent labels drawn from that same temperature-1
student distribution, each squared score norm is an unbiased estimate of its
Fisher trace at that state, restricted to the trainable parameter space (LoRA
coordinates in production). This is an uncentered second moment, not the
squared norm of an average gradient. States retain equal weight even when
label counts differ. No environment restore/branch or optimizer update is
performed to compute these moments; the saved visible history is the policy input.

The `clusters` mapping feeds `tools.r2_allocate --sigma` directly: **`sigma2`
now equals `sigma2_unclipped`**; `sigma2_clipped_DIAGNOSTIC` retains clipping at
1 as a separate diagnostic. `sample_count` counts state/label examples and
`state_count` counts states. The existing `--spec` example sets also use full
reply loss sums, but teacher/greedy labels do not justify the Fisher claim.
Recompute old token-mean/clipped sigma files before R2 allocation.

Candidate outputs contain raw/consumed gradient norm, cosine of the **training
CE gradient** with the **positive surrogate log-likelihood gradient**, the
actual restored-state AdamW parameter score `<grad F, Delta theta>`, message
statistics and source log likelihood. This cosine convention is stated in JSON;
if an analysis uses descent directions, negate it explicitly. Undefined cosines
remain null. Surrogate examples only produce diagnostic gradients, never updates.

## R3 amplitude prediction

```bash
.venv/bin/python -m tools.amplitude_prediction \
  --snapshot /local/gemma-snapshot --checkpoint /runs/cold/checkpoint-tokens-104070 \
  --verified /data/k20-s0 --surrogate data/manifest/r03_surrogate.json \
  --message batch.json --tail-updates 10 --grid 0 0.5 1 2 4 8 \
  --output results/amplitude-prediction.json
```

Use an unscaled, nonzero episode batch. Compute before observing CRN outcomes.
The tool bias-corrects the saved moments at the **current** optimizer step,
then substitutes them for the stationary mean and mean square in Corollary
7.2(a). It sums the exact mean-field coordinate formula for j=0,...,T,
including the message step. `alpha_eta=-dF*sign(mhat)*eta`; rho and lambda use
`sqrt(vhat)`. An algebraically equivalent unnormalized form handles m=0 by
continuous extension. Two curves use c*g and globally clipped c*g. The measured
c=1 clipped gradient is compared against rescaling for rounding diagnostics.

All approximations are emitted in JSON: stationarity; ratio of expected
moments in place of expected ratio; no theta-dependent tail gradients; affine
F at the current state; epsilon set to zero; no evolving bias correction;
no decay; and native-gradient scaling approximated by c times measured g.
Coordinates with v=0 are counted/excluded, and sensitive exclusions mark the
prediction incomplete. rho>1 and zero-mean coordinates are reported rather
than silently projected into the corollary's assumptions. Peaks are grid
argmaxes, with boundary flags; nontrivial zero crossings have grid brackets
and linear interpolation. No global concavity or continuous optimum is assumed.

## Continuation to p40

```bash
.venv/bin/python -m train.continue_from_state \
  --snapshot /local/gemma-snapshot --checkpoint /runs/cold/checkpoint-tokens-104070 \
  --verified /data/k20-s0 --extra data/r03/P2 \
  --target-tokens 416280 --fraction 0.25 --seed 17 \
  --output results/continue-P2
```

Use `--fraction 0` without `--extra` for the reference fidelity arm. Fraction
scheduling chooses update k when `floor(k*f)>floor((k-1)*f)`, where k=1 starts
after the saved cursor. Thus f=.25 replaces relative updates 4,8,12,... .
The seed controls uniform with-replacement extra-example draws; each batch
closes at >=512 supervised tokens. No reference update is replayed merely to
warm the optimizer.

Instead of a fraction, `--schedule schedule.json` accepts absolute 1-based
update positions `[119,123]` (sample those batches) or
`{"119":[0,1],"123":[2]}` (explicit extra-pool IDs). Prefix positions,
duplicates, empty batches, and schedule positions past the realized target
are rejected. These positions **replace**, rather than add to, reference updates.

The target is **total realized supervised-token exposure**, including the
saved prefix. Stop at the first complete example reaching the target, and
report overshoot. Replacement lengths can change the number of updates. If
the frozen reference plan is exhausted first, cycle it from its beginning;
`reference_position`, `reference_plan_index`, and `reference_cycle` disclose
this extension. No new seed trajectories are claimed. This explicit extension
rule is an implementation choice absent from the preregistration; check its
occurrence before interpreting an arm.

The effective plan and ledger record each update's pool, original reference
position, replaced indices, actual example indices, token counts and label
provenance. Final `checkpoint-tokens-N` directories contain ordinary PEFT
adapter files, tokenizer assets, `training_state.json` and `optimizer.pt`,
compatible with existing eval/merge scripts. A final `state-update-N` also
saves exact trainables and Python/CPU/CUDA RNG using the existing API;
`--save-every-update` saves every resumed update. The token payload is a
superset of train/sft's format. For complete RNG restoration use the named
update-state format (the legacy token loader only restores CPU torch RNG).

The CLI starts from the frozen reference run. Programmatic continuation can
also consume an effective plan and full saved update state; chaining a mixed
run through the CLI requires supplying its combined example bank matching
its recorded hash. No training files outside these five tools were changed.

## Validation

The bounded-surrogate/student-label follow-up passed **82 CPU tests in 25.52 s**:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/r0r3_cpu_tests.py -m pytest -q \
  tests/test_crn_tail.py tests/test_grad_stats.py tests/test_rollout_pool.py \
  tests/test_r2_allocate.py tests/test_warmth_dump.py tests/test_amplitude_prediction.py \
  tests/test_collect_corrections.py tests/test_collect_rewrites.py
```

This covers trajectory-vs-token weighting, bounded merge defaults, mock-server
label calls/failures, direct rollout CLI ingestion, prompt masking, raw label
preservation, full-reply gradient norms, state/label averaging and an exact
categorical Fisher-trace check. Two expected tiny-model PEFT warnings; no skips.
The same offline dependency overlay described below was reused, with CUDA
hidden, network/OpenAI blocked, and no installations or commits.

On 2026-10-12, **120 tests passed in 38.78 s** (52 new tests plus the relevant
existing suites and the new shared dataset loader tests). All tests used CUDA
hidden and the existing socket/DNS/OpenAI-blocking fixtures. There were no
skips. After including final state restoration in each diagnostic's wall-time
measurement, all **52 new tests passed again in 12.96 s**. PEFT issued 19
expected warnings in the combined suite for tiny models without an original
model config path. No GPU, teacher call, hpg access, commit or push occurred.

The requested `.venv/bin/python` (Python 3.11) lacks PyTorch. It was kept as
the test interpreter; an offline, read-only import-path overlay used existing
cached packages: torch 2.14.1+cpu, Transformers 5.14.1, PEFT 0.20.0, Accelerate
1.14.0, tokenizers 0.22.2 and safetensors 0.8.0. No packages were installed or
downloaded and the project venv was not changed. The local runner and exact
selected dependency paths remain in `/tmp/r0r3_cpu_tests.py` and
`/tmp/r0r3_test_dependencies.json`. The command actually run was:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/r0r3_cpu_tests.py -m pytest -q \
  tests/test_crn_tail.py tests/test_warmth_dump.py tests/test_grad_stats.py \
  tests/test_amplitude_prediction.py tests/test_continue_from_state.py \
  tests/test_adam_probe.py tests/test_stage2b_torch.py \
  tests/test_stage2b_schedule.py tests/test_stage2b_seeds.py \
  tests/test_r03_sets.py --durations=12
```

With training dependencies already available, omit `/tmp/r0r3_cpu_tests.py`.
All five `--help` entry points also passed with the unmodified project venv
and `PYTHONPATH=src:.`.

Measured tiny-model times (two-layer Gemma4Unified, LoRA r2, one CPU thread,
two examples with 2 and 3 supervised tokens; loading excluded):

| Operation | Wall seconds |
|---|---:|
| One CRN message + T=10 updates + scoring/restoration | 0.169 |
| Warmth, sensitivity, one native step, summaries/restoration | 0.091 |
| One candidate ranker + two per-example gradients | 0.097 |
| Amplitude prediction, 6 multipliers, both clipping settings | 0.114 |
| Three resumed updates and checkpoint saves, no moment warmup | 0.245 |
| Three resumed updates and checkpoint saves, with saved warmup moments | 0.178 |
| Continuation optimizer compute per update | 0.014 |

Tests establish bitwise repeated-item scores; equivalence to an independently
executed zero step and tail; intact parameters, moments, counters and RNG after
diagnostics; token-weighted union accumulation; fixed CRN pairing and missing
shard rejection; finite-difference surrogate gradients; analytic warmth bounds;
float64 stationary amplitude simulations; and bitwise f=0 continuation from
a mid-run token checkpoint, both with and without moment warmup. Continued
adapters pass the existing artifact validator and checkpoint loader.

Remaining interpretation limits: the actual 12B/GPU execution and multi-GPU
determinism were not tested. The strict log-likelihood range B still needs a
justification; observed ranges cannot supply one. Signed coordinate
contributions in the amplitude sum do not imply global concavity or guarantee
an interior peak/zero crossing. Grid peaks/interpolated crossings and any
sensitive v=0 exclusions are labeled explicitly. The continuation cycling rule
and token-threshold overshoot are disclosed implementation choices. This work
does not freeze experimental outcomes, calibrate surrogate-to-task transfer,
update PROJECT_STATE rankings, or run the experiments themselves.
