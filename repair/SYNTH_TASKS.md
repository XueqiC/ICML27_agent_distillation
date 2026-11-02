# Seed-only ALFWorld synthetic tasks

Run from this worktree with a Python environment containing the repository's
ALFWorld/TextWorld dependencies:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python tools/synth_tasks.py \
  --seeds /read-only/collection/subsets/k20/manifest.json \
  --data-root /read-only/alfworld/json_2.1.1 \
  --main-checkout /read-only/main-checkout \
  --out spike_out/new_run --per-seed 5 --g3-per-seed 3 \
  --generators G1,G2,G3 --seed 0
```

`--per-seed` counts candidate attempts per generator, not guaranteed accepted
tasks. `--g3-per-seed` overrides that count for recovery prefixes. Output must
be new or empty. Four CPU validation processes are used by default; set
`--validation-workers 1` for sequential validation. No teacher, student, GPU,
or network is used. K counts verified executed trajectories, with instance
and group counts reported separately. Failed seed retries are excluded.

G1 moves one goal-type instance, retaining the original scene and goal. Pair
evidence comes only from selected seed `canContain` and `inReceptacle` facts.
It changes both `inReceptacle` and `objectAtLocation`, preserves receptacle
state, and refuses nested objects or unrecognized dependencies. An initial
placement observed in a seed is evidence even if that pair has no
`canContain` fact in the target game; that distinction needs to be considered
when interpreting initial-layout interventions versus legal placement actions.

G2 copies an exact goal form and literal grammar `task` rule from another
selected seed, substituting type names. Only scene objects, affordances,
lamps/appliances, and native placement pairs qualify. A task type absent
from K is unavailable; the factory never borrows another train game's template.
Goals already satisfied at reset are rejected. Scene state and domain remain
unchanged. The task text is the grammar's `task` rule expanded by `intro`;
the harness neither reads `traj_data.json` for this text nor synthesizes it
in `AlfredDemangler`.

G3 uses a genuinely replayed prefix of the verified successful seed actions.
It executes a refused action, takes a wrong-type object, repeats `look` twice
at a visited location, or puts the goal object in a wrong-type receptacle.
The prefix ends immediately after the error. These are recovery situations,
not new task instances. Its recovery witness must win within 40 steps
including the prefix, on two real resets.

Before generation, the factory freezes group identities from native held-out
directory names, main-checkout manifests/listings, and the worktree's split
files. It excludes **all** valid_seen as an additional conservative restriction,
as well as all valid_unseen. It reads no held-out game or trajectory. Evaluated
task lists in run manifests are read separately from checkpoint training
provenance. Extra identity manifests can be supplied with `--heldout`.

`manifest.json` records every attempt and rejection. Only `status == "valid"`
is usable. Rejected games are kept under `rejected/`; accepted game files
live under `data_root/train/synth/`. Their `walkthrough` is empty. The
`solvable` metadata bit is set only after real witness execution; both the
validated and final file hashes are recorded. PDDL and grammar stay unchanged
during this metadata update.

The worker uses installed TextWorld Fast Downward `PddlState.replan`, the
planner behind `AlfredExpert(planner)`. Commands are executed through the
unchanged `Environment`. Failure to obtain a witness is **unknown feasibility**,
not proof of unsolvability. The G2 parent-action negative control may
coincidentally win; this is recorded explicitly.

`witnesses/` contains privileged validation episodes and is **never for
training**. `validation_requests.json`, provenance, PDDL diffs, and validation
results are also audit data. Use `training_task_view()` as the explicit
task-ID/task-text whitelist; never serialize manifest rows into prompts. G3 prefix
files contain only action/visible-observation fields plus source metadata;
their recovery suffix appears only in the witness. No training dataset or
model update is produced by this factory.

This is a TextWorld feasibility spike, not validation of Unity geometry or
a scientific performance finding. Nested movable receptacles, slicing,
capacity changes, and unseen goal schemas are outside this implementation.
