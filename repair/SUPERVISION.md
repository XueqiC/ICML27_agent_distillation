# E0.4 supervision builder

`repair/supervision.py` and `tools/build_v2_dataset.py` produce
`v2-train-interchange/1`, following the
[V2 interchange contract](../../ICML27_agent_distillation/docs/method/V2_interchange_spec.md).
They perform no environment interactions, training, or checkpoint selection.
Existing collection, execution, facts, prompts, and training modules are unchanged.

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python \
  tools/build_v2_dataset.py build --arm JOINT \
  --splits data/manifest/method_v1_splits.json \
  --inputs /path/to/joint-run --output /path/to/new-dataset \
  --rationale-teacher --budget-ledger /path/to/shared-label-budget.jsonl \
  --max-teacher-input-tokens 5000000 --max-teacher-output-tokens 500000
```

The other arms are `JOINT_COMPLETE_ONLY`, `PREFIX_EXPERT`, `OFFLINE`, and
`CONTINUE`. Decision arms require run directories with `plan.json`,
`provenance.json`, `run_manifest.json`, and `records/*.json`. OFFLINE and CONTINUE
need no run directories: their sources are frozen receipts. For these arms,
optional `--inputs` may name the frozen collection, its `subsets/k20` view, or
theta0's training output directory; arbitrary runs cannot replace the receipts.

The collection defaults to the exact `rebuild_arguments.verified_root` in the
split freeze. This worktree's freeze refers to the sibling checkout's
`data/verified/seeds_full_train_r2`. `--verified-root` supports relocation while
requiring the same frozen hashes. The configured student is read from
`--config` (default `configs/base_e4b.yaml`), never substituted.

Theta0 defaults to `e4b-tsft-k20-e32-lr5e-5`'s existing
`reports/runs/{baselines_loni,baselines}/.../stages/train-sft/attempt-000/output`
in that collection's project, in the listed order. Use `--theta0-training-dir`
on another host. Its `provenance.json`, final K20 training state, adapter config,
and adapter weights must agree; the recorded training support manifest hash
must equal the frozen K20 view. The adapter path and weights SHA256 are recorded.
No weights are loaded into a model.

## Decision supervision

JOINT accepts completed, nonvoid, prefix-verified **G** records on exact V2_pool
trials. U/G1 records are counted as exclusions. The execution record, rather than
reviewer proposals, supplies actions, observations, and environment success.
Suffix executor records must match the execution replies/actions. Replayed
student-prefix steps are also considered for JOINT, with the same eligibility
checks as subsequent steps.

Facts are recomputed with `repair.facts.FactTable` using only the stripped public
observation, executed action, and public outcome. Stored review facts, metrics,
private audit blobs, game files, and task-path semantics cannot establish local
progress. A local target requires an actually executed, admissible, legal,
nonrefused transition. An earlier refusal permanently excludes that exact action
from later JOINT targets, even if the later execution succeeds.

| Event | Required public evidence |
| --- | --- |
| `target_seen` | A previously unseen goal-object instance enters the fact table after go to/open/examine/look. Initial sightings do not qualify. |
| `target_taken` | Legal take adds the goal instance to known possession. |
| `hands_freed` | Legal move/put of a nongoal object changes hands from known occupied to free; the next goal-object take in this episode is legal and not refused. A later successful retry does not rescue an intervening failed take. |
| `transform_done` | Legal heat/cool/clean adds the corresponding state for that goal instance. |
| `placed` | Legal move/put changes that instance's location to the goal receptacle type; any required state is already verified for that same instance. |
| `lamp_used_with_target` | Legal use of a desklamp while holding the goal instance adds its lit state. |
| `task_success` | The last executed step of the environment-success execution. |

Complete-success supervision includes legal, nonrefused steps in successful
executions, excluding earlier-refused actions and repeats of the same action
with the same pre-action fact state. The state hash retains current location,
condition values, observed locations, receptacles, transformations, lamps, and
refused-action identities. It omits accumulating evidence step numbers, sighting
history, and counters, so a repeated look cannot appear novel solely because
the evidence timestamp changed.

JOINT unions these criteria **once per step**. Local eligibility takes precedence
for `source=joint_local`; `complete_success_eligible` separately records complete
eligibility. `evidence.local_event` retains the contract's scalar value, while
`evidence.local_events` retains all qualifying events in documented event order.
JOINT_COMPLETE_ONLY uses `source=joint_complete`. Event counts count each event,
not just the scalar primary event.

Inputs use `sed.react.build_messages` over the actual public prefix. Only the
appended `Guidance:` suffix is removed from observations. Existing environment
text and admissible commands retain the shared visible-field rendering. Failed
and refused replies remain context with masked loss. An unparseable historical
reply makes later examples ineligible under interchange rule 2; it is never
replaced with an invented action.

Teacher / student-after-guidance replies, and student Thoughts mentioning
guidance, hint(s), or suggest(ed/ions/ing), receive sequential L labels. The
replacement retains the original history Action line. Each step's label is
constructed once and reused in subsequent histories; future actions, observations,
reviewer guidance, success, task IDs, and audit fields never enter L prompts.
The target uses the executed action. Independent student Thoughts are retained;
an ungrounded student target becomes Action-only, while its executed history
reply remains context. Every decision supervises only its last assistant turn.

`--rationale-teacher` uses the existing capped factory and `RepairTeacher` with
the unchanged `repair.prompts.RATIONALE` prompt. Shared accounting reserves every
transport attempt before HTTP; known usage settles, unknown usage retains its
full bound. Both explicit token caps and `--budget-ledger` are mandatory for real
L calls. Each output cap is 1024 including reasoning-token accounting. The visible
rationale is limited to 64 tokenizer tokens. When no local tokenizer exists, a
conservative 64 UTF-8-byte bound replaces whitespace counting for this cap.

The existing `parse_label` and `prefix_checks` enforce schema, prior step citations,
instance grounding, and unsupported flags. Additional checks prevent guidance
references and grounding an invented instance solely through an earlier Thought.
These are structural grounding checks, not a semantic proof of every prose claim.
A missing, invalid, unsupported, overlong, or ungrounded label yields an Action-only
target with an explicit `label_source`/`label_check`; raw output and rejection are
retained. Transport, truncation, unknown-usage, quota, and budget failures abort
without a sealing manifest. No failed response becomes a model action or label.
Without `--rationale-teacher`, needed labels are Action-only without API calls.

PREFIX_EXPERT requires successful, nonvoid, prefix-verified takeover records. It
supervises only teacher steps at/after `branch_index`, dropping refused steps.
The student's prefix and earlier teacher replies remain context. Targets retain
the teacher's own visible reply, including Thought. No L call or relabeling is
performed. Ungrounded targets are excluded and counted.

## Trajectories and replay

OFFLINE selects exactly the recorded verified teacher episode for each V2_pool
trial, checking its receipt identity, episode hash, full evidence hashes, and real
environment win. CONTINUE selects the exact 20 K20 receipts used by theta0.
Every arm writes those K20 trajectories again under `replay/`, separately from
its primary examples. CONTINUE deliberately has the same K20 examples in both
locations; E0.5 chooses its mixing policy and must not count them as 40 seeds.

Each is one trajectory example with **all** assistant turns supervised. Rendering
uses the same `build_messages` loop as `train.dataset.episode_examples`; that
module exposes a token renderer rather than a standalone message renderer.
Tests capture its `encode_turn` inputs and compare canonical message bytes on
synthetic trajectories (including inadmissible actions and ungrounded Thoughts)
and, when available, two recorded K20 trajectories.
No tokenizer, model, or teacher is needed for these comparisons.

Under revised interchange rule 5, trajectory messages must byte-reproduce the
main-grid trajectory-SFT rendering, including the teacher's refused actions and
original Thoughts. Rules 2 (supervised-action admissibility) and 4 (rationale
instance grounding) apply only to `kind=decision` supervised turns. Trajectory
violations of those two criteria are informational diagnostics and do not block
sealing. All decision eligibility and validation checks remain in force.

Each directory's manifest contains `trajectory_diagnostics`, keyed by trajectory
`example_id`, including entries with both counts zero:

```json
{"trajectory_diagnostics": {"<example_id>": {
  "inadmissible_supervised_turns": 1,
  "not_yet_seen_instance_supervised_turns": 2
}}}
```

The first count compares each supervised action to the immediately preceding
user turn's admissible commands. The second counts supervised Thoughts mentioning
an instance absent from the public prefix before that turn (user observations,
admissible commands, and prior executed actions). Prior Thought claims and future
sightings do not establish grounding. Each affected turn counts once per category,
even if it names multiple unseen IDs; a turn may contribute to both categories.
Decision examples have no entries here. Primary and `replay/` manifests report
their own examples separately; CONTINUE's duplicate replay is not extra exposure.

The builder never drops a trajectory turn, rewrites teacher prose, or changes an
admissible-command list to clear a diagnostic. Split membership, guidance absence,
action parsing, public-field whitelists, full-turn supervision, and frozen evidence
checks still apply. A mandatory K20/OFFLINE trajectory failing these checks blocks
sealing and writes `validation_failures.json`, with source, task, episode reference,
and reason. All mandatory trajectories are checked before L is constructed or called.

**Revised-contract frozen-data check (2026-10-27, read-only):** all 20 K20 and
200 V2_pool/OFFLINE trajectories pass, up from 16 and 138 under the old rules.
All 220 match the main-grid rendering byte for byte at every assistant turn.

| Frozen view | Passing trajectories | Supervised turns | Inadmissible turns (trajectories) | Not-yet-seen-instance turns (trajectories) |
| --- | ---: | ---: | ---: | ---: |
| K20 / CONTINUE / replay | 20/20 | 315 | 5 (4) | 0 (0) |
| V2_pool / OFFLINE | 200/200 | 3867 | 86 (51) | 37 (15) |

There are 4 K20 and 62 OFFLINE trajectories with at least one diagnostic; the
OFFLINE categories overlap. Counts cover every supervised turn, unlike the old
first-failure report. The views contain 20 and 200 unique instances and task groups,
respectively. The check validated frozen receipt identities, environment-win
evidence and hashes, then rechecked all 18,179 input files unchanged. File writes
and network access were blocked; no dataset was sealed, model run, or teacher called.
These are data-validation counts, not model-performance results.

Split checks use exact frozen trial IDs and prohibit evaluation groups, including
sibling trials. V2's additional cross-split task-name exclusions remain enforced.
K20 uses its original frozen view: a TRAIN task and a valid_seen task with the same
basename are distinct instances/groups; K20 is not retroactively reselected.

## Artifacts, validation, and tests

The output directory is exclusively created. Never reuse an incomplete output:
its retained ledgers are evidence, and its shared budget is not reset. A completed
root and its `replay/` each contain sorted `examples.jsonl`, `examples.sha256`,
and `manifest.json`. The manifest is written last, after validation and an input
hash recheck. `example_id` is SHA256 of canonical JSON
`[arm, source, task_id, episode_ref, step]`; trajectory step is JSON null.
`example_ids_sha256` hashes the sorted ID list as canonical JSON. Identical source
paths and inputs yield identical examples/IDs; manifest creation time can differ.

Manifests contain the interchange fields plus local-event counts, unique instance
and group counts, token counts, exclusion reasons, code hashes, dependency versions,
student identity, environment configuration, teacher alias, call IDs/model IDs/
timestamps/usage, and label-artifact hashes. `training_exposure_tasks` lists groups
represented in that directory; replay has its own list. Actual training exposures
and new environment transitions are zero. Original execution/teacher provenance
remains reachable through hashed input records and episode evidence. K counts
executed seed trajectories, never the number of groups or expanded examples.

Token counting uses only the configured local snapshot (`--snapshot` supports a
hash-checked relocation), with `local_files_only=True`; no model is loaded. If it
is unavailable, counts use per-turn `ceil(chars/4)` explicitly marked as estimates.
The count scope is assistant text before chat-template control tokens. Real
validation failure is never hidden by this accounting fallback.

`--dry-run` creates an explicitly synthetic, `training_eligible=false` dataset,
using a fake L client through the same capped transport and shared accounting.
The same validation and informational trajectory diagnostics apply. Test fixtures
exercise all five arms, including unchanged trajectories with inadmissible or
ungrounded teacher turns, strict rejection of either in decision targets, and
blocking of guidance leaks and unparseable trajectory replies before any L call.

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q \
  tests/test_supervision.py tests/test_joint.py tests/test_takeover.py tests/test_facts.py
```

Tests block network/API access and use fake clients/environments. They establish
engineering behavior only, not a scientific finding or baseline reproduction.
