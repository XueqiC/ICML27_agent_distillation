# V2 training interchange format (`v2-train-interchange/1`)

Written 2026-10-27 for method plan v1 §4 (E0.4 builds it on branch `repair-v01-eval`, E0.5 consumes it on branch `repair-v01-train`).
It is the only contract between data construction and training; neither side imports the other's code.

## File layout

`<dataset_dir>/` (exclusive create, sealed):
- `examples.jsonl` — one JSON object per line (below), sorted by `example_id`.
- `manifest.json` — `{schema: "v2-train-interchange/1", arm, created, code_revision, inputs: {path: sha256}, counts: {by_source, by_kind, by_task_type}, supervised_turns, example_ids_sha256, splits_sha256, theta0: {adapter_path, adapter_weights_sha256}, training_exposure_tasks: [group ids], excluded: [{reason, count}]}`.
- `examples.sha256` — sha256 of `examples.jsonl`.

## Example object

```json
{"example_id": "<sha256 of canonical [arm, source, task_id, episode_ref, step]>",
 "arm": "JOINT|JOINT_COMPLETE_ONLY|PREFIX_EXPERT|OFFLINE|CONTINUE",
 "source": "joint_local|joint_complete|prefix_expert|offline_teacher|k20_replay",
 "kind": "decision|trajectory",
 "task_id": "train/...", "group_id": "train/...", "task_type": "...",
 "messages": [{"role": "system|user|assistant", "content": "..."}],
 "supervised_assistant_indices": [ ... ],
 "weight": 1.0,
 "evidence": {"episode_ref": "<run dir>/records/<key>.json", "step": 17, "executor": "student|student_after_guidance|teacher",
              "local_event": "target_seen|target_taken|hands_freed|transform_done|placed|lamp_used_with_target|task_success|null",
              "episode_success": true, "label_source": "student_thought|teacher_rationale|teacher_reply", "label_check": "passed"}}
```

- `messages` use the exact student prompt contract (`sed.react.build_messages`: system prompt, JSON user turns with observation + admissible_commands, assistant replies `Thought: …\nAction: …`). No `Guidance:` text may appear anywhere in `messages` (guidance is removed from observations; history replies that depended on guidance are replaced, see E0.4 rules).
- `kind = decision`: `supervised_assistant_indices` = exactly the LAST assistant message (prefix turns are context, loss-masked).
- `kind = trajectory`: all assistant messages supervised (OFFLINE teacher trajectories, K20 replay, as the main-grid trajectory SFT).
- `weight` multiplies the per-example token-normalized loss exactly as the existing trainer's sample weights (if the trainer has none, weights must all be 1.0 and mixing is done by sampling ratio in E0.5).

## Validity rules (checked by both sides)

1. Every `task_id` is in `data/manifest/method_v1_splits.json` V2_pool (or the K20 view for `k20_replay`); none in U_dev, U_mech, D_probe, D_unseen_quick or dev140.
2. No message contains `Guidance:`; every assistant message parses to an action. For `kind = decision` the supervised action is in the admissible commands of the preceding user turn.
3. Each decision example's prefix is the real executed public history up to that step (observations exactly as recorded by the environment).
4. For `kind = decision`, rationale text in the supervised turn uses only prefix information (instance ids that appear in the prefix only).
5. `kind = trajectory` examples (OFFLINE teacher trajectories, CONTINUE / replay K20 demonstrations) must reproduce the main-grid trajectory-SFT training data exactly (byte-equal rendering); rules 2 (admissibility) and 4 (grounding) do NOT apply to them, because the baselines were trained on these trajectories unchanged, including the teacher's own refused actions and Thoughts. Their counts of supervised turns that are inadmissible or cite not-yet-seen instance ids are recorded in the manifest (`trajectory_diagnostics`) for information only.

6. `manifest.counts` must contain `by_source`, `by_kind` and `by_task_type`, each equal to the counts recomputed from `examples.jsonl`. Producers may add informational keys. A consumer accepts them and checks the ones it can recompute:
   - `by_local_event` equals the counts of every element of `evidence.local_events` over decision examples (rule 8). If `local_events` is absent, it counts `evidence.local_event` instead, omitting `null`;
   - `examples` equals the number of rows;
   - `task_groups` and `unique_instances` equal the distinct `group_id` and `task_id` counts.

   Any other key is refused.

7. **CONTINUE layout.** For `arm = CONTINUE` only, the root `examples.jsonl` may be byte-identical to `replay/examples.jsonl`: both hold the 20 K20 trajectories. The arm share of each batch is drawn from the root pool and the replay share from `replay/` with the same seeded replay draw as every other arm. For every other arm, a row present in both pools is refused.
8. **Informational decision evidence.** A `kind = decision` example's `evidence` may also contain these optional fields, all audit-only and never model inputs:

   | field | type |
   |---|---|
   | `local_events` | list of the `local_event` enum values, without `null` |
   | `complete_success_eligible` | bool |
   | `public_prefix_sha256`, `fact_state_sha256`, `facts_before_sha256`, `facts_after_sha256` | 64-hex sha256 |
   | `history_rewrites` | list of objects recording replaced history Thoughts |
   | `next_legal_target_take` | int or null |

   When `local_events` is present, `local_event` equals `local_events[0]`, or `null` if the list is empty. Any other extra field is refused. `kind = trajectory` evidence has `step = null` and `local_event = null`.

Revision 2026-10-27 17:5x CDT: rule 5 added and rules 2/4 restricted to decision examples, after the first build found 4/20 K20 and 62/200 V2_pool teacher trajectories that the main grid trained on but that fail admissibility/grounding.
Revision 2026-10-29 23:2x CDT: rule 6 added after the first real E0.5 canary refused the E0.4 OFFLINE bundle ("Manifest counts mismatch": E0.4 writes the informational keys by_local_event, examples, task_groups, unique_instances).
Revision 2026-10-30 00:1x CDT: rules 7 (CONTINUE root = replay) and 8 (E0.4 informational decision evidence) added, and rule 6 counts `local_events`; found when validating the real OFFLINE/CONTINUE bundles with the E0.5 reader.
