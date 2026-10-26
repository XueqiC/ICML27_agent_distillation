# Repair pilot M0a

Later CPU milestones: [M0b-1](M0b-1.md) (splits/coverage) and
[M0b-2](M0b-2.md) (teacher stages, E1/E2 and prefix labels), and
[M0b-3](M0b-3.md) (collection/export, fixed-checkpoint probes and stage scripts;
training is on the separate 2506fa9-based branch).

This worktree package implements the environment portion of design v0.1,
sections 3, 4, 6.3, 7, 8.4 and 18. It makes no teacher calls and loads no model.
Existing harness, trainer, evaluation and keeper modules are unchanged.

- `Patch` / `PatchBlock` and `compile_patch` use original, zero-based indices.
  Invalid candidates raise `InvalidPatch(reason)`; more than four blocks raise
  `ExtendedRepair` and must be reported as a separate variant.
- `align_actions` saves the forward DP path, with ties resolved as keep,
  replace, delete, insert. `supervised_positions` indexes corrected actions;
  `supervised_decisions` maps those indices to actual policy turns and their
  newly executed public prefixes, accounting for parser feedback turns.
- `execute_actions(factory, task_id, actions, max_steps)` accepts environment
  commands or recorded ReAct replies. The shared parser classifies every turn;
  inadmissible commands are still issued. Parser feedback is explicitly marked
  `analytic_verifier` and consumes no environment step. The turn cap matches
  the runner, including parser failures. Success reads only the harness `won`
  field; unknown verification remains `"unknown"`.
- `ResultCache` is explicit and process-local. Every lookup follows a real
  reset. The key binds environment version, game content, reset fingerprint,
  canonical actions and seed. Exact parsed commands, both step limits and the
  initial audit digest additionally scope entries. Hits retain prior execution
  evidence, identify its source trajectory, and charge zero steps. Errors are
  not cached. No snapshot/restore method is used.
- `MinimalityChecker(factory, ablation_replays=12, total_replays=24)` verifies
  the original failure and candidate, performs whole-block reverts, and repeats
  the selected success without cache. Pass previous proposal/retry counts as
  `replays_used` / `ablation_replays_used`; these are per-failure totals, not
  fresh budgets per candidate. A replay slot is reserved for confirmation.
  Supply `original_trajectory` when available to check the recorded failure
  exactly. An omitted reference confirms only the current original-action
  failure and consistent resets, not equivalence to an unavailable old trace.
  `check_minimality=False` returns `BEST_OBSERVED`. A complete pass is required
  for `BLOCK_IRREDUCIBLE_WITHIN_DECLARED_BLOCKS`; unknown ablations cannot earn
  that status. Joint necessity is limited to the two declared blocks, fixed
  suffix, initial state and step limit, and is withdrawn on replay divergence.
- `BudgetLedger` uses `sed.storage.JsonlLedger`. Attempts/reservations are
  unsettled; completion, failure, rejection, retry and cache events remain in
  the append-only history. Summaries sum settled usage only, reject duplicate
  settlements and report unreported quantities separately. Evaluation steps
  have their own interaction counter. Record run-level quantities such as
  trainable parameters once, rather than on every update.
- Records serialize to JSON-compatible dictionaries and round-trip with
  `from_json(dict_or_json_text)`. IDs derive from canonical content. Full
  records are evidence, not prompts: use `TrajectoryRecord.decision_context`
  or `supervised_decisions`, which read only explicit public-field whitelists.
  Audit digests, rewards and verification labels remain outside `visible`.

The calling collection layer must authorize frozen training tasks and persist
returned records and its run provenance. The acceptance CLI does this itself;
library helpers do not infer split membership from an instruction or task name.

Run the acceptance tool from this worktree with five or more distinct training
task episodes in the existing `sed.runner` episode format:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m tools.repair_determinism_check \
  --config configs/base_e4b.yaml --data-root /path/to/alfworld \
  --episodes /path/to/episode1.json /path/to/episode2.json \
             /path/to/episode3.json /path/to/episode4.json /path/to/episode5.json \
  --output /path/to/new-report.json
```

The supplied config must point to the frozen manifest/split and match episode
seed and step limit. Task IDs come from the episodes. The tool authorizes each
task before reset, performs two uncached full replays plus truncated-action and
short-limit checks, compares recorded transitions and audit digests, and writes
every result. It exits nonzero on a mismatch or fewer than five distinct tasks.
The adjacent `<report-stem>_artifacts/` contains provenance and the budget ledger,
including hashes of this package (the existing provenance helper does not scan
new top-level packages). Use a fresh output path for each acceptance run.

The unit tests use deterministic fake backends through the real harness adapter.
Passing them establishes software behavior, not real ALFWorld determinism or a
scientific mechanism finding. Real acceptance requires the recorded training
episodes and their matching frozen data/configuration.
