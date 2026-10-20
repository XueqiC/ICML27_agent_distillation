# AD-FTP port registration — 2026-10-19

Official helper pin: `8884b80ea3d22e53a2e1e4b9fd324600d13e0430`
([source](https://github.com/Nardien/agent-distillation/tree/8884b80ea3d22e53a2e1e4b9fd324600d13e0430)).
File references below are relative to `third_party/agent-distillation/`.
The governing local specification is `docs/baselines/NEW_METHOD_SPECS_1019.md`,
AD-FTP and Shared protocol. This is an engineering port of the partial helpers
and specified paper workflow; no experimental finding or end-to-end reproduction
is claimed. No upstream runtime/model is imported or substituted.

| Official file:line / specification | Replacement/change | Reason class |
| --- | --- | --- |
| `exps_research/first_thought_prefix/build_prefix_memory.py:30-32` | Literal `response.split("\n\n")[0]`, preceded by `Thought: `. The helper's extra trailing blank line is omitted, as explicitly specified locally; paragraph text is otherwise unchanged. | local specification / small attributed reimplementation |
| `src/smolagents/agents.py:1455-1459,1489-1492`; `exps_research/unified_framework/processors/agent.py:112-116` | Prefix applies only to the first agent reply. `teachers/openai_teacher.py` sends completed Responses input messages and has no assistant continuation/prefill interface. Use the specified required-beginning system instruction; reject a first reply whose raw bytes do not start with the prefix. Never repair or prepend returned replies. | registered API fallback |
| AD-FTP data steps 1–3 | CoT prompt asks for a visible step-by-step solution plan without actions/tools, using only allowlisted task instruction and actual reset observation. This is sampled visible text, never hidden reasoning. The goal is already present in ALFWorld's observation. H0 subsequent turns use unchanged `sed.react` messages/parser. | prompt wording and ALFWorld adaptation |
| Shared teacher/model protocol | gpt-6-luna only, configured sampling parameters with 1024 output tokens for both CoT and action calls. CoT cap is an explicit implementation choice. No sampling-parameter fallback or model substitution. Returned model ID, timestamps, raw text, prompts and usage are retained. | X1/X3; output-cap registration |
| AD-FTP data steps 4–5 and cap | At most three independently reset attempts per actual support instance, one fresh CoT per attempt, at most 40 agent calls. Durable `CappedTeacher` cap 123 per task gives the cell hard cap `123 * unique_instances`. Fresh attempt cache namespaces prevent failed attempts from reusing the same sampled CoT. POSIX limiter directory is shared across cells. | exact bound; engineering partition of cell cap |
| `teachers/capped.py` convention | Failed/unknown transport usage permanently halts that cell; resume never resets a counter. Prefix mismatch is a generation rejection, consumes its attempt and permits the next one. Started interrupted environment attempts consume a slot and retain unknown outcomes; a completed successful episode survives interrupted publication. | conservative durable accounting |
| AD-FTP fallback rule; verified loader schema | Copy the selected successful execution, or the existing trusted demonstration if all three fail, to a canonical `attempt=1` publication slot. This is **not** a claim of another execution. All actual FTP attempts, raw CoT, rejected outputs, ledger/counter/blobs, transitions and original parent receipt are nested in the slot's hashed `teacher/` bundle. Fallback also copies original teacher evidence. Counts and raw attempt numbering are explicit in `ftp_attempts`, `fallback_count`, `ftp_attempt_count`, `api_attempts`, token totals and new transition totals. | layout compatibility; no fourth-attempt fiction |
| Shared K grid | Preserve the source selection's parent-slot mapping, including replacements, so existing `materialize_subsets` yields identical nested support instances. K counts the original trusted support trajectories; generated and reused executions, unique instances and groups are disclosed separately. No new tasks. | unchanged support identity |
| Full-training cell | Use the concurrent `wt/fulltrain` worktree's `all_train_v1` selection/validation schema (narrow same-repository compatibility port). Selection K counts candidates; collection K counts verified successes. Siblings count as distinct instances in the same group. Generate only for verified support successes; copy the source's failed-candidate receipts/evidence unchanged for complete coverage, with no FTP calls on those candidates. Disclose these as reused costs. Existing stratified selection semantics are unchanged. | future full-collection schema |
| AD training and SAG | Call the unchanged Kang command: r64/alpha128/dropout .05, seven text linear projections, 2e-4 linear/zero warmup, 2 epochs, batch 8→b', seed 0, exact OFF turn packing. Prefix tokens are in the saved assistant target. Existing greedy and SAG N8/T.4/top-p.8 evaluation run after the LoRA equivalence gate. | same AD recipe; existing Gemma/harness adaptation |
| Shared pipeline | Hash-bound stages and immutable identities; completed stages reused, partial stages retained. Collection work/counters survive stage retries. Source collections are read-only; publication is outside `data/verified`. CPU-only collection job; separate GPU training/evaluation allocations. | engineering |

Run CPU collection with `scripts/rai_collect_adftp.sh [--resume]`. It preflights
all three K100 collections and `seeds_full_train` before a call. Configure
`COLLECTION_ROOT`, `ADFTP_ROOT`, `ADFTP_RATE_LIMIT_DIR`, `ENV_PYTHON`, and
`COLLECT_CONFIG`; no cluster job or API call is submitted by installation/tests.
Outputs are `<ADFTP_ROOT>/<cell>/collection`, with `subsets/k5`, `k10`, `k20`,
`k50` for K100. The collection can be passed unchanged to
`baselines.kang.pipeline --verified ...` and its existing SAG evaluation.

For the method wrapper, run `ADFTP_THROUGH=collect` on CPU; run
`ADFTP_THROUGH=train` with `VERIFIED` pointing to the generated collection or its
nested view and a separate `ADFTP_OUT` per training cell. This GPU stage never
calls the teacher. `ADFTP_MODE=evaluate` evaluates both greedy and SAG on dev140;
`models.json` is published at the run root and under `training/`. No dev-based
selection. The CLI also supports `--through all` for explicitly managed local
execution; the SLURM wrapper deliberately requires separate allocations.

Training exposure is in `training/disclosure.json` and stage update ledgers;
collection costs remain in `collection/manifest.json` and the hashed bundles.
Do not sum nested views, copied evidence, or trusted fallback costs as new calls.


## 2026-10-19 engineering fix: fail loudly on teacher transport

Transport failures (HTTP errors, timeouts, incomplete/unknown responses, halted
collectors and exhausted teacher budgets) raise through the stage boundary.
Failed stage attempts retain sanitized usage, raw returned outputs, counters and
real execution prefixes; they never publish a completion with missing labels.
Completed malformed/invalid outputs keep this method's existing rejection or
masking behavior. `teacher_report.json` counts terminal transport failures,
failed HTTP attempts, parser rejections and method rejections separately. Existing
reports include the same counters; disclosure ledgers retain historical failures.

This is an **engineering fix, no algorithm change**: prompts, labeling/masking,
selection, trusted K, task scope and configured caps are unchanged. See
[`docs/TEACHER_TRANSPORT_FAILURES.md`](../../docs/TEACHER_TRANSPORT_FAILURES.md)
for recovery, non-refundable reservations and the offline regression matrix.

## 2026-10-19 baseline transport correction

**engineering fix: matches upstream client defaults**. The shared opt-in
[transport policy](../../docs/baselines/TRANSPORT_POLICY.md) supplies 600-second
read/write/pool and 5-second connection timeouts, three audited HTTP attempts
(SDK max_retries=2 behavior), and SDK exponential backoff/Retry-After handling.
Shared-key rate-limit 429s retain retry429 backoff up to 20 attempts without
spending another logical slot or method try; quota 429s halt immediately.
Exhaustion fails loudly through `teachers/stage.py`, never as a rejection.
Run/counter identities, disclosures and HTTP ledgers record the effective policy.
This entry supersedes earlier no-retry/60-second transport descriptions and
references to capped logical slots as HTTP attempts; labels, masks, prompts,
environment attempts and training are unchanged. Trusted seed acquisition
remains at 60 seconds with its existing configuration and verified identities.
