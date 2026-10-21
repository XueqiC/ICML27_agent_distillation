# SmartAD: registered ALFWorld H0 adaptation

The [shared LoRA stability deviation (2026-10-21)](../kang/DEVIATIONS.md#shared-lora-stability-deviation--2026-10-21) governs the identity-bound learning-rate/warmup overrides for SmartAD, trajectory SFT and AD-FTP.

Source: *Capacity-Aligned Agent Distillation for Small Language Models*,
Findings of ACL 2026, §§3.2–3.3 and Appendix A,
<https://aclanthology.org/2026.findings-acl.1349/>.
Local source: `docs/baselines/smartad_paper_text.txt`. No official implementation
is available; this is a paper-based implementation. Registered before any live
collection or GPU experiment. Offline tests do not establish scientific results.

| Paper / requirement | Implementation and registered adaptation |
|---|---|
| Teacher/student models and CodeAct QA/math tasks | Configured `gpt-6-luna` sampled text/actions and `google/gemma-4-12B-it@707f0a3b`; ALFWorld H0 support TRAIN tasks. No checkpoint from the paper is loaded. No teacher logits, hidden reasoning, or internals are requested. |
| §3.2: ten temperature-1 candidates per task | Ten candidate slots, each grounded in a fresh actual reset and rollout, up to 40 H0 turns. Filter on real environment wins, with no parser/inadmissible/failed tool turns. The standard ALFWorld `Nothing happens.` failure observation is also rejected. Do not replenish failed candidates; retain their raw evidence and rejection. |
| No duplicate teacher use | Collect the nine always-needed new slots first. Check the trusted source collection identity, sampling settings, H0 settings, and every successful turn's call receipt. If equal to the effective candidate sampling, reuse it as candidate 1; otherwise execute candidate 1 and record the reason. Source usage is inherited, not billed again. A completed task's reuse decision is frozen across resume. |
| Temperature unsupported by provider | Request temperature 1.0 on an actual rollout call, with 1,024 output tokens; no separate probe. Only HTTP400 with `temperature` and `unsupported_parameter`/`unsupported_value`, or `invalid_request_error` with null code, permits switching to the provider default. This is LEAP's structured-error rule. The failed request, unknown usage, and reservation remain; a durable recovery references the rejected ledger prefix. Other errors halt. `sampling_fallback.json` registers the decision and evidence. No claim about live provider acceptance is made by these tests. |
| Caps and transport | `smartad` opts into `teachers/baseline_transport.py` SDK defaults. Each of at most ten new candidates has a non-refundable `CappedTeacher` cap of 41 logical calls (40 policy calls plus at most one explicit sampling rejection); task ≤410, cell ≤410 × support instances. SDK-equivalent HTTP/429 retries stay inside a logical call and are separately ledgered, as for other baselines. Shared POSIX limiter directory across collection cells. |
| Resume | Each candidate has its own durable cache, start marker, episode, counter, ledger and hash receipt. Never restart or replace an existing candidate. A complete episode interrupted before receipt publication is reused. Unknown/interrupted transport or execution halts for investigation; no fabricated restored state or automatic resampling. |
| §3.2 / Eq. 3–4 selection | Initial base student in bf16, eval mode, no gradient, exact full preceding H0 OFF context for every assistant turn. Per-token mean NLL inside each turn, then equal mean across turns. Include the native assistant turn-end target as in the shared SFT encoding. Score all correct candidates, choose minimum; ties use stable candidate ID. Incorrect candidates retain a null score and rejection. Selection is fixed before training, never iterated or tuned on dev. |
| Zero correct candidates | Keep the existing trusted executed demonstration, preserving the support task set. This user-requested AD-FTP-style fallback is outside the paper's successful-pool assumption. Count per-task fallbacks in `smartad_selection.json` and collection manifests; preserve all failed candidates. |
| §3.3 reason/action/final | H0's executed `Action:` line, including its header and command, is action; the last assistant turn's action span is final. If multiple Action lines appear, H0 executes the last one. All other supervised text, whitespace/format tokens and native turn-end token are reason. Weights 1 / 1.5 / 2. Offsets must reproduce the exact target IDs; a token overlapping an action span takes its action/final weight. Observations, system/user text, previous turns and OFF prompt prefill stay masked at −100 with zero weight. Unlike SAD, SmartAD supervises all visible assistant text. |
| Parser-rejected teacher replies (2026-10-20) | Before segmentation, mask the complete target when H0 `parse_action(action_raw)` raises `ActionParseError`. When recorded `action_outcome` is present, use `parser_error` and fail if it disagrees with parsing the original raw reply. Retain the reply verbatim in visible history; all labels −100, segment labels absent and weights zero, including native turn end. Parsed replies retain the existing reason/action/final rules. This explicit user-registered deviation is shared with SAD and also covers trusted fallback demonstrations. Candidate correctness filtering and Eq. 3 selection are unchanged. |
| Eq. 5 | Sum per-token CE × label-position weight, divided by the sum of weights over the actual effective batch. Stream complete turns and accumulate gradients against that one denominator. This is neither a mean of turn losses nor a mean of trajectory means. No extra division by token count or number of turns. |
| Appendix A training | LoRA SFT, two epochs, AdamW 2e-4, gradient checkpointing. Shared AD/Kang unspecified details: r64/α128/dropout .05 on all seven attention/MLP language-model linear projections; frozen multimodal towers, fp32 adapter masters, bf16 base/compute; linear schedule, no warmup, seed 0, turn packing, weight decay 0, clipping 1. |
| Effective batch 4 on two GPUs | One GPU with streamed accumulation. Shared rule `b' = min(4, max(1, floor(N/8)))`, N = selected trajectories, counting each trajectory once. Final partial batches use their actual weight denominator. |
| Paper max length 10,240 | Retain 32,768 exact-context limit to avoid truncating ALFWorld trajectories. Overlength raises; no truncation in selection or training. |
| Inference | Shared unchanged H0, thinking OFF, 1,024 output tokens, 40 turns, greedy temperature 0, dev140. Shared HF/vLLM adapter equivalence gate and batch-invariant evaluation. No SAG or extra inference search. |
| K and nested/full cells | K counts trusted seed trajectories; ten candidates do not multiply K or the unique task count. One pool per K100 s0–s2, one selection per base snapshot; K5/10/20/50 are shared receipt views published after selection. Full TRAIN accepts `data/verified/seeds_full_train_r2` (or any valid full-TRAIN source path), retaining inherited failed seed attempts without collecting on them. |

The implementation reuses this repository's AD-FTP accounting/publication pattern,
Kang LoRA trainer, `train.dataset.encode_turn`, shared loss scorer, and LEAP's
existing transport error classification. It does not copy an external repository.
Frozen inputs, code hashes, dependencies, environment settings, source receipts,
returned teacher IDs/timestamps, raw outputs, token usage, negative executions,
parent seeds and training exposures remain in provenance and audit artifacts.
Training loads only the shared visible-field whitelist. Audit outcomes are used
only for correctness filtering and accounting, never as prompt input.

Parser-mask accounting is stored in each example's `label_provenance` (and thus
`example_sources.json`) and aggregated as `parser_rejection_mask` in the training
stage's `result.json` and checkpoint training state: affected `episodes`, masked
`replies`, and `supervised_tokens_removed`. Count each input episode once,
independent of epochs/rank padding. Removed tokens mean assistant-all target
positions before segmentation, including native turn end. Per-turn provenance
identifies each masked reply and its removed-token count. Fully masked turns
have no scoring forward or weight in Eq. 5; later exact-context turns preserve
their text in both scoring paths.

This deviation applies only to SAD and SmartAD. Trajectory SFT / AD
(`baselines/kang`, `baselines/adftp`) retain their official assistant-all recipe
and supervise every reply, including H0 parser rejections.

## Stage usage

All commands below describe future authorized runs; implementation validation
uses fake transport/environment and CPU models only.

```bash
# CPU only: all three K100 sources; full TRAIN also when its manifest exists.
ENV_PYTHON=/path/to/env/python bash scripts/rai_collect_smartad.sh
# Add --resume to reuse durable attempts in the same collection tree.

# One GPU: score the entire pool once; nested views publish after selection.
python -m baselines.smartad.pipeline --through select \
  --verified reports/runs/smartad-collections/k100_s0 \
  --snapshot /path/to/base-snapshot --output reports/runs/smartad-selection/k100_s0

# One GPU: use selected collection or its subsets/k5, k10, k20, k50.
python -m baselines.smartad.pipeline --through train \
  --verified reports/runs/smartad-selection/k100_s0/collection/subsets/k20 \
  --snapshot /path/to/base-snapshot --output reports/runs/smartad-k20s0

# One GPU: same verified input, snapshot and training output.
python -m baselines.smartad.pipeline --through evaluate \
  --verified reports/runs/smartad-selection/k100_s0/collection/subsets/k20 \
  --snapshot /path/to/base-snapshot --output reports/runs/smartad-k20s0
```

`scripts/hpg_smartad.slurm` accepts `SMARTAD_THROUGH=collect|select|train|evaluate`,
`SMARTAD_OUT`, and `VERIFIED`. Select consumes the pool; train/evaluate consume
the selected verified collection. Collection defaults to a CPU allocation.
Select/train/evaluate require one GPU. `SMARTAD_SUBSET_K="5 10 20 50"` is frozen
at collection; `SMARTAD_RATE_LIMIT_DIR` must be shared across collectors.
The rai launcher accepts `SMARTAD_FULL_SOURCE` for an alternate full source.
`--mode prepare --through train` binds training inputs without loading a model.

## engineering fix: relocatable selection (path map)

Selection accepts repeatable `--path-map OLD=NEW` entries, for example
`--path-map /collection-host/project=/selection-host/project`. Both paths must
be absolute; duplicate or overlapping prefixes (including destinations and
chains) are rejected. Matching uses whole path components. Relocate the trusted
source collection too when its published path is unavailable. Missing directories
and episode/teacher hash mismatches fail before scoring or copying.

The launcher accepts `SMARTAD_PATH_MAP` as whitespace-separated `OLD=NEW` entries
(paths with whitespace require the CLI); entries are passed as literal arguments,
never evaluated. The map is recorded in selection `provenance.json` details and
`result.json`, outside scoring identity. Published pool records, pool hashes,
score bindings, choice rules and selected evidence stay unchanged; the selected
collection uses relative paths and needs no map for training/evaluation.
