# SAD port registration — 2026-10-19

Source: Liu et al., *Structured Agent Distillation* (2025), as frozen in
`docs/baselines/NEW_METHOD_SPECS_1019.md`, SAD and Shared protocol. No released
repository is available in that specification. This is an independently written
paper/specification implementation; no scientific result or faithful empirical
reproduction is claimed.

| Paper section / specification | Replacement/change | Reason class |
| --- | --- | --- |
| Span-masked KL objective; user Q1 | Teacher distribution is one-hot on sampled visible tokens; KL reduces to CE. Reason/action weights both 1. Sum selected token losses across all turns in a trajectory; average only across the actual global trajectory batch. No per-span, per-turn or per-token normalization. | explicit black-box adaptation |
| Reason and action spans | A line-start `Thought:` section, including its format marker, continues to the next line-start labelled section. Supervise each `Action:` line including its marker, excluding its newline and any trailing unlabelled text. Other labelled sections and introductory text are masked. No Thought section means zero reason loss. An Action line is required. | H0 span grammar; paper has no ReAct offset parser |
| Parser-rejected teacher replies (2026-10-20) | Before span selection, mask the complete target when H0 `parse_action(action_raw)` raises `ActionParseError`. When recorded `action_outcome` is present, use `parser_error` and fail if it disagrees with parsing the original raw reply. Retain the reply verbatim in visible history; no reason, action, final or turn-end labels. Parsed replies, including inadmissible commands, keep the existing span rules. | explicit user-registered deviation shared with SmartAD |
| Span/token boundary mapping | Use tokenizer offset mappings over the **same** target IDs returned by `train.dataset.encode_turn`. A token overlapping a selected character span is selected; reject a token overlapping both reason and action spans. Thus indivisible whitespace-boundary tokens may include adjacent whitespace. Prompts, observations, prior replies and the empty Gemma OFF thought marker stay masked. | tokenizer boundary convention |
| Action loss | Include native Gemma `<turn|>` (106) in action loss and action length, exactly once per reply. No hidden thinking or model control tokens are introduced. | registered harness termination requirement |
| Curriculum equation | alpha=beta=1; gamma=0 because black-box entropy is unavailable. Complexity is the count of actually selected reason/action tokens, including labels and turn ends. Stable ascending sort once, with original verified-collection order breaking ties; every epoch uses this exact order, including partial final batches. | unspecified paper weights / explicit black-box adaptation |
| Paper optimizer | Reuse shared FSDP2 full **text** parameter trainer: fp32 masters/AdamW state, bf16 compute, fp32 reduction, FlexAttention for shared history (SDPA for selectable per-turn scoring), gradient checkpointing, frozen original multimodal towers. AdamW lr2e-5, betas(.9,.999), eps1e-8, weight decay0, clipping1.0; 3% linear warmup (`ceil(.03 * updates)`) then cosine. Seed0. | existing Gemma full-training port; unspecified optimizer defaults |
| Batch64 | `b'=min(64,max(1,floor(N/8)))` for N verified trajectories. Shared history forwards each trajectory once; per-turn scoring expands turns after trajectory batching. Both preserve the exact global trajectory mean and zero-weight rank padding. | shared X5 rule |
| Paper20k×64×512 dose; local655M rule | Use **655,000,000** logical processed input positions as the literal local 655M budget (the unrounded paper product is 655,360,000). Count every input token, including masked prompt/history and repeated turn context; exclude zero-weight rank padding from the logical budget and report physical compute separately. Never replace this with supervised tokens or 20k optimizer steps. | explicit no-truncation token-dose adaptation and decimal convention |
| Budget closure | No trajectory truncation/splitting. Finish the first complete optimizer batch reaching each threshold and the final budget; report actual positions and threshold overshoot. Derive the exact update count from the ordered lengths before constructing the warmup/cosine schedule. | indivisible batch boundary |
| Checkpoints | Export 1/16,1/8,1/4,1/2,1 budget crossings, even if multiple thresholds cross in one update. Each gets a standalone checkpoint, artifact/provenance hashes, and immediate `models.json` entry at stage and pipeline root. `final=true` denotes complete immutable export, while `budget_complete` distinguishes the endpoint. Attempt suffixes preserve entries from interrupted/retried stages. No dev selection. | existing serving schema adaptation |
| Packing | Exact Gemma OFF turn contexts, 32768 limit per context, explicit error on overlength; no truncation. As specified, shared-history scoring is now the default after its merge; `--scoring per_turn` retains separate forwards. Preserve disjoint SAD labels and masked reply text as history. Full/sliding visibility and positions are unchanged. Logical budget/checkpoint thresholds still count the original turn contexts; physical forward positions are separate. Scoring is bound into the run identity and disclosure. | 2026-10-20 execution-path change; no objective/dose change |
| Support data | Only `--verified`, including nested views and the opt-in full TRAIN schema registered with AD-FTP. No teacher client, new labels, external pool, or training on dev/test. Curriculum complexity/counts are recorded in example provenance. | fixed experiment protocol |
| Resume | Hash-bound shared stage transactions. Complete stages/checkpoints are reused. An interrupted optimization stage restarts from the pinned base in a new retained attempt, as in existing ports; it does not claim optimizer-state resume. Every failed attempt's durable update exposures remain disclosed. | existing stage-resume convention |
| Evaluation gate | Opt-in full-checkpoint check reuses existing eight fixed verified TRAIN turns, teacher-forced HF/vLLM logprobs, and exact OFF prefixes. Standalone checkpoints have no adapter-disable noise reference: require pooled mean absolute error ≤.05 and no more than ceil(.05*N) absolute errors >.5. Bind checkpoint/config/template/collection/code/server identity; fail before dev evaluation. Existing baseline launchers stay unchanged. | full-checkpoint equivalence threshold registration |
| Evaluation | By default every published checkpoint uses dev140, greedy, batch-invariant vLLM and existing seed0/H0 runner. `--mode evaluate` runs the equivalence gate before each evaluation. `--eval-fractions` / `SAD_EVAL_FRACTIONS` optionally limits evaluation to explicitly requested fractions, with the selection recorded in evaluation provenance. Standalone exports also fit `MERGED_MODEL_PATH` in `scripts/hpg_eval_adapters.slurm` after a passing gate. | shared protocol; explicit evaluation subset |

Run `python -m baselines.sad.pipeline --config ... --snapshot ... --verified ...
--output ... --gpus 4`. The wrapper is `scripts/hpg_sad.slurm`, with `SAD_THROUGH`
set to `prepare`, `train`, or `all`, and `SAD_MODE=evaluate` for evaluation.
In evaluate mode, `--eval-fractions 1` (or `SAD_EVAL_FRACTIONS=1`, also honoured
by the wrapper) selects only the final fraction, stored as `1/1` in `models.json`.
Comma lists such as `1/2,1` select multiple fractions; omitted/empty means all.
Unknown fractions fail before serving. All matching entries, including retained
training attempts, are selected in manifest order. Each evaluation invocation
records the request, selected model entries and input hashes under
`evaluations/<run_id>/provenance.json`; completed per-checkpoint stages remain reusable.
Preparation can run in a CPU allocation. SAD makes zero teacher calls. GPU count
changes require matching the SLURM allocation and `NGPUS`; one GPU is supported
for a separately authorized K20 s0 smoke, which has not been run here.

Parser-mask accounting is stored in each example's `label_provenance` (and thus
`example_sources.json`) and aggregated as `parser_rejection_mask` in the training
stage's `result.json` and checkpoint training state: affected `episodes`, masked
`replies`, and `supervised_tokens_removed`. Count each input episode once,
independent of epochs/rank padding. Removed tokens mean assistant-all target
positions before method-specific span selection, including native turn end;
the old SAD span parser could not define labels for these rejected replies.
Per-turn provenance identifies each masked reply and its removed-token count.
Fully masked turns have no separate scoring forward; later exact-context turns
retain their text, in both `per_turn` and `shared_history` scoring. Complexity
and supervised-token exposure include only remaining labels; the logical budget
continues to count target-bearing exact contexts, including their masked history.

This deviation applies only to SAD and SmartAD. Trajectory SFT / AD
(`baselines/kang`, `baselines/adftp`) retain their official assistant-all recipe
and supervise every reply, including H0 parser rejections.
