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
| Span/token boundary mapping | Use tokenizer offset mappings over the **same** target IDs returned by `train.dataset.encode_turn`. A token overlapping a selected character span is selected; reject a token overlapping both reason and action spans. Thus indivisible whitespace-boundary tokens may include adjacent whitespace. Prompts, observations, prior replies and the empty Gemma OFF thought marker stay masked. | tokenizer boundary convention |
| Action loss | Include native Gemma `<turn|>` (106) in action loss and action length, exactly once per reply. No hidden thinking or model control tokens are introduced. | registered harness termination requirement |
| Curriculum equation | alpha=beta=1; gamma=0 because black-box entropy is unavailable. Complexity is the count of actually selected reason/action tokens, including labels and turn ends. Stable ascending sort once, with original verified-collection order breaking ties; every epoch uses this exact order, including partial final batches. | unspecified paper weights / explicit black-box adaptation |
| Paper optimizer | Reuse shared FSDP2 full **text** parameter trainer: fp32 masters/AdamW state, bf16 compute, fp32 reduction, SDPA, gradient checkpointing, frozen original multimodal towers. AdamW lr2e-5, betas(.9,.999), eps1e-8, weight decay0, clipping1.0; 3% linear warmup (`ceil(.03 * updates)`) then cosine. Seed0. | existing Gemma full-training port; unspecified optimizer defaults |
| Batch64 | `b'=min(64,max(1,floor(N/8)))` for N verified trajectories. Expand turns after trajectory batching; exact global mean and zero-weight rank padding. | shared X5 rule |
| Paper20k×64×512 dose; local655M rule | Use **655,000,000** logical processed input positions as the literal local 655M budget (the unrounded paper product is 655,360,000). Count every input token, including masked prompt/history and repeated turn context; exclude zero-weight rank padding from the logical budget and report physical compute separately. Never replace this with supervised tokens or 20k optimizer steps. | explicit no-truncation token-dose adaptation and decimal convention |
| Budget closure | No trajectory truncation/splitting. Finish the first complete optimizer batch reaching each threshold and the final budget; report actual positions and threshold overshoot. Derive the exact update count from the ordered lengths before constructing the warmup/cosine schedule. | indivisible batch boundary |
| Checkpoints | Export 1/16,1/8,1/4,1/2,1 budget crossings, even if multiple thresholds cross in one update. Each gets a standalone checkpoint, artifact/provenance hashes, and immediate `models.json` entry at stage and pipeline root. `final=true` denotes complete immutable export, while `budget_complete` distinguishes the endpoint. Attempt suffixes preserve entries from interrupted/retried stages. No dev selection. | existing serving schema adaptation |
| Packing | Exact Gemma OFF turn packing, 32768 limit, explicit error on overlength; no truncation. Shared-history packing is not present in this branch and is not invented here. | spec's current trainer path |
| Support data | Only `--verified`, including nested views and the opt-in full TRAIN schema registered with AD-FTP. No teacher client, new labels, external pool, or training on dev/test. Curriculum complexity/counts are recorded in example provenance. | fixed experiment protocol |
| Resume | Hash-bound shared stage transactions. Complete stages/checkpoints are reused. An interrupted optimization stage restarts from the pinned base in a new retained attempt, as in existing ports; it does not claim optimizer-state resume. Every failed attempt's durable update exposures remain disclosed. | existing stage-resume convention |
| Evaluation gate | Opt-in full-checkpoint check reuses existing eight fixed verified TRAIN turns, teacher-forced HF/vLLM logprobs, and exact OFF prefixes. Standalone checkpoints have no adapter-disable noise reference: require pooled mean absolute error ≤.05 and no more than ceil(.05*N) absolute errors >.5. Bind checkpoint/config/template/collection/code/server identity; fail before dev evaluation. Existing baseline launchers stay unchanged. | full-checkpoint equivalence threshold registration |
| Evaluation | Every published checkpoint uses dev140, greedy, batch-invariant vLLM and existing seed0/H0 runner. `--mode evaluate` runs the equivalence gate before each evaluation. Standalone exports also fit `MERGED_MODEL_PATH` in `scripts/hpg_eval_adapters.slurm` after a passing gate. | shared protocol |

Run `python -m baselines.sad.pipeline --config ... --snapshot ... --verified ...
--output ... --gpus 4`. The wrapper is `scripts/hpg_sad.slurm`, with `SAD_THROUGH`
set to `prepare`, `train`, or `all`, and `SAD_MODE=evaluate` for evaluation.
Preparation can run in a CPU allocation. SAD makes zero teacher calls. GPU count
changes require matching the SLURM allocation and `NGPUS`; one GPU is supported
for a separately authorized K20 s0 smoke, which has not been run here.
