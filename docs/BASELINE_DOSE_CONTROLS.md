# Baseline dose controls (approved 2026-10-06)

## Why
Seed-SFT on K verified seeds dips below base at 30–60k supervised tokens and only clears it after ~100k (seedsft_sweep.md).
Every official baseline recipe trains its SFT stage for 1–3 epochs, which at K ≤ 20 lands inside that dip (Kang K20 s0: 63/140,
AgentGym BC 59, AgentEvol 62–66; base 62). An official-recipe row alone therefore cannot tell "the method's ingredient does not help"
apart from "the recipe was calibrated for thousands of trajectories and is under-dosed at few-shot K". The official rows stay the
baselines (they are what the papers specify); the controls below are labelled as controls and never replace them.

## Control: the method's ingredient on an adequately dosed SFT student
For each cell (K, support seed) the start point is the Seed-SFT checkpoint the main table reports for that cell
(K20: 40 passes; K ≤ 10: the ~400k-token point; K ≥ 50: chosen by the same rule once the v2 grid lands, never by sealed data).
Exploration stays on the K seed training tasks. Subsequent collection, filtering, pair/correction construction and aggregation
follow the existing official ports. The approved AgentEvol retraining dose/adapter exception is specified below.

| Method | Ingredient tested | Start | What changes vs official |
| --- | --- | --- | --- |
| Kang | SAG decoding (FTP has no released code) | Seed-SFT LoRA | nothing else; already run for K20 s0 (115 = greedy 115) |
| AgentGym BC / AgentEvol | explore on K tasks, keep successes, retrain | Seed-SFT merged in fp32 | BC / iteration 0 replaced; each retrain starts from base on D_m with the Seed-SFT LoRA recipe and matched passes; retrains served unmerged |
| ETO | failure/success pairs on K tasks, DPO | Seed-SFT merged in fp32 | SFT replaced; fixed init reference as requested; DPO hyperparameters unchanged |
| IPR | step-level MC rewards, iterative DPO | Seed-SFT merged in fp32 | SFT replaced, including the fixed reference and SFT MC policy; subsequent recipe unchanged |
| LEAP (privileged and 0%) | teacher corrections on student rollouts | Seed-SFT merged with LEAP official bf16 policy | iteration 0 = Seed-SFT; iterations 1–3 retain fresh official LEAP adapters and merges |

Full-FT methods start from base + LoRA delta merged in fp32 (the FSDP master precision), so training starts exactly at the
fp32 base-plus-delta weights without rounding the delta into bf16 first; serving is explicitly bf16 like every full-FT baseline.
This weight equality does not claim bitwise equivalence between unmerged LoRA and merged bf16 inference. Expected mechanism: a dosed student already fits 90%+ of its
K training tasks (run 2 training fit 93.7%), so exploration finds few failures and pairs/corrections become scarce; that is a
finding about these ingredients at few-shot K, reported with the pair/correction counts per iteration.

## Order and cost
1. Official cells first (submitted 10-05: Kang and AgentGym K5/K10 × 3 seeds + K20 s1/s2; ETO/IPR/LEAP K20 s0 validating).
2. Controls at K20 s0 for all five methods (≈ 1 GPU-day on hpg), then the cells where the official row and Seed-SFT differ most.
3. Implementation: completed locally as described below. CPU validation precedes a separate rai dry run and hpg submission;
   this implementation does not constitute a GPU experiment or scientific finding.


## Implemented interface and validation

All four pipeline CLIs accept optional `--init-checkpoint /path/to/checkpoint-tokens-N`. Supply the **final Seed-SFT adapter**
selected for the main table in this same (K, support seed) cell, with its parent `provenance.json`. The implementation does
not select a dose or checkpoint using evaluation results. The four `scripts/hpg_{agentgym,eto,ipr,leap}.slurm` launchers
forward `INIT_CHECKPOINT` only when nonempty; paths with spaces remain one argument.

Before freezing a control identity, validation checks every adapter artifact against `training_state.json`, that state's
binding to the original `sft` provenance, `final=true`, the configured pinned student repo/revision and original base metadata.
The training provenance must bind the exact same verified collection manifest file hash; collection contents are also bound
by `collection_sha256` in the run identity. Equal K alone is insufficient. Partial checkpoints, changed artifacts, other
methods' adapters, other revisions and other collections are rejected before merge or training. The supplied adapter must
use the Seed-SFT text-projection LoRA r16/a32 recipe, zero dropout, AdamW 1e-5, 512 supervised tokens/update, episode packing,
constant LR, zero weight decay and gradient clip 1. Passes come from the **realized** `training_state.json.passes`, checked
against its episode exposure; recipe settings come from the provenance that state's `identity_sha256` binds.

A separate resumable `init` stage replaces the initial training stage. Its identity binds the complete adapter artifact
hash map (including the state), original training state, training provenance hash, recipe, passes and collection hash.
`disclosure.json` adds `"control": "dose_matched_init"` and `"init": {...}`. Every `models.json` display `label` gains
` (Seed-SFT init)`; stable machine `name` values and evaluation directory names remain unchanged. Original Seed-SFT exposure
is retained under `init.training_state`; importing an init contributes zero new training exposure or trusted trajectories.

## Exact method behavior

- **AgentGym BC / AgentEvol:** skip `train-0`; publish the fp32 merged init as BC / iteration 0. Keep D_0 and exploration,
  success filtering, F_m + D_m ordering, no cross-iteration deduplication, and untrained D_4 as before. Train D_1–D_3 from
  the original pinned base with fresh Seed-SFT r16/a32 adapters on one GPU using `train.sft.train_loop` and its exposure planner.
  This is the control's retraining recipe change: episode packing, eager bf16 base, fp32 adapter masters, AdamW 1e-5,
  constant LR, 512 supervised tokens/update, complete episodes, token-mean CE, clip 1, and the init's training seed and
  max length. Match the init's realized episode passes over each D_m, not its absolute token count. A fractional pass
  endpoint rounds up to the first complete episode: `ceil(init_passes * len(D_m))`; both requested and realized passes
  are logged. Retrains are **unmerged LoRA**, served and evaluated over the original base through the existing adapter
  launcher and HF/vLLM equivalence checks. Iteration 0 alone uses the full-FT fp32 init export.
- **ETO:** skip `sft`; use the fp32 init for first exploration and DPO policy initialization. As explicitly specified for
  this control, all three DPO stages use that same init as their reference. Policy checkpoints continue between iterations;
  learning rates, beta, epochs, pair construction and the no-pairs stop remain unchanged. The existing default ETO path
  retains its per-iteration reference selection.
- **IPR:** skip `train-sft`; use the fp32 init for expert MC, fixed-reference agent MC, first policy initialization and the
  fixed DPO+NLL reference throughout six iterations. Step rewards, pairs, optimizer settings and empty-pair carry-forward
  remain unchanged.
- **LEAP privileged and 0%:** skip iteration-0 training and its usual merge stage; use the init stage's Seed-SFT merge under
  `train.merge_lora.LEAP_MERGE_POLICY` (`leap_official_bf16_merge`). Retain finite/argmax/reload checks and strict-allclose
  diagnostics. Keep the original D_0 example set for later min aggregation. Privileged expert collection/extraction and
  the 0% exclusion of oracle collection remain unchanged. Iterations 1–3 use their existing fresh r128/a64 q/v LoRA,
  teacher correction protocol, min aggregation, training and bf16 merge chain. The Seed-SFT init is recognized only as
  iteration 0 of its declared variant; it cannot masquerade as a later official iterate.

Full-FT initialization loads base and LoRA in float32 on CPU, merges there, and uses `baselines.common.fsdp_train.export_checkpoint`
with float32 text export. This preserves original base tensor names, tied-weight layout, unchanged multimodal tower bytes,
shard index and base serving assets. `validate_checkpoint` accepts this tagged init. FSDP loads fp32 policy masters directly
from these weights. Reference forward/storage stays bf16 as in the existing FSDP implementation, and the serving launcher
explicitly requests bf16 for fp32 init artifacts. Later ordinary full-FT exports remain bf16.

## Default compatibility and validation scope

Without `--init-checkpoint`, no control keys (including null placeholders), label suffixes or recipe changes are introduced;
commands, stage identities' construction, checkpoint formats and model records keep their prior defaults. Source hashes
remain truthful: edits necessarily change `code_sha256`, so a pre-change run still refuses a changed-code resume under the
existing provenance contract. Historical identity hashes are not fabricated or silently grandfathered.

CPU tests use synthetic tiny Gemma-4 snapshots, real PEFT adapter training and merge/reload, and fake environment/teacher
clients. Coverage includes exact fp32 base+delta and FSDP reload, base asset/tower preservation, both LEAP merge variants,
real AgentEvol Seed-SFT retraining, fractional pass rounding, all five CLI paths skipping SFT/BC while retaining downstream
stages and references, collection/revision/artifact refusal, identity/disclosure/labels, default schema compatibility,
unmerged adapter serving/evaluation and resume. No live API, network, GPU experiment or cluster job is part of this validation.

Validation completed 2026-10-06:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q
```

Result: **896 passed, 5 skipped, 1 warning**, 786.24 seconds. Skips: unavailable ALFWorld (1), local read-only probe workspace
(2), local run2 artifacts (1), and real seed collection (1). The warning is PEFT's existing vocabulary/config warning in
`test_lora_serving_torch.py`. `bash -n` passed for all four modified SLURM launchers; `git diff --check` passed.
