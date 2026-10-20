# STeCa port registration — 2026-10-19

Official pin: `76f6ea7131e90e4fad3603396cced9586dea2ec3`
([source](https://github.com/WangHanLinHenry/STeCa/tree/76f6ea7131e90e4fad3603396cced9586dea2ec3)).
Paper: [STeCa: Step-level Trajectory Calibration for LLM Agent Learning](https://arxiv.org/abs/2502.14276).
The frozen `docs/baselines/NEW_METHOD_SPECS_1019.md` STeCa section governs this
implementation. File references are relative to `third_party/STeCa/`.
This port reports no scientific finding or faithful end-to-end experiment. No
GPU, cluster or paid API run was performed; upstream files remain untouched.

| Official file:line / paper | Replacement/change | Reason class |
| --- | --- | --- |
| `sft/my_sft.sh:10-44`; paper §3.1 | Existing full text-parameter FSDP2 trainer, configured Gemma backbone. SFT warm-up lr3e-6, batch16→min(16,max(1,floor(N/8))), 3 epochs, cosine/.03 warmup, wd0, seed0. Released shell uses lr2e-5; the explicit 10-19 spec takes precedence. Three warm-up epochs follow IPR/released shell and are registered. | user-fixed recipe / X1/X5 |
| `sampling/alfworld/generate_response_alfworld_explore.py`; `monte_carlo_sample_alfworld.py` | Reuse the existing IPR checked expert-prefix exploration and MC implementation. Fixed SFT model for every stage, five independent reset/replay completions per step, T1/top_p1/top_k−1/max512, no discount. Shared H0 greedy exploration is OFF/max1024, 40 total turns including replay. All samples and failed/unknown attempts are retained; unknown MC is never assigned zero or replaced with immediate reward. | code reuse / X3; IPR replay and sampling registration applies |
| Paper Eq.5; `data_collection/data_cali_alfworld.py:291-327` | For zero-based branch j, compare agent value after its action with **expert value after j actions**, keyed by expert j−1. Strict decrease, delta0; ties do not qualify. Select the first such branch per task in frozen expert-step order. j0 has no preceding expert step and cannot trigger reflection; its MC is retained/accounted. Later drops are recorded but not reflected. Do not inherit released `<=`, skipped initial iterations, BERT similarity<.95 filter or final-failure-only restriction. | requested paper rule supersedes released heuristics |
| Paper §3.2 online wording vs released offline collection | IPR exhaustively executes all expert-prefix explorations and MC before detecting the first drop. Selection/calibration uses the first drop; later unused execution remains disclosed. No fictitious early stopping or restored state is claimed. | required reuse of released offline collection |
| `data_collection/data_cali_alfworld.py:336-356`; paper Appendix E.2 | Reuse the released reflection instruction, render only allowlisted H0 expert prefix plus deviated reply, and supply the demonstrated action. Teacher `gpt-6-luna` replaces GPT-4o. Teacher is black-box sampled text only, through `teachers.capped`, cap equal to detected trajectories, one call each; a persistent cell-level counter/cache/usage ledger survives stage retries. No API retries, no logprobs, no hidden reasoning request. | user-required teacher / hard accounting cap |
| `data_collection/data_cali_alfworld.py:359-382`; paper §3.2 | Keep visible reflective thought and force the verified expert action; subsequent expert observations/actions unchanged. Train the corrected step and expert suffix, mask the preceding expert history. Do not append an executed wrong action and pretend the old expert observations follow it; correction is at the original expert state. Empty/hidden-only/invalid reflection stops with raw output retained and no second call. Synthetic explanation is not a verified environment fact. | paper correction placement / real-state integrity |
| Paper §3.3 | `Dc`: first-deviation calibrated suffix paired with deviated suffix. `Ds`: expert suffix at each failed expert-prefix exploration, paired with that failed suffix, including j0; no extra released MC threshold. `De`: successful **j0 from-start** explorations paired with their experts; successful partial branches do not qualify. Repeated prefixes are exposures, not new tasks/seeds. | paper datasets / explicit mapping to exhaustive IPR branches |
| Paper Eq.2-3 | Dynamic programming DTW with cosine distance, divided by sqrt(nx²+ny²), not exponential similarity or path-length normalization. Each action includes its visible rationale, matching paper §3.1. Use the released `bert-base-uncased` CLS encoder (canonical ID `google-bert/bert-base-uncased`), eval/no-grad on CPU; local snapshot required and all bytes bound. Native 512-token encoder truncation affects distance only. The absent released nDTW implementation leaves this representation choice unspecified; use its existing BERT helper. | missing paper implementation / registered distance encoder |
| Paper's stated nDTW range | Preserve Eq.3 without clipping. With cosine distance in [0,2], its stated [0,1] range is not a mathematical guarantee; record actual distances. Rewards must remain finite and positive. | mathematical clarification; no silent formula change |
| Paper Eq.7-10; `Reinforced_train/steca_train.sh` | `rc=1+.01*d`, `rs=1+.01*d`, `re=1−.01*d`. Minimize negative **summed** assistant continuation log likelihood, with reward per trajectory; no DPO, KL, entropy bonus or reference model. Sum dataset expectations: each example scaled by N/N_kind, then global-batch mean. Empty dataset contributes zero; no oversampling. Add opt-in `steca` objective to existing trainer; one turn graph at a time and existing rank padding preserved. | faithful objective; explicit finite-dataset estimator |
| `Reinforced_train/steca_train.sh:5-46` | One reinforced epoch, lr3e-6 (SFT LR, explicitly registered), batch15→b′, constant_with_warmup/.1, wd0, seed0. Initialize from the same SFT checkpoint used for exploration/MC; bind its provenance. If all three datasets are empty, publish a zero-exposure carry-forward instead of invoking an empty trainer. | released reinforcement recipe / missing LR interpretation / empty edge case |
| Full training serialization | Existing Gemma OFF `encode_turn`/continuation encoding; assistant spans and native turn ends supervised, prefix/observations masked, no truncation at 32768. Existing fp32 text masters, bf16 compute, frozen multimodal towers, SDPA, checkpointing, AdamW/clip1 and exact trajectory batches. SFT retains existing trajectory-token-mean CE; reinforcement uses summed LL. | X3/X6/X7, existing trainer conventions |
| Evaluation | Publish warm-up and reinforced checkpoints, greedy dev140/batch-invariant seed0, no dev selection. Add an **opt-in** full-checkpoint HF/vLLM gate using the same first eight verified TRAIN turns and existing token scorers: pooled mean absolute logprob error≤.05 nats and >.5-nat outliers≤max(2,ceil(.05*tokens)). Full-model gate has no adapter-effect requirement. Bind checkpoint/teacher-free TRAIN evidence/server hashes; evaluate only on passing gate. Existing baseline evaluation defaults unchanged. | required gate for full checkpoints / registered numerical tolerances |
| Stages and support | Shared hash-bound resumable transactions; only validated `--verified` support, nested views and any future full collection accepted by the shared evidence loader. No task pool expansion, no assumption K=unique tasks, no edits under `data/verified`. Reflection ledger lives outside attempt directories so retrying cannot reset the cap. Data loader reconstructs labels/rewards from checked rollouts, raw reflection and hash-bound embedding vectors. | engineering / auditability |

Stages: `sft`, `expert-mc`, `explore`, `agent-mc`, `detect`, `reflect`, `datasets`,
`reinforced`. Use `scripts/hpg_steca.slurm` with `STECA_THROUGH`; all paths and
`NGPUS` remain fixed across jobs. Set `EMBEDDING_SNAPSHOT` to the already-local
BERT checkpoint. Run through `agent-mc` on GPUs, then through `reflect` on a CPU
partition with `--gpus=0`, then `datasets` CPU, then `reinforced` GPU. The wrapper
has no default GPU request and rejects CPU-only stages in GPU allocations; CPU
stages fail before work if required GPU stages are not complete. Tensor encoding
uses `TRAIN_PYTHON` with CUDA disabled. `--through all` cannot execute the
reflection stage while a GPU is allocated; use the stage boundaries above.
`reinforced` requires completed reflection and dataset stages. `STECA_MODE=evaluate` evaluates every
published model. Direct full-checkpoint launcher users set `FULL_EQUIVALENCE=1`
and `VERIFIED` to enable the same gate.

Offline tests cover mechanics, masks, gradients, durable caps and stage resume.
Real FSDP execution, memory fit and live serving equivalence remain untested.


## 2026-10-19 engineering fix: fail loudly on teacher transport

Transport failures (HTTP errors, timeouts, other incomplete/unknown responses, halted
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

2026-10-20: a response truncated at the step/output cap is a model output (upstream: parse error), not a transport failure.
