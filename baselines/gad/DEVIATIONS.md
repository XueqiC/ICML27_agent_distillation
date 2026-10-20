# GAD implementation choices, registered before any GPU run

Authority: `docs/baselines/NEW_METHOD_SPECS_1019.md`, GAD and Shared protocol;
SHA-256 `d6cc35f3222ee234cc9c970fe2392174365bf48e23ed51ba7e6c1ed91cee4f76`.
Upstream commits and port attribution: [NOTICE.md](NOTICE.md).
This is the user-approved ALFWorld per-turn adaptation, not a reproduction of
the paper's chat benchmark. No GPU run or scientific result is reported.

| Choice | Implementation and reason |
|---|---|
| Dataset / prompt definition (Q5) | Only the `--verified` collection's original TRAIN instances, with the common evidence validator and `registered_support` (including nested views and the full-TRAIN rule). K remains trusted executed seed trajectories. Unique instances/groups and expansion exposure are reported separately. No seed replies are silently substituted for missing on-policy teacher replies. |
| Three collection rounds | Round 0 visits histories with the configured base; round 1 with the warmup generator; round 2 with the end-of-round-1 generator. Each round greedily executes all support tasks with H0, then labels each visited history once on CPU. Round 0 supplies one CE warmup epoch; rounds 1 and 2 each supply one joint epoch. Histories are refreshed between epochs, while eight response samples are freshly drawn before **every** optimizer batch. |
| Frozen splits / failed rollouts | Shared `common.explore` executes H0 and real reset/prefix replay, and `checked_rollouts` verifies both. Successful and failed completed episodes contribute histories. Unknown/incomplete/replay-inconsistent episodes retain their evidence and halt the stage, without fabricated failure labels or replacements. There is no state branching in GAD. |
| Teacher | The configured `gpt-6-luna`, sampled text only, through `teachers.capped.from_config`; one cell-wide durable counter, hard cap `3 * 40 * unique support instances`, SDK and wrapper retries disabled. Each round/history has a distinct draw/cache key, and resume reuses it. Completed unparseable text remains a target. Failed/incomplete/unknown requests halt the cell; they consume their reservation and cannot be retried silently. Raw output, returned ID, timestamp, usage and rejection metadata stay in the ledger/cache. |
| Token limits | Group generation is capped at 1,024 tokens, overriding the launcher's 1,536 maximum. Teacher calls also set `max_output_tokens=1024`, retaining other configured sampling options. This is an API token cap, not a Gemma tokenizer count; a teacher target exceeding 1,536 Gemma tokens including turn-end stops the stage with its raw evidence retained. Complete contexts above 32,768 tokens, or prompts without room for 1,024 sampled tokens, stop without truncation. |
| Warmup ordering | Ten explicit discriminator-only updates cycle the round-0 minibatch order, with freshly sampled groups each time. Then G and D warm jointly for one complete round-0 epoch: ordinary teacher-reply CE (temperature 1) for G and BT for D. Thus D has **10 additional** warm-start updates, plus the warmup epoch's BT updates, consistent with the paper's simultaneous warmup. The spec's ten steps take precedence over the upstream global-step guard's possible indexing offset. No warmup examples are skipped to absorb those ten steps. |
| Epoch count | The local spec's 1 CE + 2 GAD epochs takes precedence over the upstream shell files' `total_epochs=2` and `total_epochs=4`. No extra epoch or dev-based selection is introduced. |
| Trainer | The shared SFT/DPO loop cannot express GRPO or a trainable discriminator. `gad.worker` ports the pinned verl update equations and uses existing FSDP2 preparation, sharding, export and DCP helpers. Full text-parameter tuning, fp32 masters/Adam state and bf16 compute; original frozen multimodal towers are retained unchanged, as in the existing full-tuning baselines. No LoRA or alternate model is silently selected. |
| Discriminator | Independent copy of the original generator's backbone, a fresh bias-free scalar head on the **last attended token** of prompt+response. Head initialized Normal(0, 0.02), seed 0 (explicit Gemma adaptation). Text backbone and head train; unused LM head is excluded where untied. No Chinese or third-party reward checkpoint is loaded. DCP retains the head, backbone and optimizer between all updates/rounds. |
| Sampling transport | Shared vLLM serving, batch invariant; `/v1/completions` takes native Gemma OFF prompt token IDs and returns eight exact token sequences with `return_token_ids=true`. Temperature 0.8, top-p 1, top-k -1; deterministic request seeds from round/batch/prompt position. Every raw group/usage is logged before validation. Missing IDs/groups stop; no text re-tokenization, action filtering, reward filtering or synthetic EOS at the cap. This API requires a vLLM build with returned token IDs. See [vLLM API](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/serving/). |
| GRPO | Unbiased within-group standard deviation + 1e-6; raw pre-update D score, no sigmoid and no environment reward. PPO clip 0.2, negative-advantage dual clip 3, log-ratio clamp [-20,20], actor low-var KL coefficient 0.001 with upstream KL clamp [-10,10]. Old-policy and reference log-probabilities use rollout temperature 0.8. One optimizer update per complete prompt minibatch, as the upstream single PPO epoch. |
| Reference | Frozen generator after CE warmup for both joint rounds, matching the upstream GAD launcher's reference initialization from the warmup actor. It is not the teacher and requests no teacher probabilities. |
| Optimizer / reduction | Both use AdamW lr 1e-6, betas (0.9,0.999), epsilon 1e-8, weight decay .01, constant LR, gradient norm clip .2, no dropout/entropy bonus. Batch is `min(256,max(1,N//8))` **prompts**, N = collected histories in that round; all eight samples remain together. Partial batches are retained. Global response-token mean implements the upstream `token-mean` objective exactly across one-sequence microbatches, rather than retaining dynamic-microbatch weighting artifacts. BT is the mean over teacher/student pairs; each teacher is paired with all eight samples. CE sees each teacher reply once per epoch. Zero-weight rank padding is compute only, never additional examples. |
| Serialized GPU phases | A training job alternates serving, discriminator update, generator update, freeing each process before the next. D scores are saved before its BT update, then used unchanged for G. This preserves algorithm order without a Ray cluster or simultaneously resident G/D/reference/vLLM. Checkpoints and Adam state are retained after each update; restart validates and resumes each completed nested transaction. Failed attempts remain separate. This is I/O-heavy and unprofiled on B200; no throughput or memory-fit claim is made. |
| Publication / evaluation | Every generator optimizer checkpoint is in cell `models.json` (more frequent than upstream `save_freq=50`). D-only states are private optimizer/reward artifacts, not policies. Separate evaluate mode uses the common HF/vLLM equivalence gate before greedy, seed-0 dev140 evaluation of every registered checkpoint. Evaluation never invokes collection/training or selects checkpoints. |
| Resources / execution | One selected stage per invocation; missing prerequisites fail rather than execute another resource class. Teacher stages check both CUDA visibility and SLURM GPU allocation. `hpg_gad.slurm` has no embedded GPU request. GPU stages receive explicit allocation at submission. Four FSDP GPUs are the wrapper default; one GPU is selectable for the prescribed future smoke, with no fit guarantee. |
| Accounting / local provenance | The common cell identity binds code, configuration, seed manifests, original snapshot content, split freeze, dependency versions, recipe, source-file hashes and training world size. Nested transactions retain raw calls, sampled token IDs, batch plans, pre-update rewards, optimizer-state hashes and completed-update token exposure. Incomplete work/unknown usage remains marked unknown. No secret environment dump or raw transport exception is saved. |

No original baseline behavior, verified data, run reports or pinned specification
is modified. All implementation tests use tiny/stub models, fake environments
and fake teacher/vLLM transports, with CUDA and network/API access disabled.


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
