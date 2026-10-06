**Baseline K ladder audit — 2026-10-06**

Both baseline preparation guards now use `REGISTERED_K = (1, 2, 5, 10, 20, 50, 100)` from [baselines/common/__init__.py](../baselines/common/__init__.py). The shared guard covers Kang, AgentGym BC/AgentEvol, IPR and both LEAP variants; ETO imports the same constant in its separate preparation path. Each retains its existing ValueError wording style with the expanded slash-separated ladder. This implements the user's 2026-10-05 widening. No optimizer, schedule, aggregation, exploration, checkpoint-selection or stopping rule was changed.

The material remaining blocker is **LEAP K=100 iteration 3**: on the three actual supports it can require **220 / 239 / 224 updates**, exceeding the unchanged `<200` precondition. **K=50 is safe on these specific collections**, with at most **105 / 115 / 111 updates per stage**. Those are bounds conditional on correction-set sizes, not observed training results. K=1 also exposes important dose and holdout edge cases described below.

Here “runs” means the current port has no additional K/cardinality blocker, given valid evidence and the existing runtime requirements. It does not certify GPU memory, serving, teacher availability, or completion of every iterative stage. “Official semantics” distinguishes already registered adaptations from newly reachable branches; all methods already use X5 and the shared model/harness adaptations. The original handoff's suggested K=5/10/20 is superseded by the explicit user instruction.

Every cell below has **runs / official-semantics effect / decision**. Symbols are expanded immediately after the table.

| Method | K=1 | K=2 | K=5 | K=10 | K=20 | K=50 | K=100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kang SFT + SAG, no FTP | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / batch cap / none |
| AgentGym BC | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / dose / none | Yes / dose / none |
| AgentGym AgentEvol | Yes / accumulation / none | Yes / accumulation / none | Yes / accumulation / none | Yes / accumulation / none | Yes / accumulation / none | Yes / accumulation / none | Yes / accumulation + cap / none |
| ETO | E / empty pairs + dose / none | E / empty pairs + dose / none | E / empty pairs + dose / none | E / empty pairs + dose / none | E / empty pairs + dose / none | E / empty pairs + dose / none | E / empty pairs + dose / none |
| IPR | Yes* / empty pairs + dose / none | Yes* / empty pairs + dose / none | Yes* / empty pairs + dose / none | Yes* / empty pairs + dose / none | Yes* / empty pairs + dose / none | Yes* / empty pairs + dose / none | Yes* / empty pairs + dose / none |
| LEAP privileged | L / H1 + warmup / A | L / H + warmup / A | L / H + warmup / A | L / H + warmup / A | L / H + warmup / A | L / H / B | Conditional / H + 200 / C |
| LEAP 0% privileged | L / H1 + warmup / A | L / H + warmup / A | L / H + warmup / A | L / H + warmup / A | L / H + warmup / A | L / H / B | Conditional / H + 200 / C |

- **dose:** the existing X5 rule changes update counts and integer warmup counts with N; no new algorithm switch or required recipe change. Its claimed “at least eight steps per epoch” does **not** hold for N<8.
- **batch cap / accumulation:** Kang reaches its original batch of 8 at N>=64; AgentEvol can reach batch 32 at D_m>=256, possible at K=100. These are existing X5 branches. Repeated successful trajectories add training exposure without increasing K or unique tasks.
- **E:** SFT runs; each of three DPO rounds requires P>0. P=0 stops the pipeline and retains the current model. At K=1 a successful rollout alone makes P=0. This is the registered empty-data adaptation, not a new minimum-task requirement.
- **Yes\*:** IPR completes its six-round orchestration even with P=0 by carrying the model forward with zero training exposure. It does not guarantee six DPO updates/stages with nonempty data.
- **L:** BC runs; later LEAP stages require every included correction set to be nonempty. An empty set stops the chain. Privileged LEAP additionally needs an available executed expert and accepted extracted state for each corrected task; 0% skips that dependency. The `<200` guard remains active.
- **H / H1:** all-data training still omits the official 85/15 trajectory holdout at every K. H1 highlights that literal official splitting at K=1 produces zero training trajectories. This is a substantive pre-existing deviation even below 200 updates.
- **A:** no new implementation decision is needed to run the registered all-data port. Retain and disclose its holdout deviation and tiny-dose schedule; do not label it literal official training. Changing either needs a separate recipe decision.
- **B:** no recipe change is needed for these K=50 supports. The old `K<=20` justification should be broadened in a later registration update to the measured, per-stage bound; a generic K=50/H40 collection is not guaranteed safe.
- **C:** before requesting an unconditional four-stage K=100 run, decide whether to retain the guard and report a blocked stage, explicitly extend the all-data/final-epoch deviation beyond 200, or restore the official train-only holdout and periodic eval/save/best-loss selection. These alternatives change different things. None was implemented here.

The numerical dose audit uses [dose, epoch_batches, warmup_count and rank_slots](../baselines/common/schedule.py) and the actual [shared training loop](../baselines/common/fsdp_train.py). For E epochs:

```text
b' = min(b_orig, max(1, N // 8))
updates_per_epoch = ceil(N / b')
total_updates = E * ceil(N / b')
```

N counts **trajectories** for Kang/AgentGym/ETO-SFT/IPR-SFT, even with turn packing; all turns of a selected trajectory are accumulated into its update. It counts current-iteration **pairs** for ETO/IPR preference training, and independent **step examples** for LEAP. Partial batches are retained and normalized by their actual size. Distributed zero-weight padding allows b'<number of ranks without inventing logical examples; there is no K>=GPU-count precondition. Empty N is rejected rather than clamped to one example.

The following entries are `b' / total optimizer steps` for the seed-only stage. They count optimizer calls, including calls at learning rate zero.

| Seed training stage | Original batch; epochs | K=1 | K=2 | K=5 | K=10 | K=20 | K=50 | K=100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kang | 8; 2 | 1 / 2 | 1 / 4 | 1 / 10 | 1 / 20 | 2 / 20 | 6 / 18 | 8 / 26 |
| AgentGym BC / AgentEvol D0 | 32; 1 | 1 / 1 | 1 / 2 | 1 / 5 | 1 / 10 | 2 / 10 | 6 / 9 | 12 / 9 |
| ETO SFT | 64; 3 | 1 / 3 | 1 / 6 | 1 / 15 | 1 / 30 | 2 / 30 | 6 / 27 | 12 / 27 |
| IPR SFT | 48; 3 | 1 / 3 | 1 / 6 | 1 / 15 | 1 / 30 | 2 / 30 | 6 / 27 | 12 / 27 |

**Kang.** N=1 yields two ordinary updates, no warmup, with linear LR multipliers 1 and 1/2. There is no task sampling, holdout or periodic-step checkpoint branch. The released SFT path's save/eval configuration is not selected by K; the port publishes the final adapter. SAG's eight samples are per-action candidate generations, independent of K, so K=1 does not require eight seed tasks. Its all-invalid and tied-vote fallbacks are unchanged; FTP remains excluded. Sources: [port pipeline](../baselines/kang/pipeline.py), [registration](../baselines/kang/DEVIATIONS.md), and pinned `third_party/agent-distillation/exps_research/finetune_sft.py:174-219,258`.

**AgentGym BC and AgentEvol.** Integer warmup is `floor(.1 * total_updates)`: seed-stage counts across the ladder are **0,0,0,1,1,0,0**. In particular K=1 BC makes one nonzero update at full initial LR; it does not silently perform zero training. AgentEvol trains D0..D3, each freshly from base for one epoch; D4 is generated but untrained. `D_(m+1)=D_m+F_m`, with 0<=F_m<=K, so K<=D_m<=(m+1)K. At K=1 each trained round has 1..4 trajectories and 1..4 updates, warmup zero. Empty F_m preserves D_m; it cannot make seed-backed training empty. For K=100, D3 can have 400 trajectories, b'=32 and 13 updates. Warmup can switch off again as b' increases; total steps are not monotonic in N. This is the existing floor rule, not a reason to impose a minimum dose. The literal upstream multi-GPU dataloader/accumulation arithmetic can yield zero planned steps at small K; the registered exact-partial-batch X5 implementation already replaces it. The upstream checkpoint/evaluation frequencies are epoch-based, with one epoch here, rather than a newly reachable 200-step branch. Sources: [pipeline](../baselines/agentgym/pipeline.py), [accumulation filter](../baselines/agentgym/data.py), [registration](../baselines/agentgym/DEVIATIONS.md), pinned `bc_trainer.py:285-321,436-505` and `train_behavioral_clone_alfworld.sh:3,15-29` under `third_party/AgentGym/agentenv/`.

**ETO.** Initial SFT warmup is `ceil(.03*U)=1` at every ladder K in the table. At K=1 its three SFT LR multipliers are 0,1,1/2. Preference training uses P from the current round, original batch 32, three epochs and `ceil(.1*U)` warmup. At K=1, P is 0 or 1: one usable failed rollout yields three DPO calls with multipliers 0,1,1; a successful rollout is a tie and produces no pair. More generally 0<=P<=K; there is no requirement P>=2 or one pair per task. Unknown execution/replay failure stops evidence preparation rather than becoming a negative example. There is no pair accumulation, hidden holdout, save-at-200 branch, or minimum-update assertion. The official script disables intermediate save/eval for both SFT and DPO, so crossing an arbitrary update count would not enable best-loss selection. Sources: [pair construction](../baselines/eto/pairs.py), [pipeline and empty-P stop](../baselines/eto/pipeline.py), [registration](../baselines/eto/DEVIATIONS.md), pinned `third_party/ETO/run_eto.sh:20-42,154-180`.

**IPR.** Seed SFT has the same counts and warmup behavior as ETO. Preference training uses original batch 48, three epochs, `ceil(.1*U)` warmup, fixed SFT reference/MC policy, current-iterate initialization, and DPO plus chosen summed NLL. P=1 is valid and yields three calls; P=0 carries forward through the six registered iterations. Exploration branches at every seed action j=0..T-1. Five MC continuations are required **per state**, not five different tasks. A single nonempty seed is sufficient, including terminal prefixes that are reset/replayed with no subsequent action. At most one preference is emitted per branch, so P<=sum(T_seed); pairs can outnumber K. With these supports the K=100 pair-stage upper bound is 120 updates (largest seed-step count 1908, batch cap 48, three epochs), and there is no 200-step restriction even for larger hypothetical pools. All five known MC outcomes and replay integrity are required; incomplete samples cannot be averaged as a smaller MC set. Sources: [rollout planning](../baselines/ipr/rollouts.py), [pair rules](../baselines/ipr/pairs.py), [six-round pipeline](../baselines/ipr/pipeline.py), [registration](../baselines/ipr/DEVIATIONS.md), pinned `third_party/IPR/run_pipeline.sh:39-61,263-291`.

Shared exploration authorizes a **nonempty** list of unique TRAIN task IDs; it does not require two tasks, all six task types, or as many tasks as workers. AgentGym and ETO submit one job per seed task; a process pool may have idle workers. IPR submits jobs per prefix/sample. The existing seed selector permits zero quotas for some types at K<6. The validated collections enforce distinct seed groups, making K=unique instances=task groups **for these collections only**; K still counts trusted executed trajectories by definition. None of these methods acquires new trusted seeds by retrying, rephrasing, accumulating successes or sampling MC continuations. See [training_tasks/run_exploration](../baselines/common/explore.py), [seed selection](../src/sed/seed_selection.py), and [evidence validation](../src/sed/seed_evidence.py).

For LEAP, the count unit and min aggregation are decisive. [step_rows](../baselines/leap/data.py) makes one example per seed/correction turn. If S0 is the sum of seed trajectory lengths and C_j is the number of usable correction turns from failed rollouts in round j, the actual training-stage size is:

```text
N_0 = S0
N_i = (i + 1) * min(S0, C_1, ..., C_i),  i=1,2,3
0 <= C_j <= 40*K
b'_i = min(32, max(1, N_i // 8))
U_i = ceil(N_i / b'_i)                    # one epoch, reset each stage
```

Thus correction rows are capped by the smallest set, including the seed set. The DEVIATIONS phrase “ALL K and all usable correction trajectories” describes source eligibility, **not** exhaustive exposure after min subsampling. Summing S0+C1+C2+C3 without subsampling overestimates training size. Conversely, computing N=K underestimates it. Both variants use the same arithmetic, fresh LoRA and fresh optimizer/scheduler at every iteration.

I read `data/verified/seeds_k100_s{0,1,2}` and their k1/k2/k5/k10/k20/k50 views, counted `len(episode['steps'])` from the manifest's verified TRAIN episodes, and checked each episode JSON SHA256 against its receipt. The episode identities are nested. The old `seeds_k20_s*/subsets/k5,k10` and K20 roots have identical episode identities/hashes and step totals to the corresponding K100 views. No new data were collected or written. The measured mean lengths at K=50 are 16.80/18.36/17.66 and at K=100 are 17.55/19.08/17.85; individual lengths span 3..40, so “10–25 turns” is only a rough average description.

Triples below are support seeds **s0 / s1 / s2**. The final column is a conservative maximum over any nonempty stage 0..3, not an observed correction/training count.

| K | Verified trajectories | Unique instances | Task groups | S0 seed step examples | Actual BC updates U0 | Maximum updates in any one stage |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 35 / 9 / 19 | 9 / 9 / 10 | 15 / 15 / 15 |
| 2 | 2 / 2 / 2 | 2 / 2 / 2 | 2 / 2 / 2 | 46 / 29 / 28 | 10 / 10 / 10 | 15 / 15 / 15 |
| 5 | 5 / 5 / 5 | 5 / 5 / 5 | 5 / 5 / 5 | 79 / 84 / 66 | 9 / 9 / 9 | 15 / 15 / 15 |
| 10 | 10 / 10 / 10 | 10 / 10 / 10 | 10 / 10 / 10 | 200 / 190 / 161 | 8 / 9 / 9 | 25 / 24 / 21 |
| 20 | 20 / 20 / 20 | 20 / 20 / 20 | 20 / 20 / 20 | 382 / 374 / 319 | 12 / 12 / 10 | 48 / 47 / 40 |
| 50 | 50 / 50 / 50 | 50 / 50 / 50 | 50 / 50 / 50 | 840 / 918 / 883 | 27 / 29 / 28 | 105 / 115 / 111 |
| 100 | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 100 / 100 | 1755 / 1908 / 1785 | 55 / 60 / 56 | 220 / 239 / 224 |

For small N, X5 steps are nonmonotonic: N=15 gives 15 updates, N=16 gives 8. The bounds above therefore evaluate `max(dose((i+1)*m,32,epochs=1)['total_steps'] for m in range(1,S0+1))` for i=1..3, plus the exact BC count. Merely evaluating dose at the largest N would give incorrect small-K bounds. Some integer minima in this envelope may not be realizable from a particular collection's full correction trajectories; the bound remains conservative.

The unchanged [LEAP trainer](../baselines/leap/train.py), both `main` and `train_loop`, refuses **U>=200**, including exactly 200. In this region b'=32, giving the exact boundary **N>=6369**. At iteration 3, N is a multiple of four, so the first blocked size is **6372**, with `min(S0,C1,C2,C3)>=1593`. This concerns each stage's optimizer counter; cumulative steps across the four fresh optimizers are not used for this condition.

| Support | K=50 upper U0/U1/U2/U3 | K=100 upper U0/U1/U2/U3 | Earliest possible blocked stage on these K=100 seeds |
| --- | --- | --- | --- |
| s0 | 27 / 53 / 79 / 105 | 55 / 110 / 165 / 220 | iteration 3, if every C_j>=1593 |
| s1 | 29 / 58 / 87 / 115 | 60 / 120 / 179 / 239 | iteration 3, if every C_j>=1593 |
| s2 | 28 / 56 / 83 / 111 | 56 / 112 / 168 / 224 | iteration 3, if every C_j>=1593 |

At K=100, iteration 3 passes this particular guard if any C_j<=1592; a zero set instead takes the empty-aggregate stop. Reaching 1593 correction turns requires at least 40 accepted failed trajectories in **each** correction round under H40, but 40 failures are not sufficient unless their total lengths reach that threshold. Privileged availability and correction acceptance can lower C_j; the 0% ablation need not have the same counts. Actual C_j for these proposed K50/K100 runs was not supplied or generated, so exceeding 200 is possible at K100, not certain.

The old horizon-only argument for K<=20 is sound: N_i<=4*40*K<=3200, at most 100 updates. Applying the same horizon-only argument to K50 would allow 8000 examples/250 updates; its safety here depends on the measured S0 bound. K100's rough 10–25-turn range allows S0=1000..2500 and a full four-set aggregate of 4000..10000, or 125..313 updates. The exact measured values above replace that rough estimate.

LEAP keeps the official **10 warmup steps**. Its actual K=1 BC sets have N=35/9/19, b'=4/1/2 and U=9/9/10. All optimizer calls occur at scheduler indices below 10: update 1 has LR=0, and the largest LR actually used is 0.8/0.8/0.9 of 3e-5. These BC runs never reach peak LR or cosine decay. K=2/5/10 BC likewise finishes within warmup; K=20 s2 does too. Smaller later correction aggregates can re-enter this regime at any K. A hypothetical one-step LEAP example set would yield a single LR-zero call, although none of the current K=1 seeds is that short. There is no warmup<=total_steps assertion. Changing warmup to force “meaningful updates” would change the recipe and was not done.

LEAP has no minimum number of distinct tasks for correction or teacher batching. At K=1 one accepted failed-trajectory correction can train; a student success, exhausted correction, or (privileged only) missing expert/extracted state can make C_j=0 and stop subsequent min aggregation. Teacher caps scale linearly: privileged extraction <=K logical calls/3K attempt slots; each of three correction rounds <=K logical calls/3K attempts. The 0% variant omits extraction. These counts are not minimum required successes. Unknown execution remains an evidence failure. Sources: [pipeline](../baselines/leap/pipeline.py), [teacher budget](../baselines/leap/teacher.py), [data aggregation](../baselines/leap/data.py), and [unchanged registration](../baselines/leap/DEVIATIONS.md).

The literal official LEAP behavior differs in two separate ways. The pinned [data builder](../third_party/leap_llm/scripts/dataproc/create_training_data.py), lines 165–167, shuffles **trajectory log files**, trains on `floor(.85*M)` files and holds out the rest before creating step examples. The [configuration](../third_party/leap_llm/configs/training_alfworld.yaml) sets 0.85 for both BC and correction data. This is a TRAIN-derived holdout named `test.json` upstream, not permission to use this repository's sealed test or dev140 for selection. Official train/eval sets are independently min-subsampled across iteration directories in [sft.py](../third_party/leap_llm/scripts/train/sft.py), lines 59–76.

| Seed K (or correction-file count M) | 1 | 2 | 5 | 10 | 20 | 50 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Official training trajectory files | 0 | 1 | 4 | 8 | 17 | 42 | 85 |
| Official held-out trajectory files | 1 | 1 | 1 | 2 | 3 | 8 | 15 |

Therefore literal official BC cannot train at K=1. Even at larger K, a correction directory with only one usable trajectory has zero training examples after the official split, whereas the registered all-data port can use it. Omitting holdout affects training exposure at **every K**, irrespective of whether eval/save is reached. Reinstating it would require freezing/disclosing the actual file/group split policy: separate upstream splits per iteration can put the same task on different sides across iterations.

The official [training launcher](../third_party/leap_llm/bash/train-sft-alfworld-iterx.sh), lines 40–67, requests one epoch, global batch 32, warmup 10, eval/save every 200 optimizer steps, `load_best_model_at_end=True`, selection by eval loss, and save-total-limit 3. When the **actual training set after holdout and min aggregation** reaches step 200, it evaluates and saves; later multiples do likewise, and the best evaluated checkpoint is loaded at the end. Between 200 and 399 updates this means step 200 is the only periodic candidate, potentially replacing later final-epoch weights. Below 200 there is no periodic candidate; the pinned `sft.py` ends after `trainer.train()` and does not explicitly perform an unconditional final save. Our final-epoch publication is already a deliberate adaptation. [Official source at the audited pin](https://github.com/sanjibanc/leap_llm/blob/a3376d89a2eeee88f03555a4174a0bdf7faf8be6/bash/train-sft-alfworld-iterx.sh#L40-L67).

**Do not assume that a port count >=200 means the holdout recipe would also reach 200.** Splitting changes the count. As an illustration only, replacing S0 by .85*S0 gives four-set step estimates of about **187 / 203 / 190** for K100 s0/s1/s2, assuming corrections are not the minimum. The exact official train seed totals depend on which 85 logs are retained: summing the shortest versus longest 85 gives bounds **1194..1696 / 1365..1820 / 1255..1710**. These ranges straddle the iteration-3 cutoff of 1593 for every support. Correction holdouts introduce further dependence. No split was invented or selected in this audit. A restored official recipe must compute actual post-split counts and apply its eval/save schedule accordingly; simply deleting the current guard would retain all-data/final-only training and would not restore official semantics.

For provenance of the count-sensitive conclusion, these are raw SHA256 hashes of the inspected collection manifest files (not canonical JSON digests):

| Manifest under `data/verified/` | SHA256 |
| --- | --- |
| `seeds_k100_s0/subsets/k50/manifest.json` | `43fb498bfbdf495e5ebc06cffd88a334f6dc00771a851183a63bc110d15b422e` |
| `seeds_k100_s1/subsets/k50/manifest.json` | `9575010967746964a262518408aa96b63f143c3d54bdbbfaea511dfe95499ab9` |
| `seeds_k100_s2/subsets/k50/manifest.json` | `a28b0f05b2948a4693443ea44335ad8f20bcb64b012378862b7df000c1bae9cf` |
| `seeds_k100_s0/manifest.json` | `c00dfdea0c38418dc7a1ad74391699ad32b9d1408a13c4fffc835204fa25b541` |
| `seeds_k100_s1/manifest.json` | `e2167c6ca8b790c2e58615407f679a0d6eb9b3a926745610fa27edba67232609` |
| `seeds_k100_s2/manifest.json` | `8fc4a46cc8bffc56c33d366ef07b233c16554d3f6072c6a40d9544025273db71` |

All five local official repository HEADs match [PINS.md](../baselines/PINS.md): Kang `8884b80`, AgentGym `3ef9235`, ETO `a2fc5da`, IPR `93335a3`, LEAP `a3376d8`. The audit used the pinned local source contents; attempted remote LEAP raw-source retrieval returned cache misses. No upstream script was executed. Other K-independent requirements remain: valid nonempty supervision, complete verified TRAIN evidence, exact reset/replay, 32768-token no-truncation limits, configured models, checkpoint identity, and the existing hardware/merge/serving gates. The [baseline cell launcher](../scripts/hpg_baseline_cells.slurm) already routes every ladder K correctly; its supported methods remain Kang and AgentGym, while ETO/IPR/LEAP use their individual launchers with `VERIFIED`.

Guard regression tests exercise both real preparation functions with fake evidence loaders: all seven ladder values pass through preparation, and K=3/200 raise the corresponding ValueError before episode reads/output creation. Existing tests cover X5 N=1, partial batches/rank padding, single-task exploration, empty iterative data, scheduler arithmetic, and LEAP aggregation. Full validation uses the requested interpreter and offline settings:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -B -m pytest -q
```

Validation result: **884 passed, 1 skipped, 1 warning in 707.36 seconds (11m47s), exit 0**. The skip is `tests/test_alfworld_integration.py:15`: optional `alfworld` is not installed in this venv. The warning is the existing PEFT config/vocabulary warning in `test_hf_logprob_alignment_and_real_peft_conversion`. The new guard tests also passed separately: **18 passed in 2.07 seconds**, and are included in the full-suite total. `git diff --check` passed. Tests use the repository's autouse network/OpenAI transport prohibition and fake clients/environments. No live experiment, production training, teacher/student API request, GPU job, commit or push was performed. `analysis/`, `teachers/`, `train/` and `data/` were not edited; their contents were only read as needed for source/count inspection and existing tests.

Existing prepared run identities bind code hashes, so this guard repair can cause preparation to reject an old output root containing `identity.json`. Preserve the old attempt evidence and use a fresh output root when identities differ; do not weaken or rewrite the provenance check. A cell that failed at the old K guard before an identity was written has no identity conflict from that failure alone.
