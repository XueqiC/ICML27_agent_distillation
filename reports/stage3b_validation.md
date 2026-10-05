# Stage 3b engineering validation — 2026-10-05

Implemented AgentGym BC/AgentEvol at `3ef9235` and Kang non-FTP SFT + SAG at
`8884b80` on Stage 3a infrastructure. This is an engineering deliverable,
not a scientific result or a claim of experimental reproduction.

No GPUs, real servers, network/OpenAI calls, job submissions, commits or pushes
were used. `third_party/`, `reports/runs/`, `data/`, and mechanism-analysis code
were not modified. Tests use fake environments, in-memory HTTP transports,
fake launcher executables, and tiny random CPU Gemma models. The existing
real tokenizer and seed artifacts are read only where Stage 3a tests require them.

## Behavior and evidence

| Component | Implemented and checked |
| --- | --- |
| Linear schedule | Same step-zero, warmup, decay and end-boundary values as Transformers. Default ratio warmup remains ceil for ETO; AgentGym supplies explicit `int(.1*steps)`, including zero; Kang supplies zero. |
| AgentGym datasets | K checked seed trajectories, allowlisted conversations only, source/parent IDs, evidence reconstruction on load; Fm prepended to all Dm; first successful task per iteration, no cross-iteration dedup. Synthetic rule compared with isolated pinned official filter AST. |
| AgentEvol pipeline | Train 0..3 from base, explore/filter after each including round 3, produce D4 without training. Empty Fm continues. Resumable stages, immutable completion hashes, retained failed attempts; evaluation separate. |
| Kang LoRA | r64/alpha128/.05, seven text projections only, frozen base/towers/head, two epochs, lr2e-4, X5 from8, linear/0 warmup; shared assistant-only SFT objective. CPU updates, save/reload and serving-schema checks. |
| SAG | Eight samples, T .4 / top_p .8 / 1024; shared parser + admissibility; stable group vote and all fallbacks; raw candidate/channel blobs and step metadata; all-sample usage. Fake harness executes exactly one winning action. |
| Greedy compatibility | Literal pre-change request body compared byte for byte. CLI omits decoding from the default student spec; default policy condition remains unchanged. Existing prompt/parser/template/caps untouched. |
| Evaluation | Existing adapter identity/equivalence gate and shared HPG launcher; `DECODING=sag` forwarded to suite. SAG requires batch-invariant server provenance; launchers bind `BATCH_INVARIANT=1`. Failed equivalence checks prevent adapter evaluation. |
| Provenance | Frozen config/data/splits/code/checkpoint hashes, installed dependencies, student identity, seed teacher evidence binding, all per-update exposures and per-iteration Dm/Fm, failed/retried stages, replay transitions and unknown costs. |

The existing full-parameter FSDP path and ETO recipe remain unchanged except for
optional accumulated SFT input and explicit integer warmup support. Kang is a
separate `--adaptation kang_lora` branch in the same training module, using the
existing loss/sampler and PEFT target helper. Adapter output includes
`adapter_config.json`, `adapter_model.safetensors`, `training_state.json`, and
parent training provenance matching the existing serving inspector.

Every deviation has official file:line, replacement and reason class in
[AgentGym DEVIATIONS](../baselines/agentgym/DEVIATIONS.md) and
[Kang DEVIATIONS](../baselines/kang/DEVIATIONS.md).
Kang's official `pyproject.toml:29` pins TRL 0.19.1; `:82` has only
`transformers>=4.0.0`. No exact historical Transformers pin exists there.
`finetune_sft.py:201-220` omits scheduler/warmup. The requested linear/0 recipe
is explicit here and checked against installed Transformers 5.14.1 defaults
(`training_args.py:767-778`) and its actual scheduler. Historical TRL source was
not installed locally and was not fetched; that verification limit is retained.

## Offline checks

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  /home/xueqi/hq/projects/tc-alignment-taskeq/.venv/bin/python -m pytest -q \
  tests/test_stage3a_torch.py tests/test_stage3b_torch.py
bash -n scripts/hpg_agentgym.slurm scripts/hpg_kang.slurm \
  scripts/hpg_eval_adapters.slurm scripts/eval_adapters.sh
git diff --check
```

Final stable-source pass: **458 passed, 5 skipped (385.87s)** in the project
venv. The torch command passed **35 tests (18.61s)** under Python 3.12.13,
torch 2.13.0, Transformers 5.14.1 and PEFT 0.20.0. The project skips are four
torch-dependent modules (torch is absent from that venv) and the optional live
ALFWorld integration test without `ALFWORLD_DATA_ROOT`. No tests were disabled.
Shell syntax, both pipeline `--help` entries and `git diff --check` pass.

An earlier full pass overlapped ongoing edits and returned 449 passed / three
resume-identity failures / five skips. All three failures correctly detected
changing repository code hashes inside mechanism resume tests. The complete
suite was rerun with hash-tracked sources held unchanged and passed as above;
no mechanism code or resume checks were changed. A focused Stage 3b + adapter
launcher pass also passed 46 tests (32.18s), before the final added accumulating
pipeline parametrization was included in the complete 458-test pass.

## Exact HPG commands (not executed)

The configured account/qos are `fsu-compsci-dept`, partition `hpg-b200`.
AgentGym requests four B200s / 32 CPUs / 480G / 12h. Kang requests one B200 /
8 CPUs / 96G / 6h. These are resource requests, not measured memory/throughput.

AgentGym K20/support-seed 0 (training seed is always 0):

```bash
source configs/hpg.env
cd "$PROJECT_ROOT"
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0"
export AGENTGYM_OUT="$PROJECT_ROOT/reports/runs/agentgym/k20-s0"
export AGENTGYM_MODE=train AGENTGYM_THROUGH=all BATCH_INVARIANT=1
sbatch --export=ALL scripts/hpg_agentgym.slurm
```

Repeat the same command/output to resume. `AGENTGYM_THROUGH=train-0` produces
BC only; boundaries also include `explore-0`, `filter-0`, through `filter-3`.
Later set `AGENTGYM_THROUGH=all` to continue the same run. `AGENTGYM_MODE=prepare`
freezes inputs without training/serving. Changing identity-bound code/config/data
requires a new output tree. Completed artifacts are checked before reuse.

For K5/support-seed 1, change the two paths before submitting:

```bash
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s1/subsets/k5"
export AGENTGYM_OUT="$PROJECT_ROOT/reports/runs/agentgym/k5-s1"
sbatch --export=ALL scripts/hpg_agentgym.slurm
```

Evaluate all completed AgentGym checkpoints independently on frozen dev140:

```bash
sbatch --gpus=1 --cpus-per-task=8 --mem=48G \
  --export=ALL,AGENTGYM_MODE=evaluate scripts/hpg_agentgym.slurm
```

This uses the shared full-checkpoint serving/evaluation path, greedy, thinking
OFF, with batch invariance enforced. Evaluation neither generates training
data nor determines which AgentEvol rounds run. `models.json` is authoritative
after retries; `datasets.json` records D0..D4 locations, and `iteration-m.json`
records Dm, Fm, the base training start and exposures. D4 is explicitly untrained.

Kang K20/support-seed 0:

```bash
export VERIFIED="$PROJECT_ROOT/data/verified/seeds_k20_s0"
export KANG_OUT="$PROJECT_ROOT/reports/runs/kang/k20-s0"
export KANG_MODE=train BATCH_INVARIANT=1
sbatch --export=ALL scripts/hpg_kang.slurm
```

The same output resumes; `KANG_MODE=prepare` freezes inputs only. For nested
K5/K10, change VERIFIED and KANG_OUT as for AgentGym. After training completes,
read the adapter path from the result and submit a **separate** SAG evaluation:

```bash
export ADAPTERS="$(CUDA_VISIBLE_DEVICES='' "$ENV_PYTHON" - "$KANG_OUT/result.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    result = json.load(handle)
print(result["name"] + "=" + result["adapter"])
PY
)"
unset MERGED_MODEL_PATH
export DECODING=sag INCLUDE_BASE=0 BATCH_INVARIANT=1
export TASKS="$PROJECT_ROOT/data/manifest/dev140.json"
export OUT="$PROJECT_ROOT/reports/runs/kang-dev140/k20-s0-sag"
sbatch --export=ALL scripts/hpg_eval_adapters.slurm
```

The existing launcher computes HF references first, starts vLLM, verifies the
adapter equivalence report, then evaluates SAG. A failed check prevents that
adapter's evaluation. For the companion SFT-only greedy evaluation, keep the
adapter and frozen tasks, set `DECODING=greedy`, and use a new OUT:

```bash
export DECODING=greedy OUT="$PROJECT_ROOT/reports/runs/kang-dev140/k20-s0-greedy"
sbatch --export=ALL scripts/hpg_eval_adapters.slurm
```

## Acceptance boundary

No real 12B training, distributed GPU collectives, B200 memory fit, long-episode
GPU throughput, r64 vLLM equivalence check or live SAG evaluation was run.
CPU correctness and artifact compatibility do not establish those results.
First hardware runs must pass the existing serving gate and retain measured
resource/provenance receipts. No Stage 3b baseline or mechanism conclusion is claimed.
