# Stage 2a validation — 2026-10-04

Stage 2a enables the base-student dev40 thinking OFF/ON pilot. It does **not**
report pilot success rates, GPU performance, a trained checkpoint, or a
scientific finding. No GPU computation, network access, live OpenAI calls,
SLURM submissions, package installations, commits, or pushes were performed.
The governing handoff and repository/Stage 1 documentation were read first.

## Implementation

| Files | Result |
| --- | --- |
| `harness/alfworld_env/`; imports in runner/tests; README | Renamed `harness/alfworld` so direct scripts cannot shadow the installed package. Adapter behavior is unchanged. |
| `src/sed/react.py`, `actions.py`, `visibility.py` | One ReAct template, conversation builder and parser. The last exact `Action:` line wins; horizontal/surrounding whitespace is stripped, without fuzzy matching. Optional visible `Thought:` text stays in history. |
| `src/sed/runner.py`, `events.py` | Legal and inadmissible parsed commands both execute verbatim. Missing/empty Action produces the specified visible parser feedback, zero environment steps, and one of 40 policy turns. Events expose `action_outcome`, `inadmissible`, and a policy metadata blob reference. Replay reproduces parser-feedback turns without stepping. |
| `teachers/openai_teacher.py` | Same ReAct conversation/parser as the student, cache schema 2 with `alfworld-react-v2` prompt version. Only sampled visible output and usage metadata are requested. |
| `students/vllm_student.py`, `src/sed/thinking.py` | Greedy non-streaming chat completions, per-request `chat_template_kwargs.enable_thinking`, configurable output cap, thinking separation, visible-only history, request/response/thinking blobs, append-only call ledger. No automatic retry, model fallback, or history truncation. |
| `src/sed/dev_subset.py`, `data/manifest/build_dev40.py`, `dev40.json` | Immutable seed-0 stratified dev subset, source/freeze/list hashes, instance/group counts. Rebuilding agrees byte-for-byte. |
| `eval/run_suite.py` | Spawned process workers with independent environment/client instances and episode directories, summary JSON and per-task TSV. Refuses sealed tasks without `--sealed-final` and refuses training-data output destinations. Teacher remains train/dev only. |
| `harness/serve_student.py`, `configs/hpg.env`, `scripts/hpg_eval_student.slurm` | Local-checkpoint vLLM supervisor, source-based parser detection, ready timeout, signal shutdown, exact B200 allocation settings, cluster defaults outside Python. Supports an explicitly supplied already-merged LoRA checkpoint for later work. |
| `pyproject.toml`, `requirements*`, `src/sed/provenance.py`, tests | New packages/entry points, direct httpx pin, expanded source/runtime provenance and offline regression coverage. |

The 40-turn budget includes parser errors. `environment_steps` counts actual
attempted calls, with confirmed and unknown transitions reported separately.
Policy/service and environment errors retain unknown success, remain in the
suite denominator, and cause a nonzero suite CLI exit after aggregation. Token
usage is retained even if the following environment step fails. Unknown usage
is never silently converted into known zero; summary numeric totals are
known-only and have separate unknown-call counts. Teacher call attempts/cache
accounting remain in their episode-local usage ledgers. Evaluation produces no
training examples or online updates.

## Frozen dev40

The builder reads the existing manifest/freeze, not environment game files.
It filters to eligible, nonsealed `valid_seen` assignments in logical `dev`.
For seed 0, task types are ranked by SHA-256 of canonical JSON
`[seed, "quota", type]`; the first four receive 7 and the last two receive 6.
Games within each type are ranked by `[seed, "game", task_id]`, and the selected
list is ordered by `[seed, "order", task_id]`. Lexical IDs break any hash ties.
The builder refuses to overwrite a different frozen artifact.

| Task type | Games |
| --- | ---: |
| look_at_obj_in_light | 7 |
| pick_and_place_simple | 7 |
| pick_clean_then_place_in_recep | 6 |
| pick_cool_then_place_in_recep | 7 |
| pick_heat_then_place_in_recep | 7 |
| pick_two_obj_and_place | 6 |

40 unique instances, 40 task groups. These are evaluation instances, not K
trusted seed trajectories. No actual sealed games were opened. The existing
manifest and split freeze are unchanged.

- List hash: `ef2d21c7f83ba4c13a1124579ad1ee9de2e199a4662e884efbe412303856d3ce`
- Manifest hash: `71339051bb6d53e343ae4a27dd97bcc08680b461d75753a665dfd5a840e11a34`
- Freeze hash: `1e704d8993b9643fa32e45404ffee0d642bc8f8194f7977c416a9a6ec015c12a`

Hashes use canonical JSON without the file's final newline. The list hash
covers the ordered `task_ids` array.

## Installed vLLM thinking shape

Read-only inspection used
`/home/xueqi/hq/projects/tc-alignment-vllm/envs/vllm-serve/.venv`, confirming
**vLLM 0.27.1 and Transformers 5.14.1** from distribution metadata. No vLLM,
Transformers, or torch module was imported to inspect the runtime. Evidence
paths and file hashes are in
[stage2a_runtime_inspection.json](stage2a_runtime_inspection.json).

The exact registered parser is `gemma4`:
`vllm/reasoning/__init__.py` maps it to
`gemma4_engine_reasoning_parser.Gemma4ParserReasoningAdapter`, backed by
`vllm/parser/gemma4.py`. The parser handles `<|channel>thought\n...<channel|>`,
strips the channel label, and preserves special tokens while parsing.

**This installation emits `message.reasoning`, not `message.reasoning_content`.**
`entrypoints/openai/chat_completion/protocol.py:61` defines `ChatMessage`,
including `reasoning: str | None`; `chat_completion/serving.py` constructs
responses with `reasoning=reasoning` and `content=content`. The same protocol
module normalizes the deprecated `reasoning_content` field to `reasoning` on
*input*. The client supports both output names for compatibility.

Schematic source-derived response shape (not a live model response):

```json
{
  "choices": [{
    "message": {"role": "assistant", "reasoning": "separate thinking text", "content": "Thought: visible explanation\nAction: look"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
}
```

Without a server reasoning parser, the client also separates raw
`<|channel>thought ... <channel|>` text. It handles an opener already present in
the prompt, an unclosed thought block, null visible content, and token-limit
finishes. Thinking-only output is a parser-error turn and cannot execute an
embedded `Action:`. `skip_special_tokens=false` is passed so the raw fallback
can retain channel boundaries. Earlier thinking and audit fields are excluded
from every subsequent prompt, while visible ReAct `Thought:` text is retained.

`entrypoints/openai/engine/protocol.py:115` defines `UsageInfo` with prompt,
completion and total counts but **no reasoning-token breakdown**. The client
records exact returned `completion_tokens_details.reasoning_tokens` if a server
supplies it. Otherwise thinking-token counts are null when thinking is present
(or output is truncated), and zero for a completed response with no thinking.
It always records the separated thinking character count, latency, prompt and
completion counts, finish reason, model ID, timestamp, and request thinking
setting. Completion counts include thinking; no tokenizer or character-based
estimate is presented as an exact reasoning-token count.

## Validation

Final project-venv command, without dataset override:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .venv/bin/python -m pytest -q
```

**91 passed, 1 skipped in 5.68 s.** The optional real-data test skipped because
`ALFWORLD_DATA_ROOT` was unset.

With installed data:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 \
.venv/bin/python -m pytest -q
```

**92 passed in 12.75 s**, including the real ALFWorld reset, three executed
steps, and reset/prefix replay. Both commands exited 0. Tests block DNS,
TCP/UDP and live OpenAI Responses calls. Spawned fake-suite workers independently
install socket guards. The fake chat server runs through `httpx.MockTransport`:
full HTTP request/response bodies are exercised without opening a network socket.

Coverage includes all Stage 1 regression areas plus last-Action parsing,
whitespace/missing/empty lines, exact admissibility, parser-feedback replay,
40 parser failures with no environment step, verbatim inadmissible execution,
structured/legacy/raw thinking shapes, thinking-only truncation, prompt field
whitelists, visible conversation history, HTTP student episodes in both modes,
real success/timeout semantics on a fake backend, sanitized HTTP failures,
usage after environment failure, dev40 determinism/tamper checks, sealed refusal,
two-process suite aggregation, unknown-success denominators, server arguments,
and readiness failure/timeout. Old assertions tied to the superseded single-
command/fatal-parser behavior were updated; the remaining Stage 1 coverage passes.

Also passed: direct-file `--help` for the episode, suite and server entry points;
`bash -n scripts/hpg_eval_student.slurm configs/hpg.env`; `git diff --check`.
The renamed adapter is otherwise byte-for-byte unchanged.

## HPG pilot commands

From the checkout on HPG, with the local snapshot and both venvs already present:

```bash
cd /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
source configs/hpg.env

sbatch --export="ALL,ENABLE_THINKING=off,OUT=$PROJECT_ROOT/reports/runs/stage2a-base-dev40-off" scripts/hpg_eval_student.slurm
sbatch --export="ALL,ENABLE_THINKING=on,OUT=$PROJECT_ROOT/reports/runs/stage2a-base-dev40-on" scripts/hpg_eval_student.slurm
```

Defaults in `configs/hpg.env`:

```text
PROJECT_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation
ALFWORLD_DATA_ROOT=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
HF_HOME=/blue/fsu-compsci-dept/xc25.fsu/hq/tools/hf-cache
VLLM_PYTHON=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment-vllm/envs/vllm-serve/.venv/bin/python
ENV_PYTHON=$PROJECT_ROOT/.venv/bin/python
MODEL_PATH=$HF_HOME/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
TASKS=$PROJECT_ROOT/data/manifest/dev40.json
WORKERS=4
MAX_TOKENS=2048
READY_TIMEOUT=900
```

`ENV_PYTHON` must contain the ALFWorld/client dependencies. These defaults are
sourceable and overridable; no HPG paths are hard-coded in Python. Both modes
use `google/gemma-4-12B-it` at the specified revision, environment seed 0, the
same frozen list and 40-turn limit, and temperature 0. `MAX_TOKENS` includes
thinking and visible ReAct output. Keep it identical between the two pilot jobs.

The job requests account/qos `fsu-compsci-dept`, partition `hpg-b200`, one GPU,
8 CPUs, 48 GB RAM, 3 hours. Serving uses `VLLM_ATTENTION_BACKEND=FLASH_ATTN`,
`VLLM_USE_FLASHINFER_SAMPLER=0`, max model length 32768, memory utilization 0.85,
and the installed `gemma4` reasoning parser. The loopback port is
`20000 + SLURM_JOB_ID % 20000`; an occupied port is refused. Readiness checks
`/v1/models` for the expected served model. Signal handlers/SLURM traps stop the
server and evaluation process group. Models are local-only/offline.

Each condition writes a fresh `$OUT` with server logs/provenance and
`eval/summary.json`, `eval/per_task.tsv`, suite provenance, the frozen list,
and per-episode events/blobs/call ledgers. Pair the TSVs by `task_id`; retain
parser, inadmissible, length-finish, service-error and unknown-token information
when choosing the future fixed thinking mode.

## Remaining runtime limits

No real Gemma inference or B200 job was run. Thus GPU memory fit, server startup,
live return text, throughput, context exhaustion frequency and OFF/ON success
rates remain unmeasured. Source inspection establishes the installed API schema,
not a measured model response. The local checkpoint path and metadata/template
hashes are recorded, but the launcher does not certify local weight bytes against
Hugging Face; the default path names the pinned snapshot. Merged-LoRA serving is
an explicit future option, not a merge/save/reload validation or a trained baseline.
