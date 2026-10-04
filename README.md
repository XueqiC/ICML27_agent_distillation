# Sufficient Experience for Black-Box Agent Distillation

This is a new research codebase for Paper 1. We study what experience lets a
student develop an independently successful agent policy, beginning with K
trusted, executed seed trajectories. **K counts trajectories, not tasks.** New
tasks, sampled teacher outputs, and real interactions are allowed and must be
accounted for. No scientific results or trained models are claimed here.

Stage 1 provides a CPU ALFWorld harness, deterministic data manifest and frozen
splits, durable event/blob storage, a black-box teacher adapter, a single-episode
runner, and replay verification. Teacher inputs and training examples use an
explicit visible-field whitelist. Hidden success labels, facts, expert plans,
task filenames, and audit records never enter prompts. The teacher adapter uses
sampled text and accounting metadata through the [official Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses);
it does not request logprobs or hidden reasoning. No API call was made during
Stage 1 implementation or validation.

The student is `google/gemma-4-12B-it` at revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. The teacher is `gpt-6-luna`.
The configuration records the user-reported canary at 2026-10-04 15:00 CDT,
returned ID `gpt-6-luna`, with no dated snapshot. Each future response records
its returned model ID and time; an alias can drift. The student checkpoint has
not been downloaded or trained here. Only post-training with permitted model
origins is in scope; see [AGENTS.md](AGENTS.md) for the hard rules and contract.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/base.yaml` | Fixed model identities, protocol, environment-variable path overrides |
| `src/sed/` | Configuration, durable storage, events, provenance, manifest, runner |
| `harness/alfworld/` | Official ALFWorld/TextWorld adapter with verified prefix replay |
| `teachers/openai_teacher.py` | Sampled-action teacher, cache, retry and usage ledger |
| `data/manifest/` | Builder, content hashes, immutable split freeze |
| `data/raw/`, `data/verified/` | Separate unverified and verified experience destinations |
| `train/dataset.py` | Allowlisted action examples with role-level supervision and label origin |
| `eval/` | Replay CLI; no online training or writes into the training pool |
| `analysis/`, `reports/` | Future mechanism analysis and recorded validation/provenance |
| `tests/` | Fake-based unit tests and one optional real-environment integration test |

## Run locally

Python 3.11 is the tested runtime. Direct dependency pins and limitations are
in [requirements-notes.md](requirements-notes.md). From the checkout, set
`PYTHONPATH=src:.`; installation is optional. Paths resolve from the working
directory. Supply your own installed dataset root; code has no cluster paths.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=src:.
export ALFWORLD_DATA_ROOT=/path/to/alfworld/data/json_2.1.1
python data/manifest/build.py
python -m pytest -q
```

Choose an eligible ID from `seed_pool_candidates` in the split freeze. Use a
Python environment containing ALFWorld and TextWorld for real episodes:

```bash
python harness/run_episode.py --task-id 'train/TASK/TRIAL' \
  --policy scripted --output reports/runs/my-scripted-episode
python eval/replay_episode.py \
  --episode reports/runs/my-scripted-episode/episode.json \
  --output reports/runs/my-replay
```

The default scripted policy repeats `look` until termination or the 40-step
limit; this tests the complete episode lifecycle, not task-solving ability.
`--actions actions.json` accepts a JSON string list, cycling if exhausted.
`--purpose diagnosis` allows dev runs; default `generation` requires train.
Sealed tasks allow frozen-policy evaluation only, never teacher generation,
branching, or diagnostic replay. The runner never promotes raw samples into
verified training data. Trusted seed selection remains empty until executed
trajectories and required intermediate facts are actually verified.

`--policy teacher` is available for later authorized API runs. It reads
`OPENAI_API_KEY`, falling back to `~/hq/secrets/llm_apis.env` without sourcing
shell commands. Do not enable it in tests. `SED_SECRETS_FILE` overrides the
credential file. The client is lazy, SDK retries are disabled, and the wrapper
attempts at most three calls by default. It stores only allowlisted text and
metadata, never raw SDK exceptions or secrets. Cache keys cover the model ID,
template, task, history, visible intervention context, and sampling settings.
Cache hits retain original response provenance and record zero new billed
tokens. Usage unknown after failed/interrupted requests remains unknown.

## Evidence and split rules

`alfworld_manifest.json` hashes every file in each discovered trial directory
under train, valid_seen, and valid_unseen. Playable eligibility follows the
installed ALFWorld task-type, movable/sliced, and solvability rules. Excluded
trials retain reasons. `valid_train` is outside this Stage 1 protocol. The
default freeze assigns eligible train games to the expandable training pool,
valid_seen to dev, and valid_unseen to sealed test. Train-derived dev is also
supported: set `splits.dev_source: train` and an explicit `dev_count` measured
in task groups, keeping sibling trials together. A changed manifest or split
requires a new explicit destination; existing frozen files are not overwritten.

Every episode writes `provenance.json`, `events.jsonl`, content-addressed
`blobs/`, and `episode.json`. A reset event costs zero environment steps; each
issued step costs one. Failed step calls are flagged as errors, and their
state outcome is unknown. Teacher usage includes reasoning-token counts only,
with request attempts in a separate append-only ledger. A durable `started`
entry without a completion is an interrupted attempt with unknown cost.

JSONL appends preserve the exact old byte prefix, use a writer lock, and
publish through fsync plus atomic rename. This provides crash-safe logical
append-only semantics on a local POSIX filesystem; it costs O(log size) per
append and is intended for Stage 1. Blobs are immutable SHA-256-addressed JSON
and are durable before events reference them. A damaged log is rejected,
never silently truncated. Large-scale or distributed logging needs a separate
storage design. Runtime data and run directories are git-ignored by default.

The harness uses `textworld.start` and official `AlfredDemangler`, with no
expert wrapper. Reset and step return `(visible, reward, terminated, truncated,
audit)`. `visible` has exactly `observation` and `admissible_commands`.
Snapshot restoration performs a real reset and every prefix action, checking
visible text, admissible sets, reward, terminal flags, and hidden audit hashes
at every point. Divergence invalidates the environment; it cannot create a
counterfactual branch. Replay records environment seed and game hash.

The initial baselines and later SmartAD/DART-SD comparisons remain future
stages. `train/dataset.py` specifies assistant-only role supervision; it does
not yet tokenize, apply model-specific loss masks, or claim a working trainer.
The old adapter and support-rollout code were read for interface reference;
no old implementation or experimental numbers were copied wholesale.

See [reports/stage1_validation.md](reports/stage1_validation.md) for measured
local counts, exact runtime, test results, episode artifacts, and remaining
limits.
