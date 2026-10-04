# Stage 1 validation — 2026-10-04

Stage 1 is implemented in this new repository. No GPU computation, OpenAI
API calls, model downloads, training, commits, or pushes were performed.
Tests block DNS, TCP/UDP connections, and the live Responses transport.
The official OpenAI documentation was read outside tests to check the API
shape. This is an infrastructure validation, not a distillation experiment.

## Deliverables

- Project layout, package metadata, direct dependency pins, README, and hard
  research rules in `AGENTS.md`; the governing handoff's sections 6, 7, 10,
  11, 12, and 14 were read first.
- `configs/base.yaml` with the specified Gemma revision and user-reported
  GPT-6-luna canary. No teacher canary was repeated here.
- Validated section-12 event rows, immutable SHA-256 JSON blobs, serialized
  writers, fsync, and atomic publication preserving existing JSONL bytes.
- ALFWorld via official `textworld.start` plus `AlfredDemangler`, without an
  expert wrapper. Only observation text and admissible commands are visible.
  Audit facts and success flags are stored separately.
- Real reset and action-prefix replay, checking every intermediate visible
  observation, admissible set, reward, terminal flags, and audit hash.
- Deterministic file-hash manifest, immutable split freeze, eligibility and
  exclusion accounting, train-only seed candidates, and task-group dev splits.
- Lazy black-box teacher with prompt/context-aware disk cache, bounded
  explicitly logged attempts, usage including reasoning-token counts, and
  sanitized failures. It never requests logprobs or hidden reasoning.
- Full-episode runner, replay CLI, role-level allowlisted training examples,
  and provenance for manifest builds, episodes, and replay runs.

## Installed runtime

ALFWorld and TextWorld are installed in:

```text
/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/.venv/bin/python
Python 3.11.7
ALFWorld 0.4.2
TextWorld 1.7.0
```

Dataset root inspected:

```text
/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1
```

These paths are discovered local runtime evidence, not application defaults.
Set `ALFWORLD_DATA_ROOT` or `--data-root` to use another installation.
The old venv does not contain pytest/PyYAML/OpenAI. For validation only, the
existing base/user site-packages were appended after its own packages. No
environment was modified and no dependencies were installed. The complete
venv inventory, exact package origins and supplemental paths are in
[stage1_environment.json](stage1_environment.json).

## Manifest counts

Counts were computed from the installed files, not taken from another project.
One eligible game is one playable trial directory; sibling trials are not
treated as independent task groups.

| Native split | Discovered trials | Playable games | Playable task groups | Frozen use |
| --- | ---: | ---: | ---: | --- |
| train | 6,374 | 3,553 | 1,465 | Expandable train / seed candidates |
| valid_seen | 251 | 140 | 137 | Dev |
| valid_unseen | 255 | 134 | 47 | Sealed test |

There are 2,821 / 111 / 121 missing-game trials in those splits respectively.
Unsupported task types and movable/sliced exclusions are also reported;
exclusion reasons can overlap. `valid_train` is not part of the requested
Stage 1 protocol. No seed trajectories have been selected or certified;
configured K=10 does not refer to any of the counts above.

Artifacts:

- [alfworld_manifest.json](../data/manifest/alfworld_manifest.json)
- [alfworld_split_freeze.json](../data/manifest/alfworld_split_freeze.json)
- Canonical manifest SHA-256:
  `71339051bb6d53e343ae4a27dd97bcc08680b461d75753a665dfd5a840e11a34`
- Canonical freeze SHA-256:
  `1e704d8993b9643fa32e45404ffee0d642bc8f8194f7977c416a9a6ec015c12a`

Canonical digests exclude the file's final newline. Run provenance also
records byte-level input-file hashes where applicable.

## Test results

Default interpreter (`/usr/local/anaconda3/bin/python`):
**53 passed, 1 skipped in 2.10s**. The only skip is the optional real
integration test because ALFWorld is not importable in this interpreter.

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:. \
  python -m pytest -q
```

ALFWorld interpreter: **54 passed in 10.23s**, including a real train reset,
three actual `look` steps, and verified prefix replay. Exact invocation:

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
ALFWORLD_DATA_ROOT=/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/data/json_2.1.1 \
PYTHONPATH=src:. \
/home/xueqi/hq/projects/tc-alignment-vllm/envs/alfworld/.venv/bin/python -c \
'import sys; sys.path.extend(["/usr/local/anaconda3/lib/python3.11/site-packages", "/home/xueqi/.local/lib/python3.11/site-packages"]); import pytest; raise SystemExit(pytest.main(["-q"]))'
```

Coverage includes schema/type checks, duplicate event rejection, missing or
corrupt blobs, concurrent writers, interrupted publication, damaged-tail
refusal, oracle stripping, prompt/training whitelists, replay divergence,
game-content changes, terminal/truncation semantics, native-split/path guards,
manifest determinism and freeze tampering, cache key isolation, cache
corruption, failed/retried/rejected generations, unknown state after failed
environment calls, and secret non-leakage.

An offline wheel build completed successfully and included `sed`, `harness`,
`teachers`, and `train`, with all three console entry points. The preexisting
base setuptools/conda integration emitted permission warnings trying to access
its read-only conda history; the build itself exited successfully. No package
installation was performed.

## Full real episode and replay

The first lexically sorted eligible train trial was run with the fixed
scripted `look` policy for the configured 40-step limit. The episode has 41
events (reset plus 40 steps), zero teacher tokens, `terminated=false`,
`truncated=true`, and `success=false`. This is a complete timeout episode,
not a successful demonstration or a trusted seed.

- [Episode and visible step sequence](runs/stage1-final/episode/episode.json)
- [Event log](runs/stage1-final/episode/events.jsonl)
- [Episode provenance](runs/stage1-final/episode/provenance.json)
- [Replay verification](runs/stage1-final/replay/result.json)
- [Replay provenance](runs/stage1-final/replay/provenance.json)

Replay performed all 40 actions in a newly reset real environment and matched
every recorded observation, admissible set, reward, terminal flag, and audit
hash. These run artifacts are retained locally under git-ignored
`reports/runs/`. The manifest builder also wrote its own provenance there.

## Limits and next-stage boundaries

No live teacher behavior was tested; the reported earlier canary is preserved
as user evidence. The configured student revision has not been downloaded,
licensed, template-checked, fine-tuned, or evaluated by this stage. The SDK
adapter is validated with fake clients only. Live service sampling support
and alias drift must be recorded when future authorized calls are made.

Storage uses local POSIX locking and atomic rename, with O(log size) append
cost. It is not a distributed/high-throughput log. There is no tokenizer,
token loss-mask implementation, trainer, scientific evaluator, or verified
seed bank yet. The loader emits explicit role-level supervision and source
labels only. Further baseline and mechanism claims require the remaining
handoff protocol, including honest data/interaction accounting and sealed
evaluation boundaries.
