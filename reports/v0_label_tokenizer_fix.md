# V0 rationale tokenizer and exact-request reuse

Worktree: `/home/xueqi/hq/projects/ICML27_wt_repair_eval`, HEAD `96605d3`.
Changes remain unstaged. No model weights, GPU execution, real teacher calls,
or writes to `/home/xueqi/hq/projects/ICML27_v0_runs` were used.

## Implementation

`tools/v0_round.py label` now uses `tools.build_v2_dataset.local_tokenizer`.
The configured student repository/revision discovers the local E4B snapshot;
`--snapshot` selects an explicit pinned local snapshot using the same V2 checks.
Missing `tokenizer.json` aborts before label phase creation or teacher construction.
`--byte-token-cap` is an explicit test override. Parallel label workers retain
the shared `LockedTokenizer` wrapper. The phase plan records the tokenizer path,
identity and tokenizer file hashes, and binds these files as inputs.

`label --reuse-labels-from <previous label phase dir>` indexes completed raw
outputs by the actual canonical bytes of `prompt.to_json()`. It verifies the
source plan and inputs, including manifest, attempts, reviews and config. An
identical config can reside in another checkout. Completed phases must pass
their artifact seal; interrupted phases can supply completed durable requests
or completed transport receipts. Started, failed and truncated transport rows
are not reusable completions.

Each reused request writes `reused: true`, `source_path`, `source_sha256`, and
the original `raw_output`; transport sources also record `source_line`. Reuse
constructs no teacher and creates no teacher usage. The current tokenizer and
all existing label checks run again. Unmatched requests call the teacher normally.
Duplicate canonical requests use the first durable receipt in sorted path order,
with completed transport receipts as fallback.

Revalidation may change an earlier rationale in the public prefix. That changes
later request bytes, so those later requests are cache misses. The offline audit
below evaluates each original recorded request independently; it does not claim
that rebuilding the whole label phase would reuse every request or need zero calls.

## Real offline audit

Read-only inputs: `ICML27_v0_runs/v0_k20_r1/label` and
`ICML27_v0_runs/v0_k20_r1/dataset/examples.jsonl`.

Tokenizer snapshot:
`/home/xueqi/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2`

`tokenizer.json` SHA-256:
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`

All **495** recorded raw outputs pass `parse_label` with the real tokenizer.
Rationales span **15–45 student tokens**, below the unchanged 64-token cap.
Running the existing prefix checks, candidate grounding and guidance-text checks
against each recorded prompt and its dataset messages gives:

| `label_check` | Original byte cap | Real tokenizer |
| --- | ---: | ---: |
| `passed` | 2 | 471 |
| `reasoning_token_cap` | 493 | 0 |
| `label_grounding_failed` | 0 | 24 |

The 24 grounding failures consist of 14 `unsupported: true` outputs, nine
outputs failing both public-instance and candidate-grounding checks, and one
failing candidate grounding alone. They remain invalid under the existing
protocol; this fix does not relax those checks.

The 495 raw outputs correspond exactly to the 495 patch-produced dataset rows.
Another 56 audit rows have no raw output and are excluded from the dataset.
The audit verified canonical request/raw-output matches to durable receipts,
verified public-prefix hashes against dataset messages, and rechecked hashes of
624 audit/receipt files after reading. Separately, the new reuse loader validated
the real source phase and indexed its **481 unique completed requests**.

The audit ran in memory, with CUDA disabled, HF/Transformers offline flags and
socket network functions blocked. New teacher calls and usage: **0**. Original
run files written: **0**. Detailed counts, token histogram and source/tokenizer
hashes are in [v0_label_tokenizer_offline_audit.json](v0_label_tokenizer_offline_audit.json).

## Validation

Interpreter: `/home/xueqi/hq/projects/ICML27_agent_distillation/.venv/bin/python`.
Both commands use `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:.`; repository test fixtures also block
network/OpenAI access and use fake clients/environments.

```text
-m pytest -q tests/test_v0*
-m pytest -q
```

V0 suite: **125 passed** in 58.27 seconds (111 existing plus 14 new regressions).
Full CPU suite: **2,780 passed / 51 failed / 102 skipped** in 2,210.44 seconds
(36m50s). Compared with **2,766 passed / 51 failed**, this adds **14 passing
tests**. All 51 failure IDs exactly match the saved baseline in
`spike_out/v0_work_logs/full_cpu_comparison.json`, also referenced by the
2,766-pass result in `reports/incident2_v0_transport.md`. No new or resolved
failures were found.

Logs: `/tmp/v0-label-fix-tests.log`, `/tmp/v0-label-fix-full-cpu.log`.
`git diff --check` passed; HEAD remains `96605d3` and the index is empty.

Unstaged files:

- `tools/v0_round.py`
- `repair/v0_labeling.py` (new)
- `tests/test_v0_pipeline.py`
- `tests/test_v0_labels.py` (new)
- `reports/v0_label_tokenizer_fix.md` (new)
- `reports/v0_label_tokenizer_offline_audit.json` (new)
