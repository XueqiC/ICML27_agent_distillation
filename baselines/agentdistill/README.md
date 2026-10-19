# AgentDistill

Frozen Gemma with an executable MCP box built by a capped gpt-6-luna manager
and creation module on the supplied VERIFIED support TRAIN tasks.
See [DEVIATIONS.md](DEVIATIONS.md) for the paper adaptation and reference audit.

Run each stage in a separate CPU job, using the same cell directory:

```bash
export PYTHONPATH="$PWD/src:$PWD"
CUDA_VISIBLE_DEVICES='' python -m baselines.agentdistill.pipeline \
  --config configs/base.yaml --snapshot "$MODEL_PATH" --verified "$VERIFIED" \
  --output "$AGENTDISTILL_OUT" --through collect
# Repeat with --through abstract, cluster, consolidate, publish, in that order.
```

`--mode prepare` validates and freezes inputs without teacher calls.
`--mode train` is the common pipeline convention; this method does no training.
Only the selected stage runs. Repeating it validates hashes and reuses results.
Keep output outside `data/verified`, `reports/runs`, and configured seed storage.

The source can be any registered verified collection, including nested
`subsets/k5` through K100 or `seeds_full_train`. Never substitute a task manifest
for `--verified`. Every cell gets its own run reservations and teacher ledger.

After publication, use a separate GPU allocation for the gate and dev140:

```bash
python -m baselines.agentdistill.pipeline \
  --config configs/base.yaml --snapshot "$MODEL_PATH" --verified "$VERIFIED" \
  --output "$AGENTDISTILL_OUT" --mode evaluate \
  --env-python "$ENV_PYTHON" --train-python "$TRAIN_PYTHON" --vllm-python "$VLLM_PYTHON"
```

`scripts/hpg_agentdistill.slurm` uses `AGENTDISTILL_THROUGH` with values
`collect`, `abstract`, `cluster`, `consolidate`, `publish`, or `evaluate`.
The wrapper requests no GPU by default. Submit the first five on a CPU
partition. Submit `evaluate` with `--partition=hpg-b200 --gpus=1` after
`publish` succeeds. Jobs must depend on their predecessor's success.

Artifacts include stage completions, all reserved runs and transitions,
teacher prompts/raw results/usage, rejected MCPs, construction outputs, the
final `box.json`, `models.json`, and `disclosure.json`. Published model records
point to the unchanged base snapshot and final box. Results are labeled
**different interface**, and training exposure is zero.

The callable interface is:

```text
Thought: Inspect the room with the available tool.
Action: mcp {"name":"inspect","arguments":{"command":"look"}}
```

Tool example (teacher-generated tools must pass the same interpreter):

```python
def inspect(env, command):
    return env.step(command)
```

Each `env.step` must be currently admissible and counts toward 40 actions.
Tools cannot access the filesystem, network, system, or hidden environment data.

Offline tests:

```bash
CUDA_VISIBLE_DEVICES='' /home/xueqi/tf5-env/bin/python -m pytest -q tests/test_agentdistill.py
```
