# M0c training branches

Implemented locally on `repair-v01-train` after `511dc5e`, retaining the
training tree based on `2506fa9`. No collection, teacher call, GPU training,
evaluation, commit or push was performed. CPU fake/tiny-model tests establish
software behavior only.

The supplied schema-2 contract is summarized in [E2-interchange.md](E2-interchange.md).
The local `docs/direction/2026-10-26_pipeline_v01_mechanism_prereg_ZH.md`
ends at section 4: requested sections 5 and 6 were absent. The explicit M0c
user instructions and supplied interchange contract govern these additions.

The reader accepts the three exact branch/protocol/theta0/replay combinations,
binds source and arm metadata, checks replay scope and original K20 evidence,
and retains the prior whitelist/approval/hash/frozen-split gates. Schema 1
can only mean P0_K20. P1 verifies the original support manifest bound to
theta0, and resolves its shared K20 references once for both arms. Its scoped
validation uses this tree's `sed.seed_evidence` subset, receipt, artifact,
episode and environment-win checks; the attempt-validation loop is adapted
from that same local reader. Parent manifests/selections are audited without
recursively loading unrelated parent executions. No evaluation code is used.

Both arms retain the 20-update, batch-four schedule: three repair episodes and
one seed episode per update (weight 0.25). RNG seed 1 samples uniformly by task
then episode with replacement. P0 samples seeds from D_repair_train; P1 samples
the original support and excludes all repair tasks/probes by task and group.
Neither token lengths nor arm targets can alter the common sampling order.

K0 loads the configured base and calls this tree's unchanged
`train.modeling.attach_lora` with Kang's registered rank 64, alpha 128,
dropout 0.05 and `LORA_TARGETS`. E-series module filtering remains owned by
that helper. The learning rate is the explicit tsft K20 override 5e-5;
shared-history scoring, loss normalization, precision, linear schedule,
gradient clipping and AdamW remain the existing trainer's recipe. AdamW is
fresh in both K0 arms, with no optimizer or adapter restoration.

LoRA attachment uses fixed seed **20261026**, independently of prior loading
RNG consumption. Initial adapter names/shapes/dtypes/tensor bytes are hashed
in both arms; ranks must agree before publication or any optimizer step.
The initial adapter is exported through the existing exporter to `update-00`
and its state hash is recorded in the run and checkpoint-zero manifests.
This artifact is for verification; K0 probes use the base model at zero.
Dry-run never allocates model or optimizer tensors and reports its initial
hash and actual parameter inventory as pending.

The existing adapter continuation and schema-1 sampling/encoding paths remain
unchanged. Under `train/ baselines/ harness/ src/`, the only difference from
`2506fa9` remains the previously reviewed optional `fsdp_train.py` hooks.

Manifest instance/group counts describe the available training and replay
pools separately; `batches.json` and `example_sources.json` bind each sampled
exposure to its task, episode and receipt. Dry-run exposure remains zero.

Validation on the final code:

```text
$ CUDA_VISIBLE_DEVICES='' /home/xueqi/hq/projects/tc-alignment/.venv/bin/python -m pytest -q tests/test_repair_*.py
406 passed, 4 warnings in 152.47s

$ git diff --stat 2506fa9 -- train baselines harness src
 baselines/common/fsdp_train.py | 44 +++++++++++++++++++++++++++++++++++++++---
 1 file changed, 41 insertions(+), 3 deletions(-)
```

The four PEFT warnings concern missing base-config paths for randomly
initialized tiny test models. Tests block network/OpenAI access and use CPU.
`git diff --check` passes; the shared paths also have no changes relative to
`511dc5e`. `_ref/` is absent. Nothing was committed.
