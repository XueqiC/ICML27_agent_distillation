# Optional shape ablations and adapter algebra

The existing recipe remains r64/alpha128, all seven projection types, dropout
0.05, two epochs and lr 2e-4. For the requested K20 trajectory-SFT experiment,
select the K20 verified collection and explicitly use `--epochs 32
--learning-rate 5e-5`. Keep the other settings identical across ablations.

Both `baselines.kang.pipeline` and `baselines.common.fsdp_train` accept:

- `--lora-rank 1..256`; alpha becomes twice this rank unless explicitly supplied.
- `--lora-alpha INT`, positive; by itself it keeps rank 64.
- `--lora-targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- `--lora-layers 0-13,20`, inclusive indices.
- `--lora-modules 22.down_proj,23.q_proj`, excluding targets/layers flags.
- `--checkpoint-updates 1 10 20`, strictly increasing, within the training dose.

The shared trainer allows these only with SFT/Kang LoRA. Missing projections
fail explicitly; some Gemma layers have no separate k/v projection. Intermediate
adapters go to `output/checkpoints/update-N`, with `final: false`, exact exposure
counters and adjacent provenance for the existing adapter-serving gate. Export
does not run a forward pass, change model mode, or consume training RNG streams.
The normal serving equivalence gate still applies before evaluation.

`scripts/hpg_kang_wsp.slurm` forwards nonempty `KANG_LORA_RANK`,
`KANG_LORA_ALPHA`, `KANG_LORA_TARGETS`, `KANG_LORA_LAYERS`, `KANG_LORA_MODULES`,
and whitespace-separated `KANG_CHECKPOINT_UPDATES`. The original
`scripts/hpg_kang.slurm` is unchanged.

Preparation seals bind **both** the pipeline identity/recipe and the effective
trainer arguments. A mismatched shape fails before model loading. To reuse
encoded data for a new shape, run CPU preparation into a **new output tree**
with the desired flags and `--reuse-prepared /path/to/source/pipeline`. The
source must have a valid seal from the same code tree; all identity fields
outside shape/checkpoint controls must match. Encoded examples are hard-linked
on the same filesystem, without tokenization or another payload copy. Training
uses the new tree and the same shape flags, omitting `--reuse-prepared`.

Omitted controls preserve the recipe, launch argv, effective trainer config and
deterministic adapter bytes. The golden CPU regression was captured using the
original tracked implementations. Full identities/provenance also hash code:
those hashes necessarily change with this implementation; no provenance hash
exclusion or relaxed old-code resume rule was introduced.

Use `python -m tools.wsp_adapter_ops --help` for the CPU tool. Examples:

```sh
python -m tools.wsp_adapter_ops spectrum ADAPTER --out spectrum.json
python -m tools.wsp_adapter_ops truncate ADAPTER --out rank16 --rank 16
python -m tools.wsp_adapter_ops truncate ADAPTER --out selected --top-modules 20 --layers 0-13,20
python -m tools.wsp_adapter_ops average ADAPTER_A ADAPTER_B --out mean --rank 32
python -m tools.wsp_adapter_ops scale ADAPTER --out scaled --factor 0.5
python -m tools.wsp_adapter_ops overlap ADAPTER_A ADAPTER_B --out overlap.json --k 1 4 16
```

Outputs must not exist. `truncate`, `average`, and `scale` create this layout:

```text
OUT/
  provenance.json
  checkpoint/
    adapter_config.json
    adapter_model.safetensors
    wsp_ops.json
    training_state.json
```

Pass `OUT/checkpoint` as the adapter directory for evaluation or another
operation. `spectrum` and `overlap` still write JSON reports and accept adapter
directories without training provenance.

Math uses float64 thin QR and a core SVD; factors are saved as float32.
Explicit averaging weights are used verbatim, without
normalization. Zero inputs have kept share 1; zero-update overlaps are null.
When k exceeds either numerical rank, overlap uses and reports the common
effective k, with chance level effective_k / dimension. Summary norm weights
are the arithmetic mean of the two squared norms.

Derivation requires every source to have `training_state.json` and its parent's
`provenance.json`, a matching provenance identity digest, matching recorded
artifact hashes, and the checkpoint metadata `supervised_tokens`, `passes`,
and `packing`. Missing or inconsistent evidence is refused before output is
created. Averaging also requires the same student `requested_repo`, `revision`,
and `base_snapshot.metadata_sha256` across all sources.

Derived provenance preserves the first source's training config and adds
`wsp_derived`: operation parameters (including output rank/alpha), tool hash,
source paths, artifact/state hashes, provenance digests, and checkpoint metadata.
For chained operations the previous derivation remains under `wsp_derived.parent`.
`wsp_ops.json` keeps its input hashes and retained-energy reports. The derived
`training_state.json` is written last, binds the new provenance and artifacts,
and explicitly marks `wsp_derived: true`. Its exposure fields come from the first
source; `wsp_sources` records each source separately. These fields do not claim
new training or sum exposure across averaged sources.

The existing evaluator accepts this schema without changes. For example, with
the usual model, verified collection, and evaluation paths configured:

```sh
ADAPTERS="name=/path/OUT/checkpoint" sbatch --export=ALL scripts/hpg_eval_adapters.slurm
```

The configured student and serving snapshot metadata must still match the
original training provenance, and the normal serving equivalence gate still
applies. CPU artifact checks do not establish HF/vLLM serving equivalence.
