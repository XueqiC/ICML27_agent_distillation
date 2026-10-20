# engineering fix: serving equivalence for Gemma-4 k_eq_v layers

HF Transformers 5.14.1 reuses the **output** of `k_proj` as the value input
on Gemma-4 full-attention layers with `attention_k_eq_v=true`. This includes
PEFT's K LoRA delta before `v_norm`. vLLM 0.27.1 copies the base K weights
into its packed K and V slots, but a `k_proj` adapter updates only the K slot.
Without a V adapter, served V therefore omits the trained K LoRA delta.

`harness/serve_student.py` automatically applies
`gemma4_k_eq_v_value_lora_v1` to every unmerged adapter. It checks every text
decoder layer: absence of `v_proj.weight` in the local snapshot's safetensors
index (or headers) must agree exactly with `text_config.attention_k_eq_v`
and `layer_types`. Any disagreement or existing V LoRA on a k_eq_v layer
is refused. For each qualifying layer with a K LoRA A/B pair, it appends
bitwise copies as V LoRA A/B, preserving dtype, shape, all original tensor
bytes, and scaling. It extends `target_modules` only when necessary. With no
qualifying adapted layers, it serves the original adapter.

Derived adapters live only in `--output/derived_adapters/NAME`. Optional
`--lora-conversion targets|gemma4` runs afterward, writing the final naming
copy in `--output/adapters/NAME`. Server provenance retains the original
checkpoint identity and records the derivation rule, affected layer indices,
derived path, and config/weight hashes, plus any naming conversion.
`tools/check_lora_serving.py` recomputes the documented transformations and
compares file hashes before accepting the server binding, including during
report validation. Validation does not write adapter files.

**HF reference and training are unchanged** and continue to use the original
adapter. No checkpoint is modified, no LoRA scale or evaluation tolerance is
changed, and this is an engineering serving-equivalence fix, not a method
change. CPU byte/provenance tests do not establish GPU numerical equivalence;
the existing HF/vLLM gate remains required for subsequent evaluations.

**Historical evaluations:** ALL earlier unmerged-LoRA vLLM evaluations served
V without the `k_proj` delta on those 8 k_eq_v full-attention layers. They
passed the gate with a smaller systematic error; those passes did not
establish faithful serving of the trained HF/PEFT model. Preserve those
results with this qualification rather than relabeling them as corrected
evaluations.
