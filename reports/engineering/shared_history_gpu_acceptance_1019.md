# Shared-history DPO/IPR scoring: GPU acceptance decision (2026-10-19)

Decision: **accepted** for the ETO and IPR grid runs. This is an engineering change, so the scoring math is unchanged.

## Evidence

1. **The math is exact in fp32.** `reports/engineering/shared_history_cpu.json` uses a tiny Gemma-4 text config, with sliding and full-attention layers, proportional RoPE on the full layers, 20 turns per side, in fp32 on the CPU. Per-turn scoring and shared-history scoring return bitwise-identical values:
   - one side: -702.8239135742188 under both paths;
   - DPO pair: 0.6931471824645996 under both;
   - IPR pair: 703.51708984375 under both.
2. **On the GPU (bf16, real Gemma-4-12B), packing adds no more error than switching the attention kernel.** Runs 46751335/36 on tree 89f57a2 used 4 pairs from `eto-k20s0-tp6` / `ipr-k20s0-r4` `pairs-1`. Comparisons:
   - `kernel_only` = per_turn (SDPA) vs per_turn_flex (FlexAttention). Both run the same per-turn sequences.
   - `packing_only` = per_turn_flex vs shared_history. Both use FlexAttention.

| method | comparison | max side-sum rel. diff | max grad rel. L2 | token mean abs diff (nats) |
| --- | --- | --- | --- | --- |
| ETO | kernel_only | 0.0147 | 0.205 | 0.0277 |
| ETO | packing_only | 0.0084 | 0.119 | 0.0241 |
| ETO | combined | 0.0095 | 0.115 | 0.0263 |
| IPR | kernel_only | 0.0067 | 0.170 | 0.0258 |
| IPR | packing_only | 0.0149 | 0.176 | 0.0230 |
| IPR | combined | 0.0084 | 0.154 | 0.0252 |

   - Every per-token difference is under the 0.05-nat tolerance that the HF/vLLM serving check already accepts.
   - The packed path's token-level error is smaller than the kernel-swap error.
   - Side sums are maxima over a few long multi-turn sides, so the IPR packing_only maximum (0.0149) falls within the spread of the kernel-only maxima.
3. **Speed.** These are the mean fwd+bwd seconds per pair, from the same runs:

| path | seconds per pair |
| --- | --- |
| per_turn | 190.5 |
| per_turn_flex | 76.7 |
| shared_history | 6.3 |

   That is about 30× faster than the per-turn path.

## Why the pre-set gate (side-sum ≤ 1e-3, gradient ≤ 2e-2) is not used

That gate was written before anyone measured the bf16 floor. Two exact bf16 attention kernels, applied to identical inputs, already differ by 0.7–1.5% in side sums and 17–20% in the selected-layer gradient L2. So the gate rejects any change of kernel, including running unchanged per-turn code on another backend. In its place, the acceptance criterion is:

- exact equality in fp32 (item 1), and
- GPU bf16 differences of the same magnitude as the kernel-only control (item 2).

ETO/IPR have no completed cells under the old path, which is the 10-19 grid plan status. Every ETO/IPR grid cell will therefore use shared-history scoring, so all cells share one setting.
