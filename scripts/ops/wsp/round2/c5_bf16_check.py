#!/usr/bin/env python
"""c5_bf16_check.py — can a merged bf16 checkpoint represent a small LoRA edit? (prereg section 18, C5 fallback route)

For every module of a PEFT LoRA adapter: delta = scale * B @ A in fp32; merged = (W + delta) rounded to bf16;
rounded_delta = merged - W. Summed over all modules it reports ||delta||_F, ||rounded_delta||_F,
cosine(rounded_delta, delta), the relative error ||rounded_delta - delta||_F / ||delta||_F and the share of entries
that changed. CPU only; reads one base tensor at a time. Usage: c5_bf16_check.py BASE_SAFETENSORS NAME=ADAPTER_DIR ...
"""
import json
import math
import sys

import torch
from safetensors import safe_open


def scale_for(cfg, module):
    r, alpha = cfg["r"], cfg["lora_alpha"]
    for pat, v in (cfg.get("rank_pattern") or {}).items():
        if module == pat or module.endswith("." + pat):
            r = v
    for pat, v in (cfg.get("alpha_pattern") or {}).items():
        if module == pat or module.endswith("." + pat):
            alpha = v
    return alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r


def main():
    base_path = sys.argv[1]
    with safe_open(base_path, "pt") as base:
        base_keys = set(base.keys())
        for spec in sys.argv[2:]:
            name, adapter = spec.split("=", 1)
            cfg = json.load(open(adapter + "/adapter_config.json"))
            s_dd = s_rr = s_dr = s_err = 0.0
            changed = total = modules = 0
            with safe_open(adapter + "/adapter_model.safetensors", "pt") as ad:
                for ka in sorted(k for k in ad.keys() if k.endswith(".lora_A.weight")):
                    module = ka[: -len(".lora_A.weight")]
                    kb = module + ".lora_B.weight"
                    wkey = module[len("base_model.model."):] + ".weight"
                    if wkey not in base_keys:
                        raise SystemExit("base weight not found for %s" % module)
                    a = ad.get_tensor(ka).double()
                    b = ad.get_tensor(kb).double()
                    delta = (scale_for(cfg, module[len("base_model.model."):]) * (b @ a))
                    w = base.get_tensor(wkey)
                    if w.dtype != torch.bfloat16 or tuple(w.shape) != tuple(delta.shape):
                        raise SystemExit("unexpected base tensor %s %s %s" % (wkey, w.dtype, tuple(w.shape)))
                    w64 = w.double()
                    rounded = (w64 + delta).to(torch.bfloat16).double() - w64
                    s_dd += float((delta * delta).sum())
                    s_rr += float((rounded * rounded).sum())
                    s_dr += float((rounded * delta).sum())
                    s_err += float(((rounded - delta) ** 2).sum())
                    changed += int((rounded != 0).sum())
                    total += delta.numel()
                    modules += 1
            print(json.dumps({
                "name": name, "modules": modules, "parameters": total,
                "delta_frobenius": math.sqrt(s_dd), "rounded_delta_frobenius": math.sqrt(s_rr),
                "cosine_rounded_vs_delta": s_dr / math.sqrt(s_dd * s_rr) if s_rr > 0 else None,
                "relative_error": math.sqrt(s_err / s_dd), "share_entries_changed": changed / total,
            }), flush=True)


if __name__ == "__main__":
    main()
