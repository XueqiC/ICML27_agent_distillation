#!/usr/bin/env python3
"""lora_spectrum.py ADAPTER_DIR OUT_JSON — CPU only, read-only on the adapter.

For every LoRA module of a PEFT adapter, the weight change is Delta = (alpha / r) * B @ A.
Singular values of Delta are computed exactly from two thin QR factorisations
(B = Qb Rb, A^T = Qa Ra  =>  Delta = Qb (Rb Ra^T) Qa^T), so no d_out x d_in matrix is formed.
Reported per module: squared Frobenius norm, share of it in the top-k singular directions, stable rank.
"""
import json
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

adir, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(adir + "/adapter_config.json"))
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
KS = (1, 2, 4, 8, 16, 32)
mods = defaultdict(dict)
with safe_open(adir + "/adapter_model.safetensors", "pt", device="cpu") as f:
    for k in f.keys():
        m = re.match(r"(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$", k)
        if not m:
            continue
        mods[m.group(1)][m.group(2)] = f.get_tensor(k).to(torch.float64)
rows = []
for name, ab in sorted(mods.items()):
    A, B = ab["A"], ab["B"]            # A: r x d_in, B: d_out x r
    _, Rb = torch.linalg.qr(B, mode="reduced")
    _, Ra = torch.linalg.qr(A.T, mode="reduced")
    s = torch.linalg.svdvals(Rb @ Ra.T) * scale
    e = (s ** 2)
    tot = float(e.sum())
    lm = re.search(r"layers\.(\d+)\.", name)
    typ = name.split(".")[-1]
    rows.append({
        "module": name, "layer": int(lm.group(1)) if lm else -1, "type": typ,
        "d_out": B.shape[0], "d_in": A.shape[1], "fro2": tot,
        "top_share": {str(k): (float(e[:k].sum()) / tot if tot > 0 else 0.0) for k in KS},
        "stable_rank": (tot / float(e[0])) if tot > 0 else 0.0,
        "s_max": float(s[0]),
    })
tot = sum(x["fro2"] for x in rows)
agg = {"r": r, "alpha": alpha, "scale": scale, "n_modules": len(rows), "fro2_total": tot,
       "targets": cfg.get("target_modules") if isinstance(cfg.get("target_modules"), list) else str(cfg.get("target_modules"))[:300]}
# share of the total squared norm kept when every module is truncated to its top k directions
agg["global_top_share"] = {str(k): sum(x["fro2"] * x["top_share"][str(k)] for x in rows) / tot for k in KS}
by_type, by_layer = defaultdict(float), defaultdict(float)
for x in rows:
    by_type[x["type"]] += x["fro2"] / tot
    by_layer[x["layer"]] += x["fro2"] / tot
agg["share_by_type"] = dict(sorted(by_type.items(), key=lambda t: -t[1]))
agg["share_by_layer"] = {str(k): v for k, v in sorted(by_layer.items())}
srt = sorted(rows, key=lambda x: -x["fro2"])
cum, n = 0.0, 0
need = {}
for x in srt:
    cum += x["fro2"] / tot
    n += 1
    for q in (0.25, 0.5, 0.75, 0.9):
        need.setdefault(str(q), n) if cum >= q else None
agg["modules_needed_for_share"] = need
agg["top_modules"] = [(x["module"].split("language_model.")[-1], round(x["fro2"] / tot, 4), round(x["stable_rank"], 2)) for x in srt[:15]]
sr = sorted(x["stable_rank"] for x in rows)
agg["stable_rank_quartiles"] = [sr[int(q * (len(sr) - 1))] for q in (0, 0.25, 0.5, 0.75, 1)]
json.dump({"summary": agg, "modules": rows}, open(out, "w"), indent=1)
print(json.dumps(agg, indent=1))
