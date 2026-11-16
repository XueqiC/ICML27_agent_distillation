#!/usr/bin/env python3
"""wsp_shared_directions.py SNAPSHOT OUT_PREFIX NAME=ADAPTER_DIR [NAME=ADAPTER_DIR ...]   (CPU, read-only on inputs)

Question: several students trained on the same data reach nearly orthogonal weight changes (Frobenius cosine ~0.05)
that still share their leading OUTPUT directions. What are those shared directions, how much of each student's change
lies in them, and which vocabulary tokens do they point at?

For every module that writes into the residual stream (mlp.down_proj, self_attn.o_proj; d_out = hidden size):
  U_s  = top-K left singular vectors of Delta_s = (alpha/r) B_s A_s for each student s (exact, via thin QR + small SVD);
  the shared directions w_1, w_2, ... are the left singular vectors of [U_1 U_2 ... U_S]; lambda_j = sigma_j^2 / S is the
  "sharedness" in [1/S, 1] (1 = the direction is in every student's top-K output subspace);
  energy_s = || w_1^T Delta_s ||^2 / || Delta_s ||_F^2 : share of student s's change that writes along w_1.
Logit lens (no forward pass, so the SIGN of a direction is not determined): tokens with the largest and the smallest
unembedding score E w_1, reported as two lists. Meaningful mainly for the last layers.
Outputs: OUT_PREFIX.json (all numbers) and OUT_PREFIX.txt (readable summary).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

K = 4
TOPTOK = 14
snapshot, out_prefix, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
names = [s.split("=", 1)[0] for s in specs]
dirs = [s.split("=", 1)[1] for s in specs]
assert len(dirs) >= 2


def load(adir):
    cfg = json.load(open(adir + "/adapter_config.json"))
    scale = cfg["lora_alpha"] / cfg["r"]
    mods = defaultdict(dict)
    with safe_open(adir + "/adapter_model.safetensors", "pt", device="cpu") as f:
        for k in f.keys():
            m = re.match(r"(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$", k)
            if m and re.search(r"(mlp\.down_proj|self_attn\.o_proj)$", m.group(1)):
                mods[m.group(1)][m.group(2)] = f.get_tensor(k).to(torch.float64)
    out = {}
    for name, ab in mods.items():
        qb, rb = torch.linalg.qr(ab["B"], mode="reduced")
        qa, ra = torch.linalg.qr(ab["A"].T, mode="reduced")
        uc, s, _ = torch.linalg.svd(rb @ ra.T)
        out[name] = (qb @ uc, s * scale)            # left singular vectors (d_out x r), singular values
    return out


students = [load(d) for d in dirs]
common = sorted(set.intersection(*[set(s) for s in students]),
                key=lambda n: (int(re.search(r"layers\.(\d+)\.", n).group(1)), n))
# unembedding = tied token embedding of the text decoder
emb = None
with safe_open(snapshot + "/model.safetensors", "pt", device="cpu") as f:
    cands = [k for k in f.keys() if k.endswith("embed_tokens.weight")]
    for k in cands:
        t = f.get_slice(k)
        shape = t.get_shape()
        if len(shape) == 2 and shape[0] > 100000 and shape[1] == students[0][common[0]][0].shape[0]:
            emb = f.get_tensor(k).to(torch.float32)
            emb_key = k
            break
assert emb is not None, cands
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def toks(ids):
    return [tok.convert_ids_to_tokens(int(i)) for i in ids]


rows = []
for name in common:
    layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
    typ = name.rsplit(".", 1)[-1]
    us = [s[name][0][:, :K] for s in students]
    w, sig, _ = torch.linalg.svd(torch.cat(us, dim=1), full_matrices=False)
    lam = (sig ** 2 / len(students)).tolist()
    w1 = w[:, 0]
    energy = []
    for s in students:
        u, sv = s[name]
        energy.append(float((((w1 @ u) * sv) ** 2).sum() / (sv ** 2).sum()))
    top1_share = [float(s[name][1][0] ** 2 / (s[name][1] ** 2).sum()) for s in students]
    score = emb @ w1.to(torch.float32)
    hi, lo = torch.topk(score, TOPTOK).indices, torch.topk(-score, TOPTOK).indices
    rows.append({"module": name, "layer": layer, "type": typ, "sharedness": lam[:K], "energy_along_w1": energy,
                 "own_top1_share": top1_share, "tokens_plus": toks(hi), "tokens_minus": toks(lo),
                 "score_std": float(score.std()), "score_max": float(score.max()), "score_min": float(score.min())})
summary = {"students": dict(zip(names, dirs)), "K": K, "unembedding_key": emb_key, "modules": len(rows)}
for typ in ("down_proj", "o_proj"):
    sel = [r for r in rows if r["type"] == typ]
    summary[typ] = {"mean_sharedness_1": sum(r["sharedness"][0] for r in sel) / len(sel),
                    "mean_sharedness_2": sum(r["sharedness"][1] for r in sel) / len(sel),
                    "mean_energy_along_w1": sum(sum(r["energy_along_w1"]) / len(r["energy_along_w1"]) for r in sel) / len(sel),
                    "mean_own_top1_share": sum(sum(r["own_top1_share"]) / len(r["own_top1_share"]) for r in sel) / len(sel)}
json.dump({"summary": summary, "modules": rows}, open(out_prefix + ".json", "w"), indent=1, ensure_ascii=False)
with open(out_prefix + ".txt", "w") as f:
    f.write(json.dumps(summary, indent=1) + "\n\n")
    f.write("layer type      sharedness(1,2)  energy_along_w1(per student)  own_top1_share(per student)\n")
    for r in rows:
        f.write("%2d %-9s %.2f %.2f   %s   %s\n" % (r["layer"], r["type"], r["sharedness"][0], r["sharedness"][1],
                " ".join("%.2f" % e for e in r["energy_along_w1"]), " ".join("%.2f" % e for e in r["own_top1_share"])))
    f.write("\nlogit lens of the first shared direction (sign undetermined: two lists per module)\n")
    for r in rows:
        if r["layer"] >= 30:
            f.write("L%d %s  [+] %s\n            [-] %s\n" % (r["layer"], r["type"], " ".join(r["tokens_plus"]), " ".join(r["tokens_minus"])))
print(json.dumps(summary, indent=1))
print("SHARED_DIRECTIONS_DONE")
