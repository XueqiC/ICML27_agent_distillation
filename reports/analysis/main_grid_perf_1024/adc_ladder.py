import json, glob, os, sys
BL = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs/baselines"
pat = sys.argv[1] if len(sys.argv) > 1 else "adc-k300-e*"
for d in sorted(glob.glob(BL + "/" + pat)):
    cell = os.path.basename(d)
    fs = sorted(glob.glob(d + "/training/stages/train-sft/attempt-*/output/updates.jsonl"))
    if not fs: print(cell, "no updates"); continue
    rows = [json.loads(l) for l in open(fs[-1]) if l.strip()]
    ep, prev, cnt = {}, 0, {}
    for r in rows:
        n = r["supervised_tokens_seen"] - prev; prev = r["supervised_tokens_seen"]
        x = ep.setdefault(r["epoch"], [0.0, 0]); x[0] += r["supervised_loss_sum"]; x[1] += n
        cnt[r["epoch"]] = cnt.get(r["epoch"], 0) + 1
    per = max(cnt.values())
    comp = [x for x in sorted(ep) if cnt[x] >= per]
    means = [ep[x][0] / ep[x][1] for x in comp]
    done = os.path.exists(d + "/training/stages/train-sft/complete.json")
    lmin = min(means); valid = means[-1] <= max(1.1 * lmin, lmin + 0.02)
    coll = [i for i in range(1, len(means)) if means[i] > 2 * min(means[:i]) and means[i] > min(means[:i]) + 0.1]
    g = [r.get("gradient_norm", r.get("grad_norm", 0)) or 0 for r in rows]
    state = "done" if done else "run "
    flag = (" collapse@ep%d" % (coll[0] + 1)) if coll else ""
    print(f"{cell:24s} {state} ep {len(comp)} per {per} updates {len(rows)} L_last {means[-1]:.4f} L_min {lmin:.4f} "
          f"{'valid' if valid else 'INVALID'}{flag} first {means[0]:.3f} gmax {max(g):.0f}")
