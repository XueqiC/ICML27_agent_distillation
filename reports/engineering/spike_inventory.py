# Inventory of loss spikes in every main-grid training run (hpg). Per-update token-mean loss l_u = loss_sum_u / dtokens_u.
# Spike at u: l_u > max(5 * median(l_{u-20..u-1}), median + 0.5). Event = first spike u plus the following updates until
# l falls back below 2 * pre-spike median. Recovered = final complete-epoch mean <= max(1.1 * min epoch mean, min + 0.02).
import glob, json, os, re, statistics as st, sys
BL = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/reports/runs/baselines"
out = []
for d in sorted(glob.glob(BL + "/*")):
    cell = os.path.basename(d)
    if not re.match(r"(tsft|adc|smc|sadc)-k(30|100|300|500)-", cell): continue
    fs = sorted(glob.glob(d + "/training/stages/train-*/attempt-*/output/updates.jsonl")) or \
         sorted(glob.glob(d + "/stages/train-*/attempt-*/output/updates.jsonl")) or sorted(glob.glob(d + "/**/updates.jsonl", recursive=True))
    if not fs: continue
    rows = [json.loads(l) for l in open(fs[-1]) if l.strip()]
    if len(rows) < 25: continue
    prev, L, G, LR, EP, IDX = 0, [], [], [], [], []
    for r in rows:
        tok = r.get("supervised_tokens_seen"); ls = r.get("supervised_loss_sum")
        if tok is None or ls is None: L.append(r.get("loss")); 
        else: n = tok - prev; prev = tok; L.append(ls / n if n else float("nan"))
        G.append(r.get("gradient_norm")); LR.append(r.get("lr", r.get("learning_rate"))); EP.append(r.get("epoch")); IDX.append(r.get("indices"))
    n = len(L); events = []; u = 20
    while u < n:
        med = st.median(L[u - 20:u]); gmed = st.median([g for g in G[u - 20:u] if g is not None] or [0])
        if L[u] > max(5 * med, med + 0.5):
            v = u
            while v < n and L[v] > 2 * med: v += 1
            peak = max(L[u:v] or [L[u]])
            events.append({"u": u, "frac": round(u / n, 3), "epoch": EP[u], "lr": LR[u], "pre_loss": round(med, 4), "peak": round(peak, 2),
                           "len": v - u, "pre_gnorm": round(gmed, 3), "gnorm_at": round(G[u] or 0, 1),
                           "gnorm_prev": round(G[u - 1] or 0, 1), "indices": IDX[u]})
            u = v + 20
        else: u += 1
    out.append({"cell": cell, "updates": n, "lr0": LR[0], "events": events, "final_loss": round(st.mean(L[-10:]), 4),
                "min10": round(min(st.mean(L[i:i + 10]) for i in range(0, n - 9)), 4)})
json.dump(out, open(sys.argv[1], "w"))
for c in out:
    print(f"{c['cell']:24s} n={c['updates']:5d} lr0={c['lr0']} final10={c['final_loss']:.4f} min10={c['min10']:.4f} spikes={len(c['events'])}")
    for e in c["events"]:
        print(f"    u={e['u']} ({e['frac']:.0%}, ep {e['epoch']}) lr={e['lr']:.2e} pre_loss={e['pre_loss']} peak={e['peak']} len={e['len']} "
              f"gnorm pre~{e['pre_gnorm']} prev={e['gnorm_prev']} at={e['gnorm_at']}")
