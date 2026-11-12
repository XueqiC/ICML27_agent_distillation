#!/usr/bin/env python3
"""Summarise a tc_census_par units.tsv (possibly partial). usage: census_report.py units.tsv [units_total]"""
import sys, collections

GiB = 2.0 ** 30
path = sys.argv[1]
units_total = int(sys.argv[2]) if len(sys.argv) > 2 else None
U = []; B = []; E = []
for line in open(path, errors="replace"):
    f = line.rstrip("\n").split("\t")
    if f[0] == "U" and len(f) >= 17:
        U.append((f[1], f[2], int(f[3]), int(f[4]), [int(x) for x in f[5:11]], [int(x) for x in f[11:17]]))
    elif f[0] == "B" and len(f) >= 4:
        B.append((f[1], int(f[2]), f[3]))
    elif f[0] == "E":
        E.append(f)
bins = ["<4K", "4K-64K", "64K-1M", "1M-64M", "64M-1G", ">=1G"]
short = lambda r: r.replace("/blue/fsu-compsci-dept/xc25.fsu/hq/", "dept:").replace("/blue/yd24f.fsu/xc25.fsu/hq/", "yd:")
print(f"units done={len(U)}" + (f" of {units_total + len(set(u[0] for u in U))}" if units_total else "") + f"  big files(>=64MiB)={len(B)}  failed units={len(E)}")
roots = collections.OrderedDict()
for r, rel, n, b, hc, hb in U:
    d = roots.setdefault(r, dict(n=0, b=0, hc=[0] * 6, hb=[0] * 6, units=0))
    d["n"] += n; d["b"] += b; d["units"] += 1
    for i in range(6):
        d["hc"][i] += hc[i]; d["hb"][i] += hb[i]
bigb = collections.Counter(); bign = collections.Counter()
for r, s, p in B:
    bigb[r] += s; bign[r] += 1
tn = tb = tbig = 0
print(f"\n{'root':44s} {'units':>6s} {'files':>11s} {'GiB':>9s} {'KB/file':>8s} {'big n':>6s} {'big GiB':>8s} {'big%':>5s}")
for r, d in roots.items():
    tn += d["n"]; tb += d["b"]; tbig += bigb[r]
    print(f"{short(r):44s} {d['units']:6d} {d['n']:11,d} {d['b']/GiB:9.1f} {d['b']/max(d['n'],1)/1024:8.0f} {bign[r]:6d} {bigb[r]/GiB:8.1f} {100*bigb[r]/max(d['b'],1):5.0f}")
print(f"{'TOTAL (so far)':44s} {len(U):6d} {tn:11,d} {tb/GiB:9.1f} {tb/max(tn,1)/1024:8.0f} {len(B):6d} {tbig/GiB:8.1f} {100*tbig/max(tb,1):5.0f}")
print("\nsize histogram over everything so far (files / GiB):")
hc = [sum(d["hc"][i] for d in roots.values()) for i in range(6)]
hb = [sum(d["hb"][i] for d in roots.values()) for i in range(6)]
for i in range(6):
    print(f"  {bins[i]:8s} {hc[i]:12,d} files  {hb[i]/GiB:9.1f} GiB")
# aggregate by a depth-3 prefix inside each root
agg = collections.defaultdict(lambda: [0, 0])
for r, rel, n, b, _, _ in U:
    k = "(shallow)" if rel == "(shallow)" else "/".join(rel.split("/")[:3])
    agg[(r, k)][0] += n; agg[(r, k)][1] += b
print("\ntop 30 depth-3 prefixes by bytes:")
for (r, k), (n, b) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
    print(f"  {b/GiB:8.1f} GiB {n:11,d} files {b/max(n,1)/1024:9.0f} KB/file  {short(r)}/{k}")
print("\ntop 20 depth-3 prefixes by files:")
for (r, k), (n, b) in sorted(agg.items(), key=lambda x: -x[1][0])[:20]:
    print(f"  {n:11,d} files {b/GiB:8.1f} GiB {b/max(n,1)/1024:9.0f} KB/file  {short(r)}/{k}")
# big files grouped by parent directory
par = collections.defaultdict(lambda: [0, 0])
for r, s, p in B:
    par[(r, p.rsplit("/", 1)[0] if "/" in p else ".")][0] += 1
    par[(r, p.rsplit("/", 1)[0] if "/" in p else ".")][1] += s
print(f"\nbig files sit in {len(par)} directories; top 25 by bytes:")
for (r, k), (n, b) in sorted(par.items(), key=lambda x: -x[1][1])[:25]:
    print(f"  {b/GiB:8.1f} GiB {n:5d} files  {short(r)}/{k}")
ext = collections.Counter(); extb = collections.Counter()
for r, s, p in B:
    e = p.rsplit("/", 1)[-1]; e = e.rsplit(".", 1)[-1] if "." in e else "(none)"
    ext[e] += 1; extb[e] += s
print("\nbig files by extension:")
for e, b in extb.most_common(12):
    print(f"  {b/GiB:8.1f} GiB {ext[e]:6d} files  .{e}")
