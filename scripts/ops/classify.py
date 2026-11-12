#!/usr/bin/env python3
"""Per-root table of what is left on /blue after the big-file moves, by class.
usage: classify.py units.tsv MOVED_LIST_DIR"""
import sys, os, re, glob, collections
units, moved_dir = sys.argv[1], sys.argv[2]
GiB = 2.0 ** 30
D = "/blue/fsu-compsci-dept/xc25.fsu/hq/"; Y = "/blue/yd24f.fsu/xc25.fsu/hq/"
KEY = {D + "ICML27_agent_distillation": "dept-ICML27_agent_distillation", D + "ICML27_v0_runs": "dept-ICML27_v0_runs",
       D + "ICML27_runs": "dept-ICML27_runs", D + "ICML27_p1_backup": "dept-ICML27_p1_backup", D + "tc-alignment": "dept-tc-alignment",
       D + "tc-alignment-vllm": "dept-tc-alignment-vllm", D + "envs/p1-venv": "dept-envs-p1-venv",
       Y + "ICML27_archive": "yd-ICML27_archive", Y + "tc-alignment": "yd-tc-alignment", Y + "_trash": "yd-_trash"}
ENV = re.compile(r"(^|/)(\.venv|site-packages|\.uv-cache|uv-pythons|node_modules|__pycache__|\.pytest_cache)(/|$)|^envs/")
TRASH = re.compile(r"(^|/)(_trash|_failed)(/|$)")
def cls(key, rel):
    if key == "dept-envs-p1-venv" or ENV.search(rel): return "env (regenerable)"
    if key == "yd-_trash" or TRASH.search(rel): return "trash"
    return "data/run files"
moved = collections.defaultdict(set)
for p in glob.glob(os.path.join(moved_dir, "*.list")):
    k = os.path.basename(p).split(".")[0]
    moved[k] |= set(l.rstrip("\n") for l in open(p))
tot = collections.defaultdict(lambda: [0, 0]); bigleft = collections.defaultdict(lambda: [0, 0]); mv = collections.defaultdict(lambda: [0, 0])
detail = collections.defaultdict(lambda: [0, 0])
for line in open(units, errors="replace"):
    f = line.rstrip("\n").split("\t")
    if f[0] == "U":
        k = KEY[f[1]]; c = cls(k, "" if f[2] == "(shallow)" else f[2])
        tot[(k, c)][0] += int(f[3]); tot[(k, c)][1] += int(f[4])
        pre = "(shallow)" if f[2] == "(shallow)" else "/".join(f[2].split("/")[:3])
        detail[(k, c, pre)][0] += int(f[3]); detail[(k, c, pre)][1] += int(f[4])
    elif f[0] == "B":
        k = KEY[f[1]]; c = cls(k, f[3]); s = int(f[2])
        # a big file found by the shallow unit is classified by its own path, units by their unit path: keep the
        # big-file bookkeeping separate and only subtract moved bytes from the per-root totals
        if f[3] in moved[k]: mv[k][0] += 1; mv[k][1] += s
        else: bigleft[(k, c)][0] += 1; bigleft[(k, c)][1] += s
print(f"{'root':32s} {'class':18s} {'files':>11s} {'GiB':>8s} {'KB/file':>8s} | {'big left':>8s} {'GiB':>7s}")
grand = collections.defaultdict(lambda: [0, 0])
for k in KEY.values():
    rn = rb = 0
    for c in ("data/run files", "trash", "env (regenerable)"):
        n, b = tot.get((k, c), [0, 0])
        if not n: continue
        bl = bigleft.get((k, c), [0, 0])
        print(f"{k:32s} {c:18s} {n:11,d} {b/GiB:8.1f} {b/max(n,1)/1024:8.0f} | {bl[0]:8d} {bl[1]/GiB:7.1f}")
        rn += n; rb += b; grand[c][0] += n; grand[c][1] += b
    print(f"{'':32s} {'= census total':18s} {rn:11,d} {rb/GiB:8.1f}          | moved {mv[k][0]} files {mv[k][1]/GiB:.1f} GiB -> left {(rb-mv[k][1])/GiB:.1f} GiB")
print()
for c, (n, b) in grand.items():
    print(f"ALL ROOTS {c:18s} {n:11,d} files {b/GiB:8.1f} GiB (census, before moves)")
print(f"moved big files: {sum(v[0] for v in mv.values())} files, {sum(v[1] for v in mv.values())/GiB:.1f} GiB")
print("\nlargest env prefixes (regenerable candidates):")
for (k, c, pre), (n, b) in sorted(((kk, vv) for kk, vv in detail.items() if kk[1].startswith("env")), key=lambda x: -x[1][1])[:16]:
    print(f"  {b/GiB:7.1f} GiB {n:9,d} files  {k}/{pre}")
print("\nlargest trash prefixes:")
for (k, c, pre), (n, b) in sorted(((kk, vv) for kk, vv in detail.items() if kk[1] == "trash"), key=lambda x: -x[1][1])[:14]:
    print(f"  {b/GiB:7.1f} GiB {n:9,d} files  {k}/{pre}")
print("\nlargest data/run prefixes by FILES:")
for (k, c, pre), (n, b) in sorted(((kk, vv) for kk, vv in detail.items() if kk[1].startswith("data")), key=lambda x: -x[1][0])[:16]:
    print(f"  {n:10,d} files {b/GiB:7.1f} GiB  {k}/{pre}")
