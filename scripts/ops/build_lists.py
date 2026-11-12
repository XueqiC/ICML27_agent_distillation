#!/usr/bin/env python3
"""Build per-root big-file lists from a census units.tsv.
usage: build_lists.py units.tsv OUTDIR BATCH [DONE_DIR]
Writes OUTDIR/<key>.<BATCH>.list (relative paths) for files >= 64 MiB that are not env/venv files and are
not already in DONE_DIR/<key>.*.list (earlier batches)."""
import sys, os, re, collections, glob
units, outdir, batch = sys.argv[1], sys.argv[2], sys.argv[3]
done_dir = sys.argv[4] if len(sys.argv) > 4 else outdir
GiB = 2.0 ** 30
D = "/blue/fsu-compsci-dept/xc25.fsu/hq/"; Y = "/blue/yd24f.fsu/xc25.fsu/hq/"
O = "/orange/yd24f.fsu/xc25.fsu/"
ROOTS = {  # root -> (key, dstroot)
    D + "ICML27_agent_distillation": ("dept-ICML27_agent_distillation", O + "icml27-cold/blue-dept/ICML27_agent_distillation"),
    D + "ICML27_v0_runs": ("dept-ICML27_v0_runs", O + "icml27-cold/blue-dept/ICML27_v0_runs"),
    D + "ICML27_runs": ("dept-ICML27_runs", O + "icml27-cold/blue-dept/ICML27_runs"),
    D + "ICML27_p1_backup": ("dept-ICML27_p1_backup", O + "icml27-cold/blue-dept/ICML27_p1_backup"),
    D + "tc-alignment": ("dept-tc-alignment", O + "tc-cold/blue-dept/tc-alignment"),
    D + "tc-alignment-vllm": ("dept-tc-alignment-vllm", O + "tc-cold/blue-dept/tc-alignment-vllm"),
    D + "envs/p1-venv": ("dept-envs-p1-venv", O + "tc-cold/blue-dept/envs/p1-venv"),
    Y + "ICML27_archive": ("yd-ICML27_archive", O + "icml27-cold/blue-yd24f/ICML27_archive"),
    Y + "tc-alignment": ("yd-tc-alignment", O + "tc-cold/blue-yd24f/tc-alignment"),
    Y + "_trash": ("yd-_trash", O + "tc-cold/blue-yd24f/_trash"),
}
ENV = re.compile(r"(^|/)(\.venv|site-packages|\.uv-cache|uv-pythons|node_modules)(/|$)|^envs/")
os.makedirs(outdir, exist_ok=True)
big = collections.defaultdict(dict)
for line in open(units, errors="replace"):
    f = line.rstrip("\n").split("\t")
    if f[0] == "B" and len(f) >= 4:
        big[f[1]][f[3]] = int(f[2])
print(f"{'key':34s} {'new':>5s} {'GiB':>8s} {'done':>5s} {'env n':>6s} {'env GiB':>8s}  dstroot")
tot_n = tot_b = 0
for root, (key, dst) in ROOTS.items():
    done = set()
    for p in glob.glob(os.path.join(done_dir, key + ".*.list")):
        done |= set(l.rstrip("\n") for l in open(p))
    env = {p: s for p, s in big[root].items() if ENV.search(p) or key == "dept-envs-p1-venv"}
    new = {p: s for p, s in big[root].items() if p not in env and p not in done}
    bad = [p for p in new if p.startswith("/") or ".." in p.split("/") or "\t" in p or "\n" in p or not p]
    assert not bad, bad[:3]
    if new:
        with open(os.path.join(outdir, f"{key}.{batch}.list"), "w") as fh:
            for p in sorted(new):
                fh.write(p + "\n")
    tot_n += len(new); tot_b += sum(new.values())
    print(f"{key:34s} {len(new):5d} {sum(new.values())/GiB:8.1f} {len(done):5d} {len(env):6d} {sum(env.values())/GiB:8.1f}  {dst}")
print(f"{'TOTAL new':34s} {tot_n:5d} {tot_b/GiB:8.1f}")
