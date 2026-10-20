#!/usr/bin/env python3
"""Throttled hpg submitter (2026-10-19, after a faculty request to keep at most 10 jobs in the queue at once).

Runs on rai and submits through `ssh hpg`. It never pre-submits dependency chains: each cell (one run directory)
has at most one job in the queue, and a cell's next stage is submitted only after the previous one COMPLETED and only
while the user's total job count (running + pending, every account, including jobs this keeper did not submit) and
the total requested GPUs stay under the caps. Stages resume from their hash-bound completed stages, so a stage that
ends in any other state blocks its cell until a human looks at it.

Files (all under --dir, re-read every loop so cells and limits can be edited while it runs):
  plan.json   {"limits": {"max_jobs": 8, "max_gpus": 16, "max_group_gpus": 4},
               "cells": [{"cell": ..., "tree": sha, "stages": [stage, ...]}, ...]}   (list order = priority)
     stage   {"name", "script", "res", "time", "gpus", "exports", optional "gate": {"pid_gone": N} | {"file": path},
              optional "tree" (overrides the cell's tree for this stage)}
     cell "hold": true keeps a cell out of submission (its queued job, if any, is still tracked)
  state.json  {cell: {"done": [stage names], "active": {"name", "id"} | null, "blocked": reason | null}}
  PAUSE       if present, nothing is submitted (tracking continues)
  keeper.log  one line per event: SUBMIT / COMPLETED / FAILED / CELL_DONE / SSH_FAIL / ERROR
"""
import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

RUNS = "/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs"
DEPT = "--account=fsu-compsci-dept --qos=fsu-compsci-dept"
GROUP = "--account=yd24f.fsu --qos=yd24f.fsu"
TERMINAL_OK = {"COMPLETED"}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


DRY = False


def log(dir_, line):
    if DRY:
        print("[dry-run, not logged]", line)
        return
    with open(dir_ / "keeper.log", "a") as f:
        f.write(f"{now()} {line}\n")


def ssh(cmd, timeout=90):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "hpg", cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh exit {r.returncode}: {r.stderr.strip()[:300]}")
    return r.stdout


def queue():
    """All of the user's jobs in any state: [(id, name, state, gpus, account)]."""
    out = ssh('squeue -u xc25.fsu -h -O "JobID:16,Name:48,State:16,tres-per-job:40,Account:24"')
    jobs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        jid, name, state, tres, acct = parts[:5]
        gpus = 0
        for item in tres.split(","):
            if item.startswith("gres/gpu"):
                gpus += int(item.rsplit(":", 1)[1])
        jobs.append((jid, name, state, gpus, acct))
    return jobs


def final_state(jid):
    out = ssh(f"sacct -j {int(jid)} -n -X -P -o State").strip().splitlines()
    return out[0].split()[0] if out else ""


def gate_open(gate):
    if not gate:
        return True
    if "pid_gone" in gate:
        return not Path(f"/proc/{int(gate['pid_gone'])}").exists()
    if "file" in gate:
        return Path(gate["file"]).exists()
    return False


def submit(cell, stage, account):
    # A stage may pin its own run tree (e.g. evaluation on a serving-fix tree while training keeps its tree).
    tree_id = stage.get("tree", cell["tree"])
    tree = f"{RUNS}/{tree_id}"
    exports = f"ALL,PROJECT_ROOT={tree}" + (f",{stage['exports']}" if stage.get("exports") else "")
    cmd = (f"cd {shlex.quote(tree)} && [ \"$(cat GIT_COMMIT)\" = {shlex.quote(tree_id)} ] && "
           f"sbatch --parsable {account} {stage['res']} --time={stage['time']} --job-name={stage['name']} "
           f"--output=logs/slurm-{stage['name']}-%j.out --export={shlex.quote(exports)} {stage['script']}")
    out = ssh(cmd).strip()
    jid = out.split(";")[0]
    if not jid.isdigit():
        raise RuntimeError(f"unexpected sbatch output {out!r}")
    return jid


def step(dir_, dry):
    plan = json.loads((dir_ / "plan.json").read_text())
    limits = plan["limits"]
    state_path = dir_ / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    jobs = queue()
    live = {j[0]: j for j in jobs}
    # 1. advance cells whose active job left the queue
    for name, st in state.items():
        act = st.get("active")
        if not act or act["id"] in live:
            continue
        fs = final_state(act["id"])
        if not fs:
            continue  # sacct lag; look again next loop
        if fs in TERMINAL_OK:
            st["done"].append(act["name"])
            log(dir_, f"COMPLETED {name} {act['name']} {act['id']}")
        else:
            st["blocked"] = f"{act['name']} {act['id']} {fs}"
            log(dir_, f"FAILED {name} {act['name']} {act['id']} {fs}")
        st["active"] = None
    # 2. fill free slots in priority order
    total = len(jobs)
    gpus = sum(j[3] for j in jobs)
    group_gpus = sum(j[3] for j in jobs if j[4].startswith("yd24f"))
    paused = (dir_ / "PAUSE").exists()
    # Low-priority cells may only use a bounded share, so core cells always find room when they become ready.
    lowprio = {c["cell"] for c in plan["cells"] if c.get("lowprio")}
    live_jobs = {j[0]: j for j in jobs}
    low_jobs = sum(1 for c, v in state.items() if c in lowprio and v.get("active") and v["active"]["id"] in live_jobs)
    low_gpus = sum(live_jobs[v["active"]["id"]][3] for c, v in state.items()
                   if c in lowprio and v.get("active") and v["active"]["id"] in live_jobs)
    for cell in plan["cells"]:
        st = state.setdefault(cell["cell"], {"done": [], "active": None, "blocked": None})
        if st["active"] or st["blocked"] or cell.get("hold"):
            continue
        todo = [s for s in cell["stages"] if s["name"] not in st["done"]]
        if not todo:
            if not st.get("finished"):
                st["finished"] = now()
                log(dir_, f"CELL_DONE {cell['cell']}")
            continue
        stage = todo[0]
        if paused or not gate_open(stage.get("gate")):
            continue
        g = int(stage["gpus"])
        if total + 1 > limits["max_jobs"] or gpus + g > limits["max_gpus"]:
            continue  # a smaller later stage may still fit
        is_low = cell.get("lowprio", False)
        if is_low and (low_jobs + 1 > limits.get("max_lowprio_jobs", 0)
                       or low_gpus + g > limits.get("max_lowprio_gpus", 0)):
            continue
        # CPU stages and GPU stages that fit the group allowance go to the group account (memory headroom on the
        # dept QOS is usually the binding limit); everything else to the department.
        if stage.get("account") == "dept":
            account = DEPT
        elif g == 0 or group_gpus + g <= limits["max_group_gpus"]:
            account = GROUP
        else:
            account = DEPT
        if dry:
            print(f"would submit {cell['cell']} {stage['name']} gpus={g} {account.split()[0]}{' (low)' if is_low else ''}")
            total += 1; gpus += g
            if is_low:
                low_jobs += 1; low_gpus += g
            continue
        jid = submit(cell, stage, account)
        st["active"] = {"name": stage["name"], "id": jid, "submitted": now()}
        total += 1; gpus += g
        if is_low:
            low_jobs += 1; low_gpus += g
        if account == GROUP:
            group_gpus += g
        log(dir_, f"SUBMIT {cell['cell']} {stage['name']} {jid} gpus={g} {account.split()[0][10:]} total={total} gpus_total={gpus}")
    if not dry:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, state_path)
    return total, gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--interval", type=int, default=180)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dir_ = Path(a.dir).resolve()
    global DRY
    DRY = a.dry_run
    lock = open(dir_ / "keeper.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if not a.dry_run else 0))
    failures = 0
    while True:
        try:
            total, gpus = step(dir_, a.dry_run)
            failures = 0
            (dir_ / "heartbeat").write_text(f"{now()} jobs={total} gpus={gpus}\n")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            failures += 1
            if failures in (1, 5) or failures % 20 == 0:
                log(dir_, f"SSH_FAIL x{failures} {str(e)[:200]}")
        except Exception as e:  # noqa: BLE001 - keep the keeper alive, but say so
            log(dir_, f"ERROR {type(e).__name__}: {str(e)[:300]}")
        if a.once or a.dry_run:
            return
        time.sleep(a.interval if failures == 0 else min(900, a.interval * (1 + failures)))


if __name__ == "__main__":
    sys.exit(main())
