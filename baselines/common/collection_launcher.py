"""Explicit teacher cells, read-only plans, and CPU shard orchestration."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

from baselines.common.collection_views import bind_views, parse_views, read, validate_shard
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import safe_config
from sed.seed_evidence import load_verified_collection
from sed.storage import digest, locked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("adftp", "smartad"), required=True)
    parser.add_argument("--cell", action="append", required=True, help="Name; repeat with corresponding --source-view")
    parser.add_argument("--source-view", action="append", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--shards", type=int)
    parser.add_argument("--subset-view", action="append", default=[], metavar="K=PATH")
    parser.add_argument("--config", default=os.environ.get("COLLECT_CONFIG", "configs/base.yaml"))
    parser.add_argument("--rate-limit-dir")
    parser.add_argument("--log-root", default=os.environ.get("LOG_ROOT", "logs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    method = args.method.upper()
    args.output_root = args.output_root or os.environ.get(method + "_ROOT", f"reports/runs/{args.method}-collections")
    args.shards = args.shards if args.shards is not None else int(os.environ.get(method + "_SHARDS", "32"))
    # Every new collector uses the same limiter; the explicit shared environment
    # overrides flags exactly as RateLimiter itself does.
    limiter = os.environ.get("TEACHER_RATE_LIMIT_DIR") or args.rate_limit_dir or "data/teacher_rate_limit"
    try:
        validate_shard(0, args.shards)
        parse_views(args.subset_view)
        if (len(args.cell) != len(args.source_view) or len(set(args.cell)) != len(args.cell)
                or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", c) for c in args.cell)):
            raise ValueError("Supply distinct safe --cell names and one --source-view per cell, in order")
        config = load_config(args.config)
        manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
        plans = []
        for cell, path in zip(args.cell, args.source_view):
            source, selection, evidence = load_verified_collection(path, manifest, freeze)
            bindings = bind_views(path, source, selection, manifest, freeze, args.subset_view)
            output = Path(args.output_root).resolve() / cell
            for directory in (output, Path(limiter).resolve()):
                if any(directory.is_relative_to(p) for p in (evidence, Path(config["paths"]["raw"]).resolve(),
                                                            Path(config["paths"]["verified"]).resolve())):
                    raise ValueError("Output/limiter must be outside source storage")
            if output.exists():
                if not args.resume or not (output / "identity.json").is_file():
                    raise ValueError("Output exists; use --resume with its frozen identity")
                old = read(output / "identity.json")
                expected = {"source_verified": str(Path(path).resolve()), "source_collection_sha256": digest(source),
                            "selection": selection, "config": safe_config(config), "method": args.method,
                            "rate_limit_dir": str(Path(limiter).resolve())}
                if any(old.get(k) != v for k, v in expected.items()) or old.get("subset_views", []) != bindings:
                    raise ValueError("Existing cell identity/source/config/limiter/views mismatch")
            commands = []
            for index in range(args.shards):
                command = [sys.executable, "-m", f"baselines.{args.method}.pipeline", "--config", args.config,
                    "--verified", str(Path(path).resolve()), "--output", str(output), "--through", "collect",
                    "--rate-limit-dir", str(Path(limiter).resolve()), "--shard-index", str(index),
                    "--shard-count", str(args.shards)]
                if args.method == "smartad":
                    command.append("--resume")
                for entry in args.subset_view:
                    command.extend(("--subset-view", entry))
                commands.append(command)
            plans.append({"cell": cell, "source_view": str(Path(path).resolve()), "output": str(output),
                          "K": source["K"], "unique_instances": source["unique_instances"],
                          "task_groups": source["task_groups"], "seed": selection["seed"],
                          "selection_rule": selection["selection_rule"], "shards": args.shards,
                          "subsets": bindings, "commands": commands})
    except (ValueError, OSError, KeyError) as error:
        # Never dump provider exceptions or process environment.
        parser.error(str(error))
    print(json.dumps({"method": args.method, "dry_run": args.dry_run,
                      "rate_limit_dir": str(Path(limiter).resolve()), "cells": plans}, indent=2), flush=True)
    if args.dry_run:
        return
    processes = []
    def stop(*_):
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise SystemExit(143)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    # Different output roots may collect concurrently with one shared limiter.
    with locked(Path(args.output_root) / f".{args.method}.launcher.lock"):
        def launch(plan):
            for index, command in enumerate(plan["commands"]):
                with tempfile.NamedTemporaryFile(prefix=f"collect_{args.method}_{plan['cell']}_shard{index}.",
                                                 suffix=".log", dir=log_root, delete=False) as log:
                    processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TEACHER_RATE_LIMIT_DIR": str(Path(limiter).resolve())}))
        for plan in plans:
            launch(plan)
        if any([p.wait() != 0 for p in processes]):
            raise SystemExit("Collection shard failed; inspect retained shard logs before resuming")
        if args.method == "smartad":
            processes.clear()
            for plan in plans:
                if not (Path(plan["output"]) / "candidates.json").exists():
                    launch(plan)  # Existing deferred candidate-01 protocol.
            if any([p.wait() != 0 for p in processes]):
                raise SystemExit("Deferred candidate shard failed; inspect retained logs")
        for plan in plans:
            path = Path(plan["output"]) / ("candidates.json" if args.method == "smartad" else "collection/manifest.json")
            if not path.exists() or read(path)["status"] != "complete":
                raise SystemExit("Collection incomplete; resume durable slots")


if __name__ == "__main__":
    main()
