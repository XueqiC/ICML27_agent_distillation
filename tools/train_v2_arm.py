"""Run one V2 continuation arm (or an explicitly synthetic CPU dry-run)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Also support `python tools/train_v2_arm.py` from an uninstalled worktree.
if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("dataset", "theta0", "snapshot", "output"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--config", default="configs/base_e4b.yaml")
    run.add_argument("--splits", help="Frozen method_v1_splits.json; default: hashed dataset input")
    run.add_argument("--updates", type=int, required=True)
    run.add_argument("--batch-examples", type=int, required=True)
    run.add_argument("--replay-ratio", type=float, required=True)
    run.add_argument("--checkpoints", nargs="+", type=int, required=True)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--gpus", type=int, choices=(1, 2), required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--dry-run-tokenizer", choices=("bytes", "snapshot"), default="bytes",
                     help="Dry-run tokenizer: synthetic bytes (default), or offline snapshot tokenizer with the tiny fake model")
    inventory = sub.add_parser("inputs", help="List local dependencies or verify their byte hashes on CPU")
    inventory.add_argument("--dataset", required=True)
    for name in ("theta0", "snapshot", "splits", "output"):
        inventory.add_argument("--" + name)
    inventory.add_argument("--config", default="configs/base_e4b.yaml")
    mode = inventory.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Print a grouped JSON file inventory with byte sizes")
    mode.add_argument("--check", action="store_true", help="Verify presence and SHA256 only; never load a model")
    for command in (run, inventory):
        command.add_argument("--repair-splits", help="Diagnostic repair split source; still checked against the method freeze hash")
        command.add_argument("--dev140", help="dev140 source; still checked against the method freeze hash")
        command.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW",
                             help="Explicit absolute directory-prefix relocation (repeatable)")
    values = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(values)
    # Match the main grid's torch.distributed.run launcher. Already-launched
    # workers must not recursively spawn; dry-run always stays on one CPU.
    if args.command == "run" and not args.dry_run and args.gpus == 2 and "RANK" not in os.environ:
        return subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node", "2", str(Path(__file__).resolve()), *values], check=False).returncode
    from repair.v2_inputs import ValidationError, Relocation, inspect_inputs, require
    from sed.config import load_config
    try:
        relocation = Relocation(args.dataset, args.relocate)
        config_path = relocation.path(args.config) if args.config else None
        if config_path is not None:
            require(config_path.is_file(), f"Input verification failed: 1 offending path; missing file: {json.dumps(str(config_path))}")
        config = load_config(config_path) if config_path is not None else None
        if args.command == "inputs":
            inspected = inspect_inputs(args.dataset, theta0=args.theta0, relocate=relocation,
                snapshot=args.snapshot, config=config, splits=args.splits,
                repair_splits=args.repair_splits, dev140=args.dev140,
                inputs=[args.config] if args.config else [], check=args.check)
            files = inspected["files"]
            result = files.listing() if args.list else {"schema": "v2-training-inputs/1", "status": "verified",
                "file_count": len(files.files), "relocation": files.report()}
            encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            if args.output:
                output = Path(args.output)
                require(not any(output.resolve().is_relative_to(p) for p in
                    (relocation.dataset, inspected["theta0"], inspected["snapshot"])),
                    "Output must be outside the sealed dataset, theta0 and base snapshot")
                with output.open("x") as handle:
                    handle.write(encoded)
            else:
                print(encoded, end="")
            return 0
        from repair.v2_train import run_arm
        run_arm(args.dataset, config, args.output, theta0=args.theta0, snapshot=args.snapshot,
            updates=args.updates, batch_examples=args.batch_examples, replay_ratio=args.replay_ratio,
            checkpoints=args.checkpoints, seed=args.seed, gpus=args.gpus, dry_run=args.dry_run,
            dry_run_tokenizer=args.dry_run_tokenizer,
            splits=args.splits, repair_splits=args.repair_splits, dev140=args.dev140,
            inputs=[args.config], relocate=relocation)
    except ValidationError as exc:
        print(f"V2 training refused: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Validation messages are local; never expose raw library exceptions.
        print(f"V2 training failed ({type(exc).__name__}); no result marked complete.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
