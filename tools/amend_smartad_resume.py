"""Offline classification/amendment of the registered SmartAD r2 cell; no launch."""
import argparse
import json
from pathlib import Path

from sed.storage import locked
from teachers import smartad_operator_interrupt, smartad_resume
from teachers.smartad_resume import display


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--amendment", default=smartad_resume.AMENDMENT,
                        choices=[smartad_resume.AMENDMENT, smartad_operator_interrupt.AMENDMENT])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--details", action="store_true", help="Include per-candidate diagnosis and hashes")
    args = parser.parse_args(argv)
    if not args.resume:
        parser.error("The in-cell amendment requires --resume")
    amend = (smartad_resume if args.amendment == smartad_resume.AMENDMENT
             else smartad_operator_interrupt).amend
    result = amend(args.output, resume=True, dry_run=True)
    if args.dry_run:
        pass
    else:
        if not result["ready"]:
            parser.error("Amendment refused; inspect --resume --dry-run --details")
        with locked(Path(args.output) / ".pipeline.lock"):
            result = amend(args.output, resume=True)
    print(json.dumps(result if args.details else display(result), indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
