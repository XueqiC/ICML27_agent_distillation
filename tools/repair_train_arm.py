"""Continue one E2 arm; --dry-run loads tokenizer and adapter headers only."""
import argparse
from pathlib import Path

from repair.e2_reader import read_collection
from repair.train_arms import run_arm
from sed.config import load_config
from train.dataset import load_tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base_e4b.yaml')
    for name in ('theta0', 'snapshot', 'collection', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--arm', choices=('N', 'F'), required=True)
    parser.add_argument('--repair-splits', default='data/manifest/repair_v01_splits.json')
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD_ROOT=LOCAL_ROOT',
                        help='Relocate evidence roots without changing their hash-bound identities')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    mappings = []
    for value in args.path_map:
        old, sep, new = value.partition('=')
        if not sep or not Path(old).is_absolute() or not Path(new).is_absolute():
            parser.error('--path-map requires two absolute roots separated by =')
        mappings.append((old, new))
    config = load_config(args.config)
    tokenizer = load_tokenizer(args.snapshot)
    collection = read_collection(args.collection, config=config, theta0=args.theta0,
        snapshot=args.snapshot, tokenizer=tokenizer, repair_splits=args.repair_splits, path_maps=mappings)
    run_arm(collection, config, args.output, theta0=args.theta0, snapshot=args.snapshot,
        tokenizer=tokenizer, arm=args.arm, dry_run=args.dry_run,
        inputs=[args.config, args.repair_splits, config['paths']['manifest'], config['paths']['split_freeze']])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
