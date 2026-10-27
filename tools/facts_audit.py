"""CPU-only public-record fact audit; never opens task/game or teacher files."""
import argparse
import glob
import json
from pathlib import Path

from repair.facts import FactTable
from sed.storage import write_json


def audit(paths):
    rows = []
    for path in sorted(set(map(str, paths))):
        row = {'path': path}
        try:
            record = json.loads(Path(path).read_text())
            record = record.get('execution', record)
            if record is None:
                row['skipped'] = 'no_execution'
            else:
                facts = FactTable.from_record(record['initial'], record['steps'])
                checks = facts.inventory_checks
                row.update(goal_parsed=facts.goal['value'] is not None,
                    inventory_observations=len(checks), inventory_agreements=sum(c['agreement'] is True for c in checks),
                    inventory_disagreements=sum(c['disagreement'] for c in checks),
                    inventory_unknown=sum(c['agreement'] is None for c in checks),
                    inventory_checks=checks,
                    refused_takes_while_holding=len(facts.refused_takes_while_holding),
                    cleared_receptacles=len(facts.cleared_receptacles()), exception=None)
        except Exception as exc:
            row['exception'] = type(exc).__name__  # Never exception text / arbitrary fields.
        rows.append(row)
    valid = [r for r in rows if 'goal_parsed' in r]
    totals = {key: sum(r[key] for r in valid) for key in
              ('inventory_observations', 'inventory_agreements', 'inventory_disagreements',
               'inventory_unknown', 'refused_takes_while_holding', 'cleared_receptacles')}
    return {'schema_version': 1, 'files': len(rows), 'audited': len(valid),
            'exceptions': sum(bool(r.get('exception')) for r in rows),
            'goals_parsed': sum(r['goal_parsed'] for r in valid),
            'goal_parse_coverage': sum(r['goal_parsed'] for r in valid) / len(valid) if valid else None,
            **totals, 'episodes': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', '--glob', nargs='+', required=True, help='Quoted glob(s) or episode/record paths')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    paths = [p for pattern in args.episodes for p in glob.glob(pattern, recursive=True) if Path(p).is_file()]
    report = audit(paths)
    report['patterns'] = args.episodes
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}))
    return int(bool(report['exceptions']) or not paths)


if __name__ == '__main__':
    raise SystemExit(main())
