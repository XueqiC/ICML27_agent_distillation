"""Rebuild ONLY the B5 control after the 11-06 rsi reboot (prereg §12 deviation record).

Mirrors tools/p1_surgery.py `controls` + surgery.conditions.build_controls for setting B,
skipping the already-frozen B0-B3 and calling the unchanged automatic_finetune + freeze.
Deltas and selection rollouts go to a fresh work dir (--output); the frozen condition is
published to $B5_CONDITIONS/B5.json, next to B0-B3. Run via rsi_p1_b5_rebuild.sh (cwd = tree).
"""
import os
import sys
from pathlib import Path

T = Path('/home/xueqi/hq/projects/ICML27_run_repair_eval_d082a6a')
sys.path[:0] = [str(T / 'tools'), str(T), str(T / 'src')]

import p1_surgery as cli
from surgery.common import Relocation, output_guard, checked
from surgery.pairs import load_pairs
from surgery.agent import load_submission
from surgery.runtime import identity_key
from surgery.conditions import automatic_finetune, freeze, restore

conditions_dir = Path(os.environ['B5_CONDITIONS']).resolve()
target = conditions_dir / 'B5.json'
if target.exists():
    raise SystemExit(f'{target} already exists')

args = cli.parser().parse_args(['controls'] + sys.argv[1:])
args.relocation = Relocation(args.relocate)
args.output = output_guard(args.output, protected=[args.relocation.path(args.round_dir),
    args.relocation.path(args.verified), args.relocation.path(args.base), args.relocation.path(args.adapter),
    conditions_dir])
if args.alfworld_data:
    args.alfworld_data = str(args.relocation.path(args.alfworld_data))
if args.trial or len(args.submission) != 1:
    raise SystemExit('official run with exactly one --submission required')
pairs = load_pairs(args.pairs, args.relocation)
if pairs['setting'] != 'B':
    raise SystemExit('B5 rebuild is for setting B only')

# == tools/p1_surgery.py controls branch
student = cli.runtime(args, pairs)
session = cli.diagnosis_session(args, pairs, student, args.output / 'selection')
def rollout(pair, edits):
    return session.rollout(session.state(pair['pair_id']), edits)
deployment = load_submission(args.relocation.path(args.submission[0]), pairs['sha256'])

# == build_controls preamble (unchanged checks), then only the B5 step
checked(deployment, 'p1-submission/1')
if (deployment['pairs_sha256'] != pairs['sha256'] or deployment['setting'] != pairs['setting']
        or identity_key(student.identity) != identity_key(deployment['student_identity'])
        or deployment['kind'] != 'global_b'):
    raise ValueError('Conditions require the matching frozen deployed submission')
diagnosis = [p for p in pairs['pairs'] if p['split'] == 'diagnosis']
restore(student, dict(vectors=deployment['vectors'], edits=deployment['submission']['edit']))
edits, selection = automatic_finetune(student, diagnosis, rollout, args.output / 'deltas')
value = freeze(target, student, pairs['sha256'], 'B', 'B5', edits, selection=selection, trial=False)
print('B5 frozen', target, value.get('sha256') if isinstance(value, dict) else None,
      'chosen', selection['chosen'], flush=True)
