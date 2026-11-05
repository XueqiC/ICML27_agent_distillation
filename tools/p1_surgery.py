#!/usr/bin/env python
"""P1 preregistered surgery: offline preparation, explicit teacher gate, frozen evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from repair.p0_pairs import DEFAULT_BASE, DEFAULT_ADAPTER, DEFAULT_ROUND, CATEGORIES
from sed.storage import write_json, file_hash
from surgery.common import Relocation, output_guard, read, checked, provenance
from surgery.pairs import DEFAULT_PAIRS, DEFAULT_VERIFIED, extract_a, extract_b, load_pairs, verified_records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['split-a', 'extract-b', 'diagnose', 'integrate', 'controls',
                                     'freeze-e4', 'evaluate', 'export', 'verify-export'])
    p.add_argument('--pairs', default=DEFAULT_PAIRS)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--round-dir', default=DEFAULT_ROUND)
    p.add_argument('--manifest-path', help='Override frozen synthetic manifest path; bytes must match')
    p.add_argument('--base', '--snapshot', default=DEFAULT_BASE)
    p.add_argument('--adapter', default=DEFAULT_ADAPTER)
    p.add_argument('--e4-checkpoint')
    p.add_argument('--verified', default=DEFAULT_VERIFIED)
    p.add_argument('--alfworld-data', default=os.environ.get('ALFWORLD_DATA_ROOT'))
    p.add_argument('--relocate', action='append', default=[], metavar='OLD=NEW')
    p.add_argument('--category', choices=CATEGORIES)
    p.add_argument('--submission', action='append', default=[])
    p.add_argument('--conditions', help='Directory containing all frozen condition JSONs')
    p.add_argument('--condition', help='One frozen condition JSON, for export')
    p.add_argument('--allow-teacher-api', action='store_true')
    p.add_argument('--config', default='configs/base_e4b.yaml')
    p.add_argument('--rate-limit-dir', help='Joint limiter directory shared by all concurrent P1 sessions')
    p.add_argument('--trial', action='store_true', help='Excluded §7 trial; condition/evaluation artifacts are marked')
    p.add_argument('--trial-limit', type=int, help='Trial-only first N evaluation units per metric')
    for name in ('tool-calls', 'rollouts', 'candidate-edits', 'environment-steps', 'input-token-cap', 'output-token-cap'):
        p.add_argument('--' + name, type=int)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--merge', action='store_true', help='CPU-only verification/merge of all evaluation shards')
    p.add_argument('--m4-result', action='append', default=[], metavar='CONDITION=JOB_DIR',
                   help='evaluate --merge: import verified standalone dev140 suite results')
    return p


def runtime(args, pairs=None, *, condition=None, base_only=False, e4=False):
    from surgery.runtime import load_runtime
    setting = 'B' if base_only else pairs['setting']
    if condition:
        e4 = condition['name'] == 'E4'
    adapter = None if setting == 'B' else (args.e4_checkpoint if e4 else args.adapter)
    expected = (condition['student_identity'].get('adapter') if condition else
                None if base_only or e4 or setting == 'B' else pairs['student'])
    serving = pairs['serving'] if pairs else dict(max_tokens=1024, max_model_len=32768)
    student = load_runtime(args.relocation.path(args.base), args.relocation.path(adapter) if adapter else None,
        expected, e4=e4, max_new_tokens=serving['max_tokens'], max_model_len=serving['max_model_len'])
    student.relocation = args.relocation
    return student


def diagnosis_session(args, pairs, student, directory, *, category=None):
    from surgery.evaluation import Manifest, environment_factory
    from surgery.session import Session
    selected = [p for p in pairs['pairs'] if p['split'] == 'diagnosis' and (category is None or p['category'] == category)]
    manifest = None
    if pairs['setting'] == 'A':
        manifest = Manifest(args.manifest_path or pairs['manifest_path'], args.relocation)
        if manifest.sha256 != pairs['manifest_sha256']:
            raise ValueError('Diagnosis manifest changed')
    return Session(student, selected, directory, env_factory=environment_factory(pairs,
        manifest=manifest, data_root=args.alfworld_data))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.relocation = Relocation(args.relocate)
    args.output = output_guard(args.output, protected=[args.relocation.path(args.round_dir),
        args.relocation.path(args.verified), args.relocation.path(args.base), args.relocation.path(args.adapter),
        *([args.relocation.path(args.e4_checkpoint)] if args.e4_checkpoint else [])])
    if args.alfworld_data:
        args.alfworld_data = str(args.relocation.path(args.alfworld_data))
    overrides = {k: getattr(args, k) for k in ('tool_calls', 'rollouts', 'candidate_edits', 'environment_steps') if getattr(args, k) is not None}
    if not args.trial and (overrides or args.trial_limit is not None or args.input_token_cap is not None or args.output_token_cap is not None):
        p.error('Reduced budgets/metric limits require --trial; official preregistration remains fixed')
    if args.command in ('diagnose', 'integrate'):
        if not args.allow_teacher_api:
            p.error('Teacher sessions require --allow-teacher-api')
        if args.output.exists():
            p.error('Teacher session requires a fresh output directory')
    if args.command == 'split-a':
        result = extract_a(args.pairs, args.relocation)
        write_json(args.output / 'pairs.json', result, exclusive=True)
        print(json.dumps(dict(counts=result['counts'], split_sha256=result['split_sha256'], sha256=result['sha256'])))
        return result
    if args.command == 'verify-export':
        from surgery.export import verify_export
        result = verify_export(args.output)
        print(json.dumps(dict(passed=True, sha256=result['sha256'])))
        return result
    if args.command == 'extract-b':
        from sed.storage import JsonlLedger
        records, inputs = verified_records(args.verified, args.relocation)
        student = runtime(args, base_only=True)
        write_json(args.output / 'provenance.json', provenance(dict(command='extract-b', inputs=inputs,
            student=student.identity, verified=str(args.relocation.path(args.verified)))), exclusive=True)
        result = extract_b(student, records, inputs=inputs, ledger=JsonlLedger(args.output / 'extraction.jsonl'))
        write_json(args.output / 'pairs.json', result, exclusive=True)
        print(json.dumps(dict(counts=result['counts'], matched=len(result['matched_pool']), sha256=result['sha256'])))
        return result
    pairs = load_pairs(args.pairs, args.relocation)
    from surgery.conditions import load_condition
    if args.command in ('diagnose', 'integrate'):
        from surgery.agent import Agent, Budget, make_transport, load_submission
        from sed.config import load_config
        kind = 'integration' if args.command == 'integrate' else 'category' if pairs['setting'] == 'A' else 'global_b'
        if kind == 'category' and not args.category:
            p.error('Setting A diagnose requires --category')
        if kind == 'integration' and pairs['setting'] != 'A':
            p.error('Integration is setting A only')
        student = runtime(args, pairs)
        session = diagnosis_session(args, pairs, student, args.output / 'tools', category=args.category if kind == 'category' else None)
        submissions = [load_submission(args.relocation.path(s), pairs['sha256']) for s in args.submission]
        config = load_config(args.config)
        # Existing rate_limit_options honors this env directory; all workers
        # receive one joint limiter while each session retains its own token cap.
        os.environ['TEACHER_RATE_LIMIT_DIR'] = str(Path(args.rate_limit_dir or args.output.parent / 'joint_teacher_rate_limit').resolve())
        transport = make_transport(config, args.category or kind, args.output, kind=kind,
            input_cap=args.input_token_cap, output_cap=args.output_token_cap)
        result = Agent(session, transport, args.output, category=args.category or kind,
            pairs_sha256=pairs['sha256'], setting=pairs['setting'], kind=kind,
            budget=Budget(kind, **overrides), submissions=submissions).run()
    elif args.command == 'controls':
        from surgery.conditions import build_controls
        from surgery.agent import load_submission
        if len(args.submission) != 1:
            p.error('controls needs one deployed --submission')
        student = runtime(args, pairs)
        session = diagnosis_session(args, pairs, student, args.output / 'selection')
        def rollout(pair, edits):
            return session.rollout(session.state(pair['pair_id']), edits)
        result = build_controls(student, pairs, load_submission(args.relocation.path(args.submission[0]), pairs['sha256']),
            args.output, rollout, trial=args.trial)
    elif args.command == 'freeze-e4':
        from surgery.conditions import freeze
        if pairs['setting'] != 'A' or not args.e4_checkpoint:
            p.error('E4 requires setting A and --e4-checkpoint')
        student = runtime(args, pairs, e4=True)
        exposure = student.identity['adapter']['training']['training_exposure']
        if any(p['task_id'] in exposure['task_ids'] or p['group_id'] in exposure['group_ids']
               for p in pairs['pairs'] if p['split'] == 'test'):
            raise ValueError('E4 training exposure overlaps test pairs')
        result = freeze(args.output / 'E4.json', student, pairs['sha256'], 'A', 'E4', [],
            selection=dict(source='separately trained E4 checkpoint; recorded V0 identity'), trial=args.trial)
    elif args.command == 'evaluate':
        from surgery.evaluation import make_plan, evaluate_shard, merge, Runtime, parse_shard, import_m4
        if not args.conditions:
            p.error('evaluate requires --conditions')
        paths = sorted(args.relocation.path(args.conditions).glob('[EB]*.json'))
        conditions = [load_condition(path, pairs['sha256'], relocation=args.relocation) for path in paths]
        if args.merge:
            result = merge(pairs, conditions, args.output, m4=import_m4(args.m4_result, conditions, args.relocation))
        else:
            shard = parse_shard(args.shard)
            plan_path = args.output / 'plan.json'
            if plan_path.exists():
                plan = checked(read(plan_path), 'p1-evaluation-plan/1')
                if plan['trial'] != args.trial or plan['pairs_sha256'] != pairs['sha256']:
                    raise ValueError('Existing evaluation plan mismatch')
            else:
                plan = make_plan(pairs, conditions, args.output, round_dir=args.round_dir,
                    manifest_path=args.manifest_path, relocation=args.relocation, data_root=args.alfworld_data,
                    shards=shard[1], trial=args.trial, trial_limit=args.trial_limit)
            class OwnedRuntime(Runtime):
                def close(self):
                    import gc
                    import torch
                    self.student.model.to('cpu')
                    del self.student
                    gc.collect()
                    torch.cuda.empty_cache()
            result = evaluate_shard(pairs, conditions, args.output, plan, shard=shard, resume=args.resume,
                runtime_factory=lambda c, d: OwnedRuntime(runtime(args, pairs, condition=c), pairs, d, plan, args.relocation))
    elif args.command == 'export':
        from surgery.export import export_model
        if not args.condition:
            p.error('export requires --condition')
        condition = load_condition(args.relocation.path(args.condition), pairs['sha256'], relocation=args.relocation)
        student = runtime(args, pairs, condition=condition)
        result = export_model(student, condition, args.relocation.path(args.base), args.output)
    else:
        raise ValueError('Unknown command')
    print(json.dumps(dict(status='complete', output=str(args.output),
        sha256=result.get('sha256') if isinstance(result, dict) else None)))
    return result


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('P1 halted: ' + type(exc).__name__ + '; inspect sanitized receipts and surgery/README.md', file=sys.stderr)
        sys.exit(1)
