"""Collect E1 once, or finalize retained E2 evidence after public human review."""
import argparse
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

from sed.config import load_config
from sed.manifest import load_frozen, verify_task_files
from sed.seed_evidence import load_verified_collection, checked_episodes
from sed.storage import digest, file_hash, write_json
from repair.artifacts import (read_json, checked_document, write_sealed, checkpoint_files,
                              provenance, tokenizer_identity, run_manifest)
from repair.export import seed_references, write_training_export
from repair.collect import collect, finalize_cases, review_reader
from repair.execute import execute_actions
from repair.runtime import make_teachers, Limits, OUTPUT_CAPS, resolve_output_caps
from repair.align import canonical_action
from repair.branches import (BRANCHES, branch_fields, collection_branch, base_checkpoint,
                             validate_coverage_branch)
from tools.repair_coverage import checked_repair_splits, coverage_tasks
from tools.repair_determinism_check import _reference_checks


def load_failures(path, config, splits, manifest, freeze, env_factory, output, *, canary=False, branch=None):
    frozen = read_json(path)
    rows = frozen.get('failures', frozen.get('failure_list'))
    if rows is None or digest(rows) != frozen.get('list_sha256', frozen.get('failure_list_sha256')):
        raise ValueError('Coverage failure-list hash differs')
    if not 2 <= len(rows) <= 20:
        raise ValueError('Need 2..20 frozen failures')
    summary = read_json(Path(path).parent / 'summary.json')
    if (summary['failure_list'] != rows or summary['failure_list_sha256'] != digest(rows)
            or not summary['branch_decision_actionable']):
        raise ValueError('Failure list differs from actionable coverage summary')
    ids_path = Path(output) / 'failure_task_ids.json'
    # Preflight artifacts are separate from the immutable collector directory.
    write_json(ids_path, [r['task_id'] for r in rows], exclusive=True)
    tasks = coverage_tasks(ids_path, manifest, freeze, repair_splits=splits)
    from repair.ledger import BudgetLedger
    ledger = BudgetLedger(Path(output) / 'preflight_budget.jsonl', context=branch_fields(branch) if branch else None)
    failures = []
    for row, task in zip(rows, tasks):
        directory = (Path(path).parent / row['episode_dir']).resolve()
        if not directory.is_relative_to(Path(path).parent.resolve()):
            raise ValueError('Coverage episode escapes its run directory')
        for name, key in [('episode.json', 'episode_sha256'), ('events.jsonl', 'events_sha256'),
                          ('provenance.json', 'provenance_sha256')]:
            if file_hash(directory / name) != row[key]:
                raise ValueError('Coverage episode evidence changed')
        episode = read_json(directory / 'episode.json')
        if (episode['task_id'] != task['task_id'] or episode['success'] is not False or
                not episode['complete'] or episode['status'] != 'ok' or
                episode['seed'] != config['environment']['seed'] or
                episode['max_episode_steps'] != config['environment']['max_episode_steps'] or
                episode['game_sha256'] != task['game_sha256']):
            raise ValueError('Frozen failure identity/outcome/config differs')
        verify_task_files(config['environment']['data_root'], task)
        if canary and failures:
            continue
        replay = execute_actions(env_factory, task['task_id'], [s['action_raw'] for s in episode['steps']],
                                 episode['max_episode_steps'], ledger=ledger)
        write_json(Path(output) / (digest(task['task_id']) + '.json'), replay.to_json(), exclusive=True)
        if not _reference_checks(replay, episode)['passed'] or replay.success is not False:
            raise ValueError('Coverage failure does not replay exactly')
        # Retain original rollout accounting and policy terminal semantics.
        # Context-overflow failures cannot silently become ordinary short
        # action-list failures; M0b-2 will retain/exclude that nonreplayable
        # terminal outcome before requesting any repair labels.
        replay = replace(replay, token_costs={k: episode.get('metrics', {}).get(k)
            for k in ('prompt_tokens', 'completion_tokens', 'thinking_tokens', 'thinking_chars')})
        if episode.get('termination_reason') == 'context_overflow':
            replay = replace(replay, termination_reason='CONTEXT_LIMIT', error_type='context_overflow')
        failures.append(replay)
    return failures, tasks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch', choices=BRANCHES, required=True)
    p.add_argument('--config', default='configs/base_e4b.yaml')
    p.add_argument('--failures', help='coverage failures.json (or summary.json)')
    p.add_argument('--repair-splits', default='data/manifest/repair_v01_splits.json')
    p.add_argument('--snapshot', help='pinned local base snapshot: tokenizer and P0_K0 identity')
    p.add_argument('--theta0', help='theta0 adapter, or base snapshot for P0_K0 (defaults to --snapshot)')
    p.add_argument('--verified', help='original K trusted teacher collection')
    p.add_argument('--data-root')
    p.add_argument('--output', required=True)
    p.add_argument('--reviews', help='JSON map: public evidence SHA256 -> explicit review decisions')
    p.add_argument('--finalize', help='existing collection.json; offline, no teachers or environments')
    p.add_argument('--max-teacher-input-tokens', type=int, default=5_000_000)
    p.add_argument('--max-teacher-output-tokens', type=int, default=500_000)
    for kind, cap in OUTPUT_CAPS.items():
        p.add_argument(f'--{kind}-max-output-tokens', type=int, default=cap,
                       help=f'{kind} per-call output cap, including reasoning (default: {cap})')
    p.add_argument('--max-failure-output-tokens', type=int, default=64000,
                   help='per-failure output ceiling, including reasoning (default: 64000)')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--canary', action='store_true', help='one diagnosis HTTP attempt, input cap 16384/output cap 8192; no proposals/retry')
    args = p.parse_args(argv)
    try:
        output_caps = resolve_output_caps({kind: getattr(args, kind + '_max_output_tokens') for kind in OUTPUT_CAPS})
        limits = Limits(output_tokens=args.max_failure_output_tokens)
    except ValueError as exc:
        p.error(str(exc))
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        p.error('Output exists; choose a new run directory')
    config = load_config(args.config)
    reviewer = review_reader(read_json(args.reviews)) if args.reviews else None
    if args.finalize:
        if args.dry_run or args.canary or not args.reviews:
            p.error('--finalize requires --reviews and excludes execution modes')
        source = checked_document(args.finalize)
        if collection_branch(source)['branch'] != args.branch:
            p.error('--branch differs from finalized collection')
        if source['canary']:
            p.error('Cannot finalize a canary')
        pairs, exclusions = finalize_cases(source['cases'], source['split'], reviewer=reviewer,
                                           seed_actions=source['seed_actions'])
        for row in pairs + exclusions:
            row.update(collection_branch(source))
        source.pop('content_sha256')
        source.setdefault('collection_exclusions', source['exclusions'])
        source.update(pairs=pairs, D_repair_train=[r['task_id'] for r in pairs], exclusions=exclusions,
                      review_source_sha256=file_hash(args.reviews), parent_collection_sha256=file_hash(args.finalize))
        source['shortfall']['train_pairs'] = 4 - len(pairs)
        output.mkdir(parents=True)
        prov = provenance(output, config, command='repair_finalize', inputs=[args.finalize, args.reviews],
                          details=collection_branch(source))
        write_sealed(output / 'collection.json', source)
        write_json(output / 'exclusions.json', exclusions, exclusive=True)
        export = write_training_export(output)
        write_json(output / 'run_manifest.json', run_manifest(prov, checkpoint=source['checkpoint'],
            tokenizer=source['tokenizer'], tasks=source['tasks'], seeds={'repair_probe': source['split']['seed']},
            parent_collection_sha256=file_hash(args.finalize), review_source_sha256=file_hash(args.reviews),
            **collection_branch(source), training_exposure=0, teacher_calls=0, environment_steps=0,
            ready_for_training=export['ready_for_training']), exclusive=True)
        return 0
    if args.dry_run:
        from repair.dry_fakes import FakeEnvironment, FakeTeacher, FakeTokenizer, fake_review, O
        tasks = [{'task_id': f'train/synthetic-{i}/trial', 'group_id': f'train/synthetic-{i}',
                  'split': 'train', 'sealed': False, 'instruction': 'Put a clean apple on table 1.',
                  'task_type': 'pick_clean_then_place_in_recep', 'parent_seed_ids': [f'fake-seed-{i}']}
                 for i in range(6)]
        failures = [execute_actions(FakeEnvironment, t['task_id'], ['Thought: Use the visible information.\nAction: ' + a for a in O], 40) for t in tasks]
        result = collect(failures, tasks, config, output, env_factory=FakeEnvironment,
            teacher_factory=lambda *a, **k: FakeTeacher(), tokenizer=FakeTokenizer(), tokenizer_id='synthetic',
            checkpoint={'name': 'SYNTHETIC_ONLY'}, reviewer=fake_review, K=6, dry_run=True, branch=args.branch,
            output_caps=output_caps, limits=limits,
            max_teacher_input_tokens=args.max_teacher_input_tokens,
            max_teacher_output_tokens=args.max_teacher_output_tokens)
    else:
        if args.branch == 'P0_K0' and not args.theta0:
            args.theta0 = args.snapshot
        if not all((args.failures, args.theta0, args.snapshot, args.verified)):
            p.error('Real collection requires --failures, --theta0, --snapshot and --verified')
        if args.data_root:
            config['environment']['data_root'] = args.data_root
        from harness.alfworld_env import Environment
        from harness.lora_adapters import inspect_adapter
        from train.dataset import load_tokenizer
        if args.branch == 'P0_K0':
            checkpoint = base_checkpoint(args.theta0, config['models']['student'])
            if checkpoint != base_checkpoint(args.snapshot, config['models']['student']):
                raise ValueError('Tokenizer snapshot differs from base theta0')
        else:
            identity = inspect_adapter('e4b-tsft-k20-e32-lr5e-5', args.theta0, config['models']['student'], args.snapshot)
            checkpoint = {'identity': identity, 'files': checkpoint_files(args.theta0)}
        manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
        splits = checked_repair_splits(args.repair_splits, manifest, freeze)
        if args.branch == 'P1_U_mech':
            from repair.support import load_original_support
            seed_collection, root, seeds = load_original_support(args.verified, manifest, freeze,
                                                                 adapter=args.theta0, splits=splits)
        else:
            seed_collection, _, root = load_verified_collection(args.verified, manifest, freeze)
            seeds = list(checked_episodes(root, seed_collection['verified']))
        coverage_root = Path(args.failures).parent
        validate_coverage_branch(args.branch, read_json(coverage_root / 'summary.json'),
            read_json(coverage_root / 'provenance.json'), read_json(coverage_root / 'plan.json'),
            seed_collection, splits, manifest, freeze, checkpoint, snapshot=args.snapshot)
        seed_refs = seed_references(args.verified, seed_collection, root, seeds)
        seed_actions = {}
        for ep in seeds:
            seed_actions.setdefault(ep['task_id'], []).append([canonical_action(s['action_parsed']) for s in ep['steps']])
        env_factory = partial(Environment, config['environment']['data_root'], split='train',
            seed=config['environment']['seed'], max_episode_steps=config['environment']['max_episode_steps'])
        preflight = output.with_name(output.name + '_preflight')
        preflight.mkdir(parents=True, exist_ok=False)
        provenance(preflight, config, command='repair_collect_preflight',
            details=branch_fields(args.branch),
            inputs=[args.config, args.failures, args.repair_splits,
                    config['paths']['manifest'], config['paths']['split_freeze']])
        failures, tasks = load_failures(args.failures, config, splits, manifest, freeze, env_factory,
                                        preflight, canary=args.canary, branch=args.branch)
        if args.branch != 'P0_K20':
            # Frozen manifests contain identities, not task text. Use only the
            # replay-checked public reset observation for the teacher's task field.
            for failure, task in zip(failures, tasks):
                task['instruction'] = failure.initial['visible']['observation']
        for t in tasks:
            t['parent_seed_ids'] = [r.get('episode_id') or 'seed_receipt_' + digest(r) for r in seed_collection['verified']
                                    if args.branch == 'P1_U_mech' or r['task_id'] == t['task_id']]
        failures = [replace(f, student_checkpoint=digest(checkpoint['files'])) for f in failures]
        tokenizer = load_tokenizer(args.snapshot)
        factory = make_teachers
        if args.canary:
            config = deepcopy(config)
            config['models']['teacher']['max_attempts'] = 1
            factory = partial(make_teachers, single_diagnosis=True)
        result = collect(failures, tasks, config, output, env_factory=env_factory,
            teacher_factory=factory, tokenizer=tokenizer, tokenizer_id=tokenizer_identity(args.snapshot, tokenizer)['id'],
            tokenizer_metadata=tokenizer_identity(args.snapshot, tokenizer),
            checkpoint=checkpoint, reviewer=reviewer, seed_actions=seed_actions, K=seed_collection['K'],
            seed_demonstrations=seed_refs, branch=args.branch,
            inputs=[args.config, args.failures, args.repair_splits, Path(args.verified) / 'manifest.json',
                    coverage_root / 'summary.json', coverage_root / 'provenance.json', coverage_root / 'plan.json',
                    config['paths']['manifest'], config['paths']['split_freeze'], *sorted(preflight.glob('*.json')),
                    preflight / 'preflight_budget.jsonl'],
            canary=args.canary, output_caps=output_caps, limits=limits,
            max_teacher_input_tokens=args.max_teacher_input_tokens,
            max_teacher_output_tokens=args.max_teacher_output_tokens)
    print(f"{result['status']}: {len(result['pairs'])} pairs; output={output}")
    if args.canary:
        return int(result['status'] != 'canary_complete')
    return int(result['status'] in {'case_error', 'canary_failed'})


if __name__ == '__main__':
    raise SystemExit(main())
