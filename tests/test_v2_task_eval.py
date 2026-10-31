"""V2 from-start evaluation: frozen synthetic tasks, fake suites, no model calls."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from data.manifest.build_repair_splits import build_splits
from repair.artifacts import checked_document, read_json, seal
from sed.config import load_config
from sed.manifest import freeze_splits, load_frozen, resolve_task
from sed.storage import canonical_bytes, digest, file_hash, write_json
from tools import v2_task_eval as cli
from test_joint_adapter_eval import trained_adapter
from test_lora_serving import adapter_bank, bound_report
from test_repair_splits import universe


@pytest.fixture
def frozen_inputs(tmp_path, universe):
    manifest, freeze, dev = universe
    splits = seal(build_splits(manifest, freeze, [], dev))
    config = load_config()
    for name, value in [('manifest', manifest), ('split_freeze', freeze)]:
        destination = tmp_path / (name + '.json')
        write_json(destination, value)
        config['paths'][name] = str(destination)
    config['environment']['data_root'] = str(tmp_path / 'fake-data')
    config_path, splits_path = tmp_path / 'config.yaml', tmp_path / 'splits.json'
    config_path.write_text(yaml.safe_dump(config))
    write_json(splits_path, splits)
    return config_path, splits_path, splits


def arguments(frozen_inputs, adapter_args, *, set_name='D_unseen_quick'):
    config_path, splits_path, _ = frozen_inputs
    return ['run', '--set', set_name, '--repair-splits', str(splits_path), '--greedy',
            '--sampled-seeds', '0', '1', '2', '--config', str(config_path),
            '--data-root', str(config_path.parent / 'fake-data'),
            '--endpoint', adapter_args.endpoint, '--model-name', adapter_args.model_name,
            '--server-provenance', adapter_args.server_provenance, '--lora-check', adapter_args.lora_check,
            '--workers', '2', '--output', adapter_args.output]


class FakeSuite:
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}
        self.stop_before = None
        self.stop_inside = None
        self.teacher_row = None

    def __call__(self, *, config, tasks_path, output, policy, spec, workers, inputs):
        assert policy == 'student' and workers == 2
        name = output.name
        if name == self.stop_before:
            raise RuntimeError('Synthetic interruption between decodings')
        output.mkdir(parents=True, exist_ok=False)
        if name == self.stop_inside:
            raise RuntimeError('Synthetic interruption within decoding')
        self.calls.append((name, deepcopy(spec)))
        listing = read_json(tasks_path)
        manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
        write_json(output / 'tasks.json', listing, exclusive=True)
        for i, tid in enumerate(listing['task_ids']):
            resolve_task(manifest, freeze, tid)
            success, status = self.outcomes.get((name, tid), (False, 'ok'))
            directory = output / 'episodes' / f'{i:03d}-{digest(tid)[:12]}'
            write_json(directory / 'episode.json', dict(task_id=tid, success=success,
                status=status, complete=status == 'ok'), exclusive=True)
        if self.teacher_row:
            write_json(output / 'episodes' / 'teacher' / 'teacher_usage.jsonl', self.teacher_row)
        summary = dict(policy=policy, student=spec, task_list_sha256=listing['list_sha256'])
        write_json(output / 'summary.json', summary, exclusive=True)
        return summary


@pytest.fixture
def suite(monkeypatch):
    fake = FakeSuite()
    monkeypatch.setattr(cli, 'run_suite', fake)
    def forbidden(*args, **kwargs):
        pytest.fail('Teacher must never be constructed')
    monkeypatch.setattr('teachers.openai_teacher.Teacher.__init__', forbidden)
    return fake


@pytest.mark.parametrize('set_name,size', [('D_unseen_quick', 24), ('D_probe', 60)])
def test_frozen_listing_adapter_identity_and_decodings(frozen_inputs, trained_adapter, suite, set_name, size):
    args, adapter, _, _ = trained_adapter
    assert cli.main(arguments(frozen_inputs, args, set_name=set_name)) == 0
    output = Path(args.output)
    listing = frozen_inputs[2]['sets'][set_name]
    assert (output / 'tasks.json').read_bytes() == canonical_bytes(listing) + b'\n'
    assert [name for name, _ in suite.calls] == ['greedy', 'sampled-s0', 'sampled-s1', 'sampled-s2']
    base = {k: v for k, v in suite.calls[0][1].items() if k != 'decoding'}
    for seed, (_, spec) in enumerate(suite.calls[1:]):
        assert spec == dict(base, decoding='sampled', sampling_seed=seed)
    plan = checked_document(output / 'plan.json')
    summary = checked_document(output / 'summary.json')
    identity = plan['identity']['evaluated_adapter']
    assert summary['evaluated_adapter'] == identity
    for key in ('name', 'path', 'adapter_weights_sha256', 'adapter_config_sha256', 'base_model', 'revision'):
        assert identity[key] == adapter[key]
    training = identity['training']
    assert identity['arm'] == training['arm'] == 'OFFLINE' and not identity['is_theta0']
    assert training['optimizer_updates'] == 3
    assert training['manifest_sha256'] == file_hash(training['manifest_path'])
    assert training['checkpoint_manifest_sha256'] == file_hash(training['checkpoint_manifest_path'])
    assert training['training_exposure']['group_ids'] == ['train/replay', 'train/unrelated']
    assert summary['set'] == set_name and summary['list_sha256'] == listing['list_sha256']
    assert len(summary['per_task']) == summary['totals']['unique_instances'] == size
    assert summary['totals']['task_groups'] == size
    assert summary['teacher_calls'] == summary['model_updates'] == summary['training_exposure'] == 0
    assert summary['global_budget']['reserved_attempts'] == 0
    assert summary['totals']['unknown_or_error_episodes'] == dict.fromkeys(summary['decodings'], 0)
    assert plan['identity']['leakage_check']['status'] == 'passed'
    assert plan['identity']['teacher_calls_allowed'] == 0 and plan['identity']['from_start']
    assert read_json(output / 'run_manifest.json')['evaluation_only']


@pytest.mark.parametrize('set_name', ['U_mech', 'U_dev', 'V2_pool', 'dev140', 'valid_unseen'])
def test_set_restriction(frozen_inputs, trained_adapter, suite, set_name):
    args = trained_adapter[0]
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(frozen_inputs, args, set_name=set_name))
    assert error.value.code == 2 and not suite.calls and not Path(args.output).exists()


@pytest.mark.parametrize('field', ['content_sha256', 'manifest_sha256', 'split_freeze_sha256',
                                   'list_sha256', 'list_manifest', 'list_freeze', 'count', 'groups', 'subset'])
def test_frozen_split_refusal(frozen_inputs, trained_adapter, suite, field):
    _, path, splits = frozen_inputs
    if field in ('content_sha256', 'manifest_sha256', 'split_freeze_sha256'):
        splits[field] = 'changed'
    else:
        listing = splits['sets']['D_unseen_quick']
        if field == 'list_sha256':
            listing['list_sha256'] = 'changed'
        elif field in ('list_manifest', 'list_freeze'):
            listing['manifest_sha256' if field == 'list_manifest' else 'split_freeze_sha256'] = 'changed'
        elif field == 'count':
            listing['task_ids'].pop()
            listing['list_sha256'] = digest(listing['task_ids'])
        elif field == 'groups':
            listing['group_ids'].reverse()
        else:
            other = splits['sets']['U_mech']
            listing['task_ids'][0], listing['group_ids'][0] = other['task_ids'][0], other['group_ids'][0]
            listing['list_sha256'] = digest(listing['task_ids'])
    if field != 'content_sha256':
        splits = seal({k: v for k, v in splits.items() if k != 'content_sha256'})
    write_json(path, splits)
    args = trained_adapter[0]
    with pytest.raises(ValueError):
        cli.main(arguments(frozen_inputs, args))
    assert not suite.calls and not Path(args.output).exists()


def test_repository_frozen_listings_are_accepted():
    config = load_config()
    manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
    path = 'data/manifest/repair_v01_splits.json'
    original = read_json(path)
    for name, size in cli.SETS.items():
        _, listing, tasks = cli.evaluation_tasks(path, manifest, freeze, name)
        assert listing == original['sets'][name] and len(tasks) == size


def test_cli_with_actual_suite_and_fake_http_environment(frozen_inputs, trained_adapter, monkeypatch):
    from eval.run_suite import run_suite
    from test_suite import fake_sampled_worker
    config_path, splits_path, _ = frozen_inputs
    config = load_config(config_path)
    manifest = read_json(config['paths']['manifest'])
    game = config_path.parent / 'fake-game.json'
    write_json(game, dict(solvable=True, win_at=1))
    for task in manifest['tasks']:
        task.update(files={'game.tw-pddl': file_hash(game)}, game_sha256=file_hash(game))
    freeze = freeze_splits(manifest)
    dev_ids = [tid for tid, row in freeze['assignments'].items() if row['split'] == 'dev']
    dev = dict(task_ids=dev_ids, list_sha256=digest(dev_ids), manifest_sha256=digest(manifest),
               split_freeze_sha256=digest(freeze))
    splits = seal(build_splits(manifest, freeze, [], dev))
    for name, value in [('manifest', manifest), ('split_freeze', freeze)]:
        write_json(config['paths'][name], value)
    write_json(splits_path, splits)
    for tid in splits['sets']['D_unseen_quick']['task_ids']:
        directory = Path(config['environment']['data_root']) / tid
        directory.mkdir(parents=True)
        (directory / 'game.tw-pddl').write_bytes(game.read_bytes())
    calls = []
    def actual_suite(**kwargs):
        calls.append(kwargs['output'].name)
        return run_suite(**kwargs, worker_fn=fake_sampled_worker)
    monkeypatch.setattr(cli, 'run_suite', actual_suite)
    args = trained_adapter[0]
    argv = arguments(frozen_inputs, args)
    assert cli.main(argv) == 0
    summary = checked_document(Path(args.output) / 'summary.json')
    assert summary['totals']['greedy_successes'] == 24
    assert summary['totals']['sampled_success_rate'] == 1
    assert calls == ['greedy', 'sampled-s0', 'sampled-s1', 'sampled-s2']
    assert cli.main([*argv, '--resume']) == 0 and len(calls) == 4


def test_all_unknown_rates_remain_null(frozen_inputs, trained_adapter, suite):
    args = trained_adapter[0]
    suite.outcomes = {(name, tid): ('unknown', 'error')
                     for tid in frozen_inputs[2]['sets']['D_unseen_quick']['task_ids']
                     for name in ['greedy', 'sampled-s0', 'sampled-s1', 'sampled-s2']}
    assert cli.main(arguments(frozen_inputs, args)) == 1
    summary = checked_document(Path(args.output) / 'summary.json')
    assert summary['totals']['sampled_success_rate'] is None
    assert summary['totals']['sampled_valid_tasks'] == 0
    assert all(r['sampled_success_mean'] is None for r in summary['per_task'])
    assert all(r['valid_episodes'] == 0 and r['success_rate'] is None
               for r in summary['totals']['per_decoding'].values())


@pytest.mark.parametrize('exposure', ['task', 'group', 'alternate_trial', 'replay', 'checkpoint'])
def test_leakage_before_output(frozen_inputs, trained_adapter, suite, exposure):
    args, adapter, run, update = trained_adapter
    tid = frozen_inputs[2]['sets']['D_unseen_quick']['task_ids'][0]
    group = str(Path(tid).parent)
    if exposure == 'task':
        run['training_exposure_tasks'] = [tid]
    elif exposure == 'group':
        run['training_exposure_tasks'] = [group]
    elif exposure == 'alternate_trial':
        run['training_exposure_tasks'] = [group + '/trial_other']
    elif exposure == 'replay':
        run['input_manifests'][1]['training_exposure_tasks'] = [group]
    else:
        update['training_exposure_tasks'] = [group]
    checkpoint = Path(adapter['path'])
    write_json(checkpoint.parent.parent / 'manifest.json', run)
    write_json(checkpoint.parent / 'manifest.json', update)
    with pytest.raises(ValueError, match='overlaps adapter training exposure'):
        cli.main(arguments(frozen_inputs, args))
    assert not suite.calls and not Path(args.output).exists()


def test_theta0_is_arm_zero_without_training_manifest(frozen_inputs, trained_adapter, bound_report, suite):
    args = trained_adapter[0]
    report, server = deepcopy(bound_report)
    adapter = server['config']['server']['lora_adapters'][0]
    adapter['name'] = args.model_name = cli.THETA0
    report['model_name'] = cli.THETA0
    report['server_provenance_sha256'] = digest(server)
    write_json(args.server_provenance, server)
    write_json(args.lora_check, report)
    assert not (Path(adapter['path']).parent / 'manifest.json').exists()
    assert cli.main(arguments(frozen_inputs, args)) == 0
    identity = checked_document(Path(args.output) / 'summary.json')['evaluated_adapter']
    assert identity['training'] is None and identity['arm'] == 'theta0'
    assert identity['arm_index'] == 0 and identity['is_theta0']


def test_failed_serving_gate_before_output(frozen_inputs, trained_adapter, suite):
    args = trained_adapter[0]
    report = read_json(args.lora_check)
    write_json(args.lora_check, dict(report, passed=False))
    with pytest.raises(ValueError):
        cli.main(arguments(frozen_inputs, args))
    assert not suite.calls and not Path(args.output).exists()


def test_unknown_error_math_is_task_mean_then_mean(frozen_inputs, trained_adapter, suite):
    args = trained_adapter[0]
    ids = frozen_inputs[2]['sets']['D_unseen_quick']['task_ids']
    suite.outcomes = {
        ('greedy', ids[0]): (True, 'ok'), ('greedy', ids[1]): (True, 'error'),
        ('greedy', ids[2]): ('unknown', 'ok'),
        ('sampled-s0', ids[0]): (True, 'ok'), ('sampled-s2', ids[0]): ('unknown', 'ok'),
        ('sampled-s0', ids[1]): (True, 'ok'), ('sampled-s1', ids[1]): ('unknown', 'ok'),
        ('sampled-s2', ids[1]): (False, 'error'),
        **{(f'sampled-s{s}', ids[2]): ('unknown', 'error') for s in range(3)}}
    assert cli.main(arguments(frozen_inputs, args)) == 1
    summary = checked_document(Path(args.output) / 'summary.json')
    a, b, c = summary['per_task'][:3]
    assert [r['sampled_success_mean'] for r in [a, b, c]] == [.5, 1, None]
    assert b['success']['greedy'] is None and b['success']['sampled-s2'] is None
    totals = summary['totals']
    assert totals['greedy_successes'] == 1 and totals['sampled_valid_tasks'] == 23
    assert totals['sampled_success_rate'] == pytest.approx(1.5 / 23)
    assert totals['unknown_or_error_episodes'] == {'greedy': 2, 'sampled-s0': 1, 'sampled-s1': 2, 'sampled-s2': 3}
    assert totals['per_decoding']['greedy']['success_rate'] == pytest.approx(1 / 22)


def test_non_quick_36_main_comparison(frozen_inputs, trained_adapter, suite):
    args = trained_adapter[0]
    quick = frozen_inputs[2]['sets']['D_unseen_quick']['task_ids']
    probe = frozen_inputs[2]['sets']['D_probe']['task_ids']
    non_quick = [tid for tid in probe if tid not in quick]
    suite.outcomes = {(name, tid): (True, 'ok') for tid in quick
                     for name in ['greedy', 'sampled-s0', 'sampled-s1', 'sampled-s2']}
    suite.outcomes.update({(f'sampled-s{s}', non_quick[0]): ('unknown', 'ok') for s in range(3)})
    assert cli.main(arguments(frozen_inputs, args, set_name='D_probe')) == 1
    summary = checked_document(Path(args.output) / 'summary.json')
    main, selection = summary['subsets']['non_quick_36'], summary['subsets']['quick_24']
    assert main['task_ids'] == non_quick and selection['task_ids'] == quick
    assert main['totals']['unique_instances'] == 36 and selection['totals']['unique_instances'] == 24
    assert main['totals']['sampled_success_rate'] == 0 and main['totals']['sampled_valid_tasks'] == 35
    assert main['totals']['greedy_successes'] == 0
    assert selection['totals']['sampled_success_rate'] == 1 and selection['totals']['greedy_successes'] == 24
    assert summary['totals']['sampled_success_rate'] == pytest.approx(24 / 59)


def test_exclusive_and_resume_only_missing_decodings(frozen_inputs, trained_adapter, suite):
    args = trained_adapter[0]
    argv = arguments(frozen_inputs, args)
    output = Path(args.output)
    suite.stop_before = 'sampled-s1'
    with pytest.raises(RuntimeError):
        cli.main(argv)
    assert [n for n, _ in suite.calls] == ['greedy', 'sampled-s0']
    assert not (output / 'summary.json').exists()
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        cli.main(argv)
    suite.stop_before = None
    assert cli.main([*argv, '--resume']) == 0
    assert [n for n, _ in suite.calls] == ['greedy', 'sampled-s0', 'sampled-s1', 'sampled-s2']
    assert all((output / p).read_bytes() == data for p, data in before.items())
    summary = (output / 'summary.json').read_bytes()
    assert cli.main([*argv, '--resume']) == 0 and len(suite.calls) == 4
    assert (output / 'summary.json').read_bytes() == summary


@pytest.mark.parametrize('change', ['seeds', 'arm', 'tasks', 'receipt', 'episode', 'summary', 'config', 'code'])
def test_resume_identity_and_evidence_refusal(frozen_inputs, trained_adapter, suite, change, monkeypatch):
    args, adapter, run, _ = trained_adapter
    argv = arguments(frozen_inputs, args)
    assert cli.main(argv) == 0
    output = Path(args.output)
    if change == 'seeds':
        argv[argv.index('--sampled-seeds') + 1] = '7'
    elif change == 'arm':
        write_json(Path(adapter['path']).parent.parent / 'manifest.json', dict(run, arm='JOINT'))
    elif change == 'tasks':
        (output / 'tasks.json').write_bytes(b'{}\n')
    elif change == 'receipt':
        (output / 'greedy' / cli.RECEIPT).unlink()
    elif change == 'episode':
        episode = next((output / 'greedy').glob('episodes/*/episode.json'))
        write_json(episode, dict(read_json(episode), success=True))
    elif change == 'summary':
        summary = read_json(output / 'summary.json')
        summary['teacher_calls'] = 1
        write_json(output / 'summary.json', seal({k: v for k, v in summary.items() if k != 'content_sha256'}))
    elif change == 'config':
        argv[argv.index('--data-root') + 1] += '-different'
    else:
        original = cli.code_hashes
        monkeypatch.setattr(cli, 'code_hashes', lambda root: dict(original(root), fake_change='new'))
    with pytest.raises(ValueError):
        cli.main([*argv, '--resume'])
    assert len(suite.calls) == 4


def test_resume_refuses_partial_existing_decoding(frozen_inputs, trained_adapter, suite):
    args = trained_adapter[0]
    argv = arguments(frozen_inputs, args)
    suite.stop_inside = 'sampled-s0'
    with pytest.raises(RuntimeError):
        cli.main(argv)
    suite.stop_inside = None
    with pytest.raises(ValueError, match='only fills missing'):
        cli.main([*argv, '--resume'])
    assert [n for n, _ in suite.calls] == ['greedy']


@pytest.mark.parametrize('status', ['started', 'success', 'cache_hit'])
def test_teacher_attempts_refused_before_summary(frozen_inputs, trained_adapter, suite, status):
    args = trained_adapter[0]
    suite.teacher_row = dict(call_id='forbidden', status=status,
                            usage=dict(input_tokens=0, output_tokens=0, reasoning_tokens=0))
    with pytest.raises(AssertionError, match='Teacher usage'):
        cli.main(arguments(frozen_inputs, args))
    assert not (Path(args.output) / 'summary.json').exists()
    assert len(suite.calls) == 1


def test_teacher_usage_checked_on_resume(frozen_inputs, trained_adapter, suite):
    args = trained_adapter[0]
    argv = arguments(frozen_inputs, args)
    assert cli.main(argv) == 0
    write_json(Path(args.output) / 'greedy' / 'episodes' / 'bad' / 'teacher_usage.jsonl',
               dict(call_id='interrupted', status='started'))
    with pytest.raises(AssertionError, match='Teacher usage'):
        cli.main([*argv, '--resume'])
    assert len(suite.calls) == 4
