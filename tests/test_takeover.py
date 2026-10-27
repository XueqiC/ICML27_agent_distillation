"""Takeover uses fake transport/environments only, including subprocess dry runs."""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair.execute import execute_actions
from repair.guidance import GROUPS
from repair.runtime import CaseStopped
from repair.takeover import (SharedTeacherBudget, execute_takeover, make_teacher_policy,
                             usage_summary, takeover_metrics)
from sed.config import load_config
from sed.react import build_messages, PROMPT_TEMPLATE, PROMPT_VERSION
from sed.storage import digest
from teachers.openai_teacher import Teacher, TeacherCallError
from tools.teacher_takeover import load_spec, main, summarize_records
from test_cg_probe import ScriptEnvironment, coverage_fixture, original
from test_teacher import FakeClient, response


@pytest.fixture
def budget(tmp_path):
    return SharedTeacherBudget(tmp_path / 'global.jsonl', input_tokens=1_000_000, output_tokens=100_000)


def policy(tmp_path, budget, task, client):
    result = make_teacher_policy(load_config('configs/base_e4b.yaml'), task,
                                 tmp_path / 'call', tmp_path, budget, client=client)
    result.teacher.sleep = lambda _: None
    result.teacher.jitter = lambda *_: 0
    return result


def test_collection_teacher_exact_prompt_options_and_public_whitelist(tmp_path, budget, task, visible):
    client = FakeClient()
    actor = policy(tmp_path, budget, task, client)
    assert type(actor.teacher) is Teacher
    config = load_config('configs/base_e4b.yaml')['models']['teacher']
    assert actor.teacher.sampling == config['sampling']
    assert actor.teacher.max_attempts == config['max_attempts']
    assert actor.teacher.timeout_seconds == config['timeout_seconds']
    assert actor.teacher.prompt_template == PROMPT_TEMPLATE
    polluted = dict(visible, audit='PRIVATE', policy_commands=['PRIVATE'])
    history = [{'visible': polluted, 'reply': 'Thought: Student prefix.\nAction: look', 'hidden': 'PRIVATE'}]
    actor(history, polluted)
    request = client.requests[0]
    assert request['input'] == build_messages(history, task, visible)
    assert request['input'][2]['content'] == history[0]['reply']
    assert request['max_output_tokens'] == config['sampling']['max_output_tokens']
    assert 'PRIVATE' not in json.dumps(request) and task['task_id'] not in json.dumps(request)
    assert budget.summary()['reported_usage']['reasoning_tokens'] == 10
    assert budget.summary()['tokens_or_unsettled_bounds']['input_tokens'] == 13
    assert usage_summary(tmp_path / 'call/usage.jsonl')['calls'][0]['returned_model_id'] == 'gpt-6-luna'


def test_repeated_runs_sample_independently_and_share_limiter(tmp_path, budget, task, visible):
    client = FakeClient([response(), response()])
    a = make_teacher_policy(load_config('configs/base_e4b.yaml'), task, tmp_path / 'run0', tmp_path, budget, client=client)
    b = make_teacher_policy(load_config('configs/base_e4b.yaml'), task, tmp_path / 'run1', tmp_path, budget, client=client)
    a([], visible)
    b([], visible)
    assert len(client.requests) == 2
    assert a.teacher.cache_dir != b.teacher.cache_dir
    assert a.teacher.limiter.path == b.teacher.limiter.path
    assert budget.summary()['reserved_attempts'] == 2


class Limit429(Exception):
    status_code = 429
    body = {'error': {'code': 'rate_limit_exceeded'}}


class Quota429(Exception):
    status_code = 429
    body = {'error': {'code': 'insufficient_quota', 'message': 'PRIVATE'}}


def test_rate_limit_retries_reserved_before_every_request(tmp_path, budget, task, visible):
    client = FakeClient([Limit429('PRIVATE'), response()])
    actor = policy(tmp_path, budget, task, client)
    waits = []
    actor.teacher.sleep = waits.append
    actor([], visible)
    summary = budget.summary()
    assert len(client.requests) == 2 and len(waits) == 1
    assert summary['reserved_attempts'] == 2 and summary['unsettled_attempts'] == 1
    assert summary['tokens_or_unsettled_bounds']['output_tokens'] == 2048 + 17
    assert summary['tokens_or_unsettled_bounds']['input_tokens'] == actor.reservation.input_bound + 13
    assert 'PRIVATE' not in (tmp_path / 'call/usage.jsonl').read_text()


def test_cap_blocks_retry_and_all_later_runs(tmp_path, task, visible):
    budget = SharedTeacherBudget(tmp_path / 'global.jsonl', input_tokens=1_000_000, output_tokens=2048)
    client = FakeClient([Limit429('PRIVATE'), response()])
    actor = policy(tmp_path, budget, task, client)
    with pytest.raises(CaseStopped):
        actor([], visible)
    assert len(client.requests) == 1
    assert budget.summary()['stop_reason'] == 'global_output_tokens_cap'
    with pytest.raises(CaseStopped):
        policy(tmp_path / 'other', budget, task, client)([], visible)
    assert len(client.requests) == 1


@pytest.mark.parametrize('failure', [Quota429('PRIVATE'), RuntimeError('PRIVATE')])
def test_quota_or_fatal_error_halts_without_empty_action(tmp_path, budget, task, visible, failure):
    client = FakeClient([failure, response()])
    actor = policy(tmp_path, budget, task, client)
    with pytest.raises(TeacherCallError):
        actor([], visible)
    assert len(client.requests) == 1
    assert budget.summary()['stop_reason'] in ('teacher_quota', 'teacher_halted')
    assert 'PRIVATE' not in (tmp_path / 'call/usage.jsonl').read_text()
    with pytest.raises(CaseStopped):
        actor([], visible)


@pytest.mark.parametrize('bad', ['empty', 'truncated', 'unknown_usage'])
def test_invalid_teacher_completion_is_never_an_action(tmp_path, budget, task, visible, bad):
    result = response('' if bad == 'empty' else 'Action: look')
    if bad == 'truncated':
        result.status = 'incomplete'
        result.incomplete_details = {'reason': 'max_output_tokens'}
    if bad == 'unknown_usage':
        result.usage = None
    actor = policy(tmp_path, budget, task, FakeClient([result]))
    with pytest.raises(TeacherCallError):
        actor([], visible)
    assert budget.summary()['stop_reason'] == 'teacher_halted'


def test_budget_resume_idempotent_settlement_and_unknown_reservations(tmp_path):
    path = tmp_path / 'budget.jsonl'
    budget = SharedTeacherBudget(path, input_tokens=100, output_tokens=50)
    budget.reserve('a', 40, 20)
    budget.reserve('b', 40, 20)
    resumed = SharedTeacherBudget(path, input_tokens=100, output_tokens=50)
    resumed.settle('b', {'input_tokens': 10, 'output_tokens': 5, 'reasoning_tokens': 1})
    resumed.settle('b', {'input_tokens': 10, 'output_tokens': 5, 'reasoning_tokens': 1})
    assert resumed.summary()['tokens_or_unsettled_bounds'] == {'input_tokens': 50, 'output_tokens': 25}
    assert resumed.summary()['unsettled_attempts'] == 1
    with pytest.raises(ValueError):
        SharedTeacherBudget(path, input_tokens=99, output_tokens=50)
    with pytest.raises(CaseStopped):
        resumed.settle('a', {'input_tokens': 101, 'output_tokens': 0})
    assert SharedTeacherBudget(path, input_tokens=100, output_tokens=50).summary()['stop_reason'] == 'global_reported_usage_exceeded_reservation'


def make_case(original):
    return {'case_id': 'case', 'task_id': original.task_id, 'group': GROUPS[5], 'branch_index': 10, 'source': {}}


def test_prefix_reset_replay_success_metrics_and_immutable_resume(original, tmp_path, budget):
    client = FakeClient([response('Action: open cabinet 2'), response('Action: take apple 1 from cabinet 2')])
    task = {'task_id': original.task_id, 'split': 'train', 'sealed': False}
    factory = partial(make_teacher_policy, load_config('configs/base_e4b.yaml'), task,
                      output=tmp_path, budget=budget, client=client)
    args = dict(mode='takeover', run=0, env_factory=ScriptEnvironment, policy_factory=factory,
                output=tmp_path / 'run', run_sha256='test', budget=budget)
    row = execute_takeover(make_case(original), original, **args)
    assert row['prefix_verified'] and not row['void'] and row['success']
    assert row['metrics']['found'] and row['metrics']['picked']
    assert row['metrics']['steps_used_after_branch'] == 2
    assert row['budget']['interaction']['evaluation_steps'] == 12
    assert len(client.requests[0]['input']) == 22
    assert row['execution']['steps'][:10] == original.to_json()['steps'][:10]
    assert execute_takeover(make_case(original), original, **args) == row
    assert len(client.requests) == 2


def test_prefix_mismatch_never_calls_teacher(original, tmp_path, budget):
    def no_teacher(*_):
        raise AssertionError('Teacher called on diverged prefix')
    fake = SimpleNamespace(failure=None, __call__=no_teacher)
    row = execute_takeover(make_case(original), original, mode='takeover', run=0,
        env_factory=partial(ScriptEnvironment, mismatch=True), policy_factory=lambda _: fake,
        output=tmp_path, run_sha256='test', budget=budget)
    assert row['void'] and not row['prefix_verified']
    assert 'handback_prefix_check' in row['void_reason']


def test_from_start_has_no_student_prefix(original, tmp_path, budget):
    client = FakeClient([response('Action: take apple 1 from cabinet 2')])
    task = {'task_id': original.task_id, 'split': 'train', 'sealed': False}
    row = execute_takeover(make_case(original), original, mode='from_start', run=0,
        env_factory=ScriptEnvironment, policy_factory=partial(make_teacher_policy,
            load_config('configs/base_e4b.yaml'), task, output=tmp_path, budget=budget, client=client),
        output=tmp_path, run_sha256='test', budget=budget)
    assert not row['void'] and row['branch_index'] == 0
    assert len(client.requests[0]['input']) == 2
    assert row['metrics']['steps_used_after_branch'] == 1


@pytest.mark.parametrize('branch_index', [1, 3, 15, 39])
def test_non_cg_branch_replay_and_suffix_metrics(original, tmp_path, budget, branch_index):
    client = FakeClient([response('Action: take apple 1 from cabinet 2')])
    task = {'task_id': original.task_id, 'split': 'train', 'sealed': False}
    case = dict(make_case(original), branch_index=branch_index)
    row = execute_takeover(case, original, mode='takeover', run=0,
        env_factory=ScriptEnvironment, policy_factory=partial(make_teacher_policy,
            load_config('configs/base_e4b.yaml'), task, output=tmp_path, budget=budget, client=client),
        output=tmp_path, run_sha256='variable-branch', budget=budget)
    assert row['prefix_verified'] and row['success'] and row['branch_index'] == branch_index
    assert row['execution']['steps'][:branch_index] == original.to_json()['steps'][:branch_index]
    assert len(client.requests[0]['input']) == 2 + 2 * branch_index
    assert row['metrics']['steps_used_after_branch'] == row['metrics']['steps_to_success'] == 1
    assert row['metrics']['steps_to_find'] == 1 and row['metrics']['picked']
    assert row['metrics']['steps_remaining_when_found'] == 39 - branch_index


def test_halted_execution_records_prefix_without_failed_reply(original, tmp_path, budget):
    task = {'task_id': original.task_id, 'split': 'train', 'sealed': False}
    client = FakeClient([Quota429('PRIVATE')])
    kwargs = dict(mode='takeover', env_factory=ScriptEnvironment,
        policy_factory=partial(make_teacher_policy, load_config('configs/base_e4b.yaml'), task,
                               output=tmp_path, budget=budget, client=client),
        output=tmp_path, run_sha256='test', budget=budget)
    failed = execute_takeover(make_case(original), original, run=0, **kwargs)
    assert failed['void'] and failed['void_reason'] == 'teacher_quota'
    assert len(failed['execution']['steps']) == 10
    remaining = execute_takeover(make_case(original), original, run=1, **kwargs)
    assert remaining['status'] == 'not_run_halted' and remaining['execution'] is None
    assert len(client.requests) == 1


def test_dry_cli_process_pool_spec_validation_resume_summary(original, tmp_path):
    coverage, _ = coverage_fixture(tmp_path, original)
    spec, output = tmp_path / 'spec.json', tmp_path / 'runs'
    assert main(['spec-from-cg', '--failures', str(coverage / 'failures.json'), '--coverage-root', str(coverage),
                 '--groups', GROUPS[5], '--dry-run', '--output', str(spec)]) == 0
    cases, _ = load_spec(spec, load_config('configs/base_e4b.yaml'), dry_run=True)
    assert 'instruction' not in cases[0]['task']  # No CG guidance-envelope duplicate goal.
    args = ['run', '--spec', str(spec), '--dry-run', '--runs', '2', '--workers', '2', '--output', str(output)]
    assert main(args) == 0
    records = {p.name: p.read_bytes() for p in (output / 'records').glob('*.json')}
    assert len(records) == 2
    assert main([*args, '--resume']) == 0
    assert records == {p.name: p.read_bytes() for p in (output / 'records').glob('*.json')}
    assert main(['summarize', '--rollouts', str(output), '--output', str(tmp_path / 'summary')]) == 0
    report = json.loads((tmp_path / 'summary/summary.json').read_text())
    assert report['per_case'][0]['runs'] == 2
    assert report['teacher_usage']['input_tokens'] == 60 * 100
    assert report['global_budget']['reserved_attempts'] == 60
    with pytest.raises(ValueError):
        main([*args, '--resume', '--max-teacher-output-tokens', '1'])
    doc = json.loads(spec.read_text())
    doc[0]['source']['episode_dir'] = '../escape'
    spec.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_spec(spec, load_config('configs/base_e4b.yaml'), dry_run=True)


def test_dry_cli_zero_budget_records_all_unrun(original, tmp_path):
    coverage, _ = coverage_fixture(tmp_path, original)
    spec, output = tmp_path / 'spec.json', tmp_path / 'runs'
    main(['spec-from-cg', '--failures', str(coverage / 'failures.json'), '--coverage-root', str(coverage),
          '--groups', GROUPS[5], '--dry-run', '--output', str(spec)])
    assert main(['run', '--spec', str(spec), '--dry-run', '--runs', '3', '--workers', '1',
                 '--max-teacher-output-tokens', '0', '--output', str(output)]) == 1
    rows = [json.loads(p.read_text()) for p in (output / 'records').glob('*.json')]
    assert sum(r['status'] == 'not_run_budget' for r in rows) == 2
    assert all(r['success'] == 'unknown' for r in rows)
    assert not list((output / 'calls').glob('*/*/usage.jsonl'))


def _reserve_in_worker(path, request):
    budget = SharedTeacherBudget(path, input_tokens=20, output_tokens=20)
    try:
        budget.reserve(str(request), 5, 5)
        return True
    except CaseStopped:
        return False


def test_process_workers_cannot_overreserve_shared_caps(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(_reserve_in_worker, [tmp_path / 'shared.jsonl'] * 10, range(10)))
    assert sum(results) == 4
    state = SharedTeacherBudget(tmp_path / 'shared.jsonl', input_tokens=20, output_tokens=20).summary()
    assert state['tokens_or_unsettled_bounds'] == {'input_tokens': 20, 'output_tokens': 20}


def test_canary_real_seam_is_one_case_one_run_with_clamped_caps(original, tmp_path, monkeypatch):
    import tools.teacher_takeover as cli
    import harness.alfworld_env as harness
    coverage, _ = coverage_fixture(tmp_path, original)
    spec, output = tmp_path / 'spec.json', tmp_path / 'canary'
    main(['spec-from-cg', '--failures', str(coverage / 'failures.json'), '--coverage-root', str(coverage),
          '--groups', GROUPS[5], '--dry-run', '--output', str(spec)])
    doc = json.loads(spec.read_text())
    doc.append(dict(doc[0], case_id='second-case'))
    spec.write_text(json.dumps(doc))
    config = load_config('configs/base_e4b.yaml')
    for name in ('manifest', 'split_freeze'):
        path = tmp_path / (name + '.json')
        path.write_text('{}')
        config['paths'][name] = str(path)
    monkeypatch.setattr(cli, 'load_config', lambda _: deepcopy(config))
    native_loader = cli.load_cases
    monkeypatch.setattr(cli, 'load_cases', lambda *a, **k: native_loader(*a, dry_run=True))
    monkeypatch.setattr(harness, 'Environment', lambda *a, **k: ScriptEnvironment())
    client = FakeClient([response('Action: take apple 1 from cabinet 2')])
    native_factory = cli.make_teacher_policy
    monkeypatch.setattr(cli, 'make_teacher_policy', lambda *a, **k: native_factory(*a, **k, client=client))
    assert main(['run', '--spec', str(spec), '--canary', '--runs', '99', '--workers', '99',
                 '--data-root', 'unused-fake-root', '--output', str(output)]) == 0
    records = list((output / 'records').glob('*.json'))
    assert len(records) == 1 and len(client.requests) == 1
    identity = json.loads((output / 'plan.json').read_text())['identity']
    assert identity['runs'] == 1 and len(identity['cases']) == 1 and not identity['dry_run']
    assert identity['caps'] == {'input_tokens': 100000, 'output_tokens': 8192}
