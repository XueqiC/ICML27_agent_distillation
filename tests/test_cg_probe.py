"""Offline CG protocol checks. No teacher, model weights, GPU, or live HTTP."""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from repair.execute import execute_actions
from repair.guidance import (GROUPS, GENERAL_REMINDER, OUTPUT_CAP, GuidancePrompt, GuidanceTeacher,
    GuidancePolicy, GuidanceSession, GuidanceValidationError, GlobalTeacherBudget, BudgetLedger,
    approve_hints, approved_cases, branch_texts, cluster_bootstrap, decision_rule, derive_metrics,
    execute_rollout, guidance_prompt, make_guidance_teacher, prefix_sha, rollout_key, validate_guidance)
from sed.config import load_config
from sed.react import build_messages
from sed.storage import canonical_bytes, digest, file_hash, write_json
from students.sampled_student import SampledStudent
from students.vllm_student import Student
from teachers.openai_teacher import TeacherResult
from tools.cg_probe import load_cases, main, select_groups, summarize_records
from test_lora_serving import adapter_bank, bound_report


COMMANDS = ['go to cabinet 1', 'open cabinet 1', 'examine cabinet 1', 'open cabinet 2',
            'go to cabinet 2', 'go to shelf 1', 'take apple 1 from cabinet 2', 'look', 'inventory']
INITIAL = ('You are in a room. You see a cabinet 1, a cabinet 2, a shelf 1, '
           'a countertop 1, and a diningtable 1.\nYour task is to: put an apple on a table.')


class ScriptEnvironment:
    seed, max_episode_steps, game_hash = 0, 40, 'synthetic_game'

    def __init__(self, mismatch=False):
        self.index, self.mismatch, self.picked = 0, mismatch, False

    def reset(self, task_id):
        self.index = 0
        return self.result(INITIAL)

    def result(self, observation, success=False):
        return ({'observation': observation, 'admissible_commands': COMMANDS},
                float(success), success, self.index == 40 and not success, {'won': success})

    def step(self, action):
        self.index += 1
        observation = 'The inspected contents show no target.'
        if self.mismatch and self.index == 4:
            observation = 'DIVERGED'
        if action == 'open cabinet 2':
            observation = 'You see an apple 1.'
        if action == 'take apple 1 from cabinet 2':
            observation, self.picked = 'You pick up the apple 1 from the cabinet 2.', True
        return self.result(observation, self.picked)

    def close(self):
        pass


@pytest.fixture
def original():
    prefix = ['go to cabinet 1', 'open cabinet 1', 'examine cabinet 1', *['look'] * 7]
    raw = ['Thought: I need the apple.\nAction: ' + a for a in [*prefix, *['look'] * 30]]
    return execute_actions(ScriptEnvironment, 'train/' + GROUPS[5] + '/trial_0', raw, 40)


def label(**changes):
    value = {'student_plan': 'Obtain the apple before cooling it.',
        'conditions': [{'condition': 'The apple is available.', 'status': 'undetermined', 'evidence_steps': []},
                       {'condition': 'Inspected contents held the object.', 'status': 'refuted', 'evidence_steps': [-1, 0, 9]}],
        'condition_hint': 'Your subgoal is obtaining the apple. Its presence remains unsupported. The inspected cabinet contents have ruled out an observed apple there.',
        'action_suggestion': {'command': 'open cabinet 2', 'reason': 'Opening it can reveal its contents.'}}
    return {**value, **changes}


@pytest.mark.parametrize('hint,reason', [
    ('The shelf could help.', 'unvisited_type'),
    ('The shelves could help.', 'unvisited_type'),
    ('Check the dining tables.', 'unvisited_type'),
    ('The countertops are unknown.', 'unvisited_type'),
    ('The cabinet 2 is empty.', 'unvisited_instance'),
    ('The cabinet 99 is empty.', 'unvisited_instance'),
    ('Maybe OPEN CABINET 2.', 'admissible_command'),
    ('word ' * 71, 'word_limit'),
])
def test_hint_validator_rejects(original, hint, reason):
    with pytest.raises(GuidanceValidationError, match=reason):
        validate_guidance(json.dumps(label(condition_hint=hint)), original)


@pytest.mark.parametrize('suggestion,reason', [
    ({'command': 'inventory', 'reason': 'Information.'}, 'verb'),
    ({'command': 'take apple 1 from cabinet 2', 'reason': 'Information.'}, 'verb'),
    ({'command': 'OPEN cabinet 2', 'reason': 'Information.'}, 'not_admissible'),
    ({'command': 'open cabinet 2', 'reason': 'word ' * 31}, 'word_limit'),
])
def test_suggestion_validator(original, suggestion, reason):
    with pytest.raises(GuidanceValidationError, match=reason):
        validate_guidance(json.dumps(label(action_suggestion=suggestion)), original)


@pytest.mark.parametrize('changes', [
    {'conditions': [{'condition': 'x', 'status': 'unknown', 'evidence_steps': []}]},
    {'conditions': [{'condition': 'x', 'status': 'satisfied', 'evidence_steps': [10]}]},
    {'conditions': [{'condition': 'x', 'status': 'satisfied', 'evidence_steps': [True]}]},
    {'conditions': [{'condition': 'x', 'status': 'satisfied', 'evidence_steps': [-2]}]},
    {'student_plan': []}, {'condition_hint': None}, {'extra': 'not allowed'},
])
def test_schema_rejections(original, changes):
    with pytest.raises(GuidanceValidationError):
        validate_guidance(json.dumps(label(**changes)), original)


def test_guidance_template_revision_1():
    from repair.guidance import TEMPLATE, TEMPLATE_VERSION
    assert TEMPLATE_VERSION == 'cg-probe/condition_guidance/2'
    assert 'YOUR OWN choice' in TEMPLATE


def test_prompt_is_public_prefix_only_and_template_is_independent(original):
    from repair.prompts import Prompt
    from repair.runtime import OUTPUT_CAPS
    before = (Prompt('diagnosis', {}).template_sha256, dict(OUTPUT_CAPS))
    poisoned = replace(original,
        initial={**original.initial, 'visible': {**original.initial['visible'], 'expert_plan': 'SECRET'}},
        steps=[{**s, 'audit_sha256': 'SECRET', 'policy_metadata': {'secret': 'SECRET'}} for s in original.steps])
    prompt = guidance_prompt(poisoned)
    assert prompt.kind == 'condition_guidance'
    assert len(prompt.payload['prefix']) == 10
    assert set(prompt.payload) == {'task', 'initial_observation', 'prefix', 'admissible_commands_now', 'steps_used', 'step_limit'}
    text = json.dumps(prompt.messages())
    assert 'SECRET' not in text and original.task_id not in text and 'synthetic_game' not in text
    assert prompt.payload['prefix'][0]['action_index'] == 0
    assert validate_guidance(json.dumps(label()), original) == label()
    assert before == (Prompt('diagnosis', {}).template_sha256, OUTPUT_CAPS)
    hidden = replace(original, steps=[{**original.steps[0], 'action_raw': '<|channel>thought\nPRIVATE<channel|>Thought: public\nAction: look'}, *original.steps[1:]])
    assert 'PRIVATE' not in json.dumps(guidance_prompt(hidden).payload)


class SequenceTeacher:
    def __init__(self, values):
        self.values, self.prompts = iter(values), []

    def generate(self, history, task, visible):
        self.prompts.append(json.loads(visible['observation']))
        value = next(self.values)
        return TeacherResult(json.dumps(value), None, {'response_status': 'completed',
            'usage': {'input_tokens': 20, 'output_tokens': 12, 'reasoning_tokens': 4},
            'returned_model_id': 'FAKE', 'timestamp': 'fake-time', 'cache_hit': False})


def session(tmp_path, original, teacher, **caps):
    task = {'task_id': original.task_id, 'split': 'train', 'sealed': False}
    budget = GlobalTeacherBudget(tmp_path / 'global.jsonl', **caps)
    return GuidanceSession(teacher, task, BudgetLedger(tmp_path / 'budget.jsonl'), budget)


@pytest.mark.parametrize('eventual_success', [False, True])
def test_retry_reason_and_missing(original, tmp_path, eventual_success):
    bad = label(condition_hint='unvisited shelves')
    teacher = SequenceTeacher([bad, bad, label() if eventual_success else bad])
    s = session(tmp_path, original, teacher)
    value, attempts = s.request(original)
    assert (value is not None) == eventual_success
    assert len(teacher.prompts) == len(attempts) == 3
    assert teacher.prompts[1]['retry_reason'] == 'condition_hint_unvisited_type'
    assert [p['retry_index'] for p in teacher.prompts] == [0, 1, 2]
    assert 'condition_hint_unvisited_type' in GuidancePrompt(**{
        'payload': teacher.prompts[1]['payload'], 'retry_reason': teacher.prompts[1]['retry_reason']}).messages()[0]['content']
    assert s.budget.summary()['reported_usage']['output_tokens'] == 36
    assert s.ledger.summary()['supervision']['teacher_reasoning_tokens'] == 12
    if not eventual_success:
        assert branch_texts(value) == {'A': None, 'B': GENERAL_REMINDER, 'C': None, 'D': None}
    assert [r['details']['raw_output'] for r in s.ledger.rows() if r['event'] == 'teacher_result'] == [a['raw_output'] for a in attempts]


@pytest.mark.parametrize('caps', [{'input_tokens': 1}, {'output_tokens': OUTPUT_CAP - 1}])
def test_cap_stops_before_first_call(original, tmp_path, caps):
    teacher = SequenceTeacher([label()])
    s = session(tmp_path, original, teacher, **caps)
    assert s.request(original)[0] is None
    assert teacher.prompts == []
    assert s.budget.calls == 0


def test_format_retry_checks_remaining_budget(original, tmp_path):
    teacher = SequenceTeacher([label(condition_hint='shelves'), label()])
    s = session(tmp_path, original, teacher, output_tokens=OUTPUT_CAP)
    value, attempts = s.request(original)
    assert value is None and len(teacher.prompts) == 1
    assert attempts[-1]['status'] == 'budget_stopped'


def test_second_format_retry_is_not_a_cached_rejection(original, tmp_path):
    from teachers.capped import CappedTeacher
    from test_capped_teacher import RawClient
    bad = label(condition_hint='shelves')
    sdk = RawClient(texts=[json.dumps(v) for v in [bad, bad, label()]])
    teacher = CappedTeacher(counter_path=tmp_path / 'counter.json', cap=21,
        cache_dir=tmp_path / 'cache', usage_ledger=tmp_path / 'usage.jsonl', run_id='cg-retry',
        client=sdk, teacher_class=GuidanceTeacher, sampling={'max_output_tokens': OUTPUT_CAP})
    value, attempts = session(tmp_path, original, teacher).request(original)
    assert value == label() and len(sdk.requests) == len(attempts) == 3
    assert len({digest(r['input']) for r in sdk.requests}) == 3


@pytest.mark.parametrize('allow_retry', [False, True])
def test_real_transport_factory_with_fake_sdk_retries_obey_global_cap(original, tmp_path, allow_retry):
    from teachers.capped import CappedTeacher
    from test_capped_teacher import SequenceClient, http_429
    from test_stage2c_teacher import FakeClock
    clock = FakeClock()
    response = SimpleNamespace(status_code=200, headers={}, parse=lambda: {
        'output_text': json.dumps(label()), 'status': 'completed', 'model': 'returned-fake-id',
        'id': 'fake', '_request_id': 'fake', 'usage': {'input_tokens': 13, 'output_tokens': 7,
        'output_tokens_details': {'reasoning_tokens': 3}}})
    sdk = SequenceClient([http_429(), response])
    teacher = CappedTeacher(counter_path=tmp_path / 'counter.json', cap=21,
        cache_dir=tmp_path / 'cache', usage_ledger=tmp_path / 'usage.jsonl', run_id='cg-test',
        client=sdk, teacher_class=GuidanceTeacher, sampling={'max_output_tokens': OUTPUT_CAP},
        clock=clock, sleep=clock.sleep, jitter=lambda lo, hi: 0)
    s = session(tmp_path, original, teacher, output_tokens=OUTPUT_CAP * (2 if allow_retry else 1))
    value, attempts = s.request(original)
    assert len(sdk.requests) == (2 if allow_retry else 1)
    assert (value is not None) == allow_retry
    assert all(r['max_output_tokens'] == OUTPUT_CAP for r in sdk.requests)
    assert all('SECRET' not in json.dumps(r) for r in sdk.requests)
    assert 'PRIVATE_PROVIDER_EXCEPTION' not in s.ledger.path.read_text()
    if allow_retry:
        assert attempts[-1]['metadata']['returned_model_id'] == 'returned-fake-id'
        assert s.budget.used['output_tokens'] == OUTPUT_CAP + 7
        assert s.budget.reported_usage['reasoning_tokens'] == 3
    constructed = make_guidance_teacher(load_config('configs/base_e4b.yaml'), tmp_path / 'factory', run_id='unused')
    assert isinstance(constructed.teacher, GuidanceTeacher)
    assert constructed.teacher.sampling['max_output_tokens'] == OUTPUT_CAP


def test_approval_hash_and_rejected_cases(tmp_path):
    path = tmp_path / 'hints.json'
    write_json(path, {'cases': [{'task_id': 'one', 'texts': branch_texts(label())},
                                {'task_id': 'two', 'texts': branch_texts(None)}]})
    decisions = tmp_path / 'decisions.tsv'
    decisions.write_text('task_id\tdecision\tnote\none\treject\tlocation advice\ntwo\tapprove\tmissing\n')
    approval = approve_hints(path, decisions)
    _, cases = approved_cases(path, approval)
    assert [c['branches'] for c in cases] == [['A', 'B'], ['A', 'B']]
    assert [c['cd_status'] for c in cases] == ['rejected', 'missing']
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA256'):
        approved_cases(path, approval)


def test_served_adapter_gate_matches_coverage_and_checks_hashes(tmp_path, bound_report, adapter_bank):
    from tools.cg_probe import serving_spec
    report, server = bound_report
    _, _, student = adapter_bank
    name = 'e4b-tsft-k20-e32-lr5e-5'
    server['config']['server']['lora_adapters'][0]['name'] = name
    report['model_name'] = name
    report['server_provenance_sha256'] = digest(server)
    server_path, report_path = tmp_path / 'server.json', tmp_path / 'report.json'
    write_json(server_path, server)
    write_json(report_path, report)
    args = SimpleNamespace(server_provenance=server_path, lora_check=report_path,
        endpoint=server['config']['server']['endpoint'], model_name=name, output=tmp_path / 'run')
    config = {'models': {'student': student}}
    cases = [{'source': {'provenance': {'config': {'server_provenance': deepcopy(server)}}}}]
    spec = serving_spec(args, config, cases)
    assert spec['model'] == name and spec['max_tokens'] == 1024 and spec['enable_thinking'] is False
    old = cases[0]['source']['provenance']['config']['server_provenance']['config']['server']['lora_adapters'][0]
    old['adapter_weights_sha256'] = 'changed'
    with pytest.raises(ValueError, match='original coverage adapter'):
        serving_spec(args, config, cases)
    report['passed'] = False
    write_json(report_path, report)
    with pytest.raises(ValueError, match='passing equivalence'):
        serving_spec(args, config, [])


@pytest.mark.parametrize('turn', [10, 11, 14])
def test_exact_student_messages_only_one_user_turn_changes(original, tmp_path, turn):
    context = original.decision_context(turn)
    history, visible = context['history'], context['visible_context']
    saved = deepcopy(context)
    requests = {}
    for branch, text in branch_texts(label()).items():
        def serve(request):
            requests[branch] = json.loads(request.content)
            return httpx.Response(200, json={'model': 'fake-model', 'choices': [{'message': {
                'content': 'Thought: Continue.\nAction: look'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
        student = SampledStudent(endpoint='http://fake/v1', model='fake-model', revision='fake',
            enable_thinking=False, log_dir=tmp_path / branch, sampling_seed=3,
            task_id=original.task_id, transport=httpx.MockTransport(serve))
        GuidancePolicy(student, original.task_id, text)(history, visible)
        student.close()
    baseline = build_messages(history, {'task_id': original.task_id}, visible)
    assert canonical_bytes(requests['A']['messages']) == canonical_bytes(baseline)
    for branch in 'BCD':
        messages = requests[branch]['messages']
        changed = [i for i, (a, b) in enumerate(zip(messages, baseline)) if a != b]
        assert changed == [21]  # system + ten (user, assistant) pairs
        assert messages[21]['role'] == 'user'
        a, b = json.loads(messages[21]['content']), json.loads(baseline[21]['content'])
        assert a == {**b, 'observation': b['observation'] + '\n\nGuidance: ' + branch_texts(label())[branch]}
        assert json.dumps(messages).count('Guidance:') == 1
        assert requests[branch]['seed'] == requests['A']['seed'] == int(digest([3, original.task_id, turn])[:8], 16)
    assert context == saved


class SuffixPolicy:
    def generate(self, history, task, visible):
        actions = ['go to cabinet 1', 'open cabinet 2', 'take apple 1 from cabinet 2']
        action = actions[len(history) - 10]
        return TeacherResult('Thought: Its location is unknown.\nAction: ' + action, action, {})


def case_for(original):
    return {'task_id': original.task_id, 'group': GROUPS[5], 'target': 'apple',
            'texts': branch_texts(label()), 'teacher_json': label(), 'cd_status': 'approved'}


def test_replay_metrics_resume_and_mismatch(original, tmp_path):
    case = case_for(original)
    kwargs = dict(branch='D', decoding='sampled', seed=0, env_factory=ScriptEnvironment,
                  policy_factory=lambda log_dir: SuffixPolicy(), output=tmp_path, run_sha256='run')
    row = execute_rollout(case, original, **kwargs)
    assert row['prefix_verified'] and not row['void']
    m = row['metrics']
    assert m['found'] and m['steps_to_find'] == 2 and m['picked'] and m['success']
    assert m['followed_suggestion'] is False and m['suggestion_reveals_target'] is True
    assert m['revisits_of_cleared'] == 1 and m['inadmissible_count'] == 0
    assert m['first_post_branch_Thought'] == 'Its location is unknown.'
    assert row['budget']['interaction']['evaluation_steps'] == 13
    assert 'Guidance:' not in json.dumps(row['execution'])
    def forbidden(*args):
        pytest.fail('Completed rollout must be skipped without env or policy creation')
    assert execute_rollout(case, original, **{**kwargs, 'env_factory': forbidden, 'policy_factory': forbidden}) == row
    with pytest.raises(ValueError, match='identity'):
        execute_rollout(case, original, **{**kwargs, 'run_sha256': 'different'})
    mismatch = execute_rollout(case, original, **{**kwargs, 'seed': 1,
        'env_factory': partial(ScriptEnvironment, mismatch=True)})
    assert mismatch['void'] and not mismatch['prefix_verified'] and mismatch['metrics'] is None
    assert mismatch['void_reason'] == 'handback_prefix_check:ValueError'
    assert len(mismatch['execution']['steps']) == 4


def test_metric_boundaries_legal_pick_and_suggestion(original):
    step = original.steps[10]
    suffix = [dict(step, action_parsed='open cabinet 2', action_raw='Thought: unknown\nAction: open cabinet 2',
                   visible={**step['visible'], 'observation': 'APPLE 3'}),
              dict(step, action_parsed='take apple 3 from cabinet 2', action_outcome='inadmissible',
                   visible={**step['visible'], 'observation': 'You pick up an apple 3.'})]
    trajectory = replace(original, steps=[*original.steps[:10], *suffix])
    m = derive_metrics(trajectory, 'apple', branch='D', suggested_command='open cabinet 2')
    assert m['steps_to_find'] == 1 and m['found'] and m['followed_suggestion']
    assert not m['picked'] and m['inadmissible_count'] == 1
    suffix[-1]['action_outcome'] = 'legal'
    suffix[-1]['visible']['observation'] = 'Nothing happens.'
    assert not derive_metrics(replace(trajectory, steps=[*original.steps[:10], *suffix]), 'apple', branch='A')['picked']
    assert derive_metrics(original, 'apple', branch='A')['suggestion_reveals_target'] is None
    # A prefix sighting cannot count as a post-branch find.
    prefix_seen = replace(original, steps=[{**original.steps[0], 'visible': {**step['visible'], 'observation': 'apple 3'}}, *original.steps[1:]])
    assert not derive_metrics(prefix_seen, 'apple', branch='A')['found']


def coverage_fixture(tmp_path, original):
    root = tmp_path / 'coverage'
    directory = root / 'attempt-01-greedy' / 'episodes' / 'case'
    directory.mkdir(parents=True)
    episode = {'task_id': original.task_id, 'episode_id': 'fake-episode', 'success': False,
        'split': 'train', 'sealed': False, 'complete': True, 'status': 'ok', 'seed': 0,
        'max_episode_steps': 40, 'game_sha256': original.game_sha256,
        'initial': original.initial, 'steps': original.steps, 'truncated': True}
    write_json(directory / 'episode.json', episode)
    write_json(directory / 'provenance.json', {'config': {}, 'details': {'parent_seed_ids': ['fake-seed']}})
    (directory / 'events.jsonl').write_text('{}\n')
    row = {'task_id': original.task_id, 'group_id': 'train/' + GROUPS[5], 'episode_id': 'fake-episode',
        'attempt': 1, 'decoding': 'greedy', 'sampling_seed': None, 'episode_dir': str(directory.relative_to(root)),
        **{k: file_hash(directory / f) for k, f in [('episode_sha256', 'episode.json'),
           ('provenance_sha256', 'provenance.json'), ('events_sha256', 'events.jsonl')]}}
    write_json(root / 'failures.json', {'failures': [row], 'list_sha256': digest([row])})
    write_json(root / 'summary.json', {'failure_list': [row], 'failure_list_sha256': digest([row]), 'branch_decision_actionable': True})
    return root, directory


@pytest.mark.parametrize('change', ['short', 'parser_error', 'already_seen', 'sampled', 'missing_group', 'tampered'])
def test_failure_import_guards(original, tmp_path, change):
    root, directory = coverage_fixture(tmp_path, original)
    path = root / 'failures.json'
    failures = json.loads(path.read_text())
    episode = json.loads((directory / 'episode.json').read_text())
    if change == 'short':
        episode['steps'] = episode['steps'][:10]
    elif change == 'parser_error':
        episode['steps'][0]['action_raw'] = 'bad reply'
    elif change == 'already_seen':
        episode['steps'][9]['visible']['observation'] = 'Apple 1 here.'
    elif change == 'sampled':
        failures['failures'][0]['decoding'] = 'sampled'
    elif change == 'missing_group':
        failures['failures'][0]['group_id'] = 'train/other'
    write_json(directory / 'episode.json', episode)
    if change != 'tampered':
        failures['failures'][0]['episode_sha256'] = file_hash(directory / 'episode.json')
    else:
        (directory / 'episode.json').write_bytes((directory / 'episode.json').read_bytes() + b' ')
    failures['list_sha256'] = digest(failures['failures'])
    write_json(path, failures)
    write_json(root / 'summary.json', {'failure_list': failures['failures'], 'failure_list_sha256': failures['list_sha256'], 'branch_decision_actionable': True})
    with pytest.raises(ValueError):
        load_cases(path, root, [GROUPS[5]], load_config('configs/base_e4b.yaml'), dry_run=True)


def test_all_cli_stages_dry_run_and_resume(original, tmp_path):
    root, _ = coverage_fixture(tmp_path, original)
    output, rollouts = tmp_path / 'hints', tmp_path / 'rollouts'
    common = ['--failures', str(root / 'failures.json'), '--coverage-root', str(root),
              '--config', 'configs/base_e4b.yaml', '--dry-run']
    assert main(['hints', *common, '--tasks', GROUPS[5], '--output', str(output)]) == 0
    document = json.loads((output / 'hints.json').read_text())
    assert document['cases'][0]['prefix_sha'] == prefix_sha(original)
    assert document['cases'][0]['texts']['C']
    decisions = tmp_path / 'decisions.tsv'
    decisions.write_text('task_id\tdecision\tnote\n' + original.task_id + '\tapprove\tfake only\n')
    assert main(['approve', '--hints', str(output / 'hints.json'), '--decisions', str(decisions)]) == 0
    argv = ['rollouts', *common, '--hints', str(output / 'hints.json'), '--approval', str(output / 'hints_approval.json'),
            '--output', str(rollouts), '--workers', '2', '--limit-cases', '1']
    assert main(argv) == 0
    paths = sorted((rollouts / 'records').glob('*.json'))
    assert len(paths) == 36
    before = {p: (file_hash(p), p.stat().st_mtime_ns) for p in paths}
    with pytest.raises(FileExistsError):
        main(argv)
    assert main([*argv, '--resume']) == 0
    assert before == {p: (file_hash(p), p.stat().st_mtime_ns) for p in paths}
    report = tmp_path / 'summary'
    assert main(['summarize', '--rollouts', str(rollouts), '--output', str(report)]) == 0
    result = json.loads((report / 'summary.json').read_text())
    assert result['expected_rollouts'] == 36 and not result['missing_rollouts']
    assert result['conclusion_mapping'] == 'Incomplete or synthetic probe; no mechanism conclusion.'
    assert {r['decoding'] for r in result['aggregate']} == {'sampled', 'greedy'}
    assert (report / 'summary.md').exists() and (report / 'per_case.tsv').exists()


def test_bootstrap_determinism_and_case_clustering():
    assert cluster_bootstrap([.25, -.125, 0, .5, .125, 1, -.5]) == cluster_bootstrap([.25, -.125, 0, .5, .125, 1, -.5])
    assert cluster_bootstrap([.25] * 7) == [.25, .25]
    assert cluster_bootstrap([]) is None


@pytest.mark.parametrize('x,y,expected', [
    ([.15, .15], [0, 0], 'C better than B'),
    ([.149, .149], [0, 0], 'no clear difference'),
    ([1, 0], [0, 0], 'no clear difference'),
    ([.5, .5, 0], [0, 0, .1], 'no clear difference'),
    ([.5, .5, .5, 0], [0, 0, 0, .1], 'C better than B'),
    ([0, 0], [.5, .5], 'no clear difference'), ([], [], 'no clear difference'),
])
def test_decision_rule_edges(x, y, expected):
    assert decision_rule(x, y, 'C', 'B') == expected


def test_summary_pairs_do_not_pool_greedy_or_silently_drop_seeds():
    plan = {'run_sha256': 'r', 'identity': {'dry_run': True}, 'cases': [
        {'task_id': t, 'group': t, 'branches': list('ABCD'), 'cd_status': 'approved'} for t in ['t1', 't2']]}
    rows = []
    for case in plan['cases']:
        for b in 'ABCD':
            for d, s in [('greedy', None), *[('sampled', s) for s in range(8)]]:
                found = b in 'CD' if d == 'sampled' else b in 'AB'
                metrics = {'found': found, 'steps_to_find': 2 if found else None, 'picked': False, 'success': False,
                    'followed_suggestion': b == 'D' if b == 'D' else None, 'suggestion_reveals_target': None,
                    'inadmissible_count': 0, 'revisits_of_cleared': 0, 'first_post_branch_action': 'look'}
                rows.append({'rollout_key': rollout_key(case['task_id'], b, d, s), 'run_sha256': 'r',
                    'case': case['task_id'], 'branch': b, 'decoding': d, 'seed': s, 'void': False,
                    'void_reason': None, 'metrics': metrics})
    report = summarize_records(plan, rows)
    paired = next(p for p in report['sampled_pairs'] if p['comparison'] == 'C-B' and p['metric'] == 'found_rate')
    assert paired['difference'] == 1 and paired['ci95'] == [1, 1] and paired['decision'] == 'C better than B'
    missing = next(r for r in rows if (r['case'], r['branch'], r['decoding'], r['seed']) == ('t1', 'C', 'sampled', 2))
    rows.remove(missing)
    report = summarize_records(plan, rows)
    pair = next(p for p in report['sampled_pairs'] if p['comparison'] == 'C-B' and p['metric'] == 'found_rate')
    assert pair['excluded_cases'] == ['t1'] and pair['cases'] == ['t2']
    assert next(r for r in report['aggregate'] if (r['branch'], r['decoding']) == ('C', 'sampled'))['found_rate'] is None
    with pytest.raises(ValueError, match='duplicate'):
        summarize_records(plan, [*rows, rows[0]])
