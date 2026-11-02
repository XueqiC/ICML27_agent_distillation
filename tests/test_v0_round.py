"""CPU-only V0 protocol tests. Existing fake environment/student/transport used."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair.dry_fakes import FakeEnvironment, O, N
from repair.execute import execute_actions
from repair.joint import FakeReviewClient, FakeStudent
from repair.ledger import BudgetLedger
from repair.records import TrajectoryRecord
from repair.takeover import SharedTeacherBudget
from repair.v0_common import BINS, QUOTAS, SyntheticManifest, completed, finish, phase
from repair.v0_data import decision_rows, choose_episodes
from repair.v0_execution import attempt, probe_report, selected_failures
from repair.v0_expand import allocate, generate
from repair.v0_review import (ReviewInvalid, WholeReviewer, WholeTransport, public_execution,
                              repair_failure, review_prompt, validate_review)
from repair.v0_synth import allocate_seed, annotate, candidates, edit_kind, validate_candidate
from sed.config import load_config
from sed.storage import digest, file_hash, write_json
from train.dataset import EMPTY_THOUGHT
from test_synth_tasks import small_game, template_seeds, FakeEnv, make_seed_receipt
from repair import synth_tasks as factory


SEED = 'train/pick_and_place_simple-Mug-None-CounterTop-1/trial_0'
TASK = 'train/synth/test'


def task_row(tid=TASK, generator='G1', bin_name='near', probe=False):
    return dict(task_id=tid, id=tid.split('/')[-1], parent_seed_task_id=SEED,
        generator=generator, bin=bin_name, group_id='synth/' + tid.split('/')[-1],
        task_type='pick_and_place_simple', split='holdout' if probe else 'train', sealed=probe,
        heldout_probe=probe, status='valid', item_fingerprint=digest(tid),
        edit=dict(destination_receptacle_type='CounterTop'))


def manifest_at(tmp_path, rows=None):
    path = tmp_path / 'manifest.json'
    write_json(path, dict(schema='v0-synthetic-tasks/1', seed_task_ids=[SEED], tasks=rows or [task_row()]))
    return SyntheticManifest(path, [SEED])


def failure():
    execution = execute_actions(FakeEnvironment, TASK, O, 40)
    return dict(attempt_id='f1', episode_id='student-f1', task_id=TASK, execution=execution.to_json(),
        executors=['student'] * len(O), prefix_length=0, success=False,
        decoding='greedy', sampling_seed=None, heldout_probe=False)


def review_value(execution, action=N[3], *, index=3):
    return dict(plan=[dict(subgoal='Find the apple', evidence_steps=[0])],
        preservable_choices=[dict(choice='Keep the route', evidence_steps=[1])],
        breaking_points=[dict(step=index, category='missing_precondition', reason='Cleaning is missing')],
        patch=dict(original_trajectory_id=execution.trajectory_id, blocks=[dict(block_id='p1', op='replace',
            start=index, end_exclusive=index + 1, actions=[action])]))


class ReviewTransport:
    def __init__(self, values):
        self.values, self.prompts = iter(values), []
    def __call__(self, prompt):
        self.prompts.append(prompt)
        return next(self.values)


class Pool:
    def __init__(self, empty=()):
        self.n = 0
        self.empty = empty
    def next_valid(self, generator, bin_name, restriction=None):
        if generator in self.empty:
            return None
        self.n += 1
        return dict(task_row('train/synth/' + str(self.n), generator, bin_name), id=str(self.n))


def test_quotas_distance_bins_probes_parent_and_disjointness(tmp_path):
    tasks, allocations = allocate_seed(Pool())
    assert len(tasks) == 17
    for bin_name, quota in QUOTAS.items():
        assert sum(t['bin'] == bin_name and not t['heldout_probe'] for t in tasks) == quota
        assert sum(t['bin'] == bin_name and t['heldout_probe'] for t in tasks) == 1
    near = [t['generator'] for t in tasks if t['bin'] == 'near' and not t['heldout_probe']]
    assert near == ['G1', 'G3'] * 3
    assert all(t['parent_seed_task_id'] == SEED for t in tasks)
    assert len({t['item_fingerprint'] for t in tasks}) == 17
    manifest_at(tmp_path, tasks)
    assert not any(a['fallback_generator'] for a in allocations)


@pytest.mark.parametrize('empty', [('G1',), ('G3',), ('G1', 'G3')])
def test_near_fallback_and_explicit_shortfall(empty):
    tasks, allocation = allocate_seed(Pool(empty))
    near = [t for t in tasks if t['bin'] == 'near']
    assert len(near) == (0 if len(empty) == 2 else 7)
    assert all(t['generator'] not in empty for t in near)
    assert any(a['fallback_reason'] for a in allocation)


def test_same_type_mid_cross_type_far_and_g5_composite():
    parent = factory.Seed(SEED, small_game(), [], {})
    seeds = template_seeds()
    for bin_name in ('mid', 'far'):
        rows = list(candidates(parent, seeds, 'G2', bin_name, 0))
        assert rows
        for game, row in rows:
            assert (row['fingerprint']['tuple'][0] == 'pick_and_place_simple') == (bin_name == 'mid')
            assert factory.init_facts(game) == factory.init_facts(parent.game)
    composite = list(candidates(parent, seeds, 'G5', 'very_far', 0))
    assert composite
    valid = [(g, r) for g, r in composite if g is not None]
    assert valid
    for game, row in valid:
        assert row['parent_seed_task_id'] == SEED
        assert row['generator'] == 'G5'
        assert row['fingerprint']['tuple'][0] != 'pick_and_place_simple'
        assert factory.section(game['pddl_problem'], 'goal') != factory.section(parent.game['pddl_problem'], 'goal')
        assert set(factory.init_facts(game)) != set(factory.init_facts(parent.game))
        obj = row['edit']['object']
        assert factory.fact_map(factory.init_facts(game), 'objectType')[obj] == row['edit']['goal']['object_type'] + 'Type'
    assert [(r['id'] if g else r['edit']) for g, r in composite] == [
        (r['id'] if g else r['edit']) for g, r in candidates(parent, seeds, 'G5', 'very_far', 0)]


@pytest.mark.parametrize('won,drift', [(True, False), (False, False), (True, True)])
def test_g5_requires_executed_witness_and_determinism(tmp_path, won, drift):
    root, source, game_path, _ = make_seed_receipt(tmp_path)
    parent = factory.Seed(SEED, small_game(), ['finish'], dict(game_path=str(game_path), game_sha256=file_hash(game_path)))
    class Env(FakeEnv):
        resets = 0
        def reset(self, task):
            type(self).resets += 1
            result = super().reset(task)
            if drift and task.startswith('train/synth/'):
                result[0]['observation'] = str(self.resets) + '\n' + result[0]['observation']
            return result
        def replan(self, infos):
            return ['finish'] if won else ['look']
    row = dict(task_row(generator='G5', bin_name='very_far'),
        task_text='Your task is to: put a mug in countertop.', fingerprint=factory.fingerprint(SEED, small_game()))
    result = validate_candidate(row, parent, tmp_path / 'out', root, 0, {}, Env)
    assert (result['status'] == 'valid') == (won and not drift)
    assert result['validation']['environment_transitions'] >= 3


@pytest.mark.parametrize('tid', ['train/other/trial', 'valid_seen/x/y', 'train/synth/unknown', 'train/synth/../escape'])
def test_attempt_authorization_before_env_or_model(tmp_path, tid):
    manifest = manifest_at(tmp_path)
    def forbidden():
        raise AssertionError('must not construct')
    with pytest.raises(ValueError, match='manifest'):
        attempt(manifest, tid, decoding='greedy', sampling_seed=None, env_factory=forbidden,
                student_factory=forbidden, directory=tmp_path / 'attempt')


@pytest.mark.parametrize('mutation', ['parent', 'probe', 'duplicate', 'group', 'path'])
def test_manifest_rejects_boundary_violations(tmp_path, mutation):
    rows = [task_row(), task_row('train/synth/probe', probe=True)]
    if mutation == 'parent':
        rows[0]['parent_seed_task_id'] = 'train/synth/other'
    elif mutation == 'probe':
        rows[1]['sealed'] = False
    elif mutation == 'duplicate':
        rows[1]['item_fingerprint'] = rows[0]['item_fingerprint']
    elif mutation == 'group':
        rows[1]['group_id'] = rows[0]['group_id']
    else:
        rows[0]['task_id'] = 'train/synth/..'
    with pytest.raises(ValueError):
        manifest_at(tmp_path, rows)


def test_attempt_g3_replays_recorded_replies_and_empty_thought_context_only(tmp_path):
    row = task_row(generator='G3')
    replies = ['Thought: Seed reply exactly.\nAction: ' + O[0], EMPTY_THOUGHT + '\nAction: look']
    trace = execute_actions(FakeEnvironment, TASK, replies, 40)
    path = tmp_path / 'prefix.json'
    write_json(path, dict(task_id=TASK, prefix_replies=replies, initial_observation=trace.initial['visible'],
        transitions=[dict(action=s['action_parsed'], observation=s['visible']) for s in trace.steps]))
    row.update(prefix_path='prefix.json', prefix_sha256=file_hash(path))
    manifest = manifest_at(tmp_path, [row])
    captured = []
    class Student(FakeStudent):
        def generate(self, history, task, visible):
            captured.append(deepcopy(history))
            return super().generate(history, task, visible)
    student = Student(replies=['Action: ' + a for a in N[1:]])
    result = attempt(manifest, TASK, decoding='greedy', sampling_seed=None, env_factory=FakeEnvironment,
                     student_factory=lambda: student, directory=tmp_path / 'attempt')
    # The shared visibility layer strips empty channel markers before prompting;
    # execution evidence retains the exact EMPTY_THOUGHT reply.
    assert [h['reply'] for h in captured[0]] == [replies[0], 'Action: look']
    assert result['execution']['steps'][1]['action_raw'] == replies[1]
    assert result['success'] is True and result['prefix_length'] == 2
    rows, excluded = decision_rows(result, manifest, bucket='S', episode_ref='attempt.json')
    assert excluded['prefix_context_only'] == 2
    assert all(r['evidence']['step'] >= 2 for r in rows)
    assert all(r['supervised_assistant_indices'] == [len(r['messages']) - 1] for r in rows)


@pytest.mark.parametrize('mutation', ['keys', 'category', 'evidence', 'boolean_index', 'three_blocks', 'wrapper', 'prefix', 'id'])
def test_review_strict_schema(tmp_path, mutation):
    trace = TrajectoryRecord.from_json(failure()['execution'])
    value = review_value(trace)
    prefix = 0
    if mutation == 'keys':
        value['success'] = True
    elif mutation == 'category':
        value['breaking_points'][0]['category'] = 'unregistered'
    elif mutation == 'evidence':
        value['plan'][0]['evidence_steps'] = [99]
    elif mutation == 'boolean_index':
        value['patch']['blocks'][0]['start'] = True
    elif mutation == 'three_blocks':
        value['patch']['blocks'] *= 3
    elif mutation == 'wrapper':
        value['patch']['blocks'][0]['actions'] = ['Action: look']
    elif mutation == 'prefix':
        prefix = 4
    else:
        value['patch']['original_trajectory_id'] = 'wrong'
    with pytest.raises(ReviewInvalid):
        validate_review(json.dumps(value), trace, prefix)


def test_review_one_format_retry_and_durable_resume(tmp_path):
    trace = TrajectoryRecord.from_json(failure()['execution'])
    transport = ReviewTransport(['bad JSON', json.dumps(review_value(trace))])
    reviewer = WholeReviewer(transport, tmp_path)
    result, attempts = reviewer.request(trace, 0, [])
    assert result['patch'] and len(attempts) == 2
    assert transport.prompts[1].retry_reason
    again, _ = reviewer.request(trace, 0, [])
    assert again == result and len(transport.prompts) == 2


def test_two_format_failures_stop(tmp_path):
    trace = TrajectoryRecord.from_json(failure()['execution'])
    reviewer = WholeReviewer(ReviewTransport(['{}', '{}']), tmp_path)
    result, calls = reviewer.request(trace, 0, [])
    assert result is None and len(calls) == 2


def test_suffix_wins_without_calling_handback_and_edit_marks(tmp_path):
    original = failure()
    trace = TrajectoryRecord.from_json(original['execution'])
    reviewer = WholeReviewer(ReviewTransport([json.dumps(review_value(trace))]), tmp_path / 'requests')
    def forbidden():
        raise AssertionError('suffix wins; no handback')
    record = repair_failure(original, reviewer=reviewer, env_factory=FakeEnvironment, student_factory=forbidden,
        ledger=BudgetLedger(tmp_path / 'environment.jsonl'), directory=tmp_path / 'repair')
    assert record['success'] is True and len(record['rounds']) == 1
    assert record['action_alignment']['distance'] == 1
    assert record['executors'] == ['student'] * 3 + ['patch'] + ['student'] * 2
    rows, _ = decision_rows(record, manifest_at(tmp_path), bucket='R', episode_ref='repair.json')
    assert len(rows) == len(N)
    assert [r['evidence']['step'] for r in rows if r['v0']['edit_decision']] == [3]
    assert rows[3]['evidence']['label_source'] == 'action_only_no_label'


def test_suffix_before_handback_and_correct_branch_state(tmp_path):
    events = []
    class Env(FakeEnvironment):
        def step(self, action):
            events.append(('env', action))
            return super().step(action)
    class Student(FakeStudent):
        def generate(self, history, task, visible):
            events.append(('student', len(history)))
            return super().generate(history, task, visible)
    original = failure()
    trace = TrajectoryRecord.from_json(original['execution'])
    reviewer = WholeReviewer(ReviewTransport([json.dumps(review_value(trace, action='look'))]), tmp_path / 'requests')
    record = repair_failure(original, reviewer=reviewer, env_factory=Env,
        student_factory=lambda: Student(replies=['Action: ' + a for a in [N[3], O[4], O[5]]]),
        ledger=BudgetLedger(tmp_path / 'environment.jsonl'), directory=tmp_path / 'repair')
    # Environment class identity is bound by reset verification.
    assert record['execution']['error_type'] == 'v0_prefix_replay_mismatch'
    # Repeat with the same environment type as the original collection.
    original['execution'] = execute_actions(Env, TASK, O, 40).to_json()
    trace = TrajectoryRecord.from_json(original['execution'])
    events.clear()
    reviewer = WholeReviewer(ReviewTransport([json.dumps(review_value(trace, action='look'))]), tmp_path / 'requests2')
    record = repair_failure(original, reviewer=reviewer, env_factory=Env,
        student_factory=lambda: Student(replies=['Action: ' + a for a in [N[3], O[4], O[5]]]),
        ledger=BudgetLedger(tmp_path / 'environment2.jsonl'), directory=tmp_path / 'repair2')
    assert record['success'] is True
    first_call = next(i for i, event in enumerate(events) if event[0] == 'student')
    assert ('env', O[-1]) in events[:first_call]  # Original suffix was actually executed first.
    assert events[first_call] == ('student', 4)  # Branch after the edited block, not after suffix.
    assert record['executors'][4:] == ['handback-student'] * 3


def test_three_round_cap_previous_executions_and_won_only(tmp_path):
    class Reviewer:
        calls = []
        def request(self, current, prefix, previous):
            self.calls.append(review_prompt(current, prefix, previous))
            return review_value(current, action='look'), []
    reviewer = Reviewer()
    record = repair_failure(failure(), reviewer=reviewer, env_factory=FakeEnvironment,
        student_factory=FakeStudent, ledger=BudgetLedger(tmp_path / 'environment.jsonl'), directory=tmp_path / 'repair')
    assert len(record['rounds']) == len(reviewer.calls) == 3
    assert record['success'] is False and not record['verified']
    assert [len(p.payload['previous_rounds']) for p in reviewer.calls] == [0, 1, 2]
    assert len(reviewer.calls[-1].payload['previous_rounds'][0]['executions']) == 2


def test_review_prompt_hidden_state_exclusion_and_transform_rules():
    trace = TrajectoryRecord.from_json(failure()['execution'])
    body = trace.to_json()
    body.pop('trajectory_id')
    body['initial'].update(witness='SECRET_WITNESS', planner='SECRET_PLANNER', internal_state='SECRET_STATE')
    body['initial']['visible']['extra.expert_plan'] = 'SECRET_EXPERT'
    trace = TrajectoryRecord.from_json(body)
    prompt = json.dumps(review_prompt(trace).messages())
    for secret in ('SECRET_WITNESS', 'SECRET_PLANNER', 'SECRET_STATE', 'SECRET_EXPERT'):
        assert secret not in prompt
    facts = public_execution(trace)['steps'][0]['facts_before']
    assert facts['object_locations'] == {}
    assert facts['receptacles'] == {}
    assert 'heat <obj> with <microwave>' in prompt


def test_refused_and_repeated_refused_excluded_from_successful_rows(tmp_path):
    class RefusalEnv(FakeEnvironment):
        def step(self, action):
            if action == 'look':
                return self.state('Nothing happens.')
            return super().step(action)
    actions = ['look', 'look', *N]
    execution = execute_actions(RefusalEnv, TASK, actions, 40)
    assert execution.success is True
    episode = dict(episode_id='repair', task_id=TASK, prefix_length=0, success=True,
                   execution=execution.to_json(), executors=['student'] * len(actions))
    rows, excluded = decision_rows(episode, manifest_at(tmp_path), bucket='R', episode_ref='r.json')
    assert all(r['evidence']['step'] not in (0, 1) for r in rows)
    assert excluded['refused_or_inadmissible'] == 2
    assert all(not r['v0']['edit_decision'] for r in rows)


def test_prefix_consistent_label_has_no_future_observations(tmp_path):
    execution = execute_actions(FakeEnvironment, TASK, N, 40)
    episode = dict(episode_id='repair', task_id=TASK, prefix_length=0, success=True,
        execution=execution.to_json(), executors=['student'] * 3 + ['patch'] + ['student'] * 2)
    prompts = []
    def labeler(prompt):
        prompts.append(prompt)
        return json.dumps(dict(reasoning='Use the visible apple.', evidence_steps=[2], unsupported=False))
    rows, _ = decision_rows(episode, manifest_at(tmp_path), bucket='R', episode_ref='r.json', labeler=labeler)
    assert len(prompts) == 1 and len(prompts[0].payload['prefix']) == 3
    assert 'You executed put' not in json.dumps(prompts[0].payload)
    assert rows[3]['evidence']['label_source'] == 'teacher_rationale'
    assert rows[4]['messages'][8]['content'].startswith('Thought: Use the visible apple.')


def test_success_choice_fewest_turns_then_id_and_probe_never_trained():
    rows = [dict(task_id=TASK, success=True, heldout_probe=False, episode_id=eid,
                 execution=dict(steps=[{}] * n)) for eid, n in [('z', 4), ('a', 4), ('b', 5)]]
    rows.append(dict(rows[0], heldout_probe=True, episode_id='probe', task_id='train/synth/probe'))
    selected = choose_episodes(rows, [])
    assert len(selected) == 1 and selected[0][1]['episode_id'] == 'a'


def test_failed_selection_greedy_first_then_sampling_order():
    rows = [dict(failure(), attempt_id=str(i), decoding=d, sampling_seed=s)
            for i, (d, s) in enumerate([('sampled', 2), ('sampled', 1), ('greedy', None), ('sampled', 0)])]
    chosen = selected_failures(rows)
    assert [(r['decoding'], r['sampling_seed']) for r in chosen] == [('greedy', None), ('sampled', 0)]


def test_exact_probe_interval_unit_is_task(tmp_path):
    tasks = [task_row(f'train/synth/p{i}', probe=True) for i in range(20)]
    rows = [dict(task_id=t['task_id'], decoding=d, sampling_seed=s, success=True)
            for t in tasks for d, s in [('greedy', None), ('sampled', 0), ('sampled', 1), ('sampled', 2)]]
    report = probe_report(tasks, rows)['by_bin']['near']
    assert report['known'] == 20 and report['failures'] == 0
    assert report['cp95_upper'] == pytest.approx(1 - .025 ** (1/20))
    assert not report['stop_eligible']


def test_allocate_preserves_total_and_rejects_negative_consistency():
    assert allocate(5, {'a': 1, 'b': 1, 'c': 1}) == {'a': 2, 'b': 2, 'c': 1}
    with pytest.raises(ValueError):
        allocate(5, {'a': -1, 'b': 2})


def test_real_capped_transport_fake_client_and_ledger_hard_stop(tmp_path):
    class Client(FakeReviewClient):
        calls = 0
        def create(self, **kwargs):
            self.calls += 1
            payload = json.loads(kwargs['input'][-1]['content'])
            trace = TrajectoryRecord.from_json(failure()['execution'])
            value = review_value(trace)
            assert payload['original_trajectory_id'] == trace.trajectory_id
            result = SimpleNamespace(output_text=json.dumps(value), model='FAKE_NO_MODEL', id='fake', status='completed',
                usage={'input_tokens': 100, 'output_tokens': 80, 'output_tokens_details': {'reasoning_tokens': 0}})
            return SimpleNamespace(status_code=200, headers={}, parse=lambda: result)
    config = load_config('configs/base_e4b.yaml')
    client = Client()
    budget = SharedTeacherBudget(tmp_path / 'budget.jsonl', input_tokens=1000000, output_tokens=10000)
    transport = WholeTransport(config, TASK, tmp_path / 'teacher', budget, client=client)
    trace = TrajectoryRecord.from_json(failure()['execution'])
    value, _ = WholeReviewer(transport, tmp_path / 'requests').request(trace, 0, [])
    assert value and client.calls == 1
    assert budget.summary()['reported_usage']['input_tokens'] == 100
    assert budget.summary()['reported_usage']['output_tokens'] == 80
    small = SharedTeacherBudget(tmp_path / 'small.jsonl', input_tokens=1, output_tokens=4096)
    blocked = WholeTransport(config, TASK, tmp_path / 'blocked', small, client=client)
    with pytest.raises(Exception):
        WholeReviewer(blocked, tmp_path / 'blocked_requests').request(trace, 0, [])
    assert client.calls == 1 and small.summary()['stop_reason']
    assert not (tmp_path / 'blocked_requests').exists()


def test_phase_resume_identity_and_evidence_tamper(tmp_path):
    directory = tmp_path / 'phase'
    identity = dict(inputs={}, decodings=[('greedy', None)])
    phase(directory, identity)
    write_json(directory / 'evidence.json', {'ok': True})
    finish(directory)
    phase(directory, identity, resume=True)
    assert completed(directory)
    with pytest.raises(ValueError):
        phase(directory, dict(identity, seed=1), resume=True)
    write_json(directory / 'evidence.json', {'ok': False})
    with pytest.raises(ValueError, match='artifacts'):
        completed(directory)


def stalled_validation_worker(connection, row, parent, directory, source_root, seed, cache, journal):
    import time
    from repair.synth_validation import offline
    from sed.storage import JsonlLedger
    offline()
    JsonlLedger(journal).append(dict(event='step_started', environment_steps=1))
    time.sleep(10)


def test_native_validation_timeout_retains_unknown_transition(tmp_path):
    from repair.v0_validation import isolated_validation
    parent = factory.Seed(SEED, small_game(), [], {})
    row = task_row()
    result = isolated_validation(row, parent, tmp_path / 'out', tmp_path, 0, {},
                                 timeout=1, worker=stalled_validation_worker)
    assert result['status'] == 'rejected'
    assert result['rejection'] == 'validation_timeout_unknown_feasibility'
    assert result['validation']['environment_transitions'] == 1
    assert result['validation']['environment_transitions_unknown'] == 1
    assert result['validation_journals']
    assert json.loads((tmp_path / 'out' / result['witness_path']).read_text())['never_for_training']


def test_validation_journal_records_actual_resets_and_steps(tmp_path):
    from repair.v0_validation import JournalExecution
    journal = tmp_path / 'journal.jsonl'
    execution = JournalExecution(0, FakeEnv, journal=journal)
    execution.reset(tmp_path, TASK, 'fake_witness')
    assert execution.plan() == ['finish']
    execution.replay(['finish'])
    execution.close()
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [r['event'] for r in rows] == ['reset_started', 'reset_completed', 'planner_started',
        'planner_completed', 'step_started', 'step_completed']
    assert rows[-1]['entry']['won'] is True
