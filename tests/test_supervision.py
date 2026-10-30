"""E0.4 synthetic records/fake transports; no real model, environment or API."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from repair.artifacts import seal
from repair.facts import FactTable
from repair.prompts import rationale_prompt
from repair.supervision import (ARMS, InvalidExample, RationaleLabeler, SplitPolicy,
    build_decisions, build_trajectory, fact_state_hash, inspect_steps,
    seal_dataset, trajectory_diagnostics, trajectory_messages, validate_example)
from repair.takeover import SharedTeacherBudget
from sed.config import load_config
from sed.storage import BlobStore, canonical_bytes, digest, file_hash, write_json
from tools.build_v2_dataset import main

TASK = 'train/pick_and_place_simple-Apple-None-CounterTop-1/trial_0'
K20 = 'train/pick_and_place_simple-Apple-None-CounterTop-2/trial_0'
INITIAL = ('You are at cabinet 1. On the cabinet 1, you see nothing. '
           'There is a countertop 1. '
           'You are not carrying anything.\nYour task is to: put an apple in countertop.')


def visible(text, commands):
    return dict(observation=text, admissible_commands=commands, audit='SECRET_AUDIT',
                policy_commands=['SECRET_POLICY'], facts='SECRET_PDDL')


def record(actions, observations, *, initial=INITIAL, success=False, executors=None,
           thoughts=None, branch=0, mode='G', task_id=TASK):
    commands = sorted(set(actions))
    execution = dict(task_id=task_id, initial={'visible': visible(initial, commands)},
        success=success, termination_reason='SUCCESS' if success else 'STEP_LIMIT', steps=[])
    decisions = []
    for i, (action, obs) in enumerate(zip(actions, observations)):
        raw = 'Thought: ' + (thoughts[i] if thoughts else 'Use the public evidence.') + '\nAction: ' + action
        execution['steps'].append(dict(action_raw=raw, action_parsed=action, action_outcome='legal',
            environment_steps=1, visible=visible(obs, commands), terminated=success and i == len(actions) - 1,
            truncated=False, audit={'won': 'SECRET_WON'}, feedback_source='real_environment'))
        if i >= branch:
            decisions.append(dict(step=i, executed_reply=raw, executed_action=action,
                executor=executors[i] if executors else 'student', review={'guidance': 'SECRET_GUIDANCE'}))
    return dict(task_id=task_id, mode=mode, execution=execution, steps=decisions, branch_index=branch,
                success=success, void=False, prefix_verified=True, status='completed')


def policy_for(*tasks):
    return SplitPolicy({'sets': {'V2_pool': {'training_eligible': True, 'task_ids': list(tasks or [TASK])}}}, [K20])


def build(row, arm='JOINT', **kwargs):
    return build_decisions(row, arm=arm, episode_ref='run/records/key.json', policy=policy_for(row['task_id']), **kwargs)


def label(reason='Use the public evidence.', steps=None, unsupported=False):
    return json.dumps(dict(reasoning=reason, evidence_steps=[-1] if steps is None else steps, unsupported=unsupported))


@pytest.mark.parametrize('action,obs,event', [
    ('go to cabinet 1', 'On the cabinet 1, you see an apple 1.', 'target_seen'),
    ('open cabinet 1', 'You open the cabinet 1. In it, you see an apple 1.', 'target_seen'),
    ('examine cabinet 1', 'On the cabinet 1, you see an apple 1.', 'target_seen'),
    ('look', 'On the cabinet 1, you see an apple 1.', 'target_seen'),
    ('take apple 1 from cabinet 1', 'You pick up the apple 1.', 'target_taken'),
    ('heat apple 1 with microwave 1', 'The apple 1 is hot.', 'transform_done'),
    ('cool apple 1 with fridge 1', 'The apple 1 is cool.', 'transform_done'),
    ('clean apple 1 with sinkbasin 1', 'The apple 1 is clean.', 'transform_done'),
    ('move apple 1 to countertop 1', 'You put the apple 1 in countertop 1.', 'placed'),
    ('put apple 1 on countertop 1', 'You put the apple 1 on countertop 1.', 'placed'),
])
def test_public_local_event_rules(action, obs, event):
    examples, _ = build(record([action], [obs]))
    assert len(examples) == 1 and examples[0]['evidence']['local_event'] == event
    assert examples[0]['source'] == 'joint_local' and not examples[0]['evidence']['episode_success']
    assert 'SECRET' not in json.dumps(examples[0]['messages'])


def test_target_seen_only_exploration_first_sighting_and_new_instance():
    row = record(['inventory', 'look', 'look', 'look'], ['On the cabinet 1, you see an apple 1.',
        'On the cabinet 1, you see an apple 1.', 'On the cabinet 1, you see an apple 2.',
        'On the cabinet 1, you see an apple 2.'])
    assert [e['evidence']['step'] for e in build(row)[0]] == [2]
    row = record(['look'], ['On the cabinet 1, you see an apple 1.'], initial=INITIAL.replace('nothing', 'an apple 1'))
    assert build(row)[0] == []


@pytest.mark.parametrize('next_refused,has_take,want', [(False, True, True), (True, True, False), (False, False, False)])
def test_hands_freed_requires_known_occupied_and_next_target_take(next_refused, has_take, want):
    actions, obs = ['put mug 1 on countertop 1'], ['You put the mug 1 on countertop 1.']
    if has_take:
        actions += ['take apple 1 from cabinet 1', 'take apple 1 from cabinet 1']
        obs += ['Nothing happens.' if next_refused else 'You pick up the apple 1.', 'You pick up the apple 1.']
    row = record(actions, obs, initial=INITIAL.replace('You are not carrying anything.', 'You are carrying: mug 1.'))
    rows = inspect_steps(row['execution'])
    assert ('hands_freed' in rows[0]['local_events']) is want
    if want:
        assert rows[0]['next_legal_target_take'] == 1
    row['execution']['initial']['visible']['observation'] = INITIAL.replace('You are not carrying anything.', '')
    assert 'hands_freed' not in inspect_steps(row['execution'])[0]['local_events']


def test_placing_requires_state_of_same_instance_and_transform_delta():
    row = record(['move apple 1 to countertop 1', 'heat apple 2 with microwave 1',
        'move apple 1 to countertop 1', 'heat apple 1 with microwave 1', 'heat apple 1 with microwave 1',
        'take apple 1 from countertop 1', 'move apple 1 to countertop 1'], ['Done.'] * 7,
        initial=INITIAL.replace('put an apple', 'put a hot apple'))
    rows = inspect_steps(row['execution'])
    assert 'placed' not in rows[0]['local_events'] and 'placed' not in rows[2]['local_events']
    assert 'transform_done' in rows[3]['local_events'] and 'transform_done' not in rows[4]['local_events']
    assert 'placed' in rows[6]['local_events']


def test_lamp_requires_target_holding_and_new_lit_delta():
    row = record(['use desklamp 1', 'take apple 1 from cabinet 1', 'use desklamp 1', 'use desklamp 1'],
        ['The light is on.', 'You pick up the apple 1.', 'The light is on.', 'The light is on.'],
        initial=INITIAL.replace('put an apple in countertop', 'examine an apple with a desklamp'))
    assert ['lamp_used_with_target' in r['local_events'] for r in inspect_steps(row['execution'])] == [False, False, True, False]


def test_success_union_dedup_refused_and_repeated_refused_targets():
    row = record(['look', 'take apple 1 from cabinet 1', 'take apple 1 from cabinet 1',
        'move apple 1 to countertop 1'], ['On the cabinet 1, you see an apple 1.', 'Nothing happens.',
        'You pick up the apple 1.', 'You put the apple 1 in countertop 1.'], success=True)
    examples, excluded = build(row)
    assert [e['evidence']['step'] for e in examples] == [0, 3]
    assert len({e['example_id'] for e in examples}) == 2
    assert excluded['refused_or_inadmissible'] == excluded['repeated_refused_action'] == 1
    assert examples[-1]['evidence']['local_events'] == ['placed', 'task_success']
    assert examples[-1]['messages'][4]['content'] == row['execution']['steps'][1]['action_raw']
    assert examples[-1]['supervised_assistant_indices'] == [len(examples[-1]['messages']) - 1]
    complete, _ = build(row, 'JOINT_COMPLETE_ONLY')
    assert [e['evidence']['step'] for e in complete] == [0, 3]
    assert {e['source'] for e in complete} == {'joint_complete'}
    row['success'] = row['execution']['success'] = False
    row['execution']['termination_reason'] = 'STEP_LIMIT'
    assert build(row, 'JOINT_COMPLETE_ONLY')[0] == [] and len(build(row)[0]) == 2


def test_no_progress_hash_ignores_evidence_time_but_retains_state():
    row = record(['look', 'look', 'open cabinet 1'], [INITIAL, INITIAL, 'You open the cabinet 1.'], success=True)
    assert inspect_steps(row['execution'])[1]['no_progress_repeat']
    examples, reasons = build(row, 'JOINT_COMPLETE_ONLY')
    assert [e['evidence']['step'] for e in examples] == [0, 2] and reasons['no_progress_repeat'] == 1
    table = FactTable(INITIAL); before = fact_state_hash(table)
    table.update({'action_parsed': 'look', 'action_outcome': 'legal', 'observation': INITIAL})
    assert fact_state_hash(table) == before
    table.update({'action_parsed': 'open cabinet 1', 'action_outcome': 'legal', 'observation': 'You open the cabinet 1.'})
    assert fact_state_hash(table) != before


def test_inadmissible_and_nonexecuted_never_targets():
    row = record(['look', 'open cabinet 1'], ['On the cabinet 1, you see an apple 1.', 'You open the cabinet 1.'], success=True)
    row['execution']['initial']['visible']['admissible_commands'] = ['inventory']
    row['execution']['steps'][0].update(action_outcome='inadmissible', environment_steps=0)
    assert [e['evidence']['step'] for e in build(row)[0]] == [1]
    row['execution']['steps'][1]['environment_steps'] = 0
    assert build(row)[0] == []


def test_guidance_removed_history_labels_are_prefix_only_and_once_per_step():
    row = record(['look', 'take apple 1 from cabinet 1', 'move apple 1 to countertop 1'],
        ['On the cabinet 1, you see an apple 1.\n\nGuidance: SECRET_HINT', 'You pick up the apple 1.', 'FUTURE_FINAL'],
        initial=INITIAL + '\n\nGuidance: SECRET_INITIAL', success=True,
        executors=['teacher', 'student_after_guidance', 'student'],
        thoughts=['SECRET_REVIEW', 'The hint says take it.', 'As sUgGeStEd, place it.'])
    original, requests, audits = deepcopy(row), [], []
    def fake(prompt):
        requests.append(prompt.to_json()); return label()
    examples, _ = build(row, labeler=fake, audit=audits.append)
    assert row == original and len(requests) == len(audits) == 3
    assert all(e['evidence']['label_source'] == 'teacher_rationale' for e in examples)
    assert 'FUTURE_FINAL' not in json.dumps(requests)
    assert 'apple 1' not in requests[0]['payload']['initial_observation']
    assert not any(s in json.dumps(requests) for s in ('Guidance:', 'SECRET', 'episode_success', 'audit'))
    for e in examples:
        assert all('Guidance:' not in m['content'] for m in e['messages'])
        assert all(m['content'].startswith('Thought: Use the public evidence.\nAction: ')
                   for m in e['messages'] if m['role'] == 'assistant')
        validate_example(e, policy_for(), source_record=row)


@pytest.mark.parametrize('raw', [label('The apple 99 is here.'), label(steps=[0]), label(unsupported=True),
                                label('Follow the hint.'), label('x' * 65), label('Action: look'), 'not json'])
def test_label_rejections_action_only_and_recorded(raw):
    row = record(['look'], ['On the cabinet 1, you see an apple 1.'], executors=['teacher'])
    audit = []
    examples, excluded = build(row, labeler=lambda p: raw, audit=audit.append)
    assert examples[0]['messages'][-1]['content'] == 'Action: look'
    assert examples[0]['evidence']['label_source'] == 'action_only_invalid_label'
    assert excluded['invalid_rationale_label'] == 1 and audit[0]['raw_output'] == raw
    assert examples[0]['evidence']['label_check'] != 'passed'


def test_no_label_and_student_own_thought_handling():
    row = record(['look', 'take apple 1 from cabinet 1'], ['On the cabinet 1, you see an apple 1.', 'Taken.'],
                 executors=['teacher', 'student'], thoughts=['Teacher check.', 'I can take apple 1.'])
    examples, _ = build(row)
    assert examples[0]['evidence']['label_source'] == 'action_only_no_label'
    assert examples[1]['messages'][2]['content'] == 'Action: look'
    assert examples[1]['messages'][-1]['content'] == row['steps'][1]['executed_reply']
    row['steps'][0]['executor'] = 'student'
    row['steps'][0]['executed_reply'] = row['execution']['steps'][0]['action_raw'] = 'Thought: apple 99 is here.\nAction: look'
    assert build(row)[0][0]['evidence']['label_source'] == 'action_only_ungrounded_student'


def test_prefix_expert_successful_only_teacher_suffix_and_own_replies():
    row = record(['look', 'look', 'take apple 1 from cabinet 1', 'move apple 1 to countertop 1'],
        ['On the cabinet 1, you see an apple 1.', 'Nothing happens.', 'Taken.', 'Placed.'],
        branch=1, mode='takeover', success=True)
    examples, reasons = build(row, 'PREFIX_EXPERT', labeler=lambda p: pytest.fail('No L in PREFIX_EXPERT'))
    assert [e['evidence']['step'] for e in examples] == [2, 3]
    assert reasons['refused_or_inadmissible'] == reasons['student_prefix_context_only'] == 1
    for e in examples:
        assert e['evidence']['label_source'] == 'teacher_reply'
        assert e['messages'][-1]['content'] == row['execution']['steps'][e['evidence']['step']]['action_raw']
        assert e['messages'][2]['content'] == row['execution']['steps'][0]['action_raw']
    for key, value in [('success', False), ('void', True), ('prefix_verified', False)]:
        bad = deepcopy(row); bad[key] = value
        assert build(bad, 'PREFIX_EXPERT')[0] == []


def test_validation_catches_action_grounding_mask_observation_and_id():
    row = record(['look'], ['On the cabinet 1, you see an apple 1.'])
    original = build(row)[0][0]
    for reply in ['Action: inventory', 'Thought: apple 99 exists.\nAction: look', 'Guidance: yes\nAction: look', 'bad reply']:
        e = deepcopy(original); e['messages'][-1]['content'] = reply
        with pytest.raises(InvalidExample): validate_example(e, policy_for(), source_record=row)
    for field, value in [('supervised_assistant_indices', []), ('example_id', 'fake'), ('task_id', K20)]:
        e = deepcopy(original); e[field] = value
        with pytest.raises(InvalidExample): validate_example(e, policy_for(), source_record=row)
    changed = deepcopy(original)
    changed['messages'][1]['content'] = canonical_bytes({'observation': 'Changed.', 'admissible_commands': ['look']}).decode()
    with pytest.raises(InvalidExample, match='executed_observation_mismatch'):
        validate_example(changed, policy_for(), source_record=row)


@pytest.mark.parametrize('arm', ARMS[:3])
@pytest.mark.parametrize('reply,reason', [
    ('Action: inventory', 'supervised_action_inadmissible'),
    ('Thought: apple 99 exists.\nAction: look', 'ungrounded_rationale'),
])
def test_decisions_still_reject_inadmissible_and_ungrounded_turns(arm, reply, reason):
    row = record(['look'], ['On the cabinet 1, you see an apple 1.'], success=True,
                 mode='takeover' if arm == 'PREFIX_EXPERT' else 'G')
    example = build(row, arm)[0][0]
    example['messages'][-1]['content'] = reply
    with pytest.raises(InvalidExample, match=reason):
        validate_example(example, policy_for(), source_record=row)


@pytest.mark.parametrize('name', ['U_dev', 'U_mech', 'D_probe', 'D_unseen_quick', 'dev140'])
def test_split_excludes_groups_across_trials_and_source_names(name):
    splits = {'sets': {'V2_pool': {'training_eligible': True, 'task_ids': [TASK]},
                      name: {'task_ids': [TASK.replace('trial_0', 'trial_9')], 'group_ids': []}}}
    with pytest.raises(InvalidExample, match='evaluation_group_overlap'): SplitPolicy(splits, [K20])
    splits['sets'].pop(name)
    splits['exclusion_sources'] = [dict(category=name, group_ids=['valid_seen/' + TASK.split('/')[1]])]
    with pytest.raises(InvalidExample, match='evaluation_group_overlap'): SplitPolicy(splits, [K20])
    with pytest.raises(InvalidExample, match='task_outside_frozen_view'):
        policy_for().check(TASK.replace('trial_0', 'trial_9'), 'joint_local')
    with pytest.raises(InvalidExample, match='non_train_task'):
        policy_for().check(TASK.replace('train/', 'valid_unseen/'), 'joint_local')


def as_episode(row):
    e = deepcopy(row['execution'])
    e.update(schema_version=2, complete=True, status='ok', success=True, split='train', sealed=False, episode_id=digest(row))
    for i, step in enumerate(e['steps']): step['event_id'] = str(i)
    return e


def compare_main_grid(episode, monkeypatch):
    import train.dataset as main_grid
    captured = []
    def capture(tokenizer, messages, reply, **kwargs):
        captured.append(deepcopy(messages) + [{'role': 'assistant', 'content': reply}])
        return dict(input_ids=[1, 2], labels=[-100, 2], attention_mask=[1, 1], supervised_tokens=1,
                    prompt_tokens=1, turn_forward_tokens=2)
    monkeypatch.setattr(main_grid, 'encode_turn', capture)
    main_grid.episode_examples(episode, object(), packing='turn')
    rendered = trajectory_messages(episode)
    for turn, messages in enumerate(captured):
        assert canonical_bytes(rendered[:3 + 2 * turn]) == canonical_bytes(messages)
    return rendered


@pytest.mark.parametrize('index', [0, 1])
def test_offline_continue_match_main_grid_two_trajectories(index, monkeypatch):
    actions = ['look', 'take apple 1 from cabinet 1'] if index == 0 else ['look', 'open cabinet 1', 'look']
    episode = as_episode(record(actions, ['On the cabinet 1, you see an apple 1.'] * len(actions), success=True))
    messages = compare_main_grid(episode, monkeypatch)
    offline = build_trajectory(episode, arm='OFFLINE', source='offline_teacher', episode_ref='episode.json', policy=policy_for())
    episode['task_id'] = K20
    replay = build_trajectory(episode, arm='CONTINUE', source='k20_replay', episode_ref='episode.json', policy=policy_for())
    assert canonical_bytes(offline['messages']) == canonical_bytes(replay['messages']) == canonical_bytes(messages)
    assert offline['supervised_assistant_indices'] == list(range(2, len(messages), 2))


@pytest.mark.parametrize('arm,source,task_id', [
    ('OFFLINE', 'offline_teacher', TASK),
    ('CONTINUE', 'k20_replay', K20),
    ('JOINT', 'k20_replay', K20),
])
@pytest.mark.parametrize('inadmissible,ungrounded', [(True, False), (False, True), (True, True)])
def test_trajectory_retains_teacher_turns_and_seals_diagnostics(
        arm, source, task_id, inadmissible, ungrounded, monkeypatch, tmp_path):
    episode = as_episode(record(['look', 'look'], ['Nothing happens.', 'Done.'],
        thoughts=['apple 99 and mug 88 exist.' if ungrounded else 'Use the public evidence.',
                  'Use the public evidence.'], task_id=task_id, success=True, executors=['teacher'] * 2))
    if inadmissible:
        episode['initial']['visible']['admissible_commands'] = ['inventory']
        episode['steps'][0]['action_outcome'] = 'inadmissible'
    original = deepcopy(episode)
    messages = compare_main_grid(episode, monkeypatch)
    example = build_trajectory(episode, arm=arm, source=source, episode_ref='episode.json', policy=policy_for())
    assert episode == original
    assert canonical_bytes(example['messages']) == canonical_bytes(messages)
    assert example['supervised_assistant_indices'] == [2, 4]
    manifest = seal_dataset(tmp_path / 'dataset', [example], policy=policy_for(), arm=arm,
        inputs={}, splits_sha256='frozen', theta0={}, excluded=Counter())
    expected = {example['example_id']: dict(inadmissible_supervised_turns=int(inadmissible),
        not_yet_seen_instance_supervised_turns=int(ungrounded))}
    assert manifest['trajectory_diagnostics'] == expected
    assert json.loads((tmp_path / 'dataset/manifest.json').read_text())['trajectory_diagnostics'] == expected
    assert json.loads((tmp_path / 'dataset/examples.jsonl').read_text()) == example
    assert manifest['excluded'] == [] and manifest['supervised_turns'] == 2


def test_trajectory_diagnostics_count_turns_using_only_prior_public_evidence():
    episode = as_episode(record(['look'] * 3, ['Nothing new.', 'You see apple 99 and mug 88.', 'Done.'],
        thoughts=['apple 99 and mug 88 exist.'] * 3, success=True))
    example = build_trajectory(episode, arm='OFFLINE', source='offline_teacher',
                               episode_ref='episode.json', policy=policy_for())
    assert trajectory_diagnostics([example]) == {example['example_id']: dict(
        inadmissible_supervised_turns=0, not_yet_seen_instance_supervised_turns=2)}


@pytest.mark.parametrize('reply,reason', [
    ('Thought: Guidance: look here.\nAction: look', 'guidance_or_message_fields'),
    ('No Action line', 'unparseable_prefix'),
])
def test_trajectory_still_rejects_guidance_and_unparseable_turns(reply, reason):
    episode = as_episode(record(['look'], ['Done.'], success=True))
    episode['steps'][0]['action_raw'] = reply
    with pytest.raises(InvalidExample, match=reason):
        build_trajectory(episode, arm='OFFLINE', source='offline_teacher',
                         episode_ref='episode.json', policy=policy_for())


def test_trajectory_still_rejects_changed_rendering_and_partial_supervision():
    episode = as_episode(record(['look', 'look'], ['Nothing new.', 'Done.'], success=True))
    example = build_trajectory(episode, arm='OFFLINE', source='offline_teacher',
                               episode_ref='episode.json', policy=policy_for())
    changed = deepcopy(example)
    changed['messages'][2]['content'] = 'Action: look'
    with pytest.raises(InvalidExample, match='executed_prefix_mismatch'):
        validate_example(changed, policy_for(), expected_messages=example['messages'])
    changed = deepcopy(example)
    changed['supervised_assistant_indices'] = [4]
    with pytest.raises(InvalidExample, match='supervision_mask'):
        validate_example(changed, policy_for())


@pytest.mark.parametrize('index', [0, 1])
def test_recorded_k20_rendering_matches_main_grid_if_available(index, monkeypatch):
    splits = json.loads(Path('data/manifest/method_v1_splits.json').read_text())
    root = Path(splits['rebuild_arguments']['verified_root'])
    if not (root / 'subsets/k20/manifest.json').is_file(): pytest.skip('External recorded collection unavailable')
    receipt = json.loads((root / 'subsets/k20/manifest.json').read_text())['verified'][index]
    compare_main_grid(json.loads((root / receipt['episode_dir'] / 'episode.json').read_text()), monkeypatch)


def test_deterministic_ids_seal_counts_tokens_and_exclusive_create(tmp_path):
    row = record(['look', 'take apple 1 from cabinet 1'], ['On the cabinet 1, you see an apple 1.', 'Taken.'], success=True)
    examples, excluded = build(row)
    assert examples == build(deepcopy(row))[0]
    path = tmp_path / 'input.json'; write_json(path, row)
    metadata = dict(arm='JOINT', inputs={str(path): file_hash(path)}, splits_sha256='frozen',
                    theta0={'adapter_path': 'theta0', 'adapter_weights_sha256': 'weights'}, excluded=excluded)
    manifest = seal_dataset(tmp_path / 'dataset', examples[::-1], policy=policy_for(), **metadata)
    written = [json.loads(line) for line in (tmp_path / 'dataset/examples.jsonl').read_text().splitlines()]
    assert written == sorted(examples, key=lambda e: e['example_id'])
    assert manifest['example_ids_sha256'] == digest([e['example_id'] for e in written])
    assert manifest['supervised_turns'] == 2 and manifest['counts']['unique_instances'] == 1
    assert manifest['trajectory_diagnostics'] == {}
    assert manifest['supervised_token_estimate']['is_estimate']
    assert (tmp_path / 'dataset/examples.sha256').read_text().strip() == file_hash(tmp_path / 'dataset/examples.jsonl')
    with pytest.raises(FileExistsError): seal_dataset(tmp_path / 'dataset', examples, policy=policy_for(), **metadata)
    path.write_text('{}')
    with pytest.raises(InvalidExample, match='input_changed_during_build'):
        seal_dataset(tmp_path / 'bad', examples, policy=policy_for(), **metadata)
    assert not (tmp_path / 'bad/manifest.json').exists()


def test_fake_label_transport_shared_caps_and_identity(tmp_path):
    config = load_config('configs/base_e4b.yaml')
    budget = SharedTeacherBudget(tmp_path / 'shared.jsonl', input_tokens=100_000, output_tokens=1024)
    prompt = rationale_prompt({'instruction': INITIAL}, INITIAL, [], 'look')
    first = RationaleLabeler(config, TASK, tmp_path / 'first', budget, dry_run=True)
    assert json.loads(first(prompt))['unsupported'] is False
    assert budget.summary()['reported_usage']['input_tokens'] == 100
    assert budget.summary()['reported_usage']['output_tokens'] == 30
    second = RationaleLabeler(config, TASK, tmp_path / 'second', budget, dry_run=True)
    from teachers.openai_teacher import TeacherCallError
    with pytest.raises(TeacherCallError): second(prompt)
    assert budget.summary()['stop_reason'] == 'global_output_tokens_cap'
    log = (tmp_path / 'first/requests.jsonl').read_text()
    assert 'FAKE_NO_MODEL' in log and 'timestamp' in log


@pytest.fixture
def dataset_inputs(tmp_path):
    """On-disk collection protocol containing only fabricated evidence."""
    from sed.seed_evidence import artifact_hashes
    root = tmp_path / 'project/data/verified/seeds_full_train_r2'
    receipts = []
    for n in range(21):
        task = TASK if n == 20 else f'train/pick_and_place_simple-Apple-None-CounterTop-K{n}/trial_0'
        episode = as_episode(record(['look', 'take apple 1 from cabinet 1'],
            ['On the cabinet 1, you see an apple 1.', 'Taken.'], task_id=task, success=True))
        directory = root / f'attempts/{n}/episode'
        blob = BlobStore(directory / 'blobs').put(dict(fact_source='real_environment', executable=True, audit={'won': True}))
        episode['steps'][-1]['verification_ref'] = blob
        write_json(directory / 'episode.json', episode)
        write_json(directory / 'provenance.json', {'synthetic': True})
        receipts.append(dict(task_id=task, group_id=task.rsplit('/', 1)[0], task_type='pick_and_place_simple',
            episode_dir=str(directory.relative_to(root)), trajectory_id=episode['episode_id'], won=True,
            episode_hashes=artifact_hashes(directory)))
    collection = dict(status='complete', verified=receipts)
    selection = dict(K=20, task_order=[r['task_id'] for r in receipts[:20]])
    identity = dict(selection=selection)
    replay = dict(K=20, status='complete', verified=receipts[:20], selection_sha256=digest(selection),
                  identity_sha256=digest(identity), source_collection={'manifest_sha256': digest(collection)})
    for name, value in [('manifest.json', collection), ('subsets/k20/manifest.json', replay),
                        ('subsets/k20/selection.json', selection), ('subsets/k20/identity.json', identity)]:
        write_json(root / name, value)
    selected = receipts[-1]
    splits = dict(sets={'V2_pool': dict(training_eligible=True, task_ids=[TASK], verified_trajectories=[dict(
        task_id=TASK, trajectory_id=selected['trajectory_id'], receipt_sha256=digest(selected),
        episode_path=str(root / selected['episode_dir'] / 'episode.json'), episode_sha256=selected['episode_hashes']['episode.json'])])},
        rebuild_arguments={'verified_root': str(root)}, input_sha256={str(root / name): file_hash(root / name)
            for name in ('manifest.json', 'subsets/k20/manifest.json', 'subsets/k20/selection.json')})
    split_path = tmp_path / 'splits.json'; write_json(split_path, seal(splits))
    config = load_config('configs/base_e4b.yaml')
    config['models']['student']['revision'] = 'synthetic-no-snapshot'
    config_path = tmp_path / 'config.json'; write_json(config_path, config)
    train = tmp_path / 'theta0'; checkpoint = train / 'checkpoint'; checkpoint.mkdir(parents=True)
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'synthetic-not-a-model')
    write_json(checkpoint / 'adapter_config.json', {'synthetic': True})
    write_json(checkpoint / 'training_state.json', dict(N_examples=20, final=True,
        artifact_sha256={n: file_hash(checkpoint / n) for n in ('adapter_model.safetensors', 'adapter_config.json')}))
    config['baseline_training'] = {'verified': str(root / 'subsets/k20')}
    write_json(train / 'provenance.json', dict(config=config, inputs={str(root / 'subsets/k20/manifest.json'): file_hash(root / 'subsets/k20/manifest.json')}))
    run = tmp_path / 'run'; identity = {'dry_run': True}
    write_json(run / 'plan.json', dict(identity=identity, run_sha256=digest(identity)))
    write_json(run / 'provenance.json', {'synthetic': True}); write_json(run / 'run_manifest.json', {'synthetic': True})
    row = record(['look', 'take apple 1 from cabinet 1'], ['On the cabinet 1, you see an apple 1.', 'Taken.'],
                 success=True, executors=['teacher', 'student'])
    row.update(run_sha256=digest(identity), run_key='test'); write_json(run / 'records/test.json', row)
    return dict(root=root, splits=split_path, config=config_path, train=train, run=run, row=row)


def cli_args(d, output, arm='JOINT'):
    args = ['build', '--arm', arm, '--splits', str(d['splits']), '--config', str(d['config']),
            '--theta0-training-dir', str(d['train']), '--output', str(output), '--dry-run']
    return args + (['--inputs', str(d['run'])] if arm in ARMS[:3] else [])


@pytest.mark.parametrize('arm', ARMS)
def test_cli_all_arms_separate_twenty_replay_and_manifest(dataset_inputs, tmp_path, arm):
    d = dataset_inputs
    if arm == 'PREFIX_EXPERT':
        d['row']['mode'] = 'takeover'; write_json(d['run'] / 'records/test.json', d['row'])
    args = cli_args(d, tmp_path / arm, arm)
    assert main(args) == 0
    manifest = json.loads((tmp_path / arm / 'manifest.json').read_text())
    replay = json.loads((tmp_path / arm / 'replay/manifest.json').read_text())
    assert manifest['schema'] == replay['schema'] == 'v2-train-interchange/1'
    assert replay['counts']['examples'] == replay['counts']['unique_instances'] == replay['counts']['task_groups'] == 20
    assert replay['counts']['by_source'] == {'k20_replay': 20}
    assert manifest['counts']['examples'] == (20 if arm == 'CONTINUE' else 1 if arm == 'OFFLINE' else 2)
    assert manifest['dry_run'] and not manifest['training_eligible']
    assert manifest['actual_training_exposures'] == manifest['new_environment_steps'] == 0
    assert manifest['code_sha256'] and manifest['dependency_versions'] and manifest['inputs']
    assert len(replay['trajectory_diagnostics']) == 20
    assert len(manifest['trajectory_diagnostics']) == (20 if arm == 'CONTINUE' else 1 if arm == 'OFFLINE' else 0)
    assert all(counts == dict(inadmissible_supervised_turns=0, not_yet_seen_instance_supervised_turns=0)
               for counts in replay['trajectory_diagnostics'].values())
    with pytest.raises(FileExistsError): main(args)


def test_cli_no_seal_on_budget_failure_or_foreign_task(dataset_inputs, tmp_path):
    d = dataset_inputs
    args = cli_args(d, tmp_path / 'bad') + ['--max-teacher-output-tokens', '0']
    with pytest.raises(Exception, match='L halted'): main(args)
    assert not (tmp_path / 'bad/manifest.json').exists()
    row = deepcopy(d['row']); row['task_id'] = K20; write_json(d['run'] / 'records/test.json', row)
    with pytest.raises(InvalidExample, match='task_outside_frozen_view'): main(cli_args(d, tmp_path / 'foreign'))
    assert not (tmp_path / 'foreign').exists()


def test_history_action_line_unchanged_and_multiline_guidance_detected():
    row = record(['look', 'take apple 1 from cabinet 1'], ['On the cabinet 1, you see an apple 1.', 'Taken.'])
    raw = 'Thought: I should explore.\nThe HINT suggests checking.\n  Action :  look  '
    row['steps'][0]['executed_reply'] = row['execution']['steps'][0]['action_raw'] = raw
    examples, _ = build(row, labeler=lambda p: label())
    assert examples[0]['messages'][-1]['content'] == 'Thought: Use the public evidence.\nAction: look'
    # build_messages applies the shared split_thinking visibility normalization.
    assert examples[1]['messages'][2]['content'] == 'Thought: Use the public evidence.\n  Action :  look'
    validate_example(examples[1], policy_for(), source_record=row)
    bad = deepcopy(examples[1]); bad['messages'][2]['content'] = 'Action: look'
    with pytest.raises(InvalidExample, match='history_rewrite_mismatch'):
        validate_example(bad, policy_for(), source_record=row)


def test_unparseable_history_is_never_fabricated():
    row = record(['look', 'take apple 1 from cabinet 1'], ['On the cabinet 1, you see an apple 1.', 'Taken.'], success=True)
    row['execution']['steps'][0].update(action_raw='No Action line', action_parsed=None, action_outcome='parser_error', environment_steps=0)
    row['steps'][0].update(executed_reply='No Action line', executed_action=None)
    examples, excluded = build(row)
    assert examples == [] and excluded['unparseable_prefix'] == 2


def refreeze_synthetic_episode(d, index, edit):
    """Update fabricated receipts only; never modifies a real frozen archive."""
    root = d['root']; collection = json.loads((root / 'manifest.json').read_text())
    receipt = collection['verified'][index]
    directory = root / receipt['episode_dir']
    episode = json.loads((directory / 'episode.json').read_text())
    edit(episode)
    write_json(directory / 'episode.json', episode)
    from sed.seed_evidence import artifact_hashes
    receipt['episode_hashes'] = artifact_hashes(directory)
    write_json(root / 'manifest.json', collection)
    replay = json.loads((root / 'subsets/k20/manifest.json').read_text())
    if index < 20:
        replay['verified'][index] = receipt
    replay['source_collection']['manifest_sha256'] = digest(collection)
    write_json(root / 'subsets/k20/manifest.json', replay)
    prov = json.loads((d['train'] / 'provenance.json').read_text())
    prov['inputs'][str(root / 'subsets/k20/manifest.json')] = file_hash(root / 'subsets/k20/manifest.json')
    write_json(d['train'] / 'provenance.json', prov)
    splits = json.loads(d['splits'].read_text()); splits.pop('content_sha256')
    for selected in splits['sets']['V2_pool']['verified_trajectories']:
        if selected['task_id'] == receipt['task_id']:
            selected.update(receipt_sha256=digest(receipt), episode_sha256=receipt['episode_hashes']['episode.json'])
    for name in ('manifest.json', 'subsets/k20/manifest.json'):
        splits['input_sha256'][str(root / name)] = file_hash(root / name)
    write_json(d['splits'], seal(splits))


@pytest.mark.parametrize('arm', ['OFFLINE', 'CONTINUE', 'JOINT'])
@pytest.mark.parametrize('issue', ['inadmissible', 'ungrounded'])
def test_cli_seals_unchanged_trajectories_with_per_example_diagnostics(dataset_inputs, tmp_path, arm, issue):
    d = dataset_inputs
    def edit(episode):
        if issue == 'inadmissible':
            episode['initial']['visible']['admissible_commands'] = ['inventory']
            episode['steps'][0]['action_outcome'] = 'inadmissible'
        else:
            episode['steps'][0]['action_raw'] = 'Thought: apple 99 exists.\nAction: look'
    for index in (0, 20):
        refreeze_synthetic_episode(d, index, edit)
    output = tmp_path / 'diagnostic-trajectories'
    assert main(cli_args(d, output, arm)) == 0
    for directory in (output, output / 'replay'):
        manifest = json.loads((directory / 'manifest.json').read_text())
        examples = [json.loads(line) for line in (directory / 'examples.jsonl').read_text().splitlines()]
        diagnostics = manifest['trajectory_diagnostics']
        assert set(diagnostics) == {e['example_id'] for e in examples if e['kind'] == 'trajectory'}
        affected = [counts for counts in diagnostics.values() if any(counts.values())]
        expected = dict(inadmissible_supervised_turns=int(issue == 'inadmissible'),
                        not_yet_seen_instance_supervised_turns=int(issue == 'ungrounded'))
        assert affected == ([] if directory == output and arm == 'JOINT' else [expected])
        for example in examples:
            if example['kind'] == 'trajectory':
                episode = json.loads(Path(example['evidence']['episode_ref']).read_text())
                assert canonical_bytes(example['messages']) == canonical_bytes(trajectory_messages(episode))
        assert manifest['excluded'] == []
    assert not (output / 'validation_failures.json').exists()


@pytest.mark.parametrize('reply,reason', [
    ('Thought: Guidance: look here.\nAction: look', 'guidance_or_message_fields'),
    ('No Action line', 'unparseable_prefix'),
])
def test_mandatory_trajectory_failure_reports_no_seal_or_label_calls(dataset_inputs, tmp_path, reply, reason):
    d = dataset_inputs
    refreeze_synthetic_episode(d, 0, lambda episode: episode['steps'][0].update(action_raw=reply))
    output = tmp_path / 'invalid-trajectory'
    with pytest.raises(InvalidExample, match='mandatory_trajectory_validation_failed'):
        main(cli_args(d, output))
    report = json.loads((output / 'validation_failures.json').read_text())
    assert report['excluded'] == [{'reason': reason, 'count': 1}]
    assert report['new_teacher_calls'] == 0 and not report['sealed']
    assert not (output / 'manifest.json').exists() and not (output / 'labels').exists()


def test_real_label_requires_explicit_shared_caps_before_io(tmp_path):
    with pytest.raises(ValueError, match='both explicit hard token caps'):
        main(['build', '--arm', 'JOINT', '--inputs', 'nonexistent', '--output', str(tmp_path / 'bad'), '--rationale-teacher'])
    assert not (tmp_path / 'bad').exists()
