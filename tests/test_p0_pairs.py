from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

import pytest

from repair.p0_pairs import (extract_pairs, stratify, DEFAULT_ROUND, DEFAULT_BASE,
    local_repair, render_state, teacher_pair, load_pairs, collateral_sample, resolve_category, CATEGORIES)
from repair.records import TrajectoryRecord
from sed.storage import write_json, digest
from p0_fakes import fake_pair, TinyTokenizer


@pytest.fixture(scope='module')
def real_pairs():
    if not (Path(DEFAULT_ROUND) / 'review/records.json').is_file():
        pytest.skip('Authorized read-only round is not mounted')
    return extract_pairs()


def test_real_record_counts_and_no_silent_drops(real_pairs):
    assert real_pairs['counts']['verified_repairs'] == 134
    assert real_pairs['schema'] == 'p0-pairs/3'
    assert real_pairs['counts']['first_listed_categories'] == dict(refused_action=53, goal_misread=26,
        missing_precondition=23, wrong_object_or_target=15, search_incomplete_or_loop=13, other=4)
    assert len(real_pairs['pairs']) == 113 and len(real_pairs['excluded']) == 21
    assert real_pairs['counts']['excluded_by_reason'] == dict(
        divergence_not_teacher_patch=18, category_other=3)
    assert real_pairs['counts']['exclusion_reason_occurrences']['category_other'] == 4
    assert Counter(p['split'] for p in real_pairs['pairs']) == {'diagnosis': 23, 'test': 90}
    assert real_pairs['counts']['alignment'] == dict(final_execution_first_divergence=113)
    assert real_pairs['counts']['category_source'] == dict(at_divergence=69, nearest_following=35, nearest_preceding=9)
    assert real_pairs['counts']['state_match'] == dict(verified=134)
    assert real_pairs['counts']['unique_instances'] == 74
    assert real_pairs['counts']['diagnosis_test_instance_overlap'] == 0
    assert real_pairs['counts']['diagnosis_test_state_overlap'] == 0
    assert real_pairs['counts']['diagnosis_test_group_overlap'] == 0
    assert real_pairs['counts']['diagnosis_test_parent_seed_task_id_overlap'] == 9
    assert all(p['reproduction']['status'] == 'not_run' for p in real_pairs['pairs'])


def test_real_category_split_counts(real_pairs):
    diag = Counter(p['category'] for p in real_pairs['pairs'] if p['split'] == 'diagnosis')
    test = Counter(p['category'] for p in real_pairs['pairs'] if p['split'] == 'test')
    assert diag == dict(goal_misread=4, missing_precondition=5, refused_action=5,
                        search_incomplete_or_loop=4, wrong_object_or_target=5)
    assert test == dict(goal_misread=23, missing_precondition=32, refused_action=14,
                        search_incomplete_or_loop=8, wrong_object_or_target=13)


def test_split_order_independent(real_pairs):
    rows = real_pairs['pairs']
    assert stratify(rows) == stratify(list(reversed(rows))) == rows
    for p in rows:
        assert len({r['split'] for r in rows if r['group_id'] == p['group_id']}) == 1


def test_real_messages_byte_identical_to_harness(real_pairs):
    from sed.react import build_messages
    for pair in real_pairs['pairs']:
        original = TrajectoryRecord.from_json(pair['audit']['original'])
        context = original.decision_context(pair['step'])
        assert pair['state']['messages'] == build_messages(context['history'], {'task_id': pair['task_id']}, context['visible_context'])
        assert pair['state']['sha256'] == digest(pair['state']['messages'])
        assert pair['a_f'] == original.steps[pair['step']]['action_parsed']


def test_real_tokenizer_render_and_span_alignment(real_pairs):
    from train.dataset import load_tokenizer
    tokenizer = load_tokenizer(DEFAULT_BASE)
    for p in real_pairs['pairs']:
        if p['split'] != 'diagnosis':
            continue
        state = render_state(p, tokenizer)
        assert state['text'] == tokenizer.apply_chat_template(p['state']['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        assert all(0 <= s['start'] < s['end'] <= len(state['text']) for s in state['spans'])
        assert any(s['label'] == 'latest_observation' for s in state['spans'])


def test_freeze_tamper_detection(tmp_path, real_pairs):
    path = tmp_path / 'pairs.json'
    write_json(path, real_pairs)
    assert load_pairs(path)['sha256'] == real_pairs['sha256']
    changed = deepcopy(real_pairs)
    changed['pairs'][0]['split'] = 'test' if changed['pairs'][0]['split'] == 'diagnosis' else 'diagnosis'
    write_json(path, changed)
    with pytest.raises(ValueError, match='freeze changed'):
        load_pairs(path)


def patch_review(original, **block):
    return dict(patch=dict(original_trajectory_id=original.trajectory_id,
                          blocks=[dict(block_id='x', **block)]))


def repaired_at(original, t=1, action='inventory'):
    repaired = original.to_json()
    repaired.pop('trajectory_id')
    repaired['steps'][t].update(action_parsed=action, action_raw='Action: ' + action)
    return repaired


def test_later_round_patch_supplies_verified_action():
    r = TrajectoryRecord.from_json(fake_pair()['audit']['original'])
    review = patch_review(r, op='replace', start=1, end_exclusive=2, actions=['go to bed 2'])
    t, ar, reason, detail = local_repair(r, review, repaired_execution=repaired_at(r),
        executors=['student', 'patch', 'student'], review_rounds=3)
    assert (t, ar, reason) == (1, 'inventory', None)
    assert detail['round_zero_action_at_divergence'] == 'go to bed 2'
    assert detail['review_rounds'] == 3 and detail['executor_at_divergence'] == 'patch'
    assert detail['prefix_equal'] and detail['state_match'] == 'verified'


def test_student_redecode_divergence_excluded():
    r = TrajectoryRecord.from_json(fake_pair()['audit']['original'])
    review = patch_review(r, op='replace', start=2, end_exclusive=3, actions=['inventory'])
    t, ar, reason, detail = local_repair(r, review, repaired_execution=repaired_at(r, 0),
        executors=['student', 'student', 'patch'])
    assert (t, ar, reason) == (0, 'inventory', 'divergence_not_teacher_patch')
    assert detail['executor_at_divergence'] == 'student'


@pytest.mark.parametrize('prefix,reason', [(1, None), (2, 'divergence_in_context_only_prefix')])
def test_student_prefix_boundary(prefix, reason):
    r = TrajectoryRecord.from_json(fake_pair()['audit']['original'])
    t, ar, actual, _ = local_repair(r, patch_review(r, op='insert_before', index=1, actions=['inventory']),
        prefix_length=prefix, repaired_execution=repaired_at(r), executors=['student', 'patch', 'student'])
    assert (t, ar, actual) == (1, 'inventory', reason)


@pytest.mark.parametrize('mode,status', [('same', 'verified'), ('visible', 'mismatch'),
    ('initial', 'mismatch'), ('audit', 'verified'), ('at_divergence', 'verified')])
def test_repaired_prefix_state_match(mode, status):
    r = TrajectoryRecord.from_json(fake_pair()['audit']['original'])
    repaired = repaired_at(r, 2)
    if mode == 'visible':
        repaired['steps'][0]['visible']['observation'] = 'Changed visible observation'
    elif mode == 'initial':
        repaired['initial']['visible']['observation'] = 'Changed initial observation'
    elif mode == 'audit':
        repaired['steps'][0]['audit_sha256'] = 'not-a-policy-field'
    elif mode == 'at_divergence':
        repaired['steps'][2]['visible']['observation'] = 'Allowed after the decision'
    _, _, reason, detail = local_repair(r, patch_review(r, op='insert_before', index=2, actions=['inventory']),
        repaired_execution=repaired, executors=['student', 'student', 'patch'])
    assert detail['state_match'] == status
    assert reason == ('repaired_prefix_state_mismatch' if status == 'mismatch' else None)


@pytest.mark.parametrize('length,reason', [(2, 'no_alternative_action_at_divergence'),
    (3, 'no_action_divergence'), (4, 'no_original_action_at_divergence')])
def test_no_shared_action_divergence_excluded(length, reason):
    r = TrajectoryRecord.from_json(fake_pair()['audit']['original'])
    execution = r.to_json()
    execution.pop('trajectory_id')
    execution['steps'] = (execution['steps'] + execution['steps'])[:length]
    t, ar, actual, _ = local_repair(r, patch_review(r, op='delete', start=1, end_exclusive=3),
        repaired_execution=execution, executors=['patch'] * length)
    assert (t, ar, actual) == (None, None, reason)


@pytest.mark.parametrize('steps,t,selected,source,distance', [
    ([6, 2, 2], 2, 1, 'at_divergence', 0),
    ([0, 8, 6], 5, 2, 'nearest_following', 1),
    ([0, 3, 8], 4, 1, 'nearest_preceding', 1),
    ([2, 6], 4, 1, 'nearest_following', 2),
    ([6, 2], 4, 0, 'nearest_following', 2),
])
def test_category_resolution_and_ties(steps, t, selected, source, distance):
    points = [dict(step=s, category=CATEGORIES[i], reason=str(i)) for i, s in enumerate(steps)]
    assert resolve_category(points, t) == (points[selected], source, distance)


def test_group_split_counts_all_categories_in_chosen_group():
    # Random(0) orders these groups g3, g4, g0, g1, g2. Each chosen
    # group contributes two pairs to EVERY category, so only g3/g4 are chosen.
    rows = [dict(pair_id=f'{g}-{c}-{i}', group_id=g, task_id=g, category=c,
                 state={'sha256': f'{g}-{c}-{i}'})
            for g in ['g0', 'g1', 'g2', 'g3', 'g4'] for c in CATEGORIES for i in range(2)]
    result = stratify(rows)
    assert {p['group_id'] for p in result if p['split'] == 'diagnosis'} == {'g3', 'g4'}
    assert Counter(p['category'] for p in result if p['split'] == 'diagnosis') == dict.fromkeys(CATEGORIES, 4)
    assert result == stratify(list(reversed(rows)))
    with pytest.raises(ValueError, match='Insufficient'):
        stratify([p for p in rows if p['category'] != CATEGORIES[0]])
    rows[0]['state']['sha256'] = next(p['state']['sha256'] for p in result if p['split'] == 'diagnosis')
    with pytest.raises(ValueError, match='overlap: state'):
        stratify(rows)


def test_teacher_payload_whitelist():
    p = fake_pair()
    p['audit']['SECRET_ORACLE'] = 'DO_NOT_LEAK'
    p['review'].update(step=3, patch='DO_NOT_LEAK', hidden_success='DO_NOT_LEAK')
    state = render_state(p, TinyTokenizer())
    payload = teacher_pair(p, state)
    assert 'DO_NOT_LEAK' not in json.dumps(payload)
    assert set(payload) == {'pair_id', 'task', 'state', 'state_id', 'spans', 'a_f', 'a_r', 'review', 'framing', 'framing_record'}
    assert 'internal_state' not in json.dumps(payload)
    assert payload['review'] == dict(step=3, category=p['category'], reason=p['review']['reason'], decision_step=1)


def test_s_bucket_200_decision_freeze(real_pairs):
    first = collateral_sample(DEFAULT_ROUND)
    assert first == collateral_sample(DEFAULT_ROUND)
    assert len(first) == 200 and len({x['pair_id'] for x in first}) == 200
    assert all(p['task_id'].startswith('train/synth/') for p in first)
    records = {r['attempt_id']: r for r in json.loads((Path(DEFAULT_ROUND) / 'attempt/records.json').read_text())}
    for p in first:
        aid = p['pair_id'].removeprefix('S-').rsplit('-', 1)[0]
        assert p['original_action_raw'] == records[aid]['execution']['steps'][p['step']]['action_raw']
    thoughts = [render_state(p, TinyTokenizer())['original_thought'] for p in first]
    assert sum(bool(r['prefix']) for r in thoughts) == 198
    assert all(r['fallback_reason'] for r in thoughts if not r['prefix'])


def test_real_final_execution_actions_and_audit(real_pairs):
    reviews = {r['attempt_id']: r for r in json.loads((Path(DEFAULT_ROUND) / 'review/records.json').read_text())}
    for pair in real_pairs['pairs']:
        row = reviews[pair['pair_id']]
        t = pair['step']
        assert pair['a_r'] == row['execution']['steps'][t]['action_parsed']
        assert row['executors'][t] == pair['alignment_audit']['executor_at_divergence'] == 'patch'
        assert pair['alignment_audit']['review_rounds'] == len(row['rounds'])
    example = next(p for p in real_pairs['pairs'] if p['pair_id'].startswith('55601f73'))
    # The v3 unit can also move earlier than the v2 divergence cited in review.
    assert (example['step'], example['a_f'], example['a_r']) == (13, 'go to bed 2', 'go to shelf 1')
    assert example['alignment_audit']['round_zero_action_at_divergence'] == 'examine bed 2'


@pytest.mark.parametrize('schema', ['p0-pairs/1', 'p0-pairs/2'])
def test_old_pair_freezes_refused(tmp_path, real_pairs, schema):
    value = dict(real_pairs, schema=schema)
    value['sha256'] = digest({k: v for k, v in value.items() if k != 'sha256'})
    path = tmp_path / 'old.json'
    write_json(path, value)
    with pytest.raises(ValueError, match='freeze changed'):
        load_pairs(path)
