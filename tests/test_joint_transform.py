"""Opt-in V2 preconditions; offline synthetic execution and a frozen V1b prefix."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair.facts import FactTable
from repair.joint import (JointPolicy, JointPrompt, JointTeacher, TEMPLATE, TEMPLATE_VERSION,
    ReviewInvalid, execute_joint, make_reviewer, public_record, review_prompt, validate_review)
from repair.joint_transform import ENVIRONMENT_RULE, TransformFactTable
from sed.config import load_config
from sed.react import build_messages
from sed.storage import canonical_bytes, digest
from tools.joint_guidance import main
from tools.teacher_takeover import main as takeover_main
from test_cg_probe import GROUPS, ScriptEnvironment, coverage_fixture, original
from test_facts import step
from test_joint import (CaptureStudent, RawClient, budget, case, context, review)
from test_teacher import response

V2 = 'joint-review/2'
GOAL = 'Your task is to: put a hot apple in a bowl.'


def status(table, key):
    return table.conditions()[key]['status']


@pytest.mark.parametrize('verb,state,appliance', [
    ('heat', 'hot', 'microwave'), ('cool', 'cool', 'fridge'), ('clean', 'clean', 'sinkbasin')])
@pytest.mark.parametrize('verb_goal', [False, True])
def test_transform_goal_ready_and_executed_completion(verb, state, appliance, verb_goal):
    goal = f'{verb} an apple and put it in a bowl' if verb_goal else f'put a {state} apple in a bowl'
    table = TransformFactTable('Your task is to: ' + goal + '.')
    assert table.conditions()['required_transform'] == dict(status='satisfied', evidence_steps=[-1],
        value=dict(command=verb, appliance_type=appliance, required_state=state))
    # Execution completion is distinct from an unobserved initial physical state.
    assert status(table, 'required_state_done') == 'undetermined'
    assert status(table, 'transform_done') == 'known_unsatisfied'
    assert status(table, 'transform_ready') == 'undetermined'
    table.update(step('take apple 1 from shelf 1'))
    assert status(table, 'transform_ready') == 'undetermined'
    table.update(step(f'go to {appliance} 2'))
    assert table.conditions()['at_required_appliance'] == dict(
        status='satisfied', evidence_steps=[-1, 1], value=f'{appliance} 2')
    assert table.conditions()['transform_ready'] == dict(status='satisfied', evidence_steps=[-1, 0, 1])
    command = f'{verb} apple 1 with {appliance} 2'
    table.update(step(command, 'Nothing happens.'))
    assert status(table, 'transform_done') == 'known_unsatisfied'
    assert status(table, 'transform_ready') == 'satisfied'
    table.update(step(command))
    assert table.conditions()['transform_done'] == table.conditions()['required_state_done']
    assert table.conditions()['transform_done']['evidence_steps'] == [3]
    assert status(table, 'transform_ready') == 'known_unsatisfied'
    table.update(step('move apple 1 to bowl 1'))
    assert status(table, 'transform_done') == 'satisfied'


@pytest.mark.parametrize('verb,state,appliance', [
    ('heat', 'hot', 'microwave'), ('cool', 'cool', 'fridge'), ('clean', 'clean', 'sinkbasin')])
@pytest.mark.parametrize('placement', ['move apple 1 to', 'put apple 1 in'])
def test_placement_open_close_never_transforms_and_retake_restores_ready(verb, state, appliance, placement):
    table = TransformFactTable.from_record(f'Your task is to: put a {state} apple in a bowl.', [
        step('take apple 1 from shelf 1'), step(f'go to {appliance} 1'),
        step(f'{placement} {appliance} 1')])
    assert table.conditions()['target_inside_appliance'] == dict(
        status='satisfied', evidence_steps=[-1, 0, 2], instances=['apple 1'])
    assert status(table, 'hands_free') == 'satisfied'
    for action, text in [(f'open {appliance} 1', f'You open the {appliance} 1. In it, you see an apple 1.'),
                         (f'close {appliance} 1', f'You close the {appliance} 1.')]:
        table.update(step(action, text))
        assert status(table, 'target_inside_appliance') == 'satisfied'
        assert status(table, 'transform_done') == 'known_unsatisfied'
        assert status(table, 'required_state_done') == 'undetermined'
        assert status(table, 'transform_ready') == 'known_unsatisfied'
    table.update(step(f'take apple 1 from {appliance} 1'))
    assert status(table, 'target_inside_appliance') == 'known_unsatisfied'
    assert status(table, 'transform_ready') == 'satisfied'
    table.update(step('move apple 1 to shelf 1'))
    assert status(table, 'target_inside_appliance') == 'known_unsatisfied'


def test_only_navigation_and_arrival_establish_current_location():
    table = TransformFactTable(GOAL + ' You see a microwave 1.')
    table.update(step('take apple 1 from shelf 1'))
    for command, text in [('open microwave 1', 'You open the microwave 1.'),
                           ('close microwave 1', 'You close the microwave 1.'),
                           ('examine microwave 1', 'The microwave 1 is closed.'),
                           ('go to microwave 1', 'Nothing happens.')]:
        table.update(step(command, text))
        assert status(table, 'at_required_appliance') == 'undetermined'
        assert status(table, 'transform_ready') == 'undetermined'
    table.update(step('look', 'You are at the microwave 1.'))
    assert status(table, 'transform_ready') == 'satisfied'
    table.update(step('go to shelf 1'))
    assert status(table, 'at_required_appliance') == 'known_unsatisfied'
    table.update(step('go to microwave 1', '', 'inadmissible'))
    assert table.conditions()['at_required_appliance']['value'] == 'shelf 1'
    table.update(step('go to microwave 1', 'You arrive at microwave 1.'))
    assert status(table, 'transform_ready') == 'satisfied'
    assert TransformFactTable(GOAL + ' You are at microwave 2.').conditions()[
        'at_required_appliance']['evidence_steps'] == [-1]


def test_unknown_goal_hold_location_and_incomplete_outcomes_stay_unknown():
    for goal in ('Your task is to: teleport home.', 'No goal',
                 'Your task is to: put an apple in a microwave.'):
        table = TransformFactTable.from_record(goal, [step('take apple 1 from shelf 1'), step('go to microwave 1')])
        assert all(status(table, k) == 'undetermined' for k in
            ('required_transform', 'transform_done', 'at_required_appliance', 'target_inside_appliance', 'transform_ready'))
    table = TransformFactTable.from_record(GOAL, [step('go to microwave 1')])
    assert status(table, 'target_held') == status(table, 'transform_ready') == 'undetermined'
    table.update(step('take apple 1 from shelf 1', '', 'parser_error'))
    assert status(table, 'target_held') == 'undetermined'
    table.update(step('take apple 1 from shelf 1'))
    table.update(step('heat apple 1 with microwave 1', '', None))
    assert status(table, 'transform_done') == status(table, 'transform_ready') == 'undetermined'
    table.update(step('heat apple 1 with microwave 1'))
    assert status(table, 'transform_done') == 'satisfied'
    table.update(step('cool apple 1 with fridge 1'))
    assert status(table, 'transform_done') == 'known_unsatisfied'
    assert status(table, 'transform_ready') == 'satisfied'


def test_inside_uses_last_known_public_location():
    table = TransformFactTable.from_record(GOAL, [step('inventory', 'You are not carrying anything.'),
        step('go to microwave 1', 'In the microwave 1, you see an apple 1.')])
    expected = dict(status='satisfied', evidence_steps=[-1, 0, 1], instances=['apple 1'])
    assert table.conditions()['target_inside_appliance'] == expected
    table.update(step('move apple 1 to microwave 1', 'Nothing happens.'))
    assert table.conditions()['target_inside_appliance'] == expected
    table.update(step('take apple 1 from microwave 1'))
    table.update(step('move apple 1 to microwave 1'))
    assert status(table, 'target_inside_appliance') == 'satisfied'
    table.update(step('examine microwave 1', 'In the microwave 1, you see nothing.'))
    assert status(table, 'target_inside_appliance') == 'undetermined'


@pytest.mark.parametrize('following,expected', [
    (step('close microwave 1', 'You close the microwave 1.'),
     dict(status='satisfied', evidence_steps=[-1, 0, 2], instances=['potato 1'])),
    (step('take potato 1 from microwave 1', 'You pick up the potato 1.'),
     dict(status='known_unsatisfied', evidence_steps=[-1, 3])),
    (step('go to countertop 1', 'On the countertop 1, you see a potato 1.'),
     dict(status='known_unsatisfied', evidence_steps=[-1, 0, 3])),
], ids=['closed_after_opening', 'taken', 'seen_elsewhere'])
def test_sighted_target_inside_appliance_tracks_later_evidence(following, expected):
    # Public sighting pattern from pick_heat_then_place_in_recep-Potato-None-GarbageCan-5.
    # The student never placed this potato in the microwave.
    goal = 'Your task is to: put a hot potato in a garbagecan.'
    prefix = [step('inventory', 'You are not carrying anything.'),
              step('go to microwave 1', 'You arrive at microwave 1. The microwave 1 is closed.'),
              step('open microwave 1',
                   'You open the microwave 1. The microwave 1 is open. In it, you see a potato 1.')]
    table = TransformFactTable.from_record(goal, prefix)
    assert not table.placements
    assert table.conditions()['target_inside_appliance'] == dict(
        status='satisfied', evidence_steps=[-1, 0, 2], instances=['potato 1'])
    table.update(following)
    assert table.conditions()['target_inside_appliance'] == expected
    assert status(table, 'transform_done') == 'known_unsatisfied'
    assert status(table, 'required_state_done') == 'undetermined'
    v1_conditions = FactTable.from_record(goal, [*prefix, following]).conditions()
    assert {key: table.conditions()[key] for key in v1_conditions} == v1_conditions


def test_v2_public_whitelist_and_incremental_derivation():
    initial = {'visible': {'observation': GOAL, 'admissible_commands': ['SECRET']},
               'task_id': 'SECRET', 'audit': {'policy_commands': ['SECRET']}}
    steps = [step('take apple 1 from shelf 1'), step('go to microwave 1')]
    expected = TransformFactTable.from_record(GOAL, steps)
    table = TransformFactTable(initial)
    for value in steps:
        table.update(dict(value, audit={'won': True}, extra={'expert_plan': 'SECRET'},
                          internal_state='SECRET', hidden_success=True))
    assert table.to_json() == expected.to_json()
    assert table.prompt_text() == expected.prompt_text()
    assert 'SECRET' not in table.prompt_text()
    detached = table.to_json()
    detached['conditions']['required_transform']['value']['command'] = 'SECRET'
    assert 'SECRET' not in table.prompt_text()


@pytest.mark.parametrize('version', [TEMPLATE_VERSION, V2])
def test_template_envelope_selection_and_retry(version, tmp_path):
    prompt = JointPrompt({'data': 'public'}, template_version=version)
    expected = TEMPLATE if version == TEMPLATE_VERSION else TEMPLATE + ENVIRONMENT_RULE
    assert prompt.messages()[0]['content'] == expected
    assert prompt.to_json()['template_version'] == version
    assert prompt.template_sha256 == digest(version + '\n' + expected)
    retry = replace(prompt, retry_reason='invalid_json', retry_index=1)
    assert retry.template_version == version
    assert retry.messages()[0]['content'] == expected + '\nOne format retry: invalid_json'
    teacher = JointTeacher(cache_dir=tmp_path / 'cache', usage_ledger=tmp_path / 'usage.jsonl',
                           run_id='fake', review_template=version)
    envelope = {'observation': canonical_bytes(prompt.to_json()).decode(), 'admissible_commands': []}
    assert teacher.prompt([], {}, envelope) == prompt.messages()
    other = replace(prompt, template_version=V2 if version == TEMPLATE_VERSION else TEMPLATE_VERSION)
    with pytest.raises(ValueError, match='envelope'):
        teacher.prompt([], {}, dict(envelope, observation=canonical_bytes(other.to_json()).decode()))
    with pytest.raises(ValueError, match='Unknown joint review template'):
        JointPrompt({}, template_version='joint-review/3')


@pytest.mark.parametrize('changes,error', [
    ({}, None), ({'decision': 'demonstrate'}, None), ({'action': 'look'}, None),
    ({'action': 'heat apple 2 with microwave 1'}, 'action_not_admissible'),
    ({'facts_cited': [{'key': 'conditions.transform_ready', 'evidence_steps': [999]}]}, 'unsupported_evidence'),
    ({'guidance': 'x ' * 61}, 'guidance_word_limit'), ({'check': 'x ' * 31}, 'check_word_limit'),
    ({'extra': True}, 'schema'), ({'facts_cited': []}, 'facts_required')])
def test_v2_uses_unchanged_validator(changes, error):
    facts = TransformFactTable.from_record(GOAL, [step('take apple 1 from shelf 1'), step('go to microwave 1')])
    value = {**review(action='heat apple 1 with microwave 1',
                     facts_cited=[dict(key='conditions.transform_ready', evidence_steps=[-1, 0, 1])]), **changes}
    commands = ['heat apple 1 with microwave 1', 'look']
    if error:
        with pytest.raises(ReviewInvalid, match=error):
            validate_review(json.dumps(value), facts, commands)
    else:
        assert validate_review(json.dumps(value), facts, commands) == value


def test_v2_G_loop_fake_transport_and_format_retry(tmp_path, budget):
    command = 'heat apple 1 with microwave 1'
    value = review(action=command, facts_cited=[dict(key='conditions.transform_ready', evidence_steps=[-1, 0, 1])])
    client = RawClient([response('{bad'), response(json.dumps(value))], budget)
    reviewer = make_reviewer(load_config('configs/base_e4b.yaml'),
        dict(task_id='train/fake/trial', split='train', sealed=False), tmp_path / 'review', tmp_path,
        budget, client=client, review_template=V2)
    student = CaptureStudent(['Action: look', 'Action: ' + command])
    policy = JointPolicy(student, 'train/fake/trial', 'G', reviewer, review_template=V2)
    history = [dict(visible=context(GOAL, ['take apple 1 from shelf 1']), reply='Action: take apple 1 from shelf 1'),
               dict(visible=context('You pick up the apple 1.', ['go to microwave 1']), reply='Action: go to microwave 1')]
    visible = context('You arrive at microwave 1.', [command, 'look'])
    assert policy(history, visible) == 'Action: ' + command
    assert student.prompts[0] == build_messages(history, {'task_id': 'train/fake/trial'}, visible)
    assert policy.steps[0]['executor'] == 'student_after_guidance'
    assert policy.steps[0]['followed_suggestion'] is True
    for attempt in policy.steps[0]['review_attempts']:
        assert attempt['request']['template_version'] == V2
        assert attempt['request']['template_sha256'] == JointPrompt({}, template_version=V2).template_sha256
    assert all(r['input'][0]['content'].startswith(TEMPLATE + ENVIRONMENT_RULE) for r in client.requests)


@pytest.mark.parametrize('explicit', [False, True])
def test_v1b_recorded_prompt_facts_and_step_record_are_byte_identical(explicit):
    fixture = json.loads((Path(__file__).parent / 'fixtures/joint_v1b_step.json').read_text())
    history, visible = fixture['history'], fixture['visible']
    initial, public = public_record(history, visible)
    facts = FactTable.from_record(initial, public)
    options = {'review_template': TEMPLATE_VERSION} if explicit else {}
    prompt = review_prompt(history, visible, fixture['proposal'], facts, fixture['max_steps'], **options)
    expected = fixture['expected']
    assert prompt.template_sha256 == expected['template_sha256']
    assert digest(prompt.messages()) == expected['messages_sha256']
    assert digest(prompt.to_json()) == expected['envelope_sha256']
    assert digest(facts.to_json()) == expected['facts_sha256']
    assert digest(facts.prompt_text()) == expected['facts_prompt_sha256']

    class RecordedReviewer:
        def request(self, sent, table, commands, attempts):
            assert sent.to_json() == fixture['review_attempts'][0]['request']
            attempts.extend(deepcopy(fixture['review_attempts']))
            return fixture['review_attempts'][-1]['parsed_review']

    policy = JointPolicy(CaptureStudent([fixture['proposal']]), 'unused', 'G', RecordedReviewer(),
                         fixture['max_steps'], **options)
    policy(history, visible)
    prior = [dict(s, visible={'observation': s['observation']}) for s in public]
    policy.finish(SimpleNamespace(steps=[*prior, fixture['executed_step']]))
    assert digest(policy.steps[0]) == expected['step_record_sha256']


@pytest.mark.parametrize('version', [TEMPLATE_VERSION, V2])
def test_cli_default_selection_resume_and_mixed_version_refusal(version, original, tmp_path):
    coverage, _ = coverage_fixture(tmp_path, original)
    spec = tmp_path / 'spec.json'
    assert takeover_main(['spec-from-cg', '--failures', str(coverage / 'failures.json'), '--coverage-root', str(coverage),
        '--groups', GROUPS[5], '--dry-run', '--output', str(spec)]) == 0
    output, ledger = tmp_path / 'run', tmp_path / 'budget.jsonl'
    common = ['run', '--spec', str(spec), '--dry-run', '--seeds', '--workers', '2',
              '--budget-ledger', str(ledger), '--output', str(output)]
    selection = [] if version == TEMPLATE_VERSION else ['--review-template', version]
    assert main([*common, *selection]) == 0
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['identity']['template_version'] == version
    assert plan['identity']['template_sha256'] == JointPrompt({}, template_version=version).template_sha256
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['template_version'] == version
    records = {p.name: p.read_bytes() for p in (output / 'records').glob('*.json')}
    for blob in records.values():
        row = json.loads(blob)
        if version == V2:
            assert row['template_version'] == V2
        else:
            assert 'template_version' not in row  # Preserve the V1 episode format.
        for turn in row['steps']:
            for attempt in turn['review_attempts']:
                assert attempt['request']['template_version'] == version
    assert main([*common, '--review-template', version, '--resume']) == 0
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match='cannot mix review template versions'):
        main([*common, '--resume', '--review-template', V2 if version == TEMPLATE_VERSION else TEMPLATE_VERSION])
    assert ledger.read_bytes() == before
    assert records == {p.name: p.read_bytes() for p in (output / 'records').glob('*.json')}


def test_episode_resume_rejects_wrong_template_even_with_same_run_hash(original, tmp_path, budget):
    args = dict(mode='U', decoding='greedy', seed=None, env_factory=ScriptEnvironment,
        student_factory=lambda _: CaptureStudent(['Action: take apple 1 from cabinet 2']),
        reviewer_factory=lambda _: pytest.fail('U constructed R'), output=tmp_path, run_sha256='fake', budget=budget)
    row = execute_joint(case(original), original, review_template=V2, **args)
    assert row['template_version'] == V2 and row['success']
    with pytest.raises(ValueError, match='Resume identity mismatch'):
        execute_joint(case(original), original, **args)
