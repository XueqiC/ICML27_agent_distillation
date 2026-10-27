"""Public prose fixtures; no environment, game files, models or API clients."""
import json

import pytest

from repair.facts import FactTable, parse_goal
from tools.facts_audit import audit, main


def step(action, text='', outcome='legal', **hidden):
    return dict(action_parsed=action, action_outcome=outcome, observation=text, **hidden)


@pytest.mark.parametrize('text,obj,state,target,count', [
    ('put a apple in a fridge', 'apple', None, 'fridge', 1),
    ('put some kettle on shelf', 'kettle', None, 'shelf', 1),
    ('put an egg in the microwave', 'egg', None, 'microwave', 1),
    ('examine a newspaper with the desklamp', 'newspaper', 'lit', 'desklamp', 1),
    ('look at some book under the desklamp', 'book', 'lit', 'desklamp', 1),
    ('put a clean lettuce in countertop', 'lettuce', 'clean', 'countertop', 1),
    ('put a hot potato in a bowl', 'potato', 'hot', 'bowl', 1),
    ('put a cool apple in microwave', 'apple', 'cool', 'microwave', 1),
    ('clean some kettle and put it in diningtable', 'kettle', 'clean', 'diningtable', 1),
    ('heat some potato and put it on countertop', 'potato', 'hot', 'countertop', 1),
    ('cool some egg and put it in microwave', 'egg', 'cool', 'microwave', 1),
    ('put two apples in fridge', 'apple', None, 'fridge', 2),
    ('find two apple and put them in a fridge', 'apple', None, 'fridge', 2),
    ('find two knives and put them on the countertop', 'knife', None, 'countertop', 2),
    ('put two potatoes in fridge', 'potato', None, 'fridge', 2),
])
def test_goal_templates(text, obj, state, target, count):
    goal = parse_goal('You are in a room.\nYour task is to: ' + text + '.')
    assert goal == {'value': {'object_type': obj, 'required_state': state,
                             'target_receptacle_type': target, 'count': count}, 'evidence_steps': [-1]}


def test_unknown_goal_never_uses_task_path_or_hidden_fields():
    table = FactTable.from_record({'observation': 'Your task is to: teleport home.',
        'task_id': 'pick_clean-Apple-Fridge', 'extra': {'expert_plan': 'SECRET'}}, [])
    assert table.goal['value'] is None
    assert table.conditions()['target_location_known']['status'] == 'undetermined'
    assert 'SECRET' not in table.prompt_text()
    assert 'unknown' in table.prompt_text()
    assert parse_goal('Your task is to: find two apples in fridge.')['value'] is None


def test_take_release_inventory_reconciliation_and_repeated_refusal():
    table = FactTable('Your task is to: put a newspaper on a desk.')
    table.update(step('take creditcard 1 from desk 1', 'You pick up the creditcard 1.'))
    for _ in range(2):
        table.update(step('take newspaper 1 from desk 1', 'Nothing happens.', 'inadmissible'))
    assert table.holding['value'] == ['creditcard 1']
    assert table.refused_actions['take newspaper 1 from desk 1'] == {'count': 2, 'evidence_steps': [1, 2]}
    assert len(table.refused_takes_while_holding) == 2
    assert table.conditions()['hands_free']['status'] == 'known_unsatisfied'
    table.update(step('inventory', 'You are carrying: a creditcard 1.'))
    assert table.inventory_checks[-1]['agreement'] is True
    table.update(step('move creditcard 1 to desk 1'))
    assert table.holding['value'] == []
    table.update(step('take newspaper 1 from desk 1'))
    table.update(step('inventory', 'You are not carrying anything.'))
    assert table.inventory_checks[-1]['disagreement']
    assert table.holding == {'value': [], 'evidence_steps': [6]}
    assert table.conditions()['target_held']['status'] == 'refuted'
    table.update(step('inventory', 'You are carrying: a newspaper 2.'))
    assert table.holding['value'] == ['newspaper 2']
    table.update(step('put newspaper 2 on desk 1'))
    assert table.holding['value'] == []
    assert table.conditions()['placed_count']['value'] == 1


def test_closed_open_empty_sightings_and_stale_locations():
    table = FactTable('Your task is to: put an apple in fridge.')
    table.update(step('go to cabinet 1', 'You arrive at cabinet 1. The cabinet 1 is closed.'))
    assert 'cabinet 1' not in table.cleared_receptacles()
    table.update(step('open cabinet 1', 'You open the cabinet 1. The cabinet 1 is open. In it, you see a cup 1, a bowl 2, and an apple 3.'))
    assert table.locations['apple 3']['value'] == 'cabinet 1'
    assert len(table.sightings) == 3
    assert 'cabinet 1' not in table.cleared_receptacles()
    table.update(step('take apple 3 from cabinet 1', 'You pick up the apple 3.'))
    assert table.cleared_receptacles()['cabinet 1']['evidence_steps'] == [-1, 1, 2]
    table.update(step('go to shelf 1', 'You arrive at shelf 1. On the shelf 1, you see nothing.'))
    assert 'shelf 1' in table.cleared_receptacles()
    table.update(step('put apple 3 in cabinet 1'))
    assert 'cabinet 1' not in table.cleared_receptacles()
    table.update(step('examine cabinet 1', 'In it, you see nothing.'))
    assert table.locations['apple 3']['value'] is None
    assert table.conditions()['target_location_known']['status'] == 'refuted'
    table.update(step('close cabinet 1', 'You close the cabinet 1.'))
    assert table.receptacles['cabinet 1']['state']['value'] == 'closed'


def test_open_without_contents_is_not_cleared_and_room_list_is_not_location():
    table = FactTable('You see a fridge 1, a cabinet 1. Your task is to: put an apple in fridge.')
    assert not table.locations
    table.update(step('open cabinet 1', 'You open the cabinet 1.'))
    assert table.receptacles['cabinet 1']['contents_observed']['value'] is True
    assert 'cabinet 1' not in table.cleared_receptacles()
    table.update(step('look', 'You see nothing.'))
    assert 'cabinet 1' in table.cleared_receptacles()


def test_closed_nothing_is_not_a_clear_and_prompt_distinguishes_no_requirement():
    table = FactTable('Your task is to: put an apple in fridge.')
    table.update(step('go to cabinet 1', 'The cabinet 1 is closed. You see nothing.'))
    assert 'cabinet 1' not in table.cleared_receptacles()
    assert table.receptacles['cabinet 1']['contents_observed']['value'] is None
    prompt = json.loads(table.prompt_text())
    assert prompt['goal']['value']['required_state'] == 'not_required'
    assert prompt['holding']['value'] == 'unknown'


def test_two_distinct_objects_counted_once_and_retake_reduces_placed():
    table = FactTable('Your task is to: find two apples and put them in a fridge.')
    for obj in ('apple 1', 'apple 1', 'apple 2'):
        table.update(step(f'take {obj} from shelf 1'))
        table.update(step(f'put {obj} in fridge 1'))
    conditions = table.conditions()
    assert conditions['second_instance']['status'] == 'satisfied'
    assert conditions['placed_count']['value'] == 2
    assert conditions['placed_count']['status'] == 'satisfied'
    table.update(step('take apple 1 from fridge 1'))
    assert table.conditions()['placed_count']['value'] == 1
    assert table.conditions()['placed_count']['status'] == 'known_unsatisfied'


@pytest.mark.parametrize('verb,state,tool', [('heat', 'hot', 'microwave'), ('cool', 'cool', 'fridge'), ('clean', 'clean', 'sinkbasin')])
def test_transform_requires_executed_legal_nonrefused_action(verb, state, tool):
    table = FactTable(f'Your task is to: put a {state} apple in a bowl.')
    table.update(step(f'{verb} apple 1 with {tool} 1', 'Nothing happens.'))
    assert table.conditions()['required_state_done']['status'] == 'undetermined'
    table.update(step(f'{verb} apple 1 with {tool} 1'))
    assert table.state_changes['apple 1'][state]['evidence_steps'] == [1]
    assert table.conditions()['required_state_done']['status'] == 'satisfied'


def test_lamp_use_and_temperature_change():
    table = FactTable('Your task is to: examine a book with the desklamp.')
    table.update(step('use desklamp 1'))
    table.update(step('take book 1 from desk 1'))
    assert table.conditions()['required_state_done']['status'] == 'undetermined'
    table.update(step('use desklamp 1'))
    assert table.lamps['desklamp 1']['evidence_steps'] == [2]
    assert table.conditions()['required_state_done']['status'] == 'satisfied'
    table.update(step('heat book 1 with microwave 1'))
    table.update(step('cool book 1 with fridge 1'))
    assert 'hot' not in table.state_changes['book 1']


def test_failed_and_parser_error_actions_do_not_change_facts():
    table = FactTable('Your task is to: put an apple in fridge.')
    table.update(step('take apple 1 from shelf 1', 'Nothing happens.', 'inadmissible'))
    table.update(step('take apple 1 from shelf 1', 'Parser feedback', 'parser_error'))
    assert table.holding['value'] is None
    assert table.conditions()['target_held']['status'] == 'undetermined'


def test_public_whitelist_incremental_roundtrip_and_prompt_determinism():
    initial = {'visible': {'observation': 'Your task is to: put an apple in a fridge.', 'admissible_commands': ['SECRET']},
               'audit': {'expert_plan': 'SECRET'}, 'task_path': 'SECRET'}
    steps = [dict(step('go to shelf 1'), visible={'observation': 'On the shelf 1, you see an apple 1.',
             'admissible_commands': ['SECRET']}, hidden='SECRET'), step('take apple 1 from shelf 1')]
    a = FactTable.from_record(initial, steps)
    b = FactTable(initial)
    for s in steps:
        b.update(s)
    assert a.to_json() == b.to_json()
    assert a.prompt_text() == b.prompt_text() == a.prompt_text()
    assert 'SECRET' not in a.prompt_text()
    detached = a.to_json()
    detached['holding']['value'].clear()
    assert a.holding['value'] == ['apple 1']


def test_audit_both_record_formats_inventory_unknown_and_errors(tmp_path):
    record = {'initial': 'Your task is to: put an apple in a fridge.',
              'steps': [step('inventory', 'You are not carrying anything.'),
                        step('take apple 1 from shelf 1'), step('inventory', 'You are carrying: an apple 1.')]}
    paths = [tmp_path / 'episode.json', tmp_path / 'rollout.json']
    paths[0].write_text(json.dumps(record))
    paths[1].write_text(json.dumps({'execution': record}))
    result = audit(paths)
    assert result['files'] == 2 and result['exceptions'] == 0
    assert result['goal_parse_coverage'] == 1
    assert result['inventory_agreements'] == 2 and result['inventory_unknown'] == 2
    assert main(['--episodes', str(tmp_path / '*.json'), '--output', str(tmp_path / 'report.json')]) == 0
    broken = tmp_path / 'bad.json'
    broken.write_text('SECRET malformed')
    assert audit([broken])['episodes'][0]['exception'] == 'JSONDecodeError'
