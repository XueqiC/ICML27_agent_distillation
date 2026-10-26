"""Explicit synthetic CLI smoke fixtures, never used for a real run or finding.

Tokenizer protocol mirrors tests/test_r03_sets.py's byte-like TinyTokenizer.
Environment executes public commands and determines terminal success itself.
"""
import json
from repair.prompts import unpack_request
from teachers.openai_teacher import TeacherResult

O = ['go to cabinet 1', 'open cabinet 1', 'take apple 1 from cabinet 1', 'look',
     'go to table 1', 'put apple 1 in/on table 1']
N = [*O[:3], 'clean apple 1 with sinkbasin 1', *O[4:]]
F = [O[1], O[0], O[2], O[4], N[3], O[5]]


class FakeEnvironment:
    seed, max_episode_steps, game_hash = 0, 40, 'synthetic-game'
    def __init__(self):
        self.held = self.clean = self.won = False
    def state(self, text):
        return ({'observation': text, 'admissible_commands': sorted(set(O + N + F))},
                0, self.won, False, {'won': self.won, 'internal_state': 'never in prompts'})
    def reset(self, task_id):
        return self.state('Put a clean apple on table 1. There are cabinet 1, table 1, sinkbasin 1 and apple 1.')
    def step(self, action):
        if action.startswith('take apple'):
            self.held = True
        if action.startswith('clean apple'):
            self.clean = True
        if action.startswith('put apple'):
            self.won = self.held and self.clean
        return self.state('You executed ' + action + '.')
    def close(self):
        pass


class FakeTokenizer:
    chat_template = 'synthetic-gemma-eseries-off-template'
    all_special_ids = [106]
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and not enable_thinking
        text = ''.join('<|turn>' + ('model' if m['role'] == 'assistant' else m['role']) + '\n' +
                       m['content'].strip() + '<turn|>\n' for m in messages)
        return text + ('<|turn>model\n' if add_generation_prompt else '')
    def encode(self, text, *, add_special_tokens):
        out = []
        for i, part in enumerate(text.split('<turn|>')):
            out.extend(([106] if i else []) + [ord(c) + 1000 for c in part])
        return out
    def decode(self, ids, *, skip_special_tokens=False):
        return ''.join('<turn|>' if n == 106 else chr(n - 1000) for n in ids)


class FakeTeacher:
    def generate(self, history, task, visible_context):
        p = unpack_request(visible_context)
        if p.kind == 'diagnosis':
            value = {'expressed_plan': [], 'preservable_choices': [], 'unmet_requirements': [],
                     'candidate_joint_changes': [], 'unknowns': ['Synthetic diagnosis.']}
        elif p.kind == 'rationale':
            value = {'reasoning': 'Use the visible information.', 'evidence_steps': [-1], 'unsupported': False}
        else:
            original_id = p.payload['original_trajectory_id']
            blocks = ([{'block_id': 'clean', 'op': 'replace', 'start': 3, 'end_exclusive': 4, 'actions': [N[3]]}]
                      if p.kind == 'proposal' else
                      [{'block_id': 'far', 'op': 'replace', 'start': 0, 'end_exclusive': 6, 'actions': F}])
            value = {'candidates': [{'original_trajectory_id': original_id, 'blocks': blocks}]}
        return TeacherResult(json.dumps(value), None, {'usage': {'input_tokens': 20, 'output_tokens': 30},
            'response_status': 'completed', 'provider_model_id': 'fake', 'returned_model_id': 'fake',
            'timestamp': 'synthetic', 'cache_hit': False})


def fake_review(public):
    return {'reviewer': 'SYNTHETIC_TEST_ONLY', 'no_rejected_actions': True, 'meaningful_route': True,
            'labels_supported': True, 'notes': 'Synthetic smoke; no human or scientific acceptance.'}


def write_seed_demonstrations(output, tasks):
    """Synthetic stand-ins for existing D0 receipts, saved only in explicit dry runs."""
    from pathlib import Path
    from repair.execute import execute_actions
    from repair.ledger import BudgetLedger
    from sed.storage import write_json, file_hash, digest
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ledger = BudgetLedger(output / 'budget.jsonl')
    refs = {}
    for i, task in enumerate(tasks):
        target = execute_actions(FakeEnvironment, task['task_id'], N, 40, ledger=ledger,
                                 interaction_kind='task_validation_steps')
        episode = {'schema_version': 2, 'episode_id': 'synthetic-seed-' + str(i),
            'task_id': task['task_id'], 'split': 'train', 'sealed': False,
            'initial': {'visible': target.initial['visible']},
            'steps': [{'event_id': f'synthetic-{i}:{j}', 'action_raw': s['action_raw'], 'visible': s['visible']}
                      for j, s in enumerate(target.steps)]}
        directory = output / str(i)
        write_json(directory / 'episode.json', episode, exclusive=True)
        write_json(directory / 'execution.json', target.to_json(), exclusive=True)
        receipt = {'task_id': task['task_id'], 'group_id': task['group_id'], 'won': True,
            'episode_dir': str(i), 'episode_hashes': {'episode.json': file_hash(directory / 'episode.json')},
            'synthetic': True}
        refs[task['task_id']] = [{'task_id': task['task_id'], 'episode_id': episode['episode_id'],
            'verified_collection_path': str(output.resolve()), 'evidence_root': str(output.resolve()),
            'episode_dir': str(i), 'episode_sha256': receipt['episode_hashes']['episode.json'],
            'receipt_sha256': digest(receipt), 'receipt': receipt}]
    write_json(output / 'manifest.json', {'synthetic': True, 'verified': [r[0]['receipt'] for r in refs.values()]},
               exclusive=True)
    for rows in refs.values():
        rows[0]['verified_manifest_sha256'] = file_hash(output / 'manifest.json')
    return refs
