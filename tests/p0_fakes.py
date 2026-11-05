"""Tiny randomly initialized native Gemma4, never a downloaded checkpoint."""
from copy import deepcopy
import json

from repair.execute import execute_actions
from repair.dry_fakes import FakeEnvironment, O
from repair.p0_introspect import StudentTools
from sed.react import build_messages
from sed.storage import digest


def batch_reply(messages, form):
    """Raw replies exercise canonical, legacy and prose-extraction paths."""
    if form == 'canonical':
        return json.dumps(dict(calls=messages))
    if form == 'array':
        return json.dumps(messages)
    if form.startswith('prose_'):
        shape = form.removeprefix('prose_')
        value = messages[0] if shape == 'object' else dict(calls=messages) if shape == 'canonical' else messages
        return "I'll compare the actions first, then test the edit." + json.dumps(value) + '\nDone.'
    separator = {'concatenated': '', 'newlines': '\n', 'whitespace': ' \t\r\n '}[form]
    return ' \n' + separator.join(json.dumps(message) for message in messages) + '\t '


class TinyTokenizer:
    all_special_ids = [1, 50, 106]

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, i = [], [], 0
        while i < len(text):
            if text.startswith('<turn|>', i):
                ids.append(106); offsets.append((i, i + 7)); i += 7
            else:
                ids.append(ord(text[i]) % 256 + 128); offsets.append((i, i + 1)); i += 1
        result = dict(input_ids=ids)
        if return_offsets_mapping:
            result['offset_mapping'] = offsets
        return result

    def encode(self, text, *, add_special_tokens=False):
        return self(text)['input_ids']

    def decode(self, ids, *, skip_special_tokens=False):
        return ''.join('<turn|>' if int(i) == 106 else chr((int(i) - 128) % 256) for i in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and not enable_thinking
        result = ''.join('<' + m['role'] + '>\n' + m['content'] + '<turn|>\n' for m in messages)
        return result + ('<assistant>\n' if add_generation_prompt else '')


def tiny_student(layers=4):
    import torch
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = Gemma4TextConfig(vocab_size=512, hidden_size=16, intermediate_size=24,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, num_kv_shared_layers=2,
        layer_types=['sliding_attention', 'full_attention'] * (layers // 2),
        hidden_size_per_layer_input=0, sliding_window=16, max_position_embeddings=4096,
        rope_parameters={'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0},
                         'full_attention': {'rope_type': 'default', 'rope_theta': 10000.0}})
    config._attn_implementation = 'sdpa'
    model = Gemma4ForCausalLM(config).to(torch.bfloat16)
    return StudentTools(model, TinyTokenizer(), identity={'fixture': 'tiny-native-gemma4', 'layers': layers},
                        max_new_tokens=4, max_model_len=8192)


def fake_pair(index=0, *, category='refused_action', split='diagnosis', step=1):
    original = execute_actions(FakeEnvironment, 'train/synth/fake',
        ['Thought: Inspect the visible room.\nAction: ' + a for a in O[:3]], 40)
    context = original.decision_context(step)
    messages = build_messages(context['history'], {'task_id': original.task_id}, context['visible_context'])
    return dict(pair_id=f'pair-{index}', attempt_id=f'pair-{index}', category=category, split=split,
        task_id=original.task_id, group_id='fake-group', parent_seed_task_id='train/seed/fake',
        original_action_raw=original.steps[step]['action_raw'],
        step=step, a_f='look', a_r='inventory', decoding='greedy', sampling_seed=None,
        alignment='final_execution_first_divergence', state_match='verified',
        category_source='at_divergence', category_distance=0,
        state=dict(context=context, messages=messages, sha256=digest(messages)),
        review=dict(step=step, category=category, reason='Public evidence only.'),
        audit=dict(original=original.to_json(), env_seed=0))


def sample_submission(direction_id, layers=None):
    return dict(mechanism='A falsifiable fixture hypothesis.',
        locus=dict(layers=layers or [0], spans=['latest_observation'], direction='fixture fail-minus-fix'),
        edit=dict(direction_id=direction_id, layers=layers or [0], targets=['mlp.down_proj'], alpha=0.5),
        predictions=[dict(metric='rollout_gain_vs_none', direction='>', threshold=0.0)])


def fake_greedy(state):
    """Deterministic continuation; success still comes from FakeEnvironment."""
    if 'You executed clean apple 1 with sinkbasin 1.' in state['text']:
        action = 'put apple 1 in/on table 1'
    elif 'You executed take apple 1 from cabinet 1.' in state['text']:
        action = 'clean apple 1 with sinkbasin 1'
    else:
        action = 'take apple 1 from cabinet 1'
    return dict(action=action, reply='Action: ' + action, prompt_tokens=1, generated_tokens=1)
