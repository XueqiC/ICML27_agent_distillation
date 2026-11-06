"""Teacher payload sizes and lossless-on-disk rounding summaries (CPU only)."""
from copy import deepcopy
import json

import pytest

from sed.storage import canonical_bytes, digest
from surgery.agent import compact, summarize_tool_result
from surgery.edits import TARGETS, rounding_summary
from tests.test_p1_agent import session


def payload_fixtures():
    """Deterministic schema fixtures, NOT measurements from the GPU trial.

    Ranking: 32 layers x seven projections. Edit: 16 tensors, one prefix.
    Long numerical values represent normal full-precision JSON float rendering.
    """
    ranking = [dict(tensor=f'{layer}:{target}', layer=layer, module=target,
                    gradient_norm=0.12345678901234567, noise_norm=0.23456789012345678,
                    snr=0.5263157894736842)
               for layer in range(32) for target in TARGETS]
    tensor = dict(parameters=1048576, native_dtype='bfloat16', fp32_changed_parameters=1048576,
                  rounded_changed_parameters=524288, rounded_finite=True,
                  original_frobenius=12.345678901234567, rounded_delta_frobenius=0.12345678901234567,
                  rounded_relative_frobenius=0.01, vanished=False)
    tensors = {f'{layer}:mlp.down_proj': dict(tensor) for layer in range(16)}
    # One tensor whose fp32 change vanished after rounding, retained in counts.
    tensors['0:mlp.down_proj'].update(rounded_changed_parameters=0,
        rounded_delta_frobenius=0.0, rounded_relative_frobenius=0.0, vanished=True)
    per_edit = dict(rounding_summary(tensors), index=0, type='project_out',
                    fp32_effect_changed_parameters=16 * 1048576,
                    rounded_effect_changed_parameters=15 * 524288, effect_vanished=False)
    receipt = dict(edited_parameters=16 * 1048576, edited_fraction=0.012345678901234567,
                   fp32_composition=True, native_execution=True, exact_rollback=True,
                   edit_sha256='e' * 64, rounding=dict(rounding_summary(tensors), per_edit=[per_edit]))
    data = {
        'gradient_attribution': dict(ranking=ranking, unavailable=[], pairs=6,
            framing='original_thought', definition='Frobenius norm of gradient of mean(log P(a_r)-log P(a_f)); native absent shared-KV tensors omitted'),
        'apply_edit': dict(receipt=receipt, before=dict(scores=[]), after=dict(scores=[]),
                           edit=[], reverted=True, framing='original_thought'),
        'rollout': dict(edit_receipt=receipt, success=False, verification='real_environment',
                        environment_steps=4, execution_sha256='f' * 64),
    }
    return {tool: dict(tool=tool, data=value, summary=tool + ': fixture',
                      sha256=digest(value), cache_key='a' * 64, cache_hit=False, framing='original_thought')
            for tool, value in data.items()}


def test_payload_sizes_and_summary_contents():
    full = payload_fixtures()
    original = deepcopy(full)
    sizes = {}
    for tool, result in full.items():
        view = summarize_tool_result(result)
        sizes[tool] = dict(before=len(canonical_bytes(result)), after=len(canonical_bytes(view)))
        assert view['sha256'] == result['sha256']
        if tool == 'gradient_attribution':
            assert view == result  # Preserve the complete ranking for decisions.
            continue
        name = 'receipt' if tool == 'apply_edit' else 'edit_receipt'
        rounding = view['data'][name]['rounding']
        for small, big in zip([rounding, *rounding['per_edit']],
                              [result['data'][name]['rounding'], *result['data'][name]['rounding']['per_edit']]):
            assert 'tensors' not in small
            assert small['tensor_count'] == 16 and small['vanished_tensor_count'] == 1
            assert small['tensor_details_on_disk']
            assert {k: v for k, v in small.items() if k in big and k != 'per_edit'} == {
                k: v for k, v in big.items() if k not in ('tensors', 'per_edit')}
        assert sizes[tool]['after'] < sizes[tool]['before'] / 4
    assert full == original
    print('P1 synthetic payload sizes (canonical UTF-8 bytes): ' + json.dumps(sizes, sort_keys=True))


@pytest.mark.parametrize('tool,name', [('apply_edit', 'receipt'), ('rollout', 'edit_receipt')])
def test_teacher_summary_leaves_real_cpu_receipts_and_cache_intact(tmp_path, tool, name):
    sess = session(tmp_path)
    edit = [dict(type='scale_neurons', layers=[0], neurons={'0': [0, 2]}, c=0.5)]
    args = dict(state='pair-0', edit=edit)
    if tool == 'apply_edit':
        args['candidates'] = ['look', 'inventory']
    result = sess.call(tool, args)
    receipt = sess.cache / (result['cache_key'] + '.json')
    before = receipt.read_bytes()
    transcript = [dict(request=dict(tool=tool, args=args), result=result)]
    original = deepcopy(transcript)
    view = compact(transcript)[0]['result']
    assert 'tensors' not in view['data'][name]['rounding']
    assert transcript == original
    assert receipt.read_bytes() == before
    assert json.loads(before)['data'][name]['rounding']['tensors']
    cached = sess.call(tool, args)
    assert cached['cache_hit']
    assert cached['data'] == result['data']
    assert receipt.read_bytes() == before
    settled = [json.loads(line) for line in sess.ledger.path.read_text().splitlines()
               if json.loads(line)['event'] == 'settled']
    assert all(row['result']['data'][name]['rounding']['tensors'] for row in settled)


def test_twenty_result_window_still_preserves_ranking_and_recent_receipt_totals():
    fixtures = payload_fixtures()
    transcript = [dict(request=dict(tool='apply_edit', args={}), result=fixtures['apply_edit']) for _ in range(3)]
    transcript += [dict(request=dict(tool='gradient_attribution', args={}), result=fixtures['gradient_attribution']) for _ in range(19)]
    transcript.append(dict(request=dict(tool='rollout', args={}), result=fixtures['rollout']))
    view = compact(transcript)
    assert all(row['result']['compacted'] for row in view[:3])
    assert view[3:-1] == transcript[3:-1]
    assert view[-1]['result']['data']['edit_receipt']['rounding']['tensor_count'] == 16
    assert transcript[-1]['result']['data']['edit_receipt']['rounding']['tensors']
