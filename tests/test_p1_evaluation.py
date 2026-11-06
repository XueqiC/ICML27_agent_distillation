from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from sed.storage import digest, write_json
from surgery.common import seal, provenance, read
from surgery.conditions import freeze
from surgery.evaluation import evaluate_shard, merge, shard_owner, metric_rate
from surgery.export import export_model, verify_export
from surgery.edits import save_delta
from train.merge_lora import next_token_logits
from tests.test_p1_student import student, project, rank1, scale, hashes


def checkpoint_student(tmp_path):
    """Use native checkpoint loading, including Gemma4's regenerated fp32 RoPE buffers."""
    from transformers import Gemma4ForCausalLM
    from surgery.student import Student
    s = student()
    path = tmp_path / 'initial'
    s.model.save_pretrained(path)
    model = Gemma4ForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    return Student(model, s.tokenizer, identity=s.identity, max_new_tokens=4, max_model_len=8192)


def fixture(tmp_path):
    s = student()
    conditions = [freeze(tmp_path / 'conditions' / (name + '.json'), s, 'pairs', 'A', name, [], trial=True) for name in ('E0', 'E1')]
    pairs = dict(setting='A', sha256='pairs')
    units = [dict(condition=c, metric=m, key=k) for c in ('E0', 'E1')
             for m in ('M1', 'M2_probe', 'M2_k20', 'M3') for k in ('one', 'two')]
    plan = seal(dict(schema='p1-evaluation-plan/1', pairs_sha256='pairs', setting='A', trial=True, shards=2,
        conditions={c['name']: c['sha256'] for c in conditions}, units=units, k20_label='original K20 tasks',
        code_sha256=provenance()['code_sha256'], dependencies=provenance()['dependencies']))
    write_json(tmp_path / 'eval/plan.json', plan)
    calls = []
    def factory(condition, directory):
        def evaluate(unit, edits):
            calls.append(unit)
            if unit['metric'] == 'M3':
                return dict(greedy='look' if unit['condition'] == 'E0' else 'inventory')
            return dict(success=unit['key'] == 'one', greedy='look', a_f='look')
        return SimpleNamespace(student=s, evaluate=evaluate)
    return pairs, conditions, plan, calls, factory


def test_shards_resume_cpu_merge_and_flip_denominator(tmp_path):
    pairs, conditions, plan, calls, factory = fixture(tmp_path)
    for i in range(2):
        result = evaluate_shard(pairs, conditions, tmp_path / 'eval', plan, shard=(i, 2), runtime_factory=factory)
        assert result['completed'] == result['assigned']
    assert len(calls) == len(plan['units'])
    assert all('rounding' in read(p)['edit_receipt'] for p in (tmp_path / 'eval/units').glob('*.json'))
    before = len(calls)
    for i in range(2):
        evaluate_shard(pairs, conditions, tmp_path / 'eval', plan, shard=(i, 2), runtime_factory=factory, resume=True)
    assert len(calls) == before
    report = merge(pairs, conditions, tmp_path / 'eval')
    assert report['metrics']['E0']['M1']['value'] == 0.5
    assert report['metrics']['E1']['M3']['value'] == 1
    assert report['metrics']['E1']['decision_flip_rate']['denominator'] == 2
    assert 'mcnemar_vs_unedited' in report['appendix']
    assert report['metrics']['E1']['M4']['value'] is None
    assert (tmp_path / 'eval/report.md').is_file()
    assert 'p_value' not in (tmp_path / 'eval/report.md').read_text()


def test_resume_requires_explicit_flag_and_rejects_corruption(tmp_path):
    pairs, conditions, plan, calls, factory = fixture(tmp_path)
    evaluate_shard(pairs, conditions, tmp_path / 'eval', plan, shard=(0, 2), runtime_factory=factory)
    with pytest.raises(FileExistsError):
        evaluate_shard(pairs, conditions, tmp_path / 'eval', plan, shard=(0, 2), runtime_factory=factory)
    path = next((tmp_path / 'eval/units').glob('*.json'))
    row = read(path); row['result']['success'] = 'tampered'; write_json(path, row)
    with pytest.raises(ValueError):
        evaluate_shard(pairs, conditions, tmp_path / 'eval', plan, shard=(0, 2), runtime_factory=factory, resume=True)


def test_missing_shard_and_wrong_shard_count(tmp_path):
    pairs, conditions, plan, calls, factory = fixture(tmp_path)
    with pytest.raises(ValueError, match='Shard'):
        evaluate_shard(pairs, conditions, tmp_path / 'eval', plan, shard=(0, 3), runtime_factory=factory)
    evaluate_shard(pairs, conditions, tmp_path / 'eval', plan, shard=(0, 2), runtime_factory=factory)
    with pytest.raises(ValueError, match='Missing'):
        merge(pairs, conditions, tmp_path / 'eval')


def test_unknowns_not_false():
    assert metric_rate([True, False, 'unknown']) == dict(value=0.5, numerator=1, denominator=2, total=3, unknown=1)
    assert metric_rate(['unknown'])['value'] is None


@pytest.mark.parametrize('sharded', [False, True])
def test_export_tiny_native_model_manifest_reload_and_tamper(tmp_path, monkeypatch, sharded):
    from transformers import Gemma4ForCausalLM
    s = checkpoint_student(tmp_path)
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    s.model.save_pretrained(snapshot)
    (snapshot / 'chat_template.jinja').write_text('fixture chat template')
    (snapshot / 'tokenizer.json').write_text('{}')
    delta = save_delta(tmp_path / 'deltas', {'0:mlp.down_proj': torch.full((16, 24), 1e-5),
        '1:mlp.down_proj': torch.full((16, 24), 1e-6)}, {})
    s.layers[1].mlp.down_proj.weight.data.fill_(1)  # This entire tensor's edit vanishes.
    edits = [project(s), rank1(s), scale(), dict(type='local_finetune', delta=delta, alpha=1)]
    condition = freeze(tmp_path / 'condition.json', s, 'pairs', 'A', 'E1', edits)
    before = hashes(s)
    originals = dict(s.model.named_parameters())
    inputs = [[130, 131, 132], [135, 136]]
    with s.edited(condition['edits']) as receipt:
        expected = {n: p.detach().clone() for n, p in s.model.named_parameters()}
        expected_logits = [next_token_logits(s.model, ids) for ids in inputs]
    if sharded:
        save = s.model.save_pretrained
        monkeypatch.setattr(s.model, 'save_pretrained', lambda p, **kw: save(p, max_shard_size='10KB', **kw))
    def loader(path):
        model = Gemma4ForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
        for name, parameter in model.named_parameters():
            assert parameter.dtype == expected[name].dtype
            assert torch.equal(parameter.detach().contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8))
        assert all(torch.equal(want, next_token_logits(model, ids)) for want, ids in zip(expected_logits, inputs))
        return model
    result = export_model(s, condition, snapshot, tmp_path / 'export',
        loader=loader, inputs=inputs)
    assert result['passed'] and result['dtype'] == 'bfloat16'
    assert result['edited_weights_exact'] and result['edit_receipt'] == receipt
    assert hashes(s) == before
    assert all(p is originals[n] for n, p in s.model.named_parameters())
    assert verify_export(tmp_path / 'export') == result
    report = read(tmp_path / 'export/consistency.json')
    assert set(report['edited_weight_checks']) == {'0:mlp.down_proj', '1:mlp.down_proj'}
    assert all(c['exact'] and c['installed_sha256'] == c['exported_sha256'] for c in report['edited_weight_checks'].values())
    assert all(c['bf16_reload']['passed'] and c['bf16_reload']['exact'] for c in report['checks'])
    (tmp_path / 'export/chat_template.jinja').write_text('modified')
    with pytest.raises(ValueError, match='manifest'):
        verify_export(tmp_path / 'export')


def test_export_rejects_byte_mismatch_even_when_logits_match(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file
    from transformers import Gemma4ForCausalLM
    s = checkpoint_student(tmp_path)
    original = s.layers[0].mlp.down_proj.weight
    original.data[0, 0] = -0.0
    before = hashes(s)
    snapshot = tmp_path / 'snapshot'; snapshot.mkdir()
    s.model.save_pretrained(snapshot)
    condition = freeze(tmp_path / 'condition.json', s, 'pairs', 'A', 'E1', [project(s, 0)])
    save = s.model.save_pretrained
    def corrupt(path, **kwargs):
        save(path, **kwargs)
        weights = load_file(path / 'model.safetensors')
        key = next(n for n in weights if n.endswith('layers.0.mlp.down_proj.weight'))
        weights[key][0, 0] = 0.0  # Numerically equal, but wrong sign bit.
        save_file(weights, path / 'model.safetensors', metadata={'format': 'pt'})
    monkeypatch.setattr(s.model, 'save_pretrained', corrupt)
    with pytest.raises(ValueError, match='byte-for-byte'):
        export_model(s, condition, snapshot, tmp_path / 'export',
            loader=lambda p: Gemma4ForCausalLM.from_pretrained(p, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa'),
            inputs=[[130, 131]])
    report = read(tmp_path / 'export/consistency.json')
    assert not report['passed'] and not report['edited_weights_exact']
    assert all(c['bf16_reload']['passed'] and c['bf16_reload']['exact'] for c in report['checks'])
    assert hashes(s) == before and s.layers[0].mlp.down_proj.weight is original
    with pytest.raises(ValueError, match='consistency'):
        verify_export(tmp_path / 'export')


def test_export_rejects_incorrect_reload(tmp_path):
    s = student()
    snapshot = tmp_path / 'snapshot'; snapshot.mkdir()
    s.model.save_pretrained(snapshot)
    condition = freeze(tmp_path / 'condition.json', s, 'pairs', 'A', 'E0', [])
    def wrong_loader(path):
        other = student().model
        other.get_output_embeddings().weight.data.fill_(0)
        return other
    with pytest.raises(ValueError, match='logits'):
        export_model(s, condition, snapshot, tmp_path / 'export', loader=wrong_loader, inputs=[[130, 131]])
    assert read(tmp_path / 'export/surgery_manifest.json')['passed'] is False


def test_probe_rollout_starts_at_initial_state(tmp_path):
    from surgery.evaluation import Runtime
    from repair.dry_fakes import FakeEnvironment
    from tests.p0_fakes import fake_greedy
    runtime = Runtime.__new__(Runtime)
    runtime.student, runtime.directory = student(), tmp_path
    observed = []
    def greedy(state):
        observed.append(state['text'])
        return fake_greedy(state)
    runtime.student.greedy = greedy
    runtime.manifest = SimpleNamespace(authorize=lambda tid, probes: None,
        environment=lambda tid, probes, seed: FakeEnvironment())
    result = runtime.probe('train/synth/fake', [])
    assert result['success'] is True
    assert result['edit_receipt']['native_execution']
    assert result['edit_receipt']['rounding']['rounded_changed_parameters'] == 0
    assert result['environment_steps'] > 0
    assert 'You executed' not in observed[0]
    assert len(list((tmp_path / 'probe_executions').glob('*.json'))) == 1


def test_m4_import_binds_condition_and_standalone_protocol(tmp_path, monkeypatch):
    from surgery.evaluation import import_m4
    condition = dict(name='E1', sha256='condition')
    exported = tmp_path / 'model'
    launch = seal(dict(schema='p1-m4-launch/1', condition_sha256='condition', task_list_sha256='tasks',
        export_manifest_sha256='export', exported=str(exported), conditions={'E1': 'condition'}))
    write_json(tmp_path / 'job/p1_launch.json', launch)
    summary = dict(episodes=140, task_list_sha256='tasks', batch_invariant=True, policy='student',
        student=dict(decoding='greedy'), success_rate=0.6, successes=84, unknown_success=0)
    write_json(tmp_path / 'job/eval/summary.json', summary)
    write_json(tmp_path / 'job/server/provenance.json', dict(config=dict(server=dict(
        merged_model=True, batch_invariant=True, model_path=str(exported)))))
    monkeypatch.setattr('surgery.export.verify_export', lambda p: dict(sha256='export'))
    result = import_m4([f'E1={tmp_path}/job'], [condition])
    assert result['E1']['value'] == 0.6
    summary['student']['decoding'] = 'sampled'
    write_json(tmp_path / 'job/eval/summary.json', summary)
    with pytest.raises(ValueError, match='protocol'):
        import_m4([f'E1={tmp_path}/job'], [condition])


def test_m4_rejects_trial_before_dev140_access(tmp_path, monkeypatch):
    from surgery.export import record_m4_launch
    monkeypatch.setattr('surgery.export.verify_export', lambda p: dict(trial=True))
    with pytest.raises(ValueError, match='Trial exports'):
        record_m4_launch('fixture', '/must-not-read-dev140', '/must-not-read-seeds', tmp_path, '/absent-conditions')
