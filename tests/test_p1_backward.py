"""CPU regression oracle: original P1 backward methods from b800d97.

LegacyStudent is deliberately uncheckpointed and returns all weight gradients;
keep it independent of the production memory helpers.
"""
from copy import deepcopy
import math
from types import SimpleNamespace
import weakref

import pytest
import torch

from repair.p0_introspect import tensor_hash
from sed.storage import digest
from surgery.student import Student
from surgery.edits import all_modules, module_at, tensor_name, save_delta, load_delta, finite
from surgery.backward import checkpoint_layers, peak_memory, PEAK_MEMORY
from tests.p0_fakes import tiny_student


class LegacyStudent(Student):
    def gradient_statistics(self, examples, *, margin=False, framing='original_thought'):
        """Stream per-pair gradients; fp32 CPU moments, no train updates."""
        import torch
        if not examples:
            raise ValueError('Choose at least one diagnosis pair')
        modules = all_modules(self)
        sums, squares = {}, {}
        with self.differentiable(modules):
            for state, a_f, a_r in examples:
                objective = self.log_probability(state, a_r, framing=framing)
                objective = objective - self.log_probability(state, a_f, framing=framing) if margin else -objective
                gradients = torch.autograd.grad(objective, [m.weight for m in modules.values()], allow_unused=True)
                for key, module, grad in zip(modules, modules.values(), gradients):
                    g = torch.zeros_like(module.weight, dtype=torch.float32, device='cpu') if grad is None else grad.detach().cpu().float()
                    if key not in sums:
                        sums[key] = torch.zeros_like(g)
                        squares[key] = 0.0
                    sums[key].add_(g)
                    squares[key] += float(g.square().sum())
        rows, n = [], len(examples)
        for key in modules:
            mean = sums.pop(key) / n
            norm = float(mean.norm())
            noise = math.sqrt(max(0, squares[key] / n - norm ** 2))
            layer, target = key.split(':')
            rows.append(dict(tensor=key, layer=int(layer), module=target, gradient_norm=norm,
                             noise_norm=noise, snr=norm / max(noise, 1e-12)))
        return rows

    def gradient_attribution(self, examples, *, framing='original_thought'):
        rows = self.gradient_statistics(examples, margin=True, framing=framing)
        from surgery.edits import TARGETS
        present = {r['tensor'] for r in rows}
        return dict(ranking=sorted(rows, key=lambda r: (-r['gradient_norm'], r['tensor'])),
            unavailable=[dict(layer=i, module=t, status='absent_native_projection')
                         for i in range(len(self.layers)) for t in TARGETS if tensor_name(i, t) not in present],
            definition='Frobenius norm of gradient of mean(log P(a_r)-log P(a_f)); native absent shared-KV tensors omitted',
            pairs=len(examples), framing=framing)

    def neuron_attribution(self, examples, layer, top_n, *, framing='original_thought'):
        import torch
        module = module_at(self, layer, 'mlp.down_proj')
        width = module.weight.shape[1]
        if type(top_n) is not int or not 1 <= top_n <= width or not examples:
            raise ValueError('top_n must be between 1 and the MLP intermediate width')
        total = torch.zeros(width)
        for state, a_f, a_r in examples:
            decision = len(self.ids(self.decision_prefix(state, framing))) - 1
            for action, sign in ((a_r, 1), (a_f, -1)):
                captured = []
                def hook(mod, args):
                    activation = args[0].detach().requires_grad_(True)
                    captured.append(activation)
                    return (activation, *args[1:])
                handle = module.register_forward_pre_hook(hook)
                try:
                    objective = self.log_probability(state, action, framing=framing)
                    grad, = torch.autograd.grad(objective, captured)
                    total += sign * (captured[0][0, decision].detach().float() * grad[0, decision].float()).cpu()
                finally:
                    handle.remove()
        total /= len(examples)
        # IDs are usable as the input-side vector b of a down_proj rank-one edit.
        norm = total.norm()
        if not torch.isfinite(norm) or norm < 1e-12:
            raise ValueError('Zero/nonfinite neuron attribution')
        vectors = {layer: total / norm}
        ident = digest({str(i): tensor_hash(v) for i, v in vectors.items()})
        self.directions[ident] = vectors
        order = sorted(range(width), key=lambda i: (-abs(float(total[i])), i))[:top_n]
        return dict(layer=layer, top_neurons=[dict(neuron=i, attribution=float(total[i])) for i in order],
            direction_id=ident, vector_space='MLP intermediate', framing=framing,
            definition='mean signed activation x gradient of repair-minus-fail sequence score at last decision-prefix token; ranked by magnitude')

    def local_finetune(self, examples, modules, *, rank, lr, steps, directory, pair_ids, framing='original_thought'):
        """Train only LoRA factors; persist fp32 BA and restore all hooks/weights."""
        import torch
        import torch.nn.functional as F
        if self.active_edit is not None:
            raise ValueError('Local finetune must start from the unedited session student')
        if (type(rank) is not int or not 1 <= rank <= 16 or type(steps) is not int or not 1 <= steps <= 200
                or finite(lr) <= 0 or not examples or len(examples) != len(pair_ids)):
            raise ValueError('Local finetune requires rank 1..16, steps 1..200, positive lr and diagnosis pairs')
        if not modules or len(set(modules)) != len(modules):
            raise ValueError('Choose distinct layer:module tensors')
        selected = {}
        for key in modules:
            layer, target = key.split(':')
            selected[key] = module_at(self, int(layer), target)
        factors, handles, hashes = {}, [], {k: tensor_hash(m.weight) for k, m in selected.items()}
        from pathlib import Path
        from sed.storage import JsonlLedger
        training_id = digest(dict(pair_ids=pair_ids, modules=modules, rank=rank, lr=lr, steps=steps,
                                  framing=framing, student=self.identity))
        ledger = JsonlLedger(Path(directory) / (training_id + '.training.jsonl'))
        training = self.model.training
        usage_before = deepcopy(self.usage)
        losses, exposures = [], []
        try:
            # Local RNG context keeps future controls/generation independent of training.
            with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == 'cuda' else []):
                torch.manual_seed(0)
                for key, module in selected.items():
                    out_dim, in_dim = module.weight.shape
                    a = torch.nn.Parameter(torch.randn(rank, in_dim, device=self.device, dtype=torch.float32) / math.sqrt(in_dim))
                    b = torch.nn.Parameter(torch.zeros(out_dim, rank, device=self.device, dtype=torch.float32))
                    factors[key] = (a, b)
                    def hook(mod, args, output, a=a, b=b):
                        return (output.float() + F.linear(F.linear(args[0].float(), a), b)).to(output.dtype)
                    handles.append(module.register_forward_hook(hook))
                optimizer = torch.optim.AdamW([p for ab in factors.values() for p in ab], lr=float(lr), weight_decay=0)
                self.model.eval()
                for i in range(steps):
                    index = i % len(examples)
                    state, _, action = examples[index]
                    ledger.append(dict(event='exposure_started', step=i, pair_id=pair_ids[index], target=action,
                                       state_sha256=digest(state), seed=0))
                    optimizer.zero_grad(set_to_none=True)
                    loss = -self.log_probability(state, action, framing=framing)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite local finetune loss')
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    exposures.append(pair_ids[index])
                    ledger.append(dict(event='updated', step=i, pair_id=pair_ids[index], loss=losses[-1],
                                       student_usage=deepcopy(self.usage)))
                deltas = {k: (b.detach() @ a.detach()).cpu().float() for k, (a, b) in factors.items()}
        except Exception as exc:
            ledger.append(dict(event='failed', error_type=type(exc).__name__,
                completed_steps=len(losses), student_usage=deepcopy(self.usage)))
            raise
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(training)
            if any(tensor_hash(m.weight) != hashes[k] for k, m in selected.items()):
                raise RuntimeError('LoRA altered a base weight')
        metadata = dict(student=self.identity, pair_ids=pair_ids, modules=modules, rank=rank, lr=lr, steps=steps,
            framing=framing, loss='-log P(Action: a_r + turn-end | state + framing)', optimizer='AdamW',
            weight_decay=0, lora_scale=1, dropout=0, seed=0, batch_size=1, pair_order='cyclic supplied order',
            losses=losses, training_exposures=exposures,
            student_usage={k: self.usage[k] - v for k, v in usage_before.items()})
        ref = save_delta(directory, deltas, metadata)
        return dict(edit=dict(type='local_finetune', delta=ref, alpha=1.0), delta_sha256=ref['sha256'],
                    training_exposures=len(exposures), losses=losses, reverted=True)

EXAMPLES = [(dict(text=text, spans=[], original_thought={'prefix': ''}), fail, repair)
            for text, fail, repair in [('room', 'look', 'inventory'), ('door', 'go north', 'look'),
                                       ('cup', 'look', 'take cup')]]


def students(dtype=torch.float32, per_layer=0, shared=2):
    base = tiny_student()
    if per_layer or shared != 2:
        from transformers import Gemma4ForCausalLM
        config = deepcopy(base.config)
        config.hidden_size_per_layer_input = per_layer
        config.vocab_size_per_layer_input = 512
        config.num_kv_shared_layers = shared
        base.model = Gemma4ForCausalLM(config).to(torch.bfloat16)
    result = [cls(deepcopy(base.model), base.tokenizer, identity=base.identity,
                  max_new_tokens=4, max_model_len=8192) for cls in (Student, LegacyStudent)]
    for s in result:
        s.model.to(dtype)
    return result


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('per_layer', [0, 4])
@pytest.mark.parametrize('margin', [False, True])
def test_statistics_exact_old_vs_new(dtype, per_layer, margin):
    s, old = students(dtype, per_layer)
    assert s.gradient_statistics(EXAMPLES, margin=margin) == old.gradient_statistics(EXAMPLES, margin=margin)
    assert s.usage == old.usage


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('per_layer', [0, 4])
def test_gradient_attribution_exact_old_vs_new(dtype, per_layer):
    s, old = students(dtype, per_layer)
    assert s.gradient_attribution(EXAMPLES) == old.gradient_attribution(EXAMPLES)
    assert s.usage == old.usage


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('layer', [0, 1, 2, 3])
@pytest.mark.parametrize('shared', [0, 2])
def test_neuron_attribution_exact_old_vs_new(dtype, layer, shared):
    s, old = students(dtype, per_layer=4, shared=shared)
    try:
        expected = old.neuron_attribution(EXAMPLES, layer, 6)
    except ValueError as exc:
        # Past the shared-KV producers, a prefix-token MLP cannot affect later
        # action tokens. Preserve the existing zero-attribution rejection too.
        with pytest.raises(ValueError, match=str(exc)):
            s.neuron_attribution(EXAMPLES, layer, 6)
    else:
        assert s.neuron_attribution(EXAMPLES, layer, 6) == expected
    assert s.usage == old.usage
    for ident in s.directions:
        assert torch.equal(s.directions[ident][layer], old.directions[ident][layer])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('modules', [['0:mlp.down_proj'],
    ['0:self_attn.k_proj', '1:mlp.up_proj', '2:self_attn.q_proj', '3:mlp.down_proj']])
def test_finetune_exact_old_vs_new(tmp_path, dtype, modules):
    s, old = students(dtype, per_layer=4)
    options = dict(rank=2, lr=0.003, steps=5, pair_ids=['a', 'b', 'c'])
    new = s.local_finetune(EXAMPLES, modules, directory=tmp_path / 'new', **options)
    prior = old.local_finetune(EXAMPLES, modules, directory=tmp_path / 'old', **options)
    assert {k: v for k, v in new.items() if k != 'edit'} == {k: v for k, v in prior.items() if k != 'edit'}
    a, b = new['edit']['delta'], prior['edit']['delta']
    assert a['metadata'] == b['metadata']
    for key, value in load_delta(a).items():
        assert torch.equal(value, load_delta(b)[key])
    assert s.usage == old.usage


def invoke(s, tool, directory):
    if tool == 'local_finetune':
        return s.local_finetune(EXAMPLES, ['0:mlp.down_proj', '3:self_attn.q_proj'], rank=2,
                                lr=0.003, steps=2, directory=directory, pair_ids=['a', 'b', 'c'])
    if tool == 'neuron_attribution':
        return s.neuron_attribution(EXAMPLES, 0, 6)
    return getattr(s, tool)(EXAMPLES)


TOOLS = ['gradient_statistics', 'gradient_attribution', 'neuron_attribution', 'local_finetune']


def snapshot(s):
    return dict(layers=list(s.layers), forwards=[dict(m.__dict__).get('forward') for m in s.layers],
        own_forward=['forward' in m.__dict__ for m in s.layers],
        parameters=[(p, p.detach().clone(), p.requires_grad, p.grad) for p in s.model.parameters()],
        buffers=[(b, b.clone()) for b in s.model.buffers()], modes=[m.training for m in s.model.modules()],
        hooks=[(m, name, dict(getattr(m, name))) for m in s.model.modules()
               for name in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks')],
        rng=torch.random.get_rng_state().clone())


def assert_restored(s, before):
    assert list(s.layers) == before['layers']
    for layer, own, forward in zip(s.layers, before['own_forward'], before['forwards']):
        assert ('forward' in layer.__dict__) == own
        assert layer.__dict__.get('forward') is forward
    for p, (original, value, flag, grad) in zip(s.model.parameters(), before['parameters']):
        assert p is original and torch.equal(p, value) and p.requires_grad == flag and p.grad is grad
        assert not p._post_accumulate_grad_hooks
    for b, (original, value) in zip(s.model.buffers(), before['buffers']):
        assert b is original and torch.equal(b, value)
    assert [m.training for m in s.model.modules()] == before['modes']
    for m, name, hooks in before['hooks']:
        assert dict(getattr(m, name)) == hooks
    assert torch.equal(torch.random.get_rng_state(), before['rng'])


@pytest.mark.parametrize('tool', TOOLS)
@pytest.mark.parametrize('failure', [None, 'forward', 'backward', 'replay'])
def test_exact_restoration_including_failure(tmp_path, monkeypatch, tool, failure):
    s, _ = students(torch.bfloat16, per_layer=4)
    # Preserve an existing instance override, observer hooks and preexisting grad.
    original = s.layers[0].forward
    s.layers[0].forward = lambda *a, **kw: original(*a, **kw)
    handle = s.layers[1].register_forward_hook(lambda *a: None)
    if tool in ('gradient_statistics', 'gradient_attribution'):
        p = s.layers[0].mlp.up_proj.weight
        p.requires_grad_(True)
        p.grad = torch.ones_like(p)
    logits = s._forward([130, 131, 132]).logits.clone()
    before = snapshot(s)
    if failure == 'replay':
        from torch.utils import checkpoint as cp
        native = cp.checkpoint
        def checkpoint(function, *a, **kw):
            replay = False
            def run(*inputs, **options):
                nonlocal replay
                if replay:
                    raise RuntimeError('fixture replay failure')
                replay = True
                return function(*inputs, **options)
            return native(run, *a, **kw)
        monkeypatch.setattr(cp, 'checkpoint', checkpoint)
    if failure:
        original_score = s.log_probability
        def fail(*a, **kw):
            result = original_score(*a, **kw)
            if failure == 'forward':
                raise RuntimeError('fixture forward failure')
            def raise_backward(grad):
                raise RuntimeError('fixture backward failure')
            if failure == 'backward':
                result.register_hook(raise_backward)
            return result
        monkeypatch.setattr(s, 'log_probability', fail)
        with pytest.raises(RuntimeError, match='fixture'):
            invoke(s, tool, tmp_path)
    else:
        invoke(s, tool, tmp_path)
    assert_restored(s, before)
    assert torch.equal(s._forward([130, 131, 132]).logits, logits)
    handle.remove()


@pytest.mark.parametrize('tool', TOOLS)
def test_checkpoint_recomputation_eval_mode_and_scored_logits(tmp_path, monkeypatch, tool):
    from torch.utils import checkpoint as cp
    s, _ = students(torch.bfloat16, per_layer=4)
    calls, forwards = [], []
    native = cp.checkpoint
    def checkpoint(function, *args, **kwargs):
        assert kwargs['use_reentrant'] is False
        assert all(not m.training for m in s.model.modules())
        calls.append(1)
        def observed(*a, **kw):
            forwards.append(1)
            return function(*a, **kw)
        return native(observed, *args, **kwargs)
    monkeypatch.setattr(cp, 'checkpoint', checkpoint)
    original = s.model.forward
    def forward(*a, **kwargs):
        assert kwargs['use_cache'] is False
        positions = kwargs['logits_to_keep']
        assert isinstance(positions, torch.Tensor) and positions.ndim == 1
        assert positions.numel() < kwargs['input_ids'].shape[1]
        return original(*a, **kwargs)
    monkeypatch.setattr(s.model, 'forward', forward)
    # Nonzero configured dropout would change training-mode results. It stays off.
    for layer in s.layers:
        layer.self_attn.attention_dropout = 0.6
    invoke(s, tool, tmp_path)
    assert calls and len(forwards) > len(calls)


def test_gradient_streaming_complete_pair_and_no_retained_graph(monkeypatch):
    s, old = students(torch.bfloat16, per_layer=4)
    expected, observed, live, activations = [], [], [], []
    native_grad = torch.autograd.grad
    def grad(*args, **kwargs):
        grads = native_grad(*args, **kwargs)
        expected.append([None if g is None else g.detach().clone() for g in grads])
        return grads
    monkeypatch.setattr(torch.autograd, 'grad', grad)
    old.gradient_statistics(EXAMPLES, margin=True)
    monkeypatch.setattr(torch.autograd, 'grad', lambda *a, **k: pytest.fail('No bulk autograd.grad'))
    modules = all_modules(s)
    handles, by_action = [], {}
    for key, m in modules.items():
        def observe(weight, key=key):
            # Only this leaf owns a gradient; all preceding leaves were released.
            assert sum(m.weight.grad is not None for m in modules.values()) == 1
            assert key not in by_action
            by_action[key] = weight.grad.detach().clone()
        m.weight.requires_grad_(True)
        handles.append(m.weight.register_post_accumulate_grad_hook(observe))
        m.weight.requires_grad_(False)
    original_score = s.log_probability
    for layer in s.layers:
        handles.append(layer.register_forward_hook(lambda m, a, output: activations.append(weakref.ref(output))))
    def score(*a, **kw):
        assert all(ref() is None for ref in live)
        assert all(ref() is None for ref in activations)
        activations.clear()
        if by_action:
            observed.append([by_action[k] for k in modules])
            by_action.clear()
        result = original_score(*a, **kw)
        live.append(weakref.ref(result))
        return result
    monkeypatch.setattr(s, 'log_probability', score)
    s.gradient_statistics(EXAMPLES, margin=True)
    observed.append([by_action[k] for k in modules])
    assert len(observed) == 2 * len(expected) == 2 * len(EXAMPLES)
    for plus, minus, pair in zip(observed[::2], observed[1::2], expected):
        assert all(torch.equal(x + y, z) for x, y, z in zip(plus, minus, pair))
    assert all(ref() is None for ref in live)
    assert all(ref() is None for ref in activations)
    for handle in handles:
        handle.remove()


def test_unused_weight_has_original_zero_statistics():
    s, old = students(torch.bfloat16)
    # Artificial dormant projection exercises the old allow_unused=True case.
    module = torch.nn.Linear(16, 8, bias=False).to(torch.bfloat16).requires_grad_(False)
    for model in (s, old):
        model.layers[2].self_attn.k_proj = deepcopy(module)
    result = s.gradient_statistics(EXAMPLES, margin=True)
    assert result == old.gradient_statistics(EXAMPLES, margin=True)
    row = next(r for r in result if r['tensor'] == '2:self_attn.k_proj')
    assert row['gradient_norm'] == row['noise_norm'] == row['snr'] == 0


@pytest.mark.parametrize('list_result', [False, True])
def test_peak_receipts_use_selected_device_and_reset_each_call(monkeypatch, list_result):
    calls = []
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda d: calls.append(('reset', d)))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda d: calls.append(('peak', d)) or 123456)
    @peak_memory
    def measured(s):
        calls.append(('work', s.device))
        return [dict(norm=1)] if list_result else dict(norm=1)
    for name in ['cuda:1', 'cuda:1', 'cpu']:
        s = SimpleNamespace(device=torch.device(name))
        result = measured(s)
        row = result[0] if list_result else result
        assert row == dict(norm=1, **({PEAK_MEMORY: 123456} if name != 'cpu' else {}))
    cuda = torch.device('cuda:1')
    assert calls == [('reset', cuda), ('work', cuda), ('peak', cuda)] * 2 + [('work', torch.device('cpu'))]


def test_gradient_attribution_propagates_statistics_peak(monkeypatch):
    s, _ = students()
    row = dict(tensor='0:mlp.down_proj', gradient_norm=1, **{PEAK_MEMORY: 123456})
    monkeypatch.setattr(s, 'gradient_statistics', lambda *a, **kw: [row])
    result = s.gradient_attribution(EXAMPLES)
    assert result[PEAK_MEMORY] == 123456
    assert PEAK_MEMORY not in result['ranking'][0]


def test_forward_only_tools_unchanged_after_backward(tmp_path, monkeypatch):
    from torch.utils import checkpoint as cp
    s, old = students(torch.bfloat16, per_layer=4)
    for tool in TOOLS:
        invoke(s, tool, tmp_path / tool)
    def forbidden(*a, **k):
        pytest.fail('Forward-only tool called checkpoint')
    monkeypatch.setattr(cp, 'checkpoint', forbidden)
    state = EXAMPLES[0][0]
    for model in (s, old):
        model.usage = dict.fromkeys(model.usage, 0)
    for tool, args, kwargs in [
        ('action_scores', (state, ['look', 'inventory']), {}),
        ('greedy', (state,), {}),
        ('logit_lens', (state, 'look', 'inventory'), {}),
        ('attention_summary', (state, [0, 1, 2, 3]), {}),
        ('patch', (state, state, [0, 2], [dict(target=-1, source=-1)], ['look']), {}),
    ]:
        assert getattr(s, tool)(*args, **kwargs) == getattr(old, tool)(*args, **kwargs)
    assert s.usage == old.usage


def test_checkpoint_scope_bypasses_inference_and_restores_nested_context(monkeypatch):
    from torch.utils import checkpoint as cp
    s, _ = students(torch.bfloat16)
    before = snapshot(s)
    expected = s._forward([130, 131]).logits.clone()
    monkeypatch.setattr(cp, 'checkpoint', lambda *a, **kw: pytest.fail('inference checkpoint'))
    with checkpoint_layers(s.layers):
        outer = [m.forward for m in s.layers]
        with checkpoint_layers(s.layers):
            assert torch.equal(s._forward([130, 131]).logits, expected)
        assert [m.forward for m in s.layers] == outer
    assert_restored(s, before)


def test_automatic_finetune_old_vs_new(tmp_path, monkeypatch):
    import surgery.conditions as conditions
    from tests.p0_fakes import fake_pair
    s, old = students(torch.bfloat16, per_layer=4)
    monkeypatch.setattr(conditions, 'LRS', (0.003,))
    monkeypatch.setattr(conditions, 'STEPS', (2,))
    monkeypatch.setattr(conditions, 'render_state', lambda *a: EXAMPLES[0][0])
    rollout = lambda *a: dict(success=True)
    edits, selection = conditions.automatic_finetune(s, [fake_pair()], rollout, tmp_path / 'new')
    legacy_edits, legacy_selection = conditions.automatic_finetune(old, [fake_pair()], rollout, tmp_path / 'old')
    assert selection['ranking'] == legacy_selection['ranking']
    assert selection['chosen'] == legacy_selection['chosen']
    assert edits[0]['delta']['metadata'] == legacy_edits[0]['delta']['metadata']
    assert edits[0]['delta']['sha256'] == legacy_edits[0]['delta']['sha256']


def test_rollout_evaluation_export_unchanged_after_backward(tmp_path, monkeypatch):
    from torch.utils import checkpoint as cp
    from transformers import Gemma4ForCausalLM
    from surgery.session import Session
    from surgery.conditions import freeze
    from surgery.evaluation import Runtime
    from surgery.export import export_model
    from surgery.common import read
    from tests.p0_fakes import fake_pair, fake_greedy
    from repair.dry_fakes import FakeEnvironment
    from sed.storage import file_hash
    s, old = students(torch.bfloat16, per_layer=4)
    for tool in TOOLS:
        invoke(s, tool, tmp_path / tool)
    monkeypatch.setattr(cp, 'checkpoint', lambda *a, **k: pytest.fail('Forward-only checkpoint'))
    # Compare native predictions during a short, scripted fake-environment rollout.
    # The random tiny model does not itself produce valid environment commands.
    pair = fake_pair()
    outputs, exports = [], []
    for name, model in [('new', s), ('old', old)]:
        model.usage = dict.fromkeys(model.usage, 0)
        directory = tmp_path / name
        native_greedy, predictions = model.greedy, []
        def policy(state):
            predictions.append(native_greedy(state))
            return fake_greedy(state)
        model.greedy = policy
        session = Session(model, [pair], directory / 'session', env_factory=lambda p: FakeEnvironment())
        result = session.rollout(session.state(pair['pair_id']), [])
        model.greedy = native_greedy
        runtime = object.__new__(Runtime)
        runtime.student = model
        runtime.collateral = {pair['pair_id']: pair}
        evaluation = runtime.evaluate(dict(metric='M3', key=pair['pair_id']), [])
        outputs.append((result, evaluation, predictions))
        condition = freeze(directory / 'condition.json', model, 'pairs', 'A', 'E0', [])
        snapshot_dir = directory / 'snapshot'
        model.model.config.save_pretrained(snapshot_dir)
        report = export_model(model, condition, snapshot_dir, directory / 'export', inputs=[[130, 131]],
            loader=lambda p: Gemma4ForCausalLM.from_pretrained(p, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa'))
        assert report['passed']
        exports.append((file_hash(directory / 'export/model.safetensors'),
                        read(directory / 'export/consistency.json')['checks']))
    assert outputs[0] == outputs[1]
    assert exports[0] == exports[1]
