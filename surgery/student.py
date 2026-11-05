"""P0 student tools plus differentiable action scoring and bounded LoRA surgery."""
from contextlib import contextmanager
from copy import deepcopy
import math

from repair.p0_introspect import StudentTools, tensor_hash
from sed.storage import digest
from surgery.common import Relocation
from surgery.edits import all_modules, module_at, tensor_name, save_delta, edited, finite


class Student(StudentTools):
    def __init__(self, *args, relocation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.relocation = relocation or Relocation()

    def edited(self, spec):
        return edited(self, spec)

    def log_probability(self, state, action, *, framing='original_thought'):
        import torch
        state = self.decision_prefix(state, framing)
        prefix = self.ids(state)
        suffix, _ = self.action_tokens(action)
        if len(prefix) + len(suffix) > self.max_model_len:
            raise ValueError('Context overflow; no truncation')
        positions = torch.arange(len(prefix) - 1, len(prefix) + len(suffix) - 1, device=self.device)
        self.usage['forward_passes'] += 1
        self.usage['forward_input_tokens'] += len(prefix) + len(suffix)
        logits = self.model(input_ids=self._tensor(prefix + suffix), use_cache=False, logits_to_keep=positions).logits[0].float()
        return logits.log_softmax(-1).gather(-1, self._tensor(suffix)[0, :, None]).sum()

    @contextmanager
    def differentiable(self, modules):
        originals = [(m.weight, m.weight.requires_grad, m.weight.grad) for m in modules.values()]
        try:
            for w, _, _ in originals:
                w.requires_grad_(True)
                w.grad = None
            yield
        finally:
            for w, flag, grad in originals:
                w.requires_grad_(flag)
                w.grad = grad

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
