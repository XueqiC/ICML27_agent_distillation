"""White-box student tools; no teacher internals, training, or remote models.

Layer indices are zero based, residuals are decoder BLOCK OUTPUTS before final
norm. Action scores include `Action: ` and turn-end 106. Direction means include
only command tokens, excluding that wrapper and the terminator.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from repair.p0_pairs import REVISION, MODEL, render_state, thought_split, state_digest, FRAMINGS
from repair.records import TrajectoryRecord
from sed.react import action_outcome, build_messages
from sed.storage import digest, file_hash, atomic_write, canonical_bytes, JsonlLedger
from sed.thinking import split_thinking


def write_receipt(path, value):
    """Publish immutable content; identical rollouts may share one receipt."""
    content = canonical_bytes(value) + b'\n'
    try:
        atomic_write(path, content, exclusive=True)
    except FileExistsError:
        if Path(path).read_bytes() != content:
            raise ValueError('Content-addressed receipt integrity error: ' + str(path)) from None


def tensor_hash(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def load_theta0(base, adapter, expected):
    """Local HF load + PEFT in-memory merge; never save edited checkpoints."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    from peft import PeftModel
    from train.dataset import load_tokenizer
    base, adapter = Path(base), Path(adapter)
    if base.name != REVISION or expected['base_model'] != MODEL or expected['revision'] != REVISION:
        raise ValueError('Pinned Gemma-4 E4B snapshot required')
    for name, key in [('adapter_model.safetensors', 'adapter_weights_sha256'),
                      ('adapter_config.json', 'adapter_config_sha256')]:
        if file_hash(adapter / name) != expected[key]:
            raise ValueError('theta0 adapter bytes differ from round')
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if config.get('base_model_name_or_path') not in (MODEL, str(base), str(base.resolve())):
        # Local historical snapshot paths are permissible only with same revision.
        if Path(config.get('base_model_name_or_path', '')).name != REVISION:
            raise ValueError('Adapter base identity mismatch')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Real model requires exactly the explicitly selected GPU')
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = Gemma4ForConditionalGeneration.from_pretrained(base, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload(safe_merge=True)
    model.eval().requires_grad_(False)
    tokenizer = load_tokenizer(base)
    identity = dict(base=MODEL, revision=REVISION, adapter=expected,
        base_files={p.name: file_hash(p) for p in sorted(base.iterdir())
                    if p.is_file() and (p.suffix in ('.json', '.jinja', '.safetensors'))},
        dtype='bfloat16', attention='sdpa', merge='PEFT in memory', greedy=True,
        dependency_versions={n: __import__('importlib.metadata', fromlist=['version']).version(n)
                             for n in ('torch', 'transformers', 'peft')})
    return model, tokenizer, identity


class StudentTools:
    def __init__(self, model, tokenizer, *, identity, max_new_tokens=1024, max_model_len=32768):
        import torch
        self.model, self.tokenizer, self.identity = model.eval(), tokenizer, identity
        self.model.requires_grad_(False)
        self.text_model = getattr(model.model, 'language_model', model.model)
        self.layers = self.text_model.layers
        self.norm = self.text_model.norm
        self.head = model.get_output_embeddings()
        self.config = model.config.get_text_config()
        self.max_new_tokens, self.max_model_len = max_new_tokens, max_model_len
        self.device = next(model.parameters()).device
        if next(model.parameters()).dtype != torch.bfloat16:
            raise ValueError('P0 requires fixed bf16 student')
        self.directions = {}
        self.thought_cache = {}
        self.active_edit = None
        self.usage = dict(forward_passes=0, forward_input_tokens=0, greedy_calls=0,
                          generation_prompt_tokens=0, generated_tokens=0)

    def layer_ids(self, layers):
        if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
                or any(type(i) is not int or not 0 <= i < len(self.layers) for i in layers)):
            raise ValueError('Invalid zero-based layer indices')
        return layers

    def ids(self, state):
        ids = self.tokenizer.encode(state['text'], add_special_tokens=False)
        if not ids or len(ids) >= self.max_model_len:
            raise ValueError('Context overflow; no truncation')
        return ids

    def _tensor(self, ids):
        import torch
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _forward(self, ids, keep=1):
        import torch
        if len(ids) > self.max_model_len:
            raise ValueError('Context overflow; no truncation')
        self.usage['forward_passes'] += 1
        self.usage['forward_input_tokens'] += len(ids)
        with torch.inference_mode():
            return self.model(input_ids=self._tensor(ids), use_cache=False, logits_to_keep=keep)

    def action_tokens(self, action):
        if type(action) is not str or not action.strip() or '\n' in action:
            raise ValueError('Candidate must be one command string')
        text = 'Action: ' + action
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded['input_ids'])
        if any(i in self.tokenizer.all_special_ids for i in ids):
            raise ValueError('Action contains model control tokens')
        positions = [i for i, (s, e) in enumerate(encoded['offset_mapping']) if e > len('Action: ')]
        if not positions:
            raise ValueError('No command tokens')
        return ids + [106], positions

    def greedy(self, state):
        import torch
        from transformers import GenerationConfig
        if state.get('_prepared'):
            state = dict(state, text=state['text'][:state['base_text_length']])
        ids = self.ids(state)
        if len(ids) + self.max_new_tokens > self.max_model_len:
            from students.vllm_student import ContextOverflow
            raise ContextOverflow(dict(rendered_prompt_tokens=len(ids), max_model_len=self.max_model_len))
        # Same native stops as the pinned generation_config; turn end is 106.
        cfg = GenerationConfig(do_sample=False, max_new_tokens=self.max_new_tokens,
                               eos_token_id=[1, 106, 50], pad_token_id=0, use_cache=True)
        self.usage['greedy_calls'] += 1
        self.usage['generation_prompt_tokens'] += len(ids)
        with torch.inference_mode():
            out = self.model.generate(self._tensor(ids), generation_config=cfg)
        self.usage['generated_tokens'] += len(out[0]) - len(ids)
        generated = out[0, len(ids):].tolist()
        if generated and generated[-1] in cfg.eos_token_id:
            generated = generated[:-1]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        reply = split_thinking(raw)[0]
        action, _ = action_outcome(reply, [])
        return dict(action=action, reply=reply, framing='free_greedy', generated_tokens=len(out[0]) - len(ids), prompt_tokens=len(ids))

    def decision_prefix(self, state, framing='original_thought'):
        if framing not in FRAMINGS:
            raise ValueError('Unknown decision framing')
        if state.get('_prepared'):
            if state['framing'] != framing:
                raise ValueError('Prepared prefix framing differs')
            return state
        override = state.get('thought_override')
        if framing == 'no_thought':
            record = dict(prefix='', fallback_reason=None)
        elif override and override['framing'] == framing:
            record = deepcopy(override['record'])
        elif framing == 'original_thought':
            record = deepcopy(state.get('original_thought', thought_split(None)))
        else:
            key = digest(dict(state=state_digest(state), edit=self.active_edit,
                              student=self.identity, framing=framing,
                              max_new_tokens=self.max_new_tokens))
            if key not in self.thought_cache:
                generated = self.greedy(state)
                self.thought_cache[key] = dict(thought_split(generated['reply']),
                                               generation=generated, cache_key=key)
            record = deepcopy(self.thought_cache[key])
        prefix = record['prefix']
        record.update(requested=framing, effective=framing if prefix else 'no_thought',
                      active_edit=deepcopy(self.active_edit))
        text = state['text'] + prefix
        spans = deepcopy(state.get('spans', []))
        if prefix:
            spans.append(dict(id='current_thought', label='current_thought',
                              start=len(state['text']), end=len(text)))
        return dict(state, text=text, spans=spans, framing=framing, framing_record=record,
                    base_text_length=len(state['text']), _prepared=True,
                    canonical_state_id=state.get('sha256', state_digest(state)),
                    sha256=digest(dict(state=state_digest(state), framing=framing, record=record)))

    def action_scores(self, state, candidates, *, include_greedy=True, framing='original_thought'):
        base_state = state
        state = self.decision_prefix(state, framing)
        import torch
        if not candidates or len(candidates) > 32 or len(set(candidates)) != len(candidates):
            raise ValueError('Use 1..32 distinct candidates')
        prefix, scores = self.ids(state), []
        for action in candidates:
            suffix, _ = self.action_tokens(action)
            # Only materialize logits at the positions predicting the suffix.
            positions = torch.arange(len(prefix) - 1, len(prefix) + len(suffix) - 1, device=self.device)
            logits = self._forward(prefix + suffix, keep=positions).logits[0].float()
            logps = logits.log_softmax(-1).gather(-1, torch.tensor(suffix, device=self.device)[:, None]).squeeze(-1)
            scores.append(dict(action=action, log_probability=float(logps.sum()), tokens=len(suffix)))
        return dict(scores=scores, greedy=self.greedy(base_state) if include_greedy else None,
                    framing=framing, framing_record=state['framing_record'],
                    scoring='log P(Action: command + <turn|> | rendered state + selected Thought prefix)')

    @contextmanager
    def capture(self, layers, positions):
        values, hooks = {}, []
        try:
            for layer in self.layer_ids(layers):
                def hook(module, args, output, layer=layer):
                    hidden = output[0] if isinstance(output, tuple) else output
                    values[layer] = hidden[:, positions, :].detach().clone()
                hooks.append(self.layers[layer].register_forward_hook(hook))
            yield values
        finally:
            for h in hooks:
                h.remove()

    def logit_lens(self, state, a_f, a_r, *, framing='original_thought'):
        state = self.decision_prefix(state, framing)
        import torch
        left, _ = self.action_tokens(a_f)
        right, _ = self.action_tokens(a_r)
        first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        if first is None:
            return dict(first_difference=None, layers=[], status='identical_actions',
                        framing=framing, framing_record=state['framing_record'])
        ids = self.ids(state) + left[:first]
        with self.capture(list(range(len(self.layers))), [-1]) as hidden:
            self._forward(ids)
        rows = []
        with torch.inference_mode():
            for i, h in hidden.items():
                logits = self.head(self.norm(h)).float()
                cap = getattr(self.config, 'final_logit_softcapping', None)
                if cap:
                    logits = cap * torch.tanh(logits / cap)
                lp = logits.log_softmax(-1)[0, 0]
                rows.append(dict(layer=i, logp_repair=float(lp[right[first]]), logp_fail=float(lp[left[first]]),
                                 difference=float(lp[right[first]] - lp[left[first]])))
        return dict(first_difference=first, fail_token=left[first], repair_token=right[first],
                    common_prefix_tokens=left[:first], layers=rows,
                    framing=framing, framing_record=state['framing_record'],
                    definition='block output -> final norm -> unembedding -> native softcap')

    def attention_summary(self, state, layers, *, framing='original_thought'):
        state = self.decision_prefix(state, framing)
        import torch
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, repeat_kv
        ids = self.ids(state)
        encoded = self.tokenizer(state['text'], add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded['input_ids']) != ids:
            raise ValueError('Span tokenizer differs')
        span_tokens = {s['id']: [i for i, (a, b) in enumerate(encoded['offset_mapping'])
            if b > s['start'] and a < s['end'] and b > a] for s in state['spans']}
        rows, hooks, queries, keys = [], [], {}, {}
        try:
            for layer in self.layer_ids(layers):
                attention = self.layers[layer].self_attn
                def capture_q(module, args, output, layer=layer):
                    queries[layer] = output[:, -1:].detach()
                hooks.append(attention.q_norm.register_forward_hook(capture_q))
                if not attention.is_kv_shared_layer:
                    def capture_k(module, args, output, layer=layer):
                        keys[layer] = output.detach()
                    hooks.append(attention.k_norm.register_forward_hook(capture_k))
                def summarize(module, args, kwargs, output, layer=layer):
                    # Capture native normalized projections. Reconstruct only
                    # ONE rotated Q and its probabilities, never an L x L map.
                    # Shared layers read actual donor K, not nonexistent k_proj.
                    cos, sin = kwargs['position_embeddings']
                    q = queries.pop(layer)
                    q = apply_rotary_pos_emb(q, cos[:, -1:], sin[:, -1:], unsqueeze_dim=2).transpose(1, 2)
                    if module.is_kv_shared_layer:
                        k = kwargs['shared_kv_states'][module.layer_type][0].to(q.device)
                    else:
                        k = keys.pop(layer)
                        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                    k = repeat_kv(k, module.num_key_value_groups)
                    scores = (q @ k.transpose(-2, -1)) * module.scaling
                    mask = kwargs.get('attention_mask')
                    if mask is not None:
                        row_mask = mask[..., -1:, :k.shape[-2]]
                        scores = scores.masked_fill(~row_mask, float('-inf')) if mask.dtype == torch.bool else scores + row_mask
                    elif module.sliding_window:
                        key_positions = torch.arange(k.shape[-2], device=q.device)
                        scores = scores.masked_fill(key_positions < len(ids) - module.sliding_window, float('-inf'))
                    probs = scores.softmax(-1, dtype=torch.float32)[0, :, 0]
                    spans = []
                    for span in state['spans']:
                        mass = probs[:, span_tokens[span['id']]].sum(-1)
                        spans.append(dict(id=span['id'], label=span['label'], tokens=len(span_tokens[span['id']]),
                            mean=float(mass.mean()), max=float(mass.max())))
                    rows.append(dict(layer=layer, kv_shared=module.is_kv_shared_layer,
                        kv_type=module.layer_type, spans=spans, total_mass=float(probs.mean(0).sum())))
                hooks.append(attention.register_forward_hook(summarize, with_kwargs=True))
            self._forward(ids)
        finally:
            for hook in hooks:
                hook.remove()
        return dict(query_position=len(ids) - 1, layers=rows,
                    framing=framing, framing_record=state['framing_record'],
                    note='Single query, fp32 probabilities; overlapping goal/observation spans are not additive.')

    def patch(self, target_state, source_state, layers, positions, candidates, *, framing='original_thought'):
        """positions = [{target: integer, source: integer}], prompt positions only."""
        target_state = self.decision_prefix(target_state, framing)
        source_state = self.decision_prefix(source_state, framing)
        target, source = self.ids(target_state), self.ids(source_state)
        if not positions:
            raise ValueError('Empty residual patch')
        tp, sp = [], []
        for item in positions:
            if set(item) != {'target', 'source'} or any(type(v) is not int for v in item.values()):
                raise ValueError('Patch positions need integer source and target')
            a, b = item['target'], item['source']
            a = a + len(target) if a < 0 else a
            b = b + len(source) if b < 0 else b
            if not 0 <= a < len(target) or not 0 <= b < len(source) or a in tp:
                raise ValueError('Patch prompt position out of range/duplicate')
            tp.append(a); sp.append(b)
        before = self.action_scores(target_state, candidates, include_greedy=False, framing=framing)
        with self.capture(layers, sp) as values:
            self._forward(source)
        hooks = []
        try:
            for layer in layers:
                def hook(module, args, output, layer=layer):
                    h = output[0] if isinstance(output, tuple) else output
                    h = h.clone()
                    h[:, tp] = values[layer].to(h)
                    return (h, *output[1:]) if isinstance(output, tuple) else h
                hooks.append(self.layers[layer].register_forward_hook(hook))
            after = self.action_scores(target_state, candidates, include_greedy=False, framing=framing)
        finally:
            for hook in hooks:
                hook.remove()
        return dict(before=before, after=after, framing=framing,
            source_framing_record=source_state['framing_record'],
            target_framing_record=target_state['framing_record'], changes=[dict(action=a['action'],
            delta_log_probability=b['log_probability'] - a['log_probability'])
            for a, b in zip(before['scores'], after['scores'])])

    def hidden_mean(self, state, action, layers, *, framing='original_thought'):
        state = self.decision_prefix(state, framing)
        prefix = self.ids(state)
        suffix, positions = self.action_tokens(action)
        with self.capture(layers, [len(prefix) + i for i in positions]) as h:
            self._forward(prefix + suffix)
        return dict(layers={layer: values.float().mean(dim=(0, 1)).cpu() for layer, values in h.items()},
                    framing=framing, framing_record=state['framing_record'])

    def direction(self, positive, negative, layers, *, framing='original_thought'):
        import torch
        self.layer_ids(layers)
        if not positive or not negative or len(positive) + len(negative) > 64:
            raise ValueError('Direction needs two nonempty sets, at most 64 items')
        means, framing_records = [], []
        for group in (positive, negative):
            hs = [self.hidden_mean(state, action, layers, framing=framing) for state, action in group]
            means.append({i: torch.stack([h['layers'][i] for h in hs]).mean(0) for i in layers})
            framing_records.append([h['framing_record'] for h in hs])
        raw = {i: means[0][i] - means[1][i] for i in layers}
        if any(not torch.isfinite(d).all() or d.norm() < 1e-12 for d in raw.values()):
            raise ValueError('Zero/nonfinite direction; cannot deploy')
        vectors = {i: d / d.norm() for i, d in raw.items()}
        ident = digest({str(i): tensor_hash(d) for i, d in vectors.items()})
        self.directions[ident] = vectors
        return dict(direction_id=ident, layers=layers, norms={str(i): float(d.norm()) for i, d in raw.items()},
                    normalized=True, framing=framing, framing_records=dict(positive=framing_records[0], negative=framing_records[1]),
                    definition='mean(positive action-token residuals) - mean(negative)')

    def projection(self, state, action, direction_id, *, framing='original_thought'):
        vectors = self.directions[direction_id]
        hidden = self.hidden_mean(state, action, list(vectors), framing=framing)
        hs = hidden['layers']
        return dict(direction_id=direction_id, framing=framing, framing_record=hidden['framing_record'], layers=[dict(layer=i,
            projection=float(hs[i] @ d), cosine=float((hs[i] @ d) / hs[i].norm().clamp_min(1e-12)))
            for i, d in vectors.items()])

    @contextmanager
    def edited(self, spec):
        import torch
        if spec is None:
            yield dict(edited_parameters=0, edited_fraction=0.0)
            return
        if self.active_edit is not None:
            raise ValueError('Nested weight edits forbidden')
        if set(spec) != {'direction_id', 'layers', 'targets', 'alpha'}:
            raise ValueError('Only output-side rank-one edit family is allowed')
        alpha = spec['alpha']
        if type(alpha) not in (int, float) or not math.isfinite(alpha):
            raise ValueError('Finite alpha required')
        layers = self.layer_ids(spec['layers'])
        targets = spec['targets']
        if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
                or not set(targets) <= {'mlp.down_proj', 'self_attn.o_proj'}):
            raise ValueError('Only down_proj and o_proj may be edited')
        vectors = self.directions[spec['direction_id']]
        if not set(layers) <= vectors.keys():
            raise ValueError('Missing layer direction')
        norm_names = {'self_attn.o_proj': 'post_attention_layernorm',
                      'mlp.down_proj': 'post_feedforward_layernorm'}
        # Validate every selected layer before any mutation, even for alpha=0.
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
        for layer in layers:
            block = self.layers[layer]
            if (getattr(block, 'enable_moe_block', False) or any(hasattr(block, n) for n in
                    ('post_feedforward_layernorm_1', 'post_feedforward_layernorm_2'))):
                raise ValueError('MoE post-feedforward norm paths are unsupported')
            for name in norm_names.values():
                norm = getattr(block, name, None)
                if not isinstance(norm, Gemma4RMSNorm) or not hasattr(norm, 'weight'):
                    raise ValueError('Expected native Gemma4 post-norm with weight: ' + name)
        originals, changed, count, projections = [], 0, 0, []
        self.active_edit = deepcopy(spec)
        try:
            with torch.no_grad():
                for layer in layers:
                    for target in targets:
                        parent, name = target.split('.')
                        w = getattr(getattr(self.layers[layer], parent), name).weight
                        d = vectors[layer].to(device=w.device, dtype=torch.float32)
                        if d.shape != (w.shape[0],) or not torch.isfinite(d).all() or abs(float(d.norm()) - 1) > 1e-4:
                            raise ValueError('Edit direction must be unit output-side vector')
                        norm_name = norm_names[target]
                        g = getattr(self.layers[layer], norm_name).weight.detach().to(device=w.device, dtype=torch.float32)
                        if g.shape != d.shape or not torch.isfinite(g).all():
                            raise ValueError('Invalid post-norm weight')
                        gd = g * d
                        length = gd.norm()
                        if not torch.isfinite(length) or length < 1e-12:
                            raise ValueError('Zero/nonfinite norm-scaled direction')
                        u = gd / length
                        projections.append(dict(layer=layer, target=target,
                            post_norm_module=f'layers.{layer}.{norm_name}',
                            scaled_direction_norm=float(length), cosine_u_d=float(u @ d)))
                        # Exact bf16 rollback (not algebraic inverse). CPU backup
                        # keeps 22-layer editing within a single GPU's memory.
                        original = w.detach().cpu().clone()
                        originals.append((w, original))
                        wf = w.float()
                        if alpha != 0:  # Preserve every original byte, including signed zeros.
                            w.copy_((wf - float(alpha) * u[:, None] * (u @ wf)[None, :]).to(w.dtype))
                        count += w.numel()
                        changed += int((w != original.to(w.device)).sum())
            total = sum(p.numel() for p in self.model.parameters())
            yield dict(edited_parameters=count, edited_fraction=count / total,
                       changed_parameters=changed, total_parameters=total, projections=projections,
                       formula='W - alpha*u*u.T*W, u=normalize(g*d)')
        finally:
            with torch.no_grad():
                for w, original in originals:
                    w.copy_(original.to(w.device))
            self.active_edit = None

    def apply_edit(self, state, candidates, edit, *, framing='original_thought'):
        before = self.action_scores(state, candidates, framing=framing)
        with self.edited(edit) as receipt:
            after = self.action_scores(state, candidates, framing=framing)
        return dict(before=before, after=after, edit=edit, receipt=receipt, reverted=True, framing=framing)


def counterfactual(state, edit):
    """Text-only intervention. It never claims the environment was restored."""
    spans = {s['id']: s for s in state['spans']}
    op = edit.get('op')
    if op == 'replace':
        s = spans[edit['span']]
        if type(edit.get('text')) is not str:
            raise ValueError('Replace needs literal text')
        nested = [dict(s, start=0, end=len(edit['text']))] if edit['text'] else []
        changes = [(s['start'], s['end'], edit['text'], nested)]
    elif op == 'delete':
        s = spans[edit['span']]
        changes = [(s['start'], s['end'], '', [])]
    elif op == 'insert':
        s = spans[edit['span']]
        if edit.get('side') not in ('before', 'after') or type(edit.get('text')) is not str:
            raise ValueError('Insert needs side before/after and literal text')
        at = s['start'] if edit['side'] == 'before' else s['end']
        new = dict(id='insert_' + digest(edit)[:12], label='teacher_insert', start=0, end=len(edit['text']))
        changes = [(at, at, edit['text'], [new])]
    elif op == 'reorder':
        names = edit['spans']
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError('Reorder needs distinct span IDs in desired order')
        destinations = sorted([spans[n] for n in names], key=lambda s: s['start'])
        if any(a['end'] > b['start'] for a, b in zip(destinations, destinations[1:])):
            raise ValueError('Reorder spans cannot overlap')
        changes = []
        for dst, name in zip(destinations, names):
            src = spans[name]
            nested = [dict(s, start=s['start'] - src['start'], end=s['end'] - src['start'])
                      for s in state['spans'] if src['start'] <= s['start'] and s['end'] <= src['end']]
            changes.append((dst['start'], dst['end'], state['text'][src['start']:src['end']], nested))
    else:
        raise ValueError('Unknown context edit')
    boundary = state.get('base_text_length', len(state['text']))
    if state.get('_prepared'):
        # Keep the free-generation state separate from the current reply. Moving
        # text across that boundary would change the meaning of free rollouts.
        selected = [spans[n] for n in edit.get('spans', [edit.get('span')])]
        regions = {s['start'] >= boundary for s in selected}
        if len(regions) > 1 or any(a < boundary < b for a, b, _, _ in changes):
            raise ValueError('Cannot reorder across state/current_thought boundary')
        thought_edit = regions == {True}
        if not thought_edit:
            boundary += sum(len(c) - (b - a) for a, b, c, _ in changes)
    changes.sort()
    result, cursor, new_spans = '', 0, []
    for start, end, content, nested in changes:
        result += state['text'][cursor:start]
        base = len(result)
        new_spans.extend(dict(s, start=base + s['start'], end=base + s['end']) for s in nested)
        result += content
        cursor = end
    result += state['text'][cursor:]
    for s in state['spans']:
        # Partial intersections have no exact surviving semantic span.
        if any(a < s['end'] and b > s['start'] for a, b, _, _ in changes):
            continue
        shift = sum(len(c) - (b - a) for a, b, c, _ in changes if b <= s['start'])
        new_spans.append(dict(s, start=s['start'] + shift, end=s['end'] + shift))
    value = dict(text=result, spans=sorted(new_spans, key=lambda s: (s['start'], s['id'])),
        pair_id=state['pair_id'], origin_pair_id=state['origin_pair_id'], counterfactual=True,
        parent_state_sha256=state['sha256'], context_edit=edit)
    if 'original_thought' in state:
        value['original_thought'] = deepcopy(state['original_thought'])
    if state.get('_prepared'):
        value['text'] = result[:boundary]
        value['spans'] = [s for s in value['spans'] if s['end'] <= boundary and s['label'] != 'current_thought']
        if thought_edit:
            record = dict(state['framing_record'], prefix=result[boundary:],
                          counterfactual_edit=deepcopy(edit), source='teacher_counterfactual')
            value['thought_override'] = dict(framing=state['framing'], record=record)
        elif 'thought_override' in state:
            value['thought_override'] = deepcopy(state['thought_override'])
    value['sha256'] = state_digest(value)
    return value



class ToolSession:
    """Capability boundary: IDs only, no paths, Python, or arbitrary state rows.

    A teacher session is constructed with diagnosis pairs ONLY. Evaluation uses
    a separate instance, and cannot be passed to P0Agent.
    """
    NAMES = ('action_scores', 'logit_lens', 'attention_summary', 'counterfactual_context',
             'patch', 'direction', 'projection', 'apply_edit', 'rollout')

    def __init__(self, student, pairs, directory, *, env_factory=None, scope='diagnosis'):
        if scope not in ('diagnosis', 'evaluation'):
            raise ValueError('Unknown tool scope')
        if scope == 'diagnosis' and any(p.get('split') != 'diagnosis' for p in pairs):
            raise ValueError('Teacher tools accept diagnosis pairs only')
        self.student, self.scope = student, scope
        self.pairs = {p['pair_id']: deepcopy(p) for p in pairs}
        self.states = {}
        self.pair_states = {}
        for p in pairs:
            state = render_state(p, student.tokenizer)
            self.states[state['sha256']] = state
            self.pair_states[p['pair_id']] = state['sha256']
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.env_factory = env_factory
        self.ledger = JsonlLedger(self.directory / 'tools.jsonl')
        self.cache = self.directory / 'cache'
        self.direction_specs = {}
        self.environment_steps = 0
        from repair.p0_pairs import provenance
        runtime = provenance()
        self.identity = dict(student=student.identity, pairs=digest(pairs), scope=scope,
            max_new_tokens=student.max_new_tokens, max_model_len=student.max_model_len,
            code_sha256=runtime['code_sha256'], dependencies=runtime['dependencies'])

    def state(self, ident):
        return self.states[self.pair_states.get(ident, ident)]

    def _group(self, values):
        if not isinstance(values, list) or any(set(v) != {'state', 'action'} for v in values):
            raise ValueError('Direction groups need state/action objects')
        return [(self.state(v['state']), v['action']) for v in values]

    def rollout(self, state, edit=None):
        from repair.execute import execute_actions
        from repair.ledger import BudgetLedger
        if self.env_factory is None:
            raise ValueError('No authorized environment factory')
        pair = self.pairs[state['origin_pair_id']]
        original = TrajectoryRecord.from_json(pair['audit']['original'])
        t = pair['step']
        calls = []
        def continuation(history, visible):
            # The first decision may receive a textual context intervention;
            # physical state ALWAYS comes from checked reset and prefix replay.
            if len(history) == t and state['counterfactual']:
                current = state
            else:
                text = self.student.tokenizer.apply_chat_template(build_messages(
                    history, {'task_id': pair['task_id']}, visible), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                current = dict(text=text)
                if len(history) == t and text != state['text']:
                    raise ValueError('Replayed prompt differs byte-for-byte')
            result = self.student.greedy(current)
            calls.append(result)
            return result['reply']
        with self.student.edited(edit) as edit_receipt:
            result = execute_actions(lambda: self.env_factory(pair), pair['task_id'],
                [s['action_raw'] for s in original.steps[:t]], 40,
                ledger=BudgetLedger(self.directory / 'environment.jsonl'),
                interaction_kind='evaluation_steps' if self.scope == 'evaluation' else 'candidate_replay_steps',
                continuation=continuation, prefix_reference=original)
        self.environment_steps += result.environment_steps
        path = self.directory / 'executions' / (result.trajectory_id + '.json')
        write_receipt(path, result.to_json())
        if result.termination_reason == 'ENV_ERROR':
            # execute_actions preserves sanitized failures. Never return a fake
            # restored branch or silently turn an unknown outcome into failure.
            raise RuntimeError('P0 replay/rollout failed; see execution receipt: ' + str(path))
        return dict(success=result.success, termination=result.termination_reason,
            actions=[s['action_parsed'] for s in result.steps[t:]],
            environment_steps=result.environment_steps, prefix_turns=t, continuation_turns=len(result.steps) - t,
            policy_tokens=sum(c['prompt_tokens'] + c['generated_tokens'] for c in calls),
            edit_receipt=edit_receipt, execution_sha256=file_hash(path),
            verification='real_environment', framing='free_greedy', counterfactual_first_decision=state['counterfactual'])

    def call(self, tool, args):
        if tool not in self.NAMES or not isinstance(args, dict):
            raise ValueError('Unknown P0 tool/schema')
        args = deepcopy(args)
        framing = 'free_greedy' if tool == 'rollout' else args.setdefault('framing', 'original_thought')
        if tool != 'rollout' and framing not in FRAMINGS:
            raise ValueError('Unknown decision framing')
        key = digest(dict(identity=self.identity, tool=tool, args=args, framing=framing, edit=self.student.active_edit))
        path = self.cache / (key + '.json')
        started = time.perf_counter()
        usage_before = dict(self.student.usage)
        self.ledger.append(dict(event='started', tool=tool, args=args, cache_key=key))
        try:
            # Resolve state capabilities BEFORE a cache lookup, including nested groups.
            for field in ('state', 'target_state', 'source_state'):
                if field in args:
                    self.state(args[field])
            for field in ('positive', 'negative'):
                if field in args:
                    self._group(args[field])
            direction_id = args.get('direction_id') or (args.get('edit') or {}).get('direction_id')
            if direction_id is not None:
                if self.scope == 'diagnosis' and direction_id not in self.direction_specs:
                    raise ValueError('Direction not constructed in this diagnosis session')
                vectors = self.student.directions[direction_id]
                if direction_id != digest({str(i): tensor_hash(d) for i, d in vectors.items()}):
                    raise ValueError('Direction content changed')
            if path.exists():
                result = json.loads(path.read_text())
                if result['cache_key'] != key or result['sha256'] != digest(result['data']):
                    raise ValueError('Corrupt tool cache')
                result['cache_hit'] = True
                if tool == 'counterfactual_context':
                    state = counterfactual(self.student.decision_prefix(self.state(args['state']), framing), args['edit'])
                    if state['sha256'] != result['data']['state_id']:
                        raise ValueError('Cached counterfactual changed')
                    self.states[state['sha256']] = state
                if tool == 'direction':
                    import torch
                    ident = result['data']['direction_id']
                    saved = json.loads((self.directory / 'directions' / (key + '.json')).read_text())
                    vectors = {int(i): torch.tensor(d, dtype=torch.float32) for i, d in saved['vectors'].items()}
                    if saved['construction'] != args or ident != digest({str(i): tensor_hash(d) for i, d in vectors.items()}):
                        raise ValueError('Cached direction changed')
                    self.student.directions[ident] = vectors
                    self.direction_specs[ident] = deepcopy(args)
            else:
                kw = dict(args)
                for field in ('state', 'target_state', 'source_state'):
                    if field in kw:
                        kw[field] = self.state(kw[field])
                if tool == 'counterfactual_context':
                    state = counterfactual(self.student.decision_prefix(kw['state'], framing), kw['edit'])
                    self.states[state['sha256']] = state
                    rendered = self.student.decision_prefix(state, framing)
                    data = dict(state_id=state['sha256'], text=rendered['text'], spans=rendered['spans'],
                                framing=framing, framing_record=rendered['framing_record'])
                elif tool == 'direction':
                    data = self.student.direction(self._group(args['positive']), self._group(args['negative']), args['layers'], framing=framing)
                    self.direction_specs[data['direction_id']] = deepcopy(args)
                    vectors = self.student.directions[data['direction_id']]
                    write_receipt(self.directory / 'directions' / (key + '.json'),
                        dict(construction=args, vectors={str(i): d.tolist() for i, d in vectors.items()}))
                elif tool == 'rollout':
                    data = self.rollout(**kw)
                else:
                    data = getattr(self.student, tool)(**kw)
                result = dict(tool=tool, framing=framing, data=data, summary=tool + ': ' + self._summary(tool, data),
                              cache_key=key, cache_hit=False, sha256=digest(data))
                write_receipt(path, result)
            self.ledger.append(dict(event='settled', tool=tool, args=args, result=result,
                wall_seconds=time.perf_counter() - started,
                student_usage={k: self.student.usage[k] - v for k, v in usage_before.items()},
                environment_steps=0 if result['cache_hit'] else result['data'].get('environment_steps', 0)))
            return result
        except Exception as exc:
            self.ledger.append(dict(event='failed', tool=tool, args=args, cache_key=key,
                student_usage={k: self.student.usage[k] - v for k, v in usage_before.items()},
                error_type=type(exc).__name__, wall_seconds=time.perf_counter() - started))
            raise

    @staticmethod
    def _summary(tool, data):
        if tool == 'action_scores':
            return 'greedy=' + str(data['greedy']['action'] if data['greedy'] else None) + '; ' + '; '.join(
                f"{s['action']} logp={s['log_probability']:.4f}" for s in data['scores'])
        if tool == 'rollout':
            return f"success={data['success']}; {data['environment_steps']} real steps including replay"
        if tool == 'counterfactual_context':
            return 'created state ' + data['state_id']
        if tool == 'direction':
            return 'created unit direction ' + data['direction_id']
        if tool == 'apply_edit':
            return f"edited {data['receipt']['edited_parameters']} parameters; reverted exactly"
        if tool == 'patch':
            return '; '.join(f"{v['action']} delta_logp={v['delta_log_probability']:.4f}" for v in data['changes'])
        if tool == 'logit_lens':
            if not data['layers']:
                return 'candidate actions are identical'
            largest = max(data['layers'], key=lambda r: abs(r['difference']))
            return f"first differing token {data['first_difference']}; largest |difference| at layer {largest['layer']}: {largest['difference']:.4f}"
        if tool == 'projection':
            return '; '.join(f"L{r['layer']} projection={r['projection']:.4f}, cosine={r['cosine']:.4f}" for r in data['layers'])
        if tool == 'attention_summary':
            return '; '.join(f"L{r['layer']} max mean span=" + max(r['spans'], key=lambda s: s['mean'])['id']
                             for r in data['layers'] if r['spans'])
        return 'completed'
