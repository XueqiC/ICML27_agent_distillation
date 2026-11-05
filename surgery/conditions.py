"""Diagnosis-only control construction; immutable conditions precede evaluation."""
from copy import deepcopy
from pathlib import Path
import math

from repair.p0_introspect import tensor_hash
from repair.p0_pairs import CATEGORIES, render_state
from sed.storage import digest
from surgery.common import checked, publish, read, provenance
from surgery.edits import load_delta, save_delta, module_at, validate_edits
from surgery.runtime import identity_key

ALPHAS = (0.25, 0.5, 1.0)
LRS = (1e-5, 3e-5, 1e-4)
STEPS = (50, 200)


def diagnosis_only(pairs):
    if not pairs or any(p['split'] != 'diagnosis' for p in pairs):
        raise ValueError('Condition selection only accepts diagnosis pairs')
    return pairs


def register(student, vectors):
    ident = digest({str(i): tensor_hash(v) for i, v in vectors.items()})
    student.directions[ident] = vectors
    return ident


def restore(student, frozen):
    import torch
    for ident, layers in frozen['vectors'].items():
        vectors = {int(i): torch.tensor(v, dtype=torch.float32) for i, v in layers.items()}
        if register(student, vectors) != ident:
            raise ValueError('Condition direction hash mismatch')
    validate_edits(student, frozen['edits'])


def freeze(path, student, pairs_sha256, setting, name, edits, *, selection=None, trial=False):
    validate_edits(student, edits)
    edits = deepcopy(edits)
    # Make deployed delta references portable with the condition directory.
    from sed.storage import atomic_write
    for edit in edits:
        if edit['type'] == 'local_finetune':
            ref = edit['delta']
            source = student.relocation.path(ref['path'])
            parent = Path(path).parent.resolve()
            if not source.is_relative_to(parent):
                target = parent / 'deltas' / (ref['sha256'] + '.safetensors')
                if target.exists():
                    from sed.storage import file_hash
                    if file_hash(target) != ref['sha256']:
                        raise ValueError('Packaged delta bytes changed')
                else:
                    atomic_write(target, source.read_bytes(), exclusive=True)
                ref['path'] = str(target)
    ids = set()
    for e in edits:
        ids.update(e.get('direction_ids', []))
        ids.update(e[k]['direction_id'] for k in ('a', 'b') if k in e)
    with student.edited(edits) as edit_receipt:
        pass
    return publish(path, dict(schema='p1-condition/1', setting=setting, name=name, trial=trial,
        pairs_sha256=pairs_sha256, student_identity=student.identity, edits=deepcopy(edits),
        edit_receipt=edit_receipt,
        vectors={i: {str(k): d.tolist() for k, d in student.directions[i].items()} for i in sorted(ids)},
        selection=selection or {}, provenance=provenance(dict(condition=name)),
        frozen_before_test=True))


def load_condition(path, pairs_sha256, *, relocation=None):
    value = checked(read(path), 'p1-condition/1')
    if value['pairs_sha256'] != pairs_sha256 or value.get('frozen_before_test') is not True:
        raise ValueError('Condition must be frozen for the same pairs before evaluation')
    for edit in value['edits']:
        if edit['type'] == 'local_finetune':
            load_delta(edit['delta'], relocation)
    return value


def random_edits(student, teacher_edits, seed, directory):
    import torch
    rng = torch.Generator(device='cpu').manual_seed(seed)
    result = deepcopy(teacher_edits)
    def random_direction(ref, layers):
        vectors = {}
        for layer in layers:
            source = student.directions[ref][layer]
            v = torch.randn(source.shape, generator=rng, dtype=torch.float32)
            vectors[layer] = v / v.norm()
        return register(student, vectors)
    for edit in result:
        kind = edit['type']
        if kind == 'project_out':
            edit['direction_ids'] = [random_direction(i, edit['layers']) for i in edit['direction_ids']]
        elif kind == 'add_rank1':
            for field in ('a', 'b'):
                edit[field]['direction_id'] = random_direction(edit[field]['direction_id'], edit['layers'])
        elif kind == 'scale_neurons':
            for layer in edit['layers']:
                width = module_at(student, layer, 'mlp.down_proj').weight.shape[1]
                edit['neurons'][str(layer)] = torch.randperm(width, generator=rng)[:len(edit['neurons'][str(layer)])].tolist()
        else:
            original = load_delta(edit['delta'], student.relocation)
            tensors = {}
            for name, delta in sorted(original.items()):
                random = torch.randn(delta.shape, generator=rng, dtype=torch.float32)
                tensors[name] = random * (delta.norm() / random.norm())
            edit['delta'] = save_delta(directory, tensors, dict(control='Gaussian matched per-tensor Frobenius',
                seed=seed, source_delta_sha256=edit['delta']['sha256']))
    validate_edits(student, result)
    return result


def score(edits, diagnosis, rollout):
    diagnosis_only(diagnosis)
    rows = [dict(pair_id=p['pair_id'], **rollout(p, edits)) for p in diagnosis]
    # Unknown outcomes remain unknown and cannot win a hyperparameter selection.
    known = all(type(r['success']) is bool for r in rows)
    return dict(success_rate=sum(r['success'] is True for r in rows) / len(rows) if known else None, rows=rows)


def choose(candidates):
    eligible = [r for r in candidates if r['evaluation']['success_rate'] is not None]
    if not eligible:
        raise ValueError('No fully verified diagnosis selection candidate')
    # First grid entry wins exact ties; no hidden tuning criterion.
    return max(eligible, key=lambda r: r['evaluation']['success_rate'])


def fixed_rule(student, diagnosis, rollout, *, setting):
    diagnosis_only(diagnosis)
    layers = list(range(20, 42))
    student.layer_ids(layers)
    groups = [diagnosis] if setting == 'B' else [[p for p in diagnosis if p['category'] == c] for c in CATEGORIES]
    edits = []
    for group in groups:
        # Stream h(s,a_f)-h(s,a_r); P0 hidden_mean preserves its exact framing
        # and teacher-forced command-token positions, without P0's 64-item cap.
        sums = {}
        for pair in diagnosis_only(group):
            state = render_state(pair, student.tokenizer)
            fail = student.hidden_mean(state, pair['a_f'], layers, framing='original_thought')['layers']
            repair = student.hidden_mean(state, pair['a_r'], layers, framing='original_thought')['layers']
            for layer in layers:
                d = fail[layer] - repair[layer]
                sums[layer] = sums.get(layer, 0) + d
        if any(v.norm() < 1e-12 for v in sums.values()):
            raise ValueError('Fixed-rule direction is zero')
        ident = register(student, {i: v / v.norm() for i, v in sums.items()})
        edits.append(dict(type='project_out', direction_ids=[ident], layers=layers,
                          targets=['mlp.down_proj'], alpha=1.0))
    candidates = []
    for alpha in ALPHAS:
        trial = [dict(e, alpha=alpha) for e in edits]
        candidates.append(dict(alpha=alpha, edits=trial, evaluation=score(trial, diagnosis, rollout)))
    best = choose(candidates)
    return best['edits'], dict(rule='category mean fail-minus-repair, original_thought, layers 20..41',
        grid=candidates, chosen_alpha=best['alpha'], selection_pair_ids=[p['pair_id'] for p in diagnosis])


def automatic_finetune(student, diagnosis, rollout, directory):
    diagnosis_only(diagnosis)
    examples = [(render_state(p, student.tokenizer), p['a_f'], p['a_r']) for p in diagnosis]
    stats = student.gradient_statistics(examples, margin=False, framing='original_thought')
    stats.sort(key=lambda r: (-r['snr'], r['tensor']))
    modules = [r['tensor'] for r in stats[:math.ceil(len(stats) / 4)]]
    candidates = []
    for lr in LRS:
        for steps in STEPS:
            trained = student.local_finetune(examples, modules, rank=16, lr=lr, steps=steps,
                directory=directory, pair_ids=[p['pair_id'] for p in diagnosis], framing='original_thought')
            edits = [trained['edit']]
            candidates.append(dict(lr=lr, steps=steps, edits=edits, evaluation=score(edits, diagnosis, rollout)))
    best = choose(candidates)
    return best['edits'], dict(snr_definition='||mean(g)||F / sqrt(mean(||g-mean(g)||F^2)); epsilon=1e-12',
        ranking=stats, top_fraction=0.25, top_count=len(modules), rank=16,
        grid=candidates, chosen=dict(lr=best['lr'], steps=best['steps']),
        selection_pair_ids=[p['pair_id'] for p in diagnosis])


def build_controls(student, pairs, deployment, directory, rollout, *, trial=False):
    checked(deployment, 'p1-submission/1')
    if (deployment['pairs_sha256'] != pairs['sha256'] or deployment['setting'] != pairs['setting']
            or identity_key(student.identity) != identity_key(deployment['student_identity'])
            or (not trial and deployment['kind'] != ('integration' if pairs['setting'] == 'A' else 'global_b'))):
        raise ValueError('Conditions require the matching frozen deployed submission')
    # Only diagnosis rows are passed to the selection functions.
    diagnosis = [p for p in pairs['pairs'] if p['split'] == 'diagnosis']
    restore(student, dict(vectors=deployment['vectors'], edits=deployment['submission']['edit']))
    prefix = 'E' if pairs['setting'] == 'A' else 'B'
    directory = Path(directory)
    values = []
    def emit(name, edits, selection=None):
        values.append(freeze(directory / (name + '.json'), student, pairs['sha256'], pairs['setting'],
            name, edits, selection=selection, trial=trial))
    emit(prefix + '0', [])
    teacher = deployment['submission']['edit']
    emit(prefix + '1', teacher, dict(deployment_sha256=deployment['sha256']))
    if trial:
        return values
    for seed in range(5):
        emit(prefix + '2_' + str(seed), random_edits(student, teacher, seed, directory / 'deltas'), dict(seed=seed))
    edits, selection = fixed_rule(student, diagnosis, rollout, setting=pairs['setting'])
    emit(prefix + '3', edits, selection)
    edits, selection = automatic_finetune(student, diagnosis, rollout, directory / 'deltas')
    emit(prefix + '5', edits, selection)
    return values


def condition_set(values, pairs, *, trial=False):
    prefix = 'E' if pairs['setting'] == 'A' else 'B'
    expected = {prefix + '0', prefix + '1'} if trial else {
        prefix + str(i) for i in (0, 1, 3, 5)} | {prefix + '2_' + str(i) for i in range(5)}
    if pairs['setting'] == 'A' and not trial:
        expected.add('E4')
    if len(values) != len(expected) or {v['name'] for v in values} != expected:
        raise ValueError('Freeze EVERY registered condition before any test-side evaluation')
    for value in values:
        checked(value, 'p1-condition/1')
        if value['pairs_sha256'] != pairs['sha256'] or value['setting'] != pairs['setting'] or value['trial'] != trial:
            raise ValueError('Condition set uses different setting/split/trial status')
    identities = [identity_key(v['student_identity']) for v in values if v['name'] != 'E4']
    if any(v != identities[0] for v in identities):
        raise ValueError('All edit conditions must start from the same frozen student')
    return {v['name']: v for v in values}
