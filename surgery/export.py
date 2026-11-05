"""Standalone bf16 publication with full-vocabulary fixed-input reload checks."""
from pathlib import Path
import gc
import shutil

from sed.react import build_messages
from sed.storage import digest, file_hash, write_json
from train.merge_lora import next_token_logits, _compare_logits
from train.modeling import copy_base_assets
from repair.p0_introspect import tensor_hash
from surgery.common import checked, publish, provenance
from surgery.conditions import restore
from surgery.edits import module_at
from surgery.runtime import identity_key


def fixed_inputs(student):
    result = []
    for observation in ('You are in a room. Look around.', 'Your task is to: put an apple on a table.'):
        text = student.tokenizer.apply_chat_template(build_messages([], {},
            dict(observation=observation, admissible_commands=['look', 'inventory'])),
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        result.append(student.tokenizer.encode(text, add_special_tokens=False))
    return result


def exported_weight_checks(student, output, tensors):
    """Compare serialized bytes to the installed tensors, including signed zero."""
    import torch
    from safetensors import safe_open
    names = {id(module): name for name, module in student.model.named_modules()}
    files = {}
    for path in sorted(Path(output).glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                if name in files:
                    raise ValueError('Duplicate exported tensor')
                files[name] = path
    checks = {}
    for key in tensors:
        layer, target = key.split(':')
        module = module_at(student, int(layer), target)
        name = names[id(module)] + '.weight'
        installed = module.weight.detach().cpu().contiguous()
        check = dict(tensor=name, installed_sha256=tensor_hash(installed),
                     dtype=str(installed.dtype).removeprefix('torch.'), shape=list(installed.shape), exact=False)
        if name in files:
            with safe_open(files[name], framework='pt', device='cpu') as handle:
                exported = handle.get_tensor(name).contiguous()
                check.update(exported_sha256=tensor_hash(exported),
                    exact=(installed.dtype == exported.dtype and installed.shape == exported.shape
                           and torch.equal(installed.view(torch.uint8), exported.view(torch.uint8))))
        checks[key] = check
    return checks


def export_model(student, condition, snapshot, output, *, loader=None, inputs=None):
    import torch
    checked(condition, 'p1-condition/1')
    if identity_key(student.identity) != identity_key(condition['student_identity']):
        raise ValueError('Export runtime differs from frozen condition')
    if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in student.model.parameters()):
        raise ValueError('Standalone export requires native bf16 parameters before evaluation')
    output, snapshot = Path(output), Path(snapshot)
    output.mkdir(parents=True, exist_ok=False)
    restore(student, condition)
    inputs = inputs or fixed_inputs(student)
    checks = []
    with student.edited(condition['edits']) as edit_receipt:
        reference = [next_token_logits(student.model, ids) for ids in inputs]
        # Serialize exactly the weights already used for evaluation. No dtype conversion.
        student.model.save_pretrained(output, safe_serialization=True)
        weight_checks = exported_weight_checks(student, output, edit_receipt['rounding']['tensors'])
        if hasattr(student.tokenizer, 'save_pretrained'):
            student.tokenizer.save_pretrained(output)
        copy_base_assets(snapshot, output)
        for name in ('config.json', 'generation_config.json'):
            if (snapshot / name).is_file():
                shutil.copyfile(snapshot / name, output / name)
    # Never keep two full models on the GPU during reload.
    device = student.device
    student.model.to('cpu')
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if loader is None:
        from transformers import Gemma4ForConditionalGeneration
        loader = lambda p: Gemma4ForConditionalGeneration.from_pretrained(p, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': str(device)})
    model = loader(output).eval()
    try:
        reloaded = [next_token_logits(model, ids) for ids in inputs]
        for before, after in zip(reference, reloaded):
            checks.append(dict(edited_to_export=_compare_logits(before, after, atol=0.125, rtol=0.01),
                bf16_reload=dict(_compare_logits(before, after, atol=0.125, rtol=0.01),
                                 exact=bool(torch.equal(before, after)))))
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        student.model.to(device)
    weights_exact = all(c['exact'] for c in weight_checks.values())
    passed = weights_exact and all(c['edited_to_export']['passed'] and c['bf16_reload']['passed'] for c in checks)
    report = dict(passed=passed, checks=checks, fixed_input_ids=inputs, fixed_input_sha256=digest(inputs),
        edited_weights_exact=weights_exact, edited_weight_checks=weight_checks, edit_receipt=edit_receipt,
        dtype='bfloat16', condition_sha256=condition['sha256'], student_identity=student.identity,
        tolerances=dict(edited_to_export=dict(atol=0.125, rtol=0.01), bf16_reload=dict(atol=0.125, rtol=0.01)),
        provenance=provenance(dict(snapshot=str(snapshot), condition=condition['name'])))
    write_json(output / 'consistency.json', report, exclusive=True)
    manifest = publish(output / 'surgery_manifest.json', dict(schema='p1-export/1', passed=passed,
        condition_name=condition['name'], condition_sha256=condition['sha256'], pairs_sha256=condition['pairs_sha256'],
        trial=condition['trial'],
        edited_weights_exact=weights_exact, edit_receipt=edit_receipt,
        files={str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()},
        fixed_input_sha256=digest(inputs), dtype='bfloat16', student_identity=student.identity))
    if not passed:
        if not weights_exact:
            raise ValueError('Exported tensors differ byte-for-byte from installed weights; failed artifact must not be served')
        raise ValueError('Export/reload logits differ; failed artifact must not be served')
    return manifest


def verify_export(directory):
    from surgery.common import read
    directory = Path(directory)
    value = checked(read(directory / 'surgery_manifest.json'), 'p1-export/1')
    if not value['passed']:
        raise ValueError('Export consistency failed')
    actual = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
              if p.is_file() and p.name != 'surgery_manifest.json'}
    if actual != value['files']:
        raise ValueError('Export file manifest changed')
    return value


def record_m4_launch(exported, tasks, verified, output, conditions_dir):
    from surgery.common import read
    from surgery.conditions import condition_set
    manifest = verify_export(exported)
    if manifest['trial']:
        raise ValueError('Trial exports may be load-tested; dev140 requires the formal condition freeze')
    conditions = [checked(read(p), 'p1-condition/1') for p in sorted(Path(conditions_dir).glob('[EB]*.json'))]
    all_conditions = condition_set(conditions, dict(setting='A' if manifest['condition_name'].startswith('E') else 'B',
        sha256=manifest['pairs_sha256']), trial=manifest['trial'])
    if all_conditions[manifest['condition_name']]['sha256'] != manifest['condition_sha256']:
        raise ValueError('M4 export differs from the complete frozen condition set')
    listing, seeds = read(tasks), read(Path(verified) / 'manifest.json')
    if (listing.get('name') != 'dev140' or len(set(listing['task_ids'])) != 140
            or listing['list_sha256'] != digest(listing['task_ids'])
            or seeds.get('K') != 20 or len(seeds['verified']) != 20):
        raise ValueError('M4 requires frozen dev140 and verified K20')
    return publish(Path(output) / 'p1_launch.json', dict(schema='p1-m4-launch/1',
        export_manifest_sha256=manifest['sha256'], condition_sha256=manifest['condition_sha256'],
        exported=str(Path(exported).resolve()), tasks_sha256=file_hash(tasks), task_list_sha256=listing['list_sha256'],
        verified_manifest_sha256=file_hash(Path(verified) / 'manifest.json'),
        conditions={n: c['sha256'] for n, c in all_conditions.items()}, trial=manifest['trial'],
        settings=dict(greedy=True, batch_invariant=True, workers=16, enable_thinking=False, max_tokens=1024),
        eval_tree='46b8527'))
