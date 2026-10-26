"""Train one E2 arm using this tree's trajectory-SFT recipe and train_loop."""
from copy import deepcopy
from math import prod
from pathlib import Path
import random
import re

from baselines.common.conversation import sequence_rows
from baselines.kang.pipeline import RECIPE
from repair.e2_reader import prepare_examples, require, branch_metadata, checked_base
from baselines.common.shared_history import pack_shared_history
from harness.lora_adapters import inspect_adapter, safetensors_header
from repair.artifacts import read_json, checkpoint_files, provenance, run_manifest
from sed.storage import digest, file_hash, write_json

THETA0_NAME = 'e4b-tsft-k20-e32-lr5e-5'
CHECKPOINTS = (0, 1, 5, 20)
TRAINING_SEED = 1
LORA_INIT_SEED = 20261026


def base_recipe(theta, config, snapshot):
    """K0 uses the registered tsft LoRA, with the K20 LR override, from scratch."""
    from train.modeling import LORA_TARGETS
    require(config['environment']['max_episode_steps'] == 40
            and config['training']['thinking_mode'] == 'off'
            and config['training']['student_step_token_cap'] == 1024,
            'Inherited H/thinking/output-cap protocol differs')
    require(config['models']['student']['requested_repo'] == 'google/gemma-4-E4B-it',
            'Repair pilot requires the configured E4B student')
    metadata = checked_base(theta, snapshot, config['models']['student'])
    recipe = {k: RECIPE[k] for k in ('adaptation', 'scheduler', 'warmup_ratio', 'warmup_steps',
                                    'weight_decay', 'packing', 'max_length')}
    recipe.update(objective='sft', learning_rate=5e-5, scoring='shared_history',
                  attention='flex_attention', compute_dtype='bfloat16', adapter_master_dtype='float32',
                  frozen_base_dtype='bfloat16', optimizer={'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8})
    return {'identity': deepcopy(theta['identity']), 'files': deepcopy(theta['files']), 'recipe': recipe,
        'sft': {k: RECIPE[k] for k in ('rank', 'alpha', 'dropout')} | {
            'targets': list(LORA_TARGETS), 'base_snapshot': metadata, 'packing': recipe['packing']},
        'trainer_state': {'gradient_clip': RECIPE['gradient_clip'],
                          'loss_reduction': 'per_example_mean_then_global_batch_mean'},
        'recipe_sources': {p: file_hash(p) for p in ('scripts/hpg_kang.slurm', 'baselines/kang/pipeline.py',
            'baselines/common/fsdp_train.py', 'baselines/common/schedule.py',
            'baselines/common/shared_history.py', 'train/dataset.py', 'train/modeling.py')},
        'parameters': None, 'trainable_modules': None, 'initialization_seed': LORA_INIT_SEED,
        'optimizer_state': {'mode': 'fresh_AdamW_for_both', 'path': None, 'sha256': None},
        'scheduler_policy': {'decision': 'fresh linear schedule over 20 updates in both arms',
            'shape': recipe['scheduler'], 'learning_rate': recipe['learning_rate'],
            'warmup_ratio': recipe['warmup_ratio'], 'warmup_steps': recipe['warmup_steps'],
            'total_updates': 20, 'restore_scheduler_state': False}}


def initial_adapter_hash(model):
    """Canonical names/shapes/dtypes and raw tensor bytes, independent of save path."""
    import hashlib
    import torch
    tensors = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require('.lora_' in name, 'Fresh model has non-LoRA trainable tensors')
            value = parameter.detach().cpu().contiguous()
            tensors[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
    require(bool(tensors), 'Missing initial adapter tensors')
    return digest(tensors)


def load_fresh_adapter(snapshot, device, inherited, student):
    import torch
    from train.modeling import load_student, attach_lora
    from baselines.common.shared_history import configure_flex_attention, validate_model
    recipe, sft = inherited['recipe'], inherited['sft']
    base = load_student(snapshot, device=str(device), dtype=recipe['frozen_base_dtype'])
    base.set_attn_implementation(recipe['attention'])
    configure_flex_attention(base, recipe.get('flex_kernel_options'))
    devices = [torch.device(device).index] if torch.device(device).type == 'cuda' else []
    # Loading a base can consume RNG differently; seed attachment itself and
    # restore the training stream. Both ranks and both arms use the same seed.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(inherited['initialization_seed'])
        model = attach_lora(base, rank=sft['rank'], alpha=sft['alpha'], dropout=sft['dropout'],
                            targets=sft['targets'])
    require(set(model.peft_config) == {'default'}, 'Expected exactly one fresh adapter')
    model.peft_config['default'].revision = student['revision']
    model.peft_config['default'].base_model_name_or_path = student['requested_repo']
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(getattr(torch, recipe['adapter_master_dtype']))
    validate_model(model)
    return model


def inherited_recipe(theta0, config, snapshot):
    if (config['environment']['max_episode_steps'] != 40 or config['training']['thinking_mode'] != 'off' or
            config['training']['student_step_token_cap'] != 1024):
        raise ValueError('Inherited H/thinking/output-cap protocol differs')
    identity = inspect_adapter(THETA0_NAME, theta0, config['models']['student'], snapshot)
    source = read_json(Path(theta0).parent / 'provenance.json')
    recipe = source['config']['baseline_training']
    # The grid changed only these run choices from Kang's registered recipe.
    # All other recipe values come from this worktree and theta0's provenance.
    required = {k: RECIPE[k] for k in ('scheduler', 'warmup_ratio', 'warmup_steps',
        'weight_decay', 'packing', 'max_length')}
    required.update(objective='sft', adaptation=RECIPE['adaptation'], learning_rate=5e-5,
                    scoring='shared_history', epochs=32, seed=RECIPE['training_seed'], K=20)
    if any(recipe.get(k) != v for k, v in required.items()):
        raise ValueError('theta0 is not the registered E4B tsft K20 e32 lr5e-5 recipe')
    if config['models']['student']['requested_repo'] != 'google/gemma-4-E4B-it':
        raise ValueError('Repair pilot requires the configured E4B student')
    adapter = read_json(Path(theta0) / 'adapter_config.json')
    for saved, key in [('r', 'rank'), ('lora_alpha', 'alpha'), ('lora_dropout', 'dropout')]:
        require(adapter[saved] == RECIPE[key] == source['config']['sft'][key],
                'theta0 LoRA recipe differs from this tree')
    require(set(adapter['target_modules']) == set(source['config']['sft']['targets']),
            'theta0 target coverage differs from training provenance')
    state = read_json(Path(theta0) / 'training_state.json')
    require(state['gradient_clip'] == RECIPE['gradient_clip']
            and state['loss_reduction'] == 'per_example_mean_then_global_batch_mean'
            and state['compute_dtype'] == recipe['compute_dtype']
            and state['master_dtype'] == recipe['adapter_master_dtype']
            and state['scoring'] == recipe['scoring'] and state['packing'] == recipe['packing'],
            'theta0 trainer state differs from inherited recipe')
    header, _ = safetensors_header(Path(theta0) / 'adapter_model.safetensors')
    parameters = {}
    for name, item in header.items():
        if name == '__metadata__':
            continue
        match = re.fullmatch(r'(base_model\.model\..*\.language_model\..+)\.lora_([AB])\.weight', name)
        if not match or match[1].split('base_model.model.', 1)[1] not in adapter['target_modules']:
            raise ValueError('Unexpected trainable adapter tensor')
        shape = item['shape']
        require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)
                and shape[0 if match[2] == 'A' else 1] == adapter['r']
                and item['dtype'] == 'F32', 'Changed adapter shape/master precision')
        parameters[match[1] + '.lora_' + match[2] + '.default.weight'] = prod(shape)
    targets = sorted({n.split('.lora_', 1)[0] for n in parameters})
    expected = {f'base_model.model.{t}.lora_{side}.default.weight'
                for t in adapter['target_modules'] for side in ('A', 'B')}
    require(set(parameters) == expected, 'Missing inherited adapter A/B tensors')
    from train.modeling import LORA_TARGETS
    if not parameters or {n.rsplit('.', 1)[-1] for n in targets} != set(LORA_TARGETS):
        raise ValueError('Incomplete inherited projection coverage')
    if len(parameters) != 2 * len(adapter['target_modules']):
        raise ValueError('Missing inherited adapter A/B tensors')
    opt = Path(theta0) / 'optimizer.pt'
    unknown = [p for p in Path(theta0).iterdir() if 'optim' in p.name.lower() and p != opt]
    if unknown:
        raise ValueError('Unrecognized optimizer state; refuse asymmetric silent restart')
    return {'identity': identity, 'source_provenance_sha256': digest(source),
        'files': checkpoint_files(theta0), 'recipe': deepcopy(recipe), 'sft': source['config']['sft'],
        'trainer_state': state,
        'recipe_sources': {p: file_hash(p) for p in ('scripts/hpg_kang.slurm',
            'baselines/kang/pipeline.py', 'baselines/common/fsdp_train.py', 'baselines/common/schedule.py',
            'baselines/common/shared_history.py', 'train/dataset.py', 'train/modeling.py')},
        'parameters': parameters, 'trainable_modules': targets,
        'optimizer_state': {'mode': 'restore_for_both' if opt.exists() else 'identical_reinitialization_for_both',
                            'path': str(opt.resolve()) if opt.exists() else None,
                            'sha256': file_hash(opt) if opt.exists() else None},
        'scheduler_policy': {'decision': 'theta0 schedule ended; restart same shape over 20 updates in both arms',
            'shape': recipe['scheduler'], 'learning_rate': recipe['learning_rate'],
            'warmup_ratio': recipe['warmup_ratio'], 'warmup_steps': recipe['warmup_steps'],
            'total_updates': 20, 'restore_scheduler_state': False}}


def batch_schedule(tasks, slots, replay_tasks=None):
    rng = random.Random(TRAINING_SEED)
    result = []
    for update in range(1, 21):
        buckets = ['repair', 'repair', 'repair', 'seed']
        rng.shuffle(buckets)
        selections = []
        for bucket in buckets:
            tid = rng.choice(replay_tasks if bucket == 'seed' and replay_tasks is not None else tasks)
            position = rng.randrange(len(slots[tid, bucket]))
            selections.append({'task_id': tid, 'bucket': bucket, 'episode_index': position,
                               'index': slots[tid, bucket][position]})
        result.append({'update': update, 'epoch': 0, 'indices': [s['index'] for s in selections],
                       'selections': selections})
    return result


def accounting(examples, batches):
    context = targets = 0
    result = {0: {'optimizer_updates': 0, 'processed_context_tokens': 0, 'supervised_target_tokens': 0}}
    for batch in batches:
        for i in batch['indices']:
            example = examples[i]
            context += len(pack_shared_history(example)['input_ids'])
            targets += sum(r['supervised_tokens'] for r in sequence_rows(example))
        if batch['update'] in CHECKPOINTS:
            result[batch['update']] = {'optimizer_updates': batch['update'], 'processed_context_tokens': context,
                                      'supervised_target_tokens': targets}
    return result


def load_existing_adapter(snapshot, theta0, device, recipe):
    """One PEFT adapter loaded trainably; never get_peft_model/attach_lora."""
    import torch
    from peft import PeftModel
    from train.modeling import load_student
    from baselines.common.shared_history import configure_flex_attention, validate_model
    base = load_student(snapshot, device=str(device), dtype=recipe['frozen_base_dtype'])
    base.set_attn_implementation(recipe['attention'])
    model = PeftModel.from_pretrained(base, theta0, is_trainable=True, local_files_only=True)
    if set(model.peft_config) != {'default'}:
        raise ValueError('Continuation must contain exactly the original default adapter')
    # Same E-series explicit coverage preservation as attach_lora in 2506fa9;
    # PEFT may otherwise minimize long target lists during loading.
    model.peft_config['default'].target_modules = set(read_json(Path(theta0) / 'adapter_config.json')['target_modules'])
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(getattr(torch, recipe['adapter_master_dtype']))
    model.config.get_text_config().use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    validate_model(model)
    configure_flex_attention(model, recipe.get('flex_kernel_options'))
    return model


def publish_checkpoint(model, output, provenance_record, state, counts, inherited):
    """Use the shared serving-compatible adapter exporter at fixed updates."""
    from baselines.common.fsdp_train import export_adapter
    update = state['updates']
    if update not in CHECKPOINTS or (update == 0 and 'initialization_seed' not in inherited):
        raise ValueError('Only fixed checkpoints (zero for fresh LoRA) may be published')
    destination = Path(output) / f'update-{update:02d}'
    destination.mkdir()
    write_json(destination / 'provenance.json', provenance_record, exclusive=True)
    saved = export_adapter(model, destination / 'checkpoint', provenance_record,
        {**state, 'adaptation': inherited['recipe']['adaptation'], 'epochs': 0 if update == 0 else 1,
         'packing': inherited['recipe']['packing'],
         'optimizer_state': inherited['optimizer_state'], 'scheduler_policy': inherited['scheduler_policy'],
         'gradient_clip': inherited['trainer_state']['gradient_clip'],
         'compute_dtype': inherited['recipe']['compute_dtype'],
         'master_dtype': inherited['recipe']['adapter_master_dtype'],
         'trainable_parameter_count': sum(inherited['parameters'].values()),
         'actual_trainable_parameter_names': sorted(inherited['parameters']),
         'actual_trainable_module_names': inherited['trainable_modules'], **counts})
    item = {'update': update, 'status': 'saved', 'path': str((destination / 'checkpoint').resolve()),
            'sha256': checkpoint_files(destination / 'checkpoint'), 'trainer_state': saved, **counts}
    if update == 0:
        item.update(initial_adapter_sha256=initial_adapter_hash(model),
                    initialization_seed=inherited['initialization_seed'], verification_only=True,
                    probe_checkpoint_zero='base_model')
    write_json(destination / 'manifest.json', item, exclusive=True)
    return item


def run_arm(collection, config, output, *, theta0, snapshot, tokenizer, arm,
            dry_run=False, inputs=()):
    """Consume only read_collection's checked evidence; train_loop owns updates."""
    import os
    document = collection['document']
    branch = branch_metadata(document)
    fresh = branch['theta0_kind'] == 'base_model'
    require(not fresh or theta0 is None or Path(theta0).resolve() == Path(snapshot).resolve(),
            'Base-model branch cannot load an SFT adapter')
    inherited = base_recipe(document['theta0'], config, snapshot) if fresh else inherited_recipe(theta0, config, snapshot)
    require(document['theta0']['files'] == inherited['files'], 'Theta0 changed after reader validation')
    examples, slots = prepare_examples(collection, tokenizer, arm,
        packing=inherited['recipe']['packing'], max_length=inherited['recipe']['max_length'])
    batches = batch_schedule(document['D_repair_train'], slots, collection['seed_replay_task_ids'])
    counts = accounting(examples, batches)
    training_tasks = {t['task_id']: t for t in [*collection['tasks'], *collection['seed_tasks']]}
    config = deepcopy(config)
    config['sft'] = inherited['sft']
    config['repair_training'] = {'arm': arm, **branch, 'inherited': inherited, 'training_seed': TRAINING_SEED,
        'seed_replay_task_ids': collection['seed_replay_task_ids'],
        'optimizer_updates_cap': 20, 'effective_batch_episodes': 4, 'repair_weight': .75,
        'seed_replay_weight': .25, 'loss_mask': 'full_trajectory', 'schedule_sha256': digest(batches),
        'examples_sha256': digest(examples),
        'loss_normalization': inherited['trainer_state']['loss_reduction']}
    rank, world, local = (int(os.environ.get(k, d)) for k, d in
                         [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    require(world == 1 if dry_run else world == 2,
            'Use one CPU process for dry-run or exactly two GPU ranks for continuation')
    output = Path(output)
    def record():
        output.mkdir(parents=True, exist_ok=False)
        prov = provenance(output, config, command='repair_train_arm',
            inputs=[*inputs, *collection['inputs'],
                    *([] if fresh else [Path(theta0).parent / 'provenance.json'])])
        write_json(output / 'batches.json', batches, exclusive=True)
        write_json(output / 'example_sources.json', [e['label_provenance'] for e in examples], exclusive=True)
        manifest = run_manifest(prov, checkpoint=inherited, tokenizer=document['tokenizer'],
            tasks=list(training_tasks.values()),
            seeds={'training': TRAINING_SEED, 'rank_seeds': [TRAINING_SEED + r for r in range(world)],
                   'environment': config['environment']['seed']},
            format='repair_e2_training_run', arm=arm, dry_run=dry_run, **branch,
            seed_replay_task_ids=collection['seed_replay_task_ids'],
            initialization_seed=inherited.get('initialization_seed'),
            initial_adapter_sha256=inherited.get('initial_adapter_sha256'),
            theta0=document['theta0'], training_collection_sha256=collection['training_collection_sha256'],
            source_collection_sha256=document['source_collection_sha256'],
            source_input_hashes=document['input_hashes'],
            teacher_calls_this_run=0, environment_transitions_this_run=0,
            status='planned' if dry_run else 'running',
            K_trusted_seed_trajectories=document['K_trusted_seed_trajectories'],
            D_repair_train=document['D_repair_train'], D_repair_probe=document['D_repair_probe'],
            repair_training_unique_instances=document['unique_instances'], repair_training_task_groups=document['task_groups'],
            training_unique_instances=len(training_tasks),
            training_task_groups=len({t['group_id'] for t in training_tasks.values()}),
            replay_trusted_seed_trajectories=sum(len(v) for v in collection['seeds'].values()),
            replay_unique_instances=len(collection['seed_tasks']),
            replay_task_groups=len({t['group_id'] for t in collection['seed_tasks']}),
            trainable_parameter_count=sum(inherited['parameters'].values()) if inherited['parameters'] is not None else None,
            actual_trainable_parameter_names=sorted(inherited['parameters']) if inherited['parameters'] is not None else None,
            actual_trainable_module_names=inherited['trainable_modules'],
            parameter_names_source=('pending_fresh_model_attachment' if fresh else 'checkpoint_safetensors_header')
                                   if dry_run else 'validated_model_named_parameters',
            optimizer_state=inherited['optimizer_state'], scheduler_policy=inherited['scheduler_policy'],
            schedule_sha256=digest(batches), encoded_examples_sha256=digest(examples),
            training_exposure=0, planned_training_exposure=80,
            checkpoints=[{'update': n, **counts[n], 'path': None, 'sha256': None,
                          'status': 'planned'} for n in (CHECKPOINTS if fresh else CHECKPOINTS[1:])])
        write_json(output / 'manifest.json', manifest, exclusive=True)
        return prov, manifest
    if dry_run:
        return record()[1]
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from transformers import set_seed
    from baselines.common.fsdp_train import train_loop, rank_zero_data
    if not torch.cuda.is_available():
        raise RuntimeError('Production continuation requires allocated GPUs; use --dry-run on CPU')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device)
    try:
        group = dist.new_group(backend='gloo', timeout=timedelta(hours=4))
        set_seed(TRAINING_SEED + rank)
        model = (load_fresh_adapter(snapshot, device, inherited, config['models']['student']) if fresh else
                 load_existing_adapter(snapshot, theta0, device, inherited['recipe']))
        actual = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
        if fresh:
            initial_hash = initial_adapter_hash(model)
            rank_hashes = [None] * world
            dist.all_gather_object(rank_hashes, initial_hash, group=group)
            require(all(h == initial_hash for h in rank_hashes), 'Fresh adapter initialization differs across ranks')
            inherited.update(parameters=actual, initial_adapter_sha256=initial_hash,
                             trainable_modules=sorted({n.split('.lora_', 1)[0] for n in actual}))
            inherited['sft']['targets'] = sorted(model.peft_config['default'].target_modules)
        else:
            require(actual == inherited['parameters'], 'Loaded trainable tensors differ from theta0; no extra adapter allowed')
        for p in model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)
        opt = inherited['optimizer_state']
        optimizer = torch.load(opt['path'], map_location='cpu', weights_only=True) if opt['path'] else None
        if optimizer is not None and 'optimizer' in optimizer:
            if 'trainable' in optimizer:
                from train.optimizer_state import assert_trainable_equal
                require(list(optimizer['trainable']) == list(actual), 'Optimizer parameter order differs')
                assert_trainable_equal(model, optimizer['trainable'])
            optimizer = optimizer['optimizer']
        prov, manifest = rank_zero_data(rank, world, record, group)
        if fresh:
            def publish_initial():
                with torch.random.fork_rng(devices=[local]):
                    item = publish_checkpoint(model, output, prov,
                        {'updates': 0, 'examples_seen': 0, 'supervised_tokens_seen': 0}, counts[0], inherited)
                require(item['initial_adapter_sha256'] == inherited['initial_adapter_sha256'],
                        'Initial adapter changed before checkpoint zero')
                manifest['checkpoints'][0].update(item)
                write_json(output / 'manifest.json', manifest)
            rank_zero_data(rank, world, publish_initial, group)
        def save(model, state):
            update = state['updates']
            if update not in CHECKPOINTS:
                return
            def publish():
                item = next(r for r in manifest['checkpoints'] if r['update'] == update)
                require(state['supervised_tokens_seen'] == counts[update]['supervised_target_tokens']
                        and state['physical_policy_forward_tokens'] == counts[update]['processed_context_tokens'],
                        'Actual training exposure differs from dry-run plan')
                # Publication must not perturb the next dropout stream.
                with torch.random.fork_rng(devices=[local]):
                    item.update(publish_checkpoint(model, output, prov, state, counts[update], inherited))
                manifest['training_exposure'] = state['examples_seen']
                write_json(output / 'manifest.json', manifest)
            rank_zero_data(rank, world, publish, group)
        recipe = inherited['recipe']
        with torch.autocast('cuda', dtype=getattr(torch, recipe['compute_dtype'])):
            state = train_loop(model, examples, output, objective=recipe['objective'], learning_rate=recipe['learning_rate'],
                scheduler=recipe['scheduler'], warmup_ratio=recipe['warmup_ratio'], warmup_steps=recipe['warmup_steps'],
                original_batch=4, seed=TRAINING_SEED, weight_decay=recipe['weight_decay'],
                packing=recipe['packing'], scoring=recipe['scoring'],
                flex_kernel_options=recipe.get('flex_kernel_options'), rank=rank, world_size=world,
                device=device, replicated_lora=True, fixed_batches=batches,
                optimizer_state=optimizer, checkpoint_callback=save)
        if rank == 0:
            manifest.update(status='complete', trainer_state=state, training_exposure=state['examples_seen'])
            write_json(output / 'manifest.json', manifest)
            return manifest
    finally:
        dist.destroy_process_group()
