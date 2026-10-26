"""Observe one served fixed checkpoint, or combine immutable per-checkpoint files."""
import argparse
from functools import partial
from pathlib import Path

from harness.alfworld_env import Environment
from repair.artifacts import read_json, checked_document
from repair.branches import collection_branch
from repair.export import export_branch
from repair.probes import run_probes, combined_table, validate_training_checkpoint, validate_serving
from sed.config import load_config
from sed.manifest import load_frozen, resolve_task, authorize_task
from students.vllm_student import context_options
from tools.repair_coverage import checked_repair_splits


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/base_e4b.yaml')
    p.add_argument('--collection')
    p.add_argument('--repair-splits', default='data/manifest/repair_v01_splits.json')
    p.add_argument('--arm', choices=['shared', 'N', 'F'])
    p.add_argument('--checkpoint', type=int, choices=[0, 1, 5, 20])
    p.add_argument('--training-manifest')
    p.add_argument('--training-collection', help='E2 export; defaults to collection sibling e2_training_collection.json')
    p.add_argument('--endpoint')
    p.add_argument('--model-name')
    p.add_argument('--server-provenance')
    p.add_argument('--lora-check')
    p.add_argument('--data-root')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--combine', nargs='+', help='checkpoint.json files; requires only --output')
    p.add_argument('--output', required=True)
    a = p.parse_args(argv)
    if a.combine:
        combined_table([checked_document(f) for f in a.combine], a.output)
        return 0
    if any(v is None for v in (a.collection, a.arm, a.checkpoint, a.endpoint, a.model_name,
                               a.server_provenance)):
        p.error('Served probes require collection, arm, checkpoint, endpoint, model-name and server-provenance')
    config = load_config(a.config)
    if a.data_root:
        config['environment']['data_root'] = a.data_root
    if config['environment']['max_episode_steps'] != 40:
        raise ValueError('Pilot requires H=40')
    config['server_provenance'] = read_json(a.server_provenance)
    collection = checked_document(a.collection)
    context = collection_branch(collection)
    base_zero = context['theta0_kind'] == 'base_model' and a.checkpoint == 0
    if not base_zero and not a.lora_check:
        p.error('Adapter probes require --lora-check')
    if base_zero and a.lora_check:
        p.error('Base checkpoint zero must not supply --lora-check')
    if a.lora_check:
        config['lora_equivalence'] = read_json(a.lora_check)
    if collection['dry_run'] or collection['canary']:
        raise ValueError('Synthetic/canary collection cannot drive real probes')
    manifest, freeze = load_frozen(config['paths']['manifest'], config['paths']['split_freeze'])
    splits = checked_repair_splits(a.repair_splits, manifest, freeze)
    for tid in collection['D_repair_probe']:
        task = resolve_task(manifest, freeze, tid)
        authorize_task(task, 'generation')
        if task['group_id'] in splits['sets']['D_probe']['group_ids']:
            raise ValueError('Independent evaluation task cannot supply a failure-state probe')
    inputs = [a.config, a.collection, a.repair_splits, a.server_provenance,
              config['paths']['manifest'], config['paths']['split_freeze']]
    if a.lora_check:
        inputs.append(a.lora_check)
    spec = dict(endpoint=a.endpoint, model=a.model_name, revision=config['models']['student']['revision'],
                enable_thinking=False, max_tokens=1024, **context_options(config))
    adapter = (validate_serving(config, spec, base_model=True, expected=collection['checkpoint'])
               if base_zero else validate_serving(config, spec))
    if a.checkpoint:
        if not a.training_manifest:
            p.error('Updated checkpoint requires --training-manifest')
        trained = read_json(a.training_manifest)
        export_path = a.training_collection or str(Path(a.collection).with_name('e2_training_collection.json'))
        export = checked_document(export_path)
        if export_branch(export) != context:
            raise ValueError('Export branch differs from collection')
        if export['theta0'] != collection['checkpoint']:
            raise ValueError('Export theta0 differs from collection')
        validate_training_checkpoint(trained, export, a.collection, export_path, arm=a.arm,
            checkpoint=a.checkpoint, adapter=adapter)
        inputs.extend([a.training_manifest, export_path])
    run_probes(collection, splits, config, a.output, arm=a.arm, checkpoint=a.checkpoint, spec=spec,
        env_factory=partial(Environment, config['environment']['data_root'], split='train',
            seed=config['environment']['seed'], max_episode_steps=40), workers=a.workers, inputs=inputs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
