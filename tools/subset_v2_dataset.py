"""Seal a task-stratified, supervised-turn-matched subset of a V2 dataset.

No labels, trajectories, or training exposures are generated. Historical label
artifacts and teacher accounting remain references to the source dataset.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory

from repair.artifacts import checked_document
from repair.supervision import (ARMS, SCHEMA, InvalidExample, SplitPolicy,
    dataset_manifest, seal_dataset, validate_example)
from sed.storage import atomic_write, canonical_bytes, digest, file_hash, write_json


RECOMPUTED = {'schema', 'created', 'counts', 'supervised_turns', 'supervised_tokens',
              'supervised_token_estimate', 'example_ids_sha256',
              'training_exposure_tasks', 'trajectory_diagnostics'}
EXAMPLE_FIELDS = {'example_id', 'arm', 'source', 'kind', 'task_id', 'group_id',
                  'task_type', 'messages', 'supervised_assistant_indices', 'weight', 'evidence'}


def require(condition, reason):
    if not condition:
        raise InvalidExample(reason)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result

    def nonfinite(_):
        raise InvalidExample('nonfinite_json')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _tokenizer(manifest):
    """Keep the source's counting convention; never download a tokenizer."""
    estimate = manifest.get('supervised_token_estimate', {})
    method = estimate.get('method', 'ceil_char/4')
    require(method in ('ceil_char/4', 'local_tokenizer'), 'unknown_token_count_method')
    info = estimate.get('tokenizer')
    if method == 'ceil_char/4':
        return None, info
    require(isinstance(info, dict) and info.get('files') and info.get('path'),
            'missing_tokenizer_identity')
    snapshot = Path(info['path'])
    for name, expected in info['files'].items():
        require(file_hash(snapshot / name) == expected, 'tokenizer_hash_mismatch')
    from train.dataset import load_tokenizer
    return load_tokenizer(snapshot), info


def _metadata(manifest, tokenizer, info):
    metadata = {k: v for k, v in manifest.items() if k not in RECOMPUTED}
    metadata.pop('examples_sha256', None)
    excluded = manifest['excluded']
    require(isinstance(excluded, list) and all(
        isinstance(e, dict) and set(e) == {'reason', 'count'} and
        isinstance(e['reason'], str) and type(e['count']) is int and e['count'] > 0
        for e in excluded), 'invalid_excluded_counts')
    metadata['excluded'] = Counter({e['reason']: e['count'] for e in excluded})
    require(len(metadata['excluded']) == len(excluded), 'duplicate_exclusion_reason')
    metadata.update(tokenizer=tokenizer, tokenizer_info=info)
    return metadata


def _read_bundle(directory):
    manifest = _json((directory / 'manifest.json').read_bytes())
    require(manifest['schema'] == SCHEMA and manifest['arm'] in ARMS, 'invalid_schema_or_arm')
    require(type(manifest['dry_run']) is bool and type(manifest['training_eligible']) is bool,
            'invalid_eligibility_flags')
    require(not manifest['dry_run'] or not manifest['training_eligible'], 'eligible_dry_run')
    raw = (directory / 'examples.jsonl').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    seal = (directory / 'examples.sha256').read_text().split()
    require(len(seal) in (1, 2) and seal[0] == sha and
            (len(seal) == 1 or seal[1] in ('examples.jsonl', '*examples.jsonl')),
            'examples_hash_mismatch')
    if 'examples_sha256' in manifest:
        require(manifest['examples_sha256'] == sha, 'manifest_examples_hash_mismatch')
    lines = raw.splitlines(keepends=True)
    require(not raw or raw.endswith(b'\n'), 'incomplete_jsonl_row')
    rows = [_json(line) for line in lines]
    require(all(isinstance(e, dict) and set(e) == EXAMPLE_FIELDS for e in rows), 'example_schema')
    ids = [e['example_id'] for e in rows]
    require(ids == sorted(set(ids)), 'unsorted_or_duplicate_example_ids')
    require(digest(ids) == manifest['example_ids_sha256'], 'example_ids_hash_mismatch')
    require(isinstance(manifest['inputs'], dict) and manifest['inputs'], 'missing_inputs')
    for name, expected in manifest['inputs'].items():
        require(file_hash(name) == expected, 'input_hash_mismatch')
    theta0 = manifest['theta0']
    require(file_hash(Path(theta0['adapter_path']) / 'adapter_model.safetensors') ==
            theta0['adapter_weights_sha256'], 'theta0_hash_mismatch')
    return dict(root=directory, manifest=manifest, rows=rows, lines=dict(zip(ids, lines)))


def verify_dataset(directory):
    """Verify both bundles, frozen membership, row contracts, and summaries."""
    directory = Path(directory).resolve()
    root = _read_bundle(directory)
    replay = _read_bundle(directory / 'replay')
    manifest, rm = root['manifest'], replay['manifest']
    reference = manifest['replay']
    require(reference['path'] == 'replay' and
            reference['manifest_sha256'] == file_hash(directory / 'replay/manifest.json') and
            reference['example_ids_sha256'] == rm['example_ids_sha256'] and
            reference['examples'] == len(replay['rows']) and
            reference['supervised_turns'] == rm['supervised_turns'], 'replay_seal_mismatch')
    require(all(e['source'] == 'k20_replay' for e in replay['rows']), 'invalid_replay_source')
    require(all(manifest[k] == rm[k] for k in
                ('arm', 'splits_sha256', 'theta0', 'dry_run', 'training_eligible')),
            'replay_identity_mismatch')
    split_paths = [p for p, sha in manifest['inputs'].items() if sha == manifest['splits_sha256']]
    require(bool(split_paths), 'missing_split_freeze')
    splits = checked_document(split_paths[0])
    rows = root['rows'] + replay['rows']
    policy = SplitPolicy(splits, [e['task_id'] for e in replay['rows']],
                         {e['task_id']: e['task_type'] for e in rows})
    for bundle in (root, replay):
        m = bundle['manifest']
        for row in bundle['rows']:
            require(row['arm'] == m['arm'], 'example_arm_mismatch')
            validate_example(row, policy)
        tokenizer, info = _tokenizer(m)
        metadata = _metadata(m, tokenizer, info)
        recomputed = dataset_manifest(bundle['rows'], **metadata)
        for key in RECOMPUTED - {'created'}:
            if key in m or key not in {'supervised_tokens', 'supervised_token_estimate', 'trajectory_diagnostics'}:
                require(canonical_bytes(m[key]) == canonical_bytes(recomputed[key]),
                        'manifest_mismatch_' + key)
        bundle.update(tokenizer=tokenizer, tokenizer_info=info, metadata=metadata)
    # These are historical references in subsets, not files required by E0.5.
    if 'subset' not in manifest:
        for name, expected in manifest.get('label_artifact_sha256', {}).items():
            require(file_hash(directory / name) == expected, 'label_artifact_hash_mismatch')
    hashes = {directory / name: file_hash(directory / name)
              for name in ('manifest.json', 'examples.jsonl', 'examples.sha256')}
    hashes.update({p: file_hash(p) for p in (directory / 'replay').rglob('*') if p.is_file()})
    root.update(policy=policy, hashes=hashes)
    return root


def select_examples(rows, target, seed):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda e: e['example_id']):
        groups[row['task_id']].append(row)
    turns = {task: sum(len(e['supervised_assistant_indices']) for e in examples)
             for task, examples in groups.items()}
    total = sum(turns.values())
    require(type(target) is int and 0 <= target < total, 'target_must_be_smaller_than_source')
    quotas = {task: target * count // total for task, count in turns.items()}
    order = sorted(turns, key=lambda task: (-(target * turns[task] % total), task))
    for task in order[:target - sum(quotas.values())]:
        quotas[task] += 1
    selected, failures = [], {}
    for task in sorted(groups):
        examples = groups[task]
        random.Random(f'{seed}:{task}').shuffle(examples)
        count = 0
        for example in examples:
            n = len(example['supervised_assistant_indices'])
            if count + n <= quotas[task]:
                selected.append(example)
                count += n
        if count != quotas[task]:
            failures[task] = dict(source_turns=turns[task], quota=quotas[task],
                                  selected_turns=count, shortfall=quotas[task] - count)
    if failures:
        print(json.dumps(dict(status='refused', reason='inexact_task_quotas',
            target_turns=target, all_examples_one_turn=all(
                len(e['supervised_assistant_indices']) == 1 for e in rows), per_task=failures)))
        raise InvalidExample('inexact_task_quotas')
    return sorted(selected, key=lambda e: e['example_id']), {
        task: [turns[task], quotas[task]] for task in sorted(groups)}


def subset(args):
    # Validate both inputs before output reservation or selection.
    source, match = verify_dataset(args.source), verify_dataset(args.match)
    sm, mm = source['manifest'], match['manifest']
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('output_exists')
    require(sm['dry_run'] or sm['training_eligible'], 'real_source_not_training_eligible')
    require(sm['dry_run'] or not mm['dry_run'], 'real_source_with_dry_run_match')
    target = mm['supervised_turns']
    selected, per_task = select_examples(source['rows'], target, args.seed)
    metadata = dict(source['metadata'])
    metadata['subset'] = dict(source=str(source['root']),
        source_manifest_sha256=source['hashes'][source['root'] / 'manifest.json'],
        source_example_ids_sha256=sm['example_ids_sha256'], source_supervised_turns=sm['supervised_turns'],
        match=str(match['root']), match_manifest_sha256=match['hashes'][match['root'] / 'manifest.json'],
        target_turns=target, seed=args.seed, stratification='task_id largest-remainder', per_task=per_task)
    ids = {e['example_id'] for e in selected}
    if 'decision_diagnostics' in metadata:
        metadata['decision_diagnostics'] = {k: v for k, v in metadata['decision_diagnostics'].items() if k in ids}
    output.mkdir(parents=True, exist_ok=False)
    # seal_dataset serializes canonical rows. Stage its result so original JSONL
    # bytes can be retained even for an equivalently formatted, re-sealed source.
    # The root manifest is always published last, after all checks and copying.
    with TemporaryDirectory(prefix='.subset-seal-', dir=output) as temporary:
        stage = Path(temporary)
        sealed = seal_dataset(stage, selected, policy=source['policy'], **metadata)
        raw = b''.join(source['lines'][e['example_id']] for e in selected)
        sha = hashlib.sha256(raw).hexdigest()
        if 'examples_sha256' in sm:
            sealed['examples_sha256'] = sha
        shutil.copytree(source['root'] / 'replay', output / 'replay')
        for path, expected in {**source['hashes'], **match['hashes']}.items():
            require(file_hash(path) == expected, 'input_changed_during_subset')
        for path, expected in source['hashes'].items():
            if path.is_relative_to(source['root'] / 'replay'):
                require(file_hash(output / path.relative_to(source['root'])) == expected,
                        'replay_copy_hash_mismatch')
        atomic_write(output / 'examples.jsonl', raw, exclusive=True)
        atomic_write(output / 'examples.sha256', (sha + '\n').encode(), exclusive=True)
    write_json(output / 'manifest.json', sealed, exclusive=True)
    print(json.dumps(dict(source=str(source['root']), match=str(match['root']),
        source_turns=sm['supervised_turns'], match_turns=mm['supervised_turns'], target_turns=target,
        selected_examples=len(selected), selected_turns=sealed['supervised_turns'],
        tasks=sealed['counts']['unique_instances'], output=str(output))))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--match', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    return subset(parser.parse_args(argv))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('E0.4 subset failed: ' + type(exc).__name__ + '; no root manifest was sealed.', file=sys.stderr)
        raise SystemExit(1) from None
