"""Validate original P1 support without opening parent full-TRAIN trajectories.

The shared reader recursively audits the full collection. Here only selection
and receipt metadata may be read outside K20; episode/teacher bytes are confined
to the supplied original support. No source demonstrations become model inputs.
"""
from pathlib import Path

from repair.artifacts import read_json
from sed.manifest import authorize_task, resolve_task
from sed.provenance import training_support_manifest
from sed.seed_evidence import (_subset_values, artifact_hashes, checked_episodes,
                               environment_won, receipt_path)
from sed.seed_selection import validate_selection
from sed.storage import digest, file_hash


def load_original_support(path, manifest, freeze, *, adapter, splits):
    path = Path(path).resolve()
    collection, selection, identity = [read_json(path / n) for n in
                                      ('manifest.json', 'selection.json', 'identity.json')]
    training = read_json(Path(adapter).resolve().parent / 'provenance.json')
    original_path = training_support_manifest(training)
    # Accept the spelling kept in --verified or an absolute provenance input path.
    hashes = [v for k, v in training['inputs'].items()
              if k == str(original_path) or k.endswith('/' + str(original_path))]
    if hashes != [file_hash(path / 'manifest.json')]:
        raise ValueError('P1 --verified must be theta0 original support, bound by training provenance')
    ids = [r['task_id'] for r in collection['verified']]
    if (collection['K'] != 20 or len(ids) != 20 or len(set(ids)) != 20
            or collection['status'] != 'complete' or collection['identity_sha256'] != digest(identity)
            or collection['selection_sha256'] != digest(selection) or identity['selection'] != selection
            or selection['manifest_sha256'] != digest(manifest)
            or selection['split_freeze_sha256'] != digest(freeze)):
        raise ValueError('P1 requires the original frozen K20 support view')
    blocked = set(splits['sets']['D_probe']['group_ids']) | {
        resolve_task(manifest, freeze, tid)['group_id'] for tid in splits['sets']['U_mech']['task_ids']}
    if any(resolve_task(manifest, freeze, tid)['group_id'] in blocked for tid in ids):
        raise ValueError('Original support overlaps U_mech/probe groups')
    root = path
    if 'source_collection' in collection:
        if path.parent.name != 'subsets' or path.name != 'k20':
            raise ValueError('Invalid original K20 view location')
        root = path.parent.parent
        source, large, source_identity = [read_json(root / n) for n in
                                         ('manifest.json', 'selection.json', 'identity.json')]
        if (source['selection_sha256'] != digest(large)
                or source['identity_sha256'] != digest(source_identity)
                or source_identity['selection'] != large or 'source_collection' in source):
            raise ValueError('Original support source metadata changed')
        expected_identity, expected = _subset_values(root, source, large, selection)
        if identity != expected_identity or collection != expected:
            raise ValueError('Original support view differs from parent receipt metadata')
    else:
        if 'extension_source' in identity or 'support_view' in identity:
            raise ValueError('P1 requires original demonstrations, not a new publication')
        validate_selection(selection, manifest, freeze)
    if any(r['task_id'] not in ids for r in collection['attempts']):
        raise ValueError('Support attempts must stay within K20')
    seen = {}
    for receipt in collection['attempts']:
        task = resolve_task(manifest, freeze, receipt['task_id'])
        authorize_task(task, 'generation')
        if receipt['group_id'] != task['group_id'] or receipt['task_type'] != task['task_type']:
            raise ValueError('Support receipt differs from frozen task')
        directory = (root / receipt['episode_dir']).resolve()
        if not directory.is_relative_to(root / 'attempts'):
            raise ValueError('Support evidence escapes attempts directory')
        if (read_json(receipt_path(root, receipt)) != receipt
                or artifact_hashes(directory) != receipt['episode_hashes']
                or artifact_hashes(directory.parent / 'teacher') != receipt['teacher_hashes']):
            raise ValueError('Support attempt evidence changed')
        episode = read_json(directory / 'episode.json') if (directory / 'episode.json').exists() else {}
        prior = seen.setdefault(receipt['task_id'], [])
        if (receipt['attempt'] != len(prior) + 1 or receipt['attempt'] > 3
                or any(r['won'] for r in prior) or receipt['won'] != environment_won(directory, episode)):
            raise ValueError('Support attempt sequence/win changed')
        prior.append(receipt)
    episodes = list(checked_episodes(root, collection['verified']))
    for receipt, episode in zip(collection['verified'], episodes, strict=True):
        task = resolve_task(manifest, freeze, receipt['task_id'])
        if (receipt not in collection['attempts'] or receipt['won'] is not True
                or episode['task_id'] != task['task_id'] or episode['game_sha256'] != task['game_sha256']
                or episode['split'] != 'train'):
            raise ValueError('Support episode differs from frozen TRAIN task')
    return collection, root, episodes


def original_support_examples(path, tokenizer, manifest, freeze, *, adapter, splits):
    """Existing turn encoder for the serving gate; no new prompt/masking rules."""
    from train.dataset import episode_examples
    _, _, episodes = load_original_support(path, manifest, freeze, adapter=adapter, splits=splits)
    return [example for episode in episodes
            for example in episode_examples(episode, tokenizer, packing='turn')]
