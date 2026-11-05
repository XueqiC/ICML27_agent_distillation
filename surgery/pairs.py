"""P1 frozen A split and teacher-forced K20 base-model extraction for B."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import random

from repair.p0_pairs import CATEGORIES, assert_disjoint, render_state
from repair.records import TrajectoryRecord
from sed.react import build_messages
from sed.storage import digest, file_hash
from surgery.common import Relocation, checked, read, seal, provenance

DEFAULT_PAIRS = 'reports/p0_tooling/extraction_v3/pairs.json'
DEFAULT_VERIFIED = '/home/xueqi/hq/projects/ICML27_agent_distillation/data/verified/seeds_full_train_r2/subsets/k20'
P0_HASH = '432fd178'
P0_DIGEST = 'cb4e16996b336e0acb30de7f31c16f05e931aec0c80f3ea747d6c8ecf71533f3'
SPLIT_RULE = ('Sorted categories; for each, sorted group_ids containing that category shuffled '
              'with a fresh Random(0). Greedily take each unselected whole group if none of '
              'its category counts exceeds max(6,floor(category_total/2)). Shared-category '
              'groups contribute to all categories. Require every target exactly; otherwise fail.')


def seed_replay_version(source_provenance):
    """Translate runner's different version-hash schema, never its state evidence.

    The seed runner predates repair.execute. Verify the original environment,
    parser, visibility and environment dependencies before constructing the P0
    replay fingerprint. P0 still performs a real reset and checks every visible
    observation, audit hash, action, reward and terminal bit against the seed.
    """
    from repair.p0_pairs import ROOT
    from importlib.metadata import version
    dependencies = {n: version(n) for n in ('alfworld', 'textworld', 'numpy')}
    if any(source_provenance['dependencies'].get(n) != v for n, v in dependencies.items()):
        raise ValueError('Seed environment dependencies differ from the replay runtime')
    reused = ('harness/alfworld_env/environment.py', 'src/sed/react.py', 'src/sed/visibility.py')
    if any(source_provenance['code_sha256'].get(p) != file_hash(ROOT / p) for p in reused):
        raise ValueError('Seed environment/parser/visibility code differs')
    return digest(dict(dependencies=dependencies, adapter='harness.alfworld_env.environment.Environment',
        backend='harness.alfworld_env.environment.TextWorldBackend',
        code={p: file_hash(ROOT / p) for p in (*reused, 'repair/align.py', 'repair/execute.py')}))


def split_a(pairs):
    groups = defaultdict(list)
    for p in pairs:
        groups[p['group_id']].append(p)
    totals = Counter(p['category'] for p in pairs)
    if set(totals) != set(CATEGORIES):
        raise ValueError('Expected all five P1 categories')
    targets = {c: max(6, n // 2) for c, n in totals.items()}
    selected, counts = set(), Counter()
    for category in sorted(CATEGORIES):
        order = sorted(g for g, ps in groups.items() if any(p['category'] == category for p in ps))
        random.Random(0).shuffle(order)
        for group in order:
            addition = Counter(p['category'] for p in groups[group])
            if group not in selected and all(counts[c] + n <= targets[c] for c, n in addition.items()):
                selected.add(group)
                counts.update(addition)
    if dict(counts) != targets:
        raise ValueError('Exact category half-counts unavailable with the registered greedy split')
    result = [dict(deepcopy(p), split='diagnosis' if p['group_id'] in selected else 'test') for p in pairs]
    assert_disjoint(result)
    return result


def counts(pairs):
    return {s: dict(pairs=sum(p['split'] == s for p in pairs),
        categories=dict(Counter(p['category'] for p in pairs if p['split'] == s)),
        unique_instances=len({p['task_id'] for p in pairs if p['split'] == s}),
        task_groups=len({p['group_id'] for p in pairs if p['split'] == s})) for s in ('diagnosis', 'test')}


def extract_a(path=DEFAULT_PAIRS, relocation=None):
    relocation = relocation or Relocation()
    path = relocation.path(path)
    source = checked(read(path), 'p0-pairs/3')
    if source['sha256'] != P0_DIGEST or not file_hash(path).startswith(P0_HASH) or len(source['pairs']) != 113:
        raise ValueError('Setting A requires the frozen 113 P0 v3 pairs')
    relocation.inputs(source['inputs'])
    pairs = split_a(source['pairs'])
    return seal(dict(schema='p1-pairs/1', setting='A', pairs=pairs,
        inputs={str(path): file_hash(path), **source['inputs']}, student=source['student'],
        serving=source['serving'], manifest_path=source['manifest_path'],
        manifest_sha256=source['manifest_sha256'], source_sha256=source['sha256'],
        counts=counts(pairs), split_rule=SPLIT_RULE,
        split_sha256=digest({p['pair_id']: p['split'] for p in pairs}),
        provenance=provenance(dict(source_pairs=str(path), K=20))))


def load_pairs(path, relocation=None, *, verify_inputs=True):
    relocation = relocation or Relocation()
    value = checked(read(relocation.path(path)), 'p1-pairs/1')
    pairs = value['pairs']
    if len({p['pair_id'] for p in pairs}) != len(pairs):
        raise ValueError('Duplicate pair IDs')
    if value['split_sha256'] != digest({p['pair_id']: p['split'] for p in pairs}):
        raise ValueError('Split hash mismatch')
    if value['setting'] == 'A':
        assert_disjoint(pairs)
    elif value['setting'] != 'B' or any(p['split'] != 'diagnosis' or p['category'] != 'none' for p in pairs):
        raise ValueError('Setting B has only uncategorized diagnosis pairs')
    if verify_inputs:
        relocation.inputs(value['inputs'])
    return value


def verified_records(directory, relocation=None):
    """Check the three frozen views and every selected episode's real win evidence."""
    from sed.seed_evidence import checked_episode
    relocation = relocation or Relocation()
    directory = relocation.path(directory)
    paths = [directory / (n + '.json') for n in ('manifest', 'selection', 'identity')]
    manifest, selection, identity = map(read, paths)
    if (manifest.get('K') != 20 or len(manifest['verified']) != 20
            or manifest['status'] != 'complete' or manifest['selection_sha256'] != digest(selection)
            or manifest['identity_sha256'] != digest(identity) or identity['selection'] != selection):
        raise ValueError('Expected a complete, identity-bound verified K20 collection')
    root = relocation.path(manifest.get('source_collection', {}).get('path', '.'), root=directory)
    inputs = {str(p): file_hash(p) for p in paths}
    if 'source_collection' in manifest:
        source = read(root / 'manifest.json')
        if digest(source) != manifest['source_collection']['manifest_sha256']:
            raise ValueError('Verified source collection changed')
        inputs[str(root / 'manifest.json')] = file_hash(root / 'manifest.json')
        if any(r not in source['verified'] for r in manifest['verified']):
            raise ValueError('Subset contains an unverified source trajectory')
    rows = []
    for receipt in manifest['verified']:
        episode = checked_episode(root, receipt)
        if (not receipt['task_id'].startswith('train/') or episode['task_id'] != receipt['task_id']
                or episode['split'] != 'train' or receipt.get('won') is not True):
            raise ValueError('Only verified TRAIN trajectories are allowed')
        source_provenance = read(root / receipt['episode_dir'] / 'provenance.json')
        replay_version = seed_replay_version(source_provenance)
        record = TrajectoryRecord(task_id=episode['task_id'], initial=episode['initial'], steps=episode['steps'],
            success=True, termination_reason='SUCCESS', env_seed=episode['seed'],
            environment_version=replay_version, game_sha256=episode['game_sha256'],
            max_steps=episode['max_episode_steps'])
        for name, sha in receipt['episode_hashes'].items():
            inputs[str(root / receipt['episode_dir'] / name)] = sha
        rows.append((record, receipt['group_id'], dict(source_environment_revision=episode['environment_revision'],
            replay_environment_version=replay_version, version_schema_translation='runner -> repair.execute; original environment code/dependencies verified')))
    if len({r[0].task_id for r in rows}) != 20:
        raise ValueError('K20 trajectories must have distinct instance IDs')
    return rows, inputs


def extract_b(student, records, *, inputs=None, ledger=None):
    pairs, matched, decisions = [], [], []
    for item in records:
        record, group = item[:2]
        for t, step in enumerate(record.steps):
            context = record.decision_context(t)
            messages = build_messages(context['history'], {'task_id': record.task_id}, context['visible_context'])
            pair = dict(pair_id='B-' + digest([record.trajectory_id, t])[:24], task_id=record.task_id,
                group_id=group, parent_seed_task_id=record.task_id, category='none', split='diagnosis', step=t,
                state=dict(context=context, messages=messages, sha256=digest(messages)),
                original_action_raw=step['action_raw'], a_r=step['action_parsed'],
                review=dict(step=t, category='none', reason='Verified trajectory action'),
                audit=dict(original=record.to_json(), env_seed=record.env_seed, **(item[2] if len(item) > 2 else {})))
            if ledger:
                ledger.append(dict(event='started', pair_id=pair['pair_id'], parent_seed=record.task_id, step=t))
            try:
                result = student.greedy(render_state(pair, student.tokenizer))
            except Exception as exc:
                if ledger:
                    ledger.append(dict(event='failed', pair_id=pair['pair_id'], error_type=type(exc).__name__,
                                       student_usage=deepcopy(student.usage)))
                raise
            pair.update(a_f=result['action'], original_action_raw=result['reply'],
                        reproduction=dict(status='executed', greedy_action=result['action'], matches=True))
            # For P0 original_thought probes, use the failing base reply, while
            # all preceding replies remain the trusted trajectory's history.
            decisions.append(dict(pair_id=pair['pair_id'], step=t, task_id=record.task_id,
                                  greedy=result, trajectory_action=step['action_parsed']))
            if ledger:
                ledger.append(dict(event='settled', **decisions[-1]))
            (matched if result['action'] == step['action_parsed'] else pairs).append(pair)
    return seal(dict(schema='p1-pairs/1', setting='B', pairs=pairs, matched_pool=matched,
        decisions=decisions, inputs=inputs or {}, student=student.identity,
        serving=dict(max_tokens=student.max_new_tokens, max_model_len=student.max_model_len),
        split_rule='All mismatched teacher-forced trajectory steps are diagnosis; no test split',
        split_sha256=digest({p['pair_id']: 'diagnosis' for p in pairs}), counts=counts(pairs),
        provenance=provenance(dict(K=20, teacher_forced_history=True)),
        student_usage=deepcopy(student.usage)))


def collateral_b(pairs, count=200):
    # Fail explicitly if fewer than 200 matched decisions; never duplicate data.
    return random.Random(0).sample(pairs['matched_pool'], count)
