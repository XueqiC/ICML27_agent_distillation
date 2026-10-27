"""Checked coverage imports and public decision points for V1a/V1b.

The evidence checks are adapted from tools.cg_probe.load_cases. CG keeps its
original loader; this adapter removes only its greedy/ten-turn/unseen limits.
No environment is opened here. Execution must still reset and verify replay.
"""
import json
from pathlib import Path
import re

from repair.artifacts import read_json
from repair.facts import FactTable, observation
from repair.records import TrajectoryRecord
from sed.manifest import authorize_task, load_frozen, resolve_task
from sed.react import action_outcome
from sed.storage import digest, file_hash


RULE_VERSION = 'dp/1'
ATTEMPT_FIELDS = ('attempt', 'decoding', 'sampling_seed')
EVIDENCE_FILES = {'episode.json': 'episode_sha256', 'events.jsonl': 'events_sha256',
                  'provenance.json': 'provenance_sha256'}
INDEX_FIELDS = {'task_id', 'task_type', 'group_id', 'episode_id', 'episode_dir',
                *ATTEMPT_FIELDS, *EVIDENCE_FILES.values(), 'success', 'status',
                'complete', 'split', 'sealed', 'seed', 'max_episode_steps'}


def _index(path):
    result = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row['episode_id']
            if key in result:
                raise ValueError('Duplicate coverage episode identity')
            # Rollout rows also contain histories/audit fields. Keep metadata only.
            result[key] = {k: v for k, v in row.items() if k in INDEX_FIELDS}
    return result


def coverage_rows(failures_path, coverage_root):
    """Return checked metadata from the full index (or a legacy failure list).

    repair_coverage writes hashes in rollouts.jsonl and attempt metadata in
    episode_index.jsonl. Join on episode_id, checking all shared fields. Never
    infer unindexed episodes by globbing directories or manufacture their hashes.
    """
    root, path = Path(coverage_root).resolve(), Path(failures_path).resolve()
    if path.parent != root:
        raise ValueError('Failure list must belong to coverage-root')
    failures, summary = read_json(path), read_json(root / 'summary.json')
    rows = failures.get('failures', failures.get('failure_list'))
    if not isinstance(rows, list) or digest(rows) != failures.get('list_sha256', failures.get('failure_list_sha256')):
        raise ValueError('Coverage failure-list hash differs')
    if (summary.get('failure_list') != rows or summary.get('failure_list_sha256') != digest(rows)
            or summary.get('branch_decision_actionable') is not True):
        raise ValueError('Failure list differs from actionable coverage summary')
    inputs = {str(path): file_hash(path), str(root / 'summary.json'): file_hash(root / 'summary.json')}
    index_path, rollout_path = root / 'episode_index.jsonl', root / 'rollouts.jsonl'
    if index_path.exists():
        indexed = _index(index_path)
        inputs[str(index_path)] = file_hash(index_path)
        if rollout_path.exists():
            rollouts = _index(rollout_path)
            inputs[str(rollout_path)] = file_hash(rollout_path)
            if indexed.keys() != rollouts.keys():
                raise ValueError('Coverage episode index differs from rollout index')
            for key, row in indexed.items():
                other = rollouts[key]
                if any(row[k] != other[k] for k in row.keys() & other.keys()):
                    raise ValueError('Coverage episode indexes disagree')
                indexed[key] = {**other, **row}
        for row in rows:
            other = indexed.get(row['episode_id'])
            if other is None or any(other.get(k) != v for k, v in row.items() if k in INDEX_FIELDS):
                raise ValueError('Coverage failure list differs from episode index')
        rows = list(indexed.values())
    elif rollout_path.exists():
        raise ValueError('Coverage rollout index lacks attempt metadata')
    if len({r['episode_id'] for r in rows}) != len(rows) or len({r['episode_dir'] for r in rows}) != len(rows):
        raise ValueError('Duplicate coverage episode identity or directory')
    for row in rows:
        if not {*ATTEMPT_FIELDS, *EVIDENCE_FILES.values()} <= row.keys():
            raise ValueError('Coverage episode lacks attempt metadata or evidence hashes')
    return rows, inputs


def checked_episode(row, root, config, *, frozen=None):
    """Verify evidence and frozen authorization before exposing public records."""
    root = Path(root).resolve()
    directory = (root / row['episode_dir']).resolve()
    if Path(row['episode_dir']).is_absolute() or not directory.is_relative_to(root):
        raise ValueError('Coverage episode escapes coverage-root')
    if Path(row['task_id']).parent.as_posix() != row['group_id'] or not row['group_id'].startswith('train/'):
        raise ValueError('Coverage task/group mismatch')
    hashes = {}
    for name, key in EVIDENCE_FILES.items():
        path = directory / name
        if not path.resolve().is_relative_to(root) or file_hash(path) != row[key]:
            raise ValueError('Coverage episode evidence changed')
        hashes[str(path)] = row[key]
    ep = read_json(directory / 'episode.json')
    if (ep['task_id'] != row['task_id'] or ep.get('episode_id') != row['episode_id']
            or type(ep.get('success')) is not bool or ep.get('split') != 'train' or ep.get('sealed')
            or ep.get('complete') is not True or ep.get('status') != 'ok'
            or ep.get('seed') != config['environment']['seed']
            or ep.get('max_episode_steps') != 40 or config['environment']['max_episode_steps'] != 40):
        raise ValueError('Frozen coverage identity/outcome/config differs')
    for key in ('success', 'complete', 'status', 'split', 'sealed', 'seed', 'max_episode_steps'):
        if key in row and row[key] != ep.get(key):
            raise ValueError('Coverage index differs from episode')
    task = (resolve_task(*frozen, row['task_id']) if frozen is not None else
            dict(task_id=row['task_id'], split='train', sealed=False, group_id=row['group_id']))
    authorize_task(task, 'generation')
    if frozen is not None and (task['source_split'] != 'train' or ep['game_sha256'] != task['game_sha256']
                               or task['group_id'] != row['group_id']):
        raise ValueError('Frozen task/game identity differs')
    if 'task_type' in row and 'task_type' in task and row['task_type'] != task['task_type']:
        raise ValueError('Frozen task type differs')
    provenance = read_json(directory / 'provenance.json')
    recorded = provenance.get('config', {}).get('r03_rollout', {})
    if any(k in recorded and recorded[k] != row[k] for k in ATTEMPT_FIELDS):
        raise ValueError('Coverage attempt provenance differs')
    if type(row['attempt']) is not int or row['attempt'] < 1 or not isinstance(row['decoding'], str):
        raise ValueError('Invalid coverage attempt metadata')
    if row['sampling_seed'] is not None and type(row['sampling_seed']) is not int:
        raise ValueError('Invalid coverage sampling seed')
    source = dict(episode_id=row['episode_id'], evidence_sha256=hashes,
                  environment_revision=ep.get('environment_revision'), provenance=provenance,
                  **{k: row[k] for k in ATTEMPT_FIELDS})
    return ep, {**task, 'parent_seed_ids': [row['episode_id']]}, source


def load_original(entry, config, *, dry_run=False, _coverage=None, _frozen=None):
    """Load any verified failed rollout; branch_index counts executed turns.

    Paths are absolute or relative to cwd; load_spec resolves spec-relative paths.
    dry_run bypasses only frozen-manifest resolution, exactly as the CG loader.
    """
    source = entry['source']
    rows, inputs = _coverage if _coverage is not None else coverage_rows(source['failures_json'], source['coverage_root'])
    matches = [r for r in rows if r['episode_dir'] == source['episode_dir']]
    if len(matches) != 1:
        raise ValueError('Requested episode missing or duplicated in coverage')
    row = matches[0]
    if row['task_id'] != entry['task_id'] or row['group_id'] != 'train/' + entry['group']:
        raise ValueError('Spec task/group differs from coverage')
    if any(k in entry and (entry[k] != row[k] or type(entry[k]) is not type(row[k])) for k in ATTEMPT_FIELDS):
        raise ValueError('Spec attempt differs from coverage')
    frozen = _frozen if _frozen is not None else (None if dry_run else load_frozen(
        config['paths']['manifest'], config['paths']['split_freeze']))
    ep, task, source = checked_episode(row, source['coverage_root'], config, frozen=frozen)
    if ep['success'] is not False:
        raise ValueError('Original coverage rollout must have failed')
    steps, n = ep['steps'], entry['branch_index']
    if type(n) is not int or not 0 <= n < min(len(steps), 40):
        raise ValueError('Branch must leave an executable continuation')
    if ep['initial'].get('terminated') or ep['initial'].get('truncated'):
        raise ValueError('Branch prefix is terminal')
    previous = ep['initial']['visible']
    if not isinstance(previous, dict):
        raise ValueError('Branch prefix lacks a public initial observation')
    for step in steps[:n]:
        action, outcome = action_outcome(step['action_raw'], previous['admissible_commands'])
        if (outcome == 'parser_error' or step.get('action_outcome') != outcome
                or step.get('action_parsed') != action or step.get('visible') is None
                or step.get('terminated') or step.get('truncated')):
            raise ValueError('Invalid, parser-error, or terminal prefix')
        previous = step['visible']
    steps = [{**s, 'feedback_source': 'analytic_verifier' if s.get('action_outcome') == 'parser_error'
              else 'real_environment'} for s in steps]
    reason = ('CONTEXT_LIMIT' if ep.get('termination_reason') == 'context_overflow' else
              'STEP_LIMIT' if ep.get('truncated') else 'FAILED')
    used = sum(s.get('environment_steps', 1) for s in steps)
    original = TrajectoryRecord(row['task_id'], ep['initial'], steps, False, reason,
        token_costs={k: ep.get('metrics', {}).get(k) for k in
                     ('prompt_tokens', 'completion_tokens', 'thinking_tokens', 'thinking_chars')},
        environment_steps=used, environment_steps_confirmed=used, game_sha256=ep['game_sha256'],
        env_seed=ep['seed'], max_steps=40, error_type='context_overflow' if reason == 'CONTEXT_LIMIT' else None)
    if 'task_type' in entry and entry['task_type'] != row.get('task_type', entry['group'].split('-')[0]):
        raise ValueError('Spec task type differs from coverage')
    if 'decision_rule_version' in entry and entry['decision_rule_version'] != RULE_VERSION:
        raise ValueError('Unsupported decision rule version')
    source['coverage_sha256'] = inputs
    return dict(task=task, original=original, group=entry['group'], target=entry['group'].split('-')[1].lower(),
                source=source)


def decision_point(initial, steps):
    """dp/1: scan the whole failed rollout's public observations and actions.

    Initial observation has event_step=-1. The earliest event wins, with ties
    ordered refused, target_seen, target_taken. Branch at event_step + 1, capped
    to len(steps)-1, including for a final event. If no target instance appears in
    any observation, use search_after_10 at min(10, len(steps)-1).
    The full rollout selects the branch; teaching inputs use only its public prefix.
    A textual instance mention counts as seen (including inventory/take results),
    without inferring a location. Goal, holdings and refusals use FactTable only.
    """
    if not steps:
        raise ValueError('Decision point needs at least one policy turn')
    facts = FactTable(initial)
    goal = facts.goal['value']
    if goal is None:
        raise ValueError('Decision point requires a publicly parsed goal')
    target = re.escape(goal['object_type']) + r' \d+\b'
    mention, take = re.compile(r'\b' + target, re.I), re.compile(r'^take ' + target + r' from ', re.I)
    events = []
    if mention.search(observation(initial)):
        events.append((-1, 1, 'target_seen'))
    seen = bool(events)
    for i, step in enumerate(steps):
        facts.update(step)
        action = step.get('action_parsed') or ''
        refused = i in facts.refused_actions.get(action, {}).get('evidence_steps', [])
        if refused and (mention.search(action) or any(r['step'] == i for r in facts.refused_takes_while_holding)):
            events.append((i, 0, 'refused'))
        if mention.search(observation(step)):
            seen = True
            events.append((i, 1, 'target_seen'))
        if take.search(action) and step.get('action_outcome') == 'legal' and not refused:
            events.append((i, 2, 'target_taken'))
    if seen:
        event, _, label = min(events)
        branch = min(event + 1, len(steps) - 1)
    else:
        branch, label = min(10, len(steps) - 1), 'search_after_10'
        event = branch - 1
    return dict(decision_point=label, decision_rule_version=RULE_VERSION, event_step=event,
                branch_index=branch, steps_remaining=40 - branch)


def coverage_stratum(successes, total):
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total or total < 1:
        raise ValueError('Coverage stratum needs 0 <= k <= n and n > 0')
    return 'excluded' if successes == total else 'floor' if successes == 0 else 'middle'
